#!/usr/bin/env python3
"""Indic Voice Agent Eval Runner.

One-command reproduction:
1. Synthesizes Hindi/Hinglish customer-support utterances to 16kHz WAV via gTTS (with noise injection for background_noise tag).
2. Transcribes using faster-whisper (tiny model, CPU).
3. Measures WER, p50/p95 latency, cost per 1k calls, and failure-mode matrix.
4. Evaluates regression thresholds against config.yaml.
5. Saves report to report/latest.md.
"""

import argparse
import os
import sys
import yaml

from src import __version__
from src.utterances import UTTERANCES
from src.audio_gen import AudioGenerator
from src.stt import SpeechToTextPipeline
from src.metrics import compute_wer
from src.report import (
    EvalResult,
    compute_eval_summary,
    check_thresholds,
    generate_markdown_report,
    save_report,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_evaluation(
    config_path: str = "config.yaml",
    report_output_path: str = "report/latest.md",
    fail_on_regression: bool = True,
    limit: int = 0,
) -> int:
    """Execute complete evaluation harness and generate report."""
    print("=" * 70)
    print(f"Indic Voice Agent Evaluation Harness v{__version__}")
    print("=" * 70)

    config = load_config(config_path)
    pipeline_cfg = config.get("pipeline", {})
    pricing_cfg = config.get("pricing", {})
    thresholds_cfg = config.get("thresholds", {})

    audio_dir = pipeline_cfg.get("audio_dir", "data/audio")
    sample_rate = pipeline_cfg.get("sample_rate", 16000)
    noise_snr_db = pipeline_cfg.get("noise_snr_db", 15.0)

    whisper_model = pipeline_cfg.get("whisper_model", "tiny")
    whisper_device = pipeline_cfg.get("whisper_device", "cpu")
    compute_type = pipeline_cfg.get("compute_type", "int8")

    cloud_rate = pricing_cfg.get("cloud_reference_rate_per_sec", 0.0004)
    provider_name = pricing_cfg.get("cloud_reference_provider", "AWS Transcribe")

    utterances = UTTERANCES[:limit] if limit > 0 else UTTERANCES
    print(f"[*] Total utterances to evaluate: {len(utterances)}")
    print(f"[*] Audio cache directory: {audio_dir}")
    print(f"[*] STT Engine: faster-whisper ({whisper_model}, device={whisper_device}, {compute_type})")

    # Step 1: Synthesize audio
    print("\n[Phase 1/2] Synthesizing audio with gTTS (and noise injection for tagged sets)...")
    audio_gen = AudioGenerator(
        output_dir=audio_dir,
        sample_rate=sample_rate,
        noise_snr_db=noise_snr_db,
    )
    wav_paths = {}
    for idx, utt in enumerate(utterances, 1):
        path = audio_gen.generate(utt, use_cache=True)
        wav_paths[utt.id] = path
        if idx % 10 == 0 or idx == len(utterances):
            print(f"  -> Synthesized {idx}/{len(utterances)} audio files")

    # Step 2: Speech-to-Text inference & timing
    print("\n[Phase 2/2] Running STT inference & measuring wall-clock latency...")
    stt = SpeechToTextPipeline(
        model_name=whisper_model,
        device=whisper_device,
        compute_type=compute_type,
    )

    results = []
    for idx, utt in enumerate(utterances, 1):
        audio_file = wav_paths[utt.id]
        stt_res = stt.transcribe_file(audio_file)

        wer = compute_wer(utt.text, stt_res.text)
        latency_ms = stt_res.duration_sec * 1000.0

        eval_res = EvalResult(
            utterance_id=utt.id,
            text=utt.text,
            hypothesis=stt_res.text,
            wer=wer,
            latency_ms=latency_ms,
            detected_language=stt_res.detected_language,
            failure_modes=utt.failure_modes,
            domain=utt.domain,
        )
        results.append(eval_res)

        print(
            f"  [{idx:02d}/{len(utterances):02d}] {utt.id} | WER: {wer:.2f} | Latency: {latency_ms:.0f}ms | Ref: '{utt.text[:28]}...' -> Hyp: '{stt_res.text[:28]}...'"
        )

    # Step 3: Compute summary metrics
    print("\n[*] Aggregating evaluation statistics...")
    summary = compute_eval_summary(results, rate_per_sec=cloud_rate)
    threshold_status = check_thresholds(summary, thresholds_cfg)

    # Step 4: Generate and save report
    report_md = generate_markdown_report(
        summary=summary,
        threshold_status=threshold_status,
        config=config,
        individual_results=results,
    )
    saved_path = save_report(report_md, output_path=report_output_path)
    print(f"[+] Evaluation report generated at: {saved_path}")

    # Step 5: Print summary to stdout
    ov = summary["overall"]
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Utterances    : {ov['count']}")
    print(f"Overall Mean WER    : {ov['mean_wer']:.3f} ({ov['mean_wer']*100:.1f}%)")
    print(f"Latency (p50)       : {ov['p50_latency_ms']:.1f} ms")
    print(f"Latency (p95)       : {ov['p95_latency_ms']:.1f} ms")
    print(f"Cost per 1k (Local) : ${ov['cost_per_1k_local']:.4f}")
    print(f"Cost per 1k ({provider_name}) : ${ov['cost_per_1k_cloud']:.4f}")
    print("-" * 70)
    print("Failure Mode Matrix:")
    print(f"{'Tag':<18} {'Count':<7} {'Mean WER':<12} {'p50 (ms)':<10} {'p95 (ms)':<10} {'Cloud Cost/1k'}")
    for tag, stats in summary["by_tag"].items():
        print(
            f"{tag:<18} {stats['count']:<7} {stats['mean_wer']:<12.3f} {stats['p50_latency_ms']:<10.1f} {stats['p95_latency_ms']:<10.1f} ${stats['cost_per_1k_cloud']:.4f}"
        )
    print("=" * 70)

    if threshold_status["passed"]:
        print("[SUCCESS] All metrics passed configured regression thresholds.")
        return 0
    else:
        print("[REGRESSION DETECTED] Failures encountered:")
        for failure in threshold_status["failures"]:
            print(f"  - {failure}")
        if fail_on_regression:
            return 1
        return 0


def main():
    parser = argparse.ArgumentParser(description="Indic Voice Agent Evaluation Harness")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--output-report", default="report/latest.md", help="Path to save latest.md report"
    )
    parser.add_argument(
        "--no-fail", action="store_true", help="Do not exit with code 1 on regression failure"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit number of utterances to run (0 for all)"
    )
    args = parser.parse_args()

    exit_code = run_evaluation(
        config_path=args.config,
        report_output_path=args.output_report,
        fail_on_regression=not args.no_fail,
        limit=args.limit,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
