"""Evaluation report generator with failure-mode matrix, regression threshold checks, and pricing models."""

from dataclasses import dataclass, field
import datetime
import os
from typing import Any, Dict, List, Optional
import numpy as np

from src.metrics import (
    compute_cost_per_1k,
    compute_latency_percentiles,
)


@dataclass
class EvalResult:
    utterance_id: str
    text: str
    hypothesis: str
    wer: float
    latency_ms: float
    detected_language: str
    failure_modes: List[str] = field(default_factory=list)
    domain: str = "general"


def compute_eval_summary(
    results: List[EvalResult],
    rate_per_sec: float = 0.0004,
    unit_calls: int = 1000,
) -> Dict[str, Any]:
    """Aggregate per-utterance results into overall metrics and failure-mode breakdown."""
    if not results:
        return {
            "overall": {
                "count": 0,
                "mean_wer": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "cost_per_1k_local": 0.0,
                "cost_per_1k_cloud": 0.0,
            },
            "by_tag": {},
        }

    all_wers = [r.wer for r in results]
    all_latencies = [r.latency_ms for r in results]
    latency_stats = compute_latency_percentiles(all_latencies)

    mean_sec = (latency_stats["mean"] / 1000.0) if latency_stats["count"] > 0 else 0.0
    cost_cloud = compute_cost_per_1k(mean_sec, rate_per_sec, unit_calls)

    summary: Dict[str, Any] = {
        "overall": {
            "count": len(results),
            "mean_wer": float(np.mean(all_wers)),
            "p50_latency_ms": latency_stats["p50"],
            "p90_latency_ms": latency_stats["p90"],
            "p95_latency_ms": latency_stats["p95"],
            "p99_latency_ms": latency_stats["p99"],
            "min_latency_ms": latency_stats["min"],
            "max_latency_ms": latency_stats["max"],
            "mean_latency_ms": latency_stats["mean"],
            "cost_per_1k_local": 0.0,
            "cost_per_1k_cloud": cost_cloud,
        },
        "by_tag": {},
    }

    # Extract all unique failure tags
    known_tags = ["clean", "code_mixing", "regional_accent", "background_noise"]
    for tag in known_tags:
        tagged = [r for r in results if tag in r.failure_modes]
        if tagged:
            t_wers = [r.wer for r in tagged]
            t_latencies = [r.latency_ms for r in tagged]
            t_stats = compute_latency_percentiles(t_latencies)
            t_mean_sec = (t_stats["mean"] / 1000.0) if t_stats["count"] > 0 else 0.0
            summary["by_tag"][tag] = {
                "count": len(tagged),
                "mean_wer": float(np.mean(t_wers)),
                "p50_latency_ms": t_stats["p50"],
                "p95_latency_ms": t_stats["p95"],
                "cost_per_1k_cloud": compute_cost_per_1k(
                    t_mean_sec, rate_per_sec, unit_calls
                ),
            }

    return summary


def check_thresholds(
    summary: Dict[str, Any], thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Verify evaluation summary against configured regression thresholds."""
    failures: List[str] = []

    overall_wer = summary.get("overall", {}).get("mean_wer", 0.0)
    p95_latency = summary.get("overall", {}).get("p95_latency_ms", 0.0)

    # Overall WER check
    if "max_wer_overall" in thresholds and overall_wer > thresholds["max_wer_overall"]:
        failures.append(
            f"Overall WER {overall_wer:.3f} exceeded threshold {thresholds['max_wer_overall']:.3f}"
        )

    # Overall p95 latency check
    if (
        "max_p95_latency_ms" in thresholds
        and p95_latency > thresholds["max_p95_latency_ms"]
    ):
        failures.append(
            f"Overall p95_latency {p95_latency:.1f}ms exceeded threshold {thresholds['max_p95_latency_ms']:.1f}ms"
        )

    # Per-tag WER checks
    by_tag = summary.get("by_tag", {})
    for tag, tag_stats in by_tag.items():
        thresh_key = f"max_wer_{tag}"
        if thresh_key in thresholds:
            t_wer = tag_stats.get("mean_wer", 0.0)
            if t_wer > thresholds[thresh_key]:
                failures.append(
                    f"Failure mode '{tag}' WER {t_wer:.3f} exceeded threshold {thresholds[thresh_key]:.3f}"
                )

    return {
        "passed": len(failures) == 0,
        "failures": failures,
    }


def generate_markdown_report(
    summary: Dict[str, Any],
    threshold_status: Dict[str, Any],
    config: Dict[str, Any],
    individual_results: Optional[List[EvalResult]] = None,
) -> str:
    """Generate comprehensive Markdown eval report."""
    pricing_cfg = config.get("pricing", {})
    pipeline_cfg = config.get("pipeline", {})
    thresholds_cfg = config.get("thresholds", {})

    provider_name = pricing_cfg.get("cloud_reference_provider", "AWS Transcribe")
    rate_sec = pricing_cfg.get("cloud_reference_rate_per_sec", 0.0004)
    model_name = pipeline_cfg.get("whisper_model", "tiny")
    compute_type = pipeline_cfg.get("compute_type", "int8")

    ov = summary.get("overall", {})
    passed = threshold_status.get("passed", False)
    status_badge = "**PASSED**" if passed else "**REGRESSED (FAILED)**"

    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Indic Voice Agent Evaluation Report",
        "",
        f"> **Run Timestamp**: {timestamp}  ",
        f"> **Model**: `faster-whisper` (`{model_name}`, CPU, `{compute_type}`)  ",
        f"> **Pipeline Status**: **{status_badge}**",
        "",
        "## 1. Executive Summary",
        "",
        "| Metric | Measured Value | Threshold | Status |",
        "| :--- | :--- | :--- | :--- |",
        f"| **Evaluated Utterances** | {ov.get('count', 0)} phrases | N/A | Completed |",
        f"| **Overall Mean WER** | {ov.get('mean_wer', 0.0):.3f} ({ov.get('mean_wer', 0.0)*100:.1f}%) | <= {thresholds_cfg.get('max_wer_overall', 0.85):.3f} | {'PASS' if ov.get('mean_wer', 0.0) <= thresholds_cfg.get('max_wer_overall', 0.85) else 'FAIL'} |",
        f"| **p50 Latency** | {ov.get('p50_latency_ms', 0.0):.1f} ms | N/A | Informational |",
        f"| **p95 Latency** | {ov.get('p95_latency_ms', 0.0):.1f} ms | <= {thresholds_cfg.get('max_p95_latency_ms', 3500.0):.1f} ms | {'PASS' if ov.get('p95_latency_ms', 0.0) <= thresholds_cfg.get('max_p95_latency_ms', 3500.0) else 'FAIL'} |",
        f"| **Cost per 1k calls (Local)** | ${ov.get('cost_per_1k_local', 0.0):.4f} | $0.00 | Free local compute |",
        f"| **Cost per 1k calls ({provider_name})** | ${ov.get('cost_per_1k_cloud', 0.0):.4f} | Reference | Published pricing |",
        "",
        "## 2. Failure Mode Matrix",
        "",
        "Breakdown across failure modes (code-mixing, regional accents, background noise, clean baseline):",
        "",
        "| Failure Mode | Utterances | Mean WER | p50 Latency | p95 Latency | Cloud Cost/1k | Regression Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    by_tag = summary.get("by_tag", {})
    for tag, stats in by_tag.items():
        t_wer = stats.get("mean_wer", 0.0)
        t_thresh = thresholds_cfg.get(f"max_wer_{tag}", 1.0)
        t_status = "PASS" if t_wer <= t_thresh else f"FAIL (> {t_thresh:.2f})"
        lines.append(
            f"| `{tag}` | {stats.get('count', 0)} | {t_wer:.3f} ({t_wer*100:.1f}%) | "
            f"{stats.get('p50_latency_ms', 0.0):.1f} ms | {stats.get('p95_latency_ms', 0.0):.1f} ms | "
            f"${stats.get('cost_per_1k_cloud', 0.0):.4f} | {t_status} |"
        )

    lines.extend([
        "",
        "## 3. Pricing Assumption & Cost Model",
        "",
        "- **Local Compute**: Assumed **$0.00/hr** for all measurements on local developer CPU hardware.",
        f"- **Cloud Baseline Reference**: `{provider_name}` with published standard pricing of **$0.024 / minute** ($0.0004 / second).",
        f"- **Cost per 1k Formula**: `1,000 calls × mean_inference_duration_sec × rate_per_second`.",
        f"- For this run, mean STT duration was **{ov.get('mean_latency_ms', 0.0)/1000.0:.3f} s**, yielding a cloud equivalent of **${ov.get('cost_per_1k_cloud', 0.0):.4f}** per 1,000 customer support calls.",
        "",
        "## 4. Regression Check: " + ("PASSED" if passed else "FAILED"),
        "",
    ])

    if passed:
        lines.append(
            "All metrics satisfied configured thresholds defined in `config.yaml`."
        )
    else:
        lines.append("### Detected Regressions:")
        for fail_msg in threshold_status.get("failures", []):
            lines.append(f"- [FAIL] {fail_msg}")

    if individual_results:
        lines.extend([
            "",
            "## 5. Sample Utterance Evaluations",
            "",
            "| ID | Tag | Domain | Reference Ground Truth | STT Hypothesis | WER | Latency |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])
        # Display sample slice
        sample_slice = individual_results[:12]
        for res in sample_slice:
            tag_str = ",".join(res.failure_modes)
            ref_clean = res.text.replace("|", "/")
            hyp_clean = res.hypothesis.replace("|", "/")
            lines.append(
                f"| `{res.utterance_id}` | `{tag_str}` | {res.domain} | {ref_clean} | {hyp_clean} | {res.wer:.2f} | {res.latency_ms:.0f}ms |"
            )

    lines.append("")
    return "\n".join(lines)


def save_report(
    report_markdown: str, output_path: str = "report/latest.md"
) -> str:
    """Save report to target markdown file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_markdown)
    return output_path
