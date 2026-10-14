import os
import pytest
from src.utterances import UTTERANCES
from src.audio_gen import AudioGenerator
from src.stt import SpeechToTextPipeline
from src.metrics import compute_wer, normalize_text
from src.report import EvalResult, compute_eval_summary, generate_markdown_report

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pipeline_slice_real_model(tmp_path):
    # Slice of 2 utterances
    test_slice = UTTERANCES[:2]
    audio_dir = str(tmp_path / "audio")
    
    # Initialize generator and STT pipeline
    audio_gen = AudioGenerator(output_dir=audio_dir, sample_rate=16000)
    stt = SpeechToTextPipeline(model_name="tiny", device="cpu", compute_type="int8")

    results = []
    for utt in test_slice:
        # Generate audio (using fallback tone or gTTS)
        wav_path = audio_gen.generate(utt, use_fallback=True)
        assert os.path.exists(wav_path)

        # Transcribe with real faster-whisper model
        stt_res = stt.transcribe_file(wav_path)
        assert stt_res.duration_sec > 0.0

        wer = compute_wer(utt.text, stt_res.text)
        assert 0.0 <= wer <= 2.0

        results.append(EvalResult(
            utterance_id=utt.id,
            text=utt.text,
            hypothesis=stt_res.text,
            wer=wer,
            latency_ms=stt_res.duration_sec * 1000.0,
            detected_language=stt_res.detected_language,
            failure_modes=utt.failure_modes,
            domain=utt.domain
        ))

    assert len(results) == 2
    summary = compute_eval_summary(results, rate_per_sec=0.0004)
    assert summary["overall"]["count"] == 2
    assert summary["overall"]["p50_latency_ms"] > 0.0
