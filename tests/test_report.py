import pytest
from src.report import (
    EvalResult,
    compute_eval_summary,
    check_thresholds,
    generate_markdown_report,
)

@pytest.fixture
def sample_results():
    return [
        EvalResult(
            utterance_id="u1",
            text="mera refund nahi aaya",
            hypothesis="mera refund nahi aaya",
            wer=0.0,
            latency_ms=250.0,
            detected_language="hi",
            failure_modes=["clean"],
            domain="banking"
        ),
        EvalResult(
            utterance_id="u2",
            text="card block kar do please",
            hypothesis="card block kar do please",
            wer=0.0,
            latency_ms=300.0,
            detected_language="hi",
            failure_modes=["clean"],
            domain="banking"
        ),
        EvalResult(
            utterance_id="u3",
            text="order cancel karo paise wapas bhejo",
            hypothesis="order cancel kardo paise wapas",
            wer=0.33,
            latency_ms=450.0,
            detected_language="hi",
            failure_modes=["code_mixing"],
            domain="ecommerce"
        ),
        EvalResult(
            utterance_id="u4",
            text="humra paisa kab aai batawa",
            hypothesis="hamara paisa kab aayega batao",
            wer=0.60,
            latency_ms=500.0,
            detected_language="hi",
            failure_modes=["regional_accent"],
            domain="banking"
        ),
        EvalResult(
            utterance_id="u5",
            text="traffic horn sound mera parcel deliver karo",
            hypothesis="sound mera parcel deliver",
            wer=0.50,
            latency_ms=600.0,
            detected_language="hi",
            failure_modes=["background_noise"],
            domain="delivery"
        ),
    ]

def test_compute_eval_summary(sample_results):
    summary = compute_eval_summary(sample_results, rate_per_sec=0.0004)
    assert summary["overall"]["count"] == 5
    assert "clean" in summary["by_tag"]
    assert "code_mixing" in summary["by_tag"]
    assert "regional_accent" in summary["by_tag"]
    assert "background_noise" in summary["by_tag"]
    assert summary["by_tag"]["clean"]["count"] == 2
    assert summary["by_tag"]["clean"]["mean_wer"] == 0.0

def test_check_thresholds_pass(sample_results):
    summary = compute_eval_summary(sample_results, rate_per_sec=0.0004)
    thresholds = {
        "max_wer_overall": 0.80,
        "max_wer_clean": 0.10,
        "max_wer_code_mixing": 0.50,
        "max_wer_regional_accent": 0.70,
        "max_wer_background_noise": 0.60,
        "max_p95_latency_ms": 1000.0,
    }
    status = check_thresholds(summary, thresholds)
    assert status["passed"] is True
    assert len(status["failures"]) == 0

def test_check_thresholds_fail_wer(sample_results):
    summary = compute_eval_summary(sample_results, rate_per_sec=0.0004)
    # Set strict threshold for regional accent WER (actual is 0.60)
    thresholds = {
        "max_wer_overall": 0.80,
        "max_wer_regional_accent": 0.40,
    }
    status = check_thresholds(summary, thresholds)
    assert status["passed"] is False
    assert any("regional_accent" in f for f in status["failures"])

def test_check_thresholds_fail_latency(sample_results):
    summary = compute_eval_summary(sample_results, rate_per_sec=0.0004)
    # Set low p95 threshold (max latency in sample is 600ms)
    thresholds = {
        "max_p95_latency_ms": 300.0,
    }
    status = check_thresholds(summary, thresholds)
    assert status["passed"] is False
    assert any("p95_latency" in f for f in status["failures"])

def test_generate_markdown_report(sample_results):
    summary = compute_eval_summary(sample_results, rate_per_sec=0.0004)
    threshold_status = {
        "passed": True,
        "failures": []
    }
    config = {
        "pricing": {
            "local_compute_per_sec": 0.0,
            "cloud_reference_provider": "AWS Transcribe",
            "cloud_reference_rate_per_sec": 0.0004,
            "calls_unit": 1000,
        },
        "pipeline": {
            "whisper_model": "tiny",
            "whisper_device": "cpu",
            "compute_type": "int8",
        }
    }
    md = generate_markdown_report(summary, threshold_status, config)
    assert "# Indic Voice Agent Evaluation Report" in md
    assert "Failure Mode Matrix" in md
    assert "clean" in md
    assert "code_mixing" in md
    assert "regional_accent" in md
    assert "background_noise" in md
    assert "AWS Transcribe" in md
    assert "Pricing Assumption" in md
    assert "Regression Check: PASSED" in md
