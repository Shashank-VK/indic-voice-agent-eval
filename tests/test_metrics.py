import pytest
from src.metrics import (
    normalize_text,
    compute_wer,
    compute_latency_percentiles,
    compute_cost_per_1k,
)

def test_normalize_text_basic():
    raw = "  Mera Order, Kahan Hai?!  "
    expected = "mera order kahan hai"
    assert normalize_text(raw) == expected

def test_normalize_text_punctuation_and_symbols():
    raw = "Account #1234 - 'Refund' status: pending (immediate)..."
    norm = normalize_text(raw)
    assert "#" not in norm
    assert "-" not in norm
    assert "'" not in norm
    assert ":" not in norm
    assert "(" not in norm
    assert ")" not in norm
    assert "." not in norm
    assert norm == "account 1234 refund status pending immediate"

def test_wer_identical():
    ref = "mera refund abhi tak nahi aaya"
    hyp = "mera refund abhi tak nahi aaya"
    assert compute_wer(ref, hyp) == 0.0

def test_wer_case_and_punct_insensitive():
    ref = "Mera refund, abhi tak nahi aaya!"
    hyp = "mera refund abhi tak nahi aaya."
    assert compute_wer(ref, hyp) == 0.0

def test_wer_single_substitution():
    ref = "mera payment fail ho gaya"  # 5 words
    hyp = "mera payment pass ho gaya"  # 1 substitution
    assert pytest.approx(compute_wer(ref, hyp), 0.001) == 0.20

def test_wer_single_deletion():
    ref = "mera payment fail ho gaya"  # 5 words
    hyp = "mera payment fail ho"       # 1 deletion
    assert pytest.approx(compute_wer(ref, hyp), 0.001) == 0.20

def test_wer_single_insertion():
    ref = "mera payment fail gaya"     # 4 words
    hyp = "mera payment fail ho gaya"  # 1 insertion
    assert pytest.approx(compute_wer(ref, hyp), 0.001) == 0.25

def test_wer_empty_hypothesis():
    ref = "kripya agent se connect karein"
    hyp = ""
    assert compute_wer(ref, hyp) == 1.0

def test_wer_both_empty():
    assert compute_wer("", "") == 0.0

def test_latency_percentiles_odd():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    p = compute_latency_percentiles(latencies)
    assert p["p50"] == 300.0
    assert p["min"] == 100.0
    assert p["max"] == 500.0
    assert p["mean"] == 300.0
    assert p["p95"] >= 400.0

def test_latency_percentiles_even():
    latencies = [10.0, 20.0, 30.0, 40.0]
    p = compute_latency_percentiles(latencies)
    assert p["p50"] == 25.0
    assert p["count"] == 4

def test_latency_percentiles_single_value():
    p = compute_latency_percentiles([150.0])
    assert p["p50"] == 150.0
    assert p["p95"] == 150.0
    assert p["mean"] == 150.0
    assert p["count"] == 1

def test_latency_percentiles_empty():
    p = compute_latency_percentiles([])
    assert p["count"] == 0
    assert p["p50"] == 0.0
    assert p["p95"] == 0.0

def test_cost_model_zero_local():
    cost = compute_cost_per_1k(mean_duration_sec=1.5, rate_per_sec=0.0)
    assert cost == 0.0

def test_cost_model_cloud_calculation():
    # 2.0s mean STT duration, $0.0004/sec ($0.024/min AWS Transcribe), 1000 calls
    # 2.0 * 0.0004 * 1000 = $0.80
    cost = compute_cost_per_1k(mean_duration_sec=2.0, rate_per_sec=0.0004, unit_calls=1000)
    assert pytest.approx(cost, 0.0001) == 0.80
