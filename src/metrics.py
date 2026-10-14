"""Evaluation metrics: Word Error Rate (WER), Latency Percentiles, and Cost Model."""

import re
from typing import Dict, List
import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text for speech recognition evaluation.

    - Convert to lowercase
    - Replace punctuation and non-alphanumeric symbols with spaces
    - Collapse redundant whitespace
    """
    if not text:
        return ""
    # Replace punctuation characters with whitespace
    cleaned = re.sub(r"[^\w\s]", " ", text.lower(), flags=re.UNICODE)
    # Collapse multiple spaces into one and strip
    return " ".join(cleaned.split())


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER = (S + D + I) / N) between reference and hypothesis.

    Calculated on normalized text.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm and not hyp_norm:
        return 0.0
    if not ref_norm and hyp_norm:
        return 1.0

    ref_words = ref_norm.split()
    hyp_words = hyp_norm.split()

    n = len(ref_words)
    m = len(hyp_words)

    # Dynamic programming matrix for Levenshtein word distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                substitution = dp[i - 1][j - 1] + 1
                insertion = dp[i][j - 1] + 1
                deletion = dp[i - 1][j] + 1
                dp[i][j] = min(substitution, insertion, deletion)

    edit_distance = dp[n][m]
    return float(edit_distance / n)


def compute_latency_percentiles(latencies_ms: List[float]) -> Dict[str, float]:
    """Compute latency statistics (min, mean, p50, p90, p95, p99, max) in milliseconds."""
    if not latencies_ms:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def compute_cost_per_1k(
    mean_duration_sec: float,
    rate_per_sec: float,
    unit_calls: int = 1000,
) -> float:
    """Calculate cost for a unit number of calls (default 1,000) given mean duration and rate.

    Cost = unit_calls * mean_duration_sec * rate_per_sec
    """
    if rate_per_sec <= 0.0 or mean_duration_sec <= 0.0:
        return 0.0
    return float(round(unit_calls * mean_duration_sec * rate_per_sec, 4))
