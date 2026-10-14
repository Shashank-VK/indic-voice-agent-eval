"""Indic Voice Agent Eval Harness.

Measures WER, p50/p95 latency, cost per 1k calls, and failure modes
(code-mixing, regional accents, background noise) across Hindi/Hinglish customer support queries.
"""

import os

# Sanitize NO_PROXY and no_proxy to avoid httpx bracketed ipv6 parse errors
for _var in ("no_proxy", "NO_PROXY"):
    _val = os.environ.get(_var)
    if _val:
        _cleaned = [
            _p.strip()
            for _p in _val.split(",")
            if "[" not in _p and "]" not in _p and not _p.strip().startswith("::")
        ]
        os.environ[_var] = ",".join(_cleaned)

__version__ = "0.1.0"
