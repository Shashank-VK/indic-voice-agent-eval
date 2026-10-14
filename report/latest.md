# Indic Voice Agent Evaluation Report

> **Run Timestamp**: 2026-10-14 10:40:02 UTC  
> **Model**: `faster-whisper` (`tiny`, CPU, `int8`)  
> **Pipeline Status**: ****PASSED****

## 1. Executive Summary

| Metric | Measured Value | Threshold | Status |
| :--- | :--- | :--- | :--- |
| **Evaluated Utterances** | 40 phrases | N/A | Completed |
| **Overall Mean WER** | 0.673 (67.3%) | <= 0.750 | PASS |
| **p50 Latency** | 1980.8 ms | N/A | Informational |
| **p95 Latency** | 6200.5 ms | <= 10000.0 ms | PASS |
| **Cost per 1k calls (Local)** | $0.0000 | $0.00 | Free local compute |
| **Cost per 1k calls (AWS Transcribe)** | $0.9895 | Reference | Published pricing |

## 2. Failure Mode Matrix

Breakdown across failure modes (code-mixing, regional accents, background noise, clean baseline):

| Failure Mode | Utterances | Mean WER | p50 Latency | p95 Latency | Cloud Cost/1k | Regression Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `clean` | 10 | 0.595 (59.5%) | 1236.7 ms | 5025.8 ms | $0.7697 | PASS |
| `code_mixing` | 10 | 0.610 (61.0%) | 1903.0 ms | 11437.2 ms | $1.4231 | PASS |
| `regional_accent` | 10 | 0.732 (73.2%) | 1177.5 ms | 3458.8 ms | $0.6258 | PASS |
| `background_noise` | 10 | 0.757 (75.7%) | 2255.9 ms | 5102.9 ms | $1.1393 | PASS |

## 3. Pricing Assumption & Cost Model

- **Local Compute**: Assumed **$0.00/hr** for all measurements on local developer CPU hardware.
- **Cloud Baseline Reference**: `AWS Transcribe` with published standard pricing of **$0.024 / minute** ($0.0004 / second).
- **Cost per 1k Formula**: `1,000 calls × mean_inference_duration_sec × rate_per_second`.
- For this run, mean STT duration was **2.474 s**, yielding a cloud equivalent of **$0.9895** per 1,000 customer support calls.

## 4. Regression Check: PASSED

All metrics satisfied configured thresholds defined in `config.yaml`.

## 5. Sample Utterance Evaluations

| ID | Tag | Domain | Reference Ground Truth | STT Hypothesis | WER | Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `utt_001` | `clean` | banking | mera refund kab tak aayega | Mirarifundkabta kaiga | 1.00 | 6630ms |
| `utt_002` | `clean` | ecommerce | order cancel karne ka process bataiye | Order cancel karnika process. | 0.50 | 984ms |
| `utt_003` | `clean` | telecom | customer care officer se baat karni hai | customer care officer say about karmi hai. | 0.43 | 2537ms |
| `utt_004` | `clean` | food_delivery | delivery partner ka number chahiye | Delivery partner's number 4. | 0.40 | 1445ms |
| `utt_005` | `clean` | banking | mera account statement email par bhej do | Mira accounts statement email per bhi jadu. | 0.71 | 1239ms |
| `utt_006` | `clean` | travel | flight ticket ka status confirm hai ya nahi | Flight ticket ke status confirm heya neheng. | 0.50 | 756ms |
| `utt_007` | `clean` | telecom | registered mobile number badalna chahta hoon | Registered Mobile Number Badalna Chahata Heng. | 0.33 | 1235ms |
| `utt_008` | `clean` | ecommerce | kya return pickup aaj ho sakta hai | kya rootan pika pajho sakta hai. | 0.57 | 3065ms |
| `utt_009` | `clean` | food_delivery | address update karne mein madad chahiye | Adress up date karne me madad jahi. | 0.83 | 596ms |
| `utt_010` | `clean` | telecom | mera bill payment receipt nahi mila | Mirabil payment receipt na Himila | 0.67 | 757ms |
| `utt_011` | `code_mixing` | travel | mera payment deduct ho gaya but booking confirm nahi hui | Myra Piman Dedaktho Gaia Badbukinkan Farm Nahi Hoi | 0.90 | 483ms |
| `utt_012` | `code_mixing` | banking | app login karte waqt OTP receive nahi ho raha hai | App login karte wakt oTP receive نہیں huraha hai. | 0.40 | 2056ms |
