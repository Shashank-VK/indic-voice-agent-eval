"""Speech-to-Text pipeline using faster-whisper on local CPU."""

import os
import time
from dataclasses import dataclass
from typing import Optional
from faster_whisper import WhisperModel


@dataclass
class STTResult:
    text: str
    duration_sec: float
    detected_language: str
    language_probability: float


class SpeechToTextPipeline:
    """Wrapper around faster-whisper WhisperModel with execution timing and error handling."""

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._model: Optional[WhisperModel] = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = "hi",
    ) -> STTResult:
        """Transcribe an audio file and measure wall-clock inference duration."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        start_time = time.perf_counter()
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            beam_size=1,  # Fast greedy decoding for real-time customer support latency
        )
        transcribed_text = " ".join(s.text.strip() for s in segments).strip()
        duration_sec = time.perf_counter() - start_time

        detected_lang = getattr(info, "language", "unknown") or "unknown"
        lang_prob = float(getattr(info, "language_probability", 1.0) or 0.0)

        return STTResult(
            text=transcribed_text,
            duration_sec=duration_sec,
            detected_language=detected_lang,
            language_probability=lang_prob,
        )
