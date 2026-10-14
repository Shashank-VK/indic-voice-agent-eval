"""Audio synthesis module using gTTS with WAV conversion, noise injection, and offline fallback."""

import math
import os
import subprocess
import tempfile
import wave
from typing import Optional
import numpy as np
import gtts

from src.utterances import Utterance


def inject_gaussian_noise(
    audio_data: np.ndarray,
    snr_db: float = 15.0,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """Inject Gaussian noise into normalized float audio [-1.0, 1.0] at target SNR."""
    if seed is not None:
        np.random.seed(seed)

    sig_power = float(np.mean(audio_data**2))
    if sig_power <= 1e-9:
        return audio_data

    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = np.random.normal(0.0, math.sqrt(noise_power), audio_data.shape).astype(
        np.float32
    )
    noisy = audio_data + noise
    return np.clip(noisy, -1.0, 1.0)


def generate_synthetic_tone_wav(
    text: str,
    output_path: str,
    duration_sec: float = 1.5,
    sample_rate: int = 16000,
) -> str:
    """Generate a deterministic synthetic tone audio file as an offline/testing fallback."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    base_freq = 200.0 + (abs(hash(text)) % 300)
    signal = 0.5 * np.sin(2 * np.pi * base_freq * t) + 0.25 * np.sin(
        2 * np.pi * (base_freq * 1.5) * t
    )
    signal_int16 = (signal * 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(signal_int16.tobytes())

    return output_path


class AudioGenerator:
    """Synthesizes speech utterances to 16kHz mono WAV files."""

    def __init__(
        self,
        output_dir: str = "data/audio",
        sample_rate: int = 16000,
        noise_snr_db: float = 15.0,
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.noise_snr_db = noise_snr_db
        os.makedirs(self.output_dir, exist_ok=True)

    def generate(
        self,
        utterance: Utterance,
        use_cache: bool = True,
        use_fallback: bool = False,
        force: bool = False,
    ) -> str:
        """Generate a WAV file for the given utterance.

        Applies noise injection if utterance has 'background_noise' tag.
        """
        wav_path = os.path.join(self.output_dir, f"{utterance.id}.wav")

        if use_cache and not force and os.path.exists(wav_path):
            return wav_path

        audio_float: Optional[np.ndarray] = None

        if not use_fallback:
            try:
                # Use Hindi voice ('hi') for realistic Indic pronunciation
                tts_lang = "hi" if utterance.lang.startswith("hi") else "en"
                tts = gtts.gTTS(text=utterance.text, lang=tts_lang)

                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
                    mp3_path = mp3_f.name
                tts.save(mp3_path)

                # Convert MP3 to 16kHz mono WAV using ffmpeg
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    mp3_path,
                    "-ar",
                    str(self.sample_rate),
                    "-ac",
                    "1",
                    "-c:a",
                    "pcm_s16le",
                    wav_path,
                ]
                res = subprocess.run(cmd, capture_output=True, check=True)

                if os.path.exists(mp3_path):
                    os.remove(mp3_path)

                with wave.open(wav_path, "rb") as wf:
                    pcm16 = np.frombuffer(
                        wf.readframes(wf.getnframes()), dtype=np.int16
                    )
                audio_float = pcm16.astype(np.float32) / 32768.0
            except Exception:
                audio_float = None

        if audio_float is None:
            generate_synthetic_tone_wav(
                text=utterance.text,
                output_path=wav_path,
                duration_sec=2.0,
                sample_rate=self.sample_rate,
            )
            with wave.open(wav_path, "rb") as wf:
                pcm16 = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
            audio_float = pcm16.astype(np.float32) / 32768.0

        # Inject noise if tagged with background_noise
        if "background_noise" in utterance.failure_modes:
            seed = abs(hash(utterance.id)) % (2**31)
            audio_float = inject_gaussian_noise(
                audio_float, snr_db=self.noise_snr_db, seed=seed
            )

        # Write final WAV file
        pcm16_final = (audio_float * 32767.0).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm16_final.tobytes())

        return wav_path
