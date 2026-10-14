import os
import wave
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.utterances import Utterance
from src.audio_gen import (
    AudioGenerator,
    inject_gaussian_noise,
    generate_synthetic_tone_wav,
)

@pytest.fixture
def temp_audio_dir(tmp_path):
    d = tmp_path / "audio"
    d.mkdir()
    return d

def test_inject_gaussian_noise():
    # Constant signal
    signal = np.ones(16000, dtype=np.float32) * 0.5
    noisy = inject_gaussian_noise(signal, snr_db=15.0, seed=42)
    assert noisy.shape == signal.shape
    # Noise should change standard deviation
    assert np.std(noisy) > np.std(signal)
    # Clipping should prevent out-of-bounds
    assert np.all(noisy >= -1.0) and np.all(noisy <= 1.0)

def test_generate_synthetic_tone_wav(temp_audio_dir):
    wav_path = str(temp_audio_dir / "tone.wav")
    generate_synthetic_tone_wav("Test phrase", wav_path, duration_sec=1.0, sample_rate=16000)
    assert os.path.exists(wav_path)
    with wave.open(wav_path, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000
        assert wf.getnframes() == 16000

def test_audio_generator_with_mocked_gtts(temp_audio_dir):
    gen = AudioGenerator(output_dir=str(temp_audio_dir), sample_rate=16000)
    utt = Utterance(
        id="test_01",
        text="Mera refund nahi aaya",
        lang="hi",
        domain="banking",
        failure_modes=["clean"]
    )
    
    # Mock gTTS to write dummy mp3
    with patch("gtts.gTTS") as mock_gtts_cls:
        mock_instance = MagicMock()
        def fake_write(fp):
            # Write a minimal dummy stream or trigger fallback
            fp.write(b"fake_mp3_data")
        mock_instance.write_to_fp.side_effect = fake_write
        mock_gtts_cls.return_value = mock_instance

        # Even if mp3 decode fails on fake data, fallback should produce a valid wav
        wav_path = gen.generate(utt)
        assert os.path.exists(wav_path)
        assert os.path.getsize(wav_path) > 0
        with wave.open(wav_path, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000

def test_audio_generator_caching(temp_audio_dir):
    gen = AudioGenerator(output_dir=str(temp_audio_dir))
    utt = Utterance(
        id="test_cache",
        text="Order status kya hai",
        lang="hi",
        domain="ecommerce",
        failure_modes=["clean"]
    )
    # Generate once
    path1 = gen.generate(utt, use_fallback=True)
    mtime1 = os.path.getmtime(path1)

    # Generate again with use_cache=True
    path2 = gen.generate(utt, use_cache=True)
    mtime2 = os.path.getmtime(path2)

    assert path1 == path2
    assert mtime1 == mtime2

def test_audio_generator_noise_injection_on_tag(temp_audio_dir):
    gen = AudioGenerator(output_dir=str(temp_audio_dir))
    utt_clean = Utterance(
        id="test_clean",
        text="Call center help",
        lang="hi",
        domain="support",
        failure_modes=["clean"]
    )
    utt_noisy = Utterance(
        id="test_noisy",
        text="Call center help",
        lang="hi",
        domain="support",
        failure_modes=["background_noise"]
    )
    path_clean = gen.generate(utt_clean, use_fallback=True)
    path_noisy = gen.generate(utt_noisy, use_fallback=True)

    with wave.open(path_clean, "rb") as wf:
        data_clean = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    with wave.open(path_noisy, "rb") as wf:
        data_noisy = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)

    # Standard deviation of noisy audio should be higher due to injected noise
    assert np.std(data_noisy) != np.std(data_clean)
