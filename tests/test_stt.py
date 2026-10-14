import os
import pytest
from unittest.mock import MagicMock, patch
from src.stt import SpeechToTextPipeline, STTResult

@pytest.fixture
def dummy_wav(tmp_path):
    import wave
    import numpy as np
    wav_path = str(tmp_path / "dummy.wav")
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        data = (np.sin(np.linspace(0, 100, 16000)) * 10000).astype(np.int16)
        wf.writeframes(data.tobytes())
    return wav_path

def test_stt_result_dataclass():
    res = STTResult(
        text="mera order kahan hai",
        duration_sec=0.245,
        detected_language="hi",
        language_probability=0.92
    )
    assert res.text == "mera order kahan hai"
    assert res.duration_sec == 0.245
    assert res.detected_language == "hi"
    assert res.language_probability == 0.92

def test_stt_pipeline_with_mocked_model(dummy_wav):
    mock_segment = MagicMock()
    mock_segment.text = " mera refund kab aayega"
    mock_info = MagicMock()
    mock_info.language = "hi"
    mock_info.language_probability = 0.95

    with patch("src.stt.WhisperModel") as mock_whisper_cls:
        mock_model = MagicMock()
        mock_model.transcribe.return_value = ([mock_segment], mock_info)
        mock_whisper_cls.return_value = mock_model

        pipeline = SpeechToTextPipeline(model_name="tiny", device="cpu", compute_type="int8")
        result = pipeline.transcribe_file(dummy_wav)

        assert isinstance(result, STTResult)
        assert result.text == "mera refund kab aayega"
        assert result.detected_language == "hi"
        assert result.language_probability == 0.95
        assert result.duration_sec >= 0.0
        mock_model.transcribe.assert_called_once()

def test_stt_pipeline_file_not_found():
    with patch("src.stt.WhisperModel"):
        pipeline = SpeechToTextPipeline(model_name="tiny", device="cpu")
        with pytest.raises(FileNotFoundError):
            pipeline.transcribe_file("/non/existent/path.wav")
