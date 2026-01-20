"""Unit tests for TranscriptionService."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.services.transcription import (
    TranscriptionService,
    get_transcription_service,
    TranscriptionError,
    ModelNotDownloadedError,
    UnsupportedModelError,
)
from app.config import SUPPORTED_MODELS, DEFAULT_MODEL


class TestTranscriptionService:
    """Tests for TranscriptionService class."""

    @pytest.fixture
    def service(self):
        """Create a fresh TranscriptionService instance."""
        return TranscriptionService()

    def test_is_supported_format_wav(self, service):
        """WAV format is supported."""
        assert service.is_supported_format("test.wav") is True
        assert service.is_supported_format("TEST.WAV") is True

    def test_is_supported_format_mp3(self, service):
        """MP3 format is supported."""
        assert service.is_supported_format("audio.mp3") is True

    def test_is_supported_format_m4a(self, service):
        """M4A format is supported."""
        assert service.is_supported_format("recording.m4a") is True

    def test_is_supported_format_flac(self, service):
        """FLAC format is supported."""
        assert service.is_supported_format("music.flac") is True

    def test_is_supported_format_ogg(self, service):
        """OGG format is supported."""
        assert service.is_supported_format("voice.ogg") is True

    def test_is_supported_format_unsupported(self, service):
        """Unsupported formats return False."""
        assert service.is_supported_format("doc.txt") is False
        assert service.is_supported_format("video.mp4") is False
        assert service.is_supported_format("image.png") is False
        assert service.is_supported_format("no_extension") is False

    def test_validate_model_supported(self, service):
        """Valid model IDs pass validation."""
        # Should not raise
        service.validate_model("mlx-community/whisper-tiny-mlx")
        service.validate_model("mlx-community/whisper-large-v3-mlx")

    def test_validate_model_unsupported(self, service):
        """Invalid model IDs raise UnsupportedModelError."""
        with pytest.raises(UnsupportedModelError) as exc_info:
            service.validate_model("not-a-real/model")

        assert exc_info.value.model_id == "not-a-real/model"

    def test_transcribe_uses_default_model(self, service, sample_audio_path):
        """Transcribe uses default model when none specified."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {"text": "test", "language": "en"}

            result = service.transcribe(str(sample_audio_path))

            # Verify default model was used
            call_kwargs = mock.transcribe.call_args[1]
            assert call_kwargs["path_or_hf_repo"] == DEFAULT_MODEL

    def test_transcribe_with_language(self, service, sample_audio_path):
        """Language parameter is passed to mlx_whisper."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {"text": "test", "language": "fr"}

            result = service.transcribe(str(sample_audio_path), language="fr")

            call_kwargs = mock.transcribe.call_args[1]
            assert call_kwargs["language"] == "fr"

    def test_transcribe_with_prompt(self, service, sample_audio_path):
        """Prompt is passed as initial_prompt to mlx_whisper."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {"text": "test", "language": "en"}

            prompt = "Technical discussion about Python"
            result = service.transcribe(str(sample_audio_path), prompt=prompt)

            call_kwargs = mock.transcribe.call_args[1]
            assert call_kwargs["initial_prompt"] == prompt

    def test_transcribe_returns_correct_structure(self, service, sample_audio_path):
        """Transcribe returns dict with text, language, model."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {"text": " Hello world.", "language": "en"}

            result = service.transcribe(str(sample_audio_path))

            assert "text" in result
            assert "language" in result
            assert "model" in result
            assert result["text"] == "Hello world."  # Stripped
            assert result["language"] == "en"
            assert result["model"] == DEFAULT_MODEL

    def test_transcribe_strips_whitespace(self, service, sample_audio_path):
        """Transcription text is stripped of whitespace."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {"text": "  Padded text  ", "language": "en"}

            result = service.transcribe(str(sample_audio_path))

            assert result["text"] == "Padded text"

    def test_transcribe_model_not_found_error(self, service, sample_audio_path):
        """Raises ModelNotDownloadedError when model is not found."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.side_effect = Exception("Model not found in cache")

            with pytest.raises(ModelNotDownloadedError) as exc_info:
                service.transcribe(str(sample_audio_path))

            assert exc_info.value.model_id == DEFAULT_MODEL

    def test_transcribe_generic_error(self, service, sample_audio_path):
        """Raises TranscriptionError for other exceptions."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.side_effect = Exception("Some random error")

            with pytest.raises(TranscriptionError):
                service.transcribe(str(sample_audio_path))


class TestGetTranscriptionService:
    """Tests for get_transcription_service singleton."""

    def test_returns_same_instance(self):
        """Singleton returns the same instance."""
        # Reset the singleton for testing
        import app.services.transcription as module

        module._service = None

        service1 = get_transcription_service()
        service2 = get_transcription_service()

        assert service1 is service2

    def test_returns_transcription_service(self):
        """Returns a TranscriptionService instance."""
        service = get_transcription_service()
        assert isinstance(service, TranscriptionService)
