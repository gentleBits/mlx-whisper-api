"""Integration tests for transcription endpoint."""

import pytest
from unittest.mock import patch
from io import BytesIO

from app.config import DEFAULT_MODEL, SUPPORTED_MODELS
from app.services.model_manager import ModelStatus


class TestTranscribeEndpoint:
    """Tests for POST /transcribe endpoint."""

    @pytest.fixture(autouse=True)
    def mock_model_ready(self):
        """Pretend supported models are validated and ready by default."""
        with patch("app.routers.transcribe.get_model_manager") as mock_get_manager:
            manager = mock_get_manager.return_value
            manager.get_model_status.side_effect = lambda model_id: ModelStatus(
                id=model_id,
                status="downloaded",
            )
            yield manager

    @pytest.fixture
    def mock_mlx_whisper(self):
        """Mock mlx_whisper.transcribe to avoid actual model inference."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.return_value = {
                "text": " This is a test transcription.",
                "language": "en",
            }
            yield mock

    def test_transcribe_wav_file(self, client, sample_audio_path, mock_mlx_whisper):
        """Successfully transcribe a WAV file."""
        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "language" in data
        assert "model" in data
        assert data["text"] == "This is a test transcription."
        mock_mlx_whisper.transcribe.assert_called_once()

    def test_transcribe_missing_file(self, client):
        """Returns 422 when no file is provided."""
        response = client.post("/transcribe")

        assert response.status_code == 422

    def test_transcribe_large_file(self, client):
        """Returns 413 for files exceeding size limit."""
        # Create a file larger than the limit (100MB default)
        # We'll use a smaller test value by patching the config
        from app.routers import transcribe as transcribe_router

        original_max = transcribe_router.MAX_FILE_SIZE

        try:
            # Set a very small limit for testing
            transcribe_router.MAX_FILE_SIZE = 100  # 100 bytes

            large_content = b"x" * 200  # 200 bytes
            large_file = BytesIO(large_content)

            response = client.post(
                "/transcribe",
                files={"file": ("large.wav", large_file, "audio/wav")},
            )

            assert response.status_code == 413
            data = response.json()
            assert "error" in data
            assert "code" in data
            assert "too large" in data["error"].lower()

        finally:
            # Restore original limit
            transcribe_router.MAX_FILE_SIZE = original_max

    @pytest.mark.parametrize("filename,content_type", [
        ("test.mp3", "audio/mpeg"),
        ("test.m4a", "audio/mp4"),
        ("test.flac", "audio/flac"),
        ("test.ogg", "audio/ogg"),
    ])
    def test_transcribe_audio_formats(self, client, mock_mlx_whisper, filename, content_type):
        """Successfully accepts various audio formats."""
        fake_audio = BytesIO(b"fake audio content")

        response = client.post(
            "/transcribe",
            files={"file": (filename, fake_audio, content_type)},
        )

        assert response.status_code == 200

    def test_transcribe_response_schema(self, client, sample_audio_path, mock_mlx_whisper):
        """Response matches expected schema."""
        from app.config import DEFAULT_MODEL

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        assert isinstance(data["text"], str)
        assert isinstance(data["language"], str)
        assert isinstance(data["model"], str)

        # Verify model is from supported list and default is used
        assert data["model"] in SUPPORTED_MODELS
        assert data["model"] == DEFAULT_MODEL

    def test_transcribe_rejects_not_downloaded_model(self, client):
        """Returns MODEL_NOT_DOWNLOADED when model is not ready."""
        with patch("app.routers.transcribe.get_model_manager") as mock_get_manager:
            mock_get_manager.return_value.get_model_status.return_value = ModelStatus(
                id=DEFAULT_MODEL,
                status="not_downloaded",
            )
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", BytesIO(b"fake audio"), "audio/wav")},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "MODEL_NOT_DOWNLOADED"

    def test_transcribe_rejects_error_model(self, client):
        """Returns MODEL_DOWNLOAD_FAILED when model validation failed."""
        with patch("app.routers.transcribe.get_model_manager") as mock_get_manager:
            mock_get_manager.return_value.get_model_status.return_value = ModelStatus(
                id=DEFAULT_MODEL,
                status="error",
                error="Validation failed: missing weights.npz",
            )
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", BytesIO(b"fake audio"), "audio/wav")},
            )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == "MODEL_DOWNLOAD_FAILED"


class TestTranscribeWithRealModel:
    """Tests that use actual MLX Whisper model inference.

    These tests are marked as slow and require a downloaded model.
    Run with: pytest -m slow
    """

    @pytest.mark.slow
    def test_transcribe_wav_file_real(self, client, sample_audio_path):
        """Actually transcribe a WAV file with a real model.

        Note: This test requires the whisper-tiny-mlx model to be downloaded.
        The sample audio is a sine wave, so transcription results may vary.
        """
        from tests.conftest import TEST_MODEL

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": TEST_MODEL},
            )

        # The endpoint should respond (even if transcription is empty for a tone)
        assert response.status_code in [200, 400]

        if response.status_code == 200:
            data = response.json()
            assert "text" in data
            assert data["model"] == TEST_MODEL

    @pytest.mark.slow
    def test_transcribe_harvard_sentences(self, client, harvard_audio_path):
        """Transcribe real speech audio and verify output contains expected phrases.

        Uses Harvard sentences audio which contains phonetically balanced sentences.
        Verifies that key phrases are recognized in the transcription.
        """
        from tests.conftest import TEST_MODEL

        with open(harvard_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("harvard.wav", f, "audio/wav")},
                data={"model": TEST_MODEL},
            )

        assert response.status_code == 200
        data = response.json()

        assert "text" in data
        assert "language" in data
        assert "model" in data
        assert data["model"] == TEST_MODEL
        assert data["language"] == "en"

        # Verify transcription contains expected phrases from Harvard sentences
        # Using partial matches since tiny model may have minor errors
        text_lower = data["text"].lower()
        expected_phrases = [
            "birch",  # "The birch canoe slid on the smooth planks"
            "sheet",  # "Glue the sheet to the dark blue background"
            "depth",  # "It is easy to tell the depth of a well"
            "chicken",  # "These days a chicken leg is a rare dish"
            "rice",  # "Rice is often served in round bowls"
            "lemon",  # "The juice of lemons makes fine punch"
            "hogs",  # "The hogs were fed chopped corn and garbage"
        ]

        matched = [phrase for phrase in expected_phrases if phrase in text_lower]
        assert len(matched) >= 5, (
            f"Expected at least 5 of {expected_phrases} in transcription, "
            f"but only found {matched}. Full text: {data['text']}"
        )
