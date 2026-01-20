"""Integration tests for transcription endpoint."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from io import BytesIO

from app.config import SUPPORTED_MODELS


class TestTranscribeEndpoint:
    """Tests for POST /transcribe endpoint."""

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

    def test_transcribe_with_language(self, client, sample_audio_path, mock_mlx_whisper):
        """Transcribe with explicit language parameter."""
        mock_mlx_whisper.transcribe.return_value = {
            "text": " Hello world.",
            "language": "en",
        }

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"language": "en"},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "en"

        # Verify language was passed to transcribe
        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("language") == "en"

    def test_transcribe_with_prompt(self, client, sample_audio_path, mock_mlx_whisper):
        """Transcribe with initial prompt for context."""
        prompt = "This is a medical transcription about diabetes."

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"prompt": prompt},
            )

        assert response.status_code == 200

        # Verify prompt was passed as initial_prompt
        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("initial_prompt") == prompt

    def test_transcribe_with_model(self, client, sample_audio_path, mock_mlx_whisper):
        """Transcribe specifying a specific model."""
        model = "mlx-community/whisper-tiny-mlx"

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": model},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == model

        # Verify model was passed to transcribe
        call_kwargs = mock_mlx_whisper.transcribe.call_args[1]
        assert call_kwargs.get("path_or_hf_repo") == model

    def test_transcribe_missing_file(self, client):
        """Returns 422 when no file is provided."""
        response = client.post("/transcribe")

        assert response.status_code == 422

    def test_transcribe_invalid_model(self, client, sample_audio_path):
        """Returns 400 for unsupported model."""
        invalid_model = "not-a-real/model"

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
                data={"model": invalid_model},
            )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "Unsupported model" in data["error"]
        assert data["details"]["model"] == invalid_model

    def test_transcribe_model_not_downloaded(self, client, sample_audio_path):
        """Returns 400 with download instructions when model is not downloaded."""
        # Use a valid model ID but mock the transcription to raise an error
        model = "mlx-community/whisper-tiny-mlx"

        with patch("app.services.transcription.mlx_whisper") as mock:
            mock.transcribe.side_effect = Exception(
                "Model mlx-community/whisper-tiny-mlx not found"
            )

            with open(sample_audio_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"model": model},
                )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "not downloaded" in data["error"].lower()
        assert data["details"]["model"] == model
        assert "download_url" in data["details"]

    def test_transcribe_empty_audio(self, client):
        """Returns 400 for empty audio file."""
        # Create an empty file
        empty_file = BytesIO(b"")

        response = client.post(
            "/transcribe",
            files={"file": ("empty.wav", empty_file, "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "empty" in data["error"].lower()

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

    def test_transcribe_unsupported_format(self, client):
        """Returns 400 for unsupported audio format."""
        # Create a fake file with unsupported extension
        fake_file = BytesIO(b"fake content")

        response = client.post(
            "/transcribe",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "unsupported" in data["error"].lower()
        assert "supported_formats" in data["details"]

    def test_transcribe_mp3_file(self, client, mock_mlx_whisper):
        """Successfully accepts MP3 file format."""
        # Create a minimal MP3-like file (just for format validation)
        fake_mp3 = BytesIO(b"fake mp3 content")

        with patch.object(
            mock_mlx_whisper, "transcribe", return_value={"text": " Test.", "language": "en"}
        ):
            response = client.post(
                "/transcribe",
                files={"file": ("test.mp3", fake_mp3, "audio/mpeg")},
            )

        # Should accept the format (actual transcription may fail with fake content)
        # but the format validation should pass
        assert response.status_code == 200

    def test_transcribe_m4a_file(self, client, mock_mlx_whisper):
        """Successfully accepts M4A file format."""
        fake_m4a = BytesIO(b"fake m4a content")

        response = client.post(
            "/transcribe",
            files={"file": ("test.m4a", fake_m4a, "audio/mp4")},
        )

        assert response.status_code == 200

    def test_transcribe_flac_file(self, client, mock_mlx_whisper):
        """Successfully accepts FLAC file format."""
        fake_flac = BytesIO(b"fake flac content")

        response = client.post(
            "/transcribe",
            files={"file": ("test.flac", fake_flac, "audio/flac")},
        )

        assert response.status_code == 200

    def test_transcribe_ogg_file(self, client, mock_mlx_whisper):
        """Successfully accepts OGG file format."""
        fake_ogg = BytesIO(b"fake ogg content")

        response = client.post(
            "/transcribe",
            files={"file": ("test.ogg", fake_ogg, "audio/ogg")},
        )

        assert response.status_code == 200

    def test_transcribe_response_schema(self, client, sample_audio_path, mock_mlx_whisper):
        """Response matches expected schema."""
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

        # Verify model is from supported list
        assert data["model"] in SUPPORTED_MODELS

    def test_transcribe_default_model(self, client, sample_audio_path, mock_mlx_whisper):
        """Uses default model when none specified."""
        from app.config import DEFAULT_MODEL

        with open(sample_audio_path, "rb") as f:
            response = client.post(
                "/transcribe",
                files={"file": ("test.wav", f, "audio/wav")},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == DEFAULT_MODEL

    def test_transcribe_strips_whitespace(self, client, sample_audio_path):
        """Transcription text is stripped of leading/trailing whitespace."""
        with patch("app.services.transcription.mlx_whisper") as mock:
            # Whisper often returns text with leading space
            mock.transcribe.return_value = {
                "text": "   Hello world.   ",
                "language": "en",
            }

            with open(sample_audio_path, "rb") as f:
                response = client.post(
                    "/transcribe",
                    files={"file": ("test.wav", f, "audio/wav")},
                )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Hello world."  # Stripped


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
