"""MLX Whisper transcription service."""

import tempfile
import os
from pathlib import Path
from typing import Optional

import mlx_whisper

from app.config import DEFAULT_MODEL, SUPPORTED_MODELS


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""

    pass


class ModelNotDownloadedError(Exception):
    """Exception raised when model is not downloaded."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is not downloaded")


class UnsupportedModelError(Exception):
    """Exception raised for unsupported model IDs."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is not supported")


class TranscriptionService:
    """Service for transcribing audio using MLX Whisper."""

    # Supported audio formats
    SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

    def __init__(self):
        """Initialize the transcription service."""
        pass

    def is_supported_format(self, filename: str) -> bool:
        """Check if the audio format is supported."""
        ext = Path(filename).suffix.lower()
        return ext in self.SUPPORTED_FORMATS

    def validate_model(self, model_id: str) -> None:
        """Validate that the model ID is supported.

        Args:
            model_id: The model identifier to validate

        Raises:
            UnsupportedModelError: If the model is not in the supported list
        """
        if model_id not in SUPPORTED_MODELS:
            raise UnsupportedModelError(model_id)

    def transcribe(
        self,
        audio_path: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Transcribe audio file to text.

        Args:
            audio_path: Path to the audio file
            model: Model identifier (defaults to DEFAULT_MODEL)
            language: Two-letter language code (e.g., "en", "fr")
            prompt: Initial prompt to guide transcription

        Returns:
            dict with keys: text, language, model

        Raises:
            UnsupportedModelError: If the model is not supported
            ModelNotDownloadedError: If the model is not downloaded
            TranscriptionError: If transcription fails
        """
        model_id = model or DEFAULT_MODEL

        # Validate model is supported
        self.validate_model(model_id)

        # Build transcription options
        transcribe_options = {}
        if language:
            transcribe_options["language"] = language
        if prompt:
            transcribe_options["initial_prompt"] = prompt

        try:
            # Call mlx_whisper transcribe
            result = mlx_whisper.transcribe(
                audio_path,
                path_or_hf_repo=model_id,
                **transcribe_options,
            )

            # Extract detected language from result
            detected_language = result.get("language", language or "unknown")

            return {
                "text": result["text"].strip(),
                "language": detected_language,
                "model": model_id,
            }

        except FileNotFoundError:
            raise TranscriptionError(f"Audio file not found: {audio_path}")
        except Exception as e:
            error_msg = str(e).lower()
            # Check if the error indicates model not downloaded
            if "not found" in error_msg and "model" in error_msg:
                raise ModelNotDownloadedError(model_id)
            if "no such file or directory" in error_msg:
                raise ModelNotDownloadedError(model_id)
            raise TranscriptionError(f"Transcription failed: {e}")

    async def transcribe_upload(
        self,
        file_content: bytes,
        filename: str,
        model: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> dict:
        """Transcribe uploaded audio content.

        Args:
            file_content: Raw bytes of the audio file
            filename: Original filename (used to detect format)
            model: Model identifier
            language: Two-letter language code
            prompt: Initial prompt to guide transcription

        Returns:
            dict with keys: text, language, model
        """
        # Get the file extension
        ext = Path(filename).suffix.lower()
        if not ext:
            ext = ".wav"  # Default to wav if no extension

        # Write to a temporary file
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name

        try:
            return self.transcribe(
                audio_path=tmp_path,
                model=model,
                language=language,
                prompt=prompt,
            )
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Singleton instance
_service: Optional[TranscriptionService] = None


def get_transcription_service() -> TranscriptionService:
    """Get the transcription service singleton."""
    global _service
    if _service is None:
        _service = TranscriptionService()
    return _service
