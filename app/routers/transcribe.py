"""Transcription endpoint router.

Provides the POST /transcribe endpoint for audio-to-text transcription.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Request

from app.config import DEFAULT_MODEL, MAX_AUDIO_SIZE_MB
from app.schemas.models import TranscriptionResponse, ErrorResponse
from app.errors import (
    APIException,
    ErrorCode,
    EmptyFileError,
    FileTooLargeError,
    ModelNotDownloadedError as APIModelNotDownloadedError,
    ModelUnsupportedError,
    TranscriptionFailedError,
    UnsupportedFormatError,
)
from app.validation import (
    validate_language,
    validate_prompt,
    validate_audio_format,
    sanitize_filename,
    SUPPORTED_AUDIO_FORMATS,
    MAX_PROMPT_LENGTH,
)
from app.services.transcription import (
    get_transcription_service,
    TranscriptionError,
    ModelNotDownloadedError,
    UnsupportedModelError,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Max file size in bytes
MAX_FILE_SIZE = MAX_AUDIO_SIZE_MB * 1024 * 1024


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe audio to text",
    description=f"""
Transcribe an audio file to text using MLX-optimized Whisper models.

### Supported Formats
WAV, MP3, M4A, FLAC, OGG

### Parameters
- **file**: Audio file to transcribe (required)
- **model**: Model ID from supported list (default: {DEFAULT_MODEL})
- **language**: Two-letter ISO 639-1 language code for transcription
- **prompt**: Text prompt to guide transcription (max {MAX_PROMPT_LENGTH} chars)

### Notes
- Maximum file size: {MAX_AUDIO_SIZE_MB}MB
- If model is not downloaded, response includes download URL
""",
    responses={
        200: {
            "description": "Successful transcription",
            "content": {
                "application/json": {
                    "example": {
                        "text": "Hello, this is a test transcription.",
                        "language": "en",
                        "model": "mlx-community/whisper-large-v3-mlx"
                    }
                }
            }
        },
        400: {
            "model": ErrorResponse,
            "description": "Invalid request - unsupported format, invalid model, or transcription error",
            "content": {
                "application/json": {
                    "examples": {
                        "unsupported_format": {
                            "summary": "Unsupported audio format",
                            "value": {
                                "error": "Unsupported audio format: .txt",
                                "code": "VALIDATION_UNSUPPORTED_FORMAT",
                                "details": {
                                    "format": ".txt",
                                    "supported_formats": [".flac", ".m4a", ".mp3", ".ogg", ".wav"]
                                }
                            }
                        },
                        "model_not_downloaded": {
                            "summary": "Model not downloaded",
                            "value": {
                                "error": "Model not downloaded: mlx-community/whisper-tiny-mlx",
                                "code": "MODEL_NOT_DOWNLOADED",
                                "details": {
                                    "model": "mlx-community/whisper-tiny-mlx",
                                    "download_url": "/models/mlx-community%2Fwhisper-tiny-mlx/download"
                                }
                            }
                        },
                        "invalid_language": {
                            "summary": "Invalid language code",
                            "value": {
                                "error": "Invalid language code: xx. Must be a two-letter ISO 639-1 code.",
                                "code": "VALIDATION_INVALID_LANGUAGE",
                                "details": {"language": "xx"}
                            }
                        }
                    }
                }
            }
        },
        413: {
            "model": ErrorResponse,
            "description": "File too large",
            "content": {
                "application/json": {
                    "example": {
                        "error": f"File too large. Maximum size is {MAX_AUDIO_SIZE_MB}MB",
                        "code": "VALIDATION_FILE_TOO_LARGE",
                        "details": {
                            "file_size_bytes": 150000000,
                            "max_size_bytes": MAX_FILE_SIZE
                        }
                    }
                }
            }
        },
        422: {
            "description": "Validation error (missing required file)"
        },
    },
)
async def transcribe_audio(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe (WAV, MP3, M4A, FLAC, OGG)"),
    model: Optional[str] = Form(
        default=None,
        description=f"Model identifier from supported list. Default: {DEFAULT_MODEL}",
    ),
    language: Optional[str] = Form(
        default=None,
        description="Two-letter ISO 639-1 language code (e.g., 'en', 'fr', 'de'). Auto-detected if not specified.",
    ),
    prompt: Optional[str] = Form(
        default=None,
        description=f"Initial prompt to guide transcription. Useful for providing context, terminology, or speaker names. Max {MAX_PROMPT_LENGTH} characters.",
    ),
) -> TranscriptionResponse:
    """Transcribe an audio file to text.

    Accepts audio files in various formats and returns the transcribed text
    using the specified MLX Whisper model.

    The transcription process:
    1. Validates the audio format and file size
    2. Validates optional parameters (language, prompt)
    3. Processes the audio through the Whisper model
    4. Returns the transcribed text with metadata

    Args:
        request: FastAPI request object (for request ID tracking)
        file: Audio file to transcribe (multipart/form-data)
        model: Model identifier from supported list
        language: Two-letter ISO 639-1 language code
        prompt: Text prompt to provide context for transcription

    Returns:
        TranscriptionResponse with transcribed text, detected language, and model used

    Raises:
        UnsupportedFormatError: If audio format is not supported
        FileTooLargeError: If file exceeds size limit
        EmptyFileError: If file is empty
        InvalidLanguageError: If language code is invalid
        PromptTooLongError: If prompt exceeds length limit
        ModelUnsupportedError: If model is not in supported list
        ModelNotDownloadedError: If model is not downloaded
        TranscriptionFailedError: If transcription fails
    """
    request_id = getattr(request.state, "request_id", None)
    service = get_transcription_service()

    # Sanitize and validate filename
    filename = sanitize_filename(file.filename)

    # Validate audio format
    validate_audio_format(filename)

    # Validate optional parameters
    validated_language = validate_language(language)
    validated_prompt = validate_prompt(prompt)

    # Read file content
    content = await file.read()

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise FileTooLargeError(len(content), MAX_FILE_SIZE)

    # Check for empty file
    if len(content) == 0:
        raise EmptyFileError()

    logger.info(
        "Transcribing audio: filename=%s, size=%d bytes, model=%s, language=%s (request_id=%s)",
        filename,
        len(content),
        model or DEFAULT_MODEL,
        validated_language or "auto",
        request_id,
    )

    try:
        result = await service.transcribe_upload(
            file_content=content,
            filename=filename,
            model=model,
            language=validated_language,
            prompt=validated_prompt,
        )

        logger.info(
            "Transcription complete: %d chars, language=%s (request_id=%s)",
            len(result["text"]),
            result["language"],
            request_id,
        )

        return TranscriptionResponse(**result)

    except UnsupportedModelError as e:
        logger.warning(
            "Unsupported model: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise ModelUnsupportedError(e.model_id)

    except ModelNotDownloadedError as e:
        logger.warning(
            "Model not downloaded: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        # URL-encode the model ID for the download URL
        encoded_model_id = e.model_id.replace("/", "%2F")
        raise APIModelNotDownloadedError(
            e.model_id,
            download_url=f"/models/{encoded_model_id}/download",
        )

    except TranscriptionError as e:
        logger.error(
            "Transcription failed: %s (request_id=%s)",
            str(e),
            request_id,
        )
        raise TranscriptionFailedError(str(e))
