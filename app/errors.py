"""Standardized error handling for the MLX Whisper API.

This module provides:
- Machine-readable error codes for all API errors
- Consistent error response structure
- Exception handlers for FastAPI
"""

import logging
from enum import Enum
from typing import Optional, Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Machine-readable error codes for API errors.

    Error codes follow the pattern: CATEGORY_SPECIFIC_ERROR
    Categories:
    - VALIDATION: Input validation errors
    - MODEL: Model-related errors
    - TRANSCRIPTION: Transcription processing errors
    - SERVER: Internal server errors
    """
    # Validation errors (400)
    VALIDATION_UNSUPPORTED_FORMAT = "VALIDATION_UNSUPPORTED_FORMAT"
    VALIDATION_EMPTY_FILE = "VALIDATION_EMPTY_FILE"
    VALIDATION_FILE_TOO_LARGE = "VALIDATION_FILE_TOO_LARGE"
    VALIDATION_INVALID_LANGUAGE = "VALIDATION_INVALID_LANGUAGE"
    VALIDATION_PROMPT_TOO_LONG = "VALIDATION_PROMPT_TOO_LONG"
    VALIDATION_INVALID_MODEL_ID = "VALIDATION_INVALID_MODEL_ID"

    # Model errors (400, 404, 409)
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_NOT_DOWNLOADED = "MODEL_NOT_DOWNLOADED"
    MODEL_ALREADY_DOWNLOADED = "MODEL_ALREADY_DOWNLOADED"
    MODEL_UNSUPPORTED = "MODEL_UNSUPPORTED"
    MODEL_DOWNLOAD_FAILED = "MODEL_DOWNLOAD_FAILED"

    # Transcription errors (400, 500)
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    TRANSCRIPTION_AUDIO_INVALID = "TRANSCRIPTION_AUDIO_INVALID"

    # Server errors (500)
    SERVER_INTERNAL_ERROR = "SERVER_INTERNAL_ERROR"
    SERVER_UNAVAILABLE = "SERVER_UNAVAILABLE"


class APIError(BaseModel):
    """Standardized API error response.

    All error responses follow this structure for consistency and
    machine-readability.
    """
    error: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Model not downloaded"]
    )
    code: ErrorCode = Field(
        ...,
        description="Machine-readable error code",
        examples=[ErrorCode.MODEL_NOT_DOWNLOADED]
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error context",
        examples=[{"model": "mlx-community/whisper-tiny-mlx", "download_url": "/models/mlx-community%2Fwhisper-tiny-mlx/download"}]
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking and debugging"
    )


class APIException(Exception):
    """Base exception for API errors.

    Provides a standard way to raise errors that will be converted
    to consistent JSON responses.
    """

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details
        super().__init__(message)

    def to_response(self, request_id: Optional[str] = None) -> APIError:
        """Convert exception to API error response."""
        return APIError(
            error=self.message,
            code=self.code,
            details=self.details,
            request_id=request_id,
        )


# Pre-defined exception classes for common errors
class ValidationError(APIException):
    """Exception for input validation errors."""

    def __init__(
        self,
        message: str,
        code: ErrorCode,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            code=code,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class UnsupportedFormatError(ValidationError):
    """Exception for unsupported audio format."""

    def __init__(self, format: str, supported_formats: list[str]):
        super().__init__(
            message=f"Unsupported audio format: {format}",
            code=ErrorCode.VALIDATION_UNSUPPORTED_FORMAT,
            details={"format": format, "supported_formats": supported_formats},
        )


class FileTooLargeError(APIException):
    """Exception for files exceeding size limit."""

    def __init__(self, file_size: int, max_size: int):
        super().__init__(
            message=f"File too large. Maximum size is {max_size // (1024 * 1024)}MB",
            code=ErrorCode.VALIDATION_FILE_TOO_LARGE,
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            details={"file_size_bytes": file_size, "max_size_bytes": max_size},
        )


class EmptyFileError(ValidationError):
    """Exception for empty files."""

    def __init__(self):
        super().__init__(
            message="Empty audio file",
            code=ErrorCode.VALIDATION_EMPTY_FILE,
        )


class InvalidLanguageError(ValidationError):
    """Exception for invalid language codes."""

    def __init__(self, language: str):
        super().__init__(
            message=f"Invalid language code: {language}. Must be a two-letter ISO 639-1 code.",
            code=ErrorCode.VALIDATION_INVALID_LANGUAGE,
            details={"language": language},
        )


class PromptTooLongError(ValidationError):
    """Exception for prompts exceeding length limit."""

    def __init__(self, length: int, max_length: int):
        super().__init__(
            message=f"Prompt too long. Maximum length is {max_length} characters.",
            code=ErrorCode.VALIDATION_PROMPT_TOO_LONG,
            details={"length": length, "max_length": max_length},
        )


class ModelNotFoundError(APIException):
    """Exception when model is not in supported list."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model not found: {model_id}",
            code=ErrorCode.MODEL_NOT_FOUND,
            status_code=status.HTTP_404_NOT_FOUND,
            details={"model": model_id},
        )


class ModelNotDownloadedError(APIException):
    """Exception when model is not downloaded."""

    def __init__(self, model_id: str, download_url: Optional[str] = None):
        details = {"model": model_id}
        if download_url:
            details["download_url"] = download_url
        super().__init__(
            message=f"Model not downloaded: {model_id}",
            code=ErrorCode.MODEL_NOT_DOWNLOADED,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


class ModelAlreadyDownloadedError(APIException):
    """Exception when model is already downloaded."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model already downloaded: {model_id}",
            code=ErrorCode.MODEL_ALREADY_DOWNLOADED,
            status_code=status.HTTP_409_CONFLICT,
            details={"model": model_id},
        )


class ModelUnsupportedError(APIException):
    """Exception for unsupported model IDs."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Unsupported model: {model_id}",
            code=ErrorCode.MODEL_UNSUPPORTED,
            status_code=status.HTTP_400_BAD_REQUEST,
            details={"model": model_id},
        )


class TranscriptionFailedError(APIException):
    """Exception when transcription fails."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        super().__init__(
            message=message,
            code=ErrorCode.TRANSCRIPTION_FAILED,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
        )


async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
    """Handle APIException and return standardized JSON response."""
    request_id = getattr(request.state, "request_id", None)

    logger.warning(
        "API error: %s (code=%s, status=%d, request_id=%s)",
        exc.message,
        exc.code.value,
        exc.status_code,
        request_id,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response(request_id=request_id).model_dump(exclude_none=True),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions with a generic error response."""
    request_id = getattr(request.state, "request_id", None)

    logger.exception(
        "Unhandled exception (request_id=%s): %s",
        request_id,
        str(exc),
    )

    error = APIError(
        error="An internal server error occurred",
        code=ErrorCode.SERVER_INTERNAL_ERROR,
        request_id=request_id,
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error.model_dump(exclude_none=True),
    )
