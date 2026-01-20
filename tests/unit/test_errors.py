"""Unit tests for error handling module."""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import status

from app.errors import (
    ErrorCode,
    APIError,
    APIException,
    ValidationError,
    UnsupportedFormatError,
    FileTooLargeError,
    EmptyFileError,
    InvalidLanguageError,
    PromptTooLongError,
    ModelNotFoundError,
    ModelNotDownloadedError,
    ModelAlreadyDownloadedError,
    ModelUnsupportedError,
    TranscriptionFailedError,
    api_exception_handler,
    unhandled_exception_handler,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_codes_are_strings(self):
        """Error codes should be string values."""
        assert isinstance(ErrorCode.VALIDATION_UNSUPPORTED_FORMAT.value, str)
        assert isinstance(ErrorCode.MODEL_NOT_FOUND.value, str)

    def test_error_code_naming_convention(self):
        """Error codes should follow CATEGORY_SPECIFIC pattern."""
        for code in ErrorCode:
            parts = code.value.split("_")
            assert len(parts) >= 2, f"Error code {code.value} should have at least 2 parts"


class TestAPIError:
    """Tests for APIError response model."""

    def test_create_minimal_error(self):
        """Create error with required fields only."""
        error = APIError(
            error="Something went wrong",
            code=ErrorCode.SERVER_INTERNAL_ERROR,
        )
        assert error.error == "Something went wrong"
        assert error.code == ErrorCode.SERVER_INTERNAL_ERROR
        assert error.details is None
        assert error.request_id is None

    def test_create_full_error(self):
        """Create error with all fields."""
        error = APIError(
            error="Model not found",
            code=ErrorCode.MODEL_NOT_FOUND,
            details={"model": "test-model"},
            request_id="abc123",
        )
        assert error.error == "Model not found"
        assert error.code == ErrorCode.MODEL_NOT_FOUND
        assert error.details == {"model": "test-model"}
        assert error.request_id == "abc123"

    def test_error_serialization(self):
        """Error should serialize to dict correctly."""
        error = APIError(
            error="Test error",
            code=ErrorCode.VALIDATION_EMPTY_FILE,
            details={"key": "value"},
        )
        data = error.model_dump(exclude_none=True)
        assert "error" in data
        assert "code" in data
        assert "details" in data
        assert "request_id" not in data  # Excluded because None


class TestAPIException:
    """Tests for APIException base class."""

    def test_create_exception(self):
        """Create exception with all parameters."""
        exc = APIException(
            message="Test error",
            code=ErrorCode.SERVER_INTERNAL_ERROR,
            status_code=500,
            details={"key": "value"},
        )
        assert exc.message == "Test error"
        assert exc.code == ErrorCode.SERVER_INTERNAL_ERROR
        assert exc.status_code == 500
        assert exc.details == {"key": "value"}

    def test_default_status_code(self):
        """Default status code should be 400."""
        exc = APIException(
            message="Test",
            code=ErrorCode.VALIDATION_EMPTY_FILE,
        )
        assert exc.status_code == status.HTTP_400_BAD_REQUEST

    def test_to_response(self):
        """to_response should create APIError."""
        exc = APIException(
            message="Test error",
            code=ErrorCode.SERVER_INTERNAL_ERROR,
            details={"key": "value"},
        )
        response = exc.to_response(request_id="req123")
        assert isinstance(response, APIError)
        assert response.error == "Test error"
        assert response.code == ErrorCode.SERVER_INTERNAL_ERROR
        assert response.details == {"key": "value"}
        assert response.request_id == "req123"


class TestValidationErrors:
    """Tests for validation error classes."""

    def test_unsupported_format_error(self):
        """UnsupportedFormatError should have correct attributes."""
        exc = UnsupportedFormatError(
            format=".txt",
            supported_formats=[".wav", ".mp3"],
        )
        assert exc.code == ErrorCode.VALIDATION_UNSUPPORTED_FORMAT
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert ".txt" in exc.message
        assert exc.details["format"] == ".txt"
        assert exc.details["supported_formats"] == [".wav", ".mp3"]

    def test_file_too_large_error(self):
        """FileTooLargeError should have correct attributes."""
        exc = FileTooLargeError(file_size=150000000, max_size=100000000)
        assert exc.code == ErrorCode.VALIDATION_FILE_TOO_LARGE
        assert exc.status_code == status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        assert exc.details["file_size_bytes"] == 150000000
        assert exc.details["max_size_bytes"] == 100000000

    def test_empty_file_error(self):
        """EmptyFileError should have correct attributes."""
        exc = EmptyFileError()
        assert exc.code == ErrorCode.VALIDATION_EMPTY_FILE
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert "empty" in exc.message.lower()

    def test_invalid_language_error(self):
        """InvalidLanguageError should have correct attributes."""
        exc = InvalidLanguageError(language="xx")
        assert exc.code == ErrorCode.VALIDATION_INVALID_LANGUAGE
        assert "xx" in exc.message
        assert exc.details["language"] == "xx"

    def test_prompt_too_long_error(self):
        """PromptTooLongError should have correct attributes."""
        exc = PromptTooLongError(length=2000, max_length=1000)
        assert exc.code == ErrorCode.VALIDATION_PROMPT_TOO_LONG
        assert "1000" in exc.message
        assert exc.details["length"] == 2000
        assert exc.details["max_length"] == 1000


class TestModelErrors:
    """Tests for model error classes."""

    def test_model_not_found_error(self):
        """ModelNotFoundError should have correct attributes."""
        exc = ModelNotFoundError(model_id="test-model")
        assert exc.code == ErrorCode.MODEL_NOT_FOUND
        assert exc.status_code == status.HTTP_404_NOT_FOUND
        assert "test-model" in exc.message
        assert exc.details["model"] == "test-model"

    def test_model_not_downloaded_error(self):
        """ModelNotDownloadedError should have correct attributes."""
        exc = ModelNotDownloadedError(
            model_id="test-model",
            download_url="/models/test-model/download",
        )
        assert exc.code == ErrorCode.MODEL_NOT_DOWNLOADED
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.details["model"] == "test-model"
        assert exc.details["download_url"] == "/models/test-model/download"

    def test_model_not_downloaded_error_no_url(self):
        """ModelNotDownloadedError without URL should work."""
        exc = ModelNotDownloadedError(model_id="test-model")
        assert "download_url" not in exc.details

    def test_model_already_downloaded_error(self):
        """ModelAlreadyDownloadedError should have correct attributes."""
        exc = ModelAlreadyDownloadedError(model_id="test-model")
        assert exc.code == ErrorCode.MODEL_ALREADY_DOWNLOADED
        assert exc.status_code == status.HTTP_409_CONFLICT
        assert exc.details["model"] == "test-model"

    def test_model_unsupported_error(self):
        """ModelUnsupportedError should have correct attributes."""
        exc = ModelUnsupportedError(model_id="invalid-model")
        assert exc.code == ErrorCode.MODEL_UNSUPPORTED
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.details["model"] == "invalid-model"


class TestTranscriptionErrors:
    """Tests for transcription error classes."""

    def test_transcription_failed_error(self):
        """TranscriptionFailedError should have correct attributes."""
        exc = TranscriptionFailedError(
            message="Audio processing failed",
            details={"reason": "corrupted file"},
        )
        assert exc.code == ErrorCode.TRANSCRIPTION_FAILED
        assert exc.status_code == status.HTTP_400_BAD_REQUEST
        assert exc.message == "Audio processing failed"
        assert exc.details == {"reason": "corrupted file"}


class TestExceptionHandlers:
    """Tests for exception handler functions."""

    @pytest.mark.asyncio
    async def test_api_exception_handler(self):
        """api_exception_handler should return JSON response."""
        # Create mock request with request_id
        request = Mock()
        request.state = Mock()
        request.state.request_id = "test123"

        exc = ModelNotFoundError(model_id="test-model")

        response = await api_exception_handler(request, exc)

        assert response.status_code == status.HTTP_404_NOT_FOUND
        # Response body is bytes, need to decode
        import json
        body = json.loads(response.body.decode())
        assert body["error"] == exc.message
        assert body["code"] == ErrorCode.MODEL_NOT_FOUND.value
        assert body["request_id"] == "test123"

    @pytest.mark.asyncio
    async def test_api_exception_handler_no_request_id(self):
        """Handler should work without request_id."""
        request = Mock()
        request.state = Mock(spec=[])  # No request_id attribute

        exc = EmptyFileError()

        response = await api_exception_handler(request, exc)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        import json
        body = json.loads(response.body.decode())
        assert "request_id" not in body

    @pytest.mark.asyncio
    async def test_unhandled_exception_handler(self):
        """unhandled_exception_handler should return 500 error."""
        request = Mock()
        request.state = Mock()
        request.state.request_id = "test456"

        exc = RuntimeError("Something unexpected happened")

        response = await unhandled_exception_handler(request, exc)

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        import json
        body = json.loads(response.body.decode())
        assert body["code"] == ErrorCode.SERVER_INTERNAL_ERROR.value
        # Should not leak internal error message
        assert "unexpected" not in body["error"].lower()
        assert body["request_id"] == "test456"
