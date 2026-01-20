"""Integration tests for standardized error handling."""

import pytest
from io import BytesIO

from app.errors import ErrorCode


class TestTranscribeErrorResponses:
    """Tests for /transcribe endpoint error responses."""

    def test_unsupported_format_returns_structured_error(self, client):
        """Unsupported format should return structured error response."""
        response = client.post(
            "/transcribe",
            files={"file": ("test.txt", b"not audio content", "text/plain")},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == ErrorCode.VALIDATION_UNSUPPORTED_FORMAT.value
        assert "error" in data
        assert "details" in data
        assert "supported_formats" in data["details"]

    def test_empty_file_returns_structured_error(self, client):
        """Empty file should return structured error response."""
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", b"", "audio/wav")},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == ErrorCode.VALIDATION_EMPTY_FILE.value
        assert "error" in data

    def test_invalid_language_returns_structured_error(self, client, sample_audio_bytes):
        """Invalid language code should return structured error response."""
        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={"language": "invalid_lang"},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == ErrorCode.VALIDATION_INVALID_LANGUAGE.value
        assert "error" in data
        assert "details" in data
        assert data["details"]["language"] == "invalid_lang"

    def test_prompt_too_long_returns_structured_error(self, client, sample_audio_bytes):
        """Prompt exceeding max length should return structured error response."""
        long_prompt = "a" * 1001  # MAX_PROMPT_LENGTH is 1000

        response = client.post(
            "/transcribe",
            files={"file": ("test.wav", sample_audio_bytes, "audio/wav")},
            data={"prompt": long_prompt},
        )

        assert response.status_code == 400
        data = response.json()
        assert data["code"] == ErrorCode.VALIDATION_PROMPT_TOO_LONG.value
        assert "error" in data
        assert "details" in data
        assert data["details"]["length"] == 1001
        assert data["details"]["max_length"] == 1000

    def test_error_response_includes_request_id_header(self, client):
        """Error responses should include X-Request-ID header."""
        response = client.post(
            "/transcribe",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestModelsErrorResponses:
    """Tests for /models endpoint error responses."""

    def test_model_not_found_returns_structured_error(self, client):
        """Invalid model ID should return structured error response."""
        response = client.get("/models/invalid-model/status")

        assert response.status_code == 404
        data = response.json()
        assert data["code"] == ErrorCode.MODEL_NOT_FOUND.value
        assert "error" in data
        assert "details" in data
        assert data["details"]["model"] == "invalid-model"

    def test_download_invalid_model_returns_structured_error(self, client):
        """Download of invalid model should return structured error response."""
        response = client.post("/models/invalid-model/download")

        assert response.status_code == 404
        data = response.json()
        assert data["code"] == ErrorCode.MODEL_NOT_FOUND.value

    def test_delete_invalid_model_returns_structured_error(self, client):
        """Delete of invalid model should return structured error response."""
        response = client.delete("/models/invalid-model")

        assert response.status_code == 404
        data = response.json()
        assert data["code"] == ErrorCode.MODEL_NOT_FOUND.value


class TestRequestIDMiddleware:
    """Tests for request ID middleware."""

    def test_request_id_in_success_response(self, client):
        """Successful requests should include X-Request-ID header."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers

    def test_request_id_in_error_response(self, client):
        """Error responses should include X-Request-ID header."""
        response = client.get("/models/invalid/status")

        assert response.status_code == 404
        assert "X-Request-ID" in response.headers

    def test_client_request_id_preserved(self, client):
        """Client-provided X-Request-ID should be preserved."""
        custom_id = "custom-request-123"
        response = client.get(
            "/health",
            headers={"X-Request-ID": custom_id},
        )

        assert response.headers["X-Request-ID"] == custom_id

    def test_request_id_in_error_body(self, client):
        """Error responses should include request_id in body."""
        response = client.get("/models/invalid/status")

        data = response.json()
        assert "request_id" in data
        assert data["request_id"] == response.headers["X-Request-ID"]


class TestHealthEndpoint:
    """Tests for /health endpoint with new response model."""

    def test_health_returns_structured_response(self, client):
        """Health endpoint should return structured response."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data == {"status": "healthy"}

    def test_health_includes_request_id(self, client):
        """Health endpoint should include request ID header."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")

        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MLX Whisper API"
        assert "0.2.0" in schema["info"]["version"]

    def test_openapi_includes_error_responses(self, client):
        """OpenAPI schema should document error responses."""
        response = client.get("/openapi.json")
        schema = response.json()

        # Check /transcribe endpoint has error responses
        transcribe_path = schema["paths"]["/transcribe"]["post"]
        assert "400" in transcribe_path["responses"]
        assert "413" in transcribe_path["responses"]

        # Check /models/{model_id}/status has 404
        status_path = schema["paths"]["/models/{model_id}/status"]["get"]
        assert "404" in status_path["responses"]

    def test_openapi_includes_tags(self, client):
        """OpenAPI schema should include endpoint tags."""
        response = client.get("/openapi.json")
        schema = response.json()

        tag_names = [tag["name"] for tag in schema["tags"]]
        assert "transcription" in tag_names
        assert "models" in tag_names
        assert "health" in tag_names

    def test_swagger_ui_available(self, client):
        """Swagger UI should be available at /docs."""
        response = client.get("/docs")

        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_available(self, client):
        """ReDoc should be available at /redoc."""
        response = client.get("/redoc")

        assert response.status_code == 200
        assert "redoc" in response.text.lower()
