"""Integration tests for standardized error handling."""

import pytest

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
