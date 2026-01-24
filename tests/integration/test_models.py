"""Integration tests for model management endpoints."""

import pytest
from unittest.mock import patch

from app.config import SUPPORTED_MODELS


class TestListModels:
    """Tests for GET /models endpoint."""

    def test_list_models(self, client):
        """GET /models returns all supported models with required fields."""
        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

        # Should have all supported models
        model_ids = [m["id"] for m in data["models"]]
        assert len(model_ids) == len(SUPPORTED_MODELS)

        valid_statuses = {"downloaded", "not_downloaded", "downloading"}
        for model in data["models"]:
            assert "id" in model
            assert "name" in model
            assert "size" in model
            assert "status" in model
            assert model["status"] in valid_statuses


class TestModelStatus:
    """Tests for GET /models/{model_id}/status endpoint."""

    def test_model_status_valid(self, client):
        """GET /models/{id}/status returns correct structure for valid model."""
        response = client.get("/models/mlx-community/whisper-tiny-mlx/status")
        assert response.status_code == 200

        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["id"] == "mlx-community/whisper-tiny-mlx"
        assert data["status"] in {"downloaded", "not_downloaded", "downloading"}

    def test_model_status_invalid_id(self, client):
        """Returns 404 for unknown model ID."""
        response = client.get("/models/not-a-real/model/status")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "not found" in data["error"].lower()

    def test_model_status_url_encoded_id(self, client):
        """Handles URL-encoded model IDs correctly."""
        response = client.get("/models/mlx-community%2Fwhisper-tiny-mlx/status")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/whisper-tiny-mlx"


class TestModelDownload:
    """Tests for POST /models/{model_id}/download endpoint."""

    def test_download_model_success(self, client):
        """POST /models/{id}/download returns 200 and starts download."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch("app.services.model_manager.snapshot_download"):
                response = client.post(
                    "/models/mlx-community/whisper-tiny-mlx/download"
                )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["id"] == "mlx-community/whisper-tiny-mlx"
        assert data["status"] == "download_started"

    def test_download_model_invalid_id(self, client):
        """Returns 404 for unknown model ID."""
        response = client.post("/models/not-a-real/model/download")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_download_model_already_downloaded(self, client, tmp_path):
        """Returns 409 if model is already downloaded."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.post("/models/mlx-community/whisper-tiny-mlx/download")

        assert response.status_code == 409
        data = response.json()
        assert "already downloaded" in data["error"].lower()


class TestModelDelete:
    """Tests for DELETE /models/{model_id} endpoint."""

    def test_delete_model_success(self, client, tmp_path):
        """DELETE /models/{id} returns 200 and removes files."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/whisper-tiny-mlx"
        assert data["status"] == "deleted"
        assert not fake_cache.exists()

    def test_delete_model_invalid_id(self, client):
        """Returns 404 for unknown model ID."""
        response = client.delete("/models/not-a-real/model")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["error"].lower()

    def test_delete_model_not_downloaded(self, client):
        """Returns 400 if model is not downloaded."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        assert response.status_code == 400
        data = response.json()
        assert "not downloaded" in data["error"].lower()
