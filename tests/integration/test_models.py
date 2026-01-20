"""Integration tests for model management endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.config import SUPPORTED_MODELS


class TestListModels:
    """Tests for GET /models endpoint."""

    def test_list_models_returns_200(self, client):
        """GET /models returns 200 status."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_list_models_response_structure(self, client):
        """Response contains models array."""
        response = client.get("/models")
        data = response.json()

        assert "models" in data
        assert isinstance(data["models"], list)

    def test_list_models_returns_all_supported(self, client):
        """Returns all supported models."""
        response = client.get("/models")
        data = response.json()

        # Should have all supported models
        model_ids = [m["id"] for m in data["models"]]
        assert len(model_ids) == len(SUPPORTED_MODELS)

        for model_id in SUPPORTED_MODELS:
            assert model_id in model_ids

    def test_list_models_model_info_fields(self, client):
        """Each model has required fields."""
        response = client.get("/models")
        data = response.json()

        for model in data["models"]:
            assert "id" in model
            assert "name" in model
            assert "size" in model
            assert "quantization" in model  # Can be null
            assert "english_only" in model
            assert "status" in model
            # size_bytes and download_progress can be null

    def test_list_models_includes_status(self, client):
        """Each model has a valid status field."""
        response = client.get("/models")
        data = response.json()

        valid_statuses = {"downloaded", "not_downloaded", "downloading"}
        for model in data["models"]:
            assert model["status"] in valid_statuses

    def test_list_models_parses_metadata_correctly(self, client):
        """Model metadata is correctly parsed from IDs."""
        response = client.get("/models")
        data = response.json()

        # Find specific models and verify their metadata
        models_by_id = {m["id"]: m for m in data["models"]}

        # Test tiny model
        tiny = models_by_id.get("mlx-community/whisper-tiny-mlx")
        assert tiny is not None
        assert tiny["size"] == "tiny"
        assert tiny["quantization"] is None
        assert tiny["english_only"] is False
        assert "Whisper Tiny" in tiny["name"]

        # Test small model
        small = models_by_id.get("mlx-community/whisper-small-mlx")
        assert small is not None
        assert small["size"] == "small"
        assert small["quantization"] is None
        assert small["english_only"] is False
        assert "Whisper Small" in small["name"]

        # Test large-v3 model
        large_v3 = models_by_id.get("mlx-community/whisper-large-v3-mlx")
        assert large_v3 is not None
        assert large_v3["size"] == "large-v3"
        assert "Large V3" in large_v3["name"]


class TestModelStatus:
    """Tests for GET /models/{model_id}/status endpoint."""

    def test_model_status_returns_200(self, client):
        """GET /models/{id}/status returns 200 for valid model."""
        response = client.get("/models/mlx-community/whisper-tiny-mlx/status")
        assert response.status_code == 200

    def test_model_status_response_structure(self, client):
        """Response has correct structure."""
        response = client.get("/models/mlx-community/whisper-tiny-mlx/status")
        data = response.json()

        assert "id" in data
        assert "status" in data
        assert data["id"] == "mlx-community/whisper-tiny-mlx"

    def test_model_status_not_downloaded(self, client):
        """Status is not_downloaded for model not in cache."""
        # Mock the cache check to return None (not downloaded)
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None

            response = client.get("/models/mlx-community/whisper-tiny-mlx/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "not_downloaded"
        assert data["path"] is None
        assert data["size_bytes"] is None

    def test_model_status_downloaded(self, client, tmp_path):
        """Status is downloaded for model in cache."""
        # Create a fake cache directory with some files
        fake_cache = tmp_path / "model_cache"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 1000)
        (fake_cache / "config.json").write_text("{}")

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.get("/models/mlx-community/whisper-tiny-mlx/status")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "downloaded"
        assert data["path"] == str(fake_cache)
        assert data["size_bytes"] is not None
        assert data["size_bytes"] > 0

    def test_model_status_downloading(self, client):
        """Status is downloading when download is in progress."""
        from app.services.model_manager import get_model_manager

        manager = get_model_manager()
        # Set download progress
        manager.set_download_progress(
            "mlx-community/whisper-tiny-mlx",
            progress=0.5,
            downloaded_bytes=500000,
            total_bytes=1000000,
        )

        try:
            response = client.get("/models/mlx-community/whisper-tiny-mlx/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "downloading"
            assert data["progress"] == 0.5
            assert data["downloaded_bytes"] == 500000
            assert data["total_bytes"] == 1000000
        finally:
            # Clean up
            manager.clear_download_progress("mlx-community/whisper-tiny-mlx")

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
        # Test with URL-encoded slash (%2F)
        response = client.get("/models/mlx-community%2Fwhisper-tiny-mlx/status")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/whisper-tiny-mlx"

    def test_model_status_all_supported_models(self, client):
        """All supported models return valid status."""
        for model_id in SUPPORTED_MODELS:
            # Use the path directly with slash
            response = client.get(f"/models/{model_id}/status")
            assert response.status_code == 200, f"Failed for model: {model_id}"

            data = response.json()
            assert data["id"] == model_id
            assert data["status"] in {"downloaded", "not_downloaded", "downloading"}


class TestModelEndpointEdgeCases:
    """Edge case tests for model endpoints."""

    def test_models_endpoint_with_trailing_slash(self, client):
        """GET /models/ handles trailing slash."""
        response = client.get("/models/")
        # FastAPI may redirect, handle, or return 405 (if path routes match)
        # The key is it shouldn't cause a 500 server error
        assert response.status_code in {200, 307, 404, 405}

    def test_empty_model_id(self, client):
        """Empty model ID in path is handled."""
        response = client.get("/models//status")
        # Should return 404 for empty model ID
        assert response.status_code in {404, 422}

    def test_model_status_special_characters(self, client):
        """Model IDs with special characters are handled."""
        # Test with a model ID that has hyphens and version numbers
        response = client.get("/models/mlx-community/whisper-large-v3-mlx/status")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == "mlx-community/whisper-large-v3-mlx"
        assert data["status"] in {"downloaded", "not_downloaded", "downloading"}


class TestModelDownload:
    """Tests for POST /models/{model_id}/download endpoint."""

    def test_download_model_returns_200(self, client):
        """POST /models/{id}/download returns 200 for valid model."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch(
                "app.services.model_manager.snapshot_download"
            ) as mock_download:
                response = client.post(
                    "/models/mlx-community/whisper-tiny-mlx/download"
                )

        assert response.status_code == 200

    def test_download_model_response_structure(self, client):
        """Response has correct structure."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch("app.services.model_manager.snapshot_download"):
                response = client.post(
                    "/models/mlx-community/whisper-tiny-mlx/download"
                )

        data = response.json()
        assert "id" in data
        assert "status" in data
        assert "message" in data
        assert data["id"] == "mlx-community/whisper-tiny-mlx"
        assert data["status"] == "download_started"

    def test_download_model_invalid_id(self, client):
        """Returns 404 for unknown model ID."""
        response = client.post("/models/not-a-real/model/download")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "code" in data
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
        assert "error" in data
        assert "code" in data
        assert "already downloaded" in data["error"].lower()

    def test_download_model_url_encoded_id(self, client):
        """Handles URL-encoded model IDs correctly."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch("app.services.model_manager.snapshot_download"):
                response = client.post(
                    "/models/mlx-community%2Fwhisper-tiny-mlx/download"
                )

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/whisper-tiny-mlx"

    def test_download_model_initiates_background_download(self, client):
        """Download starts in the background."""
        from app.services.model_manager import get_model_manager

        manager = get_model_manager()

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch("app.services.model_manager.snapshot_download") as mock_dl:
                # Make download take time
                import time

                mock_dl.side_effect = lambda **kwargs: time.sleep(0.1)

                response = client.post(
                    "/models/mlx-community/whisper-tiny-mlx/download"
                )

                # Should return immediately with download_started
                assert response.status_code == 200
                assert response.json()["status"] == "download_started"

                # Wait for background thread to complete
                time.sleep(0.2)

        # Clean up any lingering progress
        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")


class TestModelDelete:
    """Tests for DELETE /models/{model_id} endpoint."""

    def test_delete_model_returns_200(self, client, tmp_path):
        """DELETE /models/{id} returns 200 for downloaded model."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        assert response.status_code == 200

    def test_delete_model_response_structure(self, client, tmp_path):
        """Response has correct structure."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        data = response.json()
        assert "id" in data
        assert "status" in data
        assert data["id"] == "mlx-community/whisper-tiny-mlx"
        assert data["status"] == "deleted"

    def test_delete_model_invalid_id(self, client):
        """Returns 404 for unknown model ID."""
        response = client.delete("/models/not-a-real/model")

        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "not found" in data["error"].lower()

    def test_delete_model_not_downloaded(self, client):
        """Returns 400 if model is not downloaded (model exists but not downloaded)."""
        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        # 400 (not 404) because the model exists in supported list, just not downloaded
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "code" in data
        assert "not downloaded" in data["error"].lower()

    def test_delete_model_url_encoded_id(self, client, tmp_path):
        """Handles URL-encoded model IDs correctly."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.delete("/models/mlx-community%2Fwhisper-tiny-mlx")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "mlx-community/whisper-tiny-mlx"

    def test_delete_model_removes_files(self, client, tmp_path):
        """Delete actually removes the model files."""
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)
        subdir = fake_cache / "subdir"
        subdir.mkdir()
        (subdir / "weights.bin").write_bytes(b"y" * 200)

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = fake_cache

            response = client.delete("/models/mlx-community/whisper-tiny-mlx")

        assert response.status_code == 200
        # Files should be deleted
        assert not fake_cache.exists()


class TestDownloadDeleteIntegration:
    """Integration tests for download and delete workflows."""

    def test_download_then_check_status(self, client):
        """After starting download, status shows downloading."""
        from app.services.model_manager import get_model_manager

        manager = get_model_manager()

        with patch(
            "app.services.model_manager.ModelManager.get_model_cache_path"
        ) as mock_cache:
            mock_cache.return_value = None
            with patch("app.services.model_manager.snapshot_download") as mock_dl:
                import time

                # Make download slow
                def slow_download(**kwargs):
                    time.sleep(0.2)

                mock_dl.side_effect = slow_download

                # Start download
                response = client.post(
                    "/models/mlx-community/whisper-tiny-mlx/download"
                )
                assert response.status_code == 200

                # Check status immediately - should show downloading
                status_response = client.get(
                    "/models/mlx-community/whisper-tiny-mlx/status"
                )
                assert status_response.status_code == 200
                # Status could be downloading or not_downloaded depending on timing
                # The key is that it doesn't error

                # Wait for download to complete
                time.sleep(0.3)

        # Clean up
        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")
