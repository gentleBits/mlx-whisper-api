"""Unit tests for ModelManager service."""

import pytest
import time
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.services.model_manager import (
    ModelManager,
    ModelMetadata,
    ModelStatus,
    ModelNotFoundError,
    ModelAlreadyDownloadedError,
    ModelNotDownloadedError,
    ModelDownloadError,
    get_model_manager,
)
from app.config import SUPPORTED_MODELS


class TestModelManagerParsing:
    """Tests for model ID parsing."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_parse_all_supported_models(self, manager):
        """All supported models can be parsed with correct metadata."""
        for model_id in SUPPORTED_MODELS:
            meta = manager.parse_model_id(model_id)

            assert meta.id == model_id
            assert meta.size in {"tiny", "small", "large-v3"}
            assert meta.name is not None
            assert len(meta.name) > 0
            assert meta.quantization is None or meta.quantization.startswith("q") or meta.quantization.endswith("bit")
            assert meta.english_only is False

    def test_parse_quantized_model(self, manager):
        """Quantized model ID is parsed with correct quantization metadata."""
        meta = manager.parse_model_id("mlx-community/whisper-large-v3-mlx-8bit")

        assert meta.id == "mlx-community/whisper-large-v3-mlx-8bit"
        assert meta.size == "large-v3"
        assert meta.quantization == "8bit"
        assert meta.name == "Whisper Large V3 (8BIT)"
        assert meta.english_only is False


class TestModelManagerValidation:
    """Tests for model validation."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_is_model_supported(self, manager):
        """Check model support for valid and invalid IDs."""
        assert manager.is_model_supported("mlx-community/whisper-tiny-mlx") is True
        assert manager.is_model_supported("not-a-real/model") is False

    def test_validate_model(self, manager):
        """Validate raises ModelNotFoundError for invalid model only."""
        # Valid model - should not raise
        manager.validate_model("mlx-community/whisper-tiny-mlx")

        # Invalid model - should raise
        with pytest.raises(ModelNotFoundError) as exc_info:
            manager.validate_model("not-a-real/model")
        assert exc_info.value.model_id == "not-a-real/model"


class TestModelManagerStatus:
    """Tests for model status checking."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_get_model_status_not_downloaded(self, manager):
        """Status is not_downloaded when model not in cache."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            status = manager.get_model_status("mlx-community/whisper-tiny-mlx")

        assert status.id == "mlx-community/whisper-tiny-mlx"
        assert status.status == "not_downloaded"
        assert status.path is None
        assert status.size_bytes is None

    def test_get_model_status_downloaded(self, manager, tmp_path):
        """Status is downloaded when model in cache."""
        # Create a fake cache directory
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 1000)

        with patch.object(manager, "get_model_cache_path", return_value=fake_cache):
            status = manager.get_model_status("mlx-community/whisper-tiny-mlx")

        assert status.status == "downloaded"
        assert status.path == str(fake_cache)
        assert status.size_bytes == 1000

    def test_get_model_status_downloading(self, manager):
        """Status is downloading when download in progress."""
        manager.set_download_progress(
            "mlx-community/whisper-tiny-mlx",
            progress=0.5,
            downloaded_bytes=500,
            total_bytes=1000,
        )

        status = manager.get_model_status("mlx-community/whisper-tiny-mlx")

        assert status.status == "downloading"
        assert status.progress == 0.5
        assert status.downloaded_bytes == 500
        assert status.total_bytes == 1000

    def test_get_model_status_invalid_model(self, manager):
        """Status raises for invalid model."""
        with pytest.raises(ModelNotFoundError):
            manager.get_model_status("not-a-real/model")


class TestModelManagerInfo:
    """Tests for complete model info."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_get_model_info(self, manager):
        """Get complete model info combines metadata and status."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            info = manager.get_model_info("mlx-community/whisper-tiny-mlx")

        assert info["id"] == "mlx-community/whisper-tiny-mlx"
        assert info["name"] == "Whisper Tiny"
        assert info["size"] == "tiny"
        assert info["quantization"] is None
        assert info["english_only"] is False
        assert info["status"] == "not_downloaded"

    def test_list_models(self, manager):
        """List models returns all supported models."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            models = manager.list_models()

        assert len(models) == len(SUPPORTED_MODELS)

        model_ids = [m["id"] for m in models]
        for model_id in SUPPORTED_MODELS:
            assert model_id in model_ids


class TestModelManagerDownloadProgress:
    """Tests for download progress tracking."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_download_progress_lifecycle(self, manager):
        """Set and clear download progress."""
        # Clear non-existent model doesn't error
        manager.clear_download_progress("nonexistent/model")

        # Set progress
        manager.set_download_progress(
            "mlx-community/whisper-tiny-mlx",
            progress=0.5,
            downloaded_bytes=500,
            total_bytes=1000,
        )

        assert "mlx-community/whisper-tiny-mlx" in manager._download_progress
        progress = manager._download_progress["mlx-community/whisper-tiny-mlx"]
        assert progress["progress"] == 0.5
        assert progress["downloaded_bytes"] == 500
        assert progress["total_bytes"] == 1000

        # Clear progress
        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")
        assert "mlx-community/whisper-tiny-mlx" not in manager._download_progress


class TestModelManagerDirectorySize:
    """Tests for directory size calculation."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_get_directory_size(self, manager, tmp_path):
        """Calculate directory size correctly, including edge cases."""
        # Empty directory returns 0
        assert manager.get_directory_size(tmp_path) == 0
        # Nonexistent directory returns 0
        assert manager.get_directory_size(tmp_path / "nonexistent") == 0

        # Create test files
        (tmp_path / "file1.bin").write_bytes(b"x" * 100)
        (tmp_path / "file2.bin").write_bytes(b"y" * 200)
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "file3.bin").write_bytes(b"z" * 50)

        size = manager.get_directory_size(tmp_path)
        assert size == 350


class TestGetModelManager:
    """Tests for get_model_manager singleton."""

    def test_returns_same_instance(self):
        """Singleton returns the same instance."""
        # Reset the singleton for testing
        import app.services.model_manager as module

        module._manager = None

        manager1 = get_model_manager()
        manager2 = get_model_manager()

        assert manager1 is manager2

    def test_returns_model_manager(self):
        """Returns a ModelManager instance."""
        manager = get_model_manager()
        assert isinstance(manager, ModelManager)


class TestModelManagerDownload:
    """Tests for model download functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_download_model_invalid_model(self, manager):
        """Download raises for invalid model."""
        with pytest.raises(ModelNotFoundError):
            manager.download_model("not-a-real/model")

    def test_download_model_already_downloaded(self, manager, tmp_path):
        """Download raises if model already downloaded."""
        # Create a fake cache directory
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        (fake_cache / "model.bin").write_bytes(b"x" * 100)

        with patch.object(manager, "get_model_cache_path", return_value=fake_cache):
            with pytest.raises(ModelAlreadyDownloadedError) as exc_info:
                manager.download_model("mlx-community/whisper-tiny-mlx")

        assert exc_info.value.model_id == "mlx-community/whisper-tiny-mlx"

    def test_download_model_success(self, manager):
        """Download completes successfully and clears progress."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download"
            ) as mock_download:
                manager.download_model("mlx-community/whisper-tiny-mlx")

                mock_download.assert_called_once()
                call_kwargs = mock_download.call_args.kwargs
                assert call_kwargs["repo_id"] == "mlx-community/whisper-tiny-mlx"

        # Progress should be cleared after successful download
        assert "mlx-community/whisper-tiny-mlx" not in manager._download_progress

    def test_download_model_error_clears_progress(self, manager):
        """Download clears progress tracking on error."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download",
                side_effect=Exception("Network error"),
            ):
                with pytest.raises(ModelDownloadError):
                    manager.download_model("mlx-community/whisper-tiny-mlx")

        # Progress should be cleared after error
        assert "mlx-community/whisper-tiny-mlx" not in manager._download_progress

    def test_download_model_error_includes_message(self, manager):
        """Download error includes original error message."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download",
                side_effect=Exception("Connection refused"),
            ):
                with pytest.raises(ModelDownloadError) as exc_info:
                    manager.download_model("mlx-community/whisper-tiny-mlx")

        assert "Connection refused" in exc_info.value.message

    def test_is_download_in_progress(self, manager):
        """Check if download is in progress."""
        assert not manager.is_download_in_progress("mlx-community/whisper-tiny-mlx")

        manager.set_download_progress("mlx-community/whisper-tiny-mlx", progress=0.5)
        assert manager.is_download_in_progress("mlx-community/whisper-tiny-mlx")

        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")
        assert not manager.is_download_in_progress("mlx-community/whisper-tiny-mlx")

    def test_download_skipped_if_already_in_progress(self, manager):
        """Download is skipped if already in progress."""
        # Set download as in progress
        manager.set_download_progress("mlx-community/whisper-tiny-mlx", progress=0.5)

        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download"
            ) as mock_download:
                # Should return without starting another download
                manager.download_model("mlx-community/whisper-tiny-mlx")

                # snapshot_download should NOT be called
                mock_download.assert_not_called()

        # Clean up
        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")


class TestModelManagerAsyncDownload:
    """Tests for async model download functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_start_download_async_starts_thread(self, manager):
        """Async download starts a background thread."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download"
            ) as mock_download:
                # Make the mock block briefly to ensure thread starts
                mock_download.side_effect = lambda **kwargs: time.sleep(0.1)

                manager.start_download_async("mlx-community/whisper-tiny-mlx")

                # Should immediately mark as downloading
                assert manager.is_download_in_progress("mlx-community/whisper-tiny-mlx")

                # Wait for thread to complete
                time.sleep(0.2)

                # Should be cleared after completion
                assert not manager.is_download_in_progress(
                    "mlx-community/whisper-tiny-mlx"
                )

    def test_start_download_async_skipped_if_already_in_progress(self, manager):
        """Async download is skipped if already in progress."""
        manager.set_download_progress("mlx-community/whisper-tiny-mlx", progress=0.5)

        with patch.object(manager, "get_model_cache_path", return_value=None):
            with patch(
                "app.services.model_manager.snapshot_download"
            ) as mock_download:
                manager.start_download_async("mlx-community/whisper-tiny-mlx")

                # Should not start another thread
                mock_download.assert_not_called()

        manager.clear_download_progress("mlx-community/whisper-tiny-mlx")


class TestModelManagerDelete:
    """Tests for model deletion functionality."""

    @pytest.fixture
    def manager(self):
        """Create a fresh ModelManager instance."""
        return ModelManager()

    def test_delete_model_invalid_model(self, manager):
        """Delete raises for invalid model."""
        with pytest.raises(ModelNotFoundError):
            manager.delete_model("not-a-real/model")

    def test_delete_model_not_downloaded(self, manager):
        """Delete raises if model not downloaded."""
        with patch.object(manager, "get_model_cache_path", return_value=None):
            with pytest.raises(ModelNotDownloadedError) as exc_info:
                manager.delete_model("mlx-community/whisper-tiny-mlx")

        assert exc_info.value.model_id == "mlx-community/whisper-tiny-mlx"

    def test_delete_model_success(self, manager, tmp_path):
        """Delete removes model from cache, including nested directories."""
        # Create a fake cache directory with subdirectories
        fake_cache = tmp_path / "model"
        fake_cache.mkdir()
        subdir = fake_cache / "subdir"
        subdir.mkdir()
        (subdir / "model.bin").write_bytes(b"x" * 100)
        (fake_cache / "config.json").write_text("{}")

        with patch.object(manager, "get_model_cache_path", return_value=fake_cache):
            manager.delete_model("mlx-community/whisper-tiny-mlx")

        # Directory and all contents should be deleted
        assert not fake_cache.exists()

    def test_delete_model_path_not_exists(self, manager, tmp_path):
        """Delete raises if cache path returned but doesn't exist."""
        fake_cache = tmp_path / "nonexistent"

        with patch.object(manager, "get_model_cache_path", return_value=fake_cache):
            with pytest.raises(ModelNotDownloadedError):
                manager.delete_model("mlx-community/whisper-tiny-mlx")
