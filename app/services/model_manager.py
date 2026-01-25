"""Model download and status tracking service."""

import os
import re
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

from huggingface_hub import scan_cache_dir, snapshot_download, HfApi

from app.config import SUPPORTED_MODELS, HUGGINGFACE_CACHE


@dataclass
class ModelMetadata:
    """Parsed metadata from a model ID."""

    id: str
    name: str
    size: str
    quantization: Optional[str]
    english_only: bool


@dataclass
class ModelStatus:
    """Status information for a model."""

    id: str
    status: str  # "downloaded", "not_downloaded", "downloading"
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    progress: Optional[float] = None
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None


class ModelNotFoundError(Exception):
    """Exception raised when a model ID is not in the supported list."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is not supported")


class ModelAlreadyDownloadedError(Exception):
    """Exception raised when trying to download an already downloaded model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is already downloaded")


class ModelNotDownloadedError(Exception):
    """Exception raised when trying to delete a model that isn't downloaded."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"Model '{model_id}' is not downloaded")


class ModelDownloadError(Exception):
    """Exception raised when model download fails."""

    def __init__(self, model_id: str, message: str):
        self.model_id = model_id
        self.message = message
        super().__init__(f"Failed to download model '{model_id}': {message}")


class ModelManager:
    """Service for managing MLX Whisper models."""

    # Size display names
    SIZE_NAMES = {
        "tiny": "Tiny",
        "small": "Small",
        "large": "Large",
        "large-v3": "Large V3",
    }

    def __init__(self):
        """Initialize the model manager."""
        self._download_progress: dict[str, dict] = {}

    def parse_model_id(self, model_id: str) -> ModelMetadata:
        """Parse a model ID to extract metadata.

        Examples:
            mlx-community/whisper-tiny-mlx -> size=tiny
            mlx-community/whisper-small-mlx -> size=small
            mlx-community/whisper-large-v3-mlx -> size=large-v3

        Args:
            model_id: Full model identifier (e.g., "mlx-community/whisper-tiny-mlx")

        Returns:
            ModelMetadata with parsed information
        """
        # Extract the model name part after the org prefix
        if "/" in model_id:
            _, model_name = model_id.split("/", 1)
        else:
            model_name = model_id

        # Remove "whisper-" prefix
        model_name = model_name.replace("whisper-", "")

        quantization = None

        # Remove "-mlx" suffix (handles both "-mlx" at end and "-mlx-" in middle)
        model_name = model_name.replace("-mlx", "")

        # Extract quantization suffix (e.g., -q8, -q4)
        quant_match = re.search(r"-q(\d+)$", model_name)
        if quant_match:
            quantization = f"q{quant_match.group(1)}"
            model_name = model_name[: quant_match.start()]

        # Check for English-only variant
        english_only = ".en" in model_name
        model_name = model_name.replace(".en", "")

        # Determine size
        size = self._extract_size(model_name)

        # Build display name
        display_name = self._build_display_name(size, english_only, quantization)

        return ModelMetadata(
            id=model_id,
            name=display_name,
            size=size,
            quantization=quantization,
            english_only=english_only,
        )

    def _extract_size(self, model_name: str) -> str:
        """Extract the model size from the parsed name."""
        if "large-v3" in model_name:
            return "large-v3"
        elif "large" in model_name:
            return "large"
        elif "small" in model_name:
            return "small"
        elif "tiny" in model_name:
            return "tiny"
        return "unknown"

    def _build_display_name(
        self, size: str, english_only: bool, quantization: Optional[str]
    ) -> str:
        """Build a human-readable display name for the model."""
        size_name = self.SIZE_NAMES.get(size, size.title())
        name = f"Whisper {size_name}"

        if english_only:
            name += " English"

        if quantization:
            name += f" ({quantization.upper()})"

        return name

    def is_model_supported(self, model_id: str) -> bool:
        """Check if a model ID is in the supported list."""
        return model_id in SUPPORTED_MODELS

    def validate_model(self, model_id: str) -> None:
        """Validate that a model ID is supported.

        Raises:
            ModelNotFoundError: If the model is not supported
        """
        if not self.is_model_supported(model_id):
            raise ModelNotFoundError(model_id)

    def get_model_cache_path(self, model_id: str) -> Optional[Path]:
        """Get the cache path for a model if it's downloaded.

        Args:
            model_id: Model identifier (e.g., "mlx-community/whisper-tiny-mlx")

        Returns:
            Path to the cached model directory, or None if not downloaded
        """
        try:
            cache_info = scan_cache_dir(HUGGINGFACE_CACHE)
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    return Path(repo.repo_path)
        except Exception:
            pass
        return None

    def get_directory_size(self, path: Path) -> int:
        """Calculate the total size of a directory in bytes."""
        total = 0
        try:
            for entry in path.rglob("*"):
                # Skip symlinks to avoid double-counting (HuggingFace cache uses
                # symlinks in snapshots/ pointing to actual files in blobs/)
                if entry.is_file() and not entry.is_symlink():
                    total += entry.stat().st_size
        except Exception:
            pass
        return total

    def get_model_status(self, model_id: str) -> ModelStatus:
        """Get the status of a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelStatus with current status information

        Raises:
            ModelNotFoundError: If the model is not supported
        """
        self.validate_model(model_id)

        # Check if download is in progress
        if model_id in self._download_progress:
            progress_info = self._download_progress[model_id]
            return ModelStatus(
                id=model_id,
                status="downloading",
                progress=progress_info.get("progress", 0.0),
                downloaded_bytes=progress_info.get("downloaded_bytes"),
                total_bytes=progress_info.get("total_bytes"),
            )

        # Check if downloaded
        cache_path = self.get_model_cache_path(model_id)
        if cache_path and cache_path.exists():
            size_bytes = self.get_directory_size(cache_path)
            return ModelStatus(
                id=model_id,
                status="downloaded",
                path=str(cache_path),
                size_bytes=size_bytes,
            )

        return ModelStatus(
            id=model_id,
            status="not_downloaded",
        )

    def get_model_info(self, model_id: str) -> dict:
        """Get complete information for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dict with model metadata and status

        Raises:
            ModelNotFoundError: If the model is not supported
        """
        metadata = self.parse_model_id(model_id)
        status = self.get_model_status(model_id)

        return {
            "id": metadata.id,
            "name": metadata.name,
            "size": metadata.size,
            "quantization": metadata.quantization,
            "english_only": metadata.english_only,
            "status": status.status,
            "size_bytes": status.size_bytes,
            "download_progress": status.progress,
        }

    def list_models(self) -> list[dict]:
        """List all supported models with their status.

        Returns:
            List of model info dicts for all supported models
        """
        return [self.get_model_info(model_id) for model_id in SUPPORTED_MODELS]

    def set_download_progress(
        self,
        model_id: str,
        progress: float,
        downloaded_bytes: Optional[int] = None,
        total_bytes: Optional[int] = None,
    ) -> None:
        """Update download progress for a model.

        Args:
            model_id: Model identifier
            progress: Progress as a float (0.0 to 1.0)
            downloaded_bytes: Bytes downloaded so far
            total_bytes: Total bytes to download
        """
        self._download_progress[model_id] = {
            "progress": progress,
            "downloaded_bytes": downloaded_bytes,
            "total_bytes": total_bytes,
        }

    def clear_download_progress(self, model_id: str) -> None:
        """Clear download progress tracking for a model.

        Args:
            model_id: Model identifier
        """
        self._download_progress.pop(model_id, None)

    def is_download_in_progress(self, model_id: str) -> bool:
        """Check if a download is currently in progress for a model.

        Args:
            model_id: Model identifier

        Returns:
            True if download is in progress, False otherwise
        """
        return model_id in self._download_progress

    def download_model(self, model_id: str) -> None:
        """Download a model from HuggingFace Hub synchronously.

        This method blocks until the download is complete.
        For non-blocking downloads, use start_download_async.

        Args:
            model_id: Model identifier

        Raises:
            ModelNotFoundError: If the model is not supported
            ModelAlreadyDownloadedError: If the model is already downloaded
        """
        self.validate_model(model_id)

        # Check if already downloaded
        cache_path = self.get_model_cache_path(model_id)
        if cache_path and cache_path.exists():
            raise ModelAlreadyDownloadedError(model_id)

        # Check if already downloading
        if self.is_download_in_progress(model_id):
            return  # Already downloading, don't start another

        try:
            # Initialize progress tracking
            self.set_download_progress(model_id, progress=0.0)

            # Download the model using snapshot_download
            snapshot_download(
                repo_id=model_id,
                cache_dir=HUGGINGFACE_CACHE,
            )

            # Mark as complete (progress tracking will be cleared
            # since the model is now in cache)
            self.clear_download_progress(model_id)

        except Exception as e:
            self.clear_download_progress(model_id)
            raise ModelDownloadError(model_id, str(e))

    def start_download_async(self, model_id: str) -> None:
        """Start a model download in a background thread.

        This method returns immediately and the download proceeds
        in the background. Check status with get_model_status().

        Args:
            model_id: Model identifier

        Raises:
            ModelNotFoundError: If the model is not supported
            ModelAlreadyDownloadedError: If the model is already downloaded
        """
        self.validate_model(model_id)

        # Check if already downloaded
        cache_path = self.get_model_cache_path(model_id)
        if cache_path and cache_path.exists():
            raise ModelAlreadyDownloadedError(model_id)

        # Check if already downloading
        if self.is_download_in_progress(model_id):
            return  # Already downloading

        # Initialize progress tracking
        self.set_download_progress(model_id, progress=0.0)

        # Start download in background thread
        thread = threading.Thread(
            target=self._download_in_background,
            args=(model_id,),
            daemon=True,
        )
        thread.start()

    def _download_in_background(self, model_id: str) -> None:
        """Background thread function to download a model.

        Args:
            model_id: Model identifier
        """
        try:
            snapshot_download(
                repo_id=model_id,
                cache_dir=HUGGINGFACE_CACHE,
            )
            # Download complete - clear progress tracking
            self.clear_download_progress(model_id)
        except Exception as e:
            # Store error in progress tracking
            self._download_progress[model_id] = {
                "progress": 0.0,
                "error": str(e),
            }

    def delete_model(self, model_id: str) -> None:
        """Delete a downloaded model from the cache.

        Args:
            model_id: Model identifier

        Raises:
            ModelNotFoundError: If the model is not supported
            ModelNotDownloadedError: If the model is not downloaded
        """
        self.validate_model(model_id)

        # Find the model in the cache
        cache_path = self.get_model_cache_path(model_id)
        if not cache_path or not cache_path.exists():
            raise ModelNotDownloadedError(model_id)

        # Delete the model directory
        try:
            shutil.rmtree(cache_path)
        except Exception as e:
            raise RuntimeError(f"Failed to delete model '{model_id}': {e}")

        # Also try to clean up the HuggingFace cache metadata
        try:
            # The refs directory stores branch/tag references
            cache_base = Path(HUGGINGFACE_CACHE) / "hub"
            model_dir_name = f"models--{model_id.replace('/', '--')}"
            refs_path = cache_base / model_dir_name
            if refs_path.exists() and refs_path != cache_path:
                shutil.rmtree(refs_path)
        except Exception:
            # Best effort cleanup of metadata
            pass


# Singleton instance
_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the model manager singleton."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager
