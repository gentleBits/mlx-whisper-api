"""Model download, validation, and status tracking service."""

from __future__ import annotations

import gc
import json
import re
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from huggingface_hub import scan_cache_dir, snapshot_download
from mlx_whisper.load_models import load_model

from app.config import HUGGINGFACE_CACHE, SUPPORTED_MODELS


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
    status: str  # "downloaded", "not_downloaded", "downloading", "error"
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    progress: Optional[float] = None
    downloaded_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    error: Optional[str] = None


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
    """Service for managing MLX Whisper models and validation state."""

    VALIDATION_STATE_WORKING = "working"
    VALIDATION_STATE_BROKEN = "broken"

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
        self._state_lock = threading.Lock()
        configured_cache = Path(HUGGINGFACE_CACHE).expanduser()
        if configured_cache.name == "hub":
            # Backward-compatible: allow explicit hub path in env config.
            self._hf_hub_cache_path = configured_cache
            self._hf_cache_root_path = configured_cache.parent
        else:
            self._hf_cache_root_path = configured_cache
            self._hf_hub_cache_path = configured_cache / "hub"
        self._validation_state_file = (
            self._hf_cache_root_path
            / "mlx_whisper_api"
            / "model_validation_state.json"
        )
        self._validation_state = self._load_validation_state()

    def _load_validation_state(self) -> dict[str, dict[str, Any]]:
        """Load persisted validation state from disk."""
        if not self._validation_state_file.exists():
            return {}

        try:
            with self._validation_state_file.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                return raw
        except Exception:
            # Invalid or unreadable state file should not break startup.
            pass
        return {}

    def _save_validation_state(self) -> None:
        """Persist validation state to disk atomically."""
        state_dir = self._validation_state_file.parent
        state_dir.mkdir(parents=True, exist_ok=True)

        temp_path = self._validation_state_file.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(self._validation_state, f, indent=2, sort_keys=True)
        temp_path.replace(self._validation_state_file)

    def _set_validation_state(
        self,
        model_id: str,
        state: str,
        cache_fingerprint: Optional[dict[str, Any]],
        error: Optional[str],
    ) -> None:
        """Set and persist validation state for a model."""
        payload = {
            "state": state,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "cache_fingerprint": cache_fingerprint,
            "error": error,
        }
        with self._state_lock:
            self._validation_state[model_id] = payload
            self._save_validation_state()

    def _clear_validation_state(self, model_id: str) -> None:
        """Remove and persist validation state for a model."""
        with self._state_lock:
            if model_id in self._validation_state:
                del self._validation_state[model_id]
                self._save_validation_state()

    def _get_validation_state(self, model_id: str) -> Optional[dict[str, Any]]:
        """Get validation state for a model."""
        with self._state_lock:
            state = self._validation_state.get(model_id)
            return dict(state) if isinstance(state, dict) else None

    def _get_repo_cache_info(self, model_id: str) -> Any:
        """Get cached HuggingFace repo metadata for a model."""
        try:
            cache_info = scan_cache_dir(str(self._hf_hub_cache_path))
            for repo in cache_info.repos:
                if repo.repo_id == model_id:
                    return repo
        except Exception:
            pass
        return None

    def _get_latest_revision(self, repo_info: Any) -> Any:
        """Get the latest cached revision for a repo."""
        revisions = list(getattr(repo_info, "revisions", []))
        if not revisions:
            return None
        return max(revisions, key=lambda rev: getattr(rev, "last_modified", 0) or 0)

    def _get_model_cache_fingerprint(self, model_id: str) -> Optional[dict[str, Any]]:
        """Build a fingerprint of the model cache state."""
        repo_info = self._get_repo_cache_info(model_id)
        if repo_info is None:
            return None

        revision = self._get_latest_revision(repo_info)
        return {
            "commit_hash": getattr(revision, "commit_hash", None) if revision else None,
            "size_on_disk": int(getattr(repo_info, "size_on_disk", 0) or 0),
            "nb_files": int(getattr(repo_info, "nb_files", 0) or 0),
        }

    def _get_model_snapshot_path(self, model_id: str) -> Optional[Path]:
        """Get the latest snapshot path for a cached model."""
        repo_info = self._get_repo_cache_info(model_id)
        if repo_info is None:
            return None

        revision = self._get_latest_revision(repo_info)
        if revision is None:
            return None

        snapshot_path = getattr(revision, "snapshot_path", None)
        if snapshot_path is None:
            return None
        return Path(snapshot_path)

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

        # Extract quantization suffix (e.g., -q8, -q4, -8bit, -4bit)
        quant_match = re.search(r"-q(\d+)$", model_name)
        bit_match = re.search(r"-(\d+bit)$", model_name)
        if quant_match:
            quantization = f"q{quant_match.group(1)}"
            model_name = model_name[: quant_match.start()]
        elif bit_match:
            quantization = bit_match.group(1)
            model_name = model_name[: bit_match.start()]

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
        repo_info = self._get_repo_cache_info(model_id)
        if repo_info is None:
            return None
        return Path(repo_info.repo_path)

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

        # Check if download is in progress or failed.
        if model_id in self._download_progress:
            progress_info = self._download_progress[model_id]
            if progress_info.get("error"):
                return ModelStatus(
                    id=model_id,
                    status="error",
                    error=str(progress_info["error"]),
                )
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
            current_fingerprint = self._get_model_cache_fingerprint(model_id)
            validation_state = self._get_validation_state(model_id)

            if (
                validation_state
                and validation_state.get("state") == self.VALIDATION_STATE_WORKING
                and validation_state.get("cache_fingerprint") == current_fingerprint
            ):
                return ModelStatus(
                    id=model_id,
                    status="downloaded",
                    path=str(cache_path),
                    size_bytes=size_bytes,
                )

            if validation_state and validation_state.get("cache_fingerprint") != current_fingerprint:
                error_message = (
                    "Model cache changed since last validation. "
                    "Re-run download to validate it."
                )
            else:
                error_message = validation_state.get("error") if validation_state else None
                if not error_message:
                    error_message = (
                        "Model found in cache but not validated. "
                        "Re-run download to validate it."
                    )

            return ModelStatus(
                id=model_id,
                status="error",
                path=str(cache_path),
                size_bytes=size_bytes,
                error=error_message,
            )

        return ModelStatus(
            id=model_id,
            status="not_downloaded",
        )

    def validate_downloaded_model(self, model_id: str) -> None:
        """Validate that a cached model can be fully loaded."""
        self.validate_model(model_id)

        cache_path = self.get_model_cache_path(model_id)
        if not cache_path or not cache_path.exists():
            self._clear_validation_state(model_id)
            raise ModelDownloadError(model_id, "Validation failed: model is not cached")

        snapshot_path = self._get_model_snapshot_path(model_id)
        if not snapshot_path or not snapshot_path.exists():
            fingerprint = self._get_model_cache_fingerprint(model_id)
            error_message = "Validation failed: snapshot path not found in cache"
            self._set_validation_state(
                model_id=model_id,
                state=self.VALIDATION_STATE_BROKEN,
                cache_fingerprint=fingerprint,
                error=error_message,
            )
            raise ModelDownloadError(model_id, error_message)

        try:
            model = load_model(str(snapshot_path))
            del model
            gc.collect()
        except Exception as e:
            fingerprint = self._get_model_cache_fingerprint(model_id)
            error_message = f"Validation failed: {e}"
            self._set_validation_state(
                model_id=model_id,
                state=self.VALIDATION_STATE_BROKEN,
                cache_fingerprint=fingerprint,
                error=error_message,
            )
            raise ModelDownloadError(model_id, error_message)

        fingerprint = self._get_model_cache_fingerprint(model_id)
        self._set_validation_state(
            model_id=model_id,
            state=self.VALIDATION_STATE_WORKING,
            cache_fingerprint=fingerprint,
            error=None,
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
            "error": status.error,
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
        progress = self._download_progress.get(model_id)
        return progress is not None and progress.get("error") is None

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

        status = self.get_model_status(model_id)
        if status.status == "downloaded":
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
                cache_dir=str(self._hf_hub_cache_path),
            )
            self.validate_downloaded_model(model_id)
            self.clear_download_progress(model_id)

        except ModelDownloadError:
            self.clear_download_progress(model_id)
            raise
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

        status = self.get_model_status(model_id)
        if status.status == "downloaded":
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
                cache_dir=str(self._hf_hub_cache_path),
            )
            self.validate_downloaded_model(model_id)
            # Download complete - clear progress tracking
            self.clear_download_progress(model_id)
        except Exception as e:
            # Store error in progress tracking so status resolves to "error".
            if not isinstance(e, ModelDownloadError):
                error_message = f"Download failed: {e}"
                self._set_validation_state(
                    model_id=model_id,
                    state=self.VALIDATION_STATE_BROKEN,
                    cache_fingerprint=self._get_model_cache_fingerprint(model_id),
                    error=error_message,
                )
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

        self.clear_download_progress(model_id)
        self._clear_validation_state(model_id)

        # Also try to clean up the HuggingFace cache metadata
        try:
            # The refs directory stores branch/tag references
            cache_base = self._hf_hub_cache_path
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
