"""Model management endpoint router.

Provides endpoints for listing, downloading, and deleting MLX Whisper models.
"""

import logging
from urllib.parse import unquote

from fastapi import APIRouter, Path, Request

from app.schemas.models import (
    ModelInfo,
    ModelListResponse,
    ModelStatusResponse,
    DownloadResponse,
    DeleteResponse,
    ErrorResponse,
)
from app.errors import (
    ModelNotFoundError as APIModelNotFoundError,
    ModelAlreadyDownloadedError as APIModelAlreadyDownloadedError,
    ModelNotDownloadedError as APIModelNotDownloadedError,
)
from app.services.model_manager import (
    get_model_manager,
    ModelNotFoundError,
    ModelAlreadyDownloadedError,
    ModelNotDownloadedError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])


@router.get(
    "",
    response_model=ModelListResponse,
    summary="List all supported models",
    description="""
List all supported MLX Whisper models with their current status.

Each model includes:
- **id**: Unique model identifier (HuggingFace repo path)
- **name**: Human-readable name
- **size**: Model size category (tiny, base, small, medium, large, turbo)
- **quantization**: Quantization level if applicable (q4, 8bit, etc.)
- **english_only**: Whether the model only supports English
- **status**: Current status (downloaded, not_downloaded, downloading, error)
- **size_bytes**: Size on disk for downloaded models
- **download_progress**: Progress (0.0-1.0) for models being downloaded
""",
    responses={
        200: {
            "description": "List of all supported models",
        },
    },
)
async def list_models(request: Request) -> ModelListResponse:
    """List all supported models and their status.

    Returns a list of all supported MLX Whisper models with metadata
    parsed from the model ID and current download status.

    Args:
        request: FastAPI request object

    Returns:
        ModelListResponse containing list of all models with metadata and status
    """
    request_id = getattr(request.state, "request_id", None)
    manager = get_model_manager()

    models = manager.list_models()

    logger.debug(
        "Listed %d models (request_id=%s)",
        len(models),
        request_id,
    )

    return ModelListResponse(models=[ModelInfo(**m) for m in models])


@router.get(
    "/{model_id:path}/status",
    response_model=ModelStatusResponse,
    summary="Get model status",
    description="""
Get detailed status information for a specific model.

Returns:
- **downloaded**: Model is available locally with path and size
- **not_downloaded**: Model needs to be downloaded
- **downloading**: Model download in progress with progress info
- **error**: Download failed with error message
""",
    responses={
        200: {
            "description": "Model status information",
        },
        404: {
            "model": ErrorResponse,
            "description": "Model not found in supported list",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Model not found: invalid-model",
                        "code": "MODEL_NOT_FOUND",
                        "details": {"model": "invalid-model"}
                    }
                }
            }
        },
    },
)
async def get_model_status(
    request: Request,
    model_id: str = Path(
        ...,
        description="Model identifier (URL-encoded if contains '/'). Example: mlx-community%2Fwhisper-tiny-mlx",
        examples=["mlx-community/whisper-tiny-mlx"],
    ),
) -> ModelStatusResponse:
    """Get detailed status of a specific model.

    Checks the HuggingFace cache to determine if the model is downloaded
    and returns detailed status information.

    Args:
        request: FastAPI request object
        model_id: Model identifier (e.g., "mlx-community/whisper-tiny-mlx")

    Returns:
        ModelStatusResponse with detailed status information

    Raises:
        ModelNotFoundError: If model is not in the supported list
    """
    request_id = getattr(request.state, "request_id", None)

    # URL-decode the model_id (handles %2F -> /)
    model_id = unquote(model_id)

    manager = get_model_manager()

    try:
        status = manager.get_model_status(model_id)

        logger.debug(
            "Model status: %s -> %s (request_id=%s)",
            model_id,
            status.status,
            request_id,
        )

        return ModelStatusResponse(
            id=status.id,
            status=status.status,
            path=status.path,
            size_bytes=status.size_bytes,
            progress=status.progress,
            downloaded_bytes=status.downloaded_bytes,
            total_bytes=status.total_bytes,
        )
    except ModelNotFoundError as e:
        logger.warning(
            "Model not found: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise APIModelNotFoundError(e.model_id)


@router.post(
    "/{model_id:path}/download",
    response_model=DownloadResponse,
    summary="Download a model",
    description="""
Initiate download of a model from HuggingFace Hub.

The download runs in the background. Use `GET /models/{model_id}/status`
to check progress.

**Note**: Downloads may take several minutes depending on model size
and network speed. Large models (e.g., whisper-large-v3) are ~3GB.
""",
    responses={
        200: {
            "description": "Download initiated successfully",
        },
        404: {
            "model": ErrorResponse,
            "description": "Model not found in supported list",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Model not found: invalid-model",
                        "code": "MODEL_NOT_FOUND",
                        "details": {"model": "invalid-model"}
                    }
                }
            }
        },
        409: {
            "model": ErrorResponse,
            "description": "Model is already downloaded",
            "content": {
                "application/json": {
                    "example": {
                        "error": "Model already downloaded: mlx-community/whisper-tiny-mlx",
                        "code": "MODEL_ALREADY_DOWNLOADED",
                        "details": {"model": "mlx-community/whisper-tiny-mlx"}
                    }
                }
            }
        },
    },
)
async def download_model(
    request: Request,
    model_id: str = Path(
        ...,
        description="Model identifier (URL-encoded if contains '/'). Example: mlx-community%2Fwhisper-tiny-mlx",
        examples=["mlx-community/whisper-tiny-mlx"],
    ),
) -> DownloadResponse:
    """Initiate download of a model.

    The download runs in the background using a separate thread.
    Check the status with GET /models/{model_id}/status.

    Args:
        request: FastAPI request object
        model_id: Model identifier (e.g., "mlx-community/whisper-tiny-mlx")

    Returns:
        DownloadResponse with status information

    Raises:
        ModelNotFoundError: If model is not in the supported list
        ModelAlreadyDownloadedError: If model is already downloaded
    """
    request_id = getattr(request.state, "request_id", None)

    # URL-decode the model_id (handles %2F -> /)
    model_id = unquote(model_id)

    manager = get_model_manager()

    try:
        manager.start_download_async(model_id)

        logger.info(
            "Started download: %s (request_id=%s)",
            model_id,
            request_id,
        )

        return DownloadResponse(
            id=model_id,
            status="download_started",
            message=f"Model download initiated. Check /models/{model_id}/status for progress.",
        )
    except ModelNotFoundError as e:
        logger.warning(
            "Download failed - model not found: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise APIModelNotFoundError(e.model_id)
    except ModelAlreadyDownloadedError as e:
        logger.info(
            "Download skipped - already downloaded: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise APIModelAlreadyDownloadedError(e.model_id)


@router.delete(
    "/{model_id:path}",
    response_model=DeleteResponse,
    summary="Delete a model",
    description="""
Remove a downloaded model from the local cache.

This frees up disk space by removing all model files from the
HuggingFace cache directory.

**Note**: The model can be re-downloaded at any time using the
download endpoint.
""",
    responses={
        200: {
            "description": "Model deleted successfully",
        },
        404: {
            "model": ErrorResponse,
            "description": "Model not found or not downloaded",
            "content": {
                "application/json": {
                    "examples": {
                        "not_found": {
                            "summary": "Model not in supported list",
                            "value": {
                                "error": "Model not found: invalid-model",
                                "code": "MODEL_NOT_FOUND",
                                "details": {"model": "invalid-model"}
                            }
                        },
                        "not_downloaded": {
                            "summary": "Model not downloaded",
                            "value": {
                                "error": "Model not downloaded: mlx-community/whisper-tiny-mlx",
                                "code": "MODEL_NOT_DOWNLOADED",
                                "details": {"model": "mlx-community/whisper-tiny-mlx"}
                            }
                        }
                    }
                }
            }
        },
    },
)
async def delete_model(
    request: Request,
    model_id: str = Path(
        ...,
        description="Model identifier (URL-encoded if contains '/'). Example: mlx-community%2Fwhisper-tiny-mlx",
        examples=["mlx-community/whisper-tiny-mlx"],
    ),
) -> DeleteResponse:
    """Delete a downloaded model from the cache.

    Removes all model files from the HuggingFace cache directory
    to free up disk space.

    Args:
        request: FastAPI request object
        model_id: Model identifier (e.g., "mlx-community/whisper-tiny-mlx")

    Returns:
        DeleteResponse confirming deletion

    Raises:
        ModelNotFoundError: If model is not in the supported list
        ModelNotDownloadedError: If model is not downloaded
    """
    request_id = getattr(request.state, "request_id", None)

    # URL-decode the model_id (handles %2F -> /)
    model_id = unquote(model_id)

    manager = get_model_manager()

    try:
        manager.delete_model(model_id)

        logger.info(
            "Deleted model: %s (request_id=%s)",
            model_id,
            request_id,
        )

        return DeleteResponse(
            id=model_id,
            status="deleted",
        )
    except ModelNotFoundError as e:
        logger.warning(
            "Delete failed - model not found: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise APIModelNotFoundError(e.model_id)
    except ModelNotDownloadedError as e:
        logger.info(
            "Delete skipped - not downloaded: %s (request_id=%s)",
            e.model_id,
            request_id,
        )
        raise APIModelNotDownloadedError(e.model_id)
