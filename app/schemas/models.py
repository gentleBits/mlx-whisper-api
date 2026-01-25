"""Pydantic models for request/response schemas.

This module defines all API request and response schemas with
comprehensive OpenAPI documentation including examples and descriptions.
"""

from enum import Enum
from typing import Optional, Any

from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Status of a model in the system."""
    DOWNLOADED = "downloaded"
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING = "downloading"
    ERROR = "error"


class ModelSize(str, Enum):
    """Size category of a Whisper model."""
    TINY = "tiny"
    SMALL = "small"
    LARGE = "large"


class TranscriptionResponse(BaseModel):
    """Response schema for successful transcription.

    Contains the transcribed text along with metadata about
    the transcription process.
    """
    text: str = Field(
        ...,
        description="The transcribed text from the audio file",
        examples=["Hello, this is a test transcription."]
    )
    language: str = Field(
        ...,
        description="Detected or specified language code (ISO 639-1)",
        examples=["en"]
    )
    model: str = Field(
        ...,
        description="Model ID used for transcription",
        examples=["mlx-community/whisper-large-v3-mlx"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Hello, this is a test transcription.",
                    "language": "en",
                    "model": "mlx-community/whisper-large-v3-mlx"
                }
            ]
        }
    }


class ModelInfo(BaseModel):
    """Detailed information about a supported model.

    Includes metadata parsed from the model ID and current status.
    """
    id: str = Field(
        ...,
        description="Unique model identifier (HuggingFace repo path)",
        examples=["mlx-community/whisper-large-v3-mlx"]
    )
    name: str = Field(
        ...,
        description="Human-readable model name",
        examples=["Whisper Large V3"]
    )
    size: str = Field(
        ...,
        description="Model size category (tiny, small, large)",
        examples=["large"]
    )
    quantization: Optional[str] = Field(
        default=None,
        description="Quantization level if applicable",
        examples=[None]
    )
    english_only: bool = Field(
        default=False,
        description="Whether this model only supports English"
    )
    status: str = Field(
        ...,
        description="Current status: downloaded, not_downloaded, downloading, error",
        examples=["downloaded"]
    )
    size_bytes: Optional[int] = Field(
        default=None,
        description="Size in bytes for downloaded models",
        examples=[3100000000]
    )
    download_progress: Optional[float] = Field(
        default=None,
        description="Download progress (0.0 to 1.0) when downloading",
        ge=0.0,
        le=1.0,
        examples=[0.45]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mlx-community/whisper-large-v3-mlx",
                    "name": "Whisper Large V3",
                    "size": "large",
                    "quantization": None,
                    "english_only": False,
                    "status": "downloaded",
                    "size_bytes": 3100000000,
                    "download_progress": None
                }
            ]
        }
    }


class ModelListResponse(BaseModel):
    """Response containing a list of all supported models."""
    models: list[ModelInfo] = Field(
        ...,
        description="List of all supported models with their status"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "models": [
                        {
                            "id": "mlx-community/whisper-tiny-mlx",
                            "name": "Whisper Tiny",
                            "size": "tiny",
                            "quantization": None,
                            "english_only": False,
                            "status": "not_downloaded",
                            "size_bytes": None,
                            "download_progress": None
                        },
                        {
                            "id": "mlx-community/whisper-large-v3-mlx",
                            "name": "Whisper Large V3",
                            "size": "large",
                            "quantization": None,
                            "english_only": False,
                            "status": "downloaded",
                            "size_bytes": 3100000000,
                            "download_progress": None
                        }
                    ]
                }
            ]
        }
    }


class ModelStatusResponse(BaseModel):
    """Detailed status response for a specific model."""
    id: str = Field(
        ...,
        description="Model identifier",
        examples=["mlx-community/whisper-large-v3-mlx"]
    )
    status: str = Field(
        ...,
        description="Current status: downloaded, not_downloaded, downloading, error",
        examples=["downloaded"]
    )
    path: Optional[str] = Field(
        default=None,
        description="Local cache path for downloaded models",
        examples=["/Users/user/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-mlx"]
    )
    size_bytes: Optional[int] = Field(
        default=None,
        description="Total size in bytes for downloaded models",
        examples=[3100000000]
    )
    progress: Optional[float] = Field(
        default=None,
        description="Download progress (0.0 to 1.0) when downloading",
        ge=0.0,
        le=1.0,
        examples=[0.67]
    )
    downloaded_bytes: Optional[int] = Field(
        default=None,
        description="Bytes downloaded so far when downloading",
        examples=[2077000000]
    )
    total_bytes: Optional[int] = Field(
        default=None,
        description="Total bytes to download when downloading",
        examples=[3100000000]
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mlx-community/whisper-large-v3-mlx",
                    "status": "downloaded",
                    "path": "/Users/user/.cache/huggingface/hub/models--mlx-community--whisper-large-v3-mlx",
                    "size_bytes": 3100000000,
                    "progress": None,
                    "downloaded_bytes": None,
                    "total_bytes": None,
                    "error": None
                }
            ]
        }
    }


class DownloadResponse(BaseModel):
    """Response when initiating a model download."""
    id: str = Field(
        ...,
        description="Model identifier",
        examples=["mlx-community/whisper-tiny-mlx"]
    )
    status: str = Field(
        ...,
        description="Current status after initiating download",
        examples=["download_started"]
    )
    message: str = Field(
        ...,
        description="Human-readable status message",
        examples=["Model download initiated. Check /models/mlx-community%2Fwhisper-tiny-mlx/status for progress."]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mlx-community/whisper-tiny-mlx",
                    "status": "download_started",
                    "message": "Model download initiated. Check /models/mlx-community%2Fwhisper-tiny-mlx/status for progress."
                }
            ]
        }
    }


class DeleteResponse(BaseModel):
    """Response when deleting a model."""
    id: str = Field(
        ...,
        description="Model identifier",
        examples=["mlx-community/whisper-tiny-mlx"]
    )
    status: str = Field(
        ...,
        description="Status after deletion",
        examples=["deleted"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": "mlx-community/whisper-tiny-mlx",
                    "status": "deleted"
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response structure.

    All API errors follow this format for consistency.
    The 'code' field provides a machine-readable error identifier.
    """
    error: str = Field(
        ...,
        description="Human-readable error message",
        examples=["Model not downloaded"]
    )
    code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=["MODEL_NOT_DOWNLOADED"]
    )
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="Additional error context and details",
        examples=[{"model": "mlx-community/whisper-tiny-mlx", "download_url": "/models/mlx-community%2Fwhisper-tiny-mlx/download"}]
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking and debugging",
        examples=["a1b2c3d4"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "error": "Model not downloaded",
                    "code": "MODEL_NOT_DOWNLOADED",
                    "details": {
                        "model": "mlx-community/whisper-tiny-mlx",
                        "download_url": "/models/mlx-community%2Fwhisper-tiny-mlx/download"
                    },
                    "request_id": "a1b2c3d4"
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(
        ...,
        description="Health status of the service",
        examples=["healthy"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "healthy"
                }
            ]
        }
    }
