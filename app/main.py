"""FastAPI application entry point.

MLX Whisper API - REST API for audio-to-text transcription
using MLX-optimized Whisper models on Apple Silicon.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.routers import transcribe, models
from app.schemas.models import HealthResponse, ErrorResponse
from app.errors import APIException, api_exception_handler, unhandled_exception_handler
from app.middleware import RequestIDMiddleware, LoggingMiddleware, setup_logging

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
setup_logging(LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    import logging
    logger = logging.getLogger(__name__)
    logger.info("MLX Whisper API starting up...")
    yield
    # Shutdown
    logger.info("MLX Whisper API shutting down...")


# OpenAPI documentation customization
API_TITLE = "MLX Whisper API"
API_DESCRIPTION = """
REST API for audio-to-text transcription using MLX-optimized Whisper models on Apple Silicon.

## Features

- **Audio Transcription**: Transcribe audio files to text with support for multiple formats (WAV, MP3, M4A, FLAC, OGG)
- **Model Management**: List, download, and manage Whisper models from HuggingFace
- **Language Support**: Automatic language detection or specify target language
- **Prompt Support**: Provide context to guide transcription

## Error Handling

All errors return a consistent JSON structure with:
- `error`: Human-readable error message
- `code`: Machine-readable error code (e.g., `MODEL_NOT_DOWNLOADED`)
- `details`: Additional context (optional)
- `request_id`: Request ID for tracking (optional)

## Authentication

Currently, this API does not require authentication. For production deployments,
consider adding authentication middleware.
"""

API_VERSION = "0.2.0"

OPENAPI_TAGS = [
    {
        "name": "transcription",
        "description": "Audio transcription operations",
    },
    {
        "name": "models",
        "description": "Model management operations - list, download, and delete models",
    },
    {
        "name": "health",
        "description": "Health check endpoint",
    },
]

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    openapi_tags=OPENAPI_TAGS,
    lifespan=lifespan,
    responses={
        500: {
            "model": ErrorResponse,
            "description": "Internal server error",
        },
    },
)

# Add middleware (order matters - first added is outermost)
app.add_middleware(LoggingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Register exception handlers
app.add_exception_handler(APIException, api_exception_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

# Include routers
app.include_router(transcribe.router, tags=["transcription"])
app.include_router(models.router)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
    description="Check if the API service is running and healthy.",
    responses={
        200: {
            "description": "Service is healthy",
            "content": {
                "application/json": {
                    "example": {"status": "healthy"}
                }
            }
        }
    },
)
async def health_check(request: Request) -> HealthResponse:
    """Health check endpoint.

    Returns a simple status response indicating the service is running.
    This endpoint can be used for:
    - Load balancer health checks
    - Container orchestration liveness probes
    - Monitoring systems

    Returns:
        HealthResponse with status "healthy"
    """
    return HealthResponse(status="healthy")
