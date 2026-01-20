"""Middleware for the MLX Whisper API.

Provides:
- Request ID tracking for observability
- Request/response logging
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# Header name for request ID
REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request IDs to all requests.

    Generates a unique request ID for each request and:
    - Stores it in request.state for use in logging and error responses
    - Adds it to the response headers for client tracking
    - Uses client-provided X-Request-ID if present (for request tracing)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Use client-provided request ID or generate a new one
        request_id = request.headers.get(REQUEST_ID_HEADER)
        if not request_id:
            request_id = str(uuid.uuid4())[:8]  # Short ID for readability

        # Store in request state for access in handlers
        request.state.request_id = request_id

        # Process the request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers[REQUEST_ID_HEADER] = request_id

        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all requests and responses.

    Logs:
    - Incoming requests with method, path, and request ID
    - Outgoing responses with status code and duration
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get request ID (set by RequestIDMiddleware)
        request_id = getattr(request.state, "request_id", "unknown")

        # Record start time
        start_time = time.time()

        # Log incoming request
        logger.info(
            "Request: %s %s (request_id=%s)",
            request.method,
            request.url.path,
            request_id,
        )

        # Process the request
        response = await call_next(request)

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Log response
        log_level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            log_level,
            "Response: %s %s -> %d (%.2fms, request_id=%s)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            request_id,
        )

        return response


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application.

    Sets up structured logging with consistent format across all loggers.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Reduce noise from uvicorn access logs (we have our own logging)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
