"""
app/core/middleware.py

Request logging middleware.
Assigns X-Request-ID to every request, logs timing and status,
and attaches request_id to request.state for downstream logging.

Every log line for one request shares the same request_id —
searchable across both .NET and Python log streams.
"""
from __future__ import annotations

import time
import uuid

import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)

# Paths skipped from request logging — high-frequency probes create noise
SKIP_LOGGING_PATHS = {
    "/health/live",
    "/api/v1/health/live",
}


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs every request with timing and a short correlation ID.
    Adds X-Request-ID header to all responses for cross-service tracing.

    Log format:
        request_started  → method, path, request_id
        request_complete → method, path, status, elapsed_ms, request_id
        request_error    → method, path, elapsed_ms, request_id, error

    The same request_id appears in:
        - Python logs (every log line that calls request.state.request_id)
        - .NET response header X-Request-ID (for .NET to correlate)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.perf_counter()

        # Attach to request.state so any route handler can log it
        # Usage: request_id = getattr(http_request.state, "request_id", "unknown")
        request.state.request_id = request_id

        skip = request.url.path in SKIP_LOGGING_PATHS

        if not skip:
            logger.info(
                "request_started",
                method=request.method,
                path=request.url.path,
                request_id=request_id,
            )

        try:
            response = await call_next(request)
            elapsed = (time.perf_counter() - start) * 1000

            if not skip:
                logger.info(
                    "request_complete",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    elapsed_ms=round(elapsed, 1),
                    request_id=request_id,
                )

            # Add to response headers — .NET reads this for correlation
            response.headers["X-Request-ID"] = request_id
            return response

        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_error",
                method=request.method,
                path=request.url.path,
                elapsed_ms=round(elapsed, 1),
                request_id=request_id,
                error=str(exc),
            )
            raise  # never swallow — FastAPI error handling must still run
