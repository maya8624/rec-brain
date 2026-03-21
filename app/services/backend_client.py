"""
Async HTTP client for the .NET backend API.
Handles auth headers, retries, and error logging.
"""
import logging
from functools import wraps
from typing import Any

import httpx
import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import settings
from app.core.exceptions import BackendClientError

logger = structlog.get_logger(__name__)


def async_retry(func):
    """Retry on transient HTTP/network errors with exponential backoff."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (httpx.TransportError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(
            logging.getLogger(__name__), logging.WARNING),
        reraise=True,
    )
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


class BackendClient:
    """
    Singleton async HTTP client for the .NET real estate backend.
    Initialize at app startup, close at shutdown.
    """

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.BACKEND_BASE_URL,
            headers={
                "X-Api-Key": settings.BACKEND_API_KEY,
                "Content-Type": "application/json",
                "X-Service": "python-ai-service",
            },
            timeout=httpx.Timeout(10.0, connect=5.0),
        )
        logger.info("BackendClient initialized",
                    base_url=settings.BACKEND_BASE_URL)

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("BackendClient closed")

    def _ensure_ready(self) -> None:
        if not self._client:
            raise RuntimeError(
                "BackendClient not initialized. Call initialize() first."
            )

    @async_retry
    async def _request(self, method: str, path: str, **kwargs) -> Any:
        self._ensure_ready()
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(
                "request failed",
                method=method,
                path=path,
                status=e.response.status_code,
            )
            raise BackendClientError(
                str(e), status_code=e.response.status_code)

    async def get(self, path: str, params: dict | None = None) -> Any:
        return await self._request("GET", path, params=params)

    async def post(self, path: str, body: dict) -> Any:
        return await self._request("POST", path, json=body)

    async def patch(self, path: str, body: dict) -> Any:
        return await self._request("PATCH", path, json=body)


# Singleton — initialized at app startup via lifespan
backend_client = BackendClient()
