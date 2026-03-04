"""
Async HTTP client for the .NET backend API.
Handles auth headers, retries, and error logging.
"""

import httpx
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import logging
from functools import wraps
from typing import Any

from core.config import settings
from core.exceptions import BackendClientError

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
                "Authorization": f"Bearer {settings.BACKEND_API_KEY}",
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
                "BackendClient not initialized. Call initialize() first.")

    @async_retry
    async def get(self, path: str, params: dict | None = None) -> Any:
        self._ensure_ready()
        try:
            resp = await self._client.get(path, params=params)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("GET failed", path=path,
                         status=e.response.status_code)
            raise BackendClientError(
                str(e), status_code=e.response.status_code)

    @async_retry
    async def post(self, path: str, body: dict) -> Any:
        self._ensure_ready()
        try:
            resp = await self._client.post(path, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("POST failed", path=path,
                         status=e.response.status_code)
            raise BackendClientError(
                str(e), status_code=e.response.status_code)

    @async_retry
    async def patch(self, path: str, body: dict) -> Any:
        self._ensure_ready()
        try:
            resp = await self._client.patch(path, json=body)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            logger.error("PATCH failed", path=path,
                         status=e.response.status_code)
            raise BackendClientError(
                str(e), status_code=e.response.status_code)


# Singleton — initialized at app startup via lifespan
backend_client = BackendClient()
