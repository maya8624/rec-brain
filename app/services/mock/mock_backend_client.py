"""
Mock backend client — replaces the real BackendClient in local development.
Same interface as backend_client.py so tools work without any changes.
Reads from mock_data.py instead of calling the .NET API.
"""

import structlog
from typing import Any

from app.services.mock.mock_data import (
    get_availability_response,
    get_create_appointment_response,
    get_cancel_appointment_response,
)

logger = structlog.get_logger(__name__)


class MockBackendClient:
    """
    Drop-in replacement for BackendClient.
    Implements the same interface (initialize, close, get, post, patch)
    but returns mock data instead of making real HTTP calls.
    """

    async def initialize(self) -> None:
        logger.info(
            "MockBackendClient initialized — no real HTTP calls will be made")

    async def close(self) -> None:
        logger.info("MockBackendClient closed")

    async def get(self, path: str, params: dict | None = None) -> Any:
        """Route GET requests to mock data based on path."""
        log = logger.bind(method="GET", path=path, params=params)
        params = params or {}

        # GET /api/appointments/availability
        if "availability" in path:
            response = get_availability_response(
                property_id=params.get("propertyId", "PROP-001"),
                date_str=params.get("date", ""),
                time_str=params.get("time", ""),
            )
            log.info("Mock availability response",
                     available=response.get("available"))
            return response

        log.warning("Unhandled mock GET path", path=path)
        return {}

    async def post(self, path: str, body: dict) -> Any:
        """Route POST requests to mock data based on path."""
        log = logger.bind(method="POST", path=path)

        # POST /api/appointments
        if path == "/api/appointments":
            response = get_create_appointment_response(body)
            log.info("Mock appointment created",
                     appointment_id=response["appointment"]["id"])
            return response

        log.warning("Unhandled mock POST path", path=path)
        return {}

    async def patch(self, path: str, body: dict) -> Any:
        """Route PATCH requests to mock data based on path."""
        log = logger.bind(method="PATCH", path=path)

        # PATCH /api/appointments/{id}/status
        if "appointments" in path and "status" in path:
            appointment_id = path.split("/")[3]  # extract ID from path
            response = get_cancel_appointment_response(appointment_id, body)
            log.info("Mock appointment updated", appointment_id=appointment_id)
            return response

        log.warning("Unhandled mock PATCH path", path=path)
        return {}


# Singleton — same pattern as the real backend_client
mock_backend_client = MockBackendClient()
