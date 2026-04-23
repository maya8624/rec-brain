"""
HTTP client for the .NET backend booking API.
All booking state is owned by .NET — Python never stores booking records.
"""
import logging
import uuid

from app.core.constants import InternalRoutes
from app.services.backend_client import BackendClient
from app.core.exceptions import BackendClientError, BookingServiceError, BookingValidationError
from app.schemas.booking import (
    AvailableSlot,
    AvailableSlotList,
    AvailabilityResult,
    BookingConfirmation,
    BookingRequest,
    CancellationConfirmation,
)

logger = logging.getLogger(__name__)


class BookingService:
    """HTTP client for all booking operations against the .NET backend."""

    def __init__(self, client: BackendClient):
        self._client = client

    async def get_availability(self, property_id: str) -> AvailabilityResult:
        try:
            uuid.UUID(property_id)
        except ValueError as exc:
            raise BookingValidationError(
                f"property_id must be a valid UUID: {exc}") from exc

        try:
            data = await self._client.get(
                InternalRoutes.AVAILABLE,
                params={"propertyId": property_id},
            )
        except BackendClientError as exc:
            raise BookingServiceError(
                f"Failed to fetch availability: {exc}"
            ) from exc

        available_slots = self._parse_availability(data)
        count = len(available_slots)

        logger.info(
            "get_availability | property_id=%s slots=%d",
            property_id, count
        )

        return AvailabilityResult(
            success=True,
            property_id=property_id,
            available_slots=available_slots,
            slot_count=count,
        )

    async def book(self, request: BookingRequest) -> dict:
        """
        Create a confirmed inspection booking.
        Returns BookingConfirmation as dict.
        """

        payload = {
            "InspectionSlotId": request.slot_id,
            "UserId": request.user_id,
            "Notes": request.notes,
        }

        try:
            data = await self._client.post(InternalRoutes.BOOK, json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Booking failed: {exc}") from exc

        confirmation = self._parse_booking_response(data)
        logger.info("book | confirmed | id=%s", confirmation.confirmation_id)

        return confirmation.model_dump()

    async def cancel(self, confirmation_id: str, user_id: str) -> CancellationConfirmation:
        """Cancel an existing booking."""
        try:
            payload: dict = {"UserId": user_id}
            await self._client.patch(InternalRoutes.cancel(confirmation_id), json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Cancellation failed: {exc}") from exc

        logger.info("cancel | cancelled | id=%s", confirmation_id)

        return CancellationConfirmation(
            id=confirmation_id,
            success=True,
        )

    def _parse_availability(self, data: list) -> list[AvailableSlot]:
        """Parse .NET InspectionSlotDto list directly into AvailableSlot objects."""
        slots = AvailableSlotList.validate_python(data)
        return [s for s in slots if s.available and s.start_at]

    def _parse_booking_response(self, data: dict) -> BookingConfirmation:
        """Normalise .NET booking confirmation response into a typed model."""

        return BookingConfirmation(
            confirmation_id=data.get("id", ""),
            property_id=data.get("propertyId", ""),
            status=data.get("status", ""),
            agent_first_name=data.get("agentFirstName", ""),
            agent_last_name=data.get("agentLastName", ""),
            agent_phone=data.get("agentPhone"),
            start_at_utc=data.get("startAtUtc"),
            end_at_utc=data.get("endAtUtc"),
        )
