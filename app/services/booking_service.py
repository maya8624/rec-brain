"""
HTTP client for the .NET backend booking API.
All booking state is owned by .NET — Python never stores booking records.
"""
import logging

from app.core.constants import InternalRoutes
from app.services.backend_client import BackendClient
from app.core.exceptions import BackendClientError, BookingServiceError
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

    async def check_availability(self, property_id: str) -> AvailabilityResult:
        """
        Fetch available inspection slots for a property.
        """
        try:
            url = f"{InternalRoutes.AVAILABLE}/{property_id}"
            data = await self._client.get(url)
        except BackendClientError as exc:
            raise BookingServiceError(
                f"Failed to fetch availability: {exc}") from exc

        available_slots = self._parse_availability(data)
        count = len(available_slots)

        return AvailabilityResult(
            success=True,
            property_id=property_id,
            available_slots=available_slots,
            slot_count=count,
        )

    async def book(self, request: BookingRequest) -> BookingConfirmation:
        """
        Create a confirmed inspection booking.
        """
        try:
            payload = {
                "InspectionSlotId": request.slot_id,
                "UserId": request.user_id,
                "Notes": request.notes,
            }
            data = await self._client.post(InternalRoutes.BOOK, json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Booking failed: {exc}") from exc

        response = self._parse_booking_response(data)

        return response

    async def get_booking(self, confirmation_id: str, user_id: str) -> BookingConfirmation:
        """Fetch a single booking by confirmation ID."""
        try:
            url = InternalRoutes.get_booking(confirmation_id)
            data = await self._client.get(url, params={"userId": user_id})
        except BackendClientError as exc:
            raise BookingServiceError(
                f"Failed to fetch booking: {exc}") from exc

        return self._parse_booking_response(data)

    async def get_my_bookings(self, user_id: str) -> list[BookingConfirmation]:
        """Fetch all bookings for the current user."""
        try:
            data = await self._client.get(f"{InternalRoutes.MY_BOOKINGS}/{user_id}")
        except BackendClientError as exc:
            raise BookingServiceError(
                f"Failed to fetch bookings: {exc}") from exc

        if not isinstance(data, list):
            raise BookingServiceError(
                "Unexpected response format from bookings endpoint")

        return [self._parse_booking_response(item) for item in data]

    async def cancel(self, confirmation_id: str, user_id: str) -> CancellationConfirmation:
        """Cancel an existing booking."""
        try:
            payload = {"UserId": user_id}
            await self._client.patch(InternalRoutes.cancel(confirmation_id), json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Cancellation failed: {exc}") from exc

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
            property_address=data.get("propertyAddress", ""),
            status=data.get("status", ""),
            agent_first_name=data.get("agentFirstName", ""),
            agent_last_name=data.get("agentLastName", ""),
            agent_phone=data.get("agentPhone", ""),
            start_at_utc=data.get("startAtUtc"),
            end_at_utc=data.get("endAtUtc"),
        )
