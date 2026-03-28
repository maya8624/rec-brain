"""
HTTP client for the .NET backend booking API.
All booking state is owned by .NET — Python never stores booking records.
"""
import logging
from datetime import datetime

from app.core.constants import BookingEndpoints
from app.services.backend_client import BackendClient
from app.core.exceptions import BackendClientError, BookingServiceError, BookingValidationError
from app.schemas.booking import (
    AvailableSlot,
    AvailabilityResult,
    BookingConfirmation,
    BookingRequest,
    CancellationConfirmation,
    CancellationRequest,
)

logger = logging.getLogger(__name__)


class BookingService:
    """HTTP client for all booking operations against the .NET backend."""

    def __init__(self, client: BackendClient):
        self._client = client

    # ------------------------------------
    # Check availability
    # ------------------------------------

    async def get_availability(
            self,
            property_id: str,
            preferred_date: str | None = None) -> dict:
        """
        Fetch available inspection slots. Returns AvailabilityResult as dict.
        """

        self._validate_property_id(property_id)
        params = {"propertyId": property_id}

        if preferred_date:
            self._validate_date_string(preferred_date)
            params["preferredDate"] = preferred_date

        logger.info("get_availability | property=%s | date=%s",
                    property_id, preferred_date)

        try:
            data = await self._client.get(BookingEndpoints.AVAILABILITY, params=params)
        except BackendClientError as exc:
            raise BookingServiceError(
                f"Failed to fetch availability: {exc}") from exc

        slots = self._parse_availability(data)
        available_slots = [
            AvailableSlot(
                datetime=slot["datetime"],
                agent_name=slot.get("agent_name", ""),
                available=slot.get("available", True),
            )
            for slot in slots if slot.get("datetime")
        ]

        logger.info("get_availability | found %d slots",
                    len(available_slots))

        return AvailabilityResult(
            success=True,
            property_id=property_id,
            available_slots=available_slots,
            slot_count=len(available_slots),
        ).model_dump()

    # ------------------------------------
    # Booking
    # ------------------------------------

    async def book(self, request: BookingRequest) -> dict:
        """
        Create a confirmed inspection booking.
        Returns BookingConfirmation as dict.
        """

        payload = {
            "propertyId": request.property_id,
            "inspectionDatetime": request.datetime_slot,
            "contact": {
                "name": request.contact.name,
                "email": request.contact.email,
                "phone": request.contact.phone,
            },
        }

        logger.info(
            "book | property=%s | slot=%s | contact=%s",
            request.property_id, request.datetime_slot, request.contact.email,
        )

        try:
            data = await self._client.post(BookingEndpoints.BOOK, json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Booking failed: {exc}") from exc

        confirmation = self._parse_booking_response(data)

        logger.info("book | confirmed | id=%s",
                    confirmation.confirmation_id)

        return confirmation.model_dump()

    # ------------------------------------
    # Cancel booking
    # ------------------------------------

    async def cancel(self, request: CancellationRequest) -> dict:
        """Cancel an existing booking. Returns CancellationConfirmation as dict."""

        payload: dict = {"confirmationId": request.confirmation_id}

        if request.reason:
            payload["reason"] = request.reason

        logger.info("cancel | id=%s", request.confirmation_id)

        try:
            await self._client.post(BookingEndpoints.CANCEL, json=payload)
        except BackendClientError as exc:
            raise BookingServiceError(f"Cancellation failed: {exc}") from exc

        logger.info("cancel | cancelled | id=%s", request.confirmation_id)

        return CancellationConfirmation(
            confirmation_id=request.confirmation_id,
        ).model_dump()

    # ------------------------------------
    # Validataion helpers
    # ------------------------------------

    def _validate_property_id(self, property_id: str):
        if not property_id or not property_id.strip():
            raise BookingValidationError("property_id is required")

    def _validate_date_string(self, date_str: str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")

        except ValueError:
            raise BookingValidationError(
                f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD"
            ) from None

    # ------------------------------------
    # Response parsers
    # ------------------------------------

    def _parse_availability(self, data: dict) -> list[dict]:
        """Normalise .NET availability response to a clean list of slot dicts."""
        slots = data.get("availableSlots") or []

        return [
            {
                "datetime": slot.get("datetime"),
                "available": slot.get("available", True),
                "agent_name": slot.get("agentName", ""),
            }
            for slot in slots
            if slot.get("available", True)
        ]

    def _parse_booking_response(self, data: dict) -> BookingConfirmation:
        """Normalise .NET booking confirmation response into a typed model."""

        return BookingConfirmation(
            confirmation_id=data.get("confirmationId", ""),
            property_address=data.get("propertyAddress", ""),
            confirmed_datetime=data.get("confirmedDatetime", ""),
            agent_name=data.get("agentName", ""),
            agent_phone=data.get("agentPhone", ""),
        )
