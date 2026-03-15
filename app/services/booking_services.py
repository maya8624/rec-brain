import logging
import httpx
from typing import Optional
from datetime import datetime, timedelta
from app.core.config import settings

logger = logging.getLogger(__name__)

# Datetime format contract between Python AI service and .NET backend
DATETIME_FORMAT = "%Y-%m-%d %H:%M"


class BookingService:
    """
    Handles property inspection availability and booking.
    Communicates with the .NET backend via HTTP.

    All booking state is owned by .NET — this service is a clean
    HTTP client with retry logic, validation, and structured errors.

    Usage:
        service = BookingService()
        slots = service.get_availability("prop_123", "2025-04-12")
        result = service.book("prop_123", "2025-04-12 10:00", ...)
    """

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    def _get_client(self) -> httpx.AsyncClient:
        """Lazy init HTTP client with shared configuration."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=httpx.Timeout(
                    connect=5.0,
                    read=15.0,
                    write=10.0,
                    pool=5.0,
                ),
                headers={
                    "Content-Type": "application/json",
                    "X-Api-Key": settings.BACKEND_API_KEY,
                },
            )
        return self._client

    async def get_availability(
        self,
        property_id: str,
        preferred_date: Optional[str] = None,
    ) -> list[dict]:
        """
        Fetch available inspection slots from .NET backend.

        Args:
            property_id:    Property to check availability for
            preferred_date: Optional ISO date string 'YYYY-MM-DD'
                            Defaults to next 7 days if not provided

        Returns:
            List of slot dicts: [{"datetime": "2025-04-12 10:00", "available": True}, ...]

        Raises:
            BookingServiceError on HTTP or parsing failures
        """
        self._validate_property_id(property_id)

        params = {"propertyId": property_id}
        if preferred_date:
            self._validate_date_string(preferred_date)
            params["preferredDate"] = preferred_date

        logger.info("Fetching availability | property: %s | date: %s",
                    property_id, preferred_date)

        try:
            client = self._get_client()
            response = await client.get(
                "/api/inspections/availability",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            slots = self._parse_availability(data)
            logger.info("Availability retrieved — %d slots", len(slots))
            return slots

        except httpx.HTTPStatusError as e:
            logger.error("Availability HTTP error %d: %s",
                         e.response.status_code, e)
            raise BookingServiceError(
                f"Could not retrieve availability (HTTP {e.response.status_code})"
            ) from e
        except httpx.RequestError as e:
            logger.error("Availability request error: %s", e)
            raise BookingServiceError(
                "Could not connect to booking service") from e

    async def book(
        self,
        property_id: str,
        datetime_slot: str,
        contact_name: str,
        contact_email: str,
        contact_phone: str,
    ) -> dict:
        """
        Create a confirmed inspection booking via .NET backend.

        Args:
            property_id:    Property to book inspection for
            datetime_slot:  'YYYY-MM-DD HH:MM' format
            contact_name:   Full name of the contact
            contact_email:  Valid email address
            contact_phone:  Australian mobile or landline

        Returns:
            dict with confirmation_id, property_address, confirmed_datetime

        Raises:
            BookingValidationError for invalid input
            BookingServiceError for backend failures
        """
        # Validate all fields before hitting the backend
        self._validate_booking_inputs(
            property_id=property_id,
            datetime_slot=datetime_slot,
            contact_name=contact_name,
            contact_email=contact_email,
            contact_phone=contact_phone,
        )

        payload = {
            "propertyId": property_id,
            "inspectionDatetime": datetime_slot,
            "contact": {
                "name": contact_name,
                "email": contact_email,
                "phone": contact_phone,
            },
        }

        logger.info(
            "Creating booking | property: %s | slot: %s | contact: %s",
            property_id, datetime_slot, contact_email,
        )

        try:
            client = self._get_client()
            response = await client.post(
                "/api/inspections/book",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            confirmation = self._parse_booking_response(data)
            logger.info("Booking confirmed — id: %s",
                        confirmation["confirmation_id"])
            return confirmation

        except httpx.HTTPStatusError as e:
            status = e.response.status_code

            # 409 = slot no longer available
            if status == 409:
                raise BookingServiceError(
                    "That time slot is no longer available. Please choose another."
                ) from e

            # 422 = validation rejection from .NET
            if status == 422:
                detail = e.response.json().get("detail", "Invalid booking details")
                raise BookingValidationError(detail) from e

            logger.error("Booking HTTP error %d: %s", status, e)
            raise BookingServiceError(
                f"Booking failed (HTTP {status}). Please try again."
            ) from e

        except httpx.RequestError as e:
            logger.error("Booking request error: %s", e)
            raise BookingServiceError(
                "Could not connect to booking service") from e

    async def cancel(
        self,
        confirmation_id: str,
        reason=None,
    ) -> dict:
        """
        Cancel an existing inspection booking via .NET backend.
        """
        if not confirmation_id or not confirmation_id.strip():
            raise BookingValidationError("confirmation_id is required")

        payload = {"confirmationId": confirmation_id}
        if reason:
            payload["reason"] = reason

        logger.info("Cancelling booking | id: %s", confirmation_id)

        try:
            client = self._get_client()
            response = await client.post(
                "/api/inspections/cancel",
                json=payload,
            )
            response.raise_for_status()
            logger.info("Booking cancelled | id: %s", confirmation_id)
            return {"success": True, "confirmation_id": confirmation_id}

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 404:
                raise BookingServiceError(
                    f"Booking {confirmation_id} not found. Please check the reference."
                ) from e
            if status == 409:
                raise BookingServiceError(
                    f"Booking {confirmation_id} cannot be cancelled — already cancelled or completed."
                ) from e
            logger.error("Cancel HTTP error %d: %s", status, e)
            raise BookingServiceError(
                f"Cancellation failed (HTTP {status}).") from e

        except httpx.RequestError as e:
            logger.error("Cancel request error: %s", e)
            raise BookingServiceError(
                "Could not connect to booking service") from e

    async def close(self):
        """Close HTTP client — call on application shutdown."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("BookingService HTTP client closed")

    # ── Validation helpers ────────────────────────────────────────────────────

    def _validate_property_id(self, property_id: str):
        if not property_id or not property_id.strip():
            raise BookingValidationError("property_id is required")

    def _validate_date_string(self, date_str: str):
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise BookingValidationError(
                f"Invalid date format: '{date_str}'. Expected YYYY-MM-DD"
            )

    def _validate_booking_inputs(
        self,
        property_id: str,
        datetime_slot: str,
        contact_name: str,
        contact_email: str,
        contact_phone: str,
    ):
        errors = []

        if not property_id or not property_id.strip():
            errors.append("property_id is required")

        # Validate datetime format
        try:
            slot_dt = datetime.strptime(datetime_slot, DATETIME_FORMAT)
            # Must be in the future
            if slot_dt <= datetime.now():
                errors.append("Inspection datetime must be in the future")
        except ValueError:
            errors.append(
                f"Invalid datetime format: '{datetime_slot}'. Expected YYYY-MM-DD HH:MM"
            )

        if not contact_name or len(contact_name.strip()) < 2:
            errors.append("contact_name must be at least 2 characters")

        if not contact_email or "@" not in contact_email:
            errors.append("contact_email must be a valid email address")

        # Basic Australian phone validation (mobile or landline)
        phone_digits = "".join(filter(str.isdigit, contact_phone))
        if len(phone_digits) < 8:
            errors.append("contact_phone must be a valid phone number")

        if errors:
            raise BookingValidationError("; ".join(errors))

    # ── Response parsers ──────────────────────────────────────────────────────

    def _parse_availability(self, data: dict) -> list[dict]:
        """Normalise .NET availability response to a clean list of slots."""
        slots = data.get("slots") or data.get("availableSlots") or []
        return [
            {
                "datetime": slot.get("datetime") or slot.get("dateTime", ""),
                "available": slot.get("available", True),
                "agent_name": slot.get("agentName", ""),
            }
            for slot in slots
            if slot.get("available", True)
        ]

    def _parse_booking_response(self, data: dict) -> dict:
        """Normalise .NET booking confirmation response."""
        return {
            "confirmation_id": data.get("confirmationId") or data.get("id", ""),
            "property_address": data.get("propertyAddress") or data.get("address", ""),
            "confirmed_datetime": data.get("confirmedDatetime") or data.get("datetime", ""),
            "agent_name": data.get("agentName", ""),
            "agent_phone": data.get("agentPhone", ""),
        }


# ── Custom exceptions ─────────────────────────────────────────────────────────

class BookingServiceError(Exception):
    """Backend or connectivity failure — show user-friendly message."""
    pass


class BookingValidationError(BookingServiceError):
    """Invalid input data — message is safe to surface to the user."""
    pass
