"""
HTTP client for the .NET backend booking API.
All booking state is owned by .NET — Python never stores booking records.
"""
import logging
from datetime import datetime
import httpx

from app.core.constants import BookingEndpoints
from app.core.config import settings

from app.core.exceptions import (
    BookingServiceError,
    BookingValidationError,
    raise_for_booking_status,
)
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

    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------
    # HTTP client
    # ------------------------------------
    def _get_client(self) -> httpx.AsyncClient:
        """Lazy init HTTP client with shared configuration."""

        if self._client is None or self._client.is_closed:
            timeout = httpx.Timeout(
                connect=5.0,
                read=15.0,
                write=10.0,
                pool=5.0
            )

            headers = {
                "Content-Type": "application/json",
                "X-Api-Key": settings.BACKEND_API_KEY,
            }

            self._client = httpx.AsyncClient(
                base_url=settings.BACKEND_BASE_URL,
                timeout=timeout,
                headers=headers,
            )

        return self._client

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
            response = await self._get_client().get(
                BookingEndpoints.AVAILABILITY,
                params=params,
            )

            response.raise_for_status()
            slots = self._parse_availability(response.json())

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

        except httpx.HTTPStatusError as e:
            logger.error("get_availability | HTTP %d", e.response.status_code)

            raise BookingServiceError(
                f"Could not retrieve availability (HTTP {e.response.status_code})"
            ) from e

        except httpx.RequestError as e:
            logger.error("get_availability | request error: %s", e)

            raise BookingServiceError(
                "Could not connect to booking service") from e

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
            response = await self._get_client().post(
                BookingEndpoints.BOOK,
                json=payload,
            )

            response.raise_for_status()
            confirmation = self._parse_booking_response(response.json())

            logger.info("book | confirmed | id=%s",
                        confirmation.confirmation_id)

            return confirmation.model_dump()

        except httpx.HTTPStatusError as e:
            logger.error("book | HTTP %d", e.response.status_code)

            raise_for_booking_status(e)

        except httpx.RequestError as e:
            logger.error("book | request error: %s", e)

            raise BookingServiceError(
                "Could not connect to booking service") from e

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
            response = await self._get_client().post(
                BookingEndpoints.CANCEL,
                json=payload,
            )

            response.raise_for_status()
            logger.info("cancel | cancelled | id=%s", request.confirmation_id)

            return CancellationConfirmation(
                confirmation_id=request.confirmation_id,
            ).model_dump()

        except httpx.HTTPStatusError as e:
            logger.error("cancel | HTTP %d", e.response.status_code)

            raise_for_booking_status(
                e, confirmation_id=request.confirmation_id
            )

        except httpx.RequestError as e:
            logger.error("cancel | request error: %s", e)

            raise BookingServiceError(
                "Could not connect to booking service") from e

    async def close(self):
        """Close HTTP client — call on application shutdown."""

        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("BookingService HTTP client closed")

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
        """
        Normalise .NET availability response to a clean list of slot dicts.
        """
        slots = data.get("slots") or []

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
