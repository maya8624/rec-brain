"""
Unit tests for LangGraph action tools.
All tests use the make_booking_service fixture — no .NET backend required.
"""
import pytest

from app.tools.check_availability import check_availability
from app.tools.book_inspection import book_inspection
from app.tools.cancel_inspection import cancel_inspection
from app.core.exceptions import BookingServiceError, BookingValidationError
from app.schemas.booking import CancellationRequest


class TestCheckAvailability:
    async def test_success_returns_slot_count(self, make_booking_service):
        svc = make_booking_service()
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "preferred_date": "2026-04-12",
            "booking_service": svc,
        })
        assert result["success"] is True
        assert result["slot_count"] == 2

    async def test_calls_service_with_property_id(self, make_booking_service):
        svc = make_booking_service()
        await check_availability.ainvoke({
            "property_id": "prop_999",
            "booking_service": svc,
        })
        svc.get_availability.assert_called_once_with("prop_999", None)

    async def test_preferred_date_forwarded(self, make_booking_service):
        svc = make_booking_service()
        await check_availability.ainvoke({
            "property_id": "prop_123",
            "preferred_date": "2026-06-01",
            "booking_service": svc,
        })
        svc.get_availability.assert_called_once_with("prop_123", "2026-06-01")

    async def test_no_slots_returns_success_with_zero_count(self, make_booking_service):
        svc = make_booking_service(availability=[])
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "booking_service": svc,
        })
        assert result["success"] is True
        assert result["slot_count"] == 0

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(
            raise_error=BookingServiceError("Backend unavailable"))
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "booking_service": svc,
        })
        assert result["success"] is False
        assert result["error"]

    async def test_unexpected_exception_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=RuntimeError("unexpected"))
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "booking_service": svc,
        })
        assert result["success"] is False


class TestBookInspection:
    async def test_success_returns_confirmation_id(self, make_booking_service):
        svc = make_booking_service()
        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2027-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": svc,
        })
        assert result["success"] is True
        assert result["confirmation_id"] == "CONF-12345"

    async def test_success_calls_service_book_once(self, make_booking_service):
        svc = make_booking_service()
        await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2027-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": svc,
        })
        svc.book.assert_called_once()

    async def test_validation_error_returns_failure(self, make_booking_service):
        # BookingValidationError raised by the mocked service (valid date so Pydantic passes first)
        svc = make_booking_service(
            raise_error=BookingValidationError("Slot no longer available"))
        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2027-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": svc,
        })
        assert result["success"] is False
        assert "Slot no longer available" in result["error"]

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(
            raise_error=BookingServiceError("Backend unavailable")
        )

        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2027-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": svc,
        })
        assert result["success"] is False


class TestCancelInspection:
    async def test_success(self, make_booking_service):
        svc = make_booking_service()
        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "reason": "Change of plans",
            "booking_service": svc,
        })

        assert result["success"] is True
        assert result["confirmation_id"] == "CONF-12345"

    async def test_calls_service_cancel_once(self, make_booking_service):
        svc = make_booking_service()
        await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "booking_service": svc,
        })
        svc.cancel.assert_called_once()

    async def test_passes_cancellation_request_object(self, make_booking_service):
        """cancel() must receive a CancellationRequest, not raw kwargs."""
        svc = make_booking_service()
        await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "reason": "No longer needed",
            "booking_service": svc,
        })

        call_arg = svc.cancel.call_args.args[0]

        assert isinstance(call_arg, CancellationRequest)
        assert call_arg.confirmation_id == "CONF-12345"
        assert call_arg.reason == "No longer needed"

    async def test_no_reason_sends_none(self, make_booking_service):
        svc = make_booking_service()
        await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "booking_service": svc,
        })

        call_arg = svc.cancel.call_args.args[0]
        assert call_arg.reason is None

    async def test_not_found_returns_failure(self, make_booking_service):
        svc = make_booking_service(
            raise_error=BookingValidationError("Booking CONF-99999 not found")
        )

        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-99999",
            "booking_service": svc,
        })

        assert result["success"] is False
        assert result["confirmation_id"] == "CONF-99999"

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(
            raise_error=BookingServiceError("Backend unavailable")
        )

        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "booking_service": svc,
        })

        assert result["success"] is False
