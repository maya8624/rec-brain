"""
Unit tests for LangGraph action tools.
All tests use the make_booking_service fixture — no .NET backend required.
"""
import pytest

from app.tools.check_availability import check_availability
from app.tools.book_inspection import book_inspection
from app.tools.cancel_inspection import cancel_inspection
from app.tools.get_booking import get_booking
from app.core.constants import AppStateKeys
from app.core.exceptions import BookingServiceError, ToolValidationError


def _cfg(svc):
    return {"configurable": {AppStateKeys.BOOKING_SERVICE: svc, AppStateKeys.USER_ID: "test-user"}}


class TestCheckAvailability:
    async def test_success_returns_slot_count(self, make_booking_service):
        svc = make_booking_service()
        result = await check_availability.ainvoke(
            {"property_id": "prop_123"}, config=_cfg(svc)
        )
        assert result["success"] is True
        assert result["slot_count"] == 2

    async def test_calls_service_with_property_id(self, make_booking_service):
        svc = make_booking_service()
        await check_availability.ainvoke(
            {"property_id": "prop_999"}, config=_cfg(svc)
        )
        svc.check_availability.assert_called_once_with("prop_999")

    async def test_no_slots_returns_success_with_zero_count(self, make_booking_service):
        from app.schemas.booking import AvailabilityResult
        svc = make_booking_service(availability=AvailabilityResult(
            success=True, property_id="prop_123", available_slots=[], slot_count=0
        ))
        result = await check_availability.ainvoke(
            {"property_id": "prop_123"}, config=_cfg(svc)
        )
        assert result["success"] is True
        assert result["slot_count"] == 0

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=BookingServiceError("Backend unavailable"))
        result = await check_availability.ainvoke(
            {"property_id": "prop_123"}, config=_cfg(svc)
        )
        assert result["success"] is False
        assert result["error"]

    async def test_unexpected_exception_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=RuntimeError("unexpected"))
        result = await check_availability.ainvoke(
            {"property_id": "prop_123"}, config=_cfg(svc)
        )
        assert result["success"] is False

    async def test_slot_times_converted_to_sydney_time(self, make_booking_service):
        svc = make_booking_service()
        result = await check_availability.ainvoke(
            {"property_id": "prop_123"}, config=_cfg(svc)
        )
        slot = result["available_slots"][0]
        assert "T" not in slot["start_at"]
        assert "Z" not in slot["start_at"]
        assert "AEST" in slot["start_at"] or "AEDT" in slot["start_at"]
        assert "T" not in slot["end_at"]
        assert "AEST" in slot["end_at"] or "AEDT" in slot["end_at"]


_BOOK_ARGS = {"slot_id": "slot-001"}


class TestBookInspection:
    async def test_success_returns_confirmation_id(self, make_booking_service):
        svc = make_booking_service()
        result = await book_inspection.ainvoke(_BOOK_ARGS, config=_cfg(svc))
        assert result["success"] is True
        assert result["confirmation_id"] == "CONF-12345"

    async def test_success_calls_service_book_once(self, make_booking_service):
        svc = make_booking_service()
        await book_inspection.ainvoke(_BOOK_ARGS, config=_cfg(svc))
        svc.book.assert_called_once()

    async def test_validation_error_returns_failure(self, make_booking_service):
        # ToolValidationError raised by the mocked service (valid date so Pydantic passes first)
        svc = make_booking_service(raise_error=ToolValidationError("Slot no longer available"))
        result = await book_inspection.ainvoke(_BOOK_ARGS, config=_cfg(svc))
        assert result["success"] is False
        assert "Slot no longer available" in result["error"]

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=BookingServiceError("Backend unavailable"))
        result = await book_inspection.ainvoke(_BOOK_ARGS, config=_cfg(svc))
        assert result["success"] is False

    async def test_success_message_includes_email_confirmation(self, make_booking_service):
        svc = make_booking_service()
        result = await book_inspection.ainvoke(_BOOK_ARGS, config=_cfg(svc))
        assert "confirmation email" in result["message"].lower()


class TestCancelInspection:
    async def test_success(self, make_booking_service):
        svc = make_booking_service()
        result = await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-12345", "reason": "Change of plans"}, config=_cfg(svc)
        )
        assert result["success"] is True
        assert result["id"] == "CONF-12345"

    async def test_calls_service_cancel_once(self, make_booking_service):
        svc = make_booking_service()
        await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-12345"}, config=_cfg(svc)
        )
        svc.cancel.assert_called_once()

    async def test_calls_service_with_correct_args(self, make_booking_service):
        svc = make_booking_service()
        await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-12345", "reason": "No longer needed"}, config=_cfg(svc)
        )
        svc.cancel.assert_called_once_with("CONF-12345", "test-user")

    async def test_not_found_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=ToolValidationError("Booking CONF-99999 not found"))
        result = await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-99999"}, config=_cfg(svc)
        )
        assert result["success"] is False
        assert result["id"] == "CONF-99999"

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=BookingServiceError("Backend unavailable"))
        result = await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-12345"}, config=_cfg(svc)
        )
        assert result["success"] is False

    async def test_success_message_includes_email_confirmation(self, make_booking_service):
        svc = make_booking_service()
        result = await cancel_inspection.ainvoke(
            {"confirmation_id": "CONF-12345"}, config=_cfg(svc)
        )
        assert "confirmation email" in result["message"].lower()


_CONF_ID = "CONF-12345"


class TestGetBooking:
    async def test_by_confirmation_id_returns_success(self, make_booking_service):
        svc = make_booking_service()
        result = await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        assert result["success"] is True
        assert result["confirmation_id"] == _CONF_ID

    async def test_by_confirmation_id_calls_service_with_user_id(self, make_booking_service):
        svc = make_booking_service()
        await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        svc.get_booking.assert_called_once_with(_CONF_ID, "test-user")

    async def test_empty_bookings_list_returns_failure(self, make_booking_service):
        svc = make_booking_service(my_bookings=[])
        result = await get_booking.ainvoke({}, config=_cfg(svc))
        assert result["success"] is False

    async def test_no_args_returns_all_bookings(self, make_booking_service):
        svc = make_booking_service()
        result = await get_booking.ainvoke({}, config=_cfg(svc))
        assert result["success"] is True
        assert len(result["bookings"]) == 1
        assert result["bookings"][0]["confirmation_id"] == _CONF_ID

    async def test_no_args_empty_account_returns_failure(self, make_booking_service):
        svc = make_booking_service(my_bookings=[])
        result = await get_booking.ainvoke({}, config=_cfg(svc))
        assert result["success"] is False
        assert result["error"]

    async def test_service_error_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=BookingServiceError("Backend down"))
        result = await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        assert result["success"] is False
        assert "Backend down" in result["error"]

    async def test_unexpected_exception_returns_failure(self, make_booking_service):
        svc = make_booking_service(raise_error=RuntimeError("crash"))
        result = await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        assert result["success"] is False

    async def test_datetimes_formatted_as_sydney_time(self, make_booking_service):
        svc = make_booking_service()
        result = await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        assert result["start_at"]
        assert "Z" not in result["start_at"]          # not raw UTC
        assert " at " in result["start_at"]           # human-readable separator
        assert "AEST" in result["start_at"] or "AEDT" in result["start_at"]

    async def test_agent_name_joined_correctly(self, make_booking_service):
        svc = make_booking_service()
        result = await get_booking.ainvoke({"confirmation_id": _CONF_ID}, config=_cfg(svc))
        assert result["agent_name"] == "Jane Smith"
