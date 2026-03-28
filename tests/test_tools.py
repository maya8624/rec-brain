"""
Tests for action tools — check_availability, book_inspection, cancel_inspection.
Uses mock BookingService so no .NET backend needed.

Usage:
    pytest tests/test_tools.py         # run all
    pytest tests/test_tools.py -v      # verbose
    pytest tests/test_tools.py -k check_availability
"""
from app.tools.check_availability import check_availability
from app.tools.book_inspection import book_inspection
from app.tools.cancel_inspection import cancel_inspection
from app.services.booking_service import BookingService
from app.core.exceptions import BookingServiceError, BookingValidationError
import json
import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result: dict):
    print(json.dumps(result, indent=2, default=str))


# ── Factory ────────────────────────────────────────────────────────────────────

def make_booking_service(availability=None, booking_result=None, raise_error=None):
    """
    Factory for mock BookingService.
    Pass raise_error=BookingServiceError("msg") to simulate failures.
    """
    mock = AsyncMock(spec=BookingService)

    if raise_error:
        mock.get_availability.side_effect = raise_error
        mock.book.side_effect = raise_error
        mock.cancel.side_effect = raise_error
    else:
        mock.get_availability.return_value = availability or [
            {"datetime": "2026-04-12 10:00",
                "agent_name": "Jane Smith", "available": True},
            {"datetime": "2026-04-12 14:00",
                "agent_name": "Jane Smith", "available": True},
            {"datetime": "2026-04-13 11:00",
                "agent_name": "Jane Smith", "available": True},
        ]
        mock.book.return_value = booking_result or {
            "confirmation_id": "CONF-12345",
            "property_address": "123 Main St, Sydney NSW 2000",
            "confirmed_datetime": "2026-04-12 10:00",
            "agent_name": "Jane Smith",
            "agent_phone": "0412 345 678",
        }
        mock.cancel.return_value = {"success": True}

    return mock


# ── check_availability ─────────────────────────────────────────────────────────

class TestCheckAvailability:
    async def test_success(self):
        booking_service = make_booking_service()
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "preferred_date": "2026-04-12",
            "booking_service": booking_service,
        })
        pretty(result)
        assert result["success"]
        assert result["slot_count"] == 3
        print("actual call:", booking_service.get_availability.call_args)
        booking_service.get_availability.assert_called_once()

    async def test_no_slots(self):
        booking_service = make_booking_service(availability=[])
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "booking_service": booking_service,
        })
        pretty(result)
        assert result["success"]
        assert result["slot_count"] == 0
        booking_service.get_availability.assert_called_once()

    async def test_service_error(self):
        booking_service = make_booking_service(
            raise_error=BookingServiceError("Backend unavailable")
        )
        result = await check_availability.ainvoke({
            "property_id": "prop_123",
            "booking_service": booking_service,
        })
        pretty(result)
        assert not result["success"]
        assert result["error"]
        booking_service.get_availability.assert_called_once()


# ── book_inspection ────────────────────────────────────────────────────────────

class TestBookInspection:
    async def test_success(self):
        booking_service = make_booking_service()
        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2026-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": booking_service,
        })
        pretty(result)
        assert result["success"]
        assert result["confirmation_id"] == "CONF-12345"
        booking_service.book.assert_called_once_with(
            property_id="prop_123",
            datetime_slot="2026-04-12 10:00",
            contact_name="John Smith",
            contact_email="john@email.com",
            contact_phone="0412 345 678",
        )

    async def test_validation_error(self):
        booking_service = make_booking_service(
            raise_error=BookingValidationError("Invalid datetime slot")
        )
        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "invalid-date",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": booking_service,
        })
        pretty(result)
        assert not result["success"]
        assert "Invalid" in result["error"]
        booking_service.book.assert_called_once()

    async def test_service_error(self):
        booking_service = make_booking_service(
            raise_error=BookingServiceError("Backend unavailable")
        )
        result = await book_inspection.ainvoke({
            "property_id": "prop_123",
            "datetime_slot": "2026-04-12 10:00",
            "contact_name": "John Smith",
            "contact_email": "john@email.com",
            "contact_phone": "0412 345 678",
            "booking_service": booking_service,
        })
        pretty(result)
        assert not result["success"]
        booking_service.book.assert_called_once()


# ── cancel_inspection ──────────────────────────────────────────────────────────

class TestCancelInspection:
    async def test_success(self):
        booking_service = make_booking_service()
        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "reason": "Change of plans",
            "booking_service": booking_service,
        })
        pretty(result)
        assert result["success"]
        assert result["confirmation_id"] == "CONF-12345"
        booking_service.cancel.assert_called_once_with(
            confirmation_id="CONF-12345",
            reason="Change of plans",
        )

    async def test_no_reason(self):
        booking_service = make_booking_service()
        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-12345",
            "booking_service": booking_service,
        })
        pretty(result)
        assert result["success"]
        booking_service.cancel.assert_called_once_with(
            confirmation_id="CONF-12345",
        )

    async def test_not_found(self):
        booking_service = make_booking_service(
            raise_error=BookingValidationError("Booking CONF-99999 not found")
        )
        result = await cancel_inspection.ainvoke({
            "confirmation_id": "CONF-99999",
            "booking_service": booking_service,
        })
        pretty(result)
        assert not result["success"]
        booking_service.cancel.assert_called_once_with(
            confirmation_id="CONF-99999",
        )
