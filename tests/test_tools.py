"""
Tests for action tools — check_availability, book_inspection, cancel_inspection.
Uses mock BookingService so no .NET backend needed.

Usage:
    python scripts/test_tools.py                      # run all
    python scripts/test_tools.py check_availability
    python scripts/test_tools.py book_inspection
    python scripts/test_tools.py cancel_inspection
"""
from app.core.exceptions import BookingServiceError, BookingValidationError
from app.tools.cancel_inspection import cancel_inspection
from app.tools.book_inspection import book_inspection
from app.tools.check_availability import check_availability
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result: dict):
    print(json.dumps(result, indent=2, default=str))


def print_header(name: str):
    print(f"\n── {name} {'─' * (50 - len(name))}")


def make_booking_service(
    availability=None,
    booking_result=None,
    raise_error=None,
):
    """
    Factory for mock BookingService.
    Pass raise_error=BookingServiceError("msg") to simulate failures.
    """
    mock = AsyncMock()

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


# ── check_availability tests ───────────────────────────────────────────────────

async def test_check_availability_success():
    print_header("check_availability — success")
    booking_service = make_booking_service()
    result = await check_availability.ainvoke({
        "property_id": "prop_123",
        "preferred_date": "2026-04-12",
        "booking_service": booking_service,
    })
    pretty(result)
    assert result["success"], "Expected success"
    assert result["slot_count"] == 3, f"Expected 3 slots, got {result['slot_count']}"
    print("  ✓ check_availability success")


async def test_check_availability_no_slots():
    print_header("check_availability — no slots")
    booking_service = make_booking_service(availability=[])
    result = await check_availability.ainvoke({
        "property_id": "prop_123",
        "booking_service": booking_service,
    })
    pretty(result)
    assert result["success"], "Expected success even with 0 slots"
    assert result["slot_count"] == 0
    print("  ✓ check_availability no slots")


async def test_check_availability_service_error():
    print_header("check_availability — service error")
    booking_service = make_booking_service(
        raise_error=BookingServiceError("Backend unavailable")
    )
    result = await check_availability.ainvoke({
        "property_id": "prop_123",
        "booking_service": booking_service,
    })
    pretty(result)
    assert not result["success"], "Expected failure"
    assert result["error"], "Expected error message"
    print("  ✓ check_availability service error handled")


# ── book_inspection tests ──────────────────────────────────────────────────────

async def test_book_inspection_success():
    print_header("book_inspection — success")
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
    assert result["success"], "Expected success"
    assert result["confirmation_id"] == "CONF-12345"
    print("  ✓ book_inspection success")


async def test_book_inspection_validation_error():
    print_header("book_inspection — validation error")
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
    assert not result["success"], "Expected failure"
    assert "Invalid" in result["error"]
    print("  ✓ book_inspection validation error handled")


async def test_book_inspection_service_error():
    print_header("book_inspection — service error")
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
    assert not result["success"], "Expected failure"
    print("  ✓ book_inspection service error handled")


# ── cancel_inspection tests ────────────────────────────────────────────────────

async def test_cancel_inspection_success():
    print_header("cancel_inspection — success")
    booking_service = make_booking_service()
    result = await cancel_inspection.ainvoke({
        "confirmation_id": "CONF-12345",
        "reason": "Change of plans",
        "booking_service": booking_service,
    })
    pretty(result)
    assert result["success"], "Expected success"
    assert result["confirmation_id"] == "CONF-12345"
    print("  ✓ cancel_inspection success")


async def test_cancel_inspection_no_reason():
    print_header("cancel_inspection — no reason")
    booking_service = make_booking_service()
    result = await cancel_inspection.ainvoke({
        "confirmation_id": "CONF-12345",
        "booking_service": booking_service,
    })
    pretty(result)
    assert result["success"], "Expected success without reason"
    print("  ✓ cancel_inspection no reason")


async def test_cancel_inspection_not_found():
    print_header("cancel_inspection — not found")
    booking_service = make_booking_service(
        raise_error=BookingValidationError("Booking CONF-99999 not found")
    )
    result = await cancel_inspection.ainvoke({
        "confirmation_id": "CONF-99999",
        "booking_service": booking_service,
    })
    pretty(result)
    assert not result["success"], "Expected failure"
    print("  ✓ cancel_inspection not found handled")


# ── Router ─────────────────────────────────────────────────────────────────────

TESTS = {
    "check_availability":            test_check_availability_success,
    "check_availability_no_slots":   test_check_availability_no_slots,
    "check_availability_error":      test_check_availability_service_error,
    "book_inspection":               test_book_inspection_success,
    "book_inspection_validation":    test_book_inspection_validation_error,
    "book_inspection_error":         test_book_inspection_service_error,
    "cancel_inspection":             test_cancel_inspection_success,
    "cancel_inspection_no_reason":   test_cancel_inspection_no_reason,
    "cancel_inspection_not_found":   test_cancel_inspection_not_found,
}


async def run_all():
    passed = 0
    failed = 0

    for name, fn in TESTS.items():
        try:
            await fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            failed += 1

    print(f"\n── Results: {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f" | {failed} FAILED ✗")
    else:
        print(" ✓")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        asyncio.run(run_all())
    else:
        name = sys.argv[1]
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(TESTS)}")
            sys.exit(1)
        asyncio.run(TESTS[name]())
