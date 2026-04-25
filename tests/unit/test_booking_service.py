"""
Unit tests for BookingService — HTTP client for the .NET inspection backend.

BackendClient is mocked with AsyncMock so no real HTTP calls are made.
Tests verify request construction, response parsing, and error mapping.
"""
from unittest.mock import AsyncMock
import pytest

from app.services.booking_service import BookingService
from app.core.exceptions import BackendClientError, BookingServiceError, BookingValidationError
from app.schemas.booking import BookingRequest


_PROPERTY_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def make_backend_client(
    get_return: list | None = None,
    post_return: dict | None = None,
    raise_error: Exception | None = None,
):
    """AsyncMock for BackendClient — no spec to avoid signature mismatch on post()."""
    mock = AsyncMock()
    if raise_error:
        mock.get.side_effect = raise_error
        mock.post.side_effect = raise_error
        mock.patch.side_effect = raise_error
    else:
        mock.get.return_value = get_return or []
        mock.post.return_value = post_return or {}
    return mock


def make_service(client=None, **client_kwargs) -> BookingService:
    return BookingService(client=client or make_backend_client(**client_kwargs))


def _available_slot(slot_id: str = "slot-001", start: str = "2027-06-15T10:00:00Z") -> dict:
    return {
        "id": slot_id,
        "startAtUtc": start,
        "endAtUtc": "2027-06-15T11:00:00Z",
        "agentId": "agent-1",
        "capacity": 5,
        "status": "open",
        "notes": "",
    }


def _unavailable_slot(slot_id: str = "slot-002") -> dict:
    return {
        "id": slot_id,
        "startAtUtc": "2027-06-15T14:00:00Z",
        "endAtUtc": "2027-06-15T15:00:00Z",
        "agentId": "agent-1",
        "capacity": 0,
        "status": "closed",
        "notes": "",
    }


class TestGetAvailability:
    async def test_success_returns_availability_result(self):
        client = make_backend_client(get_return=[_available_slot()])

        result = await BookingService(client).check_availability(_PROPERTY_ID)

        assert result.success is True
        assert result.slot_count == 1
        assert result.property_id == _PROPERTY_ID

    async def test_available_slots_are_parsed_correctly(self):
        client = make_backend_client(get_return=[
            _available_slot("slot-001", "2027-04-12T10:00:00Z"),
            _available_slot("slot-002", "2027-04-12T14:00:00Z"),
        ])

        result = await BookingService(client).check_availability(_PROPERTY_ID)

        assert result.slot_count == 2
        assert result.available_slots[0].slot_id == "slot-001"
        assert result.available_slots[1].slot_id == "slot-002"

    async def test_unavailable_slots_are_filtered_out(self):
        client = make_backend_client(get_return=[
            _unavailable_slot("slot-001"),
            _available_slot("slot-002"),
        ])

        result = await BookingService(client).check_availability(_PROPERTY_ID)

        assert result.slot_count == 1
        assert result.available_slots[0].slot_id == "slot-002"

    async def test_invalid_property_id_raises_validation_error(self):
        svc = make_service()
        with pytest.raises(BookingValidationError, match="property_id"):
            await svc.check_availability("not-a-uuid")

    async def test_backend_client_error_raises_booking_service_error(self):
        client = make_backend_client(
            raise_error=BackendClientError("503 unavailable", 503))
        with pytest.raises(BookingServiceError):
            await BookingService(client).check_availability(_PROPERTY_ID)


class TestBook:
    def _make_request(self) -> BookingRequest:
        return BookingRequest(slot_id="slot-001", user_id="user-123")

    async def test_success_returns_booking_confirmation(self):
        client = make_backend_client(post_return={
            "id": "CONF-42",
            "propertyId": _PROPERTY_ID,
            "status": "confirmed",
            "agentFirstName": "Jane",
            "agentLastName": "Smith",
            "agentPhone": "0400 000 000",
            "startAtUtc": "2027-04-12T10:00:00Z",
            "endAtUtc": "2027-04-12T11:00:00Z",
        })

        result = await BookingService(client).book(self._make_request())

        assert result.confirmation_id == "CONF-42"
        assert result.property_id == _PROPERTY_ID
        assert result.agent_first_name == "Jane"
        assert result.agent_last_name == "Smith"

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(
            raise_error=BackendClientError("timeout", 503)
        )

        with pytest.raises(BookingServiceError):
            await BookingService(client).book(self._make_request())

    async def test_post_called_once(self):
        client = make_backend_client(post_return={"id": "C1"})

        await BookingService(client).book(self._make_request())
        client.post.assert_called_once()

    async def test_post_payload_contains_slot_and_user(self):
        client = make_backend_client(post_return={"id": "C1"})

        await BookingService(client).book(self._make_request())

        payload = client.post.call_args.kwargs.get("json", {})
        assert payload["InspectionSlotId"] == "slot-001"
        assert payload["UserId"] == "user-123"


class TestCancel:
    async def test_success_returns_cancellation_confirmation(self):
        client = make_backend_client()

        result = await BookingService(client).cancel("CONF-12345", "user-abc")

        assert result.id == "CONF-12345"
        assert result.success is True

    async def test_patch_called_with_user_id(self):
        client = make_backend_client()

        await BookingService(client).cancel("CONF-12345", "user-abc")

        client.patch.assert_called_once()
        payload = client.patch.call_args.kwargs.get("json", {})
        assert payload.get("UserId") == "user-abc"

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(raise_error=BackendClientError("404", 404))

        with pytest.raises(BookingServiceError):
            await BookingService(client).cancel("CONF-12345", "user-abc")
