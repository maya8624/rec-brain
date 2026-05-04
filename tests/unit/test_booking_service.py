"""
Unit tests for BookingService — HTTP client for the .NET inspection backend.

BackendClient is mocked with AsyncMock so no real HTTP calls are made.
Tests verify request construction, response parsing, and error mapping.
"""
from unittest.mock import AsyncMock
import pytest

from app.services.booking_service import BookingService
from app.core.exceptions import BackendClientError, BookingServiceError, ToolValidationError
from app.schemas.booking import BookingRequest


_PROPERTY_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"


def make_backend_client(
    get_return: list | dict | None = None,
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
        with pytest.raises(ToolValidationError, match="property_id"):
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


def _booking_dto(conf_id: str = "CONF-42") -> dict:
    return {
        "id": conf_id,
        "propertyId": _PROPERTY_ID,
        "propertyAddress": "42 Main St, Sydney",
        "status": "confirmed",
        "agentFirstName": "Jane",
        "agentLastName": "Smith",
        "agentPhone": "0400 000 000",
        "startAtUtc": "2027-04-12T10:00:00Z",
        "endAtUtc": "2027-04-12T11:00:00Z",
    }


class TestGetBooking:
    async def test_success_returns_booking_confirmation(self):
        client = make_backend_client(get_return=_booking_dto())
        result = await BookingService(client).get_booking("CONF-42", "user-123")
        assert result.confirmation_id == "CONF-42"
        assert result.property_id == _PROPERTY_ID

    async def test_property_address_is_parsed(self):
        client = make_backend_client(get_return=_booking_dto())
        result = await BookingService(client).get_booking("CONF-42", "user-123")
        assert result.property_address == "42 Main St, Sydney"

    async def test_agent_names_parsed(self):
        client = make_backend_client(get_return=_booking_dto())
        result = await BookingService(client).get_booking("CONF-42", "user-123")
        assert result.agent_first_name == "Jane"
        assert result.agent_last_name == "Smith"

    async def test_get_called_with_user_id_param(self):
        client = make_backend_client(get_return=_booking_dto())
        await BookingService(client).get_booking("CONF-42", "user-123")
        client.get.assert_called_once()
        params = client.get.call_args.kwargs.get("params", {})
        assert params.get("userId") == "user-123"

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(raise_error=BackendClientError("404", 404))
        with pytest.raises(BookingServiceError):
            await BookingService(client).get_booking("CONF-42", "user-123")


class TestGetMyBookings:
    async def test_success_returns_list(self):
        client = make_backend_client(get_return=[_booking_dto("CONF-1"), _booking_dto("CONF-2")])
        results = await BookingService(client).get_my_bookings("user-123")
        assert len(results) == 2
        assert results[0].confirmation_id == "CONF-1"
        assert results[1].confirmation_id == "CONF-2"

    async def test_empty_list_returns_empty(self):
        client = make_backend_client(get_return=[])
        results = await BookingService(client).get_my_bookings("user-123")
        assert results == []

    async def test_get_called_with_user_id_param(self):
        client = make_backend_client(get_return=[])
        await BookingService(client).get_my_bookings("user-123")
        client.get.assert_called_once()
        params = client.get.call_args.kwargs.get("params", {})
        assert params.get("userId") == "user-123"

    async def test_non_list_response_raises_booking_service_error(self):
        client = make_backend_client(get_return=_booking_dto())
        with pytest.raises(BookingServiceError, match="Unexpected response format"):
            await BookingService(client).get_my_bookings("user-123")

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(raise_error=BackendClientError("500", 500))
        with pytest.raises(BookingServiceError):
            await BookingService(client).get_my_bookings("user-123")

    async def test_property_address_parsed_per_item(self):
        client = make_backend_client(get_return=[_booking_dto()])
        results = await BookingService(client).get_my_bookings("user-123")
        assert results[0].property_address == "42 Main St, Sydney"


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
