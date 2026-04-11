"""
Unit tests for BookingService — HTTP client for the .NET inspection backend.

BackendClient is mocked with AsyncMock so no real HTTP calls are made.
Tests verify request construction, response parsing, and error mapping.
"""
from unittest.mock import AsyncMock
import pytest

from app.services.booking_service import BookingService
from app.core.exceptions import BackendClientError, BookingServiceError, BookingValidationError
from app.schemas.booking import BookingRequest, ContactInfo, CancellationRequest


def make_backend_client(
    get_return: dict | None = None,
    post_return: dict | None = None,
    raise_error: Exception | None = None,
):
    """AsyncMock for BackendClient — no spec to avoid signature mismatch on post()."""
    mock = AsyncMock()
    if raise_error:
        mock.get.side_effect = raise_error
        mock.post.side_effect = raise_error
    else:
        mock.get.return_value = get_return or {}
        mock.post.return_value = post_return or {}
    return mock


def make_service(client=None, **client_kwargs) -> BookingService:
    return BookingService(client=client or make_backend_client(**client_kwargs))


def _future_slot() -> str:
    return "2027-06-15 10:00"


class TestGetAvailability:
    async def test_success_returns_availability_result_dict(self):
        client = make_backend_client(get_return={
            "availableSlots": [
                {
                    "datetime": "2027-04-12 10:00",
                    "available": True, "agentName": "Jane Smith"
                },
            ]
        })

        result = await BookingService(client).get_availability("prop_123")

        assert result["success"] is True
        assert result["slot_count"] == 1
        assert result["property_id"] == "prop_123"

    async def test_available_slots_are_parsed_correctly(self):
        client = make_backend_client(get_return={
            "availableSlots": [
                {
                    "datetime": "2027-04-12 10:00",
                    "available": True, "agentName": "Jane"
                },
                {
                    "datetime": "2027-04-12 14:00",
                    "available": True, "agentName": "Jane"
                },
            ]
        })

        result = await BookingService(client).get_availability("prop_123")

        assert result["slot_count"] == 2
        assert result["available_slots"][0]["agent_name"] == "Jane"

    async def test_unavailable_slots_are_filtered_out(self):
        client = make_backend_client(get_return={
            "availableSlots": [
                {
                    "datetime": "2027-04-12 10:00",
                    "available": False, "agentName": "Jane"
                },
                {
                    "datetime": "2027-04-12 14:00",
                    "available": True, "agentName": "Jane"
                },
            ]
        })

        result = await BookingService(client).get_availability("prop_123")

        assert result["slot_count"] == 1

    async def test_empty_property_id_raises_validation_error(self):
        svc = make_service()
        with pytest.raises(BookingValidationError, match="property_id"):
            await svc.get_availability("")

    async def test_backend_client_error_raises_booking_service_error(self):
        client = make_backend_client(
            raise_error=BackendClientError("503 unavailable", 503))
        with pytest.raises(BookingServiceError):
            await BookingService(client).get_availability("prop_123")


class TestBook:
    def _make_request(self, slot: str = None) -> BookingRequest:
        return BookingRequest(
            property_id="prop_123",
            datetime_slot=slot or _future_slot(),
            contact=ContactInfo(
                name="John Smith",
                email="john@example.com",
                phone="0412 345 678",
            ),
        )

    async def test_success_returns_confirmation_dict(self):
        client = make_backend_client(post_return={
            "confirmationId": "CONF-42",
            "propertyAddress": "123 Main St",
            "confirmedDatetime": _future_slot(),
            "agentName": "Jane Smith",
            "agentPhone": "0400 000 000",
        })

        result = await BookingService(client).book(self._make_request())

        assert result["confirmation_id"] == "CONF-42"
        assert result["property_address"] == "123 Main St"
        assert result["agent_name"] == "Jane Smith"

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(
            raise_error=BackendClientError("timeout", 503)
        )

        with pytest.raises(BookingServiceError):
            await BookingService(client).book(self._make_request())

    async def test_post_called_once(self):
        client = make_backend_client(post_return={
            "confirmationId": "C1", "propertyAddress": "", "confirmedDatetime": _future_slot(),
        })

        await BookingService(client).book(self._make_request())
        client.post.assert_called_once()


class TestCancel:
    async def test_success_returns_confirmation_dict(self):
        client = make_backend_client(post_return={})

        result = await BookingService(client).cancel(
            CancellationRequest(confirmation_id="CONF-12345")
        )

        assert result["confirmation_id"] == "CONF-12345"

    async def test_reason_included_in_payload_when_provided(self):
        client = make_backend_client(post_return={})

        await BookingService(client).cancel(
            CancellationRequest(
                confirmation_id="CONF-12345",
                reason="Change of plans"
            )
        )

        call_kwargs = client.post.call_args

        # second positional arg or json kwarg contains the payload
        payload = call_kwargs.args[1] if len(
            call_kwargs.args) > 1 else call_kwargs.kwargs.get("json", {})

        assert payload.get("reason") == "Change of plans"

    async def test_reason_omitted_when_none(self):
        client = make_backend_client(post_return={})

        await BookingService(client).cancel(
            CancellationRequest(confirmation_id="CONF-12345", reason=None)
        )
        call_kwargs = client.post.call_args

        payload = call_kwargs.args[1] if len(
            call_kwargs.args) > 1 else call_kwargs.kwargs.get("json", {})

        assert "reason" not in payload

    async def test_backend_error_raises_booking_service_error(self):
        client = make_backend_client(
            raise_error=BackendClientError("404", 404)
        )

        with pytest.raises(BookingServiceError):
            await BookingService(client).cancel(
                CancellationRequest(confirmation_id="CONF-12345")
            )
