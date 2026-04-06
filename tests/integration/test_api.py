"""
Integration tests for the /api/chat endpoints.

Skip with: pytest -m unit
Run with:  pytest -m integration
"""
import pytest
from tests.integration.conftest import skip_if_no_env

pytestmark = [pytest.mark.integration, skip_if_no_env]


class TestChatEndpoint:
    async def test_valid_request_returns_200(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "What are your office hours?",
                "thread_id": "integ-test-general",
                "user_id": "test-user-01",
                "is_new_conversation": True,
            })

        assert response.status_code == 200

    async def test_response_shape(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "thread_id": "integ-test-shape",
                "user_id": "test-user-01",
                "is_new_conversation": True,
            })

        body = response.json()

        assert "reply" in body
        assert "thread_id" in body
        assert "intent" in body
        assert "tools_used" in body
        assert "booking_confirmed" in body
        assert "requires_human" in body

    async def test_thread_id_echoed(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "hello",
                "thread_id": "echo-test-thread",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )
        assert response.json()["thread_id"] == "echo-test-thread"

    async def test_empty_message_returns_422(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "",
                "thread_id": "t1",
                "user_id": "u1",
            }
        )
        assert response.status_code == 422

    async def test_missing_thread_id_returns_422(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "hello",
                "user_id": "u1",
            }
        )
        assert response.status_code == 422

    async def test_missing_user_id_returns_422(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "hello",
                "thread_id": "t1",
            }
        )
        assert response.status_code == 422

    async def test_search_intent_detected(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "Show me 3 bedroom houses in Sydney under $800k",
                "thread_id": "integ-search",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )
        body = response.json()
        assert body["intent"] == "search"

    async def test_general_intent_detected(self, client):
        response = await client.post(
            "/api/chat",
            json={
                "message": "What are your office hours?",
                "thread_id": "integ-general",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )
        body = response.json()
        assert body["intent"] == "general"

    async def test_new_conversation_starts_clean(self, client):
        """is_new_conversation=True must not inherit state from a previous thread."""
        response = await client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "thread_id": "fresh-thread-xyz",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )
        assert response.status_code == 200
        assert response.json()["booking_confirmed"] is False

    async def test_multi_turn_conversation(self, client):
        """Second message on same thread should continue state (no is_new_conversation)."""
        thread_id = "multi-turn-integ"
        await client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "thread_id": thread_id,
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )
        response = await client.post(
            "/api/chat",
            json={
                "message": "What are office hours?",
                "thread_id": thread_id,
                "user_id": "u1",
                "is_new_conversation": False,
            }
        )
        assert response.status_code == 200


class TestChatStreamEndpoint:
    async def test_returns_event_stream_content_type(self, client):
        response = await client.post(
            "/api/chat/stream",
            json={
                "message": "Hello",
                "thread_id": "stream-test",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )

        assert "text/event-stream" in response.headers.get("content-type", "")

    async def test_stream_contains_done_marker(self, client):
        response = await client.post(
            "/api/chat/stream",
            json={
                "message": "Hello",
                "thread_id": "stream-done",
                "user_id": "u1",
                "is_new_conversation": True,
            }
        )

        assert "[DONE]" in response.text
