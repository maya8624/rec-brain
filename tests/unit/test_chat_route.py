"""
Unit tests for the chat route helper functions.
These are pure functions — no HTTP server or LangGraph agent required.
"""
from langchain_core.messages import AIMessage, HumanMessage

from app.api.routes.chat import (
    _build_response,
    _extract_single_property_id,
    _to_sse_event,
)


class TestBuildResponse:
    def _make_result(self, **overrides) -> dict:
        base = {
            "messages": [AIMessage(content="Hello, how can I help you?")],
            "user_intent": "general",
            "booking_status": {"confirmed": False, "cancelled": False, "awaiting_confirmation": False},
            "booking_context": {},
            "requires_human": False,
        }
        base.update(overrides)
        return base

    def test_reply_extracted_from_last_ai_message(self):
        result = self._make_result(messages=[
            HumanMessage(content="hi"),
            AIMessage(content="Hello there!"),
        ])
        response = _build_response("thread_1", result)
        assert response.reply == "Hello there!"

    def test_last_ai_message_used_when_multiple(self):
        result = self._make_result(messages=[
            AIMessage(content="first response"),
            HumanMessage(content="follow-up"),
            AIMessage(content="second response"),
        ])
        response = _build_response("thread_1", result)
        assert response.reply == "second response"

    def test_fallback_reply_when_no_ai_message(self):
        result = self._make_result(messages=[HumanMessage(content="hi")])
        response = _build_response("thread_1", result)
        assert response.reply == "I couldn't process that request."

    def test_early_response_used_as_reply_when_no_ai_messages(self):
        """Compound intent: early_response in state must become the reply, not the generic fallback."""
        result = self._make_result(
            messages=[HumanMessage(content="find houses and book an inspection")],
            early_response="I can only handle one request at a time. Would you like to search for properties, or book an inspection?",
        )
        response = _build_response("t", result)
        assert "one request at a time" in response.reply

    def test_thread_id_echoed(self):
        response = _build_response("my-thread-99", self._make_result())
        assert response.thread_id == "my-thread-99"

    def test_requires_human_reply_when_no_ai_message(self):
        """Escalation path: no AIMessage in state → escalation reply, not generic fallback."""
        result = self._make_result(
            messages=[HumanMessage(content="book")],
            requires_human=True,
        )
        response = _build_response("t", result)
        assert "team member" in response.reply.lower()
        assert response.reply != "I couldn't process that request."

    def test_deposit_populated_from_state(self):
        result = self._make_result(deposit_result={"listing_id": "abc", "session_url": "https://stripe.com/x"})
        response = _build_response("t", result)
        assert response.deposit == {"listing_id": "abc", "session_url": "https://stripe.com/x"}

    def test_deposit_none_by_default(self):
        response = _build_response("t", self._make_result())
        assert response.deposit is None

    def test_property_id_set_when_single_search_result(self):
        result = self._make_result(search_results=[{"property_id": "abc-123"}])
        response = _build_response("t", result)
        assert response.property_id == "abc-123"

    def test_property_id_none_when_multiple_search_results(self):
        result = self._make_result(search_results=[
            {"property_id": "abc-123"},
            {"property_id": "def-456"},
        ])
        response = _build_response("t", result)
        assert response.property_id is None

    def test_property_id_none_when_no_search_results(self):
        response = _build_response("t", self._make_result())
        assert response.property_id is None


# ── _extract_single_property_id ───────────────────────────────────────────────

class TestExtractSinglePropertyId:
    def test_returns_id_when_exactly_one_result(self):
        assert _extract_single_property_id([{"property_id": "abc-123"}]) == "abc-123"

    def test_returns_none_when_multiple_results(self):
        rows = [{"property_id": "abc"}, {"property_id": "def"}]
        assert _extract_single_property_id(rows) is None

    def test_returns_none_when_empty(self):
        assert _extract_single_property_id([]) is None

    def test_returns_none_when_property_id_missing(self):
        assert _extract_single_property_id([{"address": "1 George St"}]) is None

    def test_coerces_property_id_to_str(self):
        assert _extract_single_property_id([{"property_id": 42}]) == "42"


class TestToSseEvent:
    def test_token_event(self):
        chunk = type("Chunk", (), {"content": "Hello"})()
        event = {
            "event": "on_chat_model_stream",
            "name": "",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": chunk}
        }

        result = _to_sse_event(event)
        assert result == {"type": "token", "content": "Hello"}

    def test_empty_token_content_returns_none(self):
        chunk = type("Chunk", (), {"content": ""})()
        event = {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "agent"},
            "data": {"chunk": chunk}
        }

        assert _to_sse_event(event) is None

    def test_tool_start_event(self):
        event = {"event": "on_tool_start", "name": "check_availability"}

        result = _to_sse_event(event)

        assert result == {"type": "tool_start", "tool": "check_availability"}

    def test_tool_end_event(self):
        event = {
            "event": "on_tool_end",
            "name": "book_inspection"
        }

        result = _to_sse_event(event)

        assert result == {"type": "tool_end", "tool": "book_inspection"}

    def test_internal_event_returns_none(self):
        event = {"event": "on_chain_start", "name": "some_internal_node"}
        assert _to_sse_event(event) is None
