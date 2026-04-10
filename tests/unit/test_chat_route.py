"""
Unit tests for the chat route helper functions.
These are pure functions — no HTTP server or LangGraph agent required.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.api.routes.chat import (
    _build_response,
    _extract_sources,
    _extract_tools_used,
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

    def test_intent_populated(self):
        result = self._make_result(user_intent="booking")
        response = _build_response("t", result)
        assert response.intent == "booking"

    def test_booking_confirmed_true(self):
        result = self._make_result(
            booking_status={
                "confirmed": True,
                "cancelled": False,
                "awaiting_confirmation": False
            }
        )
        response = _build_response("t", result)
        assert response.booking_confirmed is True

    def test_booking_cancelled_true(self):
        result = self._make_result(
            booking_status={
                "confirmed": False,
                "cancelled": True,
                "awaiting_confirmation": False
            }
        )
        response = _build_response("t", result)
        assert response.booking_cancelled is True

    def test_confirmation_id_from_booking_context(self):
        result = self._make_result(
            booking_context={"confirmation_id": "CONF-7"})

        response = _build_response("t", result)
        assert response.confirmation_id == "CONF-7"

    def test_requires_human_propagated(self):
        response = _build_response("t", self._make_result(requires_human=True))
        assert response.requires_human is True

    def test_sources_empty_by_default(self):
        response = _build_response("t", self._make_result())
        assert response.sources == []


# ── _extract_tools_used ────────────────────────────────────────────────────────

class TestExtractToolsUsed:
    def test_single_tool_extracted(self):
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_availability",
                    "args": {},
                    "id": "tc_1",
                    "type": "tool_call"
                }
            ],
        )
        assert _extract_tools_used([ai_msg]) == ["check_availability"]

    def test_multiple_tools_across_messages(self):
        msg1 = AIMessage(content="", tool_calls=[{
            "name": "check_availability",
            "args": {},
            "id": "tc_1",
            "type": "tool_call"
        }])

        msg2 = AIMessage(content="", tool_calls=[{
            "name": "book_inspection",
            "args": {},
            "id": "tc_2",
            "type": "tool_call"
        }])

        tools = _extract_tools_used([msg1, msg2])
        assert "check_availability" in tools
        assert "book_inspection" in tools

    def test_deduplicates_repeated_tools(self):
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "check_availability", "args": {},
                    "id": "tc_1", "type": "tool_call"
                },
                {
                    "name": "check_availability", "args": {},
                    "id": "tc_2", "type": "tool_call"
                },
            ],
        )
        assert _extract_tools_used([msg]).count("check_availability") == 1

    def test_no_tool_calls_returns_empty(self):
        assert _extract_tools_used([AIMessage(content="plain response")]) == []

    def test_ignores_non_ai_messages(self):
        assert _extract_tools_used([HumanMessage(content="hi")]) == []


class TestExtractSources:
    def test_sources_from_search_documents_tool_message(self):
        content = {
            "sources": [
                {
                    "document": "lease.pdf", "doc_type": "lease",
                    "page": "2", "relevance_score": 0.91
                },
            ]
        }

        msg = ToolMessage(
            content=json.dumps(content),
            name="search_documents",
            tool_call_id="tc_1"
        )

        sources = _extract_sources([msg])

        assert len(sources) == 1
        assert sources[0].document == "lease.pdf"
        assert sources[0].relevance_score == 0.91

    def test_ignores_non_search_documents_tools(self):
        content = {"sources": [
            {"document": "lease.pdf", "doc_type": "lease",
                "page": "1", "relevance_score": 0.9}
        ]}

        msg = ToolMessage(
            content=json.dumps(content),
            name="book_inspection",
            tool_call_id="tc_1"
        )

        assert _extract_sources([msg]) == []

    def test_returns_empty_when_no_tool_messages(self):
        assert _extract_sources([AIMessage(content="response")]) == []

    def test_malformed_json_returns_empty(self):
        msg = ToolMessage(
            content="not json }",
            name="search_documents",
            tool_call_id="tc_1"
        )

        assert _extract_sources([msg]) == []


class TestToSseEvent:
    def test_token_event(self):
        chunk = type("Chunk", (), {"content": "Hello"})()
        event = {
            "event": "on_chat_model_stream",
            "name": "",
            "data": {"chunk": chunk}
        }

        result = _to_sse_event(event)
        assert result == {"type": "token", "content": "Hello"}

    def test_empty_token_content_returns_none(self):
        chunk = type("Chunk", (), {"content": ""})()
        event = {
            "event": "on_chat_model_stream",
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
