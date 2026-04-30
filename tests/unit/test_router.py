"""
Unit tests for all LangGraph conditional edge functions in app/agents/router.py.

"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.router import (
    route_after_context,
    route_after_safety,
    route_after_search,
    route_after_tools,
    route_agent_output,
    route_intent_output,
)
from app.core.constants import Node


# ── Helpers ────────────────────────────────────────────────────────────────────

def ai_with_tool_call(tool_name: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": tool_name,
                "args": {},
                "id": "tc_1",
                "type": "tool_call"
            }
        ],
    )


def tool_message(name: str, content: dict, tool_call_id: str = "tc_1") -> ToolMessage:
    return ToolMessage(content=json.dumps(content), name=name, tool_call_id=tool_call_id)


def base_state(**kwargs) -> dict:
    return {"messages": [], "requires_human": False, "error_count": 0, **kwargs}


# ── route_intent_output ────────────────────────────────────────────────────────

class TestRouteIntentOutput:
    def test_early_response_goes_to_end(self):
        state = base_state(
            early_response="Please clarify.",
            user_intent="general"
        )

        assert route_intent_output(state) == Node.END

    def test_search_goes_to_listing_search(self):
        assert route_intent_output(base_state(
            user_intent="search"
        )) == Node.LISTING_SEARCH

    def test_document_query_goes_to_vector_search(self):
        assert route_intent_output(base_state(
            user_intent="document_query"
        )) == Node.VECTOR_SEARCH

    def test_hybrid_search_goes_to_hybrid_search(self):
        assert route_intent_output(base_state(
            user_intent="hybrid_search"
        )) == Node.HYBRID_SEARCH

    def test_booking_goes_to_agent(self):
        assert route_intent_output(base_state(
            user_intent="booking")) == Node.AGENT

    def test_cancellation_goes_to_agent(self):
        assert route_intent_output(base_state(
            user_intent="cancellation")) == Node.AGENT

    def test_general_goes_to_agent(self):
        assert route_intent_output(base_state(
            user_intent="general")) == Node.AGENT

    def test_search_then_book_goes_to_listing_search(self):
        assert route_intent_output(base_state(
            user_intent="search_then_book"
        )) == Node.LISTING_SEARCH

    def test_unknown_intent_defaults_to_agent(self):
        assert route_intent_output(base_state(
            user_intent="unknown")) == Node.AGENT

    def test_no_early_response_key(self):
        """Missing early_response key must not crash and must route normally."""
        state = {
            "messages": [],
            "user_intent": "search",
            "requires_human": False

        }
        assert route_intent_output(state) == Node.LISTING_SEARCH


class TestRouteAgentOutput:
    def test_no_ai_message_goes_to_end(self):
        state = base_state(messages=[HumanMessage(content="hello")])
        assert route_agent_output(state) == Node.END

    def test_plain_ai_response_goes_to_end(self):
        state = base_state(
            messages=[AIMessage(content="Here are the properties…")])
        assert route_agent_output(state) == Node.END

    def test_check_availability_goes_to_tools(self):
        state = base_state(messages=[ai_with_tool_call("check_availability")])
        assert route_agent_output(state) == Node.TOOLS

    def test_book_inspection_goes_to_tools(self):
        state = base_state(messages=[ai_with_tool_call("book_inspection")])
        assert route_agent_output(state) == Node.TOOLS

    def test_cancel_inspection_goes_to_tools(self):
        state = base_state(messages=[ai_with_tool_call("cancel_inspection")])
        assert route_agent_output(state) == Node.TOOLS

    def test_unrecognised_tool_goes_to_end(self):
        state = base_state(messages=[ai_with_tool_call("unknown_tool")])
        assert route_agent_output(state) == Node.END

    def test_requires_human_goes_to_end(self):
        state = base_state(
            messages=[ai_with_tool_call("book_inspection")],
            requires_human=True,
        )
        assert route_agent_output(state) == Node.END


# ── route_after_search ─────────────────────────────────────────────────────────

class TestRouteAfterSearch:
    def test_normal_goes_to_agent(self):
        assert route_after_search(base_state()) == Node.AGENT

    def test_requires_human_goes_to_end(self):
        assert route_after_search(base_state(requires_human=True)) == Node.END



# ── route_after_tools ──────────────────────────────────────────────────────────

class TestRouteAfterTools:
    def test_successful_tool_goes_to_context_update(self):
        state = base_state(messages=[
            HumanMessage(content="book"),
            ai_with_tool_call("book_inspection"),
            tool_message("book_inspection", {
                         "success": True, "confirmation_id": "CONF-1"}),
        ])
        assert route_after_tools(state) == Node.CONTEXT_UPDATE

    def test_failed_tool_goes_to_safety(self):
        state = base_state(messages=[
            HumanMessage(content="book"),
            ai_with_tool_call("book_inspection"),
            tool_message("book_inspection", {
                         "success": False, "error": "Backend down"}),
        ])
        assert route_after_tools(state) == Node.SAFETY

    def test_no_tool_messages_goes_to_safety(self):
        """No ToolMessages between last AIMessage and end → safety."""
        state = base_state(messages=[
            HumanMessage(content="book"),
            ai_with_tool_call("book_inspection"),
        ])
        assert route_after_tools(state) == Node.SAFETY

    def test_requires_human_goes_to_end(self):
        state = base_state(
            messages=[
                ai_with_tool_call("book_inspection"),
                tool_message("book_inspection", {"success": True}),
            ],
            requires_human=True,
        )
        assert route_after_tools(state) == Node.END

    def test_mixed_results_with_one_success_goes_to_context_update(self):
        """At least one success → context_update (not safety)."""
        state = base_state(messages=[
            HumanMessage(content="book"),
            ai_with_tool_call("check_availability"),
            tool_message("check_availability", {
                         "success": True, "slot_count": 3}, "tc_1"),
            tool_message("book_inspection", {
                         "success": False, "error": "timeout"}, "tc_2"),
        ])
        assert route_after_tools(state) == Node.CONTEXT_UPDATE


# ── route_after_context / route_after_safety ───────────────────────────────────

class TestRouteAfterContextAndSafety:
    def test_after_context_goes_to_agent(self):
        assert route_after_context(base_state()) == Node.AGENT

    def test_after_context_requires_human_goes_to_end(self):
        assert route_after_context(base_state(requires_human=True)) == Node.END

    def test_after_safety_goes_to_agent(self):
        assert route_after_safety(base_state()) == Node.AGENT

    def test_after_safety_requires_human_goes_to_end(self):
        assert route_after_safety(base_state(requires_human=True)) == Node.END
