"""
Unit tests for context_update_node and its private helpers.

context_update_node is synchronous — no DB or LLM required.
State is built by hand using LangChain message types.
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.nodes.context import context_update_node
from app.agents.state import BookingContext, BookingStatus
from app.core.constants import ToolNames


def _tool_msg(name: str, content: dict, tool_call_id: str = "tc_1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(content),
        name=name,
        tool_call_id=tool_call_id,
    )


def _ai_with_tool_call(name: str = "check_availability") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {},
                     "id": "tc_1", "type": "tool_call"}],
    )


def make_state(tool_messages: list, booking_context: dict | None = None, **kwargs) -> dict:
    return {
        "messages": [
            HumanMessage(content="book inspection"),
            _ai_with_tool_call(),
            *tool_messages,
        ],
        "user_intent": "booking",
        "error_count": 0,
        "requires_human": False,
        "booking_context": BookingContext(**(booking_context or {})),
        "booking_status": BookingStatus(awaiting_confirmation=False, confirmed=False, cancelled=False),
        **kwargs,
    }


class TestContextUpdateNoOp:
    def test_no_tool_messages_returns_empty(self):
        state = {
            "messages": [HumanMessage(content="book"), _ai_with_tool_call()],
            "booking_context": BookingContext(),
            "booking_status": BookingStatus(awaiting_confirmation=False, confirmed=False, cancelled=False),
            "error_count": 0,
        }
        assert context_update_node(state) == {}

    def test_failed_check_availability_returns_empty(self):
        state = make_state([_tool_msg(ToolNames.CHECK_AVAILABILITY, {
                           "success": False, "error": "timeout"})])
        result = context_update_node(state)
        assert "booking_context" not in result

    def test_failed_book_inspection_returns_empty(self):
        state = make_state([_tool_msg(ToolNames.BOOK_INSPECTION, {
                           "success": False, "error": "slot gone"})])
        result = context_update_node(state)
        assert "booking_status" not in result

    def test_failed_cancel_inspection_returns_empty(self):
        state = make_state(
            [_tool_msg(ToolNames.CANCEL_INSPECTION, {"success": False})])
        result = context_update_node(state)
        assert "booking_status" not in result


class TestHandleCheckAvailability:
    def test_success_updates_booking_context_with_slots(self):
        slots = [
            {"datetime": "2026-04-12 10:00",
                "agent_name": "Jane Smith", "available": True},
            {"datetime": "2026-04-12 14:00",
                "agent_name": "Jane Smith", "available": True},
        ]
        state = make_state([_tool_msg(ToolNames.CHECK_AVAILABILITY, {
            "success": True,
            "property_id": "prop_123",
            "available_slots": slots,
            "slot_count": 2,
        })])
        result = context_update_node(state)
        assert "booking_context" in result
        assert result["booking_context"]["available_slots"] == slots

    def test_success_backfills_property_id_if_missing(self):
        state = make_state(
            [_tool_msg(ToolNames.CHECK_AVAILABILITY, {
                "success": True,
                "property_id": "prop_555",
                "available_slots": [{"datetime": "2026-04-12 10:00"}],
                "slot_count": 1,
            })],
            booking_context={},  # no property_id yet
        )
        result = context_update_node(state)
        assert result["booking_context"]["property_id"] == "prop_555"

    def test_success_does_not_overwrite_existing_property_id(self):
        state = make_state(
            [_tool_msg(ToolNames.CHECK_AVAILABILITY, {
                "success": True,
                "property_id": "prop_NEW",
                "available_slots": [{"datetime": "2026-04-12 10:00"}],
                "slot_count": 1,
            })],
            booking_context={"property_id": "prop_ORIG"},
        )
        result = context_update_node(state)
        assert result["booking_context"]["property_id"] == "prop_ORIG"

    def test_no_slots_returns_empty(self):
        state = make_state([_tool_msg(ToolNames.CHECK_AVAILABILITY, {
            "success": True,
            "property_id": "prop_123",
            "available_slots": [],
            "slot_count": 0,
        })])
        result = context_update_node(state)
        assert "booking_context" not in result


class TestHandleBookInspection:
    def test_success_sets_confirmation_id(self):
        state = make_state([_tool_msg(ToolNames.BOOK_INSPECTION, {
            "success": True,
            "confirmation_id": "CONF-9999",
            "confirmed_datetime": "2026-04-12 10:00",
        })])
        result = context_update_node(state)
        assert result["booking_context"]["confirmation_id"] == "CONF-9999"

    def test_success_flips_booking_status_to_confirmed(self):
        state = make_state([_tool_msg(ToolNames.BOOK_INSPECTION, {
            "success": True,
            "confirmation_id": "CONF-9999",
            "confirmed_datetime": "2026-04-12 10:00",
        })])
        result = context_update_node(state)
        assert result["booking_status"]["confirmed"] is True
        assert result["booking_status"]["awaiting_confirmation"] is False
        assert result["booking_status"]["cancelled"] is False

    def test_success_resets_error_count(self):
        state = make_state(
            [_tool_msg(ToolNames.BOOK_INSPECTION, {
                "success": True,
                "confirmation_id": "CONF-9999",
                "confirmed_datetime": "2026-04-12 10:00",
            })],
            error_count=2,
        )
        result = context_update_node(state)
        assert result["error_count"] == 0


class TestHandleCancelInspection:
    def test_success_flips_status_to_cancelled(self):
        state = make_state(
            [_tool_msg(ToolNames.CANCEL_INSPECTION, {"success": True})])
        result = context_update_node(state)
        assert result["booking_status"]["cancelled"] is True
        assert result["booking_status"]["confirmed"] is False
        assert result["booking_status"]["awaiting_confirmation"] is False


class TestContextUpdateJsonResilience:
    def test_malformed_json_tool_message_does_not_crash(self):
        bad_msg = ToolMessage(
            content="not valid json {{{{",
            name=ToolNames.BOOK_INSPECTION,
            tool_call_id="tc_1",
        )
        state = {
            "messages": [
                HumanMessage(content="book"),
                _ai_with_tool_call(),
                bad_msg,
            ],
            "booking_context": BookingContext(),
            "booking_status": BookingStatus(awaiting_confirmation=False, confirmed=False, cancelled=False),
            "error_count": 0,
        }
        # Must not raise — returns {} because parsed content has no "success": True
        result = context_update_node(state)
        assert isinstance(result, dict)
