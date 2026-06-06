"""
context_update_node         — runs after the LangGraph tool node
_collect_tool_results       — parses the latest ToolMessage batch
_merge_context              — shallow-merges state context with overrides
_handle_*                   — one handler per tool, returns partial updates dict
"""

import json
import structlog
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage

from app.core.constants import ToolNames, StateKeys
from app.agents.state import (
    BookingContext,
    ConversationPhase,
    RealEstateAgentState,
)

logger = structlog.get_logger(__name__)


def context_update_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Inspect the latest batch of ToolMessages and update typed context fields.
    Returns only the keys that actually changed; returns {} when there are
    no tool results to process (LangGraph treats {} as a no-op).
    """
    recent_tool_results = _collect_tool_results(state["messages"])

    if not recent_tool_results:
        logger.debug("context_update_no_tool_results")
        return {}

    context_result: dict[str, Any] = {}

    for tool_name, tool_result in recent_tool_results:
        logger.info("context_update_processing_tool", tool=tool_name)

        handler = _TOOL_HANDLERS.get(tool_name)
        if handler:
            context = handler(state, tool_result)
            context_result.update(context)

    # Any successful tool resets the error counter so safety_node starts fresh
    if any(tool_result.get("success") for _, tool_result in recent_tool_results):
        context_result[StateKeys.ERROR_COUNT] = 0

    return context_result


def _handle_check_availability(_state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Reset BookingContext with fresh slots and property_id for the selected property."""
    if not result.get("success"):
        return {}

    slots = result.get("available_slots")
    if not slots:
        return {}

    return {
        StateKeys.BOOKING_CONTEXT: BookingContext(
            available_slots=slots,
            property_id=result.get("property_id", ""),
        ),
        StateKeys.PHASE: ConversationPhase.BOOKING_PENDING,
    }


def _handle_book_inspection(_state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Update BookingContext with confirmation details and flip status to confirmed."""
    if not result.get("success"):
        return {}

    return {
        StateKeys.BOOKING_CONTEXT: BookingContext(
            property_id=result.get("property_id", ""),
            property_address=result.get("property_address", ""),
            confirmation_id=result.get("confirmation_id", ""),
            confirmed_datetime=result.get("confirmed_datetime", ""),
            confirmed=True,
            cancelled=False,
        ),
        StateKeys.PHASE: ConversationPhase.BOOKING_CONFIRMED,
    }


def _handle_cancel_inspection(_state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Flip BookingContext to cancelled."""
    if not result.get("success"):
        return {}

    return {
        StateKeys.BOOKING_CONTEXT: BookingContext(
            cancelled=True,
            confirmed=False,
        ),
        StateKeys.PHASE: ConversationPhase.IDLE,
    }


def _handle_get_booking(_state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Persist a uniquely identified booking for follow-up actions like cancellation."""
    if not result.get("success"):
        return {}

    booking_data: dict[str, Any] | None = None
    if result.get("confirmation_id"):
        booking_data = result
    else:
        bookings = result.get("bookings") or []
        if len(bookings) == 1:
            booking_data = bookings[0]

    if not booking_data:
        return {}

    merged = _merge_context(
        _state,
        StateKeys.BOOKING_CONTEXT,
        {
            "confirmation_id": booking_data.get("confirmation_id", ""),
            "property_id": booking_data.get("property_id", ""),
            "property_address": booking_data.get("property_address", ""),
        },
    )
    return {StateKeys.BOOKING_CONTEXT: BookingContext(**merged)}


def _handle_get_deposit(_state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Store deposit result in state so it can be forwarded to the frontend via SSE."""
    return {
        StateKeys.DEPOSIT_RESULT: result if result.get("success") else None,
        StateKeys.PHASE: ConversationPhase.DEPOSIT_CONFIRMED,
    }


_TOOL_HANDLERS: dict[str, Any] = {
    ToolNames.CHECK_AVAILABILITY: _handle_check_availability,
    ToolNames.BOOK_INSPECTION:    _handle_book_inspection,
    ToolNames.CANCEL_INSPECTION:  _handle_cancel_inspection,
    ToolNames.GET_BOOKING:        _handle_get_booking,
    ToolNames.GET_DEPOSIT:        _handle_get_deposit,
}


def _collect_tool_results(messages: list[AIMessage | ToolMessage]) -> list[tuple[str, dict]]:
    """
    Walk back from the end of the message list to find the most recent
    batch of ToolMessages (everything between END and the preceding AIMessage).

    Any message type other than ToolMessage is silently skipped —
    only AIMessage acts as the stop sentinel.

    JSON parse errors fall back to a plain text dict. The fallback does NOT
    set success=True — callers must check result.get("success") themselves.
    """
    results = []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            try:
                content = json.loads(message.content)
                results.append((message.name, content))
            except (json.JSONDecodeError, TypeError):
                # logger.error("_collect_tool_results | JSON parse failed for tool=%s", message.name)
                results.append((message.name, {"output": message.content}))

    return results


def _merge_context(state: RealEstateAgentState, key: str, overrides: dict) -> dict:
    """
    Shallow-merge overrides onto the existing context stored at key.
    Copies the existing context first so state is never mutated.

    """
    base = dict(state.get(key) or {})
    base.update(overrides)
    return base
