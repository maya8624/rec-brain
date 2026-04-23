"""
context_update_node         — runs after the LangGraph tool node
_extract_recent_tool_results — parses the latest ToolMessage batch
_merge_context              — shallow-merges state context with overrides
_handle_*                   — one handler per tool, returns partial updates dict
"""

import json
import logging
from typing import Any
from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import (
    BookingContext,
    BookingStatus,
    RealEstateAgentState,
)
from app.core.constants import ToolNames, StateKeys

logger = logging.getLogger(__name__)


def context_update_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Inspect the latest batch of ToolMessages and update typed context fields.
    Returns only the keys that actually changed; returns {} when there are
    no tool results to process (LangGraph treats {} as a no-op).
    """
    recent_tool_results = _extract_recent_tool_results(state["messages"])

    if not recent_tool_results:
        logger.debug("context_update_node | no tool results found")
        return {}

    context_result: dict[str, Any] = {}

    for tool_name, tool_result in recent_tool_results:
        logger.info("context_update_node | processing tool=%s", tool_name)

        handler = _TOOL_HANDLERS.get(tool_name)
        if handler:
            context = handler(state, tool_result)
            context_result.update(context)

    # Any successful tool resets the error counter so safety_node starts fresh
    if any(tool_result.get("success") for _, tool_result in recent_tool_results):
        context_result["error_count"] = 0

    return context_result


def _handle_check_availability(state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Merge available_slots from the availability API result into BookingContext."""

    if not result.get("success"):
        return {}

    slots = result.get("available_slots") or []
    if not slots:
        return {}

    overrides: dict[str, Any] = {"available_slots": slots}

    # Only backfill property_id if we don't already have one
    existing = dict(state.get("booking_context") or {})

    if result.get("property_id") and not existing.get("property_id"):
        overrides["property_id"] = result["property_id"]

    merged = _merge_context(state, StateKeys.BOOKING_CONTEXT, overrides)

    return {StateKeys.BOOKING_CONTEXT: BookingContext(**merged)}


def _handle_book_inspection(state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Update BookingContext with confirmation details and flip status to confirmed."""

    if not result.get("success"):
        return {}

    merged = _merge_context(
        state,
        StateKeys.BOOKING_CONTEXT,
        {"confirmation_id": result.get("confirmation_id", ""),
         "confirmed_datetime": result.get("confirmed_datetime", "")}
    )

    return {
        StateKeys.BOOKING_CONTEXT: BookingContext(**merged),
        StateKeys.BOOKING_STATUS: BookingStatus(
            awaiting_confirmation=False,
            confirmed=True,
            cancelled=False,
        ),
    }


def _handle_cancel_inspection(state: RealEstateAgentState, result: dict) -> dict[str, Any]:
    """Flip BookingStatus to cancelled."""

    if not result.get("success"):
        return {}

    return {
        StateKeys.BOOKING_STATUS: BookingStatus(
            awaiting_confirmation=False,
            confirmed=False,
            cancelled=True,
        ),
    }


_TOOL_HANDLERS: dict[str, Any] = {
    ToolNames.CHECK_AVAILABILITY: _handle_check_availability,
    ToolNames.BOOK_INSPECTION:    _handle_book_inspection,
    ToolNames.CANCEL_INSPECTION:  _handle_cancel_inspection,
}


def _extract_recent_tool_results(messages: list[AIMessage | ToolMessage]) -> list[tuple[str, dict]]:
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
                content = (
                    json.loads(message.content)
                    if isinstance(message.content, str)
                    else message.content
                )
                results.append((message.name, content))

            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "_extract_recent_tool_results | JSON parse failed for tool=%s",
                    message.name,
                )
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
