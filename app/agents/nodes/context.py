"""
context_update_node — runs after the LangGraph tool node.

Parses raw ToolMessage JSON and maps results into the strongly-typed
context fields on RealEstateAgentState, so the router and subsequent
agent turns always have clean structured data rather than raw strings.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import (
    BookingContext,
    BookingStatus,
    PropertyContext,
    RealEstateAgentState,
    SearchContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def context_update_node(state: RealEstateAgentState) -> dict:
    """
    Inspect the latest batch of ToolMessages and update typed context fields.

    Returns only the keys that actually changed; returns {} when there are
    no tool results to process (LangGraph treats {} as a no-op).
    """
    messages = list(state["messages"])
    recent_tool_results = _extract_recent_tool_results(messages)

    if not recent_tool_results:
        logger.debug("context_update_node | no tool results found")
        return {}

    updates: dict[str, Any] = {}

    for tool_name, result in recent_tool_results:
        logger.info("context_update_node | processing tool=%s", tool_name)

        if tool_name == "search_listings":
            ctx = _extract_search_context(state, result)
            if ctx:
                updates["search_context"] = ctx

        elif tool_name == "get_property_details":
            ctx = _extract_property_context(result)
            if ctx:
                updates["property_context"] = ctx

        elif tool_name == "check_inspection_availability":
            ctx = _extract_availability_context(state, result)
            if ctx:
                updates["booking_context"] = ctx

        elif tool_name == "book_inspection":
            booking_ctx, booking_status = _extract_booking_confirmed(
                state, result)
            if booking_ctx:
                updates["booking_context"] = booking_ctx
            if booking_status:
                updates["booking_status"] = booking_status

        elif tool_name == "cancel_inspection":
            booking_status = _extract_cancellation_status(result)
            if booking_status:
                updates["booking_status"] = booking_status

    # Any successful tool resets the error counter so safety_node starts fresh
    if any(r.get("success") for _, r in recent_tool_results):
        updates["error_count"] = 0

    return updates


# ---------------------------------------------------------------------------
# Helpers — message parsing
# ---------------------------------------------------------------------------

def _extract_recent_tool_results(messages: list) -> list[tuple[str, dict]]:
    """
    Walk back from the end of the message list to find the most recent
    batch of ToolMessages (everything between END and the preceding AIMessage).

    JSON parse errors fall back to a plain text dict.  The fallback does NOT
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
                # Preserve the raw text but do NOT claim success
                results.append((message.name, {"output": message.content}))

    return results


# ---------------------------------------------------------------------------
# Helpers — typed context builders
# ---------------------------------------------------------------------------

def _merge_context(state: RealEstateAgentState, key: str, overrides: dict) -> dict:
    """
    Shallow-merge *overrides* onto the existing context dict stored at *key*.

    Avoids the repetitive `dict(state.get(key, {}))` pattern found in
    every extractor.
    """
    base = dict(state.get(key) or {})
    base.update(overrides)
    return base


def _extract_search_context(
    state: RealEstateAgentState,
    result: dict,
) -> SearchContext | None:
    """Merge result_count into existing SearchContext."""
    if not result.get("success"):
        return None

    merged = _merge_context(state, "search_context", {
        "last_result_count": result.get("result_count", 0),
    })
    return SearchContext(**merged)


def _extract_property_context(result: dict) -> PropertyContext | None:
    """Build PropertyContext from get_property_details tool result."""
    if not result.get("success"):
        return None

    data = result.get("property") or {}
    if not data:
        return None

    return PropertyContext(
        property_id=data.get("property_id", ""),
        address=data.get("address", ""),
        suburb=data.get("suburb", ""),
        price=data.get("price", 0.0),
        bedrooms=data.get("bedrooms", 0),
        bathrooms=data.get("bathrooms", 0),
        property_type=data.get("property_type", ""),
        agent_id=data.get("agent_id", ""),
        agent_name=data.get("agent_name", ""),
        agent_phone=data.get("agent_phone", ""),
    )


def _extract_availability_context(
    state: RealEstateAgentState,
    result: dict,
) -> BookingContext | None:
    """Merge available_slots from .NET availability API result into BookingContext."""
    if not result.get("success"):
        return None

    slots = result.get("available_slots") or []
    if not slots:
        return None

    overrides: dict[str, Any] = {"available_slots": slots}
    # Only backfill property_id if we don't already have one
    if result.get("property_id"):
        overrides.setdefault("property_id", result["property_id"])

    merged = _merge_context(state, "booking_context", overrides)
    return BookingContext(**merged)


def _extract_booking_confirmed(
    state: RealEstateAgentState,
    result: dict,
) -> tuple[BookingContext | None, BookingStatus | None]:
    """
    After a successful book_inspection call, update BookingContext with
    confirmation details and flip BookingStatus to confirmed.
    """
    if not result.get("success"):
        return None, None

    merged = _merge_context(state, "booking_context", {
        "confirmation_id": result.get("confirmation_id", ""),
        "confirmed_datetime": result.get("confirmed_datetime", ""),
    })
    booking_ctx = BookingContext(**merged)
    booking_status = BookingStatus(
        awaiting_confirmation=False,
        confirmed=True,
        cancelled=False,
    )
    return booking_ctx, booking_status


def _extract_cancellation_status(result: dict) -> BookingStatus | None:
    """Flip BookingStatus to cancelled after a successful cancel_inspection call."""
    if not result.get("success"):
        return None

    return BookingStatus(
        awaiting_confirmation=False,
        confirmed=False,
        cancelled=True,
    )
