"""
Shared utilities for all agent nodes.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys

logger = logging.getLogger(__name__)


def last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state, or ''."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else ""
    return ""


def last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    """Return the most recent AIMessage in state, or None."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


def resolve_app_service(config: RunnableConfig, attr: str, caller: str) -> Any | None:
    """
    Extract a service from RunnableConfig configurable by key name.

    Expects the service to be stored directly under config["configurable"][attr].
    Returns None and logs an error rather than raising, so callers can
    decide how to handle the missing service.
    """
    try:
        service = config.get(AppStateKeys.CONFIGURABLE, {}).get(attr)
        if service is None:
            raise ValueError(f"'{attr}' not found in configurable")
        return service

    except Exception as exc:
        logger.error("%s | could not resolve %s: %s", caller, attr, exc)
        return None


def build_tool_message(tool_call_id: str, name: str, content: dict) -> ToolMessage:
    """
    Serialise *content* as JSON and wrap it in a ToolMessage.

    Using json.dumps (not str()) ensures context_update_node can parse
    it back with json.loads without errors.
    """
    return ToolMessage(
        content=json.dumps(content),
        tool_call_id=tool_call_id,
        name=name,
    )


def error_content(error: Exception) -> dict:
    """Standard error payload shape shared by all search nodes."""
    return {"success": False, "error": str(error)}


def nodes_to_dicts(nodes: list) -> list[dict]:
    return [
        {
            "text": n.node.get_content(),
            "score": n.score,
            "metadata": n.node.metadata,
        }
        for n in nodes
    ]


def vector_payload(nodes: list) -> dict:
    return {
        "results": nodes_to_dicts(nodes),
        "result_count": len(nodes),
        "source": "vector_db",
    }


def slim_rows(rows: list[dict]) -> list[dict]:
    """Strip unused columns — keeps only what the LLM and frontend need."""
    return [
        {
            "property_id":     str(row["property_id"]) if row.get("property_id") else "",
            "listing_id":      str(row["listing_id"]) if row.get("listing_id") else "",
            "address":        row.get("address_line1", ""),
            "suburb":         row.get("suburb", ""),
            "state":          row.get("state", ""),
            "postcode":       row.get("postcode", ""),
            "price":          row.get("price", 0),
            "bedrooms":       row.get("bedrooms", 0),
            "bathrooms":      row.get("bathrooms", 0),
            "car_spaces":     row.get("car_spaces", 0),
            "property_type":  row.get("property_type", ""),
            "listing_type":   row.get("listing_type", ""),
            "listing_status": row.get("listing_status", ""),
            "agent_name":     f"{row.get('agent_first_name', '')} {row.get('agent_last_name', '')}".strip(),
            "agent_phone":    row.get("agent_phone", ""),
            "agency_name":    row.get("agency_name", ""),
        }
        for row in rows
    ]


def listing_summary(rows: list[dict]) -> str:
    """Compact one-line-per-listing text for LLM — saves tokens vs full JSON."""
    return "\n".join(
        f"{i+1}. [property_id={row['property_id']}] {row['address']}, {row['suburb']} {row['state']} — "
        f"${row['price']:,.0f} | {row['bedrooms']}bed {row['bathrooms']}bath | "
        f"{row['property_type']} | {row['listing_type']} | "
        f"Agent: {row['agent_name']} {row['agent_phone']}"
        for i, row in enumerate(rows)
    )


def format_search_reply(rows: list[dict], count: int) -> str:
    """User-facing markdown reply built in code — no LLM involved."""
    header = f"Found {count} {'property' if count == 1 else 'properties'}:\n"
    blocks = []
    for i, row in enumerate(rows, start=1):
        price = f"${row['price']:,.0f}/week" if row["listing_type"] == "Rent" else f"${row['price']:,.0f}"
        blocks.append(
            f"**{i}. {row['address']}, {row['suburb']} {row['state']}**\n"
            f"{price} | {row['bedrooms']} bed | {row['bathrooms']} bath | "
            f"{row['property_type']} | {row['listing_type']}\n"
            f"Agent: {row['agent_name']} {row['agent_phone']}"
        )
    suffix = (
        "\n\n_Showing top 10 results. Narrow your search criteria to see more specific results._"
        if count == 10 else ""
    )
    return header + "\n\n".join(blocks) + suffix
