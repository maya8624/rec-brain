"""
Shared utilities for all agent nodes.
"""

import json
import structlog
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from llama_index.core.schema import NodeWithScore

from app.schemas.rag import SourceChunk

from app.agents.state import RealEstateAgentState
from app.core.config import settings
from app.core.constants import AppStateKeys, InternalRoutes, Messages, StateKeys

logger = structlog.get_logger(__name__)


def search_error_response() -> dict:
    return {
        "messages": [AIMessage(content=Messages.SEARCH_ERROR)],
        StateKeys.SEARCH_RESULTS: [],
        StateKeys.RETRIEVED_DOCS: None,
    }


def last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state, or ''."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
            return msg.content
    return ""


def last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    """Return the most recent AIMessage in state, or None."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


def resolve_app_service(config: RunnableConfig, attr: str, caller: str) -> Any:
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
        logger.error("service_resolve_failed", caller=caller, attr=attr, error=str(exc))
        raise RuntimeError(f"{caller} | service '{attr}' not available") from exc


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


def extract_sources(nodes: list[NodeWithScore]) -> list[SourceChunk]:
    """Returns one SourceChunk per retrieved node, preserving retrieval order."""
    return [
        SourceChunk(
            file_name=n.node.metadata.get("file_name", ""),
            page=_parse_page(n.node.metadata),
            score=round(n.score or 0.0, 4),
            text=n.node.get_content(),
        )
        for n in nodes
    ]


def _parse_page(metadata: dict) -> int | None:
    raw = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
    try:
        return int(raw) if raw is not None else None
    except (ValueError, TypeError):
        return None


def vector_payload(nodes: list[NodeWithScore]) -> dict:
    result = [{
        "text": n.node.get_content(),
        "score": n.score,
        "metadata": n.node.metadata
        }
        for n in nodes
    ]

    return {
        "results": result,
        "result_count": len(nodes),
        "source": "vector_db",
    }


def slim_rows(rows: list[dict]) -> list[dict]:
    """Strip unused columns — keeps only what the LLM and frontend need."""
    base = str(settings.BACKEND_BASE_URL).rstrip("/")
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
            "pet_friendly":   row.get("pet_friendly", False),
            "agent_name":     f"{row.get('agent_first_name', '')} {row.get('agent_last_name', '')}".strip(),
            "agent_phone":    row.get("agent_phone", ""),
            "agency_name":    row.get("agency_name", ""),
            "property_url":   f"{base}{InternalRoutes.property_detail(str(row['property_id']))}" if row.get("property_id") else "",
            "image_url":      row.get("image_url"),
        }
        for row in rows
    ]


def format_listings(rows: list[dict]) -> str:
    """Build final display markdown for search results — the LLM outputs this verbatim."""
    count = len(rows)
    header = f"{count} {'property' if count == 1 else 'properties'} found."
    blocks = []
    for i, row in enumerate(rows, 1):
        price = row.get("price", 0)
        listing_type = row.get("listing_type", "")
        price_str = f"${price:,.0f} per week" if listing_type == "Rent" else f"${price:,.0f}"
        blocks.append(
            f"{i}. **[{row['address']}, {row['suburb']} {row['state']}]({row['property_url']})** [property_id={row['property_id']}]\n"
            f"   - Type: {row['property_type']}\n"
            f"   - Price: {price_str}\n"
            f"   - Bedrooms: {row['bedrooms']}\n"
            f"   - Bathrooms: {row['bathrooms']}\n"
            f"   - Pet friendly: {'Yes' if row.get('pet_friendly') else 'No'}\n"
            f"   - Agent: {row['agent_name']} {row['agent_phone']}"
        )
    return header + "\n\n" + "\n\n".join(blocks)
