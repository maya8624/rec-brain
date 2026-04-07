"""
Shared utilities for all agent nodes.

Kept internal to this package (underscore-prefixed) — import from the
node modules, not directly from here.
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState

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
    Extract a service from FastAPI app.state via RunnableConfig.

    Looks up ``request.app.state.<attr>`` from the configurable context.
    Returns None and logs an error rather than raising, so callers can
    decide how to handle the missing service.
    """
    try:
        request = config.get("configurable", {}).get("request")
        if request is None:
            raise ValueError("no 'request' key in configurable")

        service = getattr(request.app.state, attr, None)
        if service is None:
            raise ValueError(f"app.state.{attr} is not set")

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
