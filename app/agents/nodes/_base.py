"""
Shared utilities used by vector.py and sql.py.

Kept internal to this package (underscore-prefixed) — import from the
node modules, not directly from here.
"""

import json

from langchain_core.messages import AIMessage, ToolMessage

from app.agents.state import RealEstateAgentState


def last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    """Return the most recent AIMessage in state, or None."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
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
    """Standard error payload shape shared by both search nodes."""
    return {"success": False, "error": str(error)}
