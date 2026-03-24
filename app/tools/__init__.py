"""
app/tools/__init__.py
 
Single import point for all agent tools.
get_all_tools() is called by agent_node and build_graph.
 
Tool docstrings ARE the LLM's instructions for when to call each tool.
Keep them precise, example-rich, and updated when behaviour changes.
"""

from app.tools.book_inspection import book_inspection
from app.tools.cancel_inspection import cancel_inspection
from app.tools.check_availability import check_availability

_ALL_TOOLS = [
    check_availability,
    book_inspection,
    cancel_inspection,
]


def get_all_tools() -> list:
    """
    Returns all registered agent tools.
    """
    return _ALL_TOOLS
