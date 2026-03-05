"""
FastAPI dependency injection.
Injects the agent from app.state so routes stay clean.
"""

from fastapi import Request
from langgraph.graph.state import CompiledStateGraph


def get_agent(request: Request) -> CompiledStateGraph:
    """Inject the LangGraph agent built at startup."""

    return request.app.state.ai_agent
