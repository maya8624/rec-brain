"""
FastAPI dependency injection.
Injects the agent from app.state so routes stay clean.
"""

from typing import Annotated
from fastapi import HTTPException, Request
from fastapi.params import Header
from langgraph.graph.state import CompiledStateGraph
from app.core.config import settings


def get_agent(request: Request) -> CompiledStateGraph:
    """Inject the LangGraph agent built at startup."""
    return request.app.state.ai_agent


async def verify_internal_key(x_api_key: Annotated[str, Header(alias="X-API-Key")]) -> str:
    """
    Dependency — verifies the internal service API key.
    Attach to any route that should only be called by .NET backend.
    """

    if x_api_key != settings.BACKEND_API_KEY:
        raise HTTPException(
            status_code=403, detail="Forbidden")
    return x_api_key
