"""
All traffic arrives from .NET backend at this /api/chat endpoint.
No AI logic here — only HTTP concerns
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import initial_state
from app.api.dependencies import get_agent, verify_internal_key, CompiledStateGraph
from app.core.config import settings
from app.core.constants import AppStateKeys, InternalRoutes
from app.schemas.chat import ChatErrorResponse, ChatRequest, ChatResponse, PropertyListing

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse, dependencies=[Depends(verify_internal_key)])
async def chat(
        request: ChatRequest,
        http_request: Request,
        agent: CompiledStateGraph = Depends(get_agent)) -> ChatResponse:
    """
    Main chat endpoint — called by .NET backend for every user message.
    """

    try:
        config: RunnableConfig = {
            AppStateKeys.CONFIGURABLE: {
                AppStateKeys.THREAD_ID:       request.thread_id,
                AppStateKeys.USER_ID:         request.user_id,
                AppStateKeys.BOOKING_SERVICE: http_request.app.state.booking_service,
                AppStateKeys.SQL_VIEW_SERVICE: http_request.app.state.sql_view_service,
                AppStateKeys.RAG_SERVICE:     http_request.app.state.rag_service,
            }
        }

        if request.is_new_conversation:
            input_state = initial_state()
            input_state["messages"] = [HumanMessage(content=request.message)]
        else:
            # LangGraph reload existing state from checkpointer automatically
            input_state = {"messages": [HumanMessage(content=request.message)]}

        final_state = await agent.ainvoke(input=input_state, config=config)
        chat_response = _build_response(request.thread_id, final_state)
        return chat_response

    except Exception as exc:
        logger.exception(
            "chat | error | thread_id=%s | %s",
            request.thread_id, exc)

        raise HTTPException(
            status_code=500,
            detail=ChatErrorResponse(
                error="AI service error. Please try again.",
                thread_id=request.thread_id,
            ).model_dump(),  # convert Pydantic model to dict for JSON response
        ) from exc


@router.post("/stream", dependencies=[Depends(verify_internal_key)])
async def chat_stream(
        request: ChatRequest,
        http_request: Request,
        agent: CompiledStateGraph = Depends(get_agent)) -> StreamingResponse:
    """
    Streaming chat — returns tokens via Server-Sent Events.
    Connect from React using EventSource or fetch with streaming.

    SSE events:
        data: {"type": "token", "content": "Hello"}
        data: {"type": "tool_start", "tool": "search_listings"}
        data: {"type": "tool_end", "tool": "search_listings"}
        data: {"type": "error", "message": "..."}
        data: [DONE]
    """
    request_id = getattr(http_request.state, "request_id", "unknown")

    logger.info(
        "chat/stream | thread_id=%s | user=%s | id=%s",
        request.thread_id, request.user_id, request_id,
    )

    return StreamingResponse(
        _event_generator(request, http_request, agent),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable nginx buffering
            "X-Request-ID": request_id,
        },
    )


async def _event_generator(request: ChatRequest, http_request: Request, agent):
    """
    Mirrors the pattern from your original stream_agent() approach.
    """
    try:
        config = {
            AppStateKeys.CONFIGURABLE: {
                AppStateKeys.THREAD_ID:       request.thread_id,
                AppStateKeys.USER_ID:         request.user_id,
                AppStateKeys.BOOKING_SERVICE: http_request.app.state.booking_service,
                AppStateKeys.SQL_VIEW_SERVICE: http_request.app.state.sql_view_service,
                AppStateKeys.RAG_SERVICE:     http_request.app.state.rag_service,
            }
        }

        if request.is_new_conversation:
            input_state = initial_state()
            input_state["messages"] = [HumanMessage(content=request.message)]
        else:
            input_state = {"messages": [HumanMessage(content=request.message)]}

        emitted_tokens = False
        async for event in agent.astream_events(input_state, config=config, version="v2"):
            # Capture early_response from graph end event (compound intent)
            if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                output = event.get("data", {}).get("output", {})
                early = output.get("early_response")
                if early and not emitted_tokens:
                    yield f"data: {json.dumps({'type': 'token', 'content': early})}\n\n"
                continue

            sse = _to_sse_event(event)
            if sse:
                if sse.get("type") == "token":
                    emitted_tokens = True
                yield f"data: {json.dumps(sse)}\n\n"

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception(
            "chat/stream | error | thread_id=%s | %s",
            request.thread_id, exc,
        )
        yield f"data: {json.dumps({'type': 'error', 'message': 'An unexpected error occurred.'})}\n\n"


def _to_sse_event(event: dict) -> dict | None:
    """
    Converts a LangGraph astream_events dict into an SSE payload.
    Returns None for internal events we don't surface to the frontend.
    """
    kind = event.get("event", "")
    name = event.get("name", "")

    # LLM is generating a token — send it immediately
    if kind == "on_chat_model_stream":
        chunk = event.get("data", {}).get("chunk")
        if chunk and hasattr(chunk, "content") and chunk.content:
            return {"type": "token", "content": chunk.content}

    # Tool starting — frontend can show "Searching..." indicator
    if kind == "on_tool_start":
        return {"type": "tool_start", "tool": name}

    # Tool finished — frontend can hide indicator
    if kind == "on_tool_end":
        return {"type": "tool_end", "tool": name}

    return None


def _build_response(thread_id: str, final_state: dict) -> ChatResponse:
    """
    Builds a ChatResponse from the final LangGraph state dict.
    """

    ai_messages = [
        message for message in final_state.get("messages", [])
        if isinstance(message, AIMessage)
    ]

    reply = (
        ai_messages[-1].content
        if ai_messages
        else final_state.get("early_response")
        or (
            # requires_human=True: graph exited via safety escalation with no AIMessage.
            # Option B: replace this with a human_escalation_node in graph.py that appends
            # an AIMessage so the graph itself owns the escalation message — worth doing
            # when escalation needs side effects (webhook, CRM notify, staff alert, etc.).
            "I'm having trouble completing this — a team member will follow up shortly."
            if final_state.get("requires_human")
            else "I couldn't process that request."
        )
    )

    search_results = final_state.get("search_results", [])

    return ChatResponse(
        reply=reply,
        thread_id=thread_id,
        listings=_extract_listings(search_results),
        property_id=_extract_single_property_id(search_results),
    )


def _extract_single_property_id(search_results: list[dict]) -> str | None:
    """Return the property_id only when exactly one result exists — used to trigger the Book button."""
    if len(search_results) == 1:
        pid = search_results[0].get("property_id")
        return str(pid) if pid else None

    return None


def _extract_listings(rows: list[dict]) -> list[PropertyListing]:
    """Convert slim state rows into Listing response models."""
    base = str(settings.BACKEND_BASE_URL).rstrip("/")
    listings = []

    for row in rows:
        property_id = row.get("property_id")
        property_url = f"{base}{InternalRoutes.property_detail(property_id)}" if property_id else None
        listings.append(PropertyListing(**row, property_url=property_url))

    return listings
