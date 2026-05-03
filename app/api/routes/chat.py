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
from app.core.constants import AppStateKeys, InternalRoutes, Messages, StateKeys
from app.schemas.chat import ChatErrorResponse, ChatRequest, ChatResponse, PropertyListing

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])

# Nodes that build replies in code — no on_chat_model_stream events fire for these
_CODE_REPLY_NODES = frozenset({"listing_search", "hybrid_search"})


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
        data: {"type": "token",      "content": "..."}   — LLM token (append to buffer)
        data: {"type": "message",    "content": "..."}   — code-built reply (render at once)
        data: {"type": "tool_start", "tool": "..."}
        data: {"type": "tool_end",   "tool": "..."}
        data: {"type": "result",     "thread_id": "...", "listings": [...], "property_id": "..."}
        data: {"type": "error",      "message": "..."}
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

# TODO: refactor


async def _event_generator(request: ChatRequest, http_request: Request, agent):
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
        final_state: dict = {}

        async for event in agent.astream_events(input_state, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_end":
                output = event.get("data", {}).get("output") or {}
                if name == "LangGraph":
                    final_state = output
                    if not emitted_tokens and output.get("early_response"):
                        yield _sse("token", content=output["early_response"])
                        emitted_tokens = True
                elif name in _CODE_REPLY_NODES and not emitted_tokens:
                    ai_msgs = [m for m in (output.get(
                        "messages") or []) if isinstance(m, AIMessage)]
                    if ai_msgs:
                        yield _sse("message", content=ai_msgs[-1].content)
                        emitted_tokens = True
                continue

            sse = _to_sse_event(event)
            if sse:
                if sse.get("type") == "token":
                    emitted_tokens = True
                yield f"data: {json.dumps(sse)}\n\n"

        if not emitted_tokens:
            msg = Messages.ESCALATION if final_state.get(
                StateKeys.REQUIRES_HUMAN) else Messages.FALLBACK
            yield _sse("token", content=msg)

        search_results = final_state.get("search_results", [])
        yield _sse("result",
                   thread_id=request.thread_id,
                   listings=[lst.model_dump()
                             for lst in _extract_listings(search_results)],
                   property_id=_extract_single_property_id(search_results))

        yield "data: [DONE]\n\n"

    except Exception as exc:
        logger.exception(
            "chat/stream | error | thread_id=%s | %s", request.thread_id, exc)
        yield _sse("error", message="An unexpected error occurred.")


def _sse(type_: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': type_, **kwargs})}\n\n"


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
        else final_state.get(StateKeys.EARLY_RESPONSE)
        or (
            # requires_human=True: graph exited via safety escalation with no AIMessage.
            # Option B: replace this with a human_escalation_node in graph.py that appends
            # an AIMessage so the graph itself owns the escalation message — worth doing
            # when escalation needs side effects (webhook, CRM notify, staff alert, etc.).
            Messages.ESCALATION
            if final_state.get(StateKeys.REQUIRES_HUMAN)
            else Messages.FALLBACK
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
