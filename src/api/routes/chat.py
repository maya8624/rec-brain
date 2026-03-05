"""
Chat routes — POST /chat (standard) and POST /chat/stream (SSE).
No AI logic here — only HTTP concerns: parsing, response formatting, error handling.
"""

# Imports Python's built-in logging module to record application events.
import logging
import structlog
from pydantic import ValidationError

# Imports APIRouter to structure routes modularly and
# HTTPException to return error responses to the client.
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse

# Imports the exception type raised when request data doesn't match the expected Pydantic model.

from src.api.schemas.chat import ChatRequest, ChatResponse
# Imports the Pydantic models that define the structure of the incoming request and outgoing response.
from src.agent.runner import run_agent, stream_agent
from src.api.dependencies import get_agent

# Imports the business logic function that handles the AI processing (RAG/SQL)
from src.tools.search.vector_search import perform_vector_search

# Initializes a logger for this module, using the module name to trace where logs originate.
# logger = logging.getLogger(__name__)

# Creates a router instance. All routes defined here will start with /api/ai and
# be grouped under "ai-chat" in the auto-generated documentation.
router = APIRouter(prefix="/api/ai/chat", tags=["ai-chat"])
logger = structlog.get_logger(__name__)

# TODO: validate inputs on .net backend
# Defines the function that handles the request.
# FastAPI automatically validates the incoming request body against the ChatRequest model.


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, agent=Depends(get_agent)):
    """
    Standard chat endpoint — returns the full response at once.
    Suitable for simple integrations.
    """
    # attaches session_id to every log line from here, so you can filter logs by session in production.
    log = logger.bind(session_id=request.session_id)
    #  logs only the first 80 characters of the message. Avoids logging sensitive full messages.
    log.info("Chat request", message_preview=request.message[:80])

    try:
        # agent/runner.py contains the logic to invoke the agent and get a response.
        reply = await run_agent(agent, request.message, request.session_id)
        result = ChatResponse(reply=reply, session_id=request.session_id)
        return result

    except Exception as e:
        log.error("Chat failed", error=str(e))

        raise HTTPException(
            status_code=500, detail="AI service error. Please try again.") from e


@router.post("/stream")
async def chat_stream(request: ChatRequest, agent=Depends(get_agent)):
    """
    Streaming chat endpoint — returns tokens via Server-Sent Events (SSE).
    Connect from React using EventSource or the fetch API with streaming.
    """
    log = logger.bind(session_id=request.session_id)
    log.info("Stream request", message_preview=request.message[:80])

    # Instead of waiting for the full response, this sends tokens back word by word as the LLM generates them — giving the typing effect in the UI.
    # `async for token in stream_agent(...)` — iterates over tokens as they arrive from the LLM, one by one.
    # `yield f"data: {token}\n\n"` — the `\n\n` is required by the SSE protocol. React reads these chunks and appends them to the message in real time.
    # `yield "data: [DONE]\n\n"` — signals to the React frontend that streaming is finished.
    async def event_generator():
        async for token in stream_agent(agent, request.message, request.session_id):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    result = StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )

    return result


@router.post("/ask", response_model=ChatResponse)
async def ask_question(request: ChatRequest):
    # Docstring explaining what this endpoint does (receives a question, processes it, returns an answer).
    """
    Receives a question from .NET, processes it via RAG/SQL fallback,
    and returns the answer.
    """
    try:
        result = await perform_vector_search(request.question)
        return result

    except ValidationError as e:
        logger.warning("Validation error: %s", e)
        # Returns a 422 Unprocessable Entity error to the client with the validation details.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception(
            "Unexpected error processing question: %s", request.question)
        raise HTTPException(status_code=500, detail="Internal server error")


# Defines a POST endpoint at POST /api/ai/ask.
# It tells FastAPI to validate the outgoing response against the ChatResponse model.
