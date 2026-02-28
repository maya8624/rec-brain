# Imports Python's built-in logging module to record application events.
import logging

# Imports APIRouter to structure routes modularly and
# HTTPException to return error responses to the client.
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

# Imports the exception type raised when request data doesn't match the expected Pydantic model.
from pydantic import ValidationError

# Imports the Pydantic models that define the structure of the incoming request and outgoing response.
from src.domain.models import ChatRequest, ChatResponse

# Imports the business logic function that handles the AI processing (RAG/SQL)
from src.application.chat_service import handle_user_query

# Initializes a logger for this module, using the module name to trace where logs originate.
logger = logging.getLogger(__name__)

# Creates a router instance. All routes defined here will start with /api/ai and
# be grouped under "AI Chat" in the auto-generated documentation.
router = APIRouter(prefix="/api/ai", tags=["AI Chat"])

# Dependency to check for API Key


# async def verify_api_key(x_api_key: str = Header(...)):
#     if x_api_key != settings.SECRET_API_KEY:
#         raise HTTPException(
#             status_code=403, detail="Could not validate credentials")
#     return x_api_key

# Defines a POST endpoint at POST /api/ai/ask.
# It tells FastAPI to validate the outgoing response against the ChatResponse model.


@router.post("/ask", response_model=ChatResponse)
# TODO: validate inputs on .net backend
# Defines the function that handles the request.
# FastAPI automatically validates the incoming request body against the ChatRequest model.
async def ask_question(request: ChatRequest):
    # Docstring explaining what this endpoint does (receives a question, processes it, returns an answer).
    """
    Receives a question from .NET, processes it via RAG/SQL fallback,
    and returns the answer.
    """
    try:
        result = await handle_user_query(request.question)
        return result

    except ValidationError as e:
        logger.warning("Validation error: %s", e)
        # Returns a 422 Unprocessable Entity error to the client with the validation details.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception(
            "Unexpected error processing question: %s", request.question)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def root():
    return {
        "message": "Real Estate AI Service is Online",
        "docs": "/docs",
        "status": "active"
    }


@router.get("/health")
async def health_check():
    """
        Returns a simple JSON response to prove the server is running.
    """
    return {"status": "healthy"}
