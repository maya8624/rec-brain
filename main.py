
"""
FastAPI application entrypoint.
Responsibilities: app lifecycle only — startup, shutdown, router registration.
No business logic here.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import logging
import structlog

from fastapi import FastAPI, Header, HTTPException
# Imports middleware to handle CORS
from fastapi.middleware.cors import CORSMiddleware

from src.agent.builder import build_agent
from src.api.routes import chat, health
from src.core.config import settings
from src.core.logging import setup_logging
from src.services.backend_client import backend_client


logger = structlog.get_logger(__name__)

# Defines the logging level (INFO and above) and sets a specific format that
# includes the timestamp, level name, logger name, and the actual message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# Create a logger instance for this specific file
logger = logging.getLogger(__name__)


# Dependency to check for API Key
async def verify_internal_key(x_api_key: str = Header(...)):
    if x_api_key != settings.BACKEND_API_KEY:
        raise HTTPException(
            status_code=403, detail="Forbidden")
    return x_api_key

# App Lifecycle (lifespan)
# A decorator that tells FasAPI this function manages the app's life cycle
# TODO: cheeck if I need "AsyncGenerator"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Initialize shared resources on startup, clean up on shutdown.
    """
    # configures structlog
    setup_logging()  # 1. logging ready

    # ogs a message right when the server starts.
    logger.info("Starting AI service",
                version=settings.APP_VERSION,
                env=settings.ENVIRONMENT if hasattr(
                    settings, 'ENVIRONMENT') else 'development',
                mock_mode=settings.MOCK_MODE,)

    # Initialize HTTP client to .NET backend
    await backend_client.initialize()  # 3. HTTP client to .NET ready

    # Build agent once — reused across all requests
    # FastAPI's app.state is a built-in place to store shared objects. Storing the agent here means every request can access it without rebuilding it
    app.state.backend_client = backend_client
    app.state.ai_agent = build_agent()  # 4. AI agent built and stored

    logger.info("AI service ready")
    # Crucial: Everything before this runs on startup.
    # Everything after this runs on shutdown. The server runs while paused at this line.

    # The `yield` is the moment the server goes live. Everything before it is setup, everything after is cleanup.
    yield

    await backend_client.close()
    logger.info("Shutting down Real Estate AI Service")

# FastAPI App Instance: this creates the FastAPI application but doesn't start it yet.
# Metadata for the automatic API documentation (Swagger UI).
app = FastAPI(
    title="Real Estate AI Service",
    description="Microservice for semantic search and SQL fallback",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if not settings.PRODUCTION else None,
    redoc_url="/redoc" if not settings.PRODUCTION else None,
)

# Configure CORS — restrict allow_origins to your .NET app URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(","),
    allow_credentials=True,  # Allows browsers to send cookies/auth headers
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],    # Allows all headers.
)

# Register routers: tells FastAPI to use the routes imported from
app.include_router(chat.router)
app.include_router(health.router)
