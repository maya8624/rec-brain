
"""
FastAPI application entrypoint.
Responsibilities: app lifecycle only — startup, shutdown, router registration.
No business logic here.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import structlog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.builder import build_agent
from src.api.routes import chat, health
from src.core.config import settings
from src.core.logging import setup_logging
# from src.services.backend_client import backend_client
from src.services.mock import backend_client

logger = structlog.get_logger(__name__)

# Resolved once at module load — avoids repeated str() casting
ENVIRONMENT = str(settings.ENVIRONMENT)
APP_VERSION = str(settings.APP_VERSION)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in str(settings.ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

# Defines the logging level (INFO and above) and sets a specific format that
# includes the timestamp, level name, logger name, and the actual message
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
# )

# Create a logger instance for this specific file
# logger = logging.getLogger(__name__)


# ------------------------------------
# App Lifecycle
# A decorator that tells FasAPI this function manages the app's life cycle
# ------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages startup and shutdown of shared resources.
    """
    # 1. Configure structlog first — must be before any logger.info() calls
    setup_logging()

    # Logs a message right when the server starts.
    logger.info("Starting AI service",
                version=APP_VERSION,
                env=ENVIRONMENT,
                mock_mode=str(settings.MOCK_MODE),)

    try:
        # 2. Initialize HTTP client to .NET backend (or mock)
        await backend_client.initialize()

        # Build agent once — reused across all requests
        # FastAPI's app.state is a built-in place to store shared objects. Storing the agent here means every request can access it without rebuilding it
        _app.state.backend_client = backend_client
        _app.state.ai_agent = build_agent()  # 4. AI agent built and stored

        logger.info("AI service ready")
        # Crucial: Everything before this runs on startup.
        # Everything after this runs on shutdown. The server runs while paused at this line.
        # The `yield` is the moment the server goes live. Everything before it is setup, everything after is cleanup.
        yield

    except Exception as ex:
        logger.exception("Failed to start AI service", error=str(ex))
        raise

    finally:
        await backend_client.close()
        logger.info("Shutting down Real Estate AI Service")

# ------------------------------------
# FastAPI App Instance: this creates the FastAPI application but doesn't start it yet.
# Metadata for the automatic API documentation (Swagger UI).
# ------------------------------------
app = FastAPI(
    title="Real Estate AI Service",
    description="AI orchestration service for real estate search and inspection workflows.",
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if ENVIRONMENT != "production" else None,
    redoc_url="/redoc" if ENVIRONMENT != "production" else None,
)

# ------------------------------------
# Middleware
# ------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,  # Allows browsers to send cookies/auth headers
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# Register routers
# ------------------------------------
app.include_router(chat.router)
app.include_router(health.router)
