"""
main.py

FastAPI application entry point.
Responsibilities: app lifecycle only — startup, shutdown, router registration.
No business logic here.
"""
from contextlib import asynccontextmanager
import structlog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.agents.graph import build_graph
from app.infrastructure.checkpointer import PostgresCheckpointer
from app.core.middleware import RequestLoggingMiddleware
from app.api.routes import chat, health
from app.core.config import settings
from app.core.logging import setup_logging
from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.llm import get_llm
from app.infrastructure.pgvector_store import PgVectorStoreService
from app.services.booking_service import BookingService
from app.services.sql_service import SqlViewService
from app.services.rag_service import RagRetriever
from app.services.backend_client import backend_client

logger = structlog.get_logger(__name__)

# Resolved once at module load — avoids repeated str() casting in hot paths
ENVIRONMENT = str(settings.ENVIRONMENT)
APP_VERSION = str(settings.APP_VERSION)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in str(settings.ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

# ------------------------------------
# Lifespan
# ------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """
    Manages startup and shutdown of shared resources.
    Order matters — each step depends on the previous one.

    Everything before yield  = startup (runs before first request)
    Everything after yield   = shutdown (runs after last request)
    finally block            = always runs, even if yield raises
    """
    # Logging first — must be before any logger calls
    setup_logging()

    logger.info(
        "Starting AI service",
        version=APP_VERSION,
        env=ENVIRONMENT,
    )

    try:
        await backend_client.initialize()

        _app.state.backend_client = backend_client
        _app.state.booking_service = BookingService(_app.state.backend_client)
        _app.state.sql_view_service = SqlViewService(llm=get_llm())

        _app.state.rag_retriever = RagRetriever(
            vector_store_service=PgVectorStoreService(),
            embedding_service=EmbeddingService(),
        )

        _app.state.checkpointer = await PostgresCheckpointer.create()
        _app.state.ai_agent = build_graph(_app.state.checkpointer.instance)

        logger.info("AI agent ready")

        yield  # ← server is live while paused here

    except Exception as ex:
        logger.exception("Failed to start AI service", error=str(ex))
        raise

    finally:
        await backend_client.close()

        if hasattr(_app.state, "checkpointer"):
            await _app.state.checkpointer.close()

        logger.info("AI service shutdown complete")


# ------------------------------------
# FastAPI app initialization
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
# Swagger security scheme
# ------------------------------------
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    }
    schema["security"] = [{"ApiKeyAuth": []}]
    app.openapi_schema = schema
    return schema

app.openapi = custom_openapi

# ------------------------------------
# Middleware
# ------------------------------------
# X-Request-ID on every request — enables tracing across .NET → Python
app.add_middleware(RequestLoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ------------------------------------
# Routers
# ------------------------------------
app.include_router(chat.router)
app.include_router(health.router)
