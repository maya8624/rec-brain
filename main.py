# Imports Python's built-in logging module to track application events.
import logging

# Imports a tool used to manage the startup and shutdown lifecycle of the application
from contextlib import asynccontextmanager

from typing import AsyncGenerator

# Imports the main class to create your web application
from fastapi import FastAPI

# Imports middleware to handle CORS
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as ai_router
from src.config import settings

# Defines the logging level (INFO and above) and sets a specific format that
# includes the timestamp, level name, logger name, and the actual message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# Create a logger instance for this specific file
logger = logging.getLogger(__name__)

# App Lifecycle (lifespan)


# A decorator that tells FasAPI this function manages the app's life cycle
@asynccontextmanager
async def lifespan(f_app: FastAPI) -> AsyncGenerator[None, None]:
    """
    This function defines what happens when the server starts up and shuts down.
    """
    # ogs a message right when the server starts.
    logger.info("Starting Real Estate AI Service")

    # Crucial: Everything before this runs on startup.
    # Everything after this runs on shutdown. The server runs while paused at this line.
    yield

    # Logs a message when the server stops.
    logger.info("Shutting down Real Estate AI Service")

# FastAPI App Instance
# This creates the actual web aplication object
# Metadata for the automatic API documentation (Swagger UI).
app = FastAPI(
    title="Real Estate AI Service",
    description="Microservice for semantic search and SQL fallback",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if not settings.PRODUCTION else None,
    redoc_url="/redoc" if not settings.PRODUCTION else None,
)

# Use settings for CORS
origins = settings.ALLOWED_ORIGINS.split(",")

# Configure CORS — restrict allow_origins to your .NET app URL in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,

    # Allows browsers to send cookies/auth headers
    allow_credentials=True,

    # Allows all HTTP methods (GET, POST, etc.).
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],    # Allows all headers.
)

# Routing and Health Check

# Tells FastAPI to use the routes imported from
app.include_router(ai_router)
