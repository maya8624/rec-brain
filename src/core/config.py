"""
Centralised configuration using pydantic-settings.
Reads from .env file automatically — never use os.environ.get() elsewhere.

Priority order:
  1. Real environment variables  (production)
  2. .env file                   (local development)
  3. Field(default)              (fallback)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, ValidationError


# ------------------------------------
# Guard — fail fast if .env is missing
# ------------------------------------

env_path = Path(".env")

if not env_path.exists():
    raise FileNotFoundError(
        f"Critical error: Configuration file '{env_path}' not found. "
        "Please create one based on .env.example"
    )


# ------------------------------------
# Settings
# ------------------------------------

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    Required fields (Field(...)) — app crashes on startup if missing.
    Optional fields (Field(default)) — have safe fallback values.

    Usage anywhere in the project:
        from core.config import settings
        settings.GROQ_API_KEY
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # ignore unknown vars in .env (e.g. comments, extras)
        extra="ignore",
        case_sensitive=False  # GROQ_API_KEY == groq_api_key
    )

    # ------------------------------------
    # .NET Backend
    # ------------------------------------
    BACKEND_BASE_URL: str = Field(
        "http://localhost:5000",
        description="Base URL of the .NET API"
    )
    BACKEND_API_KEY: str = Field(
        "",
        description="API key for authenticating with the .NET API"
    )

    # ------------------------------------
    # App
    # ------------------------------------
    ALLOWED_ORIGINS: str = Field(
        "http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )
    PRODUCTION: bool = Field(
        False,
        description="Set to True in production — affects logging, CORS, etc."
    )
    APP_VERSION: str = Field("1.0.0")

    # ------------------------------------
    # Database
    # ------------------------------------
    DATABASE_URL: str = Field(
        ...,
        description="PostgreSQL connection string for SQL agent"
    )

    # ------------------------------------
    # Vector DB
    # ------------------------------------
    CHROMA_PATH: str = Field(
        "./chroma_db",
        description="Local path to ChromaDB storage"
    )
    SIMILARITY_THRESHOLD: float = Field(
        0.7,
        description="Minimum similarity score for vector search results (0.0 - 1.0)"
    )
    EMBEDDING_MODEL: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model for vector search"
    )

    # ------------------------------------
    # Ollama (local LLM)
    # ------------------------------------
    OLLAMA_BASE_URL: str = Field(
        "http://localhost:11434",
        description="Ollama server base URL"
    )

    # ------------------------------------
    # Groq (cloud LLM)
    # ------------------------------------
    GROQ_API_KEY: str = Field(
        ...,
        description="Groq API key — get from console.groq.com"
    )
    MODEL_NAME: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model for general chat"
    )
    LLM_TEMPERATURE: float = Field(
        0.0,
        description="LLM temperature — 0.0 = deterministic, 1.0 = creative"
    )
    LLM_MAX_TOKENS: int = Field(
        1024,
        description="Maximum tokens in LLM response"
    )

    # ------------------------------------
    # OpenAI (optional fallback)
    # ------------------------------------
    OPENAI_API_KEY: str = Field(
        "",
        description="OpenAI API key — optional, only needed if using OpenAI models"
    )

    # ------------------------------------
    # Agent / Tool calling
    # ------------------------------------
    AGENT_TOOL_CALL_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model used specifically for tool calling agent"
    )
    AGENT_TIMEOUT_SECONDS: int = Field(
        30,
        description="Max seconds before agent call times out"
    )

    # ------------------------------------
    # Mock mode — local dev without .NET
    # ------------------------------------
    MOCK_MODE: bool = Field(
        False,
        description="Set to True to use mock data instead of real .NET API calls"
    )

    # ------------------------------------
    # Logging
    # ------------------------------------
    LOG_LEVEL: str = Field(
        "INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )


# ------------------------------------
# Singleton — import this everywhere
# ------------------------------------

try:
    settings = Settings()
    print(
        f"✅ Settings loaded — env: {'production' if settings.PRODUCTION else 'development'}, mock: {settings.MOCK_MODE}")
except ValidationError as e:
    print(f"❌ Configuration validation error:\n{e}")
    raise SystemExit(1)  # exit cleanly instead of traceback
