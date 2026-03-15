"""
app/core/config.py

Centralised configuration using pydantic-settings.
Reads from .env file automatically — never use os.environ.get() elsewhere.

Priority order:
    1. Real environment variables  (production)
    2. .env file                   (local development)
    3. Field(default=...)          (fallback)
"""
from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationError


# ── Guard — fail fast if .env is missing ──────────────────────────────────────

env_path = Path(".env")

if not env_path.exists():
    raise FileNotFoundError(
        f"Configuration file '{env_path}' not found. "
        "Create one from .env.example before starting the service."
    )


# ── Settings ──────────────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    All application settings in one place.

    Required fields  Field(...)         — app refuses to start if missing
    Optional fields  Field(default=...) — safe fallbacks, override in .env

    Usage anywhere:
        from app.core.config import settings
        settings.GROQ_API_KEY
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # silently ignore unknown vars (PATH, HOME, etc)
        extra="ignore",
        case_sensitive=False,   # GROQ_API_KEY == groq_api_key
    )

    # ── App ───────────────────────────────────────────────────────────────────
    APP_VERSION: str = Field("1.0.0")
    ENVIRONMENT: str = Field(
        "development",
        description="development | staging | production",
    )
    MOCK_MODE: bool = Field(
        False,
        description="True = use mock backend client, skip real .NET API calls",
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: str = Field(
        "http://localhost:3000,http://localhost:5000",
        description="Comma-separated CORS origins — main.py splits into list",
    )

    # ── Database (PostgreSQL + pgvector) ──────────────────────────────────────
    DATABASE_URL: str = Field(
        ...,
        description="PostgreSQL connection string — used by SQL agent and app",
    )
    # Read-only connection for SQL agent — never write access
    # If not set, falls back to DATABASE_URL via effective_ai_database_url
    # Production: create a read-only PostgreSQL user and set this
    AI_DATABASE_URL: str = Field(
        "",
        description="Read-only DB URL for SQL agent. Falls back to DATABASE_URL.",
    )

    # ── Vector search ────────────────────────────────────────────────────────
    # CURRENT:  ChromaDB (local file-based)
    # FUTURE:   pgvector (inside your existing PostgreSQL — no extra service)
    # Migration: swap VECTOR_STORE from chroma -> pgvector in .env
    #            rag_service.py checks this value to pick the store
    VECTOR_STORE: str = Field(
        "chroma",
        description="Vector store backend: chroma | pgvector",
    )

    # ChromaDB — current
    CHROMA_PATH: str = Field(
        "./chroma_db",
        description="Local path to ChromaDB storage directory",
    )

    # pgvector — ready for when you migrate
    # No extra config needed — uses DATABASE_URL + pgvector extension
    # Enable with: VECTOR_STORE=pgvector in .env
    PGVECTOR_TABLE: str = Field(
        "property_embeddings",
        description="PostgreSQL table name for pgvector embeddings",
    )
    PGVECTOR_EMBED_DIM: int = Field(
        384,
        description="Embedding dimensions — must match your embedding model",
    )

    # Shared by both Chroma and pgvector
    SIMILARITY_THRESHOLD: float = Field(
        0.7,
        description="Minimum similarity score for RAG retrieval (0.0 - 1.0)",
    )
    EMBEDDING_MODEL: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model — used if OPENAI_API_KEY not set",
    )

    # ── .NET Backend ──────────────────────────────────────────────────────────
    BACKEND_BASE_URL: str = Field(
        "http://localhost:5000",
        description="Base URL of the .NET API",
    )
    BACKEND_API_KEY: str = Field(
        "",
        description="API key for authenticating with .NET API",
    )

    # ── Groq (cloud LLM) ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(
        ...,
        description="Groq API key — get from console.groq.com",
    )
    MODEL_NAME: str = Field(
        "llama-3.3-70b-versatile",
        description="Default Groq model",
    )

    # Per-use-case models — swap independently via .env without code changes
    SQL_AGENT_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Model for SQL agent — needs strong reasoning",
    )
    RAG_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Model for RAG synthesis — can use faster model",
    )
    CHAT_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Model for general chat — quality matters for UX",
    )
    AGENT_TOOL_CALL_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Model for tool calling agent",
    )

    LLM_TEMPERATURE: float = Field(
        0.0,
        description="0.0 = best for tool calling, 1.0 = creative",
    )
    LLM_MAX_TOKENS: int = Field(
        2048,
        description="Max tokens in LLM response — 2048 needed for RAG synthesis",
    )
    AGENT_TIMEOUT_SECONDS: int = Field(
        30,
        description="Max seconds before agent call times out",
    )

    # ── Ollama (local LLM — optional) ─────────────────────────────────────────
    OLLAMA_BASE_URL: str = Field(
        "http://localhost:11434",
        description="Ollama server URL — used when running LLM locally",
    )

    # ── OpenAI (optional — for embeddings) ────────────────────────────────────
    # If set: uses text-embedding-3-small for higher quality embeddings
    # If not set: falls back to local HuggingFace EMBEDDING_MODEL
    OPENAI_API_KEY: str = Field(
        "",
        description="OpenAI API key — optional, only for embedding quality upgrade",
    )

    # ── LangSmith observability (optional) ────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = Field(False)
    LANGCHAIN_API_KEY: str = Field("")
    LANGCHAIN_PROJECT: str = Field("real-estate-ai")

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = Field(
        "INFO",
        description="DEBUG | INFO | WARNING | ERROR",
    )

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def effective_ai_database_url(self) -> str:
        """
        Read-only DB URL for the SQL agent.
        Falls back to main DATABASE_URL if AI_DATABASE_URL not set.

        Production: set AI_DATABASE_URL to a read-only PostgreSQL user.
        Development: leave unset — uses main DATABASE_URL.
        """
        return self.AI_DATABASE_URL or self.DATABASE_URL

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def is_staging(self) -> bool:
        return self.ENVIRONMENT == "staging"

    @property
    def allowed_origins_list(self) -> list[str]:
        """ALLOWED_ORIGINS parsed as a Python list."""
        return [
            origin.strip()
            for origin in self.ALLOWED_ORIGINS.split(",")
            if origin.strip()
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────

try:
    settings = Settings()
    print(
        f"Settings loaded | env={settings.ENVIRONMENT} "
        f"mock={settings.MOCK_MODE} "
        f"model={settings.MODEL_NAME}"
    )
except ValidationError as e:
    print(f"Configuration error:\n{e}")
    raise SystemExit(1)   # clean exit — no ugly traceback
