# tests/conftest.py
from unittest.mock import AsyncMock, MagicMock, patch
import os
import sys

# Adds the project root to the Python path so imports work correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# MODULE-LEVEL PATCHES
# ALL patches must start BEFORE any src.* imports, because
# the import chain src.main → routes → chat_service → database
# calls create_engine() and SQLDatabase() at module level.
# ============================================================

# 1. Patch settings so Settings() doesn't fail reading a missing .env
_settings_patcher = patch(
    "src.config.settings",
    DATABASE_URL="postgresql://fake:fake@localhost/fake",
    OLLAMA_BASE_URL="http://localhost:11434",
    CHROMA_PATH="./db/chroma_db",
    MODEL_NAME="llama3.2",
    SIMILARITY_THRESHOLD=0.4,
    PRODUCTION=False,
    SECRET_API_KEY="fake-secret",
    ALLOWED_ORIGINS="*",
)
_settings_patcher.start()

# 2. Patch create_engine so no real DB connection is attempted
_engine_patcher = patch("sqlalchemy.create_engine", return_value=MagicMock())
_engine_patcher.start()

# 3. Patch SQLDatabase at the source BEFORE any import touches database.py
_sql_db_patcher = patch(
    "langchain_community.utilities.SQLDatabase", return_value=MagicMock()
)
_sql_db_patcher.start()

# 4. Patch get_llm so no real LLM connection is attempted on import
_llm_patcher = patch("src.infrastructure.llm.get_llm",
                     return_value=MagicMock())
_llm_patcher.start()

# ============================================================
# Only NOW is it safe to import anything from src.*
# The full chain: main → routes → chat_service → database
# will all use mocked versions of engine, SQLDatabase, and LLM.
# ============================================================
import pytest                                   # noqa: E402
from fastapi.testclient import TestClient       # noqa: E402
from main import app                        # noqa: E402


@pytest.fixture(autouse=True)
def patch_db_connection():
    """Patches create_engine to prevent real DB connections during tests."""
    with patch("src.infrastructure.database.create_engine") as mock_engine:
        mock_engine.return_value = MagicMock()
        yield mock_engine


@pytest.fixture
def client():
    """Provides a FastAPI test client for route-level tests."""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Provides a mocked SQLAlchemy database object."""
    db = MagicMock()
    db.get_table_info.return_value = "Table: users, Columns: id, name"
    db.run.return_value = "[('result',)]"
    return db


@pytest.fixture
def mock_llm():
    """Provides a mocked LLM object for chat_service tests."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value="SELECT * FROM listings")
    return llm


@pytest.fixture
def mock_vector_db():
    """Provides a mocked async vector search — returns no results by default (SQL fallback)."""
    mock = AsyncMock()
    mock.return_value = []  # empty → triggers SQL fallback
    return mock
