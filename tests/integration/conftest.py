"""
Integration test fixtures.

All integration tests are marked with @pytest.mark.integration and are
skipped automatically when required environment variables are absent.
This keeps `pytest -m unit` fast in CI without a live environment.
"""
import os

import pytest
from httpx import ASGITransport, AsyncClient


def _missing_env_vars() -> list[str]:
    required = ["GROQ_API_KEY", "POSTGRES_URL"]
    return [v for v in required if not os.getenv(v)]


# Skip entire module if env is not configured
skip_if_no_env = pytest.mark.skipif(
    bool(_missing_env_vars()),
    reason=f"Integration env not configured. Missing: {_missing_env_vars() or 'none'}",
)


@pytest.fixture(scope="session")
def app():
    """FastAPI app instance — initialised once for the whole integration session."""
    from main import app as _app
    return _app


@pytest.fixture
async def client(app):
    """Async HTTP test client wired directly to the FastAPI app (no TCP)."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://127.0.0.1:8000:",
    ) as c:
        yield c
