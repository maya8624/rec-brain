"""
Mock service resolver.
Returns the correct backend client based on MOCK_MODE setting.
Import backend_client from here in all tools — never directly from backend_client.py.
"""

from core.config import settings


def get_backend_client():
    """
    Returns real or mock backend client based on MOCK_MODE env var.

    MOCK_MODE=true  → MockBackendClient (mock_data.py, no HTTP calls)
    MOCK_MODE=false → BackendClient (real httpx calls to .NET API)
    """
    if settings.MOCK_MODE:
        from services.mock.mock_backend_client import mock_backend_client
        return mock_backend_client
    else:
        from services.backend_client import backend_client
        return backend_client


# Single import point for all tools
backend_client = get_backend_client()
