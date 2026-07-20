"""
Unit tests for PgVectorStoreService.create_vector_store.
"""
from unittest.mock import MagicMock, patch

import pytest

from app.infrastructure.pgvector_store import PgVectorStoreService

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _postgres_url(monkeypatch):
    monkeypatch.setattr(
        "app.infrastructure.pgvector_store.settings.POSTGRES_URL",
        "postgresql://user:pass@db.internal:5432/recbrain",
    )


class TestCreateVectorStore:
    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    def test_returns_store_on_success(self, mock_from_params):
        mock_store = MagicMock()
        mock_from_params.return_value = mock_store

        result = PgVectorStoreService().create_vector_store()

        assert result is mock_store

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    def test_async_connection_string_uses_asyncpg_driver(self, mock_from_params):
        PgVectorStoreService().create_vector_store()

        call_kwargs = mock_from_params.call_args.kwargs
        assert call_kwargs["async_connection_string"].startswith("postgresql+asyncpg://")

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    def test_sync_connection_string_passed_through_unchanged(self, mock_from_params):
        PgVectorStoreService().create_vector_store()

        call_kwargs = mock_from_params.call_args.kwargs
        assert call_kwargs["connection_string"] == "postgresql://user:pass@db.internal:5432/recbrain"

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    def test_sslmode_translated_to_ssl_param_for_asyncpg(self, mock_from_params, monkeypatch):
        monkeypatch.setattr(
            "app.infrastructure.pgvector_store.settings.POSTGRES_URL",
            "postgresql://user:pass@db.internal:5432/recbrain?sslmode=require",
        )

        PgVectorStoreService().create_vector_store()

        call_kwargs = mock_from_params.call_args.kwargs
        assert "ssl=require" in call_kwargs["async_connection_string"]
        assert "sslmode=require" not in call_kwargs["async_connection_string"]

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    @patch("app.infrastructure.pgvector_store.logger.info")
    def test_logs_host_without_credentials_on_success(self, log_info, mock_from_params):
        PgVectorStoreService().create_vector_store()

        log_info.assert_called_once_with("pgvector_connected", host="db.internal:5432")

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    def test_raises_on_failure(self, mock_from_params):
        mock_from_params.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError, match="connection refused"):
            PgVectorStoreService().create_vector_store()

    @patch("app.infrastructure.pgvector_store.PGVectorStore.from_params")
    @patch("app.infrastructure.pgvector_store.logger.exception")
    def test_logs_host_without_credentials_on_failure(self, log_exception, mock_from_params):
        mock_from_params.side_effect = RuntimeError("connection refused")

        with pytest.raises(RuntimeError):
            PgVectorStoreService().create_vector_store()

        log_exception.assert_called_once_with(
            "pgvector_connection_failed", host="db.internal:5432", error="connection refused"
        )
