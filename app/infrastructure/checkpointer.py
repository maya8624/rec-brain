"""
PostgreSQL checkpointer service for LangGraph state persistence.
Initialized once at app startup, closed on shutdown.
"""
import logging

from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import InMemorySaver
from app.core.config import settings

logger = logging.getLogger(__name__)


class PostgresCheckpointer:
    """
    Wraps AsyncPostgresSaver with explicit lifecycle management.
    Use create() to initialize, close() on shutdown.
    """

    def __init__(self, checkpointer: AsyncPostgresSaver | InMemorySaver, pool: AsyncConnectionPool | None = None):
        self._checkpointer = checkpointer
        self._pool = pool

    @classmethod
    async def create(cls) -> "PostgresCheckpointer":
        """Initialize the checkpointer and its connection pool."""
        try:
            pool = AsyncConnectionPool(
                conninfo=settings.POSTGRES_URL,
                open=False,
                kwargs={"autocommit": True, "row_factory": dict_row},
            )
            await pool.open()

            checkpointer = AsyncPostgresSaver(conn=pool)
            await checkpointer.setup()

            logger.info("AsyncPostgresSaver ready for use as checkpointer")
            return cls(checkpointer, pool)
        except Exception as exc:
            logger.critical(
                "AsyncPostgresSaver failed: %s — falling back to InMemorySaver (state will NOT persist)", exc)
            return cls(InMemorySaver())

    async def close(self) -> None:
        """Close the connection pool on app shutdown."""
        if self._pool is not None:
            await self._pool.close()
            logger.info("AsyncPostgresSaver connection pool closed")

    @property
    def instance(self) -> AsyncPostgresSaver | InMemorySaver:
        return self._checkpointer
