"""
Redis Client
 
Handles all memory persistence for the agent:
  - Session memory  (short-term, per conversation)
  - User preferences (long-term, across sessions)
  - Viewed properties (deduplication across sessions)
 
Design:
  - Uses redis.asyncio — non-blocking, compatible with FastAPI
  - Single connection pool shared across all requests
  - All failures are caught and logged — Redis down never crashes the agent
  - All data serialized as JSON strings
"""

import logging
from typing import Optional
from redis.asyncio import ConnectionPool, Redis

# TODO: structlog for consistency with the rest of the app
logger = logging.getLogger(__name__)


# ── TTL constants ─────────────────────────────────────────────────────────────

SESSION_TTL_SECONDS: int = 60 * 60 * 2              # 2 hours
PREFERENCES_TTL_SECONDS: int = 60 * 60 * 24 * 90    # 90 days
VIEWED_TTL_SECONDS: int = 60 * 60 * 24 * 30         # 30 days
# 20 turns (user + assistant)
MAX_HISTORY_MESSAGES: int = 40
MAX_VIEWED_PROPERTIES: int = 200

# ── Key schema ────────────────────────────────────────────────────────────────


class RedisKeys:
    """
    Central key schema — all Redis keys defined in one place.
    Changing a key pattern here propagates everywhere.
    """

    @staticmethod
    def session(session_id: str) -> str:
        return f"session:{session_id}:history"

    @staticmethod
    def preferences(user_id: str) -> str:
        return f"user:{user_id}:preferences"

    @staticmethod
    def viewed(user_id: str) -> str:
        return f"user:{user_id}:viewed_properties"


# ── Connection pool ───────────────────────────────────────────────────────────


_pool: Optional[ConnectionPool] = None


def init_redis(host: str = "localhost", port: int = 6379, db: int = 0) -> None:
    """
    Initialize the connection pool.
    Call once at app startup in main.py lifespan.
    """
    global _pool
    _pool = ConnectionPool(
        host=host,
        port=port,
        db=db,
        max_connections=20,
        decode_responses=True,      # always return str, not bytes
    )
    logger.info("✅ Redis pool initialized → {%s}:{%s}/{%s}", host, port, db)


def _get_client() -> Redis:
    if _pool is None:
        raise RuntimeError(
            "Redis pool not initialized. Call init_redis() at startup.")
    return Redis(connection_pool=_pool)
