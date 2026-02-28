from functools import lru_cache
import logging
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from src.config import settings

logger = logging.getLogger(__name__)

# Engine with optimized pooling for a production-ready AI Service
engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,  # Checks connection health before use
    pool_size=10,        # Persistent connections
    max_overflow=20,     # Temporary extra connections
    pool_recycle=3600,   # Recycles connections every hour
    pool_timeout=30      # Timeout if pool is exhausted
)

# Define the tables  Real Estate agent is allowed to 'see'
ALLOWED_TABLES = [
    'properties',
    'inspections',
    'market_listings',
    'open_houses',
    'agents',
    'auctions'
]


@lru_cache(maxsize=1)
def get_db_wrapper() -> SQLDatabase:
    """
    Returns a pre-configured SQLDatabase wrapper.
    Best Practice: Explicitly whitelist tables to reduce token usage and 
    prevent the model from discovering sensitive schema parts.
    """
    try:
        logger.info("Initializing SQLDatabase wrapper.")

        return SQLDatabase(
            engine,
            include_tables=ALLOWED_TABLES,
            view_support=True,  # Critical for real estate views/reporting
            sample_rows_in_table_info=3  # Helps LLM understand data types
        )
    except Exception as e:
        logger.exception("Failed to initialize SQLDatabase wrapper: %s", e)
        raise
