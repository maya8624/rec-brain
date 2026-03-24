import logging
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from app.core.config import settings
from app.core.constants import TableNames

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
# TODO: change to views for better security and to simplify schema for LLM
ALLOWED_TABLES = [
    TableNames.AGENCIES,
    TableNames.AGENTS,
    TableNames.INSPECTION_BOOKINGS,
    TableNames.LISTINGS,
    TableNames.PROPERTIES,
    TableNames.PROPERTY_ADDRESSES,
    TableNames.PROPERTY_TYPES,
    TableNames.V_LISTINGS,
]


def get_db() -> SQLDatabase:
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
            view_support=True,
            # e.g. that status is 'active' not 'Active' or 1). Reduces token usage in schema description.
            sample_rows_in_table_info=3
        )
    except Exception as e:
        logger.exception("Failed to initialize SQLDatabase wrapper: %s", e)
        raise
