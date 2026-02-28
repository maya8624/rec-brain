import logging
import os
from datetime import datetime

# Setup a specific logger for SQL tracking
sql_logger = logging.getLogger("sql_debug")
sql_logger.setLevel(logging.INFO)

# Ensure a logs directory exists
os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(
    f"logs/sql_queries_{datetime.now().strftime('%Y%m%d')}.log")

file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))

sql_logger.addHandler(file_handler)


def track_sql_steps(response: dict):
    """
    Parses agent response to find and log executed SQL.
    """
    steps = response.get("intermediate_steps", [])

    for action, observation in steps:
        # Check if the tool used was the SQL query tool
        if hasattr(action, "tool") and action.tool == "sql_db_query":
            query = action.tool_input
            sql_logger.info("QUERY: %s", query)
            # Optionally log the result/observation if it's short
            sql_logger.info("RESULT: %s...", str(observation)[:200])
