import logging
import re
import os
from typing import Optional
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_core.messages import SystemMessage
from app.infrastructure.database import get_db_wrapper
from app.infrastructure.llm import get_llm
from app.prompts.sql_agent import SQL_AGENT_SYSTEM_MESSAGE
from app.core.config import settings

logger = logging.getLogger(__name__)


class SqlAgentService:
    """
    Wraps LangChain SQL agent with real-estate-specific configuration.
    Responsible for translating natural language queries into SQL
    and returning structured results.

    Usage:
        service = SqlAgentService()
        result = service.search("5 properties under $1M in Castle Hill")
    """

    def __init__(self):
        self._agent = None

    def _get_agent(self):
        """Lazy initialisation — agent built on first use."""
        if self._agent is None:
            self._agent = self._build_agent()
        return self._agent

    def _build_agent(self):
        logger.info("Building SQL agent")

        raw_llm = get_llm()
        db = get_db_wrapper()
        is_verbose = settings.is_development

        toolkit = SQLDatabaseToolkit(db=db, llm=raw_llm)
        agent_llm = raw_llm.with_retry(stop_after_attempt=3)

        return create_sql_agent(
            llm=agent_llm,
            toolkit=toolkit,
            verbose=is_verbose,
            agent_type="openai-tools",
            system_message=SQL_AGENT_SYSTEM_MESSAGE,
            max_iterations=6,
            agent_executor_kwargs={
                "return_intermediate_steps": True,
                "handle_parsing_errors": True,
            }
        )

    def search(self, natural_language_query: str) -> dict:
        """
        Execute a natural language property search.

        Args:
            natural_language_query: Plain English query from the user
                eg: "5 houses under $1M in Castle Hill with 4 bedrooms"

        Returns:
            dict with keys:
                output       - formatted string result for the LLM
                sql_used     - the generated SQL (for logging/debugging)
                result_count - number of properties found
                success      - bool

        Raises:
            SqlAgentError on unrecoverable failures
        """
        logger.info("SQL search: %s", natural_language_query)

        try:
            agent = self._get_agent()
            raw = agent.invoke({"input": natural_language_query})

            sql_used = self._extract_sql(raw.get("intermediate_steps", []))
            result_count = self._extract_count(raw.get("output", ""))

            logger.info("SQL search complete — sql: %s | count: %d",
                        sql_used, result_count)

            return {
                "success": True,
                "output": raw.get("output", ""),
                "sql_used": sql_used,
                "result_count": result_count,
            }

        except Exception as e:
            logger.exception("SQL agent search failed: %s", e)
            # Reset agent so next call gets a fresh instance
            self._agent = None
            raise SqlAgentError(f"Property search failed: {e}") from e

    def _extract_sql(self, intermediate_steps: list) -> str:
        """Pull the generated SQL from LangChain intermediate steps."""
        for action, _ in intermediate_steps:
            if hasattr(action, "tool") and action.tool == "sql_db_query":
                return getattr(action, "tool_input", "")
        return ""

    def _extract_count(self, output: str) -> int:
        """
        Best-effort extraction of result count from agent output string.
        Looks for patterns like 'found 5 properties', '3 results', etc.
        """
        patterns = [
            r"found (\d+)",
            r"(\d+) propert",
            r"(\d+) result",
            r"(\d+) listing",
        ]
        for pattern in patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return 0


class SqlAgentError(Exception):
    """Raised when the SQL agent fails unrecoverably."""
    pass
