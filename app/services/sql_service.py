"""
Translates natural language queries into SQL and returns structured results.
"""
import logging

from langchain.schema import SystemMessage
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

from app.core.config import settings
from app.prompts.sql import SQL_AGENT_SYSTEM_MESSAGE

logger = logging.getLogger(__name__)


class SqlAgentError(Exception):
    """Raised when the SQL agent fails unrecoverably."""


class SqlAgentService:
    """ Wraps LangChain SQL agent for natural language property search."""

    def __init__(self, llm, db) -> None:
        self._llm = llm
        self._db = db
        self._agent = None

    async def _build_agent(self):
        logger.info("SqlAgentService | building SQL agent")

        toolkit = SQLDatabaseToolkit(db=self._db, llm=self._llm)
        schema_info = self._db.get_table_info()

        system_message = SystemMessage(content=f"""
            {SQL_AGENT_SYSTEM_MESSAGE.content}
            DATABASE SCHEMA: {schema_info}
        """)

        return create_sql_agent(
            llm=self._llm,
            toolkit=toolkit,
            verbose=settings.is_development,
            agent_type="tool-calling",
            system_message=system_message,
            max_iterations=5,
            agent_executor_kwargs={
                "return_intermediate_steps": True,
                "handle_parsing_errors": True,
            },
        )

    async def search(self, question: str) -> dict:
        """Execute a natural language property search."""
        logger.info("SqlAgentService | search | question=%s", question)

        try:
            if self._agent is None:
                self._agent = await self._build_agent()

            raw = await self._agent.ainvoke({"input": question})
            sql_used = self._extract_sql(raw.get("intermediate_steps", []))
            # result_count = self._extract_count(raw.get("output", 0))
            result_count = self._extract_count(
                raw.get("intermediate_steps", []),
                raw.get("output", "")
            )

            logger.info(
                "SqlAgentService | complete | sql=%s | count=%s",
                sql_used,
                result_count,
            )

            return {
                "success": True,
                "output": raw.get("output", ""),
                "sql_used": sql_used,
                "result_count": result_count,
            }

        except SqlAgentError:
            raise
        except Exception as e:
            logger.exception("SqlAgentService | failed | %s", e)
            raise SqlAgentError(f"Property search failed: {e}") from e

    async def _reset_agent(self) -> None:
        """Invalidate the cached agent so the next call rebuilds it."""
        self._agent = None

    @staticmethod
    def _extract_sql(intermediate_steps: list) -> str:
        """
        Pull the last generated SQL from LangChain intermediate steps.
        Last sql_db_query call is the most meaningful one to surface.
        """
        sql = ""

        for action, _ in intermediate_steps:
            if hasattr(action, "tool") and action.tool == "sql_db_query":
                sql = getattr(action, "tool_input", "")

        return sql

    @staticmethod
    def _extract_count(intermediate_steps: list, output: str) -> int:
        """Count rows from the last sql_db_query result."""
        for action, observation in reversed(intermediate_steps):
            if hasattr(action, "tool") and action.tool == "sql_db_query":
                if isinstance(observation, str) and observation.startswith("["):
                    try:
                        import ast
                        rows = ast.literal_eval(observation)
                        if isinstance(rows, list):
                            return len(rows)
                    except Exception:
                        pass
        return 0
