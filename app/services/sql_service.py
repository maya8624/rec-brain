"""
app/services/sql_search.py

SqlViewService — LLM-generated SQL queries scoped to v_listings only.

Flow:
    1. Receive natural language question
    2. Send question + v_listings schema to LLM
    3. LLM generates a safe SELECT query
    4. Validate query (SELECT only, v_listings only)
    5. Run query against DB and return results
"""
import logging
import re

from sqlalchemy import text
from langchain_core.messages import HumanMessage, SystemMessage

from app.infrastructure.database import engine
from app.prompts.sql import SQL_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class SqlValidationError(Exception):
    """Raised when generated SQL fails safety validation."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SqlViewService:
    """
    Executes LLM-generated SQL queries against v_listings.
    LLM generates the query — we validate and run it.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    async def search_listings(self, question: str) -> dict:
        """Generate and execute a SQL query from a natural language question."""
        logger.info("SqlViewService.search_listings | question=%.80s", question)

        try:
            sql = await self._generate_sql(question)
            logger.debug("SqlViewService | generated sql=%s", sql)

            self._validate_sql(sql)

            with engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = [dict(row._mapping) for row in result]

            result_count = len(rows)

            logger.info("SqlViewService | complete | count=%d", result_count)

            return {
                "success": True,
                "output": rows,
                "result_count": result_count,
                "sql_used": sql,
            }

        except SqlValidationError as exc:
            logger.error("SqlViewService | validation failed | %s", exc)
            return {
                "success": False,
                "output": None,
                "result_count": 0,
                "error": "Property search is temporarily unavailable.",
            }

        except Exception as exc:
            logger.exception("SqlViewService | failed | %s", exc)
            return {
                "success": False,
                "output": None,
                "result_count": 0,
                "error": "Property search is temporarily unavailable.",
            }

    async def _generate_sql(self, question: str) -> str:
        """Send question + v_listings schema to LLM. Returns raw SQL string."""
        messages = [
            SystemMessage(content=SQL_GENERATION_PROMPT),
            HumanMessage(content=question),
        ]

        response = await self._llm.ainvoke(messages)
        sql = response.content.strip()

        # Strip markdown fences if present
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]

        return sql.strip()

    @staticmethod
    def _validate_sql(sql: str) -> None:
        """
        Safety check — SELECT against v_listings only, no mutation keywords.
        Raises SqlValidationError if validation fails.
        """
        sql_upper = sql.upper().strip()

        if not sql_upper.startswith("SELECT"):
            raise SqlValidationError(f"Query must be SELECT, got: {sql[:50]}")

        if "V_LISTINGS" not in sql_upper:
            raise SqlValidationError("Query must target v_listings only")

        forbidden = {"INSERT", "UPDATE", "DELETE",
                     "DROP", "CREATE", "ALTER", "TRUNCATE"}
        words = set(re.findall(r'\b\w+\b', sql_upper))
        violations = forbidden & words
        if violations:
            raise SqlValidationError(f"Forbidden keywords: {violations}")
