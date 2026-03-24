"""
SqlViewService — LLM-generated SQL queries scoped to v_listings only.

Flow:
    1. Receive natural language question from listing_search_node
    2. Send question + v_listings schema to LLM
    3. LLM generates a safe SELECT query
    4. Validate query (SELECT only, v_listings only)
    5. Run query against DB
    6. Return results

TODO: Delete SqlAgentService (commented out below) once confirmed stable.
"""
import logging
import re

from langchain_core.messages import SystemMessage, HumanMessage

from app.prompts.sql import SQL_GENERATION_PROMPT

logger = logging.getLogger(__name__)


class SqlAgentError(Exception):
    """Raised when the SQL agent fails unrecoverably."""


class SqlViewService:
    """
    Executes LLM-generated SQL queries against v_listings.
    LLM generates the query — we validate and run it.
    """

    def __init__(self, db, llm) -> None:
        self._db = db
        self._llm = llm

    async def search_listings(self, question: str) -> dict:
        """
        Generate and execute a SQL query from a natural language question.
        """
        logger.info(
            "SqlViewService | search_listings | question=%.80s", question)

        try:
            sql = await self._generate_sql(question)
            logger.debug("SqlViewService | generated sql=%s", sql)

            self._validate_sql(sql)

            rows = self._db.run(sql)
            result_count = len(rows) if isinstance(rows, list) else 0

            logger.info("SqlViewService | complete | count=%d", result_count)

            return {
                "success": True,
                "output": rows,
                "result_count": result_count,
                "sql_used": sql,
            }

        except SqlValidationError as e:
            logger.error("SqlViewService | validation failed | %s", e)
            return {
                "success": False,
                "output": None,
                "result_count": 0,
                "error": "Property search is temporarily unavailable.",
            }

        except Exception as e:
            logger.exception("SqlViewService | failed | %s", e)
            return {
                "success": False,
                "output": None,
                "result_count": 0,
                "error": "Property search is temporarily unavailable.",
            }

    async def _generate_sql(self, question: str) -> str:
        """
        Send question + v_listings schema to LLM.
        Returns raw SQL query string.
        """
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
        Safety check — ensure query is a safe SELECT against v_listings only.
        Raises SqlValidationError if validation fails.
        """
        sql_upper = sql.upper().strip()

        # Must be a SELECT statement
        if not sql_upper.startswith("SELECT"):
            raise SqlValidationError(f"Query must be SELECT, got: {sql[:50]}")

        # Must query v_listings only
        if "V_LISTINGS" not in sql_upper:
            raise SqlValidationError("Query must target v_listings only")

        # Must not contain mutation keywords
        forbidden = {"INSERT", "UPDATE", "DELETE",
                     "DROP", "CREATE", "ALTER", "TRUNCATE"}
        words = set(re.findall(r'\b\w+\b', sql_upper))
        violations = forbidden & words
        if violations:
            raise SqlValidationError(f"Forbidden keywords: {violations}")


class SqlValidationError(Exception):
    """Raised when generated SQL fails safety validation."""


# ---------------------------------------------------------------------------
# TODO: Delete SqlAgentService once confirmed stable without it.
# No longer used in the current graph flow.
# ---------------------------------------------------------------------------

# from langchain.schema import SystemMessage as SM
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
# from app.core.config import settings
# from app.prompts.sql import SQL_AGENT_SYSTEM_MESSAGE
#
# class SqlAgentError(Exception):
#     """Raised when the SQL agent fails unrecoverably."""
#
# class SqlAgentService:
#     """Wraps LangChain SQL agent for natural language property search."""
#
#     def __init__(self, llm, db) -> None:
#         self._llm = llm
#         self._db = db
#         self._agent = None
#
#     async def _build_agent(self):
#         logger.info("SqlAgentService | building SQL agent")
#         toolkit = SQLDatabaseToolkit(db=self._db, llm=self._llm)
#         schema_info = self._db.get_table_info()
#         system_message = SM(content=f"""
#             {SQL_AGENT_SYSTEM_MESSAGE.content}
#             DATABASE SCHEMA: {schema_info}
#         """)
#         return create_sql_agent(
#             llm=self._llm,
#             toolkit=toolkit,
#             verbose=settings.is_development,
#             agent_type="tool-calling",
#             system_message=system_message,
#             max_iterations=5,
#             agent_executor_kwargs={
#                 "return_intermediate_steps": True,
#                 "handle_parsing_errors": True,
#             },
#         )
#
#     async def search(self, question: str) -> dict:
#         logger.info("SqlAgentService | search | question=%s", question)
#         try:
#             if self._agent is None:
#                 self._agent = await self._build_agent()
#             raw = await self._agent.ainvoke({"input": question})
#             sql_used = self._extract_sql(raw.get("intermediate_steps", []))
#             result_count = self._extract_count(
#                 raw.get("intermediate_steps", []),
#                 raw.get("output", "")
#             )
#             return {
#                 "success": True,
#                 "output": raw.get("output", ""),
#                 "sql_used": sql_used,
#                 "result_count": result_count,
#             }
#         except SqlAgentError:
#             raise
#         except Exception as e:
#             logger.exception("SqlAgentService | failed | %s", e)
#             raise SqlAgentError(f"Property search failed: {e}") from e
#
#     async def _reset_agent(self) -> None:
#         self._agent = None
#
#     @staticmethod
#     def _extract_sql(intermediate_steps: list) -> str:
#         sql = ""
#         for action, _ in intermediate_steps:
#             if hasattr(action, "tool") and action.tool == "sql_db_query":
#                 sql = getattr(action, "tool_input", "")
#         return sql
#
#     @staticmethod
#     def _extract_count(intermediate_steps: list, output: str) -> int:
#         for action, observation in reversed(intermediate_steps):
#             if hasattr(action, "tool") and action.tool == "sql_db_query":
#                 if isinstance(observation, list):
#                     return len(observation)
#                 if isinstance(observation, str) and observation.startswith("["):
#                     try:
#                         import ast
#                         rows = ast.literal_eval(observation)
#                         if isinstance(rows, list):
#                             return len(rows)
#                     except Exception:
#                         pass
#         return 0
