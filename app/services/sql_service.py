"""
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
from app.agents.state import SearchContext
from app.schemas.property import SearchResult

logger = logging.getLogger(__name__)

_SEARCH_ERROR = SearchResult(
    success=False, error="Property search is temporarily unavailable."
)

# Fixed SELECT columns — mirrors SQL_GENERATION_PROMPT rule 1
_SELECT_COLS = (
    "SELECT listing_id, property_id, listing_type, listing_status, price, bedrooms, bathrooms, "
    "car_spaces, property_type, title, address_line1, address_line2, suburb, state, "
    "postcode, agent_first_name, agent_last_name, agent_phone, agency_name"
)


class SqlValidationError(Exception):
    """Raised when generated SQL fails safety validation."""


class SqlViewService:
    """
    Executes LLM-generated SQL queries against v_listings.
    LLM generates the query — we validate and run it.
    """

    def __init__(self, llm) -> None:
        self._llm = llm

    async def search_listings(self, question: str) -> SearchResult:
        """Generate and execute a SQL query from a natural language question."""
        try:
            sql = await self._generate_sql(question)
            rows = self._execute_sql(sql)
            return SearchResult(success=True, output=rows, result_count=len(rows))

        except SqlValidationError as exc:
            logger.error("SqlViewService | validation failed | %s", exc)
            return _SEARCH_ERROR

        except Exception as exc:
            logger.exception("SqlViewService | failed | %s", exc)
            return _SEARCH_ERROR

    async def search_from_context(self, ctx: SearchContext) -> SearchResult:
        """
        Execute a template-built SQL query from SearchContext — no LLM call.
        Used by listing_search_node when intent_node has already extracted entities.
        """
        try:
            sql = self.build_sql_from_context(ctx)
            rows = self._execute_sql(sql)
            return SearchResult(success=True, output=rows, result_count=len(rows))

        except Exception as exc:
            logger.exception(
                "SqlViewService.search_from_context | failed | %s", exc
            )
            return _SEARCH_ERROR

    def _execute_sql(self, sql: str) -> list[dict]:
        """Validate and run a SELECT query. Returns rows as a list of dicts."""
        self._validate_sql(sql)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            return [dict(row._mapping) for row in result]

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
    def build_sql_from_context(ctx: SearchContext) -> str:
        """
        Build a deterministic SELECT query from structured SearchContext fields.
        No LLM involved — avoids the SQL_GENERATION_PROMPT token cost entirely.

        Falls back to search_listings (LLM) when search_context lacks a location
        or contains criteria this template cannot express (e.g. "near good schools").
        """
        conditions = ["is_published = true", "is_active = true"]

        if ctx.get("property_id"):
            pid = ctx["property_id"].replace("'", "")
            conditions.append(f"property_id = '{pid}'")

        if ctx.get("location"):
            loc = ctx["location"].replace("'", "")
            conditions.append(f"suburb ILIKE '%{loc}%'")

        if ctx.get("address"):
            addr = ctx["address"].replace("'", "")
            conditions.append(f"address_line1 ILIKE '%{addr}%'")

        if ctx.get("listing_type") in ("Sale", "Rent"):
            conditions.append(f"listing_type = '{ctx['listing_type']}'")

        if ctx.get("property_type"):
            pt = ctx["property_type"].replace("'", "")
            # "Unit" is not a valid DB value — Australians use it interchangeably with Apartment
            if pt.lower() == "unit":
                pt = "Apartment"
            conditions.append(f"property_type ILIKE '{pt}'")

        if ctx.get("bedrooms") is not None:
            conditions.append(f"bedrooms = {int(ctx['bedrooms'])}")

        if ctx.get("bathrooms") is not None:
            conditions.append(f"bathrooms >= {int(ctx['bathrooms'])}")

        if ctx.get("max_price") is not None:
            conditions.append(f"price <= {float(ctx['max_price'])}")

        if ctx.get("min_price") is not None:
            conditions.append(f"price >= {float(ctx['min_price'])}")

        where = " AND ".join(conditions)
        limit = min(int(ctx.get("limit") or 10), 10)
        return f"{_SELECT_COLS} FROM v_listings WHERE {where} ORDER BY price ASC LIMIT {limit}"

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
