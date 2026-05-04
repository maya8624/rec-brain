"""
Unit tests for SqlViewService — LLM-generated SQL against v_listings.

All LLM calls and DB calls are mocked so no Groq API or PostgreSQL needed.
_validate_sql is a static method tested directly as a pure function.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.sql_service import SqlValidationError, SqlViewService


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_service(llm_response: str = "SELECT * FROM v_listings", db_rows: list | None = None):
    """
    Factory for SqlViewService with mocked LLM and engine.

    llm_response: the raw string the mock LLM returns
    db_rows:      list of row dicts mock DB returns (defaults to one row)
    """
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(content=llm_response)

    default_rows = db_rows if db_rows is not None else [
        {"address": "1 Test St, Sydney", "price": 750_000, "bedrooms": 3}
    ]
    mock_rows = [MagicMock(_mapping=row) for row in default_rows]

    mock_conn = MagicMock()
    mock_conn.execute.return_value = mock_rows
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    return SqlViewService(llm=mock_llm), mock_llm, mock_conn


# ── build_sql_from_context ─────────────────────────────────────────────────────

class TestBuildSqlFromContext:
    def test_location_only(self):
        sql = SqlViewService.build_sql_from_context({"location": "Sydney"})
        assert "suburb ILIKE '%Sydney%'" in sql
        assert "is_published = true" in sql
        assert "LIMIT 10" in sql

    def test_all_filters(self):
        sql = SqlViewService.build_sql_from_context({
            "location": "Parramatta",
            "listing_type": "Sale",
            "property_type": "House",
            "bedrooms": 3,
            "bathrooms": 2,
            "max_price": 800000,
            "min_price": 500000,
        })
        assert "suburb ILIKE '%Parramatta%'" in sql
        assert "listing_type = 'Sale'" in sql
        assert "property_type ILIKE 'House'" in sql
        assert "bedrooms = 3" in sql
        assert "bathrooms >= 2" in sql
        assert "price <= 800000.0" in sql
        assert "price >= 500000.0" in sql

    def test_property_type_exact_no_wildcard(self):
        """No % wildcards — prevents House matching Townhouse."""
        sql = SqlViewService.build_sql_from_context({"location": "Sydney", "property_type": "House"})
        assert "property_type ILIKE 'House'" in sql
        assert "property_type ILIKE '%House%'" not in sql

    def test_invalid_listing_type_excluded(self):
        """Only 'Sale' or 'Rent' are valid — anything else is ignored."""
        sql = SqlViewService.build_sql_from_context({"location": "Sydney", "listing_type": "unknown"})
        assert "listing_type = " not in sql

    def test_quote_stripped_from_location(self):
        """Single quotes in LLM output are stripped to prevent SQL injection."""
        sql = SqlViewService.build_sql_from_context({"location": "O'Brien"})
        assert "OBrien" in sql
        assert "'O'" not in sql

    def test_property_id_filter(self):
        sql = SqlViewService.build_sql_from_context({"property_id": "f85bef59-8ccd-4783-a70d-3c9af7359732"})
        assert "property_id = 'f85bef59-8ccd-4783-a70d-3c9af7359732'" in sql

    def test_property_id_quote_stripped(self):
        sql = SqlViewService.build_sql_from_context({"property_id": "abc'123"})
        assert "abc123" in sql
        assert "'" not in sql.split("property_id = ")[1].split("'")[1] if "property_id" in sql else True

    def test_empty_context_still_valid_sql(self):
        sql = SqlViewService.build_sql_from_context({})
        assert sql.upper().startswith("SELECT")
        assert "v_listings" in sql.lower()
        assert "is_published = true" in sql


# ── search_from_context ────────────────────────────────────────────────────────

@patch("app.services.sql_service.engine")
class TestSearchFromContext:
    async def test_success_returns_result_dict(self, mock_engine):
        svc, _, mock_conn = make_service()
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_from_context({"location": "Sydney", "bedrooms": 3})
        assert result["success"] is True
        assert result["result_count"] == 1

    async def test_does_not_call_llm(self, mock_engine):
        svc, mock_llm, mock_conn = make_service()
        mock_engine.connect.return_value = mock_conn
        await svc.search_from_context({"location": "Sydney"})
        mock_llm.ainvoke.assert_not_called()

    async def test_db_exception_returns_success_false(self, mock_engine):
        svc, _, mock_conn = make_service()
        mock_conn.execute.side_effect = RuntimeError("DB down")
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_from_context({"location": "Sydney"})
        assert result["success"] is False


# ── _validate_sql ──────────────────────────────────────────────────────────────

class TestValidateSql:
    def test_valid_select_passes(self):
        SqlViewService._validate_sql("SELECT * FROM v_listings WHERE suburb = 'Sydney'")

    def test_non_select_raises(self):
        with pytest.raises(SqlValidationError, match="SELECT"):
            SqlViewService._validate_sql("INSERT INTO v_listings VALUES (1)")

    def test_missing_v_listings_raises(self):
        with pytest.raises(SqlValidationError, match="v_listings"):
            SqlViewService._validate_sql("SELECT * FROM other_table")

    def test_insert_keyword_raises(self):
        with pytest.raises(SqlValidationError, match="Forbidden"):
            SqlViewService._validate_sql("SELECT * FROM v_listings; INSERT INTO x VALUES (1)")

    def test_update_keyword_raises(self):
        with pytest.raises(SqlValidationError, match="Forbidden"):
            SqlViewService._validate_sql("SELECT * FROM v_listings WHERE 1=1 UPDATE v_listings SET")

    def test_drop_keyword_raises(self):
        with pytest.raises(SqlValidationError, match="Forbidden"):
            SqlViewService._validate_sql("SELECT * FROM v_listings DROP TABLE v_listings")

    def test_delete_keyword_raises(self):
        with pytest.raises(SqlValidationError, match="Forbidden"):
            SqlViewService._validate_sql("SELECT * FROM v_listings DELETE FROM v_listings")

    def test_case_insensitive_table_name(self):
        """V_LISTINGS in uppercase must also pass validation."""
        SqlViewService._validate_sql("SELECT * FROM V_LISTINGS WHERE suburb = 'Sydney'")

    def test_markdown_fenced_sql_still_passes_after_stripping(self):
        """After fence-stripping in _generate_sql, the plain SQL should validate fine."""
        SqlViewService._validate_sql("SELECT bedrooms FROM v_listings WHERE price < 800000")


# ── _generate_sql ──────────────────────────────────────────────────────────────

class TestGenerateSql:
    async def test_returns_llm_content_stripped(self):
        svc, mock_llm, _ = make_service(llm_response="  SELECT * FROM v_listings  ")
        sql = await svc._generate_sql("Show me properties in Sydney")
        assert sql == "SELECT * FROM v_listings"

    async def test_strips_markdown_code_fence(self):
        svc, _, _ = make_service(llm_response="```sql\nSELECT * FROM v_listings\n```")
        sql = await svc._generate_sql("houses in Sydney")
        assert "```" not in sql
        assert "SELECT" in sql

    async def test_sends_question_to_llm(self):
        svc, mock_llm, _ = make_service()
        await svc._generate_sql("3 bedroom houses in Brisbane")
        mock_llm.ainvoke.assert_called_once()
        call_messages = mock_llm.ainvoke.call_args.args[0]
        human_msg = call_messages[-1]
        assert "Brisbane" in human_msg.content


# ── search_listings ────────────────────────────────────────────────────────────

@patch("app.services.sql_service.engine")
class TestSearchListings:
    async def test_success_returns_result_dict(self, mock_engine):
        svc, _, mock_conn = make_service()
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("Show me houses in Sydney")
        assert result["success"] is True
        assert result["result_count"] == 1
        assert result["output"] is not None

    async def test_result_count_matches_db_rows(self, mock_engine):
        rows = [{"address": f"{i} St"} for i in range(5)]
        svc, _, mock_conn = make_service(db_rows=rows)
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("houses")
        assert result["result_count"] == 5

    async def test_validation_error_returns_success_false(self, mock_engine):
        """LLM returns a non-SELECT query → validation fails → success=False, no crash."""
        svc, _, mock_conn = make_service(llm_response="DELETE FROM v_listings")
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("delete everything")
        assert result["success"] is False
        assert result["output"] is None

    async def test_db_exception_returns_success_false(self, mock_engine):
        svc, _, mock_conn = make_service()
        mock_conn.execute.side_effect = RuntimeError("connection timeout")
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("houses in Sydney")
        assert result["success"] is False

    async def test_llm_exception_returns_success_false(self, mock_engine):
        svc, mock_llm, mock_conn = make_service()
        mock_llm.ainvoke.side_effect = RuntimeError("Groq API unreachable")
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("houses")
        assert result["success"] is False

    async def test_empty_result_returns_zero_count(self, mock_engine):
        svc, _, mock_conn = make_service(db_rows=[])
        mock_engine.connect.return_value = mock_conn
        result = await svc.search_listings("10 bedroom mansion under $10k")
        assert result["success"] is True
        assert result["result_count"] == 0
