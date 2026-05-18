"""
Unit tests for SearchService — preference-based listing search and suburb summary.
All external calls (SQL, RAG) are mocked.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.services.search_service import SearchService, _to_search_query, DISPLAY_COUNT
from app.schemas.search import TenantPreference, PreferenceSearchResponse, SuburbSummaryResponse
from app.schemas.property import SearchResult

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_listing(n: int = 1) -> list[dict]:
    return [
        {
            "address_line1": f"{i} Test St",
            "suburb": "Sydney",
            "price": 500 + i * 10,
            "bedrooms": 2,
        }
        for i in range(n)
    ]


def make_service(
    listings: list[dict] | None = None,
    search_success: bool = True,
    summary_text: str = "Great options found.",
    suburb_summary_text: str = "Bondi is a vibrant coastal suburb.",
    nodes: list | None = None,
    sql_error: Exception | None = None,
    rag_error: Exception | None = None,
):
    mock_sql = AsyncMock()
    mock_rag = AsyncMock()
    mock_llm = AsyncMock()

    rows = listings if listings is not None else make_listing(1)

    if sql_error:
        mock_sql.search_listings.side_effect = sql_error
    else:
        mock_sql.search_listings.return_value = SearchResult(
            success=search_success,
            output=rows if search_success else None,
            result_count=len(rows) if search_success else 0,
        )
    mock_sql.generate_summary.return_value = summary_text
    mock_llm.ainvoke.return_value = MagicMock(content=suburb_summary_text)

    if rag_error:
        mock_rag.aretrieve.side_effect = rag_error
    else:
        if nodes is None:
            node = MagicMock()
            node.node.get_content.return_value = "Bondi is a vibrant coastal suburb."
            nodes = [node]
        mock_rag.aretrieve.return_value = nodes

    return SearchService(sql=mock_sql, rag=mock_rag, llm=mock_llm), mock_sql, mock_rag, mock_llm


def make_pref(**kwargs) -> TenantPreference:
    defaults = {"suburbs": ["Bondi"], "maxRent": 600.0, "minBeds": 2, "maxBeds": 3}
    return TenantPreference(**{**defaults, **kwargs})


# ── _to_nl ────────────────────────────────────────────────────────────────────

class TestToSearchQuery:
    def test_single_suburb(self):
        assert "in Bondi" in _to_search_query(make_pref(suburbs=["Bondi"]))

    def test_multiple_suburbs(self):
        result = _to_search_query(make_pref(suburbs=["Bondi", "Manly", "Coogee"]))
        assert "Bondi, Manly" in result
        assert "or Coogee" in result

    def test_bedroom_range(self):
        assert "2-4 bedroom" in _to_search_query(make_pref(minBeds=2, maxBeds=4))

    def test_min_beds_only(self):
        assert "at least 3 bedroom" in _to_search_query(make_pref(minBeds=3, maxBeds=None))

    def test_max_beds_only(self):
        assert "up to 4 bedroom" in _to_search_query(make_pref(minBeds=None, maxBeds=4))

    def test_same_min_max_beds_not_shown_as_range(self):
        result = _to_search_query(make_pref(minBeds=2, maxBeds=2))
        assert "2-2 bedroom" not in result
        assert "at least 2 bedroom" in result

    def test_pet_friendly(self):
        assert "pet friendly" in _to_search_query(make_pref(petFriendly=True))

    def test_max_rent(self):
        assert "under $550/wk" in _to_search_query(make_pref(maxRent=550.0))

    def test_available_within_days(self):
        assert "available within 14 days" in _to_search_query(make_pref(availableWithinDays=14))

    def test_no_preferences_still_valid(self):
        result = _to_search_query(TenantPreference(suburbs=[]))
        assert result.startswith("Find me a")
        assert "property" in result

    def test_starts_with_find_me(self):
        assert _to_search_query(make_pref()).startswith("Find me a")


# ── search_by_preferences ─────────────────────────────────────────────────────

class TestSearchByPreferences:
    async def test_returns_preference_search_response(self):
        svc, _, _, _ = make_service()
        result = await svc.search_by_preferences(make_pref())
        assert isinstance(result, PreferenceSearchResponse)

    async def test_message_comes_from_generate_summary(self):
        svc, _, _, _ = make_service(summary_text="Found great listings!")
        result = await svc.search_by_preferences(make_pref())
        assert result.message == "Found great listings!"

    async def test_total_count_matches_all_listings(self):
        svc, _, _, _ = make_service(listings=make_listing(6))
        result = await svc.search_by_preferences(make_pref())
        assert result.total_count == 6

    async def test_display_count_capped_at_display_count(self):
        svc, _, _, _ = make_service(listings=make_listing(DISPLAY_COUNT + 2))
        result = await svc.search_by_preferences(make_pref())
        assert result.display_count == DISPLAY_COUNT

    async def test_has_more_true_when_exceeds_display_count(self):
        svc, _, _, _ = make_service(listings=make_listing(DISPLAY_COUNT + 1))
        result = await svc.search_by_preferences(make_pref())
        assert result.has_more is True

    async def test_has_more_false_when_within_display_count(self):
        svc, _, _, _ = make_service(listings=make_listing(DISPLAY_COUNT))
        result = await svc.search_by_preferences(make_pref())
        assert result.has_more is False

    async def test_empty_results(self):
        svc, _, _, _ = make_service(listings=[])
        result = await svc.search_by_preferences(make_pref())
        assert result.total_count == 0
        assert result.listings == []
        assert result.has_more is False

    async def test_generate_summary_receives_top_slice_only(self):
        listings = make_listing(DISPLAY_COUNT + 2)
        svc, mock_sql, _, _ = make_service(listings=listings)
        await svc.search_by_preferences(make_pref())
        _, called_top, called_total = mock_sql.generate_summary.call_args.args
        assert len(called_top) == DISPLAY_COUNT
        assert called_total == DISPLAY_COUNT + 2

    async def test_search_listings_called_with_nl_query(self):
        svc, mock_sql, _, _ = make_service()
        await svc.search_by_preferences(make_pref(suburbs=["Newtown"]))
        query = mock_sql.search_listings.call_args.args[0]
        assert "Newtown" in query

    async def test_failed_search_returns_empty_listings(self):
        svc, _, _, _ = make_service(search_success=False)
        result = await svc.search_by_preferences(make_pref())
        assert result.listings == []
        assert result.total_count == 0


# ── get_suburb_summary ────────────────────────────────────────────────────────

class TestGetSuburbSummary:
    async def test_returns_suburb_summary_response(self):
        svc, _, _, _ = make_service()
        result = await svc.get_suburb_summary(["Bondi"])
        assert isinstance(result, SuburbSummaryResponse)

    async def test_summary_comes_from_llm(self):
        svc, _, _, _ = make_service(suburb_summary_text="Bondi is a vibrant coastal suburb.")
        result = await svc.get_suburb_summary(["Bondi"])
        assert result.summary == "Bondi is a vibrant coastal suburb."

    async def test_llm_called_with_node_context(self):
        node = MagicMock()
        node.node.get_content.return_value = "Bondi has great beaches."
        svc, _, _, mock_llm = make_service(nodes=[node])
        await svc.get_suburb_summary(["Bondi"])
        prompt = mock_llm.ainvoke.call_args.args[0][0].content
        assert "Bondi has great beaches." in prompt

    async def test_multiple_nodes_context_passed_to_llm(self):
        nodes = []
        for text in ["First paragraph.", "Second paragraph."]:
            n = MagicMock()
            n.node.get_content.return_value = text
            nodes.append(n)
        svc, _, _, mock_llm = make_service(nodes=nodes)
        await svc.get_suburb_summary(["Bondi"])
        prompt = mock_llm.ainvoke.call_args.args[0][0].content
        assert "First paragraph." in prompt
        assert "Second paragraph." in prompt

    async def test_no_nodes_returns_none_summary(self):
        svc, _, _, _ = make_service(nodes=[])
        result = await svc.get_suburb_summary(["Bondi"])
        assert result.summary is None

    async def test_query_includes_suburb_name(self):
        svc, _, mock_rag, _ = make_service()
        await svc.get_suburb_summary(["Newtown"])
        query = mock_rag.aretrieve.call_args.args[0]
        assert "Newtown" in query

    async def test_multiple_suburbs_in_query(self):
        svc, _, mock_rag, _ = make_service()
        await svc.get_suburb_summary(["Bondi", "Manly"])
        query = mock_rag.aretrieve.call_args.args[0]
        assert "Bondi" in query
        assert "Manly" in query

    async def test_empty_suburbs_uses_fallback_query(self):
        svc, _, mock_rag, _ = make_service()
        await svc.get_suburb_summary([])
        query = mock_rag.aretrieve.call_args.args[0]
        assert query == "suburb overview"
