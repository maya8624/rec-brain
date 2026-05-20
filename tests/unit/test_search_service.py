"""
Unit tests for SearchService — preference-based listing search and suburb summary.
All external calls (SQL, RAG, LLM) are mocked — no real I/O.
"""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.search_service import SearchService, _to_search_query, DISPLAY_COUNT
from app.schemas.search import TenantPreference, PreferenceSearchResponse, SuburbSummaryResponse, SuburbProfile, SuburbRents
from app.schemas.property import SearchResult

pytestmark = pytest.mark.unit


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_listing(n: int = 1) -> list[dict]:
    return [
        {
            "listing_id": str(uuid.uuid4()),
            "property_id": str(uuid.uuid4()),
            "listing_type": "Rent",
            "listing_status": "Active",
            "price": 500 + i * 10,
            "bedrooms": 2,
            "bathrooms": 1,
            "car_spaces": 0,
            "property_type": "Apartment",
            "address_line1": f"{i} Test St",
            "suburb": "Sydney",
            "state": "NSW",
            "postcode": "2000",
        }
        for i in range(n)
    ]


def make_node(text: str = "Suburb content."):
    n = MagicMock()
    n.node.get_content.return_value = text
    return n


def make_rag(
    nodes: list | None = None,
    nodes_per_call: list[list] | None = None,
    raise_error: Exception | None = None,
) -> AsyncMock:
    """
    nodes          — same node list returned for every aretrieve call
    nodes_per_call — list of per-call return values (for multi-suburb tests)
    raise_error    — raised on every call
    """
    mock = AsyncMock()
    if raise_error:
        mock.aretrieve.side_effect = raise_error
    elif nodes_per_call is not None:
        mock.aretrieve.side_effect = nodes_per_call
    else:
        mock.aretrieve.return_value = nodes if nodes is not None else [make_node()]
    return mock


def make_sql(
    listings: list[dict] | None = None,
    search_success: bool = True,
    raise_error: Exception | None = None,
) -> AsyncMock:
    mock = AsyncMock()
    rows = listings if listings is not None else make_listing(1)
    if raise_error:
        mock.search_listings.side_effect = raise_error
    else:
        mock.search_listings.return_value = SearchResult(
            success=search_success,
            output=rows if search_success else None,
            result_count=len(rows) if search_success else 0,
        )
    return mock


def make_llm(
    content: str = "Great options found.",
    suburb_response: SuburbSummaryResponse | None = None,
) -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    structured = MagicMock()
    structured.ainvoke = AsyncMock(return_value=suburb_response or SuburbSummaryResponse(suburbs=[
        SuburbProfile(
            name="Bondi",
            description="A vibrant coastal suburb.",
            rents=SuburbRents(one_bedroom="$500/wk"),
            vacancy_rate="3.1%",
            trend="up 1.5% QoQ",
        )
    ]))
    mock.with_structured_output.return_value = structured
    return mock


def make_service(sql=None, rag=None, llm=None) -> SearchService:
    return SearchService(
        sql=sql or make_sql(),
        rag=rag or make_rag(),
        llm=llm or make_llm(),
    )


def make_pref(**kwargs) -> TenantPreference:
    defaults = {"suburbs": ["Bondi"], "maxRent": 600.0, "minBeds": 2, "maxBeds": 3}
    return TenantPreference(**{**defaults, **kwargs})


# ── _to_search_query ──────────────────────────────────────────────────────────

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
        result = await make_service().search_by_preferences(make_pref())
        assert isinstance(result, PreferenceSearchResponse)

    async def test_message_comes_from_llm(self):
        svc = make_service(llm=make_llm("Found great listings!"))
        result = await svc.search_by_preferences(make_pref())
        assert result.message == "Found great listings!"

    async def test_total_count_matches_all_listings(self):
        svc = make_service(sql=make_sql(listings=make_listing(6)))
        result = await svc.search_by_preferences(make_pref())
        assert result.total_count == 6

    async def test_display_count_capped_at_display_count(self):
        svc = make_service(sql=make_sql(listings=make_listing(DISPLAY_COUNT + 2)))
        result = await svc.search_by_preferences(make_pref())
        assert result.display_count == DISPLAY_COUNT

    async def test_has_more_true_when_exceeds_display_count(self):
        svc = make_service(sql=make_sql(listings=make_listing(DISPLAY_COUNT + 1)))
        result = await svc.search_by_preferences(make_pref())
        assert result.has_more is True

    async def test_has_more_false_when_within_display_count(self):
        svc = make_service(sql=make_sql(listings=make_listing(DISPLAY_COUNT)))
        result = await svc.search_by_preferences(make_pref())
        assert result.has_more is False

    async def test_empty_results(self):
        svc = make_service(sql=make_sql(listings=[]))
        result = await svc.search_by_preferences(make_pref())
        assert result.total_count == 0
        assert result.listings == []
        assert result.has_more is False

    async def test_generate_summary_receives_top_slice_only(self):
        sql = make_sql(listings=make_listing(DISPLAY_COUNT + 2))
        svc = make_service(sql=sql)
        with patch.object(svc, "_generate_summary", new=AsyncMock(return_value="ok")) as mock_summary:
            await svc.search_by_preferences(make_pref())
        _, called_top, called_total = mock_summary.call_args.args
        assert len(called_top) == DISPLAY_COUNT
        assert called_total == DISPLAY_COUNT + 2

    async def test_search_listings_called_with_nl_query(self):
        sql = make_sql()
        svc = make_service(sql=sql)
        await svc.search_by_preferences(make_pref(suburbs=["Newtown"]))
        query = sql.search_listings.call_args.args[0]
        assert "Newtown" in query

    async def test_failed_search_returns_empty_listings(self):
        svc = make_service(sql=make_sql(search_success=False))
        result = await svc.search_by_preferences(make_pref())
        assert result.listings == []
        assert result.total_count == 0


# ── get_suburb_summary ────────────────────────────────────────────────────────

def make_suburb_response(*names: str) -> SuburbSummaryResponse:
    return SuburbSummaryResponse(suburbs=[
        SuburbProfile(
            name=name,
            description=f"{name} is a great suburb.",
            rents=SuburbRents(one_bedroom="$500/wk"),
            vacancy_rate="2.0%",
            trend="up 1.5% QoQ",
        )
        for name in names
    ])


class TestGetSuburbSummary:
    async def test_returns_suburb_summary_response(self):
        result = await make_service().get_suburb_summary(["Bondi"])
        assert isinstance(result, SuburbSummaryResponse)

    async def test_structured_llm_result_returned(self):
        expected = make_suburb_response("Bondi")
        llm = make_llm(suburb_response=expected)
        svc = make_service(llm=llm)
        result = await svc.get_suburb_summary(["Bondi"])
        assert result.suburbs[0].name == "Bondi"
        assert result.suburbs[0].rents.one_bedroom == "$500/wk"

    async def test_empty_suburbs_returns_empty_response_without_calling_rag(self):
        rag = make_rag()
        svc = make_service(rag=rag)
        result = await svc.get_suburb_summary([])
        assert result.suburbs == []
        rag.aretrieve.assert_not_called()

    async def test_no_nodes_returns_empty_suburbs(self):
        svc = make_service(rag=make_rag(nodes=[]))
        result = await svc.get_suburb_summary(["Bondi"])
        assert result.suburbs == []

    async def test_no_nodes_does_not_call_llm(self):
        llm = make_llm()
        svc = make_service(rag=make_rag(nodes=[]), llm=llm)
        await svc.get_suburb_summary(["Bondi"])
        llm.with_structured_output.assert_not_called()

    async def test_aretrieve_called_once_per_suburb(self):
        rag = make_rag(nodes_per_call=[
            [make_node("Bondi content")],
            [make_node("Surry Hills content")],
            [make_node("Newtown content")],
        ])
        svc = make_service(rag=rag)
        await svc.get_suburb_summary(["Bondi", "Surry Hills", "Newtown"])
        assert rag.aretrieve.call_count == 3

    async def test_aretrieve_called_with_doc_type_guide(self):
        rag = make_rag()
        svc = make_service(rag=rag)
        await svc.get_suburb_summary(["Bondi"])
        assert rag.aretrieve.call_args.kwargs.get("doc_type") == "guide"

    async def test_suburb_name_in_rag_query(self):
        rag = make_rag()
        svc = make_service(rag=rag)
        await svc.get_suburb_summary(["Castle Hill"])
        query = rag.aretrieve.call_args.args[0]
        assert "Castle Hill" in query

    async def test_context_block_labelled_by_suburb(self):
        rag = make_rag(nodes=[make_node("coastal vibes")])
        llm = make_llm()
        svc = make_service(rag=rag, llm=llm)
        await svc.get_suburb_summary(["Bondi"])
        structured_mock = llm.with_structured_output.return_value
        prompt = structured_mock.ainvoke.call_args.args[0][0].content
        assert "### Bondi" in prompt
        assert "coastal vibes" in prompt

    async def test_failed_suburb_retrieval_skipped(self):
        rag = make_rag(nodes_per_call=[
            RuntimeError("timeout"),
            [make_node("Surry Hills content")],
        ])
        svc = make_service(rag=rag, llm=make_llm(suburb_response=make_suburb_response("Surry Hills")))
        result = await svc.get_suburb_summary(["Bondi", "Surry Hills"])
        assert result.suburbs[0].name == "Surry Hills"

    async def test_empty_nodes_for_one_suburb_skipped(self):
        rag = make_rag(nodes_per_call=[
            [],
            [make_node("Newtown content")],
        ])
        svc = make_service(rag=rag, llm=make_llm(suburb_response=make_suburb_response("Newtown")))
        result = await svc.get_suburb_summary(["Bondi", "Newtown"])
        assert result.suburbs[0].name == "Newtown"

    async def test_all_suburbs_failed_returns_empty(self):
        rag = make_rag(nodes_per_call=[
            RuntimeError("timeout"),
            RuntimeError("timeout"),
        ])
        svc = make_service(rag=rag)
        result = await svc.get_suburb_summary(["Bondi", "Surry Hills"])
        assert result.suburbs == []

    async def test_node_content_included_in_llm_prompt(self):
        rag = make_rag(nodes=[make_node("Great beaches and cafes.")])
        llm = make_llm()
        svc = make_service(rag=rag, llm=llm)
        await svc.get_suburb_summary(["Bondi"])
        structured_mock = llm.with_structured_output.return_value
        prompt = structured_mock.ainvoke.call_args.args[0][0].content
        assert "Great beaches and cafes." in prompt
