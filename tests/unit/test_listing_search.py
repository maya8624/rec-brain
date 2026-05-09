"""
Unit tests for listing_search_node — direct SQL search, no tool calls.
Uses make_sql_service and make_config from conftest.
"""
import pytest
from langchain_core.messages import HumanMessage

from app.schemas.property import SearchResult
from app.agents.nodes.listing import listing_search_node


class TestListingSearchSuccess:
    async def test_returns_search_results(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        assert "search_results" in result

    async def test_search_results_has_one_item(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        assert len(result["search_results"]) == 1

    async def test_search_results_returned_with_count(self, make_sql_service, make_config):
        svc = make_sql_service(result={
            "success": True,
            "output": [{"address": "1 Test St"}, {"address": "2 Test St"}],
            "result_count": 2,
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses in Melbourne")]},
            make_config(sql_service=svc),
        )

        assert "search_results" in result
        assert len(result["search_results"]) == 2

    async def test_calls_service_with_exact_question(self, make_sql_service, make_config):
        svc = make_sql_service()
        question = "3 bedroom townhouses in Parramatta under $900k"

        await listing_search_node(
            {"messages": [HumanMessage(content=question)]},
            make_config(sql_service=svc),
        )

        svc.search_listings.assert_called_once_with(question)

    async def test_context_path_uses_search_from_context(self, make_sql_service, make_config):
        """When search_context has a location, use search_from_context — no LLM call."""
        svc = make_sql_service()
        state = {
            "messages": [HumanMessage(content="3 bed houses in Sydney under $800k")],
            "search_context": {"location": "Sydney", "property_type": "House", "bedrooms": 3},
        }
        await listing_search_node(state, make_config(sql_service=svc))

        svc.search_from_context.assert_called_once_with(
            {"location": "Sydney", "property_type": "House", "bedrooms": 3}
        )
        svc.search_listings.assert_not_called()

    async def test_no_location_falls_back_to_llm(self, make_sql_service, make_config):
        """Without a location in search_context, fall back to sql_service LLM path."""
        svc = make_sql_service()
        question = "properties near good schools"
        state = {
            "messages": [HumanMessage(content=question)],
            "search_context": {},
        }
        await listing_search_node(state, make_config(sql_service=svc))

        svc.search_listings.assert_called_once_with(question)
        svc.search_from_context.assert_not_called()

    async def test_zero_results_returns_empty_search_results(self, make_sql_service, make_config):
        """0 results — search_results is an empty list."""
        svc = make_sql_service(result={
            "success": True,
            "output": [],
            "result_count": 0,
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses in Sydney")]},
            make_config(sql_service=svc),
        )

        assert "search_results" in result
        assert result["search_results"] == []

    async def test_ten_results_all_returned(self, make_sql_service, make_config):
        """All 10 results appear in search_results."""
        svc = make_sql_service(result={
            "success": True,
            "output": [{"address": f"{i} St"} for i in range(10)],
            "result_count": 10,
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="apartments in Sydney")]},
            make_config(sql_service=svc),
        )

        assert len(result["search_results"]) == 10

    async def test_success_false_returns_empty_search_results(self, make_sql_service, make_config):
        """If service reports success=False, search_results is empty."""
        svc = make_sql_service(result=SearchResult(
            success=False,
            output=None,
            result_count=0,
            error="Property search temporarily unavailable.",
        ))

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses")]},
            make_config(sql_service=svc),
        )

        assert "search_results" in result
        assert result["search_results"] == []

    async def test_retrieved_docs_cleared(self, make_sql_service, make_config):
        """listing_search_node clears retrieved_docs so stale vector results don't persist."""
        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )
        assert result.get("retrieved_docs") is None


class TestListingSearchGuards:
    async def test_no_human_message_returns_empty(self, make_sql_service, make_config):
        svc = make_sql_service()
        result = await listing_search_node(
            {"messages": []},
            make_config(sql_service=svc),
        )

        assert result == {}
        svc.search_listings.assert_not_called()

    async def test_missing_service_returns_empty(self, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses")]},
            make_config(sql_service=None),
        )

        assert result == {}

    async def test_missing_request_returns_empty(self):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses")]},
            {"configurable": {}},
        )

        assert result == {}

    async def test_service_exception_returns_empty(self, make_sql_service, make_config):
        svc = make_sql_service(
            raise_error=RuntimeError("DB connection dropped")
        )

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses")]},
            make_config(sql_service=svc),
        )

        assert result == {}
