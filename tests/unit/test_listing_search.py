"""
Unit tests for listing_search_node — direct SQL search, no tool calls.
Uses make_sql_service and make_config from conftest.
"""
from langchain_core.messages import HumanMessage

from app.agents.nodes.listing import listing_search_node


class TestListingSearchSuccess:
    async def test_returns_retrieved_docs(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], str)

    async def test_retrieved_docs_contains_result_count(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        assert "[PROPERTY SEARCH RESULTS" in result["retrieved_docs"]
        assert "1 listing(s) found" in result["retrieved_docs"]

    async def test_search_results_returned(self, make_sql_service, make_config):
        svc = make_sql_service(result={
            "success": True,
            "output": [{"address": "1 Test St"}, {"address": "2 Test St"}],
            "result_count": 2,
            "sql_used": "SELECT * FROM v_listings",
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

    async def test_zero_results_returns_retrieved_docs(self, make_sql_service, make_config):
        """0 results still returns retrieved_docs with a no-results message."""
        svc = make_sql_service(result={
            "success": True,
            "output": [],
            "result_count": 0,
            "sql_used": "SELECT * FROM v_listings",
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses in Sydney")]},
            make_config(sql_service=svc),
        )

        assert "retrieved_docs" in result
        assert "0 listing(s) found" in result["retrieved_docs"]

    async def test_zero_results_does_not_reference_previous_listings(self, make_sql_service, make_config):
        """0-result message includes instruction not to reuse previous responses."""
        svc = make_sql_service(result={
            "success": True,
            "output": [],
            "result_count": 0,
            "sql_used": "SELECT * FROM v_listings",
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="townhouses in Melbourne")]},
            make_config(sql_service=svc),
        )

        assert "Do NOT reference" in result["retrieved_docs"]

    async def test_ten_results_includes_pagination_note(self, make_sql_service, make_config):
        """When exactly 10 results are returned, a pagination note is appended."""
        svc = make_sql_service(result={
            "success": True,
            "output": [{"address": f"{i} St"} for i in range(10)],
            "result_count": 10,
            "sql_used": "SELECT * FROM v_listings LIMIT 10",
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="apartments in Sydney")]},
            make_config(sql_service=svc),
        )

        assert "top 10" in result["retrieved_docs"]

    async def test_fewer_than_ten_results_no_pagination_note(self, make_sql_service, make_config):
        """Fewer than 10 results — no pagination note needed."""
        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses in Sydney")]},
            make_config(sql_service=make_sql_service()),  # default returns 1 result
        )

        assert "top 10" not in result["retrieved_docs"]

    async def test_success_false_still_returns_retrieved_docs(self, make_sql_service, make_config):
        """If service reports success=False, the node still returns retrieved_docs."""
        svc = make_sql_service(result={
            "success": False,
            "output": None,
            "result_count": 0,
            "error": "Property search temporarily unavailable.",
        })

        result = await listing_search_node(
            {"messages": [HumanMessage(content="houses")]},
            make_config(sql_service=svc),
        )

        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], str)


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
