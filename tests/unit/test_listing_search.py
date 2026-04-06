"""
Unit tests for listing_search_node — direct SQL search, no tool calls.
Uses make_sql_service and make_config from conftest.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.listing import listing_search_node
from tests.conftest import parsed


class TestListingSearchSuccess:
    async def test_returns_system_message(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)

    async def test_result_structure(self, make_sql_service, make_config):
        result = await listing_search_node(
            {"messages": [HumanMessage(content="3 bedroom houses in Sydney")]},
            make_config(sql_service=make_sql_service()),
        )

        content = parsed(result)
        assert "search_results" in content
        assert "result_count" in content
        assert "success" in content

    async def test_result_count_matches_service(self, make_sql_service, make_config):
        svc = make_sql_service(result={
            "success": True,
            "output": [{"address": "1 Test St"}, {"address": "2 Test St"}],
            "result_count": 2,
            "sql_used": "SELECT * FROM v_listings",
        })

        content = parsed(await listing_search_node(
            {"messages": [HumanMessage(content="houses in Melbourne")]},
            make_config(sql_service=svc),
        ))

        assert content["result_count"] == 2

    async def test_calls_service_with_exact_question(self, make_sql_service, make_config):
        svc = make_sql_service()
        question = "3 bedroom townhouses in Parramatta under $900k"

        await listing_search_node(
            {"messages": [HumanMessage(content=question)]},
            make_config(sql_service=svc),
        )

        svc.search_listings.assert_called_once_with(question)

    async def test_success_false_still_returns_message(self, make_sql_service, make_config):
        """If service reports success=False (e.g. SQL error), the node still returns a message."""
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

        content = parsed(result)
        assert content["success"] is False
        assert content["error"] is not None


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
