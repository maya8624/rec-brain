"""
Tests for SqlViewService — LLM SQL generation against v_listings.

Usage:
    pytest scripts/test_sql.py           # run all
    pytest scripts/test_sql.py -v        # verbose
    pytest scripts/test_sql.py -k price  # single group or test
"""
from app.services.sql_service import SqlViewService
from app.infrastructure.database import get_db
from app.infrastructure.llm import get_llm
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result: dict):
    print(json.dumps(result, indent=2, default=str))


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def service():
    return SqlViewService(llm=get_llm(), db=get_db())


# ── Search ─────────────────────────────────────────────────────────────────────


class TestSearch:
    async def test_basic(self, service: SqlViewService):
        """Basic suburb search."""
        result = await service.search_listings("Show me properties in Sydney")
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_state_full_name(self, service: SqlViewService):
        """State as full name instead of abbreviation."""
        result = await service.search_listings("Show me properties in Queensland")
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_no_results(self, service: SqlViewService):
        """Query that should return no results."""
        result = await service.search_listings(
            "10 bedroom mansions in Broken Hill under $100k"
        )
        pretty(result)
        assert result["success"], "Expected success even with 0 results"
        assert result["result_count"] == 0, f"Expected 0 results, got {result['result_count']}"


# ── Filters ────────────────────────────────────────────────────────────────────


class TestFilters:
    async def test_bedrooms(self, service: SqlViewService):
        """Filter by bedrooms."""
        result = await service.search_listings("3 bedroom houses in Melbourne")
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_price(self, service: SqlViewService):
        """Filter by price range."""
        result = await service.search_listings("Show me houses in Brisbane under $800k")
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_price_shorthand(self, service: SqlViewService):
        """Price shorthands like $800k and $1.2m."""
        result = await service.search_listings("Houses between $500k and $1.2m in Sydney")
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_rent(self, service: SqlViewService):
        """Rental listings."""
        result = await service.search_listings(
            "Find rental apartments in Parramatta under $600 per week"
        )
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")

    async def test_full_filter(self, service: SqlViewService):
        """All filters combined."""
        result = await service.search_listings(
            "Show me 3 bedroom townhouses in Parramatta NSW for sale under $900k"
        )
        pretty(result)
        assert result["success"]
        print(f"Found {result['result_count']} results")
