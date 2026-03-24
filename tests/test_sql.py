"""
Tests for SqlViewService — LLM SQL generation against v_listings.

Usage:
    python scripts/test_sql.py                    # run all
    python scripts/test_sql.py basic              # single test
    python scripts/test_sql.py bedrooms
    python scripts/test_sql.py price
    python scripts/test_sql.py rent
    python scripts/test_sql.py no_results
"""
from app.infrastructure.llm import get_llm
from app.infrastructure.database import get_db
from app.services.sql_service import SqlViewService
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result: dict):
    print(json.dumps(result, indent=2, default=str))


def print_header(name: str):
    print(f"\n── {name} {'─' * (50 - len(name))}")


# ── Tests ──────────────────────────────────────────────────────────────────────

async def test_basic(service: SqlViewService):
    """Basic suburb search."""
    print_header("basic — suburb search")
    result = await service.search_listings("Show me properties in Sydney")
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_bedrooms(service: SqlViewService):
    """Filter by bedrooms."""
    print_header("bedrooms — 3 bed filter")
    result = await service.search_listings("3 bedroom houses in Melbourne")
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_price(service: SqlViewService):
    """Filter by price range."""
    print_header("price — under $800k")
    result = await service.search_listings(
        "Show me houses in Brisbane under $800k"
    )
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_rent(service: SqlViewService):
    """Rental listings."""
    print_header("rent — rental properties")
    result = await service.search_listings(
        "Find rental apartments in Parramatta under $600 per week"
    )
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_full_filter(service: SqlViewService):
    """All filters combined."""
    print_header("full_filter — all params")
    result = await service.search_listings(
        "Show me 3 bedroom townhouses in Parramatta NSW for sale under $900k"
    )
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_no_results(service: SqlViewService):
    """Query that should return no results."""
    print_header("no_results — unlikely query")
    result = await service.search_listings(
        "10 bedroom mansions in Broken Hill under $100k"
    )
    pretty(result)
    assert result["success"], "Expected success even with 0 results"
    print(f"  ✓ Returned {result['result_count']} results (expected 0)")


async def test_price_shorthand(service: SqlViewService):
    """Price shorthands like $800k and $1.2m."""
    print_header("price_shorthand — $800k and $1.2m")
    result = await service.search_listings(
        "Houses between $500k and $1.2m in Sydney"
    )
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


async def test_state_full_name(service: SqlViewService):
    """State as full name instead of abbreviation."""
    print_header("state_full_name — Queensland")
    result = await service.search_listings(
        "Show me properties in Queensland"
    )
    pretty(result)
    assert result["success"], "Expected success"
    print(f"  ✓ Found {result['result_count']} results")


# ── Router ─────────────────────────────────────────────────────────────────────

TESTS = {
    "basic":           test_basic,
    "bedrooms":        test_bedrooms,
    "price":           test_price,
    "rent":            test_rent,
    "full_filter":     test_full_filter,
    "no_results":      test_no_results,
    "price_shorthand": test_price_shorthand,
    "state_full_name": test_state_full_name,
}


async def run_all(service: SqlViewService):
    passed = 0
    failed = 0

    for name, fn in TESTS.items():
        try:
            await fn(service)
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n── Results: {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f" | {failed} FAILED ✗")
    else:
        print(" ✓")


if __name__ == "__main__":
    service = SqlViewService(llm=get_llm(), db=get_db())

    if len(sys.argv) < 2:
        asyncio.run(run_all(service))
    else:
        name = sys.argv[1]
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(TESTS)}")
            sys.exit(1)
        asyncio.run(TESTS[name](service))
