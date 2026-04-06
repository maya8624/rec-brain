"""
Live integration tests for SqlViewService.

Hits a real Groq LLM and real PostgreSQL — requires a configured .env.
Skip with: pytest -m unit
Run with:  pytest -m integration
"""
import pytest
from tests.integration.conftest import skip_if_no_env

pytestmark = [pytest.mark.integration, skip_if_no_env]


@pytest.fixture(scope="module")
def service():
    from app.services.sql_service import SqlViewService
    from app.infrastructure.database import get_db
    from app.infrastructure.llm import get_llm
    return SqlViewService(llm=get_llm(), db=get_db())


class TestSearchLive:
    async def test_basic_suburb_search(self, service):
        result = await service.search_listings("Show me properties in Sydney")
        assert result["success"] is True

    async def test_state_full_name(self, service):
        result = await service.search_listings("Show me properties in Queensland")
        assert result["success"] is True

    async def test_no_results_returns_success_with_zero_count(self, service):
        result = await service.search_listings("10 bedroom mansions in Broken Hill under $100k")
        assert result["success"] is True
        assert result["result_count"] == 0


class TestFiltersLive:
    async def test_bedroom_filter(self, service):
        result = await service.search_listings("3 bedroom houses in Melbourne")
        assert result["success"] is True

    async def test_price_filter(self, service):
        result = await service.search_listings("Houses in Brisbane under $800k")
        assert result["success"] is True

    async def test_price_shorthand(self, service):
        result = await service.search_listings("Houses between $500k and $1.2m in Sydney")
        assert result["success"] is True

    async def test_rental_filter(self, service):
        result = await service.search_listings("Rental apartments in Parramatta under $600 per week")
        assert result["success"] is True

    async def test_combined_filters(self, service):
        result = await service.search_listings(
            "Show me 3 bedroom townhouses in Parramatta NSW for sale under $900k"
        )
        assert result["success"] is True
