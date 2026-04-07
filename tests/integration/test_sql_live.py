"""
Live integration tests for SqlViewService.
Hits a real Groq LLM and real PostgreSQL — requires a configured .env.
"""
import pytest
from tests.integration.conftest import skip_if_no_env

from app.services.sql_service import SqlViewService
from app.infrastructure.llm import get_llm

pytestmark = [pytest.mark.integration, skip_if_no_env]


@pytest.fixture(scope="module")
def service():
    return SqlViewService(llm=get_llm())


class TestSearchLive:
    async def test_basic_suburb_search(self, service):
        result = await service.search_listings("Show me properties in Sydney")
        assert result["success"] is True

    async def test_state_full_name(self, service):
        result = await service.search_listings("Show me properties in New South Wales")
        assert result["success"] is True

    async def test_no_results_returns_success_with_zero_count(self, service):
        result = await service.search_listings("10 bedroom mansions in Broken Hill under $100k")
        assert result["success"] is True
        assert result["result_count"] == 0


class TestFiltersLive:
    async def test_bedroom_filter(self, service):
        result = await service.search_listings("3 bedroom houses in Sydney")
        assert result["success"] is True

    async def test_price_filter(self, service):
        result = await service.search_listings("Houses in Brisbane under $800k")
        assert result["success"] is True

    async def test_price_shorthand(self, service):
        result = await service.search_listings("Houses between $500k and $1.2m in Sydney")
        assert result["success"] is True

    async def test_rental_filter(self, service):
        result = await service.search_listings("Rental apartments in Parramatta under $1000 per week")
        assert result["success"] is True

    async def test_combined_filters(self, service):
        result = await service.search_listings(
            "Show me 3 bedroom townhouses in Bondi NSW for sale under $900k"
        )
        assert result["success"] is True
