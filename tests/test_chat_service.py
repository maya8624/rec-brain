# tests/test_chat_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.application import chat_service


@pytest.mark.asyncio
async def test_handle_user_query_sql_fallback(mock_db, mock_llm, mock_vector_db):
    """Test that the service falls back to SQL if vector search returns no results."""

    with patch("src.application.chat_service.db", mock_db), \
            patch("src.application.chat_service.sql_chain") as mock_chain, \
            patch("src.application.chat_service.search_similar_listings", mock_vector_db):

        # sql_chain.ainvoke is awaited in chat_service, so it must be an AsyncMock
        mock_chain.ainvoke = AsyncMock(return_value="SELECT * FROM listings")

        result = await chat_service.handle_user_query("What is my name?")

        assert result["source"] == "sql_db"
        assert result["answer"] == "[('result',)]"
        mock_db.run.assert_called()


@pytest.mark.asyncio
async def test_handle_user_query_vector_hit(mock_db, mock_vector_db):
    """Test that the service returns vector results if similarity score is below threshold."""

    mock_doc = MagicMock()
    mock_doc.page_content = "Vector Answer"
    # Score 0.1 is well below the 0.4 threshold → vector hit
    mock_vector_db.return_value = [(mock_doc, 0.1)]

    with patch("src.application.chat_service.db", mock_db), \
            patch("src.application.chat_service.search_similar_listings", mock_vector_db):

        result = await chat_service.handle_user_query("Find me a cozy apartment")

        assert result["source"] == "vector_db"
        assert result["answer"] == "Vector Answer"
        # SQL should NOT have been called
        mock_db.run.assert_not_called()
