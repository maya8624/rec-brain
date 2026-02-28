# tests/test_routes.py
from unittest.mock import AsyncMock, patch


def test_ask_endpoint(client, mock_db):
    """Test the /api/ai/ask endpoint returns a 200 with sql_db source."""
    mock_db.run.return_value = "[(1, 'Sample Property')]"

    with patch("src.application.chat_service.search_similar_listings", new=AsyncMock(return_value=[])), \
            patch("src.application.chat_service.db", mock_db), \
            patch("src.application.chat_service.sql_chain") as mock_chain:

        # sql_chain.ainvoke is awaited, so it must be an AsyncMock
        mock_chain.ainvoke = AsyncMock(return_value="SELECT * FROM listings")

        response = client.post(
            "/api/ai/ask",
            json={"session_id": "test_1", "question": "Get all properties"},
        )

    assert response.status_code == 200
    assert response.json()["source"] == "sql_db"
