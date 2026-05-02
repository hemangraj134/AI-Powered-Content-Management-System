"""
test_api.py – Integration tests for the MetaMinds FastAPI application.

Uses FastAPI's built-in TestClient (backed by httpx) to send real HTTP
requests through the full ASGI stack without starting a live server.

Heavy dependencies (SentenceTransformer, ChromaDB) are mocked via conftest.py
so these tests run without a GPU or a real vector database.
"""

import pytest
from fastapi.testclient import TestClient

# conftest.py has already patched sys.path and mocked heavy deps, so this
# import resolves to MetaMinds/main.py safely.
from main import app

client = TestClient(app)


def test_health_check():
    """
    GET / should return HTTP 200 with the expected status message.
    The gpu_available field will be False on CI runners (no GPU), which is fine.
    """
    response = client.get("/")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()
    assert data["status"] == "MetaMinds AI Server is running", (
        f"Unexpected status message: {data.get('status')}"
    )
    assert "gpu_available" in data, "Response is missing the 'gpu_available' field."


def test_search_returns_list():
    """
    POST /search/ should return HTTP 200 and a JSON list even when the
    vector database is empty (mocked to return no results).
    """
    payload = {"query": "artificial intelligence document management", "top_k": 3}
    response = client.post("/search/", json=payload)

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    assert isinstance(response.json(), list), (
        "Search endpoint should always return a JSON list."
    )
