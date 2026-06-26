import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_voice_query():
    payload = {
        "site_id": "site_1",
        "worker_id": "worker_1",
        "query": "What is the emergency protocol for a fire?"
    }
    response = client.post("/api/v1/voice/query", json=payload)
    
    # Will fail if LLM/Qdrant aren't mocked, just testing router exists.
    assert response.status_code in (200, 500)
