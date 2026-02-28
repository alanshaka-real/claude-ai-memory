import httpx
import pytest

EMBEDDING_URL = "http://localhost:8100"

@pytest.mark.integration
def test_health():
    r = httpx.get(f"{EMBEDDING_URL}/health")
    assert r.status_code == 200

@pytest.mark.integration
def test_single_embedding():
    r = httpx.post(f"{EMBEDDING_URL}/v1/embeddings", json={
        "input": "hello world",
        "model": "all-MiniLM-L6-v2"
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 1
    assert len(data["data"][0]["embedding"]) == 384

@pytest.mark.integration
def test_batch_embedding():
    r = httpx.post(f"{EMBEDDING_URL}/v1/embeddings", json={
        "input": ["hello", "world"],
        "model": "all-MiniLM-L6-v2"
    })
    assert r.status_code == 200
    data = r.json()
    assert len(data["data"]) == 2
