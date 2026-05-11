from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from atom.entrypoints.openai import api_server


class FakeColbertService:
    model_id = "lfm2-colbert-350m"
    descriptor = SimpleNamespace(
        weights_path=Path("/home/local/ai/models/registry/LiquidAI/LFM2-ColBERT-350M")
    )

    def embed_texts(self, texts):
        embeddings = [
            [float(i + 1), float(i + 2), float(i + 3)] for i, _ in enumerate(texts)
        ]
        return embeddings, 17

    def rerank(self, query, documents, top_n=None):
        scored = [
            {"index": i, "score": float(len(documents) - i), "document": doc}
            for i, doc in enumerate(documents)
        ]
        if top_n is not None:
            scored = scored[:top_n]
        return scored, 29


def _configure_retrieval_mode(monkeypatch):
    service = FakeColbertService()
    monkeypatch.setattr(api_server, "retrieval_service", service)
    monkeypatch.setattr(api_server, "engine", None)
    monkeypatch.setattr(api_server, "tokenizer", None)
    monkeypatch.setattr(api_server, "model_name", service.model_id)
    monkeypatch.setattr(
        api_server,
        "allowed_model_names",
        {
            service.model_id,
            str(service.descriptor.weights_path),
            "lfm2-colbert-350m",
        },
    )
    monkeypatch.setattr(api_server, "default_chat_template_kwargs", {})
    return TestClient(api_server.app)


def test_embeddings_route_returns_openai_shape(monkeypatch):
    client = _configure_retrieval_mode(monkeypatch)

    resp = client.post(
        "/v1/embeddings",
        json={
            "model": "lfm2-colbert-350m",
            "input": ["alpha", "beta"],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert body["model"] == "lfm2-colbert-350m"
    assert len(body["data"]) == 2
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert body["usage"] == {"prompt_tokens": 17, "total_tokens": 17}


def test_rerank_route_returns_sorted_results(monkeypatch):
    client = _configure_retrieval_mode(monkeypatch)

    resp = client.post(
        "/v1/rerank",
        json={
            "model": "lfm2-colbert-350m",
            "query": "hello",
            "documents": ["first", "second", "third"],
            "top_k": 2,
            "rid": "req-123",
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "rerank"
    assert body["model"] == "lfm2-colbert-350m"
    assert body["id"] == "req-123"
    assert len(body["results"]) == 2
    assert body["results"][0]["score"] >= body["results"][1]["score"]
    assert body["results"][0]["document"] == "first"
    assert body["usage"] == {"prompt_tokens": 29, "total_tokens": 29}


def test_chat_route_is_rejected_for_retrieval_only_model(monkeypatch):
    client = _configure_retrieval_mode(monkeypatch)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "lfm2-colbert-350m",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert resp.status_code == 400
    assert "retrieval-only ColBERT model" in resp.json()["detail"]
