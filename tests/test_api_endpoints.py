"""Smoke-тесты HTTP-эндпоинтов через FastAPI TestClient. Qdrant, LLM и поиск замоканы."""
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

import api


def _scroll_with(points: list) -> tuple:
    return points, None


def _fake_point(color: str, gender: str, category: str) -> MagicMock:
    p = MagicMock()
    p.payload = {"color": color, "gender": gender, "category": category}
    return p


def test_health_returns_ok():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])):
        with TestClient(api.app) as client:
            r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_filters_endpoint_returns_unique_sorted_values():
    pts = [
        _fake_point("Blue", "Men", "Shirts"),
        _fake_point("Red", "Women", "Dresses"),
        _fake_point("Blue", "Men", "Shirts"),
    ]
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with(pts)):
        with TestClient(api.app) as client:
            r = client.get("/filters")
    assert r.status_code == 200
    body = r.json()
    assert body["colors"] == ["Blue", "Red"]
    assert body["genders"] == ["Men", "Women"]
    assert body["categories"] == ["Dresses", "Shirts"]


def test_request_id_header_set_by_middleware():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])):
        with TestClient(api.app) as client:
            r = client.get("/health")
    assert "x-request-id" in {k.lower() for k in r.headers.keys()}


def test_search_off_topic_blocked_by_guardrail():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])), \
         patch("api.is_fashion_query", return_value=(False, "не про одежду")):
        with TestClient(api.app) as client:
            r = client.post("/search", json={
                "query": "как варить борщ",
                "guardrail": True,
                "with_llm": False,
                "reformulate": False,
            })
    assert r.status_code == 200
    body = r.json()
    assert body["products"] == []
    assert body["recommendation"]["pick"] is None
    assert "одежд" in body["recommendation"]["reason"].lower()


def test_search_returns_products_when_guardrail_disabled():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])), \
         patch("api.search", return_value=[{"score": 0.9, "title": "Shirt", "category": "Shirts",
                                             "gender": "Men", "color": "Blue", "image_path": None}]):
        with TestClient(api.app) as client:
            r = client.post("/search", json={
                "query": "blue shirt",
                "guardrail": False,
                "with_llm": False,
                "reformulate": False,
            })
    assert r.status_code == 200
    body = r.json()
    assert len(body["products"]) == 1
    assert body["products"][0]["title"] == "Shirt"


def test_rate_limit_returns_429_with_retry_after():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])), \
         patch("api.check_rate_limit", return_value=(False, 42)):
        with TestClient(api.app) as client:
            r = client.post("/search", json={"query": "x", "guardrail": False, "with_llm": False,
                                              "reformulate": False})
    assert r.status_code == 429
    assert r.headers["Retry-After"] == "42"
    assert "x-request-id" in {k.lower() for k in r.headers.keys()}


def test_rate_limit_not_applied_to_health():
    """Лимит должен срабатывать только на /search и /search/image, не на health/filters."""
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])), \
         patch("api.check_rate_limit", return_value=(False, 60)) as mock_rl:
        with TestClient(api.app) as client:
            r = client.get("/health")
    assert r.status_code == 200
    mock_rl.assert_not_called()


def test_chat_graph_endpoint_returns_mermaid():
    with patch.object(api.qdrant_client, "scroll", return_value=_scroll_with([])):
        with TestClient(api.app) as client:
            r = client.get("/chat/graph")
    assert r.status_code == 200
    body = r.json()
    assert "mermaid" in body
    src = body["mermaid"]
    # Ноды LangGraph должны фигурировать в сгенерированном mermaid-исходнике
    assert "analyze" in src
    assert "search_and_respond" in src
    assert "decline" in src
