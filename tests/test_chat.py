"""Тесты chat.py: ноды LangGraph, роутинг, session persistence."""
import json
from unittest.mock import MagicMock, patch

import chat


def setup_function():
    import cache
    cache.reset_for_tests()
    chat.reset_graph_for_tests()


def _fake_llm_resp(payload: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


def _fake_client(payload: dict) -> MagicMock:
    c = MagicMock()
    c.chat.completions.create.return_value = _fake_llm_resp(payload)
    return c


# ---------- Узлы ----------

def test_analyze_node_extracts_search_intent():
    payload = {"intent": "search", "query": "blue shirt men", "color": "Blue", "gender": "Men", "category": "Shirts"}
    with patch.object(chat, "_client", return_value=_fake_client(payload)):
        out = chat.analyze_node({"messages": [{"role": "user", "content": "синяя рубашка для мужчин"}]})
    assert out["intent"] == "search"
    assert out["search_query"] == "blue shirt men"
    assert out["filters"]["color"] == "Blue"


def test_analyze_node_handles_off_topic():
    payload = {"intent": "off_topic", "query": "", "color": None, "gender": None, "category": None}
    with patch.object(chat, "_client", return_value=_fake_client(payload)):
        out = chat.analyze_node({"messages": [{"role": "user", "content": "как варить борщ"}]})
    assert out["intent"] == "off_topic"


def test_analyze_node_fallback_on_llm_error():
    failing = MagicMock()
    failing.chat.completions.create.side_effect = RuntimeError("api down")
    with patch.object(chat, "_client", return_value=failing):
        out = chat.analyze_node({"messages": [{"role": "user", "content": "что-то"}]})
    # При ошибке LLM не блокируем юзера — пробуем поиск с исходным запросом
    assert out["intent"] == "search"
    assert out["search_query"] == "что-то"


def test_decline_node_returns_polite_refusal():
    out = chat.decline_node({"messages": []})
    assert out["products"] == []
    assert out["response"]["pick"] is None
    assert len(out["response"]["suggestions"]) == 3


def test_search_and_respond_node_calls_pipeline():
    fake_products = [{"score": 0.9, "title": "Shirt", "category": "Shirts",
                      "gender": "Men", "color": "Blue", "image_path": None}]
    fake_recommendation = {"pick": "Shirt", "reason": "good", "alternatives": "none", "suggestions": []}
    with patch.object(chat, "multi_query_search", return_value=fake_products), \
         patch.object(chat, "generate", return_value=fake_recommendation):
        out = chat.search_and_respond_node({"search_query": "blue shirt", "filters": {}})
    assert out["products"] == fake_products
    assert out["response"]["pick"] == "Shirt"


# ---------- Routing ----------

def test_route_search_to_search_node():
    assert chat._route_after_analyze({"intent": "search"}) == "search_and_respond"


def test_route_off_topic_to_decline():
    assert chat._route_after_analyze({"intent": "off_topic"}) == "decline"


# ---------- End-to-end через граф ----------

def test_run_chat_turn_search_flow_end_to_end():
    analyze_payload = {"intent": "search", "query": "blue shirt", "color": None, "gender": None, "category": None}
    fake_products = [{"score": 0.9, "title": "T", "category": "C", "gender": "G", "color": "X", "image_path": None}]
    rec = {"pick": "T", "reason": "good fit", "alternatives": "none", "suggestions": ["s1", "s2", "s3"]}

    with patch.object(chat, "_client", return_value=_fake_client(analyze_payload)), \
         patch.object(chat, "multi_query_search", return_value=fake_products), \
         patch.object(chat, "generate", return_value=rec):
        out = chat.run_chat_turn(None, "синяя рубашка")

    assert out["intent"] == "search"
    assert out["response"]["pick"] == "T"
    assert len(out["products"]) == 1
    assert out["session_id"]  # auto-generated


def test_run_chat_turn_off_topic_skips_search():
    payload = {"intent": "off_topic", "query": "", "color": None, "gender": None, "category": None}
    with patch.object(chat, "_client", return_value=_fake_client(payload)), \
         patch.object(chat, "multi_query_search") as mock_search:
        out = chat.run_chat_turn(None, "как погода")

    assert out["intent"] == "off_topic"
    assert out["response"]["pick"] is None
    mock_search.assert_not_called()


def test_run_chat_turn_preserves_session_id():
    payload = {"intent": "off_topic", "query": ""}
    with patch.object(chat, "_client", return_value=_fake_client(payload)):
        out = chat.run_chat_turn("my-session-123", "off topic")
    assert out["session_id"] == "my-session-123"


# ---------- Session persistence ----------

def test_load_session_returns_empty_without_redis():
    assert chat.load_session("any-id") == []


def test_save_session_silent_without_redis():
    chat.save_session("any-id", [{"role": "user", "content": "x"}])


def _enable_fake_redis(url: str = "redis://fake:6379/0") -> None:
    import os
    os.environ["REDIS_URL"] = url
    import cache as cache_mod
    from config import reload_settings
    reload_settings()
    cache_mod.reset_for_tests()


def _disable_fake_redis() -> None:
    import os
    os.environ.pop("REDIS_URL", None)
    import cache as cache_mod
    from config import reload_settings
    reload_settings()
    cache_mod.reset_for_tests()


def test_session_roundtrip_with_mocked_redis():
    fake_storage = {}
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_client.get.side_effect = lambda k: fake_storage.get(k)
    fake_client.setex.side_effect = lambda k, ttl, v: fake_storage.update({k: v})

    _enable_fake_redis()
    try:
        with patch("redis.from_url", return_value=fake_client):
            chat.save_session("sid1", [{"role": "user", "content": "hello"}])
            loaded = chat.load_session("sid1")
            assert loaded == [{"role": "user", "content": "hello"}]
    finally:
        _disable_fake_redis()


def test_session_history_truncated_to_max():
    messages = [{"role": "user", "content": f"msg{i}"} for i in range(20)]
    captured = {}
    fake_client = MagicMock()
    fake_client.ping.return_value = True

    def capture_setex(key, ttl, value):
        captured[key] = json.loads(value)

    fake_client.setex.side_effect = capture_setex

    _enable_fake_redis()
    try:
        with patch("redis.from_url", return_value=fake_client):
            chat.save_session("sid", messages)
        stored = next(iter(captured.values()))
        assert len(stored["messages"]) == chat.MAX_HISTORY_MESSAGES
        assert stored["messages"][0]["content"] == "msg10"  # последние 10
    finally:
        _disable_fake_redis()
