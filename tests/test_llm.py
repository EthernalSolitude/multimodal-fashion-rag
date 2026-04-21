"""Тесты llm.py: guardrail, reformulate, generate — все с моками OpenAI."""
import json
from unittest.mock import MagicMock, patch

import llm


def _fake_response(payload: dict) -> MagicMock:
    """Эмулирует OpenAI chat.completions.create response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(payload)
    return resp


def _fake_client(payload: dict) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = _fake_response(payload)
    return client


def setup_function():
    """Сбрасываем lru_cache между тестами — иначе is_fashion_query отдаёт старые значения."""
    llm.is_fashion_query.cache_clear()


def test_is_fashion_query_accepts_clothing():
    with patch.object(llm, "_client", return_value=_fake_client({"is_fashion": True, "reason": "одежда"})):
        ok, reason = llm.is_fashion_query("синяя рубашка")
    assert ok is True
    assert reason == "одежда"


def test_is_fashion_query_rejects_off_topic():
    with patch.object(llm, "_client", return_value=_fake_client({"is_fashion": False, "reason": "про погоду"})):
        ok, reason = llm.is_fashion_query("какая сегодня погода")
    assert ok is False
    assert "погод" in reason.lower()


def test_is_fashion_query_empty_string():
    ok, reason = llm.is_fashion_query("   ")
    assert ok is False
    assert "Пуст" in reason


def test_is_fashion_query_cached():
    mock_client = _fake_client({"is_fashion": True, "reason": "x"})
    with patch.object(llm, "_client", return_value=mock_client):
        llm.is_fashion_query("кроссовки Nike")
        llm.is_fashion_query("кроссовки Nike")
    assert mock_client.chat.completions.create.call_count == 1


def test_is_fashion_query_fail_open_on_exception():
    failing = MagicMock()
    failing.chat.completions.create.side_effect = RuntimeError("api down")
    with patch.object(llm, "_client", return_value=failing):
        ok, reason = llm.is_fashion_query("что угодно")
    assert ok is True
    assert "недоступен" in reason


def test_reformulate_query_prepends_original():
    with patch.object(llm, "_client", return_value=_fake_client({"queries": ["blue shirt men", "navy top"]})):
        out = llm.reformulate_query("синяя рубашка", n=2)
    assert out[0] == "синяя рубашка"
    assert "blue shirt men" in out
    assert len(out) == 3


def test_reformulate_query_returns_original_on_error():
    failing = MagicMock()
    failing.chat.completions.create.side_effect = RuntimeError("boom")
    with patch.object(llm, "_client", return_value=failing):
        out = llm.reformulate_query("платье", n=3)
    assert out == ["платье"]


def test_reformulate_query_empty_input():
    assert llm.reformulate_query("") == [""]


def test_generate_empty_products_returns_refusal():
    result = llm.generate("синяя рубашка", [])
    assert result["pick"] is None
    assert result["suggestions"]
    assert "ничего не нашлось" in result["reason"].lower()


def test_format_products_includes_all_fields():
    products = [{"title": "Blue Shirt", "category": "Shirts", "color": "Blue", "gender": "Men"}]
    text = llm._format_products("синяя рубашка", products)
    assert "Blue Shirt" in text
    assert "Shirts" in text
    assert "синяя рубашка" in text
