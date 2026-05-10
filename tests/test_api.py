"""Тесты хелперов и pydantic-моделей api.py."""
import pytest
from pydantic import ValidationError

import api


def test_clean_removes_swagger_placeholder():
    assert api._clean("string") is None
    assert api._clean("  string ") is None
    assert api._clean("STRING") is None


def test_clean_preserves_real_values():
    assert api._clean("Blue") == "Blue"
    assert api._clean("Men") == "Men"


def test_clean_handles_none_and_empty():
    assert api._clean(None) is None
    assert api._clean("") is None


def test_search_request_defaults():
    req = api.SearchRequest(query="blue shirt")
    assert req.top_k == 5
    assert req.with_llm is True
    assert req.guardrail is True
    assert req.rerank is True
    assert req.reformulate is True
    assert req.color is None


def test_search_request_overrides():
    req = api.SearchRequest(query="x", top_k=10, color="Red", guardrail=False, with_llm=False)
    assert req.top_k == 10
    assert req.color == "Red"
    assert req.guardrail is False
    assert req.with_llm is False


def test_search_request_requires_query():
    with pytest.raises(ValidationError):
        api.SearchRequest()  # type: ignore[call-arg]


def test_build_results_handles_empty():
    assert api._build_results([]) == []


def test_build_results_constructs_image_url():
    products = [{
        "score": 0.9, "title": "T", "category": "C", "gender": "M", "color": "B",
        "image_path": "./images/abc.jpg",
    }]
    out = api._build_results(products)
    assert out[0].image_url == "http://localhost:8000/images/abc.jpg"
