"""Тесты чистых хелперов из search.py (без реальных моделей — стабим через conftest)."""
from unittest.mock import MagicMock

import search


def test_is_ascii_english():
    assert search._is_ascii("blue shirt") is True


def test_is_ascii_russian():
    assert search._is_ascii("синяя рубашка") is False


def test_is_ascii_mixed():
    assert search._is_ascii("Nike кроссовки") is False


def test_is_ascii_empty():
    assert search._is_ascii("") is True


def test_build_filter_none_returns_none():
    assert search._build_filter(None) is None


def test_build_filter_all_empty_returns_none():
    assert search._build_filter({"color": None, "gender": "", "category": None}) is None


def test_build_filter_one_field():
    f = search._build_filter({"color": "Blue", "gender": None})
    assert f is not None
    assert len(f.must) == 1
    assert f.must[0].key == "color"
    assert f.must[0].match.value == "Blue"


def test_build_filter_multiple_fields():
    f = search._build_filter({"color": "Red", "gender": "Men", "category": "Shirts"})
    assert len(f.must) == 3
    keys = {c.key for c in f.must}
    assert keys == {"color", "gender", "category"}


def test_point_to_dict_rounds_score():
    point = MagicMock()
    point.score = 0.876543
    point.payload = {
        "title": "Blue Shirt",
        "category": "Shirts",
        "gender": "Men",
        "color": "Blue",
        "image_path": "./images/1.jpg",
    }
    d = search._point_to_dict(point)
    assert d["score"] == 0.877
    assert d["title"] == "Blue Shirt"
    assert d["image_path"] == "./images/1.jpg"


def test_multi_query_search_empty_returns_empty():
    assert search.multi_query_search([]) == []
