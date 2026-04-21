"""Тесты хелперов api.py."""
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
