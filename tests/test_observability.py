"""Тесты observability: request_id, контекст, timed-обёртка."""
from observability import clear_context, new_request_id, timed


def test_new_request_id_returns_nonempty_string():
    rid = new_request_id()
    assert isinstance(rid, str)
    assert len(rid) > 0


def test_request_ids_are_unique():
    ids = {new_request_id() for _ in range(50)}
    assert len(ids) == 50


def test_clear_context_does_not_raise():
    new_request_id()
    clear_context()


def test_timed_yields_and_completes():
    with timed("unit_test_stage"):
        x = 1 + 1
    assert x == 2


def test_timed_propagates_exceptions():
    import pytest
    with pytest.raises(ValueError):
        with timed("unit_test_stage"):
            raise ValueError("expected")
