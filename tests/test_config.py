"""Тесты config.py: типы, дефолты, reload."""
import os

from config import Settings, reload_settings


def test_settings_defaults_load_without_error():
    s = Settings()
    assert s.llm_backend == "api"
    assert s.qdrant_url.startswith("http")
    assert s.rerank_candidates == 20
    assert s.rate_limit_per_minute == 30
    assert s.chat_session_ttl_seconds == 3600


def test_settings_reads_env_var():
    os.environ["RATE_LIMIT_PER_MINUTE"] = "99"
    try:
        s = Settings()
        assert s.rate_limit_per_minute == 99
    finally:
        os.environ.pop("RATE_LIMIT_PER_MINUTE", None)


def test_settings_validates_int_type():
    """pydantic должен ругнуться на нечисло в int-поле."""
    import pytest
    from pydantic import ValidationError

    os.environ["RATE_LIMIT_PER_MINUTE"] = "not-a-number"
    try:
        with pytest.raises(ValidationError):
            Settings()
    finally:
        os.environ.pop("RATE_LIMIT_PER_MINUTE", None)


def test_reload_settings_picks_up_env_changes():
    os.environ["REDIS_URL"] = "redis://changed:6379/0"
    try:
        s = reload_settings()
        assert s.redis_url == "redis://changed:6379/0"
    finally:
        os.environ.pop("REDIS_URL", None)
        reload_settings()
