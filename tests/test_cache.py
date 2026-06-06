"""Тесты cache.py: fail-open поведение и детерминистичность ключей."""
import os
from unittest.mock import MagicMock, patch

import cache


def setup_function():
    """Сброс singleton перед каждым тестом — иначе протекает между тестами."""
    cache.reset_for_tests()


def test_cache_key_deterministic():
    k1 = cache.cache_key("guardrail", "blue shirt")
    k2 = cache.cache_key("guardrail", "blue shirt")
    assert k1 == k2


def test_cache_key_distinguishes_namespaces():
    k1 = cache.cache_key("guardrail", "x")
    k2 = cache.cache_key("reformulate", "x")
    assert k1 != k2


def test_cache_key_distinguishes_args():
    k1 = cache.cache_key("ns", "a")
    k2 = cache.cache_key("ns", "b")
    assert k1 != k2


def test_cache_key_format_has_prefix():
    k = cache.cache_key("guardrail", "x")
    assert k.startswith("frag:guardrail:")


def test_get_json_returns_none_when_no_redis_url():
    os.environ.pop("REDIS_URL", None)
    assert cache.get_json("ns", "any-key") is None


def test_set_json_silent_when_no_redis_url():
    os.environ.pop("REDIS_URL", None)
    cache.set_json("ns", "any-key", {"foo": "bar"})


def test_get_json_returns_none_on_connection_failure():
    os.environ["REDIS_URL"] = "redis://nonexistent-host-xyz:6379/0"
    cache.reset_for_tests()
    try:
        assert cache.get_json("ns", "key") is None
    finally:
        os.environ.pop("REDIS_URL", None)
        cache.reset_for_tests()


def test_cache_roundtrip_with_mocked_redis():
    fake_storage = {}
    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_client.get.side_effect = lambda k: fake_storage.get(k)
    fake_client.setex.side_effect = lambda k, ttl, v: fake_storage.update({k: v})

    os.environ["REDIS_URL"] = "redis://fake:6379/0"
    cache.reset_for_tests()
    try:
        with patch("redis.from_url", return_value=fake_client):
            cache.set_json("ns", "k1", {"hello": "world"})
            assert cache.get_json("ns", "k1") == {"hello": "world"}
            assert cache.get_json("ns", "missing") is None
    finally:
        os.environ.pop("REDIS_URL", None)
        cache.reset_for_tests()


def _fake_redis_for_ratelimit():
    """Fake Redis-клиент с поведением INCR/EXPIRE, нужным для check_rate_limit."""
    counters = {}
    client = MagicMock()
    client.ping.return_value = True

    def fake_incr(key):
        counters[key] = counters.get(key, 0) + 1
        return counters[key]

    client.incr.side_effect = fake_incr
    client.expire.return_value = True
    return client, counters


def test_rate_limit_allows_below_threshold():
    os.environ["REDIS_URL"] = "redis://fake:6379/0"
    cache.reset_for_tests()
    fake_client, _ = _fake_redis_for_ratelimit()
    try:
        with patch("redis.from_url", return_value=fake_client):
            for _ in range(3):
                allowed, retry = cache.check_rate_limit("test", "ip1", limit=5, window_seconds=60)
                assert allowed is True
                assert retry == 0
    finally:
        os.environ.pop("REDIS_URL", None)
        cache.reset_for_tests()


def test_rate_limit_rejects_above_threshold():
    os.environ["REDIS_URL"] = "redis://fake:6379/0"
    cache.reset_for_tests()
    fake_client, _ = _fake_redis_for_ratelimit()
    try:
        with patch("redis.from_url", return_value=fake_client):
            for _ in range(3):
                assert cache.check_rate_limit("t", "ip", limit=3, window_seconds=60)[0] is True
            allowed, retry = cache.check_rate_limit("t", "ip", limit=3, window_seconds=60)
            assert allowed is False
            assert 1 <= retry <= 60
    finally:
        os.environ.pop("REDIS_URL", None)
        cache.reset_for_tests()


def test_rate_limit_fail_open_without_redis():
    os.environ.pop("REDIS_URL", None)
    cache.reset_for_tests()
    for _ in range(100):
        allowed, _ = cache.check_rate_limit("t", "ip", limit=3, window_seconds=60)
        assert allowed is True


def test_rate_limit_isolates_by_identifier():
    os.environ["REDIS_URL"] = "redis://fake:6379/0"
    cache.reset_for_tests()
    fake_client, _ = _fake_redis_for_ratelimit()
    try:
        with patch("redis.from_url", return_value=fake_client):
            for _ in range(2):
                cache.check_rate_limit("t", "ip1", limit=2, window_seconds=60)
            # ip1 на пределе, ip2 ещё свободен
            assert cache.check_rate_limit("t", "ip1", limit=2, window_seconds=60)[0] is False
            assert cache.check_rate_limit("t", "ip2", limit=2, window_seconds=60)[0] is True
    finally:
        os.environ.pop("REDIS_URL", None)
        cache.reset_for_tests()
