"""Redis-кеш для LLM-вызовов и rate limiting с fail-open поведением.

Если REDIS_URL не задан, или Redis недоступен — все операции no-op'ятся:
get_json возвращает None, set_json молча проглатывает, rate-limit пропускает.
Это значит, что приложение работает без Redis, просто без кеша и без лимитов.
"""
import hashlib
import json
import os
import time
from typing import Any

from observability import cache_hits, cache_misses, log

_client = None
_init_attempted = False


def _get_client():
    """Lazy singleton клиента Redis. При ошибке коннекта запоминаем и больше не пробуем."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis
        c = redis.from_url(url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        c.ping()
        _client = c
        log.info("redis_connected", url=url)
        return _client
    except Exception as e:
        log.warning("redis_unavailable", error=str(e))
        return None


def cache_key(namespace: str, *parts: Any) -> str:
    """Детерминистичный ключ из namespace + произвольных аргументов."""
    raw = "|".join(str(p) for p in parts)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"frag:{namespace}:{h}"


def get_json(namespace: str, key: str) -> Any | None:
    """Прочитать JSON из кеша. None если не найдено / Redis недоступен."""
    c = _get_client()
    if c is None:
        return None
    try:
        raw = c.get(key)
        if raw is None:
            cache_misses.labels(namespace=namespace).inc()
            return None
        cache_hits.labels(namespace=namespace).inc()
        return json.loads(raw)
    except Exception as e:
        log.warning("cache_get_failed", error=str(e), key=key)
        return None


def set_json(namespace: str, key: str, value: Any, ttl_seconds: int = 86400) -> None:
    """Записать JSON-сериализуемое значение в кеш. Молча проглатывает ошибки."""
    c = _get_client()
    if c is None:
        return
    try:
        c.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))
    except Exception as e:
        log.warning("cache_set_failed", error=str(e), key=key)


def check_rate_limit(
    namespace: str, identifier: str, limit: int, window_seconds: int = 60,
) -> tuple[bool, int]:
    """Fixed-window счётчик через Redis INCR + EXPIRE.

    Возвращает (allowed, retry_after_seconds). При недоступности Redis — fail-open:
    разрешаем запрос, retry_after=0. На время outage пользователи не блокируются.
    """
    c = _get_client()
    if c is None:
        return True, 0
    try:
        now = int(time.time())
        bucket = now // window_seconds
        key = f"ratelimit:{namespace}:{identifier}:{bucket}"
        count = c.incr(key)
        if count == 1:
            # Только на первом инкременте ставим TTL — экономим RTT
            c.expire(key, window_seconds + 1)
        if count > limit:
            retry_after = window_seconds - (now % window_seconds)
            return False, max(retry_after, 1)
        return True, 0
    except Exception as e:
        log.warning("rate_limit_check_failed", error=str(e))
        return True, 0


def reset_for_tests() -> None:
    """Сброс singleton — используется в тестах, не для прода."""
    global _client, _init_attempted
    _client = None
    _init_attempted = False
