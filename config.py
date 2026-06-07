"""Типизированный конфиг через pydantic-settings.

Все настройки в одном месте, читаются из .env / env-vars при старте,
валидируются типами. Дефолты подобраны под локальный docker-compose стенд.

`settings` — это прокси-объект: атрибуты делегируются актуальному инстансу,
поэтому reload_settings() в тестах виден и из других модулей (которые
импортировали settings до перезагрузки).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # LLM
    llm_backend: str = "api"
    llm_api_base_url: str = "https://api.cerebras.ai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama3.1-8b"
    local_model_path: str = ""

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Retrieval
    rerank_candidates: int = 20
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    sparse_model_name: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # Redis (пусто = кеш и rate-limit отключены, всё работает в fail-open режиме)
    redis_url: str = ""

    # Rate limiting
    rate_limit_per_minute: int = 30

    # Chat
    chat_session_ttl_seconds: int = 3600

    # Eval
    eval_num_queries: int = 30


class _SettingsProxy:
    """Прокси к Settings: атрибуты лениво читаются с текущего инстанса.
    Нужен чтобы `from config import settings` в других модулях продолжал
    указывать на актуальный объект после reload_settings()."""

    def __init__(self) -> None:
        self._inner: Settings = Settings()

    def _reload(self) -> Settings:
        self._inner = Settings()
        return self._inner

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


settings = _SettingsProxy()


def reload_settings() -> Settings:
    """Перечитать настройки. Нужно после правки env-vars в тестах."""
    return settings._reload()
