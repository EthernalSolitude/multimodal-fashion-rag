"""Стабим тяжёлые ML-зависимости чтобы тесты не грузили CLIP/BM42/Qdrant при старте.
Также гарантируем что REDIS_URL пуст — кеш fail-open'ится, не пытается коннектиться."""
import os
import sys
from unittest.mock import MagicMock


def _install_stubs() -> None:
    # Чистим env до импорта config — иначе локальный .env подсунет настоящий REDIS_URL
    os.environ.pop("REDIS_URL", None)

    dummy_model = MagicMock()
    dummy_model.encode.return_value = [0.0] * 512
    dummy_model.predict.return_value = [0.5]

    st = MagicMock()
    st.SentenceTransformer = MagicMock(return_value=dummy_model)
    st.CrossEncoder = MagicMock(return_value=dummy_model)
    sys.modules.setdefault("sentence_transformers", st)

    fe = MagicMock()
    fe.SparseTextEmbedding = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("fastembed", fe)

    # qdrant_client.models оставляем настоящим (для _build_filter нужны Filter/FieldCondition),
    # а сам QdrantClient подменяем чтобы не коннектился к живому серверу при импорте search.py
    try:
        import qdrant_client as qc
        qc.QdrantClient = MagicMock(return_value=MagicMock())
    except ImportError:
        pass

    # После чистки env перечитаем настройки — иначе settings.redis_url помнит старое значение
    try:
        from config import reload_settings
        reload_settings()
    except ImportError:
        pass


_install_stubs()
