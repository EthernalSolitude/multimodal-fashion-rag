import math
import os

from fastembed import SparseTextEmbedding
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import settings
from observability import timed

# Аббревиатуры для backward-compat с местами где импортируют как константы
RERANK_CANDIDATES = settings.rerank_candidates

# Все тяжёлые клиенты/модели грузятся лениво: при `import search` ничего не
# коннектится и не качается. Это позволяет запускать тесты и линтер без живых
# зависимостей и не падать при `pytest --version` от битой модели.
_client = None
_text_model = None
_sparse_model = None
_image_model = None
_reranker = None


def _get_qdrant_client():
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url)
    return _client


def _get_text_model():
    global _text_model
    if _text_model is None:
        local = './models/clip-multilingual'
        _text_model = SentenceTransformer(local if os.path.exists(local) else 'sentence-transformers/clip-ViT-B-32-multilingual-v1')
    return _text_model


def _get_sparse_model():
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(settings.sparse_model_name)
    return _sparse_model


def _get_image_model():
    global _image_model
    if _image_model is None:
        path = './models/clip-vit-b32'
        _image_model = SentenceTransformer(path if os.path.exists(path) else 'clip-ViT-B-32')
    return _image_model


def _get_reranker():
    global _reranker
    if _reranker is None:
        local_path = './models/reranker'
        _reranker = CrossEncoder(local_path if os.path.exists(local_path) else settings.reranker_model)
    return _reranker


def __getattr__(name: str):
    """Backward-compat для импортов `from search import client/text_model/sparse_model`.
    Триггерит ленивую инициализацию при первом обращении."""
    if name == "client":
        return _get_qdrant_client()
    if name == "text_model":
        return _get_text_model()
    if name == "sparse_model":
        return _get_sparse_model()
    raise AttributeError(f"module 'search' has no attribute {name!r}")


def _build_filter(filters: dict | None) -> Filter | None:
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items() if v
    ]
    return Filter(must=conditions) if conditions else None


def _point_to_dict(point) -> dict:
    return {
        "score": round(point.score, 3),
        "title": point.payload.get("title"),
        "category": point.payload.get("category"),
        "gender": point.payload.get("gender"),
        "color": point.payload.get("color"),
        "image_path": point.payload.get("image_path"),
    }


def _encode_sparse(query: str) -> SparseVector:
    s = next(iter(_get_sparse_model().embed([query])))
    return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def _search_dense_only(vector: list, top_k: int, filters: dict | None, using: str = "dense"):
    with timed(f"dense_{using}"):
        return _get_qdrant_client().query_points(
            collection_name="fashion",
            query=vector,
            using=using,
            query_filter=_build_filter(filters),
            limit=top_k,
        ).points


def _search_hybrid_rrf(query: str, top_k: int, filters: dict | None):
    with timed("hybrid_rrf"):
        dense_vec = _get_text_model().encode(query).tolist()
        sparse_vec = _encode_sparse(query)
        qfilter = _build_filter(filters)
        return _get_qdrant_client().query_points(
            collection_name="fashion",
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=RERANK_CANDIDATES, filter=qfilter),
                Prefetch(query=sparse_vec, using="sparse", limit=RERANK_CANDIDATES, filter=qfilter),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
        ).points


def _rerank(query: str, points: list, top_k: int) -> list[dict]:
    if not points:
        return []
    with timed("rerank"):
        model = _get_reranker()
        pairs = [
            [query, f"{p.payload.get('title', '')} {p.payload.get('category', '')} {p.payload.get('color', '')} {p.payload.get('gender', '')}"]
            for p in points
        ]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)[:top_k]
        out = []
        for score, p in ranked:
            d = _point_to_dict(p)
            d["score"] = round(1 / (1 + math.exp(-float(score))), 3)
            out.append(d)
        return out


def search(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    rerank: bool = False,
    hybrid: bool = True,
) -> list[dict]:
    limit = RERANK_CANDIDATES if rerank else top_k
    if hybrid:
        points = _search_hybrid_rrf(query, limit, filters)
    else:
        vector = _get_text_model().encode(query).tolist()
        points = _search_dense_only(vector, limit, filters)

    if rerank:
        return _rerank(query, points, top_k)
    return [_point_to_dict(p) for p in points]


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def multi_query_search(
    queries: list[str],
    top_k: int = 5,
    filters: dict | None = None,
    rerank: bool = True,
    hybrid: bool = True,
    per_query_limit: int | None = None,
) -> list[dict]:
    """Последовательно ищем по каждому подзапросу, дедуплицируем по id,
    опционально переранжируем cross-encoder'ом."""
    if not queries:
        return []
    limit = per_query_limit or RERANK_CANDIDATES
    seen: dict[int, object] = {}
    for q in queries:
        pts = _search_hybrid_rrf(q, limit, filters) if hybrid else _search_dense_only(_get_text_model().encode(q).tolist(), limit, filters)
        for p in pts:
            if p.id not in seen or p.score > seen[p.id].score:
                seen[p.id] = p
    union = list(seen.values())
    if rerank:
        rerank_q = next((q for q in queries if _is_ascii(q)), queries[0])
        return _rerank(rerank_q, union, top_k)
    union.sort(key=lambda p: p.score, reverse=True)
    return [_point_to_dict(p) for p in union[:top_k]]


def search_by_image(image: Image.Image, top_k: int = 5, filters: dict | None = None) -> list[dict]:
    vector = _get_image_model().encode(image).tolist()
    points = _search_dense_only(vector, top_k, filters)
    return [_point_to_dict(p) for p in points]


if __name__ == "__main__":
    query = "red winter jacket for men"
    for item in search(query, rerank=True, hybrid=True):
        print(item)
