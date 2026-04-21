import math
import os

from dotenv import load_dotenv
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

from observability import timed

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SPARSE_MODEL_NAME = os.getenv("SPARSE_MODEL_NAME", "Qdrant/bm42-all-minilm-l6-v2-attentions")

client = QdrantClient(url=QDRANT_URL)
text_model = SentenceTransformer('./models/clip-multilingual')
sparse_model = SparseTextEmbedding(SPARSE_MODEL_NAME)
_image_model = None
_reranker = None


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
        _reranker = CrossEncoder(local_path if os.path.exists(local_path) else RERANKER_MODEL)
    return _reranker


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
    s = next(iter(sparse_model.embed([query])))
    return SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def _search_dense_only(vector: list, top_k: int, filters: dict | None, using: str = "dense"):
    with timed(f"dense_{using}"):
        return client.query_points(
            collection_name="fashion",
            query=vector,
            using=using,
            query_filter=_build_filter(filters),
            limit=top_k,
        ).points


def _search_hybrid_rrf(query: str, top_k: int, filters: dict | None):
    with timed("hybrid_rrf"):
        dense_vec = text_model.encode(query).tolist()
        sparse_vec = _encode_sparse(query)
        qfilter = _build_filter(filters)
        return client.query_points(
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
        vector = text_model.encode(query).tolist()
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
    """Fan-out по подзапросам, дедуп по id, опциональный rerank.
    Для rerank берём первый ASCII-запрос (cross-encoder англоязычный — MS-MARCO)."""
    if not queries:
        return []
    limit = per_query_limit or RERANK_CANDIDATES
    seen: dict[int, object] = {}
    for q in queries:
        pts = _search_hybrid_rrf(q, limit, filters) if hybrid else _search_dense_only(text_model.encode(q).tolist(), limit, filters)
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
