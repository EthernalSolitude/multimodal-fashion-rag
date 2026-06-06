"""Production-grade eval harness для retrieval-pipeline.

Что делает:
- При первом запуске генерит фиксированный test-set (100 запросов из категориальных триплетов)
  и сохраняет в eval_data/test_queries.json — последующие запуски используют тот же набор
- Прогоняет N baseline-конфигов (sparse-only, dense-only, hybrid, hybrid+rerank)
- Считает P@5, R@10, NDCG@10, MRR@10 с bootstrap 95% CI
- Пишет markdown-таблицу в eval_data/results.md + JSON в eval_data/results.json

Запуск:
    python eval_full.py                # использует существующий test-set, иначе генерит
    python eval_full.py --regenerate   # пересоздать test-set

Требует живой Qdrant с проиндексированной коллекцией fashion.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from eval_metrics import BOOTSTRAP_N, bootstrap_ci, compute_metrics
from search import (
    RERANK_CANDIDATES,
    _encode_sparse,
    _get_reranker,
    _search_hybrid_rrf,
    client,
    text_model,
)

EVAL_DATA = Path("eval_data")
QUERIES_FILE = EVAL_DATA / "test_queries.json"
RESULTS_JSON = EVAL_DATA / "results.json"
RESULTS_MD = EVAL_DATA / "results.md"

NUM_QUERIES = 100
SEED = 42


# ---------- Test set ----------

def generate_test_set() -> dict:
    print("Generating frozen test set from Qdrant...")
    random.seed(SEED)
    all_points, _ = client.scroll(collection_name="fashion", limit=10000, with_payload=True)

    groups = defaultdict(list)
    for p in all_points:
        key = (p.payload.get("color"), p.payload.get("gender"), p.payload.get("category"))
        if all(key):
            groups[key].append(p.id)

    # Берём только группы где >= 5 товаров — иначе recall@10 малоинформативен
    eligible = sorted([k for k, ids in groups.items() if len(ids) >= 5])
    random.shuffle(eligible)
    selected = eligible[:NUM_QUERIES]

    queries = []
    for color, gender, category in selected:
        queries.append({
            "query_text": f"{color} {category} for {gender}",
            "color": color,
            "gender": gender,
            "category": category,
            "relevant_ids": sorted(groups[(color, gender, category)]),
        })

    return {"version": "1.0", "seed": SEED, "num_queries": len(queries), "queries": queries}


def load_or_generate() -> list[dict]:
    if QUERIES_FILE.exists():
        print(f"Loading frozen test set from {QUERIES_FILE}")
        return json.loads(QUERIES_FILE.read_text(encoding="utf-8"))["queries"]
    EVAL_DATA.mkdir(exist_ok=True)
    data = generate_test_set()
    QUERIES_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(data['queries'])} queries to {QUERIES_FILE}")
    return data["queries"]


# ---------- Search modes (низкоуровневые, без потери id) ----------

def _search_sparse(query_text: str, limit: int):
    svec = _encode_sparse(query_text)
    return client.query_points(
        collection_name="fashion",
        query=svec,
        using="sparse",
        limit=limit,
    ).points


def _search_dense(query_text: str, limit: int):
    vec = text_model.encode(query_text).tolist()
    return client.query_points(
        collection_name="fashion",
        query=vec,
        using="dense",
        limit=limit,
    ).points


def _rerank_keep_ids(query_text: str, points: list, top_k: int) -> list:
    if not points:
        return []
    model = _get_reranker()
    pairs = [
        [query_text,
         f"{p.payload.get('title','')} {p.payload.get('category','')} "
         f"{p.payload.get('color','')} {p.payload.get('gender','')}"]
        for p in points
    ]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, points), key=lambda x: x[0], reverse=True)[:top_k]
    return [p for _, p in ranked]


def retrieve_ids(query_text: str, mode: str, rerank: bool, top_k: int = 10) -> list[int]:
    """Унифицированный интерфейс: для любой конфигурации возвращает list of point IDs."""
    limit = RERANK_CANDIDATES if rerank else top_k
    if mode == "sparse":
        points = _search_sparse(query_text, limit)
    elif mode == "dense":
        points = _search_dense(query_text, limit)
    elif mode == "hybrid":
        points = _search_hybrid_rrf(query_text, limit, None)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if rerank:
        points = _rerank_keep_ids(query_text, points, top_k)
    return [p.id for p in points[:top_k]]


# ---------- Run baselines ----------

CONFIGS = [
    ("BM25 sparse only", {"mode": "sparse", "rerank": False}),
    ("Dense CLIP only",  {"mode": "dense",  "rerank": False}),
    ("Hybrid RRF",       {"mode": "hybrid", "rerank": False}),
    ("Hybrid + cross-encoder rerank", {"mode": "hybrid", "rerank": True}),
]


def run_eval() -> dict:
    queries = load_or_generate()
    print(f"Test set: {len(queries)} queries\n")
    results = {}
    for name, cfg in CONFIGS:
        print(f"[{name}] mode={cfg['mode']} rerank={cfg['rerank']}")
        per_query: dict[str, list[float]] = {"P@5": [], "R@10": [], "NDCG@10": [], "MRR@10": []}
        for q in queries:
            retrieved = retrieve_ids(q["query_text"], cfg["mode"], cfg["rerank"])
            m = compute_metrics(retrieved, set(q["relevant_ids"]))
            for k, v in m.items():
                per_query[k].append(v)
        results[name] = {metric: bootstrap_ci(vals) for metric, vals in per_query.items()}
        for metric, (mean, lo, hi) in results[name].items():
            print(f"  {metric:<9} = {mean:.3f}  [{lo:.3f}, {hi:.3f}]")
        print()
    return results


def format_markdown(results: dict) -> str:
    lines = [
        "# Eval results",
        "",
        f"Test set: **{NUM_QUERIES} categorical queries** from indexed catalog (frozen, seed={SEED}).",
        "Each query: `{Color} {Category} for {Gender}`. Ground truth: items matching all 3 facets.",
        f"95% CI via **bootstrap** (n={BOOTSTRAP_N}).",
        "",
        "| Config | P@5 | R@10 | NDCG@10 | MRR@10 |",
        "|---|---|---|---|---|",
    ]
    for name, metrics in results.items():
        row = f"| **{name}** |"
        for m in ["P@5", "R@10", "NDCG@10", "MRR@10"]:
            mean, lo, hi = metrics[m]
            row += f" {mean:.3f} [{lo:.3f}, {hi:.3f}] |"
        lines.append(row)
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regenerate", action="store_true",
                        help="Force regenerate frozen test set (overwrites existing)")
    args = parser.parse_args()

    if args.regenerate and QUERIES_FILE.exists():
        QUERIES_FILE.unlink()

    EVAL_DATA.mkdir(exist_ok=True)
    results = run_eval()

    serializable = {name: {m: list(v) for m, v in metrics.items()} for name, metrics in results.items()}
    RESULTS_JSON.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    md = format_markdown(results)
    RESULTS_MD.write_text(md, encoding="utf-8")

    print(f"Saved {RESULTS_JSON.name} and {RESULTS_MD.name}\n")
    print(md)


if __name__ == "__main__":
    main()
