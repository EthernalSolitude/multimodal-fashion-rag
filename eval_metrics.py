"""Чистые метрики ранжирования + bootstrap CI. Без зависимости от Qdrant — легко тестировать."""
import math
import random

BOOTSTRAP_N = 1000
DEFAULT_SEED = 42


def compute_metrics(retrieved_ids: list[int], relevant_ids: set[int]) -> dict[str, float]:
    """Считает P@5, R@10, NDCG@10, MRR@10 для одного запроса."""
    rels_top10 = [1 if rid in relevant_ids else 0 for rid in retrieved_ids[:10]]

    p5 = sum(rels_top10[:5]) / 5.0

    # R@10 нормирован на min(|relevant|, 10) — иначе при relevant > 10 рецилл искусственно занижен
    denom = min(len(relevant_ids), 10) or 1
    r10 = sum(rels_top10) / denom

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(rels_top10))
    ideal_n = min(len(relevant_ids), 10)
    idcg = sum(1 / math.log2(i + 2) for i in range(ideal_n))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    mrr = 0.0
    for i, rel in enumerate(rels_top10):
        if rel:
            mrr = 1.0 / (i + 1)
            break

    return {"P@5": p5, "R@10": r10, "NDCG@10": ndcg, "MRR@10": mrr}


def bootstrap_ci(
    values: list[float], n: int = BOOTSTRAP_N, alpha: float = 0.05, seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Bootstrap 95% доверительного интервала для среднего. Возвращает (mean, lo, hi)."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    k = len(values)
    means = []
    for _ in range(n):
        sample_sum = 0.0
        for _ in range(k):
            sample_sum += values[rng.randrange(k)]
        means.append(sample_sum / k)
    means.sort()
    mean = sum(values) / k
    lo = means[int(n * alpha / 2)]
    hi = means[int(n * (1 - alpha / 2))]
    return mean, lo, hi
