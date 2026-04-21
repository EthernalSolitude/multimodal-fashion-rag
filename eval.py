import random
from collections import defaultdict

from search import client, search

random.seed(42)

print("Загружаем индекс...")
all_points, _ = client.scroll(collection_name="fashion", limit=10000, with_payload=True)

groups = defaultdict(list)
title_to_id = {}
for p in all_points:
    key = (p.payload.get("color"), p.payload.get("gender"), p.payload.get("category"))
    if all(key):
        groups[key].append(p.id)
    title_to_id[p.payload.get("title")] = p.id

print(f"Товаров: {len(all_points)}, групп: {len(groups)}")

queries = random.sample(list(groups.keys()), min(100, len(groups)))


def evaluate(use_filters: bool, hybrid: bool, rerank: bool):
    p_at_5 = []
    mrr_scores = []

    for (color, gender, category) in queries:
        query_text = f"{color} {category} for {gender}"
        relevant_ids = set(groups[(color, gender, category)])

        filters = {"color": color, "gender": gender, "category": category} if use_filters else None
        results = search(query_text, top_k=10, filters=filters, rerank=rerank, hybrid=hybrid)

        hits_5 = sum(1 for r in results[:5] if title_to_id.get(r["title"]) in relevant_ids)
        p_at_5.append(hits_5 / 5)

        mrr = 0
        for rank, r in enumerate(results, 1):
            if title_to_id.get(r["title"]) in relevant_ids:
                mrr = 1 / rank
                break
        mrr_scores.append(mrr)

    return sum(p_at_5) / len(p_at_5), sum(mrr_scores) / len(mrr_scores)


configs = [
    ("Dense only",        dict(use_filters=False, hybrid=False, rerank=False)),
    ("Dense+Sparse RRF",  dict(use_filters=False, hybrid=True,  rerank=False)),
    ("RRF + filters",     dict(use_filters=True,  hybrid=True,  rerank=False)),
    ("RRF + filt + rerank",dict(use_filters=True,  hybrid=True,  rerank=True)),
]

results = []
for name, kwargs in configs:
    print(f"[{name}]...")
    p5, mrr = evaluate(**kwargs)
    results.append((name, p5, mrr))

print("\n=== Сравнение ===")
print(f"{'Конфиг':<22} {'P@5':<8} {'MRR@10':<8}")
for name, p5, mrr in results:
    print(f"{name:<22} {p5:<8.3f} {mrr:<8.3f}")
