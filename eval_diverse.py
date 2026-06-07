"""Diverse eval: LLM генерирует свободные запросы, LLM-judge оценивает релевантность.
Показывает эффект от query reformulation + rerank на нечётких запросах,
которых не ловит синтетический eval.py на категориальных тройках."""

import json
import time

from config import settings
from llm import _client, _model, reformulate_query
from search import multi_query_search, search

NUM_QUERIES = settings.eval_num_queries
TOP_K = 5


_GEN_PROMPT = f"""Сгенерируй {NUM_QUERIES} РАЗНОРОДНЫХ запросов к fashion-магазину на русском, какие пишут реальные покупатели.
Включи:
- бренды (Nike, Adidas, Levis, Puma и т.п.)
- свободные формулировки ("что-нибудь для зала", "подарок жене")
- описательные ("лёгкая летняя рубашка", "тёплая куртка для зимы")
- конкретика (цвет + вещь + пол: "синие джинсы для мужчин")
Верни строго JSON: {{"queries": ["...", "..."]}}"""


_JUDGE_PROMPT = """Ты оценщик релевантности в fashion-поиске.
Покупатель спросил: "{query}"
Найден товар: "{title}" (категория: {category}, цвет: {color}, пол: {gender})
Оцени релевантность 0/1:
- 1 = товар адекватно отвечает на запрос
- 0 = товар не подходит
Верни строго JSON: {{"relevant": 0 или 1, "reason": "..."}}"""


def generate_queries() -> list[str]:
    resp = _client().chat.completions.create(
        model=_model(),
        messages=[{"role": "user", "content": _GEN_PROMPT}],
        response_format={"type": "json_object"},
        max_tokens=1500,
        temperature=0.9,
    )
    data = json.loads(resp.choices[0].message.content)
    return [q for q in data.get("queries", []) if q][:NUM_QUERIES]


def judge(query: str, product: dict) -> int:
    try:
        resp = _client().chat.completions.create(
            model=_model(),
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(
                query=query,
                title=product.get("title"),
                category=product.get("category"),
                color=product.get("color"),
                gender=product.get("gender"),
            )}],
            response_format={"type": "json_object"},
            max_tokens=100,
            temperature=0.0,
        )
        data = json.loads(resp.choices[0].message.content)
        return 1 if data.get("relevant") in (1, True, "1") else 0
    except Exception:
        return 0


def evaluate(config_name: str, search_fn) -> tuple[float, float]:
    p_at_5 = []
    mrr_scores = []
    for i, q in enumerate(queries, 1):
        print(f"  {config_name}: [{i}/{len(queries)}] {q[:60]}")
        results = search_fn(q)
        if not results:
            p_at_5.append(0.0)
            mrr_scores.append(0.0)
            continue
        rels = [judge(q, r) for r in results[:TOP_K]]
        p_at_5.append(sum(rels) / TOP_K)
        mrr = 0.0
        for rank, rel in enumerate(rels, 1):
            if rel:
                mrr = 1 / rank
                break
        mrr_scores.append(mrr)
        time.sleep(0.1)
    return sum(p_at_5) / len(p_at_5), sum(mrr_scores) / len(mrr_scores)


print(f"Генерируем {NUM_QUERIES} разнородных запросов через LLM...")
queries = generate_queries()
print(f"Получено {len(queries)} запросов. Примеры:")
for q in queries[:5]:
    print(f"  - {q}")
print()

configs = [
    ("Dense only",         lambda q: search(q, top_k=TOP_K, rerank=False, hybrid=False)),
    ("Hybrid RRF",         lambda q: search(q, top_k=TOP_K, rerank=False, hybrid=True)),
    ("RRF + rerank",       lambda q: search(q, top_k=TOP_K, rerank=True, hybrid=True)),
    ("Multi-query + rerank", lambda q: multi_query_search(reformulate_query(q, 3), top_k=TOP_K, rerank=True, hybrid=True)),
]

results = []
for name, fn in configs:
    print(f"\n[{name}]")
    p5, mrr = evaluate(name, fn)
    results.append((name, p5, mrr))

print("\n=== Diverse eval (LLM-judge) ===")
print(f"{'Конфиг':<24} {'P@5':<8} {'MRR@5':<8}")
for name, p5, mrr in results:
    print(f"{name:<24} {p5:<8.3f} {mrr:<8.3f}")
