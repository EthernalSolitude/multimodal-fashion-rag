"""Тесты чистых retrieval-метрик и bootstrap CI из eval_metrics.py."""
import math

from eval_metrics import bootstrap_ci, compute_metrics

# ---------- compute_metrics ----------

def test_p5_perfect_top5():
    m = compute_metrics([1, 2, 3, 4, 5], {1, 2, 3, 4, 5})
    assert m["P@5"] == 1.0


def test_p5_no_hits():
    m = compute_metrics([10, 20, 30, 40, 50], {99, 100})
    assert m["P@5"] == 0.0


def test_p5_partial_hits():
    m = compute_metrics([1, 99, 2, 99, 3], {1, 2, 3})
    assert m["P@5"] == 0.6


def test_mrr_first_position():
    m = compute_metrics([10, 20, 30], {10})
    assert m["MRR@10"] == 1.0


def test_mrr_third_position():
    m = compute_metrics([10, 20, 30], {30})
    assert abs(m["MRR@10"] - 1 / 3) < 1e-9


def test_mrr_no_hit():
    m = compute_metrics([10, 20, 30], {99})
    assert m["MRR@10"] == 0.0


def test_recall_capped_at_10():
    # 2 hits in top10, 5 relevant total → r = 2/5
    m = compute_metrics([1, 99, 2] + [99] * 7, {1, 2, 3, 4, 5})
    assert abs(m["R@10"] - 0.4) < 1e-9


def test_recall_when_few_relevant():
    # 1 hit, 1 relevant → r = 1.0
    m = compute_metrics([1, 99, 99], {1})
    assert m["R@10"] == 1.0


def test_ndcg_perfect_order():
    m = compute_metrics([1, 2, 3], {1, 2, 3})
    assert abs(m["NDCG@10"] - 1.0) < 1e-9


def test_ndcg_reversed_order_less_than_perfect():
    perfect = compute_metrics([1, 2, 3], {1, 2, 3})["NDCG@10"]
    bad = compute_metrics([99, 99, 1, 2, 3], {1, 2, 3})["NDCG@10"]
    assert bad < perfect


def test_ndcg_zero_when_no_relevant_retrieved():
    m = compute_metrics([1, 2, 3], {99})
    assert m["NDCG@10"] == 0.0


def test_metrics_handle_short_results():
    m = compute_metrics([1], {1})
    assert m["P@5"] == 0.2
    assert m["MRR@10"] == 1.0


# ---------- bootstrap_ci ----------

def test_bootstrap_constant_values():
    mean, lo, hi = bootstrap_ci([0.5] * 100)
    assert mean == 0.5
    assert lo == 0.5
    assert hi == 0.5


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert lo <= mean <= hi
    assert 0 <= lo <= 1
    assert 0 <= hi <= 1


def test_bootstrap_deterministic_with_seed():
    vals = [0.1, 0.2, 0.3, 0.4, 0.5]
    r1 = bootstrap_ci(vals, seed=42)
    r2 = bootstrap_ci(vals, seed=42)
    assert r1 == r2


def test_bootstrap_empty_returns_zeros():
    assert bootstrap_ci([]) == (0.0, 0.0, 0.0)


def test_bootstrap_narrow_ci_for_large_n():
    """С 200 одинаковых значений CI должно быть узким."""
    vals = [0.5] * 200
    _, lo, hi = bootstrap_ci(vals)
    assert hi - lo < 0.01


def test_ndcg_position_matters():
    """NDCG должен быть выше когда релевантный документ выше."""
    high = compute_metrics([1, 99, 99, 99, 99], {1})["NDCG@10"]
    low = compute_metrics([99, 99, 99, 99, 1], {1})["NDCG@10"]
    assert high > low


def test_ndcg_log2_formula():
    """Sanity check: NDCG для 1 релевантного на позиции 1 = 1.0."""
    m = compute_metrics([1], {1})
    assert m["NDCG@10"] == 1.0


def test_ndcg_known_value():
    """Релевантный на позиции 2 → DCG = 1/log2(3), IDCG=1 → NDCG = 1/log2(3)."""
    m = compute_metrics([99, 1, 99], {1})
    expected = 1 / math.log2(3)
    assert abs(m["NDCG@10"] - expected) < 1e-9
