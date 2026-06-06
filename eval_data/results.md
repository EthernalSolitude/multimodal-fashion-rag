# Eval results

Test set: **100 categorical queries** from indexed catalog (frozen, seed=42).
Each query: `{Color} {Category} for {Gender}`. Ground truth: items matching all 3 facets.
95% CI via **bootstrap** (n=1000).

| Config | P@5 | R@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| **BM25 sparse only** | 0.894 [0.856, 0.926] | 0.935 [0.907, 0.960] | 0.920 [0.891, 0.945] | 0.940 [0.910, 0.970] |
| **Dense CLIP only** | 0.654 [0.592, 0.720] | 0.646 [0.590, 0.703] | 0.661 [0.606, 0.716] | 0.813 [0.749, 0.870] |
| **Hybrid RRF** | 0.806 [0.764, 0.844] | 0.862 [0.826, 0.895] | 0.848 [0.813, 0.880] | 0.946 [0.913, 0.977] |
| **Hybrid + cross-encoder rerank** | 0.934 [0.904, 0.960] | 0.961 [0.938, 0.979] | 0.955 [0.933, 0.973] | 0.978 [0.953, 0.995] |
