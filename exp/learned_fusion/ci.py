"""Bootstrap confidence intervals over queries. Reusable by other experiments.

 from exp.learned_fusion.ci import boot_ci, paired_ci
 boot_ci([per_query_values]) -> (mean, lo, hi)
 paired_ci(a_values, b_values) -> (mean_diff, lo, hi) # same queries in the same order; None entries skipped
"""
import numpy as np


def boot_ci(vals, n=1000, seed=0):
 v = np.asarray([x for x in vals if x is not None], dtype=float)
 if len(v) == 0:
 return (float("nan"),) * 3
 rng = np.random.default_rng(seed)
 idx = rng.integers(0, len(v), size=(n, len(v)))
 m = v[idx].mean(axis=1)
 return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired_ci(a, b, n=1000, seed=0):
 pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
 if not pairs:
 return (float("nan"),) * 3
 d = np.asarray([y - x for x, y in pairs], dtype=float)
 rng = np.random.default_rng(seed)
 idx = rng.integers(0, len(d), size=(n, len(d)))
 m = d[idx].mean(axis=1)
 return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt(ci):
 return f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]"
