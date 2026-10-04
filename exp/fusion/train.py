"""Learned fusion, step 2: logistic regression (few monotone features) with query-grouped CV vs the hand formula relevance = p_ctx * p_person.
Usage: uv run python -m exp.fusion.train (sklearn)"""
import json, math, statistics as st, collections
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "features.jsonl")]
FEATS = ["p_ctx", "p_person", "siglip", "desc", "n_sources", "inv_best_rank", "cap_overlap", "year_match", "named", "caption"]
def vec(r):
 best = min([v for k, v in r.items() if k.startswith("rank_") and v > 0] or [151])
 return [r["p_ctx"], r["p_person"], r["siglip"], r["desc"], r["n_sources"] / 6, 1 / best, r["cap_overlap"], r["year_match"], r["named"], r["caption"]]
X = np.array([vec(r) for r in rows]); y = np.array([1 if r["grade"] == 2 else 0 for r in rows]); g = np.array([r["query"] for r in rows])
hand = np.array([r["p_ctx"] * r["p_person"] for r in rows])
def ndcg_at10(scores):
 by = collections.defaultdict(list)
 for s, r in zip(scores, rows):
 by[r["query"]].append((s, r["grade"]))
 vals = []
 for q, items in by.items():
 items.sort(key=lambda t: -t[0]); top = items[:10]
 dcg = sum(gr / math.log2(i + 2) for i, (_, gr) in enumerate(top))
 ideal = sorted((gr for _, gr in items), reverse=True)[:10]; idcg = sum(gr / math.log2(i + 2) for i, gr in enumerate(ideal))
 if idcg: vals.append(dcg / idcg)
 return st.mean(vals)
def p10(scores, thr):
 by = collections.defaultdict(list)
 for s, r in zip(scores, rows):
 by[r["query"]].append((s, r["grade"]))
 vals = []
 for q, items in by.items():
 shown = sorted([t for t in items if t[0] >= thr], key=lambda t: -t[0])[:10]
 if shown: vals.append(sum({2: 1, 1: .5}.get(gr, 0) for _, gr in shown) / len(shown))
 return st.mean(vals) if vals else 0
pred = np.zeros(len(rows))
for tr, te in GroupKFold(5).split(X, y, g):
 m = LogisticRegression(max_iter=2000, C=1.0).fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
print(f"pairs {len(rows)}, queries {len(set(g))}, positives {y.mean():.2f}")
print(f"{'':<22}{'nDCG@10':>9}{'P@10 @0.1':>11}{'P@10 @0.5':>11}")
print(f"{'hand p_ctx*p_person':<22}{ndcg_at10(hand):>9.3f}{p10(hand, 0.1):>11.3f}{p10(hand, 0.5):>11.3f}")
print(f"{'logistic (5-fold CV)':<22}{ndcg_at10(pred):>9.3f}{p10(pred, 0.3):>11.3f}{p10(pred, 0.6):>11.3f}")
m = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
print("coefficients:", {f: round(c, 2) for f, c in zip(FEATS, m.coef_[0])}, "intercept", round(float(m.intercept_[0]), 2))
json.dump({"features": FEATS, "coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}, open(HERE / "model.json", "w"))
# paired bootstrap on per-query nDCG difference
by_h, by_l = collections.defaultdict(list), collections.defaultdict(list)
for s_h, s_l, r in zip(hand, pred, rows):
 by_h[r["query"]].append((s_h, r["grade"])); by_l[r["query"]].append((s_l, r["grade"]))
def nd(items):
 items = sorted(items, key=lambda t: -t[0])[:10]; dcg = sum(gr / math.log2(i + 2) for i, (_, gr) in enumerate(items))
 ideal = sorted((gr for _, gr in items), reverse=True); idcg = sum(gr / math.log2(i + 2) for i, gr in enumerate(ideal)); return dcg / idcg if idcg else None
d = [nd(by_l[q]) - nd(by_h[q]) for q in by_h if nd(by_h[q]) is not None]
import random; rng = random.Random(0); bs = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(2000))
print(f"paired nDCG@10 diff (logistic - hand): {st.mean(d):+.3f} [{bs[50]:+.3f}, {bs[1949]:+.3f}]")
