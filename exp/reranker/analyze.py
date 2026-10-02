"""Calibration + ranking quality from exported pairs and scores. No models, no GPU, trivial CPU.

PYTHONPATH=. uv run python exp/reranker/analyze.py --pairs exp/reranker/data/archive_pairs.jsonl --scores exp/reranker/scores_archive/
 also accepts compare.py's partial raw_*.npy via --npy "exp/reranker/raw_websample_*.npy" (aligned to pairs order)
Per model/variant: mean per-query AUC, 2-param sigmoid calibration (grade 2 vs rest), P/R at 0.2 / 0.5 on calibrated p.
Cascade option: --cascade small,big (small ranks all, big rescored top-30 per query).
"""
import argparse, glob, json, os, statistics as st
import numpy as np


def fit_sigmoid(s, y, iters=500):
 s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
 a, b = 1.0 / (s.std() + 1e-6), -s.mean() / (s.std() + 1e-6)
 for _ in range(iters):
 p = 1 / (1 + np.exp(-(a * s + b))); w = p * (1 - p) + 1e-9
 g = np.array([((p - y) * s).sum(), (p - y).sum()])
 H = np.array([[(w * s * s).sum(), (w * s).sum()], [(w * s).sum(), w.sum()]]) + 1e-6 * np.eye(2)
 step = np.linalg.solve(H, g); a -= step[0]; b -= step[1]
 if np.abs(step).max() < 1e-8:
 break
 return float(a), float(b)


def evaluate(raw, pairs, label):
 y = np.array([1.0 if p["grade"] == 2 else 0.0 for p in pairs])
 idx = {}
 for k, p in enumerate(pairs):
 idx.setdefault(p["query_raw"], []).append(k)
 aucs = []
 for q, ks in idx.items():
 pos = [raw[k] for k in ks if pairs[k]["grade"] == 2]; neg = [raw[k] for k in ks if pairs[k]["grade"] == 0]
 if pos and neg:
 aucs.append(np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos]))
 a, b = fit_sigmoid(raw, y); prob = 1 / (1 + np.exp(-(a * raw + b)))
 def pr(t):
 s = prob >= t
 return (y[s].mean() if s.any() else 0.0, y[s].sum() / y.sum())
 # P@10-style: per query, mean precision of the top-10 by raw score (grade 2 =1, 1 = .5)
 p10 = []
 for q, ks in idx.items():
 top = sorted(ks, key=lambda k: -raw[k])[:10]
 p10.append(np.mean([{2: 1, 1: .5}.get(pairs[k]["grade"], 0) for k in top]))
 p2, r2 = pr(0.2); p5, r5 = pr(0.5)
 row = dict(model=label, auc=float(st.mean(aucs)), p10_judged=float(st.mean(p10)), p02=float(p2), r02=float(r2), p05=float(p5), r05=float(r5),
 calib_a=a, calib_b=b, n=len(pairs))
 print(f"{label:<60}{row['auc']:>6.3f}{row['p10_judged']:>7.3f}{p2:>7.2f}{r2:>7.2f}{p5:>7.2f}{r5:>7.2f} a={a:.3f} b={b:.3f}", flush=True)
 return row


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--pairs", required=True); ap.add_argument("--scores", default=None); ap.add_argument("--npy", default=None)
 ap.add_argument("--cascade", default=None); ap.add_argument("--out", default=None)
 a = ap.parse_args()
 pairs = [json.loads(l) for l in open(a.pairs)]
 print(f"{len(pairs)} pairs, {len({p['query_raw'] for p in pairs})} queries")
 print(f"{'model':<60}{'AUC':>6}{'P@10j':>7}{'P@.2':>7}{'R@.2':>7}{'P@.5':>7}{'R@.5':>7}")
 rows, raws = [], {}
 if a.npy:
 for f in sorted(glob.glob(a.npy)):
 r = np.load(f)
 if len(r) != len(pairs):
 print(f"skip {f}: {len(r)} scores vs {len(pairs)} pairs"); continue
 lab = os.path.basename(f).replace("raw_", "").replace(".npy", ""); raws[lab] = r
 rows.append(evaluate(r, pairs, lab))
 if a.scores:
 for f in sorted(glob.glob(os.path.join(a.scores, "*.jsonl"))):
 sc = [json.loads(l) for l in open(f)]
 if len(sc) != len(pairs):
 print(f"skip {f}: {len(sc)} vs {len(pairs)}"); continue
 lab = os.path.basename(f).replace(".jsonl", "")
 for v in ("raw_doc", "raw_cap"):
 r = np.array([s[v] for s in sc]); raws[f"{lab}|{v}"] = r
 rows.append(evaluate(r, pairs, f"{lab} [{'caption' if v == 'raw_cap' else 'plain'}]"))
 if a.cascade:
 small, big = a.cascade.split(",")
 rs, rb = raws[small], raws[big]
 idx = {}
 for k, p in enumerate(pairs):
 idx.setdefault(p["query_raw"], []).append(k)
 comb = rs.copy() - 1e6 # everything outside the top-30 by small drops below
 for q, ks in idx.items():
 top = sorted(ks, key=lambda k: -rs[k])[:30]
 for k in top:
 comb[k] = rb[k]
 rows.append(evaluate(comb, pairs, f"cascade {small} -> {big} (top30)"))
 if a.out:
 json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
 main()
