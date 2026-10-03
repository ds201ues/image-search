"""Paired comparison of two saved eval runs: uv run python -m eval.compare <labelA> <labelB>
Same queries, per-query differences, 95% paired bootstrap interval; a change is believable when the interval excludes 0."""
import json, random, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).parent
A, B = (json.load((HERE / "runs" / f"{l}.json").open()) for l in sys.argv[1:3])

def paired(xa, xb, n=2000):
 d = [b - a for a, b in zip(xa, xb)]
 rng = random.Random(0); ms = sorted(st.mean(rng.choice(d) for _ in d) for _ in range(n))
 return st.mean(d), ms[int(0.025 * n)], ms[int(0.975 * n)]

for thr_key in A["per_query"]:
 pa, pb = A["per_query"][thr_key], B["per_query"].get(thr_key) or B["per_query"][list(B["per_query"])[list(A["per_query"]).index(thr_key)]]
 qs = [q for q in pa if q in pb]
 print(f"\nthreshold {thr_key}: {len(qs)} shared queries ({A['label']} -> {B['label']})")
 for metric in ("p10", "p10f", "ndcg", "rec"):
 xa = [pa[q][metric] if pa[q][metric] is not None else 0.0 for q in qs if pa[q][metric] is not None or pb[q][metric] is not None]
 xb = [pb[q][metric] if pb[q][metric] is not None else 0.0 for q in qs if pa[q][metric] is not None or pb[q][metric] is not None]
 if not xa:
 continue
 m, lo, hi = paired(xa, xb)
 flag = "BETTER" if lo > 0 else "WORSE" if hi < 0 else "no clear change"
 print(f" {metric:<5} {st.mean(xa):.3f} -> {st.mean(xb):.3f} diff {m:+.3f} [{lo:+.3f}, {hi:+.3f}] {flag}")
 ea, eb = sum(pa[q]["empty"] for q in qs), sum(pb[q]["empty"] for q in qs); ja, jb = st.mean(pa[q]["junk"] for q in qs), st.mean(pb[q]["junk"] for q in qs)
 print(f" empty {ea} -> {eb} junk/q {ja:.2f} -> {jb:.2f}")
 by = {}
 for q in qs:
 by.setdefault(pa[q]["type"], []).append(((pb[q]["p10"] or 0) - (pa[q]["p10"] or 0)))
 print(" P@10 change by type: " + ", ".join(f"{t} {st.mean(v):+.2f}" for t, v in sorted(by.items())))
ka, kb = A.get("known_item") or {}, B.get("known_item") or {}
if ka and kb:
 qs = [q for q in ka if q in kb]
 h = lambda d, k: sum(1 for q in qs if d[q] and d[q] <= k) / len(qs)
 print(f"\nknown-item: hit@1 {h(ka,1):.2f} -> {h(kb,1):.2f} hit@10 {h(ka,10):.2f} -> {h(kb,10):.2f} never retrieved {sum(ka[q] is None for q in qs)} -> {sum(kb[q] is None for q in qs)}")
