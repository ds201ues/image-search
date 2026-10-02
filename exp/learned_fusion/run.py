"""E7 runner: run search variants, judge new pairs (cached, append-only), score with CIs.

uv run python -m exp.learned_fusion.run --variants base,rrf,pool100,rrf_pool100,fallback,all
Runs are cached in exp/learned_fusion/runs/<variant>.pkl (rel_min=0, limit 200: every reranked candidate with features).
"""
import argparse, json, pickle, re, statistics as st, hashlib, time
from pathlib import Path
import torch
torch.set_num_threads(4)
from app.db import connect
from eval.run_eval import load_queries
from eval.judge import grades, spent
from exp.learned_fusion import search_v as S
from exp.learned_fusion.ci import boot_ci, paired_ci, fmt

HERE = Path(__file__).parent
RUNS = HERE / "runs"; RUNS.mkdir(exist_ok=True)
VARIANTS = {"base": {}, "rrf": {"rrf": True}, "pool100": {"pool": 100}, "rrf_pool100": {"rrf": True, "pool": 100},
 "fallback": {"fallback": True}, "all": {"rrf": True, "pool": 100, "fallback": True}}
YEAR = r"\b(19[5-9]\d|20[0-4]\d)\b"


def split(q): # stable dev/test split by query hash
 return "dev" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 2 == 0 else "test"


def run_variant(name, qs, conn):
 f = RUNS / f"{name}.pkl"
 if f.exists():
 return pickle.load(f.open("rb"))
 S.OPTS.update({"rrf": False, "pool": 50, "fallback": False}); S.OPTS.update(VARIANTS[name])
 out = {}
 t0 = time.time()
 for i, q in enumerate(qs):
 r = S.search(q["query"], conn=conn, rel_min=0.0, limit=200)
 out[q["query"]] = r
 if i % 40 == 0:
 print(f" {name}: {i}/{len(qs)} {time.time() - t0:.0f}s", flush=True)
 pickle.dump(out, f.open("wb"))
 return out


def judge_all(runs, qs, conn, topn=20):
 """Judge the union of top-n of every variant; blind names + year check exactly as eval/run_eval.py."""
 files = dict(conn.execute("SELECT id, file FROM images"))
 pairs = {(q, x["id"]) for name in runs for q, r in runs[name].items() for x in r["results"][:topn]}
 names_for, year_of = {}, {}
 base = runs[next(iter(runs))]
 for q in qs:
 r = base[q["query"]]
 nm = r["required"]
 ys = re.findall(YEAR, q["query"])
 if ys:
 year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys:
 names_for[q["query"]] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]):
 g[(qq, iid)] = 0
 return g


def score(results, q, g):
 top = results[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 rel_pool = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel_pool & {x["id"] for x in results}) / len(rel_pool) if rel_pool else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not results and bool(rel_pool)), junk, len(results)


def keep_fixed(results, t):
 return [x for x in results if x["relevance"] >= t]


def keep_dyn(results, floor, ratio, k):
 if not results:
 return []
 top = results[0]["relevance"]
 out = [x for x in results if x["relevance"] >= max(floor, ratio * top)]
 have = {x["id"] for x in out}
 for x in results[:k]:
 if x["relevance"] >= floor and x["id"] not in have:
 out.append(x)
 return out


def table(rows_by_type, label):
 print(f"\n{label}")
 print(f"{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}")
 for k, ms in sorted(rows_by_type.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 print(f"{k:<16}{len(ms):>4}{(st.mean(p) if p else 0):>7.2f}{(st.mean(r) if r else 0):>8.2f}"
 f"{sum(m[2] for m in ms):>7}{st.mean(m[3] for m in ms):>8.2f}{st.mean(m[4] for m in ms):>7.1f}")


def evaluate(runs_v, qs, g, keep, label, subset=None):
 rows, per_q = {}, {}
 for q in qs:
 if subset and split(q["query"]) != subset:
 continue
 m = score(keep(runs_v[q["query"]]["results"]), q["query"], g)
 per_q[q["query"]] = m
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 if label:
 table(rows, label)
 return per_q


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--variants", default="base,rrf,pool100,rrf_pool100,fallback,all")
 ap.add_argument("--no-judge", action="store_true")
 a = ap.parse_args()
 conn = connect(); S.search("warm up", conn=conn)
 qs = load_queries()
 runs = {v: run_variant(v, qs, conn) for v in a.variants.split(",")}
 g = judge_all(runs, qs, conn)
 print(f"judged pairs available: {len(g)} new cost ${spent['usd']:.3f}")
 base_pq = None
 for v in runs:
 pq = evaluate(runs[v], qs, g, lambda r: keep_fixed(r, S.REL_MIN), f"VARIANT {v} (fixed threshold {S.REL_MIN})")
 lat = sorted(r["timings"]["total_ms"] for r in runs[v].values())
 print(f" latency p50 {lat[len(lat)//2]:.0f} p95 {lat[int(.95*(len(lat)-1))]:.0f}; rerank p50 "
 f"{sorted(r['timings']['rerank_ms'] for r in runs[v].values())[len(lat)//2]:.0f} ms")
 p10 = [m[0] for m in pq.values()]; rec = [m[1] for m in pq.values()]
 print(f" CI P@10 {fmt(boot_ci(p10))} R@pool {fmt(boot_ci(rec))}")
 if base_pq is None:
 base_pq = pq
 else:
 d10 = paired_ci([base_pq[q][0] for q in pq], [pq[q][0] for q in pq])
 dr = paired_ci([base_pq[q][1] for q in pq], [pq[q][1] for q in pq])
 print(f" vs base: dP@10 {fmt(d10)} dR@pool {fmt(dr)}")
 # dynamic cutoff sweep on dev, report on test, for the last variant
 v = list(runs)[-1]
 print(f"\nDYNAMIC CUTOFF sweep on dev split, variant {v}")
 best = None
 for floor in (0.05, 0.1, 0.2, 0.3):
 for ratio in (0.0, 0.3, 0.5, 0.7):
 for k in (0, 3, 5):
 pq = evaluate(runs[v], qs, g, lambda r: keep_dyn(r, floor, ratio, k), None, "dev")
 p10 = st.mean(m[0] for m in pq.values() if m[0] is not None); rec = st.mean(m[1] for m in pq.values() if m[1] is not None)
 junk = st.mean(m[3] for m in pq.values()); emp = sum(m[2] for m in pq.values())
 obj = p10 + rec - 0.1 * junk # balance precision and recall, penalise junk
 print(f" floor {floor:.2f} ratio {ratio:.1f} k {k}: P@10 {p10:.3f} R {rec:.3f} empty {emp} junk {junk:.2f} obj {obj:.3f}")
 if best is None or obj > best[0]:
 best = (obj, floor, ratio, k)
 _, floor, ratio, k = best
 print(f"best on dev: floor {floor} ratio {ratio} k {k}")
 for sub in ("dev", "test"):
 evaluate(runs[v], qs, g, lambda r: keep_fixed(r, S.REL_MIN), f"{sub}: fixed 0.2", sub)
 evaluate(runs[v], qs, g, lambda r: keep_dyn(r, floor, ratio, k), f"{sub}: dynamic floor {floor} ratio {ratio} k {k}", sub)
 json.dump({"floor": floor, "ratio": ratio, "k": k, "variant": v}, (HERE / "cutoff.json").open("w"))


if __name__ == "__main__":
 import os, sys
 main(); sys.stdout.flush(); os._exit(0)
