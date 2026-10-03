"""Judged evaluation: pooled results from several system variants, graded by GPT-6 Luna (cached).

uv run python -m eval.run_eval # full system, judged
uv run python -m eval.run_eval --sweep # also sweep the reranker threshold offline
uv run python -m eval.run_eval --show-misses # list wrong results shown and good results missed

Query sets: eval/gen_queries.jsonl (Luna-written from random photos) + eval/hand_queries.txt (ours, incl. user reports).
Pool per query = top 20 of: full system, SigLIP only, description-vector only, full-text only, no reranker.
Metrics (grade 2 = relevant, 1 = partial counts half):
 P@10 share of shown results (up to 10) that are relevant
 R@pool share of the pool's relevant photos the system shows (up to 60)
 empty queries with no results although the pool has a relevant photo
 junk shown results graded 0
"""
import argparse, json, re, statistics as st
from pathlib import Path
from app.db import connect
from app import search as S
from eval.judge import grades, spent

HERE = Path(__file__).parent
VARIANTS = {"full": ("vector", "desc", "text", "person", "rerank"), "siglip": ("vector", "person"),
 "desc": ("desc", "person"), "fulltext": ("text", "person"), "no_rerank": ("vector", "desc", "text", "person"),
 "ctx_only": ("vector", "desc", "text", "rerank")} # no person evidence: pool sees photos the people index missed


def keep(results, threshold):
 """Same rule as app/search.py: one threshold on relevance."""
 return [x for x in results if x["relevance"] >= threshold]


def load_queries():
 qs = []
 if (HERE / "gen_queries.jsonl").exists():
 qs += [{"query": q["query"], "type": q["type"], "source_image": q.get("source_image")} for q in map(json.loads, (HERE / "gen_queries.jsonl").open())]
 if (HERE / "hand_queries.txt").exists():
 for l in (HERE / "hand_queries.txt").open():
 if l.strip() and not l.startswith("#"):
 t, q = l.strip().split("|", 1)
 qs.append({"query": q.strip(), "type": t.strip()})
 return qs


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--sweep", action="store_true"); ap.add_argument("--show-misses", action="store_true")
 ap.add_argument("--label", default=None, help="save per-query metrics to eval/runs/<label>.json for paired comparison (eval/compare.py)")
 a = ap.parse_args()
 conn = connect(); S.search("warm up", conn=conn)
 files = dict(conn.execute("SELECT id, file FROM images"))
 qs = load_queries()

 runs = {} # (variant, query) -> results list (ctx_min=0 so thresholds can be swept offline)
 lat = []
 for q in qs:
 for v, use in VARIANTS.items():
 r = S.search(q["query"], conn=conn, use=use, rel_min=0.0 if v == "full" else None)
 runs[(v, q["query"])] = r["results"]
 if v == "full":
 lat.append(r["timings"]["total_ms"])
 pairs = {(q["query"], x["id"]) for q in qs for v in VARIANTS for x in runs[(v, q["query"])][:20]}
 # queries naming people are judged blind on names (see eval/judge.py); years are checked against the stored date
 names_for, year_of = {}, {}
 for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 # names come from the LLM parse (so the judge is never primed with a name, even for people missing from the directory);
 # the context is the English rewrite, so Hinglish queries are judged by meaning
 parsed_people = [pp["name"] for pp in (r.get("parsed") or {}).get("people") or []]
 nm = r["required"] or parsed_people
 ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
 if ys:
 year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys or S.norm(ctx) != S.norm(q["query"]):
 names_for[q["query"]] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 # a photo outside the asked year range cannot satisfy the query, whatever it shows
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]):
 g[(qq, iid)] = 0
 print(f"judged pairs: {len(pairs)} (new judging cost ${spent['usd']:.3f} over {spent['calls']} calls)")

 def score(results, q):
 shown = [x for x in results]
 top = shown[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 relevant_pool = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else (0.0 if relevant_pool else None) # empty with answers = 0
 rec = len(relevant_pool & {x["id"] for x in shown}) / len(relevant_pool) if relevant_pool else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 # fixed-denominator precision and nDCG@10 (gains 2/1/0, ideal from the judged pool)
 import math
 p10f = sum(val(x["id"]) for x in top) / 10 if relevant_pool or top else None
 gains = sorted((gr for (qq, iid), gr in g.items() if qq == q and gr), reverse=True)[:10]
 idcg = sum(gg / math.log2(i + 2) for i, gg in enumerate(gains))
 dcg = sum((g.get((q, x["id"])) or 0) / math.log2(i + 2) for i, x in enumerate(top))
 ndcg = (dcg / idcg) if idcg else None
 return p10, rec, (not shown and bool(relevant_pool)), junk, len(shown), p10f, ndcg

 def boot(vals, n=1000, seed=0):
 """95% bootstrap interval over queries."""
 import random
 vals = [v for v in vals if v is not None]
 if not vals:
 return (0, 0)
 rng = random.Random(seed); ms = []
 for _ in range(n):
 smp = [vals[rng.randrange(len(vals))] for _ in vals]; ms.append(st.mean(smp))
 ms.sort(); return (ms[int(0.025 * n)], ms[int(0.975 * n)])

 def known_item(threshold):
 """Source photo rank for Luna-written queries (judge-free): hit@1/5/10, MRR, never-retrieved."""
 ranks = []
 for q in qs:
 src = q.get("source_image")
 if not src:
 continue
 ids = [x["id"] for x in runs[("full", q["query"])]] # rel_min 0 ranking
 ranks.append(ids.index(src) + 1 if src in ids else None)
 if not ranks:
 return
 hit = lambda k: sum(1 for r in ranks if r and r <= k) / len(ranks)
 mrr = st.mean((1 / r) if r else 0 for r in ranks)
 ci = boot([1.0 if (r and r <= 10) else 0.0 for r in ranks])
 print(f"known-item ({len(ranks)} generated queries): hit@1 {hit(1):.2f} hit@5 {hit(5):.2f} hit@10 {hit(10):.2f} "
 f"[95% {ci[0]:.2f}-{ci[1]:.2f}] MRR {mrr:.2f} never retrieved {sum(r is None for r in ranks)}")
 return {q["query"]: r for q, r in zip([q for q in qs if q.get("source_image")], ranks)}

 per_query = {}

 def report(threshold, label):
 rows = {}
 for q in qs:
 res = keep(runs[("full", q["query"])], threshold)
 m = score(res, q["query"])
 per_query.setdefault(threshold, {})[q["query"]] = {"type": q["type"], "p10": m[0], "rec": m[1], "empty": m[2], "junk": m[3],
 "shown": m[4], "p10f": m[5], "ndcg": m[6]}
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 print(f"\n{label}")
 print(f"{'type':<16}{'n':>4}{'P@10':>7}{'P@10/10':>8}{'nDCG':>6}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}")
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 pf = [m[5] for m in ms if m[5] is not None]; nd = [m[6] for m in ms if m[6] is not None]
 ci = f" [P@10 95% {boot(p)[0]:.2f}-{boot(p)[1]:.2f}]" if k == "ALL" else ""
 print(f"{k:<16}{len(ms):>4}{(st.mean(p) if p else 0):>7.2f}{(st.mean(pf) if pf else 0):>8.2f}{(st.mean(nd) if nd else 0):>6.2f}"
 f"{(st.mean(r) if r else 0):>8.2f}{sum(m[2] for m in ms):>7}{st.mean(m[3] for m in ms):>8.2f}{st.mean(m[4] for m in ms):>7.1f}{ci}")
 return rows

 report(S.REL_MIN, f"FULL SYSTEM, both tiers (relevance >= {S.REL_MIN})")
 report(S.TIER_MATCH, f"TOP TIER ONLY (relevance >= {S.TIER_MATCH})")
 ki = known_item(S.REL_MIN)
 if a.label:
 (HERE / "runs").mkdir(exist_ok=True)
 json.dump({"label": a.label, "rel_min": S.REL_MIN, "tier_match": S.TIER_MATCH, "per_query": {str(k): v for k, v in per_query.items()},
 "known_item": ki}, (HERE / "runs" / f"{a.label}.json").open("w"))
 print(f"saved eval/runs/{a.label}.json")
 for v in ("siglip", "desc", "fulltext", "no_rerank"):
 allm = [score(runs[(v, q["query"])][:60], q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 print(f" variant {v:<10} P@10 {st.mean(p) if p else 0:.2f} R@pool {st.mean(r) if r else 0:.2f} junk/q {st.mean(m[3] for m in allm):.2f}")
 lat.sort()
 print(f"\nlatency ms (full): p50 {st.median(lat):.0f} p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} max {lat[-1]:.0f}")

 if a.sweep:
 print("\nthreshold sweep (ALL):")
 for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
 allm = [score(keep(runs[("full", q["query"])], t), q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 print(f" >= {t:.2f}: P@10 {st.mean(p) if p else 0:.2f} R@pool {st.mean(r) if r else 0:.2f} "
 f"empty {sum(m[2] for m in allm)} junk/q {st.mean(m[3] for m in allm):.2f}")
 if a.show_misses:
 print("\nwrong results shown (grade 0) and relevant photos missed:")
 for q in qs:
 res = keep(runs[("full", q["query"])], S.REL_MIN)
 shown = {x["id"] for x in res}
 bad = [x["id"] for x in res[:10] if g.get((q["query"], x["id"])) == 0]
 missed = [iid for (qq, iid), gr in g.items() if qq == q["query"] and gr == 2 and iid not in shown]
 if bad or missed:
 print(f" [{q['type']}] {q['query']!r}: {len(bad)} wrong shown, {len(missed)} relevant missed")


if __name__ == "__main__":
 import os, sys
 main()
 sys.stdout.flush()
 os._exit(0) # do not wait for abandoned judge threads
