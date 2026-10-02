"""Judged eval of one variant table. uv run python -m exp.caption_retrieval.run <table|images> <label>
Same metrics as eval/run_eval.py; pools only the variants whose recall sources changed (full, fulltext, desc)
so the judge cache grows but shared pairs are reused. Saves per-query results to exp/caption_retrieval/<label>.json."""
import json, re, sys, statistics as st
import torch; torch.set_num_threads(4)
from pathlib import Path
from app.db import connect
from app import search as S
from eval.run_eval import load_queries
from eval.judge import grades, spent

HERE = Path(__file__).parent
table, label = sys.argv[1], sys.argv[2]
S.TABLES["images"] = table
VARIANTS = {"full": ("vector", "desc", "text", "person", "rerank"), "fulltext": ("text", "person"), "desc": ("desc", "person")}

conn = connect(); S.search("warm up", conn=conn)
files = dict(conn.execute("SELECT id, file FROM images"))
qs = load_queries()
runs, lat = {}, []
for q in qs:
 for v, use in VARIANTS.items():
 r = S.search(q["query"], conn=conn, use=use, rel_min=0.0 if v == "full" else None)
 runs[(v, q["query"])] = r["results"]
 if v == "full":
 lat.append(r["timings"]["total_ms"])
pairs = {(q["query"], x["id"]) for q in qs for v in VARIANTS for x in runs[(v, q["query"])][:20]}
names_for, year_of = {}, {}
for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 nm = r["required"]; ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
 if ys: year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys: names_for[q["query"]] = (nm or None, ctx)
g = grades(pairs, files, names_for)
years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]): g[(qq, iid)] = 0
print(f"[{label}] judged pairs: {len(pairs)} (new judging cost ${spent['usd']:.3f} over {spent['calls']} calls)")

def score(results, q):
 top = results[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 # relevant pool = everything graded 2 for this query in the whole judge cache (shared across experiments)
 rel = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel & {x["id"] for x in results}) / len(rel) if rel else None
 return p10, rec, (not results and bool(rel)), sum(1 for x in top if g.get((q, x["id"])) == 0), len(results)

def report(th):
 rows = {}
 for q in qs:
 res = [x for x in runs[("full", q["query"])] if x["relevance"] >= th]
 m = score(res, q["query"])
 for k in (q["type"], "ALL"): rows.setdefault(k, []).append(m)
 out = [f"{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}"]
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 out.append(f"{k:<16}{len(ms):>4}{(st.mean(p) if p else 0):>7.2f}{(st.mean(r) if r else 0):>8.2f}{sum(m[2] for m in ms):>7}{st.mean(m[3] for m in ms):>8.2f}{st.mean(m[4] for m in ms):>7.1f}")
 return "\n".join(out)

lines = [f"[{label}] table={table} threshold {S.REL_MIN}", report(S.REL_MIN)]
lat.sort(); lines.append(f"latency ms: p50 {st.median(lat):.0f} p95 {lat[int(.95*(len(lat)-1))]:.0f}")
lines.append("sweep:")
for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
 allm = [score([x for x in runs[("full", q["query"])] if x["relevance"] >= t], q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 lines.append(f" >= {t:.2f}: P@10 {st.mean(p):.2f} R@pool {st.mean(r):.2f} empty {sum(m[2] for m in allm)} junk/q {st.mean(m[3] for m in allm):.2f}")
print("\n".join(lines))
(HERE / f"{label}.txt").write_text("\n".join(lines))
json.dump({q["query"]: {"type": q["type"], "full": [(x["id"], x["relevance"], g.get((q["query"], x["id"]))) for x in runs[("full", q["query"])]]} for q in qs},
 open(HERE / f"{label}.json", "w"))
