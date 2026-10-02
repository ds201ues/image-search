"""E3: judged eval of baseline vs caption-aware tables. One variant at a time.
uv run python -m exp.caption_aware_vlm.evaluate --variant base|cap|cap_lq
Writes exp/caption_aware_vlm/eval_<variant>.json with per-query metrics and the report text."""
import argparse, json, re, statistics as st, sys
import torch
torch.set_num_threads(4)
from pathlib import Path
from app.db import connect
from app import search as S
from eval.judge import grades, spent
from eval.run_eval import load_queries, keep

HERE = Path(__file__).parent
P = "exp_caption_aware_vlm"
VARIANTS = {"base": {}, "cap": {"images": f"{P}_images", "scores": f"{P}_scores"}, "cap_lq": {"images": f"{P}_images_lq", "scores": f"{P}_scores"}}
SOURCES = {"full": ("vector", "desc", "text", "person", "rerank"), "siglip": ("vector", "person"),
 "desc": ("desc", "person"), "fulltext": ("text", "person"), "no_rerank": ("vector", "desc", "text", "person")}


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); a = ap.parse_args()
 S.TABLES.update(VARIANTS[a.variant])
 conn = connect(); S.people(conn, refresh=True); S.search("warm up", conn=conn)
 files = dict(conn.execute("SELECT id, file FROM images"))
 qs = load_queries()
 runs, lat = {}, []
 for q in qs:
 for v, use in SOURCES.items():
 r = S.search(q["query"], conn=conn, use=use, rel_min=0.0 if v == "full" else None)
 runs[(v, q["query"])] = r["results"]
 if v == "full":
 lat.append(r["timings"]["total_ms"])
 pairs = {(q["query"], x["id"]) for q in qs for v in SOURCES for x in runs[(v, q["query"])][:20]}
 names_for, year_of = {}, {}
 for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 nm = r["required"]
 ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
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
 print(f"[{a.variant}] judged pairs {len(pairs)}, new judging ${spent['usd']:.3f} over {spent['calls']} calls")

 def score(results, q):
 top = results[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 rel = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel & {x["id"] for x in results}) / len(rel) if rel else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not results and bool(rel)), junk, len(results)

 rows, per_q = {}, {}
 for q in qs:
 m = score(keep(runs[("full", q["query"])], S.REL_MIN), q["query"])
 per_q[q["query"]] = {"type": q["type"], "p10": m[0], "rec": m[1], "empty": m[2], "junk": m[3], "shown": m[4],
 "top": [x["id"] for x in keep(runs[("full", q["query"])], S.REL_MIN)[:10]]}
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 lines = [f"{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}"]
 summary = {}
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 summary[k] = {"n": len(ms), "p10": st.mean(p) if p else 0, "rec": st.mean(r) if r else 0, "empty": sum(m[2] for m in ms),
 "junk": st.mean(m[3] for m in ms), "shown": st.mean(m[4] for m in ms)}
 lines.append(f"{k:<16}{len(ms):>4}{summary[k]['p10']:>7.2f}{summary[k]['rec']:>8.2f}{summary[k]['empty']:>7}{summary[k]['junk']:>8.2f}{summary[k]['shown']:>7.1f}")
 lat.sort()
 lines.append(f"latency ms: p50 {st.median(lat):.0f} p95 {lat[int(0.95 * (len(lat) - 1))]:.0f}")
 for v in ("siglip", "desc", "fulltext", "no_rerank"):
 allm = [score(runs[(v, q["query"])][:60], q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 lines.append(f" source {v:<10} P@10 {st.mean(p):.2f} R@pool {st.mean(r):.2f}")
 sweep = {}
 for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
 allm = [score(keep(runs[("full", q["query"])], t), q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 sweep[t] = (st.mean(p), st.mean(r), sum(m[2] for m in allm), st.mean(m[3] for m in allm))
 lines.append(f" >= {t:.2f}: P@10 {sweep[t][0]:.2f} R@pool {sweep[t][1]:.2f} empty {sweep[t][2]} junk/q {sweep[t][3]:.2f}")
 report = "\n".join(lines); print(report)
 (HERE / f"eval_{a.variant}.json").write_text(json.dumps({"summary": summary, "per_query": per_q, "report": report, "sweep": {str(k): v for k, v in sweep.items()},
 "judging_usd": spent["usd"]}, indent=1))
 sys.stdout.flush(); import os; os._exit(0)


if __name__ == "__main__":
 main()
