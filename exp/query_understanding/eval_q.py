"""E6 eval: baseline vs parsed variants, judged with the shared cache (eval.judge), same metrics as eval/run_eval.py."""
import json, re, statistics as st, sys, time
from pathlib import Path
from app.db import connect
from app import search as S
from eval.judge import grades, spent
from eval.run_eval import load_queries
from .search_q import search_q
from .parser import parse, stats as pstats

HERE = Path(__file__).parent
YEAR = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")


def main():
 conn = connect(); S.search("warm up", conn=conn)
 files = dict(conn.execute("SELECT id, file FROM images"))
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 qs = load_queries()
 V = {"baseline": lambda q: S.search(q, conn=conn, rel_min=0.0),
 "parsed": lambda q: search_q(q, conn, "parsed", rel_min=0.0),
 "parsed_exp": lambda q: search_q(q, conn, "parsed", expansions=True, rel_min=0.0),
 "city_boost": lambda q: search_q(q, conn, "city_boost", rel_min=0.0),
 "city_filter": lambda q: search_q(q, conn, "city_filter", rel_min=0.0)}
 runs, lat, meta = {}, {v: [] for v in V}, {}
 for q in qs:
 for v, fn in V.items():
 r = fn(q["query"])
 runs[(v, q["query"])] = r["results"]
 lat[v].append(r["timings"]["total_ms"])
 if v == "parsed":
 meta[q["query"]] = r
 # judge: names blind (required people from the variant's own parse), years from the query or the parse
 pairs = {(q["query"], x["id"]) for q in qs for v in V for x in runs[(v, q["query"])][:20]}
 names_for, year_of = {}, {}
 for q in qs:
 rb = S.search(q["query"], conn=conn, limit=1); rp = meta[q["query"]]
 nm = rp["required"] or rb["required"]
 ys = YEAR.findall(q["query"])
 p = rp["parsed"]
 if ys:
 year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 elif p["year_from"]:
 year_of[q["query"]] = (p["year_from"], p["year_to"] or p["year_from"])
 ctx = " ".join(w for w in rp["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else YEAR.sub("", q["query"]).strip()
 if nm or q["query"] in year_of:
 names_for[q["query"]] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]):
 g[(qq, iid)] = 0
 print(f"judged pairs: {len(pairs)} (new judging ${spent['usd']:.3f} over {spent['calls']} calls); parser ${pstats['usd']:.4f}")

 def score(results, q, thr):
 shown = [x for x in results if x["relevance"] >= thr]
 top = shown[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 rel = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel & {x["id"] for x in shown}) / len(rel) if rel else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not shown and bool(rel)), junk, len(shown)

 out = []
 def table(v, thr):
 rows = {}
 for q in qs:
 m = score(runs[(v, q["query"])], q["query"], thr)
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 out.append(f"\n### {v} (threshold {thr})\n\n| type | n | P@10 | R@pool | empty | junk/q | shown |\n|---|---|---|---|---|---|---|")
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 out.append(f"| {k} | {len(ms)} | {st.mean(p) if p else 0:.2f} | {st.mean(r) if r else 0:.2f} | {sum(m[2] for m in ms)} | "
 f"{st.mean(m[3] for m in ms):.2f} | {st.mean(m[4] for m in ms):.1f} |")
 for v in V:
 table(v, S.REL_MIN)
 out.append("\n### Summary at threshold %.2f\n\n| variant | P@10 | R@pool | empty | junk/q | p50 ms | p95 ms |\n|---|---|---|---|---|---|---|" % S.REL_MIN)
 for v in V:
 ms = [score(runs[(v, q["query"])], q["query"], S.REL_MIN) for q in qs]
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 l = sorted(lat[v])
 out.append(f"| {v} | {st.mean(p):.3f} | {st.mean(r):.3f} | {sum(m[2] for m in ms)} | {st.mean(m[3] for m in ms):.2f} | "
 f"{l[len(l)//2]:.0f} | {l[int(.95*(len(l)-1))]:.0f} |")
 # per-query deltas baseline vs parsed, for examples
 out.append("\n### Biggest per-query changes (parsed vs baseline, P@10)\n")
 deltas = []
 for q in qs:
 b = score(runs[("baseline", q["query"])], q["query"], S.REL_MIN)[0] or 0
 for v in ("parsed", "city_boost", "city_filter"):
 p = score(runs[(v, q["query"])], q["query"], S.REL_MIN)[0] or 0
 deltas.append((p - b, v, q["type"], q["query"], b, p, meta[q["query"]]["parsed"]["search_text"], meta[q["query"]]["parsed"]["city"]))
 deltas.sort(key=lambda d: -abs(d[0]))
 for d in deltas[:25]:
 out.append(f"- {d[0]:+.2f} [{d[1]}/{d[2]}] `{d[3]}` -> `{d[6]}` city={d[7]!r}: P@10 {d[4]:.2f} -> {d[5]:.2f}")
 # per-query raw for the writeup
 (HERE / "per_query.json").write_text(json.dumps([{"query": q["query"], "type": q["type"],
 **{v: score(runs[(v, q["query"])], q["query"], S.REL_MIN)[:2] for v in V}} for q in qs], indent=0, default=str))
 txt = "\n".join(out)
 (HERE / "eval_output.md").write_text(txt)
 print(txt)


if __name__ == "__main__":
 import os
 main(); sys.stdout.flush(); os._exit(0)
