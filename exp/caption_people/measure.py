"""Step 5: baseline vs caption-people evidence, same judged eval (eval.run_eval logic, TABLES swapped)."""
import re, statistics as st, sys
from app.db import connect
from app import search as S
from eval.run_eval import load_queries
from eval.judge import grades, spent

EXP = {"scores": "exp_caption_people_scores", "persons": "exp_caption_people_persons"}


def run_all(conn, qs):
 S.people(conn, refresh=True)
 out, lat, names_for, year_of = {}, [], {}, {}
 for q in qs:
 r = S.search(q["query"], conn=conn)
 out[q["query"]] = r
 lat.append(r["timings"]["total_ms"])
 nm = r["required"]
 ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
 if ys:
 year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys:
 names_for[q["query"]] = (nm or None, ctx)
 return out, lat, names_for, year_of


def score(g, results, q, years, year_of):
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 top = results[:10]
 rel_pool = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel_pool & {x["id"] for x in results}) / len(rel_pool) if rel_pool else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not results and bool(rel_pool)), junk, len(results)


def table(label, qs, runs, g, years, year_of):
 rows = {}
 for q in qs:
 m = score(g, runs[q["query"]]["results"], q["query"], years, year_of)
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 lines = [f"\n{label}", f"{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}"]
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 lines.append(f"{k:<16}{len(ms):>4}{(st.mean(p) if p else 0):>7.2f}{(st.mean(r) if r else 0):>8.2f}"
 f"{sum(m[2] for m in ms):>7}{st.mean(m[3] for m in ms):>8.2f}{st.mean(m[4] for m in ms):>7.1f}")
 return "\n".join(lines), rows


def main():
 conn = connect(); S.search("warm up", conn=conn)
 files = dict(conn.execute("SELECT id, file FROM images"))
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 qs = load_queries()
 base, blat, nf_b, yo = run_all(conn, qs)
 S.TABLES.update(EXP)
 exp, elat, nf_e, _ = run_all(conn, qs)
 S.TABLES.update({"scores": "image_person_scores", "persons": "persons"})
 pairs = {(q, x["id"]) for runs in (base, exp) for q, r in runs.items() for x in r["results"][:20]}
 # blind-name judging must use the same person names for both systems: prefer the experiment's linking (superset)
 nf = {**nf_b, **nf_e}
 g = grades(pairs, files, nf)
 for (qq, iid), gr in list(g.items()):
 if qq in yo and gr:
 y = years.get(iid)
 if y is None or not (yo[qq][0] <= y <= yo[qq][1]):
 g[(qq, iid)] = 0
 print(f"judged pairs {len(pairs)}, new judging ${spent['usd']:.3f} over {spent['calls']} calls")
 tb, rb = table("BASELINE (live persons: Luna + face clusters)", qs, base, g, years, yo)
 te, re_ = table("CAPTION PEOPLE (+ caption names, caption-seeded clusters)", qs, exp, g, years, yo)
 print(tb); print(te)
 for lab, lat in (("baseline", blat), ("caption_people", elat)):
 lat.sort(); print(f"latency {lab}: p50 {st.median(lat):.0f} ms p95 {lat[int(0.95 * (len(lat) - 1))]:.0f} ms")
 print("\nqueries that changed (P@10 base -> exp, required people):")
 ch = []
 for q in qs:
 b = score(g, base[q["query"]]["results"], q["query"], years, yo); e = score(g, exp[q["query"]]["results"], q["query"], years, yo)
 if (b[0] or 0) != (e[0] or 0) or b[4] != e[4]:
 ch.append((abs((e[0] or 0) - (b[0] or 0)), q, b, e))
 for _, q, b, e in sorted(ch, key=lambda x: -x[0])[:25]:
 print(f" [{q['type']}] {q['query']!r}: P@10 {b[0] if b[0] is not None else '-'} -> {e[0] if e[0] is not None else '-'}, "
 f"shown {b[4]} -> {e[4]}, junk {b[3]} -> {e[3]}; required base {base[q['query']]['required']} exp {exp[q['query']]['required']}")
 print(f"\n{len(ch)} of {len(qs)} queries changed")


if __name__ == "__main__":
 import os
 main(); sys.stdout.flush(); os._exit(0)
