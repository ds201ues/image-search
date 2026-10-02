"""E8 step 4: baseline vs collapsing near-dup / shoot siblings in the top results. Reuses the judge cache; no new judging
beyond what the baseline pool already covers (collapsing only reorders already-shown results)."""
import json, statistics as st, sys
from app.db import connect
from app import search as S
from eval.run_eval import load_queries
from eval.judge import grades

GROUP_TABLE = sys.argv[1] if len(sys.argv) > 1 else "exp_dedup_diversity_shoots" # or exp_dedup_diversity_groups
conn = connect(); S.search("warm up", conn=conn)
col = "shoot_id" if "shoots" in GROUP_TABLE else "group_id"
grp = dict(conn.execute(f"SELECT image_id, {col} FROM {GROUP_TABLE}"))
files = dict(conn.execute("SELECT id, file FROM images"))
qs = load_queries()
runs = {q["query"]: S.search(q["query"], conn=conn)["results"] for q in qs}
pairs = {(q, x["id"]) for q, res in runs.items() for x in res[:20]}
g = grades(pairs, files) # all cached from the baseline run (names/years handled there); uncached -> None

def collapse(res):
 seen, first, rest = set(), [], []
 for x in res:
 k = grp.get(x["id"], x["id"])
 (rest if k in seen else first).append(x); seen.add(k)
 return first + rest

def score(res, q):
 top = res[:10]
 val = lambda i: {2: 1.0, 1: 0.5}.get(g.get((q, i)), 0.0)
 relpool = {i for (qq, i), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(relpool & {x["id"] for x in res}) / len(relpool) if relpool else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 keys = [grp.get(x["id"], x["id"]) for x in top]
 dup_pairs = len(keys) - len(set(keys))
 distinct_rel = len({grp.get(x["id"], x["id"]) for x in top if g.get((q, x["id"])) == 2})
 return p10, rec, junk, dup_pairs, distinct_rel

rows = {}
for q in qs:
 base = score(runs[q["query"]], q["query"]); coll = score(collapse(runs[q["query"]]), q["query"])
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append((base, coll))
print(f"group table: {GROUP_TABLE}")
print(f"{'type':<16}{'n':>4} | {'P@10':>6}{'R':>6}{'junk':>6}{'dups':>6}{'dRel':>6} | {'P@10':>6}{'R':>6}{'junk':>6}{'dups':>6}{'dRel':>6} (baseline | collapsed)")
def agg(ms, i):
 v = [m[i] for m in ms if m[i] is not None]; return st.mean(v) if v else 0
for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 b = [m[0] for m in ms]; c = [m[1] for m in ms]
 print(f"{k:<16}{len(ms):>4} | {agg(b,0):>6.2f}{agg(b,1):>6.2f}{agg(b,2):>6.2f}{agg(b,3):>6.2f}{agg(b,4):>6.2f} | "
 f"{agg(c,0):>6.2f}{agg(c,1):>6.2f}{agg(c,2):>6.2f}{agg(c,3):>6.2f}{agg(c,4):>6.2f}")
changed = [(q["query"], score(runs[q["query"]], q["query"]), score(collapse(runs[q["query"]]), q["query"])) for q in qs
 if [x["id"] for x in runs[q["query"]][:10]] != [x["id"] for x in collapse(runs[q["query"]])[:10]]]
print(f"\n{len(changed)} queries whose top-10 changed; examples:")
for q, b, c in changed[:8]:
 print(f" {q!r}: P@10 {b[0]:.2f}->{c[0]:.2f}, dups {b[3]}->{c[3]}, distinct relevant {b[4]}->{c[4]}")
