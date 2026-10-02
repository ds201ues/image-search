"""Build eval/grading_pool.jsonl for the human check of the LLM judge (served at /grade).
Stratified: ~100 pairs where the judge and the live system disagree (judge 0 but shown in top-5, or judge 2 but relevance < 0.1),
~100 judge grade-1 pairs, ~100 random judged pairs; balanced across query types. Usage: uv run python -m eval.make_grading_pool
"""
import json, random
from pathlib import Path
from app.db import connect
from app import search as S
from eval.run_eval import load_queries

HERE = Path(__file__).parent
rng = random.Random(11)
J = {}
for f in ("judgments_v2.jsonl", "judgments.jsonl"):
 if (HERE / f).exists():
 for l in (HERE / f).open():
 try:
 r = json.loads(l); J.setdefault((r["query"], r["image_id"]), r["grade"])
 except Exception:
 pass
qs = load_queries(); conn = connect()
dis, g1, rnd = [], [], []
for q in qs:
 r = S.search(q["query"], conn=conn, rel_min=0.0, limit=60)
 top5 = [x["id"] for x in r["results"][:5]]
 rel = {x["id"]: x["relevance"] for x in r["results"]}
 for (qq, iid), g in J.items():
 if qq != q["query"]:
 continue
 row = {"query": qq, "image_id": iid, "type": q["type"], "judge": g}
 if (g == 0 and iid in top5) or (g == 2 and rel.get(iid, 0) < 0.1):
 dis.append(row)
 elif g == 1:
 g1.append(row)
 else:
 rnd.append(row)
rng.shuffle(dis); rng.shuffle(g1); rng.shuffle(rnd)
pool = dis[:100] + g1[:100] + rnd[:100]
rng.shuffle(pool)
(HERE / "grading_pool.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pool))
print(f"pool: {len(pool)} pairs (disagreements {min(100, len(dis))}, grade-1 {min(100, len(g1))}, random {min(100, len(rnd))}) -> eval/grading_pool.jsonl")
