"""Learned fusion, step 1: features for every judged (query, candidate) pair from the live system's recall + stage 2.
Writes exp/fusion/features.jsonl. Usage: uv run python -m exp.fusion.features
Features: p_ctx (Jev), p_person (product of required-person beliefs), per-source ranks (vector/desc/text/person/rare/concept; 0 = absent),
n_sources, rrf, siglip cosine (query text vs image), desc cosine, caption present, caption-word overlap with the query, year match, n faces,
confident-name present, query type one-hots. Label: judge grade (0/1/2). Group: query."""
import json, re
from pathlib import Path
from app.db import connect
from app import search as S
from app.embed import embed_text
from app.text import embed_query
from eval.run_eval import load_queries
HERE = Path(__file__).parent
conn = connect(); S.PARSE_TIMEOUT_MS = 60000
J = {}
for l in open("eval/judgments_v2.jsonl"):
 try:
 r = json.loads(l); J[(r["query"], r["image_id"])] = r["grade"]
 except Exception:
 pass
out = open(HERE / "features.jsonl", "w"); n = 0
for q in load_queries():
 r = S.search(q["query"], conn=conn, rel_min=0.0, limit=150)
 if not r["results"]:
 continue
 qv = embed_text([r["reranked_as"]])[0]; dv = embed_query(S.expand_cities(r["reranked_as"]))
 ids = [x["id"] for x in r["results"]]
 cos = {i: (float(a), float(b), cap, nf, nn, yr) for i, a, b, cap, nf, nn, yr in conn.execute("""
 SELECT i.id, 1 - (i.clip <=> %s), 1 - (i.desc_emb <=> %s), i.caption IS NOT NULL,
 (SELECT count(*) FROM faces f WHERE f.image_id = i.id),
 (SELECT count(*) FROM image_person_scores s WHERE s.image_id = i.id AND s.score >= 0.8),
 extract(year from i.date_published)::int
 FROM images i WHERE i.id = ANY(%s)""", (qv, dv, ids))}
 qwords = set(S.norm(q["query"]).split()) - S.STOP
 years = [int(y) for y in re.findall(S.YEAR_RE, q["query"])]
 for x in r["results"]:
 g = J.get((q["query"], x["id"]))
 if g is None:
 continue
 ranks = {}
 for w in x["why"]:
 src, rk = w.split("#"); ranks[src] = int(rk)
 sc, dc, cap, nf, nn, yr = cos[x["id"]]
 capw = set(S.norm(x.get("caption") or "").split())
 out.write(json.dumps({"query": q["query"], "type": q["type"], "image_id": x["id"], "grade": g,
 "p_ctx": x["p_ctx"], "p_person": x["p_person"] if x["p_person"] is not None else 1.0, "required": bool(r["required"]),
 **{f"rank_{s}": ranks.get(s, 0) for s in ("vector", "desc", "text", "person", "rare", "concept")},
 "n_sources": len(ranks), "siglip": sc, "desc": dc, "caption": int(cap), "cap_overlap": len(qwords & capw) / max(len(qwords), 1),
 "year_match": int(bool(years) and yr is not None and min(years) <= yr <= max(years)), "n_faces": nf, "named": int(nn > 0)}) + "\n")
 n += 1
print(f"{n} judged pairs with features -> {HERE / 'features.jsonl'}")
