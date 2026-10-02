"""Export judged (query, doc) pairs + all docs/queries to jsonl so scoring can run on a GPU machine without Postgres.

PYTHONPATH=. uv run python exp/reranker/export_pairs.py --judgments eval/judgments.jsonl --out exp/reranker/data/archive
 (set DATABASE_URL=...imgsearch and --judgments eval/judgments_websample.jsonl for the web sample)
Writes <out>_pairs.jsonl {query_raw, query (as the reranker sees it), image_id, grade, doc, doc_cap}
 <out>_docs.jsonl {image_id, doc, doc_cap} every described image (for embedders)
 <out>_queries.jsonl{query_raw, query, relevant:[ids]} for recall@50 of the desc source
Pair order = same construction as compare.py (sorted queries, judgments file order), so the raw_*.npy
files compare.py saved before it was stopped align with <out>_pairs.jsonl.
"""
import argparse, json
from pathlib import Path
from app.db import connect
from app.search import PERSON_SURE, PERSON_MIN, people, norm


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--judgments", default="eval/judgments.jsonl"); ap.add_argument("--out", default="exp/reranker/data/archive")
 a = ap.parse_args()
 Path(a.out).parent.mkdir(parents=True, exist_ok=True)
 conn = connect()
 J = {}
 for l in open(a.judgments):
 try:
 r = json.loads(l)
 except json.JSONDecodeError:
 continue
 if r.get("grade") is not None:
 J[(r["query"], r["image_id"])] = r["grade"]
 docs, docs_cap = {}, {}
 for iid, d, cap, ppl in conn.execute("""SELECT i.id, coalesce(i.doc_text, ''), coalesce(i.caption, ''),
 (SELECT string_agg(CASE WHEN s.score >= %s THEN p.name ELSE 'possibly ' || p.name END, ', ' ORDER BY s.score DESC)
 FROM image_person_scores s JOIN persons p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= %s)
 FROM images i WHERE i.doc_text IS NOT NULL""", (PERSON_SURE, PERSON_MIN)):
 t = f"People: {ppl}. {d}" if ppl else d
 docs[iid] = t; docs_cap[iid] = f"Caption: {cap} {t}" if cap else t
 P = people(conn)

 def as_reranked(q):
 links, _ = P.link(q)
 out = " " + norm(q) + " "
 for p, c, w in links:
 if c >= 0.5 and w:
 out = out.replace(f" {w} ", f" {P.names[p].lower()} ", 1)
 return out.strip()

 qs = sorted({q for q, _ in J})
 sets = {q: [(iid, g) for (qq, iid), g in J.items() if qq == q and iid in docs] for q in qs}
 sets = {q: v for q, v in sets.items() if any(g == 2 for _, g in v) and any(g == 0 for _, g in v)}
 n = 0
 with open(a.out + "_pairs.jsonl", "w") as f:
 for q, items in sets.items():
 qr = as_reranked(q)
 for iid, g in items:
 f.write(json.dumps({"query_raw": q, "query": qr, "image_id": iid, "grade": g, "doc": docs[iid], "doc_cap": docs_cap[iid]}) + "\n"); n += 1
 with open(a.out + "_docs.jsonl", "w") as f:
 for iid in docs:
 f.write(json.dumps({"image_id": iid, "doc": docs[iid], "doc_cap": docs_cap[iid]}) + "\n")
 with open(a.out + "_queries.jsonl", "w") as f:
 for q in qs:
 rel = [iid for (qq, iid), g in J.items() if qq == q and g == 2 and iid in docs]
 if rel:
 f.write(json.dumps({"query_raw": q, "query": as_reranked(q), "relevant": rel}) + "\n")
 print(f"{len(sets)} queries, {n} pairs, {len(docs)} docs -> {a.out}_*.jsonl")


if __name__ == "__main__":
 main()
