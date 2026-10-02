"""Offline reranker comparison on already-judged (query, photo) pairs. No API cost.

uv run python -m eval.compare_rerankers
For each model: per-query ranking quality (AUC: does a relevant photo score above an irrelevant one),
precision/recall at threshold 0.2/0.5 against judge grade 2, and CPU latency for 100 documents.
Documents are built exactly as app/search.py builds them (people + description).
"""
import json, statistics as st, time, sys
import numpy as np
from sentence_transformers import CrossEncoder
from app.db import connect
from app.search import PERSON_SURE, PERSON_MIN, people, norm, STOP
from eval.run_eval import load_queries

MODELS = sys.argv[1:] or ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2",
 "jinaai/jina-reranker-v1-turbo-en", "mixedbread-ai/mxbai-rerank-base-v1", "BAAI/bge-reranker-base"]


def main():
 conn = connect()
 J = {}
 for l in open("eval/judgments.jsonl"):
 r = json.loads(l); J[(r["query"], r["image_id"])] = r["grade"]
 qs = [q["query"] for q in load_queries()]
 docs = {}
 for iid, d, ppl in conn.execute("""SELECT i.id, coalesce(i.doc_text, ''),
 (SELECT string_agg(CASE WHEN s.score >= %s THEN p.name ELSE 'possibly ' || p.name END, ', ' ORDER BY s.score DESC)
 FROM image_person_scores s JOIN persons p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= %s)
 FROM images i""", (PERSON_SURE, PERSON_MIN)):
 docs[iid] = f"People: {ppl}. {d}" if ppl else d
 P = people(conn)
 def as_reranked(q): # same query rewrite as search()
 links, _ = P.link(q)
 out = " " + norm(q) + " "
 for p, c, w in links:
 if c >= 0.5 and w:
 out = out.replace(f" {w} ", f" {P.names[p].lower()} ", 1)
 return out.strip()
 sets = {q: [(iid, g) for (qq, iid), g in J.items() if qq == q and g is not None] for q in qs}
 sets = {q: v for q, v in sets.items() if any(g == 2 for _, g in v) and any(g == 0 for _, g in v)}
 print(f"{len(sets)} queries with both relevant and irrelevant judged photos, {sum(len(v) for v in sets.values())} pairs")
 sample = list(docs.values())[:100]
 print(f"{'model':<44}{'AUC':>6}{'P@.2':>7}{'R@.2':>7}{'P@.5':>7}{'R@.5':>7}{'ms/100':>8}")
 for name in MODELS:
 try:
 m = CrossEncoder(name, max_length=160, device="cpu", trust_remote_code=True)
 except Exception as e:
 print(f"{name:<44} failed to load: {str(e)[:60]}"); continue
 m.predict([("warm up", sample[0])])
 t = time.perf_counter(); m.predict([("people speaking at a press conference", d) for d in sample], batch_size=128)
 ms = (time.perf_counter() - t) * 1000
 aucs, allp = [], []
 for q, items in sets.items():
 raw = np.asarray(m.predict([(as_reranked(q), docs[i]) for i, _ in items], batch_size=128), dtype=np.float32)
 sc = raw if (raw.min() >= 0 and raw.max() <= 1) else 1 / (1 + np.exp(-raw))
 pos = [s for s, (_, g) in zip(sc, items) if g == 2]; neg = [s for s, (_, g) in zip(sc, items) if g == 0]
 aucs.append(np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos]))
 allp += [(s, g) for s, (_, g) in zip(sc, items)]
 def pr(t):
 shown = [g for s, g in allp if s >= t]; rel = [g for s, g in allp if g == 2]
 return (sum(g == 2 for g in shown) / len(shown) if shown else 0, sum(g == 2 for s, g in allp if s >= t) / len(rel))
 p2, r2 = pr(0.2); p5, r5 = pr(0.5)
 print(f"{name:<44}{st.mean(aucs):>6.3f}{p2:>7.2f}{r2:>7.2f}{p5:>7.2f}{r5:>7.2f}{ms:>8.0f}", flush=True)


if __name__ == "__main__":
 main()
