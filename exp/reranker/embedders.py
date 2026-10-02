"""Description-embedder benchmark for the `desc` recall source.
PYTHONPATH=. uv run python exp/reranker/embedders.py [--recall]
Encodes every doc_text with each model (saved to exp/reranker/desc_<model>.npy), times encoding and query latency,
and with --recall measures recall@50 of the desc-only source against judged relevant photos (grade 2) in eval/judgments.jsonl.
"""
import argparse, gc, json, re, statistics as st, time
import numpy as np, torch
from app.db import connect
torch.set_num_threads(4)
MODELS = {"bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", None, None),
 "granite-small-r2": ("ibm-granite/granite-embedding-small-english-r2", "", None, None),
 "qwen3-0.6b-384": ("Qwen/Qwen3-Embedding-0.6B", None, "query", 384)}


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--recall", action="store_true"); ap.add_argument("--models", default=",".join(MODELS))
 a = ap.parse_args()
 conn = connect()
 rows = conn.execute("SELECT id, coalesce(doc_text,''), coalesce(caption,'') FROM images WHERE doc_text IS NOT NULL ORDER BY id").fetchall()
 ids = [r[0] for r in rows]; docs = [r[1] for r in rows]; docs_cap = [f"Caption: {c} {d}" if c else d for _, d, c in rows]
 print(f"{len(ids)} docs")
 J = {}
 for l in open("eval/judgments.jsonl"):
 try:
 r = json.loads(l)
 except json.JSONDecodeError:
 continue
 if r.get("grade") == 2:
 J.setdefault(r["query"], set()).add(r["image_id"])
 from sentence_transformers import SentenceTransformer
 report = []
 for key in a.models.split(","):
 hf, qprefix, qprompt, trunc = MODELS[key]
 kw = {"truncate_dim": trunc} if trunc else {}
 m = SentenceTransformer(hf, device="cpu", trust_remote_code=True, **kw)
 for variant, D in (("desc", docs), ("desc+caption", docs_cap)):
 t = time.perf_counter()
 E = m.encode(D, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
 enc_ms = (time.perf_counter() - t) * 1000 / len(D)
 np.save(f"exp/reranker/desc_{key}_{variant}.npy", E.astype(np.float32))
 qs = list(J) or ["protest outside court"]
 t = time.perf_counter()
 for q in qs[:20]:
 m.encode([qprefix + q] if qprefix is not None else [q], prompt_name=qprompt, normalize_embeddings=True)
 q_ms = (time.perf_counter() - t) * 1000 / min(20, len(qs))
 rec = None
 if a.recall and J:
 recs = []
 qv = m.encode([(qprefix + q) if qprefix is not None else q for q in qs], prompt_name=qprompt, normalize_embeddings=True, batch_size=32)
 idpos = {i: k for k, i in enumerate(ids)}
 for q, v in zip(qs, qv):
 rel = {i for i in J[q] if i in idpos}
 if not rel:
 continue
 top = np.argsort(-(E @ v))[:50]
 recs.append(len(rel & {ids[k] for k in top}) / len(rel))
 rec = st.mean(recs)
 row = dict(model=key, variant=variant, dim=E.shape[1], enc_ms_per_doc=round(enc_ms, 1), query_ms=round(q_ms, 1), recall50=rec)
 report.append(row); print(row, flush=True)
 del m; gc.collect()
 json.dump(report, open("exp/reranker/embedders.json", "w"), indent=1)


if __name__ == "__main__":
 main()
