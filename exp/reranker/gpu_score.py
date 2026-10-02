"""Self-contained GPU scorer. Needs only: pip install sentence-transformers torch numpy (CUDA build).
No Postgres, no project imports. Copy this file + exp/reranker/data/*.jsonl to the box.

Mode 1, rerankers: python gpu_score.py rerank --pairs archive_pairs.jsonl --out scores/ [--models a@256 b@512 ...]
 writes scores/<model>__<maxlen>.jsonl, one line per pair: {i, raw_doc, raw_cap} (raw logits for doc and doc_cap)
 plus timing.json with GPU ms per 150 pairs (for the record; CPU latency is measured on the laptop).
Mode 2, embedders: python gpu_score.py embed --docs archive_docs.jsonl --queries archive_queries.jsonl --out emb/
 writes emb/<model>__doc.npy, <model>__doc_cap.npy (rows = docs order), <model>__q.npy (rows = queries order), ids.json
"""
import argparse, gc, json, os, time
import numpy as np, torch

RERANKERS = ["cross-encoder/ms-marco-MiniLM-L-6-v2@160", "cross-encoder/ms-marco-MiniLM-L-6-v2@256",
 "cross-encoder/ettin-reranker-17m-v1@256", "cross-encoder/ettin-reranker-32m-v1@256",
 "cross-encoder/ettin-reranker-68m-v1@256", "cross-encoder/ettin-reranker-150m-v1@256",
 "ibm-granite/granite-embedding-reranker-english-r2@256", "Alibaba-NLP/gte-reranker-modernbert-base@256",
 "mixedbread-ai/mxbai-rerank-base-v2@256", "mixedbread-ai/mxbai-rerank-large-v2@256",
 "BAAI/bge-reranker-v2-m3@256", "Qwen/Qwen3-Reranker-0.6B@256"]
# name: (hf id, query prefix or None, prompt_name or None, truncate_dim)
EMBEDDERS = {"bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", None, None),
 "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", None, None),
 "granite-small-r2": ("ibm-granite/granite-embedding-small-english-r2", "", None, None),
 "qwen3-0.6b-384": ("Qwen/Qwen3-Embedding-0.6B", None, "query", 384),
 "qwen3-0.6b-1024": ("Qwen/Qwen3-Embedding-0.6B", None, "query", None),
 "gte-base": ("Alibaba-NLP/gte-base-en-v1.5", "", None, None)}


def dev():
 return "cuda" if torch.cuda.is_available() else "cpu"


def rerank(a):
 pairs = [json.loads(l) for l in open(a.pairs)]
 os.makedirs(a.out, exist_ok=True)
 timing = {}
 from sentence_transformers import CrossEncoder
 for spec in a.models:
 name, _, ml = spec.partition("@"); ml = int(ml or 256)
 tag = f"{name.replace('/', '_')}__{ml}"
 if os.path.exists(f"{a.out}/{tag}.jsonl") and not a.force:
 print("skip", tag); continue
 try:
 m = CrossEncoder(name, max_length=ml, device=dev(), trust_remote_code=True,
 model_kwargs={"torch_dtype": torch.float16} if a.fp16 and dev() == "cuda" else {})
 except Exception as e:
 print(f"load failed {spec}: {str(e)[:120]}", flush=True); continue
 try:
 def pred(variant):
 raw = m.predict([(p["query"], p[variant]) for p in pairs], batch_size=a.batch, show_progress_bar=False)
 raw = np.asarray(raw, dtype=np.float64)
 if raw.ndim > 1:
 raw = raw[:, -1] if raw.shape[1] > 1 else raw[:, 0]
 return raw
 t = time.perf_counter(); rd = pred("doc"); t1 = time.perf_counter() - t
 rc = pred("doc_cap")
 sample = [(p["query"], p["doc"]) for p in pairs[:150]]
 m.predict(sample, batch_size=a.batch); 
 if dev() == "cuda": torch.cuda.synchronize()
 t = time.perf_counter(); m.predict(sample, batch_size=a.batch)
 if dev() == "cuda": torch.cuda.synchronize()
 timing[tag] = {"gpu_ms_150": (time.perf_counter() - t) * 1000, "total_s_doc": t1, "n": len(pairs)}
 with open(f"{a.out}/{tag}.jsonl", "w") as f:
 for i, (x, y) in enumerate(zip(rd, rc)):
 f.write(json.dumps({"i": i, "raw_doc": float(x), "raw_cap": float(y)}) + "\n")
 print(f"done {tag}: {len(pairs)} pairs x2, {timing[tag]['gpu_ms_150']:.0f} ms/150 on {dev()}", flush=True)
 except Exception as e:
 print(f"failed {spec}: {str(e)[:160]}", flush=True)
 del m; gc.collect(); torch.cuda.empty_cache() if dev() == "cuda" else None
 json.dump(timing, open(f"{a.out}/timing.json", "w"), indent=1)


def embed(a):
 docs = [json.loads(l) for l in open(a.docs)]
 queries = [json.loads(l) for l in open(a.queries)] if a.queries else []
 os.makedirs(a.out, exist_ok=True)
 json.dump({"ids": [d["image_id"] for d in docs], "queries": [q["query"] for q in queries]}, open(f"{a.out}/ids.json", "w"))
 from sentence_transformers import SentenceTransformer
 timing = {}
 for key in a.models:
 hf, qprefix, qprompt, trunc = EMBEDDERS[key]
 if os.path.exists(f"{a.out}/{key}__doc.npy") and not a.force:
 print("skip", key); continue
 try:
 m = SentenceTransformer(hf, device=dev(), trust_remote_code=True, **({"truncate_dim": trunc} if trunc else {}),
 model_kwargs={"torch_dtype": torch.float16} if a.fp16 and dev() == "cuda" else {})
 except Exception as e:
 print(f"load failed {key}: {str(e)[:120]}", flush=True); continue
 for variant in ("doc", "doc_cap"):
 t = time.perf_counter()
 E = m.encode([d[variant] for d in docs], batch_size=a.batch, normalize_embeddings=True, show_progress_bar=False)
 timing[f"{key}__{variant}"] = (time.perf_counter() - t) * 1000 / max(len(docs), 1)
 np.save(f"{a.out}/{key}__{variant}.npy", np.asarray(E, dtype=np.float32))
 if queries:
 qt = [(qprefix + q["query"]) if qprefix is not None else q["query"] for q in queries]
 Q = m.encode(qt, prompt_name=qprompt, normalize_embeddings=True, batch_size=a.batch, show_progress_bar=False)
 np.save(f"{a.out}/{key}__q.npy", np.asarray(Q, dtype=np.float32))
 print(f"done {key}: dim {E.shape[1]}, {timing[f'{key}__doc']:.2f} ms/doc on {dev()}", flush=True)
 del m; gc.collect(); torch.cuda.empty_cache() if dev() == "cuda" else None
 json.dump(timing, open(f"{a.out}/timing.json", "w"), indent=1)


if __name__ == "__main__":
 ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="mode", required=True)
 r = sub.add_parser("rerank"); r.add_argument("--pairs", required=True); r.add_argument("--out", default="scores")
 r.add_argument("--models", nargs="*", default=RERANKERS); r.add_argument("--batch", type=int, default=64)
 r.add_argument("--fp16", action="store_true"); r.add_argument("--force", action="store_true")
 e = sub.add_parser("embed"); e.add_argument("--docs", required=True); e.add_argument("--queries", default=None); e.add_argument("--out", default="emb")
 e.add_argument("--models", nargs="*", default=list(EMBEDDERS)); e.add_argument("--batch", type=int, default=64)
 e.add_argument("--fp16", action="store_true"); e.add_argument("--force", action="store_true")
 a = ap.parse_args()
 print("device:", dev(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
 rerank(a) if a.mode == "rerank" else embed(a)
