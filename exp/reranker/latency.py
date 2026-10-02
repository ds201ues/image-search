"""Idle-laptop CPU latency for one reranker: 150 (query, doc) pairs, 4 threads, best of 5.
PYTHONPATH=. uv run python exp/reranker/latency.py <hf id> <max_length>"""
import json, sys, time, torch
torch.set_num_threads(4)
from sentence_transformers import CrossEncoder
name, ml = sys.argv[1], int(sys.argv[2])
pairs = [json.loads(l) for l in open("exp/reranker/data/archive_pairs.jsonl")][:150]
m = CrossEncoder(name, max_length=ml, device="cpu", trust_remote_code=True)
for variant in ("doc", "doc_cap"):
 batch = [(p["query"], p[variant]) for p in pairs]
 m.predict(batch[:16], batch_size=64)
 ts = []
 for _ in range(5):
 t = time.perf_counter(); m.predict(batch, batch_size=64); ts.append((time.perf_counter() - t) * 1000)
 print(json.dumps({"model": name, "max_length": ml, "docs": variant, "ms_150_best": round(min(ts)), "ms_150_median": round(sorted(ts)[2]), "threads": 4}))
