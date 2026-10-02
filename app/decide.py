"""Stage-2 relevance backends: P(this photo satisfies the query) for each candidate document.
- minilm: local cross-encoder (app.text.rerank), the default
- jev: TypeSafe AI's decision model (System One), one Noul question per (query, document), parallel calls, cached,
 hard budget cap (JEV_BUDGET_USD). Jev reads text only, so it judges caption + description, like the reranker.
"""
import hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from .config import DATA_DIR

JEV_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL = os.getenv("JEV_MODEL", "jev-latest")
JEV_PRICE_IN = 0.042 / 1e6 # USD per input token (output is free)
JEV_BUDGET_USD = float(os.getenv("JEV_BUDGET_USD", "2.0"))
JEV_WORKERS = int(os.getenv("JEV_WORKERS", "16"))
CACHE = DATA_DIR / "jev_cache.jsonl"
INSTRUCTIONS = ('A journalist searched a newspaper photo archive for: "{q}"\n'
 "The state is the photographer's caption and a description of one photo. Does this photo satisfy that search? "
 "Every requested person, place, event, object or attribute must be present; a named person must be in the photo.")
CRITERIA = {"true": "The photo clearly satisfies the whole search", "false": "Something requested is missing, unclear or only loosely related"}

_lock = threading.Lock()
_cache = None
spent = {"usd": 0.0, "calls": 0, "tokens": 0, "ms": []}


def _key(q, doc):
 return hashlib.sha1((q + "\x1f" + doc).encode()).hexdigest()


def _load():
 global _cache
 with _lock:
 if _cache is None:
 _cache = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); _cache[r["k"]] = r["p"]
 except (json.JSONDecodeError, KeyError):
 pass
 return _cache


def jev_one(q: str, doc: str, retries=4) -> float:
 """P(relevant) from one Noul question. Raises on budget exhaustion."""
 c = _load(); k = _key(q, doc)
 if k in c:
 return c[k]
 if spent["usd"] >= JEV_BUDGET_USD:
 raise RuntimeError(f"Jev budget cap ${JEV_BUDGET_USD} reached")
 body = {"state": doc[:4000], "model": JEV_MODEL,
 "questions": {"rel": {"type": "noul", "instructions": INSTRUCTIONS.format(q=q), "criteria": CRITERIA}}}
 hdr = {"Authorization": f"Bearer {os.environ['JEV_API_KEY']}", "Content-Type": "application/json"}
 for i in range(retries):
 t = time.perf_counter()
 r = requests.post(JEV_URL, json=body, headers=hdr, timeout=30)
 if r.status_code in (429, 529, 500, 502, 503):
 time.sleep(0.5 * 2 ** i); continue
 r.raise_for_status()
 out = r.json()
 p = float(out["answers"]["rel"]["noul"])
 with _lock:
 spent["calls"] += 1; spent["ms"].append((time.perf_counter() - t) * 1000)
 tok = out.get("usage", {}).get("input_tokens", 0); spent["tokens"] += tok; spent["usd"] += tok * JEV_PRICE_IN
 c[k] = p
 with CACHE.open("a") as f:
 f.write(json.dumps({"k": k, "p": p}) + "\n")
 return p
 raise RuntimeError(f"Jev: gave up after {retries} retries (last status {r.status_code})")


def rerank_jev(q: str, docs: list[str]) -> np.ndarray:
 if not docs:
 return np.zeros(0)
 with ThreadPoolExecutor(JEV_WORKERS) as ex:
 return np.asarray(list(ex.map(lambda d: jev_one(q, d), docs)), dtype=np.float32)


def rerank(q: str, docs: list[str], backend: str | None = None) -> np.ndarray:
 backend = backend or os.getenv("RERANK_BACKEND", "minilm")
 if backend == "jev":
 return rerank_jev(q, docs)
 from .text import rerank as rr
 return rr(q, docs)
