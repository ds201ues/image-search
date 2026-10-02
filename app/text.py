"""Text models, local CPU.
- embed_docs / embed_query: description-meaning vectors (bge-small-en-v1.5, 384-d) for recall
- rerank: cross-encoder that reads (query, description) and returns calibrated relevance 0..1 for precision
"""
import threading
import numpy as np
from .config import TEXT_EMBED_MODEL, RERANK_MODEL, RERANK_MAX_LEN

_lock = threading.Lock()
_emb = _rr = None
QUERY_PREFIX = "Represent this sentence for searching relevant passages: " # bge convention


def _load_emb():
 global _emb
 with _lock:
 if _emb is None:
 from sentence_transformers import SentenceTransformer
 _emb = SentenceTransformer(TEXT_EMBED_MODEL, device="cpu")
 return _emb


def _load_rr():
 global _rr
 with _lock:
 if _rr is None:
 from sentence_transformers import CrossEncoder
 _rr = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LEN, device="cpu") # 256 so the caption + description both fit
 return _rr


def embed_docs(texts: list[str]) -> np.ndarray:
 return _load_emb().encode(texts, batch_size=32, normalize_embeddings=True)


def embed_query(q: str) -> np.ndarray:
 return _load_emb().encode([QUERY_PREFIX + q], normalize_embeddings=True)[0]


def rerank(q: str, docs: list[str]) -> np.ndarray:
 if not docs:
 return np.zeros(0)
 logits = _load_rr().predict([(q, d) for d in docs], batch_size=128)
 return 1 / (1 + np.exp(-np.asarray(logits, dtype=np.float32)))


def doc_text(v: dict | None, caption: str | None = None) -> str:
 """What the reranker and description embedding read: photographer caption first (who/where/what), then the VLM description."""
 if not v and not caption:
 return ""
 v = v or {}
 cap = f"Caption: {' '.join(caption.split())}. " if caption else ""
 people = "; ".join(", ".join(x for x in (p.get("description"), p.get("clothing"), p.get("action")) if x)
 for p in v.get("people") or [])
 parts = [v.get("summary", ""),
 f"People: {people}." if people else "",
 f"Place: {v.get('place_type', '')} {v.get('location_guess', '')}.",
 f"Event: {v.get('event_type', '')}." if v.get("event_type") else "",
 f"Activities: {', '.join(v.get('activities') or [])}." if v.get("activities") else "",
 f"Objects: {', '.join(v.get('objects') or [])}." if v.get("objects") else "",
 f"Landmarks: {', '.join(v.get('landmarks') or [])}." if v.get("landmarks") else "",
 f"Text in image: {', '.join(v.get('visible_text') or [])}." if v.get("visible_text") else "",
 f"Type: {v.get('image_type', '')}."]
 return cap + " ".join(p for p in parts if p.strip())
