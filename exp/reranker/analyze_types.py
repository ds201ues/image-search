"""Per-query-type AUC and caption-vs-plain deltas for every scored model; embedder recall@50 from GPU embeddings."""
import glob, json, os, statistics as st, sys
import numpy as np
PAIRS = "exp/reranker/data/archive_pairs.jsonl"
pairs = [json.loads(l) for l in open(PAIRS)]
types = {}
for l in open("eval/gen_queries.jsonl"):
 q = json.loads(l); types[q["query"]] = q["type"]
for l in open("eval/hand_queries.txt"):
 if l.strip() and not l.startswith("#"):
 t, q = l.strip().split("|", 1); types[q.strip()] = t.strip()
idx = {}
for k, p in enumerate(pairs):
 idx.setdefault(p["query_raw"], []).append(k)
def auc_by_type(raw):
 out = {}
 for q, ks in idx.items():
 pos = [raw[k] for k in ks if pairs[k]["grade"] == 2]; neg = [raw[k] for k in ks if pairs[k]["grade"] == 0]
 if pos and neg:
 a = np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos])
 out.setdefault(types.get(q, "?"), []).append(a); out.setdefault("ALL", []).append(a)
 return {t: round(st.mean(v), 3) for t, v in out.items()}
# only queries whose docs differ (have a caption) for the caption delta
cap_q = {p["query_raw"] for p in pairs if p["doc_cap"] != p["doc"]}
rows = []
tset = ["ALL", "person", "person_context", "scene", "attribute", "place", "event_year", "hinglish"]
print(f"{'model':<48}{'var':<8}" + "".join(f"{t[:9]:>10}" for t in tset))
for f in sorted(glob.glob("exp/reranker/scores_archive/*.jsonl")):
 sc = [json.loads(l) for l in open(f)]; lab = os.path.basename(f)[:-6].replace("cross-encoder_", "").replace("__256", "").replace("__160", "@160")
 for v in ("raw_doc", "raw_cap"):
 r = auc_by_type(np.array([s[v] for s in sc]))
 rows.append({"model": lab, "variant": v, **r})
 print(f"{lab:<48}{'plain' if v=='raw_doc' else 'caption':<8}" + "".join(f"{r.get(t, float('nan')):>10.3f}" for t in tset))
json.dump(rows, open("exp/reranker/analysis_archive_types.json", "w"), indent=1)
# embedders: recall@50 of the desc-only source
print("\nembedder recall@50 (desc-only source), docs plain vs caption")
ids = json.load(open("exp/reranker/emb_archive/ids.json")); pos = {i: k for k, i in enumerate(ids["ids"])}
qs = [json.loads(l) for l in open("exp/reranker/data/archive_queries.jsonl")]
tim = json.load(open("exp/reranker/emb_archive/timing.json"))
for m in ["bge-small", "bge-base", "granite-small-r2", "qwen3-0.6b-384", "qwen3-0.6b-1024"]:
 Q = np.load(f"exp/reranker/emb_archive/{m}__q.npy")
 out = []
 for v in ("doc", "doc_cap"):
 E = np.load(f"exp/reranker/emb_archive/{m}__{v}.npy"); rec = []
 for q, qv in zip(qs, Q):
 rel = {i for i in q["relevant"] if i in pos}
 if not rel: continue
 top = np.argsort(-(E @ qv))[:50]; rec.append(len(rel & {ids["ids"][k] for k in top}) / len(rel))
 out.append(st.mean(rec))
 print(f"{m:<20} plain {out[0]:.3f} caption {out[1]:.3f} dim {E.shape[1]} gpu ms/doc {tim[m+'__doc']:.2f}")
