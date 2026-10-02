"""E9: cost vs quality per variant.
 (a) field agreement with the baseline description
 (b) retrieval: exp_vlm_cost_images_<variant> = images with the variant's doc_text/search_text/desc_emb for the 300 sample photos,
 swapped into app.search.TABLES; rank of the source photo + judged P@10 on out/queries.jsonl.
uv run python -m exp.vlm_cost.measure [variants...]
"""
import json, sys, statistics as st
from difflib import SequenceMatcher
import torch; torch.set_num_threads(2)
from app.db import connect, flatten_vlm
from app import search as S
from app.text import doc_text, embed_docs
from eval.judge import grades, spent
from .variants import OUT, sample_ids

ALL = ["baseline", "detail_low", "no_reasoning", "thumb512", "compact", "model_gpt-5-nano", "batch"]


def load(v):
 p = OUT / f"{v}.jsonl"
 return {r["id"]: r for r in map(json.loads, p.open())} if p.exists() else {}


def expand(v: dict) -> dict:
 """Compact schema -> full-schema shape so flatten_vlm/doc_text work."""
 if "place" not in v:
 return v
 return {"recognized_people": v["recognized_people"], "summary": v["summary"], "image_type": "photo", "people_count": len(v["people"]),
 "people": [{"description": p, "clothing": "", "colors": [], "action": "", "expression": ""} for p in v["people"]],
 "setting": "", "place_type": v["place"], "location_guess": "", "landmarks": [], "objects": v["objects"],
 "activities": [], "event_type": v["event_type"], "visible_text": [], "mood": "", "time_of_day": "",
 "dominant_colors": [], "tags": v["tags"]}


def same(a, b):
 a, b = a.lower().strip(), b.lower().strip()
 return a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.85


def agreement(base, var):
 pc, place, names_p, names_r, jac, length = [], [], [], [], [], []
 for iid, b in base.items():
 if iid not in var: continue
 bv, vv = b["vlm"], expand(var[iid]["vlm"])
 pc.append(abs((bv.get("people_count") or 0) - (vv.get("people_count") or 0)) <= 1)
 place.append(same(bv.get("place_type") or "", vv.get("place_type") or "") if bv.get("place_type") else None)
 bn = {x["name"] for x in bv["recognized_people"] if x["confidence"] >= 0.6}; vn = {x["name"] for x in vv["recognized_people"] if x["confidence"] >= 0.6}
 if bn: names_r.append(len([n for n in bn if any(same(n, m) for m in vn)]) / len(bn))
 if vn: names_p.append(len([n for n in vn if any(same(n, m) for m in bn)]) / len(vn))
 bt, vt = set(map(str.lower, bv["tags"])), set(map(str.lower, vv["tags"]))
 jac.append(len(bt & vt) / max(1, len(bt | vt)))
 length.append(len(doc_text(vv).split()))
 f = lambda xs: st.mean([x for x in xs if x is not None]) if any(x is not None for x in xs) else float("nan")
 return dict(n=len(pc), people_count_ok=f(pc), place_ok=f(place), names_recall=f(names_r), names_precision=f(names_p),
 tags_jaccard=f(jac), doc_words=f(length))


def build_table(conn, v, var):
 t = f"exp_vlm_cost_images_{v.replace('-', '_').replace('.', '_')}"
 conn.execute(f"DROP TABLE IF EXISTS {t}")
 conn.execute(f"CREATE TABLE {t} AS SELECT * FROM images")
 conn.execute(f"ALTER TABLE {t} ADD PRIMARY KEY (id)")
 ids = list(var); vl = [expand(var[i]["vlm"]) for i in ids]
 docs = [doc_text(x) for x in vl]; embs = embed_docs(docs)
 with conn.cursor() as cur:
 cur.executemany(f"UPDATE {t} SET vlm = %s, doc_text = %s, desc_emb = %s, search_text = %s, tsv = to_tsvector('english', %s) WHERE id = %s",
 [(json.dumps(x), d, e, flatten_vlm(x), flatten_vlm(x), i) for x, d, e, i in zip(vl, docs, embs, ids)])
 conn.execute(f"CREATE INDEX ON {t} USING gin (tsv)"); conn.execute(f"CREATE INDEX ON {t} USING hnsw (desc_emb vector_cosine_ops)")
 conn.execute(f"CREATE INDEX ON {t} USING hnsw (clip vector_cosine_ops)")
 return t


def retrieval(conn, t, qs, files):
 S.TABLES["images"] = t
 runs = {q["query"]: S.search(q["query"], conn=conn, rel_min=0.0)["results"] for q in qs}
 S.TABLES["images"] = "images"
 pairs = {(q["query"], x["id"]) for q in qs for x in runs[q["query"]][:10]}
 names_for = {}
 for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 if r["required"]:
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in r["required"]))
 names_for[q["query"]] = (r["required"], ctx)
 g = grades(pairs, files, names_for)
 ranks, p10, hit10 = [], [], []
 for q in qs:
 res = [x for x in runs[q["query"]] if x["relevance"] >= S.REL_MIN]
 ids = [x["id"] for x in res]
 rk = ids.index(q["source_image"]) + 1 if q["source_image"] in ids else None
 ranks.append(rk); hit10.append(rk is not None and rk <= 10)
 top = res[:10]
 if top: p10.append(sum({2: 1, 1: .5}.get(g.get((q["query"], x["id"])), 0) for x in top) / len(top))
 return dict(src_hit10=st.mean(hit10), src_mrr=st.mean(1 / r if r else 0 for r in ranks), p10=st.mean(p10) if p10 else 0, shown=st.mean(len([x for x in runs[q["query"]] if x["relevance"] >= S.REL_MIN]) for q in qs))


def main():
 vs = sys.argv[1:] or ALL
 conn = connect(); S.search("warm up", conn=conn)
 base = load("baseline"); qs = [json.loads(l) for l in (OUT / "queries.jsonl").open()]
 files = dict(conn.execute("SELECT id, file FROM images"))
 rows = []
 for v in vs:
 var = load(v)
 if not var: print("skip", v); continue
 n = len(var)
 cost = 1000 * sum(r["cost"] for r in var.values()) / n
 tin = st.mean(r["tin"] for r in var.values() if r["tin"]) if any(r["tin"] for r in var.values()) else float("nan")
 tout = st.mean(r["tout"] for r in var.values() if r["tout"]) if any(r["tout"] for r in var.values()) else float("nan")
 ms = st.median(r["ms"] for r in var.values() if r["ms"]) if any(r["ms"] for r in var.values()) else float("nan")
 ag = agreement(base, var) if v != "baseline" else {}
 t = build_table(conn, v, var); rt = retrieval(conn, t, qs, files)
 rows.append(dict(variant=v, n=n, usd_per_1k=cost, tok_in=tin, tok_out=tout, ms_p50=ms, **ag, **rt))
 print(json.dumps(rows[-1]), flush=True)
 (OUT / "measure.json").write_text(json.dumps(rows, indent=1))
 print(f"judging cost ${spent['usd']:.3f}")
 hdr = ["variant", "n", "usd_per_1k", "tok_in", "tok_out", "ms_p50", "people_count_ok", "place_ok", "names_recall", "names_precision", "tags_jaccard", "doc_words", "src_hit10", "src_mrr", "p10", "shown"]
 print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
 for r in rows:
 print("| " + " | ".join(f"{r.get(h, ''):.3f}" if isinstance(r.get(h), float) else str(r.get(h, "")) for h in hdr) + " |")


if __name__ == "__main__":
 import os; main(); sys.stdout.flush(); os._exit(0)
