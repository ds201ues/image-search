"""Full-system eval with the reranker swapped (monkeypatched), on the archive DB.
PYTHONPATH=. uv run python exp/reranker/full_eval.py --model cross-encoder/ettin-reranker-68m-v1 --max-length 256 --calib a,b [--caption]
Reuses eval.run_eval query loading and eval.judge grading (append-only cache). Sweeps REL_MIN offline.
"""
import argparse, json, re, statistics as st, time
import numpy as np, torch
from app.db import connect
from app import search as S
from app import text as T
from eval.run_eval import load_queries
from eval.judge import grades, spent
torch.set_num_threads(4)


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--model", default=None); ap.add_argument("--max-length", type=int, default=256)
 ap.add_argument("--calib", default=None, help="a,b for p = sigmoid(a*raw+b); default: model's own sigmoid")
 ap.add_argument("--caption", action="store_true", help="prepend the IPTC caption to the reranker document")
 ap.add_argument("--tag", default="baseline")
 a = ap.parse_args()
 conn = connect()
 if a.model:
 from sentence_transformers import CrossEncoder
 m = CrossEncoder(a.model, max_length=a.max_length, device="cpu", trust_remote_code=True)
 ca, cb = (map(float, a.calib.split(","))) if a.calib else (1.0, 0.0)
 def rerank(q, docs):
 if not docs:
 return np.zeros(0)
 raw = np.asarray(m.predict([(q, d) for d in docs], batch_size=64), dtype=np.float32)
 return 1 / (1 + np.exp(-(ca * raw + cb)))
 S.rerank = rerank
 if a.caption:
 caps = dict(conn.execute("SELECT id, caption FROM images WHERE caption IS NOT NULL"))
 # the docs the reranker reads come from images.doc_text inside search(): use a copy with the caption prefixed
 conn.execute("DROP TABLE IF EXISTS exp_reranker_images_cap CASCADE")
 conn.execute("""CREATE TABLE exp_reranker_images_cap AS
 SELECT * FROM images""")
 conn.execute("""UPDATE exp_reranker_images_cap SET doc_text = 'Caption: ' || caption || ' ' || coalesce(doc_text,'') WHERE caption IS NOT NULL""")
 conn.execute("CREATE INDEX ON exp_reranker_images_cap USING hnsw (clip vector_cosine_ops)")
 conn.execute("CREATE INDEX ON exp_reranker_images_cap USING hnsw (desc_emb vector_cosine_ops)")
 conn.execute("CREATE INDEX ON exp_reranker_images_cap USING gin (tsv)")
 S.TABLES["images"] = "exp_reranker_images_cap"
 S.search("warm up", conn=conn)
 files = dict(conn.execute("SELECT id, file FROM images"))
 qs = load_queries()
 runs, lat = {}, []
 for q in qs:
 r = S.search(q["query"], conn=conn, rel_min=0.0)
 runs[q["query"]] = r["results"]; lat.append(r["timings"]["total_ms"])
 pairs = {(q["query"], x["id"]) for q in qs for x in runs[q["query"]][:20]}
 names_for, year_of = {}, {}
 for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 nm = r["required"]; ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
 if ys:
 year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys:
 names_for[q["query"]] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr and (years.get(iid) is None or not (year_of[qq][0] <= years[iid] <= year_of[qq][1])):
 g[(qq, iid)] = 0
 # pool of relevant photos = everything judged 2 for the query (shared cache across all experiments)
 pool = {}
 for l in open("eval/judgments.jsonl"):
 try:
 r = json.loads(l)
 except json.JSONDecodeError:
 continue
 if r.get("grade") == 2:
 pool.setdefault(r["query"], set()).add(r["image_id"])
 for (qq, iid), gr in g.items():
 if gr == 2:
 pool.setdefault(qq, set()).add(iid)
 print(f"judged pairs: {len(pairs)} (new judging cost ${spent['usd']:.3f} over {spent['calls']} calls)")

 def score(results, q):
 top = results[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 rel = pool.get(q, set())
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel & {x["id"] for x in results}) / len(rel) if rel else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not results and bool(rel)), junk, len(results)

 def report(th):
 rows = {}
 for q in qs:
 res = [x for x in runs[q["query"]] if x["relevance"] >= th][:60]
 m = score(res, q["query"])
 for k in (q["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 out = {}
 for k, ms in rows.items():
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 out[k] = dict(n=len(ms), p10=st.mean(p) if p else 0, rpool=st.mean(r) if r else 0, empty=sum(m[2] for m in ms),
 junk=st.mean(m[3] for m in ms), shown=st.mean(m[4] for m in ms))
 return out
 lat.sort()
 res = {"tag": a.tag, "model": a.model, "caption": a.caption, "calib": a.calib, "lat_p50": st.median(lat),
 "lat_p95": lat[int(0.95 * (len(lat) - 1))], "sweep": {str(t): report(t) for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)},
 "top10": {q["query"]: [x["id"] for x in runs[q["query"]][:10]] for q in qs}}
 json.dump(res, open(f"exp/reranker/full_{a.tag}.json", "w"), indent=1)
 print(f"latency p50 {res['lat_p50']:.0f} p95 {res['lat_p95']:.0f}")
 print(f"{'thr':>5}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}")
 for t, r in res["sweep"].items():
 A = r["ALL"]; print(f"{t:>5}{A['p10']:>7.2f}{A['rpool']:>8.2f}{A['empty']:>7}{A['junk']:>8.2f}{A['shown']:>7.1f}")
 best = max(res["sweep"].items(), key=lambda kv: kv[1]["ALL"]["p10"] - 0.02 * kv[1]["ALL"]["empty"])
 print(f"best threshold {best[0]} by type:")
 for k, r in sorted(best[1].items()):
 print(f" {k:<16}{r['n']:>4}{r['p10']:>7.2f}{r['rpool']:>8.2f}{r['empty']:>7}{r['junk']:>8.2f}")


if __name__ == "__main__":
 import os, sys
 main(); sys.stdout.flush(); os._exit(0)
