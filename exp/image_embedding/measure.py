"""Measure candidate image-text models against the baseline (siglip2-base in images.clip).
 python measure.py source # vector-source-only recall@50 vs judged pool, all models
 python measure.py full fgclip2_base # full system with the vector source swapped (judges new pairs)
"""
import sys, json, re, statistics as st, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(2) # laptop is thermally limited; only the reranker runs locally
sys.path.insert(0, str(Path(__file__).parent))
from app.db import connect
from app import search as S
from eval.run_eval import load_queries, VARIANTS, keep
from eval.judge import grades, spent, _load

HERE = Path(__file__).parent
COLS = "id, file, width, height, date_published, section, source_url, ref_title, vlm, vlm_model, doc_text, desc_emb, tsv, search_text, caption, city"


def relevant_sets(qs):
 g = _load()
 rel = {}
 for (q, iid), gr in g.items():
 if gr == 2:
 rel.setdefault(q, set()).add(iid)
 return {q["query"]: rel.get(q["query"], set()) for q in qs}


def qtype_buckets(qs):
 b = {}
 for q in qs:
 b.setdefault(q["type"], []).append(q["query"])
 b.setdefault("len<=3" if len(q["query"].split()) <= 3 else "len>3", []).append(q["query"])
 b.setdefault("ALL", []).append(q["query"])
 return b


def embedder(key):
 """Query vectors come from the GPU run (exp_image_embedding_<key>_queries); no local model inference."""
 conn = connect()
 Q = {q: np.asarray(v) for q, v in conn.execute(f"SELECT query, emb FROM exp_image_embedding_{key}_queries")}
 def et(ts):
 return np.stack([Q[t] for t in ts]) # KeyError = query string missing from queries.txt: add it and re-run the GPU step
 if key == "siglip2_base":
 return et, "images", "clip"
 return et, f"exp_image_embedding_{key}", "emb"


def source_recall(conn, qs, keys, k=50):
 rel = relevant_sets(qs); buckets = qtype_buckets(qs)
 out = {}
 for key in keys:
 et, tbl, col = embedder(key)
 idcol = "id" if tbl == "images" else "image_id"
 rec = {}
 for q in qs:
 if not rel[q["query"]]:
 continue
 qv = et([S.norm(q["query"])])[0]
 top = [r[0] for r in conn.execute(f"SELECT {idcol} FROM {tbl} ORDER BY {col} <=> %s LIMIT {k}", (qv,))]
 rec[q["query"]] = len(set(top) & rel[q["query"]]) / len(rel[q["query"]])
 out[key] = {b: round(st.mean([rec[x] for x in xs if x in rec]), 3) for b, xs in buckets.items() if any(x in rec for x in xs)}
 print(key, json.dumps(out[key]), flush=True)
 return out


def full_system(conn, qs, key):
 et, tbl, col = embedder(key)
 view = f"exp_image_embedding_images_{key}"
 if tbl != "images":
 conn.execute(f"DROP VIEW IF EXISTS {view}")
 conn.execute(f"CREATE VIEW {view} AS SELECT {COLS}, e.emb AS clip FROM images i JOIN {tbl} e ON e.image_id = i.id")
 S.TABLES["images"] = view
 S.embed_text = et
 files = dict(conn.execute("SELECT id, file FROM images"))
 S.search("warm up", conn=conn)
 runs, lat = {}, []
 for q in qs:
 for v, use in VARIANTS.items():
 if v in ("desc", "fulltext"):
 continue # unchanged by the image model; pooled already by the baseline run
 r = S.search(q["query"], conn=conn, use=use, rel_min=0.0 if v == "full" else None)
 runs[(v, q["query"])] = r["results"]
 if v == "full":
 lat.append(r["timings"]["total_ms"])
 pairs = {(q["query"], x["id"]) for q in qs for (v, qq), res in runs.items() if qq == q["query"] for x in res[:20]}
 names_for, year_of = {}, {}
 for q in qs:
 r = S.search(q["query"], conn=conn, limit=1)
 nm = r["required"]; ys = re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q["query"])
 if ys: year_of[q["query"]] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in r["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else r["reranked_as"]
 if nm or ys: names_for[q["query"]] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]): g[(qq, iid)] = 0
 print(f"{key}: judged {len(pairs)} pairs, new judging ${spent['usd']:.3f}")

 def score(results, q):
 top = results[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 relp = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(relp & {x["id"] for x in results}) / len(relp) if relp else None
 return p10, rec, (not results and bool(relp)), sum(1 for x in top if g.get((q, x["id"])) == 0), len(results)

 def report(th, label):
 rows = {}
 for q in qs:
 m = score(keep(runs[("full", q["query"])], th), q["query"])
 for k in (q["type"], "ALL"): rows.setdefault(k, []).append(m)
 lines = [label, f"{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}"]
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 lines.append(f"{k:<16}{len(ms):>4}{(st.mean(p) if p else 0):>7.2f}{(st.mean(r) if r else 0):>8.2f}{sum(m[2] for m in ms):>7}{st.mean(m[3] for m in ms):>8.2f}{st.mean(m[4] for m in ms):>7.1f}")
 return "\n".join(lines)

 out = [report(S.REL_MIN, f"{key} FULL SYSTEM (threshold {S.REL_MIN})")]
 sip = [score(runs[("siglip", q["query"])][:60], q["query"]) for q in qs]
 p = [m[0] for m in sip if m[0] is not None]; r = [m[1] for m in sip if m[1] is not None]
 out.append(f" variant siglip-only P@10 {st.mean(p):.2f} R@pool {st.mean(r):.2f} junk/q {st.mean(m[3] for m in sip):.2f}")
 lat.sort(); out.append(f"latency ms: p50 {st.median(lat):.0f} p95 {lat[int(0.95*(len(lat)-1))]:.0f}")
 out.append("threshold sweep (ALL):")
 for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
 allm = [score(keep(runs[("full", q["query"])], t), q["query"]) for q in qs]
 p = [m[0] for m in allm if m[0] is not None]; r = [m[1] for m in allm if m[1] is not None]
 out.append(f" >= {t:.2f}: P@10 {st.mean(p):.2f} R@pool {st.mean(r):.2f} empty {sum(m[2] for m in allm)} junk/q {st.mean(m[3] for m in allm):.2f}")
 txt = "\n".join(out); print(txt)
 (HERE / f"full_{key}.txt").write_text(txt)
 json.dump({q["query"]: [x["id"] for x in keep(runs[("full", q["query"])], S.REL_MIN)[:10]] for q in qs}, open(HERE / f"top10_{key}.json", "w"))


if __name__ == "__main__":
 conn = connect(); qs = load_queries()
 if sys.argv[1] == "source":
 keys = sys.argv[2:] or ["siglip2_base", "fgclip2_base", "siglip2_so400m"]
 json.dump(source_recall(conn, qs, keys), open(HERE / "source_recall.json", "w"), indent=1)
 else:
 full_system(conn, qs, sys.argv[2])
 import os; sys.stdout.flush(); os._exit(0)
