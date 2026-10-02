"""E10 vlm_rerank: GPT-6 Luna looks at the top-K thumbnails with the query and re-scores them ("accurate mode").

uv run python -m exp.vlm_rerank.run search # run the fast search once per eval query (rel_min=0), cache
uv run python -m exp.vlm_rerank.run point # pointwise Luna grades for top-30 candidates (cached per pair)
uv run python -m exp.vlm_rerank.run list # listwise: one multi-image call over the top-10 shown
uv run python -m exp.vlm_rerank.run eval # metrics vs baseline, agreement with the judge
uv run python -m exp.vlm_rerank.run latency # clean sequential latency sample (fresh calls)
Writes exp/vlm_rerank/*.jsonl caches and prints tables for results.md.
"""
import base64, io, json, random, re, statistics as st, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
torch.set_num_threads(4)
from openai import OpenAI
from PIL import Image
from app.config import THUMB_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT
from app.db import connect
from app import search as S
from eval.run_eval import load_queries
from eval.judge import grades

HERE = Path(__file__).parent
SEARCH_CACHE = HERE / "search_cache.json"
POINT_CACHE = HERE / "point_cache.jsonl"
LIST_CACHE = HERE / "list_cache.jsonl"
K_POINT, K_PRE, K_LIST = 20, 30, 10
REL_MIN = S.REL_MIN
YEAR_RE = r"\b(19[5-9]\d|20[0-4]\d)\b"
_lock = threading.Lock()
_client = OpenAI(timeout=60, max_retries=2)

POINT_PROMPT = """A journalist searched a news photo archive for: "{q}"
Does this photo satisfy that search? Judge what is visible. You may identify well-known public figures you recognise.
2 = clearly yes: every requested person, scene, object, attribute or place is shown
1 = partly: related, but a requested element is missing, unclear or only loosely matches
0 = no
If the search names a person, that person must visibly appear in the photo for 1 or 2. Be strict about 2."""
POINT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["grade", "reason"],
 "properties": {"grade": {"type": "integer", "enum": [0, 1, 2]}, "reason": {"type": "string"}}}
LIST_PROMPT = """A journalist searched a news photo archive for: "{q}"
You are shown {n} candidate photos, numbered 1..{n} in order. Return the numbers of the photos that satisfy the search
(every requested person, scene, object, attribute or place is shown; a named person must visibly appear), ranked best first,
in `ranking`. Leave out photos that do not satisfy it. Return an empty list if none do."""
LIST_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ranking"],
 "properties": {"ranking": {"type": "array", "items": {"type": "integer"}}}}


def b64_thumb(iid, side=512):
 im = Image.open(THUMB_DIR / f"{iid}.jpg").convert("RGB")
 im.thumbnail((side, side))
 buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
 return base64.b64encode(buf.getvalue()).decode()


def strip_year(q):
 return " ".join(re.sub(YEAR_RE, "", q).split())


def _cost(u):
 return (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6


def point_call(q, iid):
 t = time.perf_counter()
 r = _client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low", messages=[{"role": "user", "content": [
 {"type": "text", "text": POINT_PROMPT.format(q=strip_year(q))},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_thumb(iid)}", "detail": "low"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "g", "strict": True, "schema": POINT_SCHEMA}})
 out = json.loads(r.choices[0].message.content)
 return {"query": q, "image_id": iid, "grade": out["grade"], "reason": out["reason"],
 "ms": int((time.perf_counter() - t) * 1000), "cost": _cost(r.usage), "tin": r.usage.prompt_tokens, "tout": r.usage.completion_tokens}


def list_call(q, ids):
 t = time.perf_counter()
 content = [{"type": "text", "text": LIST_PROMPT.format(q=strip_year(q), n=len(ids))}]
 for k, iid in enumerate(ids, 1):
 content.append({"type": "text", "text": f"Photo {k}:"})
 content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_thumb(iid)}", "detail": "low"}})
 r = _client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low", messages=[{"role": "user", "content": content}],
 response_format={"type": "json_schema", "json_schema": {"name": "r", "strict": True, "schema": LIST_SCHEMA}})
 rk = [k for k in json.loads(r.choices[0].message.content)["ranking"] if 1 <= k <= len(ids)]
 seen, ranking = set(), []
 for k in rk:
 if k not in seen:
 seen.add(k); ranking.append(ids[k - 1])
 return {"query": q, "ids": ids, "ranking": ranking, "ms": int((time.perf_counter() - t) * 1000), "cost": _cost(r.usage),
 "tin": r.usage.prompt_tokens, "tout": r.usage.completion_tokens}


def load_jsonl(p):
 out = []
 if p.exists():
 for l in p.open():
 try:
 out.append(json.loads(l))
 except json.JSONDecodeError:
 pass
 return out


def append(p, row):
 with _lock, p.open("a") as f:
 f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------- stages
def stage_search():
 conn = connect(); S.search("warm up", conn=conn)
 qs = load_queries(); out = {}
 for q in qs:
 r = S.search(q["query"], conn=conn, rel_min=0.0, limit=60)
 out[q["query"]] = {"type": q["type"], "required": r["required"], "reranked_as": r["reranked_as"], "years": r["years"],
 "ms": r["timings"]["total_ms"],
 "results": [{"id": x["id"], "rel": x["relevance"], "p_ctx": x["p_ctx"]} for x in r["results"]]}
 SEARCH_CACHE.write_text(json.dumps(out))
 lat = sorted(v["ms"] for v in out.values())
 print(f"{len(out)} queries searched; fast search p50 {st.median(lat):.0f} ms p95 {lat[int(.95 * (len(lat) - 1))]:.0f} ms")


def stage_point():
 sc = json.loads(SEARCH_CACHE.read_text())
 have = {(r["query"], r["image_id"]) for r in load_jsonl(POINT_CACHE)}
 todo = [(q, x["id"]) for q, v in sc.items() for x in v["results"][:K_PRE] if (q, x["id"]) not in have]
 print(f"pointwise: {len(todo)} new pairs (~${len(todo) * 0.00025:.2f})")
 def one(p):
 try:
 append(POINT_CACHE, point_call(*p))
 except Exception as e:
 print("fail", p, str(e)[:80])
 with ThreadPoolExecutor(24) as ex:
 list(ex.map(one, todo))


def stage_list():
 sc = json.loads(SEARCH_CACHE.read_text())
 have = {r["query"] for r in load_jsonl(LIST_CACHE)}
 todo = []
 for q, v in sc.items():
 ids = [x["id"] for x in v["results"] if x["rel"] >= REL_MIN][:K_LIST]
 if q not in have and ids:
 todo.append((q, ids))
 print(f"listwise: {len(todo)} calls")
 def one(p):
 try:
 append(LIST_CACHE, list_call(*p))
 except Exception as e:
 print("fail", p[0], str(e)[:80])
 with ThreadPoolExecutor(12) as ex:
 list(ex.map(one, todo))


def variants(sc, point, lst):
 """query -> {variant: [image ids shown, ranked]}"""
 out = {}
 for q, v in sc.items():
 res = v["results"]
 base = [x for x in res if x["rel"] >= REL_MIN]
 pg = {iid: g for (qq, iid), g in point.items() if qq == q}
 def regrade(cands):
 keep = [x for x in cands if pg.get(x["id"], 0) >= 1]
 keep.sort(key=lambda x: (-pg[x["id"]], -x["rel"]))
 return keep
 p20 = regrade(base[:K_POINT]) + base[K_POINT:]
 p30 = regrade(res[:K_PRE])
 # pointwise after threshold, but fall back to pre-threshold candidates when the fast search shows nothing
 p20_fb = p20 if base else p30
 l10 = None
 if q in lst:
 byid = {x["id"]: x for x in base[:K_LIST]}
 l10 = [byid[i] for i in lst[q]["ranking"] if i in byid] + base[K_LIST:]
 out[q] = {"baseline": base, "point20": p20, "point20_fb": p20_fb, "point30_pre": p30, "list10": l10 if l10 is not None else base}
 return out


def stage_eval():
 conn = connect()
 sc = json.loads(SEARCH_CACHE.read_text())
 point = {(r["query"], r["image_id"]): r["grade"] for r in load_jsonl(POINT_CACHE)}
 prow = load_jsonl(POINT_CACHE)
 lst = {r["query"]: r for r in load_jsonl(LIST_CACHE)}
 V = variants(sc, point, lst)
 files = dict(conn.execute("SELECT id, file FROM images"))
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images"))
 # judge the union of everything any variant shows in its top 20, same protocol as eval/run_eval.py
 pairs = {(q, x["id"]) for q, vs in V.items() for v in vs.values() for x in v[:20]}
 names_for, year_of = {}, {}
 for q, v in sc.items():
 nm = v["required"]; ys = re.findall(YEAR_RE, q)
 if ys:
 year_of[q] = (int(min(ys)), int(max(ys)))
 ctx = " ".join(w for w in v["reranked_as"].split() if not any(w in n.lower().split() for n in nm)) if nm else v["reranked_as"]
 if nm or ys:
 names_for[q] = (nm or None, ctx)
 g = grades(pairs, files, names_for)
 for (qq, iid), gr in list(g.items()):
 if qq in year_of and gr:
 y = years.get(iid)
 if y is None or not (year_of[qq][0] <= y <= year_of[qq][1]):
 g[(qq, iid)] = 0

 def score(shown, q):
 top = shown[:10]
 val = lambda iid: {2: 1.0, 1: 0.5, 0: 0.0}.get(g.get((q, iid)), 0.0)
 rel_pool = {iid for (qq, iid), gr in g.items() if qq == q and gr == 2}
 p10 = sum(val(x["id"]) for x in top) / len(top) if top else None
 rec = len(rel_pool & {x["id"] for x in shown}) / len(rel_pool) if rel_pool else None
 junk = sum(1 for x in top if g.get((q, x["id"])) == 0)
 return p10, rec, (not shown and bool(rel_pool)), junk, len(shown)

 print(f"\n{'variant':<14}{'type':<16}{'n':>4}{'P@10':>7}{'R@pool':>8}{'empty':>7}{'junk/q':>8}{'shown':>7}")
 summary = {}
 for var in ("baseline", "point20", "point20_fb", "point30_pre", "list10"):
 rows = {}
 for q, vs in V.items():
 m = score(vs[var], q)
 for k in (sc[q]["type"], "ALL"):
 rows.setdefault(k, []).append(m)
 for k, ms in sorted(rows.items(), key=lambda kv: kv[0] == "ALL"):
 p = [m[0] for m in ms if m[0] is not None]; r = [m[1] for m in ms if m[1] is not None]
 line = (len(ms), st.mean(p) if p else 0, st.mean(r) if r else 0, sum(m[2] for m in ms), st.mean(m[3] for m in ms), st.mean(m[4] for m in ms))
 summary[(var, k)] = line
 print(f"{var:<14}{k:<16}{line[0]:>4}{line[1]:>7.2f}{line[2]:>8.2f}{line[3]:>7}{line[4]:>8.2f}{line[5]:>7.1f}")
 print()
 # agreement with the cached judge on 40 random queries
 rnd = random.Random(1); qs40 = rnd.sample(sorted(sc), 40)
 both = [(point[(q, i)], g[(q, i)]) for (q, i) in point if q in qs40 and g.get((q, i)) is not None]
 exact = sum(a == b for a, b in both) / len(both)
 bin_ = sum((a >= 1) == (b >= 1) for a, b in both) / len(both)
 rel2 = sum((a == 2) == (b == 2) for a, b in both) / len(both)
 primed = sum(1 for a, b in both if a >= 1 and b == 0); unprimed = sum(1 for a, b in both if a == 0 and b >= 1)
 print(f"agreement on {len(both)} pairs / 40 queries: exact {exact:.2f}, relevant-vs-not {bin_:.2f}, grade2-vs-not {rel2:.2f}; "
 f"rerank says relevant but judge says 0: {primed}; rerank 0 but judge relevant: {unprimed}")
 # by whether the query names a person
 named = {q for q, v in sc.items() if v["required"]}
 for lab, qs_ in (("named", named), ("unnamed", set(sc) - named)):
 b = [(point[(q, i)], g[(q, i)]) for (q, i) in point if q in qs_ and g.get((q, i)) is not None]
 if b:
 print(f" {lab}: {len(b)} pairs, relevant-vs-not agreement {sum((a >= 1) == (bb >= 1) for a, bb in b) / len(b):.2f}, "
 f"rerank relevant/judge 0: {sum(1 for a, bb in b if a >= 1 and bb == 0)}")
 # cost and per-call latency from the bulk run
 ms = sorted(r["ms"] for r in prow); cost = sum(r["cost"] for r in prow)
 print(f"\npointwise calls: {len(prow)}, ${cost / len(prow) * 1000:.2f} per 1k calls, ${cost / len(prow) * K_POINT * 1000:.2f} per 1k queries at K={K_POINT}, "
 f"per-call ms p50 {st.median(ms):.0f} p95 {ms[int(.95 * (len(ms) - 1))]:.0f}; tokens in/out per call {sum(r['tin'] for r in prow) // len(prow)}/{sum(r['tout'] for r in prow) // len(prow)}")
 lr = list(lst.values())
 if lr:
 lms = sorted(r["ms"] for r in lr); lc = sum(r["cost"] for r in lr)
 print(f"listwise calls: {len(lr)}, ${lc / len(lr) * 1000:.2f} per 1k queries, ms p50 {st.median(lms):.0f} p95 {lms[int(.95 * (len(lms) - 1))]:.0f}, "
 f"avg {st.mean(len(r['ranking']) for r in lr):.1f} of {st.mean(len(r['ids']) for r in lr):.1f} kept")
 # examples: queries where point30_pre changed the most vs baseline
 ex = []
 for q, vs in V.items():
 b, p = score(vs["baseline"], q), score(vs["point30_pre"], q)
 if b[0] is not None or p[0] is not None:
 ex.append((abs((p[0] or 0) - (b[0] or 0)), q, b, p))
 ex.sort(reverse=True)
 print("\nexamples (query, baseline P@10/shown -> point30_pre P@10/shown):")
 for d, q, b, p in ex[:8]:
 print(f" [{sc[q]['type']}] {q!r}: {b[0] if b[0] is not None else '-'}/{b[4]} -> {p[0] if p[0] is not None else '-'}/{p[4]} empty {b[2]}->{p[2]}")
 json.dump({f"{v}|{k}": s for (v, k), s in summary.items()}, (HERE / "summary.json").open("w"), indent=1)


def stage_latency(n=25):
 """Sequential, fresh calls: one query at a time, K parallel pointwise calls; and one listwise call."""
 sc = json.loads(SEARCH_CACHE.read_text())
 rnd = random.Random(3); qs = rnd.sample([q for q, v in sc.items() if sum(x["rel"] >= REL_MIN for x in v["results"]) >= 10], n)
 pw, lw = [], []
 for q in qs:
 ids = [x["id"] for x in sc[q]["results"][:K_POINT]]
 t = time.perf_counter()
 with ThreadPoolExecutor(K_POINT) as ex:
 list(ex.map(lambda i: point_call(q, i), ids))
 pw.append((time.perf_counter() - t) * 1000)
 t = time.perf_counter(); list_call(q, ids[:K_LIST]); lw.append((time.perf_counter() - t) * 1000)
 for lab, xs in (("pointwise K=20 (20 parallel calls)", pw), ("listwise K=10 (one call)", lw)):
 xs = sorted(xs); print(f"{lab}: p50 {st.median(xs):.0f} ms p95 {xs[int(.95 * (len(xs) - 1))]:.0f} ms max {xs[-1]:.0f} ms over {len(xs)} queries")


if __name__ == "__main__":
 {"search": stage_search, "point": stage_point, "list": stage_list, "eval": stage_eval, "latency": stage_latency}[sys.argv[1]]()
 sys.stdout.flush()
 import os; os._exit(0)
