"""GPT-6 Luna as relevance judge, with a cache so each (query, image) pair is paid for once.

grade 2 = clearly satisfies the search, 1 = partly, 0 = no.

Names are judged BLIND: asked "is Kohli in this photo?", the model happily finds Kohli in a pilot's cockpit.
So for queries that name people, the judge first lists who it recognises without seeing the query
(cached per image), names are compared in code, and only the rest of the query is judged with the photo.
Caveat: the judge is the same model that guessed names at index time, so shared mistakes are not caught.
"""
import json, threading
from pathlib import Path
from openai import OpenAI
from app.config import DATA_DIR, THUMB_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT
from app.vlm import _jpeg_b64

CACHE = Path(__file__).parent / "judgments_v2.jsonl" # v2: judge sees caption + date (v1 image-only: judgments.jsonl)
BLIND = Path(__file__).parent / "blind_people.jsonl"
BLIND_PROMPT = """List the well-known public figures (politicians, sportspeople, actors, business leaders, officials)
you can recognise in this news photo, with confidence 0-1. Only include people you are reasonably sure about.
Return an empty list if you recognise nobody."""
BLIND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["people"],
 "properties": {"people": {"type": "array", "items": {"type": "object", "additionalProperties": False,
 "required": ["name", "confidence"],
 "properties": {"name": {"type": "string"}, "confidence": {"type": "number"}}}}}}
_blind = None
PROMPT = """A journalist searched a news photo archive for: "{q}"
Does this photo satisfy that search? Judge what is visible, well-known public figures you recognise, AND the archive's own
record of the photo below (the photographer's caption and the date are reliable evidence for place, event and time).
{meta}
2 = clearly yes: every requested person, scene, object, attribute or place is shown
1 = partly: related, but a requested element is missing, unclear or only loosely matches
0 = no
If the search names a person, that person must actually appear in the photo for 1 or 2."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["grade", "reason"],
 "properties": {"grade": {"type": "integer", "enum": [0, 1, 2]}, "reason": {"type": "string"}}}
_lock = threading.Lock()
_cache = None
_client = None
spent = {"usd": 0.0, "calls": 0}


def _load():
 global _cache
 if _cache is None:
 _cache = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); _cache[(r["query"], r["image_id"])] = r["grade"]
 except (json.JSONDecodeError, KeyError):
 pass # partial line from a concurrent writer
 return _cache


def _img(file):
 """Judge from the 768px thumbnail when it exists (same pixels the indexer saw, no 12 MB decode)."""
 tp = THUMB_DIR / (Path(str(file)).stem + ".jpg")
 return tp if tp.exists() else DATA_DIR / file


_meta = None


def photo_meta(iid):
 """Caption and date from the index, for the judge prompt."""
 global _meta
 with _lock:
 if _meta is None:
 from app.db import connect
 _meta = {i: (c, d) for i, c, d in connect().execute("SELECT id, caption, date_published::date FROM images")}
 c, d = _meta.get(iid, (None, None))
 lines = [f"Caption: {' '.join(c.split())}" if c else "Caption: (none)", f"Date: {d.isoformat()}" if d else "Date: unknown"]
 return "\n".join(lines)


def _call(text, file, schema, name):
 global _client
 _client = _client or OpenAI(timeout=60, max_retries=3)
 r = _client.chat.completions.create(
 model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": [
 {"type": "text", "text": text},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(_img(file))}"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}})
 with _lock:
 spent["usd"] += (r.usage.prompt_tokens * VLM_PRICE_IN + r.usage.completion_tokens * VLM_PRICE_OUT) / 1e6
 spent["calls"] += 1
 return json.loads(r.choices[0].message.content)


def blind_people(iid, file):
 """Who the judge recognises, without knowing any query. Cached per image."""
 global _blind
 with _lock:
 if _blind is None:
 _blind = {}
 if BLIND.exists():
 for l in BLIND.open():
 try:
 r = json.loads(l); _blind[r["image_id"]] = r["people"]
 except (json.JSONDecodeError, KeyError):
 pass
 if iid in _blind:
 return _blind[iid]
 ppl = [p for p in _call(BLIND_PROMPT, file, BLIND_SCHEMA, "people")["people"] if p["confidence"] >= 0.7]
 with _lock:
 _blind[iid] = ppl
 with BLIND.open("a") as f:
 f.write(json.dumps({"image_id": iid, "people": ppl}) + "\n")
 return ppl


def _same(a, b):
 from difflib import SequenceMatcher
 a, b = a.lower().strip(), b.lower().strip()
 return a == b or a in b or b in a or SequenceMatcher(None, a, b).ratio() >= 0.85


def _judge_one(q, iid, file, names=None, ctx=None):
 if names is None and ctx is not None and ctx != q: # no names, but the query carried a year: judge the rest
 q_judge = ctx or q
 out = _call(PROMPT.format(q=q_judge, meta=photo_meta(iid)), file, SCHEMA, "judge")
 with _lock:
 _cache[(q, iid)] = out["grade"]
 with CACHE.open("a") as f:
 f.write(json.dumps({"query": q, "image_id": iid, "grade": out["grade"], "reason": out["reason"], "judged_as": q_judge}) + "\n")
 return
 if names: # name check: blind recognition OR the photographer's caption naming the person; then judge only the context
 seen = [p["name"] for p in blind_people(iid, file)]
 cap = (photo_meta(iid).split("\n")[0] or "").lower()
 def named(n):
 if any(_same(n, s) for s in seen):
 return True
 toks = [t for t in n.lower().replace(".", " ").split() if len(t) > 2]
 return bool(toks) and all(t in cap for t in toks) # every name word appears in the caption
 if not all(named(n) for n in names):
 grade, reason = 0, f"blind ID saw {seen or 'nobody'}, not {names}"
 elif not ctx:
 grade, reason = 2, f"blind ID saw {names}"
 else:
 out = _call(PROMPT.format(q=ctx, meta=photo_meta(iid)), file, SCHEMA, "judge")
 grade, reason = out["grade"], f"names ok; context: {out['reason']}"
 with _lock:
 _cache[(q, iid)] = grade
 with CACHE.open("a") as f:
 f.write(json.dumps({"query": q, "image_id": iid, "grade": grade, "reason": reason, "blind": True}) + "\n")
 return
 global _client
 _client = _client or OpenAI(timeout=60, max_retries=3)
 r = _client.chat.completions.create(
 model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT.format(q=q, meta=photo_meta(iid))},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(_img(file))}"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "judge", "strict": True, "schema": SCHEMA}})
 out = json.loads(r.choices[0].message.content)
 with _lock:
 spent["usd"] += (r.usage.prompt_tokens * VLM_PRICE_IN + r.usage.completion_tokens * VLM_PRICE_OUT) / 1e6
 spent["calls"] += 1
 _cache[(q, iid)] = out["grade"]
 with CACHE.open("a") as f:
 f.write(json.dumps({"query": q, "image_id": iid, "grade": out["grade"], "reason": out["reason"]}) + "\n")


CALL_DEADLINE = 90 # seconds of wall clock per judgment, whatever the HTTP layer does


def grades(pairs, files, names_for=None, workers=32):
 """pairs: [(query, image_id)] -> {pair: grade or None}; judges only uncached pairs.
 names_for: query -> (person names in the query, rest of the query) for blind name judging.
 Uses daemon threads with a wall-clock deadline per call: a connection the proxy keeps open
 (seen with a TLS-intercepting corporate proxy: SSL read never returns despite client timeouts) is abandoned, not waited on.
 Unfinished pairs come back as None and are retried on the next run."""
 import queue, time
 cache = _load()
 todo = list({p for p in pairs if p not in cache})
 if todo:
 q, started = queue.Queue(), {}
 for p in todo:
 q.put(p)

 def worker():
 while True:
 try:
 p = q.get_nowait()
 except queue.Empty:
 return
 started[p] = time.time()
 _safe(p, files, (names_for or {}).get(p[0]))
 started.pop(p, None)

 threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
 for t in threads:
 t.start()
 last = 0
 while any(t.is_alive() for t in threads):
 time.sleep(2)
 now = time.time()
 stuck = [p for p, t0 in list(started.items()) if now - t0 > CALL_DEADLINE]
 alive = sum(t.is_alive() for t in threads)
 done = len(todo) - q.qsize() - len(started)
 if done - last >= 250:
 print(f" judged {done}/{len(todo)}", flush=True); last = done
 if stuck and len(stuck) >= alive: # every live worker is stuck: give up on them
 print(f" abandoning {len(stuck)} stuck calls, {q.qsize()} pairs left for the next run", flush=True)
 break
 if stuck and q.empty() and len(stuck) >= alive:
 break
 missing = sum(1 for p in pairs if p not in cache)
 if missing:
 print(f" WARNING: {missing} pairs not judged (re-run to fill them in)", flush=True)
 return {p: cache.get(p) for p in pairs}


def _safe(p, files, nm):
 try:
 _judge_one(p[0], p[1], files[p[1]], *(nm or (None, None)))
 except Exception as e:
 print(f"judge fail {p}: {e}")
