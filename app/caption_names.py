"""Names in photographer captions -> people evidence. GPT-6 Luna text-only, cached per image in data/caption_names.jsonl.
Canonical spellings are merged with data/caption_persons.json (alias -> canonical), produced once by a Luna grouping pass.
"""
import json, re, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from .config import DATA_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT

CACHE = DATA_DIR / "caption_names.jsonl"
PERSONS = DATA_DIR / "caption_persons.json"
PROMPT = """Below is a photographer's caption (and maybe a headline) for a news photo in an Indian newspaper archive.
List the people NAMED in the text who are likely to appear IN the photo. Rules:
- Include politicians, officials, sportspeople, actors, business people, and any clearly named individual who is the subject.
- Exclude the photographer / photo credit ("Photo by X", "Pic X", "agency photo", agency names), and people named only as context (e.g. "a photo of Gandhi on the wall", "fans of Kohli" when Kohli is not present, "X's residence", "named after Y").
- canonical_full_name: the common full English name as a journalist would write it (expand initials/nicknames only if you are sure, e.g. "N Chandrababu Naidu" -> "N. Chandrababu Naidu", "Mamata" -> "Mamata Banerjee", "Nitish Kumar" stays).
- confidence 0-1 that this person is actually in the photo.
Return an empty list if nobody is named."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["people"],
 "properties": {"people": {"type": "array", "items": {"type": "object", "additionalProperties": False,
 "required": ["name_as_written", "canonical_full_name", "role_or_title", "confidence"],
 "properties": {"name_as_written": {"type": "string"}, "canonical_full_name": {"type": "string"},
 "role_or_title": {"type": "string"}, "confidence": {"type": "number"}}}}}}
_lock = threading.Lock()
spent = {"usd": 0.0, "calls": 0}


def _norm(s):
 return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace(".", " ")).split())


def _alias_map():
 if not PERSONS.exists():
 return {}
 m = {}
 for p in json.load(PERSONS.open()):
 for a in [p["name"], *p.get("aliases", [])]:
 m[_norm(a)] = p["name"]
 return m


def extract(rows, workers=32):
 """rows: [(image_id, caption, headline)] -> {image_id: [(canonical_name, confidence)]}. Only uncached rows call the API."""
 done = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); done[r["image_id"]] = r["people"]
 except (json.JSONDecodeError, KeyError):
 pass
 todo = [r for r in rows if r[0] not in done and (r[1] or r[2])]
 if todo:
 client = OpenAI(timeout=60, max_retries=3)
 def one(r):
 iid, cap, head = r
 text = (f"Headline: {head}\n" if head else "") + (f"Caption: {cap}" if cap else "")
 try:
 resp = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": PROMPT + "\n\n" + text[:3000]}],
 response_format={"type": "json_schema", "json_schema": {"name": "names", "strict": True, "schema": SCHEMA}})
 out = json.loads(resp.choices[0].message.content)["people"]; u = resp.usage
 with _lock:
 spent["usd"] += (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6; spent["calls"] += 1
 done[iid] = out
 with CACHE.open("a") as f:
 f.write(json.dumps({"image_id": iid, "people": out}) + "\n")
 except Exception as e:
 print(f" caption names fail {iid}: {e}")
 with ThreadPoolExecutor(workers) as ex:
 list(ex.map(one, todo))
 alias = _alias_map()
 out = {}
 for iid, _, _ in rows:
 ppl = {}
 for p in done.get(iid, []):
 name = (p.get("canonical_full_name") or p.get("name_as_written") or "").strip()
 if not name:
 continue
 name = alias.get(_norm(name), name)
 ppl[name] = max(ppl.get(name, 0), float(p.get("confidence", 0)))
 if ppl:
 out[iid] = list(ppl.items())
 return out
