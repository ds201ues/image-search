"""Query understanding: GPT-6 Luna (text only) turns a journalist's query into people, years, city, an English rewrite and
expansions. Cached per query in data/query_parses.jsonl (p50 ~2 s uncached, so the cache matters; see PLAN.md).
"""
import json, threading, time
from openai import OpenAI
from .config import DATA_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT

CACHE = DATA_DIR / "query_parses.jsonl"
_lock = threading.Lock()
_cache = None
_client = None
stats = {"calls": 0, "usd": 0.0}
KNOWN_CITIES = ["mumbai", "bengaluru", "chennai", "ahmedabad", "chandigarh", "delhi", "patna", "kolkata", "lucknow", "hyderabad",
 "london", "nagpur", "pune", "coimbatore", "jaipur", "mysuru", "visakhapatnam", "mangaluru", "hubballi", "vijayawada",
 "paris", "kochi", "melbourne", "berlin", "shanghai", "madurai", "beijing", "dubai", "bhopal", "srinagar",
 "thiruvananthapuram", "aurangabad", "tokyo", "madrid", "ranchi", "tiruchirappalli", "rio de janeiro", "glasgow"]
PROMPT = """You parse search queries typed by Indian newspaper journalists into a photo archive search system.
Queries are short (2-8 words), often lowercase, and may mix Hindi/Hinglish words or transliterations ("kisan andolan" = farmers protest,
"mela" = fair, "ke baad" = after, "par" = on, "wali" = wearing/with), nicknames ("didi" = Mamata Banerjee, "namo" = Narendra Modi),
and misspellings ("amit sha" = Amit Shah). Known city names in this archive: {cities}.

Return:
- people: well-known public figures the query refers to, each with canonical full English name (as on Wikipedia) and the surface words used.
 Do NOT invent people from generic words ("actor", "cm", "minister", "students"). Only include real named individuals.
- city: one of the known cities if the query clearly names one (map Bangalore→bengaluru, Mysore→mysuru, New Delhi→delhi, Bombay→mumbai), else "".
 A city mentioned as part of an organisation name (e.g. "iit madras", "bangalore mirror") does NOT count unless the photo would be taken there.
- year_from / year_to: integers if the query gives a year or range, else 0.
- event: the type of event or situation in a few English words ("protest", "press conference", "flood", "cricket match"), or "".
- visual: visual attributes in English (clothing, colours, objects, actions), or "".
- search_text: the whole query rewritten as natural English suitable for searching photo descriptions: translate Hinglish, expand
 nicknames to full names, fix misspellings, drop the year and the city word (they are handled separately), keep everything else.
 Keep it short (2-10 words). Do not add facts that are not in the query.
- expansions: 3-6 extra English search words or synonyms a description might use instead (e.g. rally → gathering, crowd, demonstration).
Query: "{q}"
"""
SCHEMA = {"type": "object", "additionalProperties": False,
 "required": ["people", "city", "year_from", "year_to", "event", "visual", "search_text", "expansions"],
 "properties": {
 "people": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["name", "surface"],
 "properties": {"name": {"type": "string"}, "surface": {"type": "string"}}}},
 "city": {"type": "string"}, "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
 "event": {"type": "string"}, "visual": {"type": "string"}, "search_text": {"type": "string"},
 "expansions": {"type": "array", "items": {"type": "string"}}}}


def _load():
 global _cache
 with _lock:
 if _cache is None:
 _cache = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); _cache[r["query"]] = r["parsed"]
 except (json.JSONDecodeError, KeyError):
 pass
 return _cache


def cached(q: str):
 """The parse if it is already cached, else None (no API call, no thread)."""
 key = " ".join(q.lower().split())
 c = _load()
 return {**c[key], "cached": True} if key in c else None


def parse(q: str) -> dict:
 """Cached parse. Never raises: on API failure returns an empty parse so search degrades to the plain query."""
 global _client
 c = _load()
 key = " ".join(q.lower().split())
 if key in c:
 return {**c[key], "cached": True}
 t = time.perf_counter()
 try:
 _client = _client or OpenAI(timeout=20, max_retries=1)
 r = _client.chat.completions.create(
 model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": PROMPT.format(q=key, cities=", ".join(KNOWN_CITIES))}],
 response_format={"type": "json_schema", "json_schema": {"name": "parse", "strict": True, "schema": SCHEMA}})
 out = json.loads(r.choices[0].message.content)
 except Exception as e:
 return {"people": [], "city": "", "year_from": 0, "year_to": 0, "event": "", "visual": "", "search_text": "",
 "expansions": [], "_ms": round((time.perf_counter() - t) * 1000), "error": str(e)[:120]}
 out["city"] = out["city"].lower().strip()
 out["_ms"] = round((time.perf_counter() - t) * 1000)
 with _lock:
 stats["calls"] += 1
 stats["usd"] += (r.usage.prompt_tokens * VLM_PRICE_IN + r.usage.completion_tokens * VLM_PRICE_OUT) / 1e6
 c[key] = out
 with CACHE.open("a") as f:
 f.write(json.dumps({"query": key, "parsed": out}) + "\n")
 return {**out, "cached": False}
