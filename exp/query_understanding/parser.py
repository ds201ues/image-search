"""E6: LLM query understanding. GPT-6 Luna (text only) turns a journalist's query into structured intent.
Cached per query in parsed.jsonl. Usage: from exp.query_understanding.parser import parse
"""
import json, re, threading, time
from pathlib import Path
from openai import OpenAI
from app.config import VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT

HERE = Path(__file__).parent
CACHE = HERE / "parsed.jsonl"
_lock = threading.Lock()
_cache = None
_client = None
stats = {"calls": 0, "usd": 0.0, "ms": []}

# city aliases seen in images.city (lowercase); the parser emits the canonical key, SQL matches any alias
CITY_ALIASES = {
 "bengaluru": ["bangalore", "bengaluru"], "mysuru": ["mysore", "mysuru"], "mangaluru": ["mangalore", "mangaluru"],
 "kolkata": ["kolkata", "kolkatta", "calcutta"], "delhi": ["delhi", "new delhi"], "mumbai": ["mumbai", "bombay", "navi mumbai"],
 "chennai": ["chennai", "madras"], "hubballi": ["hubli", "hubballi"], "thiruvananthapuram": ["thiruvananthapuram", "trivandrum"],
 "tiruchirappalli": ["trichy", "tiruchirappalli"], "kochi": ["kochi", "cochin"], "pune": ["pune", "poona"],
 "visakhapatnam": ["visakhapatnam", "vizag"], "ahmedabad": ["ahmedabad", "amdavad"], "patna": ["patna"],
}
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
 if _cache is None:
 _cache = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); _cache[r["query"]] = r["parsed"]
 except (json.JSONDecodeError, KeyError):
 pass
 return _cache


def parse(q: str) -> dict:
 global _client
 c = _load()
 key = " ".join(q.lower().split())
 if key in c:
 return c[key]
 _client = _client or OpenAI(timeout=60, max_retries=3)
 t = time.perf_counter()
 r = _client.chat.completions.create(
 model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": PROMPT.format(q=key, cities=", ".join(KNOWN_CITIES))}],
 response_format={"type": "json_schema", "json_schema": {"name": "parse", "strict": True, "schema": SCHEMA}})
 ms = (time.perf_counter() - t) * 1000
 out = json.loads(r.choices[0].message.content)
 out["city"] = out["city"].lower().strip()
 out["_ms"] = round(ms)
 with _lock:
 stats["calls"] += 1; stats["ms"].append(ms)
 stats["usd"] += (r.usage.prompt_tokens * VLM_PRICE_IN + r.usage.completion_tokens * VLM_PRICE_OUT) / 1e6
 c[key] = out
 with CACHE.open("a") as f:
 f.write(json.dumps({"query": key, "parsed": out}) + "\n")
 return out


def city_aliases(city: str) -> list[str]:
 return CITY_ALIASES.get(city, [city]) if city else []


if __name__ == "__main__":
 import sys, statistics as st
 from concurrent.futures import ThreadPoolExecutor
 sys.path.insert(0, ".")
 from eval.run_eval import load_queries
 qs = [q["query"] for q in load_queries()]
 with ThreadPoolExecutor(16) as ex:
 res = list(ex.map(parse, qs))
 if stats["ms"]:
 ms = sorted(stats["ms"])
 print(f"{stats['calls']} calls, ${stats['usd']:.4f}, latency p50 {ms[len(ms)//2]:.0f} ms p95 {ms[int(0.95*(len(ms)-1))]:.0f} ms")
 for q, p in list(zip(qs, res)):
 print(f"{q!r:48} -> {p['search_text']!r} people={[x['name'] for x in p['people']]} city={p['city']!r} "
 f"yrs={p['year_from']}-{p['year_to']} exp={p['expansions'][:4]}")
