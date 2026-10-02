"""Can Jev (choice/noul only, no text generation) replace the Luna query parse? Test the structured parts on the 198 eval queries:
city (choice over known cities + none), has_person (noul), kind (choice). Compare with the cached Luna parse; measure latency."""
import json, os, time, statistics as st
from concurrent.futures import ThreadPoolExecutor
import requests
from app import config
from app.query_parse import KNOWN_CITIES, _load
from app.decide import JEV_URL, JEV_MODEL
from eval.run_eval import load_queries
hdr = {"Authorization": f"Bearer {os.environ['JEV_API_KEY']}", "Content-Type": "application/json"}
luna = _load()
qs = [q["query"] for q in load_queries()]
CITY = {c: f"the photo was taken in {c}" for c in KNOWN_CITIES}; CITY["none"] = "no city is named, or the city word is part of an organisation name"
KIND = {"person": "a named public figure only", "person_context": "a named person plus what they do/wear/where", "scene": "an event or situation, no names",
 "attribute": "objects, clothing, colours", "place": "a place type, city or landmark", "event_year": "a subject plus a year"}
def one(q):
 body = {"state": f'Search query typed by an Indian newspaper journalist: "{q}"', "model": JEV_MODEL, "questions": {
 "city": {"type": "choice", "instructions": "Which city does the query say the photo was taken in?", "criteria": CITY},
 "has_person": {"type": "noul", "instructions": "Does the query name a specific real public figure (politician, sportsperson, actor...)? Nicknames and misspellings count.", "criteria": {"true": "a specific named individual", "false": "no named individual"}},
 "kind": {"type": "choice", "instructions": "What kind of search is this?", "criteria": KIND}}}
 t = time.perf_counter(); r = requests.post(JEV_URL, json=body, headers=hdr, timeout=30); ms = (time.perf_counter() - t) * 1000
 a = r.json()["answers"]; return q, a["city"]["choice"], a["city"]["confidence"], a["has_person"]["noul"], a["kind"]["choice"], ms, r.json()["usage"]["input_tokens"]
with ThreadPoolExecutor(16) as ex: res = list(ex.map(one, qs))
ms = sorted(x[5] for x in res); print(f"{len(res)} queries, latency p50 {ms[len(ms)//2]:.0f} ms p95 {ms[int(.95*(len(ms)-1))]:.0f} ms, cost ${sum(x[6] for x in res)*0.042/1e6:.4f}")
agree_city = agree_person = n = 0; rows = []
for q, city, cc, hp, kind, _, _ in res:
 lp = luna.get(" ".join(q.lower().split()))
 if not lp: continue
 n += 1; lc = lp["city"] or "none"; jc = city if cc >= 0.5 else "none"
 agree_city += (lc == jc); agree_person += ((hp >= 0.5) == bool(lp["people"]))
 if lc != jc or (hp >= 0.5) != bool(lp["people"]): rows.append((q, f"luna city={lc} people={[p['name'] for p in lp['people']]}", f"jev city={jc}({cc:.2f}) person={hp:.2f} kind={kind}"))
print(f"agreement with Luna parse on {n} queries: city {agree_city/n:.0%}, has_person {agree_person/n:.0%}")
print("disagreements (first 15):"); [print(" ", r) for r in rows[:15]]
json.dump([dict(zip(["query","city","city_conf","has_person","kind","ms","tokens"], x)) for x in res], open("exp/jev_parse/probe.json", "w"))
