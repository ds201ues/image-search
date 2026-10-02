"""Seed the face gallery from Wikidata: well-known people + one Commons photo each.

Two sources, merged:
 1. app/gallery_names.txt curated list of people that matter for the desk (reliable)
 2. Wikidata SPARQL bulk long tail of Indian public figures (best effort; the public endpoint
 times out often and Wikidata data is patchy, e.g. Modi has no citizenship)

uv run python -m app.gallery
"""
import argparse, time
import cv2, numpy as np, requests
from .db import connect, init_schema
from pathlib import Path
from .faces import detect_and_embed

SPARQL = "https://query.wikidata.org/sparql"
UA = {"User-Agent": "image-search-pilot/0.1 (internal newsroom research; python-requests)"}
# politician, cricketer, actor, film actor, businessperson, athlete, singer, journalist, judge
OCC = "wd:Q82955 wd:Q12299841 wd:Q33999 wd:Q10800557 wd:Q43845 wd:Q2066131 wd:Q177220 wd:Q1930187 wd:Q16533"

INDIA = f"""
SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (MAX(?l) AS ?links)
 (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {{
 ?p wdt:P31 wd:Q5; wdt:P27 wd:Q668; wdt:P18 ?img; wikibase:sitelinks ?l; wdt:P106 ?occ.
 VALUES ?occ {{ {OCC} }}
 FILTER NOT EXISTS {{ ?p wdt:P570 ?died }}
 FILTER(?l >= 25)
 OPTIONAL {{ ?p skos:altLabel ?alias FILTER(lang(?alias) = "en") }}
 SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}} GROUP BY ?p ?pLabel ORDER BY DESC(?links) LIMIT %d"""

# countries that dominate Indian world news: US, UK, Russia, China, Pakistan, Bangladesh, Sri Lanka, Nepal,
# Israel, Iran, Ukraine, France, Germany, Japan, Canada, Australia, UAE, Saudi Arabia, Afghanistan
WORLD = """
SELECT ?p ?pLabel (SAMPLE(?img) AS ?image) (MAX(?l) AS ?links)
 (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {
 VALUES ?c { wd:Q30 wd:Q145 wd:Q159 wd:Q148 wd:Q843 wd:Q902 wd:Q854 wd:Q837 wd:Q801 wd:Q794 wd:Q212
 wd:Q142 wd:Q183 wd:Q17 wd:Q16 wd:Q408 wd:Q878 wd:Q851 wd:Q889 }
 ?p wdt:P27 ?c; wdt:P31 wd:Q5; wdt:P106 wd:Q82955; wdt:P18 ?img; wikibase:sitelinks ?l.
 FILTER(?l >= 90)
 FILTER NOT EXISTS { ?p wdt:P570 ?died }
 OPTIONAL { ?p skos:altLabel ?alias FILTER(lang(?alias) = "en") }
 SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
} GROUP BY ?p ?pLabel ORDER BY DESC(?links) LIMIT %d"""


def polite_get(s, url, params=None, tries=6):
 """GET with backoff on 429/5xx, honouring Retry-After. Wikimedia rate-limits bursts."""
 for n in range(tries):
 r = s.get(url, params=params, headers=UA, timeout=60)
 if r.status_code == 200:
 return r
 if r.status_code in (429, 500, 502, 503, 504):
 wait = int(r.headers.get("Retry-After", 0) or 0) or 2 ** n * 2
 time.sleep(min(wait, 60)); continue
 r.raise_for_status()
 r.raise_for_status()
 return r


def sparql(q):
 r = requests.get(SPARQL, params={"query": q, "format": "json"}, headers=UA, timeout=120)
 r.raise_for_status()
 return r.json()["results"]["bindings"]


API = "https://www.wikidata.org/w/api.php"
NAMES_FILE = Path(__file__).with_name("gallery_names.txt")


def resolve_names(names):
 """Name -> Wikidata human with a photo, via the fast search API. Returns SPARQL-shaped rows."""
 s, out = requests.Session(), []
 for name in names:
 try:
 time.sleep(0.2)
 base = name.split(" (")[0]
 hits = polite_get(s, API, {"action": "wbsearchentities", "search": base, "language": "en",
 "type": "item", "limit": 5, "format": "json"}).json()
 ids = [h["id"] for h in hits.get("search", [])]
 if not ids:
 print(f"not found: {name}"); continue
 ents = polite_get(s, API, {"action": "wbgetentities", "ids": "|".join(ids), "props": "claims|labels|aliases|sitelinks",
 "languages": "en", "format": "json"}).json()["entities"]
 for qid in ids: # search order = relevance; take the first human with a photo
 e = ents.get(qid, {}); c = e.get("claims", {})
 human = any(x["mainsnak"].get("datavalue", {}).get("value", {}).get("id") == "Q5" for x in c.get("P31", []))
 if human and c.get("P18"):
 img = c["P18"][0]["mainsnak"]["datavalue"]["value"]
 out.append({"p": {"value": qid}, "pLabel": {"value": e["labels"].get("en", {}).get("value", name)},
 "image": {"value": "https://commons.wikimedia.org/wiki/Special:FilePath/" + img.replace(" ", "_")},
 "links": {"value": str(len(e.get("sitelinks", {})))},
 "aliases": {"value": "|".join(a["value"] for a in e.get("aliases", {}).get("en", []))}})
 break
 else:
 print(f"no photo/human: {name}")
 except Exception as ex:
 print(f"resolve fail {name}: {ex}")
 return out


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--limit", type=int, default=400, help="Indian people")
 ap.add_argument("--world", type=int, default=120, help="world politicians")
 ap.add_argument("--no-bulk", action="store_true", help="only the curated list")
 a = ap.parse_args()
 init_schema()
 names = [l.strip() for l in NAMES_FILE.open() if l.strip() and not l.startswith("#")]
 rows = resolve_names(names)
 print(f"curated: {len(rows)} of {len(names)} resolved")
 for name, q in ([] if a.no_bulk else [("india", INDIA % a.limit), ("world", WORLD % a.world)]):
 try:
 got = sparql(q); rows += got; print(f"{name}: {len(got)} people")
 except Exception as e:
 print(f"{name} query failed, skipping: {e}")
 print(f"{len(rows)} people from Wikidata")
 session = requests.Session()
 conn = connect()
 done = {r[0] for r in conn.execute("SELECT p.wikidata_id FROM persons p JOIN person_refs r ON r.person_id = p.id")}
 ok = 0
 for r in rows:
 qid = r["p"]["value"].rsplit("/", 1)[-1]
 name = r["pLabel"]["value"]
 if qid in done or name == qid:
 continue
 done.add(qid)
 aliases = [x for x in r.get("aliases", {}).get("value", "").split("|") if x]
 try:
 time.sleep(1.0)
 img = polite_get(session, r["image"]["value"] + "?width=500")
 bgr = cv2.imdecode(np.frombuffer(img.content, np.uint8), cv2.IMREAD_COLOR)
 faces = detect_and_embed(bgr) if bgr is not None else []
 except Exception as e:
 print(f"skip {name}: {e}"); continue
 if not faces:
 print(f"no face: {name}"); continue
 face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3]) # portrait: largest face
 pid = conn.execute("""
 INSERT INTO persons (name, aliases, wikidata_id, popularity) VALUES (%s, %s, %s, %s)
 ON CONFLICT (name) DO UPDATE SET aliases = EXCLUDED.aliases, popularity = EXCLUDED.popularity
 RETURNING id""", (name, aliases, qid, int(r["links"]["value"]))).fetchone()[0]
 conn.execute("INSERT INTO person_refs (person_id, emb, source) VALUES (%s, %s, %s)",
 (pid, face["emb"], "wikidata:" + qid))
 ok += 1
 if ok % 25 == 0:
 print(f"{ok} gallery faces ...")
 print(f"gallery: {ok} new people with reference faces")


if __name__ == "__main__":
 main()
