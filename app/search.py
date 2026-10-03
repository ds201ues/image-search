"""Two-stage hybrid search.

Stage 1, recall (cheap, generous): candidates from
 - SigLIP 2 text->image vectors (how it looks)
 - description-meaning vectors (what the caption + description mean)
 - Postgres full-text (exact words, incl. caption, city aliases)
 - people index (who is in it: Luna guesses + caption names + face-cluster votes, noisy-OR)
 fused with reciprocal rank fusion before the candidate cap (people candidates are never cut).
Stage 2, precision: a decision model (Jev) or cross-encoder reads (query, caption + description) -> P(context) 0..1.
 relevance = P(context) x P(each required person is in the photo)
Results come in two tiers: "match" (relevance >= TIER_MATCH) and "possible" (>= REL_MIN), so a query is rarely empty
while the top tier stays clean.

Query understanding runs in parallel with the first recall pass: the LLM parse (cached) gives canonical names, years and an
English rewrite of Hinglish; if it returns within the time box and changes the query, recall runs once more (cheap) before
stage 2. People are REQUIRED only when the parse names them (or, without a parse, when a whole name matches the directory);
ordinary words never link to a person.
"""
import os as _os, re, time, unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout
from difflib import get_close_matches
from .db import connect
from .embed import embed_text
from .text import embed_query
from .decide import rerank
from .config import QUERY_PARSE
from .query_parse import parse as parse_query
from .aliases import expand_cities, CITY_ALIASES

# Experiments swap these for their own copies (exp_* tables/views) without touching the live index.
TABLES = {"images": "images", "scores": "image_person_scores", "persons": "persons", "faces": "faces"}
PARSE_TIMEOUT_MS = int(_os.getenv("PARSE_TIMEOUT_MS", "700")) # uncached LLM parse takes ~2 s; past this we answer without it
_parse_pool = ThreadPoolExecutor(8)

POOL = 50 # candidates per recall source
MAX_RERANK = 150 # cap on candidates read by stage 2 (latency)
RRF_K = 60
_BACKEND = _os.getenv("RERANK_BACKEND", "minilm")
# thresholds per stage-2 backend (sweeps 2026-10-02: eval/integrated_report.txt, eval/jev_batched_report.txt)
REL_MIN = {"minilm": 0.30, "jev": 0.10}.get(_BACKEND, 0.3) # show at all ("possible" tier)
TIER_MATCH = {"minilm": 0.60, "jev": 0.50}.get(_BACKEND, 0.5) # top tier ("match")
PERSON_MIN = 0.50 # belief a named person is in the image for it to count as "has the person"
PERSON_SURE = 0.80 # below this the UI marks the person as "possible"
STOP = {"the", "a", "an", "in", "on", "at", "of", "and", "with", "to", "for", "is", "are", "image", "images",
 "photo", "photos", "picture", "pictures", "pic", "show", "me", "give", "find", "wearing", "near",
 "during", "inside", "outside", "from", "people", "person", "group", "some", "while", "together", "by"}
YEAR_RE = r"\b(19[5-9]\d|20[0-4]\d)\b"


def norm(s: str) -> str:
 s = unicodedata.normalize("NFKD", s)
 s = "".join(ch for ch in s if not unicodedata.combining(ch)) # Arsène -> Arsene, not "ars ne"
 return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


class People:
 """Entity linker: names -> people directory (names + aliases).
 With a parse, only the people the parse names are linked (exact or close full-name match).
 Without a parse, whole full names in the query link; a single word links only if it is a complete name word in the
 directory AND the archive shows the person in most photos whose description uses that word. A word nobody's
 description uses gets a weak prior (< 0.5), so it can widen recall but never REQUIRE the person."""

 WORD_PRIOR = 0.4

 def __init__(self, conn):
 self.conn = conn
 rows = conn.execute(f"""SELECT p.id, p.name, p.aliases, p.popularity,
 (SELECT count(*) FROM {TABLES["scores"]} s WHERE s.person_id = p.id AND s.score >= 0.5)
 FROM {TABLES["persons"]} p""").fetchall()
 self.names = {pid: name for pid, name, *_ in rows}
 rank = {pid: (seen, pop or 0) for pid, _, _, pop, seen in rows}
 better = lambda a, b: a if a is not None and rank[a] >= rank[b] else b
 self.full, self.token = {}, {}
 for pid, name, aliases, _, _ in rows:
 for ph in [name, *(aliases or [])]:
 ph = norm(re.sub(r"\(.*?\)", "", ph))
 if not ph:
 continue
 self.full[ph] = better(self.full.get(ph), pid)
 for t in ph.split():
 if len(t) >= 3 and t not in STOP:
 self.token[t] = better(self.token.get(t), pid)
 self.full_keys = list(self.full)

 def word_prior(self, word, pid):
 n, both = self.conn.execute(f"""
 SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {TABLES["scores"]} s
 WHERE s.image_id = i.id AND s.person_id = %s AND s.score >= 0.5))
 FROM {TABLES["images"]} i WHERE i.tsv @@ plainto_tsquery('english', %s)""", (pid, word)).fetchone()
 if n == 0:
 return self.WORD_PRIOR
 a = 0.25
 return (both + a * self.WORD_PRIOR) / (n + a)

 def link_name(self, name: str):
 """Canonical name from the parse -> (person_id, confidence) or None."""
 ph = norm(name)
 if ph in self.full:
 return self.full[ph], 1.0
 m = get_close_matches(ph, self.full_keys, 1, 0.86)
 if m:
 return self.full[m[0]], 0.9
 # surname-only directory entries ("Modi") or the parse giving a longer form: match if one contains the other
 toks = set(ph.split())
 cands = [k for k in self.full_keys if set(k.split()) <= toks or toks <= set(k.split())]
 if len(cands) == 1:
 return self.full[cands[0]], 0.8
 return None

 def link(self, q: str, parsed: dict | None = None):
 """-> ([(person_id, link_probability, words_used)], rest_of_query)"""
 qn = " " + norm(q) + " "
 out = []
 if parsed is not None:
 for pp in parsed.get("people") or []:
 hit = self.link_name(pp.get("name", ""))
 sf = norm(pp.get("surface", ""))
 if hit and hit[0] not in [p for p, _, _ in out]:
 out.append((hit[0], hit[1], sf))
 if sf and f" {sf} " in qn:
 qn = qn.replace(f" {sf} ", " ", 1)
 rest = " ".join(w for w in qn.split() if w not in STOP)
 return out, rest
 toks, rest, i = qn.split(), [], 0
 while i < len(toks):
 hit = None
 for n in (3, 2):
 if i + n <= len(toks):
 ph = " ".join(toks[i:i + n])
 if ph in self.full:
 hit = (self.full[ph], 1.0, n); break
 m = get_close_matches(ph, self.full_keys, 1, 0.86)
 if m:
 hit = (self.full[m[0]], 0.9, n); break
 t = toks[i]
 if not hit and t in self.token and t not in STOP:
 hit = (self.token[t], self.word_prior(t, self.token[t]), 1) # exact whole word only, no fuzzy
 if hit and hit[1] >= 0.05:
 if hit[0] not in [p for p, _, _ in out]:
 out.append((hit[0], hit[1], " ".join(toks[i:i + hit[2]])))
 i += hit[2]
 else:
 rest.append(t); i += 1
 return out, " ".join(w for w in rest if w not in STOP)


_people = None


def people(conn, refresh=False):
 global _people
 if _people is None or refresh:
 _people = People(conn)
 return _people


def _date_sql(year_from, year_to):
 sql, args = [], []
 if year_from:
 sql.append("i.date_published >= make_date(%s, 1, 1)"); args.append(year_from)
 if year_to:
 sql.append("i.date_published < make_date(%s, 1, 1)"); args.append(year_to + 1)
 return (" AND " + " AND ".join(sql)) if sql else "", args


def _understand(conn, q, parsed, year_from, year_to, person_ids):
 """Query -> required people, years, the text for recall and stage 2."""
 P = people(conn)
 q_eff = q
 if parsed:
 qn = " " + norm(q) + " "
 for pp in parsed.get("people") or []:
 sf, nm = norm(pp.get("surface", "")), pp.get("name", "")
 if sf and nm and f" {sf} " in qn:
 qn = qn.replace(f" {sf} ", f" {norm(nm)} ", 1)
 extra = [w for w in norm(parsed.get("search_text") or "").split() if w not in qn.split() and w not in STOP]
 q_eff = (qn.strip() + " " + " ".join(extra)).strip()
 if parsed.get("year_from") and not (year_from or year_to):
 year_from, year_to = parsed["year_from"], parsed.get("year_to") or parsed["year_from"]
 links, rest = P.link(q_eff, parsed)
 links += [(p, 1.0, "") for p in (person_ids or []) if p not in [x for x, _, _ in links]]
 years = [int(y) for y in re.findall(YEAR_RE, q_eff)]
 if years and not (year_from or year_to):
 year_from, year_to = min(years), max(years)
 strong = [(p, c, w) for p, c, w in links if c >= 0.5]
 ctx = " " + norm(re.sub(YEAR_RE, "", q_eff)) + " "
 for p, _, w in strong:
 if w and f" {w} " in ctx:
 ctx = ctx.replace(f" {w} ", f" {P.names[p].lower()} ", 1)
 ctx = " ".join(ctx.split())
 expansions = " ".join(parsed.get("expansions") or []) if parsed else ""
 ctx_text = expand_cities((ctx + " " + expansions).strip()) # word matching gets synonyms + city variants
 # stage 2 reads a clean English sentence: required people by canonical name + the parse's rewrite (no Hinglish/union noise)
 if parsed and parsed.get("search_text"):
 names = [P.names[p] for p, _, _ in strong]
 rewrite = norm(re.sub(YEAR_RE, "", parsed["search_text"]))
 for n in names:
 rewrite = rewrite.replace(norm(n), " ")
 ctx_rr = " ".join((" ".join(n.lower() for n in names) + " " + rewrite).split()) or ctx
 else:
 ctx_rr = ctx
 if parsed and parsed.get("city") in CITY_ALIASES and parsed["city"] not in ctx_text:
 ctx_text += " " + " ".join(CITY_ALIASES[parsed["city"]])
 return dict(q_eff=q_eff, links=links, strong=strong, pids=[p for p, _, _ in links], year_from=year_from, year_to=year_to,
 ctx=ctx, ctx_text=ctx_text, ctx_rr=ctx_rr)


def _recall(conn, U, use):
 cand, why, pscore = {}, {}, {}
 dsql, dargs = _date_sql(U["year_from"], U["year_to"])

 def add(name, ids):
 for r, iid in enumerate(ids):
 cand.setdefault(iid, 0.0); cand[iid] += 1 / (RRF_K + r + 1)
 why.setdefault(iid, []).append(f"{name}#{r + 1}")

 if U["pids"] and "person" in use:
 for iid, pid, sc in conn.execute(f"""
 SELECT s.image_id, s.person_id, s.score FROM {TABLES["scores"]} s JOIN {TABLES["images"]} i ON i.id = s.image_id
 WHERE s.person_id = ANY(%s) AND s.score >= 0.3 {dsql}""", (U["pids"], *dargs)):
 pscore.setdefault(iid, {})[pid] = sc
 add("person", sorted(pscore, key=lambda i: -min(pscore[i].values())))
 if "vector" in use:
 qv = embed_text([U["ctx"]])[0]
 add("vector", [r[0] for r in conn.execute(
 f"SELECT id FROM {TABLES['images']} i WHERE clip IS NOT NULL {dsql} ORDER BY clip <=> %s LIMIT {POOL}", (*dargs, qv))])
 if "desc" in use:
 dv = embed_query(U["ctx_text"])
 add("desc", [r[0] for r in conn.execute(
 f"SELECT id FROM {TABLES['images']} i WHERE desc_emb IS NOT NULL {dsql} ORDER BY desc_emb <=> %s LIMIT {POOL}", (*dargs, dv))])
 if "text" in use:
 add("text", [r[0] for r in conn.execute(f"""
 WITH q AS (SELECT to_tsquery('english', array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | ')) q)
 SELECT id FROM {TABLES['images']} i, q WHERE q.q IS NOT NULL AND tsv @@ q.q {dsql}
 ORDER BY ts_rank_cd(tsv, q.q, 32) DESC LIMIT {POOL}""", (U["ctx_text"], *dargs))])
 # reciprocal rank fusion before the cap; people candidates are never cut
 ids = sorted(cand, key=lambda i: (i not in pscore, -cand[i]))[:MAX_RERANK]
 return ids, why, pscore


def search(q: str, limit=60, year_from=None, year_to=None, person_ids=None, conn=None,
 use=("vector", "desc", "text", "person", "rerank"), rel_min=None, parse=None):
 rel_min = REL_MIN if rel_min is None else rel_min
 parse = QUERY_PARSE if parse is None else parse
 conn = conn or connect()
 T, t0 = {}, time.perf_counter()
 t = t0

 def lap(k):
 nonlocal t
 now = time.perf_counter(); T[k] = round((now - t) * 1000, 1); t = now

 # ---- parse in the background; first recall pass on the raw query right away
 fut = _parse_pool.submit(parse_query, q) if parse else None
 parsed = None
 if fut and fut.done(): # cached: use it from the start
 try:
 parsed = fut.result()
 except Exception:
 parsed = None
 U = _understand(conn, q, parsed, year_from, year_to, person_ids)
 lap("parse_ms")
 ids, why, pscore = _recall(conn, U, use)
 lap("recall_ms")
 if fut and parsed is None: # wait for what is left of the time box
 remaining = PARSE_TIMEOUT_MS / 1000 - (time.perf_counter() - t0)
 try:
 parsed = fut.result(timeout=max(0.0, remaining))
 except (_Timeout, Exception):
 parsed = None
 if parsed:
 U2 = _understand(conn, q, parsed, year_from, year_to, person_ids)
 if (U2["ctx_text"], U2["ctx_rr"], U2["pids"], U2["year_from"], U2["year_to"]) != (U["ctx_text"], U["ctx_rr"], U["pids"], U["year_from"], U["year_to"]):
 U = U2
 ids, why, pscore = _recall(conn, U, use)
 lap("parse_llm_ms")

 # ---- stage 2: precision
 docs = {}
 if ids:
 for iid, d, ppl in conn.execute(f"""
 SELECT i.id, coalesce(i.doc_text, ''),
 (SELECT string_agg(CASE WHEN s.score >= %s THEN p.name ELSE 'possibly ' || p.name END, ', ' ORDER BY s.score DESC)
 FROM {TABLES["scores"]} s JOIN {TABLES["persons"]} p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= %s)
 FROM {TABLES["images"]} i WHERE i.id = ANY(%s)""", (PERSON_SURE, PERSON_MIN, ids)):
 docs[iid] = f"People: {ppl}. {d}" if ppl else d
 if "rerank" in use:
 pc = dict(zip(ids, rerank(U["ctx_rr"], [docs.get(i, "") for i in ids])))
 else: # ablation without stage 2: recall rank as a stand-in
 pc = {i: 1 / (1 + min(int(w.split("#")[1]) for w in why[i] if "#" in w) / 10) for i in ids}
 lap("rerank_ms")

 results = []
 for i in ids:
 pp = 1.0
 for p, _, _ in U["strong"]:
 pp *= pscore.get(i, {}).get(p, 0.0)
 rel = float(pc[i]) * pp
 if rel >= rel_min:
 results.append({"id": i, "relevance": round(rel, 4), "p_ctx": round(float(pc[i]), 4),
 "p_person": round(pp, 4) if U["strong"] else None, "why": why.get(i, []),
 "tier": "match" if rel >= TIER_MATCH else "possible"})
 results.sort(key=lambda r: r["relevance"], reverse=True)
 results = results[:limit]
 meta = {r["id"]: r for r in results}
 rows = hydrate(conn, [r["id"] for r in results])
 for r in rows:
 r.update(meta[r["id"]])
 lap("fetch_ms")
 T["total_ms"] = round(sum(T.values()), 1)

 names = people(conn).names
 pids = U["pids"]
 present = {p for (p,) in conn.execute(f"SELECT DISTINCT person_id FROM {TABLES['scores']} WHERE person_id = ANY(%s) AND score >= %s",
 (pids, PERSON_MIN))} if pids else set()
 unlinked = [pp["name"] for pp in (parsed or {}).get("people") or [] if people(conn).link_name(pp.get("name", "")) is None]
 return {"query": q, "query_used": U["q_eff"], "parsed": parsed,
 "persons": [{"name": names[p], "p_link": round(c, 2), "words": w} for p, c, w in U["links"]],
 "persons_missing": [names[p] for p in pids if p not in present] + unlinked,
 "required": [names[p] for p, _, _ in U["strong"]], "reranked_as": U["ctx_rr"],
 "years": [U["year_from"], U["year_to"]], "timings": T, "count": len(rows), "candidates": len(ids),
 "matches": sum(1 for r in results if r["tier"] == "match"), "results": rows}


def hydrate(conn, ids):
 if not ids:
 return []
 rows = conn.execute(f"""
 SELECT i.id, i.file, i.width, i.height, i.date_published, i.section, i.source_url, i.ref_title, i.vlm, i.vlm_model,
 (SELECT coalesce(json_agg(json_build_object('name', p.name, 'score', round(s.score::numeric, 2),
 'evidence', s.evidence) ORDER BY s.score DESC), '[]')
 FROM {TABLES["scores"]} s JOIN {TABLES["persons"]} p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= 0.3),
 (SELECT coalesce(json_agg(json_build_object('name', p.name, 'score', round(f.match_score::numeric, 2), 'bbox', f.bbox,
 'cluster', f.cluster_id)), '[]')
 FROM {TABLES["faces"]} f LEFT JOIN {TABLES["persons"]} p ON p.id = f.person_id WHERE f.image_id = i.id),
 i.caption, i.city, i.byline
 FROM {TABLES["images"]} i WHERE i.id = ANY(%s)""", (ids,)).fetchall()
 by = {r[0]: r for r in rows}
 out = []
 for iid in ids:
 r = by[iid]
 ppl = r[10]
 out.append({"id": r[0], "file": r[1], "thumb": f"/thumbs/{r[0]}.jpg", "full": f"/media/{r[1]}",
 "width": r[2], "height": r[3], "date": r[4].date().isoformat() if r[4] else None,
 "section": r[5], "source_url": r[6], "ref_title": r[7], "vlm": r[8], "vlm_model": r[9],
 "people": [p["name"] for p in ppl if p["score"] >= PERSON_SURE],
 "people_possible": [p["name"] for p in ppl if PERSON_MIN <= p["score"] < PERSON_SURE],
 "people_sources": ppl, "faces": r[11], "caption": r[12], "city": r[13], "byline": r[14]})
 return out


def similar(image_id: str, limit=40, conn=None):
 conn = conn or connect()
 t = time.perf_counter()
 ids = [r[0] for r in conn.execute(f"""SELECT id FROM {TABLES["images"]} WHERE clip IS NOT NULL AND id <> %s
 ORDER BY clip <=> (SELECT clip FROM {TABLES["images"]} WHERE id = %s) LIMIT %s""", (image_id, image_id, limit))]
 ms = round((time.perf_counter() - t) * 1000, 1)
 res = hydrate(conn, ids)
 return {"query": f"similar to {image_id}", "timings": {"vector_ms": ms, "total_ms": ms}, "count": len(res), "results": res}
