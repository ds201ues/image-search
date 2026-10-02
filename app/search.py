"""Two-stage hybrid search.

Stage 1, recall (cheap, generous): candidates from
 - SigLIP 2 text->image vectors (how it looks)
 - description-meaning vectors (what the description means)
 - Postgres full-text on description (exact words)
 - people index (who is in it: Luna guesses + face-cluster votes, noisy-OR)
Stage 2, precision: a cross-encoder reads (query, description) and returns calibrated relevance 0..1.

The reranker reads the query as typed (confidently linked names written in full: "mamta" -> "Mamata Banerjee")
against the description PLUS the people the index believes are in the photo, so it judges who and what together.
 relevance = reranker(query, people + description) x P(each confidently named person is in the photo)
Weakly linked words ("khan" in "khan market protest") stay plain words. One threshold on relevance.
"""
import re, time
from difflib import get_close_matches
from .db import connect, flatten_vlm
from .embed import embed_text
from .text import embed_query, rerank
from .config import QUERY_PARSE
from .query_parse import parse as parse_query

# Experiments swap these for their own copies (exp_* tables/views) without touching the live index.
TABLES = {"images": "images", "scores": "image_person_scores", "persons": "persons", "faces": "faces"}
POOL = 50 # candidates per recall source
MAX_RERANK = 150 # cap on candidates read by the reranker (latency)
REL_MIN = 0.50 # relevance needed to show an image; 0.5 from the 2026-10-02 sweep on the archive sample (eval/integrated_report.txt)
PERSON_MIN = 0.50 # belief a named person is in the image for it to count as "has the person"
PERSON_SURE = 0.80 # below this the UI marks the person as "possible"
STOP = {"the", "a", "an", "in", "on", "at", "of", "and", "with", "to", "for", "is", "are", "image", "images",
 "photo", "photos", "picture", "pictures", "pic", "show", "me", "give", "find", "wearing", "near",
 "during", "inside", "outside", "from", "people", "person", "group", "some", "while", "together", "by"}


def norm(s: str) -> str:
 return re.sub(r"[^a-z0-9 ]+", " ", s.lower()).strip()


class People:
 """Entity linker: query words -> people directory (names + aliases), with a link probability.
 Full names (exact / close misspelling) are unambiguous. For a single word ("trump", "yogi", "cricket")
 the probability is measured from the archive: of the photos whose description uses the word, how many
 show the person ("trump" -> high, "cricket" -> Smriti Mandhana: ~0). Words no description uses
 ("kohli", "mamta") get a neutral prior."""

 WORD_PRIOR = 0.8 # single name word nobody's description uses
 FUZZY = 0.85 # factor for a close misspelling

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
 if " " in ph:
 self.full[ph] = better(self.full.get(ph), pid)
 for t in ph.split():
 if len(t) >= 2 and t not in STOP: # "xi" is a name; the link prior handles ambiguity
 self.token[t] = better(self.token.get(t), pid)
 self.full_keys, self.token_keys = list(self.full), list(self.token)

 def word_prior(self, word, pid):
 n, both = self.conn.execute(f"""
 SELECT count(*), count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {TABLES["scores"]} s
 WHERE s.image_id = i.id AND s.person_id = %s AND s.score >= 0.5))
 FROM {TABLES["images"]} i WHERE i.tsv @@ plainto_tsquery('english', %s)""", (pid, word)).fetchone()
 if n == 0:
 return self.WORD_PRIOR
 # share of photos using the word that show the person, pulled toward WORD_PRIOR by a weak prior
 # (seen once without the person -> ~0.16; always with the person -> ~0.96)
 a = 0.25
 return (both + a * self.WORD_PRIOR) / (n + a)

 def link(self, q: str):
 """-> ([(person_id, link_probability, words_used)], rest_of_query)"""
 toks, out, rest, i = norm(q).split(), [], [], 0
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
 if not hit and len(t) >= 2 and t not in STOP:
 if t in self.token:
 hit = (self.token[t], self.word_prior(t, self.token[t]), 1)
 else:
 m = get_close_matches(t, self.token_keys, 1, 0.84) if len(t) >= 4 else []
 if m:
 hit = (self.token[m[0]], self.FUZZY * self.word_prior(m[0], self.token[m[0]]), 1)
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


def search(q: str, limit=60, year_from=None, year_to=None, person_ids=None, conn=None,
 use=("vector", "desc", "text", "person", "rerank"), rel_min=None, parse=None):
 rel_min = REL_MIN if rel_min is None else rel_min
 parse = QUERY_PARSE if parse is None else parse
 conn = conn or connect()
 T, t = {}, time.perf_counter()

 def lap(k):
 nonlocal t
 now = time.perf_counter(); T[k] = round((now - t) * 1000, 1); t = now

 # ---- LLM query understanding (cached): canonical names replace nicknames/misspellings, years become a filter,
 # the English rewrite is ADDED to the original words (never replaces them), expansions feed text recall only
 q_user, parsed, expansions = q, None, ""
 if parse:
 parsed = parse_query(q)
 qn = " " + norm(q) + " "
 for pp in parsed.get("people") or []:
 sf, nm = norm(pp.get("surface", "")), pp.get("name", "")
 if sf and nm and f" {sf} " in qn:
 qn = qn.replace(f" {sf} ", f" {norm(nm)} ", 1)
 extra = [w for w in norm(parsed.get("search_text") or "").split() if w not in qn.split() and w not in STOP]
 q = (qn.strip() + " " + " ".join(extra)).strip()
 if parsed.get("year_from") and not (year_from or year_to):
 year_from, year_to = parsed["year_from"], parsed.get("year_to") or parsed["year_from"]
 expansions = " ".join(parsed.get("expansions") or [])
 lap("parse_llm_ms")

 # ---- understand the query: possible people, years, the rest
 links, rest = people(conn).link(q)
 links += [(p, 1.0, "") for p in (person_ids or []) if p not in [x for x, _, _ in links]]
 pids = [p for p, _, _ in links]
 prior = 1.0
 for _, c, _ in links:
 prior *= c
 years = [int(y) for y in re.findall(r"\b(19[5-9]\d|20[0-4]\d)\b", q)]
 if years and not (year_from or year_to):
 year_from, year_to = min(years), max(years)
 rest = re.sub(r"\b(19[5-9]\d|20[0-4]\d)\b", "", rest).strip()
 strong = [(p, c, w) for p, c, w in links if c >= 0.5] # names we require in the photo
 names_all = people(conn).names
 q_rr = " " + norm(re.sub(r"\b(19[5-9]\d|20[0-4]\d)\b", "", q)) + " "
 for p, _, w in strong:
 if w:
 q_rr = q_rr.replace(f" {w} ", f" {names_all[p].lower()} ", 1)
 q_rr = q_rr.strip()
 dsql, dargs = _date_sql(year_from, year_to)
 lap("parse_ms")

 # ---- stage 1: recall
 cand, why = {}, {}

 def add(name, ids):
 for r, iid in enumerate(ids):
 cand.setdefault(iid, None); why.setdefault(iid, []).append(f"{name}#{r + 1}")

 pscore = {}
 if pids and "person" in use:
 for iid, pid, sc in conn.execute(f"""
 SELECT s.image_id, s.person_id, s.score FROM {TABLES["scores"]} s JOIN {TABLES["images"]} i ON i.id = s.image_id
 WHERE s.person_id = ANY(%s) AND s.score >= 0.3 {dsql}""", (pids, *dargs)):
 pscore.setdefault(iid, {})[pid] = sc
 add("person", sorted(pscore, key=lambda i: -min(pscore[i].values())))
 lap("people_ms")

 ctx = q_rr
 ctx_text = (ctx + " " + expansions).strip() if expansions else ctx # synonyms help word matching, not the reranker
 if "vector" in use:
 qv = embed_text([ctx])[0]
 add("vector", [r[0] for r in conn.execute(
 f"SELECT id FROM {TABLES["images"]} i WHERE clip IS NOT NULL {dsql} ORDER BY clip <=> %s LIMIT {POOL}", (*dargs, qv))])
 if "desc" in use:
 dv = embed_query(ctx_text)
 add("desc", [r[0] for r in conn.execute(
 f"SELECT id FROM {TABLES["images"]} i WHERE desc_emb IS NOT NULL {dsql} ORDER BY desc_emb <=> %s LIMIT {POOL}", (*dargs, dv))])
 if "text" in use:
 add("text", [r[0] for r in conn.execute(f"""
 WITH q AS (SELECT to_tsquery('english', array_to_string(tsvector_to_array(to_tsvector('english', %s)), ' | ')) q)
 SELECT id FROM {TABLES["images"]} i, q WHERE q.q IS NOT NULL AND tsv @@ q.q {dsql}
 ORDER BY ts_rank_cd(tsv, q.q, 32) DESC LIMIT {POOL}""", (ctx_text, *dargs))])
 lap("recall_ms")

 # ---- stage 2: precision
 ids = list(cand)
 ids.sort(key=lambda i: i not in pscore) # people candidates are never cut by the cap
 ids = ids[:MAX_RERANK]
 docs = {}
 if ids:
 for iid, d, ppl in conn.execute(f"""
 SELECT i.id, coalesce(i.doc_text, ''),
 (SELECT string_agg(CASE WHEN s.score >= %s THEN p.name ELSE 'possibly ' || p.name END, ', ' ORDER BY s.score DESC)
 FROM {TABLES["scores"]} s JOIN {TABLES["persons"]} p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= %s)
 FROM {TABLES["images"]} i WHERE i.id = ANY(%s)""", (PERSON_SURE, PERSON_MIN, ids)):
 docs[iid] = f"People: {ppl}. {d}" if ppl else d
 if "rerank" in use:
 pc = dict(zip(ids, rerank(ctx, [docs.get(i, "") for i in ids])))
 else: # ablation without reranker: recall rank as a stand-in
 pc = {i: 1 / (1 + min(int(w.split("#")[1]) for w in why[i] if "#" in w) / 10) for i in ids}
 lap("rerank_ms")

 results = []
 for i in ids:
 pp = 1.0
 for p, _, _ in strong:
 pp *= pscore.get(i, {}).get(p, 0.0)
 rel = float(pc[i]) * pp
 if rel >= rel_min:
 results.append({"id": i, "relevance": round(rel, 4), "p_ctx": round(float(pc[i]), 4),
 "p_person": round(pp, 4) if strong else None, "why": why.get(i, [])})
 results.sort(key=lambda r: r["relevance"], reverse=True)
 results = results[:limit]
 meta = {r["id"]: r for r in results}
 rows = hydrate(conn, [r["id"] for r in results])
 for r in rows:
 r.update(meta[r["id"]])
 lap("fetch_ms")
 T["total_ms"] = round(sum(T.values()), 1)

 names = people(conn).names
 present = {p for (p,) in conn.execute(f"SELECT DISTINCT person_id FROM {TABLES["scores"]} WHERE person_id = ANY(%s) AND score >= %s",
 (pids, PERSON_MIN))} if pids else set()
 return {"query": q_user, "query_used": q, "parsed": parsed,
 "persons": [{"name": names[p], "p_link": round(c, 2), "words": w} for p, c, w in links],
 "persons_missing": [names[p] for p in pids if p not in present],
 "required": [names[p] for p, _, _ in strong], "reranked_as": ctx,
 "years": [year_from, year_to], "timings": T, "count": len(rows), "candidates": len(ids), "results": rows}


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
 FROM {TABLES["faces"]} f LEFT JOIN {TABLES["persons"]} p ON p.id = f.person_id WHERE f.image_id = i.id)
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
 "people_sources": ppl, "faces": r[11]})
 return out


def similar(image_id: str, limit=40, conn=None):
 conn = conn or connect()
 t = time.perf_counter()
 ids = [r[0] for r in conn.execute(f"""SELECT id FROM {TABLES["images"]} WHERE clip IS NOT NULL AND id <> %s
 ORDER BY clip <=> (SELECT clip FROM {TABLES["images"]} WHERE id = %s) LIMIT %s""", (image_id, image_id, limit))]
 ms = round((time.perf_counter() - t) * 1000, 1)
 res = hydrate(conn, ids)
 return {"query": f"similar to {image_id}", "timings": {"vector_ms": ms, "total_ms": ms}, "count": len(res), "results": res}
