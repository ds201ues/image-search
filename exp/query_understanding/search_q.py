"""E6 wrapper around app.search.search: LLM-parsed query -> normalised text, years, people, city boost/filter.
No edits to app/: city handling swaps app.search.TABLES["images"] to a view restricted to the city's aliases.
"""
import contextlib
from app import search as S
from app.text import embed_query as _embed_query_orig
from .parser import parse, city_aliases

CITY_VIEW = "exp_query_understanding_city_view"
CITY_BONUS = 0.15 # boost variant: relevance added to results found inside the city-restricted index
_expansion = {"text": None}


def _embed_query_expanded(q):
 """Description-vector query = normalised text + expansions (full-text would need an app change)."""
 return _embed_query_orig(q + (" " + _expansion["text"] if _expansion["text"] else ""))


@contextlib.contextmanager
def _city_scope(conn, city):
 aliases = city_aliases(city)
 if not aliases:
 yield False; return
 from psycopg import sql
 conn.execute(sql.SQL("CREATE OR REPLACE VIEW {} AS SELECT * FROM images WHERE lower(trim(city)) = ANY({}) OR lower(city) LIKE ANY({})")
 .format(sql.Identifier(CITY_VIEW), sql.Literal(aliases), sql.Literal([f"%{a}%" for a in aliases])))
 old = S.TABLES["images"]; S.TABLES["images"] = CITY_VIEW
 try:
 yield True
 finally:
 S.TABLES["images"] = old


def search_q(q, conn, mode="parsed", expansions=False, **kw):
 """mode: parsed | city_boost | city_filter. Returns app.search-style dict plus 'parsed'."""
 p = parse(q)
 text = p["search_text"] or q
 yf = p["year_from"] or None; yt = p["year_to"] or None
 if yf and not yt:
 yt = yf
 _expansion["text"] = " ".join(p["expansions"]) if expansions else None
 S.embed_query = _embed_query_expanded
 try:
 if mode == "city_filter" and p["city"]:
 with _city_scope(conn, p["city"]) as ok:
 r = S.search(text, conn=conn, year_from=yf, year_to=yt, **kw)
 else:
 r = S.search(text, conn=conn, year_from=yf, year_to=yt, **kw)
 if mode == "city_boost" and p["city"]:
 with _city_scope(conn, p["city"]) as ok:
 rc = S.search(text, conn=conn, year_from=yf, year_to=yt, **{**kw, "rel_min": 0.0})
 by = {x["id"]: x for x in r["results"]}
 for x in rc["results"]:
 x = {**x, "relevance": round(min(1.0, x["relevance"] + CITY_BONUS), 4), "why": x["why"] + ["city"]}
 if x["id"] not in by or x["relevance"] > by[x["id"]]["relevance"]:
 by[x["id"]] = x
 rel_min = kw.get("rel_min", S.REL_MIN) or 0.0
 r["results"] = sorted([x for x in by.values() if x["relevance"] >= rel_min],
 key=lambda x: -x["relevance"])[:kw.get("limit", 60)]
 r["count"] = len(r["results"])
 for k, v in rc["timings"].items():
 r["timings"][k] = round(r["timings"].get(k, 0) + v, 1)
 finally:
 S.embed_query = _embed_query_orig
 _expansion["text"] = None
 r["parsed"] = p
 r["parse_ms"] = p.get("_ms")
 return r
