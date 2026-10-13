"""uv run uvicorn app.api:app --port 8000"""
import json, time
from pathlib import Path
from fastapi import FastAPI, Query, Body
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from .config import DATA_DIR, THUMB_DIR, ROOT
from .db import connect
from . import search as S

app = FastAPI(title="Image Search")
_conn = connect()


def db():
 """One shared connection; reopen it if Postgres was restarted underneath us."""
 global _conn
 if _conn.closed:
 _conn = connect()
 return _conn


@app.on_event("startup")
def warm():
 S.search("warm up", conn=db()) # load SigLIP, text models and the people directory once


@app.get("/api/search")
def api_search(q: str, limit: int = 60, year_from: int | None = None, year_to: int | None = None,
 person: list[int] = Query(default=[]), parse: bool | None = None):
 """parse: LLM query understanding (cached); None = config default, false = plain query."""
 t = time.perf_counter()
 r = S.search(q, limit, year_from, year_to, person, conn=db(), parse=parse)
 r["timings"]["server_ms"] = round((time.perf_counter() - t) * 1000, 1)
 with (DATA_DIR / "query_log.jsonl").open("a") as f: # real queries are the next eval set
 f.write(json.dumps({"ts": time.time(), "q": q, "count": r["count"], "matches": r.get("matches"),
 "ms": r["timings"]["server_ms"], "required": r.get("required")}) + "\n")
 return r


@app.post("/api/click")
def api_click(body: dict = Body(...)):
 """UI reports which result was opened or downloaded (query, image_id, rank, action)."""
 with (DATA_DIR / "query_log.jsonl").open("a") as f:
 f.write(json.dumps({"ts": time.time(), "click": body}) + "\n")
 return {"ok": True}


# ---- human grading of (query, photo) pairs: validates the LLM judge. Pool built by eval/make_grading_pool.py.
GRADE_POOL = Path(__file__).resolve().parent.parent / "eval" / "grading_pool.jsonl"
GRADES = DATA_DIR / "human_grades.jsonl"


@app.get("/api/grade/next")
def grade_next(user: str):
 done = set()
 if GRADES.exists():
 for l in GRADES.open():
 try:
 g = json.loads(l)
 if g["user"] == user:
 done.add((g["query"], g["image_id"]))
 except Exception:
 pass
 pool = [json.loads(l) for l in GRADE_POOL.open()] if GRADE_POOL.exists() else []
 todo = [p for p in pool if (p["query"], p["image_id"]) not in done]
 if not todo:
 return {"done": True, "total": len(pool)}
 p = todo[0]
 row = db().execute("SELECT caption, date_published::date, city FROM images WHERE id = %s", (p["image_id"],)).fetchone()
 return {"done": False, "left": len(todo), "total": len(pool), "query": p["query"], "image_id": p["image_id"],
 "thumb": f"/thumbs/{p['image_id']}.jpg", "caption": row[0] if row else None,
 "date": row[1].isoformat() if row and row[1] else None, "city": row[2] if row else None}


@app.post("/api/grade")
def grade_post(body: dict = Body(...)):
 with GRADES.open("a") as f:
 f.write(json.dumps({**body, "ts": time.time()}) + "\n")
 return {"ok": True}


@app.get("/grade")
def grade_page():
 return FileResponse(ROOT / "app" / "static" / "grade.html")


@app.post("/api/refresh")
def api_refresh():
 """Reload the people directory after an ingest."""
 S.people(db(), refresh=True)
 return {"ok": True}


@app.get("/api/similar/{image_id}")
def api_similar(image_id: str, limit: int = 40):
 return S.similar(image_id, limit, conn=db())


@app.get("/api/stats")
def api_stats():
 r = db().execute("""SELECT count(*), count(clip), count(vlm), count(*) FILTER (WHERE faces_done),
 min(date_published)::date, max(date_published)::date,
 coalesce(sum(vlm_cost_usd), 0) FROM images""").fetchone()
 f = db().execute("SELECT count(*), count(person_id), count(DISTINCT cluster_id) FROM faces").fetchone()
 people = db().execute("""SELECT p.id, p.name, count(*) n FROM image_person_scores s JOIN persons p ON p.id = s.person_id
 WHERE s.score >= 0.5 GROUP BY p.id, p.name ORDER BY n DESC, p.name""").fetchall()
 return {"images": r[0], "embedded": r[1], "described": r[2], "faces_scanned": r[3],
 "date_min": r[4], "date_max": r[5], "vlm_cost_usd": round(r[6], 4),
 "faces": f[0], "faces_named": f[1], "face_clusters": f[2], "people_found": len(people),
 "gallery_size": db().execute("SELECT count(*) FROM persons").fetchone()[0],
 "people": [{"id": i, "name": n, "images": c} for i, n, c in people]}


@app.get("/original/{image_id}")
def api_original(image_id: str):
 """The archive file itself, for download. Paths in `images.file` may be absolute (archive dirs outside data/)."""
 row = db().execute("SELECT file, orig_file FROM images WHERE id = %s", (image_id,)).fetchone()
 if not row:
 return Response(status_code=404)
 path = Path(row[0]) if Path(row[0]).is_absolute() else DATA_DIR / row[0]
 if not path.exists():
 return Response(status_code=404)
 name = Path(row[1] or row[0]).name
 return FileResponse(path, filename=name)


app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")
app.mount("/media", StaticFiles(directory=DATA_DIR), name="media")


@app.get("/")
def index():
 return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/debug")
def debug_page():
 """The original testing UI: stage timings, scores, raw description fields."""
 return FileResponse(ROOT / "app" / "static" / "debug.html")
