"""uv run uvicorn app.api:app --port 8000"""
import time
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .config import DATA_DIR, THUMB_DIR, ROOT
from .db import connect
from . import search as S

app = FastAPI(title="Image Search")
conn = connect()


@app.on_event("startup")
def warm():
 S.search("warm up", conn=conn) # load SigLIP, text models and the people directory once


@app.get("/api/search")
def api_search(q: str, limit: int = 60, year_from: int | None = None, year_to: int | None = None,
 person: list[int] = Query(default=[]), parse: bool | None = None):
 """parse: LLM query understanding (cached); None = config default, false = plain query."""
 t = time.perf_counter()
 r = S.search(q, limit, year_from, year_to, person, conn=conn, parse=parse)
 r["timings"]["server_ms"] = round((time.perf_counter() - t) * 1000, 1)
 return r


@app.post("/api/refresh")
def api_refresh():
 """Reload the people directory after an ingest."""
 S.people(conn, refresh=True)
 return {"ok": True}


@app.get("/api/similar/{image_id}")
def api_similar(image_id: str, limit: int = 40):
 return S.similar(image_id, limit, conn=conn)


@app.get("/api/stats")
def api_stats():
 r = conn.execute("""SELECT count(*), count(clip), count(vlm), count(*) FILTER (WHERE faces_done),
 min(date_published)::date, max(date_published)::date,
 coalesce(sum(vlm_cost_usd), 0) FROM images""").fetchone()
 f = conn.execute("SELECT count(*), count(person_id), count(DISTINCT cluster_id) FROM faces").fetchone()
 people = conn.execute("""SELECT p.id, p.name, count(*) n FROM image_person_scores s JOIN persons p ON p.id = s.person_id
 WHERE s.score >= 0.5 GROUP BY p.id, p.name ORDER BY n DESC, p.name""").fetchall()
 return {"images": r[0], "embedded": r[1], "described": r[2], "faces_scanned": r[3],
 "date_min": r[4], "date_max": r[5], "vlm_cost_usd": round(r[6], 4),
 "faces": f[0], "faces_named": f[1], "face_clusters": f[2], "people_found": len(people),
 "gallery_size": conn.execute("SELECT count(*) FROM persons").fetchone()[0],
 "people": [{"id": i, "name": n, "images": c} for i, n, c in people]}


app.mount("/thumbs", StaticFiles(directory=THUMB_DIR), name="thumbs")
app.mount("/media", StaticFiles(directory=DATA_DIR), name="media")


@app.get("/")
def index():
 return FileResponse(ROOT / "app" / "static" / "index.html")
