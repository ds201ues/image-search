"""Ingest a folder of images. Each step is idempotent and timed.

uv run python -m app.ingest --dir data/sample # all steps
uv run python -m app.ingest --dir data/sample --steps embed,faces # no API needed
Steps: register, embed, faces, vlm, text, people
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from pathlib import Path
from PIL import Image
from .config import DATA_DIR, THUMB_DIR, FACE_MATCH_THRESHOLD
from .db import connect, init_schema, rebuild_search_text
from . import meta as M

EXTS = M.IMAGE_EXTS | M.PDF_EXTS
THUMB_SIDE = 768 # one thumbnail serves the UI, SigLIP and the VLM (which resizes to 768 anyway)
PDF_DIR = DATA_DIR / "pdf"


def thumb_path(iid: str) -> Path:
 return THUMB_DIR / f"{iid}.jpg"


def _prepare(args):
 """Worker: read metadata, extract PDF photo, write thumbnail. Returns a row dict or an error string."""
 path, rel, api = args
 try:
 iid = api.get("image_id") or path.stem
 src = path
 if path.suffix.lower() in M.PDF_EXTS:
 src = M.pdf_photo(path, PDF_DIR)
 if src is None:
 return f"{rel}: no image in PDF"
 m = M.read_all(src)
 if path is not src: # the PDF itself may carry the caption
 try:
 pm = M.read_all(path) if False else {}
 except Exception:
 pm = {}
 m = {**pm, **m}
 m = {k: (v.replace("\x00", "") if isinstance(v, str) else [x.replace("\x00", "") for x in v] if isinstance(v, list) else v)
 for k, v in m.items()}
 tp = thumb_path(iid)
 if not tp.exists():
 M.open_scaled(src, THUMB_SIDE).save(tp, "JPEG", quality=85)
 date = api.get("date_published") or m["date"]
 return dict(id=iid, file=src.resolve().relative_to(DATA_DIR.resolve()).as_posix() if src.resolve().is_relative_to(DATA_DIR.resolve()) else str(src),
 orig_file=rel, width=m["width"], height=m["height"], date=date,
 date_source="api" if api.get("date_published") else m["date_source"],
 section=api.get("section"), source_url=api.get("source_url"), ref_title=api.get("ref_title"),
 caption=m.get("caption"), headline=m.get("headline"), keywords=m.get("keywords") or None,
 city=m.get("city"), state=m.get("state"), country=m.get("country"), byline=m.get("byline"),
 credit=m.get("credit") or m.get("source"), camera=m.get("camera"),
 meta={k: v for k, v in m.items() if k not in ("date", "width", "height") and not isinstance(v, (bytes,))})
 except Exception as e:
 return f"{rel}: {type(e).__name__}: {e}"


def timed(name, n, fn):
 t = time.perf_counter()
 fn()
 s = time.perf_counter() - t
 print(f" {name:<9} {n:>6} images {s:7.1f}s {1000 * s / max(n, 1):7.0f} ms/image")
 return s


def register(conn, folder: Path, workers=8):
 """Register every image (and PDF-wrapped photo) in a folder: source metadata + thumbnail, in parallel."""
 api = {}
 mp = folder / "metadata.jsonl"
 if mp.exists():
 for line in mp.open():
 try:
 r = json.loads(line); api[r["file"]] = r
 except json.JSONDecodeError:
 pass # line still being written by the downloader
 files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS and not p.name.startswith("_")
 and (not api or p.relative_to(folder).as_posix() in api))
 have = {r[0] for r in conn.execute("SELECT id FROM images")}
 jobs = []
 for p in files:
 rel = p.relative_to(folder).as_posix()
 a = api.get(rel, {})
 if (a.get("image_id") or p.stem) not in have:
 jobs.append((p, rel, a))
 THUMB_DIR.mkdir(parents=True, exist_ok=True)
 rows, errors = [], []
 with ProcessPoolExecutor(workers) as ex:
 for r in ex.map(_prepare, jobs, chunksize=8):
 (errors if isinstance(r, str) else rows).append(r)
 with conn.cursor() as cur:
 cur.executemany("""INSERT INTO images (id, file, orig_file, width, height, date_published, date_source, section, source_url, ref_title,
 caption, headline, keywords, city, state, country, byline, credit, camera, meta)
 VALUES (%(id)s,%(file)s,%(orig_file)s,%(width)s,%(height)s,%(date)s,%(date_source)s,%(section)s,%(source_url)s,%(ref_title)s,
 %(caption)s,%(headline)s,%(keywords)s,%(city)s,%(state)s,%(country)s,%(byline)s,%(credit)s,%(camera)s,%(meta)s)
 ON CONFLICT (id) DO NOTHING""", [{**r, "meta": json.dumps(r["meta"], default=str)} for r in rows])
 for e in errors[:20]:
 print(" register error:", e)
 if len(errors) > 20:
 print(f" ... {len(errors)} register errors")
 return len(rows)


def _abs(f: str) -> Path:
 return Path(f) if f.startswith("/") else DATA_DIR / f


def step_embed(conn, batch=16):
 from .embed import embed_images
 todo = conn.execute("SELECT id, file FROM images WHERE clip IS NULL ORDER BY id").fetchall()
 def run():
 for i in range(0, len(todo), batch):
 chunk = todo[i:i + batch]
 vecs = embed_images([Image.open(thumb_path(iid)) for iid, _ in chunk])
 with conn.cursor() as cur:
 cur.executemany("UPDATE images SET clip = %s WHERE id = %s", [(v, iid) for (iid, _), v in zip(chunk, vecs)])
 return todo, run


def step_faces(conn):
 from .faces import detect_and_embed, read_bgr
 todo = conn.execute("SELECT id, file FROM images WHERE NOT faces_done ORDER BY id").fetchall()
 def run():
 import numpy as np
 from .faces import DET_SIDE
 for iid, f in todo:
 try:
 im = M.open_scaled(_abs(f), DET_SIDE) # detector runs at DET_SIDE anyway; DCT-scaled decode is 4-10x faster
 bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
 with Image.open(_abs(f)) as o:
 k = o.width / im.width
 except Exception as e:
 print(f" faces fail {iid}: {e}"); conn.execute("UPDATE images SET faces_done = true WHERE id = %s", (iid,)); continue
 faces = detect_and_embed(bgr)
 for x in faces:
 x["bbox"] = [int(round(v * k)) for v in x["bbox"]]
 with conn.cursor() as cur:
 cur.execute("DELETE FROM faces WHERE image_id = %s", (iid,))
 cur.executemany("INSERT INTO faces (image_id, bbox, det_score, emb) VALUES (%s,%s,%s,%s)",
 [(iid, x["bbox"], x["det_score"], x["emb"]) for x in faces])
 cur.execute("UPDATE images SET faces_done = true WHERE id = %s", (iid,))
 return todo, run


def step_vlm(conn, workers=8):
 from .vlm import describe
 todo = conn.execute("SELECT id, file FROM images WHERE vlm IS NULL ORDER BY id").fetchall()
 stats = {"cost": 0.0, "tin": 0, "tout": 0, "fail": 0}
 def run():
 with ThreadPoolExecutor(workers) as ex:
 futs = {ex.submit(describe, thumb_path(iid)): iid for iid, f in todo}
 for fut in as_completed(futs):
 iid = futs[fut]
 try:
 r = fut.result()
 except Exception as e:
 stats["fail"] += 1; print(f" vlm fail {iid}: {e}"); continue
 stats["cost"] += r["cost_usd"]; stats["tin"] += r["tokens_in"]; stats["tout"] += r["tokens_out"]
 conn.execute("UPDATE images SET vlm=%s, vlm_model=%s, vlm_ms=%s, vlm_cost_usd=%s WHERE id=%s",
 (json.dumps(r["vlm"]), r["model"], r["ms"], r["cost_usd"], iid))
 n = max(len(todo) - stats["fail"], 1)
 print(f" vlm: ${stats['cost']:.4f} total, ${1000 * stats['cost'] / n:.3f} per 1k images, "
 f"{stats['tin'] // n} tokens in / {stats['tout'] // n} out per image, {stats['fail']} failed")
 return todo, run


VLM_MIN_CONF = 0.3 # Luna guesses below this are noise; everything above is kept as evidence
CAPTION_WEIGHT = 0.9 # a caption naming someone is strong evidence, not proof (people get cropped out, named as context)
CAPTION_REF_CONF = 0.8 # single-face photo + one caption name at this confidence -> reference face for that person
CLUSTER_SIM = 0.55 # AuraFace cosine for "same person" when grouping faces
GALLERY_SIM = 0.40 # a face this close to a reference counts as a vote ...
GALLERY_MARGIN = 0.15 # ... if it also beats the best reference of any OTHER person by this much (open-set rule; exp/faces/measure.py:
 # 0.40 + margin 0.15 -> 85% precision like 0.55 alone, but names 9% more faces)
GALLERY_SURE = 0.55 # above this no margin is needed
CAPTION_ASSIGN_SIM = 0.30 # caption names one person, several faces: the face closest to that person's references (and this close) is theirs
REF_OUTLIER_SIM = 0.20 # a reference that resembles none of its person's other references (3+ refs) is a mislabelled face
CLUSTER_VLM_MIN = 0.6 # Luna naming a face cluster (2-3 crops side by side) counts as evidence above this confidence
CLUSTER_VLM_WEIGHT = 0.6 # alone this makes a person "possible" (>= 0.5), never "sure": 62% agreement with captions (E14)
CLUSTER_VLM_FULL = 0.55 # a face at least this similar to the crops Luna saw gets the full weight ...
CLUSTER_VLM_FLOOR = 0.40 # ... fading to half weight here; below it the name does not propagate (clusters can chain two people)


def step_text(conn, batch=64):
 """Reranker text + description-meaning vector for every described image."""
 from .text import doc_text, embed_docs
 todo = conn.execute("SELECT id, vlm, caption FROM images WHERE (vlm IS NOT NULL OR caption IS NOT NULL) AND desc_emb IS NULL").fetchall()
 def run():
 for i in range(0, len(todo), batch):
 chunk = todo[i:i + batch]
 docs = [doc_text(v, c) for _, v, c in chunk]
 vecs = embed_docs(docs)
 with conn.cursor() as cur:
 cur.executemany("UPDATE images SET doc_text = %s, desc_emb = %s WHERE id = %s",
 [(d, e, iid) for (iid, _, _), d, e in zip(chunk, docs, vecs)])
 return todo, run


def _clusters(conn):
 """Group faces of the same person (union-find over close neighbours)."""
 ids = [r[0] for r in conn.execute("SELECT id FROM faces ORDER BY id")]
 parent = {i: i for i in ids}
 def find(x):
 while parent[x] != x:
 parent[x] = parent[parent[x]]; x = parent[x]
 return x
 for a, b in conn.execute(f"""
 SELECT f.id, n.id FROM faces f, LATERAL (
 SELECT g.id FROM faces g WHERE g.id <> f.id AND g.image_id <> f.image_id
 ORDER BY g.emb <=> f.emb LIMIT 10) n
 JOIN faces g2 ON g2.id = n.id WHERE 1 - (g2.emb <=> f.emb) >= {CLUSTER_SIM}"""):
 parent[find(a)] = find(b)
 return {i: find(i) for i in ids}


def step_people(conn):
 """Who is in each image, as evidence with confidence (decided at query time, not here).
 1. Luna guesses -> image_persons(source='vlm'); caption names -> image_persons(source='caption')
 single-face photos with one confident caption name seed person_refs (gallery votes for the clusters)
 2. face clusters -> faces.cluster_id
 3. cluster votes -> image_persons(source='cluster'): what the OTHER photos of this face say
 (Luna names weighted by 1/faces-in-photo, plus gallery matches), leave-one-out so a guess
 never confirms itself.
 """
 n = conn.execute("SELECT count(*) FROM images").fetchone()[0]
 def run():
 conn.execute("DELETE FROM image_persons")
 vlm = {}
 for iid, name, conf in conn.execute("""SELECT id, x->>'name', (x->>'confidence')::real FROM images,
 jsonb_array_elements(vlm->'recognized_people') x WHERE vlm IS NOT NULL"""):
 name = " ".join((name or "").split())
 if not name or conf < VLM_MIN_CONF:
 continue
 pid = conn.execute("""INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
 RETURNING id""", (name,)).fetchone()[0]
 vlm.setdefault(iid, {})[pid] = max(conf, vlm.get(iid, {}).get(pid, 0))
 with conn.cursor() as cur:
 cur.executemany("INSERT INTO image_persons VALUES (%s, %s, 'vlm', %s)",
 [(i, p, c) for i, d in vlm.items() for p, c in d.items()])

 # caption names: evidence + reference faces
 from .caption_names import extract, spent as cap_spent
 cap_rows = conn.execute("SELECT id, caption, headline FROM images WHERE caption IS NOT NULL OR headline IS NOT NULL").fetchall()
 capn = extract(cap_rows)
 cap_pid = {}
 for iid, ppl in capn.items():
 for name, conf in ppl:
 name = " ".join(name.split())
 pid = conn.execute("""INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
 RETURNING id""", (name,)).fetchone()[0]
 cap_pid.setdefault(iid, {})[pid] = max(conf, cap_pid.get(iid, {}).get(pid, 0))
 with conn.cursor() as cur:
 cur.executemany("INSERT INTO image_persons VALUES (%s, %s, 'caption', %s) ON CONFLICT DO NOTHING",
 [(i, p, round(CAPTION_WEIGHT * c, 3)) for i, d in cap_pid.items() for p, c in d.items()])
 cur.execute("DELETE FROM person_refs WHERE source LIKE 'caption:%'")
 cur.execute("""INSERT INTO person_refs (person_id, emb, source)
 SELECT ip.person_id, f.emb, 'caption:' || f.image_id
 FROM image_persons ip JOIN faces f ON f.image_id = ip.image_id
 WHERE ip.source = 'caption' AND ip.score >= %s
 AND (SELECT count(*) FROM faces g WHERE g.image_id = ip.image_id) = 1
 AND (SELECT count(*) FROM image_persons j WHERE j.image_id = ip.image_id AND j.source = 'caption') = 1""",
 (CAPTION_WEIGHT * CAPTION_REF_CONF,))
 # caption names exactly one person but the photo has several faces: the face closest to that person's existing
 # references is theirs (Berg et al., "Names and Faces in the News"), and becomes another reference
 with conn.cursor() as cur:
 cur.execute("DELETE FROM person_refs WHERE source LIKE 'caption2:%'")
 cur.execute("""INSERT INTO person_refs (person_id, emb, source)
 SELECT person_id, emb, 'caption2:' || image_id FROM (
 SELECT ip.person_id, f.emb, f.image_id,
 (SELECT min(r.emb <=> f.emb) FROM person_refs r WHERE r.person_id = ip.person_id AND r.source LIKE 'caption:%%') AS d,
 row_number() OVER (PARTITION BY ip.image_id ORDER BY
 (SELECT min(r.emb <=> f.emb) FROM person_refs r WHERE r.person_id = ip.person_id AND r.source LIKE 'caption:%%')) AS rn
 FROM image_persons ip JOIN faces f ON f.image_id = ip.image_id
 WHERE ip.source = 'caption' AND ip.score >= %s
 AND (SELECT count(*) FROM faces g WHERE g.image_id = ip.image_id) BETWEEN 2 AND 6
 AND (SELECT count(*) FROM image_persons j WHERE j.image_id = ip.image_id AND j.source = 'caption') = 1) x
 WHERE rn = 1 AND d IS NOT NULL AND 1 - d >= %s""", (CAPTION_WEIGHT * CAPTION_REF_CONF, CAPTION_ASSIGN_SIM))
 # reference cleaning: a reference unlike every other reference of the same person (3+ refs) is a wrong face
 cur.execute("""DELETE FROM person_refs r WHERE r.source LIKE 'caption%%'
 AND (SELECT count(*) FROM person_refs o WHERE o.person_id = r.person_id) >= 3
 AND NOT EXISTS (SELECT 1 FROM person_refs o WHERE o.person_id = r.person_id AND o.id <> r.id AND 1 - (o.emb <=> r.emb) >= %s)""",
 (REF_OUTLIER_SIM,))
 nrefs, nrefs2 = conn.execute("""SELECT count(*) FILTER (WHERE source LIKE 'caption:%'), count(*) FILTER (WHERE source LIKE 'caption2:%')
 FROM person_refs""").fetchone()
 print(f" captions: {len(capn)} photos name someone ({sum(len(v) for v in capn.values())} names, "
 f"${cap_spent['usd']:.3f} new extraction), {nrefs} single-face references + {nrefs2} assigned in multi-face photos (after outlier cleaning)")

 cl = _clusters(conn)
 with conn.cursor() as cur:
 cur.executemany("UPDATE faces SET cluster_id = %s WHERE id = %s", [(c, f) for f, c in cl.items()])
 faces = conn.execute("SELECT id, image_id FROM faces").fetchall()
 nfaces = {}
 for _, iid in faces:
 nfaces[iid] = nfaces.get(iid, 0) + 1
 gallery = {}
 for fid, pid, sim, pid2, sim2 in conn.execute(f"""
 SELECT f.id, t.person_id, t.sim, t.person_id2, t.sim2 FROM faces f, LATERAL (
 SELECT (array_agg(person_id ORDER BY sim DESC))[1] AS person_id, max(sim) AS sim,
 (array_agg(person_id ORDER BY sim DESC))[2] AS person_id2,
 (array_agg(sim ORDER BY sim DESC))[2] AS sim2
 FROM (SELECT r.person_id, max(1 - (r.emb <=> f.emb)) AS sim FROM person_refs r
 WHERE r.source NOT LIKE 'vlm:%%' GROUP BY r.person_id ORDER BY sim DESC LIMIT 2) q) t
 WHERE t.sim >= {GALLERY_SIM}"""):
 if sim >= GALLERY_SURE or sim - (sim2 or 0.0) >= GALLERY_MARGIN: # open-set: clear winner or very sure
 gallery[fid] = (pid, sim)
 members = {}
 for fid, iid in faces:
 members.setdefault(cl[fid], []).append((fid, iid))

 rows, labels = [], {}
 for cid, mem in members.items():
 imgs = {iid for _, iid in mem}
 for fid, iid in mem:
 votes = {}
 for f2, i2 in mem:
 if i2 != iid: # leave-one-out: other photos' Luna names
 for pid, conf in vlm.get(i2, {}).items():
 votes[pid] = votes.get(pid, 0) + conf / nfaces[i2]
 if f2 in gallery: # gallery face matches are independent evidence, own face included
 pid, sim = gallery[f2]
 votes[pid] = votes.get(pid, 0) + sim
 denom = len(imgs - {iid}) + sum(1 for f2, _ in mem if f2 in gallery)
 if not votes or not denom:
 continue
 ranked = sorted(votes.items(), key=lambda kv: -kv[1])
 pid, v = ranked[0]
 runner = ranked[1][1] if len(ranked) > 1 else 0
 score = min(0.99, v / denom) * (1 if v >= 2 * runner else 0.5)
 rows.append((iid, pid, score))
 labels[fid] = (pid, score)
 # Luna names whole clusters from their clearest faces (independent evidence, cached)
 from .cluster_names import name_clusters, spent as cl_spent
 multi = {cid: mem for cid, mem in members.items() if len({i for _, i in mem}) >= 2}
 fmeta = {fid: (iid, bbox, w, sc) for fid, iid, bbox, w, sc in conn.execute(
 "SELECT f.id, f.image_id, f.bbox, i.width, f.det_score FROM faces f JOIN images i ON i.id = f.image_id WHERE f.cluster_id = ANY(%s)",
 (list(multi),))}
 chosen = {}
 for cid, mem in multi.items():
 seen, picks = set(), []
 for fid, _ in sorted(mem, key=lambda x: -(min(fmeta[x[0]][1][2], fmeta[x[0]][1][3]) * fmeta[x[0]][3])): # big, confident faces first
 iid, bbox, w, _ = fmeta[fid]
 if iid in seen:
 continue
 seen.add(iid); picks.append((fid, iid, bbox, w))
 if len(picks) == 3:
 break
 chosen[cid] = picks
 named = name_clusters(chosen)
 from .search import People
 directory = People(conn) # Luna's spelling -> existing person when it is the same name (alias/initials)
 cl_rows, cl_named = [], 0
 for cid, (name, conf) in named.items():
 if not name or conf < CLUSTER_VLM_MIN:
 continue
 hit = directory.link_name(name)
 if hit and hit[1] >= 0.8:
 pid = hit[0]
 else:
 pid = conn.execute("""INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
 RETURNING id""", (name,)).fetchone()[0]
 cl_named += 1
 shown = [fid for fid, *_ in chosen[cid]]
 # propagate per face, weighted by similarity to the faces Luna saw (not to the whole, possibly chained, cluster)
 for iid, sim in conn.execute("""SELECT f.image_id, max(1 - (f.emb <=> g.emb)) FROM faces f, faces g
 WHERE f.cluster_id = %s AND g.id = ANY(%s) GROUP BY f.image_id""", (cid, shown)):
 if sim >= CLUSTER_VLM_FULL:
 w = 1.0
 elif sim >= CLUSTER_VLM_FLOOR:
 w = 0.5 + 0.5 * (sim - CLUSTER_VLM_FLOOR) / (CLUSTER_VLM_FULL - CLUSTER_VLM_FLOOR)
 else:
 continue
 cl_rows.append((iid, pid, round(CLUSTER_VLM_WEIGHT * conf * w, 3)))
 best_cl = {}
 for iid, pid, sc in cl_rows:
 best_cl[(iid, pid)] = max(sc, best_cl.get((iid, pid), 0))
 with conn.cursor() as cur:
 cur.executemany("INSERT INTO image_persons VALUES (%s, %s, 'cluster_vlm', %s) ON CONFLICT DO NOTHING",
 [(i, p, sc) for (i, p), sc in best_cl.items()])
 cl_rows = list(best_cl)
 print(f" cluster naming: {len(multi)} clusters asked ({cl_spent['calls']} new calls, ${cl_spent['usd']:.3f}), "
 f"{cl_named} named, {len(cl_rows)} photo-person rows (similarity-weighted)")

 # one row per (image, person): keep the strongest face
 best = {}
 for iid, pid, sc in rows:
 best[(iid, pid)] = max(sc, best.get((iid, pid), 0))
 with conn.cursor() as cur:
 cur.executemany("INSERT INTO image_persons VALUES (%s, %s, 'cluster', %s)",
 [(i, p, sc) for (i, p), sc in best.items() if sc >= 0.2])
 cur.execute("UPDATE faces SET person_id = NULL, match_score = NULL")
 cur.executemany("UPDATE faces SET person_id = %s, match_score = %s WHERE id = %s",
 [(p, sc, f) for f, (p, sc) in labels.items()])
 rebuild_search_text(conn)
 sizes = sorted((len({i for _, i in m}) for m in members.values()), reverse=True)
 multi = sum(1 for x in sizes if x > 1)
 st = conn.execute("""SELECT count(*), count(*) FILTER (WHERE score >= 0.5), count(*) FILTER (WHERE score >= 0.8)
 FROM image_person_scores""").fetchone()
 print(f" people: {sum(len(d) for d in vlm.values())} Luna guesses; {len(faces)} faces in {len(members)} clusters "
 f"({multi} span 2+ photos, largest {sizes[0] if sizes else 0}); {len(best)} cluster votes; "
 f"image-person pairs {st[0]} (>=0.5: {st[1]}, >=0.8: {st[2]})")
 return [None] * n, run


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--dir", required=True)
 ap.add_argument("--steps", default="register,embed,faces,vlm,text,people")
 ap.add_argument("--workers", type=int, default=8, help="parallel VLM requests")
 ap.add_argument("--procs", type=int, default=8, help="parallel processes for thumbnails/metadata")
 a = ap.parse_args()
 init_schema()
 conn = connect()
 steps = a.steps.split(",")
 folder = Path(a.dir).resolve()
 print(f"ingest {folder}")
 if "register" in steps:
 t = time.perf_counter(); n = register(conn, folder, a.procs)
 print(f" register {n:>6} images {time.perf_counter() - t:7.1f}s")
 for name, fn in [("embed", step_embed), ("faces", step_faces), ("vlm", lambda c: step_vlm(c, a.workers)), ("text", step_text), ("people", step_people)]:
 if name in steps:
 todo, run = fn(conn)
 timed(name, len(todo), run)
 if "vlm" in steps and "people" not in steps:
 rebuild_search_text(conn)


if __name__ == "__main__":
 main()
