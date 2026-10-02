"""Ingest a folder of images. Each step is idempotent and timed.

uv run python -m app.ingest --dir data/sample # all steps
uv run python -m app.ingest --dir data/sample --steps embed,faces # no API needed
Steps: register, embed, faces, vlm, text, people
"""
import argparse, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from .config import DATA_DIR, THUMB_DIR, FACE_MATCH_THRESHOLD
from .db import connect, init_schema, rebuild_search_text

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def timed(name, n, fn):
 t = time.perf_counter()
 fn()
 s = time.perf_counter() - t
 print(f" {name:<9} {n:>6} images {s:7.1f}s {1000 * s / max(n, 1):7.0f} ms/image")
 return s


def register(conn, folder: Path):
 meta = {}
 mp = folder / "metadata.jsonl"
 if mp.exists():
 for line in mp.open():
 try:
 r = json.loads(line); meta[r["file"]] = r
 except json.JSONDecodeError:
 pass # line still being written by the downloader
 # only files with metadata when a metadata file exists (skips images still being downloaded)
 files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS
 and (not meta or p.relative_to(folder).as_posix() in meta))
 THUMB_DIR.mkdir(parents=True, exist_ok=True)
 rows = []
 for p in files:
 rel = p.relative_to(folder).as_posix()
 m = meta.get(rel, {})
 iid = m.get("image_id") or p.stem
 thumb = THUMB_DIR / f"{iid}.jpg"
 im = Image.open(p)
 if not thumb.exists():
 t = im.convert("RGB"); t.thumbnail((480, 480)); t.save(thumb, "JPEG", quality=82)
 rows.append((iid, p.relative_to(DATA_DIR).as_posix(), im.width, im.height,
 m.get("date_published"), m.get("section"), m.get("source_url"), m.get("ref_title")))
 with conn.cursor() as cur:
 cur.executemany("""INSERT INTO images (id, file, width, height, date_published, section, source_url, ref_title)
 VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (id) DO NOTHING""", rows)
 return len(rows)


def step_embed(conn, batch=16):
 from .embed import embed_images
 todo = conn.execute("SELECT id, file FROM images WHERE clip IS NULL ORDER BY id").fetchall()
 def run():
 for i in range(0, len(todo), batch):
 chunk = todo[i:i + batch]
 vecs = embed_images([Image.open(DATA_DIR / f) for _, f in chunk])
 with conn.cursor() as cur:
 cur.executemany("UPDATE images SET clip = %s WHERE id = %s", [(v, iid) for (iid, _), v in zip(chunk, vecs)])
 return todo, run


def step_faces(conn):
 from .faces import detect_and_embed, read_bgr
 todo = conn.execute("SELECT id, file FROM images WHERE NOT faces_done ORDER BY id").fetchall()
 def run():
 for iid, f in todo:
 faces = detect_and_embed(read_bgr(DATA_DIR / f))
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
 futs = {ex.submit(describe, DATA_DIR / f): iid for iid, f in todo}
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
CLUSTER_SIM = 0.55 # AuraFace cosine for "same person" when grouping faces
GALLERY_SIM = 0.45 # a face this close to a gallery reference counts as a vote


def step_text(conn, batch=64):
 """Reranker text + description-meaning vector for every described image."""
 from .text import doc_text, embed_docs
 todo = conn.execute("SELECT id, vlm FROM images WHERE vlm IS NOT NULL AND desc_emb IS NULL").fetchall()
 def run():
 for i in range(0, len(todo), batch):
 chunk = todo[i:i + batch]
 docs = [doc_text(v) for _, v in chunk]
 vecs = embed_docs(docs)
 with conn.cursor() as cur:
 cur.executemany("UPDATE images SET doc_text = %s, desc_emb = %s WHERE id = %s",
 [(d, e, iid) for (iid, _), d, e in zip(chunk, docs, vecs)])
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
 1. Luna guesses -> image_persons(source='vlm')
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

 cl = _clusters(conn)
 with conn.cursor() as cur:
 cur.executemany("UPDATE faces SET cluster_id = %s WHERE id = %s", [(c, f) for f, c in cl.items()])
 faces = conn.execute("SELECT id, image_id FROM faces").fetchall()
 nfaces = {}
 for _, iid in faces:
 nfaces[iid] = nfaces.get(iid, 0) + 1
 gallery = {fid: (pid, sim) for fid, pid, sim in conn.execute(f"""
 SELECT f.id, best.person_id, best.sim FROM faces f, LATERAL (
 SELECT person_id, 1 - (r.emb <=> f.emb) AS sim FROM person_refs r
 WHERE r.source NOT LIKE 'vlm:%%' ORDER BY r.emb <=> f.emb LIMIT 1) best
 WHERE best.sim >= {GALLERY_SIM}""")}
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
 a = ap.parse_args()
 init_schema()
 conn = connect()
 steps = a.steps.split(",")
 folder = Path(a.dir).resolve()
 print(f"ingest {folder}")
 if "register" in steps:
 t = time.perf_counter(); n = register(conn, folder)
 print(f" register {n:>6} images {time.perf_counter() - t:7.1f}s")
 for name, fn in [("embed", step_embed), ("faces", step_faces), ("vlm", lambda c: step_vlm(c, a.workers)), ("text", step_text), ("people", step_people)]:
 if name in steps:
 todo, run = fn(conn)
 timed(name, len(todo), run)
 if "vlm" in steps and "people" not in steps:
 rebuild_search_text(conn)


if __name__ == "__main__":
 main()
