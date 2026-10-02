"""Embed all thumbnails with a candidate model into exp_image_embedding_<key>. Usage: python embed_all.py fgclip2_base"""
import sys, time, json, gc
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
from models import MODELS
from app.db import connect
from app.config import THUMB_DIR

key = sys.argv[1]
conn = connect()
t0 = time.perf_counter(); m = MODELS[key](); load_s = time.perf_counter() - t0
ids = [r[0] for r in conn.execute("SELECT id FROM images ORDER BY id")]
probe = m.embed_images([Image.open(THUMB_DIR / f"{ids[0]}.jpg")]); dim = probe.shape[1]
tbl = f"exp_image_embedding_{key}"
conn.execute(f"CREATE TABLE IF NOT EXISTS {tbl} (image_id text PRIMARY KEY, emb vector({dim}))")
done = {r[0] for r in conn.execute(f"SELECT image_id FROM {tbl}")}
todo = [i for i in ids if i not in done]
print(f"{key}: dim {dim}, {len(todo)} to embed, load {load_s:.1f}s", flush=True)
t0 = time.perf_counter(); n = 0
B = 16
for i in range(0, len(todo), B):
 chunk = todo[i:i + B]
 vecs = m.embed_images([Image.open(THUMB_DIR / f"{iid}.jpg") for iid in chunk])
 with conn.cursor() as cur:
 cur.executemany(f"INSERT INTO {tbl} VALUES (%s, %s) ON CONFLICT DO NOTHING", [(iid, v) for iid, v in zip(chunk, vecs)])
 n += len(chunk)
 if n % 400 == 0:
 print(f" {n}/{len(todo)} {1000*(time.perf_counter()-t0)/n:.0f} ms/img", flush=True)
img_ms = 1000 * (time.perf_counter() - t0) / max(n, 1)
conn.execute(f"CREATE INDEX IF NOT EXISTS {tbl}_hnsw ON {tbl} USING hnsw (emb vector_cosine_ops)")
qs = ["modi in pink jacket", "farmers protesting on a highway", "flood", "cricket team celebrating in rain", "press conference"]
m.embed_text(qs[:1]); t0 = time.perf_counter()
for q in qs: m.embed_text([q])
txt_ms = 1000 * (time.perf_counter() - t0) / len(qs)
stats = {"model": key, "dim": dim, "img_ms": round(img_ms), "txt_ms": round(txt_ms), "n": n, "load_s": round(load_s, 1), "threads": torch.get_num_threads()}
print(json.dumps(stats))
with open(Path(__file__).parent / "timings.jsonl", "a") as f: f.write(json.dumps(stats) + "\n")
