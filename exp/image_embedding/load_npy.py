"""Load GPU-made embeddings into exp_image_embedding_<model> tables. Usage: python -m exp.image_embedding.load_npy out/"""
import sys, json
from pathlib import Path
import numpy as np
from app.db import connect

out = Path(sys.argv[1]); ids = json.load(open(out / "ids.json")); conn = connect()
queries = [l.strip() for l in open(out.parent / "queries.txt") if l.strip()] if (out.parent / "queries.txt").exists() else []
for f in sorted(out.glob("*.npy")):
 if f.stem.endswith("_queries"):
 continue
 key = f.stem; tbl = f"exp_image_embedding_{key}"
 if conn.execute("SELECT to_regclass(%s)", (tbl,)).fetchone()[0] and conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0] == len(ids):
 print(key, "already loaded"); continue
 V = np.load(f)
 conn.execute(f"DROP TABLE IF EXISTS {tbl}")
 conn.execute(f"CREATE TABLE {tbl} (image_id text PRIMARY KEY, emb vector({V.shape[1]}))")
 with conn.cursor() as cur:
 cur.executemany(f"INSERT INTO {tbl} VALUES (%s, %s)", list(zip(ids, V)))
 # no HNSW: 5k rows scan exactly in a few ms, and index builds are slow on a 2-thread budget
 qf = out / f"{key}_queries.npy"
 if qf.exists() and queries:
 Q = np.load(qf); qt = f"{tbl}_queries"
 conn.execute(f"DROP TABLE IF EXISTS {qt}"); conn.execute(f"CREATE TABLE {qt} (query text PRIMARY KEY, emb vector({Q.shape[1]}))")
 with conn.cursor() as cur:
 cur.executemany(f"INSERT INTO {qt} VALUES (%s, %s) ON CONFLICT DO NOTHING", list(zip(queries, Q)))
 print(key, V.shape, "loaded")
