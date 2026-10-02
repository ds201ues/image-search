"""E3: build exp_caption_aware_vlm_* tables from the cached Luna output.
uv run python -m exp.caption_aware_vlm.build
Tables: exp_caption_aware_vlm_images (copy of images with vlm/doc_text/search_text/desc_emb replaced; search_text_lq adds likely_queries),
 exp_caption_aware_vlm_images_lq (same but tsv built from search_text + likely_queries),
 exp_caption_aware_vlm_image_persons + view exp_caption_aware_vlm_scores (Luna names only, no face clustering: E2 covers that)."""
import json
from pathlib import Path
import torch
torch.set_num_threads(4)
from app.db import connect, flatten_vlm
from app.text import doc_text, embed_docs

HERE = Path(__file__).parent
VLM_MIN_CONF = 0.3 # same as app/ingest.py
P = "exp_caption_aware_vlm"


def main():
 conn = connect()
 cache = {}
 for l in (HERE / "vlm_cache.jsonl").open():
 try:
 r = json.loads(l); cache[r["image_id"]] = r
 except Exception:
 pass
 print(f"{len(cache)} described")
 for suffix in ("", "_lq"):
 t = f"{P}_images{suffix}"
 conn.execute(f"DROP TABLE IF EXISTS {t}")
 # tsv is a generated column in images; rebuild it explicitly here from our own search_text
 conn.execute(f"""CREATE TABLE {t} AS SELECT id, file, width, height, date_published, section, source_url, ref_title, vlm, vlm_model,
 vlm_ms, vlm_cost_usd, clip, search_text, faces_done, ingested_at, doc_text, desc_emb, caption, headline, keywords,
 city, state, country, byline, credit, camera, date_source, meta, orig_file FROM images""")
 conn.execute(f"ALTER TABLE {t} ADD PRIMARY KEY (id)")
 rows = conn.execute(f"SELECT id FROM {P}_images ORDER BY id").fetchall()
 ids = [r[0] for r in rows if r[0] in cache]
 print(f"{len(ids)} to update")
 B = 64
 for i in range(0, len(ids), B):
 chunk = ids[i:i + B]
 vlms = [cache[x]["vlm"] for x in chunk]
 docs = [doc_text(v) for v in vlms]
 vecs = embed_docs(docs)
 with conn.cursor() as cur:
 cur.executemany(f"""UPDATE {P}_images SET vlm=%s, vlm_model=%s, doc_text=%s, desc_emb=%s, search_text=%s WHERE id=%s""",
 [(json.dumps(v), cache[x]["model"], d, e, flatten_vlm(v), x) for x, v, d, e in zip(chunk, vlms, docs, vecs)])
 cur.executemany(f"""UPDATE {P}_images_lq SET vlm=%s, vlm_model=%s, doc_text=%s, desc_emb=%s, search_text=%s WHERE id=%s""",
 [(json.dumps(v), cache[x]["model"], d, e, flatten_vlm(v) + " " + " ".join(v.get("likely_queries") or []), x)
 for x, v, d, e in zip(chunk, vlms, docs, vecs)])
 if (i // B) % 10 == 0:
 print(f" {i + len(chunk)}/{len(ids)}", flush=True)
 for suffix in ("", "_lq"):
 t = f"{P}_images{suffix}"
 conn.execute(f"ALTER TABLE {t} ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(search_text, ''))) STORED")
 conn.execute(f"CREATE INDEX ON {t} USING gin (tsv)")
 conn.execute(f"CREATE INDEX ON {t} USING hnsw (clip vector_cosine_ops)")
 conn.execute(f"CREATE INDEX ON {t} USING hnsw (desc_emb vector_cosine_ops)")
 conn.execute(f"CREATE INDEX ON {t} (date_published)")

 # people evidence from the new names (same rules as app/ingest.py step_people, source 'vlm'); persons table shared with live
 conn.execute(f"DROP VIEW IF EXISTS {P}_scores"); conn.execute(f"DROP TABLE IF EXISTS {P}_image_persons")
 conn.execute(f"CREATE TABLE {P}_image_persons (image_id text, person_id int, source text, score real, PRIMARY KEY (image_id, person_id, source))")
 vlm = {}
 for iid in ids:
 for x in cache[iid]["vlm"].get("recognized_people") or []:
 name = " ".join((x.get("name") or "").split()); conf = float(x.get("confidence") or 0)
 if not name or conf < VLM_MIN_CONF:
 continue
 pid = conn.execute("""INSERT INTO persons (name) VALUES (%s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
 RETURNING id""", (name,)).fetchone()[0]
 vlm.setdefault(iid, {})[pid] = max(conf, vlm.get(iid, {}).get(pid, 0))
 with conn.cursor() as cur:
 cur.executemany(f"INSERT INTO {P}_image_persons VALUES (%s, %s, 'vlm', %s)", [(i, p, c) for i, d in vlm.items() for p, c in d.items()])
 # keep the live face-cluster votes as independent evidence so the comparison isolates the Luna-name change
 conn.execute(f"INSERT INTO {P}_image_persons SELECT image_id, person_id, source, score FROM image_persons WHERE source <> 'vlm'")
 conn.execute(f"""CREATE VIEW {P}_scores AS
 SELECT image_id, person_id, 1 - exp(sum(ln(greatest(1 - score, 1e-6)))) AS score,
 array_agg(source || ':' || round(score::numeric, 2) ORDER BY source) AS evidence
 FROM {P}_image_persons GROUP BY image_id, person_id""")
 st = conn.execute(f"""SELECT count(DISTINCT image_id) FILTER (WHERE score >= 0.8), count(*) FROM {P}_scores""").fetchone()
 base = conn.execute("SELECT count(DISTINCT image_id) FILTER (WHERE score >= 0.8), count(*) FROM image_person_scores").fetchone()
 print(f"people: {sum(len(d) for d in vlm.values())} Luna guesses (baseline 743); photos with a name >= 0.8: {st[0]} (baseline {base[0]}); pairs {st[1]} (baseline {base[1]})")


if __name__ == "__main__":
 main()
