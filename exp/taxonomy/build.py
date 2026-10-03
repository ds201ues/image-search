"""Build exp_taxonomy_images (images copy with v2 description for the sample) and report concept coverage. Usage: uv run python -m exp.taxonomy.build"""
import json
from app.db import connect, flatten_vlm, caption_text
from app.text import doc_text, embed_docs
from app.vlm_v2 import concepts_line
conn = connect()
rows = conn.execute("SELECT t.image_id, t.vlm, i.caption, i.headline, i.keywords, i.city, i.state, i.date_published FROM exp_taxonomy_vlm t JOIN images i ON i.id = t.image_id").fetchall()
print(f"v2 descriptions: {len(rows)}")
conn.execute("DROP TABLE IF EXISTS exp_taxonomy_images")
conn.execute("""CREATE TABLE exp_taxonomy_images AS SELECT id, file, width, height, date_published, section, source_url, ref_title, vlm, vlm_model, vlm_ms,
 vlm_cost_usd, clip, search_text, faces_done, ingested_at, doc_text, desc_emb, caption, headline, keywords, city, state, country, byline, credit, camera,
 date_source, meta, orig_file FROM images""")
batch = 64
for k in range(0, len(rows), batch):
 ch = rows[k:k + batch]
 st, dt, emb_src = [], [], []
 for iid, v, cap, head, kw, city, state, date in ch:
 v2 = {**v, "tags": list(dict.fromkeys((v.get("tags") or []) + [o.lower() for o in v.get("objects") or []]))}
 conc = concepts_line(v)
 st.append(" ".join([flatten_vlm(v2), conc, caption_text(cap, head, kw, city, state)]))
 d = doc_text(v2, cap, date)
 dt.append(conc + " " + d)
 emb_src.append(conc + " " + doc_text(v2, cap, date, for_embedding=True))
 vecs = embed_docs(emb_src)
 with conn.cursor() as cur:
 cur.executemany("UPDATE exp_taxonomy_images SET vlm = %s, search_text = %s, doc_text = %s, desc_emb = %s, vlm_model = 'v2' WHERE id = %s",
 [(json.dumps(v), s, d, e, iid) for (iid, v, *_), s, d, e in zip(ch, st, dt, vecs)])
conn.execute("ALTER TABLE exp_taxonomy_images ADD COLUMN tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(search_text, ''))) STORED")
conn.execute("ALTER TABLE exp_taxonomy_images ADD PRIMARY KEY (id)")
conn.execute("CREATE INDEX ON exp_taxonomy_images USING gin (tsv)"); conn.execute("CREATE INDEX ON exp_taxonomy_images USING hnsw (clip vector_cosine_ops)")
conn.execute("CREATE INDEX ON exp_taxonomy_images USING hnsw (desc_emb vector_cosine_ops)"); conn.execute("CREATE INDEX ON exp_taxonomy_images (date_published)")
# cost and concept coverage
print("tokens/cost:", conn.execute("SELECT round(avg(tokens_in)), round(avg(tokens_out)), round(sum(cost_usd)::numeric, 3) FROM exp_taxonomy_vlm").fetchone(),
 "| v1 avg out tokens ~490 at $0.20/1k (batch)")
import collections
ev = collections.Counter(r[1].get("event_type") for r in rows); print("event types:", ev.most_common(12))
roles = collections.Counter(x for r in rows for x in r[1].get("actor_roles") or []); print("roles:", roles.most_common(10))
sym = collections.Counter(s.get("label") for r in rows for s in r[1].get("symbols") or []); print("symbols:", sym.most_common(12))
# did the gap close? caption words previously missing from descriptions, now present
import re
STOP = set("the a an and or of in on at to for with by from as is are was were".split())
tok = lambda s: set(w for w in re.findall(r"[a-z][a-z\-]{2,}", (s or "").lower()) if w not in STOP)
chk = ["minister", "chief", "police", "students", "stadium", "party", "festival", "press", "conference", "election", "women", "match", "school", "college", "protest", "rally"]
old = conn.execute("SELECT i.caption, i.vlm FROM images i JOIN exp_taxonomy_vlm t ON t.image_id = i.id WHERE i.caption IS NOT NULL").fetchall()
new = {r[0]: r[1] for r in rows}
print("caption word present in description, v1 -> v2 (share of captions using the word):")
for w in chk:
 caps = [(c, v) for c, v in old if w in tok(c)]
 if not caps: continue
 v1 = sum(1 for c, v in caps if w in tok(flatten_vlm(v))) / len(caps)
 ids = conn.execute("SELECT t.image_id FROM exp_taxonomy_vlm t JOIN images i ON i.id=t.image_id WHERE i.caption IS NOT NULL").fetchall()
 v2 = sum(1 for c, v in caps if w in tok(flatten_vlm(new[conn.execute("SELECT id FROM images WHERE caption=%s LIMIT 1", (c,)).fetchone()[0]]) + " " + concepts_line(new[conn.execute("SELECT id FROM images WHERE caption=%s LIMIT 1", (c,)).fetchone()[0]]))) / len(caps) if False else None
 print(f" {w:<12} n={len(caps):>4} v1 {v1:.0%}")
