"""Build the three variant tables for E1 (copies of images; live table untouched).
a: search_text/tsv += caption, headline, keywords, city, state
b: a + desc_emb re-embedded from "Caption: ... " + doc_text
c: b + doc_text (what the reranker reads) = "Caption: ... " + doc_text
"""
import torch; torch.set_num_threads(4)
from app.db import connect
from app.text import embed_docs

def meta_text(cap, head, kws, city, state):
 parts = [cap or "", head or "", " ".join(kws or []), city or "", state or ""]
 return " ".join(p for p in parts if p).strip()

def cap_prefix(cap, head):
 c = (cap or head or "").strip()
 return f"Caption: {c} " if c else ""

conn = connect()
for v in "abc":
 conn.execute(f"DROP TABLE IF EXISTS exp_caption_retrieval_{v}")
 conn.execute(f"""CREATE TABLE exp_caption_retrieval_{v} AS
 SELECT id, file, width, height, date_published, section, source_url, ref_title, vlm, vlm_model, clip,
 search_text, doc_text, desc_emb, caption, headline, keywords, city, state, tsv::tsvector AS tsv
 FROM images""")
 conn.execute(f"ALTER TABLE exp_caption_retrieval_{v} ADD PRIMARY KEY (id)")
rows = conn.execute("SELECT id, search_text, doc_text, caption, headline, keywords, city, state FROM images ORDER BY id").fetchall()
# a, b, c: full-text gets the metadata
with conn.cursor() as cur:
 for v in "abc":
 cur.executemany(f"""UPDATE exp_caption_retrieval_{v} SET search_text = %s,
 tsv = to_tsvector('english', %s) WHERE id = %s""",
 [((st or "") + " " + meta_text(cp, h, k, ci, s), (st or "") + " " + meta_text(cp, h, k, ci, s), i)
 for i, st, dt, cp, h, k, ci, s in rows])
# b, c: description vector from caption + doc_text
docs = [cap_prefix(cp, h) + (dt or "") for i, st, dt, cp, h, k, ci, s in rows]
vecs = embed_docs(docs)
with conn.cursor() as cur:
 for v in "bc":
 cur.executemany(f"UPDATE exp_caption_retrieval_{v} SET desc_emb = %s WHERE id = %s", [(e, r[0]) for r, e in zip(rows, vecs)])
 # c: reranker doc
 cur.executemany("UPDATE exp_caption_retrieval_c SET doc_text = %s WHERE id = %s", [(d, r[0]) for r, d in zip(rows, docs)])
for v in "abc":
 conn.execute(f"CREATE INDEX ON exp_caption_retrieval_{v} USING gin (tsv)")
 conn.execute(f"CREATE INDEX ON exp_caption_retrieval_{v} USING hnsw (clip vector_cosine_ops)")
 conn.execute(f"CREATE INDEX ON exp_caption_retrieval_{v} USING hnsw (desc_emb vector_cosine_ops)")
 conn.execute(f"CREATE INDEX ON exp_caption_retrieval_{v} (date_published)")
print("built", conn.execute("select count(*) from exp_caption_retrieval_c where doc_text like 'Caption:%'").fetchone())
