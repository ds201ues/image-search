import psycopg
from pgvector.psycopg import register_vector
from .config import DATABASE_URL, CLIP_DIM, TEXT_DIM

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS images (
 id text PRIMARY KEY,
 file text NOT NULL, -- path relative to data/
 width int, height int,
 date_published timestamptz, -- from source metadata only, never from a model
 section text,
 source_url text,
 ref_title text, -- reference only (eval), never searched
 vlm jsonb, -- structured description
 vlm_model text,
 vlm_ms int,
 vlm_cost_usd double precision,
 clip vector({CLIP_DIM}), -- SigLIP image embedding
 search_text text, -- flattened VLM fields + person names
 tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(search_text, ''))) STORED,
 faces_done boolean DEFAULT false,
 ingested_at timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS images_tsv ON images USING gin (tsv);
CREATE INDEX IF NOT EXISTS images_clip ON images USING hnsw (clip vector_cosine_ops);
CREATE INDEX IF NOT EXISTS images_date ON images (date_published);
ALTER TABLE images ADD COLUMN IF NOT EXISTS doc_text text; -- what the reranker reads
ALTER TABLE images ADD COLUMN IF NOT EXISTS desc_emb vector({TEXT_DIM}); -- meaning of the description
CREATE INDEX IF NOT EXISTS images_desc_emb ON images USING hnsw (desc_emb vector_cosine_ops);

CREATE TABLE IF NOT EXISTS persons (
 id serial PRIMARY KEY,
 name text UNIQUE NOT NULL,
 aliases text[] DEFAULT '{{}}',
 wikidata_id text,
 popularity int DEFAULT 0 -- Wikidata sitelinks, used to break name ties
);

CREATE TABLE IF NOT EXISTS person_refs ( -- reference face embeddings (gallery)
 id serial PRIMARY KEY,
 person_id int REFERENCES persons(id) ON DELETE CASCADE,
 emb vector(512) NOT NULL,
 source text
);
CREATE INDEX IF NOT EXISTS person_refs_emb ON person_refs USING hnsw (emb vector_cosine_ops);

CREATE TABLE IF NOT EXISTS faces (
 id serial PRIMARY KEY,
 image_id text REFERENCES images(id) ON DELETE CASCADE,
 bbox int[4], -- x, y, w, h in original pixels
 det_score real,
 emb vector(512) NOT NULL,
 person_id int REFERENCES persons(id),
 match_score real
);
CREATE INDEX IF NOT EXISTS faces_image ON faces (image_id);
ALTER TABLE faces ADD COLUMN IF NOT EXISTS cluster_id int; -- same person across the archive
CREATE INDEX IF NOT EXISTS faces_cluster ON faces (cluster_id);

CREATE TABLE IF NOT EXISTS image_persons ( -- who is in each image, from the VLM or face matching
 image_id text REFERENCES images(id) ON DELETE CASCADE,
 person_id int REFERENCES persons(id) ON DELETE CASCADE,
 source text NOT NULL, -- evidence: 'vlm' (Luna guess) | 'cluster' (face-cluster vote)
 score real, -- that source's confidence 0..1; nothing is thrown away
 PRIMARY KEY (image_id, person_id, source)
);
CREATE INDEX IF NOT EXISTS image_persons_person ON image_persons (person_id);

-- combined belief that a person is in an image: noisy-OR over independent evidence
CREATE OR REPLACE VIEW image_person_scores AS
SELECT image_id, person_id, 1 - exp(sum(ln(greatest(1 - score, 1e-6)))) AS score,
 array_agg(source || ':' || round(score::numeric, 2) ORDER BY source) AS evidence
FROM image_persons GROUP BY image_id, person_id;
CREATE INDEX IF NOT EXISTS faces_person ON faces (person_id);
-- source metadata read from the files (IPTC / EXIF / XMP). Captions are evidence, like the VLM description.
ALTER TABLE images ADD COLUMN IF NOT EXISTS caption text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS headline text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS keywords text[];
ALTER TABLE images ADD COLUMN IF NOT EXISTS city text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS state text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS country text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS byline text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS credit text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS camera text;
ALTER TABLE images ADD COLUMN IF NOT EXISTS date_source text; -- exif_original | iptc | xmp | exif_file | caption_text | api
ALTER TABLE images ADD COLUMN IF NOT EXISTS meta jsonb; -- everything else read from the file
ALTER TABLE images ADD COLUMN IF NOT EXISTS orig_file text; -- original path when the indexed file is derived (PDF)
CREATE INDEX IF NOT EXISTS images_city ON images (lower(city));
"""


def connect():
 conn = psycopg.connect(DATABASE_URL, autocommit=True)
 conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
 register_vector(conn)
 return conn


def init_schema():
 with connect() as c:
 c.execute(SCHEMA)


def flatten_vlm(v: dict | None) -> str:
 """Turn the VLM JSON into one searchable text blob."""
 if not v:
 return ""
 # names are indexed only via image_persons (after confidence rules), so drop any the VLM leaked into tags
 names = {(x.get("name") or "").lower() for x in v.get("recognized_people") or []}
 v = {**v, "tags": [t for t in v.get("tags") or [] if t.lower() not in names]}
 parts = [v.get("summary", ""), v.get("place_type", ""), v.get("location_guess", ""),
 v.get("event_type", ""), v.get("mood", ""), v.get("setting", ""),
 v.get("time_of_day", ""), v.get("image_type", "")]
 for key in ("landmarks", "objects", "activities", "visible_text", "dominant_colors", "tags"):
 parts += v.get(key) or []
 for p in v.get("people") or []:
 parts += [p.get("description", ""), p.get("clothing", ""), p.get("action", ""), p.get("expression", "")]
 parts += p.get("colors") or []
 return " ".join(x for x in parts if x)


def caption_text(caption, headline, keywords, city, state) -> str:
 """Photographer metadata as searchable text (names included: the caption is evidence of who is there),
 plus the canonical city token so Bangalore/Bengaluru variants match."""
 from .aliases import canonical_city
 parts = [caption or "", headline or "", " ".join(keywords or []), city or "", state or ""]
 canon = canonical_city(" ".join([city or "", caption or ""]))
 if canon:
 parts.append(canon)
 return " ".join(x for x in parts if x)


def rebuild_search_text(conn, image_id: str | None = None):
 """search_text = flattened VLM description + photographer caption fields. People are ranked through image_person_scores."""
 where = "WHERE id = %s" if image_id else ""
 rows = conn.execute(f"SELECT id, vlm, caption, headline, keywords, city, state FROM images {where}",
 (image_id,) if image_id else ()).fetchall()
 with conn.cursor() as cur:
 cur.executemany("UPDATE images SET search_text = %s WHERE id = %s",
 [((flatten_vlm(v) + " " + caption_text(c, h, k, ci, st)).strip(), i) for i, v, c, h, k, ci, st in rows])
