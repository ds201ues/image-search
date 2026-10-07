"""E20 dense_desc: does a denser index-time "what is happening" text help stage 2 on newsroom queries?

Hypothesis: newsroom misses are "right topic, not the exact scene" (water tanker without a queue). The stage-2 document (caption +
structured description) rarely states the specific action. A literal, dense list of what is visible, written once at index time,
gives the text-only decision model something to judge against. Zero query-time cost.

Scope: the 3,702 photos that enter stage 2 for the 40 newsroom queries (recall is unchanged, so only their doc_text matters).
 uv run python exp/dense_desc/run.py describe # Luna, cached in exp/dense_desc/details.jsonl (~$1)
 uv run python exp/dense_desc/run.py build # exp_dense_images = images with doc_text + " Details: ..."
 uv run python -m eval.run_eval --types newsroom --images-table exp_dense_images --label newsroom_dense --show-misses
"""
import base64, io, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from openai import OpenAI
from PIL import Image
from app.config import VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT, THUMB_DIR
from app.db import connect

HERE = Path(__file__).parent
CACHE = HERE / "details.jsonl"
IDS = Path("/tmp/claude-502/newsroom_candidates.txt")
PROMPT = """You index a news photo archive. Describe exactly what is visible in this photo as 5-8 short, literal, factual sentences:
the main action (who is doing what, by role, never by name), how many people, what they hold or wear, vehicles and objects, the
setting and its condition (road, water level, crowd density, damage), weather and light, and any readable text or signs.
No interpretation, no guessing about the event or the city unless a sign says it. Plain prose, no lists, no headings."""


def describe_one(client, iid):
 im = Image.open(THUMB_DIR / f"{iid}.jpg").convert("RGB"); im.thumbnail((768, 768))
 buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
 t = time.perf_counter()
 r = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": [{"type": "text", "text": PROMPT},
 {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), "detail": "auto"}}]}])
 u = r.usage
 return {"id": iid, "details": r.choices[0].message.content.strip(), "ms": int(1000 * (time.perf_counter() - t)),
 "usd": (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6}


def describe():
 ids = IDS.read_text().split()
 done = {json.loads(l)["id"] for l in CACHE.open()} if CACHE.exists() else set()
 todo = [i for i in ids if i not in done]
 print(f"{len(ids)} photos, {len(todo)} to describe")
 client = OpenAI(timeout=90, max_retries=3)
 usd = 0.0
 with CACHE.open("a") as f, ThreadPoolExecutor(8) as ex:
 for n, rec in enumerate(ex.map(lambda i: _safe(client, i), todo), 1):
 if rec:
 f.write(json.dumps(rec) + "\n"); f.flush(); usd += rec["usd"]
 if n % 200 == 0:
 print(f" {n}/{len(todo)} ${usd:.2f}", flush=True)
 print(f"done, ${usd:.2f}")


def _safe(client, iid):
 try:
 return describe_one(client, iid)
 except Exception as e:
 print("fail", iid, str(e)[:100]); return None


def build():
 conn = connect()
 det = {json.loads(l)["id"]: json.loads(l)["details"] for l in CACHE.open()}
 conn.execute("DROP TABLE IF EXISTS exp_dense_images")
 conn.execute("CREATE TABLE exp_dense_images AS SELECT * FROM images")
 conn.execute("CREATE INDEX ON exp_dense_images (id)")
 conn.execute("CREATE INDEX ON exp_dense_images USING gin (tsv)")
 conn.execute("CREATE INDEX ON exp_dense_images USING hnsw (clip vector_cosine_ops)")
 conn.execute("CREATE INDEX ON exp_dense_images USING hnsw (desc_emb vector_cosine_ops)")
 with conn.cursor() as cur:
 cur.executemany("UPDATE exp_dense_images SET doc_text = coalesce(doc_text, '') || ' Details: ' || %s WHERE id = %s",
 [(d, i) for i, d in det.items()])
 n = conn.execute("SELECT count(*) FROM exp_dense_images WHERE doc_text LIKE '%% Details: %%'").fetchone()[0]
 print(f"exp_dense_images built, {n} photos with details")


if __name__ == "__main__":
 {"describe": describe, "build": build}[sys.argv[1]]()
