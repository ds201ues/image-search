"""Generate journalist-style test queries by showing GPT-6 Luna random archive photos.

uv run python -m eval.make_queries --n 60
Writes eval/gen_queries.jsonl. Queries describe a kind of photo a journalist would look for,
not a caption of one image, so several archive photos may satisfy each.
"""
import argparse, json, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
from app.config import DATA_DIR, VLM_MODEL
from app.db import connect
from app.vlm import _jpeg_b64

OUT = Path(__file__).parent / "gen_queries.jsonl"
PROMPT = """You are a journalist at an Indian newspaper searching the photo archive.
Looking at this photo, write ONE search query you might type that this photo would answer.
Type to write: {kind}.
- person: just a public figure's name as people type it (e.g. "yogi", "rahul gandhi", "kohli"), only if you recognise someone
- person_context: a public figure plus what they are doing / wearing / where ("modi waving at rally")
- scene: an event or situation without names ("protest outside court", "flooded street")
- attribute: objects, clothing, colours ("woman in red saree with microphone")
- place: a type of place or landmark ("parliament", "cricket stadium at night")
Write 2-7 words, lowercase, natural and slightly general (not a caption). If the type does not fit this photo, pick the closest type that does."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["query", "type"],
 "properties": {"query": {"type": "string"},
 "type": {"type": "string", "enum": ["person", "person_context", "scene", "attribute", "place"]}}}
KINDS = ["person", "person_context", "scene", "attribute", "place"]


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=60); ap.add_argument("--seed", type=int, default=7)
 a = ap.parse_args()
 rows = connect().execute("SELECT id, file FROM images WHERE vlm IS NOT NULL ORDER BY id").fetchall()
 random.Random(a.seed).shuffle(rows)
 client = OpenAI(timeout=60, max_retries=3)

 def one(k):
 iid, file = rows[k]
 r = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low", messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT.format(kind=KINDS[k % len(KINDS)])},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(DATA_DIR / file)}"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "q", "strict": True, "schema": SCHEMA}})
 return {**json.loads(r.choices[0].message.content), "source_image": iid}

 with ThreadPoolExecutor(12) as ex:
 qs = list(ex.map(one, range(min(a.n, len(rows)))))
 seen, out = set(), []
 for q in qs:
 if q["query"] not in seen:
 seen.add(q["query"]); out.append(q)
 OUT.write_text("".join(json.dumps(q) + "\n" for q in out))
 print(f"{len(out)} queries -> {OUT}")


if __name__ == "__main__":
 main()
