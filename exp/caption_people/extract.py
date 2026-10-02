"""Step 1: names in captions/headlines via GPT-6 Luna text-only. Cached in names.jsonl."""
import json, threading, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from openai import OpenAI
from app.config import VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT
from app.db import connect

HERE = Path(__file__).parent
CACHE = HERE / "names.jsonl"
PROMPT = """Below is a photographer's caption (and maybe a headline) for a news photo in an Indian newspaper archive.
List the people NAMED in the text who are likely to appear IN the photo. Rules:
- Include politicians, officials, sportspeople, actors, business people, and any clearly named individual who is the subject.
- Exclude the photographer / photo credit ("Photo by X", "Pic X", "agency photo", agency names), and people named only as context (e.g. "a photo of Gandhi on the wall", "fans of Kohli" when Kohli is not present, "X's residence", "named after Y").
- canonical_full_name: the common full English name as a journalist would write it (expand initials/nicknames only if you are sure, e.g. "N Chandrababu Naidu" -> "N. Chandrababu Naidu", "Mamata" -> "Mamata Banerjee", "Nitish Kumar" stays).
- confidence 0-1 that this person is actually in the photo.
Return an empty list if nobody is named."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["people"],
 "properties": {"people": {"type": "array", "items": {"type": "object", "additionalProperties": False,
 "required": ["name_as_written", "canonical_full_name", "role_or_title", "confidence"],
 "properties": {"name_as_written": {"type": "string"}, "canonical_full_name": {"type": "string"},
 "role_or_title": {"type": "string"}, "confidence": {"type": "number"}}}}}}
lock = threading.Lock(); spent = {"usd": 0.0, "n": 0}


def main():
 done = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); done[r["image_id"]] = r
 except json.JSONDecodeError:
 pass
 rows = connect().execute("SELECT id, caption, headline FROM images WHERE caption IS NOT NULL OR headline IS NOT NULL").fetchall()
 todo = [r for r in rows if r[0] not in done]
 print(f"{len(rows)} captioned, {len(todo)} to extract")
 client = OpenAI(timeout=60, max_retries=3)

 def one(r):
 iid, cap, head = r
 text = (f"Headline: {head}\n" if head else "") + (f"Caption: {cap}" if cap else "")
 try:
 resp = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": PROMPT + "\n\n" + text[:3000]}],
 response_format={"type": "json_schema", "json_schema": {"name": "names", "strict": True, "schema": SCHEMA}})
 out = json.loads(resp.choices[0].message.content)
 u = resp.usage
 with lock:
 spent["usd"] += (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6; spent["n"] += 1
 with CACHE.open("a") as f:
 f.write(json.dumps({"image_id": iid, "people": out["people"]}) + "\n")
 except Exception as e:
 print("fail", iid, e)

 with ThreadPoolExecutor(32) as ex:
 list(ex.map(one, todo))
 print(f"done: {spent['n']} calls, ${spent['usd']:.3f}")


if __name__ == "__main__":
 main()
