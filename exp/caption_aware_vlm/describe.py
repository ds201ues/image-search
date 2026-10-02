"""E3: describe every photo with GPT-6 Luna, giving it the photographer caption as context.
uv run python -m exp.caption_aware_vlm.describe --workers 24
Cache: exp/caption_aware_vlm/vlm_cache.jsonl (one line per image id)."""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI
from app.config import THUMB_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT, VLM_REASONING, VLM_DETAIL
from app.vlm import PROMPT as BASE_PROMPT, SCHEMA as BASE_SCHEMA, _jpeg_b64
from app.db import connect

HERE = Path(__file__).parent
CACHE = HERE / "vlm_cache.jsonl"

PROMPT = BASE_PROMPT + """

Photographer's caption and headline for this photo (may be empty, may contain typos, may name people who are
NOT visible, e.g. the person a crowd is waiting for, or a person whose convoy passed by):
CAPTION: {caption}
HEADLINE: {headline}

Use the caption as context, but describe only what is visible:
- recognized_people: give canonical full English names (e.g. "N. Chandrababu Naidu", "Mamata Banerjee"). If the caption names
 a person, include them only if a person consistent with that name (role, age, gender, appearance, setting) is actually
 visible; then your confidence may be high (0.85-0.98). If you recognise someone the caption does not name, include them
 with your own confidence. Put caption names you could NOT match to a visible person into caption_names_not_visible.
- caption_consistent: false if the caption clearly contradicts what is visible (wrong event, place, people count, object).
- location_guess / place_type / event_type may use the caption's place and event when consistent with the image.
- likely_queries: 3-5 short lowercase searches (2-6 words) a journalist would type to find THIS photo, mixing names, event,
 place and visual details (e.g. "naidu press meet chennai", "flooded road bengaluru two wheeler").
- Still do NOT guess dates."""

SCHEMA = json.loads(json.dumps(BASE_SCHEMA))
SCHEMA["properties"]["caption_consistent"] = {"type": "boolean"}
SCHEMA["properties"]["caption_names_not_visible"] = {"type": "array", "items": {"type": "string"}}
SCHEMA["properties"]["likely_queries"] = {"type": "array", "items": {"type": "string"}}
SCHEMA["required"] += ["caption_consistent", "caption_names_not_visible", "likely_queries"]

_client = OpenAI(timeout=90, max_retries=3)
_lock = threading.Lock()


def describe(iid, caption, headline):
 t = time.perf_counter()
 r = _client.chat.completions.create(
 model=VLM_MODEL,
 messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT.format(caption=caption or "", headline=headline or "")},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(THUMB_DIR / f'{iid}.jpg')}", "detail": VLM_DETAIL}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "image_index_cap", "strict": True, "schema": SCHEMA}},
 **({"reasoning_effort": VLM_REASONING} if VLM_REASONING else {}))
 u = r.usage
 return {"image_id": iid, "vlm": json.loads(r.choices[0].message.content), "model": VLM_MODEL,
 "ms": int((time.perf_counter() - t) * 1000), "tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens,
 "cost_usd": (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6}


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=24); ap.add_argument("--limit", type=int)
 a = ap.parse_args()
 done = set()
 if CACHE.exists():
 for l in CACHE.open():
 try: done.add(json.loads(l)["image_id"])
 except Exception: pass
 rows = connect().execute("SELECT id, caption, headline FROM images ORDER BY id").fetchall()
 todo = [r for r in rows if r[0] not in done][: a.limit or None]
 print(f"{len(done)} cached, {len(todo)} to do", flush=True)
 stats = {"cost": 0.0, "tin": 0, "tout": 0, "fail": 0, "n": 0}
 t0 = time.perf_counter()
 with ThreadPoolExecutor(a.workers) as ex, CACHE.open("a") as f:
 futs = {ex.submit(describe, *r): r[0] for r in todo}
 for fut in as_completed(futs):
 try:
 r = fut.result()
 except Exception as e:
 stats["fail"] += 1; print(f" fail {futs[fut]}: {str(e)[:120]}", flush=True); continue
 with _lock:
 f.write(json.dumps(r) + "\n"); f.flush()
 stats["cost"] += r["cost_usd"]; stats["tin"] += r["tokens_in"]; stats["tout"] += r["tokens_out"]; stats["n"] += 1
 if stats["n"] % 250 == 0:
 print(f" {stats['n']}/{len(todo)} ${stats['cost']:.3f} {time.perf_counter() - t0:.0f}s", flush=True)
 n = max(stats["n"], 1)
 print(f"done: {stats['n']} ok, {stats['fail']} failed, ${stats['cost']:.4f} total = ${1000 * stats['cost'] / n:.3f}/1k images, "
 f"{stats['tin'] // n} in / {stats['tout'] // n} out tokens per image, {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
 main()
