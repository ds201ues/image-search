"""E9: describe a fixed 300-photo sample with cheaper Luna configurations. Results cached per variant in out/<variant>.jsonl.

uv run python -m exp.vlm_cost.variants --variant detail_low [--workers 16]
Variants: baseline (copy from images.vlm), detail_low, no_reasoning, thumb512, compact, model:<id>, batch (submits a Batch API job)
"""
import argparse, base64, io, json, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from openai import OpenAI
import app.config as C
from app.db import connect
from app.vlm import PROMPT, SCHEMA, NAMED, PERSON

HERE = Path(__file__).parent; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
SAMPLE = HERE / "sample_ids.json"
PRICES = { # USD per 1M tokens (in, out); see results.md for sources
 "gpt-6-luna": (0.10, 0.50), "gpt-5.4-nano": (0.20, 1.25), "gpt-5-nano": (0.05, 0.40), "gpt-4.1-nano": (0.10, 0.40),
 "gpt-5.6-luna": (0.20, 1.20), "gpt-5.4-mini": (0.25, 2.00),
}
COMPACT = {"type": "object", "additionalProperties": False,
 "required": ["recognized_people", "summary", "people", "place", "event_type", "objects", "tags"],
 "properties": {"recognized_people": {"type": "array", "items": NAMED},
 "summary": {"type": "string", "description": "2-3 sentence factual, visual description: who (described), what, where, clothing colours, actions"},
 "people": {"type": "array", "items": {"type": "string"}, "description": "one line per prominent person: description, clothing with colours, action"},
 "place": {"type": "string"}, "event_type": {"type": "string"},
 "objects": {"type": "array", "items": {"type": "string"}},
 "tags": {"type": "array", "items": {"type": "string"}}}}
COMPACT_PROMPT = PROMPT.replace("tags: 15-30 lowercase search keywords", "tags: 10-20 lowercase search keywords")


def sample_ids(n=300, seed=9):
 if SAMPLE.exists():
 return json.loads(SAMPLE.read_text())
 ids = [r[0] for r in connect().execute("SELECT id FROM images WHERE vlm IS NOT NULL ORDER BY id")]
 random.Random(seed).shuffle(ids); ids = sorted(ids[:n]); SAMPLE.write_text(json.dumps(ids)); return ids


def b64(iid, side=768):
 im = Image.open(C.THUMB_DIR / f"{iid}.jpg").convert("RGB")
 if side < max(im.size):
 im.thumbnail((side, side))
 buf = io.BytesIO(); im.save(buf, "JPEG", quality=85); return base64.b64encode(buf.getvalue()).decode()


def spec(variant):
 """-> dict(model, detail, reasoning, side, schema, prompt)"""
 s = dict(model=C.VLM_MODEL, detail="auto", reasoning="low", side=768, schema=SCHEMA, prompt=PROMPT, name="image_index")
 if variant == "detail_low": s["detail"] = "low"
 elif variant == "no_reasoning": s["reasoning"] = None
 elif variant == "thumb512": s["side"] = 512
 elif variant == "compact": s.update(schema=COMPACT, prompt=COMPACT_PROMPT, name="image_compact")
 elif variant.startswith("model:"): s["model"] = variant.split(":", 1)[1]
 elif variant != "baseline": raise SystemExit(f"unknown variant {variant}")
 return s


def body(iid, s):
 msg = {"model": s["model"], "messages": [{"role": "user", "content": [
 {"type": "text", "text": s["prompt"]},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(iid, s['side'])}", "detail": s["detail"]}}]}],
 "response_format": {"type": "json_schema", "json_schema": {"name": s["name"], "strict": True, "schema": s["schema"]}}}
 if s["reasoning"]:
 msg["reasoning_effort"] = s["reasoning"]
 return msg


def one(client, iid, s):
 t = time.perf_counter()
 r = client.chat.completions.create(**body(iid, s))
 u = r.usage; pin, pout = PRICES.get(s["model"], (C.VLM_PRICE_IN, C.VLM_PRICE_OUT))
 return {"id": iid, "vlm": json.loads(r.choices[0].message.content), "ms": int((time.perf_counter() - t) * 1000),
 "tin": u.prompt_tokens, "tout": u.completion_tokens,
 "reasoning_tokens": getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
 "cost": (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6}


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--variant", required=True); ap.add_argument("--workers", type=int, default=16)
 a = ap.parse_args(); ids = sample_ids(); path = OUT / f"{a.variant.replace(':', '_')}.jsonl"
 if a.variant == "baseline":
 conn = connect(); rows = conn.execute("SELECT id, vlm, vlm_ms, vlm_cost_usd FROM images WHERE id = ANY(%s)", (ids,)).fetchall()
 with path.open("w") as f:
 for iid, v, ms, cost in rows:
 f.write(json.dumps({"id": iid, "vlm": v, "ms": ms, "cost": cost, "tin": None, "tout": None}) + "\n")
 print(f"baseline: {len(rows)} rows, ${1000 * sum(r[3] for r in rows) / len(rows):.3f} per 1k"); return
 s = spec(a.variant)
 if a.variant == "batch":
 raise SystemExit("use batch.py")
 done = {json.loads(l)["id"] for l in path.open()} if path.exists() else set()
 todo = [i for i in ids if i not in done]
 client = OpenAI(timeout=90, max_retries=3); fails = 0
 with ThreadPoolExecutor(a.workers) as ex, path.open("a") as f:
 futs = {ex.submit(one, client, i, s): i for i in todo}
 for fut in as_completed(futs):
 try:
 r = fut.result(); f.write(json.dumps(r) + "\n"); f.flush()
 except Exception as e:
 fails += 1; print("fail", futs[fut], str(e)[:120])
 rows = [json.loads(l) for l in path.open()]
 n = len(rows)
 print(f"{a.variant}: {n} ok, {fails} failed; ${1000 * sum(r['cost'] for r in rows) / n:.3f} per 1k; "
 f"tokens in {sum(r['tin'] for r in rows) // n} out {sum(r['tout'] for r in rows) // n} "
 f"(reasoning {sum(r['reasoning_tokens'] or 0 for r in rows) // n}); ms p50 {sorted(r['ms'] for r in rows)[n // 2]}")


if __name__ == "__main__":
 main()
