"""Name face clusters with GPT-6 Luna: the 2-3 clearest faces of one cluster (different photos) side by side, one call per cluster.
Independent evidence source 'cluster_vlm' in image_persons (noisy-OR with caption names, per-photo Luna guesses and face votes).
Cached per face set in data/cluster_names.jsonl, so re-clustering only re-asks for clusters whose members changed.
"""
import base64, io, json, threading
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from openai import OpenAI
from .config import DATA_DIR, THUMB_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT

CACHE = DATA_DIR / "cluster_names.jsonl"
CROP = 256 # px per face tile
CONTEXT = 2.2 # crop box = face box × this (hair, shoulders help recognition)
PROMPT = """These {n} crops show the SAME person in different news photographs from an Indian newspaper archive.
Who is this person? Only name a well-known public figure (politician, sportsperson, actor, business leader, official) you are
reasonably sure about, using the common full English name (e.g. "Narendra Modi"). Give a confidence from 0 to 1.
If you are not reasonably sure, return an empty name and confidence 0. Never guess a private or unknown person."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["name", "confidence"],
 "properties": {"name": {"type": "string"}, "confidence": {"type": "number"}}}
_lock = threading.Lock()
spent = {"usd": 0.0, "calls": 0}


def _tile(iid, bbox, orig_w):
 im = Image.open(THUMB_DIR / f"{iid}.jpg").convert("RGB")
 k = im.width / orig_w
 x, y, w, h = [v * k for v in bbox]
 cx, cy, s = x + w / 2, y + h / 2, max(w, h) * CONTEXT
 box = (int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(im.width, cx + s / 2)), int(min(im.height, cy + s / 2)))
 return im.crop(box).resize((CROP, CROP))


def strip(faces):
 """faces: [(image_id, bbox, orig_w)] -> JPEG bytes of the tiles side by side."""
 tiles = [_tile(*f) for f in faces]
 out = Image.new("RGB", (CROP * len(tiles) + 8 * (len(tiles) - 1), CROP), "white")
 for i, t in enumerate(tiles):
 out.paste(t, (i * (CROP + 8), 0))
 buf = io.BytesIO(); out.save(buf, "JPEG", quality=88)
 return buf.getvalue()


def name_clusters(clusters, workers=24):
 """clusters: {cluster_id: [(face_id, image_id, bbox, orig_w), ...]} (already the 2-3 chosen faces)
 -> {cluster_id: (name, confidence)}; only uncached face sets call the API."""
 cache = {}
 if CACHE.exists():
 for l in CACHE.open():
 try:
 r = json.loads(l); cache[r["key"]] = (r["name"], r["confidence"])
 except (json.JSONDecodeError, KeyError):
 pass
 client = OpenAI(timeout=60, max_retries=3)
 out = {}

 def one(item):
 cid, faces = item
 key = ",".join(str(f[0]) for f in sorted(faces))
 if key in cache:
 out[cid] = cache[key]; return
 try:
 img = strip([(f[1], f[2], f[3]) for f in faces])
 r = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low", messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT.format(n=len(faces))},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64.b64encode(img).decode()}", "detail": "low"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "who", "strict": True, "schema": SCHEMA}})
 o = json.loads(r.choices[0].message.content); u = r.usage
 name, conf = " ".join(o["name"].split()), float(o["confidence"])
 with _lock:
 spent["usd"] += (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6; spent["calls"] += 1
 cache[key] = (name, conf); out[cid] = (name, conf)
 with CACHE.open("a") as f:
 f.write(json.dumps({"key": key, "name": name, "confidence": conf}) + "\n")
 except Exception as e:
 print(f" cluster name fail {cid}: {e}")

 with ThreadPoolExecutor(workers) as ex:
 list(ex.map(one, clusters.items()))
 return out
