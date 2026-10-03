"""Describe the 2,000-photo sample with prompt v2 via the Batch API into exp_taxonomy_vlm. Usage: submit | collect"""
import io, json, sys, time
from openai import OpenAI
from app.config import THUMB_DIR, VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT, VLM_REASONING, VLM_DETAIL
from app.vlm import _jpeg_b64
from app.vlm_v2 import PROMPT, SCHEMA
from app.db import connect
from pathlib import Path
HERE = Path(__file__).parent; STATE = HERE / "batches.jsonl"
client = OpenAI(timeout=120, max_retries=3); conn = connect()
conn.execute("CREATE TABLE IF NOT EXISTS exp_taxonomy_vlm (image_id text PRIMARY KEY, vlm jsonb, tokens_in int, tokens_out int, cost_usd double precision)")
ids = json.load((HERE / "sample_ids.json").open())
if sys.argv[1] == "submit":
 have = {r[0] for r in conn.execute("SELECT image_id FROM exp_taxonomy_vlm")}
 todo = [i for i in ids if i not in have]
 for k in range(0, len(todo), 1000):
 chunk = todo[k:k + 1000]; buf = io.StringIO()
 for iid in chunk:
 body = {"model": VLM_MODEL, "messages": [{"role": "user", "content": [{"type": "text", "text": PROMPT},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(THUMB_DIR / f'{iid}.jpg')}", "detail": VLM_DETAIL}}]}],
 "response_format": {"type": "json_schema", "json_schema": {"name": "image_index_v2", "strict": True, "schema": SCHEMA}}}
 if VLM_REASONING: body["reasoning_effort"] = VLM_REASONING
 buf.write(json.dumps({"custom_id": iid, "method": "POST", "url": "/v1/chat/completions", "body": body}) + "\n")
 f = client.files.create(file=(f"tax_{k//1000}.jsonl", buf.getvalue().encode()), purpose="batch")
 b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions", completion_window="24h")
 STATE.open("a").write(json.dumps({"batch_id": b.id, "n": len(chunk), "status": "submitted"}) + "\n"); print("submitted", b.id, len(chunk))
else:
 rows = [json.loads(l) for l in STATE.open()]; tot = 0
 for b in rows:
 if b["status"] == "done": continue
 bb = client.batches.retrieve(b["batch_id"]); print(b["batch_id"], bb.status, bb.request_counts.completed, "/", bb.request_counts.total)
 if bb.status != "completed": continue
 for l in client.files.content(bb.output_file_id).text.splitlines():
 r = json.loads(l); resp = (r.get("response") or {}).get("body") or {}
 if not resp.get("choices"): continue
 u = resp["usage"]; cost = (u["prompt_tokens"] * VLM_PRICE_IN + u["completion_tokens"] * VLM_PRICE_OUT) / 2e6
 try: v = json.loads(resp["choices"][0]["message"]["content"])
 except Exception: continue
 conn.execute("INSERT INTO exp_taxonomy_vlm VALUES (%s,%s,%s,%s,%s) ON CONFLICT (image_id) DO NOTHING", (r["custom_id"], json.dumps(v), u["prompt_tokens"], u["completion_tokens"], cost)); tot += 1
 b["status"] = "done"
 STATE.write_text("".join(json.dumps(b) + "\n" for b in rows))
 print("collected", tot, "| in table:", conn.execute("SELECT count(*), round(avg(tokens_in)), round(avg(tokens_out)), sum(cost_usd) FROM exp_taxonomy_vlm").fetchone())
