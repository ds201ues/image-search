"""E9: Batch API run of the baseline configuration on the 300-photo sample.
uv run python -m exp.vlm_cost.batch submit | status | fetch
"""
import json, sys, time
from pathlib import Path
from openai import OpenAI
import app.config as C
from .variants import sample_ids, spec, body, PRICES, OUT

HERE = Path(__file__).parent; STATE = OUT / "batch_state.json"


def main():
 client = OpenAI(timeout=120)
 cmd = sys.argv[1]
 if cmd == "submit":
 s = spec("baseline"); req = OUT / "batch_requests.jsonl"
 with req.open("w") as f:
 for iid in sample_ids():
 f.write(json.dumps({"custom_id": iid, "method": "POST", "url": "/v1/chat/completions", "body": body(iid, s)}) + "\n")
 up = client.files.create(file=req.open("rb"), purpose="batch")
 b = client.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions", completion_window="24h")
 STATE.write_text(json.dumps({"batch_id": b.id, "submitted": time.time()})); print("submitted", b.id)
 elif cmd == "status":
 st = json.loads(STATE.read_text()); b = client.batches.retrieve(st["batch_id"])
 print(b.status, b.request_counts, f"{(time.time() - st['submitted']) / 60:.0f} min since submit")
 elif cmd == "fetch":
 st = json.loads(STATE.read_text()); b = client.batches.retrieve(st["batch_id"])
 if b.status != "completed":
 print("not done:", b.status); return
 txt = client.files.content(b.output_file_id).text
 pin, pout = PRICES[C.VLM_MODEL]; rows = []
 for l in txt.splitlines():
 r = json.loads(l); resp = r["response"]["body"]; u = resp["usage"]
 rows.append({"id": r["custom_id"], "vlm": json.loads(resp["choices"][0]["message"]["content"]), "ms": None,
 "tin": u["prompt_tokens"], "tout": u["completion_tokens"],
 "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
 "cost": (u["prompt_tokens"] * pin + u["completion_tokens"] * pout) / 2e6}) # batch = 50% off
 (OUT / "batch.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
 st["completed"] = time.time(); st["turnaround_min"] = (st["completed"] - st["submitted"]) / 60; STATE.write_text(json.dumps(st))
 print(f"batch: {len(rows)} rows, ${1000 * sum(r['cost'] for r in rows) / len(rows):.3f} per 1k, turnaround {st['turnaround_min']:.0f} min")


if __name__ == "__main__":
 main()
