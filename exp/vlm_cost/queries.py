"""E9: 60 extra Luna-written queries for sample photos (+ eval queries whose source image is in the sample) -> out/queries.jsonl"""
import json, random
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import app.config as C
from app.db import connect
from app.vlm import _jpeg_b64
from eval.make_queries import PROMPT, SCHEMA, KINDS
from .variants import sample_ids, OUT

def main():
 ids = sample_ids(); sset = set(ids)
 out = [{**json.loads(l), "src": "eval"} for l in open("eval/gen_queries.jsonl") if json.loads(l)["source_image"] in sset]
 conn = connect(); caps = dict(conn.execute("SELECT id, caption FROM images WHERE id = ANY(%s)", (ids,)))
 years = dict(conn.execute("SELECT id, extract(year from date_published)::int FROM images WHERE id = ANY(%s)", (ids,)))
 pick = [i for i in ids if i not in {q["source_image"] for q in out}]; random.Random(3).shuffle(pick); pick = pick[:60]
 client = OpenAI(timeout=60, max_retries=3)
 def one(k):
 iid = pick[k]; cap = caps.get(iid); capt = f'. Photographer\'s caption: "{cap}"' if cap else ""
 if years.get(iid): capt += f" (taken in {years[iid]})"
 r = client.chat.completions.create(model=C.VLM_MODEL, reasoning_effort="low", messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT.format(kind=KINDS[k % len(KINDS)], cap=capt)},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(C.THUMB_DIR / f'{iid}.jpg')}"}}]}],
 response_format={"type": "json_schema", "json_schema": {"name": "q", "strict": True, "schema": SCHEMA}})
 return {**json.loads(r.choices[0].message.content), "source_image": iid, "src": "e9"}
 with ThreadPoolExecutor(12) as ex:
 out += list(ex.map(one, range(len(pick))))
 seen = set(); final = []
 for q in out:
 q["query"] = " ".join(q["query"].lower().split())
 if q["query"] not in seen: seen.add(q["query"]); final.append(q)
 (OUT / "queries.jsonl").write_text("".join(json.dumps(q) + "\n" for q in final))
 print(len(final), "queries;", sum(q["src"] == "eval" for q in final), "from eval")

if __name__ == "__main__":
 main()
