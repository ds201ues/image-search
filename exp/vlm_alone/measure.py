"""Sample queries (the user's reports + people queries from the live log) under baseline vs capped views, plus a contact sheet
of Luna-only tags for eyeballing. Usage: uv run python -m exp.vlm_alone.measure"""
import random, cv2, numpy as np
from pathlib import Path
from app.db import connect
from app import search as S
from app.config import THUMB_DIR

OUT = Path(__file__).parent / "out"; OUT.mkdir(exist_ok=True)
FLAG = {"B6DCE6E6-1E30-443F-8EC5-2B0A43B41108": "Madhavan tagged Sanjay Dutt", "9014D32D-DC67-4972-A537-50B3694FF254": "Govinda tagged Sanjay Dutt"}
QUERIES = ["sanjay dutt in sunglasses", "sanjay dutt", "sanjay dutt angry", "virat kohli angry", "virat kohli fighting",
 "rahul gandhi smiling", "modi in pink jacket", "shah rukh khan", "dhoni", "ranbir kapoor", "sam pitroda", "priyanka gandhi"]
VARIANTS = {"base": "image_person_scores", "cap75": "exp_vlm_alone_scores_cap75", "cap45": "exp_vlm_alone_scores_cap45"}

def run(conn):
 S.PARSE_TIMEOUT_MS = 60000; S.search("warm up", conn=conn)
 res = {}
 for v, t in VARIANTS.items():
 S.TABLES["scores"] = t
 for q in QUERIES:
 res[(v, q)] = S.search(q, conn=conn)["results"]
 S.TABLES["scores"] = "image_person_scores"
 print("| query | base (strong/total) | cap75 | cap45 | moved out of Best under cap75 |")
 print("|---|---|---|---|---|")
 for q in QUERIES:
 def cnt(v):
 r = res[(v, q)]; return f"{sum(x['tier']=='match' for x in r)}/{len(r)}"
 b = {x["id"]: x for x in res[("base", q)]}; c = {x["id"]: x for x in res[("cap75", q)]}
 moved = [i for i, x in b.items() if x["tier"] == "match" and (i not in c or c[i]["tier"] != "match")]
 lab = ", ".join(FLAG.get(i, i[:8]) for i in moved[:6]) + (" …" if len(moved) > 6 else "")
 print(f"| {q} | {cnt('base')} | {cnt('cap75')} | {cnt('cap45')} | {len(moved)}: {lab} |")
 for iid, why in FLAG.items():
 for v in VARIANTS:
 for q in QUERIES[:3]:
 hit = next((x for x in res[(v, q)] if x["id"] == iid), None)
 if hit: print(f" {why}: '{q}' {v}: tier={hit['tier']} people={hit['people']} possible={hit['people_possible']}")

def sheet(conn, rows, name, cols=6, cell=220):
 tiles = []
 for iid, pname, ev in rows:
 im = cv2.imread(str(THUMB_DIR / f"{iid}.jpg"))
 if im is None: continue
 h, w = im.shape[:2]; s = cell / max(h, w); im = cv2.resize(im, (int(w * s), int(h * s)))
 pad = np.full((cell + 28, cell, 3), 255, np.uint8); pad[:im.shape[0], :im.shape[1]] = im
 cv2.putText(pad, f"{pname[:26]}", (3, cell + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
 cv2.putText(pad, f"{iid[:8]} {ev}", (3, cell + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90, 90, 90), 1)
 tiles.append(pad)
 while len(tiles) % cols: tiles.append(np.full_like(tiles[0], 255))
 grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
 cv2.imwrite(str(OUT / name), grid, [cv2.IMWRITE_JPEG_QUALITY, 85]); print("wrote", OUT / name)

if __name__ == "__main__":
 conn = connect()
 run(conn)
 V = "array_to_string(evidence,',') like '%%vlm:%%' and array_to_string(evidence,',') not like '%%caption%%' and array_to_string(evidence,',') not like '%%cluster%%'"
 susp = conn.execute(f"""SELECT s.image_id, p.name, s.evidence[1] FROM image_person_scores s JOIN persons p ON p.id = s.person_id
 WHERE s.score >= 0.8 AND {V} AND EXISTS (SELECT 1 FROM faces f WHERE f.image_id = s.image_id)
 AND (SELECT count(*) FROM person_refs r WHERE r.person_id = s.person_id) >= 3""").fetchall()
 allv = conn.execute(f"""SELECT s.image_id, p.name, s.evidence[1] FROM image_person_scores s JOIN persons p ON p.id = s.person_id
 WHERE s.score >= 0.8 AND {V}""").fetchall()
 rng = random.Random(1)
 sheet(conn, rng.sample(susp, 24), "suspicious_24.jpg") # Luna-only, faces present, person has refs, no face matched
 sheet(conn, rng.sample(allv, 24), "luna_only_24.jpg") # any Luna-only confirmed tag
