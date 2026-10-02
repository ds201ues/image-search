"""Step 3: persons + evidence tables. Re-runnable; copies live image_persons (vlm/cluster) when they exist."""
import json, re
from pathlib import Path
from app.db import connect

HERE = Path(__file__).parent
P, IP, V = "exp_caption_people_persons", "exp_caption_people_image_persons", "exp_caption_people_scores"


def norm(s):
 return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace(".", " ")).split())


def main():
 conn = connect()
 persons = json.loads((HERE / "persons.json").read_text())
 conn.execute(f"DROP VIEW IF EXISTS {V}; DROP TABLE IF EXISTS {IP}; DROP TABLE IF EXISTS {P}")
 conn.execute(f"""CREATE TABLE {P} (id serial PRIMARY KEY, name text UNIQUE NOT NULL, aliases text[] DEFAULT '{{}}',
 wikidata_id text, popularity int DEFAULT 0)""")
 conn.execute(f"""CREATE TABLE {IP} (image_id text, person_id int REFERENCES {P}(id), source text NOT NULL, score real,
 PRIMARY KEY (image_id, person_id, source))""")
 key = {} # normalised name/alias -> person id
 for p in persons:
 pid = conn.execute(f"INSERT INTO {P} (name, aliases) VALUES (%s, %s) RETURNING id", (p["name"], p["aliases"])).fetchone()[0]
 for n in [p["name"], *p["aliases"]]:
 key.setdefault(norm(n), pid)
 # live Luna persons: map by name, else add
 live_map = {}
 for lid, name, aliases in conn.execute("SELECT id, name, aliases FROM persons"):
 pid = key.get(norm(name)) or next((key[norm(a)] for a in (aliases or []) if norm(a) in key), None)
 if pid is None:
 pid = conn.execute(f"INSERT INTO {P} (name, aliases) VALUES (%s, %s) ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id",
 (name, aliases or [])).fetchone()[0]
 key[norm(name)] = pid
 live_map[lid] = pid
 n_live = 0
 with conn.cursor() as cur:
 for iid, lid, src, sc in conn.execute("SELECT image_id, person_id, source, score FROM image_persons").fetchall():
 cur.execute(f"INSERT INTO {IP} VALUES (%s,%s,%s,%s) ON CONFLICT (image_id, person_id, source) DO UPDATE SET score = GREATEST({IP}.score, EXCLUDED.score)",
 (iid, live_map[lid], src, sc)); n_live += 1
 # caption evidence: strong prior, not proof (off-frame names happen)
 n_cap = 0
 for l in (HERE / "names.jsonl").open():
 r = json.loads(l)
 best = {}
 for x in r["people"]:
 pid = key.get(norm(x["canonical_full_name"])) or key.get(norm(x["name_as_written"]))
 if pid and x["confidence"] >= 0.5:
 best[pid] = max(best.get(pid, 0), 0.9 * float(x["confidence"]))
 for pid, sc in best.items():
 cur.execute(f"INSERT INTO {IP} VALUES (%s,%s,'caption',%s) ON CONFLICT DO NOTHING", (r["image_id"], pid, sc)); n_cap += 1
 conn.execute(f"""CREATE VIEW {V} AS SELECT image_id, person_id, 1 - exp(sum(ln(greatest(1 - score, 1e-6)))) AS score,
 array_agg(source || ':' || round(score::numeric, 2) ORDER BY source) AS evidence FROM {IP} GROUP BY image_id, person_id""")
 print(f"persons {conn.execute(f'SELECT count(*) FROM {P}').fetchone()[0]}; live rows copied {n_live}; caption rows {n_cap}")
 print("pairs >=0.5 / >=0.8:", conn.execute(f"SELECT count(*) FILTER (WHERE score>=0.5), count(*) FILTER (WHERE score>=0.8) FROM {V}").fetchone())


if __name__ == "__main__":
 main()
