"""Step 4: caption-named single-face photos become reference faces; cluster votes with caption names only
(independent of the live Luna-based 'cluster' evidence). Writes source='caption_ref' and 'caption_cluster'."""
import numpy as np
from app.db import connect

P, IP, V = "exp_caption_people_persons", "exp_caption_people_image_persons", "exp_caption_people_scores"
CLUSTER_SIM, REF_SIM, MIN_SIDE = 0.55, 0.45, 60


def clusters(conn):
 ids = [r[0] for r in conn.execute("SELECT id FROM faces ORDER BY id")]
 parent = {i: i for i in ids}
 def find(x):
 while parent[x] != x:
 parent[x] = parent[parent[x]]; x = parent[x]
 return x
 for a, b in conn.execute(f"""SELECT f.id, n.id FROM faces f, LATERAL (SELECT g.id FROM faces g WHERE g.id <> f.id AND g.image_id <> f.image_id
 ORDER BY g.emb <=> f.emb LIMIT 10) n JOIN faces g2 ON g2.id = n.id WHERE 1 - (g2.emb <=> f.emb) >= {CLUSTER_SIM}"""):
 parent[find(a)] = find(b)
 return {i: find(i) for i in ids}


def main():
 conn = connect()
 conn.execute(f"DELETE FROM {IP} WHERE source IN ('caption_ref', 'caption_cluster')")
 cap = {} # image -> {pid: score} from captions
 for iid, pid, sc in conn.execute(f"SELECT image_id, person_id, score FROM {IP} WHERE source = 'caption'"):
 cap.setdefault(iid, {})[pid] = sc
 faces = conn.execute("SELECT id, image_id, bbox, emb FROM faces").fetchall()
 nfaces = {}
 for _, iid, _, _ in faces:
 nfaces[iid] = nfaces.get(iid, 0) + 1
 # references: one caption name (>=0.8) + one usable face
 refs = [] # (face_id, image_id, pid, emb)
 for fid, iid, bbox, emb in faces:
 if nfaces[iid] == 1 and min(bbox[2], bbox[3]) >= MIN_SIDE and len(cap.get(iid, {})) == 1:
 pid, sc = next(iter(cap[iid].items()))
 if sc >= 0.8:
 refs.append((fid, iid, pid, np.asarray(emb.to_numpy() if hasattr(emb, "to_numpy") else emb, dtype=np.float32)))
 print(f"{len(refs)} caption-seeded reference faces for {len({r[2] for r in refs})} persons")
 R = np.stack([r[3] for r in refs]) if refs else np.zeros((0, 512), np.float32)
 rpid = np.array([r[2] for r in refs]); rimg = np.array([r[1] for r in refs])
 cl = clusters(conn)
 members = {}
 for fid, iid, _, _ in faces:
 members.setdefault(cl[fid], []).append((fid, iid))
 E = {fid: np.asarray(emb.to_numpy() if hasattr(emb, "to_numpy") else emb, dtype=np.float32) for fid, _, _, emb in faces}
 # reference match per face (excluding references from the same image)
 gal = {}
 for fid, iid, _, _ in faces:
 if len(refs) == 0:
 break
 sims = R @ E[fid]
 sims[rimg == iid] = -1
 j = int(sims.argmax())
 if sims[j] >= REF_SIM:
 gal[fid] = (int(rpid[j]), float(sims[j]))
 rows = {}
 for cid, mem in members.items():
 imgs = {iid for _, iid in mem}
 for fid, iid in mem:
 votes = {}
 for f2, i2 in mem:
 if i2 != iid:
 for pid, sc in cap.get(i2, {}).items():
 votes[pid] = votes.get(pid, 0) + sc / nfaces[i2]
 if f2 in gal:
 pid, sim = gal[f2]
 votes[pid] = votes.get(pid, 0) + sim
 denom = len(imgs - {iid}) + sum(1 for f2, _ in mem if f2 in gal)
 if not votes or not denom:
 continue
 ranked = sorted(votes.items(), key=lambda kv: -kv[1])
 pid, v = ranked[0]
 runner = ranked[1][1] if len(ranked) > 1 else 0
 score = min(0.99, v / denom) * (1 if v >= 2 * runner else 0.5)
 if score >= 0.2:
 rows[(iid, pid)] = max(score, rows.get((iid, pid), 0))
 with conn.cursor() as cur:
 cur.executemany(f"INSERT INTO {IP} VALUES (%s,%s,'caption_cluster',%s) ON CONFLICT DO NOTHING",
 [(i, p, s) for (i, p), s in rows.items()])
 named_before = {r[0] for r in conn.execute(f"SELECT DISTINCT image_id FROM {IP} WHERE source <> 'caption_cluster' AND score >= 0.8")}
 named_after = {r[0] for r in conn.execute(f"SELECT DISTINCT image_id FROM {V} WHERE score >= 0.8")}
 newly = named_after - named_before
 uncaptioned = {r[0] for r in conn.execute("SELECT id FROM images WHERE caption IS NULL")}
 print(f"caption_cluster rows: {len(rows)}; photos with a >=0.8 name: {len(named_before)} -> {len(named_after)} "
 f"(+{len(newly)}, of which {len(newly & uncaptioned)} have no caption)")
 for iid, name, sc, ev in conn.execute(f"""SELECT s.image_id, p.name, round(s.score::numeric,2), s.evidence FROM {V} s JOIN {P} p ON p.id = s.person_id
 WHERE s.image_id = ANY(%s) ORDER BY s.score DESC LIMIT 8""", (list(newly & uncaptioned),)):
 print(" ", iid[:8], name, sc, ev)


if __name__ == "__main__":
 main()
