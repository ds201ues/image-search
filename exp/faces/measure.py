"""Face pipeline, measured before tuning (review item 9). No model inference: SQL + numpy over stored embeddings.
1. ROC of AuraFace cosine from caption-seeded references: same-person pairs vs different-person pairs -> threshold at 1% / 0.5% FPR.
2. Nearest-reference naming (instead of union-find chaining): for every face, nearest reference from ANOTHER photo;
 precision on photos whose caption names the person (caption hidden), coverage on uncaptioned photos.
3. Face-size histogram (what MIN_FACE drops cannot be seen; what survives can).
Usage: uv run python -m exp.faces.measure
"""
import numpy as np, collections
from app.db import connect

conn = connect()
refs = conn.execute("""SELECT r.person_id, r.emb, split_part(r.source, ':', 2) FROM person_refs r WHERE r.source LIKE 'caption:%'""").fetchall()
pid = np.array([r[0] for r in refs]); R = np.stack([np.asarray(r[1].to_numpy() if hasattr(r[1], "to_numpy") else r[1], dtype=np.float32) for r in refs]); rimg = np.array([r[2] for r in refs])
R /= np.linalg.norm(R, axis=1, keepdims=True)
print(f"caption-seeded references: {len(refs)} faces, {len(set(pid))} people, {sum(c >= 2 for c in collections.Counter(pid).values())} people with >= 2 refs")

# 1. ROC
S = R @ R.T
iu = np.triu_indices(len(refs), 1)
same = (pid[:, None] == pid[None, :])[iu]; sims = S[iu]
pos, neg = sims[same], sims[~same]
print(f"pairs: {len(pos)} same-person, {len(neg)} different-person; same median {np.median(pos):.3f}, different median {np.median(neg):.3f}")
print(f"{'thr':>5} {'TPR':>6} {'FPR':>7}")
for thr in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
 print(f"{thr:>5} {np.mean(pos >= thr):>6.2f} {np.mean(neg >= thr):>7.4f}")
for fpr in (0.01, 0.005, 0.001):
 thr = float(np.quantile(neg, 1 - fpr)); print(f"threshold at FPR {fpr:.1%}: {thr:.3f} (TPR {np.mean(pos >= thr):.2f})")

# 2. nearest-reference naming
faces = conn.execute("SELECT f.id, f.image_id, f.emb, f.bbox FROM faces f").fetchall()
F = np.stack([np.asarray(r[2].to_numpy() if hasattr(r[2], "to_numpy") else r[2], dtype=np.float32) for r in faces]); F /= np.linalg.norm(F, axis=1, keepdims=True)
fimg = np.array([r[1] for r in faces])
cap_names = collections.defaultdict(set)
for iid, p in conn.execute("SELECT image_id, person_id FROM image_persons WHERE source = 'caption' AND score >= 0.7"):
 cap_names[iid].add(p)
captioned = {r[0] for r in conn.execute("SELECT id FROM images WHERE caption IS NOT NULL")}
names = dict(conn.execute("SELECT id, name FROM persons"))
sizes = np.array([min(r[3][2], r[3][3]) for r in faces])
print(f"\nfaces: {len(faces)}; min side px: p10 {np.percentile(sizes,10):.0f} p50 {np.percentile(sizes,50):.0f} p90 {np.percentile(sizes,90):.0f}; < 48 px: {np.mean(sizes < 48):.0%}; < 64 px: {np.mean(sizes < 64):.0%}")
sim = F @ R.T
for k in range(len(faces)): # a reference from the same photo must not name itself
 sim[k, rimg == fimg[k]] = -1
best = sim.argmax(1); bsim = sim[np.arange(len(faces)), best]; bpid = pid[best]
# margin: best match vs best match to a DIFFERENT person (ignoring other refs of the same person)
sim2 = sim.copy()
for k in range(len(faces)):
 sim2[k, pid == bpid[k]] = -1
margin = bsim - sim2.max(1)
print(f"\n{'thr':>5} {'margin':>7} {'named faces':>12} {'precision on caption-named photos':>34} {'n':>6} {'uncaptioned photos gaining a name':>34}")
for thr, mg in ((0.3, 0), (0.35, 0), (0.4, 0), (0.45, 0), (0.5, 0), (0.55, 0), (0.35, 0.1), (0.4, 0.1), (0.4, 0.15), (0.45, 0.1)):
 hit = (bsim >= thr) & (margin >= mg)
 tp = fp = 0
 for k in np.where(hit)[0]:
 if fimg[k] in cap_names: # held-out check: does the caption agree?
 if bpid[k] in cap_names[fimg[k]]: tp += 1
 else: fp += 1
 unc = len({fimg[k] for k in np.where(hit)[0] if fimg[k] not in captioned})
 print(f"{thr:>5} {mg:>7} {int(hit.sum()):>12} {tp / max(tp + fp, 1):>34.3f} {tp + fp:>6} {unc:>34}")
# top gaining people at 0.4
cnt = collections.Counter(names[bpid[k]] for k in np.where(bsim >= 0.4)[0] if fimg[k] not in captioned)
print("\nmost-named people in uncaptioned photos at thr 0.4:", cnt.most_common(12))
