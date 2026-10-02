"""E8: near-duplicate groups (phash/dhash + SigLIP), shoots, VLM cost savings, contact sheet."""
import json, collections, itertools, time
from pathlib import Path
import numpy as np, imagehash
from PIL import Image
from app.db import connect
from app.config import THUMB_DIR

OUT = Path(__file__).parent
conn = connect()
ids = [r[0] for r in conn.execute("SELECT id FROM images ORDER BY id")]
N = len(ids)

# ---- perceptual hashes
hp = OUT / "hashes.json"
if hp.exists():
 H = json.load(hp.open())
else:
 H = {}
 t = time.perf_counter()
 for i, iid in enumerate(ids):
 try:
 im = Image.open(THUMB_DIR / f"{iid}.jpg")
 H[iid] = [str(imagehash.phash(im)), str(imagehash.dhash(im))]
 except Exception as e:
 print("hash fail", iid, e)
 json.dump(H, hp.open("w")); print(f"hashed {len(H)} in {time.perf_counter()-t:.0f}s")
ph = {k: imagehash.hex_to_hash(v[0]) for k, v in H.items()}
dh = {k: imagehash.hex_to_hash(v[1]) for k, v in H.items()}
# all-pairs hamming via numpy bit arrays (5k^2 = 25M, fine)
def bits(d):
 return np.array([np.array(d[i].hash).flatten() for i in ids], dtype=np.uint8)
PB, DB = bits(ph), bits(dh)
def ham_pairs(B, maxd):
 pairs = {}
 for i in range(N):
 d = np.count_nonzero(B[i + 1:] != B[i], axis=1)
 for j in np.nonzero(d <= maxd)[0]:
 pairs[(i, i + 1 + j)] = int(d[j])
 return pairs
ph_pairs = ham_pairs(PB, 12); dh_pairs = ham_pairs(DB, 12)

# ---- SigLIP cosine top-10 neighbours
sig = {}
for a, b, sim in conn.execute("""SELECT f.id, n.id, n.sim FROM images f, LATERAL (
 SELECT g.id, 1 - (g.clip <=> f.clip) AS sim FROM images g WHERE g.id <> f.id ORDER BY g.clip <=> f.clip LIMIT 10) n
 WHERE n.sim >= 0.85"""):
 key = tuple(sorted((ids.index(a), ids.index(b))))
 sig[key] = max(sig.get(key, 0), float(sim))

def groups(pairs):
 parent = list(range(N))
 def f(x):
 while parent[x] != x:
 parent[x] = parent[parent[x]]; x = parent[x]
 return x
 for a, b in pairs:
 parent[f(a)] = f(b)
 g = collections.defaultdict(list)
 for i in range(N):
 g[f(i)].append(i)
 return g

def summarize(name, pairs):
 g = groups(pairs)
 sizes = sorted((len(v) for v in g.values()), reverse=True)
 multi = sum(s for s in sizes if s > 1)
 return dict(name=name, pairs=len(pairs), groups_ge2=sum(1 for s in sizes if s > 1), imgs_in_groups=multi,
 pct=round(100 * multi / N, 1), largest=sizes[:5], vlm_saved_pct=round(100 * (multi - sum(1 for s in sizes if s > 1)) / N, 1))

rows = []
for d in (4, 6, 8, 10, 12):
 rows.append(summarize(f"phash<={d}", {k for k, v in ph_pairs.items() if v <= d}))
for d in (4, 6, 8, 10):
 rows.append(summarize(f"dhash<={d}", {k for k, v in dh_pairs.items() if v <= d}))
for s in (0.90, 0.93, 0.96, 0.98):
 rows.append(summarize(f"siglip>={s}", {k for k, v in sig.items() if v >= s}))
for r in rows:
 print(r)

# agreement
p6 = {k for k, v in ph_pairs.items() if v <= 6}
for s in (0.90, 0.93, 0.96):
 S = {k for k, v in sig.items() if v >= s}
 print(f"siglip>={s}: {len(S)} pairs, {len(S & p6)} also phash<=6 ({100*len(S&p6)/max(len(S),1):.0f}%); phash<=6 pairs not in siglip: {len(p6 - S)}")

# ---- chosen definition: phash<=6 OR siglip>=0.96 (true bursts), store groups
chosen = p6 | {k for k, v in sig.items() if v >= 0.96}
G = groups(chosen)
conn.execute("DROP TABLE IF EXISTS exp_dedup_diversity_groups")
conn.execute("CREATE TABLE exp_dedup_diversity_groups (image_id text PRIMARY KEY, group_id int, is_rep boolean)")
vlm_done = {r[0] for r in conn.execute("SELECT id FROM images WHERE vlm IS NOT NULL")}
cap = dict(conn.execute("SELECT id, caption IS NOT NULL FROM images"))
with conn.cursor() as cur:
 for gi, (root, mem) in enumerate(G.items()):
 # representative: captioned first, then lowest id
 rep = sorted(mem, key=lambda i: (not cap[ids[i]], ids[i]))[0]
 cur.executemany("INSERT INTO exp_dedup_diversity_groups VALUES (%s,%s,%s)", [(ids[i], gi, i == rep) for i in mem])
print("chosen:", summarize("phash<=6|siglip>=0.96", chosen))

# ---- shoots: same date::date AND same lower(byline), OR near-dup group
meta = {r[0]: (r[1], (r[2] or "").strip().lower()) for r in conn.execute("SELECT id, date_published::date, byline FROM images")}
key_groups = collections.defaultdict(list)
for i, iid in enumerate(ids):
 d, b = meta[iid]
 if d and b:
 key_groups[(d, b)].append(i)
shoot_pairs = set(chosen)
for mem in key_groups.values():
 for a in mem[1:]:
 shoot_pairs.add((mem[0], a))
SG = groups(shoot_pairs)
conn.execute("DROP TABLE IF EXISTS exp_dedup_diversity_shoots")
conn.execute("CREATE TABLE exp_dedup_diversity_shoots (image_id text PRIMARY KEY, shoot_id int)")
with conn.cursor() as cur:
 for si, mem in enumerate(SG.values()):
 cur.executemany("INSERT INTO exp_dedup_diversity_shoots VALUES (%s,%s)", [(ids[i], si) for i in mem])
sizes = sorted((len(v) for v in SG.values()), reverse=True)
uncap_with_sib = sum(1 for mem in SG.values() for i in mem if not cap[ids[i]] and any(cap[ids[j]] for j in mem if j != i))
uncap = sum(1 for i in ids if not cap[i])
print(f"shoots: {len(SG)} total, {sum(1 for s in sizes if s>1)} with >=2 photos covering {sum(s for s in sizes if s>1)} photos; largest {sizes[:8]}; "
 f"uncaptioned {uncap}, of which {uncap_with_sib} have a captioned sibling in their shoot")
# distribution
hist = collections.Counter(min(s, 10) for s in sizes)
print("shoot size histogram (10 = 10+):", sorted(hist.items()))

# ---- contact sheet: 10 pairs per band
def band_examples(pairs_scores, lo, hi, n=10, key=lambda v: v):
 items = [(k, v) for k, v in pairs_scores.items() if lo <= key(v) < hi]
 rng = np.random.default_rng(0)
 if len(items) > n:
 items = [items[i] for i in rng.choice(len(items), n, replace=False)]
 return items
bands = [("phash 0-3", band_examples(ph_pairs, 0, 4)), ("phash 4-6", band_examples(ph_pairs, 4, 7)), ("phash 7-10", band_examples(ph_pairs, 7, 11)),
 ("siglip 0.96-1.0", band_examples(sig, 0.96, 1.01)), ("siglip 0.93-0.96", band_examples(sig, 0.93, 0.96)),
 ("siglip 0.90-0.93", band_examples(sig, 0.90, 0.93))]
html = ["<html><body style='font-family:sans-serif;background:#111;color:#eee'><h2>E8 near-duplicate pairs by band</h2>"]
for name, items in bands:
 html.append(f"<h3>{name} ({len(items)} shown)</h3><div style='display:flex;flex-wrap:wrap;gap:12px'>")
 for (a, b), v in items:
 ia, ib = ids[a], ids[b]
 html.append(f"<div style='border:1px solid #444;padding:4px'><div style='font-size:11px'>{v if isinstance(v,int) else round(v,3)} · cap:{int(cap[ia])}/{int(cap[ib])}</div>"
 f"<img src='../../data/thumbs/{ia}.jpg' height=160> <img src='../../data/thumbs/{ib}.jpg' height=160></div>")
 html.append("</div>")
html.append("</body></html>")
(OUT / "contact_sheet.html").write_text("".join(html))
json.dump({"rows": rows}, (OUT / "stats.json").open("w"), indent=1)
