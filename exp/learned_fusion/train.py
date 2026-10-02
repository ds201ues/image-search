"""E7 Part B: learned ranker over per-candidate features, GroupKFold by query, bootstrap CIs.

uv run python -m exp.learned_fusion.train --run all
Uses exp/learned_fusion/runs/<run>.pkl (every reranked candidate with features) + eval/judgments.jsonl.
"""
import argparse, json, pickle, re, statistics as st
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from app.db import connect
from app.search import norm, STOP
from eval.run_eval import load_queries
from eval.judge import _load
from exp.learned_fusion.ci import boot_ci, paired_ci, fmt
from exp.learned_fusion.run import score, keep_fixed, keep_dyn, YEAR

HERE = Path(__file__).parent
FEATS = ["p_ctx", "p_person", "score_vector", "score_desc", "score_text", "score_person", "inv_rank_vector", "inv_rank_desc",
 "inv_rank_text", "inv_rank_person", "n_sources", "link_min", "n_strong", "has_caption", "caption_overlap",
 "year_match", "is_graphic", "n_faces", "n_cand"]
MONO = {"p_ctx": 1, "p_person": 1, "score_vector": 1, "score_desc": 1, "score_text": 1, "score_person": 1, "inv_rank_vector": 1,
 "inv_rank_desc": 1, "inv_rank_text": 1, "inv_rank_person": 1, "n_sources": 1, "caption_overlap": 1, "year_match": 1, "is_graphic": -1}


def image_side(conn):
 rows = conn.execute("""SELECT id, caption, headline, extract(year from date_published)::int, vlm->>'image_type',
 (SELECT count(*) FROM faces f WHERE f.image_id = i.id) FROM images i""").fetchall()
 return {r[0]: r[1:] for r in rows}


def build(run, qs, g, side):
 X, y, grp, meta = [], [], [], []
 for q in qs:
 qq = q["query"]; r = run[qq]
 qtok = {t for t in norm(re.sub(YEAR, "", qq)).split() if t not in STOP}
 ys = re.findall(YEAR, qq); yr = (int(min(ys)), int(max(ys))) if ys else None
 for x in r["results"]:
 f = x["features"]; cap, head, iy, itype, nf = side[x["id"]]
 ctok = {t for t in norm((cap or "") + " " + (head or "")).split() if t not in STOP}
 row = [f.get("p_ctx", 0), f.get("p_person", 1.0 if not f.get("n_strong") else 0), f.get("score_vector", 0), f.get("score_desc", 0),
 f.get("score_text", 0), f.get("score_person", 0)]
 row += [1 / f["rank_" + s] if "rank_" + s in f else 0 for s in ("vector", "desc", "text", "person")]
 row += [f.get("n_sources", 0), f.get("link_min", 1.0), f.get("n_strong", 0), int(bool(cap)),
 len(qtok & ctok) / max(len(qtok), 1), (1 if yr and iy and yr[0] <= iy <= yr[1] else 0) if yr else 0,
 int(itype in ("graphic", "collage", "illustration", "screenshot")), nf or 0, f.get("n_cand", 0)]
 X.append(row); grp.append(qq); meta.append((qq, x["id"], x["relevance"]))
 gr = g.get((qq, x["id"]))
 y.append(None if gr is None else gr)
 return np.asarray(X, float), y, np.asarray(grp), meta


def rank_eval(scores, meta, qs, g, keep):
 """Re-sort candidates by score and apply keep(); return per-query metrics."""
 by = {}
 for s, (qq, iid, rel) in zip(scores, meta):
 by.setdefault(qq, []).append({"id": iid, "relevance": float(s)})
 out = {}
 for q in qs:
 res = sorted(by.get(q["query"], []), key=lambda d: -d["relevance"])
 out[q["query"]] = score(keep(res), q["query"], g)
 return out


def main():
 ap = argparse.ArgumentParser(); ap.add_argument("--run", default="all"); a = ap.parse_args()
 conn = connect(); qs = load_queries(); g = _load()
 run = pickle.load((HERE / "runs" / f"{a.run}.pkl").open("rb"))
 X, y, grp, meta = build(run, qs, g, image_side(conn))
 lab = np.asarray([-1 if v is None else v for v in y])
 print(f"{len(X)} candidates, {np.sum(lab >= 0)} judged ({np.sum(lab == 2)} relevant), {len(set(grp))} queries")
 types = {q["query"]: q["type"] for q in qs}

 hand = np.asarray([m[2] for m in meta]) # hand formula relevance
 res = {"hand": hand, "lr": np.zeros(len(X)), "lgb": np.zeros(len(X))}
 gkf = GroupKFold(5)
 imp = np.zeros(len(FEATS))
 for tr, te in gkf.split(X, lab, grp):
 tr = tr[lab[tr] >= 0]
 ytr = (lab[tr] == 2).astype(int) # relevant vs not (partial = not)
 sc = StandardScaler().fit(X[tr])
 lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[tr]), ytr)
 res["lr"][te] = lr.predict_proba(sc.transform(X[te]))[:, 1]
 # LambdaRank with graded labels
 order = np.argsort(grp[tr], kind="stable"); tr_s = tr[order]
 groups = [len(list(v)) for _, v in __import__("itertools").groupby(grp[tr_s])]
 m = lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
 monotone_constraints=[MONO.get(f, 0) for f in FEATS], verbose=-1, label_gain=[0, 1, 3])
 m.fit(X[tr_s], lab[tr_s], group=groups)
 res["lgb"][te] = m.predict(X[te]); imp += m.feature_importances_
 # pure ranking quality: same shown count for everyone (top-10 for P@10, top-60 for recall)
 print("\nRanking quality at fixed shown count (top-60 shown, P@10 on first 10) — cross-validated")
 base_pq = None
 for name, s in res.items():
 pq = rank_eval(s, meta, qs, g, lambda r: r[:60])
 p10 = [m[0] for m in pq.values()]; rec = [m[1] for m in pq.values()]; junk = [m[3] for m in pq.values()]
 line = f"{name:<5} P@10 {fmt(boot_ci(p10))} R@60 {fmt(boot_ci(rec))} junk/q {st.mean(junk):.2f}"
 if base_pq is not None:
 line += f" dP@10 vs hand {fmt(paired_ci([base_pq[q][0] for q in pq], [pq[q][0] for q in pq]))}"
 else:
 base_pq = pq
 print(line)
 bt = {}
 for q in qs:
 bt.setdefault(q["type"], []).append(pq[q["query"]][0])
 print(" P@10 by type: " + " ".join(f"{t} {st.mean([v for v in vs if v is not None] or [0]):.2f}" for t, vs in sorted(bt.items())))
 # with cutoffs: hand formula at REL_MIN=0.2 vs LR probability threshold chosen on train folds (approx: global sweep reported)
 print("\nWith a threshold (shown = those above it); LR threshold swept (optimistic by one parameter), hand at 0.2 and best dyn from cutoff.json")
 pq = rank_eval(res["hand"], meta, qs, g, lambda r: keep_fixed(r, 0.2))
 print(f"hand@0.2 P@10 {fmt(boot_ci([m[0] for m in pq.values()]))} R {fmt(boot_ci([m[1] for m in pq.values()]))} empty {sum(m[2] for m in pq.values())} junk {st.mean(m[3] for m in pq.values()):.2f}")
 cj = HERE / "cutoff.json"
 if cj.exists():
 c = json.load(cj.open())
 pq = rank_eval(res["hand"], meta, qs, g, lambda r: keep_dyn(r, c["floor"], c["ratio"], c["k"]))
 print(f"hand dyn P@10 {fmt(boot_ci([m[0] for m in pq.values()]))} R {fmt(boot_ci([m[1] for m in pq.values()]))} empty {sum(m[2] for m in pq.values())} junk {st.mean(m[3] for m in pq.values()):.2f}")
 for t in (0.2, 0.3, 0.4, 0.5):
 pq = rank_eval(res["lr"], meta, qs, g, lambda r: keep_fixed(r, t))
 print(f"lr@{t:.1f} P@10 {fmt(boot_ci([m[0] for m in pq.values()]))} R {fmt(boot_ci([m[1] for m in pq.values()]))} empty {sum(m[2] for m in pq.values())} junk {st.mean(m[3] for m in pq.values()):.2f}")
 print("\nLightGBM feature importance (split counts, summed over folds):")
 for f, v in sorted(zip(FEATS, imp), key=lambda kv: -kv[1]):
 print(f" {f:<18}{int(v):>6}")
 # full-data LR coefficients for integration
 sc = StandardScaler().fit(X[lab >= 0]); lr = LogisticRegression(C=0.5, max_iter=2000).fit(sc.transform(X[lab >= 0]), (lab[lab >= 0] == 2).astype(int))
 print("\nLR coefficients (standardised):")
 for f, c in sorted(zip(FEATS, lr.coef_[0]), key=lambda kv: -abs(kv[1])):
 print(f" {f:<18}{c:>+7.2f}")
 json.dump({"features": FEATS, "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
 "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}, (HERE / "lr_model.json").open("w"))
 pickle.dump({"res": res, "meta": meta}, (HERE / "scores.pkl").open("wb"))


if __name__ == "__main__":
 main()
