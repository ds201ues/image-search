"""Offline reranker comparison on judged (query, photo) pairs. No API cost.

uv run python exp/reranker/compare.py --judgments eval/judgments_websample.jsonl --tag websample [--caption] [models...]
Per model: mean per-query AUC, calibrated P/R at 0.2/0.5 (2-parameter sigmoid fitted on grade 2 vs rest),
and CPU latency for 150 pairs. Documents built exactly as app/search.py builds them (people line + doc_text),
optionally with the IPTC caption prepended.
"""
import argparse, gc, json, statistics as st, sys, time
import numpy as np, torch
from app.db import connect
from app.search import PERSON_SURE, PERSON_MIN, people, norm

torch.set_num_threads(4)
DEFAULT = ["cross-encoder/ms-marco-MiniLM-L-6-v2@160", "cross-encoder/ms-marco-MiniLM-L-6-v2@256",
 "cross-encoder/ettin-reranker-17m-v1@256", "cross-encoder/ettin-reranker-32m-v1@256",
 "cross-encoder/ettin-reranker-68m-v1@256", "ibm-granite/granite-embedding-reranker-english-r2@256",
 "Alibaba-NLP/gte-reranker-modernbert-base@256", "mixedbread-ai/mxbai-rerank-base-v2@256"]


def fit_sigmoid(s, y, iters=500):
 """Logistic regression p = sigmoid(a*s + b), Newton steps."""
 s = np.asarray(s, np.float64); y = np.asarray(y, np.float64)
 a, b = 1.0 / (s.std() + 1e-6), -s.mean() / (s.std() + 1e-6)
 for _ in range(iters):
 z = a * s + b; p = 1 / (1 + np.exp(-z)); w = p * (1 - p) + 1e-9
 g = np.array([((p - y) * s).sum(), (p - y).sum()])
 H = np.array([[(w * s * s).sum(), (w * s).sum()], [(w * s).sum(), w.sum()]]) + 1e-6 * np.eye(2)
 step = np.linalg.solve(H, g); a -= step[0]; b -= step[1]
 if np.abs(step).max() < 1e-8:
 break
 return float(a), float(b)


def load_docs(conn, caption):
 docs = {}
 cap_col = "coalesce(i.caption, '')" if caption else "''"
 for iid, d, cap, ppl in conn.execute(f"""SELECT i.id, coalesce(i.doc_text, ''), {cap_col},
 (SELECT string_agg(CASE WHEN s.score >= %s THEN p.name ELSE 'possibly ' || p.name END, ', ' ORDER BY s.score DESC)
 FROM image_person_scores s JOIN persons p ON p.id = s.person_id WHERE s.image_id = i.id AND s.score >= %s)
 FROM images i""", (PERSON_SURE, PERSON_MIN)):
 t = f"People: {ppl}. {d}" if ppl else d
 docs[iid] = f"Caption: {cap} {t}" if cap else t
 return docs


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--judgments", default="eval/judgments.jsonl"); ap.add_argument("--tag", default="run")
 ap.add_argument("--caption", action="store_true"); ap.add_argument("models", nargs="*")
 a = ap.parse_args()
 models = a.models or DEFAULT
 conn = connect()
 J = {}
 for l in open(a.judgments):
 try:
 r = json.loads(l)
 except json.JSONDecodeError:
 continue
 if r.get("grade") is not None:
 J[(r["query"], r["image_id"])] = r["grade"]
 docs = load_docs(conn, a.caption)
 P = people(conn)

 def as_reranked(q):
 links, _ = P.link(q)
 out = " " + norm(q) + " "
 for p, c, w in links:
 if c >= 0.5 and w:
 out = out.replace(f" {w} ", f" {P.names[p].lower()} ", 1)
 return out.strip()

 qs = sorted({q for q, _ in J})
 sets = {q: [(iid, g) for (qq, iid), g in J.items() if qq == q and iid in docs] for q in qs}
 sets = {q: v for q, v in sets.items() if any(g == 2 for _, g in v) and any(g == 0 for _, g in v)}
 pairs = [(as_reranked(q), docs[i], g, q) for q, items in sets.items() for i, g in items]
 print(f"{len(sets)} queries with both relevant and irrelevant judged photos, {len(pairs)} pairs, caption={a.caption}", flush=True)
 sample = list(docs.values())[:150]
 lens = [len(d.split()) for d in docs.values()]
 print(f"doc words: median {st.median(lens):.0f}, p90 {sorted(lens)[int(0.9*len(lens))]}")
 out_rows = []
 print(f"{'model':<52}{'AUC':>6}{'P@.2':>7}{'R@.2':>7}{'P@.5':>7}{'R@.5':>7}{'ms/150':>8}")
 for spec in models:
 name, _, ml = spec.partition("@"); ml = int(ml or 256)
 try:
 from sentence_transformers import CrossEncoder
 m = CrossEncoder(name, max_length=ml, device="cpu", trust_remote_code=True)
 except Exception as e:
 print(f"{spec:<52} failed to load: {str(e)[:80]}", flush=True); continue
 try:
 m.predict([("warm up", sample[0])])
 ts = []
 for _ in range(3):
 t = time.perf_counter(); m.predict([("people speaking at a press conference", d) for d in sample], batch_size=64)
 ts.append((time.perf_counter() - t) * 1000)
 ms = min(ts)
 raw = np.asarray(m.predict([(q, d) for q, d, _, _ in pairs], batch_size=64), dtype=np.float64)
 if raw.ndim > 1:
 raw = raw[:, -1] if raw.shape[1] > 1 else raw[:, 0]
 y = np.array([1.0 if g == 2 else 0.0 for _, _, g, _ in pairs])
 aa, bb = fit_sigmoid(raw, y)
 prob = 1 / (1 + np.exp(-(aa * raw + bb)))
 aucs = []
 idx = {}
 for k, (_, _, g, q) in enumerate(pairs):
 idx.setdefault(q, []).append(k)
 for q, ks in idx.items():
 pos = [raw[k] for k in ks if pairs[k][2] == 2]; neg = [raw[k] for k in ks if pairs[k][2] == 0]
 aucs.append(np.mean([[1.0 if p > n else 0.5 if p == n else 0.0 for n in neg] for p in pos]))
 def pr(t):
 shown = prob >= t
 return (y[shown].mean() if shown.any() else 0.0, (y[shown].sum() / y.sum()))
 p2, r2 = pr(0.2); p5, r5 = pr(0.5)
 row = dict(model=spec, auc=float(st.mean(aucs)), p02=float(p2), r02=float(r2), p05=float(p5), r05=float(r5),
 ms150=ms, calib_a=aa, calib_b=bb, raw_min=float(raw.min()), raw_max=float(raw.max()))
 out_rows.append(row)
 print(f"{spec:<52}{row['auc']:>6.3f}{p2:>7.2f}{r2:>7.2f}{p5:>7.2f}{r5:>7.2f}{ms:>8.0f} calib a={aa:.3f} b={bb:.3f}", flush=True)
 np.save(f"exp/reranker/raw_{a.tag}_{name.replace('/', '_')}_{ml}{'_cap' if a.caption else ''}.npy", raw)
 except Exception as e:
 print(f"{spec:<52} failed: {str(e)[:100]}", flush=True)
 del m; gc.collect()
 json.dump({"pairs": [(q, d[:0], g, qq) for q, d, g, qq in pairs], "rows": out_rows},
 open(f"exp/reranker/results_{a.tag}{'_cap' if a.caption else ''}.json", "w"), indent=1)


if __name__ == "__main__":
 main()
