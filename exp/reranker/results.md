# E5 reranker — interim results (2026-10-02, 11:20)

Status: local model work stopped by the coordinator (laptop load 57); GPU scoring prepared, waiting for a GPU machine.
Everything below comes from the web-sample judgments (98 queries, 4,264 judged pairs, 1,018 docs, no captions).
Challenge-set numbers follow once `eval/baseline_report.txt` exists and the GPU scores come back.

## What exists
| file | purpose |
|---|---|
| `compare.py` | local CPU comparison (used for the partial run below; do not run again on the laptop) |
| `export_pairs.py` | DB -> `data/<set>_pairs.jsonl`, `_docs.jsonl`, `_queries.jsonl` (web sample exported; challenge pending baseline) |
| `gpu_score.py` | self-contained CUDA scorer: `rerank` (12 rerankers × plain/caption docs) and `embed` (6 embedders) modes |
| `analyze.py` | AUC, judged-P@10, 2-param sigmoid calibration, cascade simulation; no models |
| `full_eval.py` | full-system judged eval with the reranker monkeypatched (final candidates only, CPU, after baseline) |
| `RUNBOOK_gpu.md` | the exact commands |

## Web-sample ranking quality (recovered from the partial local run; pair order verified against the live log)
Docs = people line + Luna description, as app/search.py builds them. AUC = per-query, mean. P@10j = precision of the top-10 among judged pairs.
P/R at 0.2 and 0.5 are on calibrated p = sigmoid(a·raw + b), fitted on these same pairs (optimistic, in-sample).

| model | max_len | AUC | P@10j | P@.2 | R@.2 | P@.5 | R@.5 | calib a, b |
|---|---|---|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 (current) | 160 | 0.964 | 0.476 | 0.54 | 0.77 | 0.72 | 0.54 | 0.357, −0.714 |
| ms-marco-MiniLM-L-6-v2 | 256 | 0.967 | 0.478 | 0.55 | 0.78 | 0.71 | 0.54 | 0.369, −0.757 |
| ettin-reranker-17m-v1 | 256 | 0.948 | 0.463 | 0.53 | 0.68 | 0.74 | 0.46 | 1.062, −9.148 |
| ettin-reranker-32m-v1 | 256 | 0.963 | 0.469 | 0.50 | 0.66 | 0.75 | 0.45 | 1.025, −8.792 |
| ettin-reranker-68m-v1 | 256 | **0.975** | 0.485 | 0.54 | 0.71 | 0.72 | 0.46 | 0.983, −8.197 |
| granite-embedding-reranker-english-r2 | 256 | 0.964 | 0.483 | 0.57 | 0.76 | 0.74 | 0.51 | 46.96, −38.66 (sigmoid output) |
| gte-reranker-modernbert-base | 256 | 0.974 | **0.490** | 0.49 | 0.81 | 0.74 | 0.50 | 23.19, −19.96 (sigmoid output) |
| mxbai-rerank-base-v2 | 256 | not run (killed) | | | | | | |

Reading: on this (easy, 1k-image, no-caption) set the spread is small: +0.011 AUC for the best 68M/150M-class models over MiniLM-L6,
and the 17m/32m Ettin models are *below* the current model. Published Ettin gains (web search) do not show on our VLM-description documents.
MiniLM at 256 tokens is marginally better than 160, so the 160 cutoff is not costing much on web-sample docs (median 98 words).
Latencies measured in that run (1,155 ms for MiniLM/150 pairs) were taken at system load 30–57 and are **not valid**; PLAN.md's ~320 ms is the reference.
CPU latency for the final 2–3 candidates must be re-measured one model at a time when the laptop is idle.

## Hypotheses for the archive sample (to be tested with the GPU scores)
1. Captions in the document ("Caption: …" prefix) matter more than the model: names + place + event appear verbatim. Test every model with plain vs caption docs (`raw_doc` vs `raw_cap`).
2. Larger models (gte-modernbert, ettin-68m/150m, mxbai-base/large, bge-v2-m3, Qwen3-0.6B) should widen the gap on harder queries (5k photos, hinglish, event_year). If the gap stays ≤ 0.01 AUC, keep MiniLM and spend effort on E1/E7 instead.
3. Cascade ettin-17m (150) → 68m (top 30) is simulated in `analyze.py --cascade`; worth it only if 68m alone is over the 300 ms budget on CPU.

## Recommendation (interim)
Keep `cross-encoder/ms-marco-MiniLM-L-6-v2` for now; no candidate earned a swap on the web sample. Decide after the challenge-set GPU scores
(expected deltas with captions) and idle-laptop latency for at most: gte-reranker-modernbert-base, ettin-reranker-68m-v1, and the current model at 256.
Integration, if a swap wins: `RERANK_MODEL=<id>` in .env, `max_length=256` in app/text.py, and replace the bare sigmoid in `rerank()` with the
fitted `sigmoid(a·raw + b)` constants from `analysis_archive.json`, then re-sweep `REL_MIN` with `eval/run_eval.py --sweep`.
