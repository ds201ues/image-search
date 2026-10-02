# E5 reranker — results (2026-10-02)

**Question:** does a 2026 reranker beat `cross-encoder/ms-marco-MiniLM-L-6-v2` (max_length 160) on our judged pairs within the CPU budget?
Baseline context (eval/baseline_report.txt): recall is 0.96 before the reranker and 0.55 after it, so the reranker is the system's biggest loss.

## Setup
- Pairs: every judged (query, photo) pair for queries that have both relevant and irrelevant judgments.
 Archive sample: 98 queries, 4,501 pairs (grade 2 = 552, 1 = 1,987, 0 = 1,962), 2,992 pairs whose photo has an IPTC caption.
 Web sample (old 1k index, no captions): 98 queries, 4,264 pairs.
- Documents: exactly as `app/search.py` builds them (people line + Luna description), and a **caption** variant with `Caption: <IPTC caption>` prepended.
- Scoring on a GPU (fp16) with `gpu_score.py`; analysis on the laptop with `analyze.py` / `analyze_types.py` (no models).
- AUC = mean per-query AUC (grade 2 above grade 0). P@10j = precision of the top-10 among judged pairs (grade 1 counts 0.5). Calibration = 2-parameter sigmoid on raw scores (grade 2 vs rest), fitted in-sample.

## Archive sample: ranking quality (sorted by plain-doc AUC)
| model | params | AUC plain | AUC caption | Δ caption | P@10j plain | P@10j caption | calib a, b (caption) |
|---|---|---|---|---|---|---|---|
| mxbai-rerank-large-v2 (ceiling) | 1.5B | 0.881 | 0.894 | +0.014 | 0.491 | 0.491 | 15.122, -16.393 |
| bge-reranker-v2-m3 (ceiling) | 568M | 0.860 | 0.888 | +0.028 | 0.477 | 0.493 | 5.006, -3.047 |
| gte-reranker-modernbert-base | 149M | 0.857 | 0.875 | +0.017 | 0.485 | 0.491 | 15.926, -14.177 |
| Qwen3-Reranker-0.6B (ceiling) | 0.6B | 0.854 | 0.872 | +0.018 | 0.482 | 0.484 | 0.276, -1.376 |
| **ettin-reranker-68m-v1** | 69M | **0.850** | **0.873** | +0.023 | 0.476 | 0.478 | 0.648, -6.027 (plain: 0.667, -6.167) |
| ettin-reranker-150m-v1 | 151M | 0.847 | 0.873 | +0.026 | 0.474 | 0.483 | 0.645, -5.905 |
| mxbai-rerank-base-v2 | 0.5B | 0.846 | 0.845 | -0.001 | 0.474 | 0.467 | 8.999, -10.033 |
| granite-embedding-reranker-english-r2 | 149M | 0.844 | 0.870 | +0.027 | 0.469 | 0.483 | 33.979, -28.517 |
| ettin-reranker-32m-v1 | 33M | 0.813 | 0.852 | +0.039 | 0.461 | 0.471 | 0.644, -6.120 |
| MiniLM-L6-v2 @256 | 23M | 0.798 | 0.837 | +0.038 | 0.445 | 0.470 | 0.267, -1.239 |
| **MiniLM-L6-v2 @160 (current)** | 23M | **0.789** | 0.836 | +0.047 | 0.445 | 0.462 | 0.257, -1.138 |
| ettin-reranker-17m-v1 | 18M | 0.786 | 0.817 | +0.031 | 0.443 | 0.459 | 0.651, -6.285 |

Per query type (AUC, plain → caption):
| model | person | person_context | scene | attribute | place | hinglish |
|---|---|---|---|---|---|---|
| MiniLM @160 | 0.71 → 0.74 | 0.80 → 0.94 | 0.84 → 0.87 | 0.90 → 0.86 | 0.58 → 0.63 | 0.84 → 0.91 |
| ettin-68m | 0.82 → 0.84 | 0.86 → 0.92 | 0.86 → 0.88 | 0.88 → 0.93 | 0.82 → 0.79 | 0.83 → 0.85 |
| gte-modernbert-base | 0.78 → 0.81 | 0.84 → 0.90 | 0.89 → 0.89 | 0.93 → 0.93 | 0.81 → 0.76 | 0.79 → 0.91 |
| bge-v2-m3 | 0.72 → 0.79 | 0.89 → 0.93 | 0.92 → 0.94 | 0.90 → 0.92 | 0.77 → 0.79 | 0.83 → 0.85 |
(event_year has 2–3 queries; ignore.) The current model's weakest type is **place** (0.58): newer models reach ~0.80 there.

## Findings
1. **The current reranker is the worst-but-one model on the archive sample.** Every candidate ≥ 68M parameters is +0.05 to +0.09 AUC above it. The web sample (easy, 1k photos) hid this: there the spread was 0.95–0.98 for all models.
2. **Caption docs help every model** (+0.014 to +0.047 AUC; only mxbai-base is flat). The smaller the model, the bigger the gain: MiniLM +0.047, Ettin-32m +0.039, Ettin-68m +0.023. Gains are largest on person_context (+0.06 to +0.14) and hinglish; `place` sometimes drops slightly (captions name the city, which can distract a "stadium"-type query). → This is direct evidence for E1: put the caption in the reranker document.
3. **Model size beyond ~70M buys little on our documents**: ettin-68m 0.850 ≈ ettin-150m 0.847 ≈ gte-base 0.857 ≈ Qwen3-0.6B 0.854; only mxbai-large (1.5B) is clearly higher (0.881). The 17m/32m Ettin models are not an upgrade.
4. **Description embedders (desc recall source, recall@50 of judged relevant):** bge-small 0.689 (plain) / 0.700 (caption); bge-base 0.675; granite-small-r2 0.651; Qwen3-0.6B@384 0.683; Qwen3-0.6B@1024 0.736. Only the full-width Qwen3 beats bge-small, at ~12× GPU encode cost and 2.7× storage. Keep bge-small; captions in the embedded text help it slightly.

## Winner chosen for the full CPU eval
`cross-encoder/ettin-reranker-68m-v1`, max_length 256: same quality as gte-modernbert-base at half the parameters, +0.061 AUC over the current model (plain docs), and the only candidate with a chance at the latency budget.

## Full-system eval (archive sample, plain docs, judged P@10 / R@pool / empty / junk per query)
FILLED_BELOW

## CPU latency (idle laptop, 4 threads, 150 pairs, best of 5)
FILLED_BELOW

## Recommendation
FILLED_BELOW
