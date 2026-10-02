# Experiments — archive sample (5,002 photos), started 2026-10-02

Hackathon mode: many ideas in parallel, measured the same way, keep what wins. Baseline = the v2 system
(PLAN.md section 11) indexed on the archive sample with NO caption use. Every experiment reports the same table.

## Ground rules for parallel experiments
- One Postgres database `imgsearch` (DATABASE_URL in .env). The live tables (`images`, `faces`, `persons`,
 `image_persons`, `person_refs`) are read-only for experiments.
- An experiment writes only to `exp/<name>/` (code, results.md) and to DB tables/views prefixed `exp_<name>_`.
 Copy what you need: `CREATE TABLE exp_x_images AS SELECT * FROM images` then change columns there.
- Run the search through `app.search` by swapping `app.search.TABLES` (images / scores / persons / faces) to your copies.
 Do not edit `app/` or `eval/` — propose the change in results.md; the integrator applies winners.
- Measure with `eval/run_eval.py` logic (judged P@10, R@pool, empty, junk/query, latency) on eval/gen_queries.jsonl
 + eval/hand_queries.txt. The judge cache (eval/judgments.jsonl, blind_people.jsonl) is shared and append-safe;
 never rewrite it. Report new judging cost.
- CPU budget: 10 cores / 16 GB shared. Load at most one large local model per experiment; free it when done.
- Write results as: baseline row, your variant rows, per query type, plus 5 example queries that changed and why.
 State the recommendation in one line: adopt / drop / needs X.

## Metrics (same for all)
| metric | meaning |
|---|---|
| P@10 | share of shown top-10 graded relevant (grade 2 = 1, grade 1 = 0.5) |
| R@pool | share of pooled relevant photos the system shows |
| empty | queries with no result although the pool has a relevant photo |
| junk/q | shown top-10 results graded 0 |
| p50/p95 ms | server-side search latency |
| cost | $ per 1k images indexed (index-time ideas) or $ per 1k queries |

## The ten experiments

| # | name | hypothesis | main metric | type |
|---|---|---|---|---|
| E1 | caption_retrieval | Adding caption + headline + keywords + city to full-text, description vector and reranker document raises P@10 and recall on scene/place/event queries | P@10, R@pool by type | offline, cheap |
| E2 | caption_people | Names in captions (Luna text-only NER, canonicalised) become high-confidence `image_persons` evidence and face references; clusters then name the 31% uncaptioned photos | person & person_context P@10/recall, % photos with a confident name | Luna text ~$0.2 |
| E3 | caption_aware_vlm | Luna describing the image WITH the caption as context yields better people names, places and events than image-only, without parroting wrong captions | blind-judge name accuracy, P@10 | Luna ~$1.8 |
| E4 | image_embedding | A stronger CPU-feasible image-text model (from research shortlist) beats siglip2-base on SigLIP-only recall and full-system P@10 at acceptable ms/image | SigLIP-only R@pool, P@10, ms/image | local, heavy |
| E5 | reranker | A 2026 reranker (bge-v2-m3, mxbai-v2, Qwen3-Reranker…) beats ms-marco MiniLM on AUC over judged pairs within 300 ms / 150 docs | AUC, P@10, ms | local, medium |
| E6 | query_understanding | An LLM parse of the query (people, city, year range, event, visual attributes, expanded terms, Hinglish normalisation) used as filters + recall text raises P@10 on hinglish/event_year/place queries | P@10 by type, added latency | Luna text, ~$0.02/1k q |
| E7 | learned_fusion | A learned ranker (logistic / LightGBM, query-grouped CV) over per-candidate features beats the hand formula relevance = P(people) × P(context) | P@10, junk/q (CV) | offline, needs judgments |
| E8 | dedup_diversity | Near-duplicate bursts are common; collapsing them in results raises distinct relevant results, and skipping the VLM for duplicates cuts index cost | dup pairs in top-10, distinct relevant@10, % VLM calls saved | local, cheap |
| E9 | vlm_cost | detail=low / no reasoning / shorter schema / Batch API keep description quality while cutting $ per 1k images | P@10 on affected queries, $ per 1k | Luna on 300 imgs |
| E10 | vlm_rerank | Luna looking at the top-20 thumbnails with the query ("accurate mode") lifts P@10 enough to justify a 2–4 s refine step shown after the fast results | P@10 lift, ms, $ per query | Luna ~$0.001/q |

Dependencies: E7 needs the judged pools from baseline + E1 (more labels). Everything else can start once the baseline index exists.
Idea sources: our own misses (PLAN.md section 11), Kaggle retrieval competitions (see research notes in exp/research/), 2026 model surveys.

## Results log
(filled in as experiments finish; one line per experiment, link to exp/<name>/results.md)
