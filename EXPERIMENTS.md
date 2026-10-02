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
- Compute: the laptop overheats under parallel model inference (load 57 seen). Rule: NO neural model inference on the laptop
 beyond the live search path and single-process eval runs (torch threads ≤ 4, one model at a time). Anything heavier (embedding
 5k images, scoring thousands of reranker pairs, fine-tuning) runs on an ephemeral GPU machine. Jobs there are stateless: inputs rsynced up, npy/jsonl results
 rsynced back and loaded into local exp_ tables. Terminate the machine when the batch is done.
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
Baseline (archive sample, 198 queries, judged): P@10 0.59, R@pool 0.55, 24 empty, junk/q 0.36. Ablation: without reranker R@pool 0.96 / P@10 0.38;
SigLIP alone 0.85 / 0.41. event_year recall 0.00, hinglish 0.09 (people index thin: Luna named someone in 743 of 5,002 photos). eval/baseline_report.txt
- E8 dedup_diversity: DROP for this sample. 3 near-dup groups in 5,002; phash adds nothing over SigLIP; shoot grouping reaches 1 uncaptioned photo; VLM-skip saving 0.1%. Re-measure on the full larger archive. exp/dedup_diversity/results.md
- E6 query_understanding: ADOPT the parse (canonical names, years, Hinglish rewrite; union original+rewritten text to avoid over-rewrites like ramp→runway);
 city as boost only (62% of photos have a city; the filter dropped a relevant Srinagar photo). Parsed: R@pool 0.63→0.69, empties 22→15, P@10 flat (0.59→0.58);
 event_year recall 0.00→0.50, hinglish 0.15→0.45, place 0.37→0.52. Blocker: Luna text parse p50 2.2 s / p95 3.5 s, $0.13 per 1k queries → needs cache,
 a faster model, or a progressive UI. exp/query_understanding/results.md
- E9 vlm_cost (cost side final, quality pending): baseline $0.396/1k; Batch API $0.198 (12 min turnaround for 300); compact schema $0.244; detail=low $0.347;
 512px $0.351; gpt-5-nano $0.299; omitting reasoning_effort defaults to medium and costs MORE ($0.491). exp/vlm_cost/
- E1 caption_retrieval (scored from saved runs; agent cut off by spend limit): caption in full-text (a) and in desc vector (b): P@10 0.59→0.61/0.63, recall flat, place P@10 0.64→0.78.
 Caption in the RERANKER doc (c): a different operating curve: at thr 0.2 empties 26→5, R@pool 0.48→0.58, event_year recall 0.00→0.50, hinglish 0.08→0.31,
 but P@10 0.59→0.54 and junk/q 0.36→0.81; at thr 0.7: P@10 0.58, R 0.52, empty 13, junk 0.45. ADOPT a+b now; adopt c together with a recalibrated
 threshold / better reranker (E5). exp/caption_retrieval/{a,b,c,baseline}.json
- E5 reranker (eval pairs, n=4,501, GPU-scored): caption docs raise AUC for EVERY model (+0.02–0.04). MiniLM-L6 current: AUC 0.789 (plain) / 0.836 (caption).
 Best: mxbai-rerank-large-v2 0.894 (too slow on CPU), bge-reranker-v2-m3 0.888 (~8 s/150 pairs CPU, GPU or hosted only), gte-reranker-modernbert-base 0.875 (~3 s CPU),
 ettin-68m 0.873, ettin-32m 0.852. Small Ettins (17m/32m) do not beat MiniLM by enough; a real jump needs a 150M+ model → GPU serving, hosted rerank
 (Voyage ~$0.8/1k queries) or distillation (E12, not run). exp/reranker/analysis_archive.json
- E4 image_embedding (source recall@50 only, GPU-embedded): siglip2-base 0.815; fg-clip2-base 0.783 (worse); siglip2-so400m 0.869 (+5.4, attribute 0.79→0.91,
 hinglish 0.72→0.81; ~6× CPU cost, dim 1152); PE-Core-L 0.861 (scene 0.91, place worse). Recall before the reranker is already 0.96 from the 4-source union,
 so the end-to-end gain is bounded; so400m is the only candidate worth a full eval later. exp/image_embedding/source_recall.json
- E3 caption_aware_vlm (Luna pass done for 5,002 photos, $≈1.8; eval not run, agent cut off): photos with a ≥0.8 name 743→1,345 (+81%), 2,014 name guesses,
 caption_consistent=false on 45 photos (captions naming people not visible), likely_queries written for all. Needs the build+eval step. exp/caption_aware_vlm/vlm_cache.jsonl
- E2 caption_people (extraction done, measurement not run): 3,435 captions → 1,931 canonical persons with aliases. exp/caption_people/{names.jsonl,persons.json}
- E7 learned_fusion (cut off): only the dynamic-cutoff sweep landed: floor 0.3, ratio 0.5, k 0. No learned ranker trained.
- INTEGRATION (2026-10-02, live code): captions in search_text + desc vector + reranker doc (256 tokens), caption names → image_persons(source='caption')
 + caption-seeded person_refs, cached LLM query parse (names/years/rewrite/expansions; city unused). Result @thr 0.5: P@10 0.55, R@pool 0.61, empty 9,
 junk/q 0.53 vs baseline 0.59 / 0.48 / 26 / 0.36. Recall +13, empties −17, precision −4. Reranker is now the limiting stage. eval/integrated_report.txt
- E11 decision_model (Jev as stage 2, live code app/decide.py): P@10 0.55→0.72, junk/q 0.53→0.14 at thr 0.5; dominates MiniLM at every threshold;
 hit@10 0.86; cost ≈$0.34 for the whole eval. Latency p50 2.5 s is the blocker (150 calls × ~500 ms). eval/jev_report.txt, PLAN.md §12.
- E10 vlm_rerank: CANCELLED before build (user call: enough experiments; conflicts with < 1 s target; judge = same model, circular).
