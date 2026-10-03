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
- E3 caption_aware_vlm — EVALUATED 2026-10-02 with Jev stage 2 @0.2: P@10 0.67 / R 0.66 / empty 9 / junk 0.20 vs default 0.64 / 0.70 / 6 / 0.26: a wash within
 judge noise; likely_queries (doc2query) added nothing; confident-name photos 1,689 vs 1,666 (caption names are already integrated via E2). NOT adopted:
 keep the simpler image-only prompt. Option kept for later: the `caption_consistent` flag to down-weight caption names of people not visible. Earlier notes: photos with a ≥0.8 name 743→1,345 (+81%), 2,014 name guesses,
 caption_consistent=false on 45 photos (captions naming people not visible), likely_queries written for all. Needs the build+eval step. exp/caption_aware_vlm/vlm_cache.jsonl
- E2 caption_people (extraction done, measurement not run): 3,435 captions → 1,931 canonical persons with aliases. exp/caption_people/{names.jsonl,persons.json}
- E7 learned_fusion (cut off): only the dynamic-cutoff sweep landed: floor 0.3, ratio 0.5, k 0. No learned ranker trained.
- INTEGRATION (2026-10-02, live code): captions in search_text + desc vector + reranker doc (256 tokens), caption names → image_persons(source='caption')
 + caption-seeded person_refs, cached LLM query parse (names/years/rewrite/expansions; city unused). Result @thr 0.5: P@10 0.55, R@pool 0.61, empty 9,
 junk/q 0.53 vs baseline 0.59 / 0.48 / 26 / 0.36. Recall +13, empties −17, precision −4. Reranker is now the limiting stage. eval/integrated_report.txt
- E11 decision_model (Jev as stage 2, live code app/decide.py): P@10 0.55→0.72, junk/q 0.53→0.14 at thr 0.5; dominates MiniLM at every threshold;
 hit@10 0.86; cost ≈$0.34 for the whole eval. Single-call latency 2.5 s; batched 50 docs/call: same quality, p50 811 ms → ADOPTED as default stage 2 (thr 0.2: P@10 0.64, R 0.70, empty 6, junk 0.26, hit@10 0.88). eval/jev_batched_report.txt, PLAN.md §12.
- E12 jev_parse (2026-10-02): Jev cannot replace the Luna query parse (no text generation → no Hinglish rewrite, no canonical names). It classifies well:
 city 99% and has-person 97% agreement with Luna, p50 516 ms, $0.05/1k queries. Keep the time-boxed Luna parse; use Jev for city if city ever becomes a boost. exp/jev_parse/
- REVIEW FIXES (2026-10-02): linker bug (13/198 queries required a wrong person) fixed; RRF before cap; two tiers; concurrent parse; city aliases;
 judge v2 (caption+date, caption names count); empties = 0. Result: P@10 0.78 [0.74–0.82], R@pool 0.90, hit@1 0.70, hit@10 0.92, never-retrieved 7
 (from 0.64 / 0.70 / 0.64 / 0.88 / 10). Place 0.53→0.79, event_year 0.42→0.87. eval/v2_report.txt, PLAN.md §12.
- E13 faces (measured on 5k set, 2026-10-02; re-run on 23k): 478 caption-seeded references, 447 people (20 with ≥ 2 refs). Pairwise ROC: same-person median
 cosine 0.36 vs different 0.065; TPR at 0.5 only 0.24. Naming precision against the gallery (caption hidden): 0.29 @0.4, 0.63 @0.45, 0.87 @0.5, 0.89 @0.55;
 with margin ≥ 0.15 over the runner-up: 0.78 @0.4 (181 faces). Current thresholds (0.55 cluster / 0.45 gallery) are about right for THIS model; AuraFace's
 separation is the ceiling → benchmark a stronger commercially licensed model on the same reference pairs (research running). Face sizes fine (p50 183 px,
 2% < 48 px), so MIN_FACE is not the problem. exp/faces/measure.py
 Pipeline checked: RGB input (current) beats BGR on clean same-person pairs (Mamata 0.40 vs 0.25), alignment is a similarity transform to the
 ArcFace template, crops come from a 1280px decode of the original. Clean large frontal pairs (Ricciardo) reach only 0.49 → the model is modest.
 Label noise is real: some caption-seeded "same person" pairs score 0.03 (the single detected face is not the named person).
 Research (exp/research): no commercially clean open model beats AuraFace (AdaFace/InsightFace/EdgeFace/LVFace weights are research-only);
 paid routes: InsightFace commercial licence (same architecture, ~4-6x fewer pose errors) or AWS Rekognition ($1 per 1k faces; RecognizeCelebrities
 coverage of Indian figures untested). Free improvements: caption-conditioned naming (lower bar when the caption names the person), reference
 cleaning (drop refs disagreeing with their person's other refs; min face 80 px; high det score), margin rule.
- E13 faces on 33k (exp/faces/measure_33k.txt): 3,562 caption refs, 2,796 people (385 with ≥ 2 refs), same-person median 0.42. Naming precision
 (caption hidden): 0.85 @0.55 (2,923 faces) vs 0.85 @0.40+margin 0.15 (3,179 faces, 529 uncaptioned photos named vs 446). Top names gained in
 uncaptioned photos: Modi 76, Fadnavis 40, Kejriwal 39, Uddhav 34, Mamata 34. IMPLEMENTED in step_people: open-set margin rule
 (GALLERY_SIM 0.40 + GALLERY_MARGIN 0.15, or ≥ 0.55), caption-conditioned face assignment in multi-face photos (caption2 refs),
 reference outlier cleaning. Re-measure after the next people run.
- SCALE 33k (2026-10-02): P@10 0.79 / R 0.88 / empty 3 / junk 0.16 (5k: 0.78 / 0.90 / 6 / 0.15); hit@1 0.57, hit@10 0.79, never retrieved 17
 (5k: 0.70 / 0.92 / 7); p50 878 ms. Precision holds at 6.7× data; known-item rank drops because more valid photos exist. eval/scale33k_report.txt
- E14 cluster_vlm (2026-10-03): Luna names face clusters from the 3 clearest crops side by side. 2,886 clusters, $0.85, 802 named (Modi 221 photos,
 Yediyurappa 75, Siddaramaiah 75, Nitish 71, Mamata 70...). First version spread the name to every photo in the cluster: photos with a confident name
 12,253→12,605, uncaptioned photos with a name 1,303→1,736, BUT only 45% of named clusters agreed with their single-face caption names: (a) spelling
 duplicates in the directory ("A. Revanth Reddy" vs "Revanth Reddy"), (b) chained clusters mixing people who appear together (Nitish Kumar /
 Sushil Kumar Modi, Siddaramaiah / D. K. Shivakumar). Fix: match Luna's name to the directory (link_name) and propagate per face weighted by
 similarity to the crops Luna saw (full ≥ 0.55, fading to 0.40, none below). v2 result: confident-name photos 12,570 (vs 12,253 without, 12,605 v1), uncaptioned photos with a name 1,724 (vs 1,303),
 photo-level agreement with single-face caption names 62% (v1 row-level 9%). Remaining disagreements are the long tail: foreign sportspeople Luna
 confuses (James Blake/Tsonga, Murray/Djokovic, Chappell/Waugh), spelling variants (Stan/Stanislas Wawrinka) and some caption noise; the popular
 Indian figures agree. Eval at weight 0.9: person P@10 unchanged (0.80/0.88) but person_context 0.66→0.54, ALL 0.79→0.77: a lone cluster name at conf 0.99 counted as
 "sure" (0.89) despite 62% agreement. Weight lowered to 0.6 (lone cluster name = "possible", needs a second source to be sure). At 0.6: person_context 0.57 / 0.65 (top tier) vs 0.66 / 0.68
 without cluster naming and 0.54 / 0.64 at 0.9; ALL 0.77 vs 0.79 (inside the ±4 CI). Coverage: confident-name photos 12,331 (vs 12,253), uncaptioned photos
 with a name 1,475 (vs 1,303). DECISION: keep at 0.6. The eval cannot reward uncaptioned coverage (its queries were written from mostly captioned photos),
 while the cost is 2–3 person_context queries within noise; the human grading / real-query eval should settle it. Revert = CLUSTER_VLM_MIN > 1.
- BUGFIX round (2026-10-03, paired eval, 208 queries incl. 10 real): richer stage-2 doc (expression/mood/colours/time/date, caption ≤ 400 chars) +
 clean English Jev query. Result: no clear change overall (P@10 0.78→0.79 [−0.00,+0.03]); hinglish top tier +0.11, real nDCG 0.88→0.95,
 person_context +0.04; known-item hit@10 0.79→0.75, never retrieved 16→19 (the longer text dilutes the bge-small recall vector).
 Follow-up: embed a compact doc (no colours/time/date) for recall, keep the rich doc for Jev — measured below.
- E10 vlm_rerank: CANCELLED before build (user call: enough experiments; conflicts with < 1 s target; judge = same model, circular).
