# Image Search (DAM) for a newsroom photo archive — Project Plan

Last updated: 2026-10-01
Status: 5,002-file stratified sample of a larger news archive (`photos/`, indexed in Postgres db `imgsearch`). Experiments plan in EXPERIMENTS.md.

## 1. Goal

Natural-language image search for journalists over the newsroom photo archive.
A journalist types a sentence ("modi in pink jacket", "priyanka gandhi in parliament",
"farmers protesting on a highway") and gets relevant images, their metadata, and the
search time in under 1 second.

Requirements, in priority order: cheap, fast, accurate.

## 2. Known facts and constraints

| Topic | Fact |
|---|---|
| Users | Journalists. They type sentences, not keywords. |
| Data | Archive files carry IPTC/XMP metadata (measured on the 5,002-file archive sample, 2026-10-02): caption 68%, byline 64%, city 62%, headline 40%, keywords 20%. 31% have no caption. Mostly professional camera originals (~3,200 px long side). No journalist query logs. |
| Dates | 94% of files have one: EXIF DateTimeOriginal (82%), else IPTC/XMP created date, else a date typed in the caption. Sanity window 1990..today (unset camera clocks give 1980/2034/2099). Stored as a column and used as a filter. Never inferred by a model. |
| Names | Any public figure, mostly popular Indian ones (politicians, cricketers, actors, business). |
| Scale | 50,000 image sample now. Lifetime ~1 crore (10M) with daily additions. Daily volume unknown yet. |
| Infra | Own data centres with lots of CPU, RAM, storage. GPU can be provisioned if justified. |
| External APIs | Fine for testing. Production wants lowest cost, likely self-hosted. |
| Budget | Not fixed. Aim for the cheapest workable system. |
| Stack preference | None. We pick. |

## 3. Options considered

| Option | How | One-time cost for 10M | Verdict |
|---|---|---|---|
| A. Metadata only | VLM -> JSON -> full-text search | ~$1.5k–3k | Misses synonyms and visual attributes; cannot do names reliably |
| B. Image embeddings only | SigLIP 2 -> vector search | ~$100 GPU time | Great for attributes ("pink jacket"), bad at names and dates, no readable metadata |
| **C. Hybrid (chosen)** | A + B + face recognition, fused at query time | ~$2k–3.5k | Covers names, attributes, places, activities, "find similar" |
| D. Managed cloud | AWS Rekognition + vector service | ~$20k+ | 10x cost, lock-in, Western-skewed celebrity list |

Why the original idea (compress -> cheap LLM -> structured metadata -> text search) is
not enough on its own:
- Cheap VLMs name famous faces inconsistently; hosted APIs often refuse to name real people.
- Keyword search misses synonyms ("rally" vs "protest").
- Year guessed by an LLM is usually wrong.

Industry reference points:
- AP Newsroom moved to AI search that finds things "even if never tagged or captioned" (embedding-based).
- Newsroom DAMs (ResourceSpace, Immich) do names via face embeddings matched to a labelled gallery.
- Apple Photos on-device = face embeddings + fixed-vocabulary scene classifier + CLIP-style
 embedding. That is Option C minus the VLM. The VLM is the only expensive piece and exists
 to produce rich human-readable descriptions.

## 4. Architecture (Option C)

### 4.1 Ingest pipeline (once per image)

1. Download. Resize to 768px long edge. Store thumbnail. Extract EXIF date and any API metadata (date, photographer, event, filename).
2. **VLM structured JSON.** Fields: people (described, not named), clothing and colours,
 place, landmarks, objects, activity, event type, visible text, mood.
 - Pilot: Gemini 2.5 Flash-Lite via Batch API (~$0.13 per 1k images batch, ~$0.25 realtime; ~1,000 input tokens per 768px image + ~300 output tokens).
 - Production candidate: Qwen3-VL 4B (or 2B) on own CPUs. ~5–15 s/image on CPU. 50k images ≈ 140 CPU-hours. Also names well-known faces without refusals. Benchmarked against Flash-Lite on the same 500 images.
3. **SigLIP 2 image embedding**, local. Base model at 256px on CPU; large/so400m tested for quality. Matryoshka-truncate to 256 dims + quantize so 10M vectors ≈ 3GB.
4. **Faces.** Detect, embed (512-d), match against a gallery of known public figures.
 - Model: AuraFace via ONNX Runtime (commercially usable). InsightFace default models (buffalo_l, antelopev2) are non-commercial unless licensed.
 - Automated gallery bootstrap:
 a. Seed from Wikidata: Indian politicians, cricketers, actors, business leaders (2,000+ names with Wikimedia Commons photos). Download, embed, done.
 b. VLM voting: cluster all faces in the archive; if a cluster's faces were consistently named the same by the VLM, label the cluster automatically.
 c. Human review screen for unlabelled/ambiguous clusters. Optional, not blocking.
 - Expectation: top-200 figures near perfect, long tail decent and improving.

### 4.2 Search (target < 1 s)

1. Query parse: cheap LLM turns "priyanka gandhi in parliament" into
 `{person: "Priyanka Gandhi", text: "parliament"}`. Adds ~300 ms, so make it a toggle and measure. Fallback: match query words against known-people list.
2. Hard filters: person (from face layer), date range.
3. BM25 full-text on VLM metadata + vector search on SigLIP text embedding, fused with reciprocal rank fusion.
4. Rerank top 100 by SigLIP text-to-image similarity (milliseconds, local). This is what makes "pink jacket" precise.
5. Return results with per-stage timing.

### 4.3 Storage

- Postgres + pgvector + built-in full-text search, one box. Fits the RAM-heavy data centre.
- pgvector is fine to ~5–10M vectors; above that latency degrades. Keep the storage layer behind an interface so Qdrant/Milvus can replace it without touching the pipeline.
- Thumbnails in object storage or disk (10M × 50KB ≈ 500GB).

### 4.4 Stack

Python 3.12, FastAPI, Postgres 16 + pgvector, SigLIP 2 via Hugging Face (ONNX for CPU),
AuraFace via ONNX Runtime, Gemini Batch API (pilot) / Qwen3-VL via vLLM or llama.cpp (production),
small React (Vite) UI. Everything runs on one Linux box or a Mac for the pilot.

## 5. Cost estimates

| Item | 50k pilot | 10M lifetime |
|---|---|---|
| VLM metadata (Gemini Flash-Lite batch) | ~$7 | ~$1.3k–2.5k |
| VLM on own CPUs (Qwen3-VL 4B) | electricity | electricity; needs a few GPUs or weeks of a large CPU pool |
| SigLIP + face embeddings (rented GPU) | ~$2 | ~$100–200 |
| Serving box (Postgres, 64GB RAM) | own infra | own infra, or ~$200/mo cloud |
| Thumbnail storage | negligible | ~$10/mo cloud or own disk |

All-in roughly $0.20–0.35 per 1k images with the API VLM. The VLM dominates; everything else is near-free.
Self-hosted Qwen3-VL 8B on a rented A100 ≈ $0.15 per 1k but ~55 GPU-days for 10M, so the API wins on time unless policy forces on-prem.

## 6. Evaluation (we create it, no journalist queries exist)

- Take 300 random images, have the VLM describe them.
- Write 50 queries at three difficulty levels:
 1. Name only: "Rahul Gandhi"
 2. Name + attribute: "Modi in pink jacket", "Priyanka Gandhi in parliament"
 3. No-name scene: "farmers protesting on a highway", "cricket celebration in rain"
- Ground truth: stronger model judges top 20 results per query, plus spot checks.
- Metrics: precision@10, recall on name queries, latency p50/p95 per stage, cost per 1k images.
- Rerun on every model swap.

## 7. Phases

| Phase | Scope | Output |
|---|---|---|
| 0. Pilot (500 images, ~1 week) | Gold set; Flash-Lite vs Qwen3-VL on CPU; SigLIP base vs large recall; face model check | Model choices with numbers, per-image cost and CPU time |
| 1. 50k pipeline + UI | Full ingest, hybrid search, demo UI with timing | Demoable product |
| 2. Hardening | Queue, idempotent re-runs, face labelling screen, monitoring, daily ingest | Production-ready for daily additions |
| 3. Scale to 10M | Backfill (GPU weekend or CPU pool), migrate vector store if pgvector strains | Full archive searchable |

### Pilot deliverables
1. Repo with ingest pipeline: resize, metadata, Flash-Lite JSON, SigLIP embedding, face detect + gallery match.
2. CPU benchmark script for Qwen3-VL 4B and SigLIP (images/sec per core).
3. Hybrid search API with per-stage timing.
4. Evaluation harness with the 50 queries.
5. Basic UI: sentence search box, grid, metadata panel, timing breakdown, filters (person, date), "more like this".

## 8. Open questions

1. Daily new-image volume (decides CPU vs small GPU for daily embedding).
2. What exactly does the image API return besides the file (date, photographer, headline, event, filename pattern)?
3. Which GPU, if any, can be provisioned for the 10M backfill?
4. Gemini API key available for the pilot, or build self-hosted-only mode from day one?
5. Is sending compressed images to Google acceptable in production, or testing only? (Decides API vs self-hosted VLM for production.)

## 10. Pilot results (2026-10-01, 100 images, MacBook CPU)

### Decisions changed during the pilot
| Was | Now | Why |
|---|---|---|
| Gemini 2.5 Flash-Lite for descriptions | GPT-6 Luna (`gpt-6-luna`, released 2026-09-22, $0.10 / $0.50 per 1M tokens) | User choice; similar price; strong results |
| Names from a Wikidata face gallery | Names from GPT-6 Luna in the same call; faces learn from Luna | Luna named 9/10 test photos correctly (the 10th had the face hidden). Wikimedia seeding was slow, rate-limited (HTTP 429) and its data is patchy (Narendra Modi has no citizenship on Wikidata) |
| Wikidata gallery required | Optional extra (`app/gallery.py`, curated list in `app/gallery_names.txt`) | Learned faces from Luna-named single-face photos cover the same need automatically |
| SigLIP rerank of top 100 | Dropped | SigLIP is already one of the fused retrievers; reranking with the same model adds nothing |
| React UI | Single static HTML page served by FastAPI | Simpler, no build step |

### People rules (re-tune on 50k)
- Luna name at confidence ≥ 0.8 is kept. 0.6–0.8 is kept only if face matching agrees. Below 0.6 ignored.
 All pilot Luna errors (Tom Cruise for Charlie Sheen, Twinkle Khanna for Farah Khan, Mohammed Siraj for Prasidh Krishna) were below 0.8.
- Single-face photo + one Luna name at ≥ 0.9 → that face becomes a reference for the person.
- Face match needs cosine ≥ 0.5 (AuraFace). One false positive at 0.49 in the pilot.
- Names appear in search only via `image_persons`; names Luna leaks into tags are stripped.

### Search rules
Superseded by the v2 architecture in section 11. The first round of fixes (vector cutoffs, word-coverage rules,
strict/loose names, "Other matches") treated symptoms; they were removed.

### Measured
| Metric | Value |
|---|---|
| GPT-6 Luna cost | $0.35 per 1k images (~1,050 tokens in, ~490 out incl. reasoning at effort=low) → ~$18 for 50k, ~$3.5k for 10M |
| GPT-6 Luna speed | ~2–6 s per call; 100 images in 76–93 s with 8 parallel requests |
| SigLIP 2 base, CPU | ~45 ms/image (4 threads), 18 ms per text query |
| Face detect + embed, CPU | ~82 ms/image |
| Search latency | p50 23 ms, p95 28 ms server-side (100 images) |
| Eval (57 queries) | hit@1 0.98, hit@5 0.98, recall@10 0.96, precision 0.78 (1.00 on name queries), 2.5 results per query on average |

Ablation (hit@1, all queries): SigLIP alone 0.98, full-text alone 0.86, people alone 0.35 (names only).
At 100 images every retriever looks good; the test is too easy to separate them. Re-run the ablation on the 50k set before deciding what to drop.

### Next
1. Grow the sample (scrape ~1–2k images) or get the larger archive; extend eval queries with more distractors.
2. Measure pgvector + full-text latency at 50k; check name-confidence rules on a bigger set.
3. Try the Batch API for GPT-6 Luna (typically 50% cheaper) for the backfill.
4. Benchmark a self-hosted VLM (Qwen3-VL) on the same eval for the 10M cost decision.

## 11. Architecture v2 (2026-10-01): retrieve generously, judge precisely, keep evidence

### Why v1 kept needing patches
| Symptom | Root cause |
|---|---|
| "flood" returned ocean and calm water | Image similarity measures "looks alike", not "satisfies the query". No threshold on it can fix that. |
| Cutoffs and rules piled up | Three retrievers gave scores that cannot be compared, so fusion and thresholds were guesses. |
| "farah" found nothing | Names were decided once per photo and discarded (0.49 vs a 0.5 bar). Nothing accumulated across photos. |
| Hand rules for names | Query parsing was ad hoc rather than linking against a people directory. |
| Everything scored ~100% | 100 images and labels written by us cannot separate good from bad. |

### Design
```
INDEX (per image, store evidence, decide nothing)
 GPT-6 Luna JSON ─► description fields + name guesses with confidence (all kept ≥0.3)
 SigLIP 2 ─► image vector (how it looks)
 bge-small ─► description vector (what it means)
 YuNet+AuraFace ─► face vectors
CORPUS JOBS (re-run after each ingest)
 face clusters ─► same person across photos (union-find, cosine ≥0.55)
 cluster votes ─► each photo's cluster is named by the OTHER photos' Luna guesses + gallery matches
 (leave-one-out, so a guess never confirms itself)
 people belief ─► P(person in photo) = noisy-OR of independent evidence (view image_person_scores)
QUERY
 link ─► people directory (names, aliases, fuzzy) with a link probability measured from the
 archive: of photos whose description uses the word, how many show the person
 ("trump" high, "cricket"→Smriti Mandhana ~0, unseen words like "kohli" 0.8)
 rewrite ─► confidently linked words written in full ("mamta" → "mamata banerjee")
 recall ─► SigLIP + description vector + full-text (~50 each) + all photos of linked people
 precision ─► cross-encoder reads (query, "People: <names the index believes> + description")
 and judges who and what together → calibrated 0..1
 relevance ─► reranker × P(each confidently linked person is in the photo); show if ≥ 0.2
```
Uncertain names are shown as "possible" (0.5–0.8) instead of being dropped.

Lessons from building it (each was a patch first, then replaced by a system change):
- Excluding "words that appear in descriptions" from name linking broke "trump", "putin", "kohli" (descriptions
 mention them). Replaced by the measured link probability.
- Stripping names and filler words before reranking gave the reranker fragments ("speaking podium" → 0.02).
 Feeding the full query and putting names into the document fixed people + context in one model
 ("trump speaking at a podium": Trump-at-podium photos 0.95–0.98, everything else ~0).
- The judge, asked "is Kohli in this photo?", found Kohli in a pilot's cockpit. Names are now judged blind.
- Through a TLS-intercepting corporate proxy some API calls hang in an SSL read despite client timeouts. Judging uses daemon
 threads with a wall-clock deadline and resumes from cache.

### Evaluation v2
- Queries: GPT-6 Luna writes journalist-style queries from random photos (`eval/make_queries.py`), plus
 `eval/hand_queries.txt` (ours and the user's reports).
- Labels: pool top-20 from five system variants, GPT-6 Luna grades each (query, photo) 0/1/2, cached.
- Names are judged blind: the judge lists who it recognises without seeing the query. When primed
 ("is Kohli here?") it found Kohli in a pilot's cockpit.
- Metrics: P@10, recall against the pool, empty results, junk per query, latency. Threshold sweeps run offline.

### Results on 1,018 news photos, 107 queries (69 Luna-written + 38 ours), judged by GPT-6 Luna
| Query type | n | P@10 | R@pool | junk/query |
|---|---|---|---|---|
| person ("putin", "mamta") | 16 | 0.97 | 0.93 | 0.19 |
| person + context ("trump speaking at a podium") | 15 | 0.59 | 0.75 | 1.07 |
| scene | 37 | 0.86 | 0.66 | 0.05 |
| place | 18 | 0.83 | 0.69 | 0.06 |
| attribute | 21 | 0.73 | 0.83 | 0.33 |
| **all** | 107 | **0.81** | **0.75** | **0.27** |
| any single retriever alone | | 0.50–0.53 | 0.86 | 2.1–2.7 |
Latency p50 401 ms, p95 505 ms (CPU, reranker is ~80% of it). Judging cost so far ≈ $0.45.
Biggest remaining loss: the small reranker scores some relevant scene/place photos near 0
("indoor sports arena", "statement jewelry"); compared offline in `eval/compare_rerankers.py`.

### Results on 100 images, 25 hand queries (before the 1,100-image run)
| | P@10 | R@pool | junk/query |
|---|---|---|---|
| Any single retriever alone | 0.50–0.66 | 0.88–1.00 | 1.8–3.7 |
| v2 full system | 0.89 | 0.83 | 0.08 |
Remaining misses are reranker recall (stadium, airport) — next: compare a larger reranker on the 1,100 set.

## 12. Archive sample: baseline, experiments and the integrated system (2026-10-02)

Data: 5,002-file stratified sample of the larger archive. Eval: 198 queries (160 Luna-written from image+caption, 38 hand), ~14.7k judged pairs.
Experiments and findings: EXPERIMENTS.md (results log) and exp/*/results.md. Research notes: exp/research/README.md.

### Integrated system = baseline + captions everywhere + caption names + cached LLM query parse
| | P@10 | R@pool | empty | junk/q | p50 ms |
|---|---|---|---|---|---|
| baseline (no captions, no parse), thr 0.2 | 0.59 | 0.48 | 26 | 0.36 | ~400 (160-token reranker) |
| integrated, thr 0.2 | 0.53 | 0.64 | 3 | 0.70 | 746 |
| **integrated, thr 0.5 (chosen)** | 0.55 | 0.61 | 9 | 0.53 | 746 |
Per type (thr 0.2): person_context recall 0.57→0.77, scene 0.62→0.71, event_year 0.00→0.50, hinglish 0.09→0.35; place P@10 0.64→0.43 (worse).
People index: photos with a confident name 549→1,666 (caption names 2,392 rows, 478 caption-seeded reference faces, cluster votes 340→917).

### What the numbers say
- Recall and empties improved a lot; precision fell. The reranker (ms-marco MiniLM-L6) is now the limiting stage: with caption text in the
 document it over-scores loose matches (city names, "meeting room" for "high court" at 0.95). E5 measured AUC 0.84 for MiniLM vs 0.89 for
 bge-reranker-v2-m3 / mxbai-large on the same pairs, and captions help every model. Fixing this needs a bigger reranker (GPU or hosted,
 ~$0.8 per 1k queries) or distilling the judge into a small one (E12, not run).
- Query parse: adopt names/years/Hinglish rewrite (added to the original words, never replacing them); city only as a boost (62% coverage).
 Uncached parse is 2.2 s, so it is cached per query (data/query_parses.jsonl) and can be switched off (QUERY_PARSE=0, `parse=false`).
- Image model: siglip2-so400m is the only candidate worth a full eval later (+5 source recall, 6× CPU); fg-clip2 is worse; recall before
 the reranker is already ~0.9–0.96 so the end-to-end gain is bounded.
- Dedup: no near-duplicates in a 10% sample; re-check on the full pack. Luna cost: Batch API halves it ($0.20/1k); compact schema $0.24/1k.
- Heat: parallel local model inference overheated the laptop; heavy work moved to an ephemeral GPU machine, ~3.4 h.

### Decision model as stage 2 (Jev, TypeSafe AI, 2026-10-02) — eval/jev_report.txt
Stage 2 = one Noul question per (query, caption+description) to Jev (`app/decide.py`, RERANK_BACKEND=jev), MiniLM otherwise unchanged.
| stage 2 | thr | P@10 | R@pool | empty | junk/q | hit@1 | hit@10 | p50 ms |
|---|---|---|---|---|---|---|---|---|
| MiniLM (integrated) | 0.5 | 0.55 | 0.61 | 9 | 0.53 | 0.63 | 0.84 | 746 |
| Jev | 0.5 | **0.72** | 0.55 | 13 | **0.14** | 0.61 | 0.86 | 2468 |
| Jev | 0.2 | 0.63 | 0.67 | 6 | 0.30 | | | |
| Jev | 0.1 | 0.60 | 0.79 | 2 | 0.47 | | | |
| **Jev batched, 50 docs/call (chosen), thr 0.2** | 0.2 | 0.64 | 0.70 | 6 | 0.26 | **0.64** | **0.88** | **811** |
| Jev batched | 0.5 | 0.68 | 0.51 | 16 | 0.15 | 0.64 | 0.88 | 811 |
Batched judging keeps quality (same curve as single calls) and meets the latency target (p50 811 ms, p95 966 ms). Default now:
RERANK_BACKEND=jev, JEV_BATCH=50, REL_MIN 0.2 for Jev (0.5 for MiniLM). Total Jev spend for all evals ≈ $0.62.
Jev dominates MiniLM at every threshold (at 0.2 it beats MiniLM@0.5 on all four metrics). person_context P@10 0.43→0.79, scene 0.61→0.76,
attribute 0.58→0.86. Scores are decisive: 91% of pairs ≤ 0.1, 2.8% ≥ 0.5. Cost: ~20k calls, ≈$0.34 total (≈$0.0017 per query at 150
candidates; $0.042/M input tokens, no output tokens). Single-call latency was p50 2.5 s (150 calls at ~500 ms); batching 50 documents per call fixed it (latency is per call, not per token). OpenAI Decisions API (image-capable) is
the drop-in next backend once the account is enabled (403 on 2026-10-02).

### Review fixes (2026-10-02, after an independent architect review) — eval/v2_report.txt
Fixed: people linker required wrong persons from ordinary words ("goal"→Naresh Goyal, "leaders"→Leander Paes, 13/198 queries);
now only parse-named people (or whole directory names) link, never fuzzy single words; accents kept. RRF fusion before the 150 cap.
Two result tiers (match ≥ 0.5, possible ≥ 0.1). Parse runs concurrently with the first recall pass (700 ms box).
City alias tokens (Bangalore/Bengaluru…) in index and query. Query/click log. Human grading page (/grade, 300-pair pool).
Eval fixed: judge sees caption + date (v2 cache), caption names count for name queries (blind image-only check missed Kohli in action
shots), empty results count as P@10 = 0, known-item hit@k/MRR and 95% bootstrap intervals reported.
| | P@10 | R@pool | empty | junk/q | hit@1 | hit@10 | never retrieved |
|---|---|---|---|---|---|---|---|
| before fixes (Jev @0.2, v1 judge) | 0.64 | 0.70 | 6 | 0.26 | 0.64 | 0.88 | 10 |
| after fixes, both tiers (≥0.1), v2 judge | 0.78 [0.74–0.82] | 0.90 | 6 | 0.15 | **0.70** | **0.92** [0.88–0.96] | **7** |
| after fixes, top tier only (≥0.5) | 0.81 [0.75–0.86] | 0.68 | 25 | 0.03 | | | |
Per type (both tiers): place 0.79 (was 0.53), event_year 0.87 (was 0.42), hinglish 0.77, person 0.76. Zero-relevant queries in the pool 87→42.
Live latency unchanged (~0.8–1.2 s warm; eval p50 93 ms only because every Jev score was cached).

### Scale test: 33,759 photos (2026-10-02 evening) — eval/scale33k_report.txt
Full pack minus the largest files (30 GB cap): 33,759 photos indexed (Luna via Batch API $5.70 for 28,757 new photos, i.e. $0.20/1k).
Same 198 queries, same v2 judge.
| | P@10 both tiers | R@pool | empty | junk/q | P@10 top tier | hit@1 | hit@10 | never retrieved | p50 ms |
|---|---|---|---|---|---|---|---|---|---|
| 5,002 photos | 0.78 | 0.90 | 6 | 0.15 | 0.81 | 0.70 | 0.92 | 7 | ~800 |
| 33,759 photos | 0.79 [0.75–0.82] | 0.88 | 3 | 0.16 | 0.84 | 0.57 | 0.79 [0.72–0.85] | 17 | 878 (p95 1,240) |
Precision and junk held at 6.7× the data while showing twice as many results per query (17 vs 9). Known-item rank fell as expected:
the queries were written from one photo, and the bigger archive now holds several equally valid photos per query, so "the" source photo
ranks lower; 17 source photos never enter the 150 candidates (recall ceiling to investigate per source). Latency stays near the 1 s target.
People index at 33k: 78,970 faces, 2,886 clusters spanning 2+ photos (largest 221), 12,253 photos with a confident name (36%).

### Open
1. Human check of the judge: 2 journalists × 300 pairs at /grade (pool built); real query log → next eval set.
2. Reranker: Jev is stage 2; distillation into a local model (E11 plan) for vendor independence and latency (biggest precision lever). 2. Caption-aware Luna pass evaluated: no gain over the default once caption names are integrated (EXPERIMENTS.md E3); not adopted.
3. Learned fusion / dynamic cutoff (E7) not trained. Query parse is time-boxed at 700 ms (uncached parse fills the cache in the background).
4. Latency: 746 ms p50 with the 256-token reranker; keep under 1 s (consider 192 tokens or caching).

## 9. Sources

- AP AI photo library search: https://photoarchivenews.com/news/launched-ap-powers-up-ai-for-full-photo-library-search/
- Gemini API pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini 2.5 Flash-Lite pricing: https://crazyrouter.com/en/blog/gemini-2-5-flash-lite-pricing
- Qwen3-VL deployment costs: https://www.spheron.network/blog/deploy-vision-language-models-gpu-cloud/
- pgvector vs Qdrant vs Milvus 2026: https://dev.to/locionic/vector-databases-for-production-rag-2026-pinecone-vs-qdrant-vs-milvus-vs-pgvector-4fim
- Vector DB 100M benchmark: https://tensoria.fr/en/blog/vector-database-comparison
- SigLIP 2 paper: https://arxiv.org/pdf/2502.14786
- InsightFace model licensing: https://www.insightface.ai/solutions/face-recognition-licensing
- AuraFace (commercial face model): https://huggingface.co/blog/isidentical/auraface
- ResourceSpace AI Faces plugin: https://www.resourcespace.com/knowledge-base/plugins/faces
- Multimodal RAG in 2026: https://bigdataboutique.com/blog/multimodal-rag-retrieval-over-images-pdfs-and-text
- Cheapest vision LLM APIs 2026: https://www.clawrouters.com/blog/cheapest-vision-multimodal-llm-api-2026
- GPT-6 Luna pricing: https://openrouter.ai/openai/gpt-6-luna
- GPT-6 Sol and Luna release: https://www.requesty.ai/blog/gpt-6-sol-luna-pricing-release-api
