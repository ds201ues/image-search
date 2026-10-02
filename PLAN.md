# Image Search (DAM) for a newsroom photo archive — Project Plan

Last updated: 2026-10-01
Status: pilot built and running on a 100-image web sample (see section 10 for results). larger archive not received yet.

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
| Data | Images only. No captions, no IPTC text, no journalist query logs. Images come from an internal API and will be downloaded to a folder. |
| Dates | Must come from API metadata / EXIF. Stored as a column and used as a filter. Never inferred by a model. |
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
 link ─► people directory (names, aliases, fuzzy); ordinary description words never link
 recall ─► SigLIP + description vector + full-text + people, ~50 each, union
 precision ─► cross-encoder (ms-marco-MiniLM-L6, CPU) reads (query, description) → P(context) 0..1
 relevance ─► P(people in photo) × P(context); show if P(people) ≥ 0.5, or P(context) ≥ 0.2 when no one is named
```
Uncertain names are shown as "possible" (0.5–0.8) instead of being dropped.

### Evaluation v2
- Queries: GPT-6 Luna writes journalist-style queries from random photos (`eval/make_queries.py`), plus
 `eval/hand_queries.txt` (ours and the user's reports).
- Labels: pool top-20 from five system variants, GPT-6 Luna grades each (query, photo) 0/1/2, cached.
- Names are judged blind: the judge lists who it recognises without seeing the query. When primed
 ("is Kohli here?") it found Kohli in a pilot's cockpit.
- Metrics: P@10, recall against the pool, empty results, junk per query, latency. Threshold sweeps run offline.

### Results on 100 images, 25 hand queries (before the 1,100-image run)
| | P@10 | R@pool | junk/query |
|---|---|---|---|
| Any single retriever alone | 0.50–0.66 | 0.88–1.00 | 1.8–3.7 |
| v2 full system | 0.89 | 0.83 | 0.08 |
Remaining misses are reranker recall (stadium, airport) — next: compare a larger reranker on the 1,100 set.

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
