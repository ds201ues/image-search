# E4 image_embedding — results (in progress, 2026-10-02)

Question: does a stronger CPU-feasible image-text model beat `google/siglip2-base-patch16-256` on our eval?

## Setup
- Candidates (from exp/research/README.md §2): `qihoo360/fg-clip2-base` (256 patches, short walk), `google/siglip2-so400m-patch16-256`,
 `facebook/PE-Core-L14-336` (open_clip). `siglip2_base` re-embedded on the GPU for comparable timings.
- Embeddings computed on a GPU (fp16) with `gpu_embed.py` from the 768px thumbnails; loaded into `exp_image_embedding_<model>`.
 Query vectors for all 263 query strings precomputed the same way, so no local model inference was needed.
- Measurement: (a) vector source alone, recall@50 against judged grade-2 photos; (b) full system with the vector source swapped
 (view over `images` with the candidate vector as `clip`, `app.search.TABLES["images"]` + `embed_text` swapped), judged pairs appended to the shared cache.
- Local CPU notes: fg-clip2-base measured 164 ms/image and 89 ms/query on the laptop at 5 threads before local inference was halted
 (siglip2-base: 45 / 18 ms). transformers 5.18 needs the fg-clip2 text buffers rebuilt after load (see `models.py`).

## Baseline (siglip2-base in the live index), eval/baseline_report.txt
| type | n | P@10 | R@pool | empty | junk/q | shown |
|---|---|---|---|---|---|---|
| person | 19 | 0.64 | 0.64 | 2 | 0.74 | 4.9 |
| person_context | 29 | 0.37 | 0.57 | 5 | 1.03 | 2.5 |
| scene | 65 | 0.65 | 0.62 | 4 | 0.18 | 12.2 |
| attribute | 19 | 0.62 | 0.80 | 1 | 0.47 | 23.9 |
| place | 23 | 0.64 | 0.29 | 5 | 0.17 | 5.5 |
| event_year | 27 | 0.38 | 0.00 | 2 | 0.07 | 0.2 |
| hinglish | 16 | 0.71 | 0.09 | 5 | 0.00 | 0.5 |
| **ALL** | 198 | **0.59** | **0.55** | 24 | 0.36 | 7.8 |

SigLIP-only variant: P@10 0.41, R@pool 0.85, junk/q 2.71. Latency p50 934 ms / p95 1498 ms — measured while the laptop was
thermally throttled with other jobs running; not representative.

## Model comparison
(filled in when the GPU run and measurements finish)

| model | dim | GPU ms/img | GPU ms/query | source R@50 ALL | len≤3 | len>3 | hinglish | full P@10 | full R@pool | junk/q |
|---|---|---|---|---|---|---|---|---|---|---|
| siglip2_base | 768 | | | | | | | 0.59 | 0.55 | 0.36 |
| fgclip2_base | 768 | | | | | | | | | |
| siglip2_so400m | 1152 | | | | | | | | | |
| pe_core_l | 1024 | | | | | | | | | |

## Storage at 10M (pgvector halfvec)
768-d ≈ 15 GB, 1024-d ≈ 20 GB, 1152-d ≈ 23 GB. None of the candidates support Matryoshka truncation; use int8/binary + rescore if needed.

## Examples
(5 queries that changed)

## Recommendation
(pending)
