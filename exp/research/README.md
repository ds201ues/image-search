# Research notes (2026-10-02)

Three parallel research passes fed EXPERIMENTS.md. Condensed here; full reasoning in the session.

## 1. Kaggle-style ideas that transfer (ranked by expected gain / effort)
1. **Caption as evidence everywhere** (search text, description vector, reranker input, people). Found in code: `images.caption` is stored but nothing in search reads it. Risk: captions name off-frame people, so caption names are a prior, not proof. → E1, E2, E3
2. **Distil the LLM judge into the cross-encoder**: train MiniLM/Ettin on (query, doc, grade) from eval/judgments.jsonl + Luna-written synthetic queries, hard negatives = our own false positives. Rank-DistiLLM (arxiv 2405.07920), Eedi 1st, LLM Science Exam 1st (H2O). Split by query (GroupKFold). → E12
3. **LightGBM LambdaRank over stacked features** instead of relevance = P(people) × P(context): Shopee 2nd/3rd, H&M, LECR. 10–20 features, monotone constraints, min_data_in_leaf ≥ 50. → E7
4. **Shoot grouping / evidence sharing**: same date + byline or SigLIP cos > 0.92 → siblings share caption text (down-weighted) and names; GLR 2019 1st: label-aware re-ranking +4.4 mAP while plain DBA/QE gave −0.2. → E2/E8
5. **Dynamic per-query cutoff**: keep if rel ≥ max(FLOOR, ratio × top), always top-k above FLOOR (Eedi 1st, Shopee min2). → E7
6. **RRF before the MAX_RERANK cap**: today candidates are cut in source order (people, SigLIP, desc, text), so full-text is dropped first. → E7 (quick fix)
7. **Caption-seeded face naming** ("Names and Faces in the News", Berg 2004; Happywhale pseudo-label rounds): one name + one dominant face = seed; 2 rounds. → E2
8. **Junk bank penalty**: similarity to known graphics/collages (GLR 2020 1st). Small on real archive. → optional
9. **Hubness correction (QB-Norm)** on SigLIP text→image scores using a bank of synthetic queries. → optional, measure source recall@50
10. **doc2query**: Luna writes 3–5 likely journalist queries per photo at ingest, indexed in full-text + desc vector only. → E3 add-on
11. Stronger / dual text embedder (bge-base, gte-base). → E5
12. **Query-grouped CV, frozen held-out queries, bootstrap CIs** before trusting any win; ±3–5 P@10 noise at 100 queries. → eval prerequisite
13. Judge disagreements, not random pairs, to grow labels cheaply. → E12
Skip: k-reciprocal, αQE/DBA on image vectors, ArcFace/GeM fine-tuning (need instance labels).

## 2. Image-text embedding models (CPU, commercial licence)
No Apache model beats siglip2-base by much at similar cost; upgrades cost 3–9× per image for +1.5–4 COCO R@1.
Shortlist: **qihoo360/fg-clip2-base** (same cost, 196-token text, big gain on long captions, Apache, trust_remote_code),
**google/siglip2-so400m-patch16-256** (+2.2 COCO, +7.4 multilingual; ~250–300 ms/img, dim 1152),
**facebook/PE-Core-L14-336** (highest COCO 57.1 but 32-token text, ~400–500 ms/img).
Excluded: MetaCLIP 2 (CC-BY-NC), MobileCLIP2 (Apple research licence), jina-clip (CC-BY-NC), Qwen3-VL-Embedding (too slow on CPU).
Storage at 10M: 768-d halfvec ≈ 15 GB; use int8/binary + rescore rather than Matryoshka (not supported by these models).

## 3. Rerankers and text embedders (CPU)
0.5B+ rerankers (bge-v2-m3, mxbai-v2, Qwen3-Reranker, jina-v3) are 20–40× slower than MiniLM-L6: seconds per query on CPU. Offline teachers only.
Best fit: **cross-encoder/ettin-reranker-17m-v1 / 32m-v1 / 68m-v1** (May 2026, Apache, distilled from mxbai-rerank-large-v2): 17m faster than MiniLM with +0.05 NDCG@10, 32m +0.07 (above bge-v2-m3). Raw logits → fit sigmoid(a·s+b) on judgments, re-sweep threshold. Cascade option: 17m on 150 → 68m on top 30.
Also: Alibaba-NLP/gte-reranker-modernbert-base (3 s, offline ceiling). Hosted: Voyage rerank-3-lite ~$0.8 per 1k queries.
Embedders: **ibm-granite/granite-embedding-small-english-r2** (384-d drop-in, claims +5 BEIR), **Qwen/Qwen3-Embedding-0.6B** truncated to 384 (quality ceiling, ~20× encode cost, knows names).
LLM as live reranker: +0.9 s p50 at best in production (Intercom); use as judge/teacher, not live.

## 4. Commercial / closed alternatives (2026-10-03)
No vendor publishes accuracy for news-archive photo search with short named queries. Cloud vectors: Gemini Embedding 2 (MSCOCO R@1 62.9 self-reported;
independent image→image R@1 43.8 vs SigLIP2-so400m 43.4), Vertex multimodalembedding@001 (retires 2027-04), Amazon Nova MME (US-East only),
Azure AI Vision (retires 2028-09), Cohere Embed 5 (self-hostable), Voyage multimodal-3.5, Jina v4 (non-commercial). Index cost $0.06–0.35 per 1k photos,
queries ≈ $0; none names people or reads captions. People: Google Vision celebrity recognition shut down 2025-09; Azure face gated; AWS Rekognition
$1/1k calls, celebrity coverage of Indian figures unpublished, documented lookalike errors, trains on customer photos unless opted out.
Turnkey DAMs (MerlinOne NOMAD/AP, Orange Logic, Bynder=Bedrock+Rekognition, Canto, PhotoShelter, Adobe AEM, Fotoware, Perfect Memory, Veritone, Dalet):
no published precision/recall; one independent NOMAD review found named-person searches failing; prices five figures/year up. Direct benchmark plan:
Gemini Embedding 2 (~$0.5) + Voyage/Nova (~$0.5) + Rekognition celebrities+labels on 5k ($10, opt-out first) on our 198 queries, pooled judging ~$2.
