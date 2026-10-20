# image-search — working notes

Read PLAN.md first (architecture, decisions, costs). EXPERIMENTS.md is the measured adopt/drop log.

## Facts
- Users are journalists. They type sentences, not keywords.
- Dates come only from file metadata / EXIF. Never let a model infer dates.
- VLM is `gpt-6-luna` via OpenAI API (key in `.env`). A self-hosted candidate (Qwen3-VL) must be benchmarked on `eval/`, not assumed.
- Stage 2 relevance is pluggable (`app/decide.py`): `jev` (batched 50 docs/call, budget cap) or `minilm` (local). Thresholds live at the top of `app/search.py` and `app/ingest.py`. Change them only with `eval/run_eval.py --sweep` evidence.
- Do not fix a bad query with a special-case rule. Find which stage failed (recall, people evidence, reranker) using `--show-misses`, and fix that stage.
- Captions (IPTC) are evidence, like the Luna description, not ground truth.
- This public tree does not include archive photos, thumbnails, embeddings, or judged caches. Point ingest at your own folder.

## Rules
- Keep it simple: one Postgres box, one Python service, no extra infra until measurements demand it.
- Every model or threshold change must be re-measured with `eval/run_eval.py` (judged P@10, recall, junk, latency).
- Keep the vector store behind an interface so pgvector can be swapped later.
- Update PLAN.md when a decision changes.

## Environment
- Some TLS-intercepting corporate proxies break Python `urllib` SSL. Use `requests` for downloads.
- Docker Desktop (or equivalent) must be running for Postgres (`docker compose up -d`, port 5433).
- Heavy model work (bulk embedding, reranker scoring) belongs on a GPU machine, not a heat-limited laptop.
- AuraFace ONNX is exported with batch size 1; embed faces one at a time.
