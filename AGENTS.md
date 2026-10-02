# image-search — agent notes

Read PLAN.md first. It holds the agreed architecture, decisions, costs, phases and open questions.

## Project facts
- Owner: a newsroom photo archive. Users are journalists searching a photo archive with natural-language sentences.
- Architecture v2 (PLAN.md section 11): index keeps evidence (Luna description + name guesses, SigLIP image vector, bge-small description vector, faces); corpus job clusters faces and lets photos vote on names; query = people linking + generous recall (4 sources) + cross-encoder reranker; relevance = P(people) × P(context). Postgres + pgvector, FastAPI, one static HTML page.
- Dates come only from API metadata / EXIF. Never let a model infer dates.
- VLM is `gpt-6-luna` via OpenAI API (key in .env). Production self-hosted candidate (Qwen3-VL) must be benchmarked on eval/, not assumed.
- Thresholds live at the top of `app/search.py` and `app/ingest.py`. Change them only with `eval/run_eval.py --sweep` evidence.
- Do not fix a bad query with a special-case rule. Find which stage failed (recall, people evidence, reranker) using `--show-misses`, and fix that stage.
- People confidence rules and thresholds live in `app/ingest.py` (step_people) and `app/config.py`. Results and rationale in PLAN.md section 10.
- Target: search < 1 s, cheapest workable system, 50k sample now, ~10M lifetime.
- Challenge data: `photos/` (5,002 files, stratified 10% of the larger archive, sampled from the archive),
 symlinked as `data/archive`. Indexed in Postgres database `imgsearch` (DATABASE_URL in .env); the old 1k web sample lives in db `imgsearch`.
- Archive files DO carry captions: IPTC caption/headline/keywords/city/byline (~68% have a caption). `app/meta.py` reads them;
 they are stored on `images` (caption, headline, keywords, city, ...). Captions are evidence, like the Luna description, not ground truth.
- Thumbnails are 768px (`data/thumbs/<id>.jpg`) and are what SigLIP, the VLM and the judge read; originals are only decoded once at register.
- Experiments: see EXPERIMENTS.md for the ground rules (own `exp/<name>/` folder, `exp_<name>_*` tables, swap `app.search.TABLES`).

## Working rules
- Keep it simple: one Postgres box, one Python service, no extra infra until measurements demand it.
- Every model or threshold change must be re-measured with `eval/run_eval.py` (judged P@10, recall, junk, latency).
- Keep the vector store behind an interface so pgvector can be swapped for Qdrant later.
- Update PLAN.md when a decision changes or an open question gets answered.

## Sample data (stand-in until the larger archive arrives)
- `data/sample/` holds 100 lead photos scraped from public news pages on 2026-10-01 via a sample downloader (re-runnable, skips already-downloaded articles, 1 req/s).
- `metadata.jsonl` fields: image_id, file, width, height, date_published, section, source_url, image_url, article_id, ref_title, ref_description.
- `ref_title` / `ref_description` are article headlines, kept ONLY to help write test queries and judge results. Never index them (the web sample has no IPTC captions).
- Roughly 15–20% are graphics, logos or collages (web lead images), unlike raw archive photos. Real archive may differ; re-check when it lands.

## Environment gotchas
- A TLS-intercepting corporate proxy breaks Python `urllib` SSL. Use `requests` for downloads.
- Wikimedia/Wikidata rate-limit bursts (HTTP 429) and the SPARQL endpoint times out on broad queries. Use `polite_get` in app/gallery.py and standard thumb widths (500).
- Docker Desktop must be running for Postgres (`docker compose up -d`, port 5433).
- Laptop CPU is a shared, heat-limited resource (10 cores, 16 GB). Heavy model work (bulk embedding, reranker scoring, training) goes to an
 ephemeral GPU machine (a GPU machine, ephemeral GPU machine.
 Terminate it when the batch is done.
- AuraFace ONNX is exported with batch size 1; embed faces one at a time.
