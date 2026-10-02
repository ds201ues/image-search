# Photo Desk Search (pilot)

Natural-language photo search for journalists. See PLAN.md for the why, AGENTS.md for working notes.

## How it works

```
image ──► thumbnail + source metadata (date, section) ──┐
 ──► SigLIP 2 image embedding (local CPU) ├─► Postgres + pgvector
 ──► faces: YuNet detect → AuraFace embed │
 ──► GPT-6 Luna JSON: description + recognised people ──┘
 people step: confident Luna names → image_persons; single-face photos teach the face
 gallery; faces then name the same person elsewhere

query ──► names + years parsed ──► vector search (SigLIP text ↔ image)
 ──► full-text on VLM description + recognised names
 ──► face search (images containing the named person)
 ──► reciprocal rank fusion, named-person images first
```

| Piece | Choice | Licence |
|---|---|---|
| Image/text embedding | `google/siglip2-base-patch16-256` | Apache-2.0 |
| Face detection | OpenCV YuNet | MIT |
| Face recognition | AuraFace v1 (`glintr100.onnx`) | Apache-2.0 |
| Face references | learned from Luna-named photos; optional Wikidata seed | — |
| Description | `gpt-6-luna` via OpenAI API (swappable in `app/vlm.py`) | API |
| Store | Postgres 17 + pgvector (Docker) | — |

## Run it

```bash
cp .env.example .env # add OPENAI_API_KEY
docker compose up -d # Postgres on :5433
uv sync

uv run python -m app.ingest --dir data/sample # register, embed, faces, vlm, people
uv run python -m app.gallery --no-bulk # optional: reference faces from Wikidata (slow, rate-limited)
uv run uvicorn app.api:app --port 8000 # open http://localhost:8000
uv run python -m eval.run_eval --misses # quality + latency report
```

Without an API key, run `--steps register,embed,faces,people`. Search then works from embeddings (and gallery faces if seeded).

Ingest steps are idempotent; re-run any step on its own with `--steps`. To redo descriptions: `UPDATE images SET vlm = NULL`.

## Layout

```
app/config.py settings (.env)
app/db.py schema, search_text builder
app/embed.py SigLIP 2
app/faces.py YuNet + AuraFace
app/vlm.py prompt + JSON schema for the vision model
app/gallery.py Wikidata face gallery seeder
app/ingest.py pipeline CLI with per-step timings
app/search.py hybrid search
app/api.py FastAPI + static UI (app/static/index.html)
eval/ 51 test queries (name / name+context / scene) + scorer, --use for ablations
scripts/ web sample scraper
```
