# E6 query_understanding — results (2026-10-02)

**Recommendation: adopt the parse for people names, years and Hinglish rewriting; adopt city as a boost, never a hard filter;
do NOT put a 2 s Luna call on the live path — cache it and move the parse to a small/fast model (or a local rule layer) before shipping.**

## What was tested
GPT-6 Luna (text only, strict JSON) parses the query into people (canonical names), city, year range, event, visual attributes,
an English `search_text` (Hinglish/nicknames/misspellings normalised, year and city removed) and 3–6 expansion terms.
`search_q.py` wraps `app.search.search` without editing it: normalised text goes to all recall sources and the reranker,
years become the date filter, city either restricts the index (filter) or adds +0.15 relevance to hits inside the city (boost).
Live tables, same judged eval as `eval/run_eval.py` (198 queries: 160 generated + 38 hand).

## Numbers (threshold 0.20, 5,319 judged pairs, +$0.13 new judging)

| variant | P@10 | R@pool | empty | junk/q | p50 ms* | p95 ms* |
|---|---|---|---|---|---|---|
| baseline | 0.592 | 0.630 | 22 | 0.36 | 489 | 2151 |
| parsed | 0.576 | **0.689** | **15** | 0.41 | 474 | 1852 |
| parsed + expansions (desc vector only) | 0.578 | 0.688 | 15 | 0.39 | 483 | 1955 |
| parsed + city boost | 0.576 | 0.689 | 15 | 0.42 | 528 | 2087 |
| parsed + city filter | 0.598 | 0.654 | 22 | 0.34 | 460 | 1938 |

\* search latency only, excluding the parse call. **Parse call: p50 2,173 ms, p95 3,457 ms, max 5,012 ms** (201 queries, Luna reasoning=low).
Parser cost ≈ $0.00013 per query (~$0.13 per 1k queries).

Per type, baseline → parsed (P@10 / R@pool / shown per query):

| type | n | baseline | parsed | city_filter |
|---|---|---|---|---|
| event_year | 27 | 0.38 / **0.00** / 0.2 | 0.50 / **0.50** / 0.4 | 0.40 / 0.00 / 0.2 |
| hinglish | 16 | 0.71 / **0.15** / 0.5 | 0.52 / **0.45** / 3.4 | 0.62 / 0.45 / 3.1 |
| place | 23 | 0.64 / 0.37 / 5.5 | 0.61 / **0.52** / 8.5 | **0.77** / 0.36 / 5.8 |
| person | 19 | 0.64 / 0.64 | 0.67 / 0.64 | 0.67 / 0.64 |
| person_context | 29 | 0.37 / 0.57 | 0.31 / 0.53 | 0.32 / 0.53 |
| scene | 65 | 0.65 / 0.62 | 0.65 / 0.69 | 0.65 / 0.68 |

Reading: the baseline's high P@10 on event_year/hinglish is an artefact of showing almost nothing (0.2–0.5 results per query);
the parse turns empty answers into real result lists (empty 22 → 15, recall +6 points overall, +50 on event_year, +30 on hinglish).
P@10 moves within noise (±3–5 points at this query count) except where the parse over-rewrites (below).

## Why each variant behaved as it did
- **Years**: `app.search` already strips a 4-digit year, but the baseline then still searches "old delhi traffic" against a *date-filtered*
 index. The gain comes from the parse removing the city word and rewriting, which lets the reranker score the real context.
 Note the baseline's event_year recall of 0.00 is partly an eval effect: the judge marks photos outside the year as 0, and the baseline
 showed 0.2 results per query.
- **City filter** raises P@10 on place (0.64 → 0.77, junk 0.17 → 0.00) but kills recall whenever the relevant photo has no city in IPTC.
 Only 62% of photos carry a city; e.g. `srinagar disinfection drive 2020`: the relevant photo has no city field, filter → 4 results, none relevant.
 Also city aliases are messy ("bangalore bengaluru", "city names from news copy", "india; maharashtra; kolhapur").
- **City boost** (+0.15 inside the city) changed nothing measurable vs plain parsed: the reranker already ranks those photos; the boost
 only reorders within the shown set. Cheap to keep, no evidence it helps at this size.
- **Expansions** in the description vector: no effect (0.689 vs 0.688). Expansions would need to reach full-text to matter; not worth it yet.
- **People**: the parse's canonical names ("mamta" → Mamata Banerjee, "kohli" → Virat Kohli) only help if the persons table has that
 person; on this corpus the live linker already handled the hand queries, so person P@10 is flat (0.64 → 0.67).

## Five queries that changed and why
1. `actor car se utarte hue` → "actor getting out of car": P@10 0.00 → 0.90. Pure Hinglish translation; nothing in the index matched the Hindi words.
2. `old delhi traffic 2023` → "old traffic" + year 2023 + city delhi: 0.00 → 1.00. The year filter alone was not enough; removing "delhi" let the reranker see "traffic". (The rewrite "old traffic" is a parser error, "Old Delhi" is a place; it still worked.)
3. `film premiere in mumbai` → "film premiere": 0.00 → 0.57. With "mumbai" in the text, the reranker penalised descriptions that never say Mumbai.
4. `chandigarh capitol complex at night` with city **filter**: 1.00 → 0.00. The relevant photo's IPTC city is not "chandigarh". Filter risk in one line.
5. `model walking ramp in aslam khan outfit` → "...runway in Aslam Khan outfit": 1.00 → 0.50. Over-rewriting: "ramp" is the Indian term used in the Luna descriptions; "runway" is not. Same for `sportsaid athletes with duchess of cambridge 2016` → "Catherine, Princess of Wales" (+1.00 here, but the opposite can happen when the description uses the old name).

## Risks
- **Latency**: +2.2 s p50 on the live path is unacceptable against the < 1 s target. Options: (a) cache parses (journalist queries repeat), (b) a faster/smaller model
 (gpt-6-luna without reasoning, or a nano-tier model) to be measured, (c) rule layer for the cheap parts (year regex exists; a Hinglish/nickname dictionary + city alias table
 cover most hand cases) and call the LLM only when the rule layer finds Hindi words or unknown tokens, (d) progressive UI: show baseline results instantly, swap in parsed results when the parse returns.
- **Over-rewriting**: the parser replaces domain words ("ramp"→"runway", "Old Delhi"→"old") and modern names for historical ones. Mitigation: search the *union* of original and rewritten text (two recall passes, rerank once), or instruct the parser to keep the original wording and only translate non-English tokens.
- **Wrong or missing city**: never filter; at most boost. City metadata is 62% covered and inconsistently spelled.
- **Judge bias**: the judge only sees the image, so name canonicalisation is checked blind; year checks rely on the stored date (94% coverage).

## Integration steps (if adopted)
1. `app/search.py`: accept an optional pre-parsed intent `{people, year_from, year_to, search_text, city}`; when present use `search_text` as `ctx`, pass years, link `people` names through `People.link`, and run recall on both original and rewritten text (union).
2. Add a `city` recall source (+boost) using an alias table (`exp/query_understanding/parser.py: CITY_ALIASES`); normalise `images.city` at register time with the same table.
3. Parser service: `exp/query_understanding/parser.py` as is, with a persistent cache table `query_parses(query, parsed, model, ms)`; try a cheaper model and `reasoning_effort` off; measure p50/p95 again.
4. UI: fast results first, parsed results replace them when ready (the parse is independent of the first search, so start both at once).
5. Re-sweep `REL_MIN` after integration (junk/q rose 0.36 → 0.41 because more results are shown).

Files: `parser.py` (Luna parser + cache `parsed.jsonl`, 201 queries), `search_q.py` (wrapper, modes parsed / city_boost / city_filter), `eval_q.py` (judged eval), `eval_output.md` (full tables), `per_query.json`.
