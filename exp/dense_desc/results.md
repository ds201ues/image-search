# E20 dense_desc — dense index-time "what is visible" text for stage 2 (2026-10-07)

Hypothesis: newsroom misses are "right topic, not the exact scene"; a literal 5-8 sentence description of what is visible
(actions by role, counts, objects, road/water/crowd condition, light, readable text) appended to the stage-2 document gives the
text-only decision model (Jev) the specifics it lacks. Zero query-time cost.

Scope: the 3,702 photos that enter stage 2 for the 40 newsroom queries (recall untouched). Luna, reasoning low, 768px thumb,
$0.48 total ($0.13 per 1k photos). Text quality is good (literal, no names, no event guesses). Table exp_dense_images.

| newsroom, 40 queries | P@10 both tiers | R@pool | junk/q | P@10 top tier | R@pool top | shown top |
|---|---|---|---|---|---|---|
| live (newsroom_v1) | 0.757 | 0.968 | 0.15 | 0.844 | 0.561 | 11.0 |
| + details (newsroom_dense) | 0.764 | 0.929 | 0.12 | 0.837 | 0.506 | 9.4 |
Paired: P@10 +0.007 [-0.011, +0.025]; R@pool -0.039 [-0.103, +0.004]; top-tier recall -0.055 [-0.119, +0.002]. No clear change;
recall leans down (Jev reads a longer document and scores slightly fewer photos ≥ 0.5). Latency unchanged.

Decision: DROP. The specific-scene loss is not a missing-text problem; the judge and Jev read the same kind of text and still
disagree on "queue" vs "no queue". The remaining lever is a stage 2 that sees the image, which is out of scope for now (latency).
