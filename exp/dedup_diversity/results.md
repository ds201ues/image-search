# E8 dedup_diversity — results (2026-10-02)

## 1. Near-duplicates: almost none in this sample
Archive sample = 5,002 files, a stratified **10% random sample** of the larger archive. Bursts are split across the other 90%,
so duplicate counts here are a lower bound on the full pack.

| method / threshold | pairs | groups ≥2 | images in groups | % of archive | VLM calls saved |
|---|---|---|---|---|---|
| phash ≤ 6 (true near-identical) | 0 | 0 | 0 | 0.0% | 0.0% |
| phash ≤ 10 | 1 | 1 | 2 | 0.0% | 0.0% |
| dhash ≤ 8 | 6 | 3 | 9 | 0.2% | 0.1% |
| SigLIP cos ≥ 0.98 | 0 | 0 | 0 | 0.0% | 0.0% |
| **SigLIP cos ≥ 0.96** (same shoot, near-identical frame) | 3 | 3 | 6 | 0.1% | 0.1% |
| SigLIP cos ≥ 0.93 (same shoot/event, different frame) | 48 | 32 | 78 | 1.6% | 0.9% |
| SigLIP cos ≥ 0.90 (same event or same scene type) | 481 | 110 | 434 | 8.7% | — (not duplicates) |

Eyeballed (exp/dedup_diversity/contact_sheet.html, band_*.png):
- ≥ 0.96: actress at a press meet twice, Ganesh procession twice, woman with idol twice. True same-shoot pairs.
- 0.93–0.96: same football match, same cricket match, same actor shoot, same audience. Same shoot, visibly different frames. These are **shoot siblings**, not duplicates.
- 0.90–0.93: flower market vs flower market, two different weddings, golfers. Same scene type. Must not be collapsed.
- phash/dhash found nothing SigLIP didn't; at ≤ 10–12 they start matching unrelated photos. Perceptual hashes add nothing here.

Chosen definition stored in `exp_dedup_diversity_groups(image_id, group_id, is_rep)`: phash ≤ 6 OR SigLIP ≥ 0.96 → 3 groups.
Representative = captioned member first, then lowest id.

## 2. Shoots (`exp_dedup_diversity_shoots(image_id, shoot_id)`)
shoot = connected components of (same date AND same byline) OR near-dup group.

| | value |
|---|---|
| shoots | 4,908 (4,840 singletons) |
| shoots with ≥ 2 photos | 68, covering 162 photos |
| size histogram | 2: 55, 3: 7, 4: 3, 6: 2, 7: 1 |
| uncaptioned photos | 1,578 |
| uncaptioned photos with a captioned sibling in their shoot | **1** |

So in this sample, shoot grouping cannot help the 31% uncaptioned photos: those photos' siblings are not in the sample
(or they have no byline/date to group on). On the full larger archive the same date+byline key will group ~10× more.

## 3. Index-cost saving from skipping the VLM on duplicates
0.1% at the strict threshold (3 calls out of 5,002); 0.9% if we accepted shoot siblings (0.93), which is unsafe:
the siblings differ in who is speaking / batting / framed, so copying a description would be wrong for people queries.
**Not worth building for this sample.** Re-measure on the full pack; expect a few percent, still minor next to the $0.35/1k price.

## 4. Result diversity (collapse siblings in top-10): not measurable on this sample
Baseline eval (198 queries, eval/baseline_report.txt): ALL P@10 0.59, junk/q 0.36. Collapsing shoot siblings
(`exp_dedup_diversity_shoots`, 68 multi-photo shoots) changed the top-10 of **0 of 198 queries**: duplicate pairs in
top-10 were already 0.00 on average. With only 3 strict near-dup groups the strict variant cannot differ either
(run was stopped). Diversity is a non-issue at 10% sampling; it only becomes real on the full pack where bursts are intact.

Thresholds to use on the full larger archive (from the eyeballed bands):
- **collapse in results**: SigLIP cos ≥ 0.96 → "N similar" under one card. Safe; same shoot, near-identical frame.
- **shoot grouping for evidence sharing**: SigLIP ≥ 0.93 OR (same date AND same byline). Share caption text/place/event down-weighted; never share people names without a face match.
- never collapse at ≤ 0.93: those are merely the same scene type.
- perceptual hashes: skip; they found nothing SigLIP missed.

## Recommendation
**Drop dedup/diversity for the 5k sample** (0.1% duplicates, 0 queries affected, VLM-skip saving 0.1%). Re-measure on the larger archive
with the thresholds above; expect a few % duplicates there, still minor at $0.35/1k. Keep `exp_dedup_diversity_shoots` for E1/E2
(same date+byline key is the part that scales).

Integration if the full pack warrants it: `app/ingest.py` corpus job after `embed` builds `images.shoot_id` (LATERAL top-10 on clip ≥ 0.93 + date/byline union-find, as in `_clusters`);
`app/search.py` after the relevance sort keeps the first result per shoot/near-dup group in the top-10 and attaches the rest as `similar_ids`.
Artifacts: exp/dedup_diversity/{dedup.py, diversity_eval.py, contact_sheet.html, band_*.png, stats.json, diversity_shoots.txt}.
