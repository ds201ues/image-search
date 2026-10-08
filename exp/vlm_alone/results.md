# E21 vlm_alone — a Luna name guess alone should not make a confirmed tag (2026-10-08)

**Trigger.** User reported "sanjay dutt in sunglasses": R. Madhavan tagged Sanjay Dutt + Manyata Dutt (B6DCE6E6), and Govinda
tagged Sanjay Dutt in a press-conference photo whose caption names five other people (9014D32D).

**Root cause.** Both tags have one evidence row, `vlm:0.97` / `vlm:0.92`, from `recognized_people`. Noisy-OR over one source is that
number, above PERSON_SURE 0.8, so the UI shows a solid chip. Face pipeline re-run: faces detected cleanly, none within 0.34 of the
Sanjay Dutt references (vote threshold 0.40), none near anyone in the gallery; the gallery has 2 refs for Sanjay Dutt and 0 for
Salman Khan, Govinda, Manyata Dutt, Madhavan, so faces cannot veto film people. Luna re-run on B6DCE6E6: run 1 "Suniel Shetty 0.91 +
Mana Shetty 0.76", run 2 "Sanjay Dutt 0.93 + Manyata Dutt 0.82". On 9014D32D: Sanjay Dutt 0.96 / 0.91 both runs (consistently wrong).
Its confidence number is not calibrated. PLAN section 10's pilot rule ("≥0.8 kept") came from 100 images and 9/10 correct.

**Scale.** 21,008 confirmed tags; 1,878 (9%) rest on a Luna guess alone; 564 of those have detected faces and a person with ≥3 refs
yet no face matched. Top Luna-only names: Modi 58, Rahul Gandhi 45, Dhoni 44, Tendulkar 42, Sonia Gandhi 33, Kohli 30.

**Variants** (views `exp_vlm_alone_scores_cap75|cap45`, live tables untouched; `build.py`):
- cap75: a (image, person) whose only source is `vlm` is capped at 0.75 → "possible" chip, still recalled, ranks lower.
- cap45: capped at 0.45 → below PERSON_MIN, not a person hit at all. Loses 1,878 person hits.

## Results
Sample queries (`measure.py`), strong/total:
| query | base | cap75 | cap45 | moved out of Best under cap75 |
|---|---|---|---|---|
| sanjay dutt in sunglasses | 5/10 | 3/9 | 4/5 | 2: both reported photos |
| sanjay dutt | 15/20 | 13/20 | 13/16 | 2: both reported photos |
| rahul gandhi smiling | 27/60 | 23/60 | 22/53 | 4 |
| shah rukh khan | 46/60 | 41/60 | 39/46 | 5 |
| ranbir kapoor | 12/32 | 9/32 | 9/24 | 3 |
| dhoni / kohli x2 / modi pink / sam pitroda / priyanka / sanjay dutt angry | unchanged | | | 0 |

Hand check of the 12 demoted photos (`out/moved_12.jpg`): ~8 wrong tags (Sanjay Dutt tagged Rahul Gandhi, Rajasthani villagers
tagged Rahul Gandhi, Tiger Shroff tagged Shah Rukh Khan, 2000s award photo tagged Ranbir Kapoor), ~4 plausibly right. One demoted
SRK photo (66C28BA3) had cluster evidence too and moved for ranking reasons, not the cap.
Random 24 Luna-only tags where a face match was possible (`out/suspicious_24.jpg`): ~5-6 wrong (Stalin tagged Pinarayi Vijayan,
a Sikh gentleman tagged Manmohan Singh, Modi on a projection screen, a cartoon, photo-in-photo Kalam). Random 24 Luna-only overall
(`out/luna_only_24.jpg`): ~3-4 doubtful. Estimated error of Luna-only tags: 15-25%, i.e. 300-450 wrong confirmed tags in 33,759 photos.

Judged eval, people types (person, person_context, real; 58 queries; `eval_people.log`, `eval/compare.py`):
| | P@10 both tiers | R@pool | nDCG | junk/q | top tier P@10 |
|---|---|---|---|---|---|
| base | 0.72 | 0.85 | 0.75 | 0.45 | 0.78 |
| cap75 | 0.72 | 0.85 | 0.76 | 0.45 | 0.78 |
| cap45 | 0.73 | 0.82 | 0.73 | 0.47 | 0.78 |
Paired: no clear change for either (cap75 top-tier nDCG −0.012 [−0.023, −0.002], tiny). The judge is the same Luna model; blind-naming
catches some of these (it names only Salman Khan in 9014D32D, and correctly Sanjay Dutt in 32A2922B which the index tagged Rahul Gandhi)
but shares others (eval/judge.py caveat), and the wrong tags are 1-2 photos per query pool, so the aggregate cannot move. The hand-checked
sheets are the evidence here, not the judged numbers.

## Recommendation
ADOPT cap75 in `app/ingest.py` / the `image_person_scores` view: a Luna-only belief is capped at 0.75. Zero recall loss on the judged set,
the reported photos and most demoted tags are wrong ones, cost nothing, chips become dashed "Name?" so reporters verify.
DROP cap45: loses 1,878 person hits and recall for no precision gain the judge can see.
Follow-ups: (1) seed `person_refs` from Luna guesses that a caption confirms, so film people get references and faces can vote/veto;
(2) treat "faces present, person has ≥3 refs, best sim < 0.3" as mild negative evidence. Both need the same hand-check protocol, not the judge.
