"""E21 vlm_alone: a person tag whose only evidence is Luna's recognized_people guess is capped below the confirmed line.
Views only; live tables untouched. cap75 -> renders as 'possible' (dashed chip), still recalled; cap45 -> below PERSON_MIN, not a person hit."""
from app.db import connect

VIEW = """CREATE OR REPLACE VIEW exp_vlm_alone_scores_{tag} AS
SELECT image_id, person_id,
 CASE WHEN bool_and(source = 'vlm') THEN least(1 - exp(sum(ln(greatest(1 - score, 1e-6)))), {cap})
 ELSE 1 - exp(sum(ln(greatest(1 - score, 1e-6)))) END AS score,
 array_agg(source || ':' || round(score::numeric, 2) ORDER BY source) AS evidence
FROM image_persons GROUP BY image_id, person_id"""

if __name__ == "__main__":
 c = connect()
 for tag, cap in (("cap75", 0.75), ("cap45", 0.45)):
 c.execute(VIEW.format(tag=tag, cap=cap))
 c.commit()
 for tag in ("cap75", "cap45"):
 a, b, d = c.execute(f"""SELECT count(*) FILTER (WHERE b.score >= 0.8), count(*) FILTER (WHERE e.score >= 0.8),
 count(*) FILTER (WHERE b.score >= 0.5 AND e.score < 0.5)
 FROM image_person_scores b JOIN exp_vlm_alone_scores_{tag} e USING (image_id, person_id)""").fetchone()
 print(f"{tag}: confirmed tags {a} -> {b}; person hits lost {d}")
