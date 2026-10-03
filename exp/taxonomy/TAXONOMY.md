# News-photo concept taxonomy and indexing prompt v2 (draft for review, 2026-10-03)

Why: the description schema is free text; captions show it systematically misses roles (minister 97% of the time), institutions (party/club/university),
event framing (final, session, election, inauguration) and even generic concepts half the time (stadium 51%, police 50%, press conference 50%).
Evidence: exp/taxonomy/vocab_gap.txt. Principle: top-down categories from how a newsroom classifies news photos, closed vocabularies at the category
level, free text underneath. Our captions and query log are used to CHECK coverage, not to define categories. Query parsing maps into the same taxonomy,
so matching happens on concepts before words. Everything here is VISUAL (what the picture shows); place names, dates and identities stay with
captions, EXIF and the people index — the model is told not to guess them.

## 1. Event type (one primary, optional secondary) — closed list
political_rally | protest_or_demonstration | strike_or_bandh | press_conference | official_meeting_or_session | inauguration_or_launch |
oath_or_swearing_in | election_voting_or_counting | campaign_roadshow | court_or_legal | arrest_or_police_action | accident | fire |
flood_or_storm | drought_or_heat | building_collapse_or_demolition | riot_or_clash | funeral_or_mourning | religious_ritual_or_procession |
festival_celebration | wedding_or_private_celebration | sports_match | sports_training_or_practice | sports_award_or_trophy |
cultural_performance | film_or_entertainment_event | fashion_show | exhibition_or_fair | education_or_exam | health_or_hospital |
business_or_corporate_event | agriculture_or_rural_life | daily_life_or_street_scene | traffic_or_transport | construction_or_infrastructure |
nature_or_wildlife | military_or_parade | portrait_or_posed | product_or_object_shot | other
## 2. Setting — closed list
legislature_or_assembly | government_office | courtroom | police_station_or_jail | hospital | school_or_college | religious_place |
stadium_or_sports_ground | indoor_arena | stage_or_auditorium | conference_hall | hotel_or_banquet | street_or_road | highway |
market_or_shop | residential | slum_or_informal_settlement | village_or_farmland | railway_station_or_train | airport_or_aircraft |
bus_or_bus_stand | riverbank_or_waterbody | beach_or_coast | forest_or_park | construction_site | factory_or_industrial | office_or_studio |
vehicle_interior | outdoor_other | indoor_other
## 3. Actors present, by ROLE (multi-select, with count bucket) — closed list
politician_or_minister | government_official | police_or_security_forces | military | judge_or_lawyer | doctor_or_health_worker |
student | teacher_or_academic | farmer | labourer_or_worker | protester | voter | journalist_or_camera_crew | religious_figure |
athlete_or_player | coach_or_referee | actor_or_celebrity | musician_or_performer | business_person | child | elderly_person | crowd_or_public |
model_or_fashion | rescue_worker_or_firefighter | victim_or_patient | none_visible
Count bucket: 0 | 1 | 2-5 | 6-20 | 21-100 | 100+
## 4. Role cues (what tells you the role): closed list, multi-select
khadi_or_kurta_politician_attire | party_flags_or_symbols | police_uniform | military_uniform | lawyer_black_coat | medical_scrubs_or_white_coat |
school_uniform | sports_kit_or_jersey | microphones_pointed | podium_or_dais | garlands_or_shawl_felicitation | security_cordon | official_vehicle_or_convoy
## 5. Political and organisational symbols visible — open list of SHORT labels + closed type
type: party_flag | national_flag | state_emblem | team_or_club_badge | company_logo | religious_symbol | protest_banner_or_placard | none
labels: e.g. "BJP lotus", "Congress hand", "AAP broom", "Trinamool flowers", "tricolour", "RCB", "Mumbai Indians", "Reliance" — ONLY if clearly visible
## 6. Objects of news value — open list, 3–12 short nouns
e.g. EVM, ballot box, placard, barricade, lathi, water cannon, trophy, medal, microphone, garland, umbrella, sandbag, boat, ambulance, debris, hoarding
## 7. Actions — open list of 2–6 verbs/verb phrases
e.g. addressing, waving, garlanding, inaugurating (cutting ribbon / lighting lamp), protesting, marching, sitting dharna, batting, celebrating, grieving
## 8. People detail (existing): up to 6 prominent people with description, clothing + colours, action, EXPRESSION (closed: neutral | smiling | laughing | serious | angry | crying | shouting | surprised | eyes_closed | not_visible)
## 9. Scene attributes — closed
crowd_scale: none | few | dozens | hundreds | thousands; weather: clear | overcast | rain | fog_or_smog | snow | not_visible;
time_of_day: day | night | dusk_dawn | indoor_lighting; image_type: photo | collage | graphic | illustration | screenshot;
shot_type: close_up | medium | wide | aerial | silhouette
## 10. Free text (existing): summary (1–2 sentences), visible_text (signage, banners, name plates, jersey names — verbatim), landmarks (only if unmistakable), tags (15–30)
## 11. Identity (existing, unchanged): recognized_people with confidence; never names in other fields; never guess dates or city

## Query side (same taxonomy)
The query parser (Luna text, cached) adds: event_type[], setting[], actor_roles[], objects[], symbols[] from the same closed lists, alongside the
existing people / years / city / rewrite. Recall gets a 5th source: taxonomy match (event_type ∩ setting ∩ roles via SQL on JSONB), weighted by
how specific the matched categories are. Stage 2 document gets a one-line "Concepts: ..." header. Matching "parliament" → setting legislature_or_assembly
covers state assemblies by construction; "kisan andolan" → event protest + actors farmer.

## Test plan (before any backfill)
1. Stratified 2,000 photos (by event type of current description, captioned and uncaptioned), re-described with prompt v2 via Batch API (~$0.50).
2. Build exp tables (vlm_v2, doc_text_v2, taxonomy JSONB), compare on: real-query slice, scene/place/hinglish slices, the 45 inspected misses
 (did the concept now appear?), hallucination check: 100 random photos, human marks symbols/landmarks/roles that are wrong.
3. Paired eval vs the current system on the same queries. Keep if the real/scene slices improve or the concept-miss count falls, with no rise in wrong
 symbols/landmarks. Token cost per photo must stay under +40%.

## Result (2026-10-03)
See EXPERIMENTS.md E15. Coverage gains on roles/settings/sport; no measurable search change on the current eval; +27% cost; human check pending.
