"""Indexing prompt v2: the news-photo concept taxonomy (exp/taxonomy/TAXONOMY.md) as closed vocabularies, plus the existing free-text fields.
Visual only: no dates, no city guesses, names only in recognized_people."""
from .vlm import PERSON, NAMED

EVENT = ["political_rally", "protest_or_demonstration", "strike_or_bandh", "press_conference", "official_meeting_or_session",
 "inauguration_or_launch", "oath_or_swearing_in", "election_voting_or_counting", "campaign_roadshow", "court_or_legal",
 "arrest_or_police_action", "accident", "fire", "flood_or_storm", "drought_or_heat", "building_collapse_or_demolition",
 "riot_or_clash", "funeral_or_mourning", "religious_ritual_or_procession", "festival_celebration", "wedding_or_private_celebration",
 "sports_match", "sports_training_or_practice", "sports_award_or_trophy", "cultural_performance", "film_or_entertainment_event",
 "fashion_show", "exhibition_or_fair", "education_or_exam", "health_or_hospital", "business_or_corporate_event",
 "agriculture_or_rural_life", "daily_life_or_street_scene", "traffic_or_transport", "construction_or_infrastructure",
 "nature_or_wildlife", "military_or_parade", "portrait_or_posed", "product_or_object_shot", "other"]
SETTING = ["legislature_or_assembly", "government_office", "courtroom", "police_station_or_jail", "hospital", "school_or_college",
 "religious_place", "stadium_or_sports_ground", "indoor_arena", "stage_or_auditorium", "conference_hall", "hotel_or_banquet",
 "street_or_road", "highway", "market_or_shop", "residential", "slum_or_informal_settlement", "village_or_farmland",
 "railway_station_or_train", "airport_or_aircraft", "bus_or_bus_stand", "riverbank_or_waterbody", "beach_or_coast",
 "forest_or_park", "construction_site", "factory_or_industrial", "office_or_studio", "vehicle_interior", "outdoor_other", "indoor_other"]
ROLE = ["politician_or_minister", "government_official", "police_or_security_forces", "military", "judge_or_lawyer", "doctor_or_health_worker",
 "student", "teacher_or_academic", "farmer", "labourer_or_worker", "protester", "voter", "journalist_or_camera_crew", "religious_figure",
 "athlete_or_player", "coach_or_referee", "actor_or_celebrity", "musician_or_performer", "business_person", "child", "elderly_person",
 "crowd_or_public", "model_or_fashion", "rescue_worker_or_firefighter", "victim_or_patient", "none_visible"]
CUE = ["khadi_or_kurta_politician_attire", "party_flags_or_symbols", "police_uniform", "military_uniform", "lawyer_black_coat",
 "medical_scrubs_or_white_coat", "school_uniform", "sports_kit_or_jersey", "microphones_pointed", "podium_or_dais",
 "garlands_or_shawl_felicitation", "security_cordon", "official_vehicle_or_convoy"]
SYMBOL_TYPE = ["party_flag", "national_flag", "state_emblem", "team_or_club_badge", "company_logo", "religious_symbol", "protest_banner_or_placard"]
COUNT = ["0", "1", "2-5", "6-20", "21-100", "100+"]
SPORT = ["cricket", "football", "hockey", "kabaddi", "tennis", "badminton", "wrestling", "athletics", "golf", "motorsport", "boxing", "chess",
 "basketball", "volleyball", "swimming", "shooting", "archery", "cycling", "table_tennis", "other_sport", "none"]
OCCASION = ["ganesh_chaturthi", "durga_puja_or_navratri", "diwali", "holi", "eid", "christmas", "independence_day", "republic_day", "gandhi_jayanti",
 "onam", "pongal_or_sankranti", "chhath", "muharram", "guru_nanak_jayanti", "ram_navami_or_hanuman_jayanti", "new_year", "election_day",
 "budget_day", "convocation", "other_festival", "none_or_unknown"]
LANDMARK = ["gateway_of_india", "india_gate", "red_fort", "qutub_minar", "taj_mahal", "parliament_house_delhi", "rashtrapati_bhavan", "vidhana_soudha",
 "howrah_bridge", "victoria_memorial", "charminar", "marine_drive", "bandra_worli_sea_link", "golden_temple", "lotus_temple",
 "mysore_palace", "hawa_mahal", "eden_gardens", "wankhede_stadium", "chinnaswamy_stadium", "narendra_modi_stadium", "jantar_mantar",
 "mantralaya_mumbai", "bombay_high_court", "supreme_court", "other_named_landmark", "none"]
LANDMARK_CITY = {"gateway_of_india": "mumbai", "marine_drive": "mumbai", "bandra_worli_sea_link": "mumbai", "wankhede_stadium": "mumbai",
 "mantralaya_mumbai": "mumbai", "bombay_high_court": "mumbai", "india_gate": "delhi", "red_fort": "delhi", "qutub_minar": "delhi",
 "parliament_house_delhi": "delhi", "rashtrapati_bhavan": "delhi", "lotus_temple": "delhi", "jantar_mantar": "delhi",
 "supreme_court": "delhi", "taj_mahal": "agra", "vidhana_soudha": "bengaluru", "chinnaswamy_stadium": "bengaluru",
 "howrah_bridge": "kolkata", "victoria_memorial": "kolkata", "eden_gardens": "kolkata", "charminar": "hyderabad",
 "golden_temple": "amritsar", "mysore_palace": "mysuru", "hawa_mahal": "jaipur", "narendra_modi_stadium": "ahmedabad"}
EXPRESSION = ["neutral", "smiling", "laughing", "serious", "angry", "crying", "shouting", "surprised", "eyes_closed", "not_visible"]

PROMPT = """You index news photographs for an Indian newspaper's photo search used by journalists. Classify the photo into the newsroom
taxonomy below AND describe it, so it can be found by queries like "farmers protest on highway", "cm inaugurating metro line",
"students celebrating exam results", "yogi laughing", "evm strong room", "cricket team with trophy".

Rules:
- Judge ONLY what is visible. Do not guess the city, the date or the occasion from knowledge; a legislature chamber is a legislature
 chamber even if you cannot tell which state. Use "other"/"not_visible" when unsure.
- Roles are read from visual cues (attire, uniforms, podium, microphones, flags, kit). Record the cues you used.
- symbols: name party flags, national flags, team badges, company logos and religious symbols ONLY when clearly legible or unmistakable
 (e.g. "BJP lotus", "Congress hand", "tricolour", "RCB", "Reliance"). Never infer a party from a person.
- objects: 3-12 short nouns of news value (EVM, placard, barricade, lathi, trophy, garland, sandbag, boat, ambulance, debris, hoarding).
- actions: 2-6 verbs or short phrases (addressing, waving, garlanding, lighting lamp, cutting ribbon, protesting, batting, grieving).
- sport: the sport being played or shown (kit, equipment, ground), else "none". occasion: the festival or national day ONLY when the visual
 cues are unmistakable (idols, colours, lamps, flags, parade), else "none_or_unknown". landmark: ONLY an unmistakable named monument or
 building from the list, else "none"; put any other recognisable named place in landmarks_text. teams: team or club names legible on
 jerseys, badges or boards.
- recognized_people: well-known public figures you can recognise, common full English name, confidence 0-1; leave out below 0.5.
 Names go ONLY there (names written in the image go in visible_text).
- summary: 1-2 factual sentences. tags: 15-30 lowercase keywords including synonyms a journalist might type (rally/protest/demonstration/dharna).
- Use empty strings/lists for things not visible."""

def _enum(vals): return {"type": "string", "enum": vals}
def _arr(item): return {"type": "array", "items": item}
PERSON_V2 = {**PERSON, "required": PERSON["required"], "properties": {**PERSON["properties"], "expression": _enum(EXPRESSION)}}
SCHEMA = {
 "type": "object", "additionalProperties": False,
 "required": ["event_type", "event_type_secondary", "setting", "sport", "occasion", "landmark", "landmarks_text", "teams",
 "actor_roles", "actor_count", "role_cues", "symbols", "objects", "actions",
 "crowd_scale", "weather", "time_of_day", "shot_type", "image_type", "recognized_people", "summary", "people",
 "place_type", "landmarks", "visible_text", "mood", "dominant_colors", "tags"],
 "properties": {
 "event_type": _enum(EVENT), "event_type_secondary": _enum(EVENT + ["none"]), "setting": _enum(SETTING),
 "sport": _enum(SPORT), "occasion": _enum(OCCASION), "landmark": _enum(LANDMARK), "landmarks_text": _arr({"type": "string"}),
 "teams": _arr({"type": "string"}),
 "actor_roles": _arr(_enum(ROLE)), "actor_count": _enum(COUNT), "role_cues": _arr(_enum(CUE)),
 "symbols": _arr({"type": "object", "additionalProperties": False, "required": ["type", "label"],
 "properties": {"type": _enum(SYMBOL_TYPE), "label": {"type": "string"}}}),
 "objects": _arr({"type": "string"}), "actions": _arr({"type": "string"}),
 "crowd_scale": _enum(["none", "few", "dozens", "hundreds", "thousands"]),
 "weather": _enum(["clear", "overcast", "rain", "fog_or_smog", "snow", "not_visible"]),
 "time_of_day": _enum(["day", "night", "dusk_dawn", "indoor_lighting", "unknown"]),
 "shot_type": _enum(["close_up", "medium", "wide", "aerial", "silhouette"]),
 "image_type": _enum(["photo", "collage", "graphic", "illustration", "screenshot", "other"]),
 "recognized_people": _arr(NAMED), "summary": {"type": "string"}, "people": _arr(PERSON_V2),
 "place_type": {"type": "string", "description": "free text, e.g. 'assembly chamber', 'cricket stadium stands'"},
 "landmarks": _arr({"type": "string"}), "visible_text": _arr({"type": "string"}), "mood": {"type": "string"},
 "dominant_colors": _arr({"type": "string"}), "tags": _arr({"type": "string"}),
 },
}


def is_v2(v: dict | None) -> bool:
 return bool(v) and "event_type" in v


def concepts_line(v: dict) -> str:
 """One line for search_text / stage 2: taxonomy labels as words ("" for v1 descriptions)."""
 if not is_v2(v):
 return ""
 w = lambda s: (s or "").replace("_", " ")
 skip = {"", "other", "none", "none_or_unknown", "other_sport", "none_visible", "not_visible", "unknown"}
 lm = v.get("landmark") if v.get("landmark") not in skip else ""
 parts = [w(v.get("event_type")),
 w(v.get("sport")) if v.get("sport") not in skip else "",
 w(v.get("occasion")) if v.get("occasion") not in skip else "",
 w(lm) + (f" {LANDMARK_CITY[lm]}" if lm in LANDMARK_CITY else ""), ", ".join(v.get("landmarks_text") or []),
 ", ".join(v.get("teams") or []), w(v.get("event_type_secondary")) if v.get("event_type_secondary") not in (None, "none") else "",
 w(v.get("setting")), " ".join(w(r) for r in v.get("actor_roles") or []), " ".join(w(c) for c in v.get("role_cues") or []),
 " ".join(s.get("label", "") for s in v.get("symbols") or []), ", ".join(v.get("objects") or []), ", ".join(v.get("actions") or []),
 f"crowd {v.get('crowd_scale')}" if v.get("crowd_scale") not in (None, "none") else "", w(v.get("weather")) if v.get("weather") not in (None, "not_visible", "clear") else ""]
 return "Concepts: " + "; ".join(p for p in parts if p and p.strip() and p != "other") + "."
