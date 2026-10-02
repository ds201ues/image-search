"""Structured image description via an OpenAI-compatible vision model (default gpt-6-luna)."""
import base64, io, json, time
from PIL import Image
from openai import OpenAI
from .config import VLM_MODEL, VLM_PRICE_IN, VLM_PRICE_OUT, VLM_MAX_SIDE, VLM_REASONING, VLM_DETAIL

PROMPT = """You index news photographs for a newsroom photo search system used by journalists.
Describe this image as structured data so that it can be found by queries like
"leader in pink jacket playing drums", "protest outside parliament", "cricket team celebrating in rain".

Rules:
- Be concrete and visual: clothing items and their colours, actions, objects, setting.
- recognized_people: well-known public figures you can recognise (politicians, sportspeople, actors,
 business leaders, officials), using their common full English name, e.g. "Narendra Modi".
 Give a confidence from 0 to 1. Leave out anyone you are not reasonably sure about (below 0.5).
 Never invent a name for a private or unknown person.
- Names of people go ONLY in recognized_people. Never put a person's name in summary, people, or tags
 (names written in the image still go in visible_text).
- Do NOT guess dates.
- tags: 15-30 lowercase search keywords including useful synonyms (e.g. rally, protest, demonstration).
- Use empty strings or empty lists when something is not visible."""

PERSON = {
 "type": "object", "additionalProperties": False,
 "required": ["description", "clothing", "colors", "action", "expression"],
 "properties": {
 "description": {"type": "string", "description": "age range, gender presentation, notable features, e.g. 'elderly man with white beard'"},
 "clothing": {"type": "string", "description": "garments with colours, e.g. 'pink sleeveless jacket over white kurta'"},
 "colors": {"type": "array", "items": {"type": "string"}},
 "action": {"type": "string"},
 "expression": {"type": "string"},
 },
}
NAMED = {
 "type": "object", "additionalProperties": False, "required": ["name", "confidence"],
 "properties": {"name": {"type": "string"}, "confidence": {"type": "number"}},
}
SCHEMA = {
 "type": "object", "additionalProperties": False,
 "required": ["recognized_people", "summary", "image_type", "people_count", "people", "setting", "place_type",
 "location_guess", "landmarks", "objects", "activities", "event_type",
 "visible_text", "mood", "time_of_day", "dominant_colors", "tags"],
 "properties": {
 "recognized_people": {"type": "array", "items": NAMED},
 "summary": {"type": "string", "description": "1-2 sentence factual description"},
 "image_type": {"type": "string", "enum": ["photo", "collage", "graphic", "illustration", "screenshot", "other"]},
 "people_count": {"type": "integer"},
 "people": {"type": "array", "items": PERSON, "description": "up to 6 most prominent people"},
 "setting": {"type": "string", "enum": ["indoor", "outdoor", "mixed", "unknown"]},
 "place_type": {"type": "string", "description": "e.g. parliament, stadium, temple, street, press conference hall"},
 "location_guess": {"type": "string", "description": "city/landmark only if identifiable from visual clues"},
 "landmarks": {"type": "array", "items": {"type": "string"}},
 "objects": {"type": "array", "items": {"type": "string"}},
 "activities": {"type": "array", "items": {"type": "string"}},
 "event_type": {"type": "string", "description": "e.g. political rally, sports match, press conference, protest, film event"},
 "visible_text": {"type": "array", "items": {"type": "string"}},
 "mood": {"type": "string"},
 "time_of_day": {"type": "string", "enum": ["day", "night", "dusk_dawn", "indoor_lighting", "unknown"]},
 "dominant_colors": {"type": "array", "items": {"type": "string"}},
 "tags": {"type": "array", "items": {"type": "string"}},
 },
}

_client = None


def _jpeg_b64(path) -> str:
 im = Image.open(path).convert("RGB")
 im.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
 buf = io.BytesIO()
 im.save(buf, "JPEG", quality=85)
 return base64.b64encode(buf.getvalue()).decode()


def describe(path) -> dict:
 """Returns {vlm: dict, ms: int, cost_usd: float, model: str}."""
 global _client
 _client = _client or OpenAI(timeout=60, max_retries=3)
 t = time.perf_counter()
 r = _client.chat.completions.create(
 model=VLM_MODEL,
 messages=[{"role": "user", "content": [
 {"type": "text", "text": PROMPT},
 {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_jpeg_b64(path)}", "detail": VLM_DETAIL}},
 ]}],
 response_format={"type": "json_schema",
 "json_schema": {"name": "image_index", "strict": True, "schema": SCHEMA}},
 **({"reasoning_effort": VLM_REASONING} if VLM_REASONING else {}),
 )
 u = r.usage
 cost = (u.prompt_tokens * VLM_PRICE_IN + u.completion_tokens * VLM_PRICE_OUT) / 1e6
 return {"vlm": json.loads(r.choices[0].message.content), "model": VLM_MODEL,
 "ms": int((time.perf_counter() - t) * 1000), "cost_usd": cost,
 "tokens_in": u.prompt_tokens, "tokens_out": u.completion_tokens}
