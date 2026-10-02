"""Indian city name variants. One canonical token per city is added to search_text at index time and to the query at search time,
so "bangalore" finds photos whose caption says "Bengaluru" and vice versa. A general alias rule, not a per-query patch."""
import re

CITY_ALIASES = {
 "bengaluru": ["bangalore", "bengaluru", "bengalooru"], "mumbai": ["mumbai", "bombay", "navi mumbai"],
 "kolkata": ["kolkata", "kolkatta", "calcutta"], "chennai": ["chennai", "madras"], "delhi": ["delhi", "new delhi", "dilli"],
 "mysuru": ["mysore", "mysuru"], "mangaluru": ["mangalore", "mangaluru"], "hubballi": ["hubli", "hubballi"],
 "belagavi": ["belgaum", "belagavi"], "thiruvananthapuram": ["thiruvananthapuram", "trivandrum"],
 "tiruchirappalli": ["trichy", "tiruchirappalli", "tiruchi"], "kochi": ["kochi", "cochin"], "pune": ["pune", "poona"],
 "visakhapatnam": ["visakhapatnam", "vizag"], "ahmedabad": ["ahmedabad", "amdavad"], "gurugram": ["gurgaon", "gurugram"],
 "prayagraj": ["allahabad", "prayagraj"], "vadodara": ["baroda", "vadodara"], "varanasi": ["benares", "banaras", "varanasi"],
 "kozhikode": ["calicut", "kozhikode"], "shimla": ["simla", "shimla"], "puducherry": ["pondicherry", "puducherry"],
 "odisha": ["orissa", "odisha"], "bhubaneswar": ["bhubaneshwar", "bhubaneswar"], "kanpur": ["cawnpore", "kanpur"],
}
_VARIANT = {v: k for k, vs in CITY_ALIASES.items() for v in vs}


def canonical_city(text: str | None) -> str | None:
 """Canonical city token if the text contains a known variant, else None."""
 if not text:
 return None
 t = " " + re.sub(r"[^a-z ]+", " ", text.lower()) + " "
 for v, k in sorted(_VARIANT.items(), key=lambda kv: -len(kv[0])):
 if f" {v} " in t:
 return k
 return None


def expand_cities(text: str) -> str:
 """Append every variant of any city named in the text (for word matching)."""
 extra = []
 k = canonical_city(text)
 if k:
 extra = [v for v in CITY_ALIASES[k] if v not in text.lower()]
 return (text + " " + " ".join(extra)).strip() if extra else text
