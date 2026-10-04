"""Merge duplicate identities in the persons directory (conservative rules), keeping the other spelling as an alias.
Rules (same surname = last token, accent-folded, dots ignored):
 A. initials: the shorter name's single letters are the first letters of the longer name's given names, same count ("M. S. Dhoni" ~ "Mahendra Singh Dhoni")
 B. title prefix: one name = title + the other ("King Charles III" ~ "Charles III", "Sri Sri Ravi Shankar" ~ "Ravi Shankar")
 C. middle name added: shorter has >= 2 real tokens, all appear in order in the longer, and the longer's extra tokens are NOT its first token
Never: single-token names ("Kumar"), different first names ("Shilpa Jagadish Shettar" vs "Jagadish Shettar").
Usage: uv run python -m app.persons_merge [--apply]
"""
import re, sys, unicodedata, collections
from .db import connect

TITLES = {"king", "queen", "prince", "princess", "sri", "shri", "dr", "justice", "swami", "sir", "lord", "pandit", "sardar", "baba", "sant", "maulana", "mufti", "bishop", "cardinal", "pope", "sheikh", "general", "captain", "major", "colonel"}


def toks(name):
 s = unicodedata.normalize("NFKD", name); s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
 s = re.sub(r"\(.*?\)", "", s).replace(".", " ").replace(",", " ")
 return [t for t in re.sub(r"[^a-z0-9 ]+", " ", s).split() if t]


def same_person(a, b):
 ta, tb = toks(a), toks(b)
 if not ta or not tb or ta[-1] != tb[-1] or len(set(ta)) == 1 or len(set(tb)) == 1:
 return None
 short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
 # B: title prefix
 if [t for t in long_ if t not in TITLES] == short and len(short) >= 2:
 return "title"
 real_s = [t for t in short if len(t) > 1]; init_s = [t for t in short if len(t) == 1]
 real_l = [t for t in long_ if len(t) > 1]
 if len(real_l) < 2 or len(short) < 2:
 return None
 # A: initials
 if init_s and real_s == [long_[-1]] and len(init_s) == len(real_l) - 1 and all(i == w[0] for i, w in zip(init_s, real_l[:-1])):
 return "initials"
 # C: real middle name(s) added (not just initials), same first token
 init_l = [t for t in long_ if len(t) == 1]
 if not init_s and not init_l and len(real_s) >= 2 and len(real_l) > len(real_s) and real_s[0] == real_l[0] \
 and all(t in real_l for t in real_s) and [t for t in real_l if t in real_s] == real_s:
 return "middle"
 return None


def main(apply=False):
 conn = connect()
 rows = conn.execute("""SELECT p.id, p.name, (SELECT count(*) FROM image_person_scores s WHERE s.person_id = p.id AND s.score >= 0.5)
 FROM persons p""").fetchall()
 by_last = collections.defaultdict(list)
 for pid, name, n in rows:
 t = toks(name)
 if t:
 by_last[t[-1]].append((pid, name, n))
 parent = {pid: pid for pid, *_ in rows}
 def find(x):
 while parent[x] != x:
 parent[x] = parent[parent[x]]; x = parent[x]
 return x
 pairs = []
 for grp in by_last.values():
 cand = []
 for i in range(len(grp)):
 for j in range(i + 1, len(grp)):
 why = same_person(grp[i][1], grp[j][1])
 if why:
 cand.append((why, grp[i], grp[j]))
 # a name may merge with exactly one other name (an initials form matching several people is ambiguous: "S. Singh")
 deg = collections.Counter()
 for _, a, b in cand:
 deg[a[0]] += 1; deg[b[0]] += 1
 for why, a, b in cand:
 if deg[a[0]] == 1 and deg[b[0]] == 1:
 pairs.append((why, a, b)); parent[find(a[0])] = find(b[0])
 groups = collections.defaultdict(list)
 for pid, name, n in rows:
 groups[find(pid)].append((pid, name, n))
 groups = {k: sorted(v, key=lambda x: -x[2]) for k, v in groups.items() if len(v) > 1}
 print(f"{len(pairs)} pairs -> {len(groups)} merge groups covering {sum(len(v) for v in groups.values())} names")
 for v in sorted(groups.values(), key=lambda v: -sum(x[2] for x in v))[:20]:
 print(" keep", repr(v[0][1]), f"({v[0][2]})", "<-", ", ".join(f"{x[1]!r} ({x[2]})" for x in v[1:]))
 if not apply:
 print("dry run; --apply to merge"); return
 with conn.cursor() as cur:
 for v in groups.values():
 keep = v[0][0]; others = [x[0] for x in v[1:]]
 cur.execute("UPDATE persons SET aliases = (SELECT array_agg(DISTINCT a) FROM unnest(coalesce(aliases, '{}') || %s::text[]) a) WHERE id = %s",
 ([x[1] for x in v[1:]], keep))
 # move evidence; a (image, person, source) that already exists for the kept id wins
 cur.execute("UPDATE image_persons ip SET person_id = %s WHERE person_id = ANY(%s) AND NOT EXISTS (SELECT 1 FROM image_persons j WHERE j.image_id = ip.image_id AND j.person_id = %s AND j.source = ip.source)", (keep, others, keep))
 cur.execute("DELETE FROM image_persons WHERE person_id = ANY(%s)", (others,))
 cur.execute("UPDATE person_refs SET person_id = %s WHERE person_id = ANY(%s)", (keep, others))
 cur.execute("UPDATE faces SET person_id = %s WHERE person_id = ANY(%s)", (keep, others))
 cur.execute("DELETE FROM persons WHERE id = ANY(%s)", (others,))
 print("merged")


if __name__ == "__main__":
 main("--apply" in sys.argv)
