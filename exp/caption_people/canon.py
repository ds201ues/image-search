"""Step 2: merge name variants -> canonical persons with aliases. Luna over alphabetical batches + difflib."""
import json, re, collections
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from openai import OpenAI
from app.config import VLM_MODEL

HERE = Path(__file__).parent
OUT = HERE / "persons.json"
PROMPT = """These are names of people extracted from Indian news photo captions, sorted alphabetically. Some are the same
person written differently (initials with/without dots, spelling variants like Yediyurappa/Yeddyurappa, with/without first
name, honorifics, nicknames). Group ONLY the names that clearly refer to the same real person; when in doubt, keep separate
(two different people can share a surname). For each group give the best canonical full English name.
Return every input name in exactly one group."""
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["groups"],
 "properties": {"groups": {"type": "array", "items": {"type": "object", "additionalProperties": False,
 "required": ["canonical", "members"], "properties": {"canonical": {"type": "string"},
 "members": {"type": "array", "items": {"type": "string"}}}}}}}


def norm(s):
 return re.sub(r"[^a-z0-9 ]+", " ", s.lower().replace(".", " ")).split()


def main():
 rows = [json.loads(l) for l in (HERE / "names.jsonl").open()]
 cnt = collections.Counter(p["canonical_full_name"].strip() for r in rows for p in r["people"] if p["canonical_full_name"].strip())
 names = sorted(cnt, key=lambda s: " ".join(norm(s)))
 # pass A: exact normalised match (dots/case)
 parent = {n: n for n in names}
 def find(x):
 while parent[x] != x:
 parent[x] = parent[parent[x]]; x = parent[x]
 return x
 by = {}
 for n in names:
 k = " ".join(norm(n)); by.setdefault(k, []).append(n)
 for g in by.values():
 for n in g[1:]:
 parent[find(n)] = find(g[0])
 # pass B: Luna on batches of 100 (alphabetical so variants are adjacent), with 10-name overlap
 client = OpenAI(timeout=90, max_retries=3)
 batches = [names[i:i + 100] for i in range(0, len(names), 90)]
 def one(b):
 r = client.chat.completions.create(model=VLM_MODEL, reasoning_effort="low",
 messages=[{"role": "user", "content": PROMPT + "\n\n" + "\n".join(b)}],
 response_format={"type": "json_schema", "json_schema": {"name": "g", "strict": True, "schema": SCHEMA}})
 return json.loads(r.choices[0].message.content)["groups"]
 canon_votes = collections.defaultdict(collections.Counter)
 with ThreadPoolExecutor(8) as ex:
 for groups in ex.map(one, batches):
 for g in groups:
 mem = [m for m in g["members"] if m in parent]
 for m in mem[1:]:
 parent[find(m)] = find(mem[0])
 for m in mem:
 canon_votes[m][g["canonical"]] += 1
 # pass C: difflib on normalised strings sharing last token (surname), ratio >= 0.9
 keys = {n: " ".join(norm(n)) for n in names}
 bysur = collections.defaultdict(list)
 for n in names:
 t = norm(n)
 if t: bysur[t[-1]].append(n)
 merges_c = 0
 for g in bysur.values():
 for i in range(len(g)):
 for j in range(i + 1, len(g)):
 a, b = norm(g[i]), norm(g[j])
 if find(g[i]) != find(g[j]) and a[0][0] == b[0][0] and SequenceMatcher(None, a[0], b[0]).ratio() >= 0.8 \
 and SequenceMatcher(None, keys[g[i]], keys[g[j]]).ratio() >= 0.9:
 parent[find(g[i])] = find(g[j]); merges_c += 1
 groups = collections.defaultdict(list)
 for n in names:
 groups[find(n)].append(n)
 persons = []
 for mem in groups.values():
 votes = collections.Counter()
 for m in mem:
 votes.update(canon_votes.get(m, {}))
 votes[m] += cnt[m] * 0.5
 canonical = votes.most_common(1)[0][0]
 persons.append({"name": canonical, "aliases": sorted(set(mem) - {canonical}), "mentions": sum(cnt[m] for m in mem)})
 OUT.write_text(json.dumps(persons, indent=1, ensure_ascii=False))
 print(f"{len(names)} names -> {len(persons)} persons; {len(names) - len(persons)} merged (difflib pass: {merges_c})")
 for p in sorted(persons, key=lambda p: -len(p["aliases"]))[:15]:
 print(" ", p["name"], "<-", p["aliases"])


if __name__ == "__main__":
 main()
