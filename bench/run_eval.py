import json, sys, csv
from collections import defaultdict
from openai import OpenAI
from scoring import score

config, model = sys.argv[1], sys.argv[2]
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
rows = [json.loads(l) for l in open("bench/eval.jsonl", encoding="utf-8")]

MAX_TOKENS = 768   # fixed for every config

results = []
for i, r in enumerate(rows):
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": r["prompt"]}])
    out = resp.choices[0].message.content
    finish = resp.choices[0].finish_reason      # "stop" or "length"
    results.append((r["id"], r["category"], score(r, out), finish, out))
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{len(rows)} done")

path = f"results/eval_{config}.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "category", "correct", "finish", "output"])
    w.writerows(results)

tot, cnt = defaultdict(int), defaultdict(int)
for _, cat, s, _, _ in results:
    tot[cat] += s
    cnt[cat] += 1
for cat in cnt:
    print(f"{cat}: {tot[cat]}/{cnt[cat]}")
print(f"overall: {sum(tot.values())}/{sum(cnt.values())}")
print("truncated:", sum(1 for x in results if x[3] == "length"))
print("saved", path)
