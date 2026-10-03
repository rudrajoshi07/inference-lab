import json, sys, csv
from collections import defaultdict
from openai import OpenAI
from scoring import score

config, model = sys.argv[1], sys.argv[2]
client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
rows = [json.loads(l) for l in open("bench/eval.jsonl", encoding="utf-8")]

results = []
for i, r in enumerate(rows):
    resp = client.chat.completions.create(
        model=model, temperature=0, max_tokens=384,
        messages=[{"role": "user", "content": r["prompt"]}])
    out = resp.choices[0].message.content
    results.append((r["id"], r["category"], score(r, out), out))
    if (i + 1) % 20 == 0:
        print(f"{i + 1}/{len(rows)} done")

path = f"results/eval_{config}.csv"
with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "category", "correct", "finish","output"])
    w.writerows(results)

tot, cnt = defaultdict(int), defaultdict(int)
for _, cat, s, _ in results:
    tot[cat] += s;
    cnt[cat] += 1
for cat in cnt:
    print(f"{cat}: {tot[cat]}/{cnt[cat]}")
print(f"overall: {sum(tot.values())}/{sum(cnt.values())}")
print("saved", path)
