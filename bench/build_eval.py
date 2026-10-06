import json, random
from datasets import load_dataset

random.seed(42)
rows = []

gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=42).select(range(40))
for i, x in enumerate(gsm):
    ref = x["answer"].split("####")[-1].strip().replace(",", "")
    rows.append({"id": f"math-{i}", "category": "math",
                 "prompt": x["question"] + "\n\nSolve step by step. End with a final line in the form: Answer: <number>",
                 "reference": ref})

mmlu = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=42).select(range(30))
L = "ABCD"
for i, x in enumerate(mmlu):
    opts = "\n".join(f"{L[j]}. {c}" for j, c in enumerate(x["choices"]))
    rows.append({"id": f"knowledge-{i}", "category": "knowledge",
                 "prompt": f"{x['question']}\n{opts}\n\nAnswer with only the letter (A, B, C, or D).",
                 "reference": L[x["answer"]]})

names = ["Priya", "Rahul", "Anita", "Karan", "Meera", "Vikram", "Sneha", "Arjun", "Divya", "Rohan",
         "Neha", "Amit", "Kavya", "Sahil", "Pooja"]
cities = ["Pune", "Delhi", "Jaipur", "Chennai", "Kolkata", "Hisar", "Lucknow", "Indore", "Surat", "Kochi"]
templates = ["{name} is {age} years old and lives in {city}.",
             "Meet {name}, a {age}-year-old based in {city}.",
             "{name} ({age}) moved to {city} last year.",
             "Based in {city}, {name} recently turned {age}."]
for i in range(30):
    name, age, city = random.choice(names), random.randint(21, 58), random.choice(cities)
    text = random.choice(templates).format(name=name, age=age, city=city)
    rows.append({"id": f"format-{i}", "category": "format",
                 "prompt": ("Extract the person's name, age, and city from the text below. "
                            "Return only a JSON object with keys \"name\" (string), \"age\" (integer), "
                            "and \"city\" (string).\n\nText: " + text),
                 "reference": json.dumps({"name": name, "age": age, "city": city})})

with open("bench/eval.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print("wrote", len(rows), "examples")
