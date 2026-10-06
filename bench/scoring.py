import re, json

NUM = r"-?\d[\d,]*(?:\.\d+)?"

def score(row, output):
    cat, ref = row["category"], row["reference"]
    output = output or ""

    if cat == "math":
        idx = output.rfind("Answer:")
        if idx != -1:
            m = re.search(NUM, output[idx + len("Answer:"):])
            val = m.group(0) if m else None
        else:
            found = re.findall(NUM, output)
            val = found[-1] if found else None
        if val is None:
            return 0
        try:
            return int(abs(float(val.replace(",", "")) - float(ref)) < 1e-6)
        except ValueError:
            return 0

    if cat == "knowledge":
        text = output.strip()
        m = re.match(r"^\(?([ABCD])\b", text) or re.search(r"answer is\s*\(?([ABCD])\b", text, re.I)
        return int(bool(m) and m.group(1).upper() == ref)

    if cat == "format":
        text = re.sub(r"^```(?:json)?|```$", "", output.strip(), flags=re.M).strip()
        try:
            obj, exp = json.loads(text), json.loads(ref)
        except ValueError:
            return 0
        if not isinstance(obj, dict) or set(obj) != set(exp):
            return 0
        age = obj["age"]
        ok = (isinstance(age, int) and not isinstance(age, bool) and age == exp["age"]
              and str(obj["name"]).strip().lower() == exp["name"].lower()
              and str(obj["city"]).strip().lower() == exp["city"].lower())
        return int(ok)

    raise ValueError(f"unknown category {cat}")
