import asyncio, csv, json, subprocess, sys, time
import numpy as np
from openai import AsyncOpenAI

config, model = sys.argv[1], sys.argv[2]
REPEATS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
LEVELS = [1, 4, 16, 32]
N_REQUESTS = 48
MAX_TOKENS = 128          # fixed for every config

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x", timeout=900)
prompts = [json.loads(l)["prompt"] for l in open("bench/eval.jsonl", encoding="utf-8")]

async def one(prompt):
    start = time.perf_counter()
    first, chunks, tokens = None, 0, 0
    try:
        stream = await client.chat.completions.create(
            model=model, temperature=0, max_tokens=MAX_TOKENS, stream=True,
            stream_options={"include_usage": True},
            extra_body={"ignore_eos": True},
            messages=[{"role": "user", "content": prompt}])
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if first is None:
                    first = time.perf_counter() - start
                chunks += 1
            if getattr(chunk, "usage", None):
                tokens = chunk.usage.completion_tokens
    except Exception as e:
        return None
    return first, time.perf_counter() - start, (tokens or chunks)

async def run(n, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async def guarded(p):
        async with sem:
            return await one(p)
    batch = [prompts[i % len(prompts)] for i in range(n)]
    t0 = time.perf_counter()
    res = await asyncio.gather(*[guarded(p) for p in batch])
    wall = time.perf_counter() - t0
    ok = [r for r in res if r and r[0] is not None]
    ttft = np.array([r[0] for r in ok]) * 1000
    lat = np.array([r[1] for r in ok]) * 1000
    return {
        "concurrency": concurrency,
        "requests_ok": len(ok),
        "errors": n - len(ok),
        "tokens_per_sec": sum(r[2] for r in ok) / wall,
        "ttft_ms_p50": float(np.percentile(ttft, 50)),
        "ttft_ms_p95": float(np.percentile(ttft, 95)),
        "lat_ms_p50": float(np.percentile(lat, 50)),
        "lat_ms_p95": float(np.percentile(lat, 95)),
        "lat_ms_p99": float(np.percentile(lat, 99)),
    }

def gpu_mem():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

async def main():
    print("warm-up...")
    await run(8, 4)
    rows = []
    for c in LEVELS:
        for rep in range(REPEATS):
            m = await run(N_REQUESTS, c)
            m.update(config=config, repeat=rep + 1, gpu_mem_mib=gpu_mem())
            rows.append(m)
            print(f"c={c:>2} rep={rep + 1}: {m['tokens_per_sec']:.1f} tok/s | "
                  f"ttft p50 {m['ttft_ms_p50']:.0f} ms | lat p95 {m['lat_ms_p95']:.0f} ms | "
                  f"errors {m['errors']}")
    path = f"results/load_{config}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)

asyncio.run(main())
