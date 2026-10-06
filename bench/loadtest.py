import asyncio, csv, json, os, subprocess, sys, time
import numpy as np
from openai import AsyncOpenAI

config_name, model_name = sys.argv[1], sys.argv[2]
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CONCURRENCY_LEVELS = [1, 4, 16, 32]
TOTAL_REQUESTS = 48
MAX_TOKENS = 128            # fixed for every config
FIRST_ERROR = []

# optional: a long shared system prompt, used to test prefix caching
PREFIX_FILE = os.environ.get("SHARED_PREFIX_FILE")
PREFIX = open(PREFIX_FILE, encoding="utf-8").read() if PREFIX_FILE else None

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="x", timeout=900)
with open("bench/eval.jsonl", "r", encoding="utf-8") as f:
    prompts = [json.loads(line)["prompt"] for line in f]

async def send_single_request(prompt):
    start = time.perf_counter()
    ttft, tokens, chunks = None, 0, 0
    messages = [{"role": "user", "content": prompt}]
    if PREFIX:
        messages = [{"role": "system", "content": PREFIX}] + messages
    try:
        stream = await client.chat.completions.create(
            model=model_name, temperature=0, max_tokens=MAX_TOKENS, stream=True,
            stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
            messages=messages)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                chunks += 1
            if getattr(chunk, "usage", None):
                tokens = chunk.usage.completion_tokens
    except Exception as e:
        if not FIRST_ERROR:
            FIRST_ERROR.append(repr(e))
        return None
    return ttft, time.perf_counter() - start, (tokens or chunks)

async def benchmark_concurrency(n, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async def worker(p):
        async with sem:
            return await send_single_request(p)
    batch = [prompts[i % len(prompts)] for i in range(n)]
    t0 = time.perf_counter()
    res = await asyncio.gather(*[worker(p) for p in batch])
    wall = time.perf_counter() - t0
    ok = [r for r in res if r is not None and r[0] is not None]
    if not ok:
        return {"concurrency": concurrency, "requests_ok": 0, "errors": n, "tokens_per_sec": 0.0,
                "ttft_ms_p50": 0.0, "ttft_ms_p95": 0.0, "lat_ms_p50": 0.0, "lat_ms_p95": 0.0, "lat_ms_p99": 0.0}
    ttft = np.array([r[0] for r in ok]) * 1000
    lat = np.array([r[1] for r in ok]) * 1000
    return {"concurrency": concurrency, "requests_ok": len(ok), "errors": n - len(ok),
            "tokens_per_sec": sum(r[2] for r in ok) / wall,
            "ttft_ms_p50": float(np.percentile(ttft, 50)), "ttft_ms_p95": float(np.percentile(ttft, 95)),
            "lat_ms_p50": float(np.percentile(lat, 50)), "lat_ms_p95": float(np.percentile(lat, 95)),
            "lat_ms_p99": float(np.percentile(lat, 99))}

def gpu_mem():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0])

async def main():
    print("warm-up...")
    warm = await benchmark_concurrency(8, 4)
    if warm["errors"] > 0:
        print("Warm-up failed. First error:", FIRST_ERROR[:1])
        print("Check that the server is running and the model name matches what it serves.")
        sys.exit(1)
    os.makedirs("results", exist_ok=True)
    path = f"results/load_{config_name}.csv"
    writer = None
    with open(path, "w", newline="", encoding="utf-8") as f:
        for c in CONCURRENCY_LEVELS:
            for rep in range(repeats):
                m = await benchmark_concurrency(TOTAL_REQUESTS, c)
                m.update(config=config_name, repeat=rep + 1, gpu_mem_mib=gpu_mem())
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(m.keys()))
                    writer.writeheader()
                writer.writerow(m)
                f.flush()
                print(f"c={c:>2} rep={rep + 1}: {m['tokens_per_sec']:.1f} tok/s | "
                      f"ttft p50 {m['ttft_ms_p50']:.0f} ms | lat p95 {m['lat_ms_p95']:.0f} ms | errors {m['errors']}")
                if m["errors"] > 0:
                    print("  WARNING: failures. First error:", FIRST_ERROR[:1])
    print("saved", path)

asyncio.run(main())
