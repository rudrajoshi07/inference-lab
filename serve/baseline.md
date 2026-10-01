# Baseline Server Setup

This document records the exact baseline configuration used for all later comparisons.

- Hardware: Colab free T4 (15 GB)
- Model: Qwen/Qwen2.5-1.5B-Instruct, FP16
- Command:
  `python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --dtype half --max-model-len 2048 --gpu-memory-utilization 0.85 --enforce-eager --port 8000`
- GPU memory used (nvidia-smi): 12945 MiB (vLLM reserves memory up front, so this reflects its allocation, not the model weights alone)
- Notes: `--enforce-eager` disables CUDA graphs, so absolute speeds are lower than a fully optimized setup. It is kept for every config so comparisons stay fair.
- Test request: worked
