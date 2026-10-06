FROM vllm/vllm-openai:latest
ENTRYPOINT ["python3", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "Qwen/Qwen2.5-1.5B-Instruct", "--dtype", "half", "--max-model-len", "2048", "--gpu-memory-utilization", "0.85"]
