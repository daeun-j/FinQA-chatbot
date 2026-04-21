#!/bin/bash
# Launch vLLM server for Qwen2.5-7B-Instruct
#
# Requirements:
#   - NVIDIA GPU with >= 16GB VRAM (A100, A10G, L4, or RTX 4090)
#   - pip install vllm
#
# Usage:
#   bash scripts/serve_vllm.sh
#
# The server exposes an OpenAI-compatible API at http://localhost:8000/v1

MODEL="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
MAX_MODEL_LEN=4096
GPU_MEMORY_UTILIZATION=0.9

echo "Starting vLLM server..."
echo "  Model: ${MODEL}"
echo "  Port: ${PORT}"
echo "  Max context length: ${MAX_MODEL_LEN}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --dtype auto \
    --trust-remote-code

