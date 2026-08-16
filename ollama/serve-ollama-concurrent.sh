#!/usr/bin/env bash
# Track C: serve Qwen3.8-27B (GGUF Q8_0) concurrently via Ollama.
#
# Runs a PRIVATE ollama instance on :11500 so the system Ollama.app on :11434
# is untouched. Concurrency comes from OLLAMA_NUM_PARALLEL (Ollama's in-model
# slot count — its analogue of SGLang's --max-running-requests).
#
#   ./serve-ollama-concurrent.sh                  # 8 parallel slots
#   PARALLEL=16 ./serve-ollama-concurrent.sh
#
# Model: qwen3.8:atomic-q8 — AtomicChat/Qwen3.8-27B-GGUF:Q8_0 registered
# locally (see ../README.md). KV cache is allocated per slot: with num_ctx
# 8192 and 16 slots expect ~2x the KV of 8 slots — watch memory.
#
# WARNING: one model server on the GPU at a time (see BENCHMARKS.md).
set -euo pipefail

PORT="${PORT:-11500}"
PARALLEL="${PARALLEL:-8}"
MODEL="${MODEL:-qwen3.8:atomic-q8}"
CTX="${CTX:-8192}"                 # per-slot context

export OLLAMA_HOST="127.0.0.1:${PORT}"
export OLLAMA_NUM_PARALLEL="$PARALLEL"
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="10m"
export OLLAMA_CONTEXT_LENGTH="$CTX"

echo "ollama-concurrent → ${MODEL} on :${PORT}, ${PARALLEL} parallel slots, ctx ${CTX}/slot" >&2
echo "clients: OpenAI-compatible at http://127.0.0.1:${PORT}/v1" >&2
exec ollama serve
