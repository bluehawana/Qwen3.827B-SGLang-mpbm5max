#!/usr/bin/env bash
# Track B: serve a model via SGLang's native MLX backend on Apple Silicon.
#
# Build first with ../build-sglang-metal.sh (needs Xcode Metal Toolchain +
# either Rust or SGLANG_BUILD_RUST_EXTS=none — the build script handles both).
#
# STATUS (2026-08-16, tested): SERVES Qwen3.8-27B-8bit on Metal. Three walls
# were hit and solved:
#   1. Qwen3.8 is a hybrid-GDN model → default mamba `extra_buffer` cache is
#      CUDA-only. Worked around with `no_buffer` (flags below).
#   2. The mlx-community Qwen3.8-27B checkpoints are VLMs
#      (Qwen3_5ForConditionalGeneration); SGLang auto-enables multimodal and
#      dies looking for an image processor. Fixed with NO source change:
#      `--json-model-override-args '{"language_model_only": true}'` flips the
#      checkpoint-level text-only flag SGLang already honors (skips the mm
#      processor, mrope, and the vision tower entirely).
#   3. Two small MLX-backend bugs in the SGLang source (patched in
#      .sglang-src, see PATCH-NOTES.md): pool_stats_observer assumed a
#      `mamba_allocator` attr, and MlxAuxiliaryStateComponent skipped the
#      parent init that sets `mamba_checkpoint_grid`.
# Also works for pure-text MLX models:
#   MODEL=mlx-community/Meta-Llama-3.1-8B-Instruct-8bit ./serve-sglang-metal.sh
# (for those, TEXT_ONLY_OVERRIDE is harmless — drop it via OVERRIDE_ARGS='' if
# a model's config rejects unknown keys).
#
# WARNING: don't run this at the same time as oMLX — two models on one GPU
# starve each other. Stop oMLX (or your qclaude session) first for a fair run.
set -euo pipefail

SRC="${SGLANG_SRC:-$HOME/Projects/qwen3.8concurent/.sglang-src}"
PORT="${PORT:-30000}"
MAX_CONC="${MAX_CONC:-4}"           # SGLang --max-running-requests (the DGX-Spark knob)
MEM_FRAC="${MEM_FRAC:-0.85}"        # --mem-fraction-static
CTX="${CTX:-32768}"
QUANT="${QUANT:-mlx_q8}"            # mlx_q8 for the 8-bit build, mlx_q4 for 4-bit
# The mlx-community Qwen3.8 checkpoints are VLM configs; this runs them
# text-only (see STATUS above). Set OVERRIDE_ARGS='' to disable.
if [ -z "${OVERRIDE_ARGS+x}" ]; then
  OVERRIDE_ARGS='{"language_model_only": true}'
fi
EXTRA_FLAGS=()
if [ -n "$OVERRIDE_ARGS" ]; then
  EXTRA_FLAGS+=(--json-model-override-args "$OVERRIDE_ARGS")
fi

# Prefer a local MLX snapshot so SGLang doesn't re-download.
DEFAULT_SNAP="$(ls -d "$HOME"/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-8bit/snapshots/*/ 2>/dev/null | head -1)"
MODEL="${MODEL:-${DEFAULT_SNAP:-mlx-community/Qwen3.8-27B-8bit}}"

if lsof -tiTCP:8000 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "WARNING: oMLX appears to be on :8000. Running SGLang alongside it will" >&2
  echo "         halve both models' speed (shared GPU). Ctrl-C now to abort." >&2
  sleep 3
fi

source "$SRC/.venv-metal/bin/activate"
echo "SGLang-Metal → model=$MODEL port=$PORT max_running=$MAX_CONC mem_frac=$MEM_FRAC" >&2

# SGLANG_USE_MLX=1 selects the native MLX runtime (vs the slower torch.mps fallback).
# --quantization mlx_q8/mlx_q4 tells SGLang these are MLX-quantized weights;
# --mlx-enable-sampling makes temperature/top_p work on the MLX path.
exec env SGLANG_USE_MLX=1 python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --max-running-requests "$MAX_CONC" \
  --mem-fraction-static "$MEM_FRAC" \
  --context-length "$CTX" \
  --quantization "$QUANT" \
  --mlx-enable-sampling \
  --disable-cuda-graph \
  --mamba-radix-cache-strategy no_buffer \
  --disable-overlap-schedule \
  --page-size 1 \
  --trust-remote-code \
  ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
