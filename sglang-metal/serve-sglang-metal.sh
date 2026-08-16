#!/usr/bin/env bash
# Track B: serve Qwen3.8-27B via SGLang's native MLX backend on Apple Silicon.
#
# Build first with ../build-sglang-metal.sh (needs Xcode Metal Toolchain +
# either Rust or SGLANG_BUILD_RUST_EXTS=none — the build script handles both).
#
#   ./serve-sglang-metal.sh                     # 8-bit local snapshot, port 30000
#   PORT=30000 MODEL=mlx-community/Qwen3.8-27B-8bit ./serve-sglang-metal.sh
#
# WARNING: don't run this at the same time as oMLX — two 27B models on one GPU
# starve each other. Stop oMLX (or your qclaude session) first for a fair run.
set -euo pipefail

SRC="${SGLANG_SRC:-$HOME/Projects/qwen3.8concurent/.sglang-src}"
PORT="${PORT:-30000}"
MAX_CONC="${MAX_CONC:-4}"           # SGLang --max-running-requests (the DGX-Spark knob)
MEM_FRAC="${MEM_FRAC:-0.85}"        # --mem-fraction-static
CTX="${CTX:-32768}"
QUANT="${QUANT:-mlx_q8}"            # mlx_q8 for the 8-bit build, mlx_q4 for 4-bit

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
  --trust-remote-code
