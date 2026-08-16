#!/usr/bin/env bash
# Build SGLang with the native MLX/Metal backend from source (Track B).
# Long-running; safe to re-run (resumes clone). Logs to build.log.
set -uo pipefail
BUILD_DIR="${BUILD_DIR:-$HOME/Projects/qwen3.8concurent/.sglang-src}"
echo "=== SGLang-Metal build $(date) → $BUILD_DIR ==="
if [ ! -d "$BUILD_DIR/.git" ]; then
  git clone --depth 1 https://github.com/sgl-project/sglang.git "$BUILD_DIR" || exit 1
fi
cd "$BUILD_DIR" || exit 1
[ -d .venv-metal ] || uv venv -p 3.12 .venv-metal || exit 1
source .venv-metal/bin/activate
uv pip install --upgrade pip || exit 1
echo "=== building Metal AOT kernels ==="
uv run python/sglang/kernels/aot/setup_metal.py install || { echo "KERNEL BUILD FAILED"; exit 2; }
if [ -f python/pyproject_other.toml ]; then
  rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
fi
echo "=== installing sglang[all_mps] (no Rust exts) ==="
# The Rust extension modules are optional accelerators; MLX inference doesn't
# need them and we have no cargo toolchain, so skip them per SGLang's own flag.
export SGLANG_BUILD_RUST_EXTS="${SGLANG_BUILD_RUST_EXTS:-none}"
uv pip install -e "python[all_mps]" || { echo "PIP INSTALL FAILED"; exit 3; }
echo "=== verifying import ==="
python -c "import sglang; print('sglang', sglang.__version__)" || { echo "IMPORT FAILED"; exit 4; }
echo "=== BUILD OK $(date) ==="
