#!/usr/bin/env bash
# Launch oMLX tuned for concurrent serving on a 128 GB Apple Silicon Mac.
#
# The concurrency ceiling on Metal is memory + memory-bandwidth, not a batching
# limit — oMLX already does continuous batching. These flags stop the two things
# that made N=6 collapse in benchmarks:
#   1. memory-pressure thrashing at the default 'balanced' ceiling  → --memory-guard aggressive
#   2. paged-SSD boundary-cache snapshot overhead on distinct prompts → --no-cache (opt-in)
# and pin the running-request cap to where throughput actually peaks (~4).
#
#   ./serve-omlx-tuned.sh                 # 8-bit, admit 4, aggressive memory
#   ADMIT=5 QMODEL_DIR=~/.cache/huggingface/hub ./serve-omlx-tuned.sh
#   SSD_CACHE=1 ./serve-omlx-tuned.sh     # keep the SSD prefix cache on
set -euo pipefail

PORT="${OMLX_PORT:-8000}"
ADMIT="${ADMIT:-4}"                        # oMLX --max-concurrent-requests
MEM_TIER="${MEM_TIER:-aggressive}"         # safe | balanced | aggressive
SSD_CACHE="${SSD_CACHE:-0}"                # 1 = keep paged SSD cache, 0 = --no-cache
INIT_BLOCKS="${INIT_BLOCKS:-512}"

# Stop any oMLX already on this port (orphan or app-owned) so flags take effect.
if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "serve-omlx-tuned: stopping existing server on :$PORT" >&2
  omlx stop >/dev/null 2>&1 || true
  pkill -f "omlx.*serve.*--port $PORT" 2>/dev/null || true
  for _ in $(seq 1 20); do lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 1; done
fi

ARGS=(serve --port "$PORT"
  --max-concurrent-requests "$ADMIT"
  --memory-guard "$MEM_TIER"
  --initial-cache-blocks "$INIT_BLOCKS"
  --sse-keepalive-mode chunk)

if [[ "$SSD_CACHE" == "0" ]]; then
  ARGS+=(--no-cache)
  echo "serve-omlx-tuned: paged SSD cache OFF (best for many distinct prompts)" >&2
else
  echo "serve-omlx-tuned: paged SSD cache ON (best when prompts share long prefixes)" >&2
fi

echo "serve-omlx-tuned: omlx ${ARGS[*]}" >&2
exec omlx "${ARGS[@]}"
