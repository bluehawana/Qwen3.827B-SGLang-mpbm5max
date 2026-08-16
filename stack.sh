#!/usr/bin/env bash
# Bring up the whole concurrent-serving stack: tuned oMLX + the admission gateway.
#
#   ./stack.sh                 # oMLX :8000 (admit 4) + gateway :8080
#   ADMIT=4 PROXY_PORT=8080 ./stack.sh
#
# Clients then point at http://127.0.0.1:8080 (OpenAI or Anthropic API).
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")"

ADMIT="${ADMIT:-4}"
OMLX_PORT="${OMLX_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8080}"
SSD_CACHE="${SSD_CACHE:-0}"

cleanup() { echo; echo "stack: stopping…"; kill "${GW_PID:-}" 2>/dev/null || true; kill "${OMLX_PID:-}" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "stack: starting tuned oMLX on :$OMLX_PORT (admit=$ADMIT, ssd_cache=$SSD_CACHE)…"
ADMIT="$ADMIT" OMLX_PORT="$OMLX_PORT" SSD_CACHE="$SSD_CACHE" ./serve-omlx-tuned.sh &
OMLX_PID=$!

KEY="$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/.omlx/settings.json')))['auth']['api_key'])" 2>/dev/null || echo omlx)"
echo "stack: waiting for oMLX…"
until curl -fs --max-time 3 "http://127.0.0.1:$OMLX_PORT/v1/models" -H "Authorization: Bearer $KEY" >/dev/null 2>&1; do
  kill -0 "$OMLX_PID" 2>/dev/null || { echo "stack: oMLX exited"; exit 1; }
  sleep 2
done
echo "stack: oMLX up."

echo "stack: starting gateway on :$PROXY_PORT…"
ADMIT="$ADMIT" PROXY_PORT="$PROXY_PORT" UPSTREAM="http://127.0.0.1:$OMLX_PORT" python3 gateway.py &
GW_PID=$!

echo "stack: ready → clients use http://127.0.0.1:$PROXY_PORT   (Ctrl-C to stop)"
wait "$GW_PID"
