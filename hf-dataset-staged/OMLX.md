# Run with oMLX (recommended for 1–4 users)

[oMLX](https://github.com/jundot/omlx) is an MLX inference server for Apple
Silicon with continuous batching and SSD prefix caching, managed from the
macOS menu bar. In our benchmarks it had the **best single-request latency**
(5.3 s warm for 1.5k in / 64 out) and matches SGLang up to N≈4 — the sweet
spot for personal / small-team use.

## Setup

```bash
# 1. Install oMLX (menu-bar app; ships an `omlx` CLI)
#    https://github.com/jundot/omlx

# 2. Pull the MLX 8-bit build (~30 GB; oMLX discovers the HF cache automatically)
hf download mlx-community/Qwen3.8-27B-8bit

# 3. Serve (restart oMLX after new downloads so it rescans)
omlx start          # OpenAI + Anthropic compatible on http://127.0.0.1:8000
```

API key lives in `~/.omlx/settings.json` (`auth.api_key`). Model id:
`mlx-community--Qwen3.8-27B-8bit`.

## Concurrent serving

```bash
omlx serve --port 8000 --max-concurrent-requests 8 --memory-guard aggressive
```

- `--max-concurrent-requests` is the in-model cap (its `--max-running-requests`).
- SSD prefix cache: **keep it ON when clients share long prompts** (e.g. many
  Claude Code sessions — a ~20k-token system prompt becomes a cache hit);
  add `--no-cache` only for benchmark-style distinct-prompt loads.
- Measured (M-series 128 GB, distinct 1.5k prompts): 20 tok/s aggregate from
  N=4, 35.7 tok/s at N=16 decode-heavy, 16/16 success everywhere.

## Use with Claude Code

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8000 \
ANTHROPIC_AUTH_TOKEN=<api_key from ~/.omlx/settings.json> \
claude --model mlx-community--Qwen3.8-27B-8bit
```

(or `omlx launch claude`, which sets this up for you.)

## Which engine when?

| Scenario | Pick |
|---|---|
| 1–4 users / personal Claude Code | **oMLX** (best latency, SSD prompt cache) |
| Many parallel decode-heavy clients | **SGLang-MLX + our patch** (47.4 tok/s @ N=16) |
| Zero-setup / GGUF ecosystem | **Ollama** (`ollama run bluehawana/qwen3.8-27b-q8`) |

⚠️ Whichever you pick: **one model server on the GPU at a time.** Two resident
27B models cut throughput ~3× and can end in a Metal OOM (see BENCHMARKS.md).
