# Qwen3.8-27B concurrent serving on Apple Silicon (M-series, MLX)

Serve **Qwen3.8-27B** to several clients at once on a 128 GB Mac — the Metal
answer to [MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark),
which reaches ~10 concurrent on an NVIDIA DGX Spark with **SGLang**.

**Why not just port that repo?** Its *speed* comes from CUDA-only tricks —
NVFP4 W4A4 on Blackwell tensor cores, FP8 KV cache, `flashinfer`, mamba radix
cache, MTP/EAGLE speculative decoding, CUDA graphs — none of which exist on
Metal. (SGLang itself *does* now have an early Apple-Silicon path via a native
MLX backend — see **Track B** below — but it runs on the same MLX runtime as
oMLX, so the memory-bandwidth ceiling is the same; it doesn't unlock those
CUDA features.) So the durable idea to port is not the engine, it's the policy:

> Cap how many requests run inside the model at once — sized to your memory
> budget — and FIFO-queue the rest, so N clients all succeed instead of
> thrashing the machine.

This repo ships that as **Track A** (oMLX + a gateway, stable today) and
tracks the faithful engine port as **Track B** (SGLang-native-MLX,
experimental).

## What your Mac can actually do

oMLX (the MLX server) already does continuous batching. Measured on an
M-series, dense 27B, 8-bit, **distinct** prompts (~3k in / 256 out), quiet GPU:

| Concurrent in-model | Wall | Aggregate output | Per-request |
|---:|---:|---:|---:|
| 1 | 49 s | 5.3 tok/s | 5.3 |
| 2 | 65 s | 7.9 tok/s | 3.9 |
| **4** | **97 s** | **10.6 tok/s** | 2.6 |
| 6 | 203 s | 7.6 tok/s ⚠️ | 1.3 |

Decode is **memory-bandwidth-bound**, so past ~4 you gain nothing and then
*lose*: at N=6 oMLX slams into its memory ceiling (104/107 GB) and the
memory-pressure enforcer + paged-SSD-cache snapshotting thrash. **~4 in-model
is the real ceiling on this class of Mac** — not a config bug, physics.

The win this repo delivers: **admit 4 to the model, queue the rest**, so
5–6 (or more) *clients* all get served with a short wait and zero errors —
which is what "supports concurrency" means in practice.

## Pieces

| File | What it does |
|---|---|
| `gateway.py` | OpenAI + Anthropic-compatible reverse proxy. FIFO admission gate (`ADMIT`, default 4) in front of oMLX; queues overflow, streams SSE/JSON through, `/health` + `/metrics`. **Stdlib only, no pip installs.** |
| `serve-omlx-tuned.sh` | Launch oMLX tuned for concurrency: `--memory-guard aggressive`, `--max-concurrent-requests`, `--no-cache` (best for distinct prompts), sane keepalive. |
| `stack.sh` | Start tuned oMLX + gateway together; clients point at `:8080`. |
| `bench_concurrency.py` | Reproduce the table above at any N, direct or via the gateway. |

## Quick start

```bash
# 1. Models (once) — MLX builds for oMLX:
hf download mlx-community/Qwen3.8-27B-8bit        # daily driver (~30 GB)
# hf download mlx-community/Qwen3.8-27B-4bit      # faster (~16 GB)

# 2. Bring up the stack (tuned oMLX + gateway):
./stack.sh                    # oMLX :8000 (admit 4) + gateway :8080

# 3. Point any client at the gateway:
#    OpenAI:    base_url = http://127.0.0.1:8080/v1
#    Anthropic: ANTHROPIC_BASE_URL = http://127.0.0.1:8080
```

Watch it work:

```bash
curl -s http://127.0.0.1:8080/metrics | python3 -m json.tool
# {"admit_limit":4,"running":4,"waiting":2,"admitted":31,"completed":29,...}

python3 bench_concurrency.py mlx-community--Qwen3.8-27B-8bit 6 3000 256 http://127.0.0.1:8080
```

## Tuning knobs

Gateway (env): `ADMIT` (in-model limit, default 4), `MAX_QUEUE` (waiting slots
before 503, default 32), `PROXY_PORT` (8080), `UPSTREAM` (`http://127.0.0.1:8000`).

oMLX (`serve-omlx-tuned.sh` env): `ADMIT`, `MEM_TIER` (`safe|balanced|aggressive`),
`SSD_CACHE` (`0`=`--no-cache`, best for many distinct users; `1`=keep the prefix
cache, best when clients share a long system prompt), `INIT_BLOCKS`.

Rule of thumb: set the gateway `ADMIT` and oMLX `--max-concurrent-requests` to
the **same** number (default 4). Raise toward 5 only if you have RAM headroom
and your prompts are short; drop to 3 for very long contexts.

## Mapping the SGLang recipe → Metal

| DGX-Spark / SGLang | Metal / this repo |
|---|---|
| `--max-running-requests 10` | gateway `ADMIT` + oMLX `--max-concurrent-requests` (~4) |
| `--mem-fraction-static 0.95` | `--memory-guard aggressive` |
| FP8 KV cache | (not available in MLX) — smaller model tier instead |
| radix / mamba cache | oMLX paged prefix cache — off by default here for distinct prompts |
| NVFP4 W4A4 | MLX 4-bit / 8-bit quant |
| MTP EAGLE speculative decode | (not available in MLX) |
| chunked prefill 8192 | oMLX continuous batching (built-in) |

Same shape, honest about what the hardware allows.

## Track B: SGLang-native-MLX (experimental, faithful port)

As of 2026 SGLang has an official (build-from-source) Apple-Silicon path under
its "Apple Device Support 2026 Q2" roadmap, using a **native MLX backend**
(`SGLANG_USE_MLX=1`, ~5× the `torch.mps` fallback). This is the closest thing
to running the DGX-Spark repo's actual engine on a Mac — you get SGLang's
scheduler (radix cache, chunked prefill, continuous batching) instead of a
front-door gate. See `sglang-metal/` for the build + launch scripts and a
head-to-head benchmark against Track A.

Reality check: it runs on the same MLX runtime as oMLX, so peak throughput is
bounded by the same memory bandwidth. The bet is that SGLang's scheduler packs
concurrent batches better at the edge — measured, not assumed.

## Notes

- GGUF (`AtomicChat/Qwen3.8-27B-GGUF`, etc.) runs under **Ollama**, not oMLX.
  You can front an Ollama server with this same gateway (`UPSTREAM=http://127.0.0.1:11434`),
  but Ollama's own parallelism (`OLLAMA_NUM_PARALLEL`) already queues — the
  gateway then mainly adds unified metrics and a hard FIFO cap.
- Never run an Ollama model **and** oMLX at the same time — two model runtimes
  fighting over the GPU cut prefill ~15× (measured).
