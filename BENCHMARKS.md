# Qwen3.8-27B-8bit concurrency benchmarks — Apple Silicon (M-series, 128 GB)

Date: 2026-08-16 · Model: `mlx-community/Qwen3.8-27B-8bit` (MLX, ~30 GB) ·
Method: `bench_concurrency.py` — N simultaneous requests with **distinct**
~1,500-token prompts (prefix cache can't mask work), non-streaming, each
server **alone on the GPU** (this matters: see "contention" below).

## Head-to-head: oMLX vs patched SGLang-MLX

Aggregate output tok/s (and p50 latency), 64 output tokens per request:

| N concurrent | oMLX | SGLang-MLX (patched) |
|---:|---:|---:|
| 1 | **12.0** (5.3 s) | 10.3 (6.2 s) |
| 2 | **17.9** (7.1 s) | 15.9 (8.1 s) |
| 4 | **20.0** (12.8 s) | 19.8 (12.9 s) |
| 8 | 20.1 (25.2 s) | **21.6** (23.7 s) |
| 16 | 19.8 (51.1 s) | **21.0** (42.7 s) |
| **16, decode-heavy** (256 out) | 35.7 (114 s) | **47.4** (86 s) |

All cells: **N/N requests succeeded, zero errors** — both servers genuinely
handle 16 concurrent clients on this hardware.

## Findings

1. **16-concurrent works on a Mac.** Both servers complete 16/16 with no
   failures. The DGX-Spark repo's `--max-running-requests`-style pinning is
   the right shape here too (SGLang flag, oMLX `max_concurrent_requests`).
2. **The 64-out ladder plateaus ≈ 20 tok/s aggregate from N=4** — those runs
   are prefill-dominated (1.5k in / 64 out), and prefill serializes on the
   GPU. Concurrency past 4 mostly queues prefills; latency grows ~linearly.
3. **Decode-heavy is where batching pays.** At 256-out, N=16 aggregate rises
   to 35.7 (oMLX) / **47.4** (SGLang) tok/s — 4.5× the single-stream decode
   rate. SGLang's scheduler is **+33 % over oMLX** here and holds lower p50
   (86 s vs 114 s). Its radix-cache batch packing earns its keep once decode
   dominates.
4. **oMLX wins low-N latency** (warm single request: 5.3 s vs 6.2 s) — fewer
   moving parts per request. For a single Claude Code user, oMLX remains the
   better daily driver; for many parallel decode-heavy clients, patched
   SGLang is now the faster engine.
5. **Contention invalidates everything.** Earlier runs with a second 30 GB
   server resident showed "collapse at N=6" (7.6 tok/s) and ended in a Metal
   OOM. Same hardware, solo server: N=16 no collapse. Never co-load two model
   servers; benchmark nothing while another model owns the GPU.

## Reproduce

```bash
# oMLX ladder
for N in 1 2 4 8 16; do python3 bench_concurrency.py mlx-community--Qwen3.8-27B-8bit $N 1500 64 http://127.0.0.1:8000; done

# SGLang (build + patches: ./build-sglang-metal.sh, then)
MAX_CONC=16 PORT=30000 ./sglang-metal/serve-sglang-metal.sh
for N in 1 2 4 8 16; do python3 bench_concurrency.py <model-path> $N 1500 64 http://127.0.0.1:30000; done
```

Raw output: `bench-results/ladder-20260816.txt`. SGLang patches:
`sglang-metal/patches/sglang-mlx-qwen38.patch` (4 files; see PATCH-NOTES.md).
