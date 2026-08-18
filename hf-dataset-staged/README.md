---
license: apache-2.0
tags:
  - apple-silicon
  - mlx
  - sglang
  - ollama
  - qwen3.8
  - benchmark
  - inference
  - concurrency
  - speculative-decoding
pretty_name: Qwen3.8-27B concurrent serving on Apple Silicon
---

# Qwen3.8-27B concurrent serving on Apple Silicon — benchmarks, patches & recipes

Research artifacts from making **16 concurrent Qwen3.8-27B requests** work on a
single Apple Silicon Mac (M-series, 128 GB unified memory) — across **three
serving engines**, including the patches that make **SGLang's native MLX
backend** serve this model for the first time.

**Code / full history:** https://github.com/bluehawana/Qwen3.827B-SGLang-mpbm5max
**Run it with Ollama:** `ollama run bluehawana/qwen3.8-27b-q8`

## Headline results

Solo GPU, distinct ~1.5k-token prompts, 64-token outputs (aggregate tok/s):

| N concurrent | oMLX (MLX 8-bit) | SGLang-MLX patched | Ollama 0.32 (GGUF Q8_0) |
|---:|---:|---:|---:|
| 1 | **12.0** | 10.3 | 6.8 |
| 4 | **20.0** | 19.8 | 14.3 |
| 8 | 20.1 | **21.6** | 13.8 |
| 16 | 19.8 | **21.0** | 13.6 |
| 16 × 256-token outputs | 35.7 | **47.4** | 16.8 |

**Every cell: N/N requests succeeded, zero errors.** A 27B dense model
genuinely serves 16 concurrent clients on one Mac, on all three engines.

## Key findings

1. **Pin in-model concurrency, queue the rest** (`--max-running-requests` /
   `max_concurrent_requests` / `OLLAMA_NUM_PARALLEL`) — the one idea that
   ports from CUDA serving stacks to Metal.
2. **Short-output loads are prefill-bound** and plateau by N=4. Long outputs
   are where continuous batching multiplies throughput (up to 4.5× the
   single-stream decode rate).
3. **SGLang's radix-cache scheduler wins decode-heavy** (+33 % over oMLX at
   N=16); oMLX wins single-user latency; Ollama is the easiest and fully
   stable but 1.5–3× slower.
4. **GPU contention is the #1 false negative.** With a second 30 GB model
   resident, the same hardware "collapses at N=6" and eventually hits a Metal
   OOM. Most "Macs can't do concurrent serving" reports are this artifact.
   One model server at a time.

## Files

| File | Contents |
|---|---|
| `BENCHMARKS.md` | Three-way ladder + full analysis |
| `RESEARCH.md` | Community survey (vllm-mlx, mlx-lm batching, HN/X findings) |
| `OLLAMA.md` | Run + concurrent-serve with Ollama |
| `OMLX.md` | Run + concurrent-serve with oMLX (best for 1–4 users) |
| `ladder-20260816.txt` | Raw benchmark output |
| `sglang-mlx-qwen38.patch` | 4-file patch: SGLang-MLX serves Qwen3.8-27B (hybrid-GDN cache, VLM checkpoint text-only, 2 batched-load device bugs) |
| `PATCH-NOTES.md` | Why each patch exists, with crash signatures |
| `bench_concurrency.py` | Stdlib-only benchmark harness |

## Reproduce the SGLang path

```bash
git clone https://github.com/bluehawana/Qwen3.827B-SGLang-mpbm5max
cd Qwen3.827B-SGLang-mpbm5max
./build-sglang-metal.sh          # needs Xcode Metal Toolchain; auto-applies patches
MAX_CONC=16 ./sglang-metal/serve-sglang-metal.sh
```

Key launch ingredients discovered here: `SGLANG_USE_MLX=1`,
`--json-model-override-args '{"language_model_only": true}'` (runs the VLM
checkpoint text-only — there is no CLI flag for this),
`--mamba-radix-cache-strategy no_buffer` (hybrid-GDN default path is
CUDA-only), plus the included patch.

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) ·
MLX quants: [mlx-community](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit) ·
GGUF: [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF)
