---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - ollama
  - llama.cpp
  - apple-silicon
  - qwen3.8
  - q8_0
---

# Qwen3.8-27B — GGUF Q8_0

Single-file **Q8_0** (8-bit, near-lossless) GGUF of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — 28.9 GB.
Quantized by [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF);
re-hosted here alongside our Apple-Silicon serving research for one-command use.

## Run it

```bash
# Ollama (from the Ollama registry — easiest)
ollama run bluehawana/qwen3.8-27b-q8

# Ollama (straight from this repo)
ollama run hf.co/bluehawana/Qwen3.8-27B-Q8-GGUF

# llama.cpp / LM Studio / Jan: download Qwen3.8-27B-Q8_0.gguf directly
```

Needs ≥48 GB unified memory on Apple Silicon (comfortable at 64 GB+).

## Concurrent serving on a Mac

This model serves **16 concurrent requests with zero errors** on an M-series
128 GB Mac — benchmarks across Ollama / oMLX / SGLang-MLX, plus the SGLang
patches that make it possible (upstream PR
[sgl-project/sglang#35137](https://github.com/sgl-project/sglang/pull/35137)):

**→ [bluehawana/qwen3.8-27b-apple-silicon-concurrency](https://huggingface.co/datasets/bluehawana/qwen3.8-27b-apple-silicon-concurrency)**

Quick concurrent Ollama serving:
```bash
OLLAMA_HOST=127.0.0.1:11500 OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
# OpenAI-compatible endpoint: http://127.0.0.1:11500/v1
```

Credits: base model © Qwen (Apache-2.0) · Q8_0 quant by AtomicChat · benchmarks & patches by [bluehawana](https://github.com/bluehawana/Qwen3.827B-SGLang-mpbm5max).
