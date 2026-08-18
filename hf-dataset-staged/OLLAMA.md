# Run with Ollama

Yes — this model serves 16 concurrent requests under plain Ollama too
(all requests succeed; the MLX engines are 1.5–3× faster, see BENCHMARKS.md).

## One command

```bash
ollama run bluehawana/qwen3.8-27b-q8          # 28.9 GB GGUF Q8_0
# or straight from the source quant:
ollama run hf.co/AtomicChat/Qwen3.8-27B-GGUF:Q8_0
```

## Concurrent serving

```bash
OLLAMA_HOST=127.0.0.1:11500 OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
# OpenAI-compatible endpoint: http://127.0.0.1:11500/v1
```

Measured (M-series 128 GB, distinct 1.5k-token prompts): 16/16 success at
every N; aggregate ~14 tok/s at the plateau, 16.8 tok/s decode-heavy (256-out).

Credit: Q8_0 quant by [AtomicChat](https://huggingface.co/AtomicChat/Qwen3.8-27B-GGUF); base model Apache-2.0 by Qwen.
