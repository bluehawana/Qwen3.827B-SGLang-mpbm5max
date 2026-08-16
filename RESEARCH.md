# Community research: concurrent MLX serving on Apple Silicon (Aug 2026)

What HN / Reddit / X / GitHub are actually doing about our exact dilemma —
serving a ~27B Qwen to several concurrent clients on a Mac.

## The headline numbers people report (they match ours)

| Source | Setup | Scaling |
|---|---|---|
| vllm-mlx paper (EuroMLSys '26, M4 Max) | Qwen3-0.6B | 3.7× at 16 concurrent |
| vllm-mlx paper | Qwen3-8B | 2.6× at 16 concurrent |
| vllm-mlx paper | 30B-A3B MoE | 2.6× at 16; "larger models show diminishing returns due to memory bandwidth saturation" |
| oMLX docs | BatchGenerator | 4.14× at 8× concurrency (small models) |
| mlx-serve | 4-way parallel | ~1.6× |
| **ours** | **dense 27B 8-bit** | **2.0× at N=4, collapse at N=6** |

Consensus: nobody beats the bandwidth wall on dense ~27B models. Scaling
factors shrink as the model grows; our N=4 ceiling is exactly what the
community measures. The DGX-Spark repo's "10 concurrent" relies on CUDA
tricks (NVFP4 tensor cores, FP8 KV, MTP speculative decode) that have no
Metal equivalents.

## What the community optimizes instead (ranked by relevance to us)

1. **Prefix/prompt caching over raw batching.** The single biggest win
   reported for agent workloads: vLLM-mlx fork reports TTFT on a 33k-token
   context going **28 s → 0.3 s** on cache hit. Multiple Claude Code sessions
   share a ~20k-token system prompt — cache hits are the whole game.
   ⇒ For *our* use case (several qclaude sessions), oMLX's SSD prefix cache
   should stay **ON** (our `--no-cache` tuning is right only for
   distinct-prompt benchmark loads, wrong for many-Claude-Code-sessions).
2. **MoE instead of dense for multi-user.** Qwen3-30B-A3B (3B active) does
   ~128 tok/s single-stream on M4 Max — ~5× our dense-27B decode — and
   scales further before hitting bandwidth. If the goal is 5–6 users, a
   30B-A3B-class MoE serves them comfortably; a dense 27B never will.
3. **Admission control.** Same conclusion as our gateway: cap in-model
   concurrency, queue the rest. (vllm-mlx admits "up to a maximum batch
   size" in its scheduler loop — the same policy, inside the engine.)
4. **Continuous batching at token boundaries** — table stakes now; mlx-lm
   itself has it (`mlx_lm.server`, Awni Hannun demoed 4 simultaneous
   requests of Qwen3-30B on an M2 Ultra), Ollama's Apple backend is moving
   to MLX (preview).

## Server landscape for our stack (Anthropic API + Qwen3.8 VLM checkpoint)

| Server | Continuous batching | Anthropic `/v1/messages` | Multimodal (qwen3_5 VLM) | Notes |
|---|---|---|---|---|
| **oMLX** (current) | ✅ | ✅ | ✅ (mlx-vlm) | + SSD prefix cache; our Track A |
| **vllm-mlx** | ✅ | ✅ | ✅ Qwen3-VL family | `pip install vllm-mlx`; `--continuous-batching`, `--warm-prompts`, `--ssd-cache-dir`; **built-in Claude Code support**; EuroMLSys '26 paper |
| mlx-lm `mlx_lm.server` | ✅ (v0.18+) | ❌ (OpenAI only) | ❌ | simplest, official Apple |
| mlx-serve | ✅ (~1.6× @4) | ✅ | ? | no-Python native app |
| SGLang-MLX | ✅ (design) | ✅ | ❌ **blocks Qwen3.8** | roadmap confirms: VLM + hybrid-GDN not implemented on MLX yet (our Track B result) |
| llama.cpp / Ollama | partial | ❌/partial | ✅ GGUF | slower than MLX per community benchmarks (vllm-mlx +21–87%) |

**SGLang validation:** SGLang's own Apple roadmap (issue #19137) lists model
files still needing MLX rewrites and no VLM/hybrid-GDN support — our Track B
walls are upstream gaps, not our config errors.

## Actionable takeaways for this repo

1. **Keep Track A** (oMLX + gateway) as the serving path; ADMIT=4 is validated
   by every independent source.
2. **Flip SSD cache ON for Claude-Code-style workloads** (shared prefixes);
   keep `--no-cache` only for distinct-prompt loads. Make it the documented
   default split.
3. **Benchmark vllm-mlx as an oMLX alternative** — the only other server with
   the full combo we need (Anthropic API + continuous batching + qwen3_5 VLM
   + prompt caching). If its scheduler/caching beats oMLX at N=4–6, it could
   replace both oMLX *and* the gateway.
4. **If 5–6 users is a hard requirement**, offer a Qwen3-Coder-30B-A3B (MoE)
   tier next to the dense 27B — that's the community's real answer to Mac
   multi-user serving.

## Sources

- [vllm-mlx paper — Native LLM and MLLM Inference at Scale on Apple Silicon (arXiv 2601.19139)](https://arxiv.org/html/2601.19139v2)
- [vllm-mlx repo](https://github.com/waybarrios/vllm-mlx) · [HN: 464 tok/s](https://news.ycombinator.com/item?id=46642846) · [HN: fork w/ prompt caching, 28s→0.3s TTFT](https://news.ycombinator.com/item?id=47162364) · [HN: early WIP](https://news.ycombinator.com/item?id=46179060)
- [Awni Hannun on X — mlx-lm continuous batching, 4× Qwen3-30B on M2 Ultra](https://x.com/awnihannun/status/1996365940343402596)
- [HN: Ollama now powered by MLX (preview)](https://news.ycombinator.com/item?id=47582482)
- [SGLang Apple Device Support roadmap (#19137)](https://github.com/sgl-project/sglang/issues/19137) · [MLX backend issue (#17846)](https://github.com/sgl-project/sglang/issues/17846)
- [oMLX repo](https://github.com/jundot/omlx) · [mlx-serve](https://github.com/ddalcu/mlx-serve)
- [Apple Silicon LLM backends compared](https://jaesolshin.com/posts/apple-silicon-llm-backends/) · [MLX vs vLLM architecture notes](https://contracollective.com/blog/mlx-vs-vllm-production-inference-batching-2026)
