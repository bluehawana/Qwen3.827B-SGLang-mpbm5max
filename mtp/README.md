# Track D — native MTP speculative decoding for Qwen3.8-27B on Apple Silicon

**Result (M5 Max, 128 GB, oMLX, single stream):**

| Checkpoint | plain decode | **with MTP** | how |
|---|---|---|---|
| mlx-community Qwen3.8-27B **bf16** | 9.3 tok/s | **25.9 tok/s** (+178%) | head merged, `mtp_enabled` |
| mlx-community Qwen3.8-27B **8-bit** | ~17 (est.) | **35.9 tok/s** | head merged (quantized 8-bit) |
| mlx-community Qwen3.8-27B **4-bit** | ~26 (est.) | **61.6 tok/s** | head merged (bf16) |
| Ollama `qwen3.8:27b-mlx` (library) | — | **66.7 tok/s** | ships `mtp.*` already; self-drafts |
| Yukon/MLXFast challenge record (4-bit, custom-trained head + kernels) | 26 (their serial ref) | 84 median | reference point |

Same technique everywhere: the model's own multi-token-prediction head drafts 3
tokens per round, the big model verifies them in one forward pass and keeps the
longest correct prefix (acceptance 75–99 % on coding/agent traffic, 2.5–3.6
tokens per cycle). Decode on Apple Silicon is memory-bandwidth-bound — every
verify pass reads all the weights once — so tok/s scales with 1/bytes-per-pass:
bf16 (54 GB) tops out ~26, 8-bit (28 GB) ~36, 4-bit (14–21 GB) 60–85. That is
why the public 84 tok/s numbers are all 4-bit; bf16 will never get there.

This supersedes the earlier note in `bench-results/daily-driver-20260818.txt`
("MTPLX MTP shows no gain"): that MTPLX side-car was never imported into the
checkpoint index (oMLX reports *"MTPLX side-car detected but not imported"*), so
the A/B ran plain autoregressive on both sides.

## Why the mlx-community checkpoints are slow

`mlx-community/Qwen3.8-27B-{bf16,8bit,4bit}` declare `mtp_num_hidden_layers: 1`
but the converter dropped the 15 `mtp.*` tensors. oMLX says exactly this:
*"Config declares MTP layers but the weight files contain neither mtp.* tensors
nor native nextn layers."* The head is public —
[EigenLabs/Qwen3.8-27B-MTP-bf16](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-bf16)
(0.85 GB, bit-identical to Qwen's release; published for the
[Layr-Labs/qwen-3.8-mtp-challenge](https://github.com/Layr-Labs/qwen-3.8-mtp-challenge)).

Gotcha we verified against `Qwen/Qwen3.8-27B` shard bytes: that repo's README
says the weights are unchanged, but every norm gamma is already **+1-shifted**
(MLX convention). Do not shift again.

## Do it (≈2 min per checkpoint, no re-conversion)

```bash
# Python with mlx + huggingface_hub — oMLX.app bundles one:
R=/Applications/oMLX.app/Contents/Resources
alias opy='PYTHONHOME=$R/Python/cpython-3.11 PYTHONPATH=$R:$R/Python/framework-mlx-base/lib/python3.11/site-packages $R/Python/cpython-3.11/bin/python3'

opy mtp/merge_mtp.py ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-bf16/snapshots/*/ --apply
opy mtp/merge_mtp.py ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-8bit/snapshots/*/ --apply --quantize 8 64
opy mtp/merge_mtp.py ~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/ --apply   # bf16 head: better acceptance

# enable in oMLX (admin API; cookie from POST /admin/api/login {"api_key": ...})
curl -b cj -X PUT localhost:8000/admin/api/models/mlx-community--Qwen3.8-27B-bf16/settings \
     -H 'Content-Type: application/json' -d '{"mtp_enabled": true}'
# then load the model; the log should say "Speculative backend selected" and "MTP path activated"
```

Or use the ready-made checkpoints:
`bluehawana/Qwen3.8-27B-{4bit,8bit,bf16}-MTP-MLX` on Hugging Face (same
mlx-community weights + `model-mtp.safetensors`).

Ollama users need nothing: `ollama pull qwen3.8:27b-mlx` already includes the head.

## Making Claude Code fast on it (`mtp/qclaude`)

MTP fixed decode; three *prompt* problems still made Claude Code turns take
minutes. All are handled by `mtp/qclaude` (drop it in `~/.local/bin`):

1. **xhigh thinking by default.** oMLX ignores Anthropic `output_config.effort`,
   so Qwen3.8's chat template falls back to `reasoning_effort=xhigh` → 20k+
   thinking tokens per turn. `qclaude --effort medium` (default) sets it
   per-model via the admin API (`chat_template_kwargs`).
2. **Turn-2 cache bust.** Claude Code adds the `LSP` tool lazily on the second
   request; the tools block (17k tokens, rendered before the conversation)
   shifts and the whole prefix re-prefills. `--disallowedTools LSP,...` — plus a
   lean 8-tool set by default (`Workflow` alone is 22k chars); `--full-tools`
   opts out.
3. **Every-turn re-prefill in long sessions.** Claude Code appends a
   `<total_tokens>N tokens left</total_tokens>` system message after each tool
   result; oMLX hoists each new one next to the first, so at 60k+ context every
   turn re-prefilled ~50k tokens (~2 min at 430 tok/s). Fix:
   `CLAUDE_CODE_TOTAL_TOKENS_REMINDER=off`.

Fresh 3-request agentic run on `--dense` (8-bit): 29 s wall, turn-1 prefill
6.5k tokens, later turns reuse the prefix (~1k new tokens each).

MTP is single-stream: a second concurrent request makes oMLX hand off to plain
batched decode (`finish=late-join-handoff`) — for shared boxes, use Track A's
gateway or accept it.

## Files

- `merge_mtp.py` — merge the head into a checkpoint (dry-run by default).
- `qclaude` — Claude Code launcher for oMLX with the fixes above.
- `bench_stream.py` — single-stream decode benchmark (TTFT + tok/s).
- `../bench-results/mtp-20260818.txt` — raw numbers.
