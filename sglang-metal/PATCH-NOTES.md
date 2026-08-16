# SGLang-Metal patches for Qwen3.8-27B-8bit (MLX)

Date: 2026-08-16. Result: **SERVES** — SGLang's native MLX backend serves
`mlx-community/Qwen3.8-27B-8bit` on :30000 and answers chat completions
(smoke test: "Say OK" -> thinks briefly, answers "OK", ~1.9 s end-to-end).

Three walls were hit after the build succeeded; one solved by config override,
two by minimal source patches in `.sglang-src` (editable install, so patches
take effect immediately — no rebuild needed). No git commits were made.

## 1. VLM config vs. missing image processor — solved with NO source change

The mlx-community Qwen3.8-27B checkpoints declare
`Qwen3_5ForConditionalGeneration` (with `vision_config`), so
`ModelConfig` auto-resolves `enable_multimodal=True` and the tokenizer manager
dies with `ValueError: Unrecognized image processor`. There is no
`--disable-multimodal` flag — but SGLang honors a checkpoint-level
`language_model_only` boolean (the checkpoint config already carries
`"language_model_only": false`), and `--json-model-override-args` can flip it:

```
--json-model-override-args '{"language_model_only": true}'
```

Effect (all pre-existing SGLang code paths, nothing patched):
- `srt/configs/model_config.py`: `is_lm_only=True` → `is_multimodal=False`,
  mrope disabled.
- `srt/utils/hf_transformers/processor.py`: routes to `AutoTokenizer`
  instead of the (unregistered) mm processor.
- `srt/models/qwen3_vl.py` / `qwen3_5.py`: skip building/loading the vision
  tower (`vision_tower.*` weights simply ignored; `language_model.*` weights
  load fine).

Note: the CLI flag `--language-model-only` does NOT work here — it is gated to
`MuseGlimmerForConditionalGeneration` in
`server_args.py:_handle_language_model_only`. The JSON override bypasses that
gate legitimately (it is the same knob checkpoints set themselves).

## 2. `pool_stats_observer.py` — mamba stats crash on idle loop

`python/sglang/srt/managers/scheduler_components/pool_stats_observer.py`,
`_get_mamba_token_info()` (~line 249) did
`self.req_to_token_pool.mamba_allocator.available_size()`. The MLX backend's
`MlxAuxiliaryStateReqToTokenPool` has no separate `mamba_allocator`; its
`mamba_pool` (`MlxAuxiliaryStatePool`) exposes the allocator API itself.
Crash signature:

```
AttributeError: 'MlxAuxiliaryStateReqToTokenPool' object has no attribute 'mamba_allocator'
```

Patch: `getattr(pool, "mamba_allocator", None)` with fallback to
`pool.mamba_pool`. (Deliberately did NOT add a `mamba_allocator` alias on the
MLX pool class — scheduler paths probe `getattr(..., "mamba_allocator", None)`
and would then call CUDA-allocator-only methods like `alloc_group_begin`.)

## 3. `auxiliary_state.py` — `mamba_checkpoint_grid` missing on first request

`python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py`,
`MlxAuxiliaryStateComponent.__init__` bypasses `MambaComponent.__init__`
(whose `HybridReqToTokenPool` assert doesn't hold for MLX) via
`TreeComponent.__init__`, but inherited `MambaComponent` methods still read
attributes the skipped init sets. First radix-cache prefix match crashed with:

```
AttributeError: 'MlxAuxiliaryStateComponent' object has no attribute 'mamba_checkpoint_grid'
  (mem_cache/unified_cache/components/mamba_component.py:171, finalize_match_result_in_tree_core)
```

Patch: mirror the parent's initialization at the end of the MLX `__init__`:

```python
self.mamba_cache_chunk_size = mamba_cache_chunk_size()
self.mamba_checkpoint_grid = mamba_checkpoint_grid(params.page_size)
self.mamba_max_states_per_path = get_exec().mamba.mamba_max_states_per_path
```

(imports from `sglang.srt.runtime_context`; done as a local import to match
the file's dependency style). All three are set because any inherited
`MambaComponent` method may read them; `mamba_branching_seqlen` produced from
the grid IS consumed by `hardware_backend/mlx/model_runner.py:382`, so
returning a dummy/None was not an option.

## Files touched

- `.sglang-src/python/sglang/srt/managers/scheduler_components/pool_stats_observer.py` (+7/-1)
- `.sglang-src/python/sglang/srt/hardware_backend/mlx/kv_cache/auxiliary_state.py` (+13)
- `sglang-metal/serve-sglang-metal.sh` — updated STATUS comment; adds
  `--json-model-override-args '{"language_model_only": true}'` by default
  (disable with `OVERRIDE_ARGS=''`).

## Working launch recipe

```
SGLANG_USE_MLX=1 python -m sglang.launch_server --model-path <SNAP> \
  --host 127.0.0.1 --port 30000 --max-running-requests 4 \
  --mem-fraction-static 0.85 --context-length 32768 \
  --quantization mlx_q8 --mlx-enable-sampling --disable-cuda-graph \
  --mamba-radix-cache-strategy no_buffer --disable-overlap-schedule --page-size 1 \
  --trust-remote-code \
  --json-model-override-args '{"language_model_only": true}'
```

Load time: MLX weights in ~2.5 s; scheduler up in ~5 s total.
