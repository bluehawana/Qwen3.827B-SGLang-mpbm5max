#!/usr/bin/env python3
"""Merge Qwen3.8-27B's MTP (multi-token-prediction) head into an mlx-community
Qwen3.8-27B checkpoint so oMLX (`mtp_enabled`) / mlx-lm PR-990-style runtimes
can do native speculative decoding.

Why: mlx-community's Qwen3.8-27B bf16/8bit/4bit conversions declare
`mtp_num_hidden_layers: 1` in config.json but the converter strips the 15
`mtp.*` tensors, so decode runs plain autoregressive (~9 tok/s bf16 on M5 Max).
The head is public: EigenLabs/Qwen3.8-27B-MTP-bf16 (bit-identical to Qwen's
release, keys with the `mtp.` prefix removed, norms ALREADY in MLX's +1
convention -- verified against Qwen/Qwen3.8-27B; do not shift them again).

Usage (run with a Python that has mlx + huggingface_hub, e.g. oMLX's bundled one):
  python merge_mtp.py <model_dir> [--apply] [--quantize BITS GROUP] [--head PATH]

  --quantize 8 64   quantize the head's Linear weights to match an 8-bit/gs64
                    body (norms stay bf16). Leave the head bf16 for 4-bit
                    bodies (better draft acceptance; the loader only quantizes
                    modules that have `.scales` in the weights).
Writes model-mtp.safetensors, adds the keys to model.safetensors.index.json,
sets text_config.mtp_num_hidden_layers>=1. Backups: *.bak-premtp. Idempotent.
"""
import json, sys, shutil
from pathlib import Path
import mlx.core as mx

args = sys.argv[1:]
if not args or args[0].startswith("-"):
    print(__doc__); sys.exit(1)
model_dir = Path(args[0]).resolve()
apply = "--apply" in args
qbits = qgroup = None
if "--quantize" in args:
    i = args.index("--quantize"); qbits, qgroup = int(args[i+1]), int(args[i+2])
if "--head" in args:
    head_path = Path(args[args.index("--head")+1])
    if head_path.is_dir(): head_path = head_path / "model.safetensors"
else:
    from huggingface_hub import snapshot_download
    head_path = Path(snapshot_download("EigenLabs/Qwen3.8-27B-MTP-bf16")) / "model.safetensors"

idx_path = model_dir / "model.safetensors.index.json"
idx = json.loads(idx_path.read_text()); wm = idx["weight_map"]
prefix = "language_model." if any(k.startswith("language_model.") for k in wm) else ""
have = sum("mtp." in k for k in wm)
print(f"model: {model_dir.name} | key prefix: {prefix!r} | existing mtp keys: {have}")
if have:
    print("already merged — nothing to do"); sys.exit(0)

head = mx.load(str(head_path))
out = {}
for k, v in head.items():
    k = k[4:] if k.startswith("mtp.") else k          # tolerate either naming
    v = v.astype(mx.bfloat16)
    if v.ndim == 1:
        print(f"  norm {k}: mean={v.astype(mx.float32).mean().item():.3f} (kept as-is: already +1 convention)")
    out[f"{prefix}mtp.{k}"] = v
if qbits:
    q = {}
    for k, v in out.items():
        if v.ndim == 2 and k.endswith(".weight") and "norm" not in k:
            w, s, b = mx.quantize(v, group_size=qgroup, bits=qbits)
            base = k[:-len(".weight")]
            q[base+".weight"], q[base+".scales"], q[base+".biases"] = w, s, b
        else:
            q[k] = v
    out = q
size = sum(x.nbytes for x in out.values())
print(f"head tensors: {len(out)} | {size/1e9:.2f} GB" + (f" (quantized {qbits}-bit/gs{qgroup})" if qbits else " (bf16)"))
if not apply:
    print("DRY RUN — pass --apply to write"); sys.exit(0)

shard = "model-mtp.safetensors"
mx.eval(*out.values())
mx.save_safetensors(str(model_dir / shard), out, metadata={"format": "mlx"})
for k in out: wm[k] = shard
idx.setdefault("metadata", {})["total_size"] = idx["metadata"].get("total_size", 0) + size
for p in (idx_path, model_dir / "config.json"):
    bak = p.with_suffix(".json.bak-premtp")
    if not bak.exists(): shutil.copy(p, bak)
idx_path.write_text(json.dumps(idx, indent=2))
cfg_path = model_dir / "config.json"; cfg = json.loads(cfg_path.read_text())
scope = cfg.get("text_config", cfg)
scope["mtp_num_hidden_layers"] = max(int(scope.get("mtp_num_hidden_layers") or 0), 1)
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"WROTE {model_dir/shard}; index + config updated (backups *.bak-premtp).")
print("oMLX: PUT /admin/api/models/<id>/settings {\"mtp_enabled\": true} then (re)load the model.")
