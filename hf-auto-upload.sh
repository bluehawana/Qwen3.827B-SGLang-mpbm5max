#!/bin/bash
# Waits for huggingface.co to become reachable, then uploads:
#  1) the dataset update (OMLX.md + README)
#  2) the new model repo bluehawana/Qwen3.8-27B-Q8-GGUF (28.9 GB)
DS_DIR=/private/tmp/claude-501/-Users-harvad-Projects-rivivedLC105/472715f2-a10f-4f80-a955-52cba81e4ce7/scratchpad/hf-dataset
MODEL_DIR=$HOME/Projects/qwen3.8concurent/hf-model-staged
HF=$HOME/.local/bin/hf
while true; do
  if curl -fs -o /dev/null --max-time 10 https://huggingface.co 2>/dev/null; then
    echo "HF REACHABLE at $(date '+%H:%M')"
    echo "--- uploading dataset update ---"
    "$HF" upload bluehawana/qwen3.8-27b-apple-silicon-concurrency "$DS_DIR" . --repo-type dataset \
      --commit-message "Add OMLX.md run guide; engine-choice matrix" 2>&1 | tail -1
    echo "--- creating model repo ---"
    "$HF" repo create Qwen3.8-27B-Q8-GGUF --repo-type model -y 2>&1 | tail -1
    echo "--- uploading model (28.9 GB, follow symlink) ---"
    "$HF" upload bluehawana/Qwen3.8-27B-Q8-GGUF "$MODEL_DIR" . --repo-type model \
      --commit-message "Qwen3.8-27B Q8_0 GGUF + model card (quant credit: AtomicChat)" 2>&1 | tail -2
    echo "ALL UPLOADS ATTEMPTED at $(date '+%H:%M')"
    exit 0
  fi
  sleep 300
done
