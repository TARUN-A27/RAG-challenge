#!/bin/sh
# Download the model ONCE, here, so the image can ship it (the evaluation container has no network).
# ~9 GB for the 4B model. Run inside the repo's venv:  .venv/bin/pip install huggingface_hub
set -e
MODEL="${1:-Qwen/Qwen3-VL-4B-Instruct}"
python3 - "$MODEL" <<'PY'
import sys
from huggingface_hub import snapshot_download
name = sys.argv[1]
snapshot_download(name, local_dir=f"models/{name.split('/')[-1]}")
PY
