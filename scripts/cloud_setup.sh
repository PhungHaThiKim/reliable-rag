#!/usr/bin/env bash
# One-time setup of a cloud GPU machine (RunPod "PyTorch" template, Lambda, ...; Ubuntu + NVIDIA driver).
# Run from inside pypi_malware_rag/:   bash scripts/cloud_setup.sh
# Uses /workspace (RunPod persistent volume) when it exists, otherwise $HOME.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$( [ -d /workspace ] && echo /workspace || echo "$HOME" )
export HF_HOME=${HF_HOME:-$ROOT/hf}                 # model weights on the persistent volume
export PIP_BREAK_SYSTEM_PACKAGES=1                  # Ubuntu 24.04 images mark the system python as externally managed (PEP 668)
mkdir -p "$HF_HOME" "$ROOT/pypi_rag_work"

python3 -m pip install -U pip
python3 -m pip install vllm                         # brings a matching CUDA build of torch
python3 -m pip install -r requirements.txt jupyter nbconvert ipykernel openai
python3 -m ipykernel install --user --name python3 >/dev/null
(apt-get update -qq && apt-get install -y -qq git-lfs tmux) >/dev/null 2>&1 || true
git lfs install --skip-repo >/dev/null 2>&1 || true

# download weights once (safetensors + tokenizer only; skips Llama's 16 GB original/*.pth)
python3 - <<'PY'
from huggingface_hub import snapshot_download
for m in ["NousResearch/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-Coder-7B-Instruct",
          "BAAI/bge-small-en-v1.5", "BAAI/bge-reranker-base"]:
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    print("ok", m)
PY

python3 -m pytest -q -p no:cacheprovider tests/
python3 -c "import vllm, torch; print('vllm', vllm.__version__, '| torch', torch.__version__, '| GPUs', torch.cuda.device_count())"
echo "Setup done. HF_HOME=$HF_HOME. Next: tmux new -s run 'bash scripts/run_cloud.sh'"
