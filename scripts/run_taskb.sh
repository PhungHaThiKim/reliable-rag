#!/usr/bin/env bash
# Task B (notebook 04: supply-chain QA), required for the seminar conclusions. Called by run_cloud.sh on a
# single machine, or alone on its own pod (PYPI_RAG_TASK=b scripts/launch_pod.sh).
# 2+ GPUs: one reader per GPU in parallel; 1 GPU: in sequence.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$( [ -d /workspace ] && echo /workspace || echo "$HOME" )
export HF_HOME=${HF_HOME:-$ROOT/hf}
export PYPI_RAG_BASE=$(cd .. && pwd)
export PYPI_RAG_WORK=${PYPI_RAG_WORK:-$ROOT/pypi_rag_work}
export PYPI_RAG_BACKEND=${PYPI_RAG_BACKEND:-vllm}
export TOKENIZERS_PARALLELISM=false
OUT=$PYPI_RAG_WORK/executed; mkdir -p "$OUT"
READERS=("NousResearch/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct")
nb() {
  echo "[$(date +%H:%M:%S)] start $1 -> $OUT/$2.ipynb"
  jupyter nbconvert --to notebook --execute "notebooks/$1.ipynb" --output "$2" --output-dir "$OUT" \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 > "$OUT/$2.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1"
}
SC=$PYPI_RAG_WORK/supplychain                       # generate the orgs once, before readers run in parallel
[ -f "$SC/dev/stats.json" ] || python3 -m pypi_rag.cli sc-generate --seed 1 --out "$SC/dev" --n-per-group 40
[ -f "$SC/test/stats.json" ] || python3 -m pypi_rag.cli sc-generate --seed 2 --out "$SC/test" --n-per-group 100
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # local weights only (downloaded by cloud_setup.sh)
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=0 PYPI_RAG_READERS="${READERS[0]}" nb 04_supplychain_qa_colab 04_taskb_llama &
  CUDA_VISIBLE_DEVICES=1 PYPI_RAG_READERS="${READERS[1]}" nb 04_supplychain_qa_colab 04_taskb_qwen &
  wait
else
  nb 04_supplychain_qa_colab 04_taskb
fi
echo "Task B done: $PYPI_RAG_WORK/supplychain/runs"
