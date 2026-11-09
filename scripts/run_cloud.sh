#!/usr/bin/env bash
# Full pipeline, headless, on one cloud GPU machine: Task A (notebooks 00 -> 01) + Task B (notebook 04)
# -> notebooks 02, 03. Executed notebooks + logs go to $PYPI_RAG_WORK/executed/.
# Task B is REQUIRED for the seminar conclusions. On one machine it runs here; when the work is split over
# several pods (PYPI_RAG_DATASETS set), exactly one pod must run Task B: PYPI_RAG_TASK=b scripts/launch_pod.sh.
# 2+ GPUs: one reader per GPU, in parallel. 1 GPU: both readers in sequence.
# Start inside tmux so it survives a closed laptop:   tmux new -s run 'bash scripts/run_cloud.sh'
# Everything is cached: rerunning after an interruption continues where it stopped.
# Model steps run with HF_HUB_OFFLINE=1 (weights were downloaded by cloud_setup.sh): no network call while
# prompts that contain package source are processed.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$( [ -d /workspace ] && echo /workspace || echo "$HOME" )
export HF_HOME=${HF_HOME:-$ROOT/hf}
export PYPI_RAG_BASE=$(cd .. && pwd)                # parent of pypi_malware_rag/
export PYPI_RAG_WORK=${PYPI_RAG_WORK:-$ROOT/pypi_rag_work}
export PYPI_RAG_BACKEND=${PYPI_RAG_BACKEND:-vllm}
export TOKENIZERS_PARALLELISM=false
OUT=$PYPI_RAG_WORK/executed; mkdir -p "$OUT"
READERS=("NousResearch/Meta-Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-Coder-7B-Instruct")

nb() {  # nb <notebook> <output name>
  echo "[$(date +%H:%M:%S)] start $1 -> $OUT/$2.ipynb"
  jupyter nbconvert --to notebook --execute "notebooks/$1.ipynb" --output "$2" --output-dir "$OUT" \
      --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 > "$OUT/$2.log" 2>&1
  echo "[$(date +%H:%M:%S)] done  $1"
}

nb 00_data_knowledge_colab 00_data                  # CPU, needs network (upstream repos)

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # from here on: local weights only
NGPU=$(nvidia-smi -L | wc -l)
if [ "$NGPU" -ge 2 ]; then
  CUDA_VISIBLE_DEVICES=0 PYPI_RAG_READERS="${READERS[0]}" nb 01_experiments_colab 01_llama &
  CUDA_VISIBLE_DEVICES=1 PYPI_RAG_READERS="${READERS[1]}" nb 01_experiments_colab 01_qwen &
  wait
else
  nb 01_experiments_colab 01_all
fi

if [ -z "${PYPI_RAG_DATASETS:-}" ] && [ "${PYPI_RAG_SKIP_TASKB:-0}" != "1" ]; then
  bash scripts/run_taskb.sh                         # Task B on this machine
else
  echo "[run_cloud] Task B not run here (split run): one pod must run PYPI_RAG_TASK=b scripts/launch_pod.sh"
fi
nb 02_evidence_eval_colab 02_eval
nb 03_demo_seminar 03_demo
ls "$PYPI_RAG_WORK"/supplychain/runs/test_*/results.json >/dev/null 2>&1 \
  || echo "[run_cloud] WARNING: no Task B results on this machine; the seminar conclusions need them"
echo "All done. Results: $PYPI_RAG_WORK/results  | executed notebooks: $OUT"
echo "Copy back only aggregated files (results.json, report.md); datasets and Task A predictions hold malware source."
