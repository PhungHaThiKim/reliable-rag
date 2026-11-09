#!/usr/bin/env bash
# Unattended run on one RunPod pod: setup if needed, notebooks 00 -> 03, then stop the pod after a grace
# period (time to copy results). Split work between pods with environment variables:
#   PYPI_RAG_DATASETS=pyguard_obf PYPI_RAG_EXTRAS=0 bash scripts/launch_pod.sh     # pod 1
#   PYPI_RAG_DATASETS=malllm      PYPI_RAG_EXTRAS=1 bash scripts/launch_pod.sh     # pod 2 (synthetic, cases, repeat)
#   PYPI_RAG_TASK=b bash scripts/launch_pod.sh                                         # pod 3: Task B only (notebook 04)
# PYPI_RAG_GRACE=0 keeps the pod running.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$( [ -d /workspace ] && echo /workspace || echo "$HOME" )
export PYPI_RAG_WORK=${PYPI_RAG_WORK:-$ROOT/pypi_rag_work}
mkdir -p "$PYPI_RAG_WORK"
LOG=$PYPI_RAG_WORK/launch.log
{
  echo "[launch] start $(date) datasets=${PYPI_RAG_DATASETS:-all} extras=${PYPI_RAG_EXTRAS:-1}"
  python3 -c "import vllm" 2>/dev/null || bash scripts/cloud_setup.sh
  if [ "${PYPI_RAG_TASK:-a}" = "b" ]; then bash scripts/run_taskb.sh; else bash scripts/run_cloud.sh; fi
  echo "[launch] finished $(date)"
} 2>&1 | tee -a "$LOG"
GRACE=${PYPI_RAG_GRACE:-2700}
if [ "$GRACE" -gt 0 ] && command -v runpodctl >/dev/null && [ -n "${RUNPOD_POD_ID:-}" ]; then
  echo "[launch] stopping pod $RUNPOD_POD_ID in ${GRACE}s" | tee -a "$LOG"
  sleep "$GRACE"
  runpodctl stop pod "$RUNPOD_POD_ID"
fi
