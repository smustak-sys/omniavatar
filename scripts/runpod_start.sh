#!/usr/bin/env bash
# Start the OmniAvatar FastAPI server on a RunPod pod.
# Assumes weights have already been downloaded into ./pretrained_models.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export OMNI_REPO_ROOT="$REPO_ROOT"
export OMNI_STORAGE="${OMNI_STORAGE:-$REPO_ROOT/storage}"
export OMNI_CONFIG="${OMNI_CONFIG:-$REPO_ROOT/configs/inference.yaml}"
export OMNI_NPROC="${OMNI_NPROC:-1}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OMNI_STORAGE/jobs"

PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo ">> OmniAvatar API on $HOST:$PORT"
echo ">> Repo:    $OMNI_REPO_ROOT"
echo ">> Storage: $OMNI_STORAGE"
echo ">> Config:  $OMNI_CONFIG"
exec uvicorn server.app:app --host "$HOST" --port "$PORT"
