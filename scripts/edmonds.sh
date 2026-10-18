#!/usr/bin/env bash
# Usage: [NOSYNC=1] [GPUS=0] [REMOTE_DIR=...] scripts/edmonds.sh 'COMMAND STRING'
# Callers pass ONE single-quoted command string; it is expanded and run on the
# remote (edmonds) in $REMOTE_DIR (default /data/axd200066/hipinfer_pf) with PY, NVCC and
# CUDA_VISIBLE_DEVICES=${GPUS:-0} exported (PCI_BUS_ID ordering). Exit code is the remote's.
# By default src/, tests/, scripts/ and the Makefile are re-synced first, in one tar|ssh pipe: the
# three dirs are cleared remotely and extracted in place (no *.bin, notebooks or results ship). A
# failed local tar leaves the remote dirs empty until the next sync (pipefail makes it fail loudly).
# NOSYNC=1 skips the sync entirely (use it for read-only calls and while a run is in progress on
# the server). GPU 1 (RTX A6000) is never allowed.
set -euo pipefail
if [ $# -ne 1 ]; then
  echo "usage: $0 'remote command string (single-quoted)'" >&2
  exit 2
fi
gpus="${GPUS:-0}"
gpus="${gpus//[[:space:]]/}"
for x in ${gpus//,/ }; do
  case "$x" in
    ''|*[!0-9]*) echo "refusing to run: GPUS='${GPUS:-0}' has a non-numeric entry" >&2; exit 2;;
  esac
  if [ "$((10#$x))" -eq 1 ]; then
    echo "refusing to run: GPUS='${GPUS:-0}' includes GPU 1 (RTX A6000)" >&2; exit 2
  fi
done
[ -n "$gpus" ] || { echo "refusing to run: GPUS is empty" >&2; exit 2; }
REMOTE_DIR=${REMOTE_DIR:-/data/axd200066/hipinfer_pf}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${NOSYNC:-0}" != 1 ]; then
  tar -C "$HERE" --exclude=__pycache__ -cf - src tests scripts Makefile \
    | ssh -o BatchMode=yes edmonds "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && rm -rf src tests scripts && tar -xmf -"
fi
ssh -o BatchMode=yes edmonds "cd $REMOTE_DIR && export CUDA_DEVICE_ORDER=PCI_BUS_ID PY=/data/axd200066/envs/baselines/bin/python NVCC=/usr/local/cuda/bin/nvcc CUDA_VISIBLE_DEVICES=$gpus && $1"
