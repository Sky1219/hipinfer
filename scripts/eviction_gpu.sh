#!/usr/bin/env bash
# GPU steps for eviction scoring (docs/eviction.md) on edmonds, in its own directory
# /data/axd200066/hipinfer_evict, on one GPU that nothing else is using.
#
# From a local checkout (one ssh attempt per call; nothing retries):
#   scripts/eviction_gpu.sh check            nvidia-smi (every GPU, with its processes) and the thermal log
#   GPU=n scripts/eviction_gpu.sh start      sync src/ tests/ scripts/ Makefile, start the run with nohup
#   scripts/eviction_gpu.sh status           tail of the run log
#   scripts/eviction_gpu.sh fetch            copy results/2080ti/eviction/ back
# GPU is required and is never 1 (the RTX A6000). Other hipinfer sessions planned GPU 0 (int8) and
# GPU 2 (speculative decoding): run `check` first and pick a GPU with no processes.
#
# On edmonds (REMOTE=1, started by `start`): every step first prints nvidia-smi for its GPU and
# refuses to run if another process is on it; the newest thermal log in /data/axd200066/gpu_thermal/
# is shown before and after the benchmarks, with the GPU's clock-throttle reasons. Steps ($STEPS,
# default all, in order):
#   build    make for the GPU's arch
#   cpu      tests/test_eviction_cpu.py (logic emulation; SUBCACHE_DIR if a SubCache checkout exists)
#   kernels  tests/test_kernels.py, the existing kernel tests (must still print ALL PASS)
#   evict    tests/test_eviction.py with the default kernels, then the portable ones
#            (HIPINFER_DECODE_SCALAR=1 HIPINFER_FLASH_WMMA=1)
#   engine   tests/test_eviction_engine.py on $MODELS (default llama32-1b: D 64, qwen25-1.5b: D 128)
#   bench    scripts/bench_eviction.py --model <first of $MODELS>
# Results and logs: results/2080ti/eviction/ (on edmonds and, after fetch, locally).
set -uo pipefail
REMOTE_DIR=${REMOTE_DIR:-/data/axd200066/hipinfer_evict}
OUT=results/2080ti/eviction
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 edmonds"

if [ "${REMOTE:-0}" != 1 ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  cmd=${1:-}
  case "$cmd" in
    check)
      $SSH 'nvidia-smi; echo; f=$(ls -t /data/axd200066/gpu_thermal/* 2>/dev/null | head -1); echo "thermal log: ${f:-none}"; [ -n "$f" ] && tail -n 15 "$f"' ;;
    start)
      gpu=${GPU:-}
      case "$gpu" in ''|*[!0-9]*) echo "set GPU=<index> (run 'check' first)" >&2; exit 2;; esac
      [ "$gpu" = 1 ] && { echo "refusing: GPU 1 is the RTX A6000" >&2; exit 2; }
      tar -C "$HERE" --exclude=__pycache__ -cf - src tests scripts Makefile \
        | $SSH "mkdir -p $REMOTE_DIR && cd $REMOTE_DIR && rm -rf src tests scripts && tar -xmf -" || { echo "sync failed" >&2; exit 1; }
      $SSH "cd $REMOTE_DIR && mkdir -p $OUT && GPU=$gpu REMOTE=1 STEPS='${STEPS:-build cpu kernels evict engine bench}' \
            MODELS='${MODELS:-llama32-1b qwen25-1.5b}' setsid nohup bash scripts/eviction_gpu.sh > $OUT/run.log 2>&1 < /dev/null &
            sleep 2; tail -n 5 $OUT/run.log" ;;
    status)
      $SSH "tail -n ${LINES_:-40} $REMOTE_DIR/$OUT/run.log" ;;
    fetch)
      mkdir -p "$HERE/$OUT" && scp -q -r "edmonds:$REMOTE_DIR/$OUT/." "$HERE/$OUT/" && ls -l "$HERE/$OUT" ;;
    *)
      sed -n 2,26p "$0"; exit 2 ;;
  esac
  exit $?
fi

# ---------------------------------------------------------------- on edmonds
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU=${GPU:?}
[ "$GPU" = 1 ] && { echo "refusing: GPU 1 is the RTX A6000" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU
export TMPDIR=/data/axd200066/tmp
mkdir -p "$TMPDIR" "$OUT"
PY=${PY:-/data/axd200066/envs/baselines/bin/python}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
W=${WEIGHTS:-/data/axd200066/models}
STEPS=${STEPS:-build cpu kernels evict engine bench}
MODELS=${MODELS:-llama32-1b qwen25-1.5b}
FAILED=()
has() { [[ " $STEPS " == *" $1 "* ]]; }
thermal() {
  local f; f=$(ls -t /data/axd200066/gpu_thermal/* 2>/dev/null | head -1)
  echo "-- thermal log ${f:-(none found)}"; [ -n "$f" ] && tail -n 8 "$f"
  nvidia-smi -i "$GPU" -q -d PERFORMANCE | sed -n '/Clocks Event Reasons\|Clocks Throttle Reasons/,/^$/p'
}
# nvidia-smi before every job; refuse while any other process computes on this GPU
gpu_free() {
  echo "-- $(date -Is) nvidia-smi GPU $GPU before: $1"
  nvidia-smi -i "$GPU" --query-gpu=index,name,temperature.gpu,clocks.sm,clocks.max.sm,power.draw,memory.used --format=csv
  local apps; apps=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)
  if [ -n "$apps" ]; then echo "GPU $GPU is in use, not running $1:"; echo "$apps"; return 1; fi
}
step() {  # step NAME LOGFILE CMD...
  local name=$1 log=$2; shift 2
  gpu_free "$name" || { FAILED+=("$name (GPU busy)"); return; }
  echo "== $name -> $log"
  if "$@" > "$OUT/$log" 2>&1; then echo "   ok"; else echo "   FAILED (exit $?)"; FAILED+=("$name"); fi
  tail -n 4 "$OUT/$log"
}
echo "== eviction_gpu $(date -Is) on $(hostname), GPU $GPU, steps: $STEPS"
nvidia-smi
thermal

if has build; then
  CC=$($PY -c "import torch; a, b = torch.cuda.get_device_capability(); print(f'{a}{b}')")
  rm -f build/libhipinfer.so
  step build build.log make NVCC="$NVCC" NV_ARCH="-gencode arch=compute_$CC,code=sm_$CC"
  [ -f build/libhipinfer.so ] || { echo "no library; stopping"; exit 1; }
fi
if has cpu; then
  SC=""; [ -d /data/axd200066/SubCache/subcache ] && SC=/data/axd200066/SubCache
  step cpu test_eviction_cpu.log env SUBCACHE_DIR=$SC $PY tests/test_eviction_cpu.py
fi
has kernels && step kernels test_kernels.log $PY tests/test_kernels.py
if has evict; then
  step evict test_eviction.log $PY tests/test_eviction.py
  step evict-portable test_eviction_portable.log env HIPINFER_DECODE_SCALAR=1 HIPINFER_FLASH_WMMA=1 $PY tests/test_eviction.py
fi
if has engine; then
  for m in $MODELS; do
    [ -f "$W/$m.bin" ] || { echo "no $W/$m.bin; skipping engine test for $m"; FAILED+=("engine $m (no weights)"); continue; }
    step "engine $m" "test_eviction_engine_$m.log" $PY tests/test_eviction_engine.py "$W/$m.bin"
  done
fi
if has bench; then
  thermal
  m=${MODELS%% *}
  margs=(); [ -f "$W/$m.bin" ] && margs=(--model "$W/$m.bin")
  step bench bench_eviction.log $PY scripts/bench_eviction.py "${margs[@]}" --out "$OUT/bench_eviction.json"
  thermal
fi
nvidia-smi -i "$GPU" --query-gpu=index,temperature.gpu,clocks.sm --format=csv
echo "== done $(date -Is); failed: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
