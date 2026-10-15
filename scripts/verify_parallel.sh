#!/usr/bin/env bash
# Full verification of hipinfer on the two 2080 Tis, in parallel where it is safe. Run ON the server:
#   scripts/edmonds.sh 'nohup bash scripts/verify_parallel.sh > results/2080ti-wip/verify.out 2>&1 &'   (or in the foreground)
# from /data/axd200066/hipinfer_pf. Everything lands in results/<TAG> (default 2080ti-wip), never results/2080ti.
#
# Stages (each writes its own log in $OUT/_verify/):
#   0 prep    : scratch WEIGHTS_DIR of symlinks (mistral-7b -> mistral-7b-v03.bin), GPU/safety checks
#   1 build   : build once + tests/test_kernels.py once, GPU 0 only (run_all.sh with SKIP_SINGLE=1 SKIP_TP=1)
#   2 single  : two concurrent run_all.sh lanes, GPU 0 and GPU 2 (gate, full ppl, bench; SKIP_BUILD=1
#               SKIP_TP=1 SKIP_KERNEL_TESTS=1). Lane A (GPU 0): llama32-3b qwen25-0.5b; lane B (GPU 2):
#               qwen25-1.5b llama32-1b tinyllama
#   3 tp      : tensor-parallel stage on CUDA_VISIBLE_DEVICES=0,2 for all 7 models (SKIP_SINGLE=1)
#   4 compare : prefill rows (T=...) of $OUT vs results/2080ti, rows >1% slower flagged
# Exit code 1 if any stage failed (failures listed at the end, as run_all.sh does).
#
# Env: PY (python, default python), NVCC, TAG (default 2080ti-wip), BASE (comparison dir, default 2080ti),
#      MODELS_SRC (default /data/axd200066/models), WEIGHTS_DIR (scratch symlinks, default ./tmp_weights),
#      VP_WAIT_TIMEOUT (seconds to wait for a foreign process to leave a GPU before a benchmark, default 1800),
#      PPL_WINDOWS (default 16, passed through), VP_ALLOW_ANY_GPU=1 (skip the "GPUs 0 and 2 are 2080 Ti" check),
#      SINGLE_A / SINGLE_B (lane model lists), TP_ALL (TP model list).
# GPU rules: CUDA_DEVICE_ORDER=PCI_BUS_ID; every job is pinned to GPU 0 and/or 2; GPU 1 (RTX A6000) is never
#      touched, and the script refuses to start if CUDA_VISIBLE_DEVICES already lists it.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID
TAG=${TAG:-2080ti-wip}
BASE=${BASE:-2080ti}
OUT=results/$TAG
VP_WAIT_TIMEOUT=${VP_WAIT_TIMEOUT:-1800}

# ---- hook mode: called by run_all.sh (BENCH_HOOK) before every benchmark ---------------------------------------
# Waits (bounded) while a process owned by another user sits on one of $VP_GPUS (physical indices).
foreign_procs() {   # $1 = physical gpu index; prints "pid:user" for compute apps not owned by the current user
  local idx=$1 bus apps pid uid me
  me=$(id -u)   # numeric uids: `ps -o user=` truncates long names (axd200066 -> axd2000+)
  if ! bus=$(nvidia-smi -i "$idx" --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null) ||
     ! apps=$(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader,nounits 2>/dev/null); then
    echo "$(date '+%F %T') ${VP_LABEL:-bench}: nvidia-smi query failed for GPU $idx; cannot check for foreign processes" |
      tee -a "${VP_OUT:-.}/_verify/shared_gpu.log" >&2 || true
    return 0
  fi
  bus=$(echo "$bus" | tr -d ' ')
  echo "$apps" | tr -d ' ' | while IFS=, read -r b pid; do
    [ "$b" = "$bus" ] || continue
    [ -n "$pid" ] || continue
    uid=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ' || true)
    [ -n "$uid" ] || uid=unknown
    [ "$uid" = "$me" ] || echo "$pid:uid$uid"
  done || true
}
if [ "${1:-}" = "--wait-idle" ]; then
  gpus=${VP_GPUS:?VP_GPUS not set}
  log=${VP_OUT:?VP_OUT not set}/_verify/shared_gpu.log
  label=${VP_LABEL:-bench}
  # VP_WAIT_TIMEOUT is a total budget per label (lane): time spent waiting before earlier benches counts
  mkdir -p "$(dirname "$log")"
  usedf=$(dirname "$log")/wait_used_$label
  used=$(cat "$usedf" 2>/dev/null || echo 0)
  waited=$used
  while :; do
    busy=""
    for g in ${gpus//,/ }; do
      f=$(foreign_procs "$g" | tr '\n' ' ')
      [ -z "$f" ] || busy="$busy GPU$g[$f]"
    done
    if [ -z "$busy" ]; then echo "$waited" > "$usedf"; exit 0; fi
    if [ "$waited" -ge "$VP_WAIT_TIMEOUT" ]; then
      echo "$(date '+%F %T') $label: BENCHMARK RAN ON A SHARED GPU after waiting ${waited}s:$busy" | tee -a "$log"
      exit 0
    fi
    if [ $((waited % 60)) -eq 0 ]; then
      echo "$(date '+%T') $label: waiting for foreign processes to leave:$busy (${waited}s / ${VP_WAIT_TIMEOUT}s)"
    fi
    sleep 10; waited=$((waited + 10)); echo "$waited" > "$usedf"
  done
fi

# ---- main ------------------------------------------------------------------------------------------------------
PY=${PY:-python}
MODELS_SRC=${MODELS_SRC:-/data/axd200066/models}
WEIGHTS_DIR=${WEIGHTS_DIR:-$PWD/tmp_weights}
SINGLE_A=${SINGLE_A:-"llama32-3b qwen25-0.5b"}
SINGLE_B=${SINGLE_B:-"qwen25-1.5b llama32-1b tinyllama"}
TP_ALL=${TP_ALL:-"tinyllama llama32-1b qwen25-0.5b qwen25-1.5b llama32-3b mistral-7b llama31-8b"}
FAILED=()
if [ "$TAG" = 2080ti ] || [ "$TAG" = "$BASE" ]; then
  echo "refusing to run: TAG=$TAG must not be 2080ti or equal BASE=$BASE (it would overwrite the stored results)"; exit 2
fi
LOGS=$OUT/_verify
mkdir -p "$LOGS"
rm -f "$LOGS"/wait_used_* "$LOGS/shared_gpu.log"   # budgets and notes from an earlier run must not carry over
export PY WEIGHTS_DIR PPL_WINDOWS=${PPL_WINDOWS:-16}
[ -z "${NVCC:-}" ] || export NVCC

# GPU 1 is the user's A6000: refuse if it is visible, and never pass it on
CVD=${CUDA_VISIBLE_DEVICES:-}; CVD=${CVD//[[:space:]]/}
case "$CVD" in
  *[!0-9,]*) echo "refusing to run: CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES' is not a plain list of GPU indices"; exit 2;;
esac
for x in ${CVD//,/ }; do   # numeric compare, so "01" / "001" are GPU 1 too
  if [ "$((10#$x))" -eq 1 ]; then
    echo "refusing to run: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES includes GPU 1 (RTX A6000)"; exit 2
  fi
done
if [ "${VP_ALLOW_ANY_GPU:-0}" != 1 ]; then
  for g in 0 2; do
    n=$(nvidia-smi -i $g --query-gpu=name --format=csv,noheader)
    case "$n" in *2080*) ;; *) echo "GPU $g is '$n', not a 2080 Ti (set VP_ALLOW_ANY_GPU=1 to override)"; exit 2;; esac
  done
fi

# scratch weights dir: run_all expects <tag>.bin; mistral-7b is mistral-7b-v03.bin on the server
mkdir -p "$WEIGHTS_DIR"
for t in $TP_ALL; do
  src=$MODELS_SRC/$t.bin
  [ "$t" = mistral-7b ] && src=$MODELS_SRC/mistral-7b-v03.bin
  if [ ! -f "$src" ]; then echo "missing weights $src"; exit 2; fi
  ln -sfn "$src" "$WEIGHTS_DIR/$t.bin"
done

# SM clock / power log for the benchmark stages, always stopped on exit
CLK_PID=""
stop_clock() { [ -z "$CLK_PID" ] || { kill "$CLK_PID" 2>/dev/null || true; wait "$CLK_PID" 2>/dev/null || true; CLK_PID=""; }; }
start_clock() {   # $1 = stage name
  stop_clock
  nvidia-smi -i 0,2 -lms 1000 --format=csv \
    --query-gpu=timestamp,index,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
    > "$LOGS/sm_clock_$1.csv" 2>/dev/null &
  CLK_PID=$!
}
# every stage runs as its own session/process group so an interrupt can kill the python children on GPUs 0/2
PGIDS=()
kill_jobs() {   # non-interactive: a background job is not a group leader, so setsid does not fork and $! is itself
  # the session leader (with `setsid -w` the recorded pid is the waiting parent and its child leads the group)
  local p c
  for p in "${PGIDS[@]:-}"; do
    [ -n "$p" ] || continue
    # only signal pids that are still our own children (never a reused pid)
    [ "$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')" = "$$" ] || continue
    for c in $(pgrep -P "$p" 2>/dev/null || true) "$p"; do kill -TERM -- "-$c" 2>/dev/null || kill -TERM "$c" 2>/dev/null || true; done
  done
}
cleanup() { trap - EXIT INT TERM; stop_clock; kill_jobs; }
trap 'cleanup' EXIT
trap 'echo interrupted; cleanup; exit 130' INT TERM
SETSID=""
if command -v setsid >/dev/null 2>&1; then
  SETSID=setsid
  # -w makes setsid wait for the command (even if it has to fork), so `wait` returns the real exit code
  if setsid --help 2>&1 | grep -q -- '--wait'; then SETSID="setsid -w"; fi
fi
JOBPID=""
run_bg() {   # $1 = log, rest = command (env VAR=... allowed); sets JOBPID
  local log=$1; shift
  $SETSID "$@" > "$log" 2>&1 &
  JOBPID=$!; PGIDS+=("$JOBPID")
}

BENCH_HOOK_CMD="bash scripts/verify_parallel.sh --wait-idle"
export VP_OUT=$OUT VP_WAIT_TIMEOUT
SECONDS=0
stamp() { echo "[$(date '+%T') +${SECONDS}s] $*"; }
# pull "FAILED: a b c" of a run_all log into FAILED, or flag a log without the final "results in" line
collect() {   # $1 = stage label, $2 = exit code, $3 = log
  local line
  line=$(grep -E '^FAILED:' "$3" | tail -1 || true)
  if [ -n "$line" ]; then FAILED+=("$1: ${line#FAILED: }")
  elif [ "$2" -ne 0 ]; then FAILED+=("$1: run_all exited $2 (see $3)"); fi
}

stamp "stage 1: build + kernel tests (GPU 0)"
rc=0
run_bg "$LOGS/1_build_kernels.log" env CUDA_VISIBLE_DEVICES=0 SKIP_SINGLE=1 SKIP_TP=1 bash scripts/run_all.sh "$TAG"
wait "$JOBPID" || rc=$?
PGIDS=()
tail -3 "$LOGS/1_build_kernels.log"
collect "build/kernel tests" $rc "$LOGS/1_build_kernels.log"
if [ $rc -ne 0 ] || [ ! -f build/libhipinfer.so ]; then
  echo "build or kernel tests failed; not continuing to benchmarks"
  echo "FAILED: ${FAILED[*]}"; exit 1
fi

stamp "stage 2: single-GPU lanes (GPU 0: $SINGLE_A | GPU 2: $SINGLE_B)"
start_clock single
lane() {   # $1 = gpu, $2 = log, rest = models; sets JOBPID
  local g=$1 log=$2; shift 2
  run_bg "$log" env CUDA_VISIBLE_DEVICES=$g VP_GPUS=$g VP_LABEL="gpu$g" BENCH_HOOK="$BENCH_HOOK_CMD" \
    SKIP_BUILD=1 SKIP_TP=1 SKIP_KERNEL_TESTS=1 bash scripts/run_all.sh "$TAG" "$@"
}
# stale bench logs must not satisfy the comparison
for m in $SINGLE_A $SINGLE_B; do rm -f "$OUT/$m/bench.log"; done
rcA=0; rcB=0
lane 0 "$LOGS/2_single_gpu0.log" $SINGLE_A; pA=$JOBPID
lane 2 "$LOGS/2_single_gpu2.log" $SINGLE_B; pB=$JOBPID
wait "$pA" || rcA=$?
for i in "${!PGIDS[@]}"; do if [ "${PGIDS[$i]}" = "$pA" ]; then unset "PGIDS[$i]"; fi; done
wait "$pB" || rcB=$?
PGIDS=()
stop_clock
for m in $SINGLE_A $SINGLE_B; do
  [ -s "$OUT/$m/bench.log" ] || FAILED+=("$m: no fresh bench.log produced")
done
collect "single gpu0" $rcA "$LOGS/2_single_gpu0.log"
collect "single gpu2" $rcB "$LOGS/2_single_gpu2.log"
stamp "stage 2 done (rc gpu0=$rcA gpu2=$rcB)"

stamp "stage 3: tensor parallel x2 on GPUs 0,2: $TP_ALL"
start_clock tp
rc=0
run_bg "$LOGS/3_tp.log" env CUDA_VISIBLE_DEVICES=0,2 VP_GPUS=0,2 VP_LABEL=tp BENCH_HOOK="$BENCH_HOOK_CMD" \
  SKIP_BUILD=1 SKIP_SINGLE=1 SKIP_KERNEL_TESTS=1 bash scripts/run_all.sh "$TAG" $TP_ALL
wait "$JOBPID" || rc=$?
PGIDS=()
stop_clock
collect "tp2" $rc "$LOGS/3_tp.log"
stamp "stage 3 done (rc=$rc)"

stamp "stage 4: prefill comparison $OUT vs results/$BASE (rows >1% slower are flagged SLOWER)"
cmp_rows() {   # $1 = model
  local m=$1 new=$OUT/$1/bench.log old=results/$BASE/$1/bench.log
  [ -f "$new" ] || { printf '%-14s no bench.log in %s\n' "$m" "$OUT"; return 0; }
  [ -f "$old" ] || { printf '%-14s no baseline %s\n' "$m" "$old"; return 0; }
  awk -v m="$m" '
    function rows(file, dst,   line, a) {
      while ((getline line < file) > 0) {
        if (line !~ /^T= *[0-9]+ +hipinfer +[0-9.]+ ms/) continue
        sub(/^T= */, "", line); split(line, a, " "); dst[a[1]] = a[3]; if (!(a[1] in seen)) { seen[a[1]] = 1; order[++n] = a[1] }
      }
      close(file)
    }
    BEGIN {
      rows(ARGV[1], newv); rows(ARGV[2], oldv); ARGV[1] = ARGV[2] = ""
      for (i = 1; i <= n; i++) {
        t = order[i]
        if (!(t in newv) || !(t in oldv)) { printf "%-14s T=%-5s missing in %s%s\n", m, t, (t in newv ? "baseline" : "new"), (t in newv ? "" : " MISSING"); continue }
        cmpd++
        r = newv[t] / oldv[t]
        printf "%-14s T=%-5s %9.2f -> %9.2f ms  x%.3f %s\n", m, t, oldv[t], newv[t], r, (r > 1.01 ? "SLOWER" : "")
        if (r > 1.01) slow++
      }
      if (slow) print "  ^ " m ": " slow " row(s) more than 1% slower"
      printf "  %s: %d row(s) compared\n", m, cmpd
      if (cmpd == 0) { print "  NOROWS " m ": no comparable T= rows"; exit 1 }
    }' "$new" "$old"
}
in_single() { case " $SINGLE_A $SINGLE_B " in *" $1 "*) return 0;; esac; return 1; }
for m in $TP_ALL; do   # TP-only models (7B/8B) have no bench.log in either tree: informational
  cmp_rows "$m" || { in_single "$m" && FAILED+=("$m: prefill comparison had no comparable rows"); }
done > "$LOGS/4_compare.txt"
cat "$LOGS/4_compare.txt"
if grep -q MISSING "$LOGS/4_compare.txt"; then FAILED+=("prefill rows missing from new results (see $LOGS/4_compare.txt)"); fi
if grep -q SLOWER "$LOGS/4_compare.txt"; then FAILED+=("prefill rows >1% slower (see $LOGS/4_compare.txt)"); fi
[ ! -f "$LOGS/shared_gpu.log" ] || { echo "NOTE: some benchmarks ran on a shared GPU:"; cat "$LOGS/shared_gpu.log"; }

stamp "results in $OUT   (logs in $LOGS)"
if [ ${#FAILED[@]} -gt 0 ]; then
  printf 'FAILED:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "ALL OK"
