#!/usr/bin/env bash
# Speculative decoding GPU steps (docs/speculative.md) on edmonds' RTX 2080 Ti GPU 2, and only GPU 2.
#
# From a local checkout:   scripts/run_spec.sh
#   syncs src/ tests/ scripts/ Makefile to $REMOTE_DIR (default /data/axd200066/hipinfer_spec) with
#   scripts/edmonds.sh, runs this script there (REMOTE=1), copies results/2080ti/spec/ back, and checks
#   that the regression gates give the same tokens as the committed results/2080ti/<model>/gate.json.
# On edmonds (REMOTE=1), the steps in $STEPS (default: all, in this order):
#   build    make for sm_75
#   logic    tests/test_spec_logic.py (CPU, the round loop with stub engines)
#   kernels  tests/test_kernels.py, must still print ALL PASS
#   gates    tests/gate.py for llama32-1b, llama32-3b (--hf-first, as committed) and tinyllama
#   spec     tests/test_spec.py: 3B <- 1B exact match for k in $KS; 1B and TinyLlama self-drafting
#   bench    scripts/bench_spec.py: 3B <- 1B for k in $BENCH_KS, and 1B <- itself (verify overhead)
# Benchmarks refuse to start while another process computes on GPU 2 (FORCE_BENCH=1 overrides).
# Env: STEPS, KS (2,4,6,8), BENCH_KS (1,2,3,4,5,6,8), NEW (128), REPS (3), REMOTE_DIR.
set -uo pipefail
GPU=2
STEPS=${STEPS:-build logic kernels gates spec bench}
KS=${KS:-2,4,6,8}
BENCH_KS=${BENCH_KS:-1,2,3,4,5,6,8}
NEW=${NEW:-128}
REPS=${REPS:-3}
OUT=results/2080ti/spec

if [ "${REMOTE:-0}" != 1 ]; then
  HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  REMOTE_DIR=${REMOTE_DIR:-/data/axd200066/hipinfer_spec}
  rc=0
  GPUS=$GPU REMOTE_DIR=$REMOTE_DIR "$HERE/scripts/edmonds.sh" \
    "REMOTE=1 STEPS='$STEPS' KS=$KS BENCH_KS=$BENCH_KS NEW=$NEW REPS=$REPS FORCE_BENCH=${FORCE_BENCH:-0} bash scripts/run_spec.sh" || rc=$?
  mkdir -p "$HERE/$OUT"
  scp -q -r "edmonds:$REMOTE_DIR/$OUT/." "$HERE/$OUT/" || { echo "copying results back failed"; exit 1; }
  # regression: the new gate runs must match the committed ones run for run (same verdicts, same divergence steps)
  python - "$HERE" <<'EOF' || rc=1
import json, os, sys
here = sys.argv[1]
ok = True
for m in ["llama32-1b", "llama32-3b", "tinyllama"]:
    new_p = os.path.join(here, "results/2080ti/spec/regression", f"{m}_gate.json")
    if not os.path.exists(new_p):
        print(f"{m}: no new gate result"); continue
    old, new = (json.load(open(p)) for p in (os.path.join(here, "results/2080ti", m, "gate.json"), new_p))
    # gate.py records step on matching runs too since the int8 merge: compare it only where a run diverged
    key = lambda j: [(r["run"], r["identical"], None if r["identical"] else r.get("step"), r.get("accepted")) for r in j["log"]]
    same = old["pass"] == new["pass"] and key(old) == key(new)
    ok &= same
    print(f"{m}: gate pass={new['pass']} {new['identical']}/{new['runs']} identical, "
          f"{'same as committed' if same else 'DIFFERS from committed results/2080ti/' + m + '/gate.json'}")
sys.exit(0 if ok else 1)
EOF
  echo "run_spec: exit $rc (results in $OUT)"
  exit $rc
fi

# ---------------------------------------------------------------- on edmonds
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ "${CUDA_VISIBLE_DEVICES:-}" != "$GPU" ] || [ "${CUDA_DEVICE_ORDER:-}" != PCI_BUS_ID ]; then
  echo "refusing to run: needs CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU" >&2; exit 2
fi
export HF_HUB_CACHE=${HF_HUB_CACHE:-/datastore/hub}
export TMPDIR=${TMPDIR_SPEC:-/data/axd200066/tmp}  # / and /home are nearly full
mkdir -p "$TMPDIR" "$OUT/regression"
PY=${PY:-/data/axd200066/envs/baselines/bin/python}
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
W=/data/axd200066/models
exec > >(tee -a "$OUT/run.log") 2>&1
echo "== run_spec $(date -Is) on $(hostname), GPU $GPU, steps: $STEPS"
nvidia-smi -i $GPU --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit,temperature.gpu --format=csv | tee "$OUT/gpu.txt"
FAILED=()
has() { [[ " $STEPS " == *" $1 "* ]]; }
others() { nvidia-smi -i $GPU --query-compute-apps=pid,used_memory --format=csv,noheader; }

if has build; then
  echo "== build (sm_75)"
  rm -f build/libhipinfer.so
  make NVCC="$NVCC" NV_ARCH="-gencode arch=compute_75,code=sm_75" 2>&1 | grep -vE "^mkdir|warning #|Remark|^ *\^|^$|never referenced|never used" || true
  test -f build/libhipinfer.so || { echo "build failed"; exit 1; }
fi
if has logic; then
  echo "== round-loop logic (CPU)"
  $PY tests/test_spec_logic.py > "$OUT/test_spec_logic.log" 2>&1 || FAILED+=("logic")
  tail -1 "$OUT/test_spec_logic.log"
fi
if has kernels; then
  echo "== kernel tests"
  $PY tests/test_kernels.py > "$OUT/regression/test_kernels.log" 2>&1 || FAILED+=("kernel tests")
  grep FAIL "$OUT/regression/test_kernels.log" || true
  tail -1 "$OUT/regression/test_kernels.log"
fi
if has gates; then
  for spec in "llama32-1b unsloth/Llama-3.2-1B" "llama32-3b unsloth/Llama-3.2-3B --hf-first" \
              "tinyllama TinyLlama/TinyLlama-1.1B-Chat-v1.0"; do
    set -- $spec
    m=$1 hf=$2; shift 2
    echo "== $m gate"
    $PY tests/gate.py --model "$hf" --weights "$W/$m.bin" "$@" --out "$OUT/regression/${m}_gate.json" \
      > "$OUT/regression/${m}_gate.log" 2>&1 || FAILED+=("$m gate")
    grep -E "runs token-identical|GATE" "$OUT/regression/${m}_gate.log" || tail -3 "$OUT/regression/${m}_gate.log"
  done
fi
if has spec; then
  echo "== exact-match test, k in $KS"
  $PY tests/test_spec.py --model unsloth/Llama-3.2-3B --target "$W/llama32-3b.bin" --draft "$W/llama32-1b.bin" \
    --self "$W/llama32-1b.bin" "$W/tinyllama.bin" --self-model unsloth/Llama-3.2-1B TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --ks "$KS" --new "$NEW" --out "$OUT/test_spec.json" > "$OUT/test_spec.log" 2>&1 || FAILED+=("spec test")
  grep -E "FAIL|runs pass|SPEC" "$OUT/test_spec.log" | tail -5
fi
if has bench; then
  busy=$(others)
  if [ -n "$busy" ] && [ "${FORCE_BENCH:-0}" != 1 ]; then
    echo "== bench skipped: another process is using GPU $GPU:"; echo "$busy"; FAILED+=("bench (GPU busy)")
  else
    echo "== bench: Llama-3.2-3B <- 1B draft"
    $PY scripts/bench_spec.py --model unsloth/Llama-3.2-3B --target "$W/llama32-3b.bin" --draft "$W/llama32-1b.bin" \
      --ks "$BENCH_KS" --new "$NEW" --reps "$REPS" --out "$OUT/llama32-3b_llama32-1b.json" \
      > "$OUT/llama32-3b_llama32-1b.log" 2>&1 || FAILED+=("bench 3b<-1b")
    grep -vE "Loading|^$" "$OUT/llama32-3b_llama32-1b.log" | tail -15
    echo "== bench: Llama-3.2-1B <- itself (every token accepted: the cost of verification)"
    $PY scripts/bench_spec.py --model unsloth/Llama-3.2-1B --target "$W/llama32-1b.bin" --draft "$W/llama32-1b.bin" \
      --ks 1,2,4,8 --new "$NEW" --reps "$REPS" --out "$OUT/llama32-1b_self.json" \
      > "$OUT/llama32-1b_self.log" 2>&1 || FAILED+=("bench 1b self")
    grep -vE "Loading|^$" "$OUT/llama32-1b_self.log" | tail -10
  fi
fi
nvidia-smi -i $GPU --query-gpu=clocks.sm,temperature.gpu --format=csv,noheader | sed 's/^/GPU at the end: /'
if [ ${#FAILED[@]} -gt 0 ]; then echo "== FAILED: ${FAILED[*]}"; exit 1; fi
echo "== all steps passed"
