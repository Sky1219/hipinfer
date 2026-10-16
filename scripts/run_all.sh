#!/usr/bin/env bash
# Build for the visible GPU, run the kernel tests once, then for every model that fits in GPU
# memory: export weights (if missing), HF token gate, WikiText-2 perplexity vs HF, benchmarks.
#
# Usage: scripts/run_all.sh [gpu_tag] [model_tag ...]      results land in results/<gpu_tag>/<model_tag>/
# Env:   PY (python), WEIGHTS_DIR (where <model_tag>.bin live, default .), SKIP_BUILD=1, SKIP_BENCH=1,
#        SKIP_PPL=1, GEMM_SWEEP=1 (tile sweep for the first model), PPL_WINDOWS (default 16)
#        SKIP_KERNEL_TESTS=1 (skip tests/test_kernels.py), SKIP_SINGLE=1 (skip the single-GPU per-model
#        stage; build + kernel tests + TP only), BENCH_HOOK=cmd (run before every benchmark; used by
#        scripts/verify_parallel.sh to wait for a shared GPU)
# Multi-GPU (2+ visible GPUs): afterwards every model in TP_MODELS (default: the same models) also
#        runs tensor-parallel over TP GPUs (default 2): TP-vs-1-GPU test, TP benchmarks, and for
#        models that only fit sharded, perplexity vs HF. SKIP_TP=1 skips this stage.
# A failing step does not stop the run; the failures are listed at the end and the exit code is 1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
WEIGHTS_DIR=${WEIGHTS_DIR:-.}

declare -A HF=(
  [tinyllama]=TinyLlama/TinyLlama-1.1B-Chat-v1.0
  [llama32-1b]=unsloth/Llama-3.2-1B
  [qwen25-0.5b]=Qwen/Qwen2.5-0.5B
  [qwen25-1.5b]=Qwen/Qwen2.5-1.5B
  [llama32-3b]=unsloth/Llama-3.2-3B
  [llama31-8b]=unsloth/Meta-Llama-3.1-8B
  [mistral-7b]=mistralai/Mistral-7B-v0.3   # v0.3: no sliding window (v0.1's 4,096-token window is not implemented)
)
# fp16 weight bytes, used to skip models that cannot sit next to the HF reference copy
declare -A GB=([tinyllama]=2.2 [llama32-1b]=2.5 [qwen25-0.5b]=1.0 [qwen25-1.5b]=3.1 [llama32-3b]=6.4 [llama31-8b]=16.1 [mistral-7b]=14.5)
DEFAULT_MODELS="tinyllama llama32-1b qwen25-0.5b qwen25-1.5b llama32-3b mistral-7b llama31-8b"

GPU=$($PY -c "import torch; print(torch.cuda.get_device_name().replace(' ', '_'))")
MEM_GB=$($PY -c "import torch; print(torch.cuda.get_device_properties(0).total_memory / 2**30)")
TAG=${1:-$GPU}
shift || true
MODELS=${*:-$DEFAULT_MODELS}
OUT=results/$TAG
mkdir -p "$OUT"
FAILED=()

if $PY -c "import torch, sys; sys.exit(0 if torch.version.hip else 1)"; then
  # AMD / ROCm: hipcc + rocWMMA, wavefront size picked per target at compile time
  GFX=$($PY -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")
  HIPCC=${HIPCC:-$(command -v hipcc || echo /opt/rocm/bin/hipcc)}
  echo "GPU $GPU ($MEM_GB GB)  $GFX  $($HIPCC --version | grep -m1 'HIP version')"
  # RCCL (large-message all-reduce backend for multi-GPU) from the ROCm install, when present
  ROCM=$(dirname "$(dirname "$(readlink -f "$HIPCC")")")
  RCCL=""
  ls "$ROCM"/lib/librccl.so* >/dev/null 2>&1 && RCCL=1
  if [ "${SKIP_BUILD:-0}" != 1 ]; then
    make hip HIPCC="$HIPCC" AMD_ARCH="--offload-arch=$GFX" RCCL="$RCCL"
    cp build/libhipinfer_amd.so build/libhipinfer.so
  fi
else
  CC=$($PY -c "import torch; a, b = torch.cuda.get_device_capability(); print(f'{a}{b}')")
  NVCC=${NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}
  NVVER=$($NVCC --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1.\2/p')
  echo "GPU $GPU ($MEM_GB GB)  sm_$CC  nvcc $NVVER"
  # Native SASS for this GPU when the toolkit knows the arch, otherwise compute_80 PTX that
  # the driver JIT-compiles (e.g. Blackwell sm_120 needs CUDA >= 12.8).
  if $NVCC --list-gpu-arch 2>/dev/null | grep -q "compute_$CC"; then
    ARCH="-gencode arch=compute_$CC,code=sm_$CC"
  else
    echo "nvcc has no sm_$CC; building compute_80 PTX for driver JIT"
    ARCH="-gencode arch=compute_80,code=compute_80"
  fi
  # NCCL (large-message all-reduce backend for multi-GPU) from the pip wheel that ships with torch
  NCCL=$($PY -c "import nvidia.nccl as m, os; d = m.__path__[0]; print(d if os.path.exists(d + '/include/nccl.h') else '')" 2>/dev/null || true)
  if [ "${SKIP_BUILD:-0}" != 1 ]; then
    rm -f build/libhipinfer.so
    make NVCC="$NVCC" NV_ARCH="$ARCH" NCCL="$NCCL" 2>&1 | grep -vE "^mkdir" || true
  fi
fi
test -f build/libhipinfer.so

if [ "${SKIP_KERNEL_TESTS:-0}" != 1 ]; then
echo "== kernel tests"
if $PY tests/test_kernels.py > "$OUT/test_kernels.log" 2>&1; then tail -1 "$OUT/test_kernels.log"
else grep FAIL "$OUT/test_kernels.log" || true; tail -3 "$OUT/test_kernels.log"; FAILED+=("kernel tests"); fi
fi

# GB one engine needs for model $1 on one GPU; export model $1's weights unless present
need_engine() { $PY -c "print(${GB[$1]} * 1.15 + 1.5)"; }
bench_hook() { [ -z "${BENCH_HOOK:-}" ] || $BENCH_HOOK || true; }
ensure_weights() { [ -f "$WEIGHTS_DIR/$1.bin" ] || $PY scripts/export.py --model "${HF[$1]}" --out "$WEIGHTS_DIR/$1.bin"; }

first=1
SINGLE_MODELS=$MODELS; [ "${SKIP_SINGLE:-0}" != 1 ] || SINGLE_MODELS=""
for m in $SINGLE_MODELS; do
  hf=${HF[$m]:-}
  if [ -z "$hf" ]; then echo "unknown model tag $m (known: ${!HF[*]})"; exit 1; fi
  need=$($PY -c "print(${GB[$m]} * 2.3 + 3)")       # engine + HF reference copy + activations
  need1=$(need_engine $m)                            # engine alone
  NOHF=""
  if $PY -c "import sys; sys.exit(0 if $MEM_GB < $need1 else 1)"; then
    echo "== $m: needs ~${need1} GB, GPU has $MEM_GB GB -> skipped"
    continue
  elif $PY -c "import sys; sys.exit(0 if $MEM_GB < $need else 1)"; then
    echo "== $m: the HF reference copy does not fit next to the engine ($MEM_GB GB) -> HF reference first (--hf-first) for gate/ppl, hipinfer-only benchmarks"
    NOHF="--no-hf"
  fi
  W=$WEIGHTS_DIR/$m.bin
  MO=$OUT/$m
  mkdir -p "$MO"
  ensure_weights $m
  echo "== $m ($hf): HF token gate ${NOHF:+(HF first, then engine)}"
  $PY tests/gate.py --model "$hf" --weights "$W" ${NOHF:+--hf-first} --out "$MO/gate.json" > "$MO/gate.log" 2>&1 || FAILED+=("$m gate")
  grep -E "runs token-identical|GATE|Error" "$MO/gate.log" || tail -3 "$MO/gate.log"
  if [ "${SKIP_PPL:-0}" != 1 ]; then
    echo "== $m: WikiText-2 perplexity vs HF"
    $PY tests/ppl.py --model "$hf" --weights "$W" ${NOHF:+--hf-first} --windows "${PPL_WINDOWS:-16}" --out "$MO/ppl.json" > "$MO/ppl.log" 2>&1       || { tail -3 "$MO/ppl.log"; FAILED+=("$m ppl"); }
    grep -E "^window +[0-9]+:" "$MO/ppl.log" | tail -1 || true
  fi
  if [ "${SKIP_BENCH:-0}" != 1 ]; then
    echo "== $m: benchmarks"
    bench_hook
    EXTRA=""
    [ "${GEMM_SWEEP:-0}" = 1 ] && [ $first = 1 ] && EXTRA="--gemm-sweep"
    $PY scripts/bench.py --model "$hf" --weights "$W" --out "$MO/bench.json" $EXTRA $NOHF > "$MO/bench.log" 2>&1       || { tail -5 "$MO/bench.log"; FAILED+=("$m bench"); }
    grep -E "^(T=|ctx=|B=)" "$MO/bench.log" || true
  fi
  first=0
done

NGPU=$($PY -c "import torch; print(torch.cuda.device_count())")
TP=${TP:-2}
if [ "$NGPU" -ge "$TP" ] && [ "$TP" -ge 2 ] && [ "${SKIP_TP:-0}" != 1 ]; then
  echo "== tensor parallel x$TP over $NGPU GPUs"
  for m in ${TP_MODELS:-$MODELS}; do
    hf=${HF[$m]:-}
    need1=$(need_engine $m)
    needtp=$($PY -c "print(${GB[$m]} / $TP + 1.5)")   # shard (incl. replicated embedding) + KV cache + activations
    if $PY -c "import sys; sys.exit(0 if $MEM_GB < $needtp else 1)"; then
      echo "== $m: needs ~${needtp} GB per GPU at tp=$TP -> skipped"; continue
    fi
    REF=""
    $PY -c "import sys; sys.exit(0 if $MEM_GB < $need1 else 1)" && REF="--no-ref"
    W=$WEIGHTS_DIR/$m.bin
    MO=$OUT/$m
    mkdir -p "$MO"
    ensure_weights $m
    echo "== $m: tensor parallel x$TP vs 1 GPU ${REF:+(sharded only: no 1-GPU reference)}"
    $PY tests/test_tp.py --weights "$W" --tp $TP $REF --out "$MO/tp${TP}_test.json" > "$MO/tp${TP}_test.log" 2>&1       || FAILED+=("$m tp$TP test")
    grep -E "self-test|prefill logits|runs token-identical|bitwise|TP TEST|Error" "$MO/tp${TP}_test.log" || tail -3 "$MO/tp${TP}_test.log"
    if [ -n "$REF" ] && [ "${SKIP_PPL:-0}" != 1 ]; then
      echo "== $m: WikiText-2 perplexity vs HF, engine sharded x$TP"
      $PY tests/ppl.py --model "$hf" --weights "$W" --tp $TP --windows "${PPL_WINDOWS:-16}" --out "$MO/ppl_tp${TP}.json" \
        > "$MO/ppl_tp${TP}.log" 2>&1 || { tail -3 "$MO/ppl_tp${TP}.log"; FAILED+=("$m ppl tp$TP"); }
      grep -E "^window +[0-9]+:" "$MO/ppl_tp${TP}.log" | tail -1 || true
    fi
    if [ "${SKIP_BENCH:-0}" != 1 ]; then
      echo "== $m: tensor-parallel benchmarks"
      bench_hook
      NOSINGLE=""; [ -n "$REF" ] && NOSINGLE="--no-single"
      $PY scripts/bench_tp.py --weights "$W" --tp $TP $NOSINGLE --out "$MO/tp${TP}.json" > "$MO/tp${TP}.log" 2>&1         || { tail -5 "$MO/tp${TP}.log"; FAILED+=("$m tp$TP bench"); }
      grep -E "decode B=1 ctx=  128|decode B=1 ctx= 1024|prefill T= 2000|batched B= 64|^decode B=1 " "$MO/tp${TP}.log" || true
    fi
  done
fi
echo "results in $OUT"
if [ ${#FAILED[@]} -gt 0 ]; then
  echo "FAILED: ${FAILED[*]}"
  exit 1
fi
