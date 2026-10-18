#!/usr/bin/env bash
# int8 weights (W8A16, docs/quantization.md) on the visible GPU: build, kernel tests, then per model
#   export     <tag>-int8.bin (scripts/export.py --int8), if missing
#   gate       token gates: int8 vs fp16 HF (divergence allowed, reported), int8 vs HF on the same
#              int8 round-trip weights (strict), and the fp16 weights on this build (unchanged path)
#   ppl        WikiText-2 perplexity and KL: int8 vs fp16 HF, and int8 vs fake-quantized HF
#   bench      decode / prefill benchmarks of the int8 and the fp16 weights, back to back (no HF)
#   llamacpp   llama.cpp Q8_0 (and F16 when that GGUF exists) via scripts/run_baselines.sh
# Usage: scripts/run_int8.sh <gpu_tag> [model_tag ...]      results land in results/<gpu_tag>/<model_tag>/int8/
# Env:   PY, WEIGHTS_DIR (fp16 <tag>.bin), INT8_DIR (<tag>-int8.bin, exported if missing; default
#        WEIGHTS_DIR), SKIP_BUILD=1,
#        SKIP_KERNEL_TESTS=1, ONLY (subset of "gate ppl bench llamacpp"), PPL_WINDOWS (default 16),
#        LLAMA_CPP (llama.cpp checkout with build/bin/llama-bench; llamacpp skipped if unset),
#        GGUF_DIR (<tag>-q8_0.gguf, converted if missing), LLAMA_F16=1 (also <tag>-f16.gguf, converted
#        if missing, for a same-session F16 row; default 0: only when that file exists)
# A failing step does not stop the run; failures are listed at the end and the exit code is 1.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
WEIGHTS_DIR=${WEIGHTS_DIR:-.}
INT8_DIR=${INT8_DIR:-$WEIGHTS_DIR}
declare -A HF=(
  [tinyllama]=TinyLlama/TinyLlama-1.1B-Chat-v1.0
  [llama32-1b]=unsloth/Llama-3.2-1B
  [llama32-3b]=unsloth/Llama-3.2-3B
  [qwen25-0.5b]=Qwen/Qwen2.5-0.5B
  [qwen25-1.5b]=Qwen/Qwen2.5-1.5B
  [llama31-8b]=unsloth/Meta-Llama-3.1-8B
)
TAG=${1:?usage: run_int8.sh <gpu_tag> [model_tag ...]}
shift
MODELS=${*:-tinyllama llama32-1b llama32-3b}
ONLY=${ONLY:-gate ppl bench llamacpp}
want() { [[ " $ONLY " == *" $1 "* ]]; }
FAILED=()
step() {  # step <name> <log> cmd...: run, log, record a failure
  local name=$1 log=$2; shift 2
  echo "== $name"
  if "$@" > "$log" 2>&1; then tail -2 "$log"; else tail -5 "$log"; FAILED+=("$name"); fi
}

if [ "${SKIP_BUILD:-0}" != 1 ]; then
  CC=$($PY -c "import torch; a, b = torch.cuda.get_device_capability(); print(f'{a}{b}')")
  rm -f build/libhipinfer.so
  make NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}" NV_ARCH="-gencode arch=compute_$CC,code=sm_$CC"
fi
mkdir -p "results/$TAG"
if [ "${SKIP_KERNEL_TESTS:-0}" != 1 ]; then
  step "int8 CPU checks" "results/$TAG/test_int8_cpu.log" $PY tests/test_int8_cpu.py
  step "kernel tests (fp16 + int8)" "results/$TAG/test_kernels_int8.log" $PY tests/test_kernels.py
fi

for m in $MODELS; do
  model=${HF[$m]}
  fp16="$WEIGHTS_DIR/$m.bin"
  int8="$INT8_DIR/$m-int8.bin"
  out="results/$TAG/$m/int8"
  mkdir -p "$out"
  [ -f "$int8" ] || step "$m export --int8" "$out/export.log" $PY scripts/export.py --model "$model" --out "$int8" --int8
  if want gate; then
    step "$m gate int8 vs fp16 HF" "$out/gate.log" $PY tests/gate.py --model "$model" --weights "$int8" --out "$out/gate.json"
    step "$m gate int8 vs int8 HF" "$out/gate_int8ref.log" $PY tests/gate.py --model "$model" --weights "$int8" --ref int8 --out "$out/gate_int8ref.json"
    [ -f "$fp16" ] && step "$m gate fp16 (this build)" "$out/gate_fp16_weights.log" $PY tests/gate.py --model "$model" --weights "$fp16" --out "$out/gate_fp16_weights.json"
  fi
  if want ppl; then
    step "$m ppl int8 vs fp16 HF" "$out/ppl.log" $PY tests/ppl.py --model "$model" --weights "$int8" --windows "${PPL_WINDOWS:-16}" --out "$out/ppl.json"
    step "$m ppl int8 vs int8 HF" "$out/ppl_int8ref.log" $PY tests/ppl.py --model "$model" --weights "$int8" --ref int8 --windows "${PPL_WINDOWS:-16}" --out "$out/ppl_int8ref.json"
  fi
  if want bench; then
    step "$m bench int8" "$out/bench.log" $PY scripts/bench.py --model "$model" --weights "$int8" --no-hf --out "$out/bench.json"
    [ -f "$fp16" ] && step "$m bench fp16 (same session)" "$out/bench_fp16.log" $PY scripts/bench.py --model "$model" --weights "$fp16" --no-hf --skip-kernels --out "$out/bench_fp16.json"
  fi
  if want llamacpp && [ -n "${LLAMA_CPP:-}" ]; then
    GGUF_DIR=${GGUF_DIR:-$WEIGHTS_DIR}
    step "$m llama.cpp Q8_0" "$out/run_baselines_q8_0.log" env OUT="$out" ONLY=llamacpp MODEL="$model" LLAMA_CPP="$LLAMA_CPP" \
      GGUF="$GGUF_DIR/$m-q8_0.gguf" GGUF_TYPE=q8_0 LLAMA_NAME=llamacpp_q8_0 PY="$PY" scripts/run_baselines.sh "$TAG"
    if [ -f "$GGUF_DIR/$m-f16.gguf" ] || [ "${LLAMA_F16:-0}" = 1 ]; then
      step "$m llama.cpp F16" "$out/run_baselines_f16.log" env OUT="$out" ONLY=llamacpp MODEL="$model" LLAMA_CPP="$LLAMA_CPP" \
        GGUF="$GGUF_DIR/$m-f16.gguf" GGUF_TYPE=f16 LLAMA_NAME=llamacpp_f16 PY="$PY" scripts/run_baselines.sh "$TAG"
    fi
  fi
done

# HIPCC set: compile-only check of the AMD build (wave64 gfx90a and wave32 gfx1100), into its own
# library so the NVIDIA one stays in place
if [ -n "${HIPCC:-}" ]; then
  step "AMD compile check" "results/$TAG/hip_compile_int8.log" \
    $HIPCC -O3 -std=c++17 -fPIC -fvisibility=hidden -shared --offload-arch=gfx90a --offload-arch=gfx1100 -x hip src/api.cu \
    -lpthread -o build/libhipinfer_amd_check.so
fi

if [ ${#FAILED[@]} -gt 0 ]; then echo "FAILED: ${FAILED[*]}"; exit 1; fi
echo "all int8 steps passed"
