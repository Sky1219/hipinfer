#!/usr/bin/env bash
# Strong baselines next to hipinfer, on the visible GPU. Results land in results/baselines/<tag>/.
# Usage: scripts/run_baselines.sh <tag>     e.g. CUDA_VISIBLE_DEVICES=1 scripts/run_baselines.sh a6000
# Env:   PY (python with torch + transformers), LLAMA_CPP (llama.cpp checkout; built if needed),
#        GGUF (F16 GGUF; converted if missing), VLLM_PY (python with vllm; skipped if unset),
#        WEIGHTS (hipinfer .bin; hipinfer is re-run in the same session for a same-day comparison),
#        GGUF_TYPE (convert_hf_to_gguf --outtype for a missing GGUF, default f16; q8_0 for the int8
#        comparison), OUT (result directory), LLAMA_NAME / HIPINFER_NAME (result file names)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: run_baselines.sh <tag>}
PY=${PY:-python}
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}
GGUF=${GGUF:-$LLAMA_CPP/tinyllama-1.1b-chat-f16.gguf}
WEIGHTS=${WEIGHTS:-tinyllama.bin}
GGUF_TYPE=${GGUF_TYPE:-f16}
OUT=${OUT:-results/baselines/$TAG}
LLAMA_NAME=${LLAMA_NAME:-llamacpp}
HIPINFER_NAME=${HIPINFER_NAME:-hipinfer}
mkdir -p "$OUT"
# inductor/triton caches and temp files off a possibly small /tmp
SCRATCH=${SCRATCH:-$(cd .. && pwd)/tmp}
mkdir -p "$SCRATCH"
export TMPDIR=${TMPDIR:-$SCRATCH} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH/inductor} \
       TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/triton} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SCRATCH/vllm}

ONLY=${ONLY:-hipinfer llamacpp torch_compile hf_static vllm}   # subset to (re)run
want() { [[ " $ONLY " == *" $1 "* ]]; }

# llama.cpp: GPU build + F16 GGUF (GGUF_TYPE=q8_0: its 8-bit format, for hipinfer's int8 weights). Ninja when installed.
#  NVIDIA: CUDA build. Host compiler gcc-11 when present (CUDA 12.4 rejects newer gcc), else the
#    default. CUDA_ARCHS narrows the build to the GPU at hand (e.g. "80" on an A100), which is
#    several times faster than the default list.
#  AMD (PyTorch is a ROCm build): HIP build with ROCm's clang for GPU_TARGETS (default: the visible
#    GPU's gfx arch, e.g. gfx942 on an MI300X), plus rocWMMA flash attention when its headers are
#    installed, so llama.cpp gets its fastest AMD path too.
if want llamacpp && [ ! -x "$LLAMA_CPP/build/bin/llama-bench" ]; then
  [ -d "$LLAMA_CPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
  GEN=(); command -v ninja >/dev/null && GEN=(-G Ninja)
  if $PY -c "import torch, sys; sys.exit(0 if torch.version.hip else 1)"; then
    GFX=${GPU_TARGETS:-$($PY -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(':')[0])")}
    ROCM=$(hipconfig -R)
    WMMA=OFF; [ -f "$ROCM/include/rocwmma/rocwmma.hpp" ] && WMMA=ON
    HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$ROCM" cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" "${GEN[@]}" \
      -DGGML_HIP=ON -DGPU_TARGETS="$GFX" -DGGML_HIP_ROCWMMA_FATTN=$WMMA -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  else
    HOST_CC=${CC:-$(command -v gcc-11 || command -v gcc)}
    HOST_CXX=${CXX:-$(command -v g++-11 || command -v g++)}
    cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" "${GEN[@]}" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS:-75;80;86;89}" \
      -DCMAKE_C_COMPILER="$HOST_CC" -DCMAKE_CXX_COMPILER="$HOST_CXX" -DCMAKE_CUDA_HOST_COMPILER="$HOST_CXX" \
      -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  fi
  cmake --build "$LLAMA_CPP/build" -j --target llama-bench
fi
if want llamacpp && [ ! -f "$GGUF" ]; then
  SNAP=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL'))")
  $PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$SNAP" --outtype "$GGUF_TYPE" --outfile "$GGUF"
fi

run() { echo "== $1"; "${@:2}" 2>&1 | grep -vE "Loading weights|Warning: You are sending" | tee "$OUT/$1.log"; }

[ -f "$WEIGHTS" ] && want hipinfer && run "$HIPINFER_NAME" $PY scripts/bench.py --weights "$WEIGHTS" --model "$MODEL" --skip-kernels ${HIPINFER_ARGS:-} --out "$OUT/$HIPINFER_NAME.json"
want llamacpp && { run "$LLAMA_NAME" $PY scripts/bench_baselines.py llamacpp --llama-bench "$LLAMA_CPP/build/bin/llama-bench" --gguf "$GGUF" --out "$OUT/$LLAMA_NAME.json" || echo "llamacpp failed (see log)"; }
want torch_compile && { run torch_compile $PY scripts/bench_baselines.py torch_compile --model "$MODEL" --out "$OUT/torch_compile.json" || echo "torch_compile failed (see log)"; }
want hf_static && { run hf_static $PY scripts/bench_baselines.py hf_static --model "$MODEL" --out "$OUT/hf_static.json" || echo "hf_static failed (see log)"; }
if [ -n "${VLLM_PY:-}" ] && want vllm; then
  # greedy only: skip the FlashInfer sampler (JIT-compiled with nvcc on first use)
  # HF_HOME pinned so vLLM reuses the HF download already on disk (XDG_CACHE_HOME alone would move it)
  VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} XDG_CACHE_HOME=$SCRATCH/cache PATH="$(dirname "$VLLM_PY"):$PATH" run vllm "$VLLM_PY" scripts/bench_baselines.py vllm --model "$MODEL" --out "$OUT/vllm.json" || echo "vllm failed (see log)"
fi
