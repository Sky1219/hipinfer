#!/usr/bin/env bash
# Strong baselines next to hipinfer, on the visible GPU. Results land in results/baselines/<tag>/.
# Usage: scripts/run_baselines.sh <tag>     e.g. CUDA_VISIBLE_DEVICES=1 scripts/run_baselines.sh a6000
# Env:   PY (python with torch + transformers), LLAMA_CPP (llama.cpp checkout; built if needed),
#        GGUF (F16 GGUF; converted if missing), VLLM_PY (python with vllm; skipped if unset),
#        WEIGHTS (hipinfer .bin; hipinfer is re-run in the same session for a same-day comparison)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: run_baselines.sh <tag>}
PY=${PY:-python}
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
LLAMA_CPP=${LLAMA_CPP:-../llama.cpp}
GGUF=${GGUF:-$LLAMA_CPP/tinyllama-1.1b-chat-f16.gguf}
WEIGHTS=${WEIGHTS:-tinyllama.bin}
OUT=results/baselines/$TAG
mkdir -p "$OUT"
# inductor/triton caches and temp files off a possibly small /tmp
SCRATCH=${SCRATCH:-$(cd .. && pwd)/tmp}
mkdir -p "$SCRATCH"
export TMPDIR=${TMPDIR:-$SCRATCH} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$SCRATCH/inductor} \
       TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/triton} VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$SCRATCH/vllm}

ONLY=${ONLY:-hipinfer llamacpp torch_compile hf_static vllm}   # subset to (re)run
want() { [[ " $ONLY " == *" $1 "* ]]; }

# llama.cpp: CUDA build (gcc-11 host compiler for CUDA 12.4) + F16 GGUF, no quantization
if want llamacpp && [ ! -x "$LLAMA_CPP/build/bin/llama-bench" ]; then
  [ -d "$LLAMA_CPP" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_CPP"
  cmake -S "$LLAMA_CPP" -B "$LLAMA_CPP/build" -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
    -DCMAKE_C_COMPILER=${CC:-gcc-11} -DCMAKE_CXX_COMPILER=${CXX:-g++-11} -DCMAKE_CUDA_HOST_COMPILER=${CXX:-g++-11} \
    -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$LLAMA_CPP/build" -j --target llama-bench
fi
if want llamacpp && [ ! -f "$GGUF" ]; then
  SNAP=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('$MODEL'))")
  $PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$SNAP" --outtype f16 --outfile "$GGUF"
fi

run() { echo "== $1"; "${@:2}" 2>&1 | grep -vE "Loading weights|Warning: You are sending" | tee "$OUT/$1.log"; }

[ -f "$WEIGHTS" ] && want hipinfer && run hipinfer $PY scripts/bench.py --weights "$WEIGHTS" --model "$MODEL" --skip-kernels --out "$OUT/hipinfer.json"
want llamacpp && { run llamacpp $PY scripts/bench_baselines.py llamacpp --llama-bench "$LLAMA_CPP/build/bin/llama-bench" --gguf "$GGUF" --out "$OUT/llamacpp.json" || echo "llamacpp failed (see log)"; }
want torch_compile && { run torch_compile $PY scripts/bench_baselines.py torch_compile --model "$MODEL" --out "$OUT/torch_compile.json" || echo "torch_compile failed (see log)"; }
want hf_static && { run hf_static $PY scripts/bench_baselines.py hf_static --model "$MODEL" --out "$OUT/hf_static.json" || echo "hf_static failed (see log)"; }
if [ -n "${VLLM_PY:-}" ] && want vllm; then
  # greedy only: skip the FlashInfer sampler (JIT-compiled with nvcc on first use)
  VLLM_USE_FLASHINFER_SAMPLER=0 XDG_CACHE_HOME=$SCRATCH/cache PATH="$(dirname "$VLLM_PY"):$PATH" run vllm "$VLLM_PY" scripts/bench_baselines.py vllm --model "$MODEL" --out "$OUT/vllm.json" || echo "vllm failed (see log)"
fi
