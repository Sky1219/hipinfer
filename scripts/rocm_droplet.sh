#!/usr/bin/env bash
# One command for a fresh AMD GPU machine (e.g. an MI300X cloud droplet with ROCm and PyTorch):
# checks the environment, builds for the GPU, then runs the suite in order of what matters most,
# so stopping early still leaves usable results. Results: results/<tag>/ and results/baselines/<tag>/,
# packed into hipinfer_results_<tag>.tgz at the end.
#
# Usage (from the repo root, in the environment whose python has ROCm PyTorch, e.g. inside the
# image's Docker container):   scripts/rocm_droplet.sh
# Getting the code there from a local checkout (the same commit, nothing to push):
#   git archive --format=tar.gz HEAD | ssh root@<ip> "mkdir -p hipinfer && tar xzf - -C hipinfer"
# and, if PyTorch is in a container:  docker cp hipinfer <container>:/root/  then docker exec -it <container> bash
# Env: STAGES   subset/order of: check core big moe baselines pack   (default: all, in that order)
#        check      ROCm PyTorch, hipcc, rocWMMA, GPU arch, bandwidth entry; installs the few pip
#                   packages the tests need (never torch)
#        core       build, kernel tests, then TinyLlama and Llama-3.2-1B (HF token gate, perplexity,
#                   benchmarks): the "it runs on AMD" result
#        big        Llama-3.1-8B, then the other dense models (Mistral only with HF access)
#        moe        Granite MoE 1B: export, token gate, perplexity, benchmarks
#        baselines  llama.cpp (HIP build), vLLM (if importable), torch.compile and HF on
#                   BASELINE_MODELS, with hipinfer re-run beside them
#        pack       summary and the results archive
#      TAG (default from the GPU name, e.g. mi300x), PY (default python), WEIGHTS_DIR (default weights),
#      BASELINE_MODELS (default "tinyllama llama32-1b llama31-8b"), RATE (USD per GPU-hour, for the
#      cost line after each stage; default 1.99)
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
STAGES=${STAGES:-check core big moe baselines pack}
export WEIGHTS_DIR=${WEIGHTS_DIR:-weights}
BASELINE_MODELS=${BASELINE_MODELS:-tinyllama llama32-1b llama31-8b}
RATE=${RATE:-1.99}
mkdir -p "$WEIGHTS_DIR"
T0=$(date +%s)
want() { [[ " $STAGES " == *" $1 "* ]]; }
stage() { echo; echo "######## $* ($(date +%T), $(( ($(date +%s) - T0) / 60 )) min in)"; }
cost() { $PY -c "import sys; m = ($(date +%s) - $T0) / 60; print(f'-- {m:.0f} min so far, about \${m / 60 * $RATE:.2f} at \$$RATE/h')"; }

if ! $PY -c "import torch, sys; sys.exit(0 if torch.version.hip and torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "$PY has no ROCm PyTorch with a visible GPU."
  echo "On a Quick Start image, PyTorch lives in the Docker container: docker ps, then"
  echo "docker exec -it <container> bash, and run this script from the repo inside it."
  exit 1
fi
TAG=${TAG:-$($PY -c "
import re, torch
n = torch.cuda.get_device_name()
m = re.search(r'(MI\d+\w*|W\d+|RX \d+ ?\w*)', n)
print((m.group(1) if m else n).lower().replace(' ', '_'))")}
OUT=results/$TAG
mkdir -p "$OUT"

# GPU log for the whole run (temperature, power, clocks every 5 s), to check for throttling later
GPULOG=$OUT/gpu_log.txt
( while :; do echo "== $(date +%FT%T)"; (amd-smi metric -t -p -c -u 2>/dev/null || rocm-smi --showtemp --showpower --showclocks --showuse 2>/dev/null) | grep -v "^$"; sleep 5; done ) >> "$GPULOG" 2>&1 &
LOGGER=$!
trap 'kill $LOGGER 2>/dev/null' EXIT

if want check; then
  stage check
  $PY - <<'EOF'
import ast, re, torch
p = torch.cuda.get_device_properties(0)
name = torch.cuda.get_device_name()
print(f"GPU {name}  {p.gcnArchName}  {p.total_memory / 2**30:.0f} GB  "
      f"{torch.cuda.device_count()} visible  torch {torch.__version__}  HIP {torch.version.hip}")
# bench.py's table, read without importing it (that would load the library, not built yet)
known = ast.literal_eval(re.search(r"KNOWN_BW = (\{.*?\})", open("scripts/bench.py").read(), re.S).group(1))
bw = next((v for k, v in known.items() if k in name), None)
print(f"peak bandwidth entry: {bw} GB/s" if bw else
      "WARNING: no bandwidth entry for this GPU in scripts/bench.py KNOWN_BW: the %-of-floor columns will be empty")
EOF
  HIPCC=$(command -v hipcc || echo /opt/rocm/bin/hipcc)
  [ -x "$HIPCC" ] || { echo "no hipcc: install ROCm's HIP SDK (apt install rocm-hip-sdk)"; exit 1; }
  "$HIPCC" --version | grep -m1 "HIP version"
  ROCM=$(dirname "$(dirname "$(readlink -f "$HIPCC")")")
  if [ ! -f "$ROCM/include/rocwmma/rocwmma.hpp" ] && [ ! -f /opt/rocm/include/rocwmma/rocwmma.hpp ]; then
    echo "no rocWMMA headers (needed by the build): apt install rocwmma-dev"; exit 1
  fi
  $PY -c "import transformers; print('transformers', transformers.__version__)"
  $PY -m pip install -q datasets sentencepiece protobuf 2>&1 | tail -2   # no torch: the image's ROCm build stays
  df -h . | tail -1
  cost
fi

if want core; then
  stage "core: build, kernel tests, TinyLlama, Llama-3.2-1B"
  PY=$PY GEMM_SWEEP=1 scripts/run_all.sh "$TAG" tinyllama llama32-1b || echo "core: some steps failed (listed above)"
  cost
fi

if want big; then
  stage "big: Llama-3.1-8B, then the other dense models"
  MORE="llama31-8b qwen25-0.5b qwen25-1.5b llama32-3b"
  if $PY -c "from huggingface_hub import auth_check; auth_check('mistralai/Mistral-7B-v0.3')" 2>/dev/null; then
    MORE="$MORE mistral-7b"
  else
    echo "no HF access to mistralai/Mistral-7B-v0.3 (gated): skipping it (huggingface-cli login to include it)"
  fi
  PY=$PY SKIP_BUILD=1 SKIP_KERNEL_TESTS=1 scripts/run_all.sh "$TAG" $MORE || echo "big: some steps failed (listed above)"
  cost
fi

if want moe; then
  stage "moe: Granite MoE 1B"
  HFM=ibm-granite/granite-3.1-1b-a400m-base
  W=$WEIGHTS_DIR/granite-moe-1b.bin
  MO=$OUT/granite-moe-1b
  mkdir -p "$MO"
  [ -f "$W" ] || $PY scripts/export.py --model $HFM --out "$W" 2>&1 | tail -2
  $PY -c "import sys; sys.path[:0] = ['tests']; import test_kernels as t; t.test_moe(); print('MoE kernel FAILS', t.FAILS)" \
    > "$MO/test_moe.log" 2>&1; tail -1 "$MO/test_moe.log"
  $PY tests/gate.py --model $HFM --weights "$W" --out "$MO/gate.json" > "$MO/gate.log" 2>&1
  grep -E "runs token-identical|GATE" "$MO/gate.log" || tail -3 "$MO/gate.log"
  $PY tests/ppl.py --model $HFM --weights "$W" --out "$MO/ppl.json" > "$MO/ppl.log" 2>&1
  grep -E "^window +[0-9]+:" "$MO/ppl.log" | tail -1 || tail -3 "$MO/ppl.log"
  $PY scripts/bench.py --model $HFM --weights "$W" --skip-kernels --max-ctx 4096 --out "$MO/bench.json" > "$MO/bench.log" 2>&1
  grep -E "^(T=|ctx=)" "$MO/bench.log" | head -4 || tail -3 "$MO/bench.log"
  cost
fi

if want baselines; then
  stage "baselines: llama.cpp, vLLM, torch.compile, HF on: $BASELINE_MODELS"
  $PY -m pip install -q gguf 2>&1 | tail -1    # llama.cpp's converter; its requirements file would replace torch
  VLLM_PY=""
  if $PY -c "import vllm" 2>/dev/null; then VLLM_PY=$(command -v "$PY"); else echo "vLLM not importable: skipped"; fi
  declare -A HFID=([tinyllama]=TinyLlama/TinyLlama-1.1B-Chat-v1.0 [llama32-1b]=unsloth/Llama-3.2-1B
                   [qwen25-1.5b]=Qwen/Qwen2.5-1.5B [llama32-3b]=unsloth/Llama-3.2-3B [llama31-8b]=unsloth/Meta-Llama-3.1-8B)
  mkdir -p gguf
  for m in $BASELINE_MODELS; do
    [ -f "$WEIGHTS_DIR/$m.bin" ] || { echo "== $m: no hipinfer weights (run the core/big stage first), skipped"; continue; }
    echo "== $m"
    PY=$PY LLAMA_CPP=${LLAMA_CPP:-../llama.cpp} GGUF=gguf/$m-f16.gguf MODEL=${HFID[$m]} WEIGHTS=$WEIGHTS_DIR/$m.bin \
      VLLM_PY=$VLLM_PY scripts/run_baselines.sh "$TAG/$m" || echo "$m baselines: failed (see results/baselines/$TAG/$m/)"
  done
  cost
fi

if want pack; then
  stage pack
  $PY scripts/summarize.py results > "$OUT/SUMMARY.md" 2>&1 || true
  tar czf "hipinfer_results_$TAG.tgz" "$OUT" $( [ -d "results/baselines/$TAG" ] && echo "results/baselines/$TAG" )
  ls -la "hipinfer_results_$TAG.tgz"
  echo "Copy it off before destroying the machine:  scp <user>@<ip>:$(pwd)/hipinfer_results_$TAG.tgz ."
  cost
fi
