#!/usr/bin/env bash
# The int8 (W8A16) GPU run on edmonds, on the RTX 2080 Ti GPU 0 only (GPU 2 may be busy with another
# benchmark; GPU 1, the A6000, is never used). Runs scripts/run_int8.sh in its own directory
# /data/axd200066/hipinfer_q8 under nohup, so a dropped connection does not stop it:
#   scripts/edmonds_int8.sh start    sync src/ tests/ scripts/ Makefile and start the run (refuses if GPU 0
#                                    already has a compute process, unless FORCE=1)
#   scripts/edmonds_int8.sh status   progress (the step lines of run_int8.log) and GPU 0's state
#   scripts/edmonds_int8.sh fetch    copy the run's results/2080ti/ (and the log) into this checkout
# Env: MODELS (default "tinyllama llama32-1b llama32-3b"), ONLY (run_int8.sh steps, default all).
# Writes: results/2080ti/test_kernels_int8.log, test_int8_cpu.log, int8_gpu_log.csv (SM clock,
# temperature, power, utilization every 5 s), hip_compile_int8.log, and results/2080ti/<model>/int8/.
# Expected time: several hours (three models x three gates, two perplexity runs and two benchmarks each).
set -euo pipefail
cd "$(dirname "$0")/.."
R=/data/axd200066/hipinfer_q8
MODELS=${MODELS:-tinyllama llama32-1b llama32-3b}
ONLY=${ONLY:-gate ppl bench llamacpp}
export REMOTE_DIR=$R GPUS=0
case "${1:-}" in
  start)
    apps=$(NOSYNC=1 bash scripts/edmonds.sh "mkdir -p $R && nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    if [ -n "$apps" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "GPU 0 is in use, not starting (FORCE=1 to start anyway):"; echo "$apps"; exit 1
    fi
    if NOSYNC=1 bash scripts/edmonds.sh "pgrep -f 'run_int8[.]sh 2080ti' >/dev/null"; then echo "a run is already going"; exit 1; fi
    bash scripts/edmonds.sh "mkdir -p weights gguf tmp results/2080ti && \
      (nohup nvidia-smi -i 0 --query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu \
         --format=csv -l 5 > results/2080ti/int8_gpu_log.csv 2>&1 < /dev/null & echo \$! > gpu_log.pid) && \
      nohup env HF_HUB_CACHE=/datastore/hub TMPDIR=$R/tmp PATH=/data/axd200066/envs/baselines/bin:\$PATH \
        WEIGHTS_DIR=/data/axd200066/models INT8_DIR=$R/weights GGUF_DIR=$R/gguf LLAMA_CPP=/data/axd200066/llama.cpp \
        LLAMA_F16=1 HIPCC=/data/axd200066/rocm_venv/bin/hipcc ONLY='$ONLY' \
        bash -c 'bash scripts/run_int8.sh 2080ti $MODELS; echo \"run_int8 exit \$?\"; kill \$(cat gpu_log.pid)' \
        > run_int8.log 2>&1 < /dev/null & echo started; sleep 5; tail -3 run_int8.log"
    ;;
  status)
    NOSYNC=1 bash scripts/edmonds.sh "grep -E '^(== |FAILED|all int8|run_int8 exit)|ALL PASS|FAILURES|GATE|ppl HF' run_int8.log | tail -40; \
      nvidia-smi -i 0 --query-gpu=clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv"
    ;;
  fetch)
    NOSYNC=1 bash scripts/edmonds.sh "tar -cf - results/2080ti run_int8.log" | tar -xf - -C .
    mv run_int8.log results/2080ti/run_int8.log
    git status --short results/
    ;;
  *)
    echo "usage: scripts/edmonds_int8.sh start|status|fetch" >&2; exit 2
    ;;
esac
