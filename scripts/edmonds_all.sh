#!/usr/bin/env bash
# Every GPU check of the merged tree on edmonds' two RTX 2080 Tis, in one go (GPU 1, the A6000, is
# never used). Two queues run side by side under nohup, each in its own copy of the tree
# ($R/gpu0, $R/gpu2) so their builds don't collide, then one step needs both GPUs:
#   GPU 0  scripts/run_int8.sh (kernel tests, int8 gates / perplexity / benchmarks, llama.cpp Q8_0),
#          then scripts/eviction_gpu.sh (eviction kernels, engine test, benchmark)
#   GPU 2  scripts/run_spec.sh (sm_75 build, kernel tests, fp16 regression gates against the
#          committed results, speculative exact-match test, speculative benchmark)
#   both   tests/test_tp.py on Llama-3.2-3B with --tp 2, once both queues are done
# From a local checkout:
#   scripts/edmonds_all.sh check    nvidia-smi on edmonds (every GPU and its processes)
#   scripts/edmonds_all.sh start    sync src/ tests/ scripts/ Makefile into both copies and start
#                                   (refuses if GPU 0 or 2 has a compute process, unless FORCE=1)
#   scripts/edmonds_all.sh status   step lines of every log, and the two GPUs' state
#   scripts/edmonds_all.sh fetch    copy results/2080ti/ from both copies into this checkout, and
#                                   compare the new fp16 gates with the committed ones
# Env: MODELS (int8 models, default "tinyllama llama32-1b llama32-3b"), INT8_ONLY (run_int8.sh
#      steps), SPEC_STEPS (run_spec.sh steps), EVICT_STEPS (eviction_gpu.sh steps), SKIP_TP=1.
# Expected time: GPU 0 several hours (int8 is most of it), GPU 2 one to two hours.
# (_q0, _q2 and _tp are the queues themselves, run on edmonds by `start`.)
set -euo pipefail
cd "$(dirname "$0")/.."
R=/data/axd200066/hipinfer_all
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 edmonds"
MODELS=${MODELS:-tinyllama llama32-1b llama32-3b}
INT8_ONLY=${INT8_ONLY:-gate ppl bench llamacpp}
SPEC_STEPS=${SPEC_STEPS:-build logic kernels gates spec bench}
EVICT_STEPS=${EVICT_STEPS:-build cpu kernels evict engine bench}
on_edmonds() {
  export CUDA_DEVICE_ORDER=PCI_BUS_ID PY=/data/axd200066/envs/baselines/bin/python NVCC=/usr/local/cuda/bin/nvcc \
    HF_HUB_CACHE=/datastore/hub PATH=/data/axd200066/envs/baselines/bin:$PATH TMPDIR=$PWD/tmp
  mkdir -p tmp results/2080ti
}

case "${1:-}" in
  check)
    $SSH nvidia-smi
    ;;
  start)
    apps=$($SSH "nvidia-smi -i 0,2 --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader")
    if [ -n "$apps" ] && [ "${FORCE:-0}" != 1 ]; then
      echo "GPU 0 or 2 is in use, not starting (FORCE=1 to start anyway):"; echo "$apps"; exit 1
    fi
    if $SSH "pgrep -f 'edmonds_all[.]sh _' >/dev/null"; then echo "a run is already going"; exit 1; fi
    for d in gpu0 gpu2; do
      tar --exclude=__pycache__ -cf - src tests scripts Makefile \
        | $SSH "mkdir -p $R/$d && cd $R/$d && rm -rf src tests scripts && tar -xmf -"
    done
    vars="MODELS='$MODELS' INT8_ONLY='$INT8_ONLY' SPEC_STEPS='$SPEC_STEPS' EVICT_STEPS='$EVICT_STEPS' SKIP_TP=${SKIP_TP:-0}"
    # each whole background job redirected, so none of them holds the ssh session open
    $SSH "rm -f $R/gpu0/done $R/gpu2/done
      (cd $R/gpu0 && $vars setsid nohup bash scripts/edmonds_all.sh _q0) > $R/gpu0/queue.log 2>&1 < /dev/null &
      (cd $R/gpu2 && $vars setsid nohup bash scripts/edmonds_all.sh _q2) > $R/gpu2/queue.log 2>&1 < /dev/null &
      (cd $R/gpu2 && $vars setsid nohup bash scripts/edmonds_all.sh _tp) > $R/gpu2/queue_tp.log 2>&1 < /dev/null &
      sleep 10; tail -n 3 $R/gpu0/run_int8.log $R/gpu2/run_spec.log"
    ;;
  _q0)  # on edmonds, in $R/gpu0: int8, then eviction (its steps refuse while another process is on the GPU)
    on_edmonds
    export CUDA_VISIBLE_DEVICES=0
    mkdir -p weights gguf results/2080ti/eviction
    rc=0
    WEIGHTS_DIR=/data/axd200066/models INT8_DIR=$PWD/weights GGUF_DIR=$PWD/gguf LLAMA_CPP=/data/axd200066/llama.cpp \
      LLAMA_F16=1 HIPCC=/data/axd200066/rocm_venv/bin/hipcc ONLY="$INT8_ONLY" \
      bash scripts/run_int8.sh 2080ti $MODELS > run_int8.log 2>&1 || rc=$?
    echo "run_int8 exit $rc" >> run_int8.log
    rc=0
    GPU=0 REMOTE=1 STEPS="$EVICT_STEPS" bash scripts/eviction_gpu.sh > results/2080ti/eviction/run.log 2>&1 || rc=$?
    echo "eviction exit $rc" >> results/2080ti/eviction/run.log
    touch done
    ;;
  _q2)  # on edmonds, in $R/gpu2: speculative decoding and the fp16 regression gates
    on_edmonds
    export CUDA_VISIBLE_DEVICES=2 TMPDIR_SPEC=$PWD/tmp
    rc=0
    REMOTE=1 STEPS="$SPEC_STEPS" bash scripts/run_spec.sh > run_spec.log 2>&1 || rc=$?
    echo "run_spec exit $rc" >> run_spec.log
    touch done
    ;;
  _tp)  # on edmonds, in $R/gpu2 (its sm_75 build): tensor parallelism once both queues are done
    on_edmonds
    while [ ! -f $R/gpu0/done ] || [ ! -f $R/gpu2/done ]; do sleep 60; done
    [ "${SKIP_TP:-0}" = 1 ] && exit 0
    export CUDA_VISIBLE_DEVICES=0,2
    rc=0
    $PY tests/test_tp.py --weights /data/axd200066/models/llama32-3b.bin --tp 2 > results/2080ti/test_tp_merged.log 2>&1 || rc=$?
    echo "test_tp exit $rc" >> results/2080ti/test_tp_merged.log
    ;;
  status)
    $SSH "cd $R; for f in gpu0/run_int8.log gpu0/results/2080ti/eviction/run.log gpu2/run_spec.log gpu2/results/2080ti/test_tp_merged.log; do
        [ -f \$f ] || continue; echo \"=== \$f\"
        grep -E '^(== |FAILED|all int8|run_int8 exit|eviction exit|run_spec exit|test_tp exit)|ALL PASS|FAILURES|GATE|   ok|   FAILED' \$f | tail -25
      done
      ls gpu0/done gpu2/done 2>/dev/null
      nvidia-smi -i 0,2 --query-gpu=index,clocks.sm,temperature.gpu,utilization.gpu,memory.used --format=csv"
    ;;
  fetch)
    $SSH "cd $R/gpu0 && tar -cf - results/2080ti run_int8.log" | tar -xf - -C .
    mv run_int8.log results/2080ti/run_int8.log
    $SSH "cd $R/gpu2 && tar -cf - results/2080ti run_spec.log" | tar -xf - -C .
    mv run_spec.log results/2080ti/run_spec.log
    # run_spec.sh's regression check: the new fp16 gates must match the committed ones run for run
    python - . <<'EOF' || echo "gate regression: DIFFERS (see above)"
import json, os, sys
here = sys.argv[1]
ok = True
for m in ["llama32-1b", "llama32-3b", "tinyllama"]:
    new_p = os.path.join(here, "results/2080ti/spec/regression", f"{m}_gate.json")
    if not os.path.exists(new_p):
        print(f"{m}: no new gate result"); ok = False; continue
    old, new = (json.load(open(p)) for p in (os.path.join(here, "results/2080ti", m, "gate.json"), new_p))
    key = lambda j: [(r["run"], r["identical"], r.get("step"), r.get("accepted")) for r in j["log"]]
    same = old["pass"] == new["pass"] and key(old) == key(new)
    ok &= same
    print(f"{m}: gate pass={new['pass']} {new['identical']}/{new['runs']} identical, "
          f"{'same as committed' if same else 'DIFFERS from committed'}")
sys.exit(0 if ok else 1)
EOF
    git status --short results/
    ;;
  *)
    sed -n 2,21p "$0"; exit 2
    ;;
esac
