#!/usr/bin/env bash
# Re-run, on edmonds, the merged-tree benchmarks whose windows the clock log showed thermally
# throttled (results/2080ti/MERGED_RUN.md): Llama-3.2-3B int8 / fp16 / llama.cpp on GPU 0, then
# the speculative benchmarks on GPU 2. One GPU at a time, each step started on a card below 60 C
# with no compute process on it, with every GPU's clocks logged every 2 s.
# Run in a synced copy of the tree (e.g. /data/axd200066/hipinfer_all/gpu0) with its sm_75 build:
#   bash scripts/rerun_throttled.sh > rerun_throttled.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_DEVICE_ORDER=PCI_BUS_ID PY=${PY:-/data/axd200066/envs/baselines/bin/python} HF_HUB_CACHE=/datastore/hub \
  TMPDIR=$PWD/tmp PATH=/data/axd200066/envs/baselines/bin:$PATH
W=/data/axd200066/models
O=results/2080ti/rerun
mkdir -p tmp $O
nvidia-smi -i 0,1,2 --query-gpu=timestamp,index,clocks.sm,temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.active \
  --format=csv -l 2 > $O/gpu_log.csv 2>&1 < /dev/null &
logger=$!
trap 'kill $logger' EXIT
cool() {  # cool GPU: wait (up to 15 min) for the card to fall below 60 C with nothing on it
  local g=$1 t apps i
  for i in $(seq 180); do
    t=$(nvidia-smi -i "$g" --query-gpu=temperature.gpu --format=csv,noheader)
    apps=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader)
    [ -z "$apps" ] && [ "$t" -lt 60 ] && { echo "-- $(date +%T) GPU $g at $t C, idle"; return 0; }
    sleep 5
  done
  echo "-- GPU $g did not cool below 60 C (at $t C, apps: $apps)"; return 1
}
run() {  # run NAME GPU CMD...: one window, its start and end in $O/windows.txt
  local name=$1 g=$2; shift 2
  cool "$g" || { echo "== $name skipped"; return; }
  echo "$name start $(date +%T)" >> $O/windows.txt
  echo "== $name"
  CUDA_VISIBLE_DEVICES=$g "$@" || echo "   FAILED: $name"
  echo "$name end $(date +%T)" >> $O/windows.txt
}
m=llama32-3b hf=unsloth/Llama-3.2-3B
run "3b bench int8" 0 $PY scripts/bench.py --model $hf --weights weights/$m-int8.bin --no-hf --out $O/${m}_bench_int8.json \
  > $O/${m}_bench_int8.log 2>&1
run "3b bench fp16" 0 $PY scripts/bench.py --model $hf --weights $W/$m.bin --no-hf --skip-kernels --out $O/${m}_bench_fp16.json \
  > $O/${m}_bench_fp16.log 2>&1
for t in q8_0 f16; do
  run "3b llama.cpp $t" 0 env OUT=$O ONLY=llamacpp MODEL=$hf LLAMA_CPP=/data/axd200066/llama.cpp GGUF=gguf/$m-$t.gguf \
    GGUF_TYPE=$t LLAMA_NAME=${m}_llamacpp_$t PY=$PY bash scripts/run_baselines.sh 2080ti > $O/${m}_llamacpp_$t.log 2>&1
done
run "spec 3B<-1B" 2 $PY scripts/bench_spec.py --model $hf --target $W/llama32-3b.bin --draft $W/llama32-1b.bin \
  --ks 1,2,3,4,5,6,8 --new 128 --reps 3 --out $O/spec_llama32-3b_llama32-1b.json > $O/spec_llama32-3b_llama32-1b.log 2>&1
run "spec 1B<-1B" 2 $PY scripts/bench_spec.py --model unsloth/Llama-3.2-1B --target $W/llama32-1b.bin --draft $W/llama32-1b.bin \
  --ks 1,2,4,8 --new 128 --reps 3 --out $O/spec_llama32-1b_self.json > $O/spec_llama32-1b_self.log 2>&1
echo "== done $(date +%T)"
