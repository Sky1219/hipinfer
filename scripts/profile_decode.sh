#!/usr/bin/env bash
# Profile batch-1 decode of one model with Nsight Systems (see scripts/profile_decode.py).
# Usage: scripts/profile_decode.sh <out_dir> <weights.bin> [ctx] [steps]
# Env:   PY (python), NSYS (nsys binary; default: the one on PATH)
# Writes <out_dir>/plain.json (no profiler), under_nsys.json, decode.nsys-rep, decode.sqlite,
# stats_*.csv (nsys stats tables, restricted to the decode_graph NVTX range when nsys supports it)
# and summary.json.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:?usage: profile_decode.sh <out_dir> <weights.bin> [ctx] [steps]}
W=${2:?weights}
CTX=${3:-128}
STEPS=${4:-128}
PY=${PY:-python}
NSYS=${NSYS:-$(command -v nsys || true)}
[ -x "$NSYS" ] || { echo "nsys not found (set NSYS=/path/to/nsys)"; exit 1; }
mkdir -p "$OUT"
"$NSYS" --version | tee "$OUT/nsys_version.txt"

echo "== reference timing, no profiler"
$PY scripts/profile_decode.py run --weights "$W" --ctx "$CTX" --steps "$STEPS" --out "$OUT/plain.json"

echo "== nsys: CUDA + NVTX trace, graph nodes traced one by one"
"$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --force-overwrite=true -o "$OUT/decode" \
  $PY scripts/profile_decode.py run --weights "$W" --ctx "$CTX" --steps "$STEPS" --nvtx --out "$OUT/under_nsys.json"
"$NSYS" export --type=sqlite --force-overwrite=true -o "$OUT/decode.sqlite" "$OUT/decode.nsys-rep" >/dev/null

echo "== nsys stats tables"
for r in cuda_gpu_kern_sum cuda_kern_exec_sum cuda_api_sum nvtx_sum; do
  # report names and --filter-nvtx vary across nsys versions: try filtered, then whole run
  "$NSYS" stats --report "$r" --format csv --filter-nvtx decode_graph -o "$OUT/stats_graph" "$OUT/decode.sqlite" >/dev/null 2>&1 \
    || "$NSYS" stats --report "$r" --format csv -o "$OUT/stats_all" "$OUT/decode.sqlite" >/dev/null 2>&1 \
    || echo "  $r: not available in this nsys"
done
ls "$OUT"/stats_*.csv 2>/dev/null | sed 's/^/  /' || true

$PY scripts/profile_decode.py summarize --sqlite "$OUT/decode.sqlite" --meta "$OUT/plain.json" \
  --traced-meta "$OUT/under_nsys.json" --out "$OUT/summary.json"
