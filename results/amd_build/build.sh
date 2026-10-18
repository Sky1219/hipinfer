#!/usr/bin/env bash
# Compile-only AMD build of hipinfer inside AMD's ROCm dev container (no AMD GPU needed).
# Run from the repo root on any Docker host:
#   docker run --rm -v "$PWD":/src -w /src -e SRC_REV="$(git describe --always --dirty)" \
#     rocm/dev-ubuntu-24.04:7.2.4 bash results/amd_build/build.sh
# Each build runs in its own copy of the tree (so builds can run side by side) with the
# Makefile's own flags, and writes results/amd_build/<name>.log:
#   gfx90a, gfx942, gfx1100   make hip, one target each (each target's diagnostics on their own)
#   all_targets               make hip (the default AMD_ARCH: all three in one library)
#   all_targets_rccl          make hip RCCL=1
#   all_targets_wall          make hip with -Wall -Wextra added
#   resource_usage_<target>   per-kernel registers / spills / LDS / occupancy (hipcc remarks)
set -u
OUT=results/amd_build
export DEBIAN_FRONTEND=noninteractive

if [ ! -f /opt/rocm/include/rocwmma/rocwmma.hpp ] || [ ! -f /opt/rocm/include/rccl/rccl.h ]; then
  apt-get update -qq && apt-get install -y -qq rocwmma-dev rccl-dev make >/dev/null
fi

{
  echo "image:   ${IMAGE_TAG:-rocm/dev-ubuntu-24.04:7.2.4}"
  echo "rocm:    $(cat /opt/rocm/.info/version 2>/dev/null)"
  echo "rocwmma: $(dpkg-query -W -f='${Version}' rocwmma-dev 2>/dev/null)"
  echo "rccl:    $(dpkg-query -W -f='${Version}' rccl-dev 2>/dev/null)"
  echo "os:      $(. /etc/os-release; echo "$PRETTY_NAME")"
  echo "source:  ${SRC_REV:-unknown}"
  echo
  hipcc --version
} > "$OUT/toolchain.txt" 2>&1

LLVM=/opt/rocm/lib/llvm/bin
# Kernels per target in the finished library: unbundle each device code object and count kernel
# descriptors (.kd symbols), listing the MoE kernels by name.
list_kernels() {  # list_kernels <lib>
  local t; t=$(mktemp -d)
  "$LLVM/llvm-objcopy" --dump-section=.hip_fatbin="$t/fat" "$1" || return
  for b in $("$LLVM/clang-offload-bundler" --list --type=o --input="$t/fat" | grep amdgcn); do
    "$LLVM/clang-offload-bundler" --unbundle --type=o --input="$t/fat" --targets="$b" --output="$t/co"
    local kd; kd=$("$LLVM/llvm-readelf" --dyn-syms "$t/co" | grep '\.kd$')
    echo "  ${b##*--}: $(echo "$kd" | wc -l) kernels; MoE: $(echo "$kd" | grep -o -E 'moe_[a-z_]*kernel' | sort -u | tr '\n' ' ')"
  done
  rm -rf "$t"
}

build() {  # build <log name> <make args...>
  local name=$1; shift
  local log="$OUT/$name.log" w; w=$(mktemp -d)
  cp -r Makefile src "$w"
  {
    echo "\$ make $*"
    echo
    start=$(date +%s)
    make -C "$w" --no-print-directory "$@" 2>&1
    rc=$?
    echo
    echo "exit status: $rc ($(( $(date +%s) - start )) s)"
    if [ -f "$w/build/libhipinfer_amd.so" ]; then
      echo "build/libhipinfer_amd.so: $(stat -c %s "$w/build/libhipinfer_amd.so") bytes"
      list_kernels "$w/build/libhipinfer_amd.so"
    fi
  } > "$log" 2>&1
  rm -rf "$w"
  echo "$log: $(grep -c 'warning:' "$log") warnings, $(grep -c 'error:' "$log") errors, $(grep 'exit status' "$log")"
}

# hipcc's kernel-resource-usage remarks: device-only compile of every target, no link.
resource_usage() {  # resource_usage <target>
  local log="$OUT/resource_usage_$1.log"
  local cmd="hipcc -O3 -std=c++17 --offload-arch=$1 --offload-device-only -c -x hip src/api.cu -o /dev/null -Rpass-analysis=kernel-resource-usage"
  { echo "\$ $cmd"; echo; $cmd 2>&1; echo; echo "exit status: $?"; } > "$log"
  echo "$log: $(grep -c 'Function Name' "$log") kernel reports, $(grep -E 'VGPRs Spill: [1-9]|SGPRs Spill: [1-9]' "$log" | wc -l) with spills"
}

build gfx90a  hip AMD_ARCH=--offload-arch=gfx90a &
build gfx942  hip AMD_ARCH=--offload-arch=gfx942 &
build gfx1100 hip AMD_ARCH=--offload-arch=gfx1100 &
wait
build all_targets      hip &
build all_targets_rccl hip RCCL=1 &
build all_targets_wall hip HIPCC="hipcc -Wall -Wextra" &
wait
resource_usage gfx90a &
resource_usage gfx942 &
resource_usage gfx1100 &
wait
