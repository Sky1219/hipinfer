#!/usr/bin/env bash
# Print the path of an Nsight Systems CLI (nsys), installing it first if there is none. For
# Ubuntu machines where we are root (Colab): the CLI comes from NVIDIA's devtools apt repo, signed
# with the CUDA repo key; failing that, the newest nsight-systems package in the CUDA repo.
# Usage: NSYS=$(scripts/install_nsys.sh | tail -1)
set -uo pipefail

find_nsys() {
  for p in "$(command -v nsys)" /usr/local/cuda*/bin/nsys /opt/nvidia/nsight-systems*/*/bin/nsys \
           /opt/nvidia/nsight-systems*/bin/nsys /usr/local/cuda*/nsight-systems*/bin/nsys; do
    [ -n "$p" ] && [ -x "$p" ] && { echo "$p"; return 0; }
  done
  return 1
}

if ! find_nsys >/dev/null; then
  . /etc/os-release
  U=ubuntu$(echo "$VERSION_ID" | tr -d .)
  KEY=/usr/share/keyrings/cuda-archive-keyring.gpg
  if [ ! -f $KEY ]; then
    wget -q -O /tmp/cuda-keyring.deb "https://developer.download.nvidia.com/compute/cuda/repos/$U/x86_64/cuda-keyring_1.1-1_all.deb" \
      && dpkg -i /tmp/cuda-keyring.deb >/dev/null
  fi
  echo "deb [signed-by=$KEY] https://developer.download.nvidia.com/devtools/repos/$U/amd64/ /" \
    > /etc/apt/sources.list.d/nvidia-devtools.list
  apt-get update -qq >/dev/null 2>&1
  export DEBIAN_FRONTEND=noninteractive
  if ! apt-get install -y -qq nsight-systems-cli >/dev/null 2>&1; then
    PKG=$(apt-cache search --names-only '^nsight-systems-20[0-9.]*$' | cut -d' ' -f1 | sort -V | tail -1)
    [ -n "$PKG" ] && apt-get install -y -qq "$PKG" >/dev/null 2>&1
  fi
fi
find_nsys || { echo "could not find or install nsys" >&2; exit 1; }
