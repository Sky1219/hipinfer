# NVIDIA (default): plain nvcc, hip* names mapped by src/hip_compat.h
# AMD:              make hip  (needs ROCm hipcc + rocWMMA headers)
NVCC     ?= /usr/local/cuda/bin/nvcc
# Turing (T4, 2080 Ti), Ampere (A100, A6000), Ada (L4) as SASS, plus compute_80 PTX that the
# driver JIT-compiles for anything newer (e.g. Hopper, Blackwell) when the toolkit is too old.
NV_ARCH  ?= -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 \
            -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
            -gencode arch=compute_80,code=compute_80
HIPCC    ?= hipcc
AMD_ARCH ?= --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx1100
# Multi-GPU tensor parallelism always builds (with its own one-shot all-reduce). NCCL=<dir with
# include/nccl.h and lib/libnccl.so.2> adds NCCL as the large-message backend; the pip wheel works:
#   make NCCL=$$(python -c "import nvidia.nccl as m; print(m.__path__[0])")
# For AMD, make hip RCCL=1 links RCCL from the ROCm install.
NCCL     ?=
RCCL     ?=
ifneq ($(NCCL),)
  NV_NCCL = -DHI_NCCL -I$(NCCL)/include -L$(NCCL)/lib -l:libnccl.so.2 -Xlinker -rpath=$(NCCL)/lib
endif
ifneq ($(RCCL),)
  AMD_NCCL = -DHI_NCCL -lrccl
endif
SRC = src/api.cu
DEPS = src/api.cu src/kernels.cuh src/hip_compat.h

build/libhipinfer.so: $(DEPS)
	mkdir -p build
	$(NVCC) -O3 -std=c++17 -lineinfo $(NV_ARCH) -Xcompiler -fPIC,-fvisibility=hidden -shared $(SRC) $(NV_NCCL) -lpthread -o $@

hip: $(DEPS)
	mkdir -p build
	$(HIPCC) -O3 -std=c++17 -fPIC -fvisibility=hidden -shared $(AMD_ARCH) -x hip $(SRC) $(AMD_NCCL) -lpthread -o build/libhipinfer_amd.so

clean:
	rm -rf build
.PHONY: hip clean
