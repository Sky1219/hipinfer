# MoE groundwork: overnight log (2026-10-16)

Worked on a laptop with no GPU: CPU-only PyTorch 2.13, transformers 5.14.1, and (found during
the night) nvcc 12.0 inside WSL Ubuntu, which can compile but has no GPU to run on. Design:
[moe-design.md](moe-design.md).

## Status at a glance

| piece | state |
|---|---|
| design note (`docs/moe-design.md`) | written |
| export, weight-file version 3 (`scripts/export.py`) | **verified on CPU**; dense exports byte-identical to before |
| CPU reference engine (`scripts/ref_engine.py`) | **verified on CPU** against HF (below) |
| MoE kernels (`src/kernels.cuh`, end of file) | **compiles**, never run |
| engine integration (`src/api.cu`) | **compiles**, never run |
| bindings (`scripts/hipinfer.py`) and `test_moe()` in `tests/test_kernels.py` | written; test logic dry-run on CPU only |
| expert parallelism (2 GPUs), OLMoE QK-norm in the engine | design only |

## Verified on CPU

**Export.** `hf-internal-testing/tiny-random-LlamaForCausalLM` exported with the old and the new
`export.py` gives byte-identical files (`cmp`), so dense models are unaffected (still version 2).
MoE models write version 3. Export handles both transformers 5.x fused expert tensors and the 4.x
per-expert layouts; only the 5.x path has been run. One bug caught on the way: Granite 1B has
hidden = 2 x inter, so its `gate_up_proj` is square and a shape test for "transposed storage"
fired wrongly. It now decides from `down_proj`'s shape.

**Reference engine vs HF** (`python tests/test_moe_ref.py`, about 100 s, all pass; numbers in
`results/moe_ref_cpu.json`):

| model | fp32: max abs logit diff / max abs logit | fp32: top-1 at every position | routed expert sets equal to HF | fp16: greedy tokens (16 per prompt) |
|---|---|---|---|---|
| tiny-random Mixtral (4 experts, top-2, fp32 routing weights) | 3.2e-7 | 65/65 | 130/130 | 4/4 prompts identical |
| synthetic tiny OLMoE (QK-norm, 8 experts top-2, no renormalising) | 4.4e-7 | 61/61 | 122/122 | 4/4 identical |
| synthetic tiny Granite MoE (tied embeddings, all 4 multipliers, attention multiplier not a power of 2) | 1.5e-4 | 61/61 | 122/122 | 4/4 identical |
| **ibm-granite/granite-3.1-1b-a400m-base** (32 experts, top-8) | 4.3e-4 | 60/60 | 1440/1440 | **4/4 identical** |

The fp32 column runs HF and the reference in fp32 with the file's fp16 weights, so any layout
error would show up as O(1). The 1e-4 level for Granite is the export's folding of the multipliers
into fp16 weights (`embed x 12`, `lm_head / 6`, `wo x 0.22`), as intended. In fp16 the Granite
prefill logits differ from HF-on-CPU by at most 0.7% of the logit range: the usual fp16
accumulation-order noise over 24 layers. Routing is compared directly. Router logits are captured
from HF's router modules by forward hooks, and the expert sets match for every (layer, token).

**HF semantics pinned down** (transformers 5.x, default `grouped_mm` experts): router linear
rounded to fp16; Mixtral uses fp32 routing weights, Granite and OLMoE round them to fp16; the
weighted expert outputs are summed over k in fp32 and rounded once. Granite's
`residual_multiplier` is applied to the rounded MoE output. The kernels and the reference follow
this. The older `eager` path accumulates in fp16 instead.

## Compiled, never run

`src/api.cu` with the MoE kernels builds under WSL with nvcc 12.0 (gcc 13.3), for
`-gencode` sm_75, sm_80, sm_86, sm_89 and compute_80 PTX, the Makefile's default list. There are
no new warnings, and none of the 6 MoE kernels spills on any arch (`-Xptxas -v`). The router
kernel has a 128-byte stack frame for its 16-entry selection arrays. Registers on sm_75: router 50,
gate/up GEMV 62, down+combine GEMV 73, combine 54, sort 22, gather 18.

To compile under CUDA 12.0 I had to shim one **pre-existing** line in a temp copy only:
`load_shard` builds the RoPE table with `__halves2half2` on the host, which 12.0's headers reject
and 12.4 accepts. The repo is unchanged there. The script is in the session scratchpad:
it copies `src/` to `/tmp/hib`, applies the shim, and runs nvcc with `-Xptxas -v`.

A CPU dry run of `test_moe()` (shapes reduced) with Python stand-ins that copy the kernels' index
math (interleaved gate/up rows `(o/16)*32 + o%16`, expert offsets `e*2I*H` and
`(e*H + h)*I`, the counting sort and its inverse permutation, the combine, and the router's
tie-break) matched the reference exactly. This checks the test code and the indexing scheme, not
the CUDA itself.

## Untested (needs the GPU)

* Every MoE kernel's numerics, the GEMV and grouped paths of `hi_moe_experts`, graph capture of
  MoE decode (B <= 4), and decode without graphs above the GEMV limit (B > 4 runs the grouped path,
  which syncs on the per-expert counts; `decode_rank` turns graphs off for that case).
* `load_shard` on a version-3 file, and the engine end to end (`gate.py`, `ppl.py`).
* AMD (`make hip`): not compiled at all.
* Performance: nothing measured. The grouped path issues up to 2E small GEMMs per layer, each
  autotuned on first use (expect a slow first prefill).

## Tomorrow on edmonds (2080 Ti GPUs 0 and 2 only, never GPU 1)

From the repo on edmonds (or through `scripts/edmonds.sh '...'`, which syncs `src/ tests/
scripts/` to `/data/axd200066/hipinfer_pf` and exports PY, NVCC and `CUDA_VISIBLE_DEVICES=0`):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/datastore/hub
PY=/data/axd200066/envs/baselines/bin/python
make NVCC=/usr/local/cuda-12.4/bin/nvcc NV_ARCH='-gencode arch=compute_75,code=sm_75'
```

1. MoE kernels alone (fast, first thing to look at):
   ```bash
   $PY -c "import sys; sys.path[:0] = ['tests']; import test_kernels as t; t.test_moe(); print('FAILS', t.FAILS)"
   ```
2. Whole kernel suite, which also checks that the dense kernels are unchanged:
   ```bash
   $PY tests/test_kernels.py 2>&1 | tee results/2080ti/test_kernels_moe.log
   ```
3. Export Granite MoE (about 20 s; 2.8 GB) and re-run the CPU check against the same file if wanted:
   ```bash
   $PY scripts/export.py --model ibm-granite/granite-3.1-1b-a400m-base --out /data/axd200066/models/granite-moe-1b.bin
   $PY tests/test_moe_ref.py --quick
   ```
4. End to end against HF on the GPU (single sequence with and without graphs, batched B=6 on the
   grouped path, long chunked prompt):
   ```bash
   $PY tests/gate.py --model ibm-granite/granite-3.1-1b-a400m-base --weights /data/axd200066/models/granite-moe-1b.bin --out results/2080ti/granite-moe-1b/gate.json
   $PY tests/ppl.py --model ibm-granite/granite-3.1-1b-a400m-base --weights /data/axd200066/models/granite-moe-1b.bin --out results/2080ti/granite-moe-1b/ppl.json
   ```
5. Dense regression (export unchanged, engine edits in `decode_step` and `load_shard`), on a model
   that already has results (use the existing `.bin` in `/data/axd200066/models`):
   ```bash
   $PY tests/gate.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weights /data/axd200066/models/tinyllama.bin
   ```

If step 1 fails, debug order: router (indices, then weights), then `path=gemv`, then
`path=grouped` (`hi_moe_experts` forced via `hi.moe_experts(..., path=1|2)`). `ref_engine.py`'s
`moe_route` / `moe_experts` are the oracle and run on CUDA tensors as well.

## Next

1. Fix whatever step 1 to 4 turn up; then time Granite decode at B=1 against the weight-read
   floor (`hi_weight_bytes` now counts the router plus top-k experts) and against llama.cpp.
2. Expert parallelism, step 1 of the design (TP attention, experts split across the 2080 Tis,
   combine through the existing all-reduce), then OLMoE: QK-norm kernel and the 2-GPU load.
3. Device-side grouped GEMM (no host sync), so batched MoE decode can be graph-captured.
