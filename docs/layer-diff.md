# Layer-by-layer diff against Hugging Face

`tests/gate.py` and `tests/ppl.py` tell you *whether* hipinfer matches Hugging Face transformers.
[`scripts/layer_diff.py`](../scripts/layer_diff.py) tells you *where* it stops matching. It records
every sub-step of every layer on both sides for one prompt and prints the first one that is off
by more than fp16 rounding explains, plus the token positions and heads where the error sits.
So "outputs don't match HF" becomes "layer 7's RoPE is off from position 1 on".

> **Status (2026-10-16).** The HF side, the comparison and the report are tested on CPU
> ([`tests/test_layer_diff.py`](../tests/test_layer_diff.py)). The engine side
> (`hi_prefill_debug` in [`src/api.cu`](../src/api.cu), `Engine.prefill_debug` in
> [`scripts/hipinfer.py`](../scripts/hipinfer.py)) was written without a GPU toolchain and **has
> not been compiled or run yet**. The steps under [First run on the GPU](#first-run-on-the-gpu)
> build it and check it.

## What gets compared

For every prompt row (prefill, one sequence, position 0 onwards):

| point | value | shape |
|---|---|---|
| `embed` | token embedding | `[T, hidden]` |
| `layers.L.attn_norm` | input RMSNorm output | `[T, hidden]` |
| `layers.L.qkv` | q \| k \| v projections, plus bias, before RoPE | `[T, (Hq + 2 Hkv) D]` |
| `layers.L.rope` | q \| k after RoPE | `[T, (Hq + Hkv) D]` |
| `layers.L.attn` | attention output, the O projection's input | `[T, Hq D]` |
| `layers.L.o_proj` | residual stream after the O projection | `[T, hidden]` |
| `layers.L.mlp_norm` | post-attention RMSNorm output | `[T, hidden]` |
| `layers.L.gate_up` | `silu(gate) * up`, the down projection's input | `[T, inter]` |
| `layers.L.down` | residual stream after the MLP (the layer's output) | `[T, hidden]` |
| `final_norm`, `logits` | | `[T, hidden]`, `[T, vocab]` |

These are the values the engine actually materializes. The O and down GEMMs add the residual in
their epilogue, and the gate/up GEMM applies SwiGLU in its epilogue, so the engine never holds
the bare O output or separate gate and up values. The comparison gets the projection's own
output back by subtracting the residual that came before (`o_proj - previous down`,
`down - o_proj`). A broken O or down projection then moves the number by its full size instead
of being hidden inside the residual.

On the HF side the points come from forward hooks. RoPE has no module to hook, so the `rope`
point is rebuilt from the recorded q/k projections and the `(cos, sin)` the attention module
received, with HF's own `rotate_half` formula. The tests check that it equals what HF's
`apply_rotary_pos_emb` returns. They also check that causal attention over the `rope` point
reproduces the `attn` point, and that `qkv` and `gate_up` match what the engine computes from the
exported weight layout (`[q; k; v]` rows, gate/up interleaved in groups of 16).

## How "diverged" is decided

fp16 noise grows smoothly through the layers, while a bug shows up as a step. Each point's error
is the **relative L2 error of every (position, head) block** (every position, for points that
have no heads), not one number per tensor. Real checkpoints have a BOS row whose activations are
~100× the rest. A whole-tensor error is mostly that one row, so it would hide a bug in the other
rows (on TinyLlama, layer 2's `gate_up` error was 6e-5 whole-tensor but 2e-3 on its worst row).

Each block gets its own threshold. By default (`--ref-dtype fp32`, `--floor auto`) HF runs twice:
once in fp32 as the reference, and once in fp16 to measure how far honest fp16 arithmetic lands
from it, block by block. A block's threshold is `max(--min-tol, --factor × max(HF fp16's error
there, the point's median HF fp16 error))`, with defaults 2e-3 and 8. A point diverges when any
of its blocks exceeds its threshold, and the report names the first such point in execution
order. With `--floor off` every block is held to a fixed `--tol` (default 5e-2) instead.

Two details matter for residual points. In the BOS row, the fp16 rounding of `h + o` is larger
than `o` itself, so the branch is only known to about one ulp of `h`. Its denominator is therefore
floored at `|h| / 16` (`RESID_SCALE`). The median term keeps a block that fp16 happened to get
almost exactly right from getting a near-zero threshold.

Checked on CPU (TinyLlama-1.1B, 22-token prompt): a 5% perturbation of one sub-step of one layer
is named exactly for `qkv`, `attn`, `o_proj`, `gate_up` and `down` deep in the model, with
5–77× margin. A second fp16 implementation (HF with SDPA attention instead of eager) reaches at
most 0.57 of its thresholds. A 2% fault sits near the 8× threshold and is not reliably flagged;
lower `--factor` if you need that, at the cost of false positives.

## Reading the report

From the CPU demo: HF fp16 with a RoPE sign bug injected into layer 7 of TinyLlama, compared
with HF fp32 (abridged):

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0: 22 layers, hidden 2048, heads 32/4 x 64, 22 prompt tokens
HF fp16 with faults [(7, 'rope', 0.1)] vs HF fp32
layer  attn_norm       qkv      rope      attn    o_proj  mlp_norm   gate_up      down
embed   5.3e-07
    0   5.9e-04   2.6e-03   2.4e-03   3.4e-03   6.9e-04   1.3e-03   1.7e-03   1.4e-03
  ...
    6   1.3e-03   3.1e-03   1.3e-03   6.9e-03   3.2e-03   1.9e-03   2.9e-03   4.4e-03
    7   1.7e-03   2.5e-03   1.1e+00*  1.7e+00!  6.8e-01!  3.4e-01!  4.2e-01!  4.6e-01!
    8   2.5e-01!  4.0e-01!  2.8e-01!  1.1e+00!  3.9e-01!  3.5e-01!  4.2e-01!  4.5e-01!
  ...
final_norm 2.0e-01!   logits 1.2e-01!

FIRST DIVERGENCE: layer 7, rope  (layers.7.rope)
  at t=20 q16: rel 1.1e+00 > threshold 5.4e-03 (211.0x), HF fp16 error there 4.3e-04
  whole point: rel 5.7e-01, max |diff| 8.71 (ref rms 1.21), cosine 0.838770
  previous point layers.7.qkv: worst rel 2.5e-03, 0.12 of its threshold
  worst positions: t=20 1.1e+00, t=19 1.1e+00, t=2 1.1e+00, t=21 1.1e+00; position 0: 2.4e-03
  worst heads: q16 1.1e+00, q23 1.1e+00, q20 1.1e+00, q5 1.0e+00 (23 of 36 heads within 2x of the worst)
  position 0 matches (cos 1, sin 0 there): the RoPE angles / sign / frequencies, not the inputs
  119 later point(s) also over threshold (errors propagate downstream)
```

Without the fault, an fp16 run against the fixed 5e-2 tolerance (`--floor off`) ends `no
divergence`, and its closest point reaches 0.32 of the tolerance.

- The grid shows each point's worst block error. `!` means over threshold and `*` marks the
  first divergence. Everything after the first divergence inherits its error, so only `*`
  matters.
- `at t=… head`: the block that broke the threshold.
- `worst positions … position 0`: RoPE is the identity at position 0 (cos 1, sin 0), so a RoPE
  bug leaves position 0 clean and grows with position. The report says so when it sees that
  pattern.
- `worst heads`: q / k / v heads for `qkv`, q / k for `rope`. An error in only the k heads, or
  only some kv heads, points at GQA head mapping or the k part of a fused kernel.
- The exit code is 1 when anything diverged and 0 otherwise. `--json out.json` writes every
  point's numbers.

## CPU only (verified 2026-10-16)

```bash
python tests/test_layer_diff.py
```

That runs about 120 checks on random Llama (GQA) and Qwen2 (QKV bias) models plus
hf-internal-testing's tiny Llama, in under a minute. `LAYER_DIFF_REAL_MODEL=<HF id>` adds the
TinyLlama-scale check above, which takes a few minutes on CPU.

To demo the tool by injecting a fault into an HF fp16 run (`LAYER:STEP[:SCALE]`, where STEP is
any point name above):

```bash
python scripts/layer_diff.py --model hf-internal-testing/tiny-random-LlamaForCausalLM --candidate hf --device cpu --inject 1:rope
```

## First run on the GPU

These steps were written on 2026-10-16 for edmonds. Per the usual rules, use GPU 0 (a 2080 Ti),
never GPU 1 (the A6000), and always set `CUDA_DEVICE_ORDER=PCI_BUS_ID`. They use a separate
remote directory (`/data/axd200066/hipinfer_ld`), so nothing else's build in `hipinfer_pf` gets
clobbered.

**1. Ship the branch and build** (2080 Ti only, which compiles fast). Run from this checkout on
the laptop:

```bash
tar --exclude=__pycache__ -cf - src tests scripts Makefile | ssh edmonds "mkdir -p /data/axd200066/hipinfer_ld && cd /data/axd200066/hipinfer_ld && rm -rf src tests scripts && tar -xmf -"
```

```bash
ssh edmonds 'cd /data/axd200066/hipinfer_ld && make NVCC=/usr/local/cuda/bin/nvcc NV_ARCH="-gencode arch=compute_75,code=sm_75"'
```

If the build fails, the new code is the `debug dump` block above `attn_rows` and
`hi_prefill_debug` / `hi_debug_bytes` after `hi_prefill_logits`.

**2. Confirm nothing else changed.** The dump only runs inside `hi_prefill_debug`.

```bash
ssh edmonds 'cd /data/axd200066/hipinfer_ld && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/datastore/hub && /data/axd200066/envs/baselines/bin/python tests/test_kernels.py | tail -3 && /data/axd200066/envs/baselines/bin/python tests/gate.py --weights /data/axd200066/models/tinyllama.bin --new 32'
```

**3. Diff TinyLlama.** Expect `no divergence` and exit code 0.

```bash
ssh edmonds 'cd /data/axd200066/hipinfer_ld && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/datastore/hub && /data/axd200066/envs/baselines/bin/python scripts/layer_diff.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weights /data/axd200066/models/tinyllama.bin --json /data/axd200066/hipinfer_ld/layer_diff_tinyllama.json'
```

Both HF runs (fp32 reference, then fp16 floor) run on the GPU one after the other and are freed
before the engine loads. TinyLlama in fp32 is 4.4 GB. For models whose fp32 copy does not fit
in 11 GB (Llama-3.2-3B and up), add `--device cpu`: the HF runs move to the server's CPU while
the engine stays on the GPU.

**4. Negative control: make the engine wrong on purpose.** Scaling the RoPE frequencies stored
in a copy of the weight file by 1% must be reported as `layer 0, rope`, with position 0 clean.
This also checks the `rope` dump, which is the most intricate copy (k comes back out of the KV
cache):

```bash
ssh edmonds 'mkdir -p /data/axd200066/tmp && cp /data/axd200066/models/tinyllama.bin /data/axd200066/tmp/tinyllama_badrope.bin && /data/axd200066/envs/baselines/bin/python -c "import numpy as np, struct; p = \"/data/axd200066/tmp/tinyllama_badrope.bin\"; hd = struct.unpack(\"<10i\", open(p, \"rb\").read(40))[8]; a = np.memmap(p, dtype=np.float32, mode=\"r+\", offset=64, shape=(hd // 2,)); a *= 1.01; a.flush()"'
```

```bash
ssh edmonds 'cd /data/axd200066/hipinfer_ld && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/datastore/hub && /data/axd200066/envs/baselines/bin/python scripts/layer_diff.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --weights /data/axd200066/tmp/tinyllama_badrope.bin; rm /data/axd200066/tmp/tinyllama_badrope.bin'
```

**5. The other families.** Qwen2.5 covers the QKV bias, and Llama-3.2 covers llama3 RoPE scaling
and tied embeddings:

```bash
ssh edmonds 'cd /data/axd200066/hipinfer_ld && export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_CACHE=/datastore/hub && for m in "Qwen/Qwen2.5-0.5B qwen25-0.5b" "unsloth/Llama-3.2-1B llama32-1b" "unsloth/Llama-3.2-3B llama32-3b"; do set -- $m; /data/axd200066/envs/baselines/bin/python scripts/layer_diff.py --model $1 --weights /data/axd200066/models/$2.bin --device cpu --json /data/axd200066/hipinfer_ld/layer_diff_$2.json | tail -3; done'
```

### If step 3 reports a divergence

The tool works on the HF side, so either the engine really differs or the uncompiled dump is
wrong. Signs of a dump bug:

- **rel exactly 1.0 everywhere in one plane, cosine 0:** the plane was never written (the buffer
  is zeroed first). Look for a missing `dbg_step` call.
- **Only the k heads of `rope` off, q fine:** `dbg_rope`'s cache indexing
  (`((slot * hkv + h) * max_seq + pb + r0) * D`).
- **`gate_up` off everywhere but `down` fine:** the SwiGLU output column order. The engine is
  assumed to write unit order, as `down` consumes it.
- **`logits` off but `final_norm` fine:** the logits plane pointer or its row stride (`vocab`).
- **Only a few late positions off in `attn`:** a real flash-prefill tail-tile issue rather than
  the dump. Rerun with `HIPINFER_FLASH_WMMA=1` to compare the portable kernel.

`--save-ref ref.pt` keeps the HF capture, and `--load-ref ref.pt` reuses it (same prompt), so the
engine side can be iterated on without rerunning HF.

## Porting a new model family

1. Export it with `scripts/export.py` and run `layer_diff.py` on a short prompt.
2. The first divergence names the sub-step to fix. Typical first hits: `qkv` (bias, head order,
   q/k norm), `rope` (rope_type, partial rotary, interleaved vs half-split pairs: an interleaving
   bug shows as every position except 0 off), `mlp_norm` (an extra norm, Gemma's `1 + w`),
   `gate_up` (activation function), `logits` (soft-capping, tied weights).
3. The HF capture assumes the Llama module names (`input_layernorm`, `self_attn.{q,k,v,o}_proj`,
   `post_attention_layernorm`, `mlp.down_proj`, `model.norm`) and picks up Qwen3-style
   `q_norm` / `k_norm` before RoPE when present. A family with other names needs its hooks added
   in `capture_hf`.
