# Porting a model family to hipinfer

hipinfer implements one decoder architecture. A new model family is supported when its Hugging Face
checkpoint can be expressed in that architecture by the exporter, and the result passes four
levels of tests against PyTorch and HF transformers. This document describes the weight format, what the loader
checks, which config features exist today, and what each existing family needed. All line
references are to the current tree.

## The architecture the engine implements

Every layer is pre-norm: `x += O(attn(RoPE(QKV(RMSNorm(x)))))`, then
`x += down(SiLU(gate(RMSNorm(x))) * up(RMSNorm(x)))`. After the last layer come a final RMSNorm, the
lm_head and a greedy argmax. Weights and activations are fp16 with fp32 accumulation. The specifics:

- RMSNorm with a plain `w * norm(x)` weight ([src/kernels.cuh:61](../src/kernels.cuh:61)), with `eps` taken from the
  header.
- RoPE in HF's half-rotation layout: dimension `d` is paired with `d + D/2`, not with `d + 1`. The
  rotation covers the whole head (no partial rotary).
- Multi-head or grouped-query attention, causal over the whole history (no sliding window, no
  ALiBi, no soft-capping), with scale `1/sqrt(head_dim)`.
- A SwiGLU MLP.

Llama 2 / 3 / 3.1 / 3.2, TinyLlama, Mistral and Qwen2 / 2.5 fit this description. The families
with committed results are TinyLlama-1.1B, Llama-3.2-1B / 3B, Llama-3.1-8B, Qwen2.5-0.5B / 1.5B and
Mistral-7B-v0.3, plus Mistral-7B-v0.1 within its window ([scripts/run_all.sh:21](../scripts/run_all.sh:21)).
The exporter's docstring also names DeepSeek-Coder, but nothing in `results/` covers it.

## Step 1: the weight file (`scripts/export.py`, format version 2)

[scripts/export.py](../scripts/export.py) loads the HF checkpoint in fp16 and writes one flat file:
a 64-byte header, then tensors that each start on a 256-byte boundary.

**Header** (`struct.pack("<10i2f2i", ...)`, [export.py:71](../scripts/export.py:71), zero-padded to 64 bytes):

| int32 index | field | source |
|---|---|---|
| 0 | magic `0x48495046` | constant |
| 1 | version `2` | constant |
| 2–9 | vocab, hidden, inter, layers, heads, kv_heads, head_dim, max_pos | HF config; `head_dim` from `cfg.head_dim` if present, else `hidden / heads` |
| 10 | `rms_norm_eps` (float) | HF config |
| 11 | `attention_scaling` (float) | HF rotary module (1.0 unless the RoPE type scales cos/sin) |
| 12 | flags: bit 0 tied lm_head, bit 1 QKV bias | detected from the state dict |
| 13 | sliding window (0 = full attention) | `sliding_window()`, [export.py:45](../scripts/export.py:45) |

**Tensor order** ([export.py:89–105](../scripts/export.py:89)):

1. `inv_freq`, fp32 `[head_dim/2]`: the RoPE frequencies copied from HF's `model.model.rotary_emb.inv_freq`
   ([export.py:35](../scripts/export.py:35)). Because the exporter stores HF's own frequencies,
   any scaling HF applies (Llama-3 `rope_type: llama3`, linear) is already baked in. It falls back
   to `theta^(-2i/D)` only if the model has no rotary module.
2. `embed [vocab, hidden]`, then `final_norm [hidden]`.
3. `lm_head [vocab, hidden]`, written only if the lm_head is **not** tied to the embedding.
4. For each layer:
   - `attn_norm`;
   - `wqkv = cat([q_proj; k_proj; v_proj], dim 0)`;
   - `bqkv`, the same concatenation of the biases (only if flag bit 1 is set);
   - `wo`;
   - `mlp_norm`;
   - `wgu`, gate and up interleaved;
   - `wdown`.

**The gate/up interleave** ([export.py:28](../scripts/export.py:28)): rows are written as 16 gate rows, then
the same 16 up rows, then the next 16 of each. This lets one warp of the GEMV, or one 32-column tile of the GEMM,
hold a matching gate/up pair and apply SiLU(gate)·up in its epilogue without a second pass. It also means
`inter` must be a multiple of 16, and the tensor-parallel shard of `wgu` is a contiguous row range
([api.cu:974](../src/api.cu:974)).

**Detection logic:**

- *Tied embeddings* ([export.py:66](../scripts/export.py:66)): `lm_head.weight` is missing from the
  state dict, shares storage with `embed_tokens`, or is equal to it element for element.
- *QKV bias* ([export.py:68](../scripts/export.py:68)): `model.layers.0.self_attn.q_proj.bias` exists.
- *Sliding window*: `cfg.sliding_window`, but only if `use_sliding_window` is not false (Qwen2
  configs carry a window they don't use) and the window is shorter than `max_position_embeddings`.

## Step 2: the loader (`load_shard` in `src/api.cu`)

[load_shard](../src/api.cu:878) reads the header, validates it, and copies each tensor (or this
rank's shard of it) to the GPU in the order above. It refuses a file, with a message on stderr,
when:

| check | where | why |
|---|---|---|
| magic or version ≠ 2 | [api.cu:887–888](../src/api.cu:887) | stale or foreign file |
| `head_dim` not 64 or 128 | [api.cu:899](../src/api.cu:899), [api.cu:493](../src/api.cu:493) | attention kernels are templated on D = 64 / 128 only |
| `heads % kv_heads ≠ 0` or group > 8 | [api.cu:900](../src/api.cu:900) | decode attention is instantiated for G = 1..8 ([api.cu:553](../src/api.cu:553)) |
| sliding window > 0 and `max_seq` > window | [api.cu:906](../src/api.cu:906) | full attention equals windowed attention only while every sequence fits in the window |
| heads, kv_heads or inter/32 not divisible by the TP world size | [api.cu:912](../src/api.cu:912) | Megatron-style sharding by heads and MLP units |
| `max_batch × kv_heads per GPU` > 4096 | [api.cu:918](../src/api.cu:918) | fixed decode-attention counter array |

Shape limits that come from the GEMM rather than the loader: `K % 32 == 0` and `N % 16 == 0`
(`N % 32` for the SwiGLU epilogue), per the comment at [api.cu:471](../src/api.cu:471).

For tensor parallelism the loader slices as it reads:

- `wqkv` and `bqkv`: this rank's q, k and v rows (three row ranges, [api.cu:961](../src/api.cu:961)).
- `wo` and `wdown`: column ranges, through a 2-D copy.
- `wgu`: a row range.
- `lm_head`: a vocabulary row range.
- Embedding and norms: replicated.

It then builds the RoPE table on the host, in fp32: angle = `pos × inv_freq`, cos and sin
multiplied by `attention_scaling`, then rounded to fp16 ([api.cu:1003](../src/api.cu:1003)).

## Config features: what exists and what it cost

The git history is too short (8 commits) to show when each feature was added, so these counts are
the lines in the current code that exist only for that feature.

| feature | families that use it | where it lives | size |
|---|---|---|---|
| **Tied embeddings** | Qwen2.5-0.5B / 1.5B, Llama-3.2-1B / 3B | export: detect + skip write ([export.py:66–67, 92](../scripts/export.py:66)); loader: `lm_head = embed + v0·H` ([api.cu:957](../src/api.cu:957)) | 3 exporter lines, 1 loader line. The same line serves TP, because a rank's vocab slice of the embedding *is* its lm_head shard. |
| **QKV bias** | Qwen2 / 2.5 | exporter: detect + write ([export.py:68, 99–101](../scripts/export.py:68)); loader: allocate + shard ([api.cu:970–971](../src/api.cu:970)); fused decode epilogue: one add before RoPE ([kernels.cuh:214](../src/kernels.cuh:214)); bias epilogue in the three tensor-core GEMMs ([kernels.cuh:424](../src/kernels.cuh:424), [794](../src/kernels.cuh:794), [1017](../src/kernels.cuh:1017)) | about 20 new lines, plus a pointer argument threaded through existing calls (`w.bqkv` at the four QKV call sites). Most of it is the GEMM bias epilogue, which is generic: any `hi_gemm` call can take a bias. |
| **Llama-3 RoPE scaling** | Llama-3.1 / 3.2 | the exporter copies HF's `inv_freq` and `attention_scaling` ([export.py:35–42](../scripts/export.py:35)) | **no engine code.** The exporter's 8-line `rope_inv_freq` covers every rope type HF expresses as `inv_freq` + `attention_scaling`. Dynamic (NTK) scaling, which changes the frequencies with sequence length, is not covered. |
| **GQA, groups 1–8** | all (TinyLlama G=8, Qwen2.5-0.5B G=7, Qwen2.5-1.5B G=6, Llama-3.2-1B / Mistral G=4, Llama-3.2-3B G=3) | `decode_attn_mma_kernel<G, D>`, where the query heads of a group are the rows of an m16 MMA tile; switch at [api.cu:553](../src/api.cu:553) | one template instantiation per G. Groups above 8 are not instantiated and are rejected at load time. |
| **head_dim 64 and 128** | 64: TinyLlama, Qwen2.5-0.5B, Llama-3.2-1B; 128: the rest | template parameter of the flash-prefill and decode-attention kernels ([api.cu:510–518](../src/api.cu:510), [api.cu:577](../src/api.cu:577)) | two instantiations. Other sizes (e.g. 80, 96, 256) are rejected at load time. |
| **Sliding window (guard only)** | Mistral-7B-v0.1 | exporter writes the window; loader refuses `max_seq` > window | 7 exporter lines plus the check. This is a *guard*, not an implementation. |
| **Explicit `head_dim`** | any config with `head_dim ≠ hidden/heads` | [export.py:63](../scripts/export.py:63) | 1 line |

## Not supported, and how each fails

How a gap fails matters as much as the gap itself. A loud failure is safe. A silent one gives
wrong logits and is caught only by the gate and perplexity tests below.

| feature | example models | what happens today |
|---|---|---|
| Mixture-of-Experts | Mixtral, Qwen-MoE, DeepSeek-V2/V3 | **loud**: `export.py` raises `KeyError` on `mlp.gate_proj` (the experts have different names). The engine has no router or expert GEMMs. |
| Sliding-window attention beyond the window | Mistral-7B-v0.1 at > 4,096 tokens, Gemma-2 | **loud**: the loader refuses `max_seq` > window. |
| head_dim ∉ {64, 128}, GQA group > 8 | Phi-2 (80), Gemma (256) | **loud**: rejected at load time. |
| Partial rotary | Phi | **loud**: `assert inv_freq.numel() == head_dim // 2` in the exporter. |
| Fused or renamed projections | Phi-3 (`qkv_proj`), GPT-NeoX | **loud**: `KeyError` in the exporter. |
| Other norms or activations | LayerNorm models, Gemma's `(1 + w)` RMSNorm, GELU MLPs | **silent**. The exporter checks tensor names, not `model_type`, `hidden_act` or the norm type. A model whose names match Llama's but whose math differs (Gemma also scales the embedding by √hidden) exports without an error and produces wrong outputs. |
| o_proj or MLP biases | configs with `attention_bias` covering o_proj, `mlp_bias: true` | **silent**: only `q/k/v_proj.bias` are read, so other biases are dropped. |
| bf16-only numerics | any model whose activations overflow fp16 | **silent** until tested. `export.py` casts to fp16 (`torch_dtype=torch.float16`), and the engine is fp16 throughout. |
| Dynamic RoPE scaling (NTK-by-length) | | **silent**: the exported `inv_freq` is a fixed snapshot. |
| Quantized weights, paged KV cache, non-greedy sampling | | not implemented (README "Not supported") |

A cheap hardening step, not yet done, would be for `export.py` to whitelist `model_type` and assert
`hidden_act == "silu"` and the absence of unused bias tensors. That would turn the silent rows into
loud ones.

## Recipe for a new family

1. **Read the HF modeling file** for the family and compare it with the architecture section above:
   norm, activation, RoPE layout and coverage, biases, attention extras (window, soft-cap, ALiBi),
   embedding scaling, tied head.
2. **If it matches**, export and load. The exporter prints the detected config (`tied`, `qkv_bias`,
   rope type, window), and `Engine(...)` reports the same fields back from the header (`hi_config`).
3. **If it differs by a config-level feature** (a new bias, a new rope type), follow the pattern of
   the existing ones: a header flag, a conditional tensor in the export order, a pointer in `Layer`,
   and an epilogue. QKV bias, the largest of the existing features, took about 20 lines this way.
4. **If it differs structurally** (MoE, sliding window, a different norm), the work is new kernels,
   not porting. Stop and scope it separately.
5. **Add the tag** to `scripts/run_all.sh` (the `HF` and `GB` tables at lines 21–30) so that the whole
   test ladder below runs.

## The verification ladder

Each level catches a class of bug that the levels below it cannot. All four run from
`scripts/run_all.sh`.

### 1. Kernel tests: `tests/test_kernels.py`

Every kernel is checked against a PyTorch reference: fp32 math, or HF's exact fp16 order of
operations where the kernel claims to reproduce it. Coverage:

- RMSNorm at 5 shapes.
- The GEMV with each epilogue (plain, fused norm, residual, SwiGLU).
- All 27 GEMM configs (`cfg=-1` heuristic plus 0–26) and the forced split-K factors, with bias,
  residual and SwiGLU epilogues, including a partial last column tile. A sentinel row checks that
  nothing is written past the output.
- A split-K GEMM captured in a CUDA graph and replayed after its workspace was outgrown.
- argmax on NaN / -inf rows.
- RoPE + KV-cache write, prefill and per-row decode forms, with **tolerance 0** against HF's fp16
  order.
- The fused QKV+bias+RoPE+cache-write GEMV, against the unfused path, with bias on (Qwen shapes) and off.
- Flash prefill at 5 head/GQA configs × 7 (T, p0) cases, including chunked-prefill offsets.
- Decode attention at 12 (D, G) pairs × 5 length sets up to 4,000 tokens, and one work buffer reused
  across batch sizes.

**Result:** 899 checks, all OK, on both GPUs ([results/2080ti/test_kernels.log](../results/2080ti/test_kernels.log),
[results/a6000/test_kernels.log](../results/a6000/test_kernels.log)).

**For a new family:** add its shapes (hidden, inter, head_dim, G, bias) to the parametrized lists.
The QKV bias path is tested exactly this way, at Qwen2.5-1.5B's (D=128, 12/2 heads, H=1536) and
Qwen2.5-0.5B's (D=64, 14/2, H=896) shapes ([test_kernels.py:265](../tests/test_kernels.py:265)).

### 2. Greedy token gate vs HF: `tests/gate.py`

Engine and HF transformers (fp16, same GPU) generate 128 greedy tokens for each of 7 prompts: 6 short
ones, plus one 700-token prompt that exercises multi-tile attention and chunked prefill
(`max_prefill=512`). Each prompt runs single-sequence with and without CUDA graphs, and the first 6
also run together as a batch of 6 slots at different positions: 20 runs in all. The gate also
reports the max difference of the last-position prefill logits (recorded, not gated) and checks that
out-of-range slots, tokens, batch sizes and positions are rejected.

**Near-tie tolerance:** a run may diverge from HF only at a step where HF's own top-2 logit gap is
below `--tie` (default 0.05). At such a step the summation order legitimately decides the argmax, and
the run is counted as "near-tie, OK". Any other divergence fails the gate (exit code 1). Every committed
divergence is at a gap of 0, 0.0156 or 0.0313 (one A6000 Llama-3.2-1B run). The tolerance never had
to stretch to 0.05.

**What it catches:** wrong weight order, a missing bias, a wrong RoPE layout, broken
batching/slot indexing, graph-capture bugs. **What it misses:** errors that change logits without
changing the argmax over 128 tokens.

For models that don't fit next to an HF copy, `--hf-first` runs HF first, keeps its tokens, gaps
and logits on the host, frees it, and then loads the engine.

### 3. Perplexity and distribution match: `tests/ppl.py`

WikiText-2 test split, 16 non-overlapping 1,024-token windows (16,368 tokens for Llama/Mistral,
16,352 for Qwen), with both engines on the same tokens. It reports perplexity for both, top-1
agreement at every position, mean KL(HF ‖ hipinfer) and the max logit difference. This checks the
whole distribution, not only the argmax. It has **no pass/fail threshold**: the numbers are
written to `ppl.json` and judged by reading them. On the committed runs:

- PPL matches within 0.0003–0.0062 % on single GPUs, and within 0.0098 % sharded.
- Top-1 agreement is 99.74–99.91 %.
- Mean KL is 4.6e-6 to 1.6e-5.

With `--tp N`, HF runs first over the same GPUs (`device_map="auto"`). That is how the 7–8B models,
which don't fit one 11 GB card, are checked.

### 4. Tensor parallel vs one GPU: `tests/test_tp.py`

The same weights run on one GPU and sharded over N, for each all-reduce backend (one-shot, NCCL,
auto):

1. An all-reduce self-test with exact fp16 sums, including messages larger than one staging slot.
   It must be exactly 0.
2. Prefill logits at every position: max diff and top-1 agreement against one GPU.
3. Greedy decode for 5 single prompts (lengths 7–700, the last one chunked) and a batch of 8: 13 runs,
   token for token against one GPU, with the same near-tie rule applied to the 1-GPU engine's top-2 gaps.
4. At tp=2, the backends must agree with each other **bitwise** in logits and tokens.

The prompts are random token ids, not text, so this test checks equivalence of the two engines, not
model quality. With `--no-ref` (a model too big for one GPU) only checks 1 and 4 run. For those
models, `tests/ppl.py --tp 2` against HF supplies the accuracy check. **Result:** every committed
`tp2_test.json` (8 models on 2× RTX 2080 Ti) has `pass: true`. The only divergences are 2 of 13 runs
each on Llama-3.2-1B and Qwen2.5-0.5B (in every backend), all at a 1-GPU top-2 gap of 0.0078 (one fp16 step).

### Order of work for a new family

Kernel tests at the new shapes come first, then the gate, then perplexity, then TP. A failure at a
level is debugged at that level: a perplexity regression with a passing gate points at
something small and systematic (a rounding point, eps, a scale), not at the weight layout.
