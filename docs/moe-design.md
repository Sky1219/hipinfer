# Mixture-of-Experts in hipinfer: design

Status: runs on one GPU, checked on an RTX 2080 Ti against HF transformers on Granite MoE 1B
(`ibm-granite/granite-3.1-1b-a400m-base`, 32 experts, top-8). Results are in
[`results/2080ti/granite-moe-1b/`](../results/2080ti/granite-moe-1b/).

| check | result |
|---|---|
| `tests/test_kernels.py` `test_moe()`: router, top-k, GEMV and grouped expert paths vs PyTorch | all pass |
| token gate (`tests/gate.py`): 7 prompts with and without CUDA graphs, plus batched B=6 | pass: 12/20 runs token-identical; the other 8 diverge where HF's own top-2 gap is at most 0.05 |
| WikiText-2 perplexity (`tests/ppl.py`, 16 windows) | 7.7292 vs HF's 7.7294, top-1 agreement 99.4 % |
| dense models unchanged (TinyLlama token gate) | 20/20 token-identical |
| decode, B=1, ctx 128 | 2.26 ms/token, 30× HF; **63 % of the weight-read floor** (dense models: 81–87 %) |
| prefill, 2,000 tokens | 70 ms, 3.9× HF |

Not done yet: tensor or expert parallelism with MoE (rejected at load), OLMoE (needs the QK-norm
kernel), Mixtral on a GPU (too big for the GPUs tested), and AMD.

## Target models

| model | layers | hidden | experts x top-k | expert inter | attn | notes |
|---|---|---|---|---|---|---|
| `hf-internal-testing/tiny-random-MixtralForCausalLM` | 2 | 64 | 4 x 2 | 128 | 4/2 heads, D=16 | CPU math check only (D=16 is not a GPU head size) |
| `ibm-granite/granite-3.1-1b-a400m-base` | 24 | 1024 | 32 x 8 | 512 | 16/8 heads, D=64 | first GPU target, 1 GPU (2.6 GB fp16) |
| `allenai/OLMoE-1B-7B-0924` | 16 | 2048 | 64 x 8 | 1024 | 16/16 heads, D=128, QK-norm | 2-GPU expert-parallel target (13.8 GB fp16) |
| Mixtral-8x7B (later) | 32 | 4096 | 8 x 2 | 14336 | 32/8 heads | needs more memory than 2 x 11 GB in fp16 |

All three are Llama-shaped (pre-norm, RMSNorm, RoPE, GQA, SwiGLU experts); only the MLP block
changes. The differences that matter for an engine that reproduces HF's fp16 rounding:

| | Mixtral | Granite MoE | OLMoE |
|---|---|---|---|
| router | softmax over all E (fp32), top-k, renormalise | top-k of logits, softmax over the k | softmax over all E, top-k, **no** renormalise |
| routing weight dtype at combine | fp32 | fp16 | fp16 |
| scalar multipliers | none | embedding x12, residual x0.22, attention 1/64 (not 1/sqrt(D)), logits /6 | none |
| attention extras | none | none | RMSNorm over the whole q and k projections (QK-norm) |

Router math, one form for all three: logits `l = fp16(x_norm @ Wr^T)` (HF rounds the router
linear to fp16 before `.float()`), select the k largest logits (softmax is monotonic, so top-k on
probabilities and on logits pick the same experts), then

* `norm_topk` (Mixtral, Granite): `w_i = exp(l_i - m) / sum_{j in topk} exp(l_j - m)`
* otherwise (OLMoE): `w_i = exp(l_i - m) / sum_{all j} exp(l_j - m)`

and round `w` to fp16 when the model does (Granite, OLMoE). Granite's "softmax over the top-k
logits" equals Mixtral's "renormalised top-k of the full softmax" up to fp32 rounding.

Combine, as HF's default `grouped_mm` experts implementation (transformers 5.x) does it: each
expert's down projection is rounded to fp16, multiplied by its routing weight (fp32 product for
fp32 weights; fp16-rounded product for fp16 weights), summed over the k experts in fp32, rounded
to fp16 once, then added to the residual (fp16 add). The `eager` implementation instead
accumulates in fp16 in expert order (`index_add_`); the engine follows `grouped_mm`, which is the
default and what `tests/gate.py` compares against.

## Weight file: version 3 (MoE)

Dense models keep exporting version 2, byte for byte as before. MoE models export version 3:

```
header (128 bytes)
  bytes  0..55  exactly the version-2 fields (magic, version=3, vocab, hidden, inter, layers, heads,
                kv_heads, head_dim, max_pos, eps, attention_scaling, flags, sliding_window)
                inter = the per-expert intermediate size
                flags bit 0 tied lm_head, bit 1 qkv bias, bit 2 MoE (always set in v3),
                bit 3 qk_norm (OLMoE; reserved, the engine rejects it for now)
  bytes 56..63  zero
  bytes 64..79  int32 n_experts, int32 top_k, int32 moe_flags, float moe_out_scale
                moe_flags bit 0 norm_topk (renormalise the k weights), bit 1 fp16 routing weights
                moe_out_scale multiplies the MoE block output before the residual add (Granite's
                residual_multiplier; 1.0 otherwise), applied in fp16 exactly as HF does
  bytes 80..127 zero
inv_freq fp32 [D/2], embed, final_norm, lm_head (unless tied)        -- as version 2
per layer:
  attn_norm, wqkv, (bqkv), wo, mlp_norm                              -- as version 2
  (q_norm [Hq*D], k_norm [Hkv*D] if qk_norm)                         -- reserved for OLMoE
  router   [E, hidden]
  wgu      [E][2*inter, hidden]   each expert's gate/up rows interleaved in groups of 16
  wdown    [E][hidden, inter]
```

Every tensor still starts on a 256-byte boundary, so expert `e` of a layer is a contiguous block
at `offset + e * bytes_per_expert`, and an expert-parallel rank reads its experts as one row range.

**Granite's multipliers are folded at export** so the engine runs the plain Llama graph:

* `embedding_multiplier` (12): `embed = fp16(E * 12)`. HF computes `fp16(E[tok] * 12)` at run
  time, the same value, so this is exact. Granite ties `lm_head` to `E`, so the file then carries
  an untied lm_head (below).
* `logits_scaling` (6): `lm_head = fp16(E / 6)`. Not bit-exact (HF divides the fp16 logits), but
  argmax is unaffected and logits differ by about one fp16 ulp.
* `attention_multiplier` (1/64): the engine's attention uses `1/sqrt(D)` = 1/8 for D = 64, so the
  q rows of `wqkv` are scaled by `attention_multiplier * sqrt(D)` = 1/8, a power of two: exact.
  RoPE is linear, so scaling q before it is the same as after.
* `residual_multiplier` (0.22): the attention side is folded into `wo` (`fp16(Wo * 0.22)`, about
  one ulp off HF's `fp16(fp16(o) * 0.22)`); the MoE side stays in the header as `moe_out_scale`,
  applied exactly by the combine kernel (which is new code, so the extra multiply is free).

## Kernels

All MoE kernels are at the end of `src/kernels.cuh`; `tests/test_kernels.py` `test_moe()` checks each one.

1. **Router + top-k** (`moe_router_kernel`): one block per token. Reads the normalised hidden row
   (fp16), computes the E router logits with one warp per expert (fp32 accumulate, rounded to fp16
   as HF does), then thread 0 picks the top-k (ties to the lower index, like `torch.topk` on CPU),
   computes the weights (softmax over the k, or over all E), optionally rounds them to fp16, and
   writes `topk_idx[T, k]` and `topk_w[T, k]` (fp32). E <= 256, k <= 16. The router is ~0.1% of the
   layer's FLOPs; one kernel instead of GEMM + softmax + topk saves two launches per layer.

2. **Grouping by expert** (`moe_count_kernel`, `moe_scatter_kernel`): counting sort of the T*k
   (token, slot) pairs by expert. Count per expert (shared-memory atomics), exclusive prefix sum to
   `offsets[E+1]`, then scatter: `perm[offsets[e] + rank] = t*k + slot`. The order inside an
   expert does not change any result (each row is independent). A gather kernel copies the
   selected tokens' normalised rows into a contiguous `[T*k, hidden]` buffer in expert order.

3. **Batch-1 / small-batch decode: fused expert GEMV** (`moe_gemv_gu_kernel`,
   `moe_gemv_down_kernel`). For B <= 4 tokens there is nothing to group: each of the B*k
   (token, expert) pairs is one row of work. Grid `(row blocks, B*k)`; each block reads its
   pair's expert id from `topk_idx` on the device, so the launch is static and CUDA-graph
   capturable. Gate/up: the same warp-per-row-pair SwiGLU GEMV as the dense path, writing
   `act[B*k, inter]`. Down + combine: one warp per hidden output column `h` of one token computes
   the k dot products (k rows of `wdown[e_i]` against `act[pair]`), rounds each to fp16,
   multiplies by its weight, sums in fp32, rounds once, scales by `moe_out_scale`, and adds the
   residual; no atomics, deterministic, and the k expert outputs never round-trip through memory.
   Weight bytes per token: k experts x 3 x hidden x inter (Granite: 8 x 3 x 1024 x 512 x 2 B =
   25 MB per layer, 604 MB over 24 layers, against 2.4 GB for all experts).

4. **Prefill / batched decode: grouped expert GEMM.** Tokens are grouped (2), then per expert
   with rows: `hi_gemm(rows_e, wgu[e], epi 2)` then `hi_gemm(act_e, wdown[e], epi 0)` into
   `y[T*k, hidden]` in expert order, then `moe_combine_kernel` (one block per token) sums its k
   rows of `y` with the weights and adds the residual. The first draft reads the per-expert counts
   back to the host (one sync per layer) and launches up to E GEMM pairs, reusing the tuned
   tensor-core GEMMs as they are. Next step: one device-side grouped GEMM launch whose blocks walk
   `offsets` (a tile -> (expert, row tile) map built on the device), which removes the sync and
   makes batched decode graph-capturable; see "Later" below.

## Engine integration

* `Layer` gains `router, ewgu, ewdown` (base pointers of the E stacked experts); `Config` gains
  `n_experts, top_k, moe_flags, moe_scale`. Dense layers are untouched.
* `decode_step`: the attention half is unchanged. The MLP half becomes, for MoE:
  `rmsnorm -> moe_router -> moe_gemv_gu -> moe_gemv_down(+combine, +residual)` for
  B <= `HIPINFER_GEMV_MAX`, and the grouped GEMM path above it. The B <= 4 path is fully
  device-side, so decode stays graph-captured; the grouped path is run eagerly for now (the engine
  disables graph capture for MoE batches above the GEMV limit).
* `prefill_layers` / `mlp_rows`: MoE layers take the grouped GEMM path.
* Scratch buffers: `topk_idx [Mmax*k] int`, `topk_w [Mmax*k] float`, `moe_off [E+1]`,
  `moe_perm [Mmax*k]`, `moe_rows [Mmax*k, hidden]`, `moe_act [Mmax*k, inter]`,
  `moe_y [Mmax*k, hidden]`.
* Tensor parallelism with MoE is rejected at load for now (expert parallelism below replaces it).

## Expert parallelism on 2 GPUs

OLMoE-1B-7B is 13.8 GB in fp16: too big for one 11 GB 2080 Ti, but 6.4 B of its 6.9 B parameters
are experts. Split the 64 experts 32/32 and replicate the rest: 3.2 B experts + 0.5 B shared
weights = 7.4 GB per GPU.

**Step 1: TP attention + expert-sharded MoE, combine by all-reduce.** The existing tensor
parallelism leaves every rank with the full hidden states of every token after the attention
all-reduce. Each rank routes every token (the router is replicated, so all ranks agree), computes
only the (token, expert) pairs whose expert it owns, and writes a partial MoE output (zero for
tokens with no local expert); the existing all-reduce after the MLP sums the partials. No new
collective, and the decode graph stays the same shape. Load imbalance shows up as one rank
waiting at the all-reduce. Rank 0 adds the residual, as for the dense row-parallel layers.

**Step 2: data-parallel attention + all-to-all dispatch/combine** (the DeepSeek/pplx-kernels
layout, and what scales past 2 GPUs). Each rank owns its own sequences (KV cache slots) and full
attention weights; only the experts are sharded. Per MoE layer:

1. route locally; count how many rows go to each rank (`send_counts[world]`);
2. exchange counts (a tiny all-to-all, or all-gather of `world x world` ints);
3. **dispatch**: all-to-all of the selected hidden rows (fp16 `[rows, hidden]`), plus their
   (local expert id, weight, source token) metadata;
4. grouped expert GEMMs on what arrived (kernels from the single-GPU path);
5. **combine**: all-to-all back to the source ranks, then the weighted sum + residual
   (`moe_combine_kernel`, with rows indexed through the receive layout).

With NCCL first: `ncclGroupStart(); for peer: ncclSend/ncclRecv(...); ncclGroupEnd()` (NCCL has no
all-to-allv; grouped point-to-point is the standard idiom). The counts exchange forces a host
sync per layer unless receive buffers are sized for the worst case (`T*k` rows from each peer),
which is what a capturable decode needs: send fixed-size padded buffers with the counts inside.

**Step 3: custom one-shot all-to-all** for decode, where messages are a few KB and NCCL's
per-call latency dominates (same reasoning as the existing one-shot all-reduce, which beats NCCL
under ~512 KB here). Each rank writes its rows for peer p straight into p's receive buffer over P2P
(or into pinned host staging without P2P, as the all-reduce does), then raises a flag; the peer
spins on the flag, computes, and writes the results back the same way. One kernel per direction,
no host involvement, graph-capturable, and the dispatch write can be fused into the router kernel's
epilogue. The 2080 Tis on edmonds have no NVLink, so P2P (if enabled) runs over PCIe; the
pinned-host path costs two PCIe crossings but still beats NCCL for small messages.

For one sequence at batch 1, DP attention leaves one GPU idle during attention, so step 1 (TP
attention) is the better batch-1 layout on 2 GPUs; step 2 wins once there are >= 2 sequences per
GPU. The engine keeps both: the choice is per load.

**OLMoE also needs** (beyond EP): QK-norm, an RMSNorm over the whole q projection (all heads
together) and over the whole k projection, before RoPE. That breaks the fused
QKV + RoPE + cache-write GEMV epilogue (it needs a full-row reduction first), so OLMoE runs
`gemv/gemm qkv -> qk_norm kernel -> rope_cache`. Under TP attention the q-norm sum of squares
spans ranks: an all-reduce of one float per row per layer (or replicate attention, step 2).
Router: `norm_topk = 0`, fp16 weights. D = 128, 16/16 heads: supported head size.

## Running the checks

```bash
python -c "import sys; sys.path[:0] = ['tests']; import test_kernels as t; t.test_moe(); print('FAILS', t.FAILS)"
python scripts/export.py --model ibm-granite/granite-3.1-1b-a400m-base --out granite-moe-1b.bin
python tests/test_moe_ref.py --quick     # CPU reference engine vs HF on the exported file
python tests/gate.py --model ibm-granite/granite-3.1-1b-a400m-base --weights granite-moe-1b.bin
python tests/ppl.py  --model ibm-granite/granite-3.1-1b-a400m-base --weights granite-moe-1b.bin
```

If `test_moe()` fails, check the router first (indices, then weights), then the GEMV path, then the
grouped path (`hi.moe_experts(..., path=1|2)` forces one). `scripts/ref_engine.py`'s `moe_route` /
`moe_experts` are the reference and run on CUDA tensors too.

## Later

* Decode bandwidth. B=1 decode reaches 63 % of the weight-read floor against 81–87 % for the dense
  models: each token reads 8 of 32 small experts (512 × 1,024 per projection), so every GEMV is
  short and the reads are scattered. One fused launch over the selected experts is the first thing
  to try.
* Device-side grouped GEMM (one launch over all experts) for prefill and capturable batched decode.
* Expert-aware batched decode GEMV: at B = 8 with top-8 of 32 experts most experts are hit by
  more than one token; group pairs by expert so each expert's weights stream once per step.
* Shared experts (Qwen-MoE, DeepSeek): one more dense SwiGLU per layer, added in the combine.
* Load-balance stats per layer (tokens per expert) exposed through the API, for EP tuning.
