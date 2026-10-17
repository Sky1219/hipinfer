# Mixture-of-Experts in hipinfer: design

Status: runs on one GPU, checked on an RTX 2080 Ti against HF transformers on Granite MoE 1B
(`ibm-granite/granite-3.1-1b-a400m-base`, 32 experts, top-8). Results are in
[`results/2080ti/granite-moe-1b/`](../results/2080ti/granite-moe-1b/).

| check | result |
|---|---|
| `tests/test_kernels.py` `test_moe()`: router, top-k, GEMV and grouped expert paths vs PyTorch | all pass |
| token gate (`tests/gate.py`): 7 prompts with and without CUDA graphs, plus batched B=6 | **passes in all 6 runs made with the router near-tie rule** (below), with the same weights: 3 autotuned runs (12/20 token-identical each), autotuning off (15/20, 2 runs accepted by the router rule), forced split-K 2 (13/20) and forced split-K 4 (9/20, 3 accepted by the router rule). Without the rule it passed in 3 of 5 runs, depending on the GEMM configs. Both failures are router near-ties |
| WikiText-2 perplexity (`tests/ppl.py`, 16 windows) | 7.7292 vs HF's 7.7294, top-1 agreement 99.4 % |
| dense models unchanged (TinyLlama token gate) | 20/20 token-identical |
| decode, B=1, ctx 128 | 2.26 ms/token, 30× HF; **63 % of the weight-read floor** (dense models: 81–87 %) |
| prefill, 2,000 tokens | 70 ms, 3.9× HF |

The gate's MoE result used to depend on GEMM configs; the next section explains why and what the
gate does about it now.

Not done yet: tensor or expert parallelism with MoE (rejected at load), OLMoE (needs the QK-norm
kernel), Mixtral on a GPU (too big for the GPUs tested), and AMD.

## Router near-ties and the token gate

**The problem.** Without the rule below, Granite MoE passed the token gate in 3 of 5 runs on the
2080 Ti ([`gate_runs/`](../results/2080ti/granite-moe-1b/gate_runs/), plus `gate.log`). The
autotuner times every tile config on first use, so two runs can pick different configs, which sum
in different orders and round differently. Three autotuned runs passed (all three picked the same
configs). One autotuned run failed on prompt 3 (step 36, HF top-2 gap 0.1094), and so did a run
with autotuning off (`HIPINFER_GEMM_AUTOTUNE=0`; prompt 1, step 70, gap 0.0625). With the configs
held fixed, results are bit-identical between runs, so this is rounding, not a race. The dense
models passed in every run, including runs whose picks differed.

**The cause: router near-ties.** A dense model makes one discrete choice per step, the argmax. An
MoE model makes another one per token and layer: the router's top-8. When the 8th and 9th router
logits are within rounding of each other, a rounding change can pick a different expert. A
different expert changes the hidden state by far more than rounding does, and the change carries
into later positions through the KV cache. [`scripts/moe_router_trace.py`](../scripts/moe_router_trace.py)
checks this. It records the router input of every MoE layer at every position: from
`hi_prefill_debug` for the prompt and from `hi_decode_debug` for each decode step, the same kernels
the gate runs. It routes those inputs with the engine's own `hi_moe_router` and compares the expert
sets with HF's (captured by hooks on HF's router modules).

On the failing prompt 1 run (autotuning off; [`router_trace/noat_p1_all_at_once.log`](../results/2080ti/granite-moe-1b/router_trace/noat_p1_all_at_once.log)):

* 2,136 (position, layer) router points feed the token at step 70. With each side following its own
  routing, the expert sets differ at 22 of them.
* The first difference is at prompt token 9, layer 3. HF's 8th and 9th router logits are exactly
  equal there (0.30664 for experts 10 and 30), and HF takes expert 10. The engine's logits are
  0.30713 and 0.30737, so it takes expert 30.
* Many of the 22 are knock-on effects. Once one position has used a different expert, later
  positions see different hidden states. Two of the 22 have HF margins of 0.030 and 0.045.
* Re-running HF with the engine's experts at all 22 points reproduces all 128 of the engine's
  tokens. A control run that puts HF's own experts back through the same hook reproduces HF's
  tokens exactly.

The stricter test measures each difference against an HF run that already shares the engine's
routing up to that point (`router_tie_check`; described next). On this run, 8 differences remain.
Each has an HF margin between 0 and 0.0039 (at most 8 fp16 ulps of the logit). With those 8 points
routed the engine's way, HF gives the engine's tokens through step 70. On that run, the engine's
and HF's router logits differ by at most 0.0078 (median 0.0020) at any point. That is ordinary
fp16 rounding. So the hypothesis is confirmed: the engine's attention, experts and combine agree
with HF to rounding, and the token divergence comes from router choices that were ties or near-ties.

**Is there an engine rounding mismatch to fix instead?** No.
[`scripts/moe_router_check_inputs.py`](../scripts/moe_router_check_inputs.py) feeds HF's own router
inputs to `hi_moe_router`. That covers 40,392 router evaluations over the gate's 7 prompts: prefill
plus 127 greedy decode steps each
([`router_trace/router_kernel_on_hf_inputs.log`](../results/2080ti/granite-moe-1b/router_trace/router_kernel_on_hf_inputs.log)).

* The kernel rounds at the same points as HF: fp32 accumulation, then one fp16 rounding of each
  logit. Its logits differ from HF's `F.linear` by at most 0.0078. Only the summation order differs,
  and that is cuBLAS's to choose, per GPU and per shape.
* The expert sets differ at 5 of the 40,392 evaluations, all at HF margins of at most 0.0039.
* The expert combine is covered by the transplant above: with the engine's experts, HF reproduces
  all 128 of the engine's tokens.

So the router flips are unavoidable near-ties, not a mismatch the engine could remove. The gate is
the thing to change, and only for router choices that are ties at fp16 resolution.

The same holds for every divergence found on the 2080 Ti
([`router_trace/`](../results/2080ti/granite-moe-1b/router_trace/)). That includes the ones the
token rule had already accepted, and the autotuned prompt 3 failure, which forcing split-K 4 on
every GEMM reproduces exactly (same step, same gap):

| run | prompt | diverges at step | HF top-2 gap | router points adopted | max adopted HF margin | max engine-vs-HF router logit diff |
|---|---|---|---|---|---|---|
| autotune off | 1 | 70 | 0.0625 | 8 | 0.0039 | 0.0078 |
| autotune off | 4 | 43 | 0.0469 | 6 | 0.0010 | 0.0078 |
| autotuned | 0 | 29 | 0.0312 | 6 | 0.0007 | 0.0098 |
| autotuned | 1 | 56 | 0.0000 | 10 | 0.0039 | 0.0078 |
| autotuned | 4 | 43 | 0.0469 | 5 | 0.0020 | 0.0078 |
| split-K 2 | 0 | 29 | 0.0312 | 10 | 0.0042 | 0.0078 |
| split-K 2 | 4 | 43 | 0.0469 | 4 | 0.0012 | 0.0078 |
| split-K 4 | 0 | 29 | 0.0312 | 4 | 0.0015 | 0.0095 |
| split-K 4 | 1 | 75 | 0.0000 | 7 | 0.0039 | 0.0078 |
| split-K 4 | 3 | 36 | 0.1094 | 7 | 0.0015 | 0.0078 |
| split-K 4 | 4 | 43 | 0.0469 | 2 | 0.0010 | 0.0078 |

Every divergence is explained. In each case, HF with the adopted points gives the engine's tokens
through the divergence step, except autotuned prompt 1, where HF's own gap at step 56 is an exact
tie. The other prompts are token-identical in the autotune-off and autotuned runs. Under split-K
they were also identical, but that comes from a one-off scan whose logs are not committed; only the
diverging prompts were re-run with the committed script. The last column is the largest router-logit
difference at any point on the final shared-routing run, so it measures rounding noise. Prompt 4's
divergence at step 43 is worth noting: the token rule accepted it only because 0.0469 is just under
0.05. With the engine's routing, HF itself picks the engine's token, with a gap of 0.11 to 0.14.
So it was a router flip, not a token near-tie. The autotune-off and autotuned sweeps ran with the
router tolerance set to 1, which adopts every difference. No adopted margin came near 0.01, so the
default tolerance gives the same results. The split-K runs use the default.

**The gate rule.** When an MoE run diverges from HF at a step whose HF top-2 gap is at or above
`--tie`, `tests/gate.py` repeats that engine run with the router inputs dumped. It verifies that the
repeat gives the same tokens, then calls `router_tie_check`. That check runs HF on the engine's
tokens (teacher forcing, which equals free-running HF up to the divergence) and walks the router
points and tokens in execution order: for the prompt, by prefill chunk, then layer, then position;
for each decode position, by layer; each token after the routers that feed it. At the first point
where HF and the engine differ:

* a router point whose HF margin (8th minus 9th logit) is `<= --router-tie` (default 0.01) is
  *adopted*: HF re-runs with the engine's experts there, weighted by HF's own logits, and the walk
  restarts;
* a router point with a larger margin fails the run;
* a token whose HF top-2 gap is `>= --tie` fails the run; a smaller gap is a token near-tie and
  passes, as for dense models.

The run passes ("router near-tie, OK") if the walk reaches the divergence step without failing.
This is the dense rule applied to both kinds of discrete choice. Every point where the engine
differs from HF must be a choice HF itself makes on a near-tie, judged against an HF run that shares
the engine's earlier choices. If the engine's experts, attention or combine were wrong, HF would
not reproduce the engine's tokens. If the router were wrong, the differences would come with
margins above the tolerance.

The tolerance, 0.01, is the size of the rounding noise: the largest router-logit difference between
engine and HF seen on any shared-routing run is 0.0098, while the largest adopted margin is 0.0042.
As a negative control, `moe_router_trace.py --inject` adds noise to HF's attention output in one
layer, so the engine looks as if it had a bug there
([`router_trace/inject_*.log`](../results/2080ti/granite-moe-1b/router_trace/)). With 2 % noise in
layer 5, the check fails: the first non-tie difference is at prompt token 7, layer 6, with margin
0.021, and router logits on that run differ by up to 0.34. With 0.5 % noise in layer 10 or 0.2 % in
layer 20, HF's tokens through step 70 do not change at all, so no token-level gate could see those
faults. The check explains the divergence as before (12 and 8 adopted points, margins at most
0.0039). Use `scripts/layer_diff.py` to find faults that small.

Results with the rule, all with the same weights file
([`gate_runs/router_check_*.log`](../results/2080ti/granite-moe-1b/gate_runs/)):

| GEMM configs | token-identical | accepted by the router check | result |
|---|---|---|---|
| autotuned, run 1 | 12/20 | 0 | pass |
| autotuned, run 2 | 12/20 | 0 | pass |
| autotuned, run 3 | 12/20 | 0 | pass |
| autotuning off | 15/20 | 2 (prompt 1, step 70) | pass (failed without the rule) |
| split-K 2 forced on every GEMM | 13/20 | 0 | pass |
| split-K 4 forced on every GEMM | 9/20 | 3 (prompt 3 single and batched, step 36) | pass (the old autotuned prompt 3 failure) |

All other divergences in these runs are token near-ties (HF gap below 0.05). The three autotuned
runs gave the same gate results (same divergences at the same steps), so they likely cover only one point of the config space;
autotuning off and the forced split-K runs cover three more. The kernel tests pass on the same
build (991 checks, [`test_kernels.log`](../results/2080ti/granite-moe-1b/test_kernels.log)), and
TinyLlama's gate is unchanged at 20/20 ([`gate_runs/tinyllama_router_check_build.log`](../results/2080ti/granite-moe-1b/gate_runs/tinyllama_router_check_build.log));
the router check runs only for MoE models.

Limits: the check needs the HF model in memory, so `--hf-first` skips it. The hook expects HF's
router modules to return `(top_k_index, top_k_weights, router_logits)`, as Granite MoE's do in
transformers 5.x; Mixtral and OLMoE are untested. Only the 2080 Ti has been checked.

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
# router top-8 trace and near-tie check for one gate prompt (the failing one above):
HIPINFER_GEMM_AUTOTUNE=0 python scripts/moe_router_trace.py --weights granite-moe-1b.bin --prompt 1 --all-at-once
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
