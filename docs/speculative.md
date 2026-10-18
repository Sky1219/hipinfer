# Greedy speculative decoding

A small draft model (Llama-3.2-1B) proposes `k` tokens with ordinary greedy decode steps. The
target (Llama-3.2-3B) checks all `k` in one pass that starts at its current position. The
longest prefix where the target's argmax agrees with the draft is accepted, plus the target's
own next token, so every round yields between 1 and `k + 1` tokens. Both engines are separate
`Engine` objects in one process on one GPU. The two models share a tokenizer.

```python
target = hi.Engine("llama32-3b.bin", max_batch=1)
draft = hi.Engine("llama32-1b.bin", max_batch=1)
tokens = target.generate_speculative(draft, prompt_ids, n_new=128, k=4)
assert tokens == target.generate(prompt_ids, 128)   # always, token for token
```

## The guarantee, and why verification can't use the prefill path

The output equals plain greedy hipinfer output **token for token**. This does not mean "the
same up to fp16 near-ties": it is the same sequence of argmaxes over bit-identical logits.
`tests/test_spec.py` checks this on the `tests/gate.py` prompts for k = 2, 4, 6, 8.

The obvious verifier is the prefill path, `hi_prefill_logits` at offset `p0`, which already
returns logits for every position. It is not exact. Prefill runs tensor-core GEMMs and
FlashAttention, while plain decode runs fp32-FMA GEMVs and split-K decode attention. The two
round differently, so near the fp16 ties that `tests/gate.py` tolerates, a token flips. The
k/v the prefill path writes for accepted tokens also differ slightly from the ones decode would
have written, and that error carries into every later step. `scripts/bench_spec.py` still runs
this variant (`verify="prefill"`) and counts how often its output diverges from plain greedy
(see Results).

So `hi_verify` runs the target's `k + 1` rows through **the kernels a B=1 decode step uses**,
built so that each row's arithmetic is independent of how many rows there are:

| piece | B=1 decode | verify, R = k + 1 rows | why the rows match B=1 bit for bit |
|---|---|---|---|
| QKV + RoPE + cache write | fused GEMV, RMSNorm prologue | same kernel, R rows, `kv_slot` = the sequence's slot | the GEMV's per-row FMA order is fixed (same 16-byte loads per lane, same warp reduction) for every row count |
| attention | split-K decode attention, `S = decode_splits(1, Hkv)` | same kernel, R queries, **the same S**, every query reads one slot | a query's splits, partials and merge depend only on its position and S; all R rows' k/v are written before attention, and query i reads positions <= p0 + i only |
| o-proj, MLP, lm_head | GEMV | GEMV with R rows | as above |
| RMSNorm, R > 8 | fused into the GEMV | `rmsnorm_kernel`, then the unfused GEMV | the fused prologue and `rmsnorm_kernel` are the same code (256 threads, same strided sum, same `block_sum`) |
| argmax | `argmax_kernel` | `argmax_kernel`, R rows | same |

Row i's logits are therefore exactly what decode step `p0 + i` of plain greedy decoding
computes, and the k/v it leaves in the cache are exactly what that step would have left.
Positions `p0..p0+i-1` were written by verify rows `0..i-1`, so by induction the cache matches
too. Two conditions keep this true. Plain B=1 decode must take the GEMV path (the default
`HIPINFER_GEMV_MAX` >= 1, which `hi_verify` checks). The GEMV's loads in flight per lane
(`GEMV_U`, 4 by default) must be the same for 1–2 rows as for more, which they are unless the
library is built with a different `-DGEMV_U`.

Two kernel changes make this possible, and both leave existing behaviour unchanged:

* The QKV GEMV epilogue and both decode-attention kernels take a `kv_slot` (default -1, meaning
  row b uses slot b). With `kv_slot` >= 0, every row is one position of that slot.
* The GEMV accepts up to 16 rows without the fused norm (8 with it, the shared-memory limit on
  Turing), so k = 8 (9 rows) is still one pass over the weights rather than two.

The self-draft test is the direct check: a model drafting for a second copy of itself must
have **every** draft token accepted at every k, because the draft runs plain decode and the
verifier computes the same logits. A single differing logit anywhere would eventually flip
an argmax.

## One round

State: the target's cache holds positions `< p`, and `t0` is the pending token at `p` (the last
token emitted, not yet fed to either model).

1. **Draft.** Rewind the draft to length `p` with next token `t0` (`hi_rewind`). Run `k` graph
   decode steps (the existing B=1 CUDA graph, one launch per step), and read the `k` proposals
   `d1..dk` from its history.
2. **Verify.** `hi_verify(target, slot 0, [t0, d1..dk], p0 = p)` runs one captured graph per
   row count and returns the argmax `y0..yk` after each row.
3. **Accept.** Take the largest `a` with `d1..da == y0..y(a-1)`. Emit `d1..da, ya`.
4. **Roll back.** No copying is needed: the cache is contiguous per slot and attention reads
   positions `<= pos` only, so a rejected suffix is simply overwritten later. The target's
   valid prefix is now `p + a + 1` long. The draft consumed `t0, d1..d(k-1)`. If all `k` were
   accepted, it never consumed `dk`, so the next round first runs one catch-up decode step on
   it.

Near the end the round size shrinks, `k_eff = min(k, tokens still wanted - 1)`, so no row is
wasted past the last token.

## API

C (`src/api.cu`):

* `hi_verify(e, slot, tokens, R, p0, out, logits_dst, use_graph)`: the greedy next token after
  each of the R tokens fed at `p0..p0+R-1`, with R <= `hi_verify_max_rows()` (16). It writes
  the rows' k/v and leaves the slot's next token and position alone. Optionally it also copies
  the `[R, vocab]` logits to `logits_dst`. Dense models only, on one GPU.
* `hi_rewind(e, slot, len, tok)`: the cache-length reset. It keeps positions `[0, len)` and
  makes the next decode step feed `tok` at `len`.

Python (`scripts/hipinfer.py`): `Engine.verify(slot, tokens, p0)`, `Engine.rewind(slot, length,
next_token)`, and `Engine.generate_speculative(draft, prompt, n_new, k, stats=None)`.

The round loop runs on the host. Each round makes about four synchronising calls: draft decode,
draft history, and the verify copy in and out. Each decode step and each verify pass is one
CUDA graph launch.

## Cost model

Write `t_d` and `t_t` for the per-token decode times of the draft and target, and `t_v(R)` for
one verify pass of R rows. A round of `k` proposals costs about `k * t_d + t_v(k + 1)` and
yields `E = tokens_per_round` tokens, against `E * t_t` for plain decode. On a 2080 Ti the 1B
model is not that small next to the 3B model. Both are weight-bandwidth bound, and
`t_d / t_t` is about 0.4 (2.5 GB vs 6.4 GB of fp16 weights). So each draft token costs 40 % of
a target step, and the useful k is small.

## Tests and results

* [tests/test_spec_logic.py](../tests/test_spec_logic.py) runs on the CPU with no GPU. It drives
  the round loop ([scripts/speculative.py](../scripts/speculative.py)) with stub engines that keep
  a per-position token cache under the engine's contract. Every rewind, verify and first draft
  step asserts that the cache prefix it relies on holds the true sequence, so a rollback error
  fails even when it would only have cost acceptance. The test covers 758 runs over draft
  agreement rates 0 to 1, k from 1 to 15, and lengths from 1 to 64. It also checks the counts and
  the catch-up steps, and three deliberate rollback/accept bugs make it fail.
* [tests/test_spec.py](../tests/test_spec.py) needs a GPU. It runs the exact-match test (3B with a
  1B draft, k = 2, 4, 6, 8, the 7 gate prompts × 128 tokens, with CUDA graphs and once without),
  the self-draft test (1B and TinyLlama, acceptance must be exactly 1), and the argument checks.
* [scripts/bench_spec.py](../scripts/bench_spec.py) needs a GPU. Per k, it reports tokens/s
  against plain decode, acceptance, tokens per round, the share of time spent drafting and
  verifying, the cost model's prediction, and how often `verify="prefill"` diverges from plain
  greedy output.
* `scripts/run_spec.sh` runs all of the above on the 2080 Ti (GPU 2 on edmonds). It also reruns
  the kernel tests and the 1B, 3B and TinyLlama HF token gates and checks them against the
  committed results. Everything goes to `results/2080ti/spec/`.

**Not run on a GPU yet** (edmonds was unreachable when this was written). No speed-up is claimed
until `results/2080ti/spec/` holds one.

## Limits

* Greedy only. Sampling would need rejection sampling against the draft's distribution.
* Batch 1, slot 0. The verify kernels take any slot, but the round loop is single-sequence.
* Dense models on one GPU. MoE and tensor-parallel targets are rejected by `hi_verify`.
* The draft must share the target's vocabulary. The Llama-3.2 1B/3B pair does. TinyLlama has
  no smaller sibling here, so it appears only in the self-draft exactness test.
