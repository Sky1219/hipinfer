# Merged tree on the A6000 (sm_86): correctness only

Run on 2026-10-18 by `scripts/edmonds_all.sh start-a6000` from commit `439831b` (the merge of int8,
speculative decoding and eviction scoring), on an otherwise idle RTX A6000. No benchmarks: the
card throttles under sustained load ([../THERMAL.md](../THERMAL.md)), and these runs only check that
the merged code computes the same things on Ampere as on Turing.

| check | result |
|---|---|
| kernel tests (`test_kernels.log`) | ALL PASS |
| fp16 gates, TinyLlama / Llama-3.2-1B / Llama-3.2-3B / Qwen2.5-1.5B | pass, 20 / 11 / 20 / 17 of 20 runs token-identical to HF: the same runs, diverging at the same steps, as the committed `../<model>/gate.json` from before the merge |
| int8 gates vs fake-quantized HF, TinyLlama / Llama-3.2-1B | pass, 18 / 17 of 20 identical |
| int8 gates vs fp16 HF, TinyLlama / Llama-3.2-1B | pass (divergence expected and allowed), 8 / 5 of 20 identical |
| speculative decoding exact match (`test_spec.log`) | 105/105 runs identical to plain greedy |
| eviction kernels, default and portable (`test_eviction*.log`) | ALL PASS |
| eviction in the engine, Llama-3.2-1B and Qwen2.5-1.5B | ALL PASS (38 checks each) after the fix in `32c801d` |

`run.log` is the first pass, which found the eviction bug: after `hi_evict`, decode added each new
key's attention mass on top of an evicted key's stale score (4 failures per model). The two
`test_eviction_engine_*.log` files are the rerun on a build with the fix (`build_fix.log`);
`test_eviction.log` was rerun on that build too.
