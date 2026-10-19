# The merged tree on the 2080 Tis, 2026-10-18

One run of every GPU check after merging int8 weights, speculative decoding and eviction scoring
(`scripts/edmonds_all.sh`, source at commit `cf5dcc2`). Two queues ran side by side, each in its own
copy of the tree:

- **GPU 0:** `scripts/run_int8.sh` (kernel tests, int8 gates, perplexity, benchmarks, llama.cpp Q8_0
  and F16) for TinyLlama, Llama-3.2-1B and Llama-3.2-3B, then `scripts/eviction_gpu.sh`.
- **GPU 2:** `scripts/run_spec.sh` (kernel tests, the fp16 regression gates, the speculative
  exact-match test and benchmark).
- **Both GPUs:** `tests/test_tp.py --tp 2` on Llama-3.2-3B once both queues were done
  (`test_tp_merged.log`: TP TEST PASS).

## What failed, and the reruns

- **Llama-3.2-3B int8 gates and perplexity ran out of memory.** HF fp16 and the engine don't fit
  together on 11 GB. `run_int8.sh` now runs HF first for 3B and 8B (`e5d6d9f`), and those steps were
  rerun (`run_int8_3b_rerun.log`; all pass).
- **The eviction engine test failed** on every policy: decode after `hi_evict` added each new key's
  mass on top of an evicted key's stale score. Fixed in `32c801d`. `eviction/` is the rerun on the
  fixed build: every step passes.
- **The eviction benchmark ran out of memory** loading the engine. PyTorch's cache still held the
  selection benchmarks' tensors. Fixed in `4c737b9`; `eviction/bench_eviction.json` is from the fixed
  build.
- **The regression check flagged every fp16 gate as changed.** That was the check's fault: the merged
  `gate.py` records `step` on matching runs too (`5ead8de`). With that fixed, TinyLlama matches its
  committed gate run for run. Llama-3.2-1B and 3B differ in a few runs, all at HF top-2 gaps of 0
  or 0.0156. Unmerged `main`, rerun on the same GPU the same evening, also differs from its own
  committed gates on those runs (1B: 14/20 identical against 13/20 committed, merged 15/20; 3B:
  19/20 against 20/20, merged 19/20). So the exact winner of a near-tie varies between sessions
  even without code changes. On the A6000 the merged tree matched all four committed gates run for
  run ([../a6000/merged/](../a6000/merged/)).

## Clocks during the benchmarks

`gpu_log_all.csv` logs every GPU's SM clock, temperature, power and clock-event reasons every 5 s
from 21:06:49. A window runs from the previous step's result file to its own. Only samples at
≥ 50 % utilization count. "Thermal" is the share of those samples with the driver's thermal
slowdown active. Every other sample was held by the 250 W power cap, which is normal for these
cards under load.

| window | GPU | SM clock min / median | peak | thermal | verdict |
|---|---|---|---|---|---|
| TinyLlama int8 / fp16 / llama.cpp Q8_0 / F16 | 0 | 1,500–1,710 / 1,740–1,837 MHz | 67 °C | 0 % | clean (the int8 window began 40 s before the log) |
| Llama-3.2-1B int8 / fp16 / llama.cpp Q8_0 / F16 | 0 | 1,395–1,785 / 1,582–1,800 MHz | 72 °C | 0 % | clean |
| Llama-3.2-3B int8 / fp16 | 0 | 1,365 / 1,530 and 1,350 / 1,440 MHz | 88 °C | 26 % / 76 % | **throttled unevenly**, rerun |
| Llama-3.2-3B llama.cpp Q8_0 / F16 | 0 | 1,560 / 1,710 and 1,440 / 1,672 MHz | 86 °C | 0 % / 31 % | rerun with the 3B rows |
| speculative 3B ← 1B / 1B ← 1B | 2 | 1,500 / 1,545 and 1,545 / 1,612 MHz | 85 °C | 74 % / 46 % | rerun |

The reruns (`rerun/`, `scripts/rerun_throttled.sh`) went one GPU at a time. Each step started on a
card below 60 °C with no process on it, and every GPU's clocks were logged every 2 s
(`rerun/gpu_log.csv`, windows in `rerun/windows.txt`). The 2080 Tis still reach their ~84 °C
thermal target within a couple of minutes of sustained load. What the rerun fixes is that the runs
being compared now see the same clocks:

| rerun window | start | SM clock min / median | peak | thermal |
|---|---|---|---|---|
| Llama-3.2-3B int8 | 20 °C | 1,470 / 1,575 MHz | 75 °C | 0 % |
| Llama-3.2-3B fp16 | 55 °C | 1,440 / 1,545 MHz | 86 °C | 39 % |
| Llama-3.2-3B llama.cpp Q8_0 / F16 | 59–60 °C | 1,470–1,485 / 1,620 MHz | 86 °C | 28–29 % |
| speculative 3B ← 1B | 23 °C | 1,470 / 1,635 MHz | 85 °C | 44 % |
| speculative 1B ← 1B | 52 °C | 1,605 / 1,725 MHz | 84 °C | 3 % |

The rerun's numbers agree with the first run's to within 2 % (3B decode int8 6.71 vs 6.83 ms,
fp16 12.01 vs 12.16 ms; speculative speed-ups equal to three digits). The README uses the rerun
for 3B and speculative decoding and the first run for TinyLlama and Llama-3.2-1B.

The eviction benchmark ran later, alone on GPU 2 (`eviction/bench_gpu_log.csv`): median 1,770 MHz,
peak 83 °C, no thermal slowdown. Another user's job was on the A6000 at the time
(`eviction/bench_other_gpu_apps.txt`). It had no process on GPU 2.
