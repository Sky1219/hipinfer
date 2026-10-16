# RTX A6000 run of 2026-10-15: thermal check of every benchmark

The A6000 in this machine overheats under sustained load: it holds its 300 W limit for about a minute, then
reaches 90–93 °C with the fan at 90–100 % and throttles its SM clock (normally 1,800–1,920 MHz) down to as
little as 210 MHz. Each benchmark below was started on an idle card below 60 °C with no other process on it,
and its window (start marker to result file) was checked against the GPU's temperature log (one sample
every 5 s; only samples at ≥ 50 % utilization count).

| model | window (CDT) | loaded samples | time thermally throttled | SM clock min / median | peak temp | verdict |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 22:44:34-22:45:30 | 6 | 0 % | 1,560 / 1,890 MHz | 76 °C | clean |
| Llama-3.2-1B | 22:48:27-22:49:48 | 12 | 33 % | 1,485 / 1,605 MHz | 90 °C | **throttled** |
| Qwen2.5-0.5B | 22:53:53-22:54:45 | 6 | 0 % | 1,575 / 1,755 MHz | 72 °C | clean |
| Qwen2.5-1.5B | 22:57:32-22:59:11 | 13 | 31 % | 1,410 / 1,590 MHz | 91 °C | **throttled** |
| Llama-3.2-3B | 23:05:07-23:08:24 | 30 | 67 % | 420 / 1,410 MHz | 93 °C | **throttled** |
| Mistral-7B-v0.3 | 23:15:29-23:21:59 | 65 | 78 % | 210 / 450 MHz | 93 °C | **throttled** |

**What is affected:** only the speed numbers in `bench.json` of the throttled models (decode, prefill,
batched throughput, power/energy and kernel timings). SUMMARY.md marks those cells with ⚠. The token gate and
perplexity results (`gate.json`, `ppl.json`) do not depend on clock speed and are valid for every model.

Llama-3.1-8B and the llama.cpp / vLLM baselines were not run on this date; Llama-3.1-8B's `gate.json` and
`ppl.json` here are from an earlier build (2026-10-14).
