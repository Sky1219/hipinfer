# RTX A6000: thermal check of every benchmark

The A6000 in this machine overheats under sustained load: it holds its 300 W limit for about a minute, then
reaches 90–95 °C with the fan at 90–100 % and throttles its SM clock (normally 1,800–1,920 MHz) down to as
little as 210 MHz. Each benchmark below was started on an idle card below 60 °C with no other process on it,
and its window (start marker to result file) was checked against the GPU's temperature log (one sample
every 5 s; only samples at ≥ 50 % utilization count).

## hipinfer benchmarks (`<model>/bench.json`)

| model | window (CDT) | loaded samples | time thermally throttled | SM clock min / median | peak temp | verdict |
|---|---|---|---|---|---|---|
| TinyLlama-1.1B | 22:44:34-22:45:30 | 6 | 0 % | 1,560 / 1,890 MHz | 76 °C | clean |
| Llama-3.2-1B | 22:48:27-22:49:48 | 12 | 33 % | 1,485 / 1,605 MHz | 90 °C | **throttled** |
| Qwen2.5-0.5B | 22:53:53-22:54:45 | 6 | 0 % | 1,575 / 1,755 MHz | 72 °C | clean |
| Qwen2.5-1.5B | 22:57:32-22:59:11 | 13 | 31 % | 1,410 / 1,590 MHz | 91 °C | **throttled** |
| Llama-3.2-3B | 23:05:07-23:08:24 | 30 | 67 % | 420 / 1,410 MHz | 93 °C | **throttled** |
| Mistral-7B-v0.3 | 23:15:29-23:21:59 | 65 | 78 % | 210 / 450 MHz | 93 °C | **throttled** |
| Llama-3.1-8B | 00:43:22-00:51:49 | 86 | 84 % | 210 / 368 MHz | 93 °C | **throttled** |

## Engine baselines (llama.cpp, vLLM, and hipinfer re-run beside them)

A comparison is only fair if every engine in it ran unthrottled. Only TinyLlama's did; its files are in
`results/baselines/a6000/tinyllama/`. The other models' baseline runs are kept in
`results/a6000/throttled-baselines/` and are not used in any table.

| model | engine | time thermally throttled | SM clock min / median | peak temp | verdict |
|---|---|---|---|---|---|
| TinyLlama-1.1B | hipinfer | 0 % | 1,530 / 1,905 MHz | 75 °C | clean |
| TinyLlama-1.1B | llama.cpp | 0 % | 1,845 / 1,845 MHz | 75 °C | clean |
| TinyLlama-1.1B | vLLM | 0 % | 1,875 / 1,875 MHz | 80 °C | clean |
| Llama-3.2-1B | hipinfer | 0 % | 1,515 / 1,860 MHz | 82 °C | clean |
| Llama-3.2-1B | llama.cpp | 50 % | 1,710 / 1,792 MHz | 86 °C | **throttled** |
| Llama-3.2-1B | vLLM | 100 % | 1,800 / 1,800 MHz | 85 °C | **throttled** |
| Qwen2.5-1.5B | hipinfer | 0 % | 1,500 / 1,860 MHz | 86 °C | clean |
| Qwen2.5-1.5B | llama.cpp | 80 % | 1,560 / 1,590 MHz | 91 °C | **throttled** |
| Qwen2.5-1.5B | vLLM | 100 % | 1,575 / 1,620 MHz | 89 °C | **throttled** |
| Llama-3.2-3B | hipinfer | 65 % | 390 / 1,515 MHz | 93 °C | **throttled** |
| Llama-3.2-3B | llama.cpp | 100 % | 210 / 210 MHz | 95 °C | **throttled** |
| Llama-3.2-3B | vLLM | 80 % | 360 / 1,275 MHz | 93 °C | **throttled** |
| Llama-3.1-8B | hipinfer | 84 % | 210 / 360 MHz | 93 °C | **throttled** |
| Llama-3.1-8B | llama.cpp | 100 % | 210 / 210 MHz | 94 °C | **throttled** |
| Llama-3.1-8B | vLLM | 100 % | 330 / 450 MHz | 93 °C | **throttled** |

**What is affected:** only speed numbers. SUMMARY.md marks the cells of throttled hipinfer benchmarks with ⚠.
The token gate and perplexity results (`gate.json`, `ppl.json`) do not depend on clock speed and are valid for
every model.
