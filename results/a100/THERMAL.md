# A100 (Google Colab): thermal check

NVIDIA A100-SXM4-40GB on a Colab runtime, 2026-10-17. `nvidia-smi` logged temperature, power, SM clock,
utilization and both thermal-slowdown flags every 5 s for the whole run (`gpu_log.csv`, 916 samples,
00:27 to 01:43 UTC), covering the hipinfer suite and the engine comparison.

| | |
|---|---|
| samples at >= 50 % utilization | 152 |
| peak temperature | 65 °C |
| SM clock under load, min / median / max | 1,230 / 1,410 / 1,410 MHz (1,410 MHz is the A100's boost clock) |
| samples with a thermal slowdown flag set | 0 |
| board power under load, median / max | 231 / 417 W |

Every benchmark on this GPU ran unthrottled. The run is one session, so the engine comparison
(`results/baselines/a100/`) and the hipinfer suite (`results/a100/`) were measured under the same
conditions.
