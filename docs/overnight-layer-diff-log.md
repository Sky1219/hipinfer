# Overnight log: layer-by-layer diff tool (2026-10-16)

Branch `claude/charming-ptolemy-5952ba`, committed locally and not pushed. This laptop had
CPU-only PyTorch 2.13, transformers 5.14.1 and no nvcc, so everything engine-side is unverified.
Usage and tomorrow's GPU steps are in [layer-diff.md](layer-diff.md).

## What exists now

| piece | state |
|---|---|
| `scripts/layer_diff.py`: HF capture (hooks), comparison, report, CLI, fault injection | tested on CPU |
| `tests/test_layer_diff.py`: 124 checks, under 1 min; opt-in real-model check | passing |
| `src/api.cu`: `hi_prefill_debug`, `hi_debug_bytes`, dump helpers | **not compiled** |
| `scripts/hipinfer.py`: `Engine.prefill_debug` | checked against a fake lib only |
| `docs/layer-diff.md` | usage and exact edmonds commands |

## Timeline

1. **HF capture and report** (commit `7c59ab6`). Forward hooks record 10 sub-steps per layer.
   RoPE has no module, so it is rebuilt from the hooked q/k and the `(cos, sin)` kwargs. A test
   confirms it is bit-identical to HF's `apply_rotary_pos_emb`. Fault injection uses prepended
   hooks that corrupt one sub-step's value (RoPE: negated sin). On the first cut, faults at 10% were
   all found, but 2% faults in `o_proj` / `down` / k-only `qkv` were missed.
2. **Metric rework** (commit `a759403`). TinyLlama on CPU showed why whole-tensor errors fail on
   real checkpoints: the BOS row carries ~100× activations. Layer 2's `gate_up` read 6e-5
   whole-tensor while its worst row was 2e-3. Also, the fp16 rounding of `h + o` in that row
   swamps the O / down outputs, so a fixed tolerance gave false positives from layer 3 on, and a
   per-point noise floor went blind (threshold ~1.0) to a 5% O fault. Fixes, chosen by comparing
   variants offline on saved TinyLlama captures:
   - error per (position, head) block;
   - `o_proj` / `down` judged on the projection output (point minus the previous residual), with
     the denominator floored at |residual| / 16;
   - a threshold per block from HF fp16's own error at that block (×8, floored by the point's
     median and 2e-3).

   Result on TinyLlama: every 5% single-step fault (`qkv`, `attn`, `o_proj`, `gate_up`, `down`
   at various depths) is named exactly with 5–77× margin. An independent fp16 implementation
   (SDPA attention) peaks at 0.57 of its thresholds. A fixed 5e-2 tolerance (no floor) leaves a
   clean fp16 run at 0.32 of it.
3. **Engine draft** (commit `0e12315`). In `src/api.cu`: an `Engine::dbg` pointer, a
   "debug dump" block above `attn_rows` (plane layout, `dbg_copy` / `dbg_step` / `dbg_rope`),
   one `dbg_step` line after each kernel in `attn_rows` / `mlp_rows`, two `dbg_copy` lines in
   `prefill`, and `hi_debug_bytes` / `hi_prefill_debug` after `hi_prefill_logits`.
   `hi_prefill_debug` zeroes the buffer, sets `dbg`, calls the normal `prefill` with
   `all_logits` pointed at the logits plane, then clears `dbg`. It is single-GPU only, with
   T <= max_prefill (one chunk). Decode and graph capture never see it. The binding is guarded
   with `hasattr`, so `hipinfer.py` still imports against an older `.so`.
4. **Docs, RoPE hint fix, fake-engine test.** The "position 0 clean" RoPE hint compared
   position 0 with the threshold. It now compares with the worst position, since position 0
   still carries fp16 noise. A stand-in `hipinfer` module now drives `layer_diff`'s engine path,
   so the flat-buffer split and the config check are tested. An unwritten row of one plane is
   named at the right position.

## Merging with the MoE branch

The engine-side diff is small, but it touches the same functions MoE will likely touch:

- `struct Engine`: two fields appended at the end.
- `attn_rows` / `mlp_rows`: one `dbg_step(...)` line after each kernel call. If MoE rewrites
  `mlp_rows`, keep `kDbgMlpNorm` after the norm and `kDbgDown` after the residual-adding
  output. `kDbgGateUp` (dense SwiGLU, `[T, inter]`) has no MoE equivalent. Skipping it there
  leaves that plane zero, which reads as a divergence at `gate_up`. MoE support needs its own
  points (router logits, per-expert outputs), and `layer_diff.capture_hf` needs matching hooks.
- `prefill`: two `dbg_copy` lines (after `embed_kernel`, after the final norm).
- New code otherwise lives in its own block (before `attn_rows`) and after `hi_prefill_logits`.

## Not done / open

- Compiling and running the engine side: steps 1–5 in [layer-diff.md](layer-diff.md#first-run-on-the-gpu).
- Tensor-parallel dumps: rank 0's rows before the all-reduce are partial sums, so this would
  need a dump after each all-reduce.
- Decode-path dumps. The prefill path covers the same weights, but decode uses different
  kernels (GEMV, fused QKV + RoPE, split-K attention). A decode variant would need to run
  outside graph capture.
- Families with other module names (Gemma, Phi) need hooks in `capture_hf`.
