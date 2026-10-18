"""Benchmarks for one model on one GPU: each hipinfer kernel vs its PyTorch counterpart at the
model's shapes, then the end-to-end engine vs HF transformers (prefill, batch-1 decode, decode
latency vs context length, batched throughput with power draw). All timings are CUDA-event
medians; per-step decode numbers come from graph-replayed engine steps.

python scripts/bench.py --model unsloth/Llama-3.2-3B --weights llama32-3b.bin --out results/a6000/llama32-3b/bench.json

With int8 weights (version 4) the kernel section adds the int8 GEMV and GEMM next to the fp16 ones
at the same shapes, and every bandwidth floor counts the int8 bytes (hi_weight_bytes).
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import hipinfer as hi  # noqa: E402

dev = "cuda"


def timeit(fn, iters=20, warmup=3, inner=10):
    """Median GPU time per call. `inner` calls are captured into one CUDA graph and replayed,
    so host/Python launch overhead is excluded for both hipinfer and PyTorch."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / inner)
    return statistics.median(ts)


def timeit_eager(fn, iters=10, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


class Power:
    """Mean board power (W) over an interval, sampled through NVML (optional)."""

    def __init__(self):
        self.h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            self.nv = pynvml
            self.h = pynvml.nvmlDeviceGetHandleByUUID(("GPU-" + uuid).encode())
        except Exception:
            self.h = None

    def measure(self, fn):
        """Run fn(); return (its return value, mean W or None)."""
        if self.h is None:
            return fn(), None
        samples, stop = [], threading.Event()

        def poll():
            while not stop.is_set():
                samples.append(self.nv.nvmlDeviceGetPowerUsage(self.h) / 1000.0)
                time.sleep(0.02)
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        r = fn()
        stop.set()
        th.join()
        return r, (statistics.mean(samples) if samples else None)


def row(rows, kernel, role, M, shape, ours, ref, ref_name, unit_fn=None, unit=""):
    r = {"kernel": kernel, "role": role, "M": M, "shape": shape, "ours_us": ours * 1e3, "torch_us": ref * 1e3,
         "baseline": ref_name, "speedup": ref / ours}
    if unit_fn:
        r[unit] = unit_fn(ours)
        r[unit + "_torch"] = unit_fn(ref)
    rows.append(r)
    extra = f"  {unit}: {r[unit]:7.1f} vs {r[unit + '_torch']:7.1f}" if unit_fn else ""
    print(f"{kernel:13s} {shape:34s} ours {ours * 1e3:9.1f} us  {ref_name:14s} {ref * 1e3:9.1f} us  x{ref / ours:5.2f}{extra}")


def hf_rmsnorm(x, w, eps=1e-5):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def bench_kernels(rows, cfg):
    H, I, HQ, HKV, D, V = cfg["hidden"], cfg["inter"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["vocab"]
    QKV = (HQ + 2 * HKV) * D
    r16 = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16)
    print("\n== GEMM (prefill, tensor cores) ==")
    for M in [128, 512, 2048]:
        for name, N, K in [("qkv", QKV, H), ("o", H, HQ * D), ("gate_up", 2 * I, H), ("down", H, I), ("lm_head", V, H)]:
            A, W = r16(M, K), r16(N, K) * 0.02
            out = torch.empty(M, N, device=dev, dtype=torch.float16)
            ours = timeit(lambda: hi.gemm(A, W, out=out))
            ref = timeit(lambda: torch.matmul(A, W.t(), out=out))
            fl = lambda ms: 2 * M * N * K / ms / 1e9
            row(rows, "gemm", name, M, f"{name} M={M} N={N} K={K}", ours, ref, "cuBLAS", fl, "TFLOPs")
            del A, W, out

    print("\n== GEMV (decode, weight-bandwidth bound) ==")
    for M in [1, 4, 8]:
        for name, N, K, epi, nrm in [("qkv+norm", QKV, H, 0, True), ("o+res", H, HQ * D, 1, False),
                                     ("gate_up+norm+silu", 2 * I, H, 2, True), ("down+res", H, I, 1, False),
                                     ("lm_head+norm", V, H, 0, True)]:
            x, W = r16(M, K), r16(N, K) * 0.02
            nw = (torch.rand(K, device=dev) + 0.5).half()
            out = r16(M, N // 2 if epi == 2 else N)
            ours = timeit(lambda: hi.gemv(x, W, out=out, normw=nw if nrm else None, epi=epi))
            if epi == 2:
                g, u = W[:I], W[I:]
                ref_fn = lambda: F.silu(F.linear(hf_rmsnorm(x, nw), g)) * F.linear(hf_rmsnorm(x, nw), u)
            elif epi == 1:
                ref_fn = lambda: out.add_(F.linear(x, W))
            else:
                ref_fn = (lambda: F.linear(hf_rmsnorm(x, nw), W)) if nrm else (lambda: F.linear(x, W))
            ref = timeit(ref_fn)
            bw = lambda ms: N * K * 2 / ms / 1e6
            row(rows, "gemv", name, M, f"{name} M={M} N={N} K={K}", ours, ref, "torch(cuBLAS)", bw, "GB/s")

    if cfg.get("quant"):
        bench_int8_kernels(rows, cfg)

    print("\n== RMSNorm ==")
    for M in [1, 512, 4096]:
        x, w = r16(M, H), r16(H)
        ours = timeit(lambda: hi.rmsnorm(x, w))
        ref = timeit(lambda: hf_rmsnorm(x, w))
        ref2 = timeit(lambda: F.rms_norm(x, (H,), w, 1e-5))
        bw = lambda ms: 2 * M * H * 2 / ms / 1e6
        row(rows, "rmsnorm", "rmsnorm", M, f"M={M} H={H}", ours, ref, "HF eager", bw, "GB/s")
        row(rows, "rmsnorm", "rmsnorm", M, f"M={M} H={H}", ours, ref2, "F.rms_norm", bw, "GB/s")

    print("\n== RoPE (+ KV-cache write) ==")
    S = 4096
    inv = 1.0 / (10000 ** (torch.arange(0, D, 2).float() / D))
    ang = torch.arange(S).float()[:, None] * inv[None]
    cs = torch.stack([ang.cos(), ang.sin()], -1).half().to(dev)
    cos = torch.cat([cs[..., 0], cs[..., 0]], -1)
    sin = torch.cat([cs[..., 1], cs[..., 1]], -1)
    for T in [1, 512, 2048]:
        qkv = r16(T, QKV)
        kc = torch.zeros(1, HKV, S, D, device=dev, dtype=torch.float16)
        vc = torch.zeros_like(kc)
        ours = timeit(lambda: hi.rope_cache(qkv, HQ, HKV, cs, kc, vc, pos_base=0))
        pp = torch.arange(T, device=dev)

        def ref_fn():
            q = qkv[:, :HQ * D].view(T, HQ, D).transpose(0, 1)
            k = qkv[:, HQ * D:(HQ + HKV) * D].view(T, HKV, D).transpose(0, 1)
            v = qkv[:, (HQ + HKV) * D:].view(T, HKV, D).transpose(0, 1)
            c, s = cos[pp][None], sin[pp][None]
            rot = lambda t: torch.cat([-t[..., D // 2:], t[..., :D // 2]], -1)
            q2 = q * c + rot(q) * s
            k2 = k * c + rot(k) * s
            kc[0, :, :T] = k2
            vc[0, :, :T] = v
            return q2
        ref = timeit(ref_fn)
        row(rows, "rope", "rope", T, f"T={T}", ours, ref, "HF eager")

    print("\n== Attention: causal prefill (FlashAttention-style, tensor cores) ==")
    for T in [512, 1024, 2048]:
        q = r16(T, HQ * D)
        kc, vc = r16(HKV, T, D), r16(HKV, T, D)
        out = torch.empty(T, HQ * D, device=dev, dtype=torch.float16)
        ours = timeit(lambda: hi.flash_prefill(q, kc, vc, T, 0, HQ, HKV, out=out))
        qq = q.view(T, HQ, D).transpose(0, 1)[None]
        ref = timeit(lambda: F.scaled_dot_product_attention(qq, kc[None], vc[None], is_causal=True, enable_gqa=True))
        fl = lambda ms: 2 * 2 * HQ * D * T * T / 2 / ms / 1e9
        row(rows, "flash_prefill", "flash_prefill", T, f"T={T} Hq={HQ} Hkv={HKV} D={D}", ours, ref, "torch SDPA", fl, "TFLOPs")

    print("\n== Attention: decode (split-K, GQA-shared K/V reads) ==")
    for B, L in [(1, 1024), (1, 4096), (8, 1024), (8, 4096), (32, 2048)]:
        q = r16(B, HQ * D)
        kc, vc = r16(B, HKV, L, D), r16(B, HKV, L, D)
        pos = torch.full((B,), L - 1, device=dev, dtype=torch.int32)
        out = torch.empty(B, HQ * D, device=dev, dtype=torch.float16)
        work = torch.zeros(hi.lib.hi_decode_work_bytes(B, HQ, HKV, D), device=dev, dtype=torch.uint8)
        ours = timeit(lambda: hi.decode_attn(q, kc, vc, pos, HQ, HKV, out=out, work=work))
        qq = q.view(B, HQ, 1, D)
        ref = timeit(lambda: F.scaled_dot_product_attention(qq, kc, vc, enable_gqa=True))
        bw = lambda ms: 2 * B * HKV * L * D * 2 / ms / 1e6
        row(rows, "decode_attn", f"B={B}", L, f"B={B} len={L} Hq={HQ} Hkv={HKV} D={D}", ours, ref, "torch SDPA", bw, "GB/s")


def bench_int8_kernels(rows, cfg):
    """int8-weight GEMV (decode) and GEMM (prefill) against hipinfer's own fp16 kernels at the same
    shapes; GB/s counts the int8 weights plus their fp16 row scales."""
    from export import quantize_rows
    H, I, HQ, HKV, D, V = cfg["hidden"], cfg["inter"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["vocab"]
    QKV = (HQ + 2 * HKV) * D
    r16 = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16)
    print("\n== GEMV int8 weights (decode; baseline: hipinfer's fp16 GEMV) ==")
    for M in [1, 4]:
        for name, N, K, epi, nrm in [("qkv+norm", QKV, H, 0, True), ("o+res", H, HQ * D, 1, False),
                                     ("gate_up+norm+silu", 2 * I, H, 2, True), ("down+res", H, I, 1, False),
                                     ("lm_head+norm", V, H, 0, True)]:
            x, W = r16(M, K), r16(N, K) * 0.02
            q, s = quantize_rows(W)
            nw = (torch.rand(K, device=dev) + 0.5).half()
            out = r16(M, N // 2 if epi == 2 else N)
            ours = timeit(lambda: hi.gemv(x, q, out=out, normw=nw if nrm else None, epi=epi, scales=s))
            ref = timeit(lambda: hi.gemv(x, W, out=out, normw=nw if nrm else None, epi=epi))
            bw = lambda ms: (N * K + 2 * N) / ms / 1e6
            row(rows, "gemv_int8", name, M, f"{name} M={M} N={N} K={K}", ours, ref, "hipinfer fp16", bw, "GB/s")
            del x, W, q, s, out
    print("\n== GEMM int8 weights (prefill, tiles dequantized to fp16; baseline: hipinfer's fp16 GEMM) ==")
    for M in [128, 512, 2048]:
        for name, N, K in [("qkv", QKV, H), ("o", H, HQ * D), ("gate_up", 2 * I, H), ("down", H, I), ("lm_head", V, H)]:
            A, W = r16(M, K), r16(N, K) * 0.02
            q, s = quantize_rows(W)
            out = torch.empty(M, N, device=dev, dtype=torch.float16)
            ours = timeit(lambda: hi.gemm(A, q, out=out, scales=s))
            ref = timeit(lambda: hi.gemm(A, W, out=out))
            fl = lambda ms: 2 * M * N * K / ms / 1e9
            row(rows, "gemm_int8", name, M, f"{name} M={M} N={N} K={K}", ours, ref, "hipinfer fp16", fl, "TFLOPs")
            del A, W, q, s, out


def gemm_sweep(rows, cfg):
    """Time every tile config on every prefill shape (to tune gemm_pick)."""
    H, I, HQ, HKV, D, V = cfg["hidden"], cfg["inter"], cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["vocab"]
    r16 = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16)
    n = hi.lib.hi_gemm_num_configs()
    print("\n== GEMM tile-config sweep (TFLOPs; cfg -1 = heuristic) ==")
    for M in [16, 32, 64, 128, 512, 2048]:
        for name, N, K in [("qkv", (HQ + 2 * HKV) * D, H), ("o", H, HQ * D), ("gate_up", 2 * I, H), ("down", H, I), ("lm_head", V, H)]:
            A, W = r16(M, K), r16(N, K) * 0.02
            out = torch.empty(M, N, device=dev, dtype=torch.float16)
            res = {}
            for c in [-1] + list(range(n)):
                hi.lib.hi_gemm_force_config(c)
                res[c] = 2 * M * N * K / timeit(lambda: hi.gemm(A, W, out=out), iters=10) / 1e9
            hi.lib.hi_gemm_force_config(-1)
            cub = 2 * M * N * K / timeit(lambda: torch.matmul(A, W.t(), out=out), iters=10) / 1e9
            best = max(range(n), key=lambda c: res[c])
            rows.append({"M": M, "shape": name, "tflops": res, "cublas": cub, "best": best})
            print(f"M={M:5d} {name:8s} " + " ".join(f"{c}:{res[c]:5.1f}" for c in res) + f"  cuBLAS {cub:5.1f}  best={best}")


def free_mem():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return torch.cuda.mem_get_info()[0]


# datasheet DRAM bandwidth (GB/s) by device-name substring
KNOWN_BW = {"2080 Ti": 616.0, "A6000": 768.0, "T4": 320.0, "L4": 300.0, "A10": 600.0, "RTX PRO 6000": 1792.0,
            "V100": 900.0, "MI300X": 5300.0, "MI325X": 6000.0, "MI250": 1638.0, "MI210": 1638.0, "7900 XTX": 960.0,
            "W7900": 864.0, "RTX 4090": 1008.0, "RTX 3090": 936.0}


def peak_bw(device=0):
    name = torch.cuda.get_device_name(device)
    mem_gb = torch.cuda.get_device_properties(device).total_memory / 2**30
    if "A100" in name:
        return 2039.0 if mem_gb > 50 else 1555.0
    if "H100" in name:
        return 3350.0 if "SXM" in name or "HBM3" in name else 2000.0
    return next((v for k, v in KNOWN_BW.items() if k in name), None)


def engine_bytes(cfg, weight_bytes, B, max_seq, max_prefill, tp=1, slack=1.5e9):
    """Bytes one GPU needs for an engine: its weight share, its KV share, activations, logits and
    attention work (cfg as from hipinfer.read_config; heads, kv heads, inter and vocab split tp ways)."""
    H, D = cfg["hidden"], cfg["head_dim"]
    I, HQ, HKV, V = cfg["inter"] // tp, cfg["heads"] // tp, cfg["kv_heads"] // tp, -(-cfg["vocab"] // tp)
    Mmax = (max(max_prefill, B) + 127) // 128 * 128
    return weight_bytes + B * max_seq * 2 * cfg["layers"] * HKV * D * 2 \
        + Mmax * (2 * H + (HQ + 2 * HKV) * D + HQ * D + I) * 2 + B * V * 2 + B * HQ * 64 * (D + 2) * 4 + slack


def plan_engine(cfg, weight_bytes, max_seq, max_prefill, want_batch):
    """Largest batch (power of two <= want_batch) whose KV cache + activations fit next to the weights."""
    free = free_mem()
    B = want_batch
    while B >= 1:
        if engine_bytes(cfg, weight_bytes, B, max_seq, max_prefill) <= free:
            return B
        B //= 2
    return 0


def bench_e2e(res, weights, model_name, cfg, max_batch, max_ctx, peak_bw, no_hf=False):
    from transformers import AutoModelForCausalLM

    H, HKV, D, V = cfg["hidden"], cfg["kv_heads"], cfg["head_dim"], cfg["vocab"]
    kv_tok = 2 * cfg["layers"] * HKV * D * 2
    weight_bytes = os.path.getsize(weights)
    g = torch.Generator().manual_seed(0)
    rand_ids = lambda n: torch.randint(100, min(30000, V - 1), (n,), generator=g).tolist()
    e2e = res["e2e"] = {"weight_bytes": weight_bytes, "kv_bytes_per_token": kv_tok}
    max_pos = hi.max_context(cfg)  # max_pos, or the sliding window (Mistral-7B-v0.1: 4096)
    prefill_Ts = [T for T in [128, 512, 1024, 2000, 4096] if T <= min(max_pos, max_ctx)]
    ctxs = [c for c in [128, 512, 1024, 2048, 4096, 8192, 16384] if c + 256 <= min(max_pos, max_ctx)]

    # ---- phase 1: HF transformers (eager, SDPA attention) for the comparison columns
    hf_prefill, hf_decode = {}, {}
    if no_hf:
        print("\n== HF reference skipped (--no-hf) ==")
        model = None
        e2e["hf"] = False
    else:
        print("\n== HF transformers reference timings ==")
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16).cuda().eval()
    for T in (prefill_Ts if model is not None else []):
        t = torch.tensor([rand_ids(T)], device=dev)
        with torch.no_grad():
            hf_prefill[T] = timeit_eager(lambda: model(t, use_cache=False, logits_to_keep=1))
        print(f"HF prefill T={T:5d}: {hf_prefill[T]:8.2f} ms")
        del t
    for ctx in ([128, 1024] if model is not None else []):
        if ctx + 256 > max_pos:
            continue
        t = torch.tensor([rand_ids(ctx)], device=dev)

        def hf_loop(steps=64):
            with torch.no_grad():
                o = model(t, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(steps):
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / steps * 1e3
        hf_decode[ctx] = min(hf_loop() for _ in range(3))
        print(f"HF decode ctx={ctx:5d}: {hf_decode[ctx]:6.2f} ms/token")
        del t
    del model
    import gc
    gc.collect()

    # ---- phase 2: hipinfer, latency engine (batch 1, long context)
    pf_max = max(prefill_Ts)
    seq = max(max(ctxs) + 256 if ctxs else 1024, pf_max)  # the cache holds the longest context and prefill
    if plan_engine(cfg, weight_bytes, seq, pf_max, 1) < 1:
        print("not enough GPU memory for the latency engine; skipping e2e")
        return
    eng = hi.Engine(weights, max_batch=1, max_seq=seq, max_prefill=pf_max)
    power = Power()

    print("\n== Prefill latency (ms) ==")
    e2e["prefill"] = []
    for T in prefill_Ts:
        ids = rand_ids(T)
        ours = timeit_eager(lambda: eng.prefill(0, ids))
        ref = hf_prefill.get(T)
        e2e["prefill"].append({"T": T, "ours_ms": ours, "hf_ms": ref, "ours_tok_s": T / ours * 1e3})
        hf_s = f"   HF {ref:7.2f} ms   x{ref / ours:.2f}" if ref else ""
        print(f"T={T:5d}  hipinfer {ours:7.2f} ms ({T / ours * 1e3:8.0f} tok/s){hf_s}")

    print("\n== Decode, batch 1 (ms/token) ==")
    e2e["decode_b1"] = []
    for ctx in [128, 1024]:
        if ctx + 256 > max_pos:
            continue
        ids = rand_ids(ctx)
        n = 256
        res_g, res_ng = [], []
        for _ in range(3):
            eng.prefill(0, ids); res_g.append(eng.decode(1, n, True) / n)
            eng.prefill(0, ids); res_ng.append(eng.decode(1, n, False) / n)
        ours, ours_ng = min(res_g), min(res_ng)
        hf = hf_decode.get(ctx)
        kv_bytes = kv_tok * (ctx + n // 2)
        floor = (eng.weight_bytes + kv_bytes) / peak_bw / 1e6
        e2e["decode_b1"].append({"ctx": ctx, "ours_ms": ours, "ours_nograph_ms": ours_ng, "hf_ms": hf,
                                 "bw_floor_ms": floor, "bw_util": floor / ours})
        hf_s = f"   HF {hf:6.2f} ms   x{hf / ours:.1f}" if hf else ""
        print(f"ctx={ctx:5d}  hipinfer {ours:6.3f} ms (graph) / {ours_ng:6.3f} ms (no graph){hf_s}"
              f"   roofline {floor:.3f} ms -> {floor / ours * 100:.0f}% of peak BW")

    print("\n== Decode latency vs context length (batch 1, 128 new tokens) ==")
    e2e["decode_ctx"] = []
    for ctx in ctxs:
        ids = rand_ids(ctx)
        n = 128
        best = 1e9
        for _ in range(2):
            eng.prefill(0, ids)
            best = min(best, eng.decode(1, n, True) / n)
        kv_bytes = kv_tok * (ctx + n // 2)
        floor = (eng.weight_bytes + kv_bytes) / peak_bw / 1e6
        e2e["decode_ctx"].append({"ctx": ctx, "ours_ms": best, "bw_floor_ms": floor, "bw_util": floor / best,
                                  "kv_bytes": kv_bytes})
        print(f"ctx={ctx:6d}  {best:6.3f} ms/token   KV read {kv_bytes / 1e6:7.1f} MB/step   roofline {floor:.3f} ms  "
              f"{floor / best * 100:.0f}% BW")
    eng.close()

    # ---- phase 3: hipinfer, throughput engine (as many sequences as fit, ctx 512 + 128 new)
    seq = 1024
    B_max = plan_engine(cfg, weight_bytes, seq, 512, max_batch)
    print(f"\n== Batched decode throughput (ctx 512, 128 new tokens; up to {B_max} sequences fit) ==")
    e2e["decode_batched"] = []
    if B_max >= 1:
        eng = hi.Engine(weights, max_batch=B_max, max_seq=seq, max_prefill=512)
        ids = rand_ids(512)
        for B in [1, 2, 4, 8, 16, 32, 64, 128]:
            if B > B_max:
                break
            for b in range(B):
                eng.prefill(b, ids)
            eng.decode(B, 2)  # capture/warm
            for b in range(B):
                eng.prefill(b, ids)
            steps = 128
            ms, watts = power.measure(lambda: eng.decode(B, steps))
            tps = B * steps / ms * 1e3
            r = {"B": B, "ms_per_step": ms / steps, "tok_s": tps}
            extra = ""
            if watts:
                r["watts"] = watts
                r["mJ_per_token"] = watts / tps * 1e3
                extra = f"   {watts:5.0f} W   {watts / tps * 1e3:6.2f} mJ/token"
            e2e["decode_batched"].append(r)
            print(f"B={B:3d}  {ms / steps:6.3f} ms/step   {tps:8.0f} tok/s{extra}")
        eng.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="tinyllama.bin")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--peak-bw", type=float, default=None, help="GB/s; default from known GPUs")
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--max-ctx", type=int, default=16384)
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--skip-kernels", action="store_true")
    ap.add_argument("--gemm-sweep", action="store_true")
    ap.add_argument("--no-hf", action="store_true", help="skip the HF transformers reference (model too big for two copies)")
    a = ap.parse_args()
    name = torch.cuda.get_device_name()
    peak = a.peak_bw or peak_bw()
    # model config from the weight file (a tiny engine)
    e = hi.Engine(a.weights, max_batch=1, max_seq=64, max_prefill=64)
    cfg = {"vocab": e.vocab, "hidden": e.hidden, "inter": e.inter, "layers": e.layers, "heads": e.heads,
           "kv_heads": e.kv_heads, "head_dim": e.head_dim, "max_pos": e.max_pos, "window": e.window, "tied": e.tied,
           "qkv_bias": e.qkv_bias, "weight_bytes": e.weight_bytes, "quant": e.quant}
    e.close()
    res = {"gpu": name, "peak_bw_GBs": peak, "torch": torch.__version__, "model": a.model, "config": cfg, "kernels": []}
    print(name, "peak BW", peak, "GB/s |", a.model, cfg)
    if a.gemm_sweep:
        res["gemm_sweep"] = []
        gemm_sweep(res["gemm_sweep"], cfg)
    if not a.skip_kernels:
        bench_kernels(res["kernels"], cfg)
    if not a.skip_e2e:
        bench_e2e(res, a.weights, a.model, cfg, a.max_batch, a.max_ctx, peak, a.no_hf)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
