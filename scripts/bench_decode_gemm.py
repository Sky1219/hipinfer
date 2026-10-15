"""Batched-decode projection shapes: autotuned split-K GEMM vs cuBLAS. GB/s = weight bytes / time."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch
import hipinfer as hi
from bench import timeit
H, I, V = 2048, 5632, 32000
r16 = lambda *s: torch.randn(*s, device="cuda", dtype=torch.float16)
print(torch.cuda.get_device_name())
for M in [16, 32, 64]:
    tot = {"gemm_autotuned": 0, "cublas": 0}
    for name, N, K, epi, n in [("qkv", 2560, H, 0, 22), ("o", H, H, 1, 22), ("gate_up", 2 * I, H, 2, 22),
                               ("down", H, I, 1, 22), ("lm_head", V, H, 0, 1)]:
        x, W = r16(64, K), r16(N, K) * 0.02
        xm = x[:M]
        out = r16(M, N // 2 if epi == 2 else N)
        t = {}
        hi.gemm(xm, W, out=out, epi=epi)  # autotune outside graph capture
        t["gemm_autotuned"] = timeit(lambda: hi.gemm(xm, W, out=out, epi=epi))
        o2 = r16(M, N)
        t["cublas"] = timeit(lambda: torch.matmul(xm, W.t(), out=o2))
        for k in t:
            if t[k] == t[k]:
                tot[k] += t[k] * n
        gb = lambda ms: N * K * 2 / ms / 1e6
        print(f"M={M:2d} {name:8s} " + "  ".join(f"{k} {t[k]*1e3:6.1f}us {gb(t[k]):4.0f}GB/s" for k in t))
    print(f"M={M:2d} per-step projections: " + "  ".join(f"{k} {v:.2f} ms" for k, v in tot.items()))
