"""Decode attention (one new token per sequence) vs torch SDPA, at head_dim 64 (TinyLlama, GQA 32/4)
and 128 (Llama-3.2-3B, GQA 24/8). HIPINFER_DECODE_SCALAR=1 selects the portable scalar kernel."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import torch, torch.nn.functional as F
import hipinfer as hi
from bench import timeit
print(torch.cuda.get_device_name(), "kernel:", "scalar (portable)" if os.environ.get("HIPINFER_DECODE_SCALAR") else "mma.sync")
for HQ, HKV, D in [(32, 4, 64), (24, 8, 128)]:
    for B, L in [(1, 1024), (1, 4096), (8, 1024), (8, 4096), (32, 2048), (64, 1024)]:
        q = torch.randn(B, HQ * D, device="cuda", dtype=torch.float16)
        kc, vc = torch.randn(B, HKV, L, D, device="cuda", dtype=torch.float16), torch.randn(B, HKV, L, D, device="cuda", dtype=torch.float16)
        pos = torch.full((B,), L - 1, device="cuda", dtype=torch.int32)
        out = torch.empty(B, HQ * D, device="cuda", dtype=torch.float16)
        work = torch.zeros(hi.lib.hi_decode_work_bytes(B, HQ, HKV, D), device="cuda", dtype=torch.uint8)
        ours = timeit(lambda: hi.decode_attn(q, kc, vc, pos, HQ, HKV, out=out, work=work))
        qq = q.view(B, HQ, 1, D)
        ref = timeit(lambda: F.scaled_dot_product_attention(qq, kc, vc, enable_gqa=True))
        bw = lambda ms: 2 * B * HKV * L * D * 2 / ms / 1e6
        print(f"D={D:3d} B={B:3d} len={L:5d}  ours {ours*1e3:7.1f} us ({bw(ours):4.0f} GB/s)   SDPA {ref*1e3:7.1f} us ({bw(ref):4.0f} GB/s)   x{ref/ours:.2f}")
        del kc, vc
