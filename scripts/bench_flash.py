import os, sys
sys.path.insert(0, "scripts")
import torch, torch.nn.functional as F
import hipinfer as hi
from bench import timeit
HQ, HKV, D = 32, 4, 64
print(torch.cuda.get_device_name(), "kernel:", "WMMA (portable)" if os.environ.get("HIPINFER_FLASH_WMMA") else "mma.sync (register-resident)")
for T in [512, 1024, 2048, 4096]:
    q = torch.randn(T, HQ * D, device="cuda", dtype=torch.float16)
    kc, vc = torch.randn(HKV, T, D, device="cuda", dtype=torch.float16), torch.randn(HKV, T, D, device="cuda", dtype=torch.float16)
    out = torch.empty_like(q)
    ours = timeit(lambda: hi.flash_prefill(q, kc, vc, T, 0, HQ, HKV, out=out))
    qq = q.view(T, HQ, D).transpose(0, 1)[None]
    ref = timeit(lambda: F.scaled_dot_product_attention(qq, kc[None], vc[None], is_causal=True, enable_gqa=True))
    fl = lambda ms: 2 * 2 * HQ * D * T * T / 2 / ms / 1e9
    print(f"T={T:5d}  ours {ours*1e3:8.1f} us ({fl(ours):5.1f} TF)   SDPA {ref*1e3:8.1f} us ({fl(ref):5.1f} TF)   x{ref/ours:.2f}")
