"""Kernel correctness vs PyTorch references (fp32 math, or HF's exact fp16 op order where
the kernel claims to reproduce it). Covers head_dim 64 and 128, every GQA group size 1..8,
the QKV bias path, every GEMM tile config and split-K factor, argmax on NaN / -inf rows, and a
captured split-K GEMM replayed after its workspace was outgrown. Run: python tests/test_kernels.py"""
import ctypes as C
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import hipinfer as hi  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
FAILS = []


def check(name, got, ref, atol, rtol=0.0):
    err = (got.float() - ref.float()).abs()
    tol = atol + rtol * ref.float().abs()
    bad = (err > tol).sum().item()
    print(f"{'OK  ' if bad == 0 else 'FAIL'} {name:52s} max_abs_err={err.max().item():.3e}")
    if bad:
        FAILS.append(name)


def hf_rmsnorm(x, w, eps):
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def interleave(g, u):
    I, H = g.shape
    return torch.stack([g.view(I // 16, 16, H), u.view(I // 16, 16, H)], 1).reshape(2 * I, H)


def test_rmsnorm():
    for M, H in [(1, 2048), (7, 2048), (300, 4096), (5, 896), (33, 3072)]:
        x = torch.randn(M, H, device=dev, dtype=torch.float16) * 3
        w = torch.randn(H, device=dev, dtype=torch.float16)
        check(f"rmsnorm M={M} H={H} (vs HF fp16 order)", hi.rmsnorm(x, w), hf_rmsnorm(x, w, 1e-5), 2e-3, 2e-3)


def test_gemv():
    for H, I in [(2048, 5632), (896, 4864), (3072, 8192)]:
        for M in [1, 3, 8, 11]:
            x = torch.randn(M, H, device=dev, dtype=torch.float16)
            W = torch.randn(2560, H, device=dev, dtype=torch.float16) * 0.02
            check(f"gemv M={M} N=2560 K={H}", hi.gemv(x, W), x.float() @ W.float().T, 2e-3, 2e-3)
            nw = torch.rand(H, device=dev, dtype=torch.float16) + 0.5
            ref = hf_rmsnorm(x, nw, 1e-5).float() @ W.float().T
            check(f"gemv+rmsnorm M={M} K={H}", hi.gemv(x, W, normw=nw), ref, 3e-3, 3e-3)
            res = torch.randn(M, H, device=dev, dtype=torch.float16)
            W2 = torch.randn(H, I, device=dev, dtype=torch.float16) * 0.02
            a = torch.randn(M, I, device=dev, dtype=torch.float16)
            out = res.clone()
            hi.gemv(a, W2, out=out, epi=1)
            check(f"gemv+residual M={M} K={I}", out, res + (a.float() @ W2.float().T).half(), 4e-3, 2e-3)
            g = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
            u = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
            ref = F.silu((x.float() @ g.float().T).half()) * (x.float() @ u.float().T).half()
            check(f"gemv+swiglu M={M} I={I}", hi.gemv(x, interleave(g, u), normw=None, epi=2), ref, 3e-3, 3e-3)


def test_gemm_splitk():
    # batched-decode shapes through split-K (forced), every epilogue
    H, I = 2048, 5632
    for sp in [2, 4, 8]:
        hi.lib.hi_gemm_force_splits(sp)
        for M in [9, 32, 64]:
            x = torch.randn(M, H, device=dev, dtype=torch.float16)
            W = torch.randn(2560, H, device=dev, dtype=torch.float16) * 0.02
            check(f"gemm split-K={sp} M={M} N=2560", hi.gemm(x, W), x.float() @ W.float().T, 3e-3, 3e-3)
            b = torch.randn(2560, device=dev, dtype=torch.float16)
            check(f"gemm split-K={sp} +bias M={M}", hi.gemm(x, W, bias=b), x.float() @ W.float().T + b.float(), 3e-3, 3e-3)
            res = torch.randn(M, H, device=dev, dtype=torch.float16)
            a = torch.randn(M, I, device=dev, dtype=torch.float16)
            W2 = torch.randn(H, I, device=dev, dtype=torch.float16) * 0.02
            out = res.clone()
            hi.gemm(a, W2, out=out, epi=1)
            check(f"gemm split-K={sp} +residual M={M} K=5632", out, res + (a.float() @ W2.float().T).half(), 4e-3, 2e-3)
            g = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
            u = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
            ref = F.silu((x.float() @ g.float().T).half()) * (x.float() @ u.float().T).half()
            check(f"gemm split-K={sp} +swiglu M={M}", hi.gemm(x, interleave(g, u), epi=2), ref, 4e-3, 4e-3)
    hi.lib.hi_gemm_force_splits(-1)
    # One and two K tiles per split (K % 32 == 0 is the API's floor); a split
    # that does not divide the K tiles runs unsplit (hi_gemm's split_valid guard). The heuristic
    # pick may be a BK=64 tile, so BK=32 configs are also forced (6, and 26 = 256x128 2x4) to get
    # KT = 1 (K = 32 * sp) and KT = 2 (K = 64 * sp) per split; a config that cannot run here
    # (e.g. absent on AMD) falls back to a valid one in hi_gemm (cfg_valid guard)
    for cfg in [-1, 6, 26]:
        hi.lib.hi_gemm_force_config(cfg)
        for sp in [1, 2]:
            hi.lib.hi_gemm_force_splits(sp)
            for M, K in [(300, 32 * sp), (300, 64 * sp), (129, 96)]:
                x = torch.randn(M, K, device=dev, dtype=torch.float16)
                W = torch.randn(1024, K, device=dev, dtype=torch.float16) * 0.02
                check(f"gemm short-K cfg={cfg} split={sp} M={M} K={K}", hi.gemm(x, W), x.float() @ W.float().T, 3e-3, 3e-3)
    hi.lib.hi_gemm_force_splits(-1)
    hi.lib.hi_gemm_force_config(-1)
    # the weight-streaming skinny kernels (NVIDIA, configs with M <= 64), with and without split-K
    for cfg in range(hi.lib.hi_gemm_num_configs()):
        desc = C.create_string_buffer(64)
        hi.lib.hi_gemm_config_desc(cfg, desc, 64)
        if b"skinny" not in desc.value:
            continue
        hi.lib.hi_gemm_force_config(cfg)
        for sp in [1, 2, 4]:
            hi.lib.hi_gemm_force_splits(sp)
            for M in [1, 9, 16, 33, 64]:
                x = torch.randn(M, H, device=dev, dtype=torch.float16)
                W = torch.randn(2560, H, device=dev, dtype=torch.float16) * 0.02
                b = torch.randn(2560, device=dev, dtype=torch.float16)
                check(f"gemm {desc.value.decode()} split={sp} M={M} +bias", hi.gemm(x, W, bias=b), x.float() @ W.float().T + b.float(), 3e-3, 3e-3)
                res = torch.randn(M, H, device=dev, dtype=torch.float16)
                a = torch.randn(M, I, device=dev, dtype=torch.float16)
                W2 = torch.randn(H, I, device=dev, dtype=torch.float16) * 0.02
                out = res.clone()
                hi.gemm(a, W2, out=out, epi=1)
                check(f"gemm {desc.value.decode()} split={sp} M={M} +residual", out, res + (a.float() @ W2.float().T).half(), 4e-3, 2e-3)
                if b"32 weight" in desc.value:
                    g = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
                    u = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
                    ref = F.silu((x.float() @ g.float().T).half()) * (x.float() @ u.float().T).half()
                    check(f"gemm {desc.value.decode()} split={sp} M={M} +swiglu", hi.gemm(x, interleave(g, u), epi=2), ref, 4e-3, 4e-3)
        hi.lib.hi_gemm_force_splits(-1)
    hi.lib.hi_gemm_force_config(-1)


def test_gemm():
    big_shapes = [(5120, 3072), (3072, 8192)]
    big = [(torch.randn(2000, K, device=dev, dtype=torch.float16), torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02)
           for N, K in big_shapes]
    for cfg in [-1] + list(range(hi.lib.hi_gemm_num_configs())):
        hi.lib.hi_gemm_force_config(cfg)
        if cfg >= 0:
            desc = C.create_string_buffer(64)
            hi.lib.hi_gemm_config_desc(cfg, desc, 64)
        skinny = cfg >= 0 and b"skinny" in desc.value
        for M, N, K in [(128, 2560, 2048), (77, 2048, 5632), (512, 32000, 2048), (1000, 2048, 2048), (300, 1152, 896),
                        (64, 6144, 4096), (257, 4096, 14336)]:
            A = torch.randn(M, K, device=dev, dtype=torch.float16)
            W = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
            check(f"gemm cfg={cfg} M={M} N={N} K={K}", hi.gemm(A, W), A.float() @ W.float().T, 3e-3, 3e-3)
        # Real 3B prefill shapes at a non-tile-multiple M (hi_gemm falls back to a valid config for
        # one that cannot run the shape; skinny configs fall back to the same config as -1, so skip them)
        if not skinny:
            for (A, W), (N, K) in zip(big, big_shapes):
                check(f"gemm cfg={cfg} M=2000 N={N} K={K}", hi.gemm(A, W), A.float() @ W.float().T, 3e-3, 3e-3)
        M, H, I = 200, 2048, 5632
        A = torch.randn(M, H, device=dev, dtype=torch.float16)
        W = torch.randn(1152, H, device=dev, dtype=torch.float16) * 0.02
        b = torch.randn(1152, device=dev, dtype=torch.float16)
        check(f"gemm+bias cfg={cfg} M={M}", hi.gemm(A, W, bias=b), A.float() @ W.float().T + b.float(), 3e-3, 3e-3)
        g = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
        u = torch.randn(I, H, device=dev, dtype=torch.float16) * 0.02
        ref = F.silu((A.float() @ g.float().T).half()) * (A.float() @ u.float().T).half()
        check(f"gemm+swiglu cfg={cfg} M={M}", hi.gemm(A, interleave(g, u), epi=2), ref, 4e-3, 4e-3)
        res = torch.randn(M, H, device=dev, dtype=torch.float16)
        a = torch.randn(M, I, device=dev, dtype=torch.float16)
        W2 = torch.randn(H, I, device=dev, dtype=torch.float16) * 0.02
        out = res.clone()
        hi.gemm(a, W2, out=out, epi=1)
        check(f"gemm+residual cfg={cfg}", out, res + (a.float() @ W2.float().T).half(), 4e-3, 2e-3)
        # SwiGLU with a partial last column tile (2*I % 128 == 64, as for an 8-way sharded
        # Qwen2.5-0.5B): nothing may be written past column I, checked with a sentinel row below
        I2 = 608
        g = torch.randn(I2, H, device=dev, dtype=torch.float16) * 0.02
        u = torch.randn(I2, H, device=dev, dtype=torch.float16) * 0.02
        ref = F.silu((A.float() @ g.float().T).half()) * (A.float() @ u.float().T).half()
        buf = torch.full((M + 1, I2), 7.0, device=dev, dtype=torch.float16)
        hi.gemm(A, interleave(g, u), out=buf[:M], epi=2)
        check(f"gemm+swiglu cfg={cfg} M={M} I={I2} (partial tile)", buf[:M], ref, 4e-3, 4e-3)
        check(f"gemm+swiglu cfg={cfg} I={I2}: no write past the output", buf[M], torch.full_like(buf[M], 7.0), 0)
    hi.lib.hi_gemm_force_config(-1)


def test_gemm_graph_after_workspace_growth():
    """A split-K GEMM captured in a CUDA graph keeps the workspace it was captured with: a later,
    larger split-K GEMM must not free it (the replay would write into whatever reuses it)."""
    H = 2048
    hi.lib.hi_gemm_force_splits(4)
    x = torch.randn(16, H, device=dev, dtype=torch.float16)
    W = torch.randn(2560, H, device=dev, dtype=torch.float16) * 0.02
    out = torch.empty(16, 2560, device=dev, dtype=torch.float16)
    hi.gemm(x, W, out=out)  # eager: sizes the workspace
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        hi.gemm(x, W, out=out)
    A = torch.randn(1024, H, device=dev, dtype=torch.float16)
    W2 = torch.randn(8192, H, device=dev, dtype=torch.float16) * 0.02
    hi.gemm(A, W2)  # outgrows the workspace
    hi.lib.hi_gemm_force_splits(-1)
    del A, W2
    junk = torch.full((32 << 20,), 1e4, device=dev)  # may land on a freed workspace
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    check("split-K gemm graph replay after workspace growth", out, x.float() @ W.float().T, 3e-3, 3e-3)
    del junk


def test_argmax():
    """Greedy argmax vs torch.argmax: lowest index on ties, NaN wins (as in torch), and rows that
    are all -inf or all NaN give index 0 instead of an out-of-range token."""
    V = 32000
    lg = torch.randn(6, V, device=dev, dtype=torch.float16)
    lg[1, 100] = lg[1, 31999] = 20.0         # tie: lowest index
    lg[2, 5000] = float("nan")               # one NaN
    lg[2, 7] = float("nan")
    lg[3] = float("-inf")                    # all -inf
    lg[4] = float("nan")                     # all NaN
    lg[5, 31999] = float("inf")
    tok = torch.full((6,), -1, device=dev, dtype=torch.int32)
    hi.lib.hi_argmax(hi._p(lg), 6, V, hi._p(tok), hi._s())
    check("argmax (ties, NaN, all -inf, all NaN, +inf)", tok, lg.argmax(-1).int(), 0)


def rope_table(max_seq, D, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, D, 2, dtype=torch.int64).float() / D))
    ang = torch.arange(max_seq).float()[:, None] * inv[None, :]
    return torch.stack([ang.cos(), ang.sin()], -1).half().to(dev)  # [S, D/2, 2]


def hf_rope(x, cos, sin):  # x [T, H, D], cos/sin [T, D] fp16
    D = x.shape[-1]
    x1, x2 = x[..., :D // 2], x[..., D // 2:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos[:, None] + rot * sin[:, None]


def test_rope():
    for D, Hq, Hkv in [(64, 32, 4), (128, 24, 8), (128, 16, 16)]:
        S, T = 512, 37
        cs = rope_table(S, D)
        cos = torch.cat([cs[..., 0], cs[..., 0]], -1)
        sin = torch.cat([cs[..., 1], cs[..., 1]], -1)
        qkv = torch.randn(T, (Hq + 2 * Hkv) * D, device=dev, dtype=torch.float16)
        q0 = qkv[:, :Hq * D].view(T, Hq, D).clone()
        k0 = qkv[:, Hq * D:(Hq + Hkv) * D].view(T, Hkv, D).clone()
        v0 = qkv[:, (Hq + Hkv) * D:].view(T, Hkv, D).clone()
        kc = torch.zeros(3, Hkv, S, D, device=dev, dtype=torch.float16)
        vc = torch.zeros_like(kc)
        p0 = 100
        hi.rope_cache(qkv, Hq, Hkv, cs, kc, vc, pos_base=p0, slot_base=2)
        pp = torch.arange(p0, p0 + T, device=dev)
        check(f"rope D={D} q (exact HF fp16 order)", qkv[:, :Hq * D].view(T, Hq, D), hf_rope(q0, cos[pp], sin[pp]), 0)
        check(f"rope D={D} k -> cache", kc[2, :, p0:p0 + T].transpose(0, 1), hf_rope(k0, cos[pp], sin[pp]), 0)
        check(f"rope D={D} v -> cache", vc[2, :, p0:p0 + T].transpose(0, 1), v0, 0)
        # decode form: per-row positions, row b -> slot b
        qkv = torch.randn(3, (Hq + 2 * Hkv) * D, device=dev, dtype=torch.float16)
        k0 = qkv[:, Hq * D:(Hq + Hkv) * D].view(3, Hkv, D).clone()
        pos = torch.tensor([5, 400, 0], device=dev, dtype=torch.int32)
        hi.rope_cache(qkv, Hq, Hkv, cs, kc, vc, pos=pos)
        got = torch.stack([kc[b, :, pos[b]] for b in range(3)])
        check(f"rope D={D} decode per-row positions", got, hf_rope(k0, cos[pos.long()], sin[pos.long()]), 0)


def test_gemv_qkv_rope():
    for D, Hq, Hkv, H, bias in [(64, 32, 4, 2048, False), (128, 24, 8, 3072, False), (128, 12, 2, 1536, True), (64, 14, 2, 896, True)]:
        S = 512
        cs = rope_table(S, D)
        for M in [1, 3, 8]:
            x = torch.randn(M, H, device=dev, dtype=torch.float16)
            W = torch.randn((Hq + 2 * Hkv) * D, H, device=dev, dtype=torch.float16) * 0.02
            b = torch.randn((Hq + 2 * Hkv) * D, device=dev, dtype=torch.float16) if bias else None
            nw = (torch.rand(H, device=dev) + 0.5).half()
            pos = torch.randint(0, S, (M,), device=dev, dtype=torch.int32)
            # reference: separate GEMM (+bias), then the standalone RoPE/cache kernel
            qkv = hi.gemm(hi.rmsnorm(x, nw), W, bias=b)
            kc_r = torch.zeros(M, Hkv, S, D, device=dev, dtype=torch.float16)
            vc_r = torch.zeros_like(kc_r)
            hi.rope_cache(qkv, Hq, Hkv, cs, kc_r, vc_r, pos=pos)
            out = torch.zeros_like(qkv)
            kc, vc = torch.zeros_like(kc_r), torch.zeros_like(vc_r)
            hi.gemv_qkv_rope(x, W, nw, Hq, Hkv, cs, pos, kc, vc, out, bias=b)
            tag = f"D={D} G={Hq // Hkv}{' bias' if bias else ''} M={M}"
            # GEMM and GEMV sum in different orders: allow 1 fp16 ulp before the rotation
            check(f"fused qkv+rope+cache {tag}: q", out[:, :Hq * D], qkv[:, :Hq * D], 2e-2, 2e-3)
            check(f"fused qkv+rope+cache {tag}: k cache", kc, kc_r, 2e-2, 2e-3)
            check(f"fused qkv+rope+cache {tag}: v cache", vc, vc_r, 2e-2, 2e-3)


def ref_attn(q, k, v, qpos):  # q [T, Hq, D], k/v [L, Hkv, D], causal on absolute positions
    Hq, Hkv, D = q.shape[1], k.shape[1], q.shape[2]
    k = k.repeat_interleave(Hq // Hkv, 1).float()
    v = v.repeat_interleave(Hq // Hkv, 1).float()
    s = torch.einsum("thd,lhd->htl", q.float(), k) / D ** 0.5
    mask = torch.arange(k.shape[0], device=dev)[None, :] > qpos[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    return torch.einsum("htl,lhd->thd", s.softmax(-1), v)


def test_flash_prefill():
    for D, Hq, Hkv in [(64, 32, 4), (64, 32, 8), (128, 24, 8), (128, 16, 16), (128, 28, 4)]:
        S = 2048
        for T, p0 in [(1, 0), (64, 0), (100, 0), (517, 0), (1024, 0), (70, 300), (33, 1000)]:
            L = p0 + T
            q = torch.randn(T, Hq, D, device=dev, dtype=torch.float16)
            kc = torch.randn(Hkv, S, D, device=dev, dtype=torch.float16)
            vc = torch.randn(Hkv, S, D, device=dev, dtype=torch.float16)
            out = hi.flash_prefill(q.view(T, -1), kc, vc, T, p0, Hq, Hkv)
            ref = ref_attn(q, kc[:, :L].transpose(0, 1), vc[:, :L].transpose(0, 1), torch.arange(p0, L, device=dev))
            check(f"flash prefill D={D} G={Hq // Hkv} T={T} p0={p0}", out.view(T, Hq, D), ref, 2e-3, 1e-2)


def decode_attn_case(lens, Hq, Hkv, D, S, work=None):
    """Decode attention for len(lens) sequences of the given lengths vs the reference; returns the
    worst error in excess of the tolerance (<= 0 passes)."""
    B = len(lens)
    q = torch.randn(B, Hq * D, device=dev, dtype=torch.float16)
    kc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
    vc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
    pos = torch.tensor([l - 1 for l in lens], device=dev, dtype=torch.int32)
    out = hi.decode_attn(q, kc, vc, pos, Hq, Hkv, work=work).view(B, Hq, D)
    worst = 0.0
    for b, L in enumerate(lens):
        ref = ref_attn(q[b].view(1, Hq, D), kc[b, :, :L].transpose(0, 1), vc[b, :, :L].transpose(0, 1),
                       torch.tensor([L - 1], device=dev))
        worst = max(worst, ((out[b:b + 1].float() - ref).abs() - (2e-3 + 1e-2 * ref.abs())).max().item())
    return worst


def report(name, worst):
    print(f"{'OK  ' if worst <= 0 else 'FAIL'} {name:52s} worst_excess={worst:.3e}")
    if worst > 0:
        FAILS.append(name)


def test_decode_attn():
    cases = [[1], [37], [4000], [1, 300, 1024, 2049], list(range(5, 4000, 250))]
    for D, G in [(64, 8), (64, 4), (64, 1), (128, 4), (128, 3), (128, 1), (128, 7), (64, 7), (128, 8), (64, 2), (128, 6), (64, 5)]:
        Hkv = 4 if G != 1 else 16
        for lens in cases:
            B = len(lens)
            report(f"decode attn D={D} G={G} B={B} lens={lens if B <= 4 else f'{lens[0]}..{lens[-1]}'}",
                   decode_attn_case(lens, Hkv * G, Hkv, D, 4096))


def test_decode_work_reuse():
    """One work buffer shared by launches of decreasing batch size, as in an engine that serves a
    batch and then a single sequence: the split counters must not land in the larger launch's partials.
    One kv head per sequence means many splits per launch (as on a tensor-parallel rank)."""
    D, Hkv, G = 128, 1, 4
    work = torch.zeros(hi.lib.hi_decode_work_bytes(16, Hkv * G, Hkv, D), dtype=torch.uint8, device=dev)
    worst = max(decode_attn_case([2000 - 97 * b for b in range(B)], Hkv * G, Hkv, D, 2048, work=work)
                for B in [8, 1, 4, 16, 2])
    report("decode attn, one work buffer for B=8,1,4,16,2", worst)


def test_moe():
    """MoE router + experts (kernels written 2026-10-16 without a GPU; this is their first run)
    against the MoE math of scripts/ref_engine.py, which matches HF on CPU (tests/test_moe_ref.py).
    Router: expert indices and order on rows without a near-tie among the top k+1 logits, weights;
    an exact tie goes to the lower expert index. Experts: both paths (GEMV, grouped GEMM) and auto,
    given the kernel's own routing, for Granite-, Mixtral- and OLMoE-shaped layers."""
    from ref_engine import moe_experts, moe_route, r16
    g = torch.Generator(device=dev).manual_seed(0)

    def rn(*shape, s=1.0):
        return (torch.randn(*shape, generator=g, device=dev) * s).half()

    for name, E, k, H, I, flags, scale in [("granite-like", 32, 8, 1024, 512, 3, 0.22),
                                           ("mixtral-like", 8, 2, 512, 1024, 1, 1.0),
                                           ("olmoe-like", 64, 8, 2048, 1024, 2, 1.0)]:
        Wr = rn(E, H, s=0.05)
        wgu, wdown = rn(E, 2 * I, H, s=0.02), rn(E, H, I, s=0.02)
        for M in [1, 3, 4, 8, 37, 300]:
            xn, x = rn(M, H), rn(M, H)
            idx, w = hi.moe_router(xn, Wr, k, flags)
            logits = r16(xn.float() @ Wr.float().T)
            ridx, rw = moe_route(logits, k, bool(flags & 1), bool(flags & 2))
            top = logits.sort(-1, descending=True).values[:, :min(k + 1, E)]
            clear = (top[:, :-1] - top[:, 1:]).min(-1).values > 1e-2  # rows where fp32 order cannot flip a pick
            same = (idx.long() == ridx).all(-1)
            bad = (clear & ~same).sum().item()
            print(f"{'OK  ' if bad == 0 else 'FAIL'} moe router {name} M={M}: {bad} of {clear.sum().item()} clear rows differ")
            if bad:
                FAILS.append(f"moe router {name} M={M}")
            if same.any():
                check(f"moe router weights {name} M={M}", w[same], rw[same], 2e-3, 1e-2)
            ref = r16(x.float() + moe_experts(xn.float(), wgu, wdown, idx.long(), w, bool(flags & 2), scale))
            for path in (0, 1, 2):
                out = hi.moe_experts(xn, wgu, wdown, idx, w, x.clone(), flags, scale, path)
                check(f"moe experts {name} M={M} path={['auto', 'gemv', 'grouped'][path]}", out, ref, 4e-3, 4e-3)
    # exact tie: experts 2 and 5 have identical router rows and dominate -> [2, 5], weights 1/2 each
    E, k, H = 8, 2, 256
    Wr = rn(E, H, s=0.01)
    Wr[2] = Wr[5] = 0.05
    xn = (torch.rand(4, H, generator=g, device=dev) + 0.5).half()
    idx, w = hi.moe_router(xn, Wr, k, 1)
    ok = (idx.cpu() == torch.tensor([[2, 5]] * 4, dtype=torch.int32)).all().item()
    print(f"{'OK  ' if ok else 'FAIL'} moe router exact tie -> lower index first: {idx.tolist()}")
    if not ok:
        FAILS.append("moe router tie")
    check("moe router tie weights", w, torch.full_like(w, 0.5), 1e-6)


if __name__ == "__main__":
    print(torch.cuda.get_device_name())
    test_rmsnorm()
    test_gemv()
    test_gemm_splitk()
    test_gemm()
    test_gemm_graph_after_workspace_growth()
    test_argmax()
    test_rope()
    test_gemv_qkv_rope()
    test_flash_prefill()
    test_decode_attn()
    test_decode_work_reuse()
    test_moe()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)
