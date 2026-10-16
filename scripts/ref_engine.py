"""Plain-PyTorch (CPU) reference for the hipinfer engine: reads an exported weight file (version 2
dense or version 3 MoE, scripts/export.py) and computes what the GPU engine computes, from the
same layouts: wqkv rows [q; k; v], gate/up interleaved in groups of 16, RoPE from the stored
inv_freq, and for MoE the router, top-k, per-expert SwiGLU and the weighted combine.

mode "fp16" rounds to fp16 where the engine (and HF) does: RMSNorm output, every projection
output, RoPE products, SiLU, attention output, residual adds, the router logits, the routing
weights (when the model rounds them) and each expert's weighted output; all accumulation is fp32.
mode "fp32" never rounds (weights are still the file's fp16 values): for checking layouts and
math against an fp32 HF model without fp16 noise.

It is a test oracle, not fast: experts run as a loop over the experts that were hit.

python scripts/ref_engine.py --weights granite.bin --tokens 1 2 3 --new 8
"""
import argparse
import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

MAGIC = 0x48495046


def r16(t):
    """Round to fp16, keep fp32 storage."""
    return t.half().float()


def deinterleave(wgu):
    """[..., 2I, H] in groups of 16 (16 gate rows, 16 up rows) -> gate [..., I, H], up [..., I, H]."""
    n2, H = wgu.shape[-2:]
    g = wgu.reshape(*wgu.shape[:-2], n2 // 32, 2, 16, H)
    return g[..., 0, :, :].reshape(*wgu.shape[:-2], n2 // 2, H), g[..., 1, :, :].reshape(*wgu.shape[:-2], n2 // 2, H)


def swiglu(g, u, r=r16):
    # kernels.cuh silu_mul: fp16(g); fp16(silu); fp16(s * fp16(u))
    g, u = r(g), r(u)
    return r(r(g * torch.sigmoid(g)) * u)


def moe_route(logits, k, norm_topk, fp16_weights, r=r16):
    """Top-k of the (already fp16-rounded) router logits [T, E], ties to the lower expert index, in
    descending order, and their weights: softmax over the k (norm_topk) or over all E, rounded to
    fp16 when the model does. Returns (idx [T, k] long, weights [T, k] fp32)."""
    order = torch.sort(-logits, dim=-1, stable=True).indices  # stable: equal logits keep expert order
    idx = order[:, :k]
    sel = logits.gather(1, idx)
    m = sel[:, :1]
    e = torch.exp(sel - m)
    wts = e / (e.sum(-1, keepdim=True) if norm_topk else torch.exp(logits - m).sum(-1, keepdim=True))
    return idx, (r(wts) if fp16_weights else wts)


def moe_experts(xn, wgu, wdown, idx, wts, fp16_weights, scale, r=r16):
    """MoE block output (before the residual add) for normalised rows xn [T, H]: each selected
    expert's SwiGLU MLP (wgu [E, 2I, H] interleaved, wdown [E, H, I]), down projection rounded to
    fp16, times its weight (rounded to fp16 if fp16_weights), summed over the k in fp32, rounded
    once, then x scale (rounded) when scale != 1."""
    T, H = xn.shape
    contrib = torch.zeros(T, idx.shape[1], H, device=xn.device)
    for e in idx.unique().tolist():
        t, j = (idx == e).nonzero(as_tuple=True)
        gate, up = deinterleave(wgu[e].float())
        act = swiglu(xn[t] @ gate.T, xn[t] @ up.T, r)
        y = r(act @ wdown[e].float().T)
        yw = y * wts[t, j][:, None].float()
        contrib[t, j] = r(yw) if fp16_weights else yw
    out = r(contrib.sum(1))
    return r(out * scale) if scale != 1.0 else out


def read_header(path):
    """Config dict from a version 2 or 3 header."""
    with open(path, "rb") as f:
        raw = f.read(128)
    h = struct.unpack("<10i2f2i", raw[:56])
    if h[0] != MAGIC or h[1] not in (2, 3):
        raise ValueError(f"{path}: not a version-2/3 hipinfer weight file")
    keys = ["version", "vocab", "hidden", "inter", "layers", "heads", "kv_heads", "head_dim", "max_pos"]
    c = dict(zip(keys, h[1:10]), eps=h[10], attn_scaling=h[11], tied=bool(h[12] & 1), qkv_bias=bool(h[12] & 2),
             moe=bool(h[12] & 4), qk_norm=bool(h[12] & 8), window=h[13], header_bytes=64)
    if c["version"] == 3:
        E, k, mf, scale = struct.unpack("<3if", raw[64:80])
        c.update(n_experts=E, top_k=k, norm_topk=bool(mf & 1), fp16_weights=bool(mf & 2), moe_scale=scale,
                 header_bytes=128)
    return c


class RefEngine:
    def __init__(self, path, mode="fp16"):
        assert mode in ("fp16", "fp32")
        self.mode = mode
        self.c = c = read_header(path)
        H, I, V, D = c["hidden"], c["inter"], c["vocab"], c["head_dim"]
        Hq, Hkv = c["heads"], c["kv_heads"]
        QKV = (Hq + 2 * Hkv) * D
        data = np.memmap(path, dtype=np.uint8, mode="r")
        off = c["header_bytes"]

        def take(n, dtype=np.float16, shape=None):
            nonlocal off
            nb = n * np.dtype(dtype).itemsize
            a = np.frombuffer(data[off:off + nb].tobytes(), dtype=dtype)
            off += (nb + 255) // 256 * 256
            t = torch.from_numpy(a.copy()).float()  # fp16 values held exactly in fp32
            return t.view(*shape) if shape else t

        self.inv_freq = take(D // 2, np.float32)
        self.embed = take(V * H, shape=(V, H))
        self.final_norm = take(H)
        self.lm_head = self.embed if c["tied"] else take(V * H, shape=(V, H))
        self.L = []
        for _ in range(c["layers"]):
            w = dict(attn_norm=take(H), wqkv=take(QKV * H, shape=(QKV, H)))
            w["bqkv"] = take(QKV) if c["qkv_bias"] else None
            w["wo"] = take(H * Hq * D, shape=(H, Hq * D))
            w["mlp_norm"] = take(H)
            if c["moe"]:
                E = c["n_experts"]
                if c["qk_norm"]:
                    w["q_norm"], w["k_norm"] = take(Hq * D), take(Hkv * D)
                w["router"] = take(E * H, shape=(E, H))
                w["wgu"] = take(E * 2 * I * H, shape=(E, 2 * I, H))
                w["wdown"] = take(E * H * I, shape=(E, H, I))
            else:
                w["wgu"] = take(2 * I * H, shape=(2 * I, H))
                w["wdown"] = take(H * I, shape=(H, I))
            self.L.append(w)
        assert off <= data.size + 255, f"weight file too small ({data.size} < {off})"
        self.reset()

    # ---------------------------------------------------------------- rounding helpers
    def r(self, t):
        """Round to fp16 (mode fp16) or pass through (fp32)."""
        return t.half().float() if self.mode == "fp16" else t

    def mm(self, x, W):
        """x[M, K] @ W[N, K]^T with fp32 accumulation, output rounded like a GPU GEMM/GEMV."""
        return self.r(x @ W.T)

    def rmsnorm(self, x, w):
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.c["eps"])
        return self.r(w * self.r(x * r))

    def swiglu(self, g, u):
        return swiglu(g, u, self.r)

    def rope_table(self, pos):
        """(cos, sin) [T, D/2] as load_shard builds them: fp32 angle, cos/sin in double, x scaling, fp16."""
        a = pos.float()[:, None] * self.inv_freq[None, :]
        cos = (torch.cos(a.double()).float() * self.c["attn_scaling"])
        sin = (torch.sin(a.double()).float() * self.c["attn_scaling"])
        return self.r(cos), self.r(sin)

    def rope(self, x, cos, sin):
        """x [T, heads, D]; rope_pair's per-op rounding."""
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        c, s = cos[:, None, :], sin[:, None, :]
        y1 = self.r(self.r(x1 * c) - self.r(x2 * s))
        y2 = self.r(self.r(x2 * c) + self.r(x1 * s))
        return torch.cat([y1, y2], -1)

    # ---------------------------------------------------------------- model
    def reset(self):
        self.kc = [None] * self.c["layers"]
        self.vc = [None] * self.c["layers"]
        self.pos = 0
        self.last_routing = []  # per layer: (topk_idx [T, k], topk_w [T, k]) of the last forward

    def attention(self, l, w, xn, pos):
        c = self.c
        Hq, Hkv, D = c["heads"], c["kv_heads"], c["head_dim"]
        T = xn.shape[0]
        qkv = xn @ w["wqkv"].T
        if w["bqkv"] is not None:
            qkv = qkv + w["bqkv"]
        qkv = self.r(qkv)
        q, k, v = qkv.split([Hq * D, Hkv * D, Hkv * D], -1)
        if c["qk_norm"]:  # RMSNorm over the whole q (all heads) and the whole k, before RoPE
            q, k = self.rmsnorm(q, w["q_norm"]), self.rmsnorm(k, w["k_norm"])
        cos, sin = self.rope_table(pos)
        q = self.rope(q.reshape(T, Hq, D), cos, sin)
        k = self.rope(k.reshape(T, Hkv, D), cos, sin)
        v = v.reshape(T, Hkv, D)
        self.kc[l] = k if self.kc[l] is None else torch.cat([self.kc[l], k])
        self.vc[l] = v if self.vc[l] is None else torch.cat([self.vc[l], v])
        K, Vv = self.kc[l], self.vc[l]
        S = K.shape[0]
        G = Hq // Hkv
        K, Vv = K.repeat_interleave(G, 1), Vv.repeat_interleave(G, 1)  # [S, Hq, D]
        scores = torch.einsum("thd,shd->hts", q, K) / math.sqrt(D)
        causal = pos[:, None] >= torch.arange(S)[None, :]
        scores = scores.masked_fill(~causal[None], float("-inf"))
        p = torch.softmax(scores, -1)
        o = self.r(torch.einsum("hts,shd->thd", p, Vv)).reshape(T, Hq * D)
        return self.mm(o, w["wo"])

    def route(self, w, xn):
        c = self.c
        return moe_route(self.mm(xn, w["router"]), c["top_k"], c["norm_topk"], c["fp16_weights"], self.r)

    def moe(self, w, xn):
        c = self.c
        idx, wts = self.route(w, xn)
        self.last_routing.append((idx, wts))
        return moe_experts(xn, w["wgu"], w["wdown"], idx, wts, c["fp16_weights"], c["moe_scale"], self.r)

    def dense_mlp(self, w, xn):
        gate, up = deinterleave(w["wgu"])
        return self.mm(self.swiglu(xn @ gate.T, xn @ up.T), w["wdown"])

    @torch.no_grad()
    def forward(self, tokens):
        """Run `tokens` from the current position (KV cache kept); returns fp32 logits [T, vocab]
        (holding fp16 values in mode fp16)."""
        c = self.c
        tokens = torch.as_tensor(tokens, dtype=torch.long)
        T = tokens.numel()
        pos = torch.arange(self.pos, self.pos + T)
        self.last_routing = []
        x = self.embed[tokens]
        for l, w in enumerate(self.L):
            x = self.r(x + self.attention(l, w, self.rmsnorm(x, w["attn_norm"]), pos))
            xn = self.rmsnorm(x, w["mlp_norm"])
            x = self.r(x + (self.moe(w, xn) if c["moe"] else self.dense_mlp(w, xn)))
        self.pos += T
        return self.mm(self.rmsnorm(x, self.final_norm), self.lm_head)

    def generate(self, prompt, n_new):
        """Greedy: n_new tokens (the first from the prompt's last logits), plus each step's logits."""
        self.reset()
        logits = self.forward(prompt)[-1]
        out, steps = [], []
        for i in range(n_new):
            steps.append(logits)
            t = int(logits.argmax())
            out.append(t)
            if i + 1 < n_new:
                logits = self.forward([t])[-1]
        return out, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--new", type=int, default=8)
    ap.add_argument("--mode", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    eng = RefEngine(a.weights, a.mode)
    print({k: v for k, v in eng.c.items()})
    print(eng.generate(a.tokens, a.new)[0])


if __name__ == "__main__":
    main()
