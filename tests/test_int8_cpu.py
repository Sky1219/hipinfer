"""CPU checks of the int8 (W8A16) weight path, no GPU needed (docs/quantization.md):

1. scripts/export.py --int8 on a tiny random Llama with a tied head: the version-4 file parsed in
   load_shard's order holds exactly the fake_quantize_hf weights (fp16 embedding, untied int8
   lm_head, q/k/v stacked, gate/up interleaved, every scale after its rows), and nothing else.
2. quantize_rows: max |q| = 127 per non-zero row, |w - q s| <= s / 2, zero rows stay zero.
3. The GEMV's int8 -> fp32 conversion (byte permute into the mantissa of 2^23, then subtract
   2^23 + 128), emulated bit for bit for all 256 byte values and every byte position.
4. The tensor-parallel scale handling: a row-parallel (column-split) layer whose ranks all keep the
   full row scales sums to the unsplit result; a column-parallel layer takes its rows' scales.

python tests/test_int8_cpu.py
"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import export  # noqa: E402

FAILS = []


def ok(name, cond):
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        FAILS.append(name)


def test_export_layout():
    tmp = tempfile.mkdtemp()
    cfg = LlamaConfig(vocab_size=300, hidden_size=128, intermediate_size=320, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=256, tie_word_embeddings=True)
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).half()
    with torch.no_grad():
        m.model.layers[1].mlp.down_proj.weight[3] = 0  # an all-zero row
    m.save_pretrained(tmp)
    out = os.path.join(tmp, "m.bin")
    subprocess.check_call([sys.executable, os.path.join(os.path.dirname(export.__file__), "export.py"), "--model", tmp,
                           "--out", out, "--int8"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    raw = open(out, "rb").read()
    h = struct.unpack("<10i2f2i", raw[:56])
    qtype = struct.unpack("<i", raw[56:60])[0]
    ok("version-4 header: version 4, qtype 1, tied flag clear", h[1] == 4 and qtype == 1 and (h[12] & 1) == 0)
    V, H, I, L, Hq, Hkv, D = h[2], h[3], h[4], h[5], h[6], h[7], h[8]
    off = [64]

    def nxt(n):  # load_shard's next(): 256-byte aligned tensors in order
        o = off[0]
        off[0] += (n + 255) // 256 * 256
        return o

    def f16(o, n):
        return torch.from_numpy(np.frombuffer(raw, np.float16, n, o).copy())

    def proj(N, K):  # int8 rows, then the fp16 row scales
        q = torch.from_numpy(np.frombuffer(raw, np.int8, N * K, nxt(N * K)).copy()).view(N, K)
        return q, f16(nxt(N * 2), N)

    ref = LlamaForCausalLM.from_pretrained(tmp, dtype=torch.float16)
    fq = export.fake_quantize_hf(LlamaForCausalLM.from_pretrained(tmp, dtype=torch.float16))
    nxt(D // 2 * 4)  # inv_freq
    emb = f16(nxt(V * H * 2), V * H).view(V, H)
    ok("embedding stays fp16 (file and fake-quantized HF)",
       torch.equal(emb, ref.model.embed_tokens.weight.data) and torch.equal(fq.model.embed_tokens.weight.data, emb))
    f16(nxt(H * 2), H)  # final norm
    q, s = proj(V, H)
    ok("lm_head: int8 + scales, = fake-quantized HF head", torch.equal(export.dequantize_rows(q, s), fq.lm_head.weight.data))
    same = True
    for i in range(L):
        ly = fq.model.layers[i]
        f16(nxt(H * 2), H)
        wqkv = export.dequantize_rows(*proj((Hq + 2 * Hkv) * D, H))
        wo = export.dequantize_rows(*proj(H, Hq * D))
        f16(nxt(H * 2), H)
        wgu = export.dequantize_rows(*proj(2 * I, H))
        qd, sd = proj(H, I)
        same &= torch.equal(wqkv, torch.cat([ly.self_attn.q_proj.weight, ly.self_attn.k_proj.weight, ly.self_attn.v_proj.weight]))
        same &= torch.equal(wo, ly.self_attn.o_proj.weight)
        same &= torch.equal(wgu, export.interleave_gate_up(ly.mlp.gate_proj.weight, ly.mlp.up_proj.weight))
        same &= torch.equal(export.dequantize_rows(qd, sd), ly.mlp.down_proj.weight)
        if i == 1:
            same &= int(qd[3].abs().max()) == 0 and float(sd[3]) == 0.0
    ok("every layer's projections = fake-quantized HF (zero row kept)", bool(same))
    ok(f"file size = the tensors load_shard reads ({len(raw)} bytes)", off[0] == len(raw))


def test_quantize_rows():
    torch.manual_seed(1)
    W = torch.randn(300, 2048, dtype=torch.float16) * 0.02
    W[7] = 0
    W[9, 5] = 3.0  # outlier row
    q, s = export.quantize_rows(W)
    nz = W.float().abs().amax(1) > 0
    peak = q.abs().amax(1)
    ok("max |q| = 127 per non-zero row, zero rows stay zero", bool((peak[nz] == 127).all()) and int(peak[~nz].max()) == 0)
    step = s.float()[:, None]
    ok("round to nearest: |w - q s| <= s / 2", bool(((W.float() - q.float() * step).abs() <= step * 0.5005).all()))
    q2, s2 = export.quantize_rows(W, chunk=7)
    ok("row chunking does not change the result", torch.equal(q, q2) and torch.equal(s, s2))


def byte_perm(x, y, sel):
    """CUDA __byte_perm: result byte n is byte (sel >> 4n) & 7 of the 8 bytes {y:x} (x = bytes 0-3)."""
    b = [(x >> (8 * i)) & 0xFF for i in range(4)] + [(y >> (8 * i)) & 0xFF for i in range(4)]
    return sum(b[(sel >> (4 * n)) & 7] << (8 * n) for n in range(4))


def test_byte_perm_conversion():
    good = True
    for v in range(-128, 128):
        for pos in range(4):
            w = ((v & 0xFF) << (8 * pos)) | (0x5A << (8 * ((pos + 1) % 4)))  # a neighbour byte that must not leak in
            u = w ^ 0x80808080
            bits = byte_perm(u, 0x4B000000, 0x7540 | pos)
            f = np.array([bits], np.uint32).view(np.float32)[0] - np.float32(8388736.0)
            good &= f == np.float32(v)
    ok("int8 -> fp32 via byte permute + subtract: exact for all bytes and positions", bool(good))


def test_tp_scales():
    torch.manual_seed(2)
    N, K, world = 64, 512, 4
    q, s = export.quantize_rows(torch.randn(N, K, dtype=torch.float16) * 0.02)
    x = torch.randn(K, dtype=torch.float64)
    full = s.double() * (q.double() @ x)
    # row-parallel layer (o / down): rank r holds columns r*K/world.. and ALL N scales
    parts = [s.double() * (q[:, r * K // world:(r + 1) * K // world].double() @ x[r * K // world:(r + 1) * K // world])
             for r in range(world)]
    ok("row-parallel split: per-rank s * partial sums to s * full", torch.allclose(sum(parts), full, rtol=1e-12, atol=1e-12))
    # column-parallel layer (qkv / gate-up / lm_head): rank r holds rows and their scales
    rows = [s[r * N // world:(r + 1) * N // world].double() * (q[r * N // world:(r + 1) * N // world].double() @ x)
            for r in range(world)]
    ok("column-parallel split: rows with their own scales concatenate to the full result",
       torch.allclose(torch.cat(rows), full, rtol=1e-12, atol=1e-12))


if __name__ == "__main__":
    test_quantize_rows()
    test_byte_perm_conversion()
    test_tp_scales()
    test_export_layout()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)
