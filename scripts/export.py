"""Export a HF Llama-family checkpoint (Llama 2/3, TinyLlama, Mistral, Qwen2/2.5, DeepSeek-Coder...)
or a Llama-shaped Mixture-of-Experts checkpoint (Mixtral, Granite MoE, OLMoE) to the flat fp16
file read by hi_load.

Layout (version 2, dense models): 64-byte header, then tensors on 256-byte boundaries:
  header: int32 magic, version, vocab, hidden, inter, layers, heads, kv_heads, head_dim, max_pos,
          float eps, float attention_scaling, int32 flags (bit 0: tied lm_head, bit 1: qkv bias),
          int32 sliding_window (0: full attention; the engine then requires max_seq <= it)
  inv_freq        fp32 [head_dim/2]   (RoPE frequencies exactly as HF computes them, incl.
                                        llama3 / linear / dynamic-free scaling)
  embed           [vocab, hidden]
  final_norm      [hidden]
  lm_head         [vocab, hidden]      (absent when tied: the engine reuses embed)
  per layer: attn_norm, wqkv = [q; k; v] rows, (bqkv [(heads+2kv)*head_dim] if flag), wo,
             mlp_norm, wgu, wdown
wgu interleaves gate/up rows in groups of 16 (16 gate rows, then the same 16 up rows) so one
warp / one 32-column GEMM tile holds a matching pair.

Layout (version 3, MoE models; docs/moe-design.md): 128-byte header whose first 56 bytes are the
version-2 fields (inter = the per-expert intermediate size; flags bit 2: MoE, bit 3: QK-norm),
then at byte 64: int32 n_experts, int32 top_k, int32 moe_flags (bit 0: renormalise the top-k
weights, bit 1: routing weights rounded to fp16), float moe_out_scale. Tensors as version 2, except
each layer's MLP is
  (q_norm [heads*head_dim], k_norm [kv_heads*head_dim] after wo, if QK-norm)
  router [E, hidden], wgu [E][2*inter, hidden] (each expert interleaved as above), wdown [E][hidden, inter]

Layout (version 4, dense models with --int8; docs/quantization.md): the version-2 layout and header,
plus int32 qtype at byte 56 (1: int8, one fp16 scale per output row). Every projection (wqkv, wo,
wgu, wdown) and the lm_head is stored as int8 rows q [N, K] followed by fp16 scales s [N] (each on
its own 256-byte boundary), W ~ q * s with symmetric round-to-nearest (quantize_rows). The lm_head
is always written (flag bit 0 is clear): the embedding, which the engine only gathers rows from,
stays fp16. Embedding, norms and biases are fp16 as in version 2.

Granite's scalar multipliers are folded in so the engine runs the plain Llama graph: embedding x
embedding_multiplier, lm_head / logits_scaling (untying a tied head), q rows x
attention_multiplier * sqrt(head_dim) (the engine scales scores by 1/sqrt(head_dim)), wo x
residual_multiplier, and the MLP side of residual_multiplier into wdown (dense) or
moe_out_scale (MoE, applied exactly in the combine).
"""
import argparse
import math
import struct

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MAGIC = 0x48495046
VERSION = 2
VERSION_MOE = 3
VERSION_INT8 = 4
QTYPE_INT8_ROW = 1  # version-4 header field at byte 56


def quantize_rows(w, chunk=8192):
    """Symmetric round-to-nearest int8 with one fp16 scale per output row (W8A16):
    s[n] = fp16(max_k |w[n, k]| / 127), q[n, k] = clamp(round(w[n, k] / s[n]), -127, 127), so w ~ q * s.
    The rounding uses the stored fp16 scale. An all-zero row (or one whose scale underflows) gets q = 0.
    Rows are processed in chunks to bound the fp32 temporaries."""
    w = w.detach()
    q = torch.empty(w.shape, dtype=torch.int8, device=w.device)
    s = torch.empty(w.shape[0], dtype=torch.float16, device=w.device)
    for r0 in range(0, w.shape[0], chunk):
        x = w[r0:r0 + chunk].float()
        sc = (x.abs().amax(dim=1) / 127.0).half()
        sf = sc.float()[:, None]
        q[r0:r0 + chunk] = torch.where(sf > 0, torch.round(x / sf.clamp_min(1e-30)), 0.0).clamp_(-127, 127).to(torch.int8)
        s[r0:r0 + chunk] = sc
    return q, s


def dequantize_rows(q, s):
    """fp16(q * s): the weights the engine's int8 prefill GEMMs compute with."""
    return (q.float() * s.float()[:, None]).half()


def fake_quantize_hf(model):
    """Replace a HF Llama-family model's projection and lm_head weights (in place) by their int8
    round trip dequantize_rows(*quantize_rows(w)): the model an --int8 export computes, in HF, as
    a reference that differs from the engine only in summation order and where the row scale is
    applied. The embedding stays fp16; a tied lm_head is untied first. Only for models without
    folded Granite multipliers."""
    cfg = model.config
    for k in ("embedding_multiplier", "logits_scaling", "residual_multiplier", "attention_multiplier"):
        if getattr(cfg, k, None) not in (None, 1.0):
            raise SystemExit(f"fake_quantize_hf: {k} is folded into the exported weights; not supported here")
    for layer in model.model.layers:
        for lin in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj,
                    layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
            lin.weight.data = dequantize_rows(*quantize_rows(lin.weight.data)).to(lin.weight.dtype)
    head = model.lm_head.weight.data
    model.lm_head.weight = torch.nn.Parameter(dequantize_rows(*quantize_rows(head)).to(head.dtype), requires_grad=False)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    return model


def interleave_gate_up(gate, up, group=16):
    inter, hidden = gate.shape
    g = gate.reshape(inter // group, group, hidden)
    u = up.reshape(inter // group, group, hidden)
    return torch.stack([g, u], dim=1).reshape(2 * inter, hidden)


def rope_inv_freq(model, cfg, head_dim):
    """inv_freq and attention_scaling exactly as the HF rotary embedding module holds them."""
    rot = getattr(model.model, "rotary_emb", None)
    if rot is not None and hasattr(rot, "inv_freq"):
        return rot.inv_freq.detach().float().cpu(), float(getattr(rot, "attention_scaling", 1.0))
    theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
    return inv, 1.0


def sliding_window(cfg):
    """Attention window the model applies within its context (0: none). Mistral-7B-v0.1 uses 4096;
    Qwen2 configs carry a window that only applies with use_sliding_window."""
    w = getattr(cfg, "sliding_window", None) or 0
    if not getattr(cfg, "use_sliding_window", True):
        return 0
    return w if w < cfg.max_position_embeddings else 0


# Router semantics per MoE architecture (see docs/moe-design.md): renormalise the top-k weights,
# and whether HF rounds them to fp16 before the combine.
MOE_ARCH = {
    "mixtral": dict(norm_topk=True, fp16_weights=False),
    "granitemoe": dict(norm_topk=True, fp16_weights=True),   # softmax over the top-k logits
    "olmoe": dict(norm_topk=None, fp16_weights=True),        # None: config.norm_topk_prob
}


def moe_config(cfg):
    """(n_experts, top_k, moe_flags) for a MoE config, or None for a dense one."""
    E = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    if not E:
        return None
    if cfg.model_type not in MOE_ARCH:
        raise SystemExit(f"MoE architecture {cfg.model_type!r} not supported (shared experts, other routers); "
                         f"supported: {sorted(MOE_ARCH)}")
    a = MOE_ARCH[cfg.model_type]
    norm = a["norm_topk"] if a["norm_topk"] is not None else bool(cfg.norm_topk_prob)
    return E, cfg.num_experts_per_tok, (1 if norm else 0) | (2 if a["fp16_weights"] else 0)


def first(sd, *keys):
    for k in keys:
        if k in sd:
            return sd[k]
    return None


def moe_layer(sd, p, E, inter, hidden):
    """Router [E, H] and per-expert gate, up [E, I, H], down [E, H, I] of layer prefix p, from the
    transformers 5.x fused layout (experts.gate_up_proj [E, 2I, H], gate rows first) or the 4.x
    per-expert / Granite input_linear layouts."""
    router = first(sd, p + "mlp.gate.weight", p + "block_sparse_moe.router.weight", p + "block_sparse_moe.gate.weight",
                   p + "block_sparse_moe.router.layer.weight")
    gate = up = down = None
    for blk in ("mlp.experts.", "block_sparse_moe.experts."):
        if p + blk + "gate_up_proj" in sd:
            gu, down = sd[p + blk + "gate_up_proj"], sd[p + blk + "down_proj"]
            # transposed storage ([E, H, 2I] / [E, I, H]); judged by down_proj, since gate_up is
            # square when hidden == 2 * inter (Granite 1B)
            if down.shape == (E, inter, hidden) and inter != hidden:
                gu, down = gu.transpose(1, 2), down.transpose(1, 2)
            gate, up = gu[:, :inter], gu[:, inter:]
        elif p + blk + "0.w1.weight" in sd:  # 4.x Mixtral: w1 gate, w3 up, w2 down
            gate = torch.stack([sd[f"{p}{blk}{e}.w1.weight"] for e in range(E)])
            up = torch.stack([sd[f"{p}{blk}{e}.w3.weight"] for e in range(E)])
            down = torch.stack([sd[f"{p}{blk}{e}.w2.weight"] for e in range(E)])
        elif p + blk + "0.gate_proj.weight" in sd:  # 4.x OLMoE
            gate = torch.stack([sd[f"{p}{blk}{e}.gate_proj.weight"] for e in range(E)])
            up = torch.stack([sd[f"{p}{blk}{e}.up_proj.weight"] for e in range(E)])
            down = torch.stack([sd[f"{p}{blk}{e}.down_proj.weight"] for e in range(E)])
        if gate is not None:
            break
    if gate is None and p + "block_sparse_moe.input_linear.weight" in sd:  # 4.x Granite MoE
        gu = sd[p + "block_sparse_moe.input_linear.weight"]
        gate, up, down = gu[:, :inter], gu[:, inter:], sd[p + "block_sparse_moe.output_linear.weight"]
    if router is None or gate is None:
        raise SystemExit(f"{p}: MoE weights not found (keys: {[k for k in sd if k.startswith(p)]})")
    assert router.shape == (E, hidden) and gate.shape == (E, inter, hidden) and down.shape == (E, hidden, inter), \
        (router.shape, gate.shape, down.shape)
    return router, gate, up, down


def scaled(t, s):
    """fp16(t * s), the product taken in fp32 (what PyTorch does for an fp16 tensor times a scalar)."""
    return t if s == 1.0 else (t.float() * s).to(torch.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--int8", action="store_true", help="version 4: int8 projections and lm_head (W8A16, one fp16 "
                    "scale per output row, round-to-nearest)")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    sd = model.state_dict()
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    inv_freq, attn_scaling = rope_inv_freq(model, cfg, head_dim)
    assert inv_freq.numel() == head_dim // 2
    tied = "lm_head.weight" not in sd or sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr() \
        or torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
    has_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    moe = moe_config(cfg)
    if a.int8 and moe:
        raise SystemExit("--int8 supports dense models only (the MoE expert kernels are fp16)")
    qk_norm = "model.layers.0.self_attn.q_norm.weight" in sd
    if qk_norm and not moe:
        raise SystemExit("QK-norm is only supported in the MoE (version 3) layout")
    # Granite (dense or MoE) scalar multipliers; 1.0 for every other model
    emb_mult = float(getattr(cfg, "embedding_multiplier", 1.0))
    logit_div = float(getattr(cfg, "logits_scaling", 1.0))
    res_mult = float(getattr(cfg, "residual_multiplier", 1.0))
    attn_mult = getattr(cfg, "attention_multiplier", None)
    q_mult = float(attn_mult) * math.sqrt(head_dim) if attn_mult is not None else 1.0
    embed = sd["model.embed_tokens.weight"]
    lm_head = embed if tied else sd["lm_head.weight"]
    if tied and (emb_mult != 1.0 or logit_div != 1.0):
        tied = False  # embed and lm_head get different scales
    if a.int8:
        tied = False  # int8 lm_head written separately; the embedding stays fp16
    flags = (1 if tied else 0) | (2 if has_bias else 0) | (4 if moe else 0) | (8 if qk_norm else 0)
    window = sliding_window(cfg)
    version = VERSION_MOE if moe else VERSION_INT8 if a.int8 else VERSION
    hdr = struct.pack("<10i2f2i", MAGIC, version, cfg.vocab_size, cfg.hidden_size,
                      cfg.intermediate_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads,
                      head_dim, cfg.max_position_embeddings, cfg.rms_norm_eps, attn_scaling, flags, window)
    if moe:
        E, top_k, moe_flags = moe
        hdr = hdr.ljust(64, b"\0") + struct.pack("<3if", E, top_k, moe_flags, res_mult)
        hdr = hdr.ljust(128, b"\0")
    else:
        if a.int8:
            hdr += struct.pack("<i", QTYPE_INT8_ROW)
        hdr = hdr.ljust(64, b"\0")
    print(f"{a.model}: hidden {cfg.hidden_size} inter {cfg.intermediate_size} layers {cfg.num_hidden_layers} "
          f"heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} head_dim {head_dim} vocab {cfg.vocab_size} "
          f"tied={tied} qkv_bias={has_bias} rope={getattr(cfg, 'rope_scaling', None)} sliding_window={window or None}"
          + (" weights=int8 (version 4)" if a.int8 else ""))
    if moe:
        print(f"  MoE: {moe[0]} experts, top-{moe[1]}, norm_topk={bool(moe[2] & 1)} fp16_weights={bool(moe[2] & 2)} "
              f"qk_norm={qk_norm} out_scale={res_mult}")
    if (emb_mult, logit_div, res_mult, q_mult) != (1.0, 1.0, 1.0, 1.0):
        print(f"  folded: embedding x{emb_mult}, lm_head /{logit_div}, q x{q_mult}, wo x{res_mult}, "
              f"{'moe_out_scale' if moe else 'wdown x'} {res_mult}")

    H, I, Hq, D = cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, head_dim
    with open(a.out, "wb") as f:
        f.write(hdr)

        def put_bytes(b):
            f.write(b)
            f.write(b"\0" * ((-len(b)) % 256))

        def put(t):
            put_bytes(t.detach().to(torch.float16).contiguous().numpy().tobytes())

        def put_w(t):  # a projection [N, K]: fp16, or (--int8) int8 rows then their fp16 scales
            if not a.int8:
                return put(t)
            q, sc = quantize_rows(t)
            put_bytes(q.contiguous().numpy().tobytes())
            put(sc)

        put_bytes(inv_freq.contiguous().numpy().astype("float32").tobytes())
        put(scaled(embed, emb_mult))
        put(sd["model.norm.weight"])
        if not tied:
            put_w(scaled(lm_head, 1.0 / logit_div))
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            put(sd[p + "input_layernorm.weight"])
            put_w(torch.cat([scaled(sd[p + "self_attn.q_proj.weight"], q_mult), sd[p + "self_attn.k_proj.weight"],
                             sd[p + "self_attn.v_proj.weight"]], 0))
            if has_bias:
                put(torch.cat([scaled(sd[p + "self_attn.q_proj.bias"], q_mult), sd[p + "self_attn.k_proj.bias"],
                               sd[p + "self_attn.v_proj.bias"]], 0))
            put_w(scaled(sd[p + "self_attn.o_proj.weight"], res_mult))
            put(sd[p + "post_attention_layernorm.weight"])
            if not moe:
                put_w(interleave_gate_up(sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]))
                put_w(scaled(sd[p + "mlp.down_proj.weight"], res_mult))
                continue
            if qk_norm:  # QK-norm weights scale q after the norm: fold q_mult there instead
                assert q_mult == 1.0
                put(sd[p + "self_attn.q_norm.weight"])
                put(sd[p + "self_attn.k_norm.weight"])
            router, gate, up, down = moe_layer(sd, p, moe[0], I, H)
            put(router)
            put(torch.stack([interleave_gate_up(gate[e], up[e]) for e in range(moe[0])]))
            put(down)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
