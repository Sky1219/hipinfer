"""Export a HF Llama-family checkpoint (Llama 2/3, TinyLlama, Mistral, Qwen2/2.5, DeepSeek-Coder...)
to the flat fp16 file read by hi_load.

Layout (version 2): 64-byte header, then tensors on 256-byte boundaries:
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
"""
import argparse
import struct

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MAGIC = 0x48495046
VERSION = 2


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16)
    sd = model.state_dict()
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    inv_freq, attn_scaling = rope_inv_freq(model, cfg, head_dim)
    assert inv_freq.numel() == head_dim // 2
    tied = "lm_head.weight" not in sd or sd["lm_head.weight"].data_ptr() == sd["model.embed_tokens.weight"].data_ptr() \
        or torch.equal(sd["lm_head.weight"], sd["model.embed_tokens.weight"])
    has_bias = "model.layers.0.self_attn.q_proj.bias" in sd
    flags = (1 if tied else 0) | (2 if has_bias else 0)
    window = sliding_window(cfg)
    hdr = struct.pack("<10i2f2i", MAGIC, VERSION, cfg.vocab_size, cfg.hidden_size, cfg.intermediate_size,
                      cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, head_dim,
                      cfg.max_position_embeddings, cfg.rms_norm_eps, attn_scaling, flags, window)
    hdr = hdr.ljust(64, b"\0")
    print(f"{a.model}: hidden {cfg.hidden_size} inter {cfg.intermediate_size} layers {cfg.num_hidden_layers} "
          f"heads {cfg.num_attention_heads}/{cfg.num_key_value_heads} head_dim {head_dim} vocab {cfg.vocab_size} "
          f"tied={tied} qkv_bias={has_bias} rope={getattr(cfg, 'rope_scaling', None)} sliding_window={window or None}")

    with open(a.out, "wb") as f:
        f.write(hdr)

        def put_bytes(b):
            f.write(b)
            f.write(b"\0" * ((-len(b)) % 256))

        def put(t):
            put_bytes(t.detach().to(torch.float16).contiguous().numpy().tobytes())

        put_bytes(inv_freq.contiguous().numpy().astype("float32").tobytes())
        put(sd["model.embed_tokens.weight"])
        put(sd["model.norm.weight"])
        if not tied:
            put(sd["lm_head.weight"])
        for i in range(cfg.num_hidden_layers):
            p = f"model.layers.{i}."
            put(sd[p + "input_layernorm.weight"])
            put(torch.cat([sd[p + "self_attn.q_proj.weight"], sd[p + "self_attn.k_proj.weight"],
                           sd[p + "self_attn.v_proj.weight"]], 0))
            if has_bias:
                put(torch.cat([sd[p + "self_attn.q_proj.bias"], sd[p + "self_attn.k_proj.bias"],
                               sd[p + "self_attn.v_proj.bias"]], 0))
            put(sd[p + "self_attn.o_proj.weight"])
            put(sd[p + "post_attention_layernorm.weight"])
            put(interleave_gate_up(sd[p + "mlp.gate_proj.weight"], sd[p + "mlp.up_proj.weight"]))
            put(sd[p + "mlp.down_proj.weight"])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
