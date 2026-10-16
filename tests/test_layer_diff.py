"""Tests for scripts/layer_diff.py on the CPU (no GPU, no engine build): the HF capture is
self-consistent, fp16 rounding is not reported as a divergence, and a fault injected into any one
sub-step of any layer is reported at exactly that layer and sub-step, for a GQA Llama and a
Qwen2 (QKV bias) model built from random weights, plus hf-internal-testing's tiny Llama when the
Hub is reachable. Also checks the engine dump layout and the weight-layout facts the engine-side
comparison relies on. Run: python tests/test_layer_diff.py"""
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import layer_diff as ld  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def tiny(kind, seed=0, dtype=torch.float32, attn="eager"):
    from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    common = dict(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                  num_key_value_heads=2, head_dim=16, max_position_embeddings=128, rms_norm_eps=1e-6,
                  initializer_range=0.1, tie_word_embeddings=False, attn_implementation=attn)
    if kind == "llama":
        m = LlamaForCausalLM(LlamaConfig(**common))
    else:
        m = Qwen2ForCausalLM(Qwen2Config(**common, use_sliding_window=False))
        with torch.no_grad():  # HF initializes biases to zero; real Qwen2 QKV biases are not
            for n, p in m.named_parameters():
                if n.endswith(".bias"):
                    p.normal_(0, 0.1)
    return m.to(dtype).eval()


def clone(m, dtype=torch.float32, attn="eager"):
    """Same weights as m, in another precision / attention implementation."""
    c = tiny("qwen" if "Qwen" in type(m).__name__ else "llama", dtype=torch.float32, attn=attn)
    c.load_state_dict(m.state_dict())
    return c.to(dtype).eval()


IDS = torch.tensor([1, 17, 99, 3, 250, 42, 7, 7, 128, 64, 5, 200, 31, 9, 77, 160])


def test_self_consistency():
    m = tiny("llama")
    a, b = ld.capture_hf(m, IDS), ld.capture_hf(m, IDS)
    dims = ld.model_dims(m.config)
    check("capture records every point in order", list(a) == ld.point_names(dims["layers"]))
    check("capture shapes", all(a[n].shape == (len(IDS), ld.point_width(dims, n)) for n in a))
    rows = ld.compare(a, b, dims)
    check("identical runs: zero error, no divergence", all(r["rel"] == 0 for r in rows) and ld.first_divergence(rows) is None)


def test_capture_semantics():
    """The recorded points mean what the engine's do: qkv / gate_up from the exported weight
    layout, rope consumed by attention, residual stream at o_proj / down."""
    import export
    for kind in ("llama", "qwen"):
        m = tiny(kind)
        dims = ld.model_dims(m.config)
        a = ld.capture_hf(m, IDS)
        T, D, Hq, Hkv = len(IDS), dims["head_dim"], dims["heads"], dims["kv_heads"]
        sd = m.state_dict()
        errs = {}
        for l in range(dims["layers"]):
            p, q = f"layers.{l}.", f"model.layers.{l}."
            wqkv = torch.cat([sd[q + f"self_attn.{x}_proj.weight"] for x in "qkv"])
            qkv = a[p + "attn_norm"] @ wqkv.T
            if (q + "self_attn.q_proj.bias") in sd:
                qkv = qkv + torch.cat([sd[q + f"self_attn.{x}_proj.bias"] for x in "qkv"])
            errs["qkv = attn_norm @ [q;k;v]^T (+ bias)"] = (qkv - a[p + "qkv"]).abs().max()
            # SwiGLU over the export's interleaved gate/up rows, paired the way the engine's epilogue pairs them
            wgu = export.interleave_gate_up(sd[q + "mlp.gate_proj.weight"], sd[q + "mlp.up_proj.weight"])
            y = (a[p + "mlp_norm"] @ wgu.T).reshape(T, -1, 2, 16)
            act = (torch.nn.functional.silu(y[:, :, 0]) * y[:, :, 1]).reshape(T, -1)
            errs["gate_up = SwiGLU of interleaved wgu, unit order"] = (act - a[p + "gate_up"]).abs().max()
            # causal GQA attention over the rope point reproduces the attn point
            rq = a[p + "rope"][:, :Hq * D].reshape(T, Hq, D).transpose(0, 1)
            rk = a[p + "rope"][:, Hq * D:].reshape(T, Hkv, D).transpose(0, 1).repeat_interleave(Hq // Hkv, 0)
            v = a[p + "qkv"][:, (Hq + Hkv) * D:].reshape(T, Hkv, D).transpose(0, 1).repeat_interleave(Hq // Hkv, 0)
            att = torch.nn.functional.scaled_dot_product_attention(rq, rk, v, is_causal=True)
            errs["attn = causal SDPA(rope q, rope k, v)"] = (att.transpose(0, 1).reshape(T, -1) - a[p + "attn"]).abs().max()
            o = a[p + "attn"] @ sd[q + "self_attn.o_proj.weight"].T
            prev = a["embed"] if l == 0 else a[f"layers.{l - 1}.down"]
            errs["o_proj = residual + attn @ wo^T"] = (prev + o - a[p + "o_proj"]).abs().max()
            dn = a[p + "gate_up"] @ sd[q + "mlp.down_proj.weight"].T
            errs["down = o_proj + gate_up @ wdown^T"] = (a[p + "o_proj"] + dn - a[p + "down"]).abs().max()
        for what, e in errs.items():
            check(f"{kind}: {what}", e < 1e-4, f"max err {e:.1e}")


def test_rope_matches_hf_internal():
    """The recomputed rope point equals what HF's own apply_rotary_pos_emb returned."""
    from transformers.models.llama import modeling_llama as ml
    m = tiny("llama")
    seen, orig = [], ml.apply_rotary_pos_emb

    def spy(q, k, cos, sin, *a, **kw):
        out = orig(q, k, cos, sin, *a, **kw)
        seen.append(torch.cat([t[0].transpose(0, 1).reshape(q.shape[2], -1) for t in out], -1))
        return out
    ml.apply_rotary_pos_emb = spy
    try:
        a = ld.capture_hf(m, IDS)
    finally:
        ml.apply_rotary_pos_emb = orig
    if not seen:
        print("SKIP rope vs HF internal: this transformers version does not call the module-level function")
        return
    err = max((seen[l] - a[f"layers.{l}.rope"]).abs().max().item() for l in range(len(seen)))
    check("rope point == HF apply_rotary_pos_emb output", len(seen) == 4 and err == 0, f"max err {err:.1e}")


def test_fp16_not_flagged():
    for kind in ("llama", "qwen"):
        m = tiny(kind)
        ref = ld.capture_hf(m, IDS)
        dims = ld.model_dims(m.config)
        fp16 = ld.capture_hf(clone(m, torch.float16), IDS)
        rows = ld.compare(ref, fp16, dims)
        check(f"{kind}: fp16 vs fp32 within the fixed tolerance", ld.first_divergence(rows) is None,
              f"max rel {max(r['rel_max'] for r in rows):.1e}")
        # a different fp16 implementation (SDPA attention) against the eager fp16 noise floor
        floor = ld.noise_floor(ref, fp16, dims)
        sdpa = ld.capture_hf(clone(m, torch.float16, attn="sdpa"), IDS)
        rows = ld.compare(ref, sdpa, dims, floor=floor)
        worst = max(rows, key=lambda r: r["ratio"])
        check(f"{kind}: fp16 SDPA within the measured fp16 noise floor", ld.first_divergence(rows) is None,
              f"closest {worst['point']} {worst['rel']:.1e} vs {worst['thr']:.1e}")
        assert dims["layers"] == 4


def run_fault(m, ref, floor, layer, step, cand_dtype, scale=0.1):
    c = clone(m, cand_dtype)
    hs = ld.inject_fault(c, layer, step, scale)
    cand = ld.capture_hf(c, IDS)
    for h in hs:
        h.remove()
    rows = ld.compare(ref, cand, ld.model_dims(m.config), floor=floor)
    first = ld.first_divergence(rows)
    return first and first["point"], rows, cand


def test_injected_faults():
    """Every sub-step of a middle layer (and embed / final_norm / logits), injected into an fp16
    candidate, is named exactly; so is a small (5%) fault, and the same faults into fp32. (2% sits
    right at the default threshold, 8x the fp16 noise floor, and is not reliably flagged.)"""
    for kind in ("llama", "qwen"):
        m = tiny(kind)
        ref = ld.capture_hf(m, IDS)
        floor = ld.noise_floor(ref, ld.capture_hf(clone(m, torch.float16), IDS), ld.model_dims(m.config))
        cases = [(2, s) for s in ld.STEPS] + [(0, s) for s in ld.STEPS[:3]] + [(3, "down"), (0, "embed"),
                                                                                (0, "final_norm"), (0, "logits")]
        for layer, step in cases:
            want = step if step in ("embed", "final_norm", "logits") else f"layers.{layer}.{step}"
            for dt, scale in ((torch.float16, 0.1), (torch.float16, 0.05), (torch.float32, 0.1)):
                if step == "rope" and scale != 0.1:
                    continue  # the rope fault is a sign flip, not a scaled perturbation
                got, rows, _ = run_fault(m, ref, floor, layer, step, dt, scale)
                check(f"{kind}: fault {want} ({str(dt)[6:]}, scale {scale}) found at {got}", got == want)


def test_rope_localization():
    m = tiny("llama")
    ref = ld.capture_hf(m, IDS)
    got, rows, cand = run_fault(m, ref, None, 1, "rope", torch.float32)
    loc = ld.localize(ref, cand, "layers.1.rope", ld.model_dims(m.config))
    check("rope sign fault: position 0 clean, later positions off",
          loc["position0_rel"] < 1e-6 and loc["positions"][0][1] > 0.1, str(loc["positions"][:2]))
    got, rows, cand = run_fault(m, ref, None, 2, "qkv", torch.float32)
    loc = ld.localize(ref, cand, "layers.2.qkv", ld.model_dims(m.config))
    check("k-projection fault: worst heads are k heads", all(h.startswith("k") for h, _ in loc["heads"][:2]),
          str(loc["heads"]))
    report = ld.format_report(rows, ref, cand, ld.model_dims(m.config))
    check("report names the layer and sub-step", "FIRST DIVERGENCE: layer 2, qkv" in report)


def test_weight_faults():
    """Real weight corruption (a bad export / loader) is attributed to the sub-step that reads it."""
    cases = {"model.layers.1.self_attn.k_proj.weight": "layers.1.qkv",
             "model.layers.2.self_attn.v_proj.bias": "layers.2.qkv",  # zeroed: a loader that drops the bias
             "model.layers.1.self_attn.o_proj.weight": "layers.1.o_proj",
             "model.layers.0.post_attention_layernorm.weight": "layers.0.mlp_norm",
             "model.layers.3.mlp.up_proj.weight": "layers.3.gate_up",
             "model.layers.2.mlp.down_proj.weight": "layers.2.down",
             "model.layers.3.input_layernorm.weight": "layers.3.attn_norm",
             "model.norm.weight": "final_norm",
             "lm_head.weight": "logits"}
    m = tiny("qwen")
    ref = ld.capture_hf(m, IDS)
    floor = ld.noise_floor(ref, ld.capture_hf(clone(m, torch.float16), IDS), ld.model_dims(m.config))
    for wname, want in cases.items():
        c = clone(m)
        with torch.no_grad():
            w = c.get_parameter(wname)
            if wname.endswith(".bias"):
                w.zero_()
            else:
                w.add_(0.05 * w.abs().mean() * torch.randn_like(w))
        rows = ld.compare(ref, ld.capture_hf(c.half(), IDS), ld.model_dims(m.config), floor=floor)
        got = ld.first_divergence(rows)
        check(f"weight fault {wname} -> {want}", got is not None and got["point"] == want, got and got["point"])


def test_engine_layout():
    m = tiny("qwen")
    dims = ld.model_dims(m.config)
    a = ld.capture_hf(m, IDS)
    T = len(IDS)
    lay = ld.engine_layout(dims, T)
    check("layout covers every point in order", [n for n, _, _ in lay] == ld.point_names(dims["layers"]))
    check("layout planes are contiguous", all(o2 == o1 + T * w1 for (_, o1, w1), (_, o2, _) in zip(lay, lay[1:])))
    # hi_debug_bytes in src/api.cu, written out independently
    H, I, V, L, D, Hq, Hkv = (dims[k] for k in ("hidden", "inter", "vocab", "layers", "head_dim", "heads", "kv_heads"))
    per_layer = H + (Hq + 2 * Hkv) * D + (Hq + Hkv) * D + Hq * D + H + H + I + H
    check("dump size matches the C formula", ld.engine_dump_elems(dims, T) == T * (H + L * per_layer + H + V))
    back = ld.split_engine_dump(ld.pack_engine_dump(a, dims, T), dims, T)
    err = max((back[n] - a[n].half().float()).abs().max().item() for n in a)
    check("pack -> split round trip", list(back) == list(a) and err == 0)


def test_real_model():
    """Opt in with LAYER_DIFF_REAL_MODEL=<HF id> (e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0; a few
    minutes on a CPU): real checkpoints have a BOS row with activations ~100x the rest, which the
    random models above lack. A second fp16 implementation (SDPA) must stay under the thresholds
    measured from eager fp16, and 5% faults deep in the model must still be named exactly."""
    model_id = os.environ.get("LAYER_DIFF_REAL_MODEL")
    if not model_id:
        print("SKIP real-model test (set LAYER_DIFF_REAL_MODEL to an HF model id)")
        return
    from transformers import AutoTokenizer
    ids = torch.tensor(AutoTokenizer.from_pretrained(model_id)(ld.DEFAULT_PROMPT).input_ids)
    ref = ld.hf_run(model_id, "fp32", "cpu", ids)
    m = ld.load_hf(model_id, "fp16", "cpu")
    dims = ld.model_dims(m.config)
    floor = ld.noise_floor(ref, ld.capture_hf(m, ids), dims)
    L = dims["layers"]
    for layer, step in [(L // 2, "o_proj"), (L // 4, "down"), (L - 2, "attn"), (L // 3, "qkv"), (L // 2, "gate_up")]:
        hs = ld.inject_fault(m, layer, step, 0.05)
        first = ld.first_divergence(ld.compare(ref, ld.capture_hf(m, ids), dims, floor=floor))
        for h in hs:
            h.remove()
        check(f"{model_id}: 5% fault layers.{layer}.{step} found at {first and first['point']}",
              first is not None and first["point"] == f"layers.{layer}.{step}")
    m.config._attn_implementation = "sdpa"
    rows = ld.compare(ref, ld.capture_hf(m, ids), dims, floor=floor)
    worst = max(rows, key=lambda r: r["ratio"])
    check(f"{model_id}: fp16 SDPA within the eager-fp16 thresholds", ld.first_divergence(rows) is None,
          f"closest {worst['point']} at {worst['ratio']:.2f} of its threshold")


def test_cli_tiny_random():
    """The CLI end to end on hf-internal-testing's tiny Llama (skipped without the Hub)."""
    env = dict(os.environ, TRANSFORMERS_VERBOSITY="error", HF_HUB_DISABLE_SYMLINKS_WARNING="1")
    cmd = [sys.executable, os.path.join(HERE, "..", "scripts", "layer_diff.py"), "--model",
           "hf-internal-testing/tiny-random-LlamaForCausalLM", "--candidate", "hf", "--device", "cpu"]
    r = subprocess.run(cmd + ["--inject", "1:attn"], capture_output=True, text=True, env=env)
    if r.returncode not in (0, 1) or "FIRST" not in r.stdout and "no divergence" not in r.stdout:
        print("SKIP CLI test (model not available?):", (r.stderr or r.stdout).strip().splitlines()[-1:])
        return
    check("CLI: injected 1:attn reported, exit 1", r.returncode == 1 and "FIRST DIVERGENCE: layer 1, attn" in r.stdout)
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    check("CLI: clean fp16 run, exit 0", r.returncode == 0 and "no divergence" in r.stdout, r.stdout[-200:])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        print(f"== {t.__name__}", flush=True)
        t()
    print(f"\n{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
    sys.exit(1 if FAILS else 0)
