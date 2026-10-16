"""Layer-by-layer activation diff: hipinfer (or a second HF run) vs Hugging Face transformers.

tests/gate.py and tests/ppl.py say *whether* the engine matches HF; this says *where* it stops
matching. Both sides record the same points for every prompt row, in execution order:

  embed
  layers.<l>.attn_norm   input RMSNorm output                         [T, hidden]
  layers.<l>.qkv         q | k | v projections (+ bias), before RoPE    [T, (Hq + 2 Hkv) D]
  layers.<l>.rope        q | k after RoPE                              [T, (Hq + Hkv) D]
  layers.<l>.attn        attention output, the O projection's input    [T, Hq D]
  layers.<l>.o_proj      residual stream after the O projection (h + o) [T, hidden]
  layers.<l>.mlp_norm    post-attention RMSNorm output                 [T, hidden]
  layers.<l>.gate_up     silu(gate) * up, the down projection's input   [T, inter]
  layers.<l>.down        residual stream after the MLP (layer output)   [T, hidden]
  final_norm, logits                                                   [T, hidden], [T, vocab]

o_proj and down are residual-stream values because the engine fuses the residual add into
those GEMMs' epilogues (and gate_up is post-SwiGLU because the engine fuses that too).

The report is a per-layer grid of relative L2 errors, ||cand - ref|| / ||ref||, and the first
point in execution order whose error passes its threshold, with the worst token positions and
heads there. fp16 rounding grows smoothly through the layers; a bug shows up as a step. The
threshold is either a fixed --tol, or (by default when the reference is fp32) measured: HF is
also run in fp16, and a point diverges when the candidate's error exceeds --factor times HF
fp16's own error there (and --min-tol).

GPU (engine vs HF; see docs/layer-diff.md):
  python scripts/layer_diff.py --model unsloth/Llama-3.2-1B --weights llama32-1b.bin
CPU demo (HF fp16 with an injected fault vs HF fp32; no GPU needed):
  python scripts/layer_diff.py --model hf-internal-testing/tiny-random-LlamaForCausalLM \
      --candidate hf --device cpu --inject 1:rope
"""
import argparse
import json
import os
import sys
from collections import OrderedDict

import torch

STEPS = ["attn_norm", "qkv", "rope", "attn", "o_proj", "mlp_norm", "gate_up", "down"]
DEFAULT_PROMPT = "The capital of France is Paris. The quick brown fox jumps over the lazy dog, and then"


# ------------------------------------------------------------------ shapes and the engine's dump layout
def model_dims(cfg):
    """Shape of a HF config, with the key names of scripts/hipinfer.py read_config."""
    heads = cfg.num_attention_heads
    return {"vocab": cfg.vocab_size, "hidden": cfg.hidden_size, "inter": cfg.intermediate_size,
            "layers": cfg.num_hidden_layers, "heads": heads,
            "kv_heads": getattr(cfg, "num_key_value_heads", None) or heads,
            "head_dim": getattr(cfg, "head_dim", None) or cfg.hidden_size // heads}


def point_width(dims, name):
    step = name.rsplit(".", 1)[-1]
    H, D, Hq, Hkv = dims["hidden"], dims["head_dim"], dims["heads"], dims["kv_heads"]
    return {"embed": H, "attn_norm": H, "qkv": (Hq + 2 * Hkv) * D, "rope": (Hq + Hkv) * D, "attn": Hq * D,
            "o_proj": H, "mlp_norm": H, "gate_up": dims["inter"], "down": H, "final_norm": H,
            "logits": dims["vocab"]}[step]


def point_names(n_layers):
    return ["embed"] + [f"layers.{l}.{s}" for l in range(n_layers) for s in STEPS] + ["final_norm", "logits"]


def head_labels(dims, name):
    """Per-head column groups of a point ([label], width), or None for hidden-sized points."""
    step = name.rsplit(".", 1)[-1]
    Hq, Hkv = dims["heads"], dims["kv_heads"]
    labels = {"qkv": [f"q{h}" for h in range(Hq)] + [f"k{h}" for h in range(Hkv)] + [f"v{h}" for h in range(Hkv)],
              "rope": [f"q{h}" for h in range(Hq)] + [f"k{h}" for h in range(Hkv)],
              "attn": [f"h{h}" for h in range(Hq)]}.get(step)
    return labels


def engine_layout(dims, T):
    """(name, offset, width) of every plane in the buffer hi_prefill_debug fills: fp16 [T, width]
    planes back to back, in point_names order (src/api.cu dbg_plane mirrors this)."""
    out, off = [], 0
    for name in point_names(dims["layers"]):
        w = point_width(dims, name)
        out.append((name, off, w))
        off += T * w
    return out


def engine_dump_elems(dims, T):
    return sum(T * w for _, _, w in engine_layout(dims, T))


def split_engine_dump(flat, dims, T):
    """Flat hi_prefill_debug buffer -> {point: [T, width] float32}."""
    flat = flat.reshape(-1)
    assert flat.numel() == engine_dump_elems(dims, T), (flat.numel(), engine_dump_elems(dims, T))
    return OrderedDict((name, flat[off:off + T * w].reshape(T, w).float()) for name, off, w in engine_layout(dims, T))


def pack_engine_dump(acts, dims, T):
    """Inverse of split_engine_dump (fp16), for testing the layout without a GPU."""
    return torch.cat([acts[name].reshape(T, w).half().reshape(-1) for name, _, w in engine_layout(dims, T)])


# ------------------------------------------------------------------ HF capture
def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _first(x):
    return x[0] if isinstance(x, (tuple, list)) else x


def _decoder(model):
    return model.model


@torch.no_grad()
def capture_hf(model, input_ids):
    """Run a Llama-family HF causal LM on one sequence and record every point (float32, CPU).

    RoPE has no module to hook, so the rope point is recomputed from the recorded q/k projections
    and the (cos, sin) the attention module received, with HF's own formula (rotate_half). If a
    fault-injection hook alters those inputs, it must run first: inject_fault prepends its hooks."""
    dec = _decoder(model)
    dims = model_dims(model.config)
    T = input_ids.shape[-1]
    acts, tmp, hooks = OrderedDict(), {}, []

    def put(name, t):
        acts[name] = t.detach().reshape(T, -1).float().cpu()

    def fwd(mod, name):
        hooks.append(mod.register_forward_hook(lambda m, a, o: put(name, _first(o))))

    def pre(mod, name):
        hooks.append(mod.register_forward_pre_hook(lambda m, a: put(name, a[0])))

    fwd(dec.embed_tokens, "embed")
    for l, layer in enumerate(dec.layers):
        p, at = f"layers.{l}.", layer.self_attn
        fwd(layer.input_layernorm, p + "attn_norm")
        for proj in ("q", "k", "v"):
            hooks.append(getattr(at, proj + "_proj").register_forward_hook(
                lambda m, a, o, key=(l, proj): tmp.__setitem__(key, o.detach())))
        # q/k norms (Qwen3-style), when present, sit between the projection and RoPE
        for proj in ("q", "k"):
            if getattr(at, proj + "_norm", None) is not None:
                hooks.append(getattr(at, proj + "_norm").register_forward_hook(
                    lambda m, a, o, key=(l, proj + "n"): tmp.__setitem__(key, o.detach())))

        def attn_pre(m, args, kwargs, l=l):
            tmp[(l, "pe")] = kwargs.get("position_embeddings")
        hooks.append(at.register_forward_pre_hook(attn_pre, with_kwargs=True))

        def attn_post(m, args, out, l=l, p=p):
            q, k, v = tmp.pop((l, "q")), tmp.pop((l, "k")), tmp.pop((l, "v"))
            put(p + "qkv", torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1)], -1))
            D = dims["head_dim"]
            qr = tmp.pop((l, "qn"), q).reshape(T, -1, D)
            kr = tmp.pop((l, "kn"), k).reshape(T, -1, D)
            cos, sin = tmp.pop((l, "pe"))
            cos, sin = cos.reshape(T, 1, D).to(qr.dtype), sin.reshape(T, 1, D).to(qr.dtype)
            put(p + "rope", torch.cat([(qr * cos + _rotate_half(qr) * sin).reshape(T, -1),
                                       (kr * cos + _rotate_half(kr) * sin).reshape(T, -1)], -1))
        hooks.append(at.register_forward_hook(attn_post))
        pre(at.o_proj, p + "attn")
        pre(layer.post_attention_layernorm, p + "o_proj")
        fwd(layer.post_attention_layernorm, p + "mlp_norm")
        pre(layer.mlp.down_proj, p + "gate_up")
        fwd(layer, p + "down")
    fwd(dec.norm, "final_norm")
    try:
        dev = next(model.parameters()).device
        out = model(input_ids.reshape(1, T).to(dev), use_cache=False)
        put("logits", out.logits)
    finally:
        for h in hooks:
            h.remove()
    # execution order (hooks fire in that order except rope/qkv, which complete after attn_norm)
    return OrderedDict((n, acts[n]) for n in point_names(dims["layers"]) if n in acts)


# ------------------------------------------------------------------ fault injection (testing the tool)
def _perturb(t, scale, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(t.shape, generator=g, dtype=torch.float32).to(t.device, t.dtype)
    rms = t.float().pow(2).mean().sqrt().clamp_min(1e-6).to(t.dtype)
    return t + scale * rms * noise


def inject_fault(model, layer, step, scale=0.1, seed=1234):
    """Corrupt one point of one layer the way a kernel bug would: that point's value is perturbed
    by `scale` relative noise (rope: sin is negated, the classic wrong-direction rotation, which
    leaves position 0 intact; qkv: only the k projection). Returns the hook handles. The hooks are
    prepended so capture_hf, registered later, records the corrupted values."""
    dec = _decoder(model)
    if step in ("embed", "final_norm", "logits"):
        mod = {"embed": dec.embed_tokens, "final_norm": dec.norm, "logits": model.lm_head}[step]
        return [mod.register_forward_hook(lambda m, a, o: _perturb(o, scale, seed), prepend=True)]
    L = dec.layers[layer]
    at = L.self_attn
    out_hook = {"attn_norm": L.input_layernorm, "qkv": at.k_proj, "o_proj": at.o_proj,
                "mlp_norm": L.post_attention_layernorm, "down": L.mlp.down_proj}
    if step in out_hook:
        return [out_hook[step].register_forward_hook(lambda m, a, o: _perturb(o, scale, seed), prepend=True)]
    if step in ("attn", "gate_up"):
        mod = at.o_proj if step == "attn" else L.mlp.down_proj
        return [mod.register_forward_pre_hook(lambda m, a: (_perturb(a[0], scale, seed),) + tuple(a[1:]), prepend=True)]
    if step == "rope":
        def flip(m, args, kwargs):
            cos, sin = kwargs["position_embeddings"]
            kwargs["position_embeddings"] = (cos, -sin)
            return args, kwargs
        return [at.register_forward_pre_hook(flip, with_kwargs=True, prepend=True)]
    raise ValueError(f"unknown step {step!r}")


# ------------------------------------------------------------------ comparison
def _view(acts, name):
    """The value a point is judged on. o_proj and down are residual-stream values; they are judged
    on their branch (the O / down projection's own output: the point minus the residual before
    it), which a projection bug moves by its full size rather than diluted by the residual."""
    v = acts[name].double()
    if name.endswith(".o_proj") or name.endswith(".down"):
        l = int(name.split(".")[1])
        prev = f"layers.{l}.o_proj" if name.endswith(".down") else ("embed" if l == 0 else f"layers.{l - 1}.down")
        if prev in acts:
            v = v - acts[prev].double()
    return v


def _group_rel(d, r, dims, name):
    """Relative L2 error of every (position, head) block of a head-shaped point, or of every
    position of the others. Each block is judged against its own reference norm (floored at 5% of
    the mean block norm), so neither a high-norm row (real models' BOS row carries activations
    100x the rest) nor a large q part can hide an error confined to a few positions or heads."""
    T = r.shape[0]
    n = len(head_labels(dims, name) or [None])
    dn, rn = d.reshape(T, n, -1).norm(dim=-1), r.reshape(T, n, -1).norm(dim=-1)
    return dn / rn.clamp_min(0.05 * rn.mean().item() + 1e-30)


def point_error(ref, cand, name, dims):
    """(worst block relative error, whole-point relative error, branch diff, branch ref)."""
    r, c = _view(ref, name), _view(cand, name)
    if c.shape != r.shape:
        raise ValueError(f"{name}: candidate shape {tuple(c.shape)} != reference {tuple(r.shape)}")
    d = c - r
    return _group_rel(d, r, dims, name).max().item(), (d.norm() / r.norm().clamp_min(1e-30)).item(), d, r


def compare(ref, cand, dims, tol=2e-2, floor=None, factor=8.0, min_tol=2e-3):
    """One row per point present in both: errors and the threshold it is held to. `rel` (the
    worst position / head block, see _group_rel) decides; `rel_all` is the whole-point error.

    Threshold: `tol`, or with a noise floor ({point: rel} of HF fp16 against the same reference)
    max(min_tol, factor * floor[point])."""
    rows = []
    for name in ref:
        if name not in cand:
            continue
        rel, rel_all, d, r = point_error(ref, cand, name, dims)
        cos = torch.nn.functional.cosine_similarity((r + d).reshape(1, -1), r.reshape(1, -1)).item()
        thr = tol if floor is None else max(min_tol, factor * floor.get(name, tol / factor))
        rows.append({"point": name, "rel": rel, "rel_all": rel_all, "max_abs": d.abs().max().item(), "cos": cos,
                     "ref_rms": r.pow(2).mean().sqrt().item(), "thr": thr,
                     "floor": None if floor is None else floor.get(name), "diverged": not (rel <= thr)})
    return rows


def noise_floor(ref, other, dims):
    """{point: rel} of another run (e.g. HF fp16) against the reference, in compare's metric."""
    return {n: point_error(ref, other, n, dims)[0] for n in ref if n in other}


def first_divergence(rows):
    return next((r for r in rows if r["diverged"]), None)


def localize(ref, cand, name, dims, top=4):
    """Where inside one point the error sits: worst token positions and (for head-shaped points) heads."""
    _, _, d, r = point_error(ref, cand, name, dims)
    g = _group_rel(d, r, dims, name)  # [T, heads or 1]
    row = g.max(dim=1).values
    out = {"positions": [(int(i), row[i].item()) for i in row.argsort(descending=True)[:top]],
           "position0_rel": row[0].item()}
    labels = head_labels(dims, name)
    if labels:
        hr = g.max(dim=0).values
        out["heads"] = [(labels[i], hr[i].item()) for i in hr.argsort(descending=True)[:top]]
        out["heads_over_half_worst"] = int((hr > 0.5 * hr.max()).sum().item())
        out["n_heads"] = len(labels)
    return out


def _fmt(x):
    return f"{x:.1e}" if x >= 1e-9 else "0"


def format_report(rows, ref, cand, dims, title=""):
    by = {r["point"]: r for r in rows}
    first = first_divergence(rows)
    lines = [title] if title else []
    floor_note = " (threshold = max(min_tol, factor x HF-fp16 noise floor))" if rows and rows[0]["floor"] is not None else ""
    lines.append("worst relative L2 error over positions (and heads) per point; o_proj / down judged on the projection's "
                 f"output, without the residual\n'!' over threshold, '*' first divergence{floor_note}")

    def cell(name):
        r = by.get(name)
        if r is None:
            return "-"
        mark = "*" if first is not None and r["point"] == first["point"] else ("!" if r["diverged"] else " ")
        return _fmt(r["rel"]) + mark

    w = 10
    lines.append("layer " + "".join(s.rjust(w) for s in STEPS))
    lines.append(f"embed {cell('embed').rjust(w)}")
    for l in range(dims["layers"]):
        lines.append(f"{l:5d} " + "".join(cell(f"layers.{l}.{s}").rjust(w) for s in STEPS))
    lines.append(f"final_norm {cell('final_norm')}   logits {cell('logits')}")
    if first is None:
        worst = max(rows, key=lambda r: r["rel"] / r["thr"]) if rows else None
        lines.append("no divergence: every point is within its threshold" +
                     (f" (closest: {worst['point']} rel {_fmt(worst['rel'])} vs {_fmt(worst['thr'])})" if worst else ""))
        return "\n".join(lines)
    n_after = sum(r["diverged"] for r in rows) - 1
    p = first["point"]
    where = f"layer {p.split('.')[1]}, {p.split('.')[2]}" if p.startswith("layers.") else p
    lines.append("")
    lines.append(f"FIRST DIVERGENCE: {where}  ({p})")
    lines.append(f"  rel {_fmt(first['rel'])} > threshold {_fmt(first['thr'])}"
                 + (f" (HF fp16 floor {_fmt(first['floor'])})" if first["floor"] is not None else "")
                 + f"; max |diff| {first['max_abs']:.3g} (ref rms {first['ref_rms']:.3g}); cosine {first['cos']:.6f}")
    idx = [r["point"] for r in rows].index(p)
    if idx > 0:
        prev = rows[idx - 1]
        lines.append(f"  previous point {prev['point']}: rel {_fmt(prev['rel'])} (threshold {_fmt(prev['thr'])})")
    loc = localize(ref, cand, p, dims)
    lines.append("  worst positions: " + ", ".join(f"t={t} {_fmt(e)}" for t, e in loc["positions"])
                 + f"; position 0: {_fmt(loc['position0_rel'])}")
    if "heads" in loc:
        lines.append("  worst heads: " + ", ".join(f"{h} {_fmt(e)}" for h, e in loc["heads"])
                     + f" ({loc['heads_over_half_worst']} of {loc['n_heads']} heads within 2x of the worst)")
    if p.endswith(".rope") and loc["position0_rel"] < 0.1 * first["thr"]:
        lines.append("  position 0 matches (cos 1, sin 0 there): the RoPE angles / sign / frequencies, not the inputs")
    lines.append(f"  {n_after} later point(s) also over threshold (errors propagate downstream)")
    return "\n".join(lines)


def report_json(rows, ref, cand, dims):
    first = first_divergence(rows)
    out = {"first_divergence": first and first["point"], "points": rows}
    if first:
        out["localization"] = localize(ref, cand, first["point"], dims)
    return out


# ------------------------------------------------------------------ CLI
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_hf(model_id, dtype, device):
    from transformers import AutoModelForCausalLM
    # eager attention: the reference should be the plain math, not a fused kernel's
    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=DTYPES[dtype], attn_implementation="eager")
    return m.to(device).eval()


def hf_run(model_id, dtype, device, ids, inject=()):
    m = load_hf(model_id, dtype, device)
    handles = [h for (l, s, sc) in inject for h in inject_fault(m, l, s, sc)]
    acts = capture_hf(m, ids)
    for h in handles:
        h.remove()
    del m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return acts


def engine_run(weights, ids, dims):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import hipinfer as hi
    T = len(ids)
    eng = hi.Engine(weights, max_batch=1, max_seq=T + 8, max_prefill=T)
    try:
        got = {k: getattr(eng, a) for k, a in [("vocab", "vocab"), ("hidden", "hidden"), ("inter", "inter"),
                                                ("layers", "layers"), ("heads", "heads"), ("kv_heads", "kv_heads"),
                                                ("head_dim", "head_dim")]}
        if got != dims:
            raise SystemExit(f"weight file {weights} does not match the HF model: {got} vs {dims}")
        flat = eng.prefill_debug(0, ids)
        return split_engine_dump(flat.cpu(), dims, T)
    finally:
        eng.close()


def parse_inject(s):
    parts = s.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("--inject LAYER:STEP[:SCALE], e.g. 3:rope or 0:qkv:0.05 (LAYER ignored for embed/final_norm/logits)")
    return int(parts[0]), parts[1], float(parts[2]) if len(parts) == 3 else 0.1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HF model id or path")
    ap.add_argument("--weights", help="hipinfer weight file (scripts/export.py) for --candidate engine")
    ap.add_argument("--candidate", choices=["engine", "hf"], default="engine",
                    help="engine: hipinfer's hi_prefill_debug; hf: HF itself in --cand-dtype (with --inject faults)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--tokens", help="comma-separated token ids instead of --prompt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device for the HF runs")
    ap.add_argument("--ref-dtype", choices=DTYPES, default="fp32", help="HF reference precision")
    ap.add_argument("--cand-dtype", choices=DTYPES, default="fp16", help="--candidate hf precision")
    ap.add_argument("--floor", choices=["auto", "on", "off"], default="auto",
                    help="measure the HF fp16 noise floor for per-point thresholds (auto: when --ref-dtype is fp32)")
    ap.add_argument("--tol", type=float, default=2e-2, help="fixed relative-error threshold (without a noise floor)")
    ap.add_argument("--factor", type=float, default=8.0, help="threshold = factor x noise floor ...")
    ap.add_argument("--min-tol", type=float, default=2e-3, help="... but at least this")
    ap.add_argument("--inject", type=parse_inject, action="append", default=[],
                    help="LAYER:STEP[:SCALE] fault in the candidate (--candidate hf only)")
    ap.add_argument("--save-ref", help="save the HF reference capture (torch.save) for reuse")
    ap.add_argument("--load-ref", help="reuse a saved HF reference capture (skips the reference HF run)")
    ap.add_argument("--json", help="write the full per-point table and localization here")
    a = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    dims = model_dims(AutoConfig.from_pretrained(a.model))
    if a.tokens:
        ids = [int(t) for t in a.tokens.split(",")]
    else:
        ids = AutoTokenizer.from_pretrained(a.model)(a.prompt).input_ids
    T = len(ids)
    t_ids = torch.tensor(ids)
    print(f"{a.model}: {dims['layers']} layers, hidden {dims['hidden']}, heads {dims['heads']}/{dims['kv_heads']} "
          f"x {dims['head_dim']}, {T} prompt tokens", flush=True)

    if a.load_ref:
        saved = torch.load(a.load_ref)
        if saved["ids"] != ids:
            raise SystemExit(f"{a.load_ref} was captured for different tokens")
        ref = saved["acts"]
    else:
        ref = hf_run(a.model, a.ref_dtype, a.device, t_ids)
        if a.save_ref:
            torch.save({"model": a.model, "dtype": a.ref_dtype, "ids": ids, "acts": ref}, a.save_ref)
    floor = None
    if a.floor == "on" or (a.floor == "auto" and a.ref_dtype == "fp32"):
        floor = noise_floor(ref, hf_run(a.model, "fp16", a.device, t_ids), dims)

    if a.candidate == "engine":
        if not a.weights:
            raise SystemExit("--candidate engine needs --weights (scripts/export.py output)")
        if a.inject:
            raise SystemExit("--inject applies to --candidate hf")
        cand = engine_run(a.weights, ids, dims)
        title = f"hipinfer ({a.weights}) vs HF {a.ref_dtype}"
    else:
        cand = hf_run(a.model, a.cand_dtype, a.device, t_ids, a.inject)
        title = f"HF {a.cand_dtype}" + (f" with faults {a.inject}" if a.inject else "") + f" vs HF {a.ref_dtype}"

    rows = compare(ref, cand, dims, tol=a.tol, floor=floor, factor=a.factor, min_tol=a.min_tol)
    print(format_report(rows, ref, cand, dims, title=title))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        with open(a.json, "w") as f:
            json.dump({"model": a.model, "tokens": ids, "candidate": a.candidate,
                       **report_json(rows, ref, cand, dims)}, f, indent=1)
    sys.exit(1 if first_divergence(rows) else 0)


if __name__ == "__main__":
    main()
