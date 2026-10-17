"""Router top-k trace for an MoE model: hipinfer vs HF transformers along one greedy generation.

tests/gate.py fails a run when the engine's greedy tokens leave HF's at a step where HF's top-2
logit gap is above the near-tie tolerance. An MoE model makes a second kind of discrete choice
that rounding can flip: the router's top-k. This script records both sides' top-k expert sets at
every (position, layer) and the margin between the k-th and (k+1)-th router logit:

  engine: the router input (mlp_norm output) of every MoE layer, from hi_prefill_debug for the
          prompt and hi_decode_debug for each decode step (no graph; the same kernels as the
          gate's runs), routed with hi_moe_router, the engine's own kernel
  HF:     forward hooks on the router modules, along the same greedy loop as tests/gate.py

It then explains the first token divergence (router_tie_check, which tests/gate.py also uses):
HF is re-run on the engine's tokens, and wherever HF's expert set differs from the engine's, the
engine's set is adopted if HF's own margin there is a near-tie (<= router_tie), one point at a
time in execution order, so each margin is measured on an HF run that shares the engine's routing
up to that point. Adopted points get the engine's experts with HF's own logits as weights. The
divergence is explained if this HF run then gives the engine's token at the divergence step (or
HF's top-2 gap there is a token near-tie). --all-at-once also reports HF with the engine's sets
at every differing point at once, and a control run that puts HF's own sets back through the
same hook (it must reproduce HF's tokens).

python scripts/moe_router_trace.py --model ibm-granite/granite-3.1-1b-a400m-base \
    --weights granite-moe-1b.bin --prompt 1 [--new 128] [--router-tie 0.01] [--out trace.json]
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from layer_diff import _perturb, model_dims, split_engine_dump  # noqa: E402


def fp16_ulp(v):
    """Spacing of fp16 values at |v| (normal range)."""
    return 2.0 ** (math.floor(math.log2(max(abs(v), 2.0 ** -14))) - 10)


def topk_margin(row, k):
    """Top-k experts of one logit row (ties to the lower index, like the engine's kernel), and the
    margin between the k-th and (k+1)-th logit."""
    order = sorted(range(len(row)), key=lambda e: (-row[e], e))
    return frozenset(order[:k]), row[order[k - 1]] - row[order[k]]


class HFRouters:
    """Hooks on every router module: record each row's router logits by absolute position, and
    replace the routing at the (position, layer) points in `force`. The router modules return
    (top_k_index, top_k_weights, router_logits) (Granite MoE, transformers 5.x). A forced point
    gets the given experts, ordered by HF's logits, with weights from HF's own logits computed as
    the module does (softmax over the k, cast to the weights' dtype)."""

    def __init__(self, model, E, k):
        self.mods = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
        self.E, self.k, self.pos0 = E, k, 0
        self.logits, self.force = {}, {}
        self.hooks = [m.register_forward_hook(self._hook(l)) for l, m in enumerate(self.mods)]

    def _hook(self, l):
        def hook(mod, args, out):
            idx, w, logits = out
            rows = logits.reshape(-1, self.E)
            host = rows.float().cpu()
            for r in range(rows.shape[0]):
                self.logits[(self.pos0 + r, l)] = host[r].tolist()
            pts = [(r, self.force[(self.pos0 + r, l)]) for r in range(rows.shape[0]) if (self.pos0 + r, l) in self.force]
            if not pts:
                return None
            idx, w = idx.clone(), w.clone()
            fi, fw = idx.reshape(-1, self.k), w.reshape(-1, self.k)
            for r, experts in pts:
                sel = torch.tensor(sorted(experts, key=lambda e: (-host[r, e].item(), e)), device=idx.device)
                fi[r] = sel.to(idx.dtype)
                fw[r] = torch.softmax(rows[r].gather(-1, sel), dim=-1).to(w.dtype)
            return idx, w, logits
        return hook

    def close(self):
        for h in self.hooks:
            h.remove()


def hf_run(model, routers, ids, n, feed=None):
    """tests/gate.py hf_greedy (n tokens, per-step argmax and top-2 gap), with the router hooks told
    each forward's first position. With `feed`, the next input is feed[i] instead of HF's argmax
    (teacher forcing on the engine's tokens)."""
    out, gaps = [], []
    routers.logits = {}
    with torch.no_grad():
        routers.pos0 = 0
        o = model(torch.tensor([ids], device="cuda"), use_cache=True)
        past, logits = o.past_key_values, o.logits[0, -1]
        for i in range(n):
            top = logits.float().topk(2)
            gaps.append((top.values[0] - top.values[1]).item())
            out.append(int(logits.argmax()))
            if i == n - 1:
                break
            routers.pos0 = len(ids) + i
            t = feed[i] if feed is not None else out[-1]
            o = model(torch.tensor([[t]], device="cuda"), past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[0, -1]
    return out, gaps


def engine_trace(eng, dims, prompts, n):
    """The engine's greedy run of n tokens for each prompt, slot b = prompt b, decoded together as
    one batch of len(prompts) (one prompt: tests/gate.py's single-sequence run; six: its batched
    run). Returns per prompt (tokens, xn) with xn[l] the [len + n - 1, hidden] fp16 router inputs
    of layer l. Long prompts are prefilled in max_prefill chunks, as hi_prefill does."""
    L, B = dims["layers"], len(prompts)
    xn = [[[] for _ in range(L)] for _ in prompts]
    for b, ids in enumerate(prompts):
        for c0 in range(0, len(ids), eng.max_prefill):
            chunk = ids[c0:c0 + eng.max_prefill]
            acts = split_engine_dump(eng.prefill_debug(b, chunk, p0=c0), dims, len(chunk))
            for l in range(L):
                xn[b][l].append(acts[f"layers.{l}.mlp_norm"])
    for _ in range(n - 1):
        acts = split_engine_dump(eng.decode_debug(B), dims, B)
        for l in range(L):
            for b in range(B):
                xn[b][l].append(acts[f"layers.{l}.mlp_norm"][b:b + 1])
    return [(eng.history(b, len(ids), n), [torch.cat(x).half().to(eng.device) for x in xn[b]])
            for b, ids in enumerate(prompts)]


def engine_routing(hi, eng, routers, xn):
    """{(position, layer): (expert set, k-th minus (k+1)-th margin, logit row)} of the engine. The
    set comes from hi_moe_router; the logits are recomputed in torch (fp32 accumulate, rounded to
    fp16 as the kernel does) for the margins, and the set they give is checked against the kernel's."""
    k, out, mismatch = eng.top_k, {}, 0
    for l, X in enumerate(xn):
        Wr = routers.mods[l].weight
        idx, _ = hi.moe_router(X, Wr, k, eng.moe_flags)
        lg = (X.float() @ Wr.float().t()).half().float().cpu().tolist()
        for p, s in enumerate(idx.cpu().tolist()):
            rc, margin = topk_margin(lg[p], k)
            mismatch += rc != frozenset(s)
            out[(p, l)] = (frozenset(s), margin, lg[p])
    return out, mismatch


def first_diff(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


def router_tie_check(model, routers, ids, hi_tokens, hi_route, step, tie, router_tie, max_prefill=512, log=print):
    """Explain the engine's token at `step` (its first divergence from HF) by router near-ties.
    HF runs on the engine's tokens; events are taken in an order where every point comes after
    all the points it depends on: the prompt by prefill chunk, then layer, then position (within a
    layer the positions do not depend on each other's routing); then each decode position's layers;
    the token at step s after every router of positions < len(ids) + s. At the first event
    where HF and the engine differ: a router point whose HF margin is <= router_tie is adopted
    (HF re-runs with the engine's experts there) and the walk restarts; any other router
    difference, or a token at steps <= `step` whose HF top-2 gap is >= tie, ends it unexplained.
    Returns (explained, {"adopted": [...], "reason": str, "hf_gap": gap at step on the final run})."""
    T, L, k = len(ids), len(routers.mods), routers.k
    prefill_order = sorted(((p, l) for p in range(T) for l in range(L)), key=lambda x: (x[0] // max_prefill, x[1], x[0]))
    routers.force, adopted = {}, []
    try:
        while True:
            hf_tok, gaps = hf_run(model, routers, ids, step + 1, feed=hi_tokens)
            ev = None
            for s in range(step + 1):
                for p, l in (prefill_order if s == 0 else [(T + s - 1, l) for l in range(L)]):  # routers before token s
                    if (p, l) in routers.force:
                        continue
                    hf_set, hf_margin = topk_margin(routers.logits[(p, l)], k)
                    if hf_set != hi_route[(p, l)][0]:
                        ev = ("router", p, l, hf_set, hf_margin)
                        break
                if ev:
                    break
                if hf_tok[s] != hi_tokens[s]:
                    ev = ("token", s)
                    break
            # router-logit disagreement on this run, which shares the engine's routing at every point so
            # far: the rounding noise a router margin is compared with
            d = sorted(max(abs(x - y) for x, y in zip(routers.logits[q], hi_route[q][2])) for q in routers.logits)
            info = {"adopted": adopted, "hf_gap": gaps[step],
                    "router_logit_max_diff": {"median": d[len(d) // 2], "p99": d[int(len(d) * 0.99)], "max": d[-1]}}
            if ev is None:
                info["reason"] = f"HF with {len(adopted)} adopted router near-ties gives the engine's tokens through step {step}"
                return True, info
            if ev[0] == "token":
                s = ev[1]
                ok = gaps[s] < tie
                info["reason"] = (f"after {len(adopted)} adopted router near-ties HF differs at step {s} with top-2 gap "
                                  f"{gaps[s]:.4f} ({'token near-tie' if ok else 'not a near-tie'})")
                return ok, info
            _, p, l, hf_set, hf_margin = ev
            row = routers.logits[(p, l)]
            hi_set, hi_margin, hi_row = hi_route[(p, l)]
            pt = {"pos": p, "layer": l, "hf_margin": hf_margin, "engine_margin": hi_margin,
                  "hf_only": sorted(hf_set - hi_set), "engine_only": sorted(hi_set - hf_set),
                  "hf_logits": {e: row[e] for e in sorted(hf_set ^ hi_set)},
                  "engine_logits": {e: hi_row[e] for e in sorted(hf_set ^ hi_set)},
                  "fp16_ulp": fp16_ulp(sorted(row, reverse=True)[k - 1])}
            where = f"prompt token {p}" if p < T else f"decode input of step {p - T + 1}"
            log(f"    router difference at pos {p} ({where}) layer {l}: HF-only {pt['hf_only']} engine-only "
                f"{pt['engine_only']}, HF margin {hf_margin:.5f} ({hf_margin / pt['fp16_ulp']:.0f} ulp), "
                f"engine margin {hi_margin:.5f} -> {'adopt' if hf_margin <= router_tie else 'NOT a near-tie'}")
            if hf_margin > router_tie:
                info["reason"] = f"router difference at pos {p} layer {l} with HF margin {hf_margin:.4f} > {router_tie}"
                info["rejected"] = pt
                return False, info
            adopted.append(pt)
            routers.force[(p, l)] = hi_set
    finally:
        routers.force = {}


def main():
    import hipinfer as hi
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, os.path.join(HERE, "..", "tests"))
    from gate import PROMPTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--prompt", type=int, default=1, help="index into tests/gate.py's prompts (6: the long one)")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--tie", type=float, default=0.05, help="token near-tie (tests/gate.py --tie)")
    ap.add_argument("--router-tie", type=float, default=0.01, help="router near-tie (tests/gate.py --router-tie)")
    ap.add_argument("--all-at-once", action="store_true", help="also run HF with every engine set at once, plus a control")
    ap.add_argument("--inject", default=None, metavar="LAYER:SCALE",
                    help="negative control: perturb HF's attention output projection of one layer by SCALE relative "
                         "noise (layer_diff.py's fault), so the engine looks like it has a bug there")
    ap.add_argument("--gemm-splits", type=int, default=0, help="force split-K for every GEMM (changes the prefill "
                    "rounding, as a different autotuner pick does; 0: the normal choice)")
    ap.add_argument("--out", default=None, help="write the trace as JSON")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    if a.inject:
        l, scale = a.inject.split(":")
        model.model.layers[int(l)].self_attn.o_proj.register_forward_hook(lambda m, x, o: _perturb(o, float(scale), 1234))
    dims = model_dims(model.config)
    prompts = [tok(p).input_ids for p in PROMPTS] + [tok(" ".join(PROMPTS * 12)).input_ids[:700]]
    ids = prompts[a.prompt]
    T, L = len(ids), dims["layers"]

    if a.gemm_splits:
        hi.lib.hi_gemm_force_splits(a.gemm_splits)
    eng = hi.Engine(a.weights, max_batch=8, max_seq=1024, max_prefill=512)  # tests/gate.py's engine
    E, k = eng.n_experts, eng.top_k
    assert E > 0, "not an MoE model"
    routers = HFRouters(model, E, k)
    assert len(routers.mods) == L, [type(m).__name__ for m in model.modules()]

    hf_tok, hf_gaps = hf_run(model, routers, ids, a.new)
    hf_logits = routers.logits
    (hi_tok, xn), = engine_trace(eng, dims, [ids], a.new)
    gate_tok = eng.generate(ids, a.new, graph=True)
    hi_route, recompute_mismatch = engine_routing(hi, eng, routers, xn)
    div = first_diff(hi_tok, hf_tok)
    print(f"{a.model}: prompt {a.prompt} (len {T}), {L} layers, {E} experts top-{k}, GPU {torch.cuda.get_device_name()}, "
          f"HIPINFER_GEMM_AUTOTUNE={os.environ.get('HIPINFER_GEMM_AUTOTUNE', '1')}"
          + (f", split-K forced to {a.gemm_splits}" if a.gemm_splits else ""))
    print(f"traced engine run {'equals' if gate_tok == hi_tok else 'DIFFERS FROM'} the graph-captured run "
          f"(first diff {first_diff(gate_tok, hi_tok)}); torch recompute of the engine's router sets disagrees "
          f"with hi_moe_router at {recompute_mismatch} points")
    if div is None:
        print(f"engine and HF tokens identical for {a.new} steps")
    else:
        print(f"tokens diverge at step {div}: HF {hf_tok[div]} (gap {hf_gaps[div]:.4f}), engine {hi_tok[div]}")
    last = T + (div if div is not None else a.new - 1)  # positions 0 .. last - 1 feed the compared tokens

    # free-running comparison: both sides' own routing, up to the divergence
    pts, flips = [], []
    for p in range(last):
        for l in range(L):
            hf_set, hf_margin = topk_margin(hf_logits[(p, l)], k)
            hi_set, hi_margin, hi_row = hi_route[(p, l)]
            d = max(abs(x - y) for x, y in zip(hi_row, hf_logits[(p, l)]))
            pts.append((hf_margin, d))
            if hf_set != hi_set:
                flips.append({"pos": p, "layer": l, "hf_margin": hf_margin, "engine_margin": hi_margin,
                              "hf_set": sorted(hf_set), "engine_set": sorted(hi_set)})
    flips.sort(key=lambda f: (0, f["pos"] // eng.max_prefill, f["layer"], f["pos"]) if f["pos"] < T else (1, f["pos"], f["layer"]))
    dl, hm = sorted(d for _, d in pts), sorted(m for m, _ in pts)
    print(f"{len(pts)} (position, layer) router points feed the compared tokens. Engine vs HF router logits, "
          f"max |diff| per point: median {dl[len(dl) // 2]:.4f}, p99 {dl[int(len(dl) * 0.99)]:.4f}, max {dl[-1]:.4f}. "
          f"HF margins: {sum(m == 0 for m in hm)} exact ties, {sum(m < 0.002 for m in hm)} < 0.002, "
          f"{sum(m < 0.01 for m in hm)} < 0.01")
    print(f"expert sets differ at {len(flips)} points (each side on its own routing history)")
    for f in flips:
        print(f"  pos {f['pos']} layer {f['layer']}: HF margin {f['hf_margin']:.5f}, engine margin {f['engine_margin']:.5f}, "
              f"HF-only {sorted(set(f['hf_set']) - set(f['engine_set']))}, "
              f"engine-only {sorted(set(f['engine_set']) - set(f['hf_set']))}")
    result = {"model": a.model, "gpu": torch.cuda.get_device_name(), "prompt": a.prompt, "prompt_len": T, "new": a.new,
              "gemm_autotune": os.environ.get("HIPINFER_GEMM_AUTOTUNE", "1"), "gemm_splits": a.gemm_splits, "traced_equals_graph_run": gate_tok == hi_tok,
              "divergence_step": div, "hf_gap_at_divergence": hf_gaps[div] if div is not None else None,
              "hf_tokens": hf_tok, "engine_tokens": hi_tok, "router_points": len(pts),
              "recompute_mismatch": recompute_mismatch,
              "router_logit_max_diff": {"median": dl[len(dl) // 2], "p99": dl[int(len(dl) * 0.99)], "max": dl[-1]},
              "free_running_flips": flips}

    if div is not None:
        print(f"router near-tie check (router tie {a.router_tie}, token tie {a.tie}):")
        ok, info = router_tie_check(model, routers, ids, hi_tok, hi_route, div, a.tie, a.router_tie, eng.max_prefill)
        print(f"  -> {'EXPLAINED' if ok else 'NOT explained'}: {info['reason']}; HF top-2 gap at step {div} now "
              f"{info['hf_gap']:.4f}")
        n = info["router_logit_max_diff"]
        print(f"  engine vs HF router logits on that run, max |diff| per point: median {n['median']:.4f}, "
              f"p99 {n['p99']:.4f}, max {n['max']:.4f}")
        margins = [x["hf_margin"] for x in info["adopted"]]
        if margins:
            print(f"  adopted {len(margins)} points, HF margins max {max(margins):.5f}, "
                  f"in ulps max {max(x['hf_margin'] / x['fp16_ulp'] for x in info['adopted']):.0f}")
        result["router_tie_check"] = {"explained": ok, **info}

    if a.all_at_once and flips:
        runs = {}
        for name, key in [("control: HF's own sets at the differing points", "hf_set"),
                          ("the engine's sets at every differing point", "engine_set")]:
            routers.force = {(f["pos"], f["layer"]): frozenset(f[key]) for f in flips}
            t, g = hf_run(model, routers, ids, a.new)
            routers.force = {}
            print(f"HF with {name}: first diff vs HF {first_diff(t, hf_tok)}, vs engine {first_diff(t, hi_tok)}, "
                  f"{sum(x == y for x, y in zip(t, hi_tok))}/{a.new} tokens equal to the engine's")
            runs[name] = {"tokens": t, "first_diff_vs_hf": first_diff(t, hf_tok), "first_diff_vs_engine": first_diff(t, hi_tok)}
        result["all_at_once"] = runs
    routers.close()
    eng.close()
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(result, open(a.out, "w"), indent=1, default=sorted)


if __name__ == "__main__":
    main()
