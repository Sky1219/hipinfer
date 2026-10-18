"""End-to-end gate: hipinfer greedy decoding vs HF transformers (fp16, same GPU), any supported model.

Checks (1) last-position prefill logits, (2) greedy tokens for single-sequence decode with and
without graphs, (3) batched decode (B slots, different prompt lengths) against per-prompt HF.
A divergence is accepted only if HF's top-2 logit gap at that step is below `--tie` (an fp16
near-tie where summation order legitimately decides). Exit code 1 on any real mismatch.

MoE models make a second discrete choice per token and layer, the router's top-k, and a near-tie
there can flip by rounding just as a token can. When an MoE run diverges at a gap >= `--tie`, the
gate re-runs it with the router inputs dumped and asks whether router near-ties explain it
(scripts/moe_router_trace.py router_tie_check): HF re-runs on the engine's tokens, adopting the
engine's expert set at each point, in execution order, where HF's own k-th vs (k+1)-th router
logit margin is <= `--router-tie`. The run is accepted ("router near-tie, OK") only if this HF
then gives the engine's tokens through the divergence step, or diverges from them at a token
near-tie. Any router difference with a larger margin fails. docs/moe-design.md has the evidence.

python tests/gate.py --model unsloth/Llama-3.2-3B --weights llama32-3b.bin [--new 128] [--hf-first] [--out gate.json]

int8 weights (weight-file version 4, docs/quantization.md) compute a different model, so against
fp16 HF (--ref fp16, the default) their greedy tokens may diverge: each divergence is logged with
its step and HF's gap and accepted, and the summary reports how many tokens matched before it.
--ref int8 instead runs HF on the same int8 round-trip weights (scripts/export.py
fake_quantize_hf), which the engine must then match exactly like an fp16 engine matches fp16 HF.

--hf-first runs all HF generations first, keeps tokens, top-2 gaps and the prefill logits on the host,
frees HF, then loads the engine: for models whose weights fit on the GPU once but not twice. The
comparison is the same as without the flag, except that the MoE router check needs the HF model
and is skipped.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import hipinfer as hi  # noqa: E402

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Explain why the sky is blue in simple terms for a ten year old child.",
    "Once upon a time, in a small village at the edge of a dark forest, there lived",
    "List three advantages of paged KV caches for LLM serving:\n1.",
    "Q: What is 17 * 23?\nA:",
]


def hf_greedy(model, ids, n):
    """Greedy tokens plus per-step top-2 logit gaps."""
    out, gaps = [], []
    with torch.no_grad():
        o = model(torch.tensor([ids], device="cuda"), use_cache=True)
        past, logits = o.past_key_values, o.logits[0, -1]
        first_logits = logits.clone().cpu()
        for _ in range(n):
            top = logits.float().topk(2)
            gaps.append((top.values[0] - top.values[1]).item())
            t = int(logits.argmax())
            out.append(t)
            o = model(torch.tensor([[t]], device="cuda"), past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[0, -1]
    return out, gaps, first_logits


def compare(name, got, ref, gaps, tie, log, explain=None, allow=False):
    """explain(step) -> (ok, info): the MoE router check, tried when the token gap is not a near-tie.
    allow: a divergence that is not a near-tie is logged and accepted (int8 engine vs fp16 HF)."""
    for i, (a, b) in enumerate(zip(got, ref)):
        if a != b:
            ok = gaps[i] < tie
            entry = {"run": name, "identical": False, "step": i, "hf_gap": gaps[i]}
            verdict = "near-tie, OK" if ok else "MISMATCH"
            if not ok and explain is not None:
                ok, info = explain(i)
                entry["router_check"] = info
                verdict = f"router near-tie, OK ({info['reason']})" if ok else f"MISMATCH (router check: {info['reason']})"
            if not ok and allow:
                ok = entry["allowed_divergence"] = True
                verdict = "int8 weights vs the fp16 reference: allowed"
            print(f"  {name}: diverges at step {i} (hf gap {gaps[i]:.4f}) -> {verdict}")
            entry["accepted"] = ok
            log.append(entry)
            return ok
    print(f"  {name}: {len(ref)}/{len(ref)} tokens identical")
    log.append({"run": name, "identical": True, "step": len(ref)})
    return True


def router_explainer(model, eng, prompts, batch, b, got, a, traces):
    """explain(step) for compare(): router_tie_check on slot b of the engine run that decoded
    prompts[batch] together. The engine run is repeated with the router inputs dumped (cached per
    batch in `traces`) and must give the same tokens as the run being checked."""
    import moe_router_trace as mrt

    def explain(step):
        key = tuple(batch)
        if key not in traces:
            traces[key] = mrt.engine_trace(eng, mrt.model_dims(model.config), [prompts[i] for i in batch], a.new)
        tokens, xn = traces[key][b]
        if tokens != got:
            return False, {"explained": False, "reason": "the traced engine run gave different tokens, not checked"}
        routers = mrt.HFRouters(model, eng.n_experts, eng.top_k)
        try:
            route, _ = mrt.engine_routing(hi, eng, routers, xn)
            ok, info = mrt.router_tie_check(model, routers, prompts[batch[b]], tokens, route, step, a.tie,
                                            a.router_tie, eng.max_prefill, log=lambda *x: None)
        finally:
            routers.close()
        info["explained"] = ok
        m = [p["hf_margin"] for p in info["adopted"]]
        info["reason"] = f"{info['reason']}; max adopted HF router margin {max(m) if m else 0:.4f}"
        return ok, info
    return explain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--tie", type=float, default=0.05)
    ap.add_argument("--router-tie", type=float, default=0.01, help="MoE: router k-th vs (k+1)-th logit margin "
                    "below which the engine's expert set may differ from HF's (see the docstring)")
    ap.add_argument("--hf-first", action="store_true", help="run HF generation first, free it, then load the engine")
    ap.add_argument("--ref", choices=["fp16", "int8"], default="fp16",
                    help="HF reference weights: fp16 (an int8 engine may then diverge), or the exporter's int8 round "
                    "trip (fake-quantized HF; the gate is then as strict as for fp16)")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16)
    if a.ref == "int8":
        from export import fake_quantize_hf
        fake_quantize_hf(model)
    model = model.cuda().eval()
    prompts = [tok(p).input_ids for p in PROMPTS]
    # one long prompt exercising multi-tile attention and chunked prefill (max_prefill=512)
    prompts.append(tok(" ".join(PROMPTS * 12)).input_ids[:700])
    hf_refs = None
    if a.hf_first:  # HF phase first; tokens, gaps and the first logits stay on the host
        hf_refs = [hf_greedy(model, ids, a.new) for ids in prompts]
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"HF phase done; free GPU memory {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB", flush=True)
    eng = hi.Engine(a.weights, max_batch=8, max_seq=1024, max_prefill=512)
    print(f"{a.model}: layers {eng.layers} hidden {eng.hidden} heads {eng.heads}/{eng.kv_heads} head_dim {eng.head_dim} "
          f"vocab {eng.vocab} tied={eng.tied} qkv_bias={eng.qkv_bias} weights={'int8' if eng.quant else 'fp16'} "
          f"HF reference={a.ref}")
    if not eng.quant and a.ref == "int8":
        raise SystemExit("--ref int8 needs an int8 (version 4) weight file")
    allow = bool(eng.quant) and a.ref == "fp16"
    if allow:
        print("int8 engine vs fp16 HF: divergences are reported, not failed")
    moe = eng.n_experts > 0 and not hf_refs  # the router check needs the HF model
    if eng.n_experts and hf_refs:
        print("MoE: --hf-first frees the HF model, so the router near-tie check is skipped")
    traces = {}

    def explainer(batch, b, got):
        return router_explainer(model, eng, prompts, batch, b, got, a, traces) if moe else None

    ok = True
    refs, log, logit_diffs = [], [], []
    for i, ids in enumerate(prompts):
        ref, gaps, ref_logits = hf_refs[i] if hf_refs else hf_greedy(model, ids, a.new)
        refs.append((ref, gaps))
        for graph in (True, False):
            got = eng.generate(ids, a.new, graph=graph)
            if graph:
                # logits after prefill are overwritten by decode; redo the prefill to read them
                eng.prefill(0, ids)
                d = (eng.logits(1)[0].float().cpu() - ref_logits.float()).abs().max().item()
                logit_diffs.append(d)
                print(f"prompt {i} (len {len(ids)}): prefill logits max |diff| = {d:.4f}")
            ok &= compare(f"prompt {i} single graph={graph}", got, ref, gaps, a.tie, log, explainer([i], 0, got), allow)

    # batched decode: slots 0..B-1 hold different prompts at different positions
    B = 6
    firsts = [eng.prefill(b, prompts[b]) for b in range(B)]
    eng.decode(B, a.new - 1)
    gots = [[firsts[b]] + eng.history(b, len(prompts[b]) + 1, a.new - 1) for b in range(B)]
    for b, got in enumerate(gots):
        ok &= compare(f"batched B={B} slot {b}", got, refs[b][0], refs[b][1], a.tie, log, explainer(range(B), b, got), allow)

    # argument checks: out-of-range slots, tokens, batch sizes and positions raise instead of
    # writing past the KV cache
    p = prompts[0]
    bad_calls = [("prefill past max_seq", lambda: eng.prefill(0, p, p0=1024 - len(p) + 1)),
                 ("prefill slot >= max_batch", lambda: eng.prefill(8, p)),
                 ("token outside vocab", lambda: eng.prefill(0, [eng.vocab])),
                 ("decode B > max_batch", lambda: eng.decode(9, 1)),
                 ("decode past max_seq", lambda: (eng.prefill(0, p, p0=1024 - len(p) - 1), eng.decode(1, 2)))]
    for what, fn in bad_calls:
        try:
            fn()
            print(f"  {what}: not rejected -> FAIL")
            ok = False
        except ValueError:
            print(f"  {what}: rejected")
    eng.close()
    n_ident = sum(1 for r in log if r["identical"])
    n_router = sum(1 for r in log if r.get("router_check", {}).get("explained"))
    n_allowed = sum(1 for r in log if r.get("allowed_divergence"))
    prefix = [r["step"] for r in log]  # tokens identical to HF before the first difference
    print(f"\n{n_ident}/{len(log)} runs token-identical, max prefill-logit |diff| {max(logit_diffs):.4f}"
          + (f", {n_router} accepted by the router near-tie check" if n_router else "")
          + (f", {n_allowed} allowed divergences (int8 vs fp16); tokens matched before the first difference: "
             f"mean {sum(prefix) / len(prefix):.1f}, min {min(prefix)} of {a.new}" if allow else ""))
    print("GATE PASS" if ok else "GATE FAIL")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"model": a.model, "gpu": torch.cuda.get_device_name(), "new_tokens": a.new, "pass": ok,
                   "identical": n_ident, "runs": len(log), "router_accepted": n_router, "router_tie": a.router_tie, "max_prefill_logit_diff": max(logit_diffs),
                   "engine_weights": "int8" if eng.quant else "fp16", "hf_ref": a.ref, "allow_divergence": allow,
                   "allowed_divergences": n_allowed, "matched_prefix": prefix,
                   "prompt_lens": [len(p) for p in prompts], "log": log}, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
