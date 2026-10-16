"""CPU check of the MoE export + reference engine (scripts/ref_engine.py) against HF transformers.
No GPU needed. For each model:

1. fp32: HF in fp32 (weights rounded through fp16, as the file holds them) vs the reference in
   fp32 mode, all positions of each prompt: logits max |diff|, top-1 agreement, and the routed
   expert sets of every layer and token (HF's router logits, top-k) -> checks layouts and math.
2. fp16: HF in fp16 vs the reference in fp16 mode (the engine's rounding points): greedy tokens
   from each prompt, accepted at a divergence only if HF's top-2 gap there is below --tie (an fp16
   near-tie, as tests/gate.py does), plus prefill-logit max |diff|.

Models: hf-internal-testing/tiny-random-MixtralForCausalLM (top-2 of 4, fp32 routing weights),
synthetic tiny OLMoE (QK-norm, top-2 of 8 without renormalising) and tiny Granite MoE (tied
embeddings, all four Granite multipliers, a non-power-of-two attention multiplier), both random
init, and ibm-granite/granite-3.1-1b-a400m-base (skip with --quick).

python tests/test_moe_ref.py [--quick] [--new 16] [--out results/moe_ref_cpu.json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from ref_engine import RefEngine  # noqa: E402

TEXT = ["The capital of France is", "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
        "Q: What is 17 * 23?\nA:", "Once upon a time, in a small village at the edge of a dark forest, there lived"]
FAILS = []


def say(ok, msg):
    print(f"{'OK  ' if ok else 'FAIL'} {msg}", flush=True)
    if not ok:
        FAILS.append(msg)
    return ok


def synthetic(kind, root):
    """Random-init tiny OLMoE / Granite MoE saved under root; returns the directory."""
    torch.manual_seed(0)
    from transformers import AutoModelForCausalLM
    if kind == "olmoe":
        from transformers import OlmoeConfig
        cfg = OlmoeConfig(vocab_size=1024, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=4, num_experts=8, num_experts_per_tok=2,
                          norm_topk_prob=False, max_position_embeddings=256, initializer_range=0.1)
    else:
        from transformers import GraniteMoeConfig
        cfg = GraniteMoeConfig(vocab_size=1024, hidden_size=64, intermediate_size=32, num_hidden_layers=2,
                               num_attention_heads=4, num_key_value_heads=2, num_local_experts=8,
                               num_experts_per_tok=3, max_position_embeddings=256, initializer_range=0.1,
                               embedding_multiplier=12.0, logits_scaling=6.0, residual_multiplier=0.22,
                               attention_multiplier=0.03, tie_word_embeddings=True)
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():  # make the norms non-trivial so their layout matters
        for n, p in model.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.3 * torch.randn_like(p))
    d = os.path.join(root, f"tiny-{kind}")
    model.save_pretrained(d)
    return d


def export(model, out):
    subprocess.run([sys.executable, os.path.join(HERE, "..", "scripts", "export.py"), "--model", model, "--out", out],
                   check=True, stdout=subprocess.DEVNULL)


def prompts_for(model_id, vocab):
    if not os.path.isdir(model_id):  # hub models have tokenizers; the synthetic ones do not
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        return [tok(t).input_ids for t in TEXT]
    g = torch.Generator().manual_seed(1)
    return [torch.randint(3, vocab, (n,), generator=g).tolist() for n in (5, 17, 9, 30)]


def hf_greedy(model, ids, n):
    out, gaps = [], []
    with torch.no_grad():
        o = model(torch.tensor([ids]), use_cache=True)
        past, logits = o.past_key_values, o.logits[0, -1]
        first = logits.float().clone()
        for _ in range(n):
            top = logits.float().topk(2)
            gaps.append((top.values[0] - top.values[1]).item())
            t = int(logits.argmax())
            out.append(t)
            o = model(torch.tensor([[t]]), past_key_values=past, use_cache=True)
            past, logits = o.past_key_values, o.logits[0, -1]
    return out, gaps, first


def check_model(name, model_id, path, a, report):
    from transformers import AutoModelForCausalLM
    print(f"\n=== {name} ({model_id})", flush=True)
    r = {"model": f"synthetic {name}" if os.path.isdir(model_id) else model_id}
    hf16 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).eval()
    hf32 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).float().eval()
    print(f"  HF experts implementation: {hf16.config._experts_implementation}")
    prompts = prompts_for(model_id, hf32.config.vocab_size)

    # 1. fp32 math
    ref = RefEngine(path, "fp32")
    c = ref.c
    print(f"  {c['n_experts']} experts top-{c['top_k']} norm_topk={c['norm_topk']} fp16_weights={c['fp16_weights']} "
          f"qk_norm={c['qk_norm']} moe_scale={c['moe_scale']}")
    worst, agree, total, route_bad, route_n = 0.0, 0, 0, 0, 0
    # router logits [T, E] of every layer, captured from the router modules (Mixtral and OLMoE
    # return them first, Granite last; the scores / weights are [T, k])
    captured = []

    def hook(mod, inp, out):
        captured.append(next(t for t in out if t.is_floating_point() and t.shape[-1] == c["n_experts"]))
    hooks = [m.register_forward_hook(hook) for n, m in hf32.named_modules() if type(m).__name__.endswith("TopKRouter")]
    assert len(hooks) == c["layers"], [type(m).__name__ for m in hf32.modules()]
    for ids in prompts:
        captured.clear()
        with torch.no_grad():
            o = hf32(torch.tensor([ids]))
        want = o.logits[0].float()
        ref.reset()
        got = ref.forward(ids)
        scale = want.abs().max().item()
        worst = max(worst, (got - want).abs().max().item() / scale)
        agree += (got.argmax(-1) == want.argmax(-1)).sum().item()
        total += len(ids)
        for l, rl in enumerate(captured):  # per layer [T, E]; compare expert sets
            rl = rl.float().reshape(len(ids), -1)
            top = rl.topk(c["top_k"] + 1, dim=-1)
            for t in range(len(ids)):
                want_set = set(top.indices[t, :c["top_k"]].tolist())
                got_set = set(ref.last_routing[l][0][t].tolist())
                gap = (top.values[t, c["top_k"] - 1] - top.values[t, c["top_k"]]).item()
                route_n += 1
                if want_set != got_set and gap > 1e-4:
                    route_bad += 1
    for h in hooks:
        h.remove()
    r["fp32"] = {"max_rel_logit_diff": worst, "top1_agree": agree / total, "routing_mismatch": route_bad,
                 "routing_checked": route_n}
    say(worst < a.fp32_tol, f"{name} fp32: logits max |diff| / max|logit| = {worst:.2e} (tol {a.fp32_tol:.0e})")
    say(agree == total, f"{name} fp32: top-1 at every position {agree}/{total}")
    say(route_bad == 0, f"{name} fp32: routed expert sets equal HF's for {route_n - route_bad}/{route_n} (layer, token)")

    # 2. fp16 rounding points, greedy decode
    ref = RefEngine(path, "fp16")
    runs, worst16 = [], 0.0
    for i, ids in enumerate(prompts):
        want, gaps, first = hf_greedy(hf16, ids, a.new)
        got, steps = ref.generate(ids, a.new)
        d = (steps[0] - first).abs().max().item()
        worst16 = max(worst16, d / first.abs().max().item())
        div = next((s for s, (x, y) in enumerate(zip(got, want)) if x != y), None)
        ok = div is None or gaps[div] < a.tie
        runs.append({"prompt": i, "identical": div is None, "step": div, "hf_gap": None if div is None else gaps[div]})
        say(ok, f"{name} fp16 prompt {i} (len {len(ids)}): "
            + ("%d/%d greedy tokens identical" % (a.new, a.new) if div is None else
               f"diverges at step {div}, HF top-2 gap {gaps[div]:.4f} -> {'near-tie' if ok else 'MISMATCH'}")
            + f", prefill logits max |diff| {d:.4f}")
    r["fp16"] = {"runs": runs, "max_rel_prefill_logit_diff": worst16}
    report[name] = r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 1.3B Granite MoE")
    ap.add_argument("--new", type=int, default=16)
    ap.add_argument("--tie", type=float, default=0.05)
    ap.add_argument("--fp32-tol", type=float, default=2e-3, help="max |logit diff| relative to max |logit|")
    ap.add_argument("--keep", default=None, help="directory to keep exported files in (default: a temp dir)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    report = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = a.keep or tmp
        os.makedirs(root, exist_ok=True)
        models = [("tiny-mixtral", "hf-internal-testing/tiny-random-MixtralForCausalLM"),
                  ("tiny-olmoe", synthetic("olmoe", root)),
                  ("tiny-granitemoe", synthetic("granitemoe", root))]
        if not a.quick:
            models.append(("granite-moe-1b", "ibm-granite/granite-3.1-1b-a400m-base"))
        for name, mid in models:
            path = os.path.join(root, f"{name}.bin")
            if not os.path.exists(path):
                export(mid, path)
            check_model(name, mid, path, a, report)
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"pass": not FAILS, "fails": FAILS, "models": report}, open(a.out, "w"), indent=1)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
