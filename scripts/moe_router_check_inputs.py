"""Engine router kernel vs HF's router on identical inputs: does hi_moe_router round like HF?

Hooks HF's router modules along the greedy runs of tests/gate.py's prompts, feeds each router's own
input (HF's normalised hidden states) to hi_moe_router, and compares the expert sets with HF's
top-k. With the same input, any difference can only come from the router dot products' summation
order, so it must sit at an HF margin within one or two fp16 ulps of the logit.

python scripts/moe_router_check_inputs.py [--model ibm-granite/granite-3.1-1b-a400m-base] [--new 128]
"""
import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.join(HERE, "..", "tests")]
import hipinfer as hi  # noqa: E402
from gate import PROMPTS  # noqa: E402
from moe_router_trace import fp16_ulp, topk_margin  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.1-1b-a400m-base")
    ap.add_argument("--new", type=int, default=128)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    mods = [m for m in model.modules() if type(m).__name__.endswith("TopKRouter")]
    k, E = mods[0].top_k, mods[0].num_experts
    flags = hi.MOE_NORM_TOPK | hi.MOE_FP16_WEIGHTS  # Granite (scripts/export.py)
    stats = {"points": 0, "set_diff": [], "logit_diff_max": 0.0, "weight_diff_max": 0.0}

    def hook(mod, args, out):
        x = args[0].reshape(-1, args[0].shape[-1]).contiguous()
        idx, w, logits = out
        hidx, hw = hi.moe_router(x, mod.weight, k, flags)
        lg = logits.reshape(-1, E).float()
        mine = (x.float() @ mod.weight.float().t()).half().float()  # the kernel's rounding points
        stats["logit_diff_max"] = max(stats["logit_diff_max"], (mine - lg).abs().max().item())
        for r in range(x.shape[0]):
            stats["points"] += 1
            row = lg[r].tolist()
            hf_set, margin = topk_margin(row, k)
            if set(hidx[r].tolist()) != hf_set:
                stats["set_diff"].append((margin, margin / fp16_ulp(sorted(row, reverse=True)[k - 1])))
            else:  # same experts: compare the routing weights expert by expert
                hw_by = dict(zip(hidx[r].tolist(), hw[r].tolist()))
                d = max(abs(hw_by[e] - v) for e, v in zip(idx.reshape(-1, k)[r].tolist(), w.reshape(-1, k)[r].float().tolist()))
                stats["weight_diff_max"] = max(stats["weight_diff_max"], d)
    hooks = [m.register_forward_hook(hook) for m in mods]
    prompts = [tok(p).input_ids for p in PROMPTS] + [tok(" ".join(PROMPTS * 12)).input_ids[:700]]
    with torch.no_grad():
        for ids in prompts:
            o = model(torch.tensor([ids], device="cuda"), use_cache=True)
            past, logits = o.past_key_values, o.logits[0, -1]
            for _ in range(a.new - 1):
                o = model(logits.argmax().reshape(1, 1), past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[0, -1]
    for h in hooks:
        h.remove()
    d = stats["set_diff"]
    print(f"{a.model}: {stats['points']} router evaluations on HF's own inputs ({len(prompts)} gate prompts, "
          f"prefill + {a.new - 1} decode steps), GPU {torch.cuda.get_device_name()}")
    print(f"router logits, fp32-accumulate-then-fp16 (the kernel's rounding) vs HF's F.linear: max |diff| "
          f"{stats['logit_diff_max']:.6f}")
    print(f"hi_moe_router expert set differs from HF's at {len(d)} points"
          + (f"; HF margins there: max {max(m for m, _ in d):.6f} ({max(u for _, u in d):.1f} fp16 ulp)" if d else ""))
    print(f"routing weights where the sets agree: max |diff| {stats['weight_diff_max']:.2e}")


if __name__ == "__main__":
    main()
