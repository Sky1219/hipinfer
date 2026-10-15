"""End-to-end gate: hipinfer greedy decoding vs HF transformers (fp16, same GPU), any supported model.

Checks (1) last-position prefill logits, (2) greedy tokens for single-sequence decode with and
without graphs, (3) batched decode (B slots, different prompt lengths) against per-prompt HF.
A divergence is accepted only if HF's top-2 logit gap at that step is below `--tie` (an fp16
near-tie where summation order legitimately decides). Exit code 1 on any real mismatch.

python tests/gate.py --model unsloth/Llama-3.2-3B --weights llama32-3b.bin [--new 128] [--hf-first] [--out gate.json]

--hf-first runs all HF generations first, keeps tokens, top-2 gaps and the prefill logits on the host,
frees HF, then loads the engine: for models whose weights fit on the GPU once but not twice. The
comparison is the same as without the flag.
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


def compare(name, got, ref, gaps, tie, log):
    for i, (a, b) in enumerate(zip(got, ref)):
        if a != b:
            ok = gaps[i] < tie
            print(f"  {name}: diverges at step {i} (hf gap {gaps[i]:.4f}) -> {'near-tie, OK' if ok else 'MISMATCH'}")
            log.append({"run": name, "identical": False, "step": i, "hf_gap": gaps[i], "accepted": ok})
            return ok
    print(f"  {name}: {len(ref)}/{len(ref)} tokens identical")
    log.append({"run": name, "identical": True})
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--tie", type=float, default=0.05)
    ap.add_argument("--hf-first", action="store_true", help="run HF generation first, free it, then load the engine")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
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
          f"vocab {eng.vocab} tied={eng.tied} qkv_bias={eng.qkv_bias}")
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
            ok &= compare(f"prompt {i} single graph={graph}", got, ref, gaps, a.tie, log)

    # batched decode: slots 0..B-1 hold different prompts at different positions
    B = 6
    firsts = [eng.prefill(b, prompts[b]) for b in range(B)]
    eng.decode(B, a.new - 1)
    for b in range(B):
        got = [firsts[b]] + eng.history(b, len(prompts[b]) + 1, a.new - 1)
        ok &= compare(f"batched B={B} slot {b}", got, refs[b][0], refs[b][1], a.tie, log)

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
    print(f"\n{n_ident}/{len(log)} runs token-identical, max prefill-logit |diff| {max(logit_diffs):.4f}")
    print("GATE PASS" if ok else "GATE FAIL")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"model": a.model, "gpu": torch.cuda.get_device_name(), "new_tokens": a.new, "pass": ok,
                   "identical": n_ident, "runs": len(log), "max_prefill_logit_diff": max(logit_diffs),
                   "prompt_lens": [len(p) for p in prompts], "log": log}, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
