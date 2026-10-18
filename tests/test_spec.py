"""Speculative decoding must reproduce plain greedy decoding token for token (docs/speculative.md).

1. Target + draft (Llama-3.2-3B with a 1B draft by default): on the tests/gate.py prompts, the
   output of Engine.generate_speculative(draft, prompt, n, k) must equal Engine.generate(prompt, n)
   exactly for every k (default 2, 4, 6, 8), with CUDA graphs, and without them for one k.
2. Self-drafting (each --self model drafts for a second copy of itself): the verify pass computes
   exactly what a B=1 decode step computes, so every draft token must be accepted (acceptance 1.0)
   at every k. This checks the bit-exactness of verify() directly: a single differing logit
   anywhere would eventually flip a token.
3. Argument checks: too many verify rows, a verify past max_seq, and a bad rewind raise.

python tests/test_spec.py --model unsloth/Llama-3.2-3B --target llama32-3b.bin --draft llama32-1b.bin \
    [--self llama32-1b.bin tinyllama.bin] [--new 128] [--ks 2,4,6,8] [--out test_spec.json]
--model only supplies the tokenizer for the prompts (the HF model is never loaded).
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import hipinfer as hi  # noqa: E402
from gate import PROMPTS  # noqa: E402


def gate_prompts(tok):
    """tests/gate.py's prompts: six short ones and the 700-token one (multi-tile attention)."""
    prompts = [tok(p).input_ids for p in PROMPTS]
    prompts.append(tok(" ".join(PROMPTS * 12)).input_ids[:700])
    return prompts


def engine(path):
    return hi.Engine(path, max_batch=1, max_seq=1024, max_prefill=512)


def run_pair(target, draft, prompts, n, ks, log, expect_all_accepted=False, name=""):
    ok = True
    for i, ids in enumerate(prompts):
        ref = target.generate(ids, n)
        runs = [(k, True) for k in ks] + [(ks[len(ks) // 2], False)]
        for k, graph in runs:
            st = {}
            got = target.generate_speculative(draft, ids, n, k, graph=graph, stats=st)
            same = got == ref
            step = next((j for j, (a, b) in enumerate(zip(got, ref)) if a != b), None)
            full = st["accepted"] == st["drafted"]
            good = same and (full or not expect_all_accepted)
            ok &= good
            print(f"  {name} prompt {i} (len {len(ids)}) k={k} graph={graph}: "
                  f"{'identical' if same else f'DIFFERS at step {step}'}, accepted {st['accepted']}/{st['drafted']} "
                  f"({st['acceptance']:.3f}), {st['tokens_per_round']:.2f} tokens/round"
                  + ("" if good else "  -> FAIL"), flush=True)
            log.append({"pair": name, "prompt": i, "prompt_len": len(ids), "k": k, "graph": graph,
                        "identical": same, "first_diff": step, "accepted": st["accepted"], "drafted": st["drafted"],
                        "rounds": st["rounds"], "pass": good})
    return ok


def arg_checks(eng):
    ok = True
    p = [1, 2, 3]
    eng.prefill(0, p)
    bad = [("verify with MAX_VERIFY + 1 rows", lambda: eng.verify(0, [1] * (eng.MAX_VERIFY + 1), 3)),
           ("verify past max_seq", lambda: eng.verify(0, [1, 2], eng.max_seq - 1)),
           ("verify token outside vocab", lambda: eng.verify(0, [eng.vocab], 3)),
           ("rewind past max_seq", lambda: eng.rewind(0, eng.max_seq, 1)),
           ("rewind slot >= max_batch", lambda: eng.rewind(eng.max_batch, 3, 1))]
    for what, fn in bad:
        try:
            fn()
            print(f"  {what}: not rejected -> FAIL")
            ok = False
        except ValueError:
            print(f"  {what}: rejected")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-3B", help="tokenizer for the prompts")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--self", nargs="*", default=[], help="weight files to self-draft (target = draft model)")
    ap.add_argument("--self-model", nargs="*", default=None,
                    help="tokenizer per --self file (default: --model for all)")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--ks", default="2,4,6,8")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    log, ok = [], True

    prompts = gate_prompts(AutoTokenizer.from_pretrained(a.model))
    target, draft = engine(a.target), engine(a.draft)
    pair = f"{os.path.basename(a.target)} <- {os.path.basename(a.draft)}"
    print(f"{pair} on {torch.cuda.get_device_name()}, {len(prompts)} prompts x {a.new} tokens, k in {ks}")
    ok &= run_pair(target, draft, prompts, a.new, ks, log, name=pair)
    ok &= arg_checks(target)
    target.close()
    draft.close()

    for j, path in enumerate(a.self):
        tok_name = a.self_model[j] if a.self_model else a.model
        sp = gate_prompts(AutoTokenizer.from_pretrained(tok_name))
        t, d = engine(path), engine(path)
        name = f"{os.path.basename(path)} <- itself"
        print(f"{name}: every draft token must be accepted")
        ok &= run_pair(t, d, sp, a.new, ks, log, expect_all_accepted=True, name=name)
        t.close()
        d.close()

    n_pass = sum(r["pass"] for r in log)
    print(f"\n{n_pass}/{len(log)} runs pass")
    print("SPEC PASS" if ok else "SPEC FAIL")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"gpu": torch.cuda.get_device_name(), "target": a.target, "draft": a.draft, "self": a.self,
                   "new_tokens": a.new, "ks": ks, "pass": ok, "runs": len(log), "passed": n_pass, "log": log},
                  open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
