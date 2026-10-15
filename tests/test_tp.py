"""Tensor parallelism vs one GPU on the same weights, for each all-reduce backend.

1. all-reduce self-test (exact sums, incl. messages larger than one staging slot)
2. prefill logits of every position: max |diff| and top-1 agreement against the 1-GPU engine
3. greedy decode, single sequence (GEMV path, CUDA graph) and batched B=8 (GEMM path), token for
   token against the 1-GPU engine; a divergence is accepted only at a near-tie (1-GPU top-2
   logit gap < --tie, tracked per slot and step), where the different fp16 summation order
   legitimately decides
4. backends agree with each other bitwise at tp=2 (both round a two-term fp32 sum once)

With --no-ref (a model too big for one GPU) only 1 and 4 run; use tests/ppl.py --tp for accuracy.

python tests/test_tp.py --weights tinyllama.bin [--tp 2] [--devices 0 1] [--out tp.json]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import hipinfer as hi  # noqa: E402

LENS = [7, 33, 200, 400, 700]   # 700 > max_prefill 512: chunked prefill (logits compared for the others)
B = 8


def prompts(vocab):
    g = torch.Generator().manual_seed(0)
    return [torch.randint(100, vocab - 100, (n,), generator=g).tolist() for n in LENS + [11, 90, 300, 5]]


def greedy_with_gaps(eng, ids, n):
    """Greedy tokens one step at a time, with the top-2 logit gap at every step."""
    toks, gaps = [eng.prefill(0, ids)], []
    for i in range(n):
        top = eng.logits(1)[0].float().topk(2)
        gaps.append((top.values[0] - top.values[1]).item())
        if i == n - 1:
            break
        eng.decode(1, 1)
        toks.append(eng.history(0, len(ids) + 1 + i, 1)[0])
    return toks, gaps


def batched(eng, ps, n):
    firsts = [eng.prefill(b, ps[b]) for b in range(B)]
    eng.decode(B, n - 1)
    return [[firsts[b]] + eng.history(b, len(ps[b]) + 1, n - 1) for b in range(B)]


def batched_with_gaps(eng, ps, n):
    """batched(), one step at a time, with every slot's top-2 logit gap at every step."""
    firsts = [eng.prefill(b, ps[b]) for b in range(B)]
    gaps = [[] for _ in range(B)]
    for i in range(n):
        top = eng.logits(B).float().topk(2, -1).values
        for b in range(B):
            gaps[b].append((top[b, 0] - top[b, 1]).item())
        if i < n - 1:
            eng.decode(B, 1)
    toks = [[firsts[b]] + eng.history(b, len(ps[b]) + 1, n - 1) for b in range(B)]
    return toks, gaps


def compare(got, ref, gaps, tie):
    for i, (a, b) in enumerate(zip(got, ref)):
        if a != b:
            g = gaps[i] if gaps is not None else float("nan")
            return {"identical": False, "step": i, "ref_gap": g, "accepted": gaps is not None and g < tie}
    return {"identical": True, "accepted": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--devices", type=int, nargs="*", default=None)
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--tie", type=float, default=0.05)
    ap.add_argument("--no-ref", action="store_true", help="skip the 1-GPU reference (model does not fit one GPU)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    kw = dict(max_batch=B, max_seq=1024, max_prefill=512)
    res = {"weights": os.path.basename(a.weights), "tp": a.tp, "gpus": [torch.cuda.get_device_name(d) for d in
           (a.devices or range(a.tp))], "backends": {}}
    ok = True

    ref = None
    if not a.no_ref:
        eng = hi.Engine(a.weights, **kw)
        ps = prompts(eng.vocab)
        ref = {"logits": [eng.prefill_logits(0, p).cpu() for p in ps[:len(LENS)] if len(p) <= 512],
               "single": [greedy_with_gaps(eng, p, a.new) for p in ps[:len(LENS)]],
               "batched": batched_with_gaps(eng, ps, a.new)}
        eng.close()
        torch.cuda.empty_cache()

    tp = hi.Engine(a.weights, tp=a.tp, devices=a.devices, **kw)
    info = tp.tp_info()
    res["tp_info"] = info
    print(f"{res['weights']}: tp={a.tp} on {res['gpus']}  p2p={info['p2p']} nccl={info['nccl']}")
    ps = prompts(tp.vocab)
    modes = ["oneshot"] + (["nccl", "auto"] if info["nccl"] else [])
    outs = {}
    for mode in modes:
        tp.set_allreduce(mode)
        r = res["backends"][mode] = {}
        sizes = [8, tp.hidden, 8 * tp.hidden, 512 * tp.hidden, (3 << 20) + 8]
        errs = [tp.allreduce_check(n) for n in sizes]
        r["allreduce_max_err"] = max(errs)
        ok &= max(errs) == 0
        print(f"[{mode}] all-reduce self-test sizes {sizes}: max err {max(errs)}")
        logits = [tp.prefill_logits(0, p) for p in ps[:len(LENS)] if len(p) <= 512]
        single = [tp.generate(p, a.new) for p in ps[:len(LENS)]]
        bat = batched(tp, ps, a.new)
        outs[mode] = (logits, single, bat)
        if ref is None:
            continue
        diffs, agree, n_pos = [], 0, 0
        for lt, lr in zip(logits, ref["logits"]):
            lt = lt.cpu().float()
            lr = lr.float()
            diffs.append((lt - lr).abs().max().item())
            agree += (lt.argmax(-1) == lr.argmax(-1)).sum().item()
            n_pos += lt.shape[0]
        r["prefill_max_logit_diff"] = max(diffs)
        r["prefill_top1_agreement"] = agree / n_pos
        print(f"[{mode}] prefill logits vs 1 GPU: max |diff| {max(diffs):.4f}, top-1 agreement {100 * agree / n_pos:.2f}% "
              f"of {n_pos} positions")
        runs = []
        for i, (got, (rt, gaps)) in enumerate(zip(single, ref["single"])):
            c = compare(got, rt, gaps, a.tie)
            runs.append({"run": f"single prompt {i} (len {LENS[i]})", "kind": "single", **c})
        ref_toks, ref_gaps = ref["batched"]
        for b in range(B):
            runs.append({"run": f"batched B={B} slot {b}", "kind": "batched", **compare(bat[b], ref_toks[b], ref_gaps[b], a.tie)})
        for x in runs:
            tag = "identical" if x["identical"] else \
                f"diverges at step {x['step']} (1-GPU gap {x['ref_gap']:.4f}) -> {'near-tie, OK' if x['accepted'] else 'MISMATCH'}"
            print(f"[{mode}]   {x['run']}: {tag}")
        r["runs"] = runs
        r["identical"] = sum(x["identical"] for x in runs)
        bad = [x for x in runs if not x["accepted"]]
        ok &= not bad
        print(f"[{mode}] {r['identical']}/{len(runs)} runs token-identical to 1 GPU" + (f", {len(bad)} MISMATCH" if bad else ""))
    if len(outs) > 1 and a.tp == 2:
        m0 = modes[0]
        for m in modes[1:]:
            same = all(torch.equal(x, y) for x, y in zip(outs[m0][0], outs[m][0])) and outs[m0][1] == outs[m][1] \
                and outs[m0][2] == outs[m][2]
            res["backends"][m]["bitwise_equal_to_" + m0] = same
            print(f"{m} vs {m0}: {'bitwise identical' if same else 'DIFFERENT'} logits and tokens")
            ok &= same
    tp.close()
    res["pass"] = bool(ok)
    print("TP TEST PASS" if ok else "TP TEST FAIL")
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
