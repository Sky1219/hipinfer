"""Speculative decoding benchmark (docs/speculative.md): batch-1 decode tokens/s of
Engine.generate_speculative against plain greedy decoding, the draft acceptance rate, and the
per-step costs that explain the speed-up, per k. Both engines share one GPU.

For every prompt of tests/gate.py (n new tokens each), after a warm-up:
  plain: prefill, then n-1 graph decode steps, timed on the host (wall clock, like the spec loop).
  spec:  generate_speculative(draft, prompt, n, k); the loop after both prefills is timed.
tokens/s = (n - 1) / time, summed over the prompts; reps runs, the median is reported.

Cost model: t_draft (draft ms/token), t_target (target ms/token), t_verify(R) (one verify pass of R
rows, wall clock incl. its host round trip). A round of k proposals costs about
k * t_draft + t_verify(k + 1) (+ one draft catch-up step after each fully accepted round) and yields
tokens_per_round tokens; predicted speed-up = tokens_per_round * t_target / round cost.

Also counts how often verifying with the prefill path (verify="prefill": hi_prefill_logits, GEMM
numerics) diverges from plain greedy output: the reason verify() uses the decode kernels.

python scripts/bench_spec.py --model unsloth/Llama-3.2-3B --target llama32-3b.bin --draft llama32-1b.bin \
    [--ks 1,2,3,4,5,6,8] [--new 128] [--reps 3] [--out results/2080ti/spec/llama32-3b_1b.json]
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import hipinfer as hi  # noqa: E402
from test_spec import engine, gate_prompts  # noqa: E402


def gpu_state():
    """SM clock and other compute processes on the current GPU (nvidia-smi index from CUDA_VISIBLE_DEVICES)."""
    idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    q = lambda *args: subprocess.run(["nvidia-smi", "-i", idx, *args], capture_output=True, text=True).stdout.strip()
    try:
        return {"sm_clock_mhz": q("--query-gpu=clocks.sm", "--format=csv,noheader,nounits"),
                "power_w": q("--query-gpu=power.draw", "--format=csv,noheader,nounits"),
                "compute_apps": q("--query-compute-apps=pid,used_memory", "--format=csv,noheader")}
    except OSError:
        return {}


def plain_decode_s(eng, prompts, n):
    t = 0.0
    for ids in prompts:
        eng.prefill(0, ids)
        t0 = time.perf_counter()
        eng.decode(1, n - 1)
        t += time.perf_counter() - t0
    return t


def spec_decode(target, draft, prompts, n, k, refs):
    tot = {"loop_s": 0.0, "draft_s": 0.0, "verify_s": 0.0, "accepted": 0, "drafted": 0, "rounds": 0, "catchups": 0, "tokens": 0}
    exact = True
    for ids, ref in zip(prompts, refs):
        st = {}
        got = target.generate_speculative(draft, ids, n, k, stats=st)
        exact &= got == ref
        for key in tot:
            tot[key] += st[key]
    tot["acceptance"] = tot["accepted"] / tot["drafted"]
    tot["tokens_per_round"] = (tot["tokens"] - len(prompts)) / tot["rounds"]
    tot["exact"] = exact
    return tot


def step_costs(target, draft, ids, max_rows, reps=20):
    """ms per decode step of each engine (graph, GPU time), and wall ms per verify pass of R rows."""
    target.prefill(0, ids)
    draft.prefill(0, ids)
    t_target = target.decode(1, 64) / 64
    t_draft = draft.decode(1, 64) / 64
    p = len(ids)
    verify = {}
    for R in range(1, max_rows + 1):
        rows = ids[:R] if len(ids) >= R else (ids * R)[:R]
        target.verify(0, rows, p)  # capture
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            target.verify(0, rows, p)
            ts.append(time.perf_counter() - t0)
        verify[R] = statistics.median(ts) * 1e3
    return t_target, t_draft, verify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Llama-3.2-3B", help="tokenizer for the prompts")
    ap.add_argument("--target", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--ks", default="1,2,3,4,5,6,8")
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--short", action="store_true", help="the six short gate prompts only (no 700-token prompt)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",")]
    prompts = gate_prompts(AutoTokenizer.from_pretrained(a.model))
    if a.short:
        prompts = prompts[:-1]
    target, draft = engine(a.target), engine(a.draft)
    n = a.new
    print(f"{os.path.basename(a.target)} <- {os.path.basename(a.draft)} on {torch.cuda.get_device_name()}: "
          f"{len(prompts)} prompts x {n} tokens, reps {a.reps}", flush=True)
    before = gpu_state()
    print(f"GPU before: {before}", flush=True)

    refs = [target.generate(ids, n) for ids in prompts]
    t_target, t_draft, t_verify = step_costs(target, draft, prompts[0], max(ks) + 1)
    print(f"decode step: target {t_target:.3f} ms, draft {t_draft:.3f} ms (ratio {t_draft / t_target:.3f})")
    print("verify pass (wall ms) by rows: " + ", ".join(f"{R}: {v:.3f}" for R, v in t_verify.items()), flush=True)

    plain_decode_s(target, prompts, n)  # warm-up
    plain = [plain_decode_s(target, prompts, n) for _ in range(a.reps)]
    tokens = len(prompts) * (n - 1)
    plain_tps = tokens / statistics.median(plain)
    print(f"plain: {plain_tps:.1f} tok/s (median of {[round(tokens / t, 1) for t in plain]})", flush=True)

    rows = []
    for k in ks:
        spec_decode(target, draft, prompts[:1], n, k, refs[:1])  # warm-up / graph capture
        runs = [spec_decode(target, draft, prompts, n, k, refs) for _ in range(a.reps)]
        med = sorted(runs, key=lambda r: r["loop_s"])[len(runs) // 2]
        tps = tokens / med["loop_s"]
        # cost model: k drafts + a verify of k+1 rows per round, plus the draft catch-up steps
        cost = (k + med["catchups"] / med["rounds"]) * t_draft + t_verify[k + 1]
        pred = med["tokens_per_round"] * t_target / cost
        row = {"k": k, "tok_s": tps, "speedup": tps / plain_tps, "acceptance": med["acceptance"],
               "tokens_per_round": med["tokens_per_round"], "rounds": med["rounds"], "exact": all(r["exact"] for r in runs),
               "draft_share": med["draft_s"] / med["loop_s"], "verify_share": med["verify_s"] / med["loop_s"],
               "predicted_speedup": pred, "tok_s_runs": [tokens / r["loop_s"] for r in runs]}
        rows.append(row)
        print(f"k={k}: {tps:.1f} tok/s, speed-up {row['speedup']:.3f}x (model {pred:.3f}x), acceptance {row['acceptance']:.3f}, "
              f"{row['tokens_per_round']:.2f} tokens/round, draft {row['draft_share']:.0%} / verify {row['verify_share']:.0%} "
              f"of the loop, exact={row['exact']}", flush=True)

    # verify through the prefill path instead: same algorithm, GEMM / flash-prefill numerics
    prefill_div = []
    for k in ks:
        for i, (ids, ref) in enumerate(zip(prompts, refs)):
            got = target.generate_speculative(draft, ids, n, k, verify="prefill")
            step = next((j for j, (x, y) in enumerate(zip(got, ref)) if x != y), None)
            prefill_div.append({"k": k, "prompt": i, "first_diff": step})
    n_div = sum(d["first_diff"] is not None for d in prefill_div)
    print(f"verify='prefill': {n_div}/{len(prefill_div)} runs diverge from plain greedy output", flush=True)

    after = gpu_state()
    print(f"GPU after: {after}")
    target.close()
    draft.close()
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"gpu": torch.cuda.get_device_name(), "target": os.path.basename(a.target),
                   "draft": os.path.basename(a.draft), "prompts": len(prompts), "prompt_lens": [len(p) for p in prompts],
                   "new_tokens": n, "reps": a.reps, "t_target_ms": t_target, "t_draft_ms": t_draft,
                   "t_verify_ms": t_verify, "plain_tok_s": plain_tps, "plain_tok_s_runs": [tokens / t for t in plain],
                   "spec": rows, "prefill_verify_divergent": n_div, "prefill_verify_runs": len(prefill_div),
                   "prefill_verify": prefill_div, "gpu_before": before, "gpu_after": after}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
