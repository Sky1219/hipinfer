"""Multi-GPU benchmark: tensor parallelism over `--tp` GPUs vs the same engine on one GPU.

1. all-reduce latency per backend (one-shot kernel vs NCCL) at decode and prefill message sizes
2. batch-1 decode latency vs context length (ms/token, % of the per-GPU bandwidth floor)
3. batched decode throughput (ctx 512, 128 new tokens)
4. prefill latency
The 1-GPU columns are skipped with --no-single (a model that only fits sharded).

python scripts/bench_tp.py --weights llama32-3b.bin --tp 2 --out results/2080ti/llama32-3b/tp2.json
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import hipinfer as hi  # noqa: E402
from bench import engine_bytes, peak_bw  # noqa: E402


class Target:
    """Where engines go: one GPU (tp=1) or `tp` GPUs; fits() checks memory before loading."""

    def __init__(self, weights, cfg, tp, devices):
        self.weights, self.cfg, self.tp, self.devices = weights, cfg, tp, devices
        emb = cfg["vocab"] * cfg["hidden"] * 2
        head = 0 if cfg["tied"] else emb
        # weight bytes resident per GPU: layers split tp ways, embedding replicated, lm_head split
        self.shard = (os.path.getsize(weights) - emb - head) / tp + emb + head / tp

    def fits(self, B, seq, pf):
        need = engine_bytes(self.cfg, self.shard, B, seq, pf, tp=self.tp, slack=1.2e9)
        torch.cuda.empty_cache()
        return all(torch.cuda.mem_get_info(d)[0] >= need for d in self.devices)

    def make(self, B, seq, pf):
        return hi.Engine(self.weights, max_batch=B, max_seq=seq, max_prefill=pf, tp=self.tp,
                         devices=self.devices)


def rand_ids(n, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(100, min(30000, vocab - 1), (n,), generator=g).tolist()


def wall(fn, iters=5):
    fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def bench_engine(tag, tgt, ctxs, prefill_Ts, batches, peak, modes):
    """Latency engine (B=1, long context) then throughput engine. Returns rows tagged per backend
    ("auto": one-shot kernel for decode-sized messages, NCCL above; "nccl": NCCL only; None on one
    GPU). Prefill is also timed without the compute/all-reduce overlap."""
    cfg, tp = tgt.cfg, tgt.tp
    out = {"decode_b1": [], "prefill": [], "decode_batched": []}
    V, kv_tok = cfg["vocab"], 2 * cfg["layers"] * cfg["kv_heads"] * cfg["head_dim"] * 2

    def select(eng, mode):
        if mode:
            eng.set_allreduce(mode)
        return tag + (f" {mode}" if mode else "")

    ctxs = list(ctxs)
    # the cache must hold the longest decode context and the longest prefill
    seq = lambda: max(max(ctxs) + 256, max(prefill_Ts))
    while ctxs and not tgt.fits(1, seq(), max(prefill_Ts)):
        print(f"[{tag}] ctx {ctxs.pop()} does not fit; dropped", flush=True)
    if not ctxs:
        return out
    eng = tgt.make(1, seq(), max(prefill_Ts))
    for mode in modes:
        name = select(eng, mode)
        for ctx in ctxs:
            ids = rand_ids(ctx, V)
            n = 128
            best = 1e9
            for _ in range(3):
                eng.prefill(0, ids)
                best = min(best, eng.decode(1, n, True) / n)
            floor = (eng.weight_bytes + kv_tok * (ctx + n // 2) / tp) / peak / 1e6 if peak else None
            out["decode_b1"].append({"engine": name, "tp": tp, "allreduce": mode, "ctx": ctx, "ms": best,
                                     "bw_floor_ms": floor, "bw_util": floor / best if floor else None})
            util = f"   {floor / best * 100:3.0f}% of the per-GPU BW floor" if floor else ""
            print(f"[{name:14s}] decode B=1 ctx={ctx:5d}: {best:7.3f} ms/token{util}", flush=True)
        for overlap in ([True, False] if mode == "auto" else [True]):
            if mode:
                eng.set_overlap(overlap)
            for T in prefill_Ts:
                ids = rand_ids(T, V, 1)
                ms = wall(lambda: eng.prefill(0, ids))
                out["prefill"].append({"engine": name, "tp": tp, "allreduce": mode, "overlap": overlap and tp > 1, "T": T,
                                       "ms": ms, "tok_s": T / ms * 1e3})
                ov = "" if not mode else (" overlap" if overlap else " no overlap")
                print(f"[{name:14s}] prefill T={T:5d}{ov}: {ms:8.2f} ms ({T / ms * 1e3:7.0f} tok/s)", flush=True)
        if mode:
            eng.set_overlap(True)
    eng.close()
    Bmax = max(batches)
    while Bmax >= 1 and not tgt.fits(Bmax, 1024, 512):
        Bmax //= 2
    if Bmax < 1:
        return out
    eng = tgt.make(Bmax, 1024, 512)
    ids = rand_ids(512, V, 2)
    for mode in modes:
        name = select(eng, mode)
        for B in [b for b in batches if b <= Bmax]:
            for b in range(B):
                eng.prefill(b, ids)
            eng.decode(B, 2)
            for b in range(B):
                eng.prefill(b, ids)
            steps = 128
            ms = eng.decode(B, steps)
            out["decode_batched"].append({"engine": name, "tp": tp, "allreduce": mode, "B": B, "ms_per_step": ms / steps,
                                          "tok_s": B * steps / ms * 1e3})
            print(f"[{name:14s}] batched B={B:3d}: {ms / steps:7.3f} ms/step {B * steps / ms * 1e3:8.0f} tok/s", flush=True)
    eng.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--devices", type=int, nargs="*", default=None)
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--no-single", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    devices = a.devices or list(range(a.tp))
    peak = peak_bw(devices[0])
    cfg = hi.read_config(a.weights)  # from the header: the model may not fit one GPU
    max_ctx = hi.max_context(cfg)  # max_pos, or the sliding window (Mistral-7B-v0.1: 4096)
    ctxs = [c for c in [128, 1024, 4096, 8192, 16384] if c + 256 <= min(max_ctx, a.max_ctx)]
    prefill_Ts = [T for T in [512, 2000] if T <= max_ctx]
    batches = [1, 4, 8, 16, 32, 64]
    res = {"gpus": [torch.cuda.get_device_name(d) for d in devices], "peak_bw_GBs": peak, "weights": os.path.basename(a.weights),
           "config": cfg, "tp": a.tp}
    print(res["gpus"], "|", res["weights"], cfg, flush=True)

    # 1. all-reduce latency
    eng = hi.Engine(a.weights, max_batch=1, max_seq=64, max_prefill=64, tp=a.tp, devices=devices)
    info = res["tp_info"] = eng.tp_info()
    print("tp info:", info)
    ar_modes = ["oneshot"] + (["nccl"] if info["nccl"] else [])
    modes = ["auto"] + (["nccl"] if info["nccl"] else [])
    H = cfg["hidden"]
    res["allreduce"] = []
    print("\n== all-reduce latency (us), fp16, in place ==")
    for label, n in [("decode B=1", H), ("decode B=8", 8 * H), ("decode B=64", 64 * H), ("prefill T=512", 512 * H),
                     ("prefill T=2000", 2000 * H)]:
        row = {"what": label, "bytes": 2 * n}
        for m in ar_modes:
            eng.set_allreduce(m, 1 << 40)
            row[m + "_us"] = eng.allreduce_bench(n, iters=50 if n > 64 * H else 200)
        res["allreduce"].append(row)
        print(f"{label:15s} {2 * n / 1024:8.0f} KB  " + "  ".join(f"{m} {row[m + '_us']:8.1f}" for m in ar_modes), flush=True)
    eng.close()

    # 2-4. one GPU vs tensor parallel
    res["single"] = None
    if not a.no_single:
        print("\n== 1 GPU ==")
        res["single"] = bench_engine("1 GPU", Target(a.weights, cfg, 1, devices[:1]), ctxs, prefill_Ts, batches, peak,
                                     [None])
    print(f"\n== tensor parallel x{a.tp} ==")
    res["tp_rows"] = bench_engine(f"TP x{a.tp}", Target(a.weights, cfg, a.tp, devices), ctxs, prefill_Ts, batches, peak,
                                  modes)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
