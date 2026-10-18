"""CPU checks of the eviction selection logic (docs/eviction.md); no GPU or build needed.

The kernels' algorithms, emulated step for step in PyTorch, against the references in
eviction_ref.py:
  - radix select (block_kth + block_select_k): k smallest keys, ties to the lower index, on
    order-preserving uint32 keys of fp32 values (negative values, zeros and many ties included)
  - wfl_top2_kernel's decomposition: 16 threads per row, each with a running top-2 over its
    columns (strict >, increasing column order), merged by an xor butterfly with t2_merge
  - the incremental rescan of wfl_round_kernel: only rows whose best or second-best kept column
    was just removed are recomputed; decisions must equal recomputing every row every round,
    also on similarity matrices full of exact ties
And, with SUBCACHE_DIR set, eviction_ref.ref_wfl_prune against SubCache's policies.wfl_prune.
This checks the logic only: the CUDA code itself is checked by tests/test_eviction.py on a GPU.

Run: python tests/test_eviction_cpu.py"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from eviction_ref import first_max, ref_select, ref_sim, ref_top2, ref_wfl_prune  # noqa: E402

torch.manual_seed(0)
FAILS = []


def check_eq(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name:64s} {detail}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------------------------ radix select
def f2key(x):
    """evict.cuh f2key on a float32 tensor -> int64 holding the uint32 key."""
    u = x.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where((u & 0x80000000) != 0, (~u) & 0xFFFFFFFF, u | 0x80000000)


def block_select_k(keys, k):
    """block_kth + block_select_k on one problem: keys int64 [n] (0xffffffff = ineligible).
    Returns a bool mask of the selected slots."""
    prefix, mask, kk = 0, 0, k
    for shift in range(28, -1, -4):
        sel = (keys & mask) == prefix
        hist = torch.bincount((keys[sel] >> shift) & 15, minlength=16).tolist()
        cum, b = 0, 0
        while b < 15:
            if cum + hist[b] >= kk:
                break
            cum += hist[b]
            b += 1
        prefix |= b << shift
        kk -= cum
        mask |= 15 << shift
    T, take = prefix, kk
    eq = keys == T
    rank = torch.cumsum(eq.long(), 0) - eq.long()  # exclusive count of equal keys before each slot
    return (keys < T) | (eq & (rank < take))


def test_select():
    for n, k, lo, hi, levels, signed in [(5, 2, 0, 5, 0, False), (1000, 100, 4, 900, 0, False), (1000, 100, 4, 900, 7, False),
                                         (4096, 1, 0, 4096, 0, True), (4096, 4095, 0, 4096, 3, True),
                                         (3000, 1900, 100, 2000, 0, True), (777, 300, 10, 700, 2, False)]:
        for largest in (True, False):
            ok = True
            for _ in range(4):
                s = torch.randn(1, n) if signed else torch.rand(1, n)
                if levels:
                    s = (s * levels).floor() / levels  # many exact ties (and -0.0 / +0.0 pairs when signed)
                u = f2key(s[0, lo:hi])
                keys = (0xFFFFFFFF - u) if largest else u  # ~u on uint32
                got = torch.zeros(1, n, dtype=torch.uint8)
                got[0, lo:hi] = block_select_k(keys, k).to(torch.uint8)
                ok &= torch.equal(got, ref_select(s, lo, hi, k, largest))
            check_eq(f"radix select n={n} k={k} [{lo},{hi}) {'largest' if largest else 'smallest'}"
                     f"{f' {levels}-level ties' if levels else ''}{' signed' if signed else ''}", ok)


# ------------------------------------------------------------------ top-2 decomposition
def t2_better(v, i, u, j):
    return v > u or (v == u and i < j)


def t2_merge(x, y):
    xb1, xa1, xb2, xa2 = x
    yb1, ya1, yb2, ya2 = y
    if t2_better(yb1, ya1, xb1, xa1):
        return (yb1, ya1, xb1, xa1) if t2_better(xb1, xa1, yb2, ya2) else (yb1, ya1, yb2, ya2)
    return (xb1, xa1, yb1, ya1) if t2_better(yb1, ya1, xb2, xa2) else (xb1, xa1, xb2, xa2)


def kernel_top2_row(srow, kept, n):
    """wfl_top2_kernel for one row: thread tx owns columns c0 + tx*4 + c (64-column tiles)."""
    lanes = []
    for tx in range(16):
        t = (-3.0, 0x7FFFFFFF, -3.0, -1)
        for c0 in range(0, n, 64):
            for c in range(4):
                col = c0 + tx * 4 + c
                if col >= n or not kept[col]:
                    continue
                s = float(srow[col])
                b1, a1, b2, a2 = t
                if s > b1:
                    t = (s, col, b1, a1)
                elif s > b2:
                    t = (b1, a1, s, col)
        lanes.append(t)
    for off in (1, 2, 4, 8):  # xor butterfly; every lane ends with the full merge
        lanes = [t2_merge(lanes[i], lanes[i ^ off]) for i in range(16)]
    assert all(l == lanes[0] for l in lanes), "butterfly lanes disagree"
    return lanes[0]


def test_top2():
    for n, levels, frac_kept in [(37, 0, 1.0), (200, 4, 0.7), (300, 2, 0.5), (130, 0, 0.3), (64, 1, 0.9), (2, 0, 1.0)]:
        sim = torch.rand(1, n, n)
        if levels:
            sim = (sim * levels).round() / levels  # exact ties everywhere
        sim[0].fill_diagonal_(1.0)
        kept = torch.rand(1, n) < frac_kept
        kept[0, :2] = True
        best, arg, second = ref_top2(sim, kept)
        # reference second index: lowest index among kept columns != arg holding the second value
        m2 = sim.masked_fill(~kept[:, None, :], -1.0).scatter(-1, arg[..., None], -5.0)
        _, arg2 = first_max(m2)
        ok = True
        for i in range(n):
            b1, a1, b2, a2 = kernel_top2_row(sim[0, i], kept[0], n)
            ok &= b1 == float(best[0, i]) and a1 == int(arg[0, i]) and max(b2, 0.0) == max(float(second[0, i]), 0.0)
            if int(kept[0].sum()) >= 2:
                ok &= a2 == int(arg2[0, i])
        check_eq(f"top-2 decomposition n={n}{f' {levels}-level ties' if levels else ''} kept {frac_kept}", ok)


# ------------------------------------------------------------------ incremental rescan
def prune_incremental(sim, w, cand, held, m, mix, step, rounds=None, incremental=True):
    """wfl_prune with the kernels' state machine: top-2 kept per row, rows rescanned only when
    their best or second-best column was removed in the last round."""
    Pn, n = w.shape
    kept = cand | held
    n_remove = int(cand.sum(1).min()) - m
    if rounds:
        step = max(step, math.ceil(n_remove / rounds))
    best = torch.empty(Pn, n)
    sec = torch.empty(Pn, n)
    arg = torch.zeros(Pn, n, dtype=torch.long)
    arg2 = torch.zeros(Pn, n, dtype=torch.long)
    dirty = torch.ones(Pn, n, dtype=torch.bool)
    removed, rescans = 0, 0
    while removed < n_remove:
        b, a, s2 = ref_top2(sim, kept)
        m2 = sim.masked_fill(~kept[:, None, :], -1.0).scatter(-1, a[..., None], -5.0)
        _, a2 = first_max(m2)
        a2 = torch.where(s2 >= 0, a2, torch.full_like(a2, -1))  # fewer than two kept columns
        best[dirty], arg[dirty], sec[dirty], arg2[dirty] = b[dirty], a[dirty], s2[dirty], a2[dirty]
        rescans += int(dirty.sum())
        loss = torch.zeros_like(w).scatter_add_(1, arg, w * (best - sec.clamp_min(0.0)).clamp_min(0.0))
        loss = (mix * loss + (1.0 - mix) * w).masked_fill(~(kept & cand), float("inf"))
        k = min(step, n_remove - removed)
        gone = torch.sort(loss, dim=1, stable=True).indices[:, :k]
        kept.scatter_(1, gone, False)
        removed += k
        hit = torch.zeros_like(kept).scatter_(1, gone, True)
        dirty = (hit.gather(1, arg) | (hit.gather(1, arg2.clamp_min(0)) & (arg2 >= 0))) if incremental else torch.ones_like(kept)
    return kept & cand, rescans


def test_incremental():
    for Pn, n, D, m, step, levels, rounds in [(4, 64, 16, 20, 8, 0, None), (4, 200, 32, 60, 1, 0, None),
                                              (4, 200, 32, 60, 8, 4, None), (2, 400, 32, 50, 8, 2, None),
                                              (3, 300, 16, 40, 5, 0, 4)]:
        k = torch.randn(Pn, n, D)
        v = torch.randn(Pn, n, D)
        src, dst = torch.randint(0, n, (n // 3,)), torch.randint(0, n, (n // 3,))
        k[:, dst] = k[:, src] + 0.05 * torch.randn(Pn, n // 3, D)  # near-copies: rows share neighbours
        v[:, dst] = v[:, src] + 0.05 * torch.randn(Pn, n // 3, D)
        sim = ref_sim(k, v)
        if levels:
            sim = (sim * levels).round() / levels  # exact ties in the similarity
        w = torch.rand(Pn, n) ** 3
        w = w / w.sum(1, keepdim=True)
        held = torch.zeros(Pn, n, dtype=torch.bool)
        held[:, :4] = held[:, -16:] = True
        cand = ~held
        ref = ref_wfl_prune(sim, w, cand.clone(), held, m, 0.5, step, rounds)
        inc, scans = prune_incremental(sim, w, cand.clone(), held, m, 0.5, step, rounds)
        full, full_scans = prune_incremental(sim, w, cand.clone(), held, m, 0.5, step, rounds, incremental=False)
        check_eq(f"incremental rescan P={Pn} n={n} step={step}{f' {levels}-level sim ties' if levels else ''}",
                 torch.equal(inc, ref) and torch.equal(full, ref),
                 f"rows rescanned {scans} vs {full_scans} for full rescans ({scans / full_scans:.1%})")


def test_vs_subcache():
    root = os.environ.get("SUBCACHE_DIR")
    if not root:
        print("skip ref_wfl_prune vs SubCache policies.wfl_prune (set SUBCACHE_DIR)")
        return
    sys.path.insert(0, root)
    from subcache import policies
    for Pn, n, m, step, rounds, mix in [(4, 64, 20, 8, None, 0.5), (8, 200, 60, 8, None, 0.5), (4, 513, 100, 8, 3, 0.5),
                                        (4, 300, 50, 1, None, 0.3), (4, 300, 50, 8, None, 1.0)]:
        k, v = torch.randn(Pn, n, 128), torch.randn(Pn, n, 128)
        w = torch.rand(Pn, n) ** 3
        w = w / w.sum(1, keepdim=True)
        held = torch.zeros(Pn, n, dtype=torch.bool)
        held[:, :4] = held[:, -16:] = True
        sim = policies.slot_similarity(k, v)
        same_sim = torch.equal(sim, ref_sim(k, v))
        a = policies.wfl_prune(sim, w, ~held, held, m, mix=mix, step=step, rounds=rounds)
        b = ref_wfl_prune(sim, w, ~held, held, m, mix=mix, step=step, rounds=rounds)
        check_eq(f"ref_wfl_prune == SubCache wfl_prune P={Pn} n={n} m={m} step={step} mix={mix}",
                 torch.equal(a, b) and same_sim, f"{(a != b).sum().item()} differ; same sim: {same_sim}")


if __name__ == "__main__":
    test_select()
    test_top2()
    test_incremental()
    test_vs_subcache()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)
