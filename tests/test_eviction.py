"""Eviction scoring and selection kernels (docs/eviction.md) vs PyTorch references.

Scoring: the attention output, log-sum-exp, logits and per-key scores (H2O mass summed over the
query heads of a kv head; window rows and window mass) of the tracked decode kernels and of the
two-pass prefill, against a float64 softmax. Reports the max abs (and relative) error of each.
Selection: hi_select_k and the WFL prune kernels against PyTorch versions of the same rules
(SubCache policies.wfl_prune, ties to the lower index); decisions must match exactly, and the
incremental rescan must give the same decisions as rescanning every row each round.

Run: python tests/test_eviction.py   (HIPINFER_DECODE_SCALAR=1 HIPINFER_FLASH_WMMA=1 for the
portable attention kernels). Optional: SUBCACHE_DIR=<SubCache checkout> also checks the WFL
reference below against SubCache's own policies.wfl_prune."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import hipinfer as hi  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from eviction_ref import ref_select, ref_sim, ref_top2, ref_unit, ref_wfl_prune  # noqa: E402

torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
FAILS = []
LOG2E = 1.4426950408889634


def check(name, got, ref, atol, rtol=0.0):
    got, ref = got.double(), ref.double()
    err = (got - ref).abs()
    bad = (err > atol + rtol * ref.abs()).sum().item()
    rel = (err / ref.abs().clamp_min(1e-30)).max().item() if err.numel() else 0.0
    print(f"{'OK  ' if bad == 0 else 'FAIL'} {name:60s} max_abs={err.max().item() if err.numel() else 0:.2e} "
          f"max_rel={rel:.2e}")
    if bad:
        FAILS.append(name)


def check_eq(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name:60s} {detail}")
    if not ok:
        FAILS.append(name)


def ref_softmax(q, k, qpos):
    """q [T, Hq, D], k [L, Hkv, D], causal on absolute positions qpos [T]. float64:
    probs [Hq, T, L], log2-domain logits [Hq, T, L] and log-sum-exp [Hq, T]."""
    Hq, Hkv, D = q.shape[1], k.shape[1], q.shape[2]
    kk = k.repeat_interleave(Hq // Hkv, 1).double()
    s = torch.einsum("thd,lhd->htl", q.double(), kk) / math.sqrt(D)
    mask = torch.arange(k.shape[0], device=q.device)[None, :] > qpos[:, None]
    s = s.masked_fill(mask[None], float("-inf"))
    lse = torch.logsumexp(s, -1)
    return (s - lse[..., None]).exp(), s * LOG2E, lse * LOG2E


def group_sum(p, Hkv):  # [Hq, T, L] -> [Hkv, L]
    Hq, T, L = p.shape
    return p.view(Hkv, Hq // Hkv, T, L).sum((1, 2))


# ------------------------------------------------------------------ prefill: LSE + score pass
def test_prefill():
    for D, Hq, Hkv in [(64, 32, 4), (128, 24, 8), (128, 16, 16), (128, 28, 4), (64, 8, 8)]:
        S = 2048
        for T, p0 in [(1, 0), (64, 0), (100, 0), (517, 0), (70, 300), (33, 1000)]:
            L = p0 + T
            tag = f"D={D} G={Hq // Hkv} T={T} p0={p0}"
            q = torch.randn(T, Hq, D, device=dev, dtype=torch.float16)
            kc = torch.randn(Hkv, S, D, device=dev, dtype=torch.float16)
            vc = torch.randn(Hkv, S, D, device=dev, dtype=torch.float16)
            out, lse = hi.flash_prefill_lse(q.view(T, -1), kc, vc, T, p0, Hq, Hkv)
            plain = hi.flash_prefill(q.view(T, -1), kc, vc, T, p0, Hq, Hkv)
            check_eq(f"prefill {tag}: output with lse == without", torch.equal(out, plain))
            p, _, lse_ref = ref_softmax(q, kc[:, :L].transpose(0, 1), torch.arange(p0, L, device=dev))
            check(f"prefill {tag}: lse (log2)", lse, lse_ref, 1e-4, 1e-5)
            W = min(32, T)
            score = torch.zeros(Hkv, S, device=dev)
            wscore = torch.zeros(Hkv, S, device=dev)
            win = torch.zeros(Hq, W, S, device=dev, dtype=torch.float16)
            hi.prefill_scores(q.view(T, -1), kc, T, p0, Hq, Hkv, lse, score, win=win, win_r0=T - W, wscore=wscore)
            ref = group_sum(p, Hkv)
            check(f"prefill {tag}: H2O score", score[:, :L], ref, 1e-6, 1e-4)
            check_eq(f"prefill {tag}: nothing past the cached keys", score[:, L:].abs().max().item() == 0)
            mass = score[:, :L].double().sum(1)
            check(f"prefill {tag}: sum of scores = G*T", mass, torch.full_like(mass, Hq // Hkv * T), 1e-3, 1e-5)
            check(f"prefill {tag}: window score", wscore[:, :L], group_sum(p[:, T - W:], Hkv), 1e-6, 1e-4)
            check(f"prefill {tag}: window rows (fp16)", win[:, :, :L], p[:, T - W:].half(), 1e-3, 1e-3)
            # scores accumulate (+=) and a second call adds the same again
            hi.prefill_scores(q.view(T, -1), kc, T, p0, Hq, Hkv, lse, score)
            check(f"prefill {tag}: second call accumulates", score[:, :L], 2 * ref, 2e-6, 1e-4)


# ------------------------------------------------------------------ decode: tracked split-K
def test_decode():
    cases = [[1], [37], [4000], [1, 300, 1024, 2049], list(range(5, 4000, 250))]
    for D, G in [(64, 8), (64, 4), (64, 1), (128, 4), (128, 3), (128, 1), (128, 7), (128, 8), (64, 2), (128, 6)]:
        Hkv = 4 if G != 1 else 16
        Hq, S = Hkv * G, 4096
        for lens in cases:
            for with_shift in (False, True):
                B = len(lens)
                tag = (f"D={D} G={G} B={B} lens={lens if B <= 4 else f'{lens[0]}..{lens[-1]}'}"
                       f"{' shift' if with_shift else ''}")
                q = torch.randn(B, Hq * D, device=dev, dtype=torch.float16)
                kc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
                vc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
                # cache holds lens[b] keys; with a shift the position is lens[b] - 1 + shift[b]
                shift = torch.randint(0, 50, (B,), device=dev, dtype=torch.int32) if with_shift else None
                pos = torch.tensor([l - 1 for l in lens], device=dev, dtype=torch.int32)
                if with_shift:
                    pos = pos + shift
                score = torch.zeros(B, Hkv, S, device=dev)
                W = 3
                win = torch.zeros(B, Hq, W, S, device=dev, dtype=torch.float16)
                win_row = torch.tensor([(b % (W + 1)) - 1 for b in range(B)], device=dev, dtype=torch.int32)  # -1, 0, 1, 2, -1...
                wscore = torch.zeros(B, Hkv, S, device=dev)
                out, lg, lse = hi.decode_attn_track(q, kc, vc, pos, Hq, Hkv, score, shift=shift, win=win, win_row=win_row,
                                                    wscore=wscore)
                if not with_shift:
                    plain = hi.decode_attn(q, kc, vc, pos, Hq, Hkv)
                    check_eq(f"decode {tag}: output == untracked kernel (bitwise)", torch.equal(out, plain))
                err = dict(out=0.0, lse=0.0, lg=0.0, score=0.0, score_rel=0.0, win=0.0, wscore=0.0)
                bad = []
                for b, Lb in enumerate(lens):
                    qb = q[b].view(1, Hq, D)
                    k = kc[b, :, :Lb].transpose(0, 1)
                    p, s2, lse_ref = ref_softmax(qb, k, torch.tensor([Lb - 1], device=dev))
                    v = vc[b, :, :Lb].transpose(0, 1).repeat_interleave(G, 1).double()
                    o_ref = torch.einsum("htl,lhd->thd", p, v)
                    e_out = (out[b].view(1, Hq, D).double() - o_ref).abs()
                    if (e_out > 2e-3 + 1e-2 * o_ref.abs()).any():
                        bad.append("out")
                    err["out"] = max(err["out"], e_out.max().item())
                    err["lse"] = max(err["lse"], (lse[b].double() - lse_ref[:, 0]).abs().max().item())
                    err["lg"] = max(err["lg"], (lg[b, :, :Lb].double() - s2[:, 0]).abs().max().item())
                    ref = group_sum(p, Hkv)
                    e = (score[b, :, :Lb].double() - ref).abs()
                    if (e > 1e-6 + 1e-4 * ref).any() or score[b, :, Lb:].abs().max().item() != 0:
                        bad.append("score")
                    err["score"] = max(err["score"], e.max().item())
                    err["score_rel"] = max(err["score_rel"], (e / ref.clamp_min(1e-30)).max().item())
                    wr = int(win_row[b])
                    if wr >= 0:
                        err["win"] = max(err["win"], (win[b, :, wr, :Lb].double() - p[:, 0]).abs().max().item())
                        err["wscore"] = max(err["wscore"], (wscore[b, :, :Lb].double() - ref).abs().max().item())
                    elif wscore[b].abs().max().item() != 0 or win[b].abs().max().item() != 0:
                        bad.append("window written for a non-window query")
                if err["lse"] > 1e-4 or err["lg"] > 1e-4 or err["win"] > 1e-3 or err["wscore"] > 1e-5:
                    bad.append("lse/logits/window")
                print(f"{'OK  ' if not bad else 'FAIL'} decode {tag:52s} out={err['out']:.1e} lse={err['lse']:.1e} "
                      f"logit={err['lg']:.1e} score={err['score']:.1e} (rel {err['score_rel']:.1e}) "
                      f"win={err['win']:.1e} wscore={err['wscore']:.1e} {' '.join(bad)}")
                if bad:
                    FAILS.append(f"decode {tag}")


def test_decode_graph():
    """The tracked decode is graph-capturable: replays add the same scores as eager calls."""
    B, Hkv, G, D, S = 2, 4, 4, 128, 2048
    Hq = Hkv * G
    q = torch.randn(B, Hq * D, device=dev, dtype=torch.float16)
    kc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
    vc = torch.randn(B, Hkv, S, D, device=dev, dtype=torch.float16)
    pos = torch.tensor([1500, 700], device=dev, dtype=torch.int32)
    shift = torch.tensor([3, 0], device=dev, dtype=torch.int32)
    out = torch.empty(B, Hq * D, device=dev, dtype=torch.float16)
    work = torch.zeros(hi.lib.hi_decode_work_bytes(B, Hq, Hkv, D), dtype=torch.uint8, device=dev)
    lg = torch.empty(B, Hq, S, device=dev)
    lse = torch.empty(B, Hq, device=dev)
    eager = torch.zeros(B, Hkv, S, device=dev)
    for _ in range(3):
        hi.decode_attn_track(q, kc, vc, pos, Hq, Hkv, eager, shift=shift, out=out, work=work, lg=lg, lse=lse)
    graphed = torch.zeros(B, Hkv, S, device=dev)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        hi.decode_attn_track(q, kc, vc, pos, Hq, Hkv, graphed, shift=shift, out=out, work=work, lg=lg, lse=lse)
    graphed.zero_()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    check_eq("decode tracked: 3 graph replays == 3 eager calls (bitwise)", torch.equal(eager, graphed))


# ------------------------------------------------------------------ selection
def test_select():
    for Pn, n, lo, hi_, k, levels in [(3, 5, 0, 5, 2, 0), (8, 1000, 4, 900, 100, 0), (8, 1000, 4, 900, 100, 7),
                                      (5, 10000, 0, 10000, 1, 0), (5, 10000, 0, 10000, 9999, 3), (16, 4096, 4, 4064, 1000, 50),
                                      (2, 3000, 100, 2000, 1900, 0)]:
        for largest in (True, False):
            score = torch.rand(Pn, n, device=dev)
            if levels:  # many exact ties
                score = (score * levels).floor() / levels
            got = hi.select_k(score, lo, hi_, k, largest)
            ref = ref_select(score, lo, hi_, k, largest)
            check_eq(f"select_k P={Pn} n={n} [{lo},{hi_}) k={k} {'largest' if largest else 'smallest'}"
                     f"{f' ties({levels} levels)' if levels else ''}", torch.equal(got, ref),
                     f"{(got != ref).sum().item()} flags differ")


def wfl_case(Pn, n, D, m, sink, recent, mix, step, rounds, power, pool_extra=0):
    """One random WFL problem set; pool_extra > 0 puts the n ground-set slots among n + pool_extra
    cached ones (gidx), as hi_evict's pool does."""
    Lc = n + pool_extra
    k = torch.randn(Pn, Lc, D, device=dev, dtype=torch.float16)
    v = torch.randn(Pn, Lc, D, device=dev, dtype=torch.float16)
    # some near-copies, so that redundancy matters
    src = torch.randint(0, Lc, (Pn, Lc // 4), device=dev)
    dst = torch.randint(0, Lc, (Pn, Lc // 4), device=dev)
    for p in range(Pn):
        k[p, dst[p]] = (k[p, src[p]].float() + 0.05 * torch.randn_like(k[p, src[p]].float())).half()
        v[p, dst[p]] = (v[p, src[p]].float() + 0.05 * torch.randn_like(v[p, src[p]].float())).half()
    score = torch.rand(Pn, Lc, device=dev) ** 3
    if pool_extra:
        gidx = torch.stack([torch.randperm(Lc, device=dev)[:n].sort().values for _ in range(Pn)]).int()
    else:
        gidx = torch.arange(n, device=dev, dtype=torch.int32).expand(Pn, n).contiguous()
    status = torch.zeros(Pn, n, dtype=torch.uint8, device=dev)
    status[:, :sink] = 1
    if recent:
        status[:, n - recent:] = 1
    g = gidx.long()
    kk = torch.stack([k[p, g[p]] for p in range(Pn)])
    vv = torch.stack([v[p, g[p]] for p in range(Pn)])
    w = torch.stack([score[p, g[p]] for p in range(Pn)])
    w = w / w.sum(1, keepdim=True).clamp_min(1e-9)
    sim = ref_sim(kk, vv, power)
    cand, held = status == 0, status == 1
    tag = (f"P={Pn} n={n} D={D} m={m} mix={mix} step={step} rounds={rounds} power={power}"
           f"{f' pool {n}/{Lc}' if pool_extra else ''}")

    # first round's top-2 state (every row), against the reference
    _, _, work = hi.wfl_prune(k, v, score, status, m, mix, step, rounds, power, max_rounds=0, gidx=gidx)
    best, arg, second = ref_top2(sim, cand | held)
    view = lambda name: hi.wfl_work_view(work, Pn, n, D, name)
    kb, ka, ks = view("best"), view("arg").long(), view("sec")
    check(f"wfl {tag}: normalized keys", view("kn"), ref_unit(kk), 1e-5)
    check(f"wfl {tag}: weights", view("w"), w, 1e-7, 1e-5)
    check(f"wfl {tag}: best", kb, best, 1e-5)
    check(f"wfl {tag}: second (clamped at 0)", ks.clamp_min(0), second.clamp_min(0), 1e-5)
    # an argmax may differ only where another column is within float noise of the best
    near = (sim.masked_fill(~(cand | held)[:, None, :], -1.0) >= best[..., None] - 2e-6).sum(-1) > 1
    check_eq(f"wfl {tag}: best index (exact unless a near-tie)", bool(((ka == arg) | near).all()),
             f"{(ka != arg).sum().item()} differ, {near.sum().item()} rows with a near-tie")

    got, rounds_run, _ = hi.wfl_prune(k, v, score, status, m, mix, step, rounds, power, gidx=gidx)
    full, _, _ = hi.wfl_prune(k, v, score, status, m, mix, step, rounds, power, incremental=False, gidx=gidx)
    ref = ref_wfl_prune(sim, w, cand.clone(), held, m, mix, step, rounds)
    check_eq(f"wfl {tag}: kept set == reference", torch.equal(got, ref),
             f"{rounds_run} rounds, {(got != ref).sum().item()} slots differ, kept {got.sum(1).tolist()[:4]}...")
    check_eq(f"wfl {tag}: incremental rescan == full rescan", torch.equal(got, full), f"{(got != full).sum().item()} differ")


def test_wfl():
    wfl_case(4, 64, 64, 20, 4, 8, 0.5, 8, 0, 1)
    wfl_case(4, 64, 128, 1, 4, 8, 0.5, 1, 0, 1)
    wfl_case(8, 200, 128, 60, 4, 32, 0.5, 8, 0, 1)
    wfl_case(8, 200, 128, 60, 4, 32, 0.5, 8, 0, 2)
    wfl_case(8, 200, 64, 60, 4, 32, 0.0, 8, 0, 1)   # mix 0: plain H2O order
    wfl_case(8, 200, 64, 60, 4, 32, 1.0, 8, 0, 1)   # mix 1: facility location only
    wfl_case(6, 513, 128, 100, 4, 64, 0.5, 8, 3, 1)  # rounds cap: ~117 removed per round
    wfl_case(6, 513, 128, 400, 0, 0, 0.5, 1, 0, 1)   # no protected slots, one per round
    wfl_case(4, 300, 128, 80, 4, 32, 0.5, 8, 0, 1, pool_extra=700)
    wfl_case(2, 1500, 128, 300, 4, 64, 0.5, 64, 0, 1)


def test_wfl_vs_subcache():
    """The reference above against SubCache's own policies.wfl_prune (only with SUBCACHE_DIR)."""
    root = os.environ.get("SUBCACHE_DIR")
    if not root:
        print("skip wfl reference vs SubCache policies.wfl_prune (set SUBCACHE_DIR)")
        return
    sys.path.insert(0, root)
    from subcache import policies
    for Pn, n, m, step, rounds in [(4, 64, 20, 8, None), (8, 200, 60, 8, None), (4, 513, 100, 8, 3)]:
        k = torch.randn(Pn, n, 128, device=dev)
        v = torch.randn(Pn, n, 128, device=dev)
        w = torch.rand(Pn, n, device=dev) ** 3
        w = w / w.sum(1, keepdim=True)
        held = torch.zeros(Pn, n, dtype=torch.bool, device=dev)
        held[:, :4] = held[:, -16:] = True
        sim = policies.slot_similarity(k, v)
        a = policies.wfl_prune(sim, w, ~held, held, m, step=step, rounds=rounds)
        b = ref_wfl_prune(sim, w, ~held, held, m, step=step, rounds=rounds)
        check_eq(f"wfl reference == SubCache wfl_prune P={Pn} n={n} m={m}", torch.equal(a, b), f"{(a != b).sum().item()} differ")


if __name__ == "__main__":
    print(torch.cuda.get_device_name(), "| decode kernel:",
          "scalar" if os.environ.get("HIPINFER_DECODE_SCALAR") else "mma (NVIDIA) / scalar (AMD)",
          "| prefill kernel:", "wmma" if os.environ.get("HIPINFER_FLASH_WMMA") else "mma (NVIDIA) / wmma (AMD)")
    test_prefill()
    test_decode()
    test_decode_graph()
    test_select()
    test_wfl()
    test_wfl_vs_subcache()
    print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
    sys.exit(1 if FAILS else 0)
