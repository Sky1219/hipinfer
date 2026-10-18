"""Eviction inside the engine (docs/eviction.md), on a real model.

  1. Tracking must not change outputs: greedy tokens and final logits with tracking on equal
     tracking off, bitwise (prefill over two chunks, then decode with and without a graph).
  2. Score invariants: every query's softmax sums to 1, so per (layer, kv head) the H2O scores
     sum to G * (queries so far) and the window scores to G * window. Graph and eager decode give
     bitwise equal scores.
  3. hi_evict decisions: H2O / window / WFL (with and without a pool) keep exactly the slots that
     PyTorch computes from the engine's own scores, keys and values (same rules as
     tests/test_eviction.py), and the kept keys, values and scores move to the front intact.
  4. Decode after eviction writes the new token's K / V at cache index position - evicted, and
     the scores keep summing as in 2.

Usage: python tests/test_eviction_engine.py MODEL.bin [--prompt 700] [--capacity 256]"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
import hipinfer as hi  # noqa: E402
from test_eviction import check, check_eq, FAILS  # noqa: E402
from eviction_ref import ref_sim, ref_wfl_prune  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--prompt", type=int, default=700)
ap.add_argument("--steps", type=int, default=40)
ap.add_argument("--capacity", type=int, default=256)
ap.add_argument("--sink", type=int, default=4)
ap.add_argument("--recent", type=int, default=32)
ap.add_argument("--window", type=int, default=32)
args = ap.parse_args()

torch.manual_seed(0)
e = hi.Engine(args.model, max_batch=1, max_seq=2048, max_prefill=512)
G = e.heads // e.kv_heads
prompt = torch.randint(100, e.vocab - 100, (args.prompt,)).tolist()
print(f"{os.path.basename(args.model)}: layers {e.layers}, heads {e.heads}/{e.kv_heads}, head_dim {e.head_dim}; "
      f"prompt {args.prompt} tokens (max_prefill 512: two chunks), {args.steps} decode steps")


def run(track, graph):
    if track:
        e.set_tracking(True, args.window)
    first = e.prefill(0, prompt)
    e.decode(1, args.steps, graph)
    toks = [first] + e.history(0, len(prompt) + 1, args.steps)
    return toks, e.logits(1).clone()


# 1. tracking does not change outputs
t0, l0 = run(False, True)
t1, l1 = run(True, True)
check_eq("tokens: tracking on == off (graph decode)", t0 == t1)
check_eq("final logits: tracking on == off (bitwise)", torch.equal(l0, l1))

# 2. invariants, graph vs eager
n = e.cache_len(0)
Q = args.prompt + args.steps
check_eq("cache length = prompt + decode steps", n == Q, f"{n} vs {Q}")
graph_scores = [e.scores(l, 0, n) for l in range(e.layers)]
for l in [0, e.layers // 2, e.layers - 1]:
    s = graph_scores[l].double().sum(1)
    check(f"layer {l}: sum of H2O scores = G * queries", s, torch.full_like(s, G * Q), 1e-2, 1e-5)
    ws = e.scores(l, 0, n, window=True).double().sum(1)
    check(f"layer {l}: sum of window scores = G * window", ws, torch.full_like(ws, G * args.window), 1e-4, 1e-5)
_, _ = run(True, False)
eager_scores = [e.scores(l, 0, n) for l in range(e.layers)]
check_eq("scores: graph decode == eager decode (bitwise)", all(torch.equal(a, b) for a, b in zip(graph_scores, eager_scores)))


# 3. eviction decisions
def snapshot():
    n = e.cache_len(0)
    S = torch.stack([e.scores(l, 0, n) for l in range(e.layers)])          # [L, hkv, n]
    WS = torch.stack([e.scores(l, 0, n, window=True) for l in range(e.layers)])
    KV = [e.cache_read(l, 0, n) for l in range(e.layers)]
    K = torch.stack([kv[0] for kv in KV])                                   # [L, hkv, n, D]
    V = torch.stack([kv[1] for kv in KV])
    return n, S, WS, K, V


def ref_keep(policy, n, S, WS, K, V, cap, sink, recent, mix=0.5, step=8, pool=0.0):
    """Kept slot indices [L, hkv, cap] by the rules of hi_evict / SubCache select_keep."""
    sink = min(sink, cap)
    recent = min(recent, cap - sink)
    m = cap - sink - recent
    Pn = e.layers * e.kv_heads
    slot = torch.arange(n, device=S.device)
    prot = (slot < sink) | (slot >= n - recent)
    sc = (WS if policy == "window" else S).reshape(Pn, n)
    if policy in ("h2o", "window"):
        sub = sc[:, sink:n - recent]
        sel = torch.sort(-sub, dim=1, stable=True).indices[:, :m] + sink
        keep = prot.expand(Pn, n).clone()
        keep.scatter_(1, sel, True)
    else:
        ncand = n - sink - recent
        kpool = min(int(pool * m), ncand) if pool > 0 else ncand
        h2o = S.reshape(Pn, n)
        keep = torch.zeros(Pn, n, dtype=torch.bool, device=S.device)
        for p in range(Pn):
            if kpool < ncand:
                sel = torch.sort(-h2o[p, sink:n - recent], stable=True).indices[:kpool] + sink
                ground = torch.cat([slot[prot], sel]).sort().values
            else:
                ground = slot
            k = K.reshape(Pn, n, -1)[p, ground].float()[None]
            v = V.reshape(Pn, n, -1)[p, ground].float()[None]
            w = h2o[p, ground][None]
            w = w / w.sum().clamp_min(1e-9)
            held = prot[ground][None]
            kept = ref_wfl_prune(ref_sim(k, v, 1), w, ~held, held, m, mix, step)
            keep[p, ground[(kept | held)[0]]] = True
    idx = keep.float().topk(cap, dim=1).indices.sort(dim=1).values
    return idx.view(e.layers, e.kv_heads, cap).cpu()


cases = [("h2o", {}), ("window", {}), ("wfl", {}), ("wfl", {"pool": 2.0})]
for policy, kw in cases:
    run(True, True)  # fresh sequence, same state each time
    n, S, WS, K, V = snapshot()
    evicted, kept, ms = e.evict(0, args.capacity, policy, sink=args.sink, recent=args.recent, **kw)
    tag = f"{policy}{' pool ' + str(kw['pool']) if kw else ''} {n} -> {args.capacity}"
    check_eq(f"evict {tag}: evicted count", evicted == n - args.capacity, f"{evicted}")
    ref = ref_keep(policy, n, S, WS, K, V, args.capacity, args.sink, args.recent, **kw)
    check_eq(f"evict {tag}: kept slots == PyTorch from the engine's state", torch.equal(kept, ref),
             f"{(kept != ref).sum().item()} of {kept.numel()} differ; select {ms['select_ms']:.2f} ms, "
             f"compact {ms['compact_ms']:.2f} ms")
    # the kept rows moved to the front intact
    n2, S2, WS2, K2, V2 = snapshot()
    kd = kept.to(K.device)
    idx = kd[..., None].expand(-1, -1, -1, K.shape[-1])
    check_eq(f"evict {tag}: cache length", n2 == args.capacity)
    check_eq(f"evict {tag}: keys / values moved intact", torch.equal(K2, K.gather(2, idx)) and torch.equal(V2, V.gather(2, idx)))
    check_eq(f"evict {tag}: scores moved intact", torch.equal(S2, S.gather(2, kd)) and torch.equal(WS2, WS.gather(2, kd)))
    # 4. decode after eviction
    mass_before = S2.double().sum(-1)
    e.decode(1, 8, True)
    n3, S3, _, K3, _ = snapshot()
    check_eq(f"evict {tag}: decode appends at index position - evicted", n3 == args.capacity + 8
             and torch.equal(K3[:, :, :args.capacity], K2) and bool((K3[:, :, args.capacity:].abs().sum(-1) > 0).all()))
    check(f"evict {tag}: 8 more steps add 8 * G of score mass", S3.double().sum(-1) - mass_before,
          torch.full_like(mass_before, 8 * G), 1e-3, 1e-5)

print(f"tokens after the last eviction: {e.history(0, args.prompt + args.steps + 1, 8)}")
print("\nALL PASS" if not FAILS else f"\n{len(FAILS)} FAILURES: {FAILS}")
sys.exit(1 if FAILS else 0)
