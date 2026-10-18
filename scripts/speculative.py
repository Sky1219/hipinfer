"""Greedy speculative decoding round loop (docs/speculative.md), pure Python so that
tests/test_spec_logic.py can drive it with stub engines on the CPU. Engine.generate_speculative
(scripts/hipinfer.py) calls generate(). Engine methods used, all on slot 0:

  target: prefill(slot, tokens) -> next token; verify(slot, rows, p0, graph) -> argmax after each
          row; prefill_logits(slot, rows, p0) (verify="prefill" only); rewind(slot, length, tok)
  draft:  prefill(slot, tokens); rewind(slot, length, tok); decode(B, steps, graph);
          history(slot, start, n)
"""
import time


def generate(target, draft, prompt, n_new, k, graph=True, stats=None, verify="decode", max_rows=16):
    """n_new greedy tokens of `target` after `prompt`, the same as plain greedy decoding gives,
    with `draft` proposing up to k tokens per round. Each round:
      1. the draft, rewound to the pending token t0 at position p, runs k greedy steps -> d1..dk;
      2. one target verify pass over [t0, d1..dk] at p gives the target's argmax y0..yk after each;
      3. the longest prefix with d(i+1) == y(i) is accepted, plus the target's own next token;
      4. both caches are rolled back past the rejected tokens by setting their lengths.
    stats (a dict) receives counts and host timings."""
    assert n_new >= 1 and 1 <= k < max_rows and verify in ("decode", "prefill")
    t_start = time.perf_counter()
    seq = list(prompt) + [target.prefill(0, prompt)]  # seq[-1]: the pending token, in neither cache yet
    out = [seq[-1]]
    draft.prefill(0, prompt)
    dlen = len(prompt)  # the draft's cache holds seq[:dlen]
    t_draft = t_verify = 0.0
    rounds = drafted = accepted = catchups = 0
    t_loop = time.perf_counter()
    while len(out) < n_new:
        p = len(seq) - 1  # position of the pending token
        kk = min(k, n_new - len(out) - 1)  # no point drafting past the last token wanted
        t0 = time.perf_counter()
        props = []
        if kk > 0:
            for q in range(dlen, p):  # tokens the draft never consumed (the last one of a fully accepted round)
                draft.rewind(0, q, seq[q])
                draft.decode(1, 1, graph)
                catchups += 1
            draft.rewind(0, p, seq[p])
            draft.decode(1, kk, graph)
            props = draft.history(0, p + 1, kk)
        t1 = time.perf_counter()
        rows = [seq[p]] + props
        if verify == "decode":
            ys = target.verify(0, rows, p, graph)
        else:
            ys = target.prefill_logits(0, rows, p).argmax(-1).tolist()
        t2 = time.perf_counter()
        a = 0
        while a < kk and props[a] == ys[a]:
            a += 1
        new = props[:a] + [ys[a]]
        seq += new
        out += new
        if kk > 0:  # the draft consumed seq[p] and props[:kk-1]; entries past the accepted prefix are stale
            dlen = p + 1 + min(a, kk - 1)
        rounds += 1
        drafted += kk
        accepted += a
        t_draft += t1 - t0
        t_verify += t2 - t1
    p = len(seq) - 1
    target.rewind(0, p, seq[p])  # leave the target where plain decoding would continue
    t_end = time.perf_counter()
    if stats is not None:
        stats.update(rounds=rounds, drafted=drafted, accepted=accepted, catchups=catchups, tokens=len(out),
                     acceptance=accepted / drafted if drafted else float("nan"),
                     tokens_per_round=(len(out) - 1) / rounds if rounds else float("nan"),
                     prefill_s=t_loop - t_start, loop_s=t_end - t_loop, draft_s=t_draft, verify_s=t_verify)
    return out
