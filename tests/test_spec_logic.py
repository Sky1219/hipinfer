"""CPU test of the speculative round loop (scripts/speculative.py) with stub engines: no GPU, no
libhipinfer.so. tests/test_spec.py checks the real engines on a GPU.

The stubs keep a per-position token cache with the same contract as the engine (prefill,
decode steps that consume d_tok at d_pos, verify rows at p0.., rewind = set the cache length)
and compute each "model" output as a deterministic function of the cached prefix. A stale cache
entry therefore changes the stub's output, as it would a real model's. On top of that, every
rewind, every verify and every first draft step asserts that the positions the caller relies on
hold the true sequence (the target's own greedy output), so a rollback mistake fails loudly even
when it would only have cost acceptance.

Checked, for many draft agreement rates, k and lengths: the output equals the target's plain
greedy output; acceptance is 1 for a draft that always agrees and 0 for one that never does;
the counts add up (tokens = 1 + accepted + rounds); verify never gets more than k + 1 rows; a
draft catch-up step happens exactly after each fully accepted round; and the target is left
rewound to the pending token.

python tests/test_spec_logic.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import speculative  # noqa: E402

V = 50  # small vocabulary: plenty of repeated contexts


def mix(*xs):
    h = 1469598103934665603
    for x in xs:
        h = ((h ^ (x + 0x9E37)) * 1099511628211) % (1 << 61)
    return h


def target_fn(prefix):
    return mix(1, *prefix[-4:], len(prefix) % 7) % V


def make_draft_fn(agree, seed):
    """Agrees with target_fn on a deterministic pseudo-random `agree` fraction of contexts."""
    def f(prefix):
        t = target_fn(prefix)
        if (mix(seed, *prefix[-4:], len(prefix)) % 10000) < agree * 10000:
            return t
        return (t + 1 + mix(seed + 1, *prefix[-3:]) % (V - 1)) % V
    return f


class StubEngine:
    """Slot 0 only. cache[i] = token fed at position i (None: never written)."""

    def __init__(self, fn, truth, max_seq=4096, vocab=V):
        self.fn, self.truth, self.max_seq, self.vocab = fn, truth, max_seq, vocab
        self.cache = [None] * max_seq
        self.pos = self.tok = None
        self.hist = [None] * max_seq
        self.calls = []
        self.fresh = False  # the next decode step is the first after a rewind / prefill

    def _check_prefix(self, n, what):
        got = self.cache[:n]
        want = self.truth()[:n]
        assert got == want, f"{what}: cache prefix of length {n} is stale:\n {got}\n {want}"

    def prefill(self, slot, tokens, p0=0):
        assert slot == 0 and p0 == 0
        self.cache[:len(tokens)] = tokens
        self.pos, self.tok = len(tokens), self.fn(list(tokens))
        self.hist[self.pos] = self.tok
        self.fresh = True
        self.calls.append(("prefill", len(tokens)))
        return self.tok

    def rewind(self, slot, length, tok):
        assert slot == 0 and 0 <= length < self.max_seq and 0 <= tok < self.vocab
        self._check_prefix(length, "rewind")
        assert tok == self.truth()[length], "rewind: the next token is not the true one"
        self.pos, self.tok, self.fresh = length, tok, True
        self.calls.append(("rewind", length))

    def decode(self, B, steps, graph=True):
        assert B == 1 and steps >= 0 and self.pos + steps <= self.max_seq
        for i in range(steps):
            if i == 0 and self.fresh:
                self._check_prefix(self.pos, "decode")
            self.cache[self.pos] = self.tok
            self.tok = self.fn(self.cache[:self.pos + 1])
            self.pos += 1
            self.hist[self.pos] = self.tok
        self.fresh = False
        self.calls.append(("decode", steps))
        return 0.0

    def history(self, slot, start, n):
        return self.hist[start:start + n]

    def verify(self, slot, rows, p0, graph=True):
        assert slot == 0 and 1 <= len(rows) and p0 + len(rows) <= self.max_seq
        self._check_prefix(p0, "verify")
        assert rows[0] == self.truth()[p0], "verify: row 0 is not the pending token"
        ys = []
        for i, t in enumerate(rows):
            self.cache[p0 + i] = t
            ys.append(self.fn(self.cache[:p0 + i + 1]))
        self.calls.append(("verify", len(rows)))
        return ys


def plain_greedy(prompt, n):
    seq = list(prompt)
    for _ in range(n):
        seq.append(target_fn(seq))
    return seq[len(prompt):]


def run_case(prompt, n, k, agree, seed):
    ref = plain_greedy(prompt, n + 20)
    truth = lambda: list(prompt) + ref  # noqa: E731
    target = StubEngine(target_fn, truth)
    draft = StubEngine(make_draft_fn(agree, seed), truth)
    st = {}
    got = speculative.generate(target, draft, prompt, n, k, stats=st)
    assert got == ref[:n], f"output differs (n={n} k={k} agree={agree}):\n {got}\n {ref[:n]}"
    assert st["tokens"] == n == 1 + st["accepted"] + st["rounds"], st
    assert all(r <= k + 1 for c, r in target.calls if c == "verify")
    assert sum(1 for c, _ in target.calls if c == "verify") == st["rounds"]
    if agree == 1.0 and st["drafted"]:
        assert st["accepted"] == st["drafted"], st
    if agree == 0.0:
        assert st["accepted"] == 0, st
    assert target.pos == len(prompt) + n - 1 and target.tok == ref[n - 1], "target not left at the pending token"
    return st


def main():
    rng = random.Random(0)
    cases = 0
    for agree in (0.0, 0.3, 0.6, 0.8, 0.95, 1.0):
        for k in (1, 2, 3, 4, 6, 8, 15):
            for n in (1, 2, 3, 5, 17, 64):
                for seed in range(3):
                    prompt = [rng.randrange(V) for _ in range(rng.randrange(1, 12))]
                    run_case(prompt, n, k, agree, seed)
                    cases += 1
    # a draft that always agrees: every round but the last yields k + 1 tokens
    st = run_case([3, 1, 4, 1, 5], 41, 4, 1.0, 0)
    assert st["rounds"] == 8 and st["catchups"] == 7, st  # 1 + 8 * 5 = 41 tokens; a catch-up before rounds 2..8
    # a draft that never agrees: one token per round, no catch-ups
    st = run_case([2, 7, 1], 30, 6, 0.0, 0)
    assert st["rounds"] == 29 and st["catchups"] == 0, st
    print(f"{cases + 2} stub runs: output identical to plain greedy, caches never stale, counts consistent")
    print("SPEC LOGIC PASS")


if __name__ == "__main__":
    main()
