"""CPU check of scripts/profile_decode.py's summarizer on a synthetic Nsight Systems SQLite export
with known answers: every GEMV takes t0 + bytes / BW, fixed gaps between kernels and between
steps, one kernel outside the NVTX range. No GPU or nsys needed.

python tests/test_profile_decode.py
"""
import json
import os
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import profile_decode as pd  # noqa: E402

CFG = {"vocab": 1000, "hidden": 256, "inter": 512, "layers": 3, "heads": 4, "kv_heads": 2, "head_dim": 64,
       "tied": False, "qkv_bias": False}
T0, BW, GAP, STEP_GAP, ATTN, MISC, STEPS, CTX = 2000, 1000.0, 1500, 4000, 3000, 1000, 12, 128  # ns, GB/s


def build(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    con.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, shortName INT, demangledName INT)")
    con.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT, eventType INT)")
    names = ["embed_kernel", "gemv_kernel", "decode_attn_mma_kernel", "argmax_kernel"]
    con.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    gb = pd.gemv_bytes(CFG)
    t = 10_000
    con.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, 1, 1)", (t, t + 500))  # before the range
    r0 = t = t + 5000
    kernels = []
    for _ in range(STEPS):
        seq = [(0, MISC)]
        for _ in range(CFG["layers"]):
            seq += [(1, T0 + gb["qkv"] / BW), (2, ATTN), (1, T0 + gb["o"] / BW), (1, T0 + gb["gate_up"] / BW),
                    (1, T0 + gb["down"] / BW)]
        seq += [(1, T0 + gb["lm_head"] / BW), (3, MISC)]
        for j, (n, d) in enumerate(seq):
            kernels.append((t, t + round(d), n, n))
            t += round(d) + (GAP if j < len(seq) - 1 else STEP_GAP)
    con.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)", kernels)
    con.execute("INSERT INTO NVTX_EVENTS VALUES (?, ?, 'decode_graph', NULL, 59)", (r0 - 100, t))
    con.commit()
    con.close()
    return gb


def main():
    with tempfile.TemporaryDirectory() as d:
        db = os.path.join(d, "t.sqlite")
        gb = build(db)
        weight = sum(gb[r] for r in pd.GEMV_ROLES) * CFG["layers"] + gb["lm_head"]
        meta = {"gpu": "fake", "weights": "x.bin", "ctx": CTX, "peak_bw_GBs": 2 * BW, "read_bw_GBs": 1.5 * BW,
                "graph_ms": 0.1, "nograph_ms": 0.12, "config": CFG, "floor_ms": weight / (2 * BW) / 1e6}
        json.dump(meta, open(os.path.join(d, "m.json"), "w"))

        class A:
            sqlite, meta, traced_meta, out = db, os.path.join(d, "m.json"), None, os.path.join(d, "s.json")
        pd.summarize(A)
        s = json.load(open(A.out))["decode_graph"]

    L = CFG["layers"]
    n_k = 5 * L + 3
    close = lambda a, b, tol=1e-3: abs(a - b) <= tol * max(1.0, abs(b))
    assert s["kernels_per_step"] == n_k and s["steps_analyzed"] == STEPS - 1, s
    assert s["gemv_roles_by_launch_order"]
    gap_ms = ((n_k - 1) * GAP + STEP_GAP) / 1e6
    assert close(s["gap_ms"], gap_ms), (s["gap_ms"], gap_ms)
    assert close(s["gaps"]["between_steps_ms"], STEP_GAP / 1e6)
    assert close(s["gaps"]["median_us"], GAP / 1e3)
    busy = (2 * MISC + L * ATTN + (4 * L + 1) * T0 + weight / BW) / 1e6
    assert close(s["kernel_sum_ms"], busy, 1e-3), (s["kernel_sum_ms"], busy)
    assert close(s["wall_ms"], busy + gap_ms, 1e-3)
    assert close(s["gemv"]["down"]["median_GBs"], gb["down"] / (T0 + gb["down"] / BW), 1e-3)
    assert close(s["gemv_fit"]["t0_us"], T0 / 1e3, 1e-2), s["gemv_fit"]
    assert close(s["gemv_fit"]["stream_GBs"], BW, 1e-2), s["gemv_fit"]
    sf = s["shortfall"]
    assert close(sf["gaps_share"], gap_ms / (busy + gap_ms - meta["floor_ms"]), 1e-3)
    kv = sum(2 * CFG["kv_heads"] * CFG["head_dim"] * 2 * (CTX + i + 1) for i in range(STEPS - 1)) / (STEPS - 1)
    attn_over = L * (ATTN - kv / meta["peak_bw_GBs"])
    assert close(sf["fixed_in_kernel_ms"], ((4 * L + 1) * T0 + 2 * MISC + attn_over) / 1e6, 1e-2), sf
    print("OK   profile summary: counts, gaps, per-role bandwidth, t0/BW fit and shortfall split match the synthetic trace")


if __name__ == "__main__":
    main()
