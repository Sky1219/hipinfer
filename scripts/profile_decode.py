"""Batch-1 decode under Nsight Systems: where a token's time goes, kernel by kernel.

  run        Load the engine, prefill `ctx` random tokens, then decode `steps` tokens from the CUDA
             graph and `steps // 2` without it, each inside an NVTX range (decode_graph,
             decode_nograph). Writes the engine's own ms/token, the model shape and a measured DRAM
             read bandwidth to --out. Run it once plainly (the reference timing) and once under
             `nsys profile --trace=cuda,nvtx --cuda-graph-trace=node` (scripts/profile_decode.sh).
  summarize  Read the nsys SQLite export and report, per decode step inside an NVTX range:
             - kernel count, the sum of kernel time, and the step's wall time (first kernel start to
               the next step's first kernel start);
             - the gaps between kernels (inside a step, and between steps), with their distribution;
             - per GEMV role (qkv, o, gate_up, down, lm_head) the median time and achieved bandwidth,
               bytes = the weight matrix; decode attention's bandwidth from the KV bytes it reads;
             - a fit of GEMV time = t0 + bytes / BW over every GEMV in the step: t0 is the fixed cost
               a kernel pays whatever its size (fill, tail), BW the streaming rate;
             - the shortfall against the bandwidth floor, split into gaps, fixed in-kernel cost and
               streaming below peak.

python scripts/profile_decode.py run --weights tinyllama.bin --out prof/plain.json
python scripts/profile_decode.py summarize --sqlite prof/decode.sqlite --meta prof/plain.json \
    --traced-meta prof/under_nsys.json --out prof/summary.json
"""
import argparse
import json
import os
import sqlite3
import statistics
import sys

GEMV_ROLES = ("qkv", "o", "gate_up", "down")  # per layer, in decode_step's launch order


# ------------------------------------------------------------------------------------------- run

def read_bandwidth(torch, gib=2, iters=10):
    """GB/s of a read-only sweep (sum over a large fp16 tensor) and of a device-to-device copy:
    the practical ceilings next to the datasheet figure."""
    x = torch.empty(gib * 2**30 // 2, dtype=torch.float16, device="cuda").normal_()
    y = torch.empty_like(x)

    def best(fn):
        fn()
        ts = []
        for _ in range(iters):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); fn(); b.record(); b.synchronize()
            ts.append(a.elapsed_time(b))
        return min(ts)
    read = x.numel() * 2 / best(lambda: x.sum(dtype=torch.float32)) / 1e6
    copy = 2 * x.numel() * 2 / best(lambda: y.copy_(x)) / 1e6
    del x, y
    torch.cuda.empty_cache()
    return read, copy


def run(a):
    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    import hipinfer as hi
    from bench import peak_bw

    eng = hi.Engine(a.weights, max_batch=1, max_seq=a.max_seq, max_prefill=a.ctx)
    assert eng.n_experts == 0, "dense models only: MoE decode has a different kernel sequence"
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(100, min(30000, eng.vocab - 1), (a.ctx,), generator=g).tolist()
    # warm up: capture the graph, first touches
    eng.prefill(0, ids); eng.decode(1, 8, True)
    eng.prefill(0, ids); eng.decode(1, 8, False)

    def timed(graph, steps, name):
        eng.prefill(0, ids)
        torch.cuda.synchronize()
        if a.nvtx:
            torch.cuda.nvtx.range_push(name)
        ms = eng.decode(1, steps, graph)
        if a.nvtx:
            torch.cuda.nvtx.range_pop()
        return ms / steps

    res = {"gpu": torch.cuda.get_device_name(), "peak_bw_GBs": peak_bw(), "weights": a.weights,
           "ctx": a.ctx, "steps": a.steps, "nograph_steps": a.steps // 2,
           "config": {k: getattr(eng, k) for k in ("vocab", "hidden", "inter", "layers", "heads", "kv_heads",
                                                    "head_dim", "tied", "qkv_bias")},
           "weight_bytes": eng.weight_bytes, "kv_bytes_per_token": eng.kv_bytes_per_token}
    if a.nvtx:  # under the profiler: one ranged run of each, the trace is the measurement
        res["graph_ms"] = timed(True, a.steps, "decode_graph")
        res["nograph_ms"] = timed(False, a.steps // 2, "decode_nograph")
    else:  # reference: best of 3, like scripts/bench.py
        res["graph_ms"] = min(timed(True, a.steps, "") for _ in range(3))
        res["nograph_ms"] = min(timed(False, a.steps // 2, "") for _ in range(3))
    eng.close()
    if not a.nvtx:
        res["read_bw_GBs"], res["copy_bw_GBs"] = read_bandwidth(torch)
    kv = eng.kv_bytes_per_token * (a.ctx + a.steps // 2)
    res["floor_ms"] = (eng.weight_bytes + kv) / res["peak_bw_GBs"] / 1e6
    print(f"{res['gpu']}: graph {res['graph_ms']:.3f} ms/token, no graph {res['nograph_ms']:.3f}, "
          f"floor {res['floor_ms']:.3f} ms -> {res['floor_ms'] / res['graph_ms'] * 100:.0f}% of peak BW"
          + (f"; measured read {res['read_bw_GBs']:.0f} GB/s, copy {res['copy_bw_GBs']:.0f} GB/s" if not a.nvtx else ""))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)


# ------------------------------------------------------------------------------------- summarize

def load_trace(path):
    """Kernels [(start, end, short name, full name)] sorted by start, NVTX ranges {name: (start, end)}."""
    con = sqlite3.connect(path)
    q = lambda sql: con.execute(sql).fetchall()
    tables = {r[0] for r in q("SELECT name FROM sqlite_master WHERE type='table'")}
    cols = {r[1] for r in q("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    short = "shortName" if "shortName" in cols else "demangledName"
    full = "demangledName" if "demangledName" in cols else short
    kernels = q(f"SELECT k.start, k.end, s.value, f.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                f"JOIN StringIds s ON s.id = k.{short} JOIN StringIds f ON f.id = k.{full} ORDER BY k.start")
    ranges = {}
    if "NVTX_EVENTS" in tables:
        ncols = {r[1] for r in q("PRAGMA table_info(NVTX_EVENTS)")}
        text, join = ("COALESCE(n.text, s.value)", "LEFT JOIN StringIds s ON s.id = n.textId") if "textId" in ncols \
            else ("n.text", "")
        for start, end, name in q(f"SELECT n.start, n.end, {text} FROM NVTX_EVENTS n {join} "
                                  f"WHERE n.end IS NOT NULL AND n.end > n.start"):
            if name:
                ranges.setdefault(name, (start, end))
    con.close()
    return kernels, ranges


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else None


def fit_line(xs, ys):
    """Least squares y = a + b x."""
    n, mx, my = len(xs), statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def gemv_bytes(c):
    H, I, D, V = c["hidden"], c["inter"], c["head_dim"], c["vocab"]
    return {"qkv": (c["heads"] + 2 * c["kv_heads"]) * D * H * 2, "o": H * c["heads"] * D * 2,
            "gate_up": 2 * I * H * 2, "down": H * I * 2, "lm_head": V * H * 2}


def summarize_range(kernels, meta, ctx0):
    """Per-step numbers for the kernels of one NVTX range (ns timestamps)."""
    c, peak = meta["config"], meta["peak_bw_GBs"]
    L = c["layers"]
    steps, cur = [], None
    for k in kernels:
        if k[2].startswith("embed_kernel"):
            cur = []
            steps.append(cur)
        if cur is not None:
            cur.append(k)
    if len(steps) < 3:
        return {"error": f"only {len(steps)} decode steps found (no embed_kernel?)"}
    counts = [len(s) for s in steps]
    n_k = statistics.mode(counts)
    full = steps[:-1]  # every step but the last has a successor, so a full period
    period = [steps[i + 1][0][0] - s[0][0] for i, s in enumerate(full)]
    busy = [sum(k[1] - k[0] for k in s) for s in full]
    intra_gaps = [s[j + 1][0] - s[j][1] for s in full for j in range(len(s) - 1)]
    inter_gaps = [steps[i + 1][0][0] - s[-1][1] for i, s in enumerate(full)]
    gaps_by_next = {}
    for s in full:
        for j in range(len(s) - 1):
            gaps_by_next.setdefault(s[j + 1][2], []).append(s[j + 1][0] - s[j][1])
    n = len(full)
    wall_ms = statistics.fmean(period) / 1e6
    busy_ms = statistics.fmean(busy) / 1e6
    gap_ms = wall_ms - busy_ms

    # GEMV roles by launch order: [qkv, o, gate_up, down] per layer, then lm_head
    gb = gemv_bytes(c)
    roles = {r: [] for r in (*GEMV_ROLES, "lm_head")}
    attn, other = [], {}
    role_order_ok = True
    for i, s in enumerate(full):
        gemvs = [k for k in s if "gemv" in k[2]]
        if len(gemvs) != 4 * L + 1:
            role_order_ok = False
            continue
        for j, k in enumerate(gemvs):
            roles["lm_head" if j == 4 * L else GEMV_ROLES[j % 4]].append(k[1] - k[0])
        pos = ctx0 + i  # the position this step writes; attention reads pos + 1 cached tokens
        kv = 2 * c["kv_heads"] * c["head_dim"] * 2 * (pos + 1)
        attn += [(k[1] - k[0], kv) for k in s if "decode_attn" in k[2]]
        for k in s:
            if "gemv" not in k[2] and "decode_attn" not in k[2]:
                other.setdefault(k[2], []).append(k[1] - k[0])

    per_role, fx, fy = {}, [], []
    for r, ds in roles.items():
        if not ds:
            continue
        per_step = 1 if r == "lm_head" else L
        bw = [gb[r] / d for d in ds]  # bytes / ns = GB/s
        per_role[r] = {"bytes": gb[r], "per_step": per_step, "median_us": statistics.median(ds) / 1e3,
                       "median_GBs": statistics.median(bw), "pct_peak": statistics.median(bw) / peak * 100,
                       "ms_per_step": statistics.fmean(ds) * per_step / 1e6}
        if "read_bw_GBs" in meta:
            per_role[r]["pct_measured_read"] = statistics.median(bw) / meta["read_bw_GBs"] * 100
        fx += [gb[r]] * len(ds)
        fy += ds
    res = {"steps_analyzed": n, "kernels_per_step": n_k, "kernel_count_range": [min(counts), max(counts)],
           "wall_ms": wall_ms, "kernel_sum_ms": busy_ms, "gap_ms": gap_ms, "gap_share_of_wall": gap_ms / wall_ms,
           "gaps": {"intra_step_ms": sum(intra_gaps) / n / 1e6, "between_steps_ms": sum(inter_gaps) / n / 1e6,
                    "median_us": statistics.median(intra_gaps + inter_gaps) / 1e3,
                    "p90_us": pct(intra_gaps + inter_gaps, 90) / 1e3, "max_us": max(intra_gaps + inter_gaps) / 1e3,
                    "mean_us": statistics.fmean(intra_gaps + inter_gaps) / 1e3,
                    "between_steps_median_us": statistics.median(inter_gaps) / 1e3,
                    "mean_us_before": {k: statistics.fmean(v) / 1e3 for k, v in sorted(gaps_by_next.items())}},
           "gemv_roles_by_launch_order": role_order_ok, "gemv": per_role}
    if attn:
        # time above reading the KV cache at peak: at short contexts nearly all of it is fixed cost
        res["decode_attn"] = {"median_us": statistics.median(d for d, _ in attn) / 1e3,
                              "median_GBs": statistics.median(b / d for d, b in attn),
                              "ms_per_step": sum(d for d, _ in attn) / n / 1e6,
                              "over_kv_floor_ms_per_step": sum(d - b / peak for d, b in attn) / n / 1e6}
    res["other_kernels_ms_per_step"] = {k: sum(v) / n / 1e6 for k, v in other.items()}
    if len(set(fx)) >= 2:
        t0, inv_bw = fit_line(fx, fy)  # ns, ns per byte
        res["gemv_fit"] = {"t0_us": t0 / 1e3, "stream_GBs": 1 / inv_bw if inv_bw > 0 else None,
                           "t0_ms_per_step": t0 * (4 * L + 1) / 1e6}

    # the shortfall against the floor: gaps, fixed cost inside kernels (GEMV t0, attention above its
    # KV floor, embed/argmax), and the rest = GEMVs streaming below peak
    floor = meta["floor_ms"]
    short = wall_ms - floor
    split = {"floor_ms": floor, "pct_of_floor": floor / wall_ms * 100, "shortfall_ms": short,
             "gaps_ms": gap_ms, "gaps_share": gap_ms / short if short > 0 else None,
             "kernels_over_floor_ms": busy_ms - floor,
             "pct_of_floor_if_no_gaps": floor / busy_ms * 100}
    if "gemv_fit" in res:
        fixed = res["gemv_fit"]["t0_ms_per_step"] + sum(res["other_kernels_ms_per_step"].values()) \
            + res.get("decode_attn", {}).get("over_kv_floor_ms_per_step", 0.0)
        split["fixed_in_kernel_ms"] = fixed
        split["streaming_below_peak_ms"] = short - gap_ms - fixed
        split["per_kernel_fixed_share"] = (gap_ms + fixed) / short if short > 0 else None
    res["shortfall"] = split
    return res


def verdict(s):
    """Reading of the launch-overhead hypothesis; thresholds from docs/colab-followup.md."""
    gs, ks = s["shortfall"].get("gaps_share"), s["shortfall"].get("per_kernel_fixed_share")
    if gs is None:
        return "no shortfall: decode is at the floor"
    if gs >= 0.5:
        return "supported: gaps between kernels are most of the shortfall"
    if ks is not None and ks >= 0.5:
        return ("partly: gaps alone are a minority, but gaps plus the fixed per-kernel cost inside kernels "
                "are most of the shortfall (fewer, larger kernels would help; launch latency alone is not it)")
    if gs <= 0.2:
        return "refuted: the time is inside the kernels (streaming below peak), not between them"
    return "mixed: gaps are a real but minor part of the shortfall"


def summarize(a):
    meta = json.load(open(a.meta))
    kernels, ranges = load_trace(a.sqlite)
    out = {"gpu": meta["gpu"], "weights": meta["weights"], "ctx": meta["ctx"], "peak_bw_GBs": meta["peak_bw_GBs"],
           "read_bw_GBs": meta.get("read_bw_GBs"), "copy_bw_GBs": meta.get("copy_bw_GBs"),
           "reference_ms": {"graph": meta["graph_ms"], "nograph": meta["nograph_ms"]}, "floor_ms": meta["floor_ms"]}
    if a.traced_meta:
        t = json.load(open(a.traced_meta))
        out["traced_engine_ms"] = {"graph": t["graph_ms"], "nograph": t["nograph_ms"]}
        out["profiler_inflation"] = {"graph": t["graph_ms"] / meta["graph_ms"],
                                     "nograph": t["nograph_ms"] / meta["nograph_ms"]}
    for name in ("decode_graph", "decode_nograph"):
        if name not in ranges:
            out[name] = {"error": f"NVTX range {name} not in the trace (ranges: {sorted(ranges)[:10]})"}
            continue
        r0, r1 = ranges[name]
        ks = [k for k in kernels if k[0] >= r0 and k[1] <= r1]
        s = summarize_range(ks, meta, meta["ctx"])
        if "error" not in s:
            s["verdict"] = verdict(s)
        out[name] = s
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print_summary(out)


def print_summary(out):
    print(f"\n{out['gpu']}  {os.path.basename(out['weights'])}  ctx {out['ctx']}  "
          f"peak {out['peak_bw_GBs']:.0f} GB/s" + (f", measured read {out['read_bw_GBs']:.0f} GB/s" if out.get("read_bw_GBs") else ""))
    print(f"reference (no profiler): graph {out['reference_ms']['graph']:.3f} ms/token, "
          f"no graph {out['reference_ms']['nograph']:.3f}; floor {out['floor_ms']:.3f} ms")
    if "profiler_inflation" in out:
        print(f"engine time under nsys: x{out['profiler_inflation']['graph']:.3f} (graph), "
              f"x{out['profiler_inflation']['nograph']:.3f} (no graph)")
    for name in ("decode_graph", "decode_nograph"):
        s = out[name]
        if "error" in s:
            print(f"{name}: {s['error']}")
            continue
        sf, g = s["shortfall"], s["gaps"]
        print(f"\n== {name}: {s['steps_analyzed']} steps, {s['kernels_per_step']} kernels/step")
        print(f"  wall {s['wall_ms']:.3f} ms = kernels {s['kernel_sum_ms']:.3f} + gaps {s['gap_ms']:.3f} "
              f"({s['gap_share_of_wall'] * 100:.0f}% of wall; {g['intra_step_ms']:.3f} inside steps, "
              f"{g['between_steps_ms']:.3f} between)")
        print(f"  gap per kernel: median {g['median_us']:.2f} us, p90 {g['p90_us']:.2f}, max {g['max_us']:.1f}")
        print(f"  floor {sf['floor_ms']:.3f} ms = {sf['pct_of_floor']:.0f}% of floor; without gaps "
              f"{sf['pct_of_floor_if_no_gaps']:.0f}%; gaps are "
              + (f"{sf['gaps_share'] * 100:.0f}% of the {sf['shortfall_ms']:.3f} ms shortfall" if sf["gaps_share"] is not None else "-"))
        for r, v in s["gemv"].items():
            extra = f"  {v['pct_measured_read']:4.0f}% of measured" if "pct_measured_read" in v else ""
            print(f"  gemv {r:8s} {v['bytes'] / 1e6:7.1f} MB  {v['median_us']:7.1f} us  {v['median_GBs']:6.0f} GB/s  "
                  f"{v['pct_peak']:4.0f}% of peak{extra}   x{v['per_step']} = {v['ms_per_step']:.3f} ms/step")
        if "decode_attn" in s:
            d = s["decode_attn"]
            print(f"  decode_attn      {d['median_us']:7.1f} us  {d['median_GBs']:6.0f} GB/s (KV)   {d['ms_per_step']:.3f} ms/step")
        if "fixed_in_kernel_ms" in sf:
            print(f"  shortfall {sf['shortfall_ms']:.3f} ms = gaps {sf['gaps_ms']:.3f} + fixed cost inside kernels "
                  f"{sf['fixed_in_kernel_ms']:.3f} + streaming below peak {sf['streaming_below_peak_ms']:.3f}")
        if "gemv_fit" in s:
            f = s["gemv_fit"]
            print(f"  GEMV fit: t0 {f['t0_us']:.2f} us per kernel ({f['t0_ms_per_step']:.3f} ms/step), streaming "
                  + (f"{f['stream_GBs']:.0f} GB/s" if f["stream_GBs"] else "-"))
        print(f"  verdict: {s['verdict']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--weights", required=True)
    r.add_argument("--ctx", type=int, default=128)
    r.add_argument("--steps", type=int, default=128)
    r.add_argument("--max-seq", type=int, default=2048)
    r.add_argument("--nvtx", action="store_true", help="mark the decode runs with NVTX ranges (under nsys)")
    r.add_argument("--out", required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--sqlite", required=True)
    s.add_argument("--meta", required=True, help="JSON from the plain run")
    s.add_argument("--traced-meta", help="JSON from the run under nsys (profiler overhead check)")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    main()
