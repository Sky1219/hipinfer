"""Cost of eviction scoring and selection (docs/eviction.md).

  decode    tracked vs plain decode attention (one layer's kernels; tracked = attention that also
            stores logits + LSE, plus the finish kernel that adds the weights to the scores)
  prefill   flash prefill vs flash prefill + LSE + the key-parallel score pass
  select    H2O top-k and WFL prune (kernels) over every (layer, kv head) of a model, and the
            PyTorch matrix version of WFL prune (the SubCache algorithm on the GPU) on the same input
  engine    (--model) decode step time with tracking off / on at long context, and hi_evict cost

Kernel times use bench.timeit (median, launches captured into a CUDA graph); selection and engine
times are CUDA-event medians of whole calls including their launches.
Usage: python scripts/bench_eviction.py [--model M.bin] [--out results/.../eviction_bench.json] [--quick]"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
import torch  # noqa: E402
import hipinfer as hi  # noqa: E402
from bench import timeit  # noqa: E402
from eviction_ref import ref_sim, ref_wfl_prune  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model")
ap.add_argument("--out")
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
R = {"gpu": torch.cuda.get_device_name(), "decode": [], "prefill": [], "select": [], "engine": []}
SHAPES = {"llama32-1b": (32, 8, 64, 16), "llama32-3b": (24, 8, 128, 28), "qwen25-1.5b": (12, 2, 128, 28),
          "llama31-8b": (32, 8, 128, 32)}  # Hq, Hkv, D, layers
print(R["gpu"], "| decode kernel:", "scalar" if os.environ.get("HIPINFER_DECODE_SCALAR") else "default")


def ev_time(fn, reps=7):
    """Median CUDA-event ms of fn() (host launches included)."""
    fn()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


# ------------------------------------------------------------------ decode
lens = [1024, 4096] if args.quick else [1024, 4096, 16384, 32768]
for name, (Hq, Hkv, D, _) in SHAPES.items():
    for B in (1, 8):
        for L in lens:
            if B * Hkv * L * D * 4 > 3 * 2 ** 30:
                continue
            q = torch.randn(B, Hq * D, device=dev, dtype=torch.float16)
            kc = torch.randn(B, Hkv, L, D, device=dev, dtype=torch.float16)
            vc = torch.randn_like(kc)
            pos = torch.full((B,), L - 1, device=dev, dtype=torch.int32)
            out = torch.empty(B, Hq * D, device=dev, dtype=torch.float16)
            work = torch.zeros(hi.lib.hi_decode_work_bytes(B, Hq, Hkv, D), device=dev, dtype=torch.uint8)
            lg = torch.empty(B, Hq, L, device=dev)
            lse = torch.empty(B, Hq, device=dev)
            score = torch.zeros(B, Hkv, L, device=dev)
            t0 = timeit(lambda: hi.decode_attn(q, kc, vc, pos, Hq, Hkv, out=out, work=work))
            t1 = timeit(lambda: hi.decode_attn_track(q, kc, vc, pos, Hq, Hkv, score, out=out, work=work, lg=lg, lse=lse))
            r = dict(model=name, B=B, len=L, plain_us=t0 * 1e3, track_us=t1 * 1e3, overhead=t1 / t0 - 1)
            R["decode"].append(r)
            print(f"decode {name:12s} B={B} len={L:6d}: plain {t0*1e3:7.1f} us, tracked {t1*1e3:7.1f} us (+{100*(t1/t0-1):5.1f}%)")
            kc = vc = lg = None  # free before the next shape

# ------------------------------------------------------------------ prefill
for name, (Hq, Hkv, D, _) in SHAPES.items():
    for T in ([512, 2048] if args.quick else [512, 2048, 4096]):
        q = torch.randn(T, Hq * D, device=dev, dtype=torch.float16)
        kc = torch.randn(Hkv, T, D, device=dev, dtype=torch.float16)
        vc = torch.randn_like(kc)
        out = torch.empty(T, Hq * D, device=dev, dtype=torch.float16)
        lse = torch.empty(Hq, T, device=dev)
        score = torch.zeros(Hkv, T, device=dev)
        t0 = timeit(lambda: hi.flash_prefill(q, kc, vc, T, 0, Hq, Hkv, out=out))
        t1 = timeit(lambda: hi.flash_prefill_lse(q, kc, vc, T, 0, Hq, Hkv, out=out, lse=lse))
        t2 = timeit(lambda: hi.prefill_scores(q, kc, T, 0, Hq, Hkv, lse, score))
        R["prefill"].append(dict(model=name, T=T, flash_ms=t0, flash_lse_ms=t1, score_pass_ms=t2, overhead=(t1 + t2) / t0 - 1))
        print(f"prefill {name:12s} T={T:5d}: flash {t0:7.3f} ms, +lse {t1:7.3f} ms, score pass {t2:7.3f} ms "
              f"(+{100*((t1+t2)/t0-1):5.1f}% attention time)")

# ------------------------------------------------------------------ selection
for name, (Hq, Hkv, D, layers) in SHAPES.items():
    Pn = layers * Hkv
    for n in ([1024, 2048] if args.quick else [1024, 2048, 4096]):
        k = torch.randn(Pn, n, D, device=dev, dtype=torch.float16)
        v = torch.randn_like(k)
        score = torch.rand(Pn, n, device=dev) ** 3
        sink, recent = 4, n // 16
        status = torch.zeros(Pn, n, dtype=torch.uint8, device=dev)
        status[:, :sink] = 1
        status[:, n - recent:] = 1
        flags = torch.zeros(Pn, n, dtype=torch.uint8, device=dev)
        cap = n // 8  # a 12.5% budget, as SubCache's speed runs
        m = cap - sink - recent
        th = ev_time(lambda: hi.select_k(score, sink, n - recent, m, True, flags=flags))
        work = torch.empty(hi.lib.hi_wfl_work_bytes(Pn, n, D), dtype=torch.uint8, device=dev)
        rows = [("h2o top-k", th, None)]
        for label, mm, step in [(f"wfl prune to {cap} (step 8)", m, 8), ("wfl prune: evict 64 (step 8)", n - sink - recent - 64, 8)]:
            nr = [0]

            def go():
                nr[0] = hi.wfl_prune(k, v, score, status, mm, 0.5, step, 0, 1, work=work)[1]
            t = ev_time(go, reps=3)
            ref_t = None
            sim_bytes = Pn * n * n * 4
            if sim_bytes * 3 < torch.cuda.mem_get_info()[0] and n <= 2048:
                w = score / score.sum(1, keepdim=True)
                held = status == 1

                def ref():
                    ref_wfl_prune(ref_sim(k, v), w, ~held, held, mm, 0.5, step)
                ref_t = ev_time(ref, reps=3)
            rows.append((label + f", {nr[0]} rounds", t, ref_t))
        for label, t, ref_t in rows:
            R["select"].append(dict(model=name, problems=Pn, n=n, what=label, kernel_ms=t, torch_ms=ref_t))
            print(f"select {name:12s} P={Pn:3d} n={n:5d} {label:42s} {t:9.2f} ms"
                  + (f"   PyTorch matrix version {ref_t:9.2f} ms (x{ref_t / t:.1f})" if ref_t else ""))
        k = v = work = None

# ------------------------------------------------------------------ engine
if args.model:
    # the selection benchmarks above leave their tensors in PyTorch's cache; the engine allocates
    # with hipMalloc, outside it
    torch.cuda.empty_cache()
    print(f"engine: free GPU memory {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB", flush=True)
    e = hi.Engine(args.model, max_batch=1, max_seq=8192, max_prefill=1024)
    prompt = torch.randint(100, e.vocab - 100, (4096,)).tolist()
    for track in (False, True):
        e.set_tracking(track, 32)
        e.prefill(0, prompt)
        e.decode(1, 8, True)  # capture
        ms = e.decode(1, 128, True) / 128
        R["engine"].append(dict(what=f"decode step, ctx ~4.1k, tracking {'on' if track else 'off'}", ms=ms))
        print(f"engine decode step at ~4.1k context, tracking {'on ' if track else 'off'}: {ms:.3f} ms")
    for policy, kw in [("h2o", {}), ("wfl", {}), ("wfl", {"pool": 2.0, "rounds": 8})]:
        e.prefill(0, prompt)
        n = e.cache_len(0)
        t = time.perf_counter()
        ev, _, ms = e.evict(0, 1024, policy, sink=4, recent=64, **kw)
        wall = (time.perf_counter() - t) * 1e3
        e.decode(1, 8, True)
        step = e.decode(1, 128, True) / 128
        what = f"evict {n} -> 1024, {policy}{' ' + str(kw) if kw else ''}"
        R["engine"].append(dict(what=what, select_ms=ms["select_ms"], compact_ms=ms["compact_ms"], wall_ms=wall,
                                decode_after_ms=step))
        print(f"engine {what}: select {ms['select_ms']:.2f} ms, compact {ms['compact_ms']:.2f} ms "
              f"(call {wall:.1f} ms wall); decode step after: {step:.3f} ms")

if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(R, open(args.out, "w"), indent=1)
    print("wrote", args.out)
