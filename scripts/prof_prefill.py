"""Where does prefill time go? Profiles `reps` engine prefills with torch.profiler (CUPTI captures the
kernels the ctypes library launches), one profiler session per prefill, and prints:
  - per kernel name: ms per prefill, % of the kernel sum, calls; then a family summary
    (gemm / flash_prefill / other, substring match on the name);
  - the GPU timeline of each prefill: span (first GPU activity start to last end), busy (union of
    kernel / memcpy / memset intervals), idle = span - busy, the number of gaps > 5 us, and the
    10 largest gaps with the activity on each side;
  - the median wall time of `reps` unprofiled prefills (CUDA events), so wall - span is the host /
    edge overhead outside the GPU timeline;
  - the GPU's SM clock during both phases (nvidia-smi, every 200 ms), so a clock drop is visible.

python scripts/prof_prefill.py --weights llama32-3b.bin --T 2000 --reps 5
The clock log reads GPU CUDA_VISIBLE_DEVICES[0] (default 0) as an nvidia-smi index, which matches
the CUDA device only with CUDA_DEVICE_ORDER=PCI_BUS_ID (scripts/edmonds.sh exports it).
"""
import argparse
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(__file__))
import hipinfer as hi  # noqa: E402

GAP_US = 5.0


def family(name):
    if "gemm" in name:
        return "gemm"
    if "flash_prefill" in name:
        return "flash_prefill"
    return "other"


def gpu_events(prof):
    return [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def collect(events, reps):
    """{name: [total ms per prefill, calls per prefill]} from the profiler's CUDA events."""
    tot, cnt = defaultdict(float), defaultdict(int)
    for e in events:
        tot[e.name] += e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total
        cnt[e.name] += e.count
    return {n: [tot[n] / 1e3 / reps, cnt[n] / reps] for n in tot}


def timeline(events):
    """GPU timeline of one prefill: span, busy (union of intervals), gaps [(us, before, after)] longer than GAP_US."""
    iv = sorted((e.time_range.start, e.time_range.end, e.name) for e in events)
    span = iv[-1][1] - iv[0][0] if iv else 0.0
    busy, gaps = 0.0, []
    cur_s, cur_e, cur_n = iv[0] if iv else (0.0, 0.0, "")
    for s, e, n in iv[1:]:
        if s > cur_e:  # a gap: the union so far ends at cur_e (activity cur_n), the next starts at s
            busy += cur_e - cur_s
            if s - cur_e > GAP_US:
                gaps.append((s - cur_e, cur_n, n))
            cur_s, cur_e, cur_n = s, e, n
        elif e > cur_e:
            cur_e, cur_n = e, n
    busy += cur_e - cur_s
    return span, busy, gaps


class ClockLog:
    """nvidia-smi SM-clock samples every 200 ms in the background; stats per [t0, t1] window."""

    def __init__(self):
        self.idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
        self.f = tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False)
        try:
            self.p = subprocess.Popen(["nvidia-smi", "-i", self.idx, "--query-gpu=timestamp,clocks.sm",
                                       "--format=csv,noheader,nounits", "-lms", "200"],
                                      stdout=self.f, stderr=subprocess.DEVNULL)
        except OSError:
            self.p = None

    def stop(self):
        if self.p:
            self.p.terminate()
            self.p.wait()
        self.f.flush()
        self.f.seek(0)
        self.samples = []
        for line in self.f:
            try:
                ts, mhz = line.strip().split(", ")
                self.samples.append((datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(mhz)))
            except ValueError:
                continue
        self.f.close()
        os.unlink(self.f.name)

    def window(self, t0, t1):
        v = [m for t, m in self.samples if t0 <= t <= t1]
        if not v:
            return "no samples"
        return f"SM clock min {min(v)} / median {statistics.median(v):.0f} / max {max(v)} MHz ({len(v)} samples)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    eng = hi.Engine(a.weights, max_batch=1, max_seq=max(a.T, 64), max_prefill=max(a.T, 64))
    clk = None
    try:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(100, 30000, (a.T,), generator=g).tolist()
        for _ in range(2):  # warm-up (autotuner)
            eng.prefill(0, ids)
        torch.cuda.synchronize()
        clk = ClockLog()
        time.sleep(0.5)  # let the logger start

        # profiled: one session per prefill, so each trace is exactly one prefill's GPU timeline
        t0 = time.time()
        all_ev, lines = [], []
        for _ in range(a.reps):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                eng.prefill(0, ids)
                torch.cuda.synchronize()
            ev = gpu_events(prof)
            all_ev += ev
            lines.append(timeline(ev))
        t1 = time.time()

        # unprofiled wall time, same process
        walls = []
        for _ in range(a.reps):
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            eng.prefill(0, ids)
            e.record()
            e.synchronize()
            walls.append(s.elapsed_time(e))
        t2 = time.time()
        time.sleep(0.3)
    finally:
        if clk is not None:
            clk.stop()
        eng.close()

    ks = collect(all_ev, a.reps)
    if not ks:
        print(f"PROFILER CAPTURE FAILED: no CUDA kernels recorded. unprofiled wall {statistics.median(walls):.2f} ms")
        sys.exit(1)
    total = sum(v[0] for v in ks.values())
    print(f"T={a.T} reps={a.reps}  kernel-time sum {total:.2f} ms/prefill")
    print(f"{'ms':>9s} {'% of kernel sum':>15s} {'calls':>7s}  kernel")
    fam = defaultdict(float)
    for n, (ms, c) in sorted(ks.items(), key=lambda kv: -kv[1][0]):
        fam[family(n)] += ms
        print(f"{ms:9.3f} {ms / total * 100:15.2f} {c:7.1f}  {n[:110]}")
    print("\n== family summary (% of kernel sum) ==")
    for f in ["gemm", "flash_prefill", "other"]:
        print(f"{f:14s} {fam[f]:9.3f} ms {fam[f] / total * 100:6.2f} %")
    print(f"{'total':14s} {total:9.3f} ms")
    others = sorted(((ms, n) for n, (ms, _) in ks.items() if family(n) == "other"), reverse=True)
    print("\n== top 'other' kernels ==")
    for ms, n in others[:5]:
        print(f"{ms:9.3f} ms  {n[:110]}")

    print(f"\n== GPU timeline per profiled prefill (gaps > {GAP_US:g} us counted) ==")
    print(f"{'rep':>3s} {'span ms':>9s} {'busy ms':>9s} {'idle ms':>9s} {'idle %':>7s} {'gaps>' + format(GAP_US, 'g') + 'us':>9s} {'idle in them ms':>16s}")
    for i, (span, busy, gaps) in enumerate(lines):
        print(f"{i:3d} {span / 1e3:9.3f} {busy / 1e3:9.3f} {(span - busy) / 1e3:9.3f} {(span - busy) / span * 100:7.2f}"
              f" {len(gaps):9d} {sum(gp[0] for gp in gaps) / 1e3:16.3f}")
    spans = [ln[0] / 1e3 for ln in lines]
    idles = [(ln[0] - ln[1]) / 1e3 for ln in lines]
    wall = statistics.median(walls)
    print(f"median: span {statistics.median(spans):.3f} ms, idle {statistics.median(idles):.3f} ms")
    print(f"unprofiled wall (CUDA events, median of {a.reps}): {wall:.3f} ms   "
          f"[{', '.join(f'{w:.2f}' for w in walls)}]")
    print(f"wall - median span (host / edge overhead outside the GPU timeline): {wall - statistics.median(spans):.3f} ms")
    print(f"profiled phase:   {clk.window(t0, t1)}")
    print(f"unprofiled phase: {clk.window(t1, t2)}")

    mid = sorted(range(len(lines)), key=lambda i: idles[i])[len(lines) // 2]
    print(f"\n== 10 largest gaps, rep {mid} (median idle) ==")
    for us, before, after in sorted(lines[mid][2], reverse=True)[:10]:
        print(f"{us:9.1f} us  after {before[:60]:60s} -> before {after[:60]}")


if __name__ == "__main__":
    main()
