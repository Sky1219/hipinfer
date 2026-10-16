"""Markdown summary of results/<gpu>/<model>/{bench,gate,ppl}.json across GPUs and models.

python scripts/summarize.py [results_dir] > results/SUMMARY.md
"""
import glob
import json
import os
import re
import sys

MODEL_ORDER = ["tinyllama", "qwen25-0.5b", "llama32-1b", "qwen25-1.5b", "llama32-3b", "mistral-7b", "mistral-7b-v0.1",
               "llama31-8b"]
MODEL_NAME = {"tinyllama": "TinyLlama-1.1B", "qwen25-0.5b": "Qwen2.5-0.5B", "llama32-1b": "Llama-3.2-1B",
              "qwen25-1.5b": "Qwen2.5-1.5B", "llama32-3b": "Llama-3.2-3B", "mistral-7b": "Mistral-7B-v0.3",
              "mistral-7b-v0.1": "Mistral-7B-v0.1 (window ignored)",
              "llama31-8b": "Llama-3.1-8B"}


# Per GPU, from results/<gpu dir>/thermal.json (written after checking each benchmark window against
# the GPU's temperature log): {model: {"verdict": "clean" | "minor" | "THROTTLED", ...}}
THERMAL, THERMAL_DIR = {}, {}


def load(root):
    """-> {gpu_name: {model_tag: {"bench":..., "gate":..., "ppl":...}}} (first directory per GPU wins)."""
    gpus, seen_dirs = {}, {}
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d) or os.path.basename(d) in ("baselines", "tuning"):
            continue
        models = {}
        gpu = None
        for md in sorted(glob.glob(os.path.join(d, "*"))):
            if not os.path.isdir(md):
                continue
            tag = os.path.basename(md)
            e = {}
            for k in ("bench", "gate", "ppl"):
                f = os.path.join(md, k + ".json")
                if os.path.exists(f):
                    e[k] = json.load(open(f))
                    gpu = gpu or e[k].get("gpu")
            if e:
                models[tag] = e
        if gpu and models:
            if gpu in seen_dirs:
                continue
            seen_dirs[gpu] = d
            gpus[gpu] = models
            tf = os.path.join(d, "thermal.json")
            if os.path.exists(tf):
                THERMAL[gpu], THERMAL_DIR[gpu] = json.load(open(tf)), os.path.basename(d)
    return gpus


def short(gpu):
    return gpu.replace("NVIDIA ", "").replace("GeForce ", "")


def fmt_tokps(v):
    return f"{v / 1000:.1f}k" if v >= 10000 else f"{v:.0f}"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    gpus = load(root)
    if not gpus:
        sys.exit("no results found")
    names = list(gpus)
    all_models = [m for m in MODEL_ORDER if any(m in gpus[g] for g in names)]
    all_models += sorted(m for g in names for m in gpus[g] if m not in all_models)
    hdr = "| | " + " | ".join(short(g) for g in names) + " |"

    def mark(cells, m):
        """Flag a GPU's cell when that model's benchmark ran thermally throttled."""
        return [c + " ⚠" if c != "-" and THERMAL.get(g, {}).get(m, {}).get("verdict") == "THROTTLED" else c
                for c, g in zip(cells, names)]
    sep = "|---" * (len(names) + 1) + "|"

    def cfg_of(m):
        for g in names:
            b = gpus[g].get(m, {}).get("bench")
            if b:
                return b["config"]
        return None

    print("## Models\n")
    print("| model | params (fp16 GB) | layers | hidden | heads (q/kv) | head_dim | vocab | notes |")
    print("|---|---|---|---|---|---|---|---|")
    for m in all_models:
        c = cfg_of(m)
        if not c:
            continue
        notes = []
        if c.get("tied"):
            notes.append("tied embeddings")
        if c.get("qkv_bias"):
            notes.append("QKV bias")
        if c["heads"] == c["kv_heads"]:
            notes.append("MHA")
        else:
            notes.append(f"GQA group {c['heads'] // c['kv_heads']}")
        print(f"| {MODEL_NAME.get(m, m)} | {c['weight_bytes'] / 1e9:.2f} | {c['layers']} | {c['hidden']} | {c['heads']}/{c['kv_heads']} | "
              f"{c['head_dim']} | {c['vocab']} | {', '.join(notes)} |")

    print("\n## Correctness vs HF transformers (fp16)\n")
    print("Greedy gate: 7 prompts x 128 tokens, single-sequence with and without CUDA graphs, plus a 6-sequence batch "
          "(20 runs). A run that diverges only where HF's own top-2 logits are within 0.05 counts as a near-tie. "
          "Perplexity: WikiText-2 test, non-overlapping 1024-token windows, both engines on the same tokens.\n")
    print("| model | GPU | gate | runs token-identical | PPL hipinfer | PPL HF | top-1 agreement | mean KL(HF, hipinfer) | max abs logit diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in all_models:
        for g in names:
            e = gpus[g].get(m, {})
            ga, pp = e.get("gate"), e.get("ppl")
            if not ga and not pp:
                continue
            gate = ("PASS" if ga["pass"] else "**FAIL**") if ga else "-"
            ident = f'{ga["identical"]}/{ga["runs"]}' if ga else "-"
            if pp:
                print(f"| {MODEL_NAME.get(m, m)} | {short(g)} | {gate} | {ident} | {pp['ppl_hipinfer']:.4f} | {pp['ppl_hf']:.4f} | "
                      f"{100 * pp['top1_agreement']:.2f}% | {pp['mean_kl_hf_to_hipinfer']:.1e} | {pp['max_logit_diff']:.3f} |")
            else:
                print(f"| {MODEL_NAME.get(m, m)} | {short(g)} | {gate} | {ident} | - | - | - | - | - |")

    print("\n## Decode, batch 1 (ms/token, greedy, 256 new tokens after a 128-token prompt)\n")
    print("Floor = (weight bytes + KV bytes read) / peak DRAM bandwidth. HF = transformers eager with a DynamicCache.\n")
    for g in names:
        if any(v.get("verdict") == "THROTTLED" for v in THERMAL.get(g, {}).values()):
            print(f"⚠ = {short(g)} benchmark ran partly thermally throttled; this and every speed table below flags it. "
                  f"Details: results/{THERMAL_DIR[g]}/THERMAL.md.\n")
    print("| model | " + " | ".join(short(g) for g in names) + " |")
    print(sep)
    print("| peak DRAM BW (GB/s) | " + " | ".join(f'{next(e["bench"] for e in gpus[g].values() if "bench" in e)["peak_bw_GBs"]:.0f}' if any("bench" in e for e in gpus[g].values()) else "-" for g in names) + " |")
    for m in all_models:
        cells = []
        for g in names:
            b = gpus[g].get(m, {}).get("bench")
            d = next((x for x in b["e2e"].get("decode_b1", []) if x["ctx"] == 128), None) if b and "e2e" in b else None
            if d and d.get("hf_ms"):
                cells.append(f'**{d["ours_ms"]:.2f}** ({100 * d["bw_util"]:.0f}% BW; HF {d["hf_ms"]:.1f}, {d["hf_ms"] / d["ours_ms"]:.1f}x)')
            elif d:
                cells.append(f'**{d["ours_ms"]:.2f}** ({100 * d["bw_util"]:.0f}% BW; HF copy does not fit)')
            else:
                cells.append("-")
        if any(c != "-" for c in cells):
            print(f"| {MODEL_NAME.get(m, m)} | " + " | ".join(mark(cells, m)) + " |")

    print("\n## Decode throughput, batched (tok/s; ctx 512 + 128 new tokens; largest batch that fits)\n")
    print("| model | " + " | ".join(short(g) for g in names) + " |")
    print(sep)
    for m in all_models:
        cells = []
        for g in names:
            b = gpus[g].get(m, {}).get("bench")
            rows = b["e2e"].get("decode_batched", []) if b and "e2e" in b else []
            if rows:
                r8 = next((x for x in rows if x["B"] == 8), None)
                top = rows[-1]
                s = f'B=8: {fmt_tokps(r8["tok_s"])}; ' if r8 else ""
                s += f'B={top["B"]}: **{fmt_tokps(top["tok_s"])}** ({top["ms_per_step"]:.1f} ms/step'
                if "mJ_per_token" in top:
                    s += f', {top["mJ_per_token"]:.0f} mJ/tok at {top["watts"]:.0f} W'
                cells.append(s + ")")
            else:
                cells.append("-")
        if any(c != "-" for c in cells):
            print(f"| {MODEL_NAME.get(m, m)} | " + " | ".join(mark(cells, m)) + " |")

    print("\n## Prefill (prompt tokens/s; ms in parentheses; HF = transformers eager, SDPA attention)\n")
    print("| model | T | " + " | ".join(short(g) for g in names) + " |")
    print("|---" + sep)
    for m in all_models:
        Ts = sorted({p["T"] for g in names for p in gpus[g].get(m, {}).get("bench", {}).get("e2e", {}).get("prefill", [])})
        for T in Ts:
            cells = []
            for g in names:
                b = gpus[g].get(m, {}).get("bench")
                p = next((x for x in b["e2e"].get("prefill", []) if x["T"] == T), None) if b and "e2e" in b else None
                if p and p.get("hf_ms"):
                    cells.append(f'{fmt_tokps(p["ours_tok_s"])} ({p["ours_ms"]:.1f} ms; HF {p["hf_ms"]:.1f} ms, {p["hf_ms"] / p["ours_ms"]:.2f}x)')
                elif p:
                    cells.append(f'{fmt_tokps(p["ours_tok_s"])} ({p["ours_ms"]:.1f} ms)')
                else:
                    cells.append("-")
            print(f"| {MODEL_NAME.get(m, m)} | {T} | " + " | ".join(mark(cells, m)) + " |")

    print("\n## Decode latency vs context length (batch 1, ms/token; % of the bandwidth floor incl. KV reads)\n")
    print("| model | ctx | " + " | ".join(short(g) for g in names) + " |")
    print("|---" + sep)
    for m in all_models:
        ctxs = sorted({p["ctx"] for g in names for p in gpus[g].get(m, {}).get("bench", {}).get("e2e", {}).get("decode_ctx", [])})
        for ctx in ctxs:
            cells = []
            for g in names:
                b = gpus[g].get(m, {}).get("bench")
                p = next((x for x in b["e2e"].get("decode_ctx", []) if x["ctx"] == ctx), None) if b and "e2e" in b else None
                cells.append(f'{p["ours_ms"]:.2f} ({100 * p["bw_util"]:.0f}%)' if p else "-")
            print(f"| {MODEL_NAME.get(m, m)} | {ctx} | " + " | ".join(mark(cells, m)) + " |")

    print("\n## Kernels vs PyTorch (speedup = torch time / hipinfer time; graph-replayed; TinyLlama shapes)\n")
    ref = "tinyllama" if "tinyllama" in all_models else all_models[0]
    print(hdr); print(sep)
    rows = [("gemm", "qkv", 2048, None, "GEMM qkv M=2048 vs cuBLAS"),
            ("gemm", "gate_up", 2048, None, "GEMM gate_up M=2048 vs cuBLAS"),
            ("gemm", "down", 512, None, "GEMM down M=512 vs cuBLAS"),
            ("gemm", "o", 128, None, "GEMM o M=128 vs cuBLAS"),
            ("gemv", "gate_up+norm+silu", 1, None, "GEMV norm+gate_up+SwiGLU, M=1"),
            ("gemv", "down+res", 1, None, "GEMV down+residual, M=1"),
            ("gemv", "lm_head+norm", 8, None, "GEMV norm+lm_head, M=8"),
            ("rmsnorm", "rmsnorm", 4096, "F.rms_norm", "RMSNorm 4096 rows vs F.rms_norm"),
            ("rope", "rope", 512, None, "RoPE+KV write, 512 tok vs HF"),
            ("flash_prefill", "flash_prefill", 1024, None, "causal attention T=1024 vs SDPA"),
            ("flash_prefill", "flash_prefill", 2048, None, "causal attention T=2048 vs SDPA"),
            ("decode_attn", "B=1", 4096, None, "decode attention B=1 L=4096 vs SDPA"),
            ("decode_attn", "B=8", 4096, None, "decode attention B=8 L=4096 vs SDPA")]
    for kernel, role, M, base, label in rows:
        cells = []
        for g in names:
            b = gpus[g].get(ref, {}).get("bench")
            k = next((x for x in (b["kernels"] if b else []) if x["kernel"] == kernel and x.get("role") == role and x.get("M") == M
                      and (base is None or x["baseline"] == base)), None)
            if not k:
                cells.append("-")
                continue
            extra = f' ({k["TFLOPs"]:.0f} TF)' if "TFLOPs" in k else f' ({k["GB/s"]:.0f} GB/s)' if "GB/s" in k else ""
            cells.append(f'{k["speedup"]:.2f}x{extra}')
        print(f"| {label} | " + " | ".join(mark(cells, ref)) + " |")

    print("\n## GEMM vs cuBLAS at every model's shapes (TFLOPs, M=2048 prefill rows; ratio = hipinfer / cuBLAS)\n")
    print("| model | shape | " + " | ".join(short(g) for g in names) + " |")
    print("|---" + sep)
    for m in all_models:
        for role in ("qkv", "o", "gate_up", "down", "lm_head"):
            cells = []
            for g in names:
                b = gpus[g].get(m, {}).get("bench")
                k = next((x for x in (b["kernels"] if b else []) if x["kernel"] == "gemm" and x["role"] == role and x["M"] == 2048), None)
                cells.append(f'{k["TFLOPs"]:.0f} / {k["TFLOPs_torch"]:.0f} ({k["speedup"]:.2f})' if k else "-")
            if any(c != "-" for c in cells):
                print(f"| {MODEL_NAME.get(m, m)} | {role} | " + " | ".join(mark(cells, m)) + " |")

    baselines(gpus, os.path.join(root, "baselines"))


ENGINES = [("hipinfer", "**hipinfer**"),
           ("llamacpp", "llama.cpp (F16 GGUF, CUDA, best of FA on/off)"),
           ("torch_compile", "torch.compile, gpt-fast style (reduce-overhead + static KV)"),
           ("hf_static", "HF transformers, StaticCache + torch.compile"),
           ("vllm", "vLLM (V1 engine, CUDA graphs)")]


def baselines(gpus, bdir):
    """Same-session engine comparison from results/baselines/<gpu>/*.json (TinyLlama, scripts/run_baselines.sh)
    and results/baselines/<gpu>/<model>/*.json (other models)."""
    data = {}  # (gpu name, model tag) -> {engine: {"decode": {ctx: ms}, "prefill": {T: ms}}}

    def add(f, model):
        r = json.load(open(f))
        eng = os.path.splitext(os.path.basename(f))[0]
        g = data.setdefault((r["gpu"], model), {})
        if eng == "hipinfer":  # bench.py rerun next to the baselines (same session)
            e2e = r["e2e"]
            g["hipinfer"] = {"decode": {x["ctx"]: x["ours_ms"] for x in e2e["decode_b1"]},
                             "prefill": {x["T"]: x["ours_ms"] for x in e2e["prefill"]},
                             "floor": {x["ctx"]: x["bw_floor_ms"] for x in e2e["decode_b1"]}}
        else:
            g[eng] = {"decode": {x["ctx"]: x["ms"] for x in r.get("decode_b1", [])},
                      "prefill": {x["T"]: x["ms"] for x in r.get("prefill", []) if "note" not in x}}

    for f in sorted(glob.glob(os.path.join(bdir, "*", "*.json"))):
        if os.path.basename(os.path.dirname(f)) != "previous_hipinfer":
            add(f, "tinyllama")
    for f in sorted(glob.glob(os.path.join(bdir, "*", "*", "*.json"))):
        add(f, os.path.basename(os.path.dirname(f)))
    if not data:
        return
    # hipinfer numbers from the main run when a (GPU, model) has no same-session rerun
    for (gname, model), g in data.items():
        b = gpus.get(gname, {}).get(model, {}).get("bench")
        if b and "hipinfer" not in g:
            e2e = b["e2e"]
            g["hipinfer"] = {"decode": {x["ctx"]: x["ours_ms"] for x in e2e["decode_b1"]},
                             "prefill": {x["T"]: x["ours_ms"] for x in e2e["prefill"]},
                             "floor": {x["ctx"]: x["bw_floor_ms"] for x in e2e["decode_b1"]}}

    def dcell(v, h, ctx):
        s = f"{v:.2f} ms"
        if h and "floor" in h and ctx in h["floor"]:
            s += f", {100 * h['floor'][ctx] / v:.0f}% BW"
        if h and ctx in h["decode"]:
            s += f" (hipinfer {v / h['decode'][ctx]:.2f}x faster)" if v > h["decode"][ctx] else \
                 f" (**{h['decode'][ctx] / v:.2f}x faster than hipinfer**)" if v < h["decode"][ctx] else ""
        return s

    def pcell(v, h, T):
        s = f"{v:.1f} ms"
        if h and T in h["prefill"] and abs(v - h["prefill"][T]) > 1e-9:
            hv = h["prefill"][T]
            s += f" (hipinfer {v / hv:.2f}x faster)" if v > hv else f" (**{hv / v:.2f}x faster than hipinfer**)"
        return s

    models = [m for m in MODEL_ORDER if any(k[1] == m for k in data)] + sorted({k[1] for k in data} - set(MODEL_ORDER))
    print("\n## hipinfer vs other engines (fp16 weights, batch 1, same GPU)\n")
    print("Decode: 256 greedy tokens after a ctx-token prompt, ms/token (% BW = weight+KV bytes / peak BW / time). "
          "Prefill: one prompt of T random tokens, ms. llama.cpp: `llama-bench` (pp / tg256 at depth ctx), F16 GGUF, "
          "best of flash-attn on/off and ubatch 512/2048. torch.compile / HF static: decode step in a CUDA graph, KV "
          "cache sized ctx+256. vLLM: decode = (generate 257 tokens - generate 1 token) / 256. The TinyLlama rows "
          "were measured in one session by scripts/run_baselines.sh; the other models' hipinfer rows come from the "
          "main run (results/<gpu>/<model>/bench.json). Raw JSON/logs: results/baselines/.\n")
    for m in models:
        names = [k[0] for k in data if k[1] == m]
        names = sorted(set(names), key=lambda n: list(gpus).index(n) if n in gpus else 99)
        hdr = "| | " + " | ".join(short(n) for n in names) + " |"
        sep = "|---" * (len(names) + 1) + "|"
        print(f"\n### {MODEL_NAME.get(m, m)}\n")
        for title, key, points, cell in (("decode ms/token", "decode", (128, 1024), dcell),
                                         ("prefill ms", "prefill", (128, 512, 2000), pcell)):
            print(hdr); print(sep)
            for p in points:
                for eng, label in ENGINES:
                    if not any(p in data[(n, m)].get(eng, {}).get(key, {}) for n in names):
                        continue
                    cells = []
                    for n in names:
                        e, h = data[(n, m)].get(eng), data[(n, m)].get("hipinfer")
                        v = e and e[key].get(p)
                        cells.append(cell(v, h, p) if v else "-")
                    print(f"| {title}, {key} {p}: {label} | " + " | ".join(cells) + " |")
            print()


def tensor_parallel(root):
    """results/<gpu>/<model>/tp<N>.json (scripts/bench_tp.py), tp<N>_test.json (tests/test_tp.py),
    ppl_tp<N>.json (tests/ppl.py --tp)."""
    runs = {}  # (gpu dir, tp) -> {model: {"bench":, "test":, "ppl":}}
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*tp[0-9]*.json"))):
        m = re.fullmatch(r"(ppl_)?tp(\d+)(_test)?", os.path.basename(f)[:-5])
        if not m:
            continue
        kind = "ppl" if m[1] else "test" if m[3] else "bench"
        tp = int(m[2])
        gdir, model = os.path.basename(os.path.dirname(os.path.dirname(f))), os.path.basename(os.path.dirname(f))
        runs.setdefault((gdir, tp), {}).setdefault(model, {})[kind] = json.load(open(f))
    if not runs:
        return
    print("\n## Tensor parallelism (multi-GPU)\n")
    print("Megatron-style sharding over N GPUs in one process (one worker thread per GPU): QKV/gate/up "
          "column-parallel, O/down row-parallel with an all-reduce after each, lm_head vocabulary-parallel. "
          "All-reduce `auto` = hipinfer's one-shot kernel for messages up to 512 KB (decode), NCCL above "
          "(prefill); `NCCL` = NCCL only. Prefill all-reduces overlap with compute (two row halves on two "
          "streams) unless noted. % floor = this GPU's shard of the weights + its KV share / peak BW / time.\n")
    for (gdir, tp), models in runs.items():
        order = [m for m in MODEL_ORDER if m in models] + sorted(set(models) - set(MODEL_ORDER))
        anyb = next((models[m]["bench"] for m in order if "bench" in models[m]), None)
        gpu_s = " + ".join(short(g) for g in anyb["gpus"]) if anyb else gdir
        print(f"\n### {gpu_s} (`{gdir}`), tp={tp}\n")
        if anyb and anyb.get("tp_info"):
            ti = anyb["tp_info"]
            print(f"Peer-to-peer access between the GPUs: {'yes' if ti['p2p'] else 'no (one-shot all-reduce stages through pinned host memory)'}; "
                  f"NCCL built in: {'yes' if ti['nccl'] else 'no'}.\n")
        if anyb and anyb.get("allreduce"):
            print(f"All-reduce latency, fp16, in place ({MODEL_NAME.get(order[0], order[0])} hidden size {anyb['config']['hidden']}):\n")
            print("| message | size | one-shot kernel | NCCL | one-shot speedup |\n|---|---|---|---|---|")
            for r in anyb["allreduce"]:
                o, n = r.get("oneshot_us"), r.get("nccl_us")
                nccl = f"{n:.1f} us" if n else "-"
                sp = f"{n / o:.2f}x" if o and n else "-"
                print(f"| {r['what']} | {r['bytes'] / 1024:.0f} KB | {o:.1f} us | {nccl} | {sp} |")
            print()

        def rows(b, part, key, overlap=True):
            """1-GPU rows by key, and tp rows by backend then key (prefill: with or without overlap)."""
            s = {x[key]: x for x in (b.get("single") or {}).get(part, [])}
            t = {}
            for x in (b.get("tp_rows") or {}).get(part, []):
                if x.get("overlap", True) == overlap:
                    t.setdefault(x["allreduce"], {})[x[key]] = x
            return s, t

        print(f"Batch-1 decode, ms/token (1 GPU vs tp={tp}):\n")
        print(f"| model | ctx | 1 GPU | tp={tp} auto | tp={tp} NCCL only | speedup | % floor (tp) |\n|---|---|---|---|---|---|---|")
        for m in order:
            b = models[m].get("bench")
            if not b:
                continue
            s, t = rows(b, "decode_b1", "ctx")
            auto = t.get("auto", {})
            for ctx in sorted(auto):
                one = s.get(ctx, {}).get("ms")
                a = auto[ctx]["ms"]
                nc = t.get("nccl", {}).get(ctx, {}).get("ms")
                util = auto[ctx].get("bw_util")
                print(f"| {MODEL_NAME.get(m, m)} | {ctx} | {f'{one:.2f}' if one else 'does not fit'} | **{a:.2f}** | "
                      f"{f'{nc:.2f}' if nc else '-'} | {f'{one / a:.2f}x' if one else '-'} | {f'{100 * util:.0f} %' if util else '-'} |")
        print(f"\nPrefill, ms (1 GPU vs tp={tp}):\n")
        print(f"| model | tokens | 1 GPU | tp={tp} auto, overlapped | tp={tp} auto, no overlap | tp={tp} NCCL only | speedup |\n|---|---|---|---|---|---|---|")
        for m in order:
            b = models[m].get("bench")
            if not b:
                continue
            s, t = rows(b, "prefill", "T")
            no_ov = rows(b, "prefill", "T", overlap=False)[1].get("auto", {})
            for T in sorted(t.get("auto", {})):
                one = s.get(T, {}).get("ms")
                a = t["auto"][T]["ms"]
                nov = no_ov.get(T, {}).get("ms")
                nc = t.get("nccl", {}).get(T, {}).get("ms")
                print(f"| {MODEL_NAME.get(m, m)} | {T} | {f'{one:.1f}' if one else 'does not fit'} | **{a:.1f}** | "
                      f"{f'{nov:.1f}' if nov else '-'} | {f'{nc:.1f}' if nc else '-'} | {f'{one / a:.2f}x' if one else '-'} |")
        print(f"\nBatched decode throughput, tok/s (ctx 512 + 128 new tokens):\n")
        print(f"| model | B | 1 GPU | tp={tp} auto | tp={tp} NCCL only | speedup |\n|---|---|---|---|---|---|")
        for m in order:
            b = models[m].get("bench")
            if not b:
                continue
            s, t = rows(b, "decode_batched", "B")
            for B in sorted(t.get("auto", {})):
                one = s.get(B, {}).get("tok_s")
                a = t["auto"][B]["tok_s"]
                nc = t.get("nccl", {}).get(B, {}).get("tok_s")
                print(f"| {MODEL_NAME.get(m, m)} | {B} | {fmt_tokps(one) if one else '-'} | **{fmt_tokps(a)}** | "
                      f"{fmt_tokps(nc) if nc else '-'} | {f'{a / one:.2f}x' if one else '-'} |")
        print(f"\nCorrectness of the sharded engine (tests/test_tp.py: tp={tp} vs the same weights on 1 GPU; "
              "tests/ppl.py --tp for models that only fit sharded):\n")
        print("| model | greedy runs token-identical to 1 GPU | prefill top-1 agreement | max abs logit diff | "
              "backends bitwise equal | all-reduce self-test | PPL sharded / HF |\n|---|---|---|---|---|---|---|")
        for m in order:
            te, pp = models[m].get("test"), models[m].get("ppl")
            if not te and not pp:
                continue
            be = (te or {}).get("backends", {})
            a = be.get("auto") or be.get("oneshot") or {}
            runs_ = a.get("runs")
            if not runs_:
                ident = "no 1-GPU reference"
            else:
                ident = f"{a['identical']}/{len(runs_)}"
                if a["identical"] < len(runs_) and all(r["accepted"] for r in runs_):
                    ident += " (rest at near-ties)"
            agree = f"{100 * a['prefill_top1_agreement']:.2f} %" if "prefill_top1_agreement" in a else "-"
            diff = f"{a['prefill_max_logit_diff']:.3f}" if "prefill_max_logit_diff" in a else "-"
            bit = [v for k, v in a.items() if k.startswith("bitwise_equal")]
            bits = "yes" if bit and all(bit) else ("no" if bit else "-")
            errs = [b.get("allreduce_max_err", 0) for b in be.values()]
            st = ("exact" if errs and max(errs) == 0 else f"max err {max(errs)}") if errs else "-"
            ppl = f"{pp['ppl_hipinfer']:.4f} / {pp['ppl_hf']:.4f}" if pp else "-"
            print(f"| {MODEL_NAME.get(m, m)} | {ident} | {agree} | {diff} | {bits} | {st} | {ppl} |")
        print()


if __name__ == "__main__":
    main()
    tensor_parallel(sys.argv[1] if len(sys.argv) > 1 else "results")
