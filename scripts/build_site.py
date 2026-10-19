"""Build the results page (site/index.html) from results/: one self-contained HTML file, no
external scripts or fonts. Published to GitHub Pages by .github/workflows/pages.yml.

python scripts/build_site.py [results_dir] [out_dir]
"""
import html
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from summarize import MODEL_NAME, MODEL_ORDER, load  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
REPO = "https://github.com/Sky1219/hipinfer"
SKIP = {"mistral-7b-v0.1"}  # kept in results/ for the record; the page shows v0.3 only
TP_ONLY = ["llama31-8b", "mistral-7b"]


def jload(path):
    return json.load(open(path)) if os.path.exists(path) else None


def at(rows, key, val, field):
    return next((r[field] for r in rows or [] if r.get(key) == val), None)


def name(m):
    return MODEL_NAME.get(m, m)


def collect(root):
    gpus = load(root)
    t = next((g for g in gpus if "2080 Ti" in g), None)
    ti = gpus.get(t, {})
    models = [m for m in MODEL_ORDER if m in ti and "bench" in ti[m] and m not in SKIP]
    llama = {}
    for m in models:
        f = os.path.join(root, "baselines", "2080ti", "llamacpp.json") if m == "tinyllama" else \
            os.path.join(root, "baselines", "2080ti", m, "llamacpp.json")
        llama[m] = jload(f)

    decode, prefill, ctx_lines, batched = [], [], [], []
    for m in models:
        e = ti[m]["bench"]["e2e"]
        lc = llama[m] or {}
        decode.append({"label": name(m), "hipinfer": at(e["decode_b1"], "ctx", 128, "ours_ms"),
                       "llamacpp": at(lc.get("decode_b1"), "ctx", 128, "ms"),
                       "floor": at(e["decode_b1"], "ctx", 128, "bw_floor_ms"),
                       "util": at(e["decode_b1"], "ctx", 128, "bw_util")})
        prefill.append({"label": name(m), "hipinfer": at(e["prefill"], "T", 2000, "ours_ms"),
                        "llamacpp": at(lc.get("prefill"), "T", 2000, "ms")})
        ctx_lines.append({"label": name(m), "points": [[r["ctx"], round(100 * r["bw_util"], 1), round(r["ours_ms"], 2)]
                                                       for r in e["decode_ctx"]]})
        b = max(e["decode_batched"], key=lambda r: r["tok_s"]) if e["decode_batched"] else None
        batched.append({"label": name(m), "B": b and b["B"], "tok_s": b and b["tok_s"]})

    tp, ar = [], None
    for m in models + [m for m in TP_ONLY if m not in models]:
        r = jload(os.path.join(root, "2080ti", m, "tp2.json"))
        if not r:
            continue
        rows = r["tp_rows"]
        auto = [x for x in rows["decode_b1"] if x["allreduce"] == "auto"]
        one = (r.get("single") or {}).get("decode_b1")
        pf1 = (r.get("single") or {}).get("prefill")
        pf2 = [x for x in rows["prefill"] if x["allreduce"] == "auto" and x.get("overlap", True)]
        bt2 = [x for x in rows["decode_batched"] if x["allreduce"] == "auto"]
        tp.append({"label": name(m), "one": at(one, "ctx", 128, "ms"), "two": at(auto, "ctx", 128, "ms"),
                   "util": at(auto, "ctx", 128, "bw_util"),
                   "pf_one": at(pf1, "T", 2000, "ms"), "pf_two": at(pf2, "T", 2000, "ms"),
                   "bt_two": max((x["tok_s"] for x in bt2), default=None)})
        if ar is None and r.get("allreduce"):
            ar = {"hidden": r["config"]["hidden"], "model": name(m), "rows": r["allreduce"],
                  "p2p": r.get("tp_info", {}).get("p2p")}

    correct = []
    for g in gpus:
        for m in MODEL_ORDER:
            if m in SKIP or m not in gpus[g]:
                continue
            gt, pp = gpus[g][m].get("gate"), gpus[g][m].get("ppl")
            if gt or pp:
                correct.append({"model": name(m), "gpu": g.replace("NVIDIA ", "").replace("GeForce ", ""), "setup": "1 GPU",
                                "gate": gt and f"{gt['identical']}/{gt['runs']}", "pass": gt and gt["pass"],
                                "ppl_hi": pp and pp["ppl_hipinfer"], "ppl_hf": pp and pp["ppl_hf"],
                                "top1": pp and pp["top1_agreement"], "kl": pp and pp["mean_kl_hf_to_hipinfer"]})
    for m in TP_ONLY:
        pp = jload(os.path.join(root, "2080ti", m, "ppl_tp2.json"))
        if pp:
            correct.append({"model": name(m), "gpu": "2× RTX 2080 Ti", "setup": "sharded over 2 GPUs", "gate": None,
                            "pass": None, "ppl_hi": pp["ppl_hipinfer"], "ppl_hf": pp["ppl_hf"],
                            "top1": pp["top1_agreement"], "kl": pp["mean_kl_hf_to_hipinfer"]})
    return {"gpu": t, "decode": decode, "prefill": prefill, "ctx": ctx_lines, "batched": batched, "tp": tp, "ar": ar,
            "correct": correct, "a100": collect_a100(root, gpus), "int8": collect_int8(root),
            "spec": collect_spec(root)}


ENGINES = [("hipinfer", "hipinfer"), ("llamacpp", "llama.cpp"), ("vllm", "vLLM"), ("torch_compile", "torch.compile")]


def collect_a100(root, gpus):
    """Same-session engine comparison on the A100 from results/baselines/a100/<model>/<engine>.json. A missing file
    is an engine that didn't run; a baseline whose greedy tokens differ from HF computes another model (not valid)."""
    bdir = os.path.join(root, "baselines", "a100")
    rows, gpu, vllm = [], None, None
    for m in MODEL_ORDER:
        if not os.path.isdir(os.path.join(bdir, m)):
            continue
        row = {"label": name(m), "decode": {}, "prefill": {}, "why": {}}
        for eng, _ in ENGINES:
            r = jload(os.path.join(bdir, m, eng + ".json"))
            match = (r or {}).get("greedy_match_vs_hf")
            if r is None or (match and match.split("/")[0] != match.split("/")[1]):
                row["why"][eng] = "not run" if r is None else f"not valid ({match} tokens match HF)"
                continue
            if eng == "hipinfer":
                e = r["e2e"]
                gpu = r["gpu"]
                row["decode"][eng] = at(e["decode_b1"], "ctx", 128, "ours_ms")
                row["prefill"][eng] = at(e["prefill"], "T", 2000, "ours_ms")
                row["floor"] = at(e["decode_b1"], "ctx", 128, "bw_floor_ms")
            else:
                row["decode"][eng] = at(r.get("decode_b1"), "ctx", 128, "ms")
                row["prefill"][eng] = at(r.get("prefill"), "T", 2000, "ms")  # vLLM's includes 1 generated token
                vllm = vllm or r.get("vllm")
        rows.append(row)
    if not rows:
        return None
    big = (gpus.get(gpu, {}).get("mistral-7b") or {}).get("bench")  # hipinfer only, no other engines
    big = big and {"label": name("mistral-7b"), "ms": at(big["e2e"]["decode_b1"], "ctx", 128, "ours_ms"),
                   "util": at(big["e2e"]["decode_b1"], "ctx", 128, "bw_util")}
    return {"gpu": gpu, "vllm": vllm, "rows": rows, "big": big}


def ratios(rows, kind, engines, faster):
    """engine/hipinfer time ratios (faster=False) or hipinfer/engine (faster=True) over every valid pair"""
    out = []
    for r in rows:
        h = r[kind].get("hipinfer")
        for e in engines:
            v = r[kind].get(e)
            if h and v:
                out.append(h / v if faster else v / h)
    return out


WHY = {("qwen25-1.5b", "torch_compile"): "my torch.compile baseline left out Qwen2's QKV bias, so its greedy tokens "
                                         "didn't match HF (0/32) and its numbers are excluded. The bias is fixed in "
                                         "scripts/bench_baselines.py but hasn't been re-run",
       ("llama32-3b", "vllm"): "the Colab disk filled before vLLM could load Llama-3.2-3B"}


def span(v, d=2, unit="×"):
    return f"{min(v):.{d}f}–{max(v):.{d}f}{unit}" if v else "–"


def a100_facts(a, root):
    """the A100 numbers the page's text quotes, all computed from the result files"""
    rows = a["rows"]
    util = [r["floor"] / r["decode"]["hipinfer"] for r in rows if r.get("floor") and r["decode"].get("hipinfer")]
    others = [r["floor"] / r["decode"][e] for r in rows for e in ("llamacpp", "vllm")
              if r.get("floor") and r["decode"].get(e)]
    gemm, hf = [], []
    tags = {name(m): m for m in MODEL_ORDER}
    for r in rows:
        b = jload(os.path.join(root, "a100", tags[r["label"]], "bench.json"))
        gemm += [k["speedup"] for k in (b or {}).get("kernels", []) if k["kernel"] == "gemm" and k["M"] == 2048]
        h = jload(os.path.join(root, "baselines", "a100", tags[r["label"]], "hipinfer.json"))
        hf += [x["ours_ms"] / x["hf_ms"] for x in (h or {}).get("e2e", {}).get("prefill", []) if x["T"] == 2000]
    return {"util": util, "others": others, "gemm": gemm, "hf": hf,
            "lc_dec": ratios(rows, "decode", ["llamacpp"], False), "vl_dec": ratios(rows, "decode", ["vllm"], False),
            "tc_dec": ratios(rows, "decode", ["torch_compile"], False),
            "lc_pf": ratios(rows, "prefill", ["llamacpp"], False),
            "fast_pf": ratios(rows, "prefill", ["vllm", "torch_compile"], True)}


def a100_tables(a):
    engs = ENGINES[1:]

    def cell(r, kind, e, d):
        v, h = r[kind].get(e), r[kind].get("hipinfer")
        if e in r["why"]:
            return r["why"][e].split(" (")[0]
        if not (v and h):
            return fnum(v, d)
        return f"{v:.{d}f} ({v / h:.2f}× slower)" if v > h else f"{v:.{d}f} ({h / v:.2f}× faster)"

    vl = f"vLLM {a['vllm']}" if a["vllm"] else "vLLM"
    dec = table(["model", "hipinfer ms/token", "% of floor"] + [n.replace("vLLM", vl) + ", vs hipinfer" for _, n in engs],
                [[r["label"], fnum(r["decode"].get("hipinfer")),
                  f"{100 * r['floor'] / r['decode']['hipinfer']:.0f}%" if r.get("floor") and r["decode"].get("hipinfer") else "–"]
                 + [cell(r, "decode", e, 2) for e, _ in engs] for r in a["rows"]])
    pf = table(["model", "hipinfer ms"] + [n.replace("vLLM", vl + " (incl. 1 generated token)") + ", vs hipinfer"
                                           for _, n in engs],
               [[r["label"], fnum(r["prefill"].get("hipinfer"), 1)] + [cell(r, "prefill", e, 1) for e, _ in engs]
                for r in a["rows"]])
    return dec, pf


def a100_section(d, root):
    a = d["a100"]
    if not a:
        return ""
    f = a100_facts(a, root)
    dec_tbl, pf_tbl = a100_tables(a)
    gpu = html.escape(a["gpu"].replace("NVIDIA ", ""))
    vl = f"vLLM {html.escape(a['vllm'])}" if a["vllm"] else "vLLM"
    ti = [r["util"] for r in d["decode"] if r["util"] and r["label"] != name("qwen25-0.5b")]
    notes = []
    tags = {name(m): m for m in MODEL_ORDER}
    for r in a["rows"]:
        for e, n in ENGINES:
            if e in r["why"]:
                why = WHY.get((tags[r["label"]], e), r["why"][e])
                notes.append(f"<li><b>{html.escape(n)}, {html.escape(r['label'])}: {html.escape(r['why'][e].split(' (')[0])}.</b> "
                             f"{html.escape(why[0].upper() + why[1:])}.</li>")
    big = a.get("big")
    big_txt = f" {html.escape(big['label'])} also ran on the A100 (no other engines): {big['ms']:.2f} ms/token, " \
              f"{100 * big['util']:.0f}% of the floor." if big and big["ms"] and big["util"] else ""
    return f"""
<h2>On an A100</h2>
<p>{gpu} (1,555 GB/s) on Google Colab, with llama.cpp, {vl} and torch.compile run in the same session and no thermal
throttling (<a href="%%REPO%%/blob/main/results/a100/THERMAL.md">THERMAL.md</a>). hipinfer was re-run next to the other
engines, so every row compares runs a few minutes apart. fp16 weights, batch 1. llama.cpp is <code>llama-bench</code> on
an F16 GGUF, best of flash-attention on and off; vLLM uses its V1 engine with CUDA graphs; torch.compile is a
gpt-fast-style static KV cache in <code>reduce-overhead</code> mode. A baseline counts only if its greedy tokens match
HF transformers.</p>
<div class="card"><div class="ctitle">Decode latency per token on an A100 (lower is better)</div>
<div class="csub">ms/token at a 128-token context</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp</span><span><i class="sw" style="background:var(--s3)"></i>{vl}</span>
<span><i class="sw" style="background:var(--s4)"></i>torch.compile</span><span><i class="mk"></i>bandwidth floor</span></div>
<div class="chart" id="c-a100-decode"></div>
<details><summary>Data table</summary>{dec_tbl}</details></div>
<div class="card"><div class="ctitle">Prefill time for 2,000 tokens on an A100 (lower is better)</div>
<div class="csub">ms; vLLM's time is an end-to-end <code>generate()</code> call, so it includes one generated token</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp</span><span><i class="sw" style="background:var(--s3)"></i>{vl}</span>
<span><i class="sw" style="background:var(--s4)"></i>torch.compile</span></div>
<div class="chart" id="c-a100-prefill"></div>
<details><summary>Data table</summary>{pf_tbl}</details></div>
<div class="note"><ul>{"".join(notes)}</ul></div>
<p><b>Decode:</b> hipinfer is {span(f["lc_dec"])} faster than llama.cpp, {span(f["vl_dec"])} faster than vLLM and
{span(f["tc_dec"])} faster than torch.compile on the models each one ran. It is further from the floor than on the
2080 Ti ({span([100 * u for u in f["util"]], 0, "%")} on these 1–3B models, against {span([100 * u for u in ti], 0, "%")}).
The weights stream about 2.5× faster, but each kernel launch costs about the same, so the fixed per-launch time is a
bigger share of a token.{big_txt} llama.cpp and vLLM sit further below the floor still
({span([100 * u for u in f["others"]], 0, "%")}).</p>
<p><b>Long prompts are where hipinfer loses.</b> It is {span(f["lc_pf"])} faster than llama.cpp at 2,000 tokens, but
vLLM and torch.compile are {span(f["fast_pf"])} faster than hipinfer there, and HF transformers is
{span(f["hf"])} faster too. All three use cuBLAS and FlashAttention-2, and on the A100 my
GEMMs run at only {span(f["gemm"])} cuBLAS (2,048-row prefill shapes of these four models).</p>
"""


def a100_limit(d, root):
    a = d["a100"]
    if not a:
        return ""
    f = a100_facts(a, root)
    return (f"<li><b>Launch overhead on the A100.</b> Batch-1 decode reaches only {span([100 * u for u in f['util']], 0, '%')} "
            f"of the bandwidth floor on the 1–3B models: the A100 streams weights fast enough that fixed per-launch time "
            f"is a large share of each token.</li>"
            f"<li><b>Long-prompt prefill on the A100.</b> At 2,000 tokens vLLM and torch.compile are {span(f['fast_pf'])} "
            f"faster than hipinfer; hipinfer's GEMMs run at {span(f['gemm'])} cuBLAS on that GPU.</li>")


INT8 = ["tinyllama", "llama32-1b", "llama32-3b"]


def collect_int8(root):
    """int8 weights vs fp16 and llama.cpp Q8_0, all measured back to back in one session. Speed from
    results/2080ti/<model>/int8/; Llama-3.2-3B's from rerun/, started on a cool card (MERGED_RUN.md)."""
    rows = []
    for m in INT8:
        d = os.path.join(root, "2080ti", m, "int8")
        if m == "llama32-3b":
            rr = os.path.join(root, "2080ti", "rerun", "llama32-3b_")
            b, f, q = (jload(rr + x + ".json") for x in ("bench_int8", "bench_fp16", "llamacpp_q8_0"))
        else:
            b, f, q = (jload(os.path.join(d, x + ".json")) for x in ("bench", "bench_fp16", "llamacpp_q8_0"))
        pp, pr, gt = (jload(os.path.join(d, x + ".json")) for x in ("ppl", "ppl_int8ref", "gate_int8ref"))
        if not (b and f and q):
            continue
        e, g = b["e2e"], f["e2e"]
        row = {"label": name(m), "decode": {}, "prefill": {}, "util": [], "util_fp16": []}
        for c in (128, 1024):
            row["decode"][c] = {"int8": at(e["decode_b1"], "ctx", c, "ours_ms"),
                                "fp16": at(g["decode_b1"], "ctx", c, "ours_ms"),
                                "q8": at(q["decode_b1"], "ctx", c, "ms")}
            row["util"].append(at(e["decode_b1"], "ctx", c, "bw_util"))
            row["util_fp16"].append(at(g["decode_b1"], "ctx", c, "bw_util"))
        row["floor"] = at(e["decode_b1"], "ctx", 128, "bw_floor_ms")
        for t in (128, 512, 2000):
            row["prefill"][t] = {"int8": at(e["prefill"], "T", t, "ours_ms"), "fp16": at(g["prefill"], "T", t, "ours_ms"),
                                 "q8": at(q["prefill"], "T", t, "ms")}
        row["acc"] = pp and pr and gt and {
            "ppl_hf": pp["ppl_hf"], "ppl_hi": pp["ppl_hipinfer"], "chg": pp["ppl_change_pct"],
            "kl": pp["mean_kl_hf_to_hipinfer"], "ppl_ref": pr["ppl_hf"], "chg_ref": pr["ppl_change_pct"],
            "kl_ref": pr["mean_kl_hf_to_hipinfer"], "gate": f"{gt['identical']}/{gt['runs']}", "pass": gt["pass"]}
        rows.append(row)
    return rows or None


def sci(v):
    m, e = f"{v:.1e}".split("e")
    return f"{m}e{int(e)}"


def int8_facts(rows):
    dec = lambda c, a, b: [r["decode"][c][a] / r["decode"][c][b] for r in rows]
    pf = lambda ts: [r["prefill"][t]["int8"] / r["prefill"][t]["fp16"] - 1 for r in rows for t in ts]
    acc = [r["acc"] for r in rows if r["acc"]]
    return {"fp16": dec(128, "fp16", "int8"), "fp16_1k": dec(1024, "fp16", "int8"),
            "q8": dec(128, "q8", "int8"), "q8_1k": dec(1024, "q8", "int8"),
            "util": [u for r in rows for u in r["util"]], "util_fp16": [u for r in rows for u in r["util_fp16"]],
            "pf_short": pf([128]), "pf_long": pf([512, 2000]),
            "pf_q8": [r["prefill"][t]["q8"] / r["prefill"][t]["int8"] for r in rows for t in r["prefill"]],
            "kl_ref": [a["kl_ref"] for a in acc], "kl": [a["kl"] for a in acc],
            "chg_ref": [abs(a["chg_ref"]) for a in acc], "acc": acc}


def int8_section(d):
    rows = d["int8"]
    if not rows:
        return ""
    f = int8_facts(rows)
    pct = lambda v: span([100 * x for x in v], 0, "%")
    dec_tbl = table(["model", "context", "int8 ms/token", "fp16 ms/token", "int8 speed-up", "llama.cpp Q8_0 ms/token",
                     "int8 vs Q8_0", "int8 % of floor"],
                    [[r["label"], f"{c:,}", fnum(x["int8"]), fnum(x["fp16"]),
                      f"{x['fp16'] / x['int8']:.2f}×", fnum(x["q8"]), f"{x['q8'] / x['int8']:.2f}× faster",
                      f"{100 * r['util'][i]:.0f}%"]
                     for c, i in ((128, 0), (1024, 1)) for r in rows for x in [r["decode"][c]]])
    pf_tbl = table(["model", "tokens", "int8 ms", "fp16 ms", "int8 vs fp16", "llama.cpp Q8_0 ms", "int8 vs Q8_0"],
                   [[r["label"], f"{t:,}", fnum(x["int8"], 1), fnum(x["fp16"], 1),
                     f"{100 * (x['int8'] / x['fp16'] - 1):.0f}% slower", fnum(x["q8"], 1),
                     f"{x['q8'] / x['int8']:.2f}× faster"] for r in rows for t, x in r["prefill"].items()])
    acc_tbl = table(["model", "PPL HF fp16", "PPL hipinfer int8", "change", "PPL HF, same int8 weights", "change",
                     "mean KL vs HF int8", "greedy runs identical to HF int8"],
                    [[r["label"], fnum(a["ppl_hf"], 4), fnum(a["ppl_hi"], 4), f"{a['chg']:+.2f}%", fnum(a["ppl_ref"], 4),
                      f"{a['chg_ref']:+.4f}%", f"{a['kl_ref']:.1e}", a["gate"]] for r in rows for a in [r["acc"]] if a])
    n = len(rows)
    return f"""
<h2>int8 weights (W8A16)</h2>
<p>The projections and the lm_head are stored as int8 with one fp16 scale per output row (symmetric round-to-nearest);
activations, the KV cache, norms and the embedding stay fp16. Batch-1 decode reads every weight once per token, so int8
halves the bytes the GEMVs stream. Decode converts the weights to fp32 in registers; prefill dequantizes each weight tile
to fp16 on its way into shared memory and runs the existing tensor-core GEMM. RTX 2080 Ti, batch 1. hipinfer's int8 and
fp16 weights and llama.cpp's Q8_0 GGUF were measured back to back in one session, so the fp16 times here differ slightly
from the charts above. Llama-3.2-3B's rows come from a rerun started on a cool card, where the int8 and fp16 runs' median
clocks are within 2 % (<a href="%%REPO%%/blob/main/results/2080ti/MERGED_RUN.md">MERGED_RUN.md</a>).</p>
<div class="card"><div class="ctitle">Decode latency per token with int8 weights (lower is better)</div>
<div class="csub">ms/token at a 128-token context; the marker is int8's own bandwidth floor</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer int8</span>
<span><i class="sw" style="background:var(--s3)"></i>hipinfer fp16</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp Q8_0</span><span><i class="mk"></i>int8 bandwidth floor</span></div>
<div class="chart" id="c-int8-decode"></div>
<details><summary>Data table</summary>{dec_tbl}</details></div>
<div class="card"><div class="ctitle">Prefill time for 2,000 tokens with int8 weights (lower is better)</div>
<div class="csub">ms, same session</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer int8</span>
<span><i class="sw" style="background:var(--s3)"></i>hipinfer fp16</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp Q8_0</span></div>
<div class="chart" id="c-int8-prefill"></div>
<details><summary>Data table</summary>{pf_tbl}</details></div>
<p><b>Decode is {span(f["fp16"])} faster than fp16, not 2×.</b> The weight bytes halve, but the KV-cache reads and each
kernel's fixed cost don't, and the conversion costs instructions: int8 decode reaches {pct(f["util"])} of its bandwidth
floor, against {pct(f["util_fp16"])} for fp16. At a 1,024-token context it is {span(f["fp16_1k"])} faster than fp16.
Against llama.cpp's Q8_0 it is {span(f["q8"])} faster at 128 tokens of context and {span(f["q8_1k"])} at 1,024.</p>
<p><b>Prefill is slower than fp16</b>: {pct(f["pf_short"])} at 128 tokens and {pct(f["pf_long"])} at 512–2,000, because
each weight tile is converted on its way to shared memory. It is still {span(f["pf_q8"])} faster than llama.cpp's Q8_0
prefill at every length measured.</p>
<p><b>Accuracy.</b> Against fp16 HF, WikiText-2 perplexity rises by
{" / ".join(f"{a['chg']:.2f} %" for a in f["acc"])}, the cost of rounding the weights. Against HF running the same rounded
weights it is within {max(f["chg_ref"]):.3f} % (mean KL {sci(min(f["kl_ref"]))} to {sci(max(f["kl_ref"]))}), and the strict
token gate passes with {" / ".join(a["gate"].split("/")[0] for a in f["acc"])} of 20 runs identical, so the int8 kernels add
no error of their own beyond fp16 rounding.</p>
<div class="card"><div class="ctitle">int8 perplexity and token gate</div>
<div class="csub">WikiText-2, 16 × 1,024-token windows; the gate is 7 prompts × 128 tokens, 20 runs, against HF on the same int8 weights</div>
{acc_tbl}</div>
<div class="note"><ul>
<li><b>Q8_0 is not the same format.</b> It keeps one scale per 32 weights instead of one per row, so it reads about 6 %
more bytes and should round a little more accurately. llama.cpp's perplexity was not measured, and there is no 4-bit
comparison.</li>
<li><b>Scope.</b> {n} models, one GPU type (RTX 2080 Ti), batch 1, int8 weights only (no int8 activations or KV cache).</li>
</ul></div>
"""


def spec_passes(path):
    """'105/105 runs pass' from a tests/test_spec.py log -> (passed, runs)"""
    if not os.path.exists(path):
        return None
    m = re.search(r"(\d+)/(\d+) runs pass", open(path, encoding="utf-8", errors="replace").read())
    return m and (int(m.group(1)), int(m.group(2)))


def collect_spec(root):
    """Llama-3.2-3B with a Llama-3.2-1B draft, and Llama-3.2-1B drafting for itself, from the cool-card rerun"""
    rr = os.path.join(root, "2080ti", "rerun")
    p, s = jload(os.path.join(rr, "spec_llama32-3b_llama32-1b.json")), jload(os.path.join(rr, "spec_llama32-1b_self.json"))
    if not p:
        return None
    return {"plain": p["plain_tok_s"], "t_target": p["t_target_ms"], "t_draft": p["t_draft_ms"],
            "t_verify": {int(k): v for k, v in p["t_verify_ms"].items()},
            "rows": [{"k": r["k"], "tok_s": r["tok_s"], "speedup": r["speedup"], "pred": r["predicted_speedup"],
                      "acc": r["acceptance"], "tpr": r["tokens_per_round"], "exact": r["exact"]} for r in p["spec"]],
            "pv": (p["prefill_verify_divergent"], p["prefill_verify_runs"]),
            "self": s and {"speedup": [r["speedup"] for r in s["spec"]], "ks": [r["k"] for r in s["spec"]],
                           "acc": [r["acceptance"] for r in s["spec"]],
                           "pv": (s["prefill_verify_divergent"], s["prefill_verify_runs"])},
            "tests": {"RTX 2080 Ti": spec_passes(os.path.join(root, "2080ti", "spec", "test_spec.log")),
                      "RTX A6000": spec_passes(os.path.join(root, "a6000", "merged", "test_spec.log"))}}


def spec_section(d):
    s = d["spec"]
    if not s:
        return ""
    rows = s["rows"]
    best = max(rows, key=lambda r: r["speedup"])
    top = [r["k"] for r in rows if f"{r['speedup']:.2f}" == f"{best['speedup']:.2f}"]
    err = max(abs(r["pred"] / r["speedup"] - 1) for r in rows)
    tbl = table(["k", "tok/s", "speed-up", "predicted by the cost model", "draft tokens accepted", "tokens per round"],
                [[r["k"], fnum(r["tok_s"], 1), f"{r['speedup']:.2f}×", f"{r['pred']:.2f}×", f"{100 * r['acc']:.0f}%",
                  fnum(r["tpr"])] for r in rows])
    tests = [f"{v[0]} of {v[1]} on the {html.escape(g)}" for g, v in s["tests"].items() if v]
    rv = max(s["t_verify"])
    sf = s["self"]
    self_txt = f"""<li><b>Self-draft</b> (Llama-3.2-1B drafting for itself): every draft token accepted, and
{min(sf["speedup"]):.2f}× (k = {sf["ks"][0]}) to {max(sf["speedup"]):.2f}× (k = {sf["ks"][-1]}) plain speed. This is the
overhead of drafting with a model as expensive as the target; the run exists to check exactness.</li>""" \
        if sf and all(a == 1 for a in sf["acc"]) else ""
    pv_txt = f"""<li><b>Verifying through the prefill path would have broken exactness.</b> Its GEMMs round differently
from decode: it diverged from plain greedy output in {sf["pv"][0]} of {sf["pv"][1]} self-draft runs ({s["pv"][0]} of
{s["pv"][1]} with the 3B target).</li>""" if sf else ""
    return f"""
<h2>Speculative decoding</h2>
<p>Greedy speculative decoding with two engines on one GPU. Llama-3.2-1B drafts <code>k</code> tokens with its normal
graph-captured decode steps, and Llama-3.2-3B checks all of them in one pass. The longest agreeing prefix is accepted,
plus the target's own next token, and both caches roll back by resetting their lengths. The target verifies through the
batch-1 decode kernels, so every row's logits are bit-identical to the decode step they replace and the output is the
same as plain greedy decoding <b>token for token</b>, with no near-tie tolerance: <code>tests/test_spec.py</code> passes
{" and ".join(tests)} (7 prompts × 128 tokens at k = 2, 4, 6, 8, with and without CUDA graphs, plus self-drafting).</p>
<div class="card"><div class="ctitle">Speed-up over plain greedy decoding (higher is better)</div>
<div class="csub">Llama-3.2-3B with a Llama-3.2-1B draft, RTX 2080 Ti, 7 prompts × 128 new tokens; plain greedy decoding gives
{s["plain"]:.1f} tok/s</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>measured</span>
<span><i class="sw" style="background:var(--s3)"></i>cost model</span><span><i class="mk"></i>plain greedy decoding</span></div>
<div class="chart" id="c-spec"></div>
<details><summary>Data table</summary>{tbl}</details></div>
<p><b>Up to {best["speedup"]:.2f}× at k = {" and ".join(map(str, top))}.</b> The draft is not much cheaper than the target:
both are weight-bandwidth bound, so a 1B decode step costs {100 * s["t_draft"] / s["t_target"]:.0f} % of a 3B one
({s["t_draft"]:.2f} vs {s["t_target"]:.2f} ms), and gains level off around k = 5. Verification is cheap: checking {rv}
rows in one pass costs {s["t_verify"][rv] / s["t_target"]:.2f}× a single 3B decode step
({s["t_verify"][rv]:.1f} vs {s["t_target"]:.1f} ms), because the weights are read once for all rows. The cost model,
<code>k·t_draft + t_verify(k + 1)</code> per round, is within {100 * err:.0f} % at every k.</p>
<div class="note"><ul>
{self_txt}
{pv_txt}
<li><b>Scope.</b> One model pair on one GPU type (RTX 2080 Ti), greedy, batch 1, dense models only; the draft must
share the target's vocabulary.</li>
</ul></div>
"""


def git_info():
    sha = os.environ.get("GITHUB_SHA")
    try:
        sha = sha or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        date = subprocess.check_output(["git", "log", "-1", "--format=%cs", sha], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (sha or "")[:7], date


def fnum(v, d=2):
    return "–" if v is None else f"{v:,.{d}f}"


def table(head, rows):
    h = "".join(f"<th>{html.escape(c)}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def tiles(d):
    utils = [r["util"] for r in d["decode"] if r["util"] and r["label"] != name("qwen25-0.5b")]
    speed = [r["llamacpp"] / r["hipinfer"] for r in d["decode"] if r["llamacpp"] and r["hipinfer"]]
    tps = [r["one"] / r["two"] for r in d["tp"] if r["one"] and r["two"]]
    ppl = [abs(r["ppl_hi"] - r["ppl_hf"]) / r["ppl_hf"] * 100 for r in d["correct"] if r["ppl_hi"]]
    big = next((r for r in d["tp"] if r["label"] == name("llama31-8b")), None)
    out = []
    if utils:
        out.append(("Batch-1 decode, share of the memory-bandwidth limit", f"{100 * min(utils):.0f}–{100 * max(utils):.0f}%",
                    "models of 1B+ parameters, RTX 2080 Ti"))
    if speed:
        out.append(("Batch-1 decode vs llama.cpp", f"{min(speed):.2f}–{max(speed):.2f}× faster",
                    f"{len(speed)} models, same GPU, fp16"))
    if tps:
        out.append(("Two GPUs vs one, batch-1 decode", f"up to {max(tps):.2f}× faster", "tensor parallelism, 2× RTX 2080 Ti"))
    if big and big["two"]:
        out.append(("Llama-3.1-8B on two 11 GB cards", f"{big['two']:.1f} ms/token", "16 GB of weights; doesn't fit one card"))
    vl = ratios(d["a100"]["rows"], "decode", ["vllm"], False) if d["a100"] else []
    if vl:
        out.append(("Batch-1 decode on an A100 vs vLLM", f"{min(vl):.2f}–{max(vl):.2f}× faster",
                    f"{len(vl)} models, one session; slower on long-prompt prefill"))
    if d["int8"]:
        f = int8_facts(d["int8"])
        out.append(("int8 weights vs fp16, batch-1 decode", f"{span(f['fp16'])} faster",
                    f"and {span(f['q8'])} faster than llama.cpp Q8_0; RTX 2080 Ti"))
    if d["spec"]:
        best = max(r["speedup"] for r in d["spec"]["rows"])
        out.append(("Speculative decoding, Llama-3.2-3B with a 1B draft", f"up to {best:.2f}× faster",
                    "output identical to greedy; one model pair, RTX 2080 Ti"))
    if ppl:
        out.append(("WikiText-2 perplexity vs HF transformers", f"within {max(ppl):.3f}%", f"{len(ppl)} model/GPU checks"))
    return "".join(f'<div class="tile"><div class="tl">{html.escape(a)}</div><div class="tv">{html.escape(b)}</div>'
                   f'<div class="ts">{html.escape(c)}</div></div>' for a, b, c in out)


def build(root, out_dir):
    d = collect(root)
    sha, date = git_info()
    gpu = (d["gpu"] or "").replace("NVIDIA ", "").replace("GeForce ", "")
    dec_tbl = table(["model", "hipinfer ms/token", "llama.cpp ms/token", "bandwidth floor ms", "% of floor"],
                    [[r["label"], fnum(r["hipinfer"]), fnum(r["llamacpp"]), fnum(r["floor"]),
                      f"{100 * r['util']:.0f}%" if r["util"] else "–"] for r in d["decode"]])
    ctx_tbl = table(["model"] + [f"ctx {p[0]:,}" for p in d["ctx"][0]["points"]] if d["ctx"] else ["model"],
                    [[r["label"]] + [f"{p[2]:.2f} ms ({p[1]:.0f}%)" for p in r["points"]] for r in d["ctx"]])
    pf_tbl = table(["model", "hipinfer ms", "llama.cpp ms"],
                   [[r["label"], fnum(r["hipinfer"], 1), fnum(r["llamacpp"], 1)] for r in d["prefill"]])
    tp_tbl = table(["model", "1 GPU ms/token", "2 GPUs ms/token", "speedup", "prefill 2,000 tok: 1 GPU → 2 GPUs (ms)",
                    "2-GPU peak batched tok/s"],
                   [[r["label"], fnum(r["one"]) if r["one"] else "doesn't fit", fnum(r["two"]),
                     f"{r['one'] / r['two']:.2f}×" if r["one"] and r["two"] else "–",
                     f"{fnum(r['pf_one'], 0) if r['pf_one'] else '–'} → {fnum(r['pf_two'], 0)}", fnum(r["bt_two"], 0)]
                    for r in d["tp"]])
    ar_tbl = table(["message", "size", "hipinfer one-shot kernel", "NCCL", "speedup"],
                   [[r["what"], f"{r['bytes'] / 1024:,.0f} KB", f"{r['oneshot_us']:.1f} µs",
                     f"{r['nccl_us']:.1f} µs" if r.get("nccl_us") else "–",
                     f"{r['nccl_us'] / r['oneshot_us']:.2f}×" if r.get("nccl_us") else "–"] for r in d["ar"]["rows"]]) \
        if d["ar"] else ""
    cor_tbl = table(["model", "GPU", "setup", "greedy runs token-identical to HF", "PPL hipinfer", "PPL HF", "top-1 agreement",
                     "mean KL(HF ‖ hipinfer)"],
                    [[r["model"], r["gpu"], r["setup"], r["gate"] or "–", fnum(r["ppl_hi"], 4), fnum(r["ppl_hf"], 4),
                      f"{100 * r['top1']:.2f}%" if r["top1"] else "–", f"{r['kl']:.1e}" if r["kl"] else "–"]
                     for r in d["correct"]])
    data = json.dumps({k: d[k] for k in ("decode", "prefill", "ctx", "tp", "a100", "int8", "spec")}).replace("</", "<\\/")
    page = PAGE
    for k, v in {"%%TILES%%": tiles(d), "%%GPU%%": html.escape(gpu), "%%DEC_TBL%%": dec_tbl, "%%CTX_TBL%%": ctx_tbl,
                 "%%PF_TBL%%": pf_tbl, "%%TP_TBL%%": tp_tbl, "%%AR_TBL%%": ar_tbl, "%%COR_TBL%%": cor_tbl,
                 "%%A100%%": a100_section(d, root), "%%A100_LIMIT%%": a100_limit(d, root),
                 "%%INT8%%": int8_section(d), "%%SPEC%%": spec_section(d),
                 "%%DATA%%": data, "%%REPO%%": REPO, "%%SHA%%": sha, "%%DATE%%": date}.items():
        page = page.replace(k, v)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "index.html")
    open(out, "w", encoding="utf-8", newline="\n").write(page)
    print("wrote", out, f"({len(page) / 1024:.0f} KB)")


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>hipinfer results</title>
<meta name="description" content="Benchmarks and correctness results for hipinfer, an LLM inference engine written from scratch in HIP.">
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --ring: rgba(11,11,11,0.10);
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --ring: rgba(255,255,255,0.10);
    --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 32px 16px 56px; }
h1 { font-size: 30px; margin: 0 0 4px; letter-spacing: -0.01em; }
h2 { font-size: 20px; margin: 44px 0 6px; }
p { margin: 6px 0 12px; color: var(--ink2); max-width: 72ch; }
a { color: var(--s1); }
.lede { font-size: 17px; color: var(--ink2); margin-bottom: 22px; }
.links { display: flex; gap: 18px; flex-wrap: wrap; font-size: 14px; margin-bottom: 26px; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 14px 14px 12px; }
.tl { font-size: 13px; color: var(--ink2); }
.tv { font-size: 20px; font-weight: 600; margin: 4px 0 2px; font-variant-numeric: proportional-nums; }
.ts { font-size: 12px; color: var(--muted); }
.card { background: var(--surface); border: 1px solid var(--ring); border-radius: 10px; padding: 16px 16px 10px; margin-top: 12px; }
.ctitle { font-weight: 600; font-size: 15px; }
.csub { font-size: 13px; color: var(--ink2); margin: 2px 0 10px; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; color: var(--ink2); margin-bottom: 6px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; }
.lk { width: 16px; height: 0; border-top: 2px solid; display: inline-block; }
.mk { width: 2px; height: 14px; background: var(--ink); display: inline-block; }
.chart { position: relative; width: 100%; }
.chart svg { display: block; width: 100%; height: auto; overflow: visible; }
.chart text { font-size: 12px; fill: var(--ink2); font-variant-numeric: tabular-nums; }
.chart .lbl { fill: var(--ink); font-size: 13px; }
.chart .ax { fill: var(--muted); font-size: 11px; }
.tip { position: absolute; pointer-events: none; background: var(--surface); color: var(--ink); border: 1px solid var(--ring);
  border-radius: 8px; padding: 7px 10px; font-size: 13px; box-shadow: 0 4px 14px rgba(0,0,0,0.12); white-space: nowrap;
  opacity: 0; transition: opacity 80ms; z-index: 2; }
.tip b { font-weight: 600; font-variant-numeric: tabular-nums; }
.tip .row { display: flex; align-items: center; gap: 7px; }
details { margin: 8px 0 4px; font-size: 13px; }
summary { cursor: pointer; color: var(--ink2); }
.tbl { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; margin-top: 8px; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px 6px 0; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td { font-variant-numeric: tabular-nums; }
.note { font-size: 13px; color: var(--ink2); background: var(--surface); border: 1px solid var(--ring); border-radius: 10px;
  padding: 12px 16px; margin-top: 12px; }
.note ul { margin: 6px 0 0; padding-left: 18px; }
.note li { margin: 4px 0; }
footer { margin-top: 48px; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<main>
<h1>hipinfer</h1>
<div class="lede">An LLM inference engine written from scratch in HIP: tensor-core GEMMs, fused GEMV / RMSNorm / RoPE
kernels, FlashAttention-style attention, a graph-captured decode loop, and tensor parallelism across GPUs. One source
tree builds with <code>nvcc</code> for NVIDIA and <code>hipcc</code> for AMD. These are its measured results.</div>
<div class="links"><a href="%%REPO%%">Source on GitHub</a><a href="%%REPO%%#readme">README</a>
<a href="%%REPO%%/blob/main/results/SUMMARY.md">All tables (SUMMARY.md)</a><a href="%%REPO%%/tree/main/results">Raw results</a></div>

<div class="tiles">%%TILES%%</div>

<h2>Batch-1 decode</h2>
<p>Generating one token at a time is memory-bound: every token reads all the weights once. The <b>bandwidth floor</b>
is the bytes read per token divided by the GPU's peak bandwidth, the fastest any engine could go. %%GPU%%, fp16 weights,
128-token prompt. llama.cpp is <code>llama-bench</code> on an F16 GGUF, best of flash-attention on and off.</p>
<div class="card"><div class="ctitle">Decode latency per token (lower is better)</div>
<div class="csub">ms/token at a 128-token context</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp</span><span><i class="mk"></i>bandwidth floor</span></div>
<div class="chart" id="c-decode"></div>
<details><summary>Data table</summary>%%DEC_TBL%%</details></div>

<div class="card"><div class="ctitle">Share of the bandwidth floor as context grows (higher is better)</div>
<div class="csub">batch-1 decode, % of the floor including KV-cache reads</div>
<div class="legend" id="l-ctx"></div>
<div class="chart" id="c-ctx"></div>
<details><summary>Data table</summary>%%CTX_TBL%%</details></div>

<h2>Prefill</h2>
<p>Processing a 2,000-token prompt is compute-bound, so it measures the GEMMs and attention. Measured inside the full
benchmark suite, where the 2080 Ti's power cap holds it at 1.4–1.55 GHz; a short isolated run of Llama-3.2-3B gives
273.6 ms instead of 289.7.</p>
<div class="card"><div class="ctitle">Prefill time for 2,000 tokens (lower is better)</div>
<div class="csub">ms, %%GPU%%</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>hipinfer</span>
<span><i class="sw" style="background:var(--s2)"></i>llama.cpp</span></div>
<div class="chart" id="c-prefill"></div>
<details><summary>Data table</summary>%%PF_TBL%%</details></div>
%%A100%%

<h2>Two GPUs: tensor parallelism</h2>
<p>Each layer is split across both GPUs, Megatron-style, with an all-reduce after the attention and MLP outputs. These
two RTX 2080 Tis sit on different CPU sockets with no peer-to-peer path, so every byte between them crosses host memory:
close to the worst case for multi-GPU inference. Llama-3.1-8B and Mistral-7B don't fit one 11 GB card and run only sharded.</p>
<div class="card"><div class="ctitle">Decode latency per token, one GPU vs two (lower is better)</div>
<div class="csub">ms/token at a 128-token context</div>
<div class="legend"><span><i class="sw" style="background:var(--s1)"></i>1 GPU</span>
<span><i class="sw" style="background:var(--s3)"></i>2 GPUs</span></div>
<div class="chart" id="c-tp"></div>
<details><summary>Data table</summary>%%TP_TBL%%</details></div>
<div class="card"><div class="ctitle">All-reduce latency: hipinfer's one-shot kernel vs NCCL</div>
<div class="csub">fp16, in place, two GPUs. Decode-sized messages use the one-shot kernel; large prefill messages use NCCL.</div>
%%AR_TBL%%</div>
%%INT8%%
%%SPEC%%

<h2>Correctness</h2>
<p>Speed only counts if the output is right. <b>Greedy runs</b>: 7 prompts × 128 tokens, single-sequence with and without
CUDA graphs plus a batch of 6, compared token for token with HF transformers; any divergence lands where HF's own top-2
logits are tied or one fp16 step apart. <b>Perplexity</b>: WikiText-2, 1,024-token windows, both engines on the same tokens.</p>
<div class="card">%%COR_TBL%%</div>

<h2>Limits of these measurements</h2>
<div class="note"><ul>
<li><b>Thermal throttling.</b> Due to resource constraints and degrading GPUs, runs were at times throttled due to
overheating. The RTX A6000 reaches 90–93 °C under sustained load and throttles its SM clock to as low as 210 MHz: its
Llama-3.2-1B, Qwen2.5-1.5B, Llama-3.2-3B and Mistral-7B-v0.3 benchmarks ran partly throttled (listed in the
<a href="%%REPO%%#thermal-throttling">README</a> and <a href="%%REPO%%/blob/main/results/a6000/THERMAL.md">THERMAL.md</a>),
so the main charts here use the RTX 2080 Ti. The A100 ran in one Colab session with no throttling
(<a href="%%REPO%%/blob/main/results/a100/THERMAL.md">THERMAL.md</a>). Correctness results are unaffected.</li>
%%A100_LIMIT%%
<li><b>AMD support.</b> Because this project was built in HIP, it has full AMD support, and compiles properly. However, I had no available AMD GPUs and the cloud solutions available were either at capacity or too expensive.</li>
<li><b>Peer-to-peer multi-GPU.</b> The P2P all-reduce path is implemented but untested; only two GPUs have been tested.</li>
</ul></div>

<footer>Generated from <a href="%%REPO%%/tree/%%SHA%%/results">results/ at %%SHA%%</a> (%%DATE%%) by scripts/build_site.py.
The code was written with Claude (AI pair programmer); design decisions, benchmarks and claims were reviewed by the author,
Akash Dasarraju.</footer>
</main>
<script type="application/json" id="data">%%DATA%%</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById("data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  };
  const txt = (s, attrs, parent) => { const t = el("text", attrs, parent); t.textContent = s; return t; };
  const niceMax = v => { const p = Math.pow(10, Math.floor(Math.log10(v))); for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= v) return m * p; return 10 * p; };
  const fmt = (v, d) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

  function tooltip(host) {
    const t = document.createElement("div");
    t.className = "tip";
    host.appendChild(t);
    return {
      show(x, y, title, rows) {
        t.replaceChildren();
        const h = document.createElement("div"); h.textContent = title; h.style.color = "var(--ink2)"; t.appendChild(h);
        for (const [color, value, label, line] of rows) {
          const r = document.createElement("div"); r.className = "row";
          const k = document.createElement("i");
          if (line) { k.className = "lk"; k.style.borderColor = color; } else { k.className = "sw"; k.style.background = color; }
          const b = document.createElement("b"); b.textContent = value;
          const s = document.createElement("span"); s.textContent = label;
          r.append(k, b, s); t.appendChild(r);
        }
        t.style.opacity = 1;
        const w = t.offsetWidth, hw = host.clientWidth;
        t.style.left = Math.max(0, Math.min(hw - w, x + 12)) + "px";
        t.style.top = Math.max(0, y - t.offsetHeight - 10) + "px";
      },
      hide() { t.style.opacity = 0; },
    };
  }

  // Horizontal grouped bars: rows [{label, values:{key: number|null}, marker}], series [{key, label, color}]
  function bars(host, rows, series, opts) {
    host.replaceChildren();
    const W = Math.max(280, host.clientWidth), narrow = W < 560;
    const labelW = narrow ? 104 : 150, right = 66, bh = 14, gap = 2, pad = 16, top = 4, axisH = 22;
    const gh = series.length * bh + (series.length - 1) * gap;
    const H = top + rows.length * (gh + pad) + axisH;
    let mx = 0;
    for (const r of rows) { for (const s of series) if (r.values[s.key] != null) mx = Math.max(mx, r.values[s.key]); if (r.marker) mx = Math.max(mx, r.marker); }
    mx = niceMax(mx);
    const X = v => labelW + (W - labelW - right) * v / mx;
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria });
    const tip = tooltip(host);
    for (let i = 0; i <= 4; i++) {
      const v = mx * i / 4, x = X(v);
      el("line", { x1: x, x2: x, y1: top, y2: H - axisH, style: `stroke:var(${i ? "--grid" : "--axis"});stroke-width:1` }, svg);
      txt(fmt(v, v < 10 && v % 1 ? 1 : 0), { x, y: H - 6, "text-anchor": "middle", class: "ax" }, svg);
    }
    rows.forEach((r, i) => {
      const y0 = top + i * (gh + pad) + pad / 2;
      txt(r.label, { x: labelW - 10, y: y0 + gh / 2 + 4, "text-anchor": "end", class: "lbl" }, svg);
      series.forEach((s, j) => {
        const v = r.values[s.key], y = y0 + j * (bh + gap);
        if (v == null) {
          txt((r.missing && r.missing[s.key]) || opts.missing || "–", { x: (r.marker ? X(r.marker) : X(0)) + 6, y: y + bh - 3, class: "ax", "font-style": "italic" }, svg);
          return;
        }
        const x0 = X(0), x1 = Math.max(X(v), x0 + 2), rr = Math.min(4, x1 - x0);
        el("path", { d: `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + bh - rr}Q${x1},${y + bh} ${x1 - rr},${y + bh}H${x0}Z`,
                     style: `fill:var(${s.color})` }, svg);
        txt(fmt(v, opts.digits), { x: x1 + 6, y: y + bh - 3 }, svg);
        const hit = el("rect", { x: x0, y: y - gap, width: Math.max(x1 - x0, 24) + 40, height: bh + 2 * gap, fill: "transparent" }, svg);
        const u = opts.unit === "×" ? "×" : " " + opts.unit;
        const rowsT = [[`var(${s.color})`, fmt(v, opts.digits) + u, s.label]];
        if (r.marker) rowsT.push(["var(--ink)", fmt(r.marker, opts.digits) + u, opts.markerLabel]);
        hit.addEventListener("pointermove", ev => { const b = host.getBoundingClientRect(); tip.show(ev.clientX - b.left, ev.clientY - b.top, r.label, rowsT); });
        hit.addEventListener("pointerleave", () => tip.hide());
      });
      if (r.marker) {
        const x = X(r.marker);
        el("line", { x1: x, x2: x, y1: y0 - 3, y2: y0 + gh + 3, style: "stroke:var(--ink);stroke-width:2;pointer-events:none" }, svg);
      }
    });
    host.prepend(svg);
  }

  // Lines over a log2 x axis, crosshair tooltip with every series at the hovered x
  function lines(host, series, opts) {
    host.replaceChildren();
    const W = Math.max(280, host.clientWidth), H = 260, L = 44, R = 16, T = 10, B = 28;
    const xs = [...new Set(series.flatMap(s => s.points.map(p => p[0])))].sort((a, b) => a - b);
    const lx = v => Math.log2(v), x0 = lx(xs[0]), x1 = lx(xs[xs.length - 1]);
    const X = v => L + (W - L - R) * (lx(v) - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - (v - opts.yMin) / (opts.yMax - opts.yMin));
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.aria });
    for (let v = opts.yMin; v <= opts.yMax; v += opts.yStep) {
      el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), style: `stroke:var(${v === opts.yMin ? "--axis" : "--grid"});stroke-width:1` }, svg);
      txt(v + "%", { x: L - 6, y: Y(v) + 4, "text-anchor": "end", class: "ax" }, svg);
    }
    for (const v of xs) txt(v >= 1024 ? v / 1024 + "k" : String(v), { x: X(v), y: H - 8, "text-anchor": "middle", class: "ax" }, svg);
    for (const s of series) {
      el("polyline", { points: s.points.map(p => `${X(p[0])},${Y(p[1])}`).join(" "),
                       style: `fill:none;stroke:var(${s.color});stroke-width:2;stroke-linejoin:round;stroke-linecap:round` }, svg);
      const p = s.points[s.points.length - 1];
      el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, style: `fill:var(${s.color});stroke:var(--surface);stroke-width:2` }, svg);
    }
    const cross = el("line", { y1: T, y2: H - B, style: "stroke:var(--axis);stroke-width:1;opacity:0" }, svg);
    const tip = tooltip(host);
    const hit = el("rect", { x: L, y: T, width: W - L - R, height: H - T - B, fill: "transparent" }, svg);
    hit.addEventListener("pointermove", ev => {
      const b = svg.getBoundingClientRect(), px = (ev.clientX - b.left) * W / b.width;
      const v = xs.reduce((a, c) => Math.abs(X(c) - px) < Math.abs(X(a) - px) ? c : a, xs[0]);
      cross.setAttribute("x1", X(v)); cross.setAttribute("x2", X(v)); cross.style.opacity = 1;
      const rowsT = series.map(s => { const p = s.points.find(q => q[0] === v); return p && [`var(${s.color})`, p[1].toFixed(0) + "%", `${s.label} (${p[2].toFixed(2)} ms)`, true]; }).filter(Boolean)
                          .sort((a, c) => parseFloat(c[1]) - parseFloat(a[1]));
      const hb = host.getBoundingClientRect();
      tip.show(ev.clientX - hb.left, ev.clientY - hb.top, `context ${v.toLocaleString("en-US")} tokens`, rowsT);
    });
    hit.addEventListener("pointerleave", () => { tip.hide(); cross.style.opacity = 0; });
    host.prepend(svg);
  }

  const COLORS = ["--s1", "--s2", "--s3", "--s4", "--s5"];
  function render() {
    bars(document.getElementById("c-decode"),
         D.decode.map(r => ({ label: r.label, values: { h: r.hipinfer, l: r.llamacpp }, marker: r.floor })),
         [{ key: "h", label: "hipinfer", color: "--s1" }, { key: "l", label: "llama.cpp", color: "--s2" }],
         { unit: "ms", digits: 2, markerLabel: "bandwidth floor", aria: "Decode latency per token, hipinfer vs llama.cpp" });
    const ctx = D.ctx.slice(0, COLORS.length).map((s, i) => ({ label: s.label, color: COLORS[i], points: s.points }));
    const lg = document.getElementById("l-ctx");
    lg.replaceChildren(...ctx.map(s => { const sp = document.createElement("span"); const k = document.createElement("i"); k.className = "lk"; k.style.borderColor = `var(${s.color})`; sp.append(k, document.createTextNode(s.label)); return sp; }));
    lines(document.getElementById("c-ctx"), ctx, { yMin: 50, yMax: 100, yStep: 10, aria: "Share of the bandwidth floor vs context length" });
    bars(document.getElementById("c-prefill"),
         D.prefill.map(r => ({ label: r.label, values: { h: r.hipinfer, l: r.llamacpp } })),
         [{ key: "h", label: "hipinfer", color: "--s1" }, { key: "l", label: "llama.cpp", color: "--s2" }],
         { unit: "ms", digits: 1, aria: "Prefill time for 2,000 tokens" });
    bars(document.getElementById("c-tp"),
         D.tp.map(r => ({ label: r.label, values: { a: r.one, b: r.two } })),
         [{ key: "a", label: "1 GPU", color: "--s1" }, { key: "b", label: "2 GPUs", color: "--s3" }],
         { unit: "ms", digits: 2, missing: "doesn't fit one card", aria: "Decode latency per token, one GPU vs two" });
    if (D.a100) {
      const ser = [["hipinfer", "hipinfer", "--s1"], ["llamacpp", "llama.cpp", "--s2"], ["vllm", "vLLM", "--s3"],
                   ["torch_compile", "torch.compile", "--s4"]].map(([key, label, color]) => ({ key, label, color }));
      const miss = r => Object.fromEntries(Object.entries(r.why).map(([k, v]) => [k, v.split(" (")[0]]));
      bars(document.getElementById("c-a100-decode"),
           D.a100.rows.map(r => ({ label: r.label, values: r.decode, marker: r.floor, missing: miss(r) })), ser,
           { unit: "ms", digits: 2, markerLabel: "bandwidth floor", aria: "A100 decode latency per token, four engines" });
      bars(document.getElementById("c-a100-prefill"),
           D.a100.rows.map(r => ({ label: r.label, values: r.prefill, missing: miss(r) })), ser,
           { unit: "ms", digits: 1, aria: "A100 prefill time for 2,000 tokens, four engines" });
    }
    if (D.spec) {
      bars(document.getElementById("c-spec"),
           D.spec.rows.map(r => ({ label: `k = ${r.k}`, values: { m: r.speedup, p: r.pred }, marker: 1 })),
           [{ key: "m", label: "measured", color: "--s1" }, { key: "p", label: "cost model", color: "--s3" }],
           { unit: "×", digits: 2, markerLabel: "plain greedy decoding", aria: "Speculative decoding speed-up per k" });
    }
    if (D.int8) {
      const ser = [["int8", "hipinfer int8", "--s1"], ["fp16", "hipinfer fp16", "--s3"], ["q8", "llama.cpp Q8_0", "--s2"]]
        .map(([key, label, color]) => ({ key, label, color }));
      bars(document.getElementById("c-int8-decode"),
           D.int8.map(r => ({ label: r.label, values: r.decode["128"], marker: r.floor })), ser,
           { unit: "ms", digits: 2, markerLabel: "int8 bandwidth floor", aria: "Decode latency per token, int8 vs fp16 vs llama.cpp Q8_0" });
      bars(document.getElementById("c-int8-prefill"),
           D.int8.map(r => ({ label: r.label, values: r.prefill["2000"] })), ser,
           { unit: "ms", digits: 1, aria: "Prefill time for 2,000 tokens, int8 vs fp16 vs llama.cpp Q8_0" });
    }
  }
  render();
  // redraw at the new width whenever a chart's box changes size (window resize, late layout, zoom)
  let raf, widths = "";
  const hosts = ["c-decode", "c-ctx", "c-prefill", "c-tp", "c-a100-decode", "c-a100-prefill", "c-int8-decode", "c-int8-prefill", "c-spec"]
    .map(id => document.getElementById(id)).filter(Boolean);
  const key = () => hosts.map(h => h.clientWidth).join(",");
  widths = key();
  new ResizeObserver(() => {
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => { const k = key(); if (k !== widths) { widths = k; render(); } });
  }).observe(document.querySelector("main"));
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results"),
          sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "site"))
