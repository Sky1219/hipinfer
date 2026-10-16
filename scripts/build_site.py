"""Build the results page (site/index.html) from results/: one self-contained HTML file, no
external scripts or fonts. Published to GitHub Pages by .github/workflows/pages.yml.

python scripts/build_site.py [results_dir] [out_dir]
"""
import html
import json
import os
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
            "correct": correct}


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
    data = json.dumps({k: d[k] for k in ("decode", "prefill", "ctx", "tp")}).replace("</", "<\\/")
    page = PAGE
    for k, v in {"%%TILES%%": tiles(d), "%%GPU%%": html.escape(gpu), "%%DEC_TBL%%": dec_tbl, "%%CTX_TBL%%": ctx_tbl,
                 "%%PF_TBL%%": pf_tbl, "%%TP_TBL%%": tp_tbl, "%%AR_TBL%%": ar_tbl, "%%COR_TBL%%": cor_tbl,
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

<h2>Correctness</h2>
<p>Speed only counts if the output is right. <b>Greedy runs</b>: 7 prompts × 128 tokens, single-sequence with and without
CUDA graphs plus a batch of 6, compared token for token with HF transformers; any divergence lands where HF's own top-2
logits are tied or one fp16 step apart. <b>Perplexity</b>: WikiText-2, 1,024-token windows, both engines on the same tokens.</p>
<div class="card">%%COR_TBL%%</div>

<h2>Not measured yet</h2>
<div class="note"><ul>
<li><b>RTX A6000 speed.</b> Its correctness results are above. Its speed run was discarded: the GPU was shared with another job
and thermally throttled (93 °C, SM clock at 210 of 2,100 MHz).</li>
<li><b>AMD.</b> The code compiles for MI200, MI300 and RDNA3, but it hasn't run on AMD hardware.</li>
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
          txt(opts.missing || "–", { x: X(0) + 6, y: y + bh - 3, class: "ax", "font-style": "italic" }, svg);
          return;
        }
        const x0 = X(0), x1 = Math.max(X(v), x0 + 2), rr = Math.min(4, x1 - x0);
        el("path", { d: `M${x0},${y}H${x1 - rr}Q${x1},${y} ${x1},${y + rr}V${y + bh - rr}Q${x1},${y + bh} ${x1 - rr},${y + bh}H${x0}Z`,
                     style: `fill:var(${s.color})` }, svg);
        txt(fmt(v, opts.digits), { x: x1 + 6, y: y + bh - 3 }, svg);
        const hit = el("rect", { x: x0, y: y - gap, width: Math.max(x1 - x0, 24) + 40, height: bh + 2 * gap, fill: "transparent" }, svg);
        const rowsT = [[`var(${s.color})`, fmt(v, opts.digits) + " " + opts.unit, s.label]];
        if (r.marker) rowsT.push(["var(--ink)", fmt(r.marker, opts.digits) + " " + opts.unit, opts.markerLabel]);
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
  }
  render();
  // redraw at the new width whenever a chart's box changes size (window resize, late layout, zoom)
  let raf, widths = "";
  const hosts = ["c-decode", "c-ctx", "c-prefill", "c-tp"].map(id => document.getElementById(id));
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
