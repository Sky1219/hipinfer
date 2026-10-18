"""Build hipinfer_colab_followup.ipynb: a second Colab A100 run (about 1 h) that fills the gaps in
results/a100 and results/baselines/a100. Most valuable first, in case the run is stopped early:

  1. build, and the MoE kernel test (test_moe failed to import scripts/ref_engine.py last time)
  2. Nsight Systems profile of batch-1 decode on TinyLlama and Llama-3.2-3B (is the A100's 61-74 %
     of the bandwidth floor launch overhead?)
  3. Granite-3.1-1B-A400M (MoE) end to end: token gate with the router near-tie rule, perplexity,
     benchmarks
  4. torch.compile on Qwen2.5-1.5B with the QKV bias fixed, hipinfer re-run beside it
  5. vLLM on Llama-3.2-3B (the disk filled last time), hipinfer re-run beside it

Sources are embedded as in scripts/build_colab.py (base64 tarball, LF endings, fixed metadata), so
the same sources always give the same notebook. docs/colab-followup.md says how to read the results.

python scripts/build_colab_followup.py           write the notebook, then check it
python scripts/build_colab_followup.py --check   check the committed notebook is current and valid
"""
import argparse
import ast
import base64
import io
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_colab import FILES, ROOT, code, md, source_blob, unpack_cells  # noqa: E402

NAME = "hipinfer_colab_followup.ipynb"
FOLLOWUP_FILES = FILES + ["scripts/profile_decode.py", "scripts/profile_decode.sh", "scripts/install_nsys.sh"]


def cells(blob):
    return [
        md("""
# hipinfer on Colab: the A100 follow-up run

Fills the gaps left by the first A100 run (`results/a100`, `results/baselines/a100`), most valuable
first, so stopping early still leaves the top results:

| step | what | about |
|---|---|---|
| 1 | build, MoE kernel test | 5 min |
| 2 | Nsight Systems profile of batch-1 decode, TinyLlama and Llama-3.2-3B: kernel time vs gaps, GEMV bandwidth | 10 min |
| 3 | Granite-3.1-1B-A400M (MoE): token gate with the router near-tie rule, perplexity, benchmarks | 10 min |
| 4 | torch.compile on Qwen2.5-1.5B with the QKV bias fixed (must match HF on 32/32 tokens), hipinfer beside it | 10 min |
| 5 | vLLM on Llama-3.2-3B, hipinfer beside it | 10 min |

**Runtime → Change runtime type → A100.** Then Runtime → Run all. About 1 h. Every step is timed in
`timeline.txt`, a failing step is logged and skipped rather than stopping the run, and the results
are zipped after every step to `/content/hipinfer_results_<tag>.zip` (Files pane), so they survive an
early stop. The last cell downloads the zip; send it back. `docs/colab-followup.md` in the repo says
what each step measures and how to read it.
"""),
        code("""
import os
os.environ["HF_HOME"] = "/root/.cache/huggingface"  # pinned before anything imports huggingface_hub: vLLM shares it
!nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
!nvcc --version | tail -2
!df -h /content | tail -1
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| capability", torch.cuda.get_device_capability())
if "A100" not in torch.cuda.get_device_name():
    print("WARNING: not an A100; the results will not line up with results/a100")
"""),
        *unpack_cells(blob, "/content/hipinfer"),
        md("""
**Setup.** Settings, helpers, the GPU log (`nvidia-smi` every 5 s with the throttle reasons) and,
in the background, the vLLM install: its own environment (it pins its own PyTorch), made with
`virtualenv` because Colab's Python has no `ensurepip`. It overlaps the build; the timed steps wait
for it to finish so it doesn't load the CPU while they measure.
"""),
        code(r'''
!pip install -q --no-cache-dir datasets nvidia-ml-py virtualenv
import contextlib, glob, json, re, shutil, subprocess, sys, time, traceback
from huggingface_hub import constants

# steps to run; turn one off here to skip it
RUN_PROFILE, RUN_GRANITE, RUN_TORCH_COMPILE, RUN_VLLM = True, True, True, True
VLLM_VERSION = "0.31.0"  # the version in the committed A100 comparison ("" = latest)
try:  # the HF_TOKEN Colab secret, exported so the scripts' own processes see it (all models here are ungated)
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    print("no HF_TOKEN Colab secret (fine: every model here is ungated)")

gpu = torch.cuda.get_device_name()
TAG = "a100_followup" if "A100" in gpu else "colab_" + gpu.replace(" ", "_") + "_followup"
R, W, BL = f"results/{TAG}", "/content/w", f"results/baselines/{TAG}"
ZIP = f"/content/hipinfer_results_{TAG}.zip"
os.makedirs(R, exist_ok=True)
os.makedirs(W, exist_ok=True)
HF = {"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama32-3b": "unsloth/Llama-3.2-3B",
      "qwen25-1.5b": "Qwen/Qwen2.5-1.5B", "granite-moe-1b": "ibm-granite/granite-3.1-1b-a400m-base"}
sh = get_ipython().system
FAILED = []


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{R}/timeline.txt", "a") as f:
        f.write(line + "\n")


def pack():
    """Zip what there is so far (the profiler's SQLite exports are left out: nsys re-creates them)."""
    sh(f"rm -f {ZIP} && zip -qr {ZIP} {R} {BL} -x '*.sqlite' 2>/dev/null")


@contextlib.contextmanager
def step(name):
    """Time a step; an exception is logged as a failure and Run all carries on."""
    t0 = time.time()
    log(f"== {name} ({shutil.disk_usage('/content').free / 1e9:.0f} GB free disk)")
    try:
        yield
    except Exception as e:
        traceback.print_exc()
        FAILED.append(f"{name}: {e}")
        log(f"   FAILED: {e}")
    log(f"   {name}: {time.time() - t0:.0f} s")
    pack()


def quiet(cmd, logfile, show):
    """Run cmd with its output in logfile, print the lines matching the regex `show`; True on exit 0."""
    rc = subprocess.run(f"{cmd} > {logfile} 2>&1", shell=True).returncode
    lines = open(logfile, errors="replace").read().splitlines()
    print("\n".join([l for l in lines if re.search(show, l)][-30:] or lines[-5:]))
    if rc:
        log(f"   exit code {rc}: {logfile}")
    return rc == 0


def weights(m):
    """hipinfer weights for model m, exported from its HF checkpoint (downloaded to the shared cache) if missing."""
    w = f"{W}/{m}.bin"
    if not os.path.exists(w):
        if not quiet(f"python scripts/export.py --model {HF[m]} --out {w}", f"{R}/export_{m}.log", r"wrote|Error"):
            raise RuntimeError(f"export of {m} failed")
    return w


def cleanup(m):
    """Delete model m's HF download and hipinfer weights, so the disk holds one model at a time."""
    shutil.rmtree(os.path.join(constants.HF_HUB_CACHE, "models--" + HF[m].replace("/", "--")), ignore_errors=True)
    if os.path.exists(f"{W}/{m}.bin"):
        os.remove(f"{W}/{m}.bin")
    log(f"   deleted {m}'s files: {shutil.disk_usage('/content').free / 1e9:.0f} GB free disk")


# GPU log: older drivers call the reasons clocks_throttle_reasons.*; fall back to fewer fields if one is unknown
why = "clocks_event_reasons" if subprocess.run("nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv",
                                               shell=True, capture_output=True).returncode == 0 else "clocks_throttle_reasons"
fields = "timestamp,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,memory.used," + ",".join(
    f"{why}.{r}" for r in ("active", "sw_power_cap", "hw_slowdown", "sw_thermal_slowdown", "hw_thermal_slowdown",
                           "hw_power_brake_slowdown"))
if subprocess.run(f"nvidia-smi --query-gpu={fields} --format=csv", shell=True, capture_output=True).returncode:
    fields = f"timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,{why}.sw_thermal_slowdown,{why}.hw_thermal_slowdown"
if "gpulog" not in globals() or gpulog.poll() is not None:
    gpulog = subprocess.Popen(f"nvidia-smi --query-gpu={fields} --format=csv -l 5 >> {R}/gpu_log.csv", shell=True)

with open(f"{R}/env.txt", "w") as f:
    f.write(subprocess.run("nvidia-smi; nvcc --version; python -c 'import torch, transformers; "
                           "print(torch.__version__, transformers.__version__)'",
                           shell=True, capture_output=True, text=True).stdout)

# vLLM in the background (virtualenv: Colab's Python has no ensurepip); the pinned version, else the latest
VLLM_PY = "/content/vllm_env/bin/python"
vllm_job = None
if RUN_VLLM and subprocess.run(f"{VLLM_PY} -c 'import vllm'", shell=True, capture_output=True).returncode:
    pin = f"vllm=={VLLM_VERSION}" if VLLM_VERSION else "vllm"
    vllm_job = subprocess.Popen(
        f"python -m virtualenv -q /content/vllm_env && (/content/vllm_env/bin/pip install -q --no-cache-dir {pin} "
        f"|| /content/vllm_env/bin/pip install -q --no-cache-dir vllm) && {VLLM_PY} -c 'import vllm; print(\"vllm\", vllm.__version__)'",
        shell=True, stdout=open(f"{R}/vllm_install.log", "w"), stderr=subprocess.STDOUT)
    log("vLLM install started in the background")


def wait_vllm():
    if vllm_job is not None and vllm_job.poll() is None:
        log("   waiting for the vLLM install to finish (so it doesn't load the CPU during timings)")
        vllm_job.wait()


print(f"{gpu}, results in {R} and {BL}, {shutil.disk_usage('/content').free / 1e9:.0f} GB free disk")
'''),
        md("""
**1. Build and MoE kernel test.** Builds for this GPU (`sm_80` on an A100), then `test_moe()` from
`tests/test_kernels.py`: the MoE router and both expert paths (GEMV, grouped GEMM) against
`scripts/ref_engine.py`'s MoE math, for Granite-, Mixtral- and OLMoE-shaped layers. Last time it
failed to import `ref_engine`; the other kernel tests already passed on this GPU
(`results/a100/test_kernels.log`).
"""),
        code(r'''
with step("build + MoE kernel test"):
    sh(f"chmod +x scripts/*.sh && PY=python SKIP_SINGLE=1 SKIP_TP=1 SKIP_KERNEL_TESTS=1 scripts/run_all.sh {TAG}")
    if not os.path.exists("build/libhipinfer.so"):
        raise RuntimeError("build failed")
    test = ("import sys; sys.path.insert(0, 'tests'); import test_kernels as t; print(t.torch.cuda.get_device_name()); "
            "t.test_moe(); print('\\nALL PASS' if not t.FAILS else f'\\n{len(t.FAILS)} FAILURES: {t.FAILS}'); "
            "sys.exit(1 if t.FAILS else 0)")
    if not quiet(f'python -c "{test}"', f"{R}/test_kernels_moe.log", r"FAIL|ALL PASS|Error"):
        FAILED.append("MoE kernel test")
'''),
        md("""
**2. Profile batch-1 decode with Nsight Systems** (installed if missing), TinyLlama and Llama-3.2-3B,
128 tokens after a 128-token prompt. The README puts the A100's lower share of the bandwidth floor
(61–74 %, against 81–87 % on a 2080 Ti) down to per-kernel launch overhead. Decode replays a CUDA
graph, so nsys traces every graph node (`--cuda-graph-trace=node`). Per token: the kernel count, the
sum of kernel time against wall time, the gaps between kernels, and each GEMV's achieved bandwidth.
The run is repeated without the profiler for reference timings and the measured DRAM read bandwidth.
Results: `profile/<model>/summary.json`, the `nsys stats` tables (`stats_*.csv`) and the `.nsys-rep`.
"""),
        code(r'''
NSYS = None
if RUN_PROFILE:
    with step("install nsys"):
        r = subprocess.run("bash scripts/install_nsys.sh", shell=True, capture_output=True, text=True)
        NSYS = r.stdout.strip().splitlines()[-1] if r.returncode == 0 and r.stdout.strip() else None
        print(NSYS or r.stdout[-2000:] + r.stderr[-2000:])
        if not NSYS:
            raise RuntimeError("no nsys: profile skipped")
    wait_vllm()
for m in (["tinyllama", "llama32-3b"] if NSYS else []):
    with step(f"profile {m}"):
        out = f"{R}/profile/{m}"
        os.makedirs(f"{R}/profile", exist_ok=True)
        if os.path.exists(f"{out}/summary.json"):
            log("   done earlier in this runtime, skipped")
        else:
            sh(f"NSYS={NSYS} PY=python scripts/profile_decode.sh {out} {weights(m)} 128 128 2>&1 | tee {out}.log")
            if not os.path.exists(f"{out}/summary.json"):
                raise RuntimeError(f"no summary, see {out}.log")
    if not (m == "llama32-3b" and RUN_VLLM):  # step 5 benchmarks the 3B again: keep its files until then
        cleanup(m)
'''),
        md("""
**3. Granite-3.1-1B-A400M (MoE) end to end**, the first MoE run on the A100: export, the HF token gate
(with the router near-tie rule, `--router-tie 0.01`: a divergence is accepted only if router near-ties
explain it), WikiText-2 perplexity against HF over 16 windows, and the benchmarks.
"""),
        code(r'''
m = "granite-moe-1b"
if RUN_GRANITE:
    with step(f"{m}: token gate, perplexity, benchmarks"):
        o, w = f"{R}/{m}", weights(m)
        os.makedirs(o, exist_ok=True)
        if not quiet(f"python tests/gate.py --model {HF[m]} --weights {w} --router-tie 0.01 --out {o}/gate.json",
                     f"{o}/gate.log", r"diverges|runs token-identical|GATE|Error"):
            FAILED.append(f"{m} gate")
        if not quiet(f"python tests/ppl.py --model {HF[m]} --weights {w} --windows 16 --out {o}/ppl.json",
                     f"{o}/ppl.log", r"^window +16:|ppl|Error"):
            FAILED.append(f"{m} ppl")
        wait_vllm()
        if not quiet(f"python scripts/bench.py --model {HF[m]} --weights {w} --out {o}/bench.json",
                     f"{o}/bench.log", r"^(T=|ctx=|B=)|Error"):
            FAILED.append(f"{m} bench")
    cleanup(m)
'''),
        md("""
**4. torch.compile on Qwen2.5-1.5B**, now with Qwen2's QKV bias (last time it was left out, the tokens
didn't match HF and the numbers were thrown away). It must match HF eager on 32/32 greedy tokens.
hipinfer is benchmarked again in the same session (`ONLY=` in `scripts/run_baselines.sh`).
"""),
        code(r'''
m = "qwen25-1.5b"
if RUN_TORCH_COMPILE:
    with step(f"{m}: torch.compile with the QKV bias, hipinfer beside it"):
        wait_vllm()
        sh(f'ONLY="hipinfer torch_compile" PY=python MODEL={HF[m]} WEIGHTS={weights(m)} SCRATCH=/content/tmp '
           f'scripts/run_baselines.sh {TAG}/{m}')
        tc = f"{BL}/{m}/torch_compile.json"
        match = json.load(open(tc)).get("greedy_match_vs_hf") if os.path.exists(tc) else None
        log(f"   torch.compile greedy tokens vs HF: {match}" + ("" if match == "32/32" else "  <- not 32/32: not valid"))
        if match != "32/32":
            FAILED.append(f"{m} torch.compile tokens {match}")
    cleanup(m)
'''),
        md("""
**5. vLLM on Llama-3.2-3B** (last time the disk filled before vLLM could load it), with `HF_HOME`
pinned so vLLM loads the same download, and hipinfer benchmarked again beside it.
"""),
        code(r'''
m = "llama32-3b"
if RUN_VLLM:
    with step(f"{m}: vLLM, hipinfer beside it"):
        wait_vllm()
        print(open(f"{R}/vllm_install.log").read()[-1500:] if os.path.exists(f"{R}/vllm_install.log") else "")
        if subprocess.run(f"{VLLM_PY} -c 'import vllm'", shell=True).returncode:
            raise RuntimeError(f"vLLM install failed, see {R}/vllm_install.log")
        sh(f'ONLY="hipinfer vllm" PY=python MODEL={HF[m]} WEIGHTS={weights(m)} VLLM_PY={VLLM_PY} '
           f'HF_HOME={os.environ["HF_HOME"]} SCRATCH=/content/tmp scripts/run_baselines.sh {TAG}/{m}')
        if not os.path.exists(f"{BL}/{m}/vllm.json"):
            raise RuntimeError(f"no vllm.json, see {BL}/{m}/vllm.log")
    cleanup(m)
'''),
        md("**Summary**, then download the zip (also at `/content/hipinfer_results_<tag>.zip`)."),
        code(r'''
gpulog.terminate()
for f in sorted(glob.glob(f"{R}/profile/*/summary.json")):
    s = json.load(open(f))
    for k in ("decode_graph", "decode_nograph"):
        d = s.get(k, {})
        if "shortfall" in d:
            sf = d["shortfall"]
            print(f"{f.split('/')[-2]:11s} {k:15s} {d['wall_ms']:.3f} ms/token = kernels {d['kernel_sum_ms']:.3f} + gaps "
                  f"{d['gap_ms']:.3f} | {d['kernels_per_step']} kernels | {sf['pct_of_floor']:.0f}% of floor, "
                  f"{sf['pct_of_floor_if_no_gaps']:.0f}% without gaps | {d['verdict']}")
for f in sorted(glob.glob(f"{R}/*/gate.json") + glob.glob(f"{R}/*/ppl.json")):
    print(f, json.dumps({k: v for k, v in json.load(open(f)).items() if not isinstance(v, (list, dict))})[:300])
for f in sorted(glob.glob(f"{BL}/*/*.json")):
    d = json.load(open(f))
    print(f, "decode", [round(x.get("ms", x.get("ours_ms", 0)), 3) for x in d.get("decode_b1", d.get("e2e", {}).get("decode_b1", []))],
          d.get("greedy_match_vs_hf", ""), d.get("vllm", ""))
print("\nFAILED:", FAILED if FAILED else "nothing")
log("run finished")
pack()
from google.colab import files
files.download(ZIP)
'''),
    ]


def notebook_text(blob):
    nb = {"cells": cells(blob), "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
                                             "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    return json.dumps(nb, indent=1) + "\n"


def check(text):
    """Notebook JSON valid, every code cell valid Python once IPython magics are blanked, and the
    embedded tarball holds exactly FOLLOWUP_FILES with the sources' (LF) contents."""
    nb = json.loads(text)
    n = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        py = re.sub(r"^(\s*)[!%].*$", r"\1pass", src, flags=re.M)
        try:
            ast.parse(py)
        except SyntaxError as e:
            raise SystemExit(f"cell {i}: {e}\n{src}")
        n += 1
        m = re.search(r'BLOB = "([A-Za-z0-9+/=]+)"', src)
        if m:
            tar = tarfile.open(fileobj=io.BytesIO(base64.b64decode(m.group(1))), mode="r:gz")
            names = tar.getnames()
            assert names == FOLLOWUP_FILES, set(names) ^ set(FOLLOWUP_FILES)
            for f in names:
                want = open(os.path.join(ROOT, f), "rb").read().replace(b"\r\n", b"\n")
                assert tar.extractfile(f).read() == want, f"embedded {f} differs from the source"
            print(f"embedded sources: {len(names)} files, all match")
    print(f"{n} code cells parse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="check the committed notebook instead of writing it")
    a = ap.parse_args()
    blob = source_blob(FOLLOWUP_FILES)
    text = notebook_text(blob)
    out = os.path.join(ROOT, NAME)
    if a.check:
        current = open(out, encoding="utf-8").read()
        if current.replace("\r\n", "\n") != text:
            raise SystemExit(f"{NAME} is out of date: run python scripts/build_colab_followup.py")
        print(f"{NAME} is current")
    else:
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print("wrote", out, f"({len(blob) / 1024:.0f} KB embedded)")
    check(open(out, encoding="utf-8").read())


if __name__ == "__main__":
    main()
