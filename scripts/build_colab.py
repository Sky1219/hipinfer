"""Build hipinfer_colab.ipynb (one GPU) and hipinfer_kaggle.ipynb (Kaggle's free 2x T4: the
single-GPU suite plus tensor parallelism over both GPUs, including Llama-3.1-8B, which only fits
sharded): self-contained notebooks that unpack the sources (embedded as a base64 tarball, so they
run without cloning), build for the attached GPUs and run scripts/run_all.sh. Re-run this after
changing any source file.

python scripts/build_colab.py
"""
import base64
import io
import json
import os
import tarfile

ROOT = os.path.join(os.path.dirname(__file__), "..")
FILES = ["Makefile", "src/hip_compat.h", "src/kernels.cuh", "src/api.cu", "scripts/hipinfer.py",
         "scripts/export.py", "scripts/bench.py", "scripts/run_all.sh", "scripts/summarize.py",
         "tests/test_kernels.py", "tests/gate.py", "tests/ppl.py", "tests/test_tp.py", "scripts/bench_tp.py",
         "scripts/run_baselines.sh", "scripts/bench_baselines.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def source_blob():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in FILES:
            tar.add(os.path.join(ROOT, f), arcname=f)
    return base64.b64encode(buf.getvalue()).decode()


def unpack_cells(blob, dest, extra=""):
    """Markdown + code cell that unpack the embedded sources into dest and cd there."""
    return [md("Unpack the sources (embedded below so no clone is needed)."),
            code(f"""
import base64, io, tarfile, os
BLOB = "{blob}"
os.makedirs("{dest}", exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BLOB)), mode="r:gz").extractall("{dest}")
%cd {dest}
{extra}
""")]


def write_nb(name, cells, metadata, blob):
    nb = {"cells": cells, "metadata": {**metadata, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = os.path.join(ROOT, name)
    json.dump(nb, open(out, "w"), indent=1)
    print("wrote", out, f"({len(blob) / 1024:.0f} KB embedded)")


def kaggle(blob):
    cells = [
        md("""
# hipinfer on Kaggle: 2x T4, tensor parallelism

LLM inference engine written from scratch in HIP, built here with `nvcc`. Kaggle's free **GPU T4 x2**
machine runs both halves of the benchmark: every model on one T4, then sharded over both T4s with
tensor parallelism (Megatron-style column/row-parallel layers, a custom one-shot all-reduce kernel for
decode-sized messages, NCCL for prefill-sized ones, prefill all-reduces overlapped with compute).
Llama-3.1-8B (16 GB of fp16 weights) does not fit one 15 GB T4 and runs only sharded.

**Settings -> Accelerator -> GPU T4 x2**, and **Internet on** (weights come from Hugging Face).
Run all. It takes ~2 h (most of it HF reference runs and exporting the 8B weights). Results land
in `/kaggle/working/hipinfer_results_kaggle_2xT4.zip` (Output tab).
"""),
        code("""
!nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
!nvidia-smi topo -m
!nvcc --version | tail -2
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPUs", torch.cuda.device_count())
"""),
        *unpack_cells(blob, "/kaggle/working/hipinfer"),
        md("""
Single-GPU suite (kernel tests, HF token gate, perplexity vs HF, benchmarks) for `MODELS`, then the
tensor-parallel stage for `TP_MODELS`: TP-vs-1-GPU token test, all-reduce latency, decode/prefill/batched
benchmarks against one GPU, and for the 8B (sharded only) perplexity vs HF spread over both GPUs. Weights
and the HF cache go to /tmp (the 20 GB /kaggle/working limit is too small for the 8B).
"""),
        code("""
!pip install -q datasets nvidia-ml-py
import os
os.environ["HF_HOME"] = "/tmp/hf"
MODELS = "tinyllama llama32-1b qwen25-1.5b llama32-3b"
TP_MODELS = "tinyllama llama32-1b qwen25-1.5b llama32-3b llama31-8b"
!chmod +x scripts/run_all.sh
!mkdir -p /tmp/w
!PY=python WEIGHTS_DIR=/tmp/w PPL_WINDOWS=8 TP=2 TP_MODELS="{TP_MODELS}" scripts/run_all.sh kaggle_2xT4 {MODELS}
"""),
        code("""
!python scripts/summarize.py results
!zip -qr /kaggle/working/hipinfer_results_kaggle_2xT4.zip results
"""),
    ]
    write_nb("hipinfer_kaggle.ipynb", cells, {"kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True,
                                                         "isGpuEnabled": True}}, blob)


def colab(blob):
    cells = [
        md("""
# hipinfer on Colab

LLM inference engine written from scratch in HIP (FlashAttention-style attention, tensor-core GEMM,
fused GEMV/RMSNorm/RoPE kernels), built here with `nvcc` for the attached NVIDIA GPU.

**Runtime → Change runtime type → a GPU.** An **A100** is the most useful: it fits every model next
to its HF reference copy and has the bandwidth to show launch overheads. L4 and T4 work too, with fewer
models. The CPU and TPU runtimes cannot run it: the kernels are GPU code.

Run all cells. The notebook sizes itself to the GPU:

| GPU memory | models | engine comparison (llama.cpp, vLLM, torch.compile, HF) |
|---|---|---|
| 40 GB+ (A100) | all 7, up to Llama-3.1-8B | on, 5 models |
| 22–39 GB (L4) | up to Llama-3.2-3B | on, up to 3B |
| less (T4) | 4 small models | off |

Expected time on an A100: about 1–1.5 h for the hipinfer suite (kernel tests, HF token gate, WikiText-2
perplexity and benchmarks per model; the HF reference runs take most of it), plus about 1 h for the engine
comparison. It ends by downloading `hipinfer_results_<gpu>.zip`; send that back.
"""),
        code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!nvcc --version | tail -2
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| capability", torch.cuda.get_device_capability())
"""),
        *unpack_cells(blob, "/content/hipinfer", "!ls -R src scripts tests"),
        md("""
**Settings.** Chosen from the GPU's memory; edit `MODELS`, `RUN_BASELINES` or `BASELINE_MODELS` to
override. Weights are exported from the HF checkpoints on first use (`unsloth/Llama-3.2-*` and
`unsloth/Meta-Llama-3.1-8B` are ungated mirrors of Meta's weights). Mistral-7B is gated: accept its terms
on huggingface.co and add an `HF_TOKEN` Colab secret (key icon, left sidebar) to include it; without that it
is skipped.
"""),
        code("""
!pip install -q datasets nvidia-ml-py
import os, shutil, subprocess, torch
gpu = torch.cuda.get_device_name().replace(" ", "_")
mem_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
cap = "%d%d" % torch.cuda.get_device_capability()
if mem_gb >= 38:
    MODELS = "tinyllama llama32-1b qwen25-0.5b qwen25-1.5b llama32-3b mistral-7b llama31-8b"
elif mem_gb >= 22:
    MODELS = "tinyllama llama32-1b qwen25-0.5b qwen25-1.5b llama32-3b"
else:
    MODELS = "tinyllama llama32-1b qwen25-0.5b qwen25-1.5b"
try:  # Mistral's checkpoint is gated: needs its terms accepted on HF and an HF_TOKEN Colab secret
    from huggingface_hub import auth_check
    auth_check("mistralai/Mistral-7B-v0.3")
except ImportError:
    pass
except Exception:
    print("no access to mistralai/Mistral-7B-v0.3 (gated) -> skipping mistral-7b")
    MODELS = MODELS.replace(" mistral-7b", "")
RUN_BASELINES = mem_gb >= 22
BASELINE_MODELS = [m for m in ["tinyllama", "llama32-1b", "qwen25-1.5b", "llama32-3b", "llama31-8b"] if m in MODELS.split()]
print(f"{gpu}: {mem_gb:.0f} GB, sm_{cap}, {shutil.disk_usage('/content').free / 1e9:.0f} GB free disk")
print("hipinfer suite:", MODELS)
print("engine comparison:", BASELINE_MODELS if RUN_BASELINES else "off")
"""),
        md("""
**hipinfer suite**: build for this GPU, kernel tests (vs PyTorch), then per model the HF token gate,
WikiText-2 perplexity vs HF and the benchmarks. `GEMM_SWEEP=1` also times every GEMM tile config for
the first model. A background `nvidia-smi` log records temperature and clocks every 5 s for the whole
run (`results/colab_<gpu>/gpu_log.csv`), so the results can be checked for thermal throttling.
"""),
        code("""
os.makedirs(f"results/colab_{gpu}", exist_ok=True)
# older drivers call the throttle fields clocks_throttle_reasons.*
why = "clocks_event_reasons" if subprocess.run("nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv",
                                               shell=True, capture_output=True).returncode == 0 else "clocks_throttle_reasons"
gpulog = subprocess.Popen(
    "nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,clocks.sm,utilization.gpu,"
    f"{why}.sw_thermal_slowdown,{why}.hw_thermal_slowdown "
    f"--format=csv -l 5 > results/colab_{gpu}/gpu_log.csv", shell=True)
!chmod +x scripts/run_all.sh scripts/run_baselines.sh
!PY=python GEMM_SWEEP=1 PPL_WINDOWS=8 scripts/run_all.sh colab_{gpu} {MODELS}
"""),
        md("""
**Engine comparison** (skipped when `RUN_BASELINES` is off): llama.cpp, vLLM, torch.compile and HF
transformers on the same GPU and models, with hipinfer re-run beside them in the same session.

- llama.cpp is built from source for this GPU only (about 5 min). Its converter makes an F16 GGUF per
  model, which is deleted after that model to save disk.
- vLLM goes in its own virtual environment (about 5 min to install), because it pins its own PyTorch
  and would otherwise replace the one hipinfer is built against.
- A model is skipped if the disk is too full for its GGUF.
"""),
        code("""
sh = get_ipython().system
if RUN_BASELINES:
    # llama.cpp's converter needs only these (its requirements file would replace Colab's PyTorch)
    sh("pip install -q sentencepiece protobuf ninja")
    sh("[ -x /content/vllm_env/bin/python ] || (python -m venv /content/vllm_env && /content/vllm_env/bin/pip install -q vllm)")
    sh("/content/vllm_env/bin/python -c 'import vllm; print(\\"vllm\\", vllm.__version__)'")
    HF = {"tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama32-1b": "unsloth/Llama-3.2-1B",
          "qwen25-1.5b": "Qwen/Qwen2.5-1.5B", "llama32-3b": "unsloth/Llama-3.2-3B",
          "llama31-8b": "unsloth/Meta-Llama-3.1-8B"}
    GB = {"tinyllama": 2.2, "llama32-1b": 2.5, "qwen25-1.5b": 3.1, "llama32-3b": 6.4, "llama31-8b": 16.1}
    for m in BASELINE_MODELS:
        free = shutil.disk_usage("/content").free / 1e9
        if free < GB[m] * 1.5 + 5:
            print(f"== skipping {m}: only {free:.0f} GB free disk")
            continue
        gguf = f"/content/{m}-f16.gguf"
        print(f"== {m}")
        sh(f"PY=python LLAMA_CPP=/content/llama.cpp CUDA_ARCHS={cap} GGUF={gguf} MODEL={HF[m]} WEIGHTS={m}.bin "
           f"VLLM_PY=/content/vllm_env/bin/python SCRATCH=/content/tmp scripts/run_baselines.sh colab_{gpu}/{m}")
        if os.path.exists(gguf):
            os.remove(gguf)
"""),
        md("Summary tables (the same script produces results/SUMMARY.md in the repo), then download the results."),
        code("""
gpulog.terminate()
!python scripts/summarize.py results
"""),
        code("""
!zip -qr hipinfer_results_{gpu}.zip results
from google.colab import files
files.download(f"hipinfer_results_{gpu}.zip")
"""),
    ]
    write_nb("hipinfer_colab.ipynb", cells, {"accelerator": "GPU", "colab": {"gpuType": "A100"}}, blob)


if __name__ == "__main__":
    b = source_blob()
    colab(b)
    kaggle(b)
