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
         "tests/test_kernels.py", "tests/gate.py", "tests/ppl.py", "tests/test_tp.py", "scripts/bench_tp.py"]


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

**Runtime → Change runtime type → a GPU** (T4, L4, A100 or G4). The CPU and TPU runtimes cannot run
it: the kernels are GPU code (CUDA on NVIDIA, ROCm on AMD).

Run all cells. Per model the run takes ~10 min (kernel tests once, then HF token gate, WikiText-2 perplexity
vs HF, and benchmarks; most of the time is the HF reference). `MODELS` below picks the models; the defaults
fit a 15 GB T4. On an L4 (24 GB) add `llama32-3b`; on an A100 add `mistral-7b llama31-8b` too. It ends by
downloading `hipinfer_results_<gpu>.zip`; send that back to compare GPUs.
"""),
        code("""
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
!nvcc --version | tail -2
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| capability", torch.cuda.get_device_capability())
"""),
        *unpack_cells(blob, "/content/hipinfer", "!ls -R src scripts tests"),
        md("""
Build, kernel tests (vs PyTorch), then per model: HF token gate, WikiText-2 perplexity vs HF, benchmarks.
`GEMM_SWEEP=1` also times every GEMM tile config for the first model (adds ~3 min). Weights are exported
from the HF checkpoints on first use (`unsloth/Llama-3.2-*` are ungated mirrors of Meta's weights).
"""),
        code("""
!pip install -q datasets nvidia-ml-py
import torch
gpu = torch.cuda.get_device_name().replace(" ", "_")
MODELS = "tinyllama llama32-1b qwen25-0.5b qwen25-1.5b"   # + llama32-3b (24 GB GPUs), mistral-7b llama31-8b (40 GB+)
!chmod +x scripts/run_all.sh
!PY=python GEMM_SWEEP=1 PPL_WINDOWS=8 scripts/run_all.sh colab_{gpu} {MODELS}
"""),
        md("Summary tables (the same script produces results/SUMMARY.md in the repo)."),
        code("""
!python scripts/summarize.py results
"""),
        code("""
!zip -qr hipinfer_results_{gpu}.zip results
from google.colab import files
files.download(f"hipinfer_results_{gpu}.zip")
"""),
    ]
    write_nb("hipinfer_colab.ipynb", cells, {"accelerator": "GPU", "colab": {"gpuType": "T4"}}, blob)


if __name__ == "__main__":
    b = source_blob()
    colab(b)
    kaggle(b)
