"""Strong baselines for the hipinfer end-to-end numbers (TinyLlama-1.1B, fp16, batch 1, greedy).

  llamacpp       llama.cpp `llama-bench` on an F16 GGUF (no quantization): prefill (pp) at 128/512/2000
                 tokens and decode (tg256) at context depth 128/1024. Best of flash-attn on/off and
                 ubatch 512/2048 is reported.
  torch_compile  gpt-fast-style hand-written Llama forward with a static KV cache. Decode step is
                 torch.compile(mode="reduce-overhead", fullgraph=True), i.e. inductor kernels replayed
                 from a CUDA graph. Greedy tokens are checked against HF eager.
  hf_static      HF transformers model with StaticCache, forward compiled the same way.
  vllm           vLLM LLM.generate, batch 1, greedy, 256 tokens after a 128/1024-token prompt.

Decode matches scripts/bench.py: prefill `ctx` random tokens, then time 256 greedy decode steps
(so the average context is ctx + 128); ms/token = wall time / 256 with a sync at both ends.

python scripts/bench_baselines.py torch_compile --out results/baselines/<gpu>/torch_compile.json
python scripts/bench_baselines.py llamacpp --llama-bench /path/llama-bench --gguf model-f16.gguf --out ...
See scripts/run_baselines.sh for the full run.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time

MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CTXS = (128, 1024)
PREFILLS = (128, 512, 2000)
N_DECODE = 256


def gpu_name():
    import torch
    return torch.cuda.get_device_name()


def dump(out, res):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


# ---------------------------------------------------------------------------------------- llama.cpp

def run_llamacpp(a):
    def bench(extra):
        cmd = [a.llama_bench, "-m", a.gguf, "-ngl", "99", "-r", str(a.reps), "-o", "json"] + extra
        print("$", " ".join(cmd), flush=True)
        p = subprocess.run(cmd, capture_output=True, text=True)
        sys.stderr.write(p.stderr[-2000:])
        if p.returncode:
            raise SystemExit(f"llama-bench failed ({p.returncode})")
        return json.loads(p.stdout)

    res = {"engine": "llama.cpp", "gpu": None, "runs": [], "prefill": [], "decode_b1": []}
    pp = bench(["-p", ",".join(map(str, PREFILLS)), "-n", "0", "-fa", "0,1", "-ub", "512,2048", "-b", "2048"])
    tg = bench(["-p", "0", "-n", str(N_DECODE), "-d", ",".join(map(str, CTXS)), "-fa", "0,1"])
    res["runs"] = pp + tg
    res["gpu"] = pp[0].get("gpu_info")
    res["build"] = f'{pp[0].get("build_commit")} ({pp[0].get("build_number")})'
    res["model_type"] = pp[0].get("model_type")
    for T in PREFILLS:
        rs = [r for r in pp if r["n_prompt"] == T]
        best = min(rs, key=lambda r: r["avg_ns"])
        res["prefill"].append({"T": T, "ms": best["avg_ns"] / 1e6, "tok_s": best["avg_ts"],
                               "flash_attn": best.get("flash_attn"), "n_ubatch": best.get("n_ubatch"),
                               "all_ms": {f'fa{r.get("flash_attn")}_ub{r.get("n_ubatch")}': r["avg_ns"] / 1e6 for r in rs}})
    for ctx in CTXS:
        rs = [r for r in tg if r.get("n_depth", 0) == ctx]
        best = min(rs, key=lambda r: r["avg_ns"])
        res["decode_b1"].append({"ctx": ctx, "ms": best["avg_ns"] / 1e6 / N_DECODE, "tok_s": best["avg_ts"],
                                 "flash_attn": best.get("flash_attn"),
                                 "all_ms": {f'fa{r.get("flash_attn")}': r["avg_ns"] / 1e6 / N_DECODE for r in rs}})
    for p in res["prefill"]:
        print(f'llama.cpp prefill T={p["T"]:5d}  {p["ms"]:8.2f} ms  (fa={p["flash_attn"]} ub={p["n_ubatch"]})')
    for d in res["decode_b1"]:
        print(f'llama.cpp decode  ctx={d["ctx"]:5d}  {d["ms"]:6.3f} ms/token  (fa={d["flash_attn"]})')
    dump(a.out, res)


# ------------------------------------------------------------------------- gpt-fast-style torch model

def build_gptfast(hf):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = hf.config
    L, Hd, Hq, Hkv = cfg.num_hidden_layers, cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
    D, eps = Hd // Hq, cfg.rms_norm_eps
    theta = getattr(cfg, "rope_theta", None) or cfg.rope_parameters["rope_theta"]
    sd = hf.state_dict()

    def norm(x, w):  # HF LlamaRMSNorm rounding
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
        return w * h.to(x.dtype)

    def rot_half(x):
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return torch.cat((-x2, x1), -1)

    class Llama(nn.Module):
        def __init__(self):
            super().__init__()
            g = lambda k: sd[k].detach()
            self.emb = g("model.embed_tokens.weight")
            self.lm_head = g("lm_head.weight")
            self.norm_w = g("model.norm.weight")
            self.layers = []
            for i in range(L):
                p = f"model.layers.{i}."
                self.layers.append(dict(
                    wqkv=torch.cat([g(p + "self_attn.q_proj.weight"), g(p + "self_attn.k_proj.weight"),
                                    g(p + "self_attn.v_proj.weight")]),
                    # QKV bias (Qwen2); None for Llama
                    bqkv=torch.cat([g(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
                    if p + "self_attn.q_proj.bias" in sd else None,
                    wo=g(p + "self_attn.o_proj.weight"),
                    w13=torch.cat([g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")]),
                    w2=g(p + "mlp.down_proj.weight"),
                    n1=g(p + "input_layernorm.weight"), n2=g(p + "post_attention_layernorm.weight")))
            self.kc = self.vc = None

        def setup_cache(self, S):
            dev = self.emb.device
            self.S = S
            self.kc = [torch.zeros(1, Hkv, S, D, dtype=torch.float16, device=dev) for _ in range(L)]
            self.vc = [torch.zeros(1, Hkv, S, D, dtype=torch.float16, device=dev) for _ in range(L)]
            inv = 1.0 / (theta ** (torch.arange(0, D, 2, device=dev).float() / D))
            f = torch.outer(torch.arange(S, device=dev).float(), inv)
            f = torch.cat((f, f), -1)
            self.cos, self.sin = f.cos().half(), f.sin().half()
            self.mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=dev))

        def forward(self, ids, pos):  # ids [1, T], pos [T]
            T = ids.shape[1]
            x = F.embedding(ids, self.emb)
            cos, sin = self.cos[pos][None, None], self.sin[pos][None, None]
            mask = self.mask[pos][None, None]
            for i, l in enumerate(self.layers):
                h = norm(x, l["n1"])
                q, k, v = F.linear(h, l["wqkv"], l["bqkv"]).split([Hq * D, Hkv * D, Hkv * D], -1)
                q = q.view(1, T, Hq, D).transpose(1, 2)
                k = k.view(1, T, Hkv, D).transpose(1, 2)
                v = v.view(1, T, Hkv, D).transpose(1, 2)
                q = q * cos + rot_half(q) * sin
                k = k * cos + rot_half(k) * sin
                self.kc[i].index_copy_(2, pos, k)
                self.vc[i].index_copy_(2, pos, v)
                kk = self.kc[i].repeat_interleave(Hq // Hkv, 1)
                vv = self.vc[i].repeat_interleave(Hq // Hkv, 1)
                o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask)
                x = x + F.linear(o.transpose(1, 2).reshape(1, T, Hq * D), l["wo"])
                gu = F.linear(norm(x, l["n2"]), l["w13"])
                gt, up = gu.chunk(2, -1)
                x = x + F.linear(F.silu(gt) * up, l["w2"])
            return F.linear(norm(x[:, -1:], self.norm_w), self.lm_head)

    return Llama()


def run_torch_compile(a):
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    dev = "cuda"
    res = {"engine": "torch.compile (gpt-fast style)", "gpu": gpu_name(), "torch": torch.__version__,
           "prefill": [], "decode_b1": []}
    hf = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    torch.set_grad_enabled(False)
    m = build_gptfast(hf)
    g = torch.Generator().manual_seed(0)
    rand_ids = lambda n: torch.randint(100, 30000, (1, n), generator=g).to(dev)

    def decode_step(tok, pos):
        return m(tok, pos).argmax(-1)

    step = torch.compile(decode_step, mode="reduce-overhead", fullgraph=True)
    prefill_c = torch.compile(m.forward, fullgraph=True, dynamic=False)

    # correctness: 32 greedy tokens vs HF eager
    m.setup_cache(256)
    prompt = rand_ids(32)
    ref = hf.generate(prompt, max_new_tokens=32, do_sample=False, min_new_tokens=32)[0, 32:].tolist()
    tok = m(prompt, torch.arange(32, device=dev)).argmax(-1)
    ours = [tok.item()]
    pos = torch.tensor([32], device=dev)
    for _ in range(31):
        tok = step(tok, pos).clone()
        pos += 1
        ours.append(tok.item())
    match = sum(x == y for x, y in zip(ours, ref))
    res["greedy_match_vs_hf"] = f"{match}/32"
    print(f"greedy tokens vs HF eager: {match}/32 match")

    for ctx in CTXS:
        S = math.ceil((ctx + N_DECODE) / 64) * 64
        m.setup_cache(S)
        torch._dynamo.reset()
        step = torch.compile(decode_step, mode="reduce-overhead", fullgraph=True)
        ids = rand_ids(ctx)
        times = []
        t_c = time.perf_counter()
        for rep in range(4):  # rep 0 = compile + cudagraph warmup
            tok = m(ids, torch.arange(ctx, device=dev)).argmax(-1)
            pos = torch.tensor([ctx], device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_DECODE):
                tok = step(tok, pos).clone()
                pos += 1
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_DECODE * 1e3)
            if rep == 0:
                print(f"  compile+first run ctx={ctx}: {time.perf_counter() - t_c:.0f} s", flush=True)
        ms = min(times[1:])
        res["decode_b1"].append({"ctx": ctx, "ms": ms, "cache_len": S, "all_ms": times[1:]})
        print(f"torch.compile decode ctx={ctx:5d}  {ms:6.3f} ms/token", flush=True)

    m.setup_cache(max(PREFILLS))
    for T in PREFILLS:
        ids = rand_ids(T)
        pos = torch.arange(T, device=dev)
        for _ in range(3):
            prefill_c(ids, pos)
        ts = []
        for _ in range(10):
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); prefill_c(ids, pos); e1.record(); e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        res["prefill"].append({"T": T, "ms": statistics.median(ts)})
        print(f"torch.compile prefill T={T:5d}  {statistics.median(ts):8.2f} ms", flush=True)
    dump(a.out, res)


# -------------------------------------------------------------------------- HF StaticCache + compile

def run_hf_static(a):
    import torch
    from transformers import AutoModelForCausalLM, StaticCache

    dev = "cuda"
    res = {"engine": "HF StaticCache + torch.compile", "gpu": gpu_name(), "torch": torch.__version__,
           "decode_b1": []}
    import transformers
    res["transformers"] = transformers.__version__
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev).eval()
    torch.set_grad_enabled(False)
    g = torch.Generator().manual_seed(0)
    fwd = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    for ctx in CTXS:
        S = math.ceil((ctx + N_DECODE) / 64) * 64
        cache = StaticCache(config=model.config, max_cache_len=S)
        ids = torch.randint(100, 30000, (1, ctx), generator=g).to(dev)
        times = []
        for rep in range(4):
            cache.reset()
            o = model(ids, past_key_values=cache, cache_position=torch.arange(ctx, device=dev), use_cache=True,
                      logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
            cp = torch.tensor([ctx], device=dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_DECODE):
                o = fwd(tok, past_key_values=cache, cache_position=cp, use_cache=True)
                tok = o.logits[:, -1:].argmax(-1).clone()
                cp += 1
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) / N_DECODE * 1e3)
        ms = min(times[1:])
        res["decode_b1"].append({"ctx": ctx, "ms": ms, "cache_len": S, "all_ms": times[1:]})
        print(f"HF static+compile decode ctx={ctx:5d}  {ms:6.3f} ms/token", flush=True)
    dump(a.out, res)


# ---------------------------------------------------------------------------------------------- vLLM

def run_vllm(a):
    from vllm import LLM, SamplingParams, TokensPrompt
    import random

    llm = LLM(a.model, dtype="float16", max_model_len=2048, gpu_memory_utilization=a.gpu_mem,
              enable_prefix_caching=False)
    import torch
    res = {"engine": "vLLM", "gpu": gpu_name(), "decode_b1": [], "prefill": []}
    import vllm
    res["vllm"] = vllm.__version__
    rng = random.Random(0)
    sp = lambda n: SamplingParams(temperature=0, max_tokens=n, min_tokens=n, ignore_eos=True)

    def gen(ids, n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate([TokensPrompt(prompt_token_ids=ids)], sp(n), use_tqdm=False)
        return (time.perf_counter() - t0) * 1e3

    for T in PREFILLS:
        ids = [rng.randrange(100, 30000) for _ in range(T)]
        gen(ids, 1)
        ts = [gen([rng.randrange(100, 30000) for _ in range(T)], 1) for _ in range(5)]
        res["prefill"].append({"T": T, "ms": statistics.median(ts), "note": "end-to-end generate() with 1 output token"})
        print(f"vLLM prefill(+1 tok) T={T:5d}  {statistics.median(ts):8.2f} ms", flush=True)
    for ctx in CTXS:
        ts = []
        for _ in range(3):
            ids = [rng.randrange(100, 30000) for _ in range(ctx)]
            t_pre = gen(ids, 1)
            t_all = gen(ids[:-1] + [ids[-1] ^ 1], N_DECODE + 1)  # different last token, no prefix caching anyway
            ts.append((t_all - t_pre) / N_DECODE)
        ms = min(ts)
        res["decode_b1"].append({"ctx": ctx, "ms": ms, "all_ms": ts,
                                 "method": "(generate 257 tok - generate 1 tok) / 256"})
        print(f"vLLM decode ctx={ctx:5d}  {ms:6.3f} ms/token", flush=True)
    dump(a.out, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["llamacpp", "torch_compile", "hf_static", "vllm"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--llama-bench", default="llama-bench")
    ap.add_argument("--gguf")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    a = ap.parse_args()
    {"llamacpp": run_llamacpp, "torch_compile": run_torch_compile, "hf_static": run_hf_static,
     "vllm": run_vllm}[a.which](a)


if __name__ == "__main__":
    main()
