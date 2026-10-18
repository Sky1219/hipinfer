"""WikiText-2 perplexity: hipinfer vs HF transformers (fp16), same tokens, same GPU.

Greedy-token agreement (tests/gate.py) only exercises the argmax; this compares the full
distribution at every position of real text: perplexity of both engines, top-1 agreement,
mean KL(HF || hipinfer) and the largest logit difference.

python tests/ppl.py --model unsloth/Llama-3.2-1B --weights llama32-1b.bin [--windows 16] [--seq 1024] [--out ppl.json]

int8 weights (version 4): with the default --ref fp16 the numbers are the quantization's cost (perplexity
change and KL against fp16 HF); --ref int8 runs HF on the exporter's int8 round-trip weights
(fake-quantized), which isolates the engine's own error from the quantization error.

With --tp N the engine is sharded over N GPUs and HF runs first (device_map="auto" over the same
GPUs), its logits parked in host memory, so a model too big for one GPU can be checked.
--hf-first does the same on one GPU (HF, logits to the host, HF freed, then the engine), for models whose
weights fit once but not twice; the numbers are the same computation, only reordered. Host memory needed:
windows * seq * vocab * 2 bytes (16 * 1024 * 128256 * 2 = 4.2 GB for Llama-3).
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import hipinfer as hi  # noqa: E402


def wikitext_ids(tok, split="test"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    return tok(text, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=16, help="number of non-overlapping windows (0 = whole test set)")
    ap.add_argument("--tp", type=int, default=1, help="shard the engine over this many GPUs")
    ap.add_argument("--hf-first", action="store_true", help="tp=1: run HF first, park its logits on the host, free it, then load the engine")
    ap.add_argument("--ref", choices=["fp16", "int8"], default="fp16",
                    help="HF reference weights: fp16, or the exporter's int8 round trip (fake-quantized)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    def hf_model(**kw):
        m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, **kw)
        if a.ref == "int8":
            from export import fake_quantize_hf
            fake_quantize_hf(m)
        return m

    tok = AutoTokenizer.from_pretrained(a.model)
    ids = wikitext_ids(tok)
    bos = tok.bos_token_id
    n_win = len(ids) // (a.seq - 1) if a.windows == 0 else min(a.windows, len(ids) // (a.seq - 1))
    print(f"{a.model}: {len(ids)} test tokens, {n_win} windows of {a.seq}")

    windows = []
    for w in range(n_win):
        chunk = ids[w * (a.seq - 1):(w + 1) * (a.seq - 1)]
        windows.append(([bos] if bos is not None else []) + chunk)
    phased = a.tp > 1 or a.hf_first
    if phased:  # two phases: HF over all visible GPUs first, logits kept on the host
        if a.tp > 1:
            model = hf_model(device_map="auto").eval()
        else:
            model = hf_model().cuda().eval()
        dev0 = next(model.parameters()).device
        with torch.no_grad():
            hf_logits = [model(torch.tensor(c, device=dev0)[None], use_cache=False).logits[0].cpu() for c in windows]
        del model
        import gc
        gc.collect()  # accelerate's dispatch hooks keep the weights alive until collected
        torch.cuda.empty_cache()
        print("free GPU memory before the engine:", [f"{torch.cuda.mem_get_info(d)[0] / 2**30:.1f} GB"
                                                    for d in range(a.tp)], flush=True)
    else:
        model = hf_model().cuda().eval()
    eng = hi.Engine(a.weights, max_batch=1, max_seq=a.seq + 8, max_prefill=a.seq, tp=a.tp)
    nll_hf = nll_hi = 0.0
    n_tok = agree = 0
    kl_sum = 0.0
    max_diff = 0.0
    for w in range(n_win):
        chunk = windows[w]
        t = torch.tensor(chunk, device=eng.device)
        if phased:
            lh = hf_logits[w].to(eng.device)
        else:
            with torch.no_grad():
                lh = model(t[None], use_cache=False).logits[0]  # [T, V] fp16
        li = eng.prefill_logits(0, chunk)                    # [T, V] fp16
        tgt = t[1:]
        # fp32 in row chunks: a 150k-token vocabulary makes [T, V] fp32 copies too big for 11 GB cards
        for r0 in range(0, tgt.numel(), 128):
            r1 = min(r0 + 128, tgt.numel())
            lh32, li32, tg = lh[r0:r1].float(), li[r0:r1].float(), tgt[r0:r1]
            nll_hf += F.cross_entropy(lh32, tg, reduction="sum").item()
            nll_hi += F.cross_entropy(li32, tg, reduction="sum").item()
            agree += (lh32.argmax(-1) == li32.argmax(-1)).sum().item()
            kl_sum += F.kl_div(F.log_softmax(li32, -1), F.log_softmax(lh32, -1), log_target=True, reduction="sum").item()
            max_diff = max(max_diff, (lh32 - li32).abs().max().item())
            del lh32, li32
        del lh, li
        n_tok += tgt.numel()
        print(f"window {w:3d}: ppl HF {math.exp(nll_hf / n_tok):.4f}  hipinfer {math.exp(nll_hi / n_tok):.4f}  "
              f"top-1 agree {100 * agree / n_tok:.3f}%  mean KL {kl_sum / n_tok:.2e}  max |dlogit| {max_diff:.3f}", flush=True)
    eng.close()
    out = {"model": a.model, "gpu": torch.cuda.get_device_name(), "tp": a.tp, "seq": a.seq, "windows": n_win, "tokens": n_tok,
           "engine_weights": "int8" if eng.quant else "fp16", "hf_ref": a.ref,
           "ppl_hf": math.exp(nll_hf / n_tok), "ppl_hipinfer": math.exp(nll_hi / n_tok),
           "ppl_change_pct": 100 * (math.exp((nll_hi - nll_hf) / n_tok) - 1),
           "top1_agreement": agree / n_tok, "mean_kl_hf_to_hipinfer": kl_sum / n_tok, "max_logit_diff": max_diff}
    print(json.dumps(out, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
