"""ctypes binding for build/libhipinfer.so (kernels on torch tensors + the engine)."""
import ctypes as C
import os

import torch

_LIB = os.environ.get("HIPINFER_LIB", os.path.join(os.path.dirname(__file__), "..", "build", "libhipinfer.so"))
lib = C.CDLL(os.path.abspath(_LIB))

P, I, F = C.c_void_p, C.c_int, C.c_float
IP = C.POINTER(C.c_int)
lib.hi_rmsnorm.argtypes = [P, P, P, I, I, F, P]
lib.hi_gemv.argtypes = [P, I, P, F, P, P, I, I, I, I, I, P]
lib.hi_gemv_qkv_rope.argtypes = [P, I, P, F, P, P, P, I, I, I, I, I, I, P, P, P, P, I, P]
lib.hi_decode_work_bytes.argtypes = [I, I, I, I]
lib.hi_decode_work_bytes.restype = C.c_size_t
lib.hi_gemm_force_splits.argtypes = [I]
lib.hi_gemm.argtypes = [P, I, P, P, P, I, I, I, I, I, P]
lib.hi_gemm_force_config.argtypes = [I]
lib.hi_gemm_num_configs.restype = I
lib.hi_gemm_config_desc.argtypes = [I, C.c_char_p, I]
lib.hi_rope_cache.argtypes = [P, I, I, I, I, I, P, P, I, I, P, P, I, P]
lib.hi_flash_prefill.argtypes = [P, I, P, P, I, P, I, I, I, I, I, I, P]
lib.hi_decode_attn.argtypes = [P, I, P, P, I, P, P, I, I, I, I, I, P, P]
lib.hi_argmax.argtypes = [P, I, I, P, P]
lib.hi_load.argtypes = [C.c_char_p, I, I, I]
lib.hi_load.restype = P
lib.hi_load_tp.argtypes = [C.c_char_p, I, I, I, I, IP]
lib.hi_load_tp.restype = P
lib.hi_tp_info.argtypes = [P, C.POINTER(C.c_longlong)]
lib.hi_tp_set_allreduce.argtypes = [P, I, C.c_longlong]
lib.hi_tp_set_overlap.argtypes = [P, I]
lib.hi_tp_allreduce_check.argtypes = [P, C.c_size_t]
lib.hi_tp_allreduce_check.restype = F
lib.hi_tp_allreduce_bench.argtypes = [P, C.c_size_t, I]
lib.hi_tp_allreduce_bench.restype = F
lib.hi_free.argtypes = [P]
lib.hi_config.argtypes = [P, IP]
lib.hi_weight_bytes.argtypes = [P]
lib.hi_weight_bytes.restype = C.c_size_t
lib.hi_prefill.argtypes = [P, I, IP, I, I]
lib.hi_prefill.restype = I
lib.hi_prefill_logits.argtypes = [P, I, IP, I, I, P]
lib.hi_prefill_logits.restype = I
lib.hi_decode.argtypes = [P, I, I, I]
lib.hi_decode.restype = F
lib.hi_history.argtypes = [P, I, I, I, IP]
lib.hi_history.restype = I
lib.hi_logits.argtypes = [P, I, P]
lib.hi_logits.restype = I
lib.hi_set_state.argtypes = [P, I, IP, IP]
lib.hi_set_state.restype = I
lib.hi_sync.argtypes = [P]
# MoE
lib.hi_moe_config.argtypes = [P, IP]
lib.hi_moe_router.argtypes = [P, I, P, I, I, I, I, I, P, P, P]
lib.hi_moe_work_bytes.argtypes = [I, I, I, I, I]
lib.hi_moe_work_bytes.restype = C.c_size_t
lib.hi_moe_experts.argtypes = [P, I, P, P, P, P, I, I, I, I, I, I, F, P, I, I, P, P]
MOE_NORM_TOPK, MOE_FP16_WEIGHTS = 1, 2  # moe_flags bits
if hasattr(lib, "hi_prefill_debug"):  # debug dump (scripts/layer_diff.py); absent from builds that predate it
    lib.hi_debug_bytes.argtypes = [P, I]
    lib.hi_debug_bytes.restype = C.c_size_t
    lib.hi_prefill_debug.argtypes = [P, I, IP, I, I, P]
    lib.hi_prefill_debug.restype = I
if hasattr(lib, "hi_decode_debug"):
    lib.hi_decode_debug.argtypes = [P, I, P]
    lib.hi_decode_debug.restype = I
# Eviction scoring and selection (docs/eviction.md)
LL = C.c_longlong
lib.hi_flash_prefill_lse.argtypes = [P, I, P, P, I, P, I, I, I, I, I, I, P, P]
lib.hi_prefill_scores.argtypes = [P, I, P, I, I, I, I, I, I, P, P, P, I, I, I, P, P]
lib.hi_decode_attn_track.argtypes = [P, I, P, P, I, P, P, P, I, I, I, I, I, P, P, P, P, P, P, I, P, P]
lib.hi_select_k.argtypes = [P, LL, LL, I, I, I, I, I, I, P, I, P]
lib.hi_wfl_work_bytes.argtypes = [I, I, I]
lib.hi_wfl_work_bytes.restype = C.c_size_t
lib.hi_wfl_work_offset.argtypes = [I, I, I, I]
lib.hi_wfl_work_offset.restype = C.c_size_t
lib.hi_wfl_prune.argtypes = [P, P, LL, LL, P, LL, LL, I, P, P, I, I, I, I, I, F, I, I, I, I, I, P, P, P]
lib.hi_wfl_prune.restype = I
lib.hi_set_tracking.argtypes = [P, I, I]
lib.hi_set_tracking.restype = I
lib.hi_cache_len.argtypes = [P, I]
lib.hi_cache_len.restype = I
lib.hi_scores.argtypes = [P, I, I, I, I, P]
lib.hi_scores.restype = I
lib.hi_cache_read.argtypes = [P, I, I, I, P, P]
lib.hi_cache_read.restype = I
lib.hi_evict.argtypes = [P, I, I, I, I, I, F, I, I, F, I, IP, C.POINTER(F)]
lib.hi_evict.restype = I
EVICT_POLICIES = {"h2o": 0, "wfl": 1, "window": 2}


def _s():
    return C.c_void_p(torch.cuda.current_stream().cuda_stream)


def _p(t):
    return C.c_void_p(t.data_ptr()) if t is not None else None


def rmsnorm(x, w, eps=1e-5):
    y = torch.empty_like(x)
    lib.hi_rmsnorm(_p(x), _p(w), _p(y), x.shape[0], x.shape[1], eps, _s())
    return y


def gemv(x, W, out=None, normw=None, eps=1e-5, epi=0):
    M, K = x.shape
    N = W.shape[0]
    if out is None:
        out = torch.empty(M, N // 2 if epi == 2 else N, dtype=x.dtype, device=x.device)
    lib.hi_gemv(_p(x), x.stride(0), _p(normw), eps, _p(W), _p(out), out.stride(0), M, N, K, epi, _s())
    return out


def gemv_qkv_rope(x, W, normw, Hq, Hkv, cs, pos, kc, vc, out, bias=None, eps=1e-5):
    """Fused RMSNorm + QKV GEMV (+ bias) + RoPE + KV-cache write. Row m -> cache slot m at pos[m]."""
    D = kc.shape[-1]
    lib.hi_gemv_qkv_rope(_p(x), x.stride(0), _p(normw), eps, _p(W), _p(bias), _p(out), out.stride(0), x.shape[0],
                         x.shape[1], Hq, Hkv, D, _p(cs), _p(pos), _p(kc), _p(vc), kc.shape[2], _s())
    return out


def gemm(A, W, out=None, epi=0, bias=None):
    M, K = A.shape
    N = W.shape[0]
    if out is None:
        out = torch.empty(M, N // 2 if epi == 2 else N, dtype=A.dtype, device=A.device)
    lib.hi_gemm(_p(A), A.stride(0), _p(W), _p(bias), _p(out), out.stride(0), M, N, K, epi, _s())
    return out


def rope_cache(qkv, Hq, Hkv, cs, kc, vc, pos=None, pos_base=0, slot_base=0):
    """qkv [M, (Hq+2Hkv)*D] (q rotated in place); kc/vc [slots, Hkv, max_seq, D]; cs [max_seq, D/2, 2] fp16."""
    D = kc.shape[-1]
    lib.hi_rope_cache(_p(qkv), qkv.stride(0), qkv.shape[0], Hq, Hkv, D, _p(cs), _p(pos), pos_base, slot_base,
                      _p(kc), _p(vc), kc.shape[2], _s())


def flash_prefill(q, kc, vc, T, p0, Hq, Hkv, out=None):
    """q [T, ldq] with heads at column h*D; kc/vc [Hkv, max_seq, D] for one slot."""
    D = kc.shape[-1]
    if out is None:
        out = torch.empty(T, Hq * D, dtype=torch.float16, device=q.device)
    lib.hi_flash_prefill(_p(q), q.stride(0), _p(kc), _p(vc), kc.shape[1], _p(out), out.stride(0), T, p0, Hq, Hkv, D, _s())
    return out


def decode_attn(q, kc, vc, pos, Hq, Hkv, out=None, work=None):
    """q [B, ldq]; kc/vc [B, Hkv, max_seq, D]; pos [B] int32 (attends to keys 0..pos)."""
    B = q.shape[0]
    D = kc.shape[-1]
    if out is None:
        out = torch.empty(B, Hq * D, dtype=torch.float16, device=q.device)
    if work is None:
        work = torch.zeros(lib.hi_decode_work_bytes(B, Hq, Hkv, D), dtype=torch.uint8, device=q.device)
    lib.hi_decode_attn(_p(q), q.stride(0), _p(kc), _p(vc), kc.shape[2], _p(pos), _p(out), out.stride(0),
                       B, Hq, Hkv, D, _p(work), _s())
    return out


def flash_prefill_lse(q, kc, vc, T, p0, Hq, Hkv, out=None, lse=None):
    """flash_prefill that also returns each row's log2-domain log-sum-exp, lse [Hq, T] fp32."""
    D = kc.shape[-1]
    if out is None:
        out = torch.empty(T, Hq * D, dtype=torch.float16, device=q.device)
    if lse is None:
        lse = torch.empty(Hq, T, dtype=torch.float32, device=q.device)
    lib.hi_flash_prefill_lse(_p(q), q.stride(0), _p(kc), _p(vc), kc.shape[1], _p(out), out.stride(0), T, p0, Hq, Hkv, D,
                             _p(lse), _s())
    return out, lse


def prefill_scores(q, kc, T, p0, Hq, Hkv, lse, score, win=None, win_r0=None, win_off=0, wscore=None):
    """Add each key's attention mass from the T rows to score [Hkv, max_seq] fp32 (second prefill
    pass; lse from flash_prefill_lse). Rows >= win_r0 are window rows: their softmax rows go to
    win [Hq, W, max_seq] fp16 (row win_off + row - win_r0) and their mass is added to wscore."""
    D = kc.shape[-1]
    W = win.shape[1] if win is not None else 0
    if win_r0 is None:
        win_r0 = T
    lib.hi_prefill_scores(_p(q), q.stride(0), _p(kc), kc.shape[1], T, p0, Hq, Hkv, D, _p(lse), _p(score), _p(win), win_r0,
                          win_off, W, _p(wscore), _s())


def decode_attn_track(q, kc, vc, pos, Hq, Hkv, score, shift=None, out=None, work=None, lg=None, lse=None, win=None,
                      win_row=None, wscore=None):
    """decode_attn over pos[b] - shift[b] + 1 cached keys, adding every key's attention weight
    (summed over the query heads of its kv head) to score [B, Hkv, max_seq] fp32. Returns
    (out, lg [B, Hq, max_seq] log2-domain logits, lse [B, Hq] log2-domain). Window: win_row [B]
    int32 (-1 none); win [B, Hq, W, max_seq] fp16; wscore [B, Hkv, max_seq]."""
    B, D, S = q.shape[0], kc.shape[-1], kc.shape[2]
    dev = q.device
    if out is None:
        out = torch.empty(B, Hq * D, dtype=torch.float16, device=dev)
    if work is None:
        work = torch.zeros(lib.hi_decode_work_bytes(B, Hq, Hkv, D), dtype=torch.uint8, device=dev)
    if lg is None:
        lg = torch.empty(B, Hq, S, dtype=torch.float32, device=dev)
    if lse is None:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=dev)
    W = win.shape[2] if win is not None else 0
    lib.hi_decode_attn_track(_p(q), q.stride(0), _p(kc), _p(vc), S, _p(pos), _p(shift), _p(out), out.stride(0), B, Hq, Hkv,
                             D, _p(work), _p(lg), _p(lse), _p(score), _p(win), _p(win_row), W, _p(wscore), _s())
    return out, lg, lse


def select_k(score, lo, hi, k, largest=True, flags=None):
    """score [P, n] fp32 -> flags [P, n] uint8 with 1 at the k slots of [lo, hi) with the largest
    (or smallest) score per row; ties go to the lower slot."""
    Pn, n = score.shape
    if flags is None:
        flags = torch.zeros(Pn, n, dtype=torch.uint8, device=score.device)
    lib.hi_select_k(_p(score), score.stride(0), 0, 1, Pn, lo, hi, k, int(largest), _p(flags), flags.stride(0), _s())
    return flags


WFL_WORK = ("kn", "vn", "w", "best", "sec", "acc", "arg", "arg2", "hit", "rl", "rcount", "kept")


def wfl_prune(k, v, score, status, m, mix=0.5, step=8, rounds=0, power=1, incremental=True, max_rounds=-1, gidx=None,
              work=None):
    """policies.wfl_prune on the GPU. k, v [P, n, D] fp16 (one problem per row; slot i of problem p
    is k[p, gidx[p, i]] when gidx [P, n] int32 is given), score [P, n] fp32 (H2O weights, normalized
    inside), status [P, n] uint8 (0 candidate, 1 held; every problem the same number of
    candidates). Returns (kept candidates [P, n] bool, rounds, work)."""
    Pn, L, D = k.shape
    n = status.shape[1]
    dev = k.device
    ncand = int((status == 0).sum(1)[0])
    if gidx is None:
        gidx = torch.arange(n, dtype=torch.int32, device=dev).expand(Pn, n).contiguous()
    if work is None:
        work = torch.empty(lib.hi_wfl_work_bytes(Pn, n, D), dtype=torch.uint8, device=dev)
    out = torch.zeros(Pn, n, dtype=torch.uint8, device=dev)
    r = lib.hi_wfl_prune(_p(k), _p(v), L, 0, _p(score), score.stride(0), 0, 1, _p(gidx), _p(status), Pn, n, D,
                         ncand, m, mix, step, rounds, power, int(incremental), max_rounds, _p(out), _p(work), _s())
    if r < 0:
        raise ValueError("hi_wfl_prune: bad arguments (see stderr)")
    return out.bool(), r, work


def wfl_work_view(work, P, n, D, name):
    """A named array of the WFL workspace (WFL_WORK) as a tensor."""
    i = WFL_WORK.index(name)
    off = lib.hi_wfl_work_offset(P, n, D, i)
    if name in ("kn", "vn"):
        return work[off:off + P * n * D * 4].view(torch.float32).view(P, n, D)
    if name == "rcount":
        return work[off:off + P * 4].view(torch.int32)
    if name == "kept":
        return work[off:off + P * n].view(P, n)
    dt = torch.float32 if name in ("w", "best", "sec", "acc") else torch.int32
    return work[off:off + P * n * 4].view(dt).view(P, n)


def moe_router(xn, Wr, k, flags):
    """Top-k experts of each normalised row xn [M, H] under router Wr [E, H]: (idx [M, k] int32,
    weights [M, k] fp32), descending."""
    M, H = xn.shape
    idx = torch.empty(M, k, dtype=torch.int32, device=xn.device)
    w = torch.empty(M, k, dtype=torch.float32, device=xn.device)
    lib.hi_moe_router(_p(xn), xn.stride(0), _p(Wr), M, Wr.shape[0], H, k, flags, _p(idx), _p(w), _s())
    return idx, w


def moe_experts(xn, wgu, wdown, idx, w, x, flags, scale=1.0, path=0):
    """x += MoE(xn) in place, given the routing: wgu [E, 2I, H] (gate/up interleaved per expert),
    wdown [E, H, I]. path 0 auto, 1 GEMV, 2 grouped GEMM."""
    M, H = xn.shape
    E, I, k = wgu.shape[0], wdown.shape[2], idx.shape[1]
    work = torch.empty(lib.hi_moe_work_bytes(M, k, E, H, I), dtype=torch.uint8, device=xn.device)
    lib.hi_moe_experts(_p(xn), xn.stride(0), _p(wgu), _p(wdown), _p(idx), _p(w), M, E, k, H, I, flags, scale, _p(x),
                       x.stride(0), path, _p(work), _s())
    return x


ALLREDUCE = {"auto": 0, "nccl": 1, "oneshot": 2}


def read_config(path):
    """Model config from a weight file's header (scripts/export.py), without loading the weights."""
    import struct
    with open(path, "rb") as f:
        raw = f.read(128)
    h = struct.unpack("<10i2f2i", raw[:56])
    if h[0] != 0x48495046 or h[1] not in (2, 3):
        raise ValueError(f"{path}: not a version-2/3 hipinfer weight file")
    keys = ["vocab", "hidden", "inter", "layers", "heads", "kv_heads", "head_dim", "max_pos"]
    cfg = dict(zip(keys, h[2:10]), eps=h[10], tied=bool(h[12] & 1), qkv_bias=bool(h[12] & 2), window=h[13],
               moe=bool(h[12] & 4))
    if h[1] == 3:  # MoE (inter is per expert)
        cfg["n_experts"], cfg["top_k"], moe_flags, cfg["moe_scale"] = struct.unpack("<3if", raw[64:80])
    return cfg


def max_context(cfg):
    """Longest sequence the engine can serve for this model: max_pos, or the sliding window
    (which the engine does not implement, so sequences must fit inside it)."""
    return min(cfg["max_pos"], cfg["window"]) if cfg.get("window") else cfg["max_pos"]


class Engine:
    """tp > 1 shards the model over `devices` (default: the first tp visible GPUs) with tensor
    parallelism; the API is the same, and logits come back on devices[0]. With tp=1 the engine
    lives on devices[0] (default: the current device). Bad arguments (slot, batch size, token,
    or a position past max_seq) raise ValueError; the reason is printed on stderr."""

    def __init__(self, path, max_batch=8, max_seq=2048, max_prefill=1024, tp=1, devices=None):
        if devices is None:
            devices = list(range(tp)) if tp > 1 else [torch.cuda.current_device()]
        devices = list(devices)
        assert len(devices) == tp
        self.tp = tp
        self.device = torch.device("cuda", devices[0])
        self.h = lib.hi_load_tp(path.encode(), max_batch, max_seq, max_prefill, tp, (C.c_int * tp)(*devices))
        if not self.h:
            raise RuntimeError("hi_load failed")
        cfg = (C.c_int * 11)()
        lib.hi_config(self.h, cfg)
        (self.vocab, self.hidden, self.inter, self.layers, self.heads, self.kv_heads, self.head_dim, self.max_pos,
         tied, qkv_bias, self.window) = list(cfg)
        self.tied, self.qkv_bias = bool(tied), bool(qkv_bias)
        moe = (C.c_int * 3)()
        lib.hi_moe_config(self.h, moe)
        self.n_experts, self.top_k, self.moe_flags = list(moe)  # zeros for a dense model
        self.max_seq, self.max_batch, self.max_prefill = max_seq, max_batch, max_prefill
        self.weight_bytes = lib.hi_weight_bytes(self.h)
        self.kv_bytes_per_token = 2 * self.layers * self.kv_heads * self.head_dim * 2  # whole model, all GPUs

    def tp_info(self):
        v = (C.c_longlong * 6)()
        lib.hi_tp_info(self.h, v)
        return {"world": v[0], "p2p": bool(v[1]), "nccl": bool(v[2]),
                "allreduce": {k: n for n, k in ALLREDUCE.items()}[v[3]], "oneshot_max_bytes": v[4],
                "prefill_overlap": bool(v[5])}

    def set_overlap(self, on):
        """Overlap prefill all-reduces with compute (two row halves, two streams)."""
        lib.hi_tp_set_overlap(self.h, int(on))

    def set_allreduce(self, mode, oneshot_max=-1):
        """'auto' (one-shot up to oneshot_max bytes, NCCL above), 'nccl' or 'oneshot'."""
        lib.hi_tp_set_allreduce(self.h, ALLREDUCE[mode], oneshot_max)

    def allreduce_check(self, n):
        return lib.hi_tp_allreduce_check(self.h, n)

    def allreduce_bench(self, n, iters=200):
        """Microseconds per all-reduce of n fp16 values."""
        return lib.hi_tp_allreduce_bench(self.h, n, iters)

    @staticmethod
    def _ok(r, what):
        if r < 0:
            raise ValueError(f"hipinfer {what}: bad arguments (see stderr)")
        return r

    def prefill(self, slot, tokens, p0=0):
        arr = (C.c_int * len(tokens))(*tokens)
        return self._ok(lib.hi_prefill(self.h, slot, arr, len(tokens), p0), "prefill")

    def prefill_logits(self, slot, tokens, p0=0):
        """Prefill and return the fp16 logits of every position, [T, vocab] on the GPU."""
        arr = (C.c_int * len(tokens))(*tokens)
        out = torch.empty(len(tokens), self.vocab, dtype=torch.float16, device=self.device)
        self._ok(lib.hi_prefill_logits(self.h, slot, arr, len(tokens), p0, C.c_void_p(out.data_ptr())), "prefill_logits")
        return out

    def prefill_debug(self, slot, tokens, p0=0):
        """Prefill (one GPU, len(tokens) <= max_prefill) and return every layer's intermediates as
        one flat fp16 tensor on the GPU; scripts/layer_diff.py split_engine_dump names the planes.
        Not yet verified on a GPU (written before hi_prefill_debug was first compiled)."""
        if not hasattr(lib, "hi_prefill_debug"):
            raise RuntimeError("this libhipinfer.so has no hi_prefill_debug; rebuild it (make)")
        arr = (C.c_int * len(tokens))(*tokens)
        out = torch.empty(lib.hi_debug_bytes(self.h, len(tokens)) // 2, dtype=torch.float16, device=self.device)
        self._ok(lib.hi_prefill_debug(self.h, slot, arr, len(tokens), p0, C.c_void_p(out.data_ptr())), "prefill_debug")
        return out

    def decode_debug(self, B):
        """One decode step on slots [0, B) without a graph; returns its dump in the prefill_debug
        layout with B rows (only MoE layers fill theirs: mlp_norm, the router's input, and down)."""
        if not hasattr(lib, "hi_decode_debug"):
            raise RuntimeError("this libhipinfer.so has no hi_decode_debug; rebuild it (make)")
        out = torch.empty(lib.hi_debug_bytes(self.h, B) // 2, dtype=torch.float16, device=self.device)
        self._ok(lib.hi_decode_debug(self.h, B, C.c_void_p(out.data_ptr())), "decode_debug")
        return out

    def decode(self, B, steps, graph=True):
        return self._ok(lib.hi_decode(self.h, B, steps, int(graph)), "decode")

    def history(self, slot, start, n):
        arr = (C.c_int * n)()
        self._ok(lib.hi_history(self.h, slot, start, n, arr), "history")
        return list(arr)

    def logits(self, B):
        out = torch.empty(B, self.vocab, dtype=torch.float16, device=self.device)
        self._ok(lib.hi_logits(self.h, B, C.c_void_p(out.data_ptr())), "logits")
        return out

    def set_state(self, tokens, positions):
        """Next input token and its position for slots [0, len(tokens))."""
        B = len(tokens)
        assert len(positions) == B
        self._ok(lib.hi_set_state(self.h, B, (C.c_int * B)(*tokens), (C.c_int * B)(*positions)), "set_state")

    # ---- eviction (docs/eviction.md); one GPU only
    def set_tracking(self, on=True, window=0):
        """Eviction scoring on / off (off by default). window > 0 also keeps each key's mass from
        the last `window` prompt tokens. Scores start at the next fresh prefill (p0 = 0)."""
        self._ok(lib.hi_set_tracking(self.h, int(on), window), "set_tracking")

    def cache_len(self, slot):
        return self._ok(lib.hi_cache_len(self.h, slot), "cache_len")

    def scores(self, layer, slot, n=None, window=False):
        """[kv_heads, n] fp32 (on the GPU): H2O scores (or window scores) of cache indices [0, n)."""
        n = self.cache_len(slot) if n is None else n
        out = torch.empty(self.kv_heads, n, dtype=torch.float32, device=self.device)
        self._ok(lib.hi_scores(self.h, int(window), layer, slot, n, C.c_void_p(out.data_ptr())), "scores")
        return out

    def cache_read(self, layer, slot, n=None):
        """(k, v), each [kv_heads, n, head_dim] fp16 on the GPU, of cache indices [0, n)."""
        n = self.cache_len(slot) if n is None else n
        k = torch.empty(self.kv_heads, n, self.head_dim, dtype=torch.float16, device=self.device)
        v = torch.empty_like(k)
        self._ok(lib.hi_cache_read(self.h, layer, slot, n, C.c_void_p(k.data_ptr()), C.c_void_p(v.data_ptr())), "cache_read")
        return k, v

    def evict(self, slot, capacity, policy="h2o", sink=4, recent=0, mix=0.5, step=8, rounds=0, pool=0.0, power=1):
        """Evict `slot` down to `capacity` cached slots per (layer, kv head); see hi_evict. Returns
        (evicted per head, kept [layers, kv_heads, capacity] old cache indices, {select_ms, compact_ms})."""
        kept = (C.c_int * (self.layers * self.kv_heads * capacity))()
        ms = (C.c_float * 2)()
        r = self._ok(lib.hi_evict(self.h, slot, capacity, EVICT_POLICIES[policy], sink, recent, mix, step, rounds, pool,
                                  power, kept, ms), "evict")
        kt = torch.tensor(list(kept), dtype=torch.int64).view(self.layers, self.kv_heads, capacity) if r > 0 else None
        return r, kt, {"select_ms": ms[0], "compact_ms": ms[1]}

    def generate(self, prompt, n_new, graph=True):
        """Greedy: returns n_new tokens (first comes from prefill)."""
        first = self.prefill(0, prompt)
        if n_new > 1:
            self.decode(1, n_new - 1, graph)
        return [first] + self.history(0, len(prompt) + 1, n_new - 1)

    def close(self):
        if self.h:
            lib.hi_free(self.h)
            self.h = None
