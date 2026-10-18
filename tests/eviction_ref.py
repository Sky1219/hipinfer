"""PyTorch references for the eviction selection kernels (docs/eviction.md); any device.

The WFL rules are SubCache's policies.slot_similarity / wfl_prune, with ties made deterministic
the way the kernels break them: the lowest index wins, in the argmax and in the removal order."""
import math

import torch
import torch.nn.functional as F


def ref_select(score, lo, hi, k, largest):
    """flags [P, n] uint8: the k slots of [lo, hi) with the largest / smallest score, ties to the
    lower index."""
    Pn, n = score.shape
    sub = score[:, lo:hi]
    order = torch.sort(-sub if largest else sub, dim=1, stable=True).indices[:, :k]
    flags = torch.zeros(Pn, n, dtype=torch.uint8, device=score.device)
    flags.scatter_(1, order + lo, 1)
    return flags


def ref_unit(x):
    """As slot_similarity: centre per problem (over the slots), then L2-normalize; fp32."""
    x = x.float()
    return F.normalize(x - x.mean(dim=1, keepdim=True), dim=-1)


def ref_sim(k, v, power=1):
    """[P, n, n]: (max(cos_k, 0) * max(cos_v, 0)) ** power, 1 on the diagonal."""
    kn, vn = ref_unit(k), ref_unit(v)
    sim = (torch.matmul(kn, kn.transpose(1, 2)).clamp_min(0.0) * torch.matmul(vn, vn.transpose(1, 2)).clamp_min(0.0)) ** power
    eye = torch.eye(sim.shape[1], device=sim.device, dtype=torch.bool)
    return sim.masked_fill(eye, 1.0)


def first_max(x):
    """Max along -1 and the lowest index holding it."""
    mx = x.amax(-1)
    idx = torch.arange(x.shape[-1], device=x.device).expand_as(x)
    return mx, torch.where(x == mx[..., None], idx, x.shape[-1]).amin(-1)


def ref_top2(sim, kept):
    """Best kept neighbour (value, lowest index) and the second-best value of every row; columns
    that are not kept count as -1 (as wfl_prune's masked_fill)."""
    masked = sim.masked_fill(~kept[:, None, :], -1.0)
    best, arg = first_max(masked)
    second = masked.scatter(-1, arg[..., None], -1.0).amax(-1)
    return best, arg, second


def ref_wfl_prune(sim, w, cand, held, m, mix=0.5, step=8, rounds=None):
    """policies.wfl_prune: start from every candidate kept and remove the `step` cheapest per
    round until m remain. Returns the kept candidates [P, n] bool."""
    kept = cand | held
    n_remove = int(cand.sum(1).min()) - m
    if rounds:
        step = max(step, math.ceil(n_remove / rounds))
    removed = 0
    while removed < n_remove:
        best, arg, second = ref_top2(sim, kept)
        loss = torch.zeros_like(w).scatter_add_(1, arg, w * (best - second.clamp_min(0.0)).clamp_min(0.0))
        loss = (mix * loss + (1.0 - mix) * w).masked_fill(~(kept & cand), float("inf"))
        k = min(step, n_remove - removed)
        kept.scatter_(1, torch.sort(loss, dim=1, stable=True).indices[:, :k], False)
        removed += k
    return kept & cand
