"""Re-identification by ranking: for each query, where does its true match land among candidates?"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from fingerprints.data import Embeddings

# Which candidates a query is ranked against. "dx+site" on Test 2 is the headline number:
# a win there cannot come from matching the hospital (stain, scanner) or the disease.
CONTROLS = {
    "all": [],
    "site": ["site"],
    "dx": ["diagnosis"],
    "dx+site": ["diagnosis", "site"],
}


def allowed_mask(meta: pd.DataFrame, control: str):
    cols = CONTROLS[control]
    if not cols:
        return None
    mask = np.ones((len(meta), len(meta)), bool)
    for c in cols:
        v = meta[c].astype(str).to_numpy()
        mask &= v[:, None] == v[None, :]
    return mask


def ranks_of_truth(Q: np.ndarray, C: np.ndarray, allowed=None):
    """Q[i] and C[i] belong to the same patient. Rank 1 = the true match is the most similar."""
    S = Q @ C.T
    true = np.diag(S)
    better = S > true[:, None]
    if allowed is None:
        n_cand = np.full(len(Q), len(C))
    else:
        allowed = allowed | np.eye(len(Q), dtype=bool)
        better &= allowed
        n_cand = allowed.sum(1)
    return better.sum(1) + 1, n_cand


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (center - half, center + half)


def summarize(ranks, n_cand, min_candidates: int = 2, n_sims: int = 2000, seed: int = 0) -> dict:
    """Top-1/top-10 vs chance. Queries with fewer than min_candidates candidates are dropped
    (a lone candidate is a free win). p_value: chance of doing this well by guessing."""
    keep = n_cand >= min_candidates
    r, n = ranks[keep], n_cand[keep]
    if len(r) == 0:
        return dict(n_queries=0)
    hits = int((r == 1).sum())
    p_chance = 1.0 / n
    sims = (np.random.default_rng(seed).random((n_sims, len(r))) < p_chance).sum(1)
    out = dict(
        n_queries=int(len(r)),
        mean_candidates=float(n.mean()),
        top1=hits / len(r),
        top10=float((r <= 10).mean()),
        chance_top1=float(p_chance.mean()),
        chance_top10=float((np.minimum(10, n) / n).mean()),
        median_rank=float(np.median(r)),
        mrr=float((1.0 / r).mean()),
        top1_ci95=wilson(hits, len(r)),
        p_value=float((1 + (sims >= hits).sum()) / (n_sims + 1)),
    )
    out["lift_top1"] = out["top1"] / out["chance_top1"]
    return out


def test1(e: Embeddings, control: str = "all", **kw) -> dict:
    """Slide A vs slide B (different specimen) of the same patient."""
    m = e.has_B
    meta = e.patients[m].reset_index(drop=True)
    return summarize(*ranks_of_truth(e.A[m], e.B[m], allowed_mask(meta, control)), **kw)


def test2(e: Embeddings, control: str = "all", **kw) -> dict:
    """Report vs slide A of the same patient."""
    m = e.has_R
    meta = e.patients[m].reset_index(drop=True)
    return summarize(*ranks_of_truth(e.R[m], e.A[m], allowed_mask(meta, control)), **kw)


def fmt(s: dict) -> str:
    if not s.get("n_queries"):
        return "n/a (no queries with >=2 candidates)"
    lo, hi = s["top1_ci95"]
    return (f"top-1 {s['top1']:.3f} [{lo:.3f}-{hi:.3f}] vs chance {s['chance_top1']:.3f} "
            f"(x{s['lift_top1']:.1f}, p={s['p_value']:.4f}) | top-10 {s['top10']:.3f} vs "
            f"{s['chance_top10']:.3f} | n={s['n_queries']}, ~{s['mean_candidates']:.0f} candidates")
