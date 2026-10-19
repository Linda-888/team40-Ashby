"""Fixes for three limitations, all on the embeddings already on disk (no new slides, no GPU).

Part 1  stats    Test 1 with stronger statistics, for every model and control (all, dx, dx+site):
                 top-1, mean reciprocal rank (MRR), and per-query AUROC (how often the true owner
                 out-scores a random other candidate; 0.5 = chance, no ceiling from small pools).
                 Exact permutation p-values (owners shuffled within the control's groups), then Holm
                 correction across all models x controls. Also the reverse direction (frozen -> FFPE).
Part 2  attack   A trained attacker. Patients are split 4 ways (x10 repeats); the attacker learns an
                 FFPE -> frozen map from the training patients' pairs and attacks the held-out patients
                 (candidates = held-out frozen slides only, so chance = 1/pool). Methods: zero-shot,
                 domain centering, CORAL (whiten + recolor), orthogonal Procrustes, ridge regression.
Part 3  pooling  How the slide is summarized from its 200 tile embeddings: mean (as in the paper),
                 mean of fewer tiles (25/50/100), centered mean, tile matching (Chamfer: each tile's
                 best match on the other slide), and tile voting (each tile votes for the slide holding
                 its nearest tile).

    python 15_robustness.py --out ~/robustness                      # all parts, all models
    python 15_robustness.py --parts stats attack --models conch musk
Writes <out>/robustness.json and <out>/summary.md. Uses the clean patient list (data/patients_clean.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import data, paths  # noqa: E402
from fingerprints.data import normalize  # noqa: E402
from fingerprints.matching import allowed_mask  # noqa: E402

CONTROLS = ("all", "dx", "dx+site")
GROUP_COLS = {"all": [], "dx": ["diagnosis"], "dx+site": ["diagnosis", "site"]}


# ------------------------------------------------------------------------------ shared scoring

def ranks_from_S(S, allowed=None, truth=None):
    """S[i, j] = similarity of query i to candidate j; candidate truth[i] is the owner (default i).
    Returns rank of the owner (1 = best) and the number of candidates per query."""
    n = len(S)
    truth = np.arange(n) if truth is None else truth
    true = S[np.arange(n), truth]
    better = S > true[:, None]
    if allowed is None:
        return better.sum(1) + 1, np.full(n, S.shape[1])
    better &= allowed
    return better.sum(1) + 1, allowed.sum(1)


def metrics(ranks, n_cand):
    keep = n_cand >= 2
    r, c = ranks[keep], n_cand[keep]
    auroc = (c - r) / (c - 1)  # share of other candidates the owner beats
    return dict(n=int(keep.sum()), mean_candidates=float(c.mean()), top1=float((r == 1).mean()),
                chance_top1=float((1 / c).mean()), top10=float((r <= 10).mean()),
                chance_top10=float((np.minimum(10, c) / c).mean()), mrr=float((1 / r).mean()),
                chance_mrr=float(np.mean([np.mean(1 / np.arange(1, k + 1)) for k in c])),
                auroc=float(auroc.mean()))


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


# ------------------------------------------------------------------------------ Part 1

def group_ids(meta, control):
    cols = GROUP_COLS[control]
    if not cols:
        return np.zeros(len(meta), int)
    key = meta[cols].astype(str).agg("|".join, axis=1)
    return key.astype("category").cat.codes.to_numpy()


def permutation_test(S, allowed, groups, observed, n_perm, rng):
    """Shuffle who owns which candidate within each group; p = share of shuffles that do as well."""
    hits = np.zeros(n_perm)
    mrr = np.zeros(n_perm)
    auc = np.zeros(n_perm)
    idx_by_g = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for b in range(n_perm):
        truth = np.arange(len(S))
        for idx in idx_by_g:
            truth[idx] = rng.permutation(idx)
        m = metrics(*ranks_from_S(S, allowed, truth))
        hits[b], mrr[b], auc[b] = m["top1"], m["mrr"], m["auroc"]
    p = lambda null, obs: float((1 + (null >= obs - 1e-12).sum()) / (n_perm + 1))  # noqa: E731
    return dict(p_top1=p(hits, observed["top1"]), p_mrr=p(mrr, observed["mrr"]), p_auroc=p(auc, observed["auroc"]))


def part_stats(models, n_perm, seed):
    out, p_top1, p_mrr = {}, {}, {}
    for m in models:
        e = data.load(m)
        e = e.subset(e.has_B)
        A, B, meta = normalize(e.A), normalize(e.B), e.patients
        out[m] = {}
        for direction, (Q, C) in (("FFPE->frozen", (A, B)), ("frozen->FFPE", (B, A))):
            S = Q @ C.T
            for control in CONTROLS if direction == "FFPE->frozen" else ("all",):
                allowed = allowed_mask(meta, control)
                if allowed is not None:
                    allowed = allowed | np.eye(len(S), dtype=bool)
                obs = metrics(*ranks_from_S(S, allowed))
                obs.update(permutation_test(S, allowed, group_ids(meta, control), obs, n_perm,
                                            np.random.default_rng(seed)))
                key = f"{direction}/{control}"
                out[m][key] = obs
                if direction == "FFPE->frozen":
                    p_top1[f"{m}/{control}"] = obs["p_top1"]
                    p_mrr[f"{m}/{control}"] = obs["p_mrr"]
                print(f"[stats] {m:8s} {key:22s} top-1 {obs['top1']:.3f} (x{obs['top1'] / obs['chance_top1']:.1f}) "
                      f"MRR {obs['mrr']:.3f} vs {obs['chance_mrr']:.3f}  AUROC {obs['auroc']:.3f}  "
                      f"p_top1 {obs['p_top1']:.4f}  p_mrr {obs['p_mrr']:.4f}", flush=True)
    h1, h2 = holm(p_top1), holm(p_mrr)
    for k in p_top1:
        m, control = k.split("/", 1)
        out[m][f"FFPE->frozen/{control}"]["holm_top1"] = h1[k]
        out[m][f"FFPE->frozen/{control}"]["holm_mrr"] = h2[k]
    return out


# ------------------------------------------------------------------------------ Part 2

def inv_sqrt_and_sqrt(X, shrink):
    C = np.cov(X, rowvar=False)
    d = C.shape[0]
    C = (1 - shrink) * C + shrink * np.trace(C) / d * np.eye(d)
    w, v = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    return (v / np.sqrt(w)) @ v.T, (v * np.sqrt(w)) @ v.T


def transform(method, Atr, Btr, Ate, Bte, shrink, alpha):
    if method == "zero-shot":
        return Ate, Bte
    ma, mb = Atr.mean(0), Btr.mean(0)
    Ac, Bc, Qc, Cc = Atr - ma, Btr - mb, Ate - ma, Bte - mb
    if method == "centering":
        return Qc, Cc
    if method == "CORAL":
        wa, _ = inv_sqrt_and_sqrt(Ac, shrink)
        _, cb = inv_sqrt_and_sqrt(Bc, shrink)
        return Qc @ wa @ cb, Cc
    if method == "Procrustes":
        W, _ = orthogonal_procrustes(Ac, Bc)
        return Qc @ W, Cc
    if method == "ridge":
        d = Ac.shape[1]
        W = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(d), Ac.T @ Bc)
        return Qc @ W, Cc
    raise ValueError(method)


METHODS = ("zero-shot", "centering", "CORAL", "Procrustes", "ridge")


def part_attack(models, folds, repeats, seed, shrink, alpha):
    out = {}
    for m in models:
        e = data.load(m)
        e = e.subset(e.has_B)
        A, B = normalize(e.A), normalize(e.B)
        n = len(A)
        rows = {k: [] for k in METHODS}
        for r in range(repeats):
            perm = np.random.default_rng(seed + r).permutation(n)
            for te in np.array_split(perm, folds):
                tr = np.setdiff1d(perm, te)
                for k in METHODS:
                    Q, C = transform(k, A[tr], B[tr], A[te], B[te], shrink, alpha)
                    rows[k].append(metrics(*ranks_from_S(normalize(Q) @ normalize(C).T)))
        out[m] = {}
        for k, rs in rows.items():
            s = {f: float(np.mean([x[f] for x in rs])) for f in rs[0]}
            s["lift_top1"] = s["top1"] / s["chance_top1"]
            out[m][k] = s
            print(f"[attack] {m:8s} {k:11s} top-1 {s['top1']:.3f} vs {s['chance_top1']:.3f} (x{s['lift_top1']:.1f})  "
                  f"MRR {s['mrr']:.3f}  AUROC {s['auroc']:.3f}  pool ~{s['mean_candidates']:.0f}", flush=True)
    return out


# ------------------------------------------------------------------------------ Part 3

def load_tiles(model, sid):
    f = paths.emb_dir(model) / "tiles" / f"{sid}.npy"
    return normalize(np.load(f).astype(np.float32)) if f.exists() else None


def tile_scores(TA, TB):
    """Chamfer and vote similarity matrices (n x n) between slide-A tiles and slide-B tiles."""
    counts = np.array([len(t) for t in TB])
    offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
    allB = np.concatenate(TB)
    owner = np.repeat(np.arange(len(TB)), counts)
    n = len(TA)
    chamfer, vote = np.zeros((n, n), np.float32), np.zeros((n, n), np.float32)
    for i, ta in enumerate(TA):
        M = ta @ allB.T  # (tiles_i, all B tiles)
        a2b = np.maximum.reduceat(M, offsets, axis=1).mean(0)  # each A tile's best match per slide
        b2a = np.add.reduceat(M.max(0), offsets) / counts      # each B tile's best match in slide i
        chamfer[i] = 0.5 * (a2b + b2a)
        vote[i] = np.bincount(owner[M.argmax(1)], minlength=n)
    return chamfer, vote


def part_pooling(models, seed):
    out = {}
    for m in models:
        e = data.load(m)
        e = e.subset(e.has_B)
        meta = e.patients
        TA = [load_tiles(m, s) for s in meta.slide_a]
        TB = [load_tiles(m, s) for s in meta.slide_b]
        ok = np.array([a is not None and b is not None and len(a) > 0 and len(b) > 0 for a, b in zip(TA, TB)], bool)
        if ok.sum() < 10:
            print(f"[pooling] {m}: tile embeddings missing, skipped")
            continue
        TA = [t for t, k in zip(TA, ok) if k]
        TB = [t for t, k in zip(TB, ok) if k]
        meta = meta[ok].reset_index(drop=True)
        allowed = allowed_mask(meta, "dx+site") | np.eye(len(TA), dtype=bool)
        rng = np.random.default_rng(seed)
        mats = {}
        mean = lambda T: normalize(np.stack([t.mean(0) for t in T]))  # noqa: E731
        mats["mean of 200 tiles (paper)"] = [mean(TA) @ mean(TB).T]
        for k in (25, 50, 100):
            reps = []
            for _ in range(5):
                sub = lambda T: [t[rng.choice(len(t), min(k, len(t)), replace=False)] for t in T]  # noqa: E731
                reps.append(mean(sub(TA)) @ mean(sub(TB)).T)
            mats[f"mean of {k} tiles"] = reps
        mu = np.concatenate(TA + TB).mean(0)
        cen = lambda T: normalize(np.stack([(t - mu).mean(0) for t in T]))  # noqa: E731
        mats["centered mean"] = [cen(TA) @ cen(TB).T]
        t0 = time.time()
        chamfer, vote = tile_scores(TA, TB)
        mats["tile matching (Chamfer)"] = [chamfer]
        mats["tile voting"] = [vote + 1e-3 * chamfer]  # Chamfer breaks ties between equal vote counts
        out[m] = {"n": int(len(TA)), "seconds_tile_matching": round(time.time() - t0, 1)}
        for name, Ss in mats.items():
            res = {}
            for control, al in (("all", None), ("dx+site", allowed)):
                ms = [metrics(*ranks_from_S(S, al)) for S in Ss]
                s = {f: float(np.mean([x[f] for x in ms])) for f in ms[0]}
                s["lift_top1"] = s["top1"] / s["chance_top1"]
                res[control] = s
            out[m][name] = res
            a = res["all"]
            print(f"[pooling] {m:8s} {name:26s} top-1 {a['top1']:.3f} (x{a['lift_top1']:.1f})  top-10 {a['top10']:.3f}  "
                  f"MRR {a['mrr']:.3f}  AUROC {a['auroc']:.3f} | same hospital x{res['dx+site']['lift_top1']:.2f}", flush=True)
    return out


# ------------------------------------------------------------------------------ report

def summary_md(res):
    L = ["# Robustness checks (clean patients)", ""]
    if "stats" in res:
        L += ["## Part 1: stronger statistics, Test 1 (FFPE -> frozen)", "",
              "AUROC = share of other candidates the true owner out-scores (0.5 = chance). p from "
              "permutations within the control's groups; Holm-corrected across all models x controls.", "",
              "| Model | Candidates | Top-1 (x chance) | MRR (chance) | AUROC | p top-1 (Holm) | p MRR (Holm) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for m, d in res["stats"].items():
            for c in CONTROLS:
                s = d[f"FFPE->frozen/{c}"]
                L.append(f"| {m} | {c} (~{s['mean_candidates']:.0f}) | {s['top1']:.1%} (x{s['top1'] / s['chance_top1']:.1f}) | "
                         f"{s['mrr']:.3f} ({s['chance_mrr']:.3f}) | {s['auroc']:.3f} | {s['holm_top1']:.4f} | {s['holm_mrr']:.4f} |")
            s = d["frozen->FFPE/all"]
            L.append(f"| {m} | reverse: frozen -> FFPE | {s['top1']:.1%} (x{s['top1'] / s['chance_top1']:.1f}) | "
                     f"{s['mrr']:.3f} ({s['chance_mrr']:.3f}) | {s['auroc']:.3f} | {s['p_top1']:.4f} (raw) | {s['p_mrr']:.4f} (raw) |")
        L.append("")
    if "attack" in res:
        L += ["## Part 2: trained attacker (held-out patients, 4 folds x 10 repeats)", "",
              "| Model | Method | Top-1 | Chance | x chance | MRR | AUROC |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for m, d in res["attack"].items():
            for k, s in d.items():
                L.append(f"| {m} | {k} | {s['top1']:.1%} | {s['chance_top1']:.1%} | {s['lift_top1']:.1f} | {s['mrr']:.3f} | {s['auroc']:.3f} |")
        L.append("")
    if "pooling" in res:
        L += ["## Part 3: how the slide is summarized from its tiles", "",
              "| Model | Pooling | Top-1 (x chance) | Top-10 | MRR | AUROC | Same hospital (x chance) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for m, d in res["pooling"].items():
            for k, s in d.items():
                if not isinstance(s, dict):
                    continue
                a = s["all"]
                L.append(f"| {m} | {k} | {a['top1']:.1%} (x{a['lift_top1']:.1f}) | {a['top10']:.1%} | {a['mrr']:.3f} | "
                         f"{a['auroc']:.3f} | x{s['dx+site']['lift_top1']:.2f} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=list(paths.MODELS))
    ap.add_argument("--parts", nargs="+", default=["stats", "attack", "pooling"], choices=["stats", "attack", "pooling"])
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--shrink", type=float, default=0.5, help="covariance shrinkage for CORAL")
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge penalty")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="~/robustness")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    models = [m for m in args.models if (paths.emb_dir(m) / "slides.npz").exists()]
    print(f"models: {models}; patients: {paths.patients_csv()}", flush=True)
    res = {}
    if "stats" in args.parts:
        res["stats"] = part_stats(models, args.perms, args.seed)
    if "attack" in args.parts:
        res["attack"] = part_attack(models, args.folds, args.repeats, args.seed, args.shrink, args.alpha)
    if "pooling" in args.parts:
        res["pooling"] = part_pooling(models, args.seed)
    res["settings"] = vars(args)
    (out / "robustness.json").write_text(json.dumps(res, indent=2))
    (out / "summary.md").write_text(summary_md(res))
    print(f"-> {out / 'robustness.json'}, {out / 'summary.md'}")


if __name__ == "__main__":
    main()
