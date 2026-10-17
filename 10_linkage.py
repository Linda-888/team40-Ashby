"""Test 2 as a learned-alignment attack: learn the report -> slide map on one set of patients (the
attacker's public paired data, e.g. TCGA), then re-identify a disjoint set of patients.

Plain cosine in the model's image-text space need not link a report to its own slide. An attacker
who holds paired (report, slide) embeddings from the same public model can fit a linear map and try
again on new patients. We sweep how many training pairs the attacker has.

The test patients are fixed within each split, so chance does not change with training size.
Controls:
  - candidates restricted to the same diagnosis (dx) and the same diagnosis + hospital (dx+site);
  - "<method>, shuffled pairs": the same map trained on pairs shuffled within diagnosis. It keeps
    disease-level alignment but breaks patient identity, so beating it means matching the person;
  - --split site: train and test patients come from different hospitals (no shared stain/scanner);
    --split project: train on one cancer type, test on the other.
Test 1 (slide A -> slide B) runs the same way for comparison.

    python 10_linkage.py --model pathgen
    python 10_linkage.py --model pathgen --split site
Writes results/<model>/linkage[_<split>].json and .png.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import orthogonal_procrustes  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.data import normalize  # noqa: E402
from fingerprints.matching import allowed_mask, ranks_of_truth, summarize  # noqa: E402

CONTROLS = ("all", "dx", "dx+site")
SHUFFLED = ", shuffled pairs"  # negative control: same method, pairs shuffled within diagnosis
LEARNED = ("procrustes", "ridge")
METHODS = ("cosine", *LEARNED, *(m + SHUFFLED for m in LEARNED))
ALPHAS = np.logspace(-3, 3, 13)


def make_split(meta, mode, rng, test_frac, seed):
    """(train pool, test) patient indices."""
    n = len(meta)
    if mode == "random":  # stratified by diagnosis
        dx = meta.diagnosis.astype(str).to_numpy()
        test = []
        for g in np.unique(dx):
            idx = np.flatnonzero(dx == g)
            rng.shuffle(idx)
            test += list(idx[:int(round(test_frac * len(idx)))])
        test = np.array(sorted(test))
    elif mode == "site":  # whole hospitals go to test until it holds test_frac of the patients
        site = meta.site.astype(str).to_numpy()
        sites = np.unique(site)
        rng.shuffle(sites)
        chosen, count = [], 0
        for s in sites:
            if count >= test_frac * n:
                break
            chosen.append(s)
            count += (site == s).sum()
        test = np.flatnonzero(np.isin(site, chosen))
    else:  # project: alternate which cancer type is held out
        project = meta.project.astype(str).to_numpy()
        names = np.unique(project)
        if len(names) < 2:
            raise SystemExit("--split project needs patients from two projects")
        test = np.flatnonzero(project == names[seed % len(names)])
    return np.setdiff1d(np.arange(n), test), test


def shuffle_within(idx, groups, rng):
    """Permute idx within each diagnosis: disease-level pairing survives, patient identity does not."""
    out = idx.copy()
    for g in np.unique(groups[idx]):
        m = np.flatnonzero(groups[idx] == g)
        out[m] = rng.permutation(idx[m])
    return out


def link(Q, C, train, targets, test, method):
    """Queries of the test patients mapped into the candidates' space using train pairs (Q[train] -> C[targets])."""
    if method == "cosine" or len(train) < 2:
        return Q[test]
    if method == "procrustes":
        W, _ = orthogonal_procrustes(Q[train], C[targets])
        return normalize(Q[test] @ W)
    ridge = RidgeCV(alphas=ALPHAS).fit(Q[train], C[targets])
    return normalize(ridge.predict(Q[test]))


def run_test(Q, C, meta, args):
    dx = meta.diagnosis.astype(str).to_numpy()
    grid = sorted(set(args.grid))
    # scores[method][control][seed][i] -> summary dict (or None when the pool is too small)
    scores = {m: {c: [] for c in CONTROLS} for m in METHODS}
    test_sizes, pool_sizes = [], []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        pool, test = make_split(meta, args.split, rng, args.test_frac, seed)
        test_sizes.append(len(test))
        pool_sizes.append(len(pool))
        meta_test = meta.iloc[test].reset_index(drop=True)
        masks = {c: allowed_mask(meta_test, c) for c in CONTROLS}
        order = rng.permutation(pool)  # nested training sets: each larger set contains the smaller ones
        for m in METHODS:
            for c in CONTROLS:
                scores[m][c].append([])
        for n_train in grid:
            if n_train > len(pool):
                for m in METHODS:
                    for c in CONTROLS:
                        scores[m][c][-1].append(None)
                continue
            train = order[:n_train]
            shuffled = shuffle_within(train, dx, rng)
            for m in METHODS:
                control = m.endswith(SHUFFLED)
                Qhat = link(Q, C, train, shuffled if control else train, test, m.removesuffix(SHUFFLED))
                for c in CONTROLS:
                    scores[m][c][-1].append(summarize(*ranks_of_truth(Qhat, C[test], masks[c]), n_sims=200))

    curves = {}
    for m in METHODS:
        curves[m] = {}
        for c in CONTROLS:
            def stat(key):
                return np.array([[np.nan if s is None or not s.get("n_queries") else s[key] for s in run]
                                 for run in scores[m][c]], dtype=float)
            top1, chance, p = stat("top1"), stat("chance_top1"), stat("p_value")
            curves[m][c] = {
                "top1_mean": np.nanmean(top1, 0).tolist(), "top1_sd": np.nanstd(top1, 0).tolist(),
                "top10_mean": np.nanmean(stat("top10"), 0).tolist(),
                "chance_mean": np.nanmean(chance, 0).tolist(),
                "lift_mean": (np.nanmean(top1, 0) / np.nanmean(chance, 0)).tolist(),
                "frac_seeds_p_lt_0.05": np.nanmean(p < 0.05, 0).tolist(),
            }
    return {"grid": grid, "seeds": args.seeds, "test_size_mean": float(np.mean(test_sizes)),
            "pool_size_min": int(min(pool_sizes)), "curves": curves}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--split", choices=["random", "site", "project"], default="random")
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--grid", type=int, nargs="+", default=[0, 5, 10, 20, 30, 50, 75, 100, 120, 150])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--reports-name", default="reports")
    args = ap.parse_args()

    e = data.load(args.model, args.reports_name)
    tests = {}
    if e.has_R.sum() > 20:
        tests["test2"] = (e.R[e.has_R], e.A[e.has_R], e.patients[e.has_R].reset_index(drop=True))
    if e.has_B.sum() > 20:
        tests["test1"] = (e.A[e.has_B], e.B[e.has_B], e.patients[e.has_B].reset_index(drop=True))

    out = {"model": args.model, "split": args.split, "test_frac": args.test_frac, "tests": {}}
    for name, (Q, C, meta) in tests.items():
        res = run_test(Q, C, meta, args)
        out["tests"][name] = res
        label = "report -> slide A" if name == "test2" else "slide A -> slide B"
        print(f"\n{name} ({label}), split={args.split}, ~{res['test_size_mean']:.0f} test patients, "
              f"{args.seeds} seeds. top-1 (x chance) by training pairs:")
        for c in CONTROLS:
            print(f"  candidates={c}")
            print("    n_train " + "".join(f"{m.replace(SHUFFLED, ' (shuf)'):>22s}" for m in METHODS))
            for i, n_train in enumerate(res["grid"]):
                cells = []
                for m in METHODS:
                    cur = res["curves"][m][c]
                    t, lift = cur["top1_mean"][i], cur["lift_mean"][i]
                    cells.append(f"{'n/a':>22s}" if np.isnan(t) else f"{t:>11.3f} (x{lift:4.1f})  ")
                print(f"    {n_train:7d} " + "".join(cells))

    suffix = "" if args.split == "random" else f"_{args.split}"
    dst = paths.results_dir(args.model) / f"linkage{suffix}.json"
    dst.write_text(json.dumps(out, indent=2))
    plot(out, paths.results_dir(args.model) / f"linkage{suffix}.png")
    print(f"-> {dst} and linkage{suffix}.png")


def plot(out, dst):
    names = list(out["tests"])
    fig, axes = plt.subplots(len(names), 2, figsize=(12, 4.2 * len(names)), squeeze=False)
    color = {"cosine": "C0", "procrustes": "C1", "ridge": "C2"}
    for row, name in zip(axes, names):
        res = out["tests"][name]
        for ax, c in zip(row, ("all", "dx")):
            for m in METHODS:  # shuffled-pair controls share their method's color, dashed
                cur = res["curves"][m][c]
                mean, sd = np.array(cur["top1_mean"]), np.array(cur["top1_sd"])
                base = m.removesuffix(SHUFFLED)
                ls = "--" if m.endswith(SHUFFLED) else (":" if m == "cosine" else "-")
                ax.plot(res["grid"], mean, ls=ls, marker="o", ms=3, color=color[base], label=m)
                if m in LEARNED:
                    ax.fill_between(res["grid"], mean - sd, mean + sd, color=color[base], alpha=0.15)
            ax.plot(res["grid"], res["curves"]["cosine"][c]["chance_mean"], color="gray", lw=1, label="chance")
            what = "report -> slide" if name == "test2" else "slide -> slide"
            ax.set_title(f"{out['model']} {what}, split={out['split']}, candidates={c}")
            ax.set_xlabel("attacker's training pairs (disjoint patients)")
            ax.set_ylabel("top-1 re-identification")
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(dst, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
