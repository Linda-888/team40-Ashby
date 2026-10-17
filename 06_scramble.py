"""Step 8, Defense 1: rotate each release with its own secret key, then break it with Procrustes.

For each test, the query side and the candidate side get different random rotations. The attacker
learns k true pairs, fits the rotation, and we score re-identification on the remaining patients.
Two attacks: plain Procrustes (the plan), and PCA-per-release + Procrustes, which needs far fewer
pairs because PCA coordinates do not change under a rotation. Each k is compared with no scrambling
on the same n-k patients, since the candidate pool (and so the chance level) shrinks as k grows.
    python 06_scramble.py --model conch
Writes results/<model>/scramble.json and scramble.png (match rate vs number of known pairs).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.defenses import procrustes_attack, random_rotation  # noqa: E402
from fingerprints.matching import allowed_mask, fmt, ranks_of_truth, summarize  # noqa: E402

PLOT_CONTROLS = ("all", "dx+site")
CLEAR = "no scrambling"


def score(Q, C, meta, control):
    return summarize(*ranks_of_truth(Q, C, allowed_mask(meta, control)), n_sims=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200])
    ap.add_argument("--pca-dims", type=int, default=32, help="dims for the PCA + Procrustes attack")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--reports-name", default="reports")
    args = ap.parse_args()
    attacks = {"procrustes": 0, f"pca{args.pca_dims}+procrustes": args.pca_dims}

    e = data.load(args.model, args.reports_name)
    tests = {}
    if e.has_B.sum() > 10:
        tests["test1"] = (e.A[e.has_B], e.B[e.has_B], e.patients[e.has_B].reset_index(drop=True))
    if e.has_R.sum() > 10:
        tests["test2"] = (e.R[e.has_R], e.A[e.has_R], e.patients[e.has_R].reset_index(drop=True))

    out = {"model": args.model, "ks": args.ks, "seeds": args.seeds, "attacks": list(attacks), "tests": {}}
    for name, (Q, C, meta) in tests.items():
        n, d = Q.shape
        ks = [k for k in args.ks if k <= n - 10]
        rng = np.random.default_rng(0)
        res = {"n": n, "ks": ks, "clear": {}, "same_key": {}, "curves": {}, "k_to_half": {}}
        key = random_rotation(d, rng)
        for c in PLOT_CONTROLS:
            res["clear"][c] = score(Q, C, meta, c)
            # A hospital that reuses one key for both releases: rotation keeps every dot product.
            res["same_key"][c] = score(Q @ key, C @ key, meta, c)
        print(f"\n{name}: clear       {fmt(res['clear']['all'])}")
        print(f"{name}: same key    {fmt(res['same_key']['all'])}")

        # "no scrambling" is scored on the same n-k patients as each attack: fewer candidates means a
        # higher chance level, so comparing against the full-set baseline would flatter the attack.
        runs = {a: {c: [] for c in PLOT_CONTROLS} for a in [CLEAR, *attacks]}  # [seed][k] -> summary
        for seed in range(args.seeds):
            rng = np.random.default_rng(1000 + seed)
            Qr, Cr = Q @ random_rotation(d, rng), C @ random_rotation(d, rng)
            perm = rng.permutation(n)
            for a in runs:
                for c in PLOT_CONTROLS:
                    runs[a][c].append([])
            for k in ks:
                known, rest = perm[:k], perm[k:]  # score only patients the attacker didn't know
                meta_rest = meta.iloc[rest].reset_index(drop=True)
                for c in PLOT_CONTROLS:
                    runs[CLEAR][c][-1].append(score(Q[rest], C[rest], meta_rest, c))
                for a, dims in attacks.items():
                    Qhat, Chat = procrustes_attack(Qr, Cr, known, pca_dims=min(dims, d))
                    for c in PLOT_CONTROLS:
                        runs[a][c][-1].append(score(Qhat[rest], Chat[rest], meta_rest, c))

        for a in runs:
            res["curves"][a] = {}
            for c in PLOT_CONTROLS:
                top1 = np.array([[s.get("top1", np.nan) for s in run] for run in runs[a][c]])
                chance = np.array([[s.get("chance_top1", np.nan) for s in run] for run in runs[a][c]])
                res["curves"][a][c] = {"mean": np.nanmean(top1, 0).tolist(), "sd": np.nanstd(top1, 0).tolist(),
                                       "chance": np.nanmean(chance, 0).tolist()}
        clear_k = np.array(res["curves"][CLEAR]["all"]["mean"])
        chance_k = np.array(res["curves"][CLEAR]["all"]["chance"])
        print(f"{name}: no scrambling, same n-k patients: "
              + ", ".join(f"k={k}: {m:.3f}" for k, m in zip(ks, clear_k)))
        for a in attacks:
            mean_all = np.array(res["curves"][a]["all"]["mean"])
            # Smallest k where the attack gets back half of what is above chance on the same patients.
            gain = (mean_all - chance_k) / np.maximum(clear_k - chance_k, 1e-9)
            # Only meaningful where no scrambling is clearly above chance (2x) on those patients.
            res["k_to_half"][a] = next((k for k, g, cl, ch in zip(ks, gain, clear_k, chance_k)
                                        if g >= 0.5 and cl >= 2 * ch), None)
            print(f"{name}: different keys + {a} (top-1, all): "
                  + ", ".join(f"k={k}: {m:.3f}" for k, m in zip(ks, mean_all)))
            print(f"{name}:   known pairs to recover half of the above-chance match rate: {res['k_to_half'][a]}")
        out["tests"][name] = res

    dst = paths.results_dir(args.model) / "scramble.json"
    dst.write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, len(PLOT_CONTROLS), figsize=(6.5 * len(PLOT_CONTROLS), 4.5), squeeze=False)
    styles = {CLEAR: "--", **dict(zip(attacks, ["-.", "-"]))}
    for ax, c in zip(axes[0], PLOT_CONTROLS):
        for ti, (name, res) in enumerate(out["tests"].items()):
            color = f"C{ti}"
            for a, ls in styles.items():
                cur = res["curves"][a][c]
                m, s = np.array(cur["mean"]), np.array(cur["sd"])
                label = f"{name}: {a}" if a == CLEAR else f"{name}: keys + {a}"
                ax.plot(res["ks"], m, ls=ls, marker="o", ms=3, color=color, label=label)
                if a != CLEAR:
                    ax.fill_between(res["ks"], m - s, m + s, color=color, alpha=0.15)
            ax.plot(res["ks"], res["curves"][CLEAR][c]["chance"], color=color, ls=":", lw=1,
                    label=f"{name}: chance")
        ax.set_xlabel("known true pairs (k)")
        ax.set_ylabel("top-1 match rate")
        ax.set_title(f"{args.model}: candidates = {c}")
        ax.set_xscale("symlog", linthresh=5)
        ax.set_xlim(left=0)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.results_dir(args.model) / "scramble.png", dpi=160)
    print(f"-> {dst} and scramble.png")


if __name__ == "__main__":
    main()
