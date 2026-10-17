"""Step 9, Defense 2: remove the fingerprint (noise, or projecting out identity directions).

At each strength: Test 1 and Test 2 match rates (privacy) and subtype classifier balanced
accuracy (utility). A fix only counts if it blocks BOTH tests. The sweet spot is the strength
with the best utility among those where both tests are at most --max-lift x chance.

Cross-fitted: patients are split into two halves (balanced by diagnosis). Identity directions are
fit on one half and the defense is scored on the other, then the halves swap and results average.
Fitting and scoring on the same patients would let a k >= n removal erase every in-sample fingerprint
trivially, and would leak the diagnosis labels used for fitting into the classifier's CV.
    python 07_remove.py --model conch
Writes results/<model>/removal.json, removal_<method>.png, tradeoff.png.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.defenses import add_noise, identity_directions, project_out, subtype_utility  # noqa: E402
from fingerprints.matching import test1, test2  # noqa: E402

PRIVACY_CONTROL = "all"  # the realistic attacker ranks against the whole release


def halves(e, seed=0):
    """Fold id (0/1) per patient, balanced by diagnosis."""
    rng = np.random.default_rng(seed)
    dx = e.patients.diagnosis.astype(str).to_numpy()
    fold = np.zeros(e.n, int)
    for g in np.unique(dx):
        idx = np.flatnonzero(dx == g)
        rng.shuffle(idx)
        fold[idx[1::2]] = 1
    return fold


def defended(fit, test, method, strength, rng, removal):
    """The defense applied to the `test` patients; removal directions come from the `fit` patients."""
    if method == "noise":
        return test.with_(A=add_noise(test.A, strength, rng), B=add_noise(test.B, strength, rng),
                          R=add_noise(test.R, strength, rng))
    U = identity_directions(fit, int(strength), removal)
    return test.with_(A=project_out(test.A, U), B=project_out(test.B, U), R=project_out(test.R, U))


def evaluate(e2, labels):
    row = {}
    for name, fn, has in (("test1", test1, e2.has_B), ("test2", test2, e2.has_R)):
        if has.sum() >= 2:
            for control in (PRIVACY_CONTROL, "dx+site"):
                s = fn(e2, control, n_sims=200)
                row[f"{name}_{control}_top1"] = s.get("top1", np.nan)
                row[f"{name}_{control}_chance"] = s.get("chance_top1", np.nan)
                row[f"{name}_{control}_lift"] = s.get("lift_top1", np.nan)
    row["utility"] = subtype_utility(e2.A, labels)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5])
    ap.add_argument("--ks", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 50, 100, 200, 300])
    ap.add_argument("--removal", choices=["mean", "xcov"], default="mean")
    ap.add_argument("--noise-seeds", type=int, default=3)
    ap.add_argument("--max-lift", type=float, default=2.0)
    ap.add_argument("--reports-name", default="reports")
    args = ap.parse_args()

    e = data.load(args.model, args.reports_name)
    labels = e.patients.diagnosis.astype(str).to_numpy()
    if len(np.unique(labels)) < 2:
        print("warning: one diagnosis only, so utility is n/a. Use 2+ projects (e.g. LUAD + LUSC) "
              "or pass --clinical to 01_pick_patients.py")
    fold = halves(e)
    parts = [(e.subset(fold != h), e.subset(fold == h)) for h in (0, 1)]  # (fit, test)
    n_fit = min(len(f.patients) for f, _ in parts)
    print(f"cross-fitted on two halves of {e.n} patients (each scored on ~{e.n // 2}, so chance is higher)")
    grids = {"noise": args.sigmas, "remove": [k for k in args.ks if k < e.d and k <= n_fit]}

    out = {"model": args.model, "removal_method": args.removal, "max_lift": args.max_lift, "methods": {}}
    for method, grid in grids.items():
        rows = []
        for strength in grid:
            reps = args.noise_seeds if method == "noise" and strength > 0 else 1
            runs = [evaluate(defended(fit, test, method, strength, np.random.default_rng(10 * s + h), args.removal),
                             test.patients.diagnosis.astype(str).to_numpy())
                    for s in range(reps) for h, (fit, test) in enumerate(parts)]
            row = {k: float(np.nanmean([r.get(k, np.nan) for r in runs])) for k in runs[0]}
            row["strength"] = strength
            rows.append(row)
            print(f"{method:6s} {strength:>6g}: T1 top-1 {row.get('test1_all_top1', np.nan):.3f} "
                  f"(x{row.get('test1_all_lift', np.nan):.1f})  T2 top-1 {row.get('test2_all_top1', np.nan):.3f} "
                  f"(x{row.get('test2_all_lift', np.nan):.1f})  utility {row['utility']:.3f}", flush=True)
        ok = [r for r in rows
              if all(r.get(f"{t}_{PRIVACY_CONTROL}_lift", 0) <= args.max_lift for t in ("test1", "test2"))]
        sweet = max(ok, key=lambda r: np.nan_to_num(r["utility"], nan=-1)) if ok else None
        out["methods"][method] = {"rows": rows, "sweet_spot": sweet, "baseline_utility": rows[0]["utility"]}
        if sweet:
            print(f"  sweet spot: strength {sweet['strength']:g}, utility {sweet['utility']:.3f} "
                  f"(was {rows[0]['utility']:.3f})")
        else:
            print(f"  no strength brings both tests under {args.max_lift}x chance")

    (paths.results_dir(args.model) / "removal.json").write_text(json.dumps(out, indent=2))
    plot(out, args.model)
    print(f"-> {paths.results_dir(args.model)}/removal.json, removal_*.png, tradeoff.png")


def plot(out, model):
    rd = paths.results_dir(model)
    xlabel = {"noise": "noise strength (sigma)", "remove": "identity directions removed (k)"}
    for method, res in out["methods"].items():
        rows = res["rows"]
        x = [r["strength"] for r in rows]
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for t in ("test1", "test2"):
            key = f"{t}_{PRIVACY_CONTROL}_top1"
            if key in rows[0]:
                line, = ax.plot(x, [r[key] for r in rows], marker="o", label=f"{t} top-1")
                ax.plot(x, [r[f"{t}_{PRIVACY_CONTROL}_chance"] for r in rows], ls=":",
                        color=line.get_color(), label=f"{t} chance")
        ax.plot(x, [r["utility"] for r in rows], marker="s", color="black", label="subtype balanced acc.")
        if res["sweet_spot"]:
            ax.axvline(res["sweet_spot"]["strength"], color="green", ls="--", lw=1, label="sweet spot")
        ax.set_xscale("symlog", linthresh=0.1 if method == "noise" else 1)
        ax.set_xlim(left=0)
        ax.set_xlabel(xlabel[method])
        ax.set_ylabel("rate")
        ax.set_title(f"{model}: {method}")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(rd / f"removal_{method}.png", dpi=160)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for method, res in out["methods"].items():
        rows = res["rows"]
        priv = [max(r.get("test1_all_lift", 0), r.get("test2_all_lift", 0)) for r in rows]
        ax.plot(priv, [r["utility"] for r in rows], marker="o", label=method)
        if res["sweet_spot"]:
            s = res["sweet_spot"]
            ax.scatter([max(s.get("test1_all_lift", 0), s.get("test2_all_lift", 0))], [s["utility"]],
                       s=150, facecolors="none", edgecolors="green", zorder=5)
    ax.axvline(out["max_lift"], color="green", ls="--", lw=1)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(left=0)
    ax.set_xlabel("re-identification (worst of Test 1/2, x chance)   <- more private")
    ax.set_ylabel("subtype balanced accuracy")
    ax.set_title(f"{model}: privacy vs accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(rd / "tradeoff.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
