"""Feature 2: safe ways to share. Compare release formats on privacy (can an attacker find a patient?)
and usefulness (can a researcher still tell LUAD from LUSC?).

Formats:
  raw           one slide-A embedding per patient (what hospitals share today)
  groups-k      the average of every k patients with the same diagnosis (k-anonymous prototypes)
                  --grouping similar: groups of look-alike patients (MDAV microaggregation; better utility)
                  --grouping random:  random groups (needed for the DP format's guarantee)
  dp-k-eps      random groups of k, each average gets Gaussian noise (differential privacy, delta=1e-5)
  predictions   only the hospital model's P(LUSC) per slide

Attacker: holds a patient's OTHER slide (slide B, a different tissue block) and searches the release.
  raw / predictions: is the top match that patient?      chance = 1 / n patients
  groups / dp:       is the top match that patient's group?  chance = 1 / n groups
  (a group hit only says "one of these >= k people", so it is a weaker leak than a raw hit)
Usefulness: 5-fold CV by patient. The release is built from the training patients only; a logistic
regression trained on it classifies the held-out patients' own slide-A embeddings (balanced accuracy).
For 'predictions' it is the accuracy of the released predictions themselves.

    python 13_safe_release.py --models musk pathgen quilt plip --out safe_release
Writes <out>/<model>.json, <out>/summary.md and <out>/frontier.png (if matplotlib is installed).
Only numpy/pandas/scipy/scikit-learn are needed, so it runs in a small venv on a login node.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import data  # noqa: E402
from fingerprints.data import normalize  # noqa: E402

DELTA = 1e-5


# ------------------------------------------------------------------------------ grouping

def mdav(X: np.ndarray, k: int) -> list[np.ndarray]:
    """Microaggregation (MDAV): groups of >= k mutually similar rows. Returns lists of row indices."""
    left = np.arange(len(X))
    groups = []
    while len(left) >= 2 * k:
        # the row farthest from the remaining centroid, plus its k-1 nearest neighbours
        c = X[left].mean(0)
        far = left[np.argmax(((X[left] - c) ** 2).sum(1))]
        g = left[np.argsort(((X[left] - X[far]) ** 2).sum(1))[:k]]
        groups.append(g)
        left = np.setdiff1d(left, g)
    if len(left):
        groups.append(left)  # k..2k-1 leftovers form the last group
    return groups


def random_groups(n: int, k: int, rng) -> list[np.ndarray]:
    idx = rng.permutation(n)
    m = max(1, n // k)
    return [idx[i::m] for i in range(m)]  # m groups, sizes k..2k-1


def make_groups(X, dx, k, how, rng):
    """Groups never mix diagnoses, so each released average carries one label."""
    out, labels = [], []
    for g in np.unique(dx):
        idx = np.flatnonzero(dx == g)
        if len(idx) < k:
            continue
        parts = mdav(X[idx], k) if how == "similar" else random_groups(len(idx), k, rng)
        out += [idx[p] for p in parts]
        labels += [g] * len(parts)
    return out, np.array(labels)


def dp_sigma(n_min: int, eps: float, clip: float = 1.0) -> float:
    """Gaussian mechanism for a mean of >= n_min rows with norm <= clip (replace-one sensitivity 2C/n).
    The classic formula; strictly proven for eps <= 1, a standard approximation above that."""
    return (2 * clip / n_min) * math.sqrt(2 * math.log(1.25 / DELTA)) / eps


def release(X, dx, fmt, k, eps, how, rng):
    """Returns (items, labels, members) where members[j] = patient rows inside released item j."""
    if fmt == "raw":
        return X, dx, [np.array([i]) for i in range(len(X))]
    groups, labels = make_groups(X, dx, k, "random" if fmt == "dp" else how, rng)
    items = np.stack([X[g].mean(0) for g in groups])
    if fmt == "dp":
        n_min = min(len(g) for g in groups)
        items = items + rng.normal(0, dp_sigma(n_min, eps), items.shape)
    return items, labels, groups


# ------------------------------------------------------------------------------ scoring

def privacy(Q, items, members):
    """Attacker query Q[i] (patient i's slide B) vs every released item. Hit = top item contains i."""
    S = normalize(Q) @ normalize(items).T
    owner = np.full(len(Q), -1)
    for j, m in enumerate(members):
        owner[m] = j
    keep = owner >= 0
    S, owner = S[keep], owner[keep]
    true = S[np.arange(len(S)), owner]
    rank = (S > true[:, None]).sum(1) + 1
    hit = float((rank == 1).mean())
    chance = 1.0 / len(items)
    return dict(top1=hit, chance=chance, lift=hit / chance, top10=float((rank <= 10).mean()),
                median_rank=float(np.median(rank)), n_items=len(items),
                min_group=int(min(len(m) for m in members)))


def clf():
    return LogisticRegression(max_iter=3000, C=1.0)


def usefulness(X, dx, fmt, k, eps, how, seed):
    """Balanced accuracy on held-out patients of a classifier trained on a release of the others."""
    pred = np.empty(len(X), dtype=object)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    for tr, te in cv.split(X, dx):
        items, labels, _ = release(X[tr], dx[tr], fmt, k, eps, how, rng)
        if len(np.unique(labels)) < 2:
            return float("nan")
        pred[te] = clf().fit(normalize(items), labels).predict(normalize(X[te]))
    return float(balanced_accuracy_score(dx, pred.astype(str)))


def predictions_only(A, B, dx, seed):
    """Release = cross-fitted P(LUSC) per patient. Attacker scores slide B with a model trained on
    the other half of patients and matches it to the closest released probability."""
    classes = np.unique(dx)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    p_rel = np.zeros(len(A))
    for tr, te in cv.split(A, dx):
        p_rel[te] = clf().fit(A[tr], dx[tr]).predict_proba(A[te])[:, 1]
    util = float(balanced_accuracy_score(dx, np.where(p_rel > 0.5, classes[1], classes[0])))
    p_att = np.zeros(len(B))
    for tr, te in cv.split(A, dx):
        p_att[te] = clf().fit(A[tr], dx[tr]).predict_proba(B[te])[:, 1]
    D = np.abs(p_att[:, None] - p_rel[None, :])
    rank = (D < np.diag(D)[:, None]).sum(1) + 1
    hit = float((rank == 1).mean())
    chance = 1.0 / len(A)
    return util, dict(top1=hit, chance=chance, lift=hit / chance, top10=float((rank <= 10).mean()),
                      median_rank=float(np.median(rank)), n_items=len(A), min_group=1)


# ------------------------------------------------------------------------------ main

def run_model(model, ks, epss, how, seeds):
    e = data.load(model)
    e = e.subset(e.has_B)
    A, B = normalize(e.A), normalize(e.B)
    dx = e.patients.diagnosis.astype(str).to_numpy()
    rows = []

    def add(name, fmt, k=0, eps=0.0):
        privs, utils = [], []
        for s in range(seeds):
            rng = np.random.default_rng(100 + s)
            items, _, members = release(A, dx, fmt, k, eps, how, rng)
            privs.append(privacy(B, items, members))
            utils.append(usefulness(A, dx, fmt, k, eps, how, s))
        p = {key: float(np.mean([x[key] for x in privs])) for key in privs[0]}
        rows.append(dict(format=name, kind=fmt, k=k, eps=eps, utility=float(np.nanmean(utils)), **p))
        r = rows[-1]
        print(f"[{model}] {name:22s} found at #1 {r['top1']:6.1%} vs {r['chance']:6.1%} chance "
              f"(x{r['lift']:4.1f}) | hidden among >= {r['min_group']:2.0f} | LUAD/LUSC acc {r['utility']:.2f}",
              flush=True)

    add("raw (today)", "raw")
    for k in ks:
        add(f"groups of {k}", "groups", k=k)
    for k in ks:
        for eps in epss:
            add(f"groups of {k} + DP eps={eps:g}", "dp", k=k, eps=eps)
    us, ps = [], []
    for s in range(seeds):
        u, p = predictions_only(A, B, dx, s)
        us.append(u)
        ps.append(p)
    p = {key: float(np.mean([x[key] for x in ps])) for key in ps[0]}
    rows.append(dict(format="predictions only", kind="predictions", k=1, eps=0.0, utility=float(np.mean(us)), **p))
    r = rows[-1]
    print(f"[{model}] {'predictions only':22s} found at #1 {r['top1']:6.1%} vs {r['chance']:6.1%} chance "
          f"(x{r['lift']:4.1f}) | LUAD/LUSC acc {r['utility']:.2f}", flush=True)
    return dict(model=model, n_patients=int(len(A)), grouping=how, delta=DELTA, seeds=seeds, rows=rows)


def summary_md(results):
    lines = ["# Safe ways to share: privacy vs usefulness", "",
             "Found at #1 = attacker holding a patient's other slide picks the released item that contains them. "
             "Chance = 1 / number of released items. Accuracy = LUAD vs LUSC balanced accuracy on held-out patients "
             "(0.50 = coin flip).", ""]
    for res in results:
        lines += [f"## {res['model']} ({res['n_patients']} patients, grouping: {res['grouping']})", "",
                  "| Format | Found at #1 | Chance | x chance | Hidden among at least | Accuracy |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in res["rows"]:
            lines.append(f"| {r['format']} | {r['top1']:.1%} | {r['chance']:.1%} | {r['lift']:.1f}x | "
                         f"{r['min_group']:.0f} | {r['utility']:.2f} |")
        lines.append("")
    lines += ["Notes: DP uses the classic Gaussian mechanism (delta = 1e-5); its formula is proven for eps <= 1 and "
              "is a standard approximation above that. Group membership itself is not hidden by DP here, only "
              "each average's value. A group hit reveals 'one of these k people', not the person."]
    return "\n".join(lines)


def plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping the chart (pip install matplotlib)")
        return
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.2), squeeze=False)
    colors = {"raw": "#c0392b", "groups": "#2471a3", "dp": "#7d3c98", "predictions": "#1e8449"}
    for ax, res in zip(axes[0], results):
        for kind in colors:
            rs = [r for r in res["rows"] if r["kind"] == kind]
            if rs:
                ax.scatter([max(r["lift"], 0.1) for r in rs], [r["utility"] for r in rs], s=45,
                           color=colors[kind], label=kind, zorder=3)
        for r in res["rows"]:
            ax.annotate(r["format"].replace("groups of ", "k=").replace(" + DP eps=", ", eps="),
                        (max(r["lift"], 0.1), r["utility"]), fontsize=6, xytext=(4, 2), textcoords="offset points")
        ax.axvline(1, color="grey", ls=":", lw=1)
        ax.axhline(0.5, color="grey", ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("attacker success, x chance  (left = more private)")
        ax.set_ylabel("LUAD vs LUSC accuracy  (up = more useful)")
        ax.set_title(res["model"])
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    print(f"-> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["musk", "pathgen", "quilt", "plip"])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--eps", type=float, nargs="+", default=[1, 4, 8])
    ap.add_argument("--grouping", choices=["similar", "random"], default="similar")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="safe_release")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for m in args.models:
        try:
            res = run_model(m, args.ks, args.eps, args.grouping, args.seeds)
        except FileNotFoundError as err:
            print(f"skip {m}: {err}")
            continue
        (out / f"{m}.json").write_text(json.dumps(res, indent=2))
        results.append(res)
    if not results:
        sys.exit("no model had embeddings")
    (out / "summary.md").write_text(summary_md(results))
    print(f"-> {out / 'summary.md'}")
    plot(results, out / "frontier.png")


if __name__ == "__main__":
    main()
