"""Deck figures and key numbers from every model's results (run after 05, 06, 07 and 10).

  1. fig1_ladder       Test 1 and zero-shot Test 2 under stricter candidate pools (all / diagnosis / hospital).
  2. fig2_test2        Test 2 zero-shot vs learned map (random split, unseen hospitals, same hospital, control).
  3. fig3_scorecard    Every model side by side, colored by how much it leaks.
  4. fig4_linkage      Learned-map curves against the attacker's number of training pairs.
  5. fig5_defenses     Privacy vs utility for noise and identity-direction removal.
  6. fig6_scrambling   Test 1 under secret-key scrambling against known pairs.
Titles only describe; key_numbers.md states the takeaways computed from the data, so a figure can
never claim more than the numbers show. All re-identification numbers are "x chance" (top-1 / chance).

    python 11_figures.py            # -> results/figures/*.png and results/figures/key_numbers.md
"""
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from fingerprints import paths  # noqa: E402

LABEL = {"conch": "CONCH", "musk": "MUSK", "pathgen": "PathGen-CLIP", "quilt": "QuiltNet", "plip": "PLIP"}
COLOR = {"conch": "#1f77b4", "musk": "#9467bd", "pathgen": "#2ca02c", "quilt": "#ff7f0e", "plip": "#d62728"}
LEARNED = ("procrustes", "ridge")
T1 = "#3b6ea8"  # Test 1 blue
T2 = "#e08a2c"  # Test 2 orange

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def load_all():
    res = {}
    for m in paths.MODELS:
        d = paths.root() / "results" / m
        r = {name: json.loads((d / f"{name}.json").read_text())
             for name in ("tests", "scramble", "removal", "linkage", "linkage_site") if (d / f"{name}.json").exists()}
        if "tests" in r:
            res[m] = r
    return res


def lift(s):
    if not s or not s.get("n_queries"):
        return math.nan
    return s["top1"] / s["chance_top1"]


def lift_ci(s):
    lo, hi = s["top1_ci95"]
    return max(lo, 1e-9) / s["chance_top1"], hi / s["chance_top1"]


def finite(xs):
    return [x for x in xs if x == x]


def offsets(n, width=0.7):
    return [(j - (n - 1) / 2) * width / max(n, 1) for j in range(n)]


def log_y(ax, values, lo=0.5):
    vals = finite(values) or [1.0]
    ax.set_yscale("log")
    ax.set_ylim(lo, max(2.0, max(vals) * 1.6))
    ticks = [t for t in (0.5, 1, 2, 5, 10, 20, 50) if lo <= t <= max(2.0, max(vals) * 1.6)]
    ax.set_yticks(ticks, [f"{t:g}x" for t in ticks])
    ax.axhline(1, color="gray", ls="--", lw=1)


# ---- linkage helpers ---------------------------------------------------------------------------

def best_learned(link, test="test2"):
    """(method, index of the largest training size with data) for the best learned map, random split."""
    t = link["tests"][test]
    i = max(j for j, v in enumerate(t["curves"]["cosine"]["all"]["top1_mean"]) if v == v)
    m = max(LEARNED, key=lambda k: t["curves"][k]["all"]["lift_mean"][i])
    return m, i


def linkage_lift(link, method, control, i=None, test="test2"):
    cur = link["tests"][test]["curves"][method][control]["lift_mean"]
    if i is None:
        i = max(j for j, v in enumerate(cur) if v == v)
    return cur[i], link["tests"][test]["grid"][i]


# ---- figures -----------------------------------------------------------------------------------

def fig_ladder(res, dst):
    controls = [("all", "all patients"), ("dx", "same diagnosis"), ("dx+site", "same diagnosis\n+ same hospital")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    allv = []
    for ax, test, title in zip(axes, ("test1", "test2"),
                               ("Test 1: slide -> the same patient's other specimen",
                                "Test 2: report -> the same patient's slide (zero-shot)")):
        models = [m for m in res if res[m]["tests"].get(test)]
        for j, m in enumerate(models):
            for i, (c, _) in enumerate(controls):
                s = res[m]["tests"][test].get(c)
                v = lift(s)
                if v != v:
                    continue
                lo, hi = lift_ci(s)
                x = i + offsets(len(models))[j]
                ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="o", color=COLOR[m], ms=7, capsize=3,
                            label=LABEL[m] if i == 0 else None)
                ax.annotate(f"{v:.1f}x", (x, hi), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
                allv.append(hi)
        ax.set_xticks(range(len(controls)), [lab for _, lab in controls])
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.6, len(controls) - 0.4)
    log_y(axes[0], allv)
    axes[0].set_ylabel("re-identification (top-1, x chance)")
    axes[0].text(-0.55, 1.05, "chance", color="gray", fontsize=8)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Re-identification as the candidate pool gets stricter (top-1 vs chance, 95% CI)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)


def fig_test2(res, dst):
    cats = ["Test 1\nslide -> slide\n(reference)", "Test 2\nzero-shot", "Test 2\nlearned map",
            "learned map,\nunseen hospitals", "learned map,\nsame hospital", "learned map,\nshuffled pairs\n(control)"]
    models = [m for m in res if "linkage" in res[m]]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(12, 4.8))
    allv = []
    for j, m in enumerate(models):
        r, link = res[m], res[m]["linkage"]
        meth, i = best_learned(link)
        vals = [lift(r["tests"]["test1"].get("all")), lift(r["tests"]["test2"].get("all")),
                linkage_lift(link, meth, "all", i)[0],
                linkage_lift(r["linkage_site"], meth, "all")[0] if "linkage_site" in r else math.nan,
                linkage_lift(link, meth, "dx+site", i)[0],
                linkage_lift(link, meth + ", shuffled pairs", "all", i)[0]]
        xs = [k + offsets(len(models))[j] for k in range(len(cats))]
        ax.scatter(xs, vals, color=COLOR[m], s=55, zorder=3, label=f"{LABEL[m]} ({meth}, {link['tests']['test2']['grid'][i]} pairs)")
        for x, v in zip(xs, vals):
            if v == v:
                ax.annotate(f"{v:.1f}x", (x, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
        allv += vals
    ax.axvspan(-0.5, 0.5, color=T1, alpha=0.07)
    ax.axvspan(0.5, len(cats) - 0.5, color=T2, alpha=0.07)
    ax.set_xticks(range(len(cats)), cats, fontsize=8)
    ax.set_xlim(-0.5, len(cats) - 0.5)
    log_y(ax, allv)
    ax.set_ylabel("re-identification (top-1, x chance)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("Test 2: zero-shot vs a learned report -> slide map, with Test 1 as reference",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return True


def scorecard_rows(r):
    """(label, x-chance value, note) rows for one model."""
    t, rows = r["tests"], []
    rows.append(("Test 1  slide -> slide", lift(t["test1"].get("all")), ""))
    rows.append(("Test 1  same hospital", lift(t["test1"].get("dx+site")), ""))
    rows.append(("Test 2  report -> slide, zero-shot", lift(t["test2"].get("all")), ""))
    if "linkage" in r:
        meth, i = best_learned(r["linkage"])
        v, n = linkage_lift(r["linkage"], meth, "all", i)
        rows.append(("Test 2  learned map", v, f"{n} pairs"))
        if "linkage_site" in r:
            v, n = linkage_lift(r["linkage_site"], meth, "all")
            rows.append(("Test 2  learned map, unseen hospitals", v, f"{n} pairs"))
        rows.append(("Test 2  learned map, same hospital", linkage_lift(r["linkage"], meth, "dx+site", i)[0], ""))
    if "scramble" in r and "test1" in r["scramble"]["tests"]:
        sc = r["scramble"]["tests"]["test1"]
        cur = sc["curves"]["procrustes"]["all"]
        ks = sc["ks"]
        lifts = [mu / ch for mu, ch in zip(cur["mean"], cur["chance"])]
        rows.append(("Test 1  scrambled, 0 known pairs", lifts[0], ""))
        k50 = min(range(len(ks)), key=lambda j: abs(ks[j] - 50))
        rows.append((f"Test 1  scrambled, {ks[k50]} known pairs", lifts[k50], ""))
    if "removal" in r:
        noise = r["removal"]["methods"]["noise"]["rows"]
        target = next((row for row in noise if row.get("test1_all_lift", 99) <= 2), noise[-1])
        rows.append(("Test 1  after noise", target["test1_all_lift"],
                     f"sigma {target['strength']:g}, acc {noise[0]['utility']:.2f}->{target['utility']:.2f}"))
        rem = r["removal"]["methods"]["remove"]["rows"]
        rows.append(("Test 1  after removing identity dirs", rem[-1]["test1_all_lift"],
                     f"k={rem[-1]['strength']:g}, acc {rem[-1]['utility']:.2f}"))
    return rows


def fig_scorecard(res, dst):
    models = list(res)
    labels = [lab for lab, _, _ in scorecard_rows(res[models[0]])]
    grid = np.full((len(labels), len(models)), np.nan)
    notes = [["" for _ in models] for _ in labels]
    for j, m in enumerate(models):
        for lab, v, note in scorecard_rows(res[m]):
            if lab.startswith("Test 1  scrambled,") and lab not in labels:  # k may differ per model
                lab = next(x for x in labels if x.startswith("Test 1  scrambled,") and "0 known" not in x)
            if lab in labels:
                i = labels.index(lab)
                grid[i, j], notes[i][j] = v, note
    fig, ax = plt.subplots(figsize=(max(9.0, 2.2 * len(models) + 4.5), 0.55 * len(labels) + 1.4))
    shown = np.clip(np.nan_to_num(grid, nan=1.0), 0.5, 30)
    ax.imshow(shown, cmap="RdYlGn_r", norm=LogNorm(vmin=0.7, vmax=20), aspect="auto")
    for i in range(len(labels)):
        for j in range(len(models)):
            v = grid[i, j]
            txt = "n/a" if v != v else f"{v:.1f}x"
            if notes[i][j]:
                txt += f"\n{notes[i][j]}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(models)), [LABEL[m] for m in models])
    ax.set_yticks(range(len(labels)), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("Privacy scorecard (x chance: green = near chance, red = leaks)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)


def fig_linkage(res, dst):
    models = [m for m in res if "linkage" in res[m]]
    if not models:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, control, title in zip(axes, ("all", "dx+site"), ("candidates: all test patients",
                                                            "candidates: same diagnosis + same hospital")):
        for m in models:
            meth, _ = best_learned(res[m]["linkage"])
            for key, ls, lab in (("linkage", "-", "learned map"), ("linkage_site", "--", "unseen hospitals")):
                if key not in res[m]:
                    continue
                t = res[m][key]["tests"]["test2"]
                ax.plot(t["grid"], t["curves"][meth][control]["lift_mean"], ls=ls, marker="o", ms=3,
                        color=COLOR[m], label=f"{LABEL[m]}: {lab}")
            t = res[m]["linkage"]["tests"]["test2"]
            ax.plot(t["grid"], t["curves"][meth + ", shuffled pairs"][control]["lift_mean"], ls=":",
                    color=COLOR[m], alpha=0.7, label=f"{LABEL[m]}: shuffled-pairs control")
        ax.axhline(1, color="gray", ls="--", lw=1)
        ax.set_xlabel("attacker's training pairs (other patients)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("report -> slide re-identification (x chance)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Learned report -> slide map vs the attacker's training pairs", fontweight="bold")
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return True


def fig_defenses(res, dst):
    models = [m for m in res if "removal" in res[m]]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m in models:
        for method, ls, mk in (("noise", "-", "o"), ("remove", "--", "s")):
            rows = res[m]["removal"]["methods"][method]["rows"]
            xs = [row["test1_all_lift"] for row in rows]
            ys = [row["utility"] for row in rows]
            ax.plot(xs, ys, ls=ls, marker=mk, ms=4, color=COLOR[m],
                    label=f"{LABEL[m]}: {'add noise' if method == 'noise' else 'remove identity directions'}")
        base = res[m]["removal"]["methods"]["noise"]["rows"][0]
        ax.scatter([base["test1_all_lift"]], [base["utility"]], marker="*", s=180, color=COLOR[m],
                   edgecolor="k", zorder=5)
    ax.set_xscale("log")
    ax.axvline(2, color="green", ls="--", lw=1)
    ax.text(2.05, 0.02, "goal: < 2x", color="green", fontsize=8, transform=ax.get_xaxis_transform())
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.text(0.01, 0.505, "coin flip (2 subtypes)", color="gray", fontsize=8, transform=ax.get_yaxis_transform())
    ax.set_xlabel("Test 1 re-identification, x chance  (left = more private)")
    ax.set_ylabel("subtype classifier balanced accuracy  (up = more useful)")
    ax.set_title("Privacy vs utility for two defenses (star = original embeddings)", fontweight="bold",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return True


def fig_scrambling(res, dst):
    models = [m for m in res if "scramble" in res[m] and "test1" in res[m]["scramble"]["tests"]]
    if not models:
        return False
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for m in models:
        sc = res[m]["scramble"]["tests"]["test1"]
        for key, ls, lab in (("no scrambling", ":", "no scrambling"), ("procrustes", "-", "secret keys + attack")):
            cur = sc["curves"][key]["all"]
            ax.plot(sc["ks"], [mu / ch for mu, ch in zip(cur["mean"], cur["chance"])], ls=ls, marker="o", ms=3,
                    color=COLOR[m], label=f"{LABEL[m]}: {lab}")
    ax.axhline(1, color="gray", ls="--", lw=1)
    ax.set_xscale("symlog", linthresh=5)
    ax.set_xlim(left=0)
    ax.set_xlabel("true pairs the attacker knows")
    ax.set_ylabel("Test 1 re-identification (x chance)")
    ax.set_title("Test 1 under secret-key scrambling vs known pairs", fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    return True


def takeaways(res):
    """One line per scorecard row: range across models, so the deck quotes data, not hopes."""
    rows = {}
    for m, r in res.items():
        for lab, v, _ in scorecard_rows(r):
            rows.setdefault(lab.replace("  ", " "), []).append((LABEL[m], v))
    lines = ["## Across models (x chance)", ""]
    for lab, vals in rows.items():
        vs = finite([v for _, v in vals])
        if vs:
            lines.append(f"- {lab}: {min(vs):.1f}x to {max(vs):.1f}x  ("
                         + ", ".join(f"{n} {v:.1f}x" for n, v in vals if v == v) + ")")
    lines.append("")
    return lines


def key_numbers(res):
    lines = ["# Key numbers (re-identification as top-1, x chance)", "", *takeaways(res)]
    for m, r in res.items():
        t = r["tests"]
        n = t["test1"]["all"]["n_queries"]
        lines.append(f"## {LABEL[m]} ({n} patients)")
        for test, name in (("test1", "Test 1 slide -> slide"), ("test2", "Test 2 report -> slide, zero-shot")):
            for c, cname in (("all", "all"), ("dx", "same diagnosis"), ("dx+site", "same diagnosis + hospital")):
                s = t[test].get(c)
                if s and s.get("n_queries"):
                    lo, hi = s["top1_ci95"]
                    lines.append(f"- {name}, {cname}: {s['top1']:.1%} vs {s['chance_top1']:.1%} chance "
                                 f"({lift(s):.1f}x, 95% CI {lo:.1%}-{hi:.1%}, p={s['p_value']:.4f}, n={s['n_queries']})")
        for lab, v, note in scorecard_rows(r)[3:]:
            lines.append(f"- {lab.replace('  ', ' ')}: {v:.1f}x" + (f" ({note})" if note else ""))
        lines.append("")
    return "\n".join(lines)


def main():
    res = load_all()
    if not res:
        print("no results yet: run 05_tests.py (and 06, 07, 10) first")
        return
    out = paths.root() / "results" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    made = []
    fig_ladder(res, out / "fig1_ladder.png")
    made.append("fig1_ladder")
    for name, fn in (("fig2_test2", fig_test2), ("fig4_linkage", fig_linkage),
                     ("fig5_defenses", fig_defenses), ("fig6_scrambling", fig_scrambling)):
        if fn(res, out / f"{name}.png"):
            made.append(name)
    fig_scorecard(res, out / "fig3_scorecard.png")
    made.append("fig3_scorecard")
    (out / "key_numbers.md").write_text(key_numbers(res))
    print(f"models: {[LABEL[m] for m in res]}")
    print(f"-> {out}: {', '.join(sorted(made))}, key_numbers.md")


if __name__ == "__main__":
    main()
