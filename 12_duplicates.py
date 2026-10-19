"""Step 12 (sanity check): are slide A and slide B really different specimens?

TCGA can hold two scans of the same piece of tissue under different barcodes (for example a frozen
section filed as both TS1 and DX1). Matching such a pair is matching a slide to a copy of itself.
For each patient this compares the tissue outline on the two thumbnails: crop each to its tissue
(ignoring specks), resize to a common grid and score the overlap (IoU, best over rotations and
flips). The same section twice scores far above what two different patients' slides score. Then
Test 1 is re-run on the patients whose pair looks like two different specimens.
    python 12_duplicates.py                 # flag pairs above the 99th percentile of other-patient pairs
    python 12_duplicates.py --threshold 0.7
Writes data/patients_clean.csv (all patients minus the flagged ones; every analysis then uses it, see
fingerprints/paths.py), results/duplicates.csv, results/duplicates.json and results/duplicates_examples.png.
Run it after tiling (it needs the thumbnails) and before 05-11.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.matching import CONTROLS, fmt, test1  # noqa: E402
from fingerprints.tiling import tissue_mask  # noqa: E402

GRID = 64


def outline(slide_id, width=256):
    """Tissue mask cropped to its tissue and resized to GRID x GRID (None if no thumbnail)."""
    f = paths.thumbs_dir() / f"{slide_id}.jpg"
    if not f.exists():
        return None
    im = Image.open(f).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    mask, _ = tissue_mask(np.asarray(im))
    lab, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.01 * sizes.sum()))  # drop specks
    ys, xs = np.nonzero(keep)
    crop = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.float32)
    small = np.asarray(Image.fromarray(crop * 255).resize((GRID, GRID), Image.BILINEAR)) > 127
    return small


def iou(a, b):
    """Best overlap over the 8 rotations and flips (a scanner may store the slide turned)."""
    best = 0.0
    for k in range(4):
        r = np.rot90(b, k)
        for c in (r, r[:, ::-1]):
            u = (a | c).sum()
            best = max(best, (a & c).sum() / u if u else 0.0)
    return float(best)


def montage(rows, dst, title):
    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 2.1 * len(rows)))
    for ax_row, r in zip(np.atleast_2d(axes), rows):
        for ax, sid in zip(ax_row, (r.slide_a, r.slide_b)):
            ax.imshow(Image.open(paths.thumbs_dir() / f"{sid}.jpg"))
            ax.set_title(sid, fontsize=7)
            ax.axis("off")
        ax_row[0].text(-0.05, 0.5, f"IoU {r.iou:.2f}", transform=ax_row[0].transAxes, ha="right", va="center",
                       fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, help="IoU above which a pair counts as the same section "
                                                     "(default: 99th percentile of other-patient pairs)")
    ap.add_argument("--n-null", type=int, default=5, help="other-patient pairs per patient for the baseline")
    args = ap.parse_args()

    pts = pd.read_csv(paths.all_patients_csv(), dtype={"site": str})
    A = {s: outline(s) for s in pts.slide_a}
    B = {s: outline(s) for s in pts.slide_b}
    ok = pts[[A[a] is not None and B[b] is not None for a, b in zip(pts.slide_a, pts.slide_b)]].reset_index(drop=True)
    ok["iou"] = [iou(A[a], B[b]) for a, b in zip(ok.slide_a, ok.slide_b)]

    rng = np.random.default_rng(0)
    null = [iou(A[ok.slide_a[i]], B[ok.slide_b[j]])
            for i in range(len(ok)) for j in rng.choice(np.delete(np.arange(len(ok)), i), args.n_null, replace=False)]
    thr = args.threshold if args.threshold is not None else float(np.quantile(null, 0.99))
    ok["same_section"] = ok.iou > thr
    n_dup = int(ok.same_section.sum())
    print(f"{len(ok)} pairs with both thumbnails. Other-patient IoU: median {np.median(null):.2f}, "
          f"99th pct {np.quantile(null, 0.99):.2f}. Same-patient IoU: median {ok.iou.median():.2f}.")
    print(f"threshold {thr:.2f}: {n_dup}/{len(ok)} pairs look like the same section scanned twice")
    print(ok.groupby("pair_kind").same_section.agg(["sum", "count"]).to_string())

    out = {"threshold": thr, "n_pairs": len(ok), "n_same_section": n_dup,
           "null_iou": {"median": float(np.median(null)), "p99": float(np.quantile(null, 0.99))},
           "same_patient_iou_median": float(ok.iou.median()),
           "by_pair_kind": {k: {"same_section": int(g.same_section.sum()), "n": int(len(g))}
                            for k, g in ok.groupby("pair_kind")},
           "test1_without_same_section": {}}

    dup_ids = set(ok.patient_id[ok.same_section])
    clean = pts[~pts.patient_id.isin(dup_ids)]
    clean.to_csv(paths.clean_patients_csv(), index=False)
    out["n_clean"] = len(clean)
    print(f"{len(clean)} patients -> {paths.clean_patients_csv()} (the analyses use this list from now on)")
    for m in paths.MODELS:
        if not (paths.emb_dir(m) / "slides.npz").exists():
            continue
        e = data.load(m)
        keep = ~e.patients.patient_id.isin(dup_ids).to_numpy()
        out["test1_without_same_section"][m] = {}
        print(f"\n{m}: Test 1 on the {keep.sum()} patients whose slides look like different specimens")
        for control in CONTROLS:
            s = test1(e.subset(keep), control)
            out["test1_without_same_section"][m][control] = s
            print(f"  {control:8s} {fmt(s)}")

    res = paths.root() / "results"
    ok.sort_values("iou", ascending=False).to_csv(res / "duplicates.csv", index=False)
    (res / "duplicates.json").write_text(json.dumps(out, indent=2))
    top = ok.sort_values("iou", ascending=False)
    montage(list(top.head(6).itertuples()) + list(top.tail(2).itertuples()), res / "duplicates_examples.png",
            f"Slide A vs slide B: six highest and two lowest outline overlaps (flag above IoU {thr:.2f})")
    print(f"-> {res}/duplicates.csv, duplicates.json, duplicates_examples.png")


if __name__ == "__main__":
    main()
