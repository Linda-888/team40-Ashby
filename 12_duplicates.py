"""Step 12 (sanity check): are slide A and slide B really different pieces of tissue?

TCGA can file one piece of tissue under two barcodes: the same glass scanned twice (a frozen section
filed as both TS1 and DX1), or an "FFPE" slide cut from the same piece as the frozen slide (often two
serial sections side by side). Matching such a pair is matching tissue to itself. For each patient
this compares the two thumbnails, whole and section by section (best over rotations and flips), on
two scores, each judged against the same score for slides of two different patients:
  outline  overlap (IoU) of the tissue outlines: catches the same glass scanned twice;
  layout   correlation of the light/dark layout inside the tissue: catches a re-cut of the same piece,
           where pale tissue changes the outline but dense regions and holes stay in place.
A pair above the 99th percentile of other-patient pairs on either score is flagged and removed, and
Test 1 is re-run on the rest.
    python 12_duplicates.py
Writes data/patients_clean.csv (all patients minus the flagged ones; every analysis then uses it, see
fingerprints/paths.py), results/duplicates.csv, duplicates.json, duplicates_examples.png and
duplicates_borderline.png. Run it after tiling (it needs the thumbnails) and before 05-11.
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
from fingerprints.matching import CONTROLS, fmt, ranks_of_truth, test1  # noqa: E402
from fingerprints.tiling import tissue_mask  # noqa: E402

OUTLINE_GRID, LAYOUT_GRID = 64, 32


def _crop(x, mask):
    ys, xs = np.nonzero(mask)
    return x[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _resize(x, n):
    return np.asarray(Image.fromarray(x.astype(np.float32)).resize((n, n), Image.BILINEAR))


def _sections(keep):
    """The whole tissue plus each section on its own: fragments merged by dilating, pieces at least
    20% of the largest (a slide often carries two serial sections of one piece)."""
    parts = [keep]
    sec, n = ndimage.label(ndimage.binary_dilation(keep, iterations=4))
    if n > 1:
        area = ndimage.sum(keep, sec, range(1, n + 1))
        parts += [keep & (sec == i + 1) for i in np.flatnonzero(area >= 0.2 * area.max())]
    return parts


def _drop_specks(mask, frac):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(sizes >= frac * sizes.sum()))


def views(slide_id, width=512):
    """(outline shapes, layout maps) for one slide; both empty if there is no thumbnail or no tissue."""
    f = paths.thumbs_dir() / f"{slide_id}.jpg"
    if not f.exists():
        return [], []
    im = Image.open(f).convert("RGB")
    im = im.resize((width, max(1, round(im.height * width / im.width))))
    mask, _ = tissue_mask(np.asarray(im))
    keep = _drop_specks(mask, 0.01)
    if not keep.any():
        return [], []
    outline = [_resize(_crop(p, p), OUTLINE_GRID) > 0.5 for p in _sections(keep)]

    # layout: how dark the tissue is, on a looser mask that also keeps pale (e.g. alveolar) tissue
    gray = np.asarray(im.convert("L"), np.float32) / 255
    glass = float(np.median(gray[~mask])) if (~mask).any() else 0.95
    loose = _drop_specks(ndimage.binary_closing(gray < glass - 0.04, iterations=2), 0.02)
    layout = []
    for p in _sections(loose if loose.any() else keep):
        d = _resize(_crop((1 - gray) * p, p), LAYOUT_GRID)
        layout.append((d - d.mean()) / (d.std() + 1e-6))
    return outline, layout


def _turns(b):
    """The 8 rotations and flips (a scanner may store the slide turned)."""
    for k in range(4):
        r = np.rot90(b, k)
        yield r
        yield r[:, ::-1]


def best_iou(sa, sb):
    return max(float((a & c).sum() / max((a | c).sum(), 1)) for a in sa for b in sb for c in _turns(b))


def best_corr(sa, sb):
    return max(float((a * c).mean()) for a in sa for b in sb for c in _turns(b))


def montage(rows, dst, title, extra=None):
    fig, axes = plt.subplots(len(rows), 2, figsize=(7, 2.1 * len(rows)))
    for ax_row, r in zip(np.atleast_2d(axes), rows):
        for ax, sid in zip(ax_row, (r.slide_a, r.slide_b)):
            ax.imshow(Image.open(paths.thumbs_dir() / f"{sid}.jpg"))
            ax.set_title(sid, fontsize=7)
            ax.axis("off")
        note = f"\n{extra[r.patient_id]}" if extra else ""
        ax_row[0].text(-0.05, 0.5, f"outline {r.outline:.2f}\nlayout {r.layout:.2f}{note}", transform=ax_row[0].transAxes,
                       ha="right", va="center", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(dst, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="flag a pair above this quantile of other-patient pairs on either score")
    ap.add_argument("--n-null", type=int, default=10, help="other-patient pairs per patient for the baseline")
    args = ap.parse_args()

    pts = pd.read_csv(paths.all_patients_csv(), dtype={"site": str})
    V = {s: views(s) for s in pd.concat([pts.slide_a, pts.slide_b])}
    ok = pts[[bool(V[a][0]) and bool(V[b][0]) for a, b in zip(pts.slide_a, pts.slide_b)]].reset_index(drop=True)
    ok["outline"] = [best_iou(V[a][0], V[b][0]) for a, b in zip(ok.slide_a, ok.slide_b)]
    ok["layout"] = [best_corr(V[a][1], V[b][1]) for a, b in zip(ok.slide_a, ok.slide_b)]

    rng = np.random.default_rng(0)
    others = [(ok.slide_a[i], ok.slide_b[j]) for i in range(len(ok))
              for j in rng.choice(np.delete(np.arange(len(ok)), i), args.n_null, replace=False)]
    null = {"outline": np.array([best_iou(V[a][0], V[b][0]) for a, b in others]),
            "layout": np.array([best_corr(V[a][1], V[b][1]) for a, b in others])}
    thr = {k: float(np.quantile(v, args.quantile)) for k, v in null.items()}
    ok["margin"] = np.maximum(ok.outline - thr["outline"], ok.layout - thr["layout"])
    ok["same_tissue"] = ok.margin > 0
    n_dup = int(ok.same_tissue.sum())
    for k in null:
        print(f"{k:7s}: other-patient median {np.median(null[k]):.2f}, {args.quantile:.0%} {thr[k]:.2f}; "
              f"same-patient median {ok[k].median():.2f}; above the cut {int((ok[k] > thr[k]).sum())}")
    print(f"{n_dup}/{len(ok)} pairs look like the same piece of tissue (rescanned, or re-cut)")
    print(ok.groupby("pair_kind").same_tissue.agg(["sum", "count"]).to_string())

    out = {"quantile": args.quantile, "thresholds": thr, "n_pairs": len(ok), "n_same_tissue": n_dup,
           "n_by_score": {k: int((ok[k] > thr[k]).sum()) for k in null},
           "null_median": {k: float(np.median(v)) for k, v in null.items()},
           "by_pair_kind": {k: {"same_tissue": int(g.same_tissue.sum()), "n": int(len(g))}
                            for k, g in ok.groupby("pair_kind")},
           "test1_without_same_tissue": {}}

    dup_ids = set(ok.patient_id[ok.same_tissue])
    hits = {}  # patient -> models that rank the true slide B first, after cleaning
    clean = pts[~pts.patient_id.isin(dup_ids)]
    clean.to_csv(paths.clean_patients_csv(), index=False)
    out["n_clean"] = len(clean)
    print(f"{len(clean)} patients -> {paths.clean_patients_csv()} (the analyses use this list from now on)")
    for m in paths.MODELS:
        if not (paths.emb_dir(m) / "slides.npz").exists():
            continue
        e = data.load(m)
        keep = ~e.patients.patient_id.isin(dup_ids).to_numpy()
        out["test1_without_same_tissue"][m] = {}
        print(f"\n{m}: Test 1 on the {keep.sum()} patients whose slides look like different pieces of tissue")
        for control in CONTROLS:
            s = test1(e.subset(keep), control)
            out["test1_without_same_tissue"][m][control] = s
            print(f"  {control:8s} {fmt(s)}")
        es = e.subset(keep)
        r, _ = ranks_of_truth(es.A[es.has_B], es.B[es.has_B])
        for pid in es.patients.patient_id[es.has_B].to_numpy()[r == 1]:
            hits.setdefault(pid, []).append(m)

    out["top1_hits"] = hits
    res = paths.root() / "results"
    top = ok.sort_values("margin", ascending=False)
    top.to_csv(res / "duplicates.csv", index=False)
    (res / "duplicates.json").write_text(json.dumps(out, indent=2))
    montage(list(top.head(6).itertuples()) + list(top.tail(2).itertuples()), res / "duplicates_examples.png",
            "Slide A vs slide B: six most and two least alike (flag above the other-patient 99th percentile)")
    above, below = top[top.same_tissue].tail(4), top[~top.same_tissue].head(4)
    montage(list(above.itertuples()) + list(below.itertuples()), res / "duplicates_borderline.png",
            "Around the cut: the four least alike flagged pairs, then the four most alike kept pairs")
    if hits:
        order = sorted(hits, key=lambda p: -len(hits[p]))[:16]
        rows = list(ok.set_index("patient_id").loc[order].reset_index().itertuples())
        montage(rows, res / "top1_pairs.png", "Clean pairs the attack ranks first (models that do)",
                {p: f"{len(v)} model{'s' if len(v) > 1 else ''}" for p, v in hits.items()})
    print(f"-> {res}/duplicates.csv, duplicates.json, duplicates_examples.png, duplicates_borderline.png, top1_pairs.png")


if __name__ == "__main__":
    main()
