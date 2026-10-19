"""Step 13 (sanity check): does pen ink on the glass drive the match?

Pathologists mark slides with pen. Ink is saturated, so the tissue mask can keep tiles that are mostly
ink, and two slides of one patient marked the same way could match on the ink alone. For every slide
this scores each tile's ink: the share of pixels that are green to cyan (hue 0.20-0.58, saturated),
near-black, or saturated pure red, colors H&E does not produce. Tiles above --max-ink are dropped,
each slide embedding is re-averaged from the rest (as in 03_embed_images.py), and Test 1 is re-run
on the analysis patients (the clean list, see 12_duplicates.py).
    python 13_ink.py
    python 13_ink.py --max-ink 0.02
Writes results/ink.json and results/ink_examples.png (the inkiest tiles, to check the rule by eye).
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.matching import CONTROLS, fmt, ranks_of_truth, test1  # noqa: E402


def hsv(rgb):
    """rgb uint8 (..., 3) -> h, s, v in [0, 1]."""
    x = rgb.astype(np.float32) / 255
    mx, mn = x.max(-1), x.min(-1)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    dd = np.maximum(d, 1e-6)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4)) / 6
    return np.where(d > 0, h, 0), s, mx


def ink_fraction(tiles, step=4):
    """Share of ink pixels per tile (tiles: n x H x W x 3 uint8), on every step-th pixel."""
    h, s, v = hsv(tiles[:, ::step, ::step])
    green_blue = (h >= 0.20) & (h <= 0.58) & (s >= 0.25) & (v >= 0.15)
    black = v < 0.15
    red = ((h < 0.03) | (h > 0.97)) & (s > 0.75) & (v > 0.35)
    return (green_blue | black | red).mean(axis=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-ink", type=float, default=0.05, help="drop tiles with more ink pixels than this share")
    args = ap.parse_args()

    pts = data.load(paths.MODELS[0]).patients  # analysis patients with slide A
    slides = list(dict.fromkeys(list(pts.slide_a) + list(pts.slide_b)))
    keep, worst = {}, []
    for k, sid in enumerate(slides):
        f = paths.tiles_dir() / f"{sid}.npz"
        if not f.exists():
            continue
        tiles = np.load(f)["tiles"]
        frac = ink_fraction(tiles)
        keep[sid] = frac <= args.max_ink
        for i in np.argsort(frac)[-2:]:
            worst.append((float(frac[i]), sid, tiles[i]))
        worst = sorted(worst, key=lambda w: -w[0])[:24]
        if k % 50 == 0:
            print(f"[{k}/{len(slides)}] slides scored", flush=True)
    n_tiles = sum(len(v) for v in keep.values())
    n_drop = sum(int((~v).sum()) for v in keep.values())
    inky = [s for s, v in keep.items() if (~v).any()]
    print(f"{n_drop}/{n_tiles} tiles ({n_drop / n_tiles:.1%}) above {args.max_ink:.0%} ink, on {len(inky)}/{len(keep)} slides")
    out = {"max_ink": args.max_ink, "n_slides": len(keep), "n_tiles": n_tiles, "n_dropped": n_drop,
           "slides_with_ink": len(inky), "test1_ink_free": {}, "test1_same_patients": {}, "top1_hits_ink_free": {}}

    for m in paths.MODELS:
        if not (paths.emb_dir(m) / "slides.npz").exists():
            continue
        e = data.load(m)

        def clean(sid, fallback):
            f = paths.emb_dir(m) / "tiles" / f"{sid}.npy"
            if sid not in keep or not f.exists() or not keep[sid].any():
                return fallback
            x = np.load(f).astype(np.float32)[keep[sid]].mean(0)
            return x / np.linalg.norm(x)

        A = np.stack([clean(s, a) for s, a in zip(e.patients.slide_a, e.A)])
        B = np.stack([clean(s, b) if hb else b for s, b, hb in zip(e.patients.slide_b, e.B, e.has_B)])
        f = e.with_(A=A, B=B)
        out["test1_ink_free"][m], out["test1_same_patients"][m] = {}, {}
        print(f"\n{m}: Test 1 without ink tiles (before -> after)")
        for control in CONTROLS:
            before, after = test1(e, control), test1(f, control)
            out["test1_same_patients"][m][control], out["test1_ink_free"][m][control] = before, after
            print(f"  {control:8s} {fmt(before)}\n  {'':8s} {fmt(after)}")
        r, _ = ranks_of_truth(f.A[f.has_B], f.B[f.has_B])
        out["top1_hits_ink_free"][m] = f.patients.patient_id[f.has_B].to_numpy()[r == 1].tolist()

    res = paths.root() / "results"
    (res / "ink.json").write_text(json.dumps(out, indent=2))
    fig, axes = plt.subplots(4, 6, figsize=(12, 8.5))
    for ax, (frac, sid, t) in zip(axes.flat, worst):
        ax.imshow(t)
        ax.set_title(f"{sid[:12]}  ink {frac:.0%}", fontsize=7)
        ax.axis("off")
    for ax in axes.flat[len(worst):]:
        ax.axis("off")
    fig.suptitle(f"The inkiest tiles (dropped above {args.max_ink:.0%} ink pixels)", fontsize=10)
    fig.tight_layout()
    fig.savefig(res / "ink_examples.png", dpi=90)
    print(f"-> {res}/ink.json, ink_examples.png")


if __name__ == "__main__":
    main()
