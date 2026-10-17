"""Step 10: heatmaps of where on a slide the identity signal sits.

For a few correctly re-identified patients (rank 1 in Test 1 and/or Test 2), every tile t of
slide A gets
    score_t = tile_t . match - mean_j(tile_t . other_j)
where match is the patient's slide B (Test 1) or report (Test 2) embedding and other_j are the
other candidates' B / report embeddings. Tiles have length 1, so a red tile looks more like the
true match than like the average other patient. The mean tile score divided by the length of the
mean tile is exactly the slide's margin, so the map breaks down why the slide matched.

The summary asks which tissue carries identity, using stain proxies computed from the tile pixels
(no annotations): tissue fraction, hematoxylin (nuclei) and eosin (stroma) density.
    python 09_heatmaps.py --model conch
    python 09_heatmaps.py --model conch --control dx+site --patients TCGA-05-4244 TCGA-38-4625
Writes results/<model>/heatmaps/<test>_<patient>.png, heatmaps_summary.png and heatmaps.json.
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from fingerprints import data, paths  # noqa: E402
from fingerprints.matching import CONTROLS, allowed_mask, ranks_of_truth  # noqa: E402

TESTS = {"test1": "Test 1: slide A tiles vs slide B", "test2": "Test 2: slide A tiles vs the report"}
# Diverging blue <-> red around a neutral gray: red tiles point toward the true match.
CMAP = LinearSegmentedColormap.from_list("identity", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#a32a2a"])
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
INK, MUTED, GRID, OUTLINE_LOW = "#0b0b0b", "#52514e", "#d9d8d4", "#8a8986"
# Ruifrok & Johnston stain vectors (as in skimage.color): optical density -> H, E, DAB amounts.
HED_FROM_RGB = np.linalg.inv(np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11],
                                       [0.27, 0.57, 0.78]])).astype(np.float32)
FEATURES = {"tissue": "tissue fraction (vs glass, air)",
            "hematoxylin": "hematoxylin (nuclei density)",
            "eosin": "eosin (stroma, cytoplasm)"}
TILE_PX, GAP = 96, 6  # tile size and spacing in the tile strips
MIN_BOX = 0.016  # smallest tile drawn on a map, as a fraction of the map's longer side


def candidates(e, test, control):
    """As in 05_tests.py: who takes part in the test, the rank of each one's true match, and which
    other patients are its candidates."""
    has = e.has_B if test == "test1" else e.has_R
    idx = np.flatnonzero(has)
    Q, C = (e.A[idx], e.B[idx]) if test == "test1" else (e.R[idx], e.A[idx])
    allowed = allowed_mask(e.patients.iloc[idx].reset_index(drop=True), control)
    ranks, n_cand = ranks_of_truth(Q, C, allowed)
    others = np.ones((len(idx), len(idx)), bool) if allowed is None else allowed
    np.fill_diagonal(others, False)
    return dict(idx=idx, pos={int(i): p for p, i in enumerate(idx)}, ranks=ranks, n_cand=n_cand,
                others=others, match=e.B if test == "test1" else e.R)


def pick(cands, available, n, seed):
    """Random patients among the rank-1 hits of every test first, then hits of any one test."""
    hits = [{int(c["idx"][p]) for p in np.flatnonzero((c["ranks"] == 1) & (c["n_cand"] >= 2))} & available
            for c in cands.values()]
    both = set.intersection(*hits)
    rng = np.random.default_rng(seed)
    return [int(i) for group in (both, set.union(*hits) - both) for i in rng.permutation(sorted(group))][:n]


def slide_files(model, sid):
    return (paths.emb_dir(model) / "tiles" / f"{sid}.npy", paths.tiles_dir() / f"{sid}.npz",
            paths.thumbs_dir() / f"{sid}.jpg", paths.thumbs_dir() / f"{sid}.txt")


def tile_features(pixels):
    """Stain proxies per tile, from pixels alone. High hematoxylin ~ cellular tissue (tumor, immune
    cells); high eosin ~ stroma; low tissue fraction ~ tissue edges, air spaces, glass."""
    od = -np.log((pixels[:, ::2, ::2].astype(np.float32) + 1) / 256)
    tissue = od.mean(-1) > 0.15
    hed = od @ HED_FROM_RGB
    n = np.maximum(tissue.sum((1, 2)), 1)
    return {"tissue": tissue.mean((1, 2)),
            "hematoxylin": (hed[..., 0] * tissue).sum((1, 2)) / n,
            "eosin": (hed[..., 1] * tissue).sum((1, 2)) / n}


def load_slide(model, sid):
    """Tile embeddings, tile boxes in thumbnail pixels, the thumbnail, and the tile pixels if the
    tiles npz still holds them."""
    f_emb, f_tiles, f_jpg, f_ds = slide_files(model, sid)
    X = np.load(f_emb).astype(np.float32)
    z = np.load(f_tiles)
    if len(X) == 0 or len(X) != len(z["coords"]):
        print(f"skip {sid}: {len(X)} tile embeddings but {len(z['coords'])} tile coordinates")
        return None
    ds = float(f_ds.read_text())
    pixels = z["tiles"] if "tiles" in z.files else None
    if pixels is not None and len(pixels) != len(X):
        pixels = None
    return dict(X=X, xy=z["coords"] / ds, side=max(float(z["tile_size0"]) / ds, 2.0),
                thumb=np.asarray(Image.open(f_jpg).convert("RGB")), pixels=pixels,
                features=tile_features(pixels) if pixels is not None else None)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else None


def finite(x):
    return float(x) if x is not None and math.isfinite(x) else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return finite(np.median(xs)) if xs else None


def tile_stats(scores, s):
    pos = np.sort(np.clip(scores, 0, None))[::-1]
    k = max(1, math.ceil(0.1 * len(scores)))
    st = dict(n_tiles=len(scores), mean_tile_score=float(scores.mean()),
              frac_positive=float((scores > 0).mean()),
              top10_share=float(pos[:k].sum() / pos.sum()) if pos.sum() > 0 else None)
    if s["features"] is not None:
        st["spearman"] = {f: spearman(scores, v) for f, v in s["features"].items()}
    return st


def montage(tiles):
    row = np.full((TILE_PX, len(tiles) * (TILE_PX + GAP) - GAP, 3), 255, np.uint8)
    for j, tile in enumerate(tiles):
        x = j * (TILE_PX + GAP)
        row[:, x:x + TILE_PX] = np.asarray(Image.fromarray(tile).convert("RGB").resize((TILE_PX, TILE_PX)))
    return row


def strip(ax, row, labels, title):
    ax.imshow(row)
    ax.set_xticks([j * (TILE_PX + GAP) + TILE_PX / 2 for j in range(len(labels))], labels,
                  fontsize=7, color=MUTED)
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_slide(dst, s, scores, title, k):
    th, tw = s["thumb"].shape[:2]
    xy, side = s["xy"], s["side"]
    # Crop to the tiled area so the tiles are big enough to see.
    pad = 0.05 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) + side
    x0, y0 = (max(int(v - pad), 0) for v in xy.min(0))
    x1 = min(int(xy[:, 0].max() + side + pad) + 1, tw)
    y1 = min(int(xy[:, 1].max() + side + pad) + 1, th)
    ext = (x0, x1, y1, y0)
    # On a 1024 px thumbnail a 20x tile is ~5 px wide, so draw small tiles enlarged around their center.
    box = max(side, MIN_BOX * max(x1 - x0, y1 - y0))
    corner = xy + (side - box) / 2
    heat = np.full((th, tw), np.nan, np.float32)
    for t in np.argsort(np.abs(scores)):  # strongest tiles painted last, on top
        x, y = np.maximum(corner[t], 0)
        heat[int(y):int(math.ceil(y + box)), int(x):int(math.ceil(x + box))] = scores[t]
    vmax = float(np.percentile(np.abs(scores), 98)) or 1e-6
    order = np.argsort(scores)
    k = min(k, len(scores) // 2)
    top, bottom = order[::-1][:k], order[:k]
    strips = s["pixels"] is not None and k > 0

    h_map = float(np.clip(5.3 * (y1 - y0) / (x1 - x0), 2.5, 7))  # each map panel is ~5.3 in wide
    fig = plt.figure(figsize=(12, h_map + (3.4 if strips else 0.9)))
    gs = fig.add_gridspec(3 if strips else 1, 3, width_ratios=[1, 1, 0.035], wspace=0.06, hspace=0.3,
                          height_ratios=[h_map, 1.15, 1.15] if strips else [1])
    crop = s["thumb"][y0:y1, x0:x1]
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax0.imshow(crop, extent=ext)
    ax1.imshow(crop, extent=ext)
    im = ax1.imshow(np.ma.masked_invalid(heat[y0:y1, x0:x1]), extent=ext, cmap=CMAP, vmin=-vmax, vmax=vmax,
                    alpha=0.85, interpolation="nearest")
    for idxs, color in ((top, INK), (bottom, OUTLINE_LOW)):
        for t in idxs:
            ax1.add_patch(Rectangle(corner[t], box, box, fill=False, ec=color, lw=1.4))
    ax0.set_title("slide A", fontsize=9, color=INK)
    notes = [f"{len(scores)} tiles"] + ([f"drawn {box / side:.1f}x their size"] if box > 1.05 * side else []) \
        + (["outlined: the strips below"] if strips else [])
    ax1.set_title(f"identity score per tile ({'; '.join(notes)})", fontsize=9, color=INK)
    for ax in (ax0, ax1):
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("tile · match − mean(tile · others)    (red = points to the true match)",
                 fontsize=8, color=MUTED)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    if strips:
        for r, (idxs, label) in enumerate(((top, "most identifying (black outline)"),
                                           (bottom, "least identifying (gray outline)")), 1):
            strip(fig.add_subplot(gs[r, :2]), montage(s["pixels"][idxs]),
                  [f"{scores[t]:+.3f}" for t in idxs], f"{len(idxs)} {label} tiles, with their scores")
    fig.suptitle(title, fontsize=10, color=INK, x=0.06, ha="left")
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)


def zscore(x):
    return (x - x.mean()) / (x.std() or 1)


def quartiles(slides):
    """Mean within-slide z-score of the tiles in each within-slide quartile of each feature. None for
    a feature that never varies within a slide (e.g. tissue fraction when every tile is all tissue)."""
    out = {}
    for f in FEATURES:
        if all(s["features"][f].std() == 0 for s, _ in slides):
            out[f] = None
            continue
        zs, bins = [], []
        for s, scores in slides:
            zs.append(zscore(scores))
            pct = pd.Series(s["features"][f]).rank(pct=True).to_numpy()
            bins.append(np.clip(np.ceil(pct * 4).astype(int) - 1, 0, 3))
        z, b = np.concatenate(zs), np.concatenate(bins)
        out[f] = {"mean": [finite(z[b == q].mean()) if (b == q).any() else None for q in range(4)],
                  "sem": [finite(z[b == q].std() / math.sqrt((b == q).sum())) if (b == q).sum() > 1 else None
                          for q in range(4)]}
    return out


def gallery(slides, n):
    """Round robin over slides: each one's best tile, then each one's second best, ..."""
    orders = [np.argsort(scores)[::-1] for _, scores in slides]
    picks = [(j, o[r]) for r in range(max(map(len, orders))) for j, o in enumerate(orders) if r < len(o)]
    return picks[:n]


def plot_summary(dst, model, control, res, n_gallery=10):
    fig = plt.figure(figsize=(18, 4.6 * len(res)))
    gs = fig.add_gridspec(len(res), 3, width_ratios=[1, 1.15, 2.3], wspace=0.3, hspace=0.55)
    for r, (t, rt) in enumerate(res.items()):
        stats = rt["patients"]
        ax = fig.add_subplot(gs[r, 0])
        pids = list(stats)[::-1]
        ax.barh(pids, [100 * stats[p]["frac_positive"] for p in pids], height=0.6, color=SERIES[0])
        ax.axvline(50, color=MUTED, ls="--", lw=1)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% of tiles pointing toward the true match (dashed: half)", fontsize=8, color=MUTED)
        ax.set_title(f"{TESTS[t]}\nhow spread out is identity?", loc="left", fontsize=9, color=INK)
        style(ax, "x")

        slides = [(s, sc) for s, sc in rt["slides"] if s["features"] is not None]
        ax = fig.add_subplot(gs[r, 1])
        if not slides:
            ax.text(0.5, 0.5, "tile pixels not found in $TF_WORK/tiles", ha="center", va="center",
                    transform=ax.transAxes, color=MUTED)
            ax.axis("off")
            continue
        for (f, label), color, marker in zip(FEATURES.items(), SERIES, "os^"):
            q = rt["quartiles"][f]
            if q is None:
                continue
            m = [np.nan if v is None else v for v in q["mean"]]
            se = [0 if v is None else v for v in q["sem"]]
            ax.errorbar([1, 2, 3, 4], m, yerr=se, color=color, marker=marker, ms=7, lw=2, capsize=3, label=label)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks([1, 2, 3, 4], ["lowest 25%", "", "", "highest 25%"], fontsize=8)
        ax.set_xlabel("tiles binned by feature, within each slide", fontsize=8, color=MUTED)
        ax.set_ylabel("identity score (z within slide)", fontsize=8, color=MUTED)
        n_tiles = sum(len(sc) for _, sc in slides)
        ax.set_title(f"which tissue carries it? ({n_tiles} tiles, {len(slides)} slides;\n"
                     "stain proxies from pixels, not annotations)", loc="left", fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False)
        style(ax, "y")

        sub = gs[r, 2].subgridspec(2, 1, hspace=0.45)
        for k, (sign, label) in enumerate(((1, "most"), (-1, "least"))):
            picks = gallery([(s, sign * sc) for s, sc in slides], n_gallery)
            strip(fig.add_subplot(sub[k]), montage([slides[j][0]["pixels"][t] for j, t in picks]),
                  [f"{zscore(slides[j][1])[t]:+.1f}z" for j, t in picks],
                  f"{label} identifying tiles, top of each slide in turn (z within slide)")
    fig.suptitle(f"{model}: where the identity signal sits ({control} candidates)", fontsize=12,
                 color=INK, x=0.07, ha="left")
    fig.savefig(dst, dpi=150, bbox_inches="tight")
    plt.close(fig)


def style(ax, grid):
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--n-patients", type=int, default=6)
    ap.add_argument("--patients", nargs="+", help="patient ids to draw instead of random rank-1 hits")
    ap.add_argument("--control", default="all", choices=CONTROLS,
                    help="candidate set: decides who counts as matched and who the 'others' are")
    ap.add_argument("--top-tiles", type=int, default=6, help="tiles per strip on each patient's figure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reports-name", default="reports")
    args = ap.parse_args()

    e = data.load(args.model, args.reports_name)
    cands = {t: candidates(e, t, args.control) for t, has in (("test1", e.has_B), ("test2", e.has_R))
             if has.sum() >= 2}
    if not cands:
        raise SystemExit("need slide B or a report for at least 2 patients")
    available = {i for i, sid in enumerate(e.patients.slide_a)
                 if all(f.exists() for f in slide_files(args.model, sid))}
    print(f"{len(available)}/{e.n} slide A have tile embeddings, tile coordinates and a thumbnail")
    if args.patients:
        ids = {p: i for i, p in enumerate(e.patients.patient_id)}
        chosen = [ids[p] for p in args.patients if ids.get(p) in available]
        for p in sorted(set(args.patients) - {e.patients.patient_id[i] for i in chosen}):
            print(f"skip {p}: not in patients.csv, or slide A lacks tiles / a thumbnail")
    else:
        chosen = pick(cands, available, args.n_patients, args.seed)
        if len(chosen) < args.n_patients:
            print(f"only {len(chosen)} rank-1 hits ({args.control} candidates) have tiles and a thumbnail")
    if not chosen:
        raise SystemExit("no patient to draw")

    hdir = paths.results_dir(args.model) / "heatmaps"
    hdir.mkdir(exist_ok=True)
    res = {t: {"patients": {}, "slides": []} for t in cands}
    for i in chosen:
        row = e.patients.iloc[i]
        s = load_slide(args.model, row.slide_a)
        if s is None:
            continue
        for t, c in cands.items():
            p = c["pos"].get(i)
            if p is None or not c["others"][p].any():
                continue
            w = c["match"][i] - c["match"][c["idx"][c["others"][p]]].mean(0)
            scores = s["X"] @ w
            m = s["X"].mean(0)
            st = dict(slide_a=row.slide_a, rank=int(c["ranks"][p]), n_candidates=int(c["n_cand"][p]),
                      slide_margin=float(e.A[i] @ w), margin_from_tiles=float(scores.mean() / np.linalg.norm(m)),
                      tile_mean_cos=float(m @ e.A[i] / np.linalg.norm(m)), **tile_stats(scores, s))
            if st["tile_mean_cos"] < 0.99:
                print(f"warning {row.slide_a}: slide embedding is not the mean of its tiles "
                      f"(cos {st['tile_mean_cos']:.3f}); rerun 03_embed_images.py")
            res[t]["patients"][row.patient_id] = st
            res[t]["slides"].append((s, scores))
            title = (f"{args.model}   {TESTS[t]}\n{row.patient_id} ({row.diagnosis}, site {row.site}): true match "
                     f"ranked {st['rank']} of {st['n_candidates']} ({args.control} candidates); slide margin "
                     f"{st['slide_margin']:+.3f}; {st['frac_positive']:.0%} of tiles point toward the match")
            plot_slide(hdir / f"{t}_{row.patient_id}.png", s, scores, title, args.top_tiles)

    out = {"model": args.model, "control": args.control, "tests": {}}
    for t, rt in res.items():
        stats = rt["patients"]
        if not stats:
            continue
        with_px = [(s, sc) for s, sc in rt["slides"] if s["features"] is not None]
        rt["quartiles"] = quartiles(with_px) if with_px else {}
        summ = {"median_frac_positive": median(v["frac_positive"] for v in stats.values()),
                "median_top10_share": median(v["top10_share"] for v in stats.values()),
                "median_spearman": {f: median(v.get("spearman", {}).get(f) for v in stats.values())
                                    for f in FEATURES},
                "quartile_mean_z": rt["quartiles"], "patients": stats}
        out["tests"][t] = summ
        print(f"\n{TESTS[t]}: {len(stats)} patients, median {summ['median_frac_positive']:.0%} of tiles point "
              f"toward the true match" + (f"; the top 10% of tiles carry {summ['median_top10_share']:.0%} of "
                                          f"the positive score" if summ["median_top10_share"] is not None else ""))
        if with_px:
            print("  within-slide Spearman (median over patients): " + ", ".join(
                f"{f} {v:+.2f}" for f, v in summ["median_spearman"].items() if v is not None))
    res = {t: rt for t, rt in res.items() if rt["patients"]}
    if not res:
        raise SystemExit("nothing drawn")
    plot_summary(paths.results_dir(args.model) / "heatmaps_summary.png", args.model, args.control, res)
    (paths.results_dir(args.model) / "heatmaps.json").write_text(json.dumps(out, indent=2))
    print(f"-> {hdir}/<test>_<patient>.png, {paths.results_dir(args.model)}/heatmaps_summary.png, heatmaps.json")


if __name__ == "__main__":
    main()
