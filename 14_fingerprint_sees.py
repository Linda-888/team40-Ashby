"""Feature 1: what does the fingerprint see?

Part A  (labels, quick, CPU)
    For patients whose slide A finds their own slide B at rank 1 (with --id-model, default MUSK), score
    every tile of slide A as in 09_heatmaps.py:
        score_t = tile_t . B_true - mean_j(tile_t . B_other_j)
    The top tiles are the "identity tiles". PLIP labels every tile zero-shot (tumor, stroma, normal
    lung, immune cells, necrosis, blood/mucus) from its existing PLIP tile embedding and text prompts.
    Output: tissue mix of identity tiles vs all tiles, and a grid of real identity tiles with labels.

Part B  (stain, heavier: run as a CPU batch job, slurm/stain_cpu.sbatch)
    Re-embed the same tiles with PLIP twice: as they are, and after Macenko stain normalization
    (every tile mapped to one reference H&E colour). Re-run Test 1 on both. If matching survives the
    colour cleaning, the fingerprint is not just staining.

    python 14_fingerprint_sees.py labels --out ~/fingerprint_sees
    python 14_fingerprint_sees.py stain  --out ~/fingerprint_sees --tiles-per-slide 100 --threads 64
Needs: numpy pandas scipy scikit-learn pillow matplotlib torch transformers (CPU is fine).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import data, paths  # noqa: E402
from fingerprints.data import normalize  # noqa: E402
from fingerprints.matching import allowed_mask, fmt, ranks_of_truth, summarize  # noqa: E402

LABEL_MODEL = "plip"
PROMPTS = {
    "tumor": ["an H&E image of lung adenocarcinoma", "an H&E image of squamous cell carcinoma",
              "an H&E image of tumor cells", "an H&E image of malignant epithelium"],
    "stroma": ["an H&E image of tumor stroma", "an H&E image of fibrous connective tissue",
               "an H&E image of desmoplastic stroma"],
    "normal lung": ["an H&E image of normal lung alveoli", "an H&E image of benign lung parenchyma"],
    "immune cells": ["an H&E image of lymphocytes", "an H&E image of inflammatory cells"],
    "necrosis": ["an H&E image of necrosis", "an H&E image of necrotic tumor"],
    "blood / mucus": ["an H&E image of blood", "an H&E image of red blood cells", "an H&E image of mucus"],
}
COLORS = ["#c0392b", "#2471a3", "#1e8449", "#7d3c98", "#7f8c8d", "#d68910"]


# ------------------------------------------------------------------------------ PLIP

class Plip:
    def __init__(self, threads: int):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(threads)
        self.torch = torch
        self.model = CLIPModel.from_pretrained("vinid/plip").eval()
        self.proc = CLIPProcessor.from_pretrained("vinid/plip")

    def texts(self, texts):
        with self.torch.no_grad():
            tok = self.proc.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt")
            return normalize(self.model.get_text_features(**tok).numpy())

    def images(self, tiles, batch=64):
        out = []
        with self.torch.no_grad():
            for i in range(0, len(tiles), batch):
                px = self.proc.image_processor(images=list(tiles[i:i + batch]), return_tensors="pt")["pixel_values"]
                out.append(self.model.get_image_features(pixel_values=px).numpy())
        return normalize(np.concatenate(out))


def class_prototypes(plip):
    names = list(PROMPTS)
    protos = np.stack([normalize(plip.texts(PROMPTS[c]).mean(0, keepdims=True))[0] for c in names])
    return names, protos


# ------------------------------------------------------------------------------ Macenko

HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAXC_REF = np.array([1.9705, 1.0308])


def macenko(img: np.ndarray, Io=240, alpha=1, beta=0.15):
    """Macenko et al. 2009 stain normalization of one uint8 RGB tile to a fixed reference.
    Returns (normalized tile, ok). Tiles with too little stained tissue come back unchanged."""
    shape = img.shape
    od = -np.log((img.reshape(-1, 3).astype(np.float64) + 1) / Io)
    odhat = od[~np.any(od < beta, axis=1)]
    if len(odhat) < 500:
        return img, False
    _, vecs = np.linalg.eigh(np.cov(odhat.T))
    plane = vecs[:, 1:3]
    proj = odhat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
    v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
    he = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    maxc = np.array([np.percentile(conc[0], 99), np.percentile(conc[1], 99)])
    if not np.all(np.isfinite(maxc)) or np.any(maxc <= 1e-6):
        return img, False
    conc = conc / maxc[:, None] * MAXC_REF[:, None]
    out = Io * np.exp(-HE_REF @ conc)
    return np.clip(out, 0, 255).T.reshape(shape).astype(np.uint8), True


# ------------------------------------------------------------------------------ helpers

def tile_emb(model, sid):
    f = paths.emb_dir(model) / "tiles" / f"{sid}.npy"
    return normalize(np.load(f).astype(np.float32)) if f.exists() else None


def tile_pixels(sid):
    f = paths.tiles_dir() / f"{sid}.npz"
    if not f.exists():
        return None
    z = np.load(f)
    return z["tiles"] if "tiles" in z.files else None


def rank1_patients(e, min_patients):
    m = e.has_B
    idx = np.flatnonzero(m)
    ranks, _ = ranks_of_truth(e.A[idx], e.B[idx])
    order = np.argsort(ranks, kind="stable")
    chosen = [i for i in order if ranks[i] == 1]
    if len(chosen) < min_patients:
        chosen = list(order[:min_patients])
    return idx, ranks, [int(i) for i in chosen]


# ------------------------------------------------------------------------------ Part A

def part_labels(args):
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    e = data.load(args.id_model)
    idx, ranks, chosen = rank1_patients(e, args.min_patients)
    B = e.B[idx]
    plip = Plip(args.threads)
    names, protos = class_prototypes(plip)

    counts = {k: np.zeros(len(names)) for k in ("identity", "all", "least")}
    per_patient, gallery = {}, []
    for c in chosen:
        p = e.patients.iloc[idx[c]]
        X = tile_emb(args.id_model, p.slide_a)
        L = tile_emb(LABEL_MODEL, p.slide_a)
        if X is None or L is None or len(X) != len(L):
            print(f"skip {p.patient_id}: tile embeddings missing or misaligned")
            continue
        sims = X @ normalize(B).T  # (tiles, patients)
        others = np.delete(np.arange(len(B)), c)
        score = sims[:, c] - sims[:, others].mean(1)
        lab = np.argmax(L @ protos.T, 1)
        top = np.argsort(score)[::-1][:args.top_tiles]
        low = np.argsort(score)[:args.top_tiles]
        counts["identity"] += np.bincount(lab[top], minlength=len(names))
        counts["all"] += np.bincount(lab, minlength=len(names))
        counts["least"] += np.bincount(lab[low], minlength=len(names))
        per_patient[p.patient_id] = dict(rank=int(ranks[c]), diagnosis=str(p.diagnosis), site=str(p.site),
                                         identity_tiles={names[i]: int((lab[top] == i).sum()) for i in range(len(names))})
        px = tile_pixels(p.slide_a)
        if px is not None and len(px) == len(X):
            gallery.append((p.patient_id, [(px[t], names[lab[t]]) for t in top[:args.gallery_tiles]]))

    if not per_patient:
        sys.exit("no patients had aligned tile embeddings")
    mix = {k: {names[i]: float(v[i] / v.sum()) for i in range(len(names))} for k, v in counts.items()}
    res = dict(id_model=args.id_model, label_model=LABEL_MODEL, n_patients=len(per_patient),
               top_tiles=args.top_tiles, prompts=PROMPTS, tissue_mix=mix, patients=per_patient)
    (out / "labels.json").write_text(json.dumps(res, indent=2))

    print(f"\n{len(per_patient)} patients (identity model {args.id_model}, labels by {LABEL_MODEL} zero-shot)")
    print(f"{'tissue':14s} {'identity tiles':>15s} {'all tiles':>10s} {'least-identity':>15s}")
    for n in names:
        print(f"{n:14s} {mix['identity'][n]:15.0%} {mix['all'][n]:10.0%} {mix['least'][n]:15.0%}")
    plot_labels(mix, names, gallery, args, out, len(per_patient))


def plot_labels(mix, names, gallery, args, out, n_patients):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.2))
    groups = [("identity", f"top {args.top_tiles} identity tiles"), ("all", "all tiles"),
              ("least", f"bottom {args.top_tiles} tiles")]
    left = np.zeros(len(groups))
    for i, n in enumerate(names):
        vals = np.array([mix[g][n] for g, _ in groups])
        ax.barh(range(len(groups)), vals, left=left, color=COLORS[i % len(COLORS)], label=n)
        for y, (v, l0) in enumerate(zip(vals, left)):
            if v >= 0.06:
                ax.text(l0 + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(range(len(groups)), [g[1] for g in groups])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of tiles (PLIP zero-shot label)")
    ax.set_title(f"What tissue carries identity? ({args.id_model}, {n_patients} patients)", fontsize=10)
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "tissue_mix.png", dpi=170)
    plt.close(fig)

    rows = gallery[:args.gallery_patients]
    if rows:
        k = max(len(r[1]) for r in rows)
        fig, axes = plt.subplots(len(rows), k, figsize=(1.7 * k, 1.95 * len(rows)), squeeze=False)
        for r, (pid, tiles) in enumerate(rows):
            for j in range(k):
                ax = axes[r, j]
                ax.axis("off")
                if j < len(tiles):
                    ax.imshow(tiles[j][0])
                    ax.set_title(tiles[j][1], fontsize=7)
            axes[r, 0].text(-0.08, 0.5, pid, transform=axes[r, 0].transAxes, rotation=90,
                            ha="right", va="center", fontsize=7)
        fig.suptitle("Tiles that gave each patient away (label = PLIP estimate)", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / "identity_tiles.png", dpi=170)
        plt.close(fig)
    print(f"-> {out / 'labels.json'}, tissue_mix.png, identity_tiles.png")


# ------------------------------------------------------------------------------ Part B

def part_stain(args):
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    pats = pd.read_csv(paths.patients_csv(), dtype={"site": str})
    if args.max_patients:
        pats = pats.iloc[:args.max_patients]
    plip = Plip(args.threads)
    rng = np.random.default_rng(0)
    raw, norm, keep, n_ok, n_all = [], [], [], 0, 0
    t0 = time.time()
    for i, p in enumerate(pats.itertuples(), 1):
        embs = []
        for sid in (p.slide_a, p.slide_b):
            px = tile_pixels(sid)
            if px is None or len(px) == 0:
                break
            sel = px[np.sort(rng.choice(len(px), min(args.tiles_per_slide, len(px)), replace=False))]
            normed = []
            for t in sel:
                nt, ok = macenko(t)
                normed.append(nt)
                n_ok += ok
                n_all += 1
            embs.append((normalize(plip.images(sel).mean(0, keepdims=True))[0],
                         normalize(plip.images(np.stack(normed)).mean(0, keepdims=True))[0]))
        if len(embs) == 2:
            raw.append((embs[0][0], embs[1][0]))
            norm.append((embs[0][1], embs[1][1]))
            keep.append(i - 1)
        if i % 10 == 0:
            print(f"[{i}/{len(pats)} {time.time() - t0:.0f}s] Macenko ok on {n_ok}/{n_all} tiles", flush=True)
        if i == 1 and args.save_example:
            sel_show = sel[:6]
            np.save(out / "example_tiles.npy", np.stack([sel_show, np.stack([macenko(t)[0] for t in sel_show])]))

    meta = pats.iloc[keep].reset_index(drop=True)
    res = dict(label_model=LABEL_MODEL, n_patients=len(keep), tiles_per_slide=args.tiles_per_slide,
               macenko_ok=n_ok / max(n_all, 1), tests={})
    print(f"\n{len(keep)} patients, {args.tiles_per_slide} tiles per slide, PLIP")
    for name, pairs in (("as is", raw), ("stain normalized", norm)):
        A = np.stack([a for a, _ in pairs])
        B = np.stack([b for _, b in pairs])
        res["tests"][name] = {}
        for control in ("all", "dx", "dx+site"):
            s = summarize(*ranks_of_truth(A, B, allowed_mask(meta, control)))
            res["tests"][name][control] = s
            print(f"  {name:17s} {control:8s} {fmt(s)}")
    (out / "stain.json").write_text(json.dumps(res, indent=2))
    plot_stain(res, out)
    print(f"-> {out / 'stain.json'}, stain.png")


def plot_stain(res, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"all": "all patients", "dx": "same diagnosis", "dx+site": "same diagnosis + hospital"}
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.38
    for j, (name, color) in enumerate((("as is", "#7f8c8d"), ("stain normalized", "#2471a3"))):
        lifts = [res["tests"][name][c]["lift_top1"] for c in labels]
        bars = ax.bar(np.arange(3) + (j - 0.5) * w, lifts, w, color=color, label=name)
        for b, v in zip(bars, lifts):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}x", ha="center", va="bottom", fontsize=8)
    ax.axhline(1, color="black", lw=0.8, ls=":")
    ax.text(2.45, 1.05, "chance", fontsize=7, ha="right")
    ax.set_xticks(range(3), list(labels.values()), fontsize=8)
    ax.set_ylabel("slide finds same patient (x chance)")
    ax.set_title("Does the fingerprint survive stain cleaning? (PLIP)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "stain.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="part", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--id-model", default="musk", choices=paths.MODELS)
    a.add_argument("--top-tiles", type=int, default=10)
    a.add_argument("--min-patients", type=int, default=10)
    a.add_argument("--gallery-patients", type=int, default=6)
    a.add_argument("--gallery-tiles", type=int, default=5)
    b = sub.add_parser("stain")
    b.add_argument("--tiles-per-slide", type=int, default=100)
    b.add_argument("--max-patients", type=int, default=0)
    b.add_argument("--save-example", action="store_true", default=True)
    for p in (a, b):
        p.add_argument("--out", default="~/fingerprint_sees")
        p.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    part_labels(args) if args.part == "labels" else part_stain(args)


if __name__ == "__main__":
    main()
