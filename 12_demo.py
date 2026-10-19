"""Live demo: attack, then defend. A judge picks a patient; the tool searches the "anonymous" release of
frozen-section slide embeddings for that patient's FFPE slide. Then switch on each defense and run it again.

Uses the same embeddings and the same defense code as steps 06-07, so the numbers match the deck.

Two ways to run it:

  # A) Pack a small offline bundle on Delta (embeddings + metadata + small thumbnails, a few MB) ...
  source activate.sh main
  python 12_demo.py --export demo_bundle.npz --models pathgen quilt plip      # add conch musk once embedded
  # ... copy it to a laptop and present from there (no cluster, no VPN, no GPU needed):
  pip install gradio numpy pandas scipy scikit-learn pillow
  python 12_demo.py --bundle demo_bundle.npz

  # B) Or run straight from $TF_ROOT on Delta (Open OnDemand / port forward, or --share for a public link)
  python 12_demo.py --models pathgen quilt plip

Query = slide A (tumor FFPE). Candidates = slide B (tumor frozen, a different tissue block) of every patient.
Defense presets are in PRESETS below; edit them to match the settings you show on the defense slides.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints.data import Embeddings, normalize  # noqa: E402
from fingerprints.defenses import (add_noise, identity_directions, procrustes_attack,  # noqa: E402
                                   project_out, random_rotation, subtype_utility)
from fingerprints.matching import ranks_of_truth  # noqa: E402

# (label, kind, strength). Order = order in the dropdown, so it doubles as the demo script.
PRESETS = [
    ("No defense", "none", 0),
    ("Secret-key scrambling: attacker knows 0 pairs", "scramble", 0),
    ("Secret-key scrambling: attacker knows 50 pairs", "scramble", 50),
    ("Noise, sigma 0.5", "noise", 0.5),
    ("Noise, sigma 2 (blocks matching)", "noise", 2.0),
    ("Remove 100 identity directions", "remove", 100),
]
PRESET = {label: (kind, s) for label, kind, s in PRESETS}
SEED = 0
THUMB_PX = 320
SHOW_TOP = 5


# ---------------------------------------------------------------------------------- data

class Store:
    """Per-model embeddings for patients that have both slides, plus thumbnails as JPEG bytes."""

    def __init__(self, meta: pd.DataFrame, emb: dict[str, tuple[np.ndarray, np.ndarray]], thumbs: dict[str, bytes]):
        self.meta, self.emb, self.thumbs = meta.reset_index(drop=True), emb, thumbs
        self.models = list(emb)

    def embeddings(self, model: str) -> Embeddings:
        A, B = self.emb[model]
        n, d = A.shape
        R = np.full((n, d), np.nan, np.float32)
        return Embeddings(self.meta, A, B, R, np.ones(n, bool), np.zeros(n, bool))

    def thumb(self, slide_id: str):
        b = self.thumbs.get(slide_id)
        return Image.open(io.BytesIO(b)).convert("RGB") if b else None


def _jpeg(path: Path) -> bytes | None:
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    img.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def from_tf_root(models) -> Store:
    from fingerprints import data, paths

    emb, meta = {}, None
    for m in models:
        if not (paths.emb_dir(m) / "slides.npz").exists():
            print(f"skip {m}: no slides.npz")
            continue
        e = data.load(m)
        e = e.subset(e.has_B)
        if meta is None:
            meta = e.patients
        # Keep patients present for every model, in one order.
        meta = meta[meta.patient_id.isin(e.patients.patient_id)]
        emb[m] = (e.patients.patient_id.to_numpy(), e.A, e.B)
    if not emb:
        sys.exit("no model has slide embeddings yet")
    meta = meta.reset_index(drop=True)
    aligned = {}
    for m, (pids, A, B) in emb.items():
        idx = pd.Series(np.arange(len(pids)), index=pids)[meta.patient_id].to_numpy()
        aligned[m] = (normalize(A[idx]), normalize(B[idx]))
    thumbs = {}
    for sid in pd.concat([meta.slide_a, meta.slide_b]):
        b = _jpeg(paths.thumbs_dir() / f"{sid}.jpg")
        if b:
            thumbs[sid] = b
    print(f"{len(meta)} patients with both slides, models {list(aligned)}, {len(thumbs)} thumbnails")
    return Store(meta, aligned, thumbs)


def export(store: Store, out: str) -> None:
    keys = list(store.thumbs)
    blobs = [np.frombuffer(store.thumbs[k], np.uint8) for k in keys]
    offsets = np.cumsum([0] + [len(b) for b in blobs])
    arrays = {f"A__{m}": a.astype(np.float16) for m, (a, _) in store.emb.items()}
    arrays |= {f"B__{m}": b.astype(np.float16) for m, (_, b) in store.emb.items()}
    np.savez_compressed(
        out, meta_json=np.array(store.meta.to_json(orient="records")), models=np.array(store.models),
        thumb_keys=np.array(keys), thumb_offsets=offsets,
        thumb_bytes=np.concatenate(blobs) if blobs else np.zeros(0, np.uint8), **arrays)
    print(f"-> {out} ({Path(out).stat().st_size / 1e6:.1f} MB)")


def from_bundle(path: str) -> Store:
    z = np.load(path)
    meta = pd.DataFrame(json.loads(str(z["meta_json"])))
    for c in ("site", "patient_id", "diagnosis", "slide_a", "slide_b"):
        meta[c] = meta[c].astype(str)
    emb = {m: (normalize(z[f"A__{m}"].astype(np.float32)), normalize(z[f"B__{m}"].astype(np.float32)))
           for m in z["models"].tolist()}
    off, raw = z["thumb_offsets"], z["thumb_bytes"]
    thumbs = {k: raw[off[i]:off[i + 1]].tobytes() for i, k in enumerate(z["thumb_keys"].tolist())}
    return Store(meta, emb, thumbs)


# ------------------------------------------------------------------------------ defenses

def halves(dx: np.ndarray) -> np.ndarray:
    """Same diagnosis-balanced split as 07_remove.py."""
    rng = np.random.default_rng(SEED)
    fold = np.zeros(len(dx), int)
    for g in np.unique(dx):
        idx = np.flatnonzero(dx == g)
        rng.shuffle(idx)
        fold[idx[1::2]] = 1
    return fold


class Release:
    """One defended release: what the attacker sees. Q = queries (slide A), C = candidates (slide B).
    known = patients whose pairs the attacker already has (scrambling only); they are not searched."""

    def __init__(self, e: Embeddings, kind: str, strength: float):
        n = e.n
        dx = e.patients.diagnosis.astype(str).to_numpy()
        self.known = np.zeros(n, bool)
        rng = np.random.default_rng(SEED)
        if kind == "none":
            Q, C, shared = e.A, e.B, e.A
        elif kind == "noise":
            Q, C = add_noise(e.A, strength, rng), add_noise(e.B, strength, rng)
            shared = Q
        elif kind == "remove":
            # Cross-fitted, like 07_remove.py: each half is cleaned with directions fit on the other half.
            fold = halves(dx)
            Q, C = e.A.copy(), e.B.copy()
            for h in (0, 1):
                U = identity_directions(e.subset(fold != h), int(strength), "mean")
                Q[fold == h], C[fold == h] = project_out(e.A[fold == h], U), project_out(e.B[fold == h], U)
            shared = Q
        elif kind == "scramble":
            # Each release gets its own secret rotation. The attacker aligns them with k known pairs.
            QA, CB = e.A @ random_rotation(e.d, rng), e.B @ random_rotation(e.d, rng)
            k = int(min(strength, n - 2))
            if k:
                self.known[rng.choice(n, k, replace=False)] = True
            Q, C = procrustes_attack(QA, CB, np.flatnonzero(self.known))
            shared = QA  # what the hospital's own users get: a rotated but intact release
        else:
            raise ValueError(kind)
        self.Q, self.C = normalize(Q), normalize(C)
        self.search = ~self.known
        idx = np.flatnonzero(self.search)
        ranks, n_cand = ranks_of_truth(self.Q[idx], self.C[idx])
        self.top1 = float((ranks == 1).mean())
        self.top10 = float((ranks <= 10).mean())
        self.chance1 = 1 / len(idx)
        self.chance10 = min(10, len(idx)) / len(idx)
        self.utility = subtype_utility(shared, dx)

    def search_for(self, i: int):
        """Rank slide B of every searchable patient by similarity to patient i's slide A."""
        idx = np.flatnonzero(self.search)
        if i not in idx:
            idx = np.append(idx, i)
        sims = self.C[idx] @ self.Q[i]
        order = idx[np.argsort(-sims)]
        rank = int(np.flatnonzero(order == i)[0]) + 1
        return order, sims[np.argsort(-sims)], rank, len(idx)


# ------------------------------------------------------------------------------------ app

def build_app(store: Store):
    import gradio as gr

    meta = store.meta

    @lru_cache(maxsize=64)
    def release(model: str, preset: str) -> Release:
        return Release(store.embeddings(model), *PRESET[preset])

    @lru_cache(maxsize=16)
    def safe_picks(model: str) -> list[int]:
        """Patients the undefended attack finds at rank 1: dependable for a live demo."""
        r = release(model, PRESETS[0][0])
        return [i for i in range(len(meta)) if r.search_for(i)[2] == 1]

    def label(i: int) -> str:
        p = meta.iloc[i]
        return f"{p.patient_id} | {p.diagnosis} | hospital {p.site}"

    labels = [label(i) for i in range(len(meta))]
    index_of = {l: i for i, l in enumerate(labels)}

    def run(model, patient, preset):
        if not patient:
            return None, "Pick a patient first.", [], ""
        i = index_of[patient]
        r = release(model, preset)
        order, sims, rank, n_cand = r.search_for(i)
        p = meta.iloc[i]
        if rank == 1:
            verdict = f"## Found. The true patient is **#1 of {n_cand}**."
        elif rank <= 10:
            verdict = f"## Shortlisted. The true patient is **#{rank} of {n_cand}**."
        else:
            verdict = f"## Lost. The true patient is **#{rank} of {n_cand}**."
        verdict += f"\nRandom guessing finds them 1 time in {n_cand}."
        gallery = []
        for pos, (j, s) in enumerate(zip(order[:SHOW_TOP], sims[:SHOW_TOP]), 1):
            q = meta.iloc[j]
            tag = "SAME PATIENT" if j == i else "different patient"
            gallery.append((store.thumb(q.slide_b), f"#{pos} {q.patient_id} ({tag}), similarity {s:.3f}"))
        if rank > SHOW_TOP:
            gallery.append((store.thumb(p.slide_b), f"True match: #{rank} {p.patient_id}"))
        base = release(model, PRESETS[0][0])
        stats = (
            f"**This release, all {int(r.search.sum())} searchable patients** ({model})\n\n"
            f"| | Found at #1 | In top 10 | LUAD vs LUSC accuracy |\n|---|---|---|---|\n"
            f"| No defense | {base.top1:.1%} ({base.top1 / base.chance1:.0f}x chance) | {base.top10:.0%} | "
            f"{base.utility:.2f} |\n"
            f"| **{preset}** | **{r.top1:.1%} ({r.top1 / r.chance1:.1f}x chance)** | **{r.top10:.0%}** | "
            f"**{r.utility:.2f}** |\n"
            f"| Random guessing | {r.chance1:.1%} | {r.chance10:.0%} | 0.50 |"
        )
        if r.known.any():
            stats += f"\n\nThe attacker already knows {int(r.known.sum())} pairs; those patients are not searched."
        return store.thumb(p.slide_a), verdict, gallery, stats

    def random_patient(model):
        picks = safe_picks(model) or list(range(len(meta)))
        return labels[int(np.random.default_rng().choice(picks))]

    with gr.Blocks(title="Tissue Fingerprints") as app:
        gr.Markdown("# Tissue Fingerprints\nPick a patient's FFPE slide. We search the *anonymous* release of "
                    "frozen-section embeddings (a different tissue block) for that same person.")
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(store.models, value=store.models[0], label="Foundation model")
                patient = gr.Dropdown(labels, label="Query patient (slide A, FFPE)")
                pick = gr.Button("Pick a demo-safe patient")
                preset = gr.Dropdown([p[0] for p in PRESETS], value=PRESETS[0][0], label="Defense on the release")
                go = gr.Button("Run the attack", variant="primary")
                query_img = gr.Image(label="Query slide", height=260)
            with gr.Column(scale=2):
                verdict = gr.Markdown()
                gallery = gr.Gallery(label=f"Top {SHOW_TOP} matches in the anonymous release (slide B)",
                                     columns=3, height=420)
                stats = gr.Markdown()
        pick.click(random_patient, model, patient)
        for trigger in (go.click, preset.change, model.change):
            trigger(run, [model, patient, preset], [query_img, verdict, gallery, stats])
    return app, release


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", help="run from a bundle made with --export (e.g. on a laptop)")
    ap.add_argument("--export", help="write a bundle to this path instead of launching")
    ap.add_argument("--models", nargs="+", default=["conch", "musk", "pathgen", "quilt", "plip"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="public gradio.live link (needs internet)")
    ap.add_argument("--warm", action="store_true", help="precompute every preset before launching")
    args = ap.parse_args()

    store = from_bundle(args.bundle) if args.bundle else from_tf_root(args.models)
    if args.export:
        export(store, args.export)
        return
    app, release = build_app(store)
    if args.warm:
        for m in store.models:
            for p in PRESETS:
                r = release(m, p[0])
                print(f"{m:8s} {p[0]:50s} top-1 {r.top1:.1%} ({r.top1 / r.chance1:.1f}x)  "
                      f"top-10 {r.top10:.0%}  utility {r.utility:.2f}", flush=True)
    app.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
