"""Step 4: embed every tile with one model; slide embedding = mean of tiles, then length 1.

Writes emb/<model>/tiles/<slide_id>.npy (float16, one row per tile, same order as the tile
coords, for heatmaps) and emb/<model>/slides.npz (slide_ids, X). Resumes where it stopped.

    python 03_embed_images.py --model conch      # tf-main env
    python 03_embed_images.py --model musk       # tf-musk env
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from fingerprints import paths
from fingerprints.models import load_encoder


class SlideTiles(torch.utils.data.Dataset):
    """One item = all tiles of one slide, preprocessed in DataLoader workers (CPU)."""

    def __init__(self, slide_ids, transform):
        self.ids, self.transform = slide_ids, transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        tiles = np.load(paths.tiles_dir() / f"{sid}.npz")["tiles"]
        if len(tiles) == 0:
            return sid, None
        return sid, torch.stack([self.transform(Image.fromarray(t)) for t in tiles])


def first(batch):
    return batch[0]


def aggregate(model: str, slide_ids) -> None:
    ids, X = [], []
    for sid in slide_ids:
        f = paths.emb_dir(model) / "tiles" / f"{sid}.npy"
        if f.exists():
            m = np.load(f).astype(np.float32).mean(0)
            ids.append(sid)
            X.append(m / np.linalg.norm(m))
    if not ids:
        print("no slide embeddings yet")
        return
    out = paths.emb_dir(model) / "slides.npz"
    paths.save_atomic(out, lambda fh: np.savez(fh, slide_ids=np.array(ids), X=np.stack(X)))
    print(f"{len(ids)}/{len(slide_ids)} slides -> {out}  (dim {X[0].shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    all_ids = pd.read_csv(paths.slides_csv()).slide_id.tolist()
    have_tiles = [s for s in all_ids if (paths.tiles_dir() / f"{s}.npz").exists()]
    todo = [s for s in have_tiles if not (paths.emb_dir(args.model) / "tiles" / f"{s}.npy").exists()]
    print(f"{len(all_ids)} slides, {len(have_tiles)} tiled, {len(todo)} to embed with {args.model}", flush=True)

    if todo:
        enc = load_encoder(args.model, args.device)
        loader = torch.utils.data.DataLoader(SlideTiles(todo, enc.transform), batch_size=1,
                                             collate_fn=first, num_workers=args.workers, prefetch_factor=2)
        t0 = time.time()
        for i, (sid, x) in enumerate(loader, 1):
            if x is None:
                print(f"skip {sid}: no tissue tiles")
                continue
            feats = np.concatenate([enc.embed_images(x[j:j + args.batch_size])
                                    for j in range(0, len(x), args.batch_size)])
            dst = paths.emb_dir(args.model) / "tiles" / f"{sid}.npy"
            paths.save_atomic(dst, lambda fh: np.save(fh, feats.astype(np.float16)))
            if i % 20 == 0 or i == len(todo):
                rate = i / (time.time() - t0)
                print(f"[{i}/{len(todo)}] {rate:.2f} slides/s, ~{(len(todo) - i) / rate / 60:.0f} min left", flush=True)

    aggregate(args.model, all_ids)


if __name__ == "__main__":
    main()
