"""Step 3: cut 200 random 224px tissue tiles at ~20x from every slide in data/slides.csv.

Writes $TF_WORK/tiles/<slide_id>.npz (tiles uint8 + level-0 coords for heatmaps) and a
thumbnail per slide in $TF_ROOT/thumbs/. Already-tiled slides are skipped, so re-running resumes.

    python 02_tile.py --workers 16                       # all slides
    python 02_tile.py --shard 0 --num-shards 4 ...       # one Slurm array task (slurm/tile.sbatch)
"""
import argparse
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

from fingerprints import paths
from fingerprints.tiling import tile_slide


def tile_one(job):
    slide_id, path, opts = job
    out = paths.tiles_dir() / f"{slide_id}.npz"
    if out.exists():
        return slide_id, -1, ""
    try:
        import openslide

        slide = openslide.OpenSlide(path)
        seed = opts["seed"] + zlib.crc32(slide_id.encode())  # stable per slide, differs across slides
        r = tile_slide(slide, n_tiles=opts["n_tiles"], tile_px=opts["tile_px"],
                       target_mag=opts["mag"], min_tissue=opts["min_tissue"], seed=seed)
        slide.close()
        thumb = r.pop("thumb")
        paths.save_atomic(out, lambda f: np.savez(f, **r))
        # Smaller thumbnail for heatmaps; its downsample is relative to level 0.
        img = Image.fromarray(thumb)
        img.thumbnail((1024, 1024))
        paths.thumbs_dir().mkdir(parents=True, exist_ok=True)
        img.save(paths.thumbs_dir() / f"{slide_id}.jpg", quality=90)
        ds = r["thumb_downsample"] * thumb.shape[1] / img.width
        (paths.thumbs_dir() / f"{slide_id}.txt").write_text(f"{ds}\n")
        note = "" if len(r["tiles"]) >= opts["n_tiles"] else f"only {len(r['tiles'])} tissue tiles"
        return slide_id, len(r["tiles"]), note
    except Exception as e:  # one bad slide must not kill the job
        return slide_id, 0, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tiles", type=int, default=200)
    ap.add_argument("--tile-px", type=int, default=224)
    ap.add_argument("--mag", type=float, default=20.0)
    ap.add_argument("--min-tissue", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    slides = pd.read_csv(paths.slides_csv()).sort_values("slide_id")
    slides = slides.iloc[args.shard::args.num_shards]
    opts = dict(n_tiles=args.n_tiles, tile_px=args.tile_px, mag=args.mag,
                min_tissue=args.min_tissue, seed=args.seed)
    jobs = [(r.slide_id, r.path, opts) for r in slides.itertuples()]
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} slides, {args.workers} workers", flush=True)

    log, t0 = [], time.time()
    with Pool(args.workers) as pool:
        for i, (sid, n, note) in enumerate(pool.imap_unordered(tile_one, jobs), 1):
            log.append(dict(slide_id=sid, n_tiles=n, note=note))
            if note or i % 20 == 0:
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {sid}: {n} {note}", flush=True)

    log = pd.DataFrame(log)
    log.to_csv(paths.tiles_dir() / f"tile_log_shard{args.shard}.csv", index=False)
    done = log[log.n_tiles != 0]
    print(f"done: {len(done)} ok ({(log.n_tiles == -1).sum()} already tiled), "
          f"{(log.n_tiles == 0).sum()} failed, {log.note.str.startswith('only').sum()} short")


if __name__ == "__main__":
    main()
