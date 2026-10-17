"""Step 1: load each model and embed 5 tiles + 2 sentences, to prove weights and envs work.

    source activate.sh main; python 00_check_models.py --models conch pathgen
    source activate.sh musk; python 00_check_models.py --models musk
"""
import argparse
import sys
import time

import numpy as np

from fingerprints import paths
from fingerprints.models import load_encoder


def sample_tiles(n=5):
    """Real tiles if step 3 has run already, otherwise pink-ish noise (enough to test the plumbing)."""
    tiles = sorted(paths.tiles_dir().glob("*.npz")) if paths.tiles_dir().exists() else []
    if tiles:
        print(f"using tiles from {tiles[0].name}")
        return np.load(tiles[0])["tiles"][:n]
    rng = np.random.default_rng(0)
    base = np.array([200, 120, 170], dtype=np.float32)
    return np.clip(base + rng.normal(0, 30, (n, 224, 224, 3)), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(paths.MODELS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    tiles = sample_tiles()
    texts = ["lung adenocarcinoma with acinar growth pattern", "normal lung parenchyma"]
    failed = {}
    for name in args.models:
        t0 = time.time()
        try:  # one missing model (e.g. gated access still pending) must not hide the others
            enc = load_encoder(name, args.device)
            img = enc.embed_tiles(tiles)
            txt = enc.embed_texts(texts)
            assert img.shape[1] == txt.shape[1], "image and text must share one space"
        except Exception as e:
            failed[name] = repr(e)
            print(f"[{name}] FAILED: {e!r}", flush=True)
            continue
        print(f"[{name}] device={enc.device} load+embed {time.time() - t0:.1f}s")
        print(f"  image {img.shape}, norms {np.linalg.norm(img, axis=1).round(3)}")
        print(f"  text  {txt.shape}, norms {np.linalg.norm(txt, axis=1).round(3)}")
        print(f"  image-text cosine:\n{np.round(img @ txt.T, 3)}", flush=True)
    ok = [m for m in args.models if m not in failed]
    print(f"OK: {ok}  FAILED: {list(failed)}")
    sys.exit(1 if failed and not ok else 0)


if __name__ == "__main__":
    main()
