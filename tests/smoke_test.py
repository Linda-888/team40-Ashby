"""End-to-end check without GPUs, slides or model weights: barcodes, pair choice, chunking, tiling on
a fake slide, then steps 05-09 on synthetic embeddings with a planted patient fingerprint.

    python tests/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fingerprints import barcodes  # noqa: E402
from fingerprints.text import chunk_text  # noqa: E402
from fingerprints.tiling import tile_slide  # noqa: E402


def check_barcodes():
    bc = barcodes.parse("TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01.svs")
    assert (bc.patient, bc.site, bc.sample_type, bc.kind, bc.number) == ("TCGA-05-4244", "05", "01", "DX", 1)
    ts = barcodes.parse("TCGA-05-4244-01A-01-TS1.svs")
    bs = barcodes.parse("TCGA-05-4244-01A-01-BS1.svs")
    assert ts.specimen == bs.specimen, "top/bottom of one frozen piece are the same specimen"
    assert ts.specimen != bc.specimen
    assert barcodes.project_from_path("/data/TCGA-LUAD/abc/TCGA-05-4244-01Z-00-DX1.svs") == "TCGA-LUAD"
    assert barcodes.patient_id("tcga-bp-5195.25c0b433.pdf") == "TCGA-BP-5195"
    print("barcodes ok")


def check_chunks():
    text = "Lung, left upper lobe: invasive adenocarcinoma. " * 30
    chunks = chunk_text(text, 20)
    assert all(len(c.split()) <= 20 for c in chunks) and len(" ".join(chunks).split()) == len(text.split())
    assert chunk_text(" ".join(["word"] * 50), 20) == [" ".join(["word"] * 20)] * 2 + [" ".join(["word"] * 10)]
    print("chunking ok")


class FakeSlide:
    """Minimal OpenSlide stand-in: a 40x 'slide' that is white glass with a pink tissue ellipse."""

    def __init__(self, w=6000, h=4000):
        rng = np.random.default_rng(0)
        img = Image.new("RGB", (w, h), (245, 245, 245))
        ImageDraw.Draw(img).ellipse((1000, 800, 4500, 3200), fill=(200, 110, 170))
        arr = np.asarray(img).astype(np.int16) + rng.integers(-15, 15, (h, w, 3))
        self.img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
        self.dimensions = (w, h)
        self.level_downsamples = (1.0, 4.0)
        self.properties = {"openslide.objective-power": "40"}

    def get_best_level_for_downsample(self, ds):
        return 1 if ds >= 4 else 0

    def get_thumbnail(self, size):
        t = self.img.copy()
        t.thumbnail(size)
        return t

    def read_region(self, loc, level, size):
        ds = self.level_downsamples[level]
        x, y = loc
        box = (x, y, x + int(size[0] * ds), y + int(size[1] * ds))
        return self.img.crop(box).resize(size).convert("RGBA")


def check_tiling():
    r = tile_slide(FakeSlide(), n_tiles=20, seed=1)
    assert r["tiles"].shape == (20, 224, 224, 3), r["tiles"].shape
    assert r["tile_size0"] == 448 and r["level"] == 0
    cx, cy = r["coords"][:, 0] + 224, r["coords"][:, 1] + 224  # tile centers
    inside = ((cx - 2750) / 1750) ** 2 + ((cy - 2000) / 1200) ** 2 <= 1.15
    assert inside.all(), "every tile should sit on the tissue ellipse"
    assert r["tiles"].reshape(-1, 3).mean(0)[1] < 150, "tiles should be pink, not white glass"
    print(f"tiling ok ({r['n_candidates']} candidate cells, Otsu threshold {r['sat_threshold']:.2f})")


def check_pick_patients(tmp: Path, env):
    names = []
    for i in range(6):
        pid = f"TCGA-{10 + i % 2:02d}-{1000 + i:04d}"
        names += [f"/x/TCGA-LUAD/{pid}-01Z-00-DX1.svs", f"/x/TCGA-LUAD/{pid}-01A-01-TS1.svs",
                  f"/x/TCGA-LUAD/{pid}-01A-01-BS1.svs"]
    names += ["/x/TCGA-LUAD/TCGA-99-0001-01A-01-TS1.svs", "/x/TCGA-LUAD/TCGA-99-0001-01A-01-BS1.svs"]
    lst = tmp / "files.txt"
    lst.write_text("\n".join(names))
    subprocess.run([sys.executable, str(REPO / "01_pick_patients.py"), "--file-list", str(lst)],
                   env=env, check=True, capture_output=True)
    p = pd.read_csv(tmp / "data" / "patients.csv")
    assert len(p) == 6, "the patient with only TS+BS (one frozen piece) must be dropped"
    assert (p.pair_kind == "DX01-vs-TS01").all()
    assert len(pd.read_csv(tmp / "data" / "slides.csv")) == 12
    print("patient selection ok")


def make_synthetic(tmp: Path, n=240, d=128, r=12, seed=0):
    """Like real foundation-model embeddings, signal lives in a low-rank subspace (r of d dims):
    diagnosis + site + patient fingerprint + noise, then a little ambient noise. Reports share
    the fingerprint (and the same image-text space) but not the site."""
    rng = np.random.default_rng(seed)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)  # noqa: E731
    P = np.linalg.qr(rng.normal(size=(d, r)))[0]  # latent -> embedding space
    dx = rng.integers(0, 2, n)
    site = rng.integers(0, 8, n)
    dx_vec, site_vec = 1.5 * rng.normal(size=(2, r)), rng.normal(size=(8, r))
    finger = rng.normal(size=(n, r))
    img_off, txt_off = 0.3 * rng.normal(size=d), 0.3 * rng.normal(size=d)
    embed = lambda L, off: unit(L @ P.T + 0.05 * rng.normal(size=(n, d)) + off)  # noqa: E731
    A = embed(dx_vec[dx] + site_vec[site] + 0.8 * finger + rng.normal(size=(n, r)), img_off)
    B = embed(dx_vec[dx] + site_vec[site] + 0.8 * finger + rng.normal(size=(n, r)), img_off)
    R = embed(dx_vec[dx] + 0.6 * finger + rng.normal(size=(n, r)), txt_off)

    pids = [f"TCGA-{s:02d}-{i:04d}" for i, s in enumerate(site)]
    pd.DataFrame(dict(patient_id=pids, project="TCGA-X", diagnosis=np.where(dx, "LUSC", "LUAD"),
                      site=[f"{s:02d}" for s in site], slide_a=[p + "-01Z-00-DX1" for p in pids],
                      slide_b=[p + "-01A-01-TS1" for p in pids], pair_kind="DX01-vs-TS01")
                 ).to_csv(tmp / "data" / "patients.csv", index=False)
    emb = tmp / "emb" / "conch"
    emb.mkdir(parents=True)
    ids = np.array([p + "-01Z-00-DX1" for p in pids] + [p + "-01A-01-TS1" for p in pids])
    np.savez(emb / "slides.npz", slide_ids=ids, X=np.concatenate([A, B]).astype(np.float32))
    keep = rng.random(n) < 0.9  # some patients have no report
    np.savez(emb / "reports.npz", patient_ids=np.array(pids)[keep], X=R[keep].astype(np.float32))


def make_tiles(tmp: Path, n_patients=40, n_tiles=24, d=128, seed=1):
    """Fake step 3-4 outputs for the first patients' slide A: tile coords on a tissue ellipse, tile
    pixels, a thumbnail, and unit-length tile embeddings whose mean is the slide embedding (as in
    03_embed_images.py). Tiles on the left half are dark purple 'tumor' and point toward the
    patient's own slide B and report; tiles on the right are pink 'stroma' and point away."""
    rng = np.random.default_rng(seed)
    unit = lambda X: X / np.linalg.norm(X, axis=-1, keepdims=True)  # noqa: E731
    p = pd.read_csv(tmp / "data" / "patients.csv")
    emb = tmp / "emb" / "conch"
    z = np.load(emb / "slides.npz")
    slides = dict(zip(z["slide_ids"].tolist(), z["X"]))
    z = np.load(emb / "reports.npz")
    reports = dict(zip(z["patient_ids"].tolist(), z["X"]))
    B = np.stack([slides[s] for s in p.slide_b])
    R_mean = np.mean(list(reports.values()), axis=0)
    for sub in ("emb/conch/tiles", "tiles", "thumbs"):
        (tmp / sub).mkdir(parents=True)

    ds, size0 = 10.0, 448  # a 6000x4000 level-0 slide, 600x400 thumbnail
    gx, gy = np.meshgrid(np.arange(0, 6000 - size0, size0), np.arange(0, 4000 - size0, size0))
    cx, cy = gx.ravel() + size0 / 2, gy.ravel() + size0 / 2
    on_tissue = ((cx - 2750) / 1750) ** 2 + ((cy - 2000) / 1200) ** 2 <= 1
    cells = np.stack([gx.ravel(), gy.ravel()], 1)[on_tissue]
    thumb = Image.new("RGB", (600, 400), (245, 245, 245))
    ImageDraw.Draw(thumb).ellipse((100, 80, 450, 320), fill=(225, 140, 185))
    tumor = Image.new("RGB", (600, 400), (245, 245, 245))
    ImageDraw.Draw(tumor).ellipse((100, 80, 450, 320), fill=(95, 60, 150))
    thumb.paste(tumor.crop((0, 0, 275, 400)), (0, 0))

    for i in range(n_patients):
        sid, pid = p.slide_a[i], p.patient_id[i]
        coords = cells[rng.choice(len(cells), n_tiles, replace=False)].astype(np.int32)
        hot = coords[:, 0] + size0 / 2 < 2750
        pixels = np.where(hot[:, None, None, None], (95, 60, 150), (225, 140, 185)) + rng.integers(
            -20, 20, (n_tiles, 32, 32, 3))
        v = unit(B[i] - B.mean(0)) + (unit(reports[pid] - R_mean) if pid in reports else 0)
        a = np.where(hot, 1.0, -1.0)
        T = unit(slides[sid] + 0.4 * (a - a.mean())[:, None] * unit(v) + 0.05 * rng.normal(size=(n_tiles, d)))
        T = T.astype(np.float16)
        np.save(emb / "tiles" / f"{sid}.npy", T)
        m = T.astype(np.float32).mean(0)
        slides[sid] = m / np.linalg.norm(m)
        np.savez(tmp / "tiles" / f"{sid}.npz", tiles=np.clip(pixels, 0, 255).astype(np.uint8), coords=coords,
                 tile_size0=size0)
        thumb.save(tmp / "thumbs" / f"{sid}.jpg", quality=90)
        (tmp / "thumbs" / f"{sid}.txt").write_text(f"{ds}\n")
    ids = list(slides)
    np.savez(emb / "slides.npz", slide_ids=np.array(ids), X=np.stack([slides[s] for s in ids]))


def run(script, env, *args):
    r = subprocess.run([sys.executable, str(REPO / script), "--model", "conch", *args] if script != "08_scorecard.py"
                       else [sys.executable, str(REPO / script)], env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stdout, r.stderr)
        raise SystemExit(f"{script} failed")
    return r.stdout


def main():
    check_barcodes()
    check_chunks()
    check_tiling()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        (tmp / "data").mkdir()
        env = {**os.environ, "TF_ROOT": str(tmp), "TF_WORK": str(tmp), "PYTHONPATH": str(REPO),
               "MPLBACKEND": "Agg"}
        check_pick_patients(tmp, env)
        make_synthetic(tmp)

        out = run("05_tests.py", env)
        print(out)
        import json
        tests = json.loads((tmp / "results/conch/tests.json").read_text())
        for name in ("test1", "test2"):
            s = tests[name]["all"]
            assert s["top1"] > 5 * s["chance_top1"], f"{name} should find the planted fingerprint"
        assert tests["test2"]["dx+site"]["n_queries"] > 0

        out = run("06_scramble.py", env, "--seeds", "2", "--ks", "0", "5", "20", "60", "150",
                  "--pca-dims", "16")
        print(out)
        sc = json.loads((tmp / "results/conch/scramble.json").read_text())["tests"]["test2"]
        assert sc["same_key"]["all"]["top1"] == sc["clear"]["all"]["top1"], "same key changes nothing"
        plain, pca = (sc["curves"][a]["all"] for a in ("procrustes", "pca16+procrustes"))
        assert plain["mean"][0] < 3 * plain["chance"], "different keys should drop to ~chance"
        assert plain["mean"][-1] > 0.5 * sc["clear"]["all"]["top1"], "Procrustes should break it"
        assert pca["mean"][2] > plain["mean"][2], "PCA + Procrustes should need fewer pairs"

        out = run("07_remove.py", env, "--sigmas", "0", "0.5", "1", "3", "--ks", "0", "5", "20",
                  "--noise-seeds", "1")
        print(out)
        rm = json.loads((tmp / "results/conch/removal.json").read_text())["methods"]
        noise = rm["noise"]["rows"]
        assert noise[-1]["test2_all_top1"] < noise[0]["test2_all_top1"], "noise should cut matching"
        assert noise[0]["utility"] > 0.9, "clean embeddings should classify the planted diagnosis"

        print(run("08_scorecard.py", env))
        for f in ("scramble.png", "removal_noise.png", "removal_remove.png", "tradeoff.png"):
            assert (tmp / "results/conch" / f).stat().st_size > 10_000, f

        make_tiles(tmp)
        print(run("09_heatmaps.py", env))
        hm = json.loads((tmp / "results/conch/heatmaps.json").read_text())["tests"]
        assert len(hm["test1"]["patients"]) == 6, "6 patients by default"
        for name, res in hm.items():
            for pid, st in res["patients"].items():
                png = tmp / "results/conch/heatmaps" / f"{name}_{pid}.png"
                assert png.stat().st_size > 10_000, png
                assert st["tile_mean_cos"] > 0.999, "slide embedding is the mean of its tiles"
                assert abs(st["margin_from_tiles"] - st["slide_margin"]) < 1e-4, "tile scores add up to the margin"
        assert all(hm["test1"]["patients"][p]["rank"] == 1 or hm["test2"]["patients"].get(p, {}).get("rank") == 1
                   for p in hm["test1"]["patients"]), "only correctly matched patients are drawn"
        assert hm["test1"]["median_spearman"]["hematoxylin"] > 0.5, "planted identity sits on the purple tiles"
        assert (tmp / "results/conch/heatmaps_summary.png").stat().st_size > 50_000
        run("09_heatmaps.py", env, "--control", "dx+site", "--n-patients", "2", "--top-tiles", "0")
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
