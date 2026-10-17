"""Tissue mask + random ~20x tiles from a whole-slide image (works on any OpenSlide-like object)."""
from __future__ import annotations

import numpy as np
from PIL import Image

DARK = 0.15  # HSV value below this is black border / scanner edge, not tissue


def otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    mids = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * mids)
    mu0 = m0 / np.maximum(w0, 1)
    mu1 = (m0[-1] - m0) / np.maximum(w1, 1)
    return float(mids[np.argmax(w0 * w1 * (mu0 - mu1) ** 2)])


def _sat_val(rgb: np.ndarray):
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255
    return hsv[..., 1], hsv[..., 2]


def tissue_mask(rgb: np.ndarray, min_sat: float = 0.05):
    """Stained tissue is saturated, glass background is near-white. Returns (mask, threshold)."""
    s, v = _sat_val(rgb)
    thr = max(otsu(s), min_sat)
    return (s > thr) & (v > DARK), thr


def tissue_fraction(rgb: np.ndarray, sat_threshold: float) -> float:
    s, v = _sat_val(rgb)
    return float(((s > sat_threshold) & (v > DARK)).mean())


def magnification(slide, target_mag: float, tile_px: int) -> dict:
    """Which pyramid level to read and how many pixels, so a tile covers tile_px at target_mag."""
    props = slide.properties
    if "openslide.objective-power" in props:
        base = float(props["openslide.objective-power"])
    elif "openslide.mpp-x" in props:
        base = 10.0 / float(props["openslide.mpp-x"])  # 0.25 um/px ~ 40x, 0.5 ~ 20x
    else:
        base = 40.0
    ds = base / target_mag
    level = slide.get_best_level_for_downsample(ds * 1.01)  # tolerate 3.9999x-style downsamples
    size0 = int(round(tile_px * ds))  # tile footprint in level-0 pixels
    read_px = max(1, int(round(size0 / slide.level_downsamples[level])))
    return dict(base_mag=base, level=level, size0=size0, read_px=read_px)


def _read_rgb(slide, x, y, level, read_px, tile_px) -> np.ndarray:
    region = slide.read_region((x, y), level, (read_px, read_px))
    rgb = Image.new("RGB", region.size, (255, 255, 255))
    rgb.paste(region, mask=region.split()[3])  # transparent (unscanned) area -> white
    if read_px != tile_px:
        rgb = rgb.resize((tile_px, tile_px), Image.Resampling.LANCZOS)
    return np.asarray(rgb)


def tile_slide(slide, n_tiles=200, tile_px=224, target_mag=20.0, min_tissue=0.5, seed=0,
               thumb_max=2048) -> dict:
    W, H = slide.dimensions
    mag = magnification(slide, target_mag, tile_px)
    size0 = mag["size0"]

    thumb = np.asarray(slide.get_thumbnail((thumb_max, thumb_max)).convert("RGB"))
    th, tw = thumb.shape[:2]
    mask, thr = tissue_mask(thumb)
    sx, sy = W / tw, H / th

    # Tissue fraction of every grid cell at once, via an integral image of the thumbnail mask.
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    gx, gy = np.meshgrid(np.arange(0, max(W - size0, 0) + 1, size0),
                         np.arange(0, max(H - size0, 0) + 1, size0))
    gx, gy = gx.ravel(), gy.ravel()
    x0 = np.clip((gx / sx).astype(int), 0, tw - 1)
    y0 = np.clip((gy / sy).astype(int), 0, th - 1)
    x1 = np.clip(np.ceil((gx + size0) / sx).astype(int), x0 + 1, tw)
    y1 = np.clip(np.ceil((gy + size0) / sy).astype(int), y0 + 1, th)
    frac = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((x1 - x0) * (y1 - y0))
    cand = np.flatnonzero(frac >= min_tissue)
    np.random.default_rng(seed).shuffle(cand)

    tiles, coords = [], []
    for i in cand:  # the thumbnail is coarse, so re-check each tile at full resolution
        x, y = int(gx[i]), int(gy[i])
        rgb = _read_rgb(slide, x, y, mag["level"], mag["read_px"], tile_px)
        if tissue_fraction(rgb, thr) < min_tissue:
            continue
        tiles.append(rgb)
        coords.append((x, y))
        if len(tiles) == n_tiles:
            break

    return dict(
        tiles=np.stack(tiles) if tiles else np.zeros((0, tile_px, tile_px, 3), np.uint8),
        coords=np.array(coords, dtype=np.int32).reshape(-1, 2),  # level-0 top-left of each tile
        tile_size0=size0, level=mag["level"], base_mag=mag["base_mag"], target_mag=target_mag,
        sat_threshold=thr, n_candidates=len(cand), thumb=thumb, thumb_downsample=(sx + sy) / 2,
    )
