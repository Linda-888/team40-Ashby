"""Download a TCGA cohort from the GDC (open access) when the cluster doesn't have one.

Picks patients that have both a tumor FFPE diagnostic slide (DX) and a tumor frozen slide (TS/BS/MS),
balanced across projects, skipping very large pairs, and saves them as
<out>/<project>/<file_name> so 01_pick_patients.py can read the project from the path.
Also fetches the TCGA-Reports text corpus (Kefeli et al., 2024) as <reports>, and only picks
patients who have a report there.
Standard library only, so it runs on a login node without any env.

    python3 tcga_download.py --n-patients 60                     # TCGA-LUAD + TCGA-LUSC
    python3 tcga_download.py --n-patients 200 --max-pair-gb 2.5  # bigger run; resumes partial files
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from fingerprints import barcodes
from fingerprints.barcodes import choose_pair

GDC = "https://api.gdc.cancer.gov"
REPORTS_URL = ("https://data.mendeley.com/public-files/datasets/hyg5xkznpx/files/"
               "60abe141-9352-4a54-943c-3d015eabefea/file_downloaded")  # TCGA_Reports.csv.zip


def query_slides(projects):
    body = {
        "filters": {"op": "and", "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": projects}},
            {"op": "=", "content": {"field": "data_type", "value": "Slide Image"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ]},
        "fields": "file_id,file_name,file_size,cases.project.project_id",
        "format": "JSON",
        "size": 50000,
    }
    req = urllib.request.Request(f"{GDC}/files", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        hits = json.load(r)["data"]["hits"]
    return [dict(file_id=h["file_id"], file_name=h["file_name"], size=int(h["file_size"]),
                 project=h["cases"][0]["project"]["project_id"]) for h in hits]


def download(url, dst, size=None, retries=5):
    """Stream to dst with resume (HTTP Range) and retries; skip if already complete."""
    for attempt in range(retries):
        have = os.path.getsize(dst + ".part") if os.path.exists(dst + ".part") else 0
        if os.path.exists(dst) and (size is None or os.path.getsize(dst) == size):
            return dst
        try:
            headers = {"User-Agent": "curl/8.0"}  # Mendeley refuses Python's default User-Agent
            if have:
                headers["Range"] = f"bytes={have}-"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=300) as r, open(dst + ".part", "ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range: start over
                    f.seek(0)
                    f.truncate()
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if size is not None and os.path.getsize(dst + ".part") != size:
                raise IOError(f"size mismatch {os.path.getsize(dst + '.part')} != {size}")
            os.replace(dst + ".part", dst)
            return dst
        except Exception as e:  # network hiccup: back off and resume
            print(f"  retry {attempt + 1} {os.path.basename(dst)}: {e!r}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed: {url}")


def fetch_reports(path):
    if os.path.exists(path):
        print(f"reports already at {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    zpath = path + ".zip"
    download(REPORTS_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as src, open(path, "wb") as dst:
            dst.write(src.read())
    os.remove(zpath)
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    print(f"reports -> {path} (columns: {header})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", nargs="+", default=["TCGA-LUAD", "TCGA-LUSC"])
    ap.add_argument("--n-patients", type=int, default=60, help="total, split evenly across projects")
    ap.add_argument("--max-pair-gb", type=float, default=2.0, help="skip patients whose two slides exceed this")
    ap.add_argument("--out", default=os.environ.get("TF_SLIDES_DIR"))
    ap.add_argument("--reports", default=os.environ.get("TF_REPORTS_CSV"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    args = ap.parse_args()
    if not args.out:
        sys.exit("set TF_SLIDES_DIR (source config.sh) or pass --out")

    files = query_slides(args.projects)
    print(f"GDC: {len(files)} open slide images in {args.projects}")
    by_patient = defaultdict(list)
    for f in files:
        bc = barcodes.parse(f["file_name"])
        if bc:
            by_patient[(f["project"], bc.patient)].append((bc, f))

    # Reports first (small), so we only pick patients who have one: Test 2 needs it.
    with_report = None
    if args.reports:
        fetch_reports(args.reports)
        with open(args.reports, newline="") as f:
            with_report = {barcodes.patient_id(row[0]) for row in csv.reader(f)}

    rng = random.Random(args.seed)
    per_project = {p: [] for p in args.projects}
    for (project, pid), slides in sorted(by_patient.items()):
        if with_report is not None and pid not in with_report:
            continue
        pair = choose_pair(slides, "dx_vs_frozen")
        if pair and sum(f["size"] for _, f in pair) <= args.max_pair_gb * 1e9:
            per_project[project].append([f for _, f in pair])
    quota = args.n_patients // len(args.projects)
    chosen = []
    for project, pairs in per_project.items():
        rng.shuffle(pairs)
        print(f"{project}: {len(pairs)} eligible patients, taking {min(quota, len(pairs))}")
        chosen += pairs[:quota]
    todo = [f for pair in chosen for f in pair]
    total = sum(f["size"] for f in todo)
    print(f"{len(chosen)} patients, {len(todo)} slides, {total / 1e9:.1f} GB -> {args.out}", flush=True)
    if args.dry_run:
        return

    for p in args.projects:
        os.makedirs(os.path.join(args.out, p), exist_ok=True)
    t0, done_bytes = time.time(), 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(download, f"{GDC}/data/{f['file_id']}",
                            os.path.join(args.out, f["project"], f["file_name"]), f["size"]): f for f in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            fut.result()
            done_bytes += f["size"]
            rate = done_bytes / 1e6 / (time.time() - t0)
            print(f"[{i}/{len(todo)}] {f['file_name'][:40]}  {done_bytes / 1e9:.1f}/{total / 1e9:.1f} GB "
                  f"({rate:.0f} MB/s)", flush=True)
    print("all slides downloaded")


if __name__ == "__main__":
    main()
