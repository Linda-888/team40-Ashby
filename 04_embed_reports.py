"""Step 5: embed each patient's report text with the same model's text encoder.

Input: $TF_REPORTS_CSV with a patient id column and a text column. Any column holding a TCGA
barcode works as the id (e.g. TCGA-Reports' patient_filename); only the first TCGA-XX-XXXX is used.
Reports are split into chunks under the token limit, embedded, averaged, then set to length 1.
If you shortened reports with GPT-OSS on Lumen, put them in a column and pass --text-col.

    python 04_embed_reports.py --model conch
    python 04_embed_reports.py --model conch --text-col summary --out-name reports_summary
"""
import argparse
import sys

import numpy as np
import pandas as pd

from fingerprints import barcodes, paths
from fingerprints.models import load_encoder
from fingerprints.text import chunk_text

ID_COLUMNS = ("patient_id", "patient_filename", "case_submitter_id", "submitter_id", "filename")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=paths.MODELS)
    ap.add_argument("--reports", default=str(paths.reports_csv()))
    ap.add_argument("--id-col", default=None, help=f"default: first of {ID_COLUMNS} present")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--out-name", default="reports", help="emb/<model>/<out-name>.npz")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.reports, dtype=str)
    id_col = args.id_col or next((c for c in ID_COLUMNS if c in df.columns), None)
    if id_col is None or args.text_col not in df.columns:
        sys.exit(f"need an id column ({ID_COLUMNS}) and '{args.text_col}'; found {list(df.columns)}")
    df["pid"] = df[id_col].map(barcodes.patient_id)
    df = df.dropna(subset=["pid", args.text_col])
    # One report per patient; if there are several, join them.
    reports = df.groupby("pid")[args.text_col].apply(lambda s: " ".join(s)).to_dict()

    wanted = set(pd.read_csv(paths.patients_csv()).patient_id)
    pids = sorted(p for p in reports if p in wanted)
    print(f"{len(reports)} patients with reports, {len(pids)} of them in patients.csv", flush=True)
    if not pids:
        sys.exit("no overlap between report ids and patients.csv")

    enc = load_encoder(args.model, args.device)
    chunks, owner = [], []
    for i, pid in enumerate(pids):
        cs = chunk_text(reports[pid], enc.max_words)
        chunks += cs
        owner += [i] * len(cs)
    owner = np.array(owner)
    print(f"{len(chunks)} chunks ({len(chunks) / len(pids):.1f} per report)", flush=True)

    Z = enc.embed_texts(chunks)
    X = np.zeros((len(pids), Z.shape[1]), np.float32)
    np.add.at(X, owner, Z)
    keep = np.bincount(owner, minlength=len(pids)) > 0
    X = X[keep] / np.linalg.norm(X[keep], axis=1, keepdims=True)
    pids = np.array(pids)[keep]

    out = paths.emb_dir(args.model) / f"{args.out_name}.npz"
    paths.save_atomic(out, lambda fh: np.savez(fh, patient_ids=pids, X=X))
    print(f"{len(pids)} report embeddings -> {out}")


if __name__ == "__main__":
    main()
