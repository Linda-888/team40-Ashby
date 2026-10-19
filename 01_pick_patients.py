"""Step 2: choose patients with two different specimens and write data/patients.csv + data/slides.csv.

    python 01_pick_patients.py --projects TCGA-LUAD TCGA-LUSC
    python 01_pick_patients.py --projects TCGA-BRCA --clinical clinical.tsv --mode tumor_vs_normal

Slide A is always the tumor slide (FFPE diagnostic when there is one), so Test 2 compares
each report with the slide it describes. Slide B is a different physical specimen:
  dx_vs_frozen     A = tumor FFPE (DX), B = tumor frozen (TS/BS/MS)  [most patients have both]
  tumor_vs_normal  A = tumor,           B = normal tissue (sample 11)
Diagnosis defaults to the TCGA project (so LUAD vs LUSC gives free subtype labels);
pass a GDC clinical TSV to use primary_diagnosis instead.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from fingerprints import barcodes, paths
from fingerprints.barcodes import choose_pair

def _str(x):
    """pandas gives NaN for empty cells; treat it as missing."""
    return x if isinstance(x, str) and x else None


def find_slides(slides_dir: str, file_list: str | None, pattern: str):
    if file_list:
        return [line.strip() for line in open(file_list) if line.strip()]
    print(f"scanning {slides_dir} for {pattern} ...", flush=True)
    return [str(p) for p in Path(slides_dir).rglob(pattern)]


def load_clinical(path: str) -> pd.DataFrame:
    """GDC clinical.tsv, old (case_submitter_id) or new (cases.submitter_id) column names."""
    df = pd.read_csv(path, sep="\t", dtype=str, na_values=["'--", "--", "not reported", "Not Reported"])
    pick = lambda *names: next((n for n in names if n in df.columns), None)  # noqa: E731
    cols = {
        "patient_id": pick("case_submitter_id", "cases.submitter_id", "submitter_id"),
        "project": pick("project_id", "project.project_id"),
        "diagnosis": pick("primary_diagnosis", "diagnoses.primary_diagnosis"),
    }
    if cols["patient_id"] is None:
        sys.exit(f"{path}: no case_submitter_id / cases.submitter_id column")
    out = pd.DataFrame({k: df[v] if v else None for k, v in cols.items()})
    # Several rows per case (one per diagnosis/treatment): keep the first non-empty value of each column.
    return out.groupby("patient_id", as_index=False).first()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides-dir", default=os.environ.get("TF_SLIDES_DIR"))
    ap.add_argument("--file-list", help="text file with one slide path per line (skips the directory scan)")
    ap.add_argument("--pattern", default="*.svs")
    ap.add_argument("--projects", nargs="*", help="keep only these TCGA projects, e.g. TCGA-LUAD TCGA-LUSC")
    ap.add_argument("--clinical", help="GDC clinical TSV for project / primary_diagnosis")
    ap.add_argument("--mode", choices=["dx_vs_frozen", "tumor_vs_normal"], default="dx_vs_frozen")
    ap.add_argument("--max-patients", type=int, help="random subset for a quick first run")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not args.slides_dir and not args.file_list:
        sys.exit("set TF_SLIDES_DIR in config.sh or pass --slides-dir / --file-list")

    by_patient = defaultdict(list)
    project_of = {}
    unparsed = 0
    for path in find_slides(args.slides_dir, args.file_list, args.pattern):
        bc = barcodes.parse(os.path.basename(path))
        if bc is None:
            unparsed += 1
            continue
        by_patient[bc.patient].append((bc, path))
        project_of.setdefault(bc.patient, barcodes.project_from_path(path))
    print(f"{sum(map(len, by_patient.values()))} TCGA slides, {len(by_patient)} patients "
          f"({unparsed} files without a slide barcode)")

    clinical = load_clinical(args.clinical).set_index("patient_id") if args.clinical else None

    rows, slide_rows = [], []
    for pid, slides in sorted(by_patient.items()):
        project = project_of.get(pid)
        diagnosis = None
        if clinical is not None and pid in clinical.index:
            project = project or _str(clinical.at[pid, "project"])
            diagnosis = _str(clinical.at[pid, "diagnosis"])
        if args.projects and project not in args.projects:
            continue
        pair = choose_pair(slides, args.mode)
        if pair is None:
            continue
        (a, a_path), (b, b_path) = pair
        rows.append(dict(patient_id=pid, project=project or "unknown",
                         diagnosis=diagnosis or project or "unknown",
                         site=a.site, slide_a=a.barcode, slide_b=b.barcode,
                         pair_kind=f"{a.kind}{a.sample_type}-vs-{b.kind}{b.sample_type}"))
        for role, bc, p in (("A", a, a_path), ("B", b, b_path)):
            slide_rows.append(dict(slide_id=bc.barcode, patient_id=pid, role=role, path=p,
                                   kind=bc.kind, sample_type=bc.sample_type, site=bc.site))

    if not rows:
        sys.exit("no patients with two different specimens; check --projects / --mode / the slide paths")
    patients = pd.DataFrame(rows)
    if args.max_patients and len(patients) > args.max_patients:
        patients = patients.sample(args.max_patients, random_state=args.seed).sort_values("patient_id")
    slides = pd.DataFrame(slide_rows)
    slides = slides[slides.patient_id.isin(patients.patient_id)]
    dupes = slides.slide_id.duplicated()
    if dupes.any():
        print(f"warning: dropping {dupes.sum()} duplicate slide barcodes")
        slides = slides[~dupes]

    paths.all_patients_csv().parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(paths.all_patients_csv(), index=False)
    paths.clean_patients_csv().unlink(missing_ok=True)  # a new patient list needs a new 12_duplicates.py run
    slides.to_csv(paths.slides_csv(), index=False)

    site_n = patients.site.value_counts()
    print(f"\n{len(patients)} patients, {len(slides)} slides -> {paths.all_patients_csv()}")
    print(f"pair kinds:\n{patients.pair_kind.value_counts().to_string()}")
    print(f"diagnoses:\n{patients.diagnosis.value_counts().to_string()}")
    print(f"sites: {len(site_n)} total, {(site_n >= 2).sum()} with >=2 patients "
          f"(covering {site_n[site_n >= 2].sum()} patients, usable for the same-site control)")
    print(f"largest sites:\n{site_n.head(10).to_string()}")


if __name__ == "__main__":
    main()
