"""Load one model's embeddings aligned to patients.csv."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from fingerprints import paths


@dataclass
class Embeddings:
    patients: pd.DataFrame  # one row per patient: patient_id, site, diagnosis, ...
    A: np.ndarray           # slide A (tumor; the slide the report describes), (n, d)
    B: np.ndarray           # slide B (a different specimen), rows valid where has_B
    R: np.ndarray           # report, rows valid where has_R
    has_B: np.ndarray
    has_R: np.ndarray

    @property
    def n(self) -> int:
        return len(self.patients)

    @property
    def d(self) -> int:
        return self.A.shape[1]

    def with_(self, **arrays) -> "Embeddings":
        """Same patients, transformed embeddings (for defenses)."""
        return replace(self, **arrays)

    def subset(self, mask) -> "Embeddings":
        m = np.asarray(mask, bool)
        return Embeddings(self.patients[m].reset_index(drop=True), self.A[m], self.B[m], self.R[m],
                          self.has_B[m], self.has_R[m])


def _lookup(npz_path, id_key):
    z = np.load(npz_path)
    return {k: x for k, x in zip(z[id_key].tolist(), z["X"].astype(np.float32))}


def load(model: str, reports_name: str = "reports") -> Embeddings:
    patients = pd.read_csv(paths.patients_csv(), dtype={"site": str})
    slides = _lookup(paths.emb_dir(model) / "slides.npz", "slide_ids")
    rep_path = paths.emb_dir(model) / f"{reports_name}.npz"
    reports = _lookup(rep_path, "patient_ids") if rep_path.exists() else {}

    patients = patients[patients.slide_a.isin(slides)].reset_index(drop=True)
    A = np.stack([slides[s] for s in patients.slide_a])
    d = A.shape[1]
    nan = np.full(d, np.nan, np.float32)
    B = np.stack([slides.get(s, nan) for s in patients.slide_b])
    R = np.stack([reports.get(p, nan) for p in patients.patient_id])
    e = Embeddings(patients, A, B, R, ~np.isnan(B[:, 0]), ~np.isnan(R[:, 0]))
    print(f"[{model}] {e.n} patients with slide A, {e.has_B.sum()} with slide B, "
          f"{e.has_R.sum()} with a report; dim {d}")
    return e


def normalize(X: np.ndarray) -> np.ndarray:
    return X / np.linalg.norm(X, axis=1, keepdims=True)
