"""Defense 1 (secret-key rotation + the Procrustes attack) and Defense 2 (noise, direction removal)."""
from __future__ import annotations

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fingerprints.data import Embeddings, normalize


# ---- Defense 1: secret-key scrambling ---------------------------------------------------------

def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random orthogonal matrix (QR of a Gaussian, with the sign fix)."""
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))


def procrustes_attack(Q_rot, C_rot, known, pca_dims: int = 0):
    """Attacker knows k true (query, candidate) pairs. Returns (queries, candidates) in a shared space.

    pca_dims=0: plain Procrustes. Fit W with Q_rot @ W ~ C_rot on the known pairs (full dimension,
                so it needs many pairs when k << d).
    pca_dims=p: stronger. PCA each release on its own first. A rotation only rotates the principal
                axes, so PCA coordinates are the same as before scrambling (up to sign), and Procrustes
                then only has to align p dimensions.
    """
    if pca_dims:
        def pca_coords(X):
            Xc = X - X.mean(0)
            _, _, vt = np.linalg.svd(Xc, full_matrices=False)
            return Xc @ vt[:pca_dims].T
        Q_rot, C_rot = pca_coords(Q_rot), pca_coords(C_rot)
    if len(known):
        W, _ = orthogonal_procrustes(Q_rot[known], C_rot[known])
        Q_rot = Q_rot @ W
    return normalize(Q_rot), normalize(C_rot)


# ---- Defense 2: removing the fingerprint ------------------------------------------------------

def add_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with expected norm sigma (embeddings have norm 1), then renormalize."""
    if sigma == 0:
        return X
    return normalize(X + sigma * rng.standard_normal(X.shape) / np.sqrt(X.shape[1]))


def center_within(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's mean: removes disease signal (and the image-text modality gap)."""
    out = X.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= X[m].mean(0)
    return out


def identity_directions(e: Embeddings, k: int, method: str = "mean") -> np.ndarray:
    """Top-k directions that carry patient identity rather than diagnosis. Returns (d, k).

    mean: the plan's version. Average each patient's (within-diagnosis-centered) slide A, slide B
          and report, then PCA of those patient means.
    xcov: directions where a patient's two views agree (cross-covariance of A with B and of R with A),
          i.e. exactly what the matching attacks exploit.
    """
    if k == 0:
        return np.zeros((e.d, 0))
    dx = e.patients.diagnosis.astype(str).to_numpy()
    Ac = center_within(e.A, dx)
    Bc = np.full_like(e.B, np.nan)
    Bc[e.has_B] = center_within(e.B[e.has_B], dx[e.has_B])
    Rc = np.full_like(e.R, np.nan)
    if e.has_R.any():
        Rc[e.has_R] = center_within(e.R[e.has_R], dx[e.has_R])

    if method == "mean":
        M = center_within(np.nanmean(np.stack([Ac, Bc, Rc]), axis=0), dx)
        _, _, vt = np.linalg.svd(M, full_matrices=False)
        return vt[:k].T
    if method == "xcov":
        C = np.zeros((e.d, e.d))
        for X, Y, m in ((Ac, Bc, e.has_B), (Rc, Ac, e.has_R)):
            if m.sum() > 1:
                xy = X[m].T @ Y[m] / m.sum()
                C += (xy + xy.T) / 2
        w, v = np.linalg.eigh(C)
        return v[:, np.argsort(w)[::-1][:k]]
    raise ValueError(method)


def project_out(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    if U.shape[1] == 0:
        return X
    return normalize(X - (X @ U) @ U.T)


# ---- What the defense costs -------------------------------------------------------------------

def subtype_utility(X: np.ndarray, labels, n_splits: int = 5, seed: int = 0) -> float:
    """Balanced accuracy of a logistic-regression subtype classifier, 5-fold CV.
    One slide per patient, so folds are already split by patient."""
    labels = np.asarray(labels).astype(str)
    vals, counts = np.unique(labels, return_counts=True)
    keep = np.isin(labels, vals[counts >= n_splits])
    if len(np.unique(labels[keep])) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, X[keep], labels[keep], cv=cv)
    return float(balanced_accuracy_score(labels[keep], pred))
