"""Economic Complexity Index (ECI) and Product Complexity Index (PCI) from the matrix M."""

import logging

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import ArpackError, eigsh

logger = logging.getLogger(__name__)

EIGENVALUE_K = 6


def safe_inverse(x: np.ndarray) -> np.ndarray:
    """Return 1/x where x > 0, else 0."""
    out = np.zeros_like(x, dtype=np.float64)
    mask = x > 0
    out[mask] = 1.0 / x[mask]
    return out


def compute_eigendecomposition(A: csr_matrix, d: np.ndarray, k: int = EIGENVALUE_K) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of M~ = D^-1 A (A symmetric, d = row sums of the
    underlying normalisation) via the similar symmetric matrix
    S = D^-1/2 A D^-1/2.

    Returns eigenvalues (descending) and the eigenvectors of M~.
    """
    n = A.shape[0]
    d_inv_sqrt = np.sqrt(safe_inverse(d))
    Dm = diags(d_inv_sqrt, format="csr")
    S = Dm @ A @ Dm
    S = (S + S.T) / 2  # smooth numerical asymmetry only

    k_eig = min(k, n - 2)
    if k_eig < 2:
        raise ValueError(f"Insufficient dimension for eigendecomposition: n={n}")

    try:
        logger.info("Computing sparse eigendecomposition (k=%d).", k_eig)
        vals, vecs = eigsh(S, k=k_eig, which="LA")
    except ArpackError as e:
        logger.warning("eigsh failed (%s), using dense eigh.", e)
        vals, vecs = eigh(S.toarray())

    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    vecs = d_inv_sqrt[:, None] * vecs  # back to eigenvectors of M~

    logger.info("Largest eigenvalues: %s", np.round(vals[:3], 6))
    if vals[1] > 1 - 1e-6:
        logger.warning("Second eigenvalue is about 1: the graph may be disconnected, ECI/PCI not meaningful.")

    return vals, vecs


def standardize(values: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


def compute_eci(
    M: csr_matrix, countries: list, k_c: np.ndarray, k_p: np.ndarray, k: int = EIGENVALUE_K
) -> pd.DataFrame:
    """Compute the Economic Complexity Index; it correlates positively with diversity."""
    # A_cc' = sum_p M_cp M_c'p / k_p   (symmetric);  M~ = D_c^-1 A
    A_cc = M @ diags(safe_inverse(k_p), format="csr") @ M.T

    _, eigenvectors = compute_eigendecomposition(A_cc, k_c, k)

    eci = standardize(eigenvectors[:, 1])

    # Ensure positive correlation with diversity
    corr = np.corrcoef(eci, k_c)[0, 1]
    if np.isfinite(corr) and corr < 0:
        eci = -eci

    eci_df = pd.DataFrame({
        "country_i": countries,
        "diversity": k_c.astype(int),
        "eci": eci
    }).sort_values("eci", ascending=False).reset_index(drop=True)

    logger.info("ECI computed for %d countries. Top 10:\n%s", len(eci_df), eci_df.head(10).to_string(index=False))
    return eci_df


def compute_pci(
    M: csr_matrix, products: list, k_c: np.ndarray, k_p: np.ndarray, k: int = EIGENVALUE_K
) -> pd.DataFrame:
    """Compute the Product Complexity Index; it correlates negatively with ubiquity."""
    # A_pp' = sum_c M_cp M_cp' / k_c   (symmetric);  M~ = D_p^-1 A
    A_pp = M.T @ diags(safe_inverse(k_c), format="csr") @ M

    _, eigenvectors = compute_eigendecomposition(A_pp, k_p, k)

    pci = standardize(eigenvectors[:, 1])

    # Ensure negative correlation with ubiquity
    corr = np.corrcoef(pci, k_p)[0, 1]
    if np.isfinite(corr) and corr > 0:
        pci = -pci

    pci_df = pd.DataFrame({
        "product_k": products,
        "ubiquity": k_p.astype(int),
        "pci": pci
    }).sort_values("pci", ascending=False).reset_index(drop=True)

    logger.info("PCI computed for %d products. Top 10:\n%s", len(pci_df), pci_df.head(10).to_string(index=False))
    return pci_df
