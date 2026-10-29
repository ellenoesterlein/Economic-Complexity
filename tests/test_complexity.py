import logging

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from economic_complexity.complexity import compute_eci, compute_pci


def _random_m(n_countries: int = 40, n_products: int = 120, seed: int = 0) -> csr_matrix:
    """A nested-like M: more diverse countries also export the less ubiquitous products."""
    rng = np.random.default_rng(seed)
    skill = np.linspace(0.1, 0.9, n_countries)[:, None]
    difficulty = np.linspace(0.05, 0.95, n_products)[None, :]
    p = 1 / (1 + np.exp((difficulty - skill) * 12))
    return csr_matrix((rng.random((n_countries, n_products)) < p).astype(np.float64))


def _sums(M: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(M.sum(axis=1)).ravel(), np.asarray(M.sum(axis=0)).ravel()


def test_eci_is_a_z_score_that_rises_with_diversity():
    M = _random_m()
    k_c, k_p = _sums(M)

    eci = compute_eci(M, list(range(M.shape[0])), k_c, k_p)

    assert eci["eci"].mean() == pytest.approx(0, abs=1e-9)
    assert eci["eci"].std(ddof=0) == pytest.approx(1, abs=1e-6)
    assert np.corrcoef(eci["eci"], eci["diversity"])[0, 1] > 0.8
    assert eci["eci"].is_monotonic_decreasing


def test_pci_falls_with_ubiquity():
    M = _random_m()
    k_c, k_p = _sums(M)

    pci = compute_pci(M, list(range(M.shape[1])), k_c, k_p)

    assert pci["pci"].mean() == pytest.approx(0, abs=1e-9)
    assert np.corrcoef(pci["pci"], pci["ubiquity"])[0, 1] < -0.8


def test_disconnected_matrix_logs_a_warning(caplog):
    block = _random_m(10, 30).toarray()
    M = csr_matrix(np.block([[block, np.zeros_like(block)], [np.zeros_like(block), block]]))
    k_c, k_p = _sums(M)

    with caplog.at_level(logging.WARNING):
        compute_eci(M, list(range(M.shape[0])), k_c, k_p)

    assert "disconnected" in caplog.text


def test_too_few_countries_raises():
    M = csr_matrix(np.ones((3, 3)))
    k_c, k_p = _sums(M)

    with pytest.raises(ValueError, match="Insufficient dimension"):
        compute_eci(M, [0, 1, 2], k_c, k_p)
