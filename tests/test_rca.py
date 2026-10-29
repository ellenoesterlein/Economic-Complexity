from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_complexity.rca import build_mcp_matrix, compute_rca


def _write_baci(path: Path, rows: list[tuple[int, int, int, float]]) -> Path:
    """Rows of (exporter i, importer j, product k as BACI integer, value v)."""
    df = pd.DataFrame(rows, columns=["i", "j", "k", "v"])
    df.insert(0, "t", 2024)
    df["q"] = 1.0
    df.to_csv(path, index=False)
    return path


# Two exporters, two products, values split over two importers each:
# exporter 1: 10121 -> 30, 20110 -> 10; exporter 2: 10121 -> 10, 20110 -> 50
ROWS = [
    (1, 3, 10121, 20.0), (1, 4, 10121, 10.0), (1, 3, 20110, 10.0),
    (2, 3, 10121, 10.0), (2, 4, 20110, 25.0), (2, 3, 20110, 25.0),
]


@pytest.mark.parametrize("chunksize", [2, 1_000])
def test_rca_is_the_balassa_index(tmp_path, chunksize):
    rca = compute_rca(_write_baci(tmp_path / "baci.csv", ROWS), chunksize, product_level=6)

    got = rca.set_index(["i", "k"])["rca"]
    assert got[(1, "010121")] == pytest.approx((30 / 40) / (40 / 100))
    assert got[(1, "020110")] == pytest.approx((10 / 40) / (60 / 100))
    assert got[(2, "010121")] == pytest.approx((10 / 60) / (40 / 100))
    assert got[(2, "020110")] == pytest.approx((50 / 60) / (60 / 100))
    assert rca.set_index(["i", "k"])["mcp"].to_dict() == {
        (1, "010121"): 1, (1, "020110"): 0, (2, "010121"): 0, (2, "020110"): 1
    }


def test_product_level_truncates_the_zero_padded_code(tmp_path):
    rca = compute_rca(_write_baci(tmp_path / "baci.csv", ROWS), 1_000, product_level=4)

    assert sorted(rca["k"].unique()) == ["0101", "0201"]


def test_mcp_matrix_gives_diversity_and_ubiquity():
    rca = pd.DataFrame({
        "i": [1, 1, 2, 3, 3],
        "k": ["a", "b", "b", "a", "c"],
        "mcp": [1, 1, 1, 0, 1],
    })

    M, countries, products, k_c, k_p = build_mcp_matrix(rca)

    assert countries == [1, 2, 3]
    assert products == ["a", "b", "c"]
    np.testing.assert_array_equal(M.toarray(), [[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(k_c, [2, 1, 1])
    np.testing.assert_array_equal(k_p, [1, 2, 1])
