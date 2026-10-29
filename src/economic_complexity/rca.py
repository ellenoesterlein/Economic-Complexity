"""Revealed comparative advantage (RCA) and the binary country-product matrix M from BACI trade data."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def compute_rca(input_file: Path, chunksize: int, product_level: int) -> pd.DataFrame:
    """Compute the Balassa RCA per exporter and product from a BACI CSV file."""
    usecols = ["i", "k", "v"]
    chunks = []

    for chunk in pd.read_csv(input_file, chunksize=chunksize, usecols=usecols):
        # BACI stores k as integer -> restore leading zeros (e.g. 010121)
        chunk["k"] = chunk["k"].astype(str).str.zfill(6).str[:product_level]
        aggregated = chunk.groupby(["i", "k"], as_index=False).agg(x_cp=("v", "sum"))
        chunks.append(aggregated)

    exports = pd.concat(chunks, ignore_index=True)
    exports = exports.groupby(["i", "k"], as_index=False)["x_cp"].sum()

    X_c = exports.groupby("i")["x_cp"].sum().rename("X_c")
    X_p = exports.groupby("k")["x_cp"].sum().rename("X_p")
    X_world = exports["x_cp"].sum()

    exports = exports.merge(X_c, on="i").merge(X_p, on="k")

    exports["share_cp"] = exports["x_cp"] / exports["X_c"]
    exports["share_p_world"] = exports["X_p"] / X_world
    exports["rca"] = exports["share_cp"] / exports["share_p_world"]
    exports["mcp"] = (exports["rca"] >= 1).astype(np.int8)

    logger.info("RCA computed for %s country-product pairs.", f"{len(exports):,}")
    return exports


def build_mcp_matrix(rca_df: pd.DataFrame) -> tuple[csr_matrix, list, list, np.ndarray, np.ndarray]:
    """Build the sparse matrix M (countries x products, 1 where RCA >= 1), diversity k_c and ubiquity k_p."""
    m = rca_df.loc[rca_df["mcp"] == 1, ["i", "k"]].copy()

    countries = sorted(m["i"].unique())
    products = sorted(m["k"].unique())

    c_index = {c: idx for idx, c in enumerate(countries)}
    p_index = {p: idx for idx, p in enumerate(products)}

    row = m["i"].map(c_index).to_numpy()
    col = m["k"].map(p_index).to_numpy()
    data = np.ones(len(m), dtype=np.float64)

    M = csr_matrix((data, (row, col)), shape=(len(countries), len(products)), dtype=np.float64)

    k_c = np.asarray(M.sum(axis=1)).ravel()
    k_p = np.asarray(M.sum(axis=0)).ravel()

    logger.info("Matrix M: %s countries x %s products, %s entries with RCA >= 1.",
                f"{len(countries):,}", f"{len(products):,}", f"{len(m):,}")

    return M, countries, products, k_c, k_p
