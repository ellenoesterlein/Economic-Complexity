"""
Economic Complexity Index (ECI) and Product Complexity Index (PCI) Calculator.

Computes RCA, ECI, and PCI from BACI trade data using sparse matrix eigendecomposition.
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

# ============================================================
# CONFIGURATION
# ============================================================

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    """Configuration parameters."""
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT / "output"

    YEAR = 2024
    CHUNKSIZE = 1_000_000
    PRODUCT_LEVEL = 6
    EIGENVALUE_K = 6

    INPUT_FILENAME = "BACI_HS22_Y{year}_V202601.csv"
    OUTPUT_FILENAME = "ECI_PCI_HS{product_level}_Y{year}.xlsx"

    @classmethod
    def setup_directories(cls):
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_input_file(cls) -> Path:
        return cls.DATA_DIR / cls.INPUT_FILENAME.format(year=cls.YEAR)

    @classmethod
    def get_output_file(cls) -> Path:
        return cls.OUTPUT_DIR / cls.OUTPUT_FILENAME.format(
            product_level=cls.PRODUCT_LEVEL,
            year=cls.YEAR
        )


# ============================================================
# UTILITIES
# ============================================================

def safe_inverse(x: np.ndarray) -> np.ndarray:
    """Return 1/x where x > 0, else 0."""
    out = np.zeros_like(x, dtype=np.float64)
    mask = x > 0
    out[mask] = 1.0 / x[mask]
    return out


def compute_eigendecomposition(A: csr_matrix,
                               d: np.ndarray,
                               k: int = 6) -> Tuple[np.ndarray, np.ndarray]:
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
        logger.info(f"   Computing sparse eigendecomposition (k={k_eig})...")
        vals, vecs = eigsh(S, k=k_eig, which="LA")
    except Exception as e:
        logger.warning(f"   eigsh failed ({e}), using dense eigh...")
        vals, vecs = eigh(S.toarray())

    idx = np.argsort(-vals)
    vals, vecs = vals[idx], vecs[:, idx]
    vecs = d_inv_sqrt[:, None] * vecs  # back to eigenvectors of M~

    logger.info(f"   Largest eigenvalues: {np.round(vals[:3], 6)}")
    if vals[1] > 1 - 1e-6:
        logger.warning("   λ2 ≈ 1: graph may be disconnected, ECI/PCI not meaningful!")

    return vals, vecs


def standardize(values: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    return (values - np.mean(values)) / (np.std(values) + 1e-12)


# ============================================================
# STEP 1: RCA COMPUTATION
# ============================================================

def compute_rca(input_file: Path, chunksize: int, product_level: int) -> pd.DataFrame:
    """Compute Revealed Comparative Advantage from BACI data."""
    logger.info("=" * 70)
    logger.info("STEP 1: Computing RCA...")
    logger.info("=" * 70)

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

    logger.info(f"✅ RCA computed for {len(exports):,} country-product pairs")
    return exports


# ============================================================
# STEP 2: MCP MATRIX CONSTRUCTION
# ============================================================

def build_mcp_matrix(rca_df: pd.DataFrame) -> Tuple[csr_matrix, list, list, np.ndarray, np.ndarray]:
    """Build MCP matrix and compute diversity/ubiquity."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Building MCP matrix...")
    logger.info("=" * 70)

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

    logger.info(f"   Matrix: {len(countries):,} countries × {len(products):,} products")
    logger.info(f"   RCA≥1 entries: {len(m):,}")

    return M, countries, products, k_c, k_p


# ============================================================
# STEP 3: ECI COMPUTATION
# ============================================================

def compute_eci(M: csr_matrix, countries: list, k_c: np.ndarray, k_p: np.ndarray) -> pd.DataFrame:
    """Compute Economic Complexity Index."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Computing ECI...")
    logger.info("=" * 70)

    # A_cc' = sum_p M_cp M_c'p / k_p   (symmetric);  M~ = D_c^-1 A
    A_cc = M @ diags(safe_inverse(k_p), format="csr") @ M.T

    _, eigenvectors = compute_eigendecomposition(A_cc, k_c, Config.EIGENVALUE_K)

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

    logger.info(f"✅ ECI computed for {len(eci_df)} countries")
    logger.info(f"\n   Top 10 complex countries:\n{eci_df.head(10).to_string(index=False)}")

    return eci_df


# ============================================================
# STEP 4: PCI COMPUTATION
# ============================================================

def compute_pci(M: csr_matrix, products: list, k_c: np.ndarray, k_p: np.ndarray) -> pd.DataFrame:
    """Compute Product Complexity Index."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Computing PCI...")
    logger.info("=" * 70)

    # A_pp' = sum_c M_cp M_cp' / k_c   (symmetric);  M~ = D_p^-1 A
    A_pp = M.T @ diags(safe_inverse(k_c), format="csr") @ M

    _, eigenvectors = compute_eigendecomposition(A_pp, k_p, Config.EIGENVALUE_K)

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

    logger.info(f"✅ PCI computed for {len(pci_df)} products")
    logger.info(f"\n   Top 10 complex products:\n{pci_df.head(10).to_string(index=False)}")

    return pci_df


# ============================================================
# MAIN
# ============================================================

def main():
    Config.setup_directories()

    input_file = Config.get_input_file()
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    rca_df = compute_rca(input_file, Config.CHUNKSIZE, Config.PRODUCT_LEVEL)
    M, countries, products, k_c, k_p = build_mcp_matrix(rca_df)

    eci_df = compute_eci(M, countries, k_c, k_p)
    pci_df = compute_pci(M, products, k_c, k_p)

    output_file = Config.get_output_file()
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        eci_df.to_excel(writer, sheet_name="ECI_ranking", index=False)
        pci_df.to_excel(writer, sheet_name="PCI_ranking", index=False)
        rca_df[rca_df["mcp"] == 1].sort_values("rca", ascending=False).to_excel(
            writer, sheet_name="RCA>=1", index=False
        )

    logger.info(f"\n✅ Results saved: {output_file}")


if __name__ == "__main__":
    main()