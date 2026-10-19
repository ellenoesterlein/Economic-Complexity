"""
Economic Complexity Index (ECI) and Product Complexity Index (PCI) Calculator.

Computes RCA, ECI, and PCI from BACI trade data using sparse matrix eigendecomposition.
"""

import os
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigs
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
    EIGENVALUE_K = 30
    SPARSE_EIGS_MAXITER = 500

    # Input/output filenames
    INPUT_FILENAME = "BACI_HS22_Y{year}_V202601.csv"
    OUTPUT_FILENAME = "ECI_PCI_HS{product_level}_Y{year}.xlsx"

    @classmethod
    def setup_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_input_file(cls) -> Path:
        """Get input file path."""
        return cls.DATA_DIR / cls.INPUT_FILENAME.format(year=cls.YEAR)

    @classmethod
    def get_output_file(cls) -> Path:
        """Get output file path."""
        return cls.OUTPUT_DIR / cls.OUTPUT_FILENAME.format(
            product_level=cls.PRODUCT_LEVEL,
            year=cls.YEAR
        )


# ============================================================
# UTILITIES
# ============================================================

def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Safely divide arrays, returning NaN where denominator is zero."""
    return np.divide(numerator, denominator,
                     where=denominator != 0,
                     out=np.full_like(numerator, np.nan, dtype=float))


def compute_eigendecomposition(matrix: csr_matrix,
                               k: int,
                               n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute eigendecomposition using sparse or dense method.

    Falls back to dense eigendecomposition if sparse fails.
    """
    try:
        k_eig = min(n - 2, k)
        if k_eig < 2:
            raise ValueError(f"Insufficient dimension for eigendecomposition: {k_eig}")

        logger.info(f"   Computing sparse eigendecomposition (k={k_eig})...")
        eigenvalues, eigenvectors = eigs(
            matrix, k=k_eig, which="LM", v0=np.ones(n), maxiter=Config.SPARSE_EIGS_MAXITER
        )
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

    except Exception as e:
        logger.warning(f"   Sparse failed ({e}), switching to dense eigendecomposition...")
        matrix_dense = matrix.toarray()
        matrix_dense = np.nan_to_num(matrix_dense, nan=0.0, posinf=0.0, neginf=0.0)
        matrix_dense = (matrix_dense + matrix_dense.T) / 2
        eigenvalues, eigenvectors = eigh(matrix_dense)

    # Sort by descending eigenvalue
    idx = np.argsort(-eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx]


def standardize(values: np.ndarray) -> np.ndarray:
    """Standardize values (z-score normalization)."""
    mean, std = np.mean(values), np.std(values)
    return (values - mean) / (std + 1e-12)


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
        chunk["k"] = chunk["k"].astype(str).str[:product_level]
        aggregated = chunk.groupby(["i", "k"], as_index=False).agg(x_cp=("v", "sum"))
        chunks.append(aggregated)

    exports = pd.concat(chunks, ignore_index=True)
    exports = exports.groupby(["i", "k"], as_index=False)["x_cp"].sum()

    # Compute shares and RCA
    X_c = exports.groupby("i")["x_cp"].sum().rename("X_c")
    X_p = exports.groupby("k")["x_cp"].sum().rename("X_p")
    X_world = exports["x_cp"].sum()

    exports = exports.merge(X_c, on="i").merge(X_p, on="k")

    # Nutze Pandas Division direkt
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
    logger.info("Building MCP matrix...")
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
    """Compute ECI"""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Computing ECI...")
    logger.info("=" * 70)

    k_c_inv = np.zeros_like(k_c, dtype=np.float64)
    k_c_inv[k_c > 0] = 1.0 / k_c[k_c > 0]

    k_p_inv = np.zeros_like(k_p, dtype=np.float64)
    k_p_inv[k_p > 0] = 1.0 / k_p[k_p > 0]

    logger.info("   Computing M̃^cc' matrix...")
    Mtilde_cc = M @ diags(k_p_inv, format="csr") @ M.T
    Mtilde_cc = diags(k_c_inv, format="csr") @ Mtilde_cc

    eigenvalues, eigenvectors = compute_eigendecomposition(Mtilde_cc, Config.EIGENVALUE_K, len(countries))

    eci_raw = eigenvectors[:, 1].copy()
    eci = standardize(eci_raw)

    # Ensure negative correlation with diversity
    corr = np.corrcoef(eci, k_c)[0, 1]
    if np.isfinite(corr) and corr < 0:
        eci = -eci

    eci_df = pd.DataFrame({
        "country_i": countries,
        "diversity": k_c.astype(int),
        "eci": eci
    }).sort_values("eci", ascending=False).reset_index(drop=True)

    logger.info(f"✅ ECI computed for {len(eci_df)} countries")
    logger.info(f"\n   Top 10 komplexe Länder:\n{eci_df.head(10).to_string(index=False)}")

    return eci_df


# ============================================================
# STEP 4: PCI COMPUTATION
# ============================================================

def compute_pci(M: csr_matrix, products: list, k_c: np.ndarray, k_p: np.ndarray) -> pd.DataFrame:
    """Compute Product Complexity Index."""
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Computing PCI...")
    logger.info("=" * 70)

    k_c_inv = np.zeros_like(k_c, dtype=np.float64)
    k_c_inv[k_c > 0] = 1.0 / k_c[k_c > 0]

    k_p_inv = np.zeros_like(k_p, dtype=np.float64)
    k_p_inv[k_p > 0] = 1.0 / k_p[k_p > 0]

    logger.info("   Computing M̃^pp' matrix...")
    Mtilde_pp = M.T @ diags(k_c_inv, format="csr") @ M
    Mtilde_pp = diags(k_p_inv, format="csr") @ Mtilde_pp

    eigenvalues, eigenvectors = compute_eigendecomposition(Mtilde_pp, Config.EIGENVALUE_K, len(products))

    pci_raw = eigenvectors[:, 1].copy()
    pci = standardize(pci_raw)

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
    logger.info(f"\n   Top 10 komplexe Produkte:\n{pci_df.head(10).to_string(index=False)}")

    return pci_df


# ============================================================
# MAIN
# ============================================================

def main():
    """Main execution."""
    Config.setup_directories()

    input_file = Config.get_input_file()

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        logger.error(f"Expected location: {input_file}")
        return

    # Compute RCA
    rca_df = compute_rca(input_file, Config.CHUNKSIZE, Config.PRODUCT_LEVEL)

    # Build MCP matrix
    M, countries, products, k_c, k_p = build_mcp_matrix(rca_df)

    # Compute indices
    eci_df = compute_eci(M, countries, k_c, k_p)
    pci_df = compute_pci(M, products, k_c, k_p)

    # Export results
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