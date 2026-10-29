"""
Calculate RCA, ECI and PCI for one year of CEPII BACI trade data (HS22, release V202601).

Reads ``<data-dir>/BACI_HS22_Y<year>_V202601.csv`` and writes ``<out-dir>/ECI_PCI_HS<level>_Y<year>.xlsx``
with the sheets ``ECI_ranking``, ``PCI_ranking`` and ``RCA>=1``.

Examples:
    python -m economic_complexity.cli --year 2024
    python -m economic_complexity.cli --year 2023 --product-level 4 --data-dir D:/baci
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from economic_complexity.complexity import EIGENVALUE_K, compute_eci, compute_pci
from economic_complexity.rca import build_mcp_matrix, compute_rca

logger = logging.getLogger(__name__)

INPUT_FILENAME = "BACI_HS22_Y{year}_V202601.csv"
OUTPUT_FILENAME = "ECI_PCI_HS{product_level}_Y{year}.xlsx"


def run(input_file: Path, output_file: Path, chunksize: int, product_level: int, k: int) -> None:
    rca_df = compute_rca(input_file, chunksize, product_level)
    M, countries, products, k_c, k_p = build_mcp_matrix(rca_df)

    eci_df = compute_eci(M, countries, k_c, k_p, k)
    pci_df = compute_pci(M, products, k_c, k_p, k)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        eci_df.to_excel(writer, sheet_name="ECI_ranking", index=False)
        pci_df.to_excel(writer, sheet_name="PCI_ranking", index=False)
        rca_df[rca_df["mcp"] == 1].sort_values("rca", ascending=False).to_excel(
            writer, sheet_name="RCA>=1", index=False
        )

    logger.info("Wrote %s.", output_file)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="economic-complexity", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--year", type=int, default=2024, help="BACI year (default: 2024).")
    parser.add_argument(
        "--product-level", type=int, default=6, choices=(2, 4, 6), help="HS digits of a product (default: 6)."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/baci"), help="Folder of the BACI CSV (default: data/baci)."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output folder (default: outputs).")
    parser.add_argument("--chunksize", type=int, default=1_000_000, help="BACI rows read per chunk.")
    parser.add_argument(
        "--eigenvalues", type=int, default=EIGENVALUE_K, help=f"Eigenvalues computed (default: {EIGENVALUE_K})."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_file = args.data_dir / INPUT_FILENAME.format(year=args.year)
    if not input_file.exists():
        logger.error("Input file not found: %s", input_file)
        return 1
    output_file = args.out_dir / OUTPUT_FILENAME.format(product_level=args.product_level, year=args.year)
    run(input_file, output_file, args.chunksize, args.product_level, args.eigenvalues)
    return 0


if __name__ == "__main__":
    sys.exit(main())
