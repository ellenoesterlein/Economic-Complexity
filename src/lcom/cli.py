"""
Price Route B methanol (LCOM) from promoted BOA LCOE files.

For every LCOE file it writes a result NetCDF, an assumptions register (CSV) and a provenance
record (JSON) to ``<out-dir>/<run>/``. The LCOE year priced is 2050. The BOA cost workbook of the
run (``boa_cost_data.xlsx``) must be next to the LCOE file, or be given with ``--cost-workbook``.

Examples:
    python -m lcom.cli data/boa/<run>/optimal_lcoe_wy2025_01MWkm2_cov0p9_2025_2060.nc
    python -m lcom.cli FILE --wacc 0.08              # the paper's discount rate for every pixel
    python -m lcom.cli FILE --load-factor 0.95
"""

import argparse
import logging
import sys
from pathlib import Path

from lcom.run import LcoeFileError, run

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="lcom", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("lcoe_files", nargs="+", type=Path, help="Promoted BOA LCOE files (optimal_lcoe_*.nc).")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"), help="Output folder (default: outputs).")
    parser.add_argument(
        "--cost-workbook",
        type=Path,
        default=None,
        help="BOA cost workbook of the LCOE run; default: boa_cost_data.xlsx next to each LCOE file.",
    )
    parser.add_argument(
        "--load-factor", type=float, default=None, help="Plant load factor in (0, 1]; default: the LCOE coverage."
    )
    parser.add_argument(
        "--wacc", type=float, default=None, help="One cost of capital for every pixel instead of the regional WACC."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    try:
        for lcoe_file in args.lcoe_files:
            run(lcoe_file, args.out_dir, args.cost_workbook, args.load_factor, args.wacc)
    except (FileNotFoundError, LcoeFileError, ValueError) as e:
        logger.error(str(e))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
