"""
Run the RCA, ECI and PCI calculation with the folders next to this script.

Input: ``data/BACI_HS22_Y<year>_V202601.csv``, output: ``output/ECI_PCI_HS<level>_Y<year>.xlsx``. The code is in
the ``economic_complexity`` package (``src/economic_complexity``); install the project first (``uv sync``).
Other options pass through, for example ``--year 2023``.
"""

import sys
from pathlib import Path

from economic_complexity.cli import main

SCRIPT_DIR = Path(__file__).parent

if __name__ == "__main__":
    sys.exit(main(["--data-dir", str(SCRIPT_DIR / "data"), "--out-dir", str(SCRIPT_DIR / "output"), *sys.argv[1:]]))
