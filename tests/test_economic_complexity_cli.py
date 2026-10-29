import logging
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from economic_complexity.cli import main

SCRIPT = Path(__file__).parents[1] / "workflow" / "scripts" / "PCI, ECI, RCA.py"


def _write_baci(data_dir: Path, year: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = 4_000
    df = pd.DataFrame({
        "t": year,
        "i": rng.integers(1, 30, n),
        "j": rng.integers(1, 30, n),
        "k": rng.choice([10121, 20110, 30211, 840710, 854231, 870323, 271019, 90111], n),
        "v": rng.gamma(1.0, 100.0, n),
        "q": 1.0,
    })
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / f"BACI_HS22_Y{year}_V202601.csv", index=False)


def test_cli_writes_the_three_sheets(tmp_path):
    _write_baci(tmp_path / "data", 2022)

    code = main(["--year", "2022", "--data-dir", str(tmp_path / "data"), "--out-dir", str(tmp_path / "out")])

    assert code == 0
    sheets = pd.read_excel(tmp_path / "out" / "ECI_PCI_HS6_Y2022.xlsx", sheet_name=None)
    assert list(sheets) == ["ECI_ranking", "PCI_ranking", "RCA>=1"]
    assert list(sheets["ECI_ranking"].columns) == ["country_i", "diversity", "eci"]
    assert list(sheets["PCI_ranking"].columns) == ["product_k", "ubiquity", "pci"]
    assert (sheets["RCA>=1"]["rca"] >= 1).all()


def test_cli_without_input_file_fails(tmp_path, caplog):
    with caplog.at_level(logging.ERROR):
        code = main(["--year", "2022", "--data-dir", str(tmp_path), "--out-dir", str(tmp_path / "out")])

    assert code == 1
    assert "Input file not found" in caplog.text
    assert not (tmp_path / "out").exists()


def test_script_reads_the_folder_next_to_it(monkeypatch, caplog):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--year", "1900"])

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert exit_info.value.code == 1
    assert str(SCRIPT.parent / "data" / "BACI_HS22_Y1900_V202601.csv") in caplog.text
