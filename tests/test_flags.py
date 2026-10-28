import logging
from pathlib import Path

import pandas as pd
import pytest

from lcom import paper_inputs
from lcom.flags import ALL_FLAGS, CHAIN_FLAGS, COST_WORKBOOK_FLAGS, STANDING_FLAGS, check_cost_workbook_flags, log_flags


def _capex_sheet(columns: tuple[str, str], overrides: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    overrides = overrides or {}
    rows = []
    for flag in COST_WORKBOOK_FLAGS:
        value = overrides.get((flag.region, flag.tech), flag.workbook_value)
        rows.append([flag.region, flag.tech, value])
    return pd.DataFrame(rows, columns=[columns[0], columns[1], 2024])


def _write(path: Path, sheet: pd.DataFrame) -> Path:
    with pd.ExcelWriter(path) as writer:
        sheet.to_excel(writer, sheet_name="RES CAPEX projections", index=False)
    return path


def test_flag_keys_are_unique_and_cover_the_input_flags():
    keys = [f.key for f in ALL_FLAGS]
    assert len(keys) == len(set(keys))
    used = {flag for value in paper_inputs.PAPER_VALUES for flag in value.flags}
    assert used <= {f.key for f in CHAIN_FLAGS}


@pytest.mark.parametrize("columns", [("irena region", "tech"), ("irena_region", "Technology")])
def test_unchanged_workbook_matches_every_reviewed_flag(tmp_path, columns):
    path = _write(tmp_path / "costs.xlsx", _capex_sheet(columns))

    results = check_cost_workbook_flags(path)

    assert len(results) == len(COST_WORKBOOK_FLAGS)
    assert all(r.unchanged for r in results)


def test_changed_workbook_value_marks_the_flag_stale(tmp_path, caplog):
    path = _write(tmp_path / "costs.xlsx", _capex_sheet(("irena region", "tech"), {("China", "Solar PV"): 530.0}))

    results = check_cost_workbook_flags(path)
    with caplog.at_level(logging.WARNING):
        log_flags(results)

    stale = [r for r in results if not r.unchanged]
    assert [r.flag.key for r in stale] == ["capex.china.solar"]
    assert "DATA FLAG STALE" in caplog.text


def test_every_standing_and_chain_flag_is_logged(tmp_path, caplog):
    path = _write(tmp_path / "costs.xlsx", _capex_sheet(("irena region", "tech")))

    with caplog.at_level(logging.WARNING):
        log_flags(check_cost_workbook_flags(path))

    for flag in (*STANDING_FLAGS, *CHAIN_FLAGS):
        assert flag.key in caplog.text


def test_workbook_without_the_year_column_raises(tmp_path):
    sheet = _capex_sheet(("irena region", "tech")).rename(columns={2024: 2025})
    path = _write(tmp_path / "costs.xlsx", sheet)

    with pytest.raises(ValueError, match="no 2024 column"):
        check_cost_workbook_flags(path)
