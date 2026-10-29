import logging
from pathlib import Path

import pandas as pd
import pytest

from lcom import paper_inputs
from lcom.flags import (
    ALL_FLAGS,
    CHAIN_FLAGS,
    COST_WORKBOOK_FLAGS,
    LCOE_CEILING_NOT_APPLIED,
    STANDING_FLAGS,
    check_cost_workbook_flags,
    lcoe_file_flags,
    log_flags,
)


def _capex_sheet(columns: tuple[str, str], overrides: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per flag; ``overrides`` maps a flag key to another workbook value."""
    overrides = overrides or {}
    rows = []
    for flag in COST_WORKBOOK_FLAGS:
        rows.append([flag.region, flag.subregion, flag.tech, overrides.get(flag.key, flag.workbook_value)])
    return pd.DataFrame(rows, columns=[columns[0], "subregion code", columns[1], 2024])


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
    path = _write(tmp_path / "costs.xlsx", _capex_sheet(("irena region", "tech"), {"capex.china.solar": 472.0}))

    results = check_cost_workbook_flags(path)
    with caplog.at_level(logging.WARNING):
        log_flags(results)

    stale = [r for r in results if not r.unchanged]
    assert [r.flag.key for r in stale] == ["capex.china.solar"]
    assert "DATA FLAG STALE" in caplog.text


def test_subregion_rows_do_not_hide_the_region_row(tmp_path):
    sheet = _capex_sheet(("irena region", "tech"))
    other_province = pd.DataFrame([["China", "CHN:CN-GS", "Solar PV", 450.5]], columns=sheet.columns)
    path = _write(tmp_path / "costs.xlsx", pd.concat([sheet, other_province], ignore_index=True))

    results = check_cost_workbook_flags(path)

    assert len(results) == len(COST_WORKBOOK_FLAGS)
    assert all(r.unchanged for r in results)


@pytest.mark.parametrize("drop_column", [True, False])
def test_subregion_flag_is_left_out_for_a_workbook_without_subregions(tmp_path, drop_column):
    sheet = _capex_sheet(("irena region", "tech"))
    sheet = sheet[sheet["subregion code"].isna()]
    if drop_column:
        sheet = sheet.drop(columns="subregion code")
    path = _write(tmp_path / "costs.xlsx", sheet)

    results = check_cost_workbook_flags(path)

    assert "capex.china.provinces" not in [r.flag.key for r in results]
    assert all(r.unchanged for r in results)


@pytest.mark.parametrize(
    ("attrs", "flagged"),
    [({"capacity_ceiling_applied": 0}, True), ({}, True), ({"capacity_ceiling_applied": 1}, False)],
)
def test_capacity_ceiling_flag_follows_the_lcoe_file(attrs, flagged, caplog):
    flags = lcoe_file_flags(attrs)
    with caplog.at_level(logging.WARNING):
        log_flags([], flags)

    assert (LCOE_CEILING_NOT_APPLIED in flags) is flagged
    assert (LCOE_CEILING_NOT_APPLIED.key in caplog.text) is flagged


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
