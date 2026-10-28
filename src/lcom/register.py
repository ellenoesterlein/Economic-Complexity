"""
The assumptions register and provenance record written beside every result.

``<stem>_assumptions_register.csv`` has one row per input the run used. ``<stem>_provenance.json``
holds the run-level facts: which LCOE file and cost workbook the result rests on, the process data,
the currency rate, the load factor and WACC basis, and every data flag.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from lcom.flags import CHAIN_FLAGS, STANDING_FLAGS, FlagResult
from lcom.paper_inputs import ChainInput

REGISTER_COLUMNS = ["stage", "parameter", "year", "value", "unit", "source", "status", "basis", "data_flags"]


def input_rows(inputs: dict[tuple[str, str], ChainInput], year: int) -> list[dict[str, Any]]:
    """Register rows for the process inputs, in the units the chain used."""
    return [
        {
            "stage": i.stage,
            "parameter": i.parameter,
            "year": year,
            "value": i.value,
            "unit": i.unit,
            "source": i.source,
            "status": i.status,
            "basis": i.basis,
            "data_flags": ";".join(i.flags),
        }
        for i in inputs.values()
    ]


def write_register(rows: list[dict[str, Any]], path: Path) -> None:
    pd.DataFrame(rows, columns=REGISTER_COLUMNS).to_csv(path, index=False, encoding="utf-8")


def flag_records(results: list[FlagResult]) -> list[dict[str, Any]]:
    records = []
    for result in results:
        record = {k: v for k, v in asdict(result.flag).items() if v is not None and v != ""}
        record.update(found_in_workbook=result.found, review_still_holds=result.unchanged)
        records.append(record)
    for flag in (*STANDING_FLAGS, *CHAIN_FLAGS):
        records.append({k: v for k, v in asdict(flag).items() if v is not None and v != ""})
    return records


def write_provenance(provenance: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
