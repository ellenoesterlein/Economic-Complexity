# Economic-Complexity

This repository has two parts:

- **Economic complexity**: revealed comparative advantage (RCA), Economic Complexity Index (ECI) and Product
  Complexity Index (PCI) from CEPII BACI trade data.
- **Levelised cost of methanol (LCOM)**: hydrogen, CO2 (direct air capture) and methanol cost per 0.25 degree pixel,
  calculated from the levelised cost of electricity (LCOE) of the Baseload Optimisation Atlas (BOA).

## Installation

Python 3.13 or later. With [uv](https://docs.astral.sh/uv/):

```
uv sync                 # economic complexity only
uv sync --extra lcom    # also the LCOM chain (installs BOA from GitHub)
```

## Economic complexity (ECI, PCI)

```
python "workflow/scripts/PCI, ECI, RCA.py"
```

- Input: `workflow/scripts/data/BACI_HS22_Y<YEAR>_V202601.csv`. Download it from CEPII (see
  `workflow/Readme.txt`). The year is set in `Config.YEAR` in the script.
- Output: `workflow/scripts/output/ECI_PCI_HS<level>_Y<YEAR>.xlsx` with the sheets `ECI_ranking`, `PCI_ranking`
  and `RCA>=1`.

## Levelised cost of methanol (LCOM)

The chain reads a promoted BOA LCOE file and calculates per pixel: LCOE -> hydrogen (PEM electrolysis) -> CO2
(direct air capture) -> methanol (CO2 hydrogenation). Process data: Wenzel et al., *Nature Communications* 17, 8495
(2026), doi [10.1038/s41467-026-76865-x](https://doi.org/10.1038/s41467-026-76865-x).

Set `BOA_DATA_ROOT` to the BOA data folder that holds the LCOE files and the cost workbooks.
