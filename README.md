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
python -m economic_complexity.cli --year 2024
```

| Option | Effect |
|---|---|
| `--year YEAR` | BACI year (default 2024) |
| `--product-level N` | HS digits of a product: 2, 4 or 6 (default 6) |
| `--data-dir DIR` | Folder of the BACI file (default `data/baci/`) |
| `--out-dir DIR` | Output folder (default `outputs/`) |
| `--chunksize N` | BACI rows read per chunk (default 1,000,000) |
| `--eigenvalues N` | Eigenvalues computed (default 6) |

- Input: `<data-dir>/BACI_HS22_Y<YEAR>_V202601.csv`. Download it from CEPII (see
  [data/baci/README.md](data/baci/README.md), which also describes the country and product lookup tables).
- Output: `<out-dir>/ECI_PCI_HS<level>_Y<YEAR>.xlsx` with the sheets `ECI_ranking`, `PCI_ranking` and `RCA>=1`.

Method (`src/economic_complexity/`):

1. `rca.compute_rca`: sums the exports per exporter and product. Product codes get their leading zeros back and are
   cut to the product level. `M_cp = 1` where the Balassa RCA is 1 or more.
2. `rca.build_mcp_matrix`: the sparse matrix M (countries x products), diversity `k_c` and ubiquity `k_p`.
3. `complexity.compute_eci` / `compute_pci`: the eigenvectors of `D^-1 A`, with `A = M diag(1/k_p) M^T` for
   countries and `A = M^T diag(1/k_c) M` for products, come from the similar symmetric matrix `D^-1/2 A D^-1/2`.
   ECI and PCI are the z-scored second eigenvector. ECI correlates positively with diversity, PCI negatively with
   ubiquity. A second eigenvalue near 1 (a disconnected graph) gives a warning.

## Levelised cost of methanol (LCOM)

The chain reads a promoted BOA LCOE file and calculates per 0.25 degree pixel: LCOE -> hydrogen (PEM electrolysis)
-> CO2 (direct air capture, DAC) -> methanol (direct CO2 hydrogenation). This is Route B of Wenzel et al.,
*Nature Communications* 17, 8495 (2026), doi [10.1038/s41467-026-76865-x](https://doi.org/10.1038/s41467-026-76865-x).
All process values are the paper's 2050 values, so the chain prices the LCOE year 2050.

### Run

Put the two files of one BOA run in `data/boa/<run>/` (git-ignored):

| File | From the BOA data root |
|---|---|
| `optimal_lcoe_*.nc` | `lcoe-only/<run>/` |
| `boa_cost_data.xlsx` | `costs/<cost_set>/`, with `<cost_set>` the `cost_set` attribute of the LCOE file |

```
uv sync --extra lcom
python -m lcom.cli data/boa/<run>/optimal_lcoe_wy2025_01MWkm2_cov0p9_2025_2060.nc
```

| Option | Effect |
|---|---|
| `--out-dir DIR` | Output folder (default `outputs/`); results go to `DIR/<run>/` |
| `--cost-workbook PATH` | The cost workbook, if it is not next to the LCOE file |
| `--load-factor X` | Plant load factor in (0, 1]; default: the LCOE run's `coverage_fraction` |
| `--wacc X` | One cost of capital for every pixel, in place of the regional WACC |

For each LCOE file the run writes `lcom_<scenario>_y2050[_lf…][_wacc…].nc` (all cost layers in USD, the WACC and the
LCOE), `…_assumptions_register.csv` (one row per input with value, unit and source) and `…_provenance.json`.
It accepts LCOE files of BOA 1.8.0 and of BOA 1.9 or later (`schema_version` 1), and refuses a workbook that
differs from the one that priced the LCOE. BOA's cost cache goes to `cache_costs/` next to the workbook.

### Method

Per pixel, with `e` the LCOE (USD/MWh), `w` the WACC, `CF` the load factor, `p` the DAC relative productivity and
`CRF(w, n) = w / (1 - (1 + w)^-n)`. FOM is an absolute annual cost per unit of capacity.

```
LCOH = e / eta_el + (capex_el * CRF(w, n_el) + fom_el) / (eta_el * 8760 * CF)               USD/MWh_H2
CO2  = (capex_dac * CRF(w, n_dac) + fom_dac) / (CF * p)                                     USD/tCO2
       + e * (elec_dac + heat_dac / COP) / 1000
       + (capex_hp * CRF(w, n_hp) + fom_hp) * heat_dac / (8760 * CF)
LCOM = LCOH / eta_syn + e * elec_syn + co2_syn * CO2                                        USD/MWh_MeOH
       + (capex_syn * CRF(w, n_syn) + fom_syn) / (8760 * CF)
```

- **Pixels:** only pixels with `status == 1`, a positive finite LCOE and a cost key.
- **WACC:** the `Renewables` cost of capital of the pixel's cost key in the LCOE run's cost workbook, for every
  stage. Province keys fall back to the country and unknown keys to the global average, as in BOA.
- **Load factor:** every plant runs `8760 * CF` hours on the firm BOA power.
- **DAC:** CAPEX is per (tCO2/a) of nameplate capacity, so the plant terms divide by the tonnes captured, `CF * p`.
  All DAC heat comes from an air-source heat pump.
- **Units:** lower heating values 33.33 kWh/kg (hydrogen) and 5.528 kWh/kg (methanol).

### Inputs (Route B, 2050)

The paper's values are January 2024 EUR; the chain converts them at **1.08238 USD/EUR** (ECB, 2024 annual
average). FOM is the paper's share of CAPEX.

| Stage | Parameter | Paper value | Value in the chain | Source |
|---|---|---|---|---|
| PEM electrolysis | CAPEX | 553 EUR/kW_el | 598.6 USD/kW_el | Suppl. Table 2 (IEA 2019) |
| | FOM | 2 % of CAPEX | 11.97 USD/kW_el/a | Suppl. Table 2 (IRENA 2018) |
| | Efficiency | 70 % (LHV) | 0.70 | Suppl. Table 2 (IEA 2019) |
| | Lifetime | 20 a | 20 a | Suppl. Table 2 |
| CO2 hydrogenation | CAPEX | 210 EUR/(t_MeOH/a) | 360.2 USD/kW_MeOH | Suppl. Table 2; Suppl. Fig. 2 (plants > 500 kt/a) |
| | FOM | 3.5 % of CAPEX | 12.61 USD/kW_MeOH/a | Suppl. Table 2 |
| | H2 demand | 3.2 mol/mol | 0.8237 kWh_MeOH/kWh_H2 | Methods, Process chain model |
| | CO2 demand | 1.06 mol/mol | 0.26338 kg/kWh_MeOH | Methods |
| | Electricity | 0.72 kWh/kg_MeOH | 0.13025 kWh/kWh_MeOH | Methods |
| | Lifetime | 20 a | 20 a | Suppl. Table 2 |
| DAC | CAPEX | 316 EUR/(tCO2/a) | 342.0 USD/(tCO2/a) | Suppl. Table 2 (Wenzel et al. 2025) |
| | FOM | 4 % of CAPEX | 13.68 USD/(tCO2/a)/a | Suppl. Table 2 |
| | Electricity | 0.4478 MWh/tCO2 | 447.8 kWh/tCO2 | figshare, median of 20,265 regions |
| | Heat (90 C) | 2.0908 MWh/tCO2 | 2,090.8 kWh/tCO2 | figshare, median of 20,265 regions |
| | Relative productivity | median 1.005 | 1 | set by this chain |
| | Lifetime | 20 a | 20 a | Suppl. Table 2 |
| Heat pump (air source) | CAPEX | 665 EUR/kW_th | 719.8 USD/kW_th | Suppl. Table 2 (Grosse et al. 2017) |
| | FOM | 2 % of CAPEX | 14.40 USD/kW_th/a | Suppl. Table 2 |
| | COP | `T_H/(T_H - T_C) * 0.5`, `T_H` = 100 C | 2.298 | figshare, median of 20,265 regions; Methods, Eq. 6 |
| | Lifetime | 25 a | 25 a | Suppl. Table 2 |

Methanol CAPEX per (t/a) becomes CAPEX per kW_MeOH with `8760 h/a / 5527.8 kWh/t`. The figshare DAC values include
the paper's scaling to future energy demand.

### Not modelled

- Waste heat integration and the electric boiler of the paper; weather dependence of DAC energy, productivity and
  heat pump COP.
- Water supply, storage, compression, transport, taxes, stack replacement, degradation, minimum load, regional
  differences in process CAPEX.
- The flexible, off-grid operation of the paper: every plant runs at constant load on firm BOA power.
- The BOA capacity ceiling, when the LCOE file does not apply it: results are then unconstrained optima.
- LCOE years other than 2050.

### Data sources

| Data | Source | Licence |
|---|---|---|
| LCOE, WACC (cost workbook) | Baseload Optimisation Atlas (BOA), [github.com/systemiqofficial/baseload-optimisation-atlas](https://github.com/systemiqofficial/baseload-optimisation-atlas) | Apache-2.0 |
| Process data | Wenzel et al. (2026), main text and Supplementary Information | CC BY 4.0 |
| DAC and heat pump medians | Wenzel et al. (2026), figshare doi 10.6084/m9.figshare.30664253 | CC BY 4.0 |
| EUR/USD rate | ECB data portal, `EXR.A.USD.EUR.SP00.A` | ECB reuse policy |
