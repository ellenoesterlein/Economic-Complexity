# BACI

CEPII BACI, version 202601 (release 2026-01-22): trade flows per year, exporter, importer and product, in the
Harmonized System 6-digit nomenclature (HS22). Values in thousand USD, quantities in metric tons.

Download: <http://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37>. Put the yearly files
`BACI_HS22_Y<year>_V202601.csv` in this folder. Git ignores them (several GB); it tracks only this file and
the two lookup tables.

| Variable | Content |
|---|---|
| `t` | year |
| `i` | exporter |
| `j` | importer |
| `k` | product (HS6 code, stored as an integer without leading zeros) |
| `v` | value |
| `q` | quantity |

Lookup tables of the release (tracked; the code does not read them):

- `country_codes_V202601.csv`: `country_code` (the BACI `i` and `j`), `country_name`, `country_iso2`,
  `country_iso3`.
- `product_codes_HS22_V202601.csv`: `code` (6 digits, the BACI `k`), `description`.

Reference: Gaulier, G. and Zignago, S. (2010). BACI: International Trade Database at the Product-Level. The
1994-2007 Version. CEPII Working Paper, N°2010-23.
