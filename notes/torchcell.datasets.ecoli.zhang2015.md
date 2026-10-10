---
id: unvjhmpren980xhfu2jnh4b
title: Zhang2015
desc: ''
updated: 1791614203318
created: 1791614203318
---

## 2026.10.10 - CeCaFDB settled: a flux compendium NOT admitted (row 54)

Source: `torchcell/datasets/ecoli/zhang2015.py` (no dataset class; a provenance record).
Tests: `tests/torchcell/datasets/ecoli/test_zhang2015.py` (24 hermetic, 3 `--data`).
Measurement: `experiments/036-dataset-fixes-before-kg-build/scripts/cecafdb2015_release_inventory.py`,
results beside it as `cecafdb2015_release_inventory.json`, `cecafdb2015_workbooks.csv`
and `cecafdb2015_accession_probe.json`.
Schedule row: `experiments/database/scripts/build_bacteria_candidate_datasets_table.py`
under `name="CeCaFDB flux compendium"`, E. coli, tier 3, `status="aggregation"`.

Zhang et al., "CeCaFDB: a curated database for the documentation, visualization and
comparative analysis of central carbon metabolic flux distributions explored by
13C-fluxomics", Nucleic Acids Res. 43:D549-D557, doi:10.1093/nar/gku1137, PMC4383945,
citation key `zhangCeCaFDBCuratedDatabase2015`. Not in Zotero and not in the literature
mirror; nothing was added to Zotero.

**DECISION: NOT LOADED.** No dataset class, adapter, conf, graph class or schema change.
The deliverable is the raw-mirror deposit, the measurement, and this record, in the
MCF2Chem 2023 pattern ([[torchcell.datasets.ecoli.cai2023]]).

### Mirror route: raw-mirror deposit

`$DATA_ROOT/torchcell-raw/zhangCeCaFDBCuratedDatabase2015/` with `manifest.json`, 36 files:

| path | retrieval | sha256 |
|---|---|---|
| `paper/PMC4383945.1.txt` | `pmc_cloud_object("PMC4383945.1/PMC4383945.1.txt")` | `c6bd5a73...` |
| `paper/PMC4383945.1.pdf` | `pmc_cloud_object("PMC4383945.1/PMC4383945.1.pdf")` | `e476a508...` |
| `data/download_action.html` | `direct_url("http://www.cecafdb.org/download_action")` | `58825a49...` |
| `data/xls/<file>.xls` x 33 | `direct_url("http://www.cecafdb.org/download_files/<file>")` | pinned per file in `WORKBOOKS` |

Only the 33 workbooks of the two schedule hosts are deposited; the other 85 (34
organisms) are out of scope and listed in the manifest's `si_expected`. Every retrieval
was re-run end to end into a scratch directory by `retrieve_release()` on 2026-10-10 and
all 36 hashes matched. The workbook server reports `Last-Modified: Sun, 12 Apr 2015
06:54:22 GMT`, so the files have not changed since the release.

The accession answers over plain http (probe 2026-10-10: `/` 200, `/download_action` 200).
Over https the certificate does not verify; with verification off the host serves a
different application, CeCaFLUX. The schedule's "expired-TLS error on 2026-09-24" is that
https failure; the http site is live.

### What the release is, measured

| quantity | E. coli | P. putida |
|---|---|---|
| references (workbooks) | 31 | 2 |
| cases (flux maps) | 297 | 3 |
| numeric flux cells | 8,144 | 88 |
| `X`/`x` placeholder cells | 717 | 6 |
| single-case workbooks | 5 | 1 |
| cases whose `Strains` cell names a genomes-tier strain | 33 | 1 |

All 33 workbooks: one sheet, the same `V1.0F` template, one unit (`relative flux`), 0
columns beyond the per-case values, 0 cells with an uncertainty word. 118 references on
the Download page, one workbook each. Every workbook's `Experiment Name (ID)` equals the
reference the Download page files it under (`all_attributed = true`).

The paper's Table 1 row says 32 E. coli references (`Escherichia coli\t32\t297\t297`);
the release links 31, and the site's own statistics table says 31. The case count, 297,
agrees. The P. putida row (`Pseudomonas putida\t2\t3\t3`) agrees with the release.

### The four questions

**Per-source attribution: YES.** One workbook per reference, the reference named inside
it, the source DOI resolved through PubMed for all 33 (26 by unique title match, 7 by
hand-picked PMID confirmed with `esummary`), pinned in `WORKBOOKS`.

**Per-reaction values with intervals: NO.** 0 of 33 workbooks carry an interval column or
an uncertainty term, and the PMC full text contains 0 occurrences of "standard
deviation", "confidence" or "uncertainty". The values are also not the source's own
numbers:

```
The lumped reactions in these studies were broken down into their original forms, as in the KEGG Reaction Database (36), and the flux value was mapped to its precisely corresponding reaction.
```

```
The ‘Flux value’, the core of the database, displays the quantity of the flux value relative to the substrate uptake rate. In the case of multiple substrates, the sum of all the substrates uptake rates is set to 100.
```

So the reaction set is the curators' KEGG rendering and the scale is the curators'
renormalization. The meaning of the 723 `X`/`x` cells is not stated anywhere in the
release; they are counted, not interpreted.

**Hosts: 31 E. coli references, 2 P. putida.** 190 of the 297 E. coli cases (64.0%) are
one reference, Haverkorn van Rijsewijk et al. 2011 (Mol Syst Biol 7:477,
doi:10.1038/msb.2011.9). Its `Strains` row is a 190-long consecutive integer run,
`Escherichia coli BW25113` through `Escherichia coli BW25302`, while its `Genotype` row
gives every case the BW25113 genotype string with a regulator name appended (95 distinct
genotypes over glucose and galactose). The strain label does not name the strain its own
genotype row describes. Overall only 33 of 297 E. coli cases name a genomes-tier strain
(MG1655, BW25113, W3110, REL606); others name JM101, ML308, K1060, JCL1225 and similar
strains with no assembly in the tier.

**Can `FluxPhenotype` carry them honestly: NO.** Field by field (`FLUX_FIELD_FIT`):

| field | carried | measured |
|---|---|---|
| `FluxPhenotype.net_flux` | yes | per-case numbers keyed by KEGG code |
| `FluxPhenotype.net_flux_lower` / `net_flux_upper` | no | 0 of 33 workbooks |
| `FluxPhenotype.confidence_level` | no | 0 cells, 0 paper words |
| `FluxPhenotype.measurement_type` | yes | `relative flux`, uptake = 100 |
| `FluxExperimentReference.phenotype_reference` | no | 6 single-case workbooks have no parent map |
| `FluxExperiment.genotype` | no | 33 of 297 E. coli cases name a tier strain; Haverkorn labels are a fill run |

The schema would accept a point map with a typed interval gap, so the refusal is not a
schema limit. It is the same rule MCF2Chem was settled on: the landed aggregations (Lim
2022, Borchert 2024) store numbers the aggregator recomputed from raw data, and CeCaFDB
transcribes, renormalizes and re-maps published numbers. The Borchert 2024 pattern (name
the originating study per record) would be satisfiable here, but it would name a study
whose own release is where the interval and the genotype live, at which point the source
paper is the loadable row. The Li 2021 refusal (#800) applies on top for the six
single-case workbooks.

### Duplication, measured

- 33 source DOIs against every DOI named anywhere in the 50 bacterial loader modules that
  name one (74 DOIs, a superset of the served papers'): **0 shared**. Ishii 2007, the one
  served E. coli flux map, is not a CeCaFDB reference.
- Against the bacteria candidate table: **1 shared**, `10.1038/msb.2011.9`, schedule row 69
  "Haverkorn van Rijsewijk 2011" (Supplementary Tables 2 and 3). The 64% of CeCaFDB's
  E. coli cases that come from it should be loaded from row 69's own release, not from
  this transcription.

Verdict: **subsumed by its sources**, with no net-new measurement to load.

### Verification

L0 (deposit hashes) and L3 (provenance audit) are the levels a record without a store
has; L1, L2 and L4 have no store to read.

| level | check | result |
|---|---|---|
| L0 | 36 deposited files re-hashed by `release_inventory()` and `retrieve_release()` | 36/36 match |
| L3 | 9 `SourcedValue` quotes audited against `paper/PMC4383945.1.txt` | 9/9 pass |
| L1/L2/L4 | no LMDB is built | not applicable |

Schema impact (`scripts/schema_impact_check.py --base origin/main`): "No schema contract
changes vs origin/main."
