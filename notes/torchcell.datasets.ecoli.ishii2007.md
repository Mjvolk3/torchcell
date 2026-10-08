---
id: j1mdi2o6mmroqtj2psqz1au
title: Ishii2007
desc: ''
updated: 1791496039878
created: 1791496039878
---

## 2026.10.08 - Retrieval, three loaded arms, and the two typed gaps

`torchcell/datasets/ecoli/ishii2007.py`, row 35 of the bacterial schedule
([[plan.bacteria-ontology-genome]]). Three dataset classes, all with
`REFERENCE_STRAIN = "BW25113"` and taking `ecoli_genome`:

| class | experiment | phenotype | records |
|---|---|---|---|
| `MetabolomeIshii2007Dataset` | `BacterialMetaboliteExperiment` | `MetabolitePhenotype` | 24 |
| `ProteomeIshii2007Dataset` | `BacterialProteinAbundanceExperiment` | `ProteinAbundancePhenotype` | 24 |
| `FluxIshii2007Dataset` | `FluxExperiment` | `FluxPhenotype` | 24 |

Adapters `ishii2007_metabolome_adapter.py`, `ishii2007_proteome_adapter.py` and
`ishii2007_flux_adapter.py` with one conf each, and the three cases in
`tests/torchcell/adapters/_bacterial_adapter_cases.py` at positions 13, 14 and 15,
matching `kg_bacteria.yaml`. The flux adapter is the first served consumer of the
`flux phenotype` graph class.

### The publisher is blocked and the paper's own web site is not

The row's accession read "Science supporting online material; no repository accession
found", so the first question was whether anything is scriptable. Measured 2026-10-08 by
`experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py`
(`--network`):

| route | result |
|---|---|
| `science.org/doi/suppl/10.1126/science.1132067` | HTTP 403, Cloudflare JS challenge |
| `science.org/doi/10.1126/science.1132067` | HTTP 403, Cloudflare JS challenge |
| `science.org/doi/pdf/...?ijkey=9Ix5kSaD9ei06&keytype=ref` (the publisher's OWN free-access link, printed on the project web site) | HTTP 403, Cloudflare JS challenge |
| `sciencemag.org/cgi/content/full/1132067/DC1` (the URL the paper prints) | HTTP 403, Cloudflare JS challenge |
| PMC id converter | "Identifier not found in PMC" |
| `http://ecoli.iab.keio.ac.jp/` (the paper's reference 21) | HTTP 200, no challenge |

The `ijkey` link failing the same way is what shows the wall is a bot block at the edge
and not an entitlement, so no credential would help. `scripts/lit_capture_si.py --dry-run`
independently classifies the key `manual` on route `aaas`.

The DATA, though, is not in that supplement. Reference 21 is the Keio "Escherichia coli
Multi-omics Database", v1.0.0 released 2007.05.10, and it serves the whole release over
plain HTTP. Seven files, all HTTP 200, measured by HEAD:

| file | bytes | consumed |
|---|---|---|
| `Quantitative_data.xls` | 824,320 | yes |
| `Flux_GC-MS_data.xls` | 214,016 | yes |
| `Metabolome_UK_data.xls` | 17,547,264 | no, unidentified peaks |
| `DNAArray_data.xls` | 2,789,376 | no, see below |
| `DNAArray_raw-data.zip` | 15,122,608 | no, raw array output |
| `2D-DIGE_ratio_data.xls` | 1,842,688 | no, keyed by gel spot |
| `2D-DIGE_ID_list.xls` | 114,176 | no |

Both consumed files are `RetrievalMethod.direct_url` and were retrieved by their own
recorded retriever into the raw mirror
(`$DATA_ROOT/torchcell-raw/ishiiMultipleHighThroughputAnalyses2007/`). So the row is NOT
retrieval-gated for data. It IS retrieval-gated for the supplement's TEXT, and the
manual recipe for that sits in the mirror's `si_expected`, in the form #788 uses: open
the article in a browser, follow "Supplementary Materials", download the SOM PDF,
deposit under `si/` with `RetrievalMethod.manual_browser` and the sha256 of the bytes
that arrive.

### The flux family needed no schema change, and that is now proven by a build

A sibling agent established that a 13C MFA dataset needs no schema change. Confirmed
end to end here: `FluxPhenotype` / `FluxExperiment` / `FluxExperimentReference` all
existed with no consumer, `flux phenotype` was already declared in
`biocypher/config/torchcell_schema_config.yaml`, and `cell_adapter.py` already had
`_flux_phenotype_node` and `_get_flux_phenotype_reference_nodes`. Nothing in
`schema.py`, the schema config, the Biolink head ontology or the adapter base changed.
**No new graph node class, so no Biolink parent to choose, and nothing that could force
a knowledge-graph rebuild.**

The release publishes ONE number per reaction and no interval at all, so
`net_flux_lower`, `net_flux_upper` and `confidence_level` are all `None` with a typed
`ProvenanceGap` on `confidence_level`, and `label_statistic_name` stays `None`. The fit's
INPUT is mirrored as the second workbook ("Mass distributions of proteinogenic amino
acids measured by GC-MS.", one sheet per culture), and `_check_fit_inputs` refuses a
record whose mass-distribution sheet is absent, so a stored fitted flux always names the
data it was fitted to. The fitting PROCEDURE is in the unmirrored supplement.

Seven `Exch.` rows are exchange (reversibility) coefficients, not fluxes. They are
written to `preprocess/exchange_coefficients.csv` and never reach `net_flux`; the sheet
says so in as many words ("\"Exch\" denotes exchange coefficient of corresponding
reaction."). A `-` is "reaction excluded from model" and is stored as KEY ABSENCE: the
zwf record, hand-checked in the tests, has no `G6P -> 6PG` key at all, because that is
the reaction its own deletion removes.

### The medium is the one unpinned environment value

No mirrored byte names the medium. The paper says "in glucose-limited chemostat
cultures" and nothing more; the project web site's workbooks state units, replicate
structure and the cell volume and dry weight used to convert to mM, but no recipe. So:

- `MEDIUM` is built in the loader module, NOT added to `MEDIA_LIBRARY`, as one
  `composition_deferred` component whose `defers_to` names the unmirrored supplement.
  The library holds recipes; this object holds the absence of one.
- `is_synthetic=True` is **a reading, not a statement**: glucose being the sole
  growth-limiting substrate whose concentration the dilution rate sets requires a
  defined medium, since another carbon source would relieve the limitation. It is
  recorded in `preprocess/build_accounting.json` under `unpinned_environment_values`,
  the way the Rachwalski loader records its one unpinned medium.
- `temperature=None` with a typed `ProvenanceGap`, not a guessed 37 C.
- `aerobicity="aerobic"` is NOT the field default left in place. `check_aerobic` asserts
  at build time that every loaded culture has a positive oxygen uptake rate in the
  release's own `Specific_Rates` sheet (minimum over the 24 loaded: 1.06 mmol/gDCW/h).

There is deliberately no `ProvenanceGap` on `media` itself: `ProvenanceGapMixin` refuses
a gap on a field that holds a value, which is the right rule and is asserted in a test.

### Records, and the arithmetic of what is dropped

Each sheet's sample columns are classified in one fixed order, so every dropped column
carries exactly one reason:

| rule | metabolite | protein | flux |
|---|---|---|---|
| `no_data_in_this_layer` | GR04x | GR04x, KO05x | GR04x, KO05x |
| `reference_sample` | 5 RF columns | 6 RF columns | 4 RF columns |
| `culture_not_batch` | GR01-GR04 | GR01-GR04 | GR01-GR04 |
| `duplicate_culture_same_genotype_and_environment` | KO05x | - | - |
| **kept** | **24** | **24** | **24** |

35 - 1 - 5 - 4 - 1 = 24; 36 - 2 - 6 - 4 = 24; 34 - 2 - 4 - 4 = 24.

`culture_not_batch` is the landed rule name: the wild type at 0.1, 0.4, 0.5 and 0.7 h-1
differs from the reference and from itself ONLY in dilution rate, and `Environment` has
no dilution-rate slot (issue #753, and `PhysicalFactor` has no member for it either).
Four real cultures per layer are therefore not served, and that is the cost of #753 on
this row.

pfkA is the one disruptant cultured twice ("pfkA disruptant was cultured twice."), as
KO05x (`pfkA_1`) and KO05 (`pfkA_2`). Both are pfkA at 0.2 h-1, so they would collide on
(genotype, environment). `PREFERRED_DUPLICATE_SAMPLES = {"KO05"}` names the keeper
explicitly rather than letting column order decide, because column order would have kept
KO05x: KO05 is the pfkA culture measured in ALL THREE served layers, which is what keeps
the three datasets' pfkA records the same culture. KO05x has metabolite and mRNA data
only and no oxygen uptake rate.

`tpiA` is a measured absence worth recording: the Information sheet says it "could not
be cultured because of reactor wash out at a dilution rate of 0.2h-1", so the panel's 24
is 25 attempted.

### The reference is series-matched where the release states a series

The workbook labels every sample with a measurement series and labels the trailing
column block "Control (Wild type, cultured at a dilution rate of 0.2h-1)". So each
metabolite and protein record's `experiment_reference` is the reference column of its
OWN series, not a pooled average: series 1-5 map to RF02, RF03, RF04, RF05, RF06 in the
Metabolite sheet, and series 1-6 to RF02 plus five RF03 columns in the Protein sheet.
The Protein sheet repeats the Sample ID `RF03` across five series columns, which is why
the drop ledger names columns by a per-COLUMN label (`RF03@BN`) and not by sample id.

The Flux sheet carries no series row, so that arm stores ONE reference fit, `RF03` --
the one reference culture measured in all three served layers. **This is a documented
representative choice and is flagged for review.** Averaging the four released reference
fits was rejected on principle: an average of four separate network fits is not a fit.
The other three are written whole to `preprocess/flux_reference_fits.csv`.

### Two reference restrictions, both forced by an L3 invariant, both measured

Each culture detects its own set of species, so a record and its reference do not key
identically.

- Metabolite: `verify_metabolite_dataset`'s L3 wants the reference keys to be a SUBSET
  of the record's. The reference is restricted to the metabolites the record also
  detected. **Nothing is lost on the record side** (all 3,433 stored values kept); 319
  reference baselines are dropped, and each of those had no record key to baseline.
- Protein: `verify_protein_dataset`'s L3 wants EQUALITY, so each record stores the
  proteins its own series reference also detected. Measured: **1 abundance of 1,349** is
  dropped on the record side (and 18 reference baselines), written to
  `preprocess/abundances_without_reference.json`.
  `EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE = 1` is pinned so a corrected re-export that
  drops more stops the build.

A blank cell is "not detected" per the Information sheet, so neither side's drop is read
as a zero.

### Sourcing and identifiers

Every statistic carries a verbatim quote plus the sha256 and section of the file it came
from, audited at verification time; 24 `SOURCED_VALUES`, ten against the Ishii `paper.md`
(sha256 `1d30256f408a939cf1cf400c03f36a124a8274627393602cabcb92cf72394065`), two against
Baba 2006, and the rest against the `Information` sheet of the pinned workbook.

- **Background strain**, through the deferral chain Ishii -> "(13)" -> Baba 2006
  (mirrored): Ishii says only "E. coli K-12"; Baba says "E. coli K-12 strain BW25113".
  Records pin `ecoli_K12_BW25113_ASM75055v1` and the `ecoli_k12_bw25113_locus_tag`
  namespace.
- **Protein replicate count**, the one range that needed resolving. The sheet says the
  CV captures "duplicate sample preparation and independent LC-MS/MS measurements" and
  separately that 1, 2 or 3 peptides per protein were used, which could put the CV over
  as many as six values. No per-record peptide column exists, so the
  **conservative lower end** was taken: n = 2, the larger SE, per the range-resolution
  rule. `SE = level * CV / 100 / sqrt(2)`, and `nan` where the sheet reports a level and
  no CV. The peptide counts go to `preprocess/protein_peptides.csv`; a peptide count is
  not a replicate count.
- **Identifiers, measured.** 24 of 24 disruptant symbols resolve to BW25113 locus tags
  (23 through the gene-symbol layer, `gapC` through a gene synonym onto a non-gene
  feature), no collision, no ambiguity. 66 of the 67 quantified protein symbols resolve;
  `gpmG` is retired on this annotation and carries NO value in any of the 36 Protein
  columns, so it never becomes a key. `check_gpmg_unused` asserts that emptiness rather
  than trusting it. 58 distinct locus tags are stored as abundance keys across the 24
  records.
- **The CE-TOFMS protocol is read from the workbook, not hardcoded.** The Metabolite
  sheet encodes anion / cation / nucleotide as the FILL COLOUR of the name cell and the
  Information sheet prints the legend as three coloured swatches. The loader reads the
  legend and refuses a workbook whose protocol row counts are not 204 / 314 / 61. This
  matters because exactly one name, `Citrate`, is released twice (anion, with no values;
  nucleotide, with values), and a bare-name key would let one row overwrite the other.
  Both are stored as `Citrate (anion)` / `Citrate (nucleotide)`.
- **An unexplained marker, recorded not resolved.** Three metabolite columns carry series
  `1*` where every other column carries a bare number, and the workbook prints no
  footnote for the asterisk. It is stripped for series matching and kept verbatim on the
  `SampleColumn`.

### Verification, L0 to L4

`python -m torchcell.datasets.ecoli.ishii2007 verify --dataset <slug>`, all three PASS:

| level | metabolome | proteome | flux |
|---|---|---|---|
| L0 structural | 24 records | 24 records | 24 records |
| L1 count | 24 = 24 | 24 = 24 | 24 = 24 |
| L1 uniqueness | 24 unique strains | 24 unique ORFs | (no family rule) |
| L2 value fidelity | 3,433 values | 1,348 values | 1,010 values |
| L2 se non-negative | 0 (no SE released) | 1,291 values | n/a |
| L3 reference | subset for 2,820 values | key-matched for 1,348 | interval not half-stated |
| L3 measurement type | single | single | single |
| L3 provenance audit | 24 quotes | 24 quotes | 24 quotes |
| L4 containment | 24 of 24 loci | 64 of 64 identifiers | 24 of 24 loci |

`metabolome_ishii2007` is registered in `METABOLITE_DATASETS` and `proteome_ishii2007` in
`BACTERIAL_PROTEIN_ABUNDANCE_DATASETS`, so `run_all` reaches them. The flux arm has no
family runner (no flux dataset has ever been served), so its L0-L3 rows come from the
shared `levels.py` helpers plus a flux-specific L3 that asserts no record carries a
half-stated interval.

### What is NOT loaded, and why

- **qRT-PCR mRNA (85 transcripts, copy number per ug total RNA).** A typed SCHEMA GAP,
  not a retrieval problem: the bytes are mirrored and the layer would serve 24 records.
  No phenotype holds an absolute transcript abundance. `MicroarrayExpressionPhenotype`
  stores `log2(sample/reference)`, `RNASeqExpressionPhenotype` requires TPM AND raw
  mapped-read counts, and `PseudobulkExpressionPhenotype` is a log2 ratio with a cell
  count. Filing qRT-PCR copy numbers under any of them would misreport the assay. The
  honest fix is a new additive `TranscriptAbundancePhenotype` plus its experiment pair
  and graph class, which is a separate change.
- **Genome-wide DNA array (4,213 oligos, seven sample-vs-control Control/Sample/Ratio
  blocks: pgm, pgi, gapC, zwf, rpe, WT 0.5, WT 0.7).** This one WOULD fit
  `MicroarrayExpressionPhenotype`, but there is no `BacterialMicroarrayExpressionExperiment`
  pair to carry the assembly pin, and the two WT arms would drop under
  `culture_not_batch` anyway, leaving five records.
- **2D-DIGE** ratios are keyed by gel spot rather than by protein, and the paper itself
  calls that layer semiquantitative.
- **Unknown-peak CE-TOFMS matrices** have no metabolite identity to key on.
- **DNA-array raw data** is raw instrument output.

### Things worth knowing next time

- The release is a legacy BIFF `.xls` and nothing in the environment WRITES that format
  (`xlwt` is absent). The hermetic test therefore fakes the narrow `xlrd` surface the
  loader reads, fill colours included, rather than skipping the colour logic.
- `self.name` on an `ExperimentDataset` is the CLASS name and is what lands on
  `dataset_name`; the dev-tree slug is separate. These classes carry an explicit
  `SLUG` ClassVar because the record counts and roots are keyed by slug.

Generating script for every number above:
`experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py`,
results in `experiments/036-dataset-fixes-before-kg-build/results/ishii2007_release_inventory.json`.
