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
from, audited at verification time; 23 `SOURCED_VALUES`: eight against the Ishii `paper.md`
(sha256 `1d30256f408a939cf1cf400c03f36a124a8274627393602cabcb92cf72394065`), two against
Baba 2006, and thirteen against the `Information` sheet of the pinned workbook.

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
| L3 provenance audit | 23 quotes | 23 quotes | 23 quotes |
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

## 2026.10.09 - #753 names the dilution rate, so the wild-type arm is served

`Environment.dilution_rate_per_hour` landed with issue #753 (validated finite and
strictly positive or absent). Every environment this loader builds now sets it, the
`culture_not_batch` drop rule is retired, and all three arms go from **24 to 28
records**: the 24 Keio disruptants at 0.2 h-1 plus the wild type at 0.1, 0.4, 0.5 and
0.7 h-1.

### Which column is which rate, measured instead of inferred from GR01..GR04

The `GR` numbering says nothing about the rate, so it is not what the loader reads. The
release writes the rate into the Sample Name, in two places that must agree:

| sample id | `IDs` sheet Sample Name | data sheets' name row | rate (h-1) |
|---|---|---|---|
| `GR01` | `WT, 0.1h-1` | `WT, 0.1h-1` | 0.1 |
| `GR02` | `WT, 0.4h-1` | `WT, 0.4h-1` | 0.4 |
| `GR03` | `WT, 0.5h-1` | `WT, 0.5h-1` | 0.5 |
| `GR04` | `WT, 0.7h-1` | `WT, 0.7h-1` | 0.7 |
| `GR04x` | `WT, 0.7h-1` | blank in all three served sheets | 0.7 |
| `RF01`-`RF08` | `WT, 0.2h-1` | `WT(Mar)`, `WT(Jun)`, ... | 0.2 |

The ordering does happen to be `0.1, 0.4, 0.5, 0.7`, but that is a measurement, not an
assumption: `check_dilution_rate_arm` reads the `IDs` roster, refuses a data sheet whose
own name row contradicts it, and refuses a parsed rate set that is not the paper's arm
minus the 0.2 h-1 reference rate. `GR04x`'s name cell is blank in the Protein,
Metabolite and Flux sheets, so for that column the roster is the only label, which is why
the roster is the source and the data sheet the cross-check.

### The quotes, with their sha256

Ishii `paper.md`, sha256
`1d30256f408a939cf1cf400c03f36a124a8274627393602cabcb92cf72394065`:

- the arm, all five rates (`SOURCED_VALUES["dilution_rate_arm"]`): "To allow a
  comparison of the effects of these genetic perturbations with the effects of
  environmental perturbations, wild-type cells were examined at several different
  dilution rates (0.1, 0.2, 0.4, 0.5, and $0 . 7 \ \mathrm { h o u r s } ^ { - 1 }$ ."
- the reference rate (`SOURCED_VALUES["dilution_rate_per_hour"]`): "The cells were grown
  at a single fixed dilution rate of 0.2 hours−1 in glucose-limited chemostat cultures,
  and wildtype cells cultured at the same specific growth rate were used as a reference
  sample for comparison."
- why the rate is an environment and not a protocol detail: "In chemostat cultures, the
  concentration of growth-limiting substrate can be controlled by the dilution rate $( I
  4 )$ . the dilution rate was thus varied in this study from an almost glucose-starved
  state to a nearly unlimited glucose supply."

`Quantitative_data.xls`, sha256
`2b7663f505af2137d31697a4d316eb1f7ffd24274f73e4844d27d03348313888`:

- `Information` sheet, the Samples table, which assigns the GR block and the control
  block to the same sheet: "Wild type, cultured at various dilution rates" (against
  "Column BA-AJ" for mRNA and Protein, "Column AB-AF" for the other layers) and
  "Control (Wild type, cultured at a dilution rate of 0.2h-1)" (against "Column BL-" and
  "Column AH-").
- `IDs` sheet, the Memo of the one column that is a second culture at an already-served
  rate (`SOURCED_VALUES["gr04x_second_mrna_measurement"]`, new): "Used for 2nd
  measurement of mRNAs."

### The reference for a dilution-rate record is the 0.2 h-1 wild type, and it is honest

The decision was: is there a legitimate reference for a wild-type culture at 0.1 h-1, or
should these cultures be refused? **There is one, and the release names it.** The
Samples table gives every sheet ONE control block, "Control (Wild type, cultured at a
dilution rate of 0.2h-1)", and it covers the GR block as well as the disruptants. The
`IDs` sheet then puts each GR column in a measurement series that has an `RF` column, so
the series match the other 24 records use resolves for these four too:

| arm | GR01 | GR02 | GR03 | GR04 |
|---|---|---|---|---|
| Metabolite (series 5, 5, 5, 4) | RF06 | RF06 | RF06 | RF05 |
| Protein (series 4, 4, 4, 4) | RF03 | RF03 | RF03 | RF03 |
| Flux (no series row) | RF03 | RF03 | RF03 | RF03 |

So no denominator is invented. It IS a **cross-environment reference**, and the stored
bytes say so rather than hiding it: the record's `environment` carries its own rate while
its `environment_reference` carries 0.2 h-1. That is exactly the comparison the paper
asks for ("To allow a comparison of the effects of these genetic perturbations with the
effects of environmental perturbations"). Nothing was refused on this row.

### The empty genotype does not collide, measured

A wild-type record carries `Genotype(perturbations=[])`, so four records per arm share
one genotype. The adapter's content-addressed experiment id is the sha256 of the whole
serialized experiment, environment included (`cell_adapter._experiment_node`), so the
dilution rate is what separates them. Measured on the built stores: **28 distinct ids
over 28 records in each of the three arms**, with four records carrying no perturbation
and rates `[0.1, 0.4, 0.5, 0.7]`. There is no fifth wild-type experiment to collide
with, because the 0.2 h-1 wild type is only ever an `ExperimentReference`.

`verify_metabolite_dataset`'s L1 `genotype_uniqueness` keyed on the strain alone, which
four identical empty genotypes fail. It now takes `environment_keyed=True` (additive,
default False, threaded into `_l1_orf_uniqueness` as `per_environment`) and keys on
(strain, environment), which is what a record of a genotype IN an environment means. For
a dataset whose records share one environment the key is the strain plus a constant, so
no other dataset's verdict or unique-key count changes (asserted in
`tests/torchcell/verification/test_metabolite_verification.py`). The protein verifier
needed nothing: its L1 iterates deleted ORFs, and an empty genotype contributes none.

### The recomputed column arithmetic

Regenerated by
`experiments/036-dataset-fixes-before-kg-build/scripts/ishii2007_release_inventory.py`
into `.../results/ishii2007_release_inventory.json`, from the loader's own
`classify_columns`:

| layer | sample columns | disruptant / dilution / reference | empty | reference | duplicate | kept |
|---|---|---|---|---|---|---|
| Metabolite | 35 | 25 / 5 / 5 | 1 (`GR04x`) | 5 | 1 (`KO05x`) | **28** |
| Protein | 36 | 25 / 5 / 6 | 2 (`GR04x`, `KO05x`) | 6 | 0 | **28** |
| Flux | 34 | 25 / 5 / 4 | 2 (`GR04x`, `KO05x`) | 4 | 0 | **28** |
| mRNA (not served) | 45 | 25 / 5 / 15 | 0 | 15 | 2 (`KO05x`, `GR04x`) | 28 |

35 - 1 - 5 - 1 = 28; 36 - 2 - 6 = 28; 34 - 2 - 4 = 28. The old line read
"35 - 1 - 5 - 4 - 1 = 24; 36 - 2 - 6 - 4 = 24; 34 - 2 - 4 - 4 = 24"; the four
dilution-rate columns are no longer a subtrahend.

`GR04x` stays dropped under `no_data_in_this_layer` in all three served sheets, and the
`pfkA` duplicate rule is untouched (`KO05` is still the kept pfkA culture in all three
arms, `pfkA_2`, at 0.2 h-1).

**A second duplicate pair surfaced, and it is in the unserved layer.** `GR04` and
`GR04x` are two wild-type cultures at ONE rate (Culture Date 38693 against 38631, with
`GR04x`'s Memo "Used for 2nd measurement of mRNAs."), so with the rate in the identity
they are a `(genotype, dilution rate)` collision exactly like the pfkA pair. In the three
served sheets `GR04x` is empty and never reaches the duplicate rule; in the mRNA layer
both carry data, so `PREFERRED_DUPLICATE_SAMPLES` now names `GR04` alongside `KO05`, for
the same reason (`GR04` is the culture measured in all three served layers). This changes
nothing in the served arms and makes the mRNA layer measurable.

### Record counts, before and after

| dataset | before | after | wild-type records | rates served |
|---|---|---|---|---|
| `metabolome_ishii2007` | 24 | **28** | 4 | 0.1, 0.2, 0.4, 0.5, 0.7 |
| `proteome_ishii2007` | 24 | **28** | 4 | 0.1, 0.2, 0.4, 0.5, 0.7 |
| `flux_ishii2007` | 24 | **28** | 4 | 0.1, 0.2, 0.4, 0.5, 0.7 |

Rebuilt with `python -m torchcell.database.build_dataset_lmdb --dataset <Class>
--retire-existing`; `--list-stale --include-private` names none of the three afterwards.
What the four new records hold:

| rate (h-1) | metabolites (reference) | proteins | reactions |
|---|---|---|---|
| 0.1 | 189 (124) | 57 | 42 |
| 0.4 | 157 (123) | 56 | 40 |
| 0.5 | 135 (123) | 57 | 40 |
| 0.7 | 126 (107) | 56 | 40 |

Side effects, all measured: distinct stored protein keys 58 -> 59; metabolome reference
baselines dropped (metabolites the record did not detect) 319 -> 345; protein reference
baselines dropped 18 -> 20; `EXPECTED_PROTEIN_KEYS_WITHOUT_REFERENCE` **unchanged at 1**
(still `KO19`'s `BW25113_4090`).

### Verification, L0 to L4, after the rebuild

`python -m torchcell.datasets.ecoli.ishii2007 verify --dataset <slug>`, all three PASS:

| level | metabolome | proteome | flux |
|---|---|---|---|
| L0 structural | 28 records | 28 records | 28 records |
| L1 count | 28 = 28 | 28 = 28 | 28 = 28 |
| L1 uniqueness | 28 unique (strain, environment) pairs | 24 unique knocked-out ORFs | (no family rule) |
| L2 value fidelity | 4,040 values | 1,574 values | 1,172 values |
| L2 se non-negative | 0 (no SE released) | 1,513 values | n/a |
| L3 reference | key-subset for 3,297 values | key-matched for 1,574 | interval not half-stated |
| L3 measurement type | single | single | single |
| L3 provenance audit | 24 quotes | 24 quotes | 24 quotes |
| L4 containment | 24 of 24 loci | 65 of 65 identifiers | 24 of 24 loci |

The quote count is 24 rather than 23 because of the new `IDs`-sheet Memo value; the L4
protein count is 65 rather than 64 because of the one extra stored protein key.

### Does the field unblock Schmidt 2016 and Lamoureux 2023? Measured, and yes

Measured from the built drop ledgers and the raw mirrors; **neither file was edited
here**, both belong to another PR.

- **Schmidt 2016, all three proteome arms: 4 conditions each, fully unblocked.** The
  `culture_not_batch` rule names `Chemostat µ=0.5`, `Chemostat µ=0.35`,
  `Chemostat µ=0.20`, `Chemostat µ=0.12` in `proteome_schmidt2016`, and
  `chemostat µ=0.12 / 0.20 / 0.35 / 0.5` in `proteome_srm_set1_schmidt2016` and
  `proteome_srm_set2_schmidt2016` (set 2 writes `µ=0.2`). The rate is in the released
  condition label itself, so every one of the four has a rate to carry: 14 -> 18 records
  for `proteome_schmidt2016` and `proteome_srm_set2_schmidt2016`, 11 -> 15 for
  `proteome_srm_set1_schmidt2016`, if their loaders set the field. Their
  `needed_addition` text, "a culture-mode / dilution-rate slot on Environment
  (closure-changing)", now names a field that exists for the dilution-rate half.
- **Lamoureux 2023: 3 of 3 samples unblocked.** `rnaseq_lamoureux2023`'s
  `culture_not_batch` holds exactly `p1k_00049`, `p1k_00092` and `p1k_00093`. All three
  are `Culture Type = Chemostat` with the rate stated verbatim in `Additional Details`:
  "chemostat w/ dilution rate 0.31 h^-1" for `p1k_00049` and `p1k_00092`, "chemostat w/
  dilution rate 0.44 h^-1" for `p1k_00093`. The two 0.31 h-1 samples do not collide with
  each other either, because their environments differ elsewhere too (`p1k_00049`:
  `NH4Cl(10mM)` plus `sauer trace element mixture`; `p1k_00092`: `NH4Cl(1)`, no trace
  mixture). The mirror holds 5 chemostat rows in all; the other two (`p1k_00096`,
  `p1k_00097`) are dropped earlier under `point_mutation_allele` (rpoB E546V) and stay
  dropped. The 82 `Fed-batch` rows are **not** unblocked: a fed-batch culture has no
  steady-state dilution rate, and none reaches this rule anyway.
- `rnaseq_public_k12_lamoureux2023` carries 164 samples under the same rule name, but
  there that rule also covers `bioreactor`, blank and `Mid-to-late exponential` culture
  labels. How many of the 164 state a dilution rate is **not measured here**.

### Things worth knowing next time

- The synthetic fixture now carries an `IDs` sheet and narrows `DILUTION_RATE_ARM` to
  `(0.1, 0.2, 0.7)`, because it holds one served rate and one empty column rather than
  the release's five.
- `records.csv` changed shape: `sample_id, perturbed_gene_symbol,
  culture_name_verbatim, dilution_rate_per_hour, series_verbatim, reference_sample_id`
  (the flux arm has no `series_verbatim`). `perturbed_gene_symbol` is empty for a
  wild-type culture, which is the honest value, and `build_accounting.json` gained
  `reference_dilution_rate_per_hour`, `dilution_rates_loaded`,
  `dilution_rate_by_sample_column` and `retired_drop_rules`.
- The `Flux_GC-MS_data.xls` labeling sheets are named by CULTURE NAME for the
  disruptants (`galM` ... `talB`, `pfkA_1`, `pfkA_2`) and by SAMPLE ID for the
  dilution-rate and reference cultures (`GR01`-`GR04`, `RF03`-`RF06`), so
  `_check_fit_inputs` looks each up by kind. `WT, 0.1h-1` is not a sheet name in that
  workbook.
- `torchcell/datamodels/identity.py::environment_identity` now projects
  `dilution_rate_per_hour` (landed by the sibling #753 agent). Measured over this
  module's own `environment()`: the five rates give **5 of 5 distinct environment node
  ids** (`identity_sha256(environment_identity(environment(rate)))`), so the four
  recovered records get four environment nodes in the graph rather than collapsing onto
  one. Nothing in the dataset build reads that projection, so the dev stores did not
  need a second rebuild for it.
