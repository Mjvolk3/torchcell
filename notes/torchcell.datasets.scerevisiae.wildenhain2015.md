---
id: 0s6gui1z01a87158vovj3e3
title: Wildenhain2015
desc: ''
updated: 1789272406505
created: 1789272406505
---

## 2026.09.12 - Serve-50 rebuild: AID description mirrored, screens counted, category mapped

Loader: `torchcell/datasets/scerevisiae/wildenhain2015.py`. Adapter:
`torchcell/adapters/wildenhain2015_adapter.py` +
`torchcell/adapters/conf/env_chemgen_wildenhain2015_adapter.yaml`. Tests:
`tests/torchcell/datasets/scerevisiae/test_wildenhain2015.py`,
`tests/torchcell/adapters/test_wildenhain2015_adapter.py`.

### Why the previous build could not be served

It failed L0 on 100% of its records (`Media.is_synthetic Field required`). Beyond that:
the medium was a free-text `Media(name="synthetic complete (SC), 2% glucose")` that did not
join the shipped `SC`; `Temperature(value=30.0)` came from a PubChem AID description that
was in no mirror and carried no hash; `n_samples = 2 x released rows` counted OD reads of
re-exported duplicate rows; no uncertainty was stored at all; 16 ORFs were spelled two ways
(`TOR1` and `Tor1`), splitting the perturbation node; the released Inactive / Active /
Inconclusive curation verdict was ingested silently; and `download()` fetched a 151 MB FTP
container whose own sha256 was never pinned.

### Raw mirror and provenance chain

`$DATA_ROOT/torchcell-raw/wildenhainPredictionSynergismChemicalGenetic2015/` holds the two
files the loader consumes:

- `data/1159580.csv.gz`, sha256 `c461c679...`, retrieved with
  `torchcell.literature.retrieve.zip_member` from the NCBI FTP range archive
  `1159001_1160000.zip`, whose OWN sha256
  (`d1fd5dc2bf7c526ad9845e0a14ae9981256fb820aaf4228b48a3ba0724ee59b0`, measured 2026-09-13)
  is asserted before the member is read, so a re-packed upstream archive fails loudly.
- `data/aid_1159580_description.json`, sha256 `23c5f8c5...`, the PUG-REST assay
  description. This is the artifact that turns the screening protocol from an unsourced
  default into a sourced value.

`download()` links both out of the mirror and verifies each sha256.

The AID description is the ONLY source for several values: "All strains were grown and
screened in synthetic complete (SC) medium with 2% glucose." (the paper's own medium
sentence has "All fungal species" as its subject, which in context follows the fungal
PATHOGEN isolates), and "Plates were incubated at 30 C without shaking for approximately
18 h or until culture saturation was achieved for the solvent controls." Grepping the
paper for a 30 C statement returns nothing; its only temperature is 37 C, for HeLa/HEK
cells. The 13 module-level `SourcedValue` constants audit clean against the mirror each
one names (AID description in the raw mirror, `paper.md` sha256 `f46409eb...` in the
library mirror), which the loader test asserts.

### Screens, and the measured correction to the previous build

The export re-emits the same datapoint under two gene-symbol spellings (`MDH1` and `mdh1`):
of the 46,195 cells with more than one released row, 33,483 contain byte-identical
duplicates. Two measurements settle how to count:

- within a cell, rows sharing a `z_score` share every other data column too (they differ
  only in the `sym` casing), and
- deduplicating on the full datapoint identity and deduplicating on the z string give the
  IDENTICAL partition, with zero multi-datapoint cells sharing a z.

So the z string is the datapoint key. After deduplication the screens-per-cell histogram is
{1: 412,368, 2: 14,579, 3: 1,617, 4: 7, 5: 2}, i.e. 16,205 multi-screen cells, not the
46,195 the raw row count suggests. `n_samples` is now the number of contributing SCREENS
with `sample_unit=screen` (the AID says the z is computed from "normalized average reads
per screen", so the technical-duplicate OD pair is already inside one z), the uncertainty
is the sample SD across screens, and a single-screen cell carries a typed `ProvenanceGap`
on `environment_response_uncertainty` and `environment_response_se` instead of a silent
None. A sample SD of exactly 0 is impossible by construction and the loader raises if one
appears.

This also removes the `non replicate` flag's effect on `n_samples`: counting screens rather
than reads is what the flag broke. 2,320 kept cells have every contributing screen flagged
(the two OD reads of each disagreed); they are KEPT and recorded here rather than dropped,
since the release retained them after its own "> 3 MAD" outlier filter.

### Released curation verdict -> ResponseCategory

The `PUBCHEM_ACTIVITY_OUTCOME` and `bioactivity` columns are now typed rather than
discarded, sourced from the AID's own column definition ("sensitive if compound decreases
fitness or resistant if compound increases fitness compared to negative control"):

| released (outcome, bioactivity) | `category` | cells |
|---|---|---|
| Inactive, "" | `no_change` | 381,659 |
| Active, sensitive | `sensitive` | 43,785 |
| Active, resistant | `resistant` | 920 |
| Inconclusive, sensitive / resistant | `not_determined` | 829 |
| screens disagree | `not_determined` | 1,380 |

`category_label` keeps the source words verbatim ("Active / sensitive",
"Active / Inactive / sensitive" when screens disagree), so the mapping is auditable and
the 976 Inconclusive rows are visible instead of silent.

### Retention rules and counts

428,573 (ORF, compound-identity) cells, kept **428,206**. The one rule
(`preprocess/dropped_records.json`): `compound_without_a_structure_identifier` removes the
367 cells of the 5 SID-only compounds (no CID and no SMILES, so no InChIKey, CID or ChEBI
id exists). Nothing else is dropped: all 5,173 CIDs resolve through the pinned identity
table to a canonical PubChem name AND an InChIKey, which is what moves 427,177 records from
"CID + SMILES only" to joinable, and all 242 released ORFs are current R64 genes.

### Encoding

- **Medium**: the shared `MEDIA_LIBRARY["SC"]` object, which now carries its D-glucose at
  20 g/L (= the AID's 2%), so the medium matches a library object exactly instead of being
  free text.
- **Compounds**: `resolved_compound(identity, pubchem_cid=..., smiles=...)`. `Compound.name`
  is the canonical PubChem title (`vanillin`), not the identifier-posing-as-a-name
  `"CID 1183"` the previous build stored. Dose `20 uM`; `Solvent(name="DMSO")` with a typed
  vehicle `Compound` and `percent=None` (the final DMSO fraction is not released, and
  back-computing it from "2 uL of 1 mM stock into 100 uL" would assume a neat-DMSO stock).
- **Genotype**: `BarcodedKanMxDeletionPerturbation` with `collection="Euroscarf deletion
  collection"` and `barcode=None` (the release publishes no barcode).
  `perturbed_gene_name` is the GENOME's standard name, which collapses the `TOR1` / `Tor1`
  split.
- **Reference**: BY4741 at z = 0 in the SAME compound environment. That is now a sourced
  statement rather than an assumption: the z is standardized within a screen, so 0 is that
  screen's own normalized-growth center. The reference `units` says exactly that.
- `Environment.duration_generations` carries a typed `ProvenanceGap`: an 18 h OD growth to
  saturation doses exposure in hours, not doublings.

### Verifier result (verbatim)

Run with the STREAMING gate (`stream: True`), which the runners entry needs for this size;
report written to `preprocess/verification_report.json`. The rebuild also collapses the
stale 42 GB `experiment_reference_index.json` (5,178 references x 428,573 booleans, a
pre-sparse artifact) to 212 MB: the whole dataset tree is 2.0 GB, down from 41 GB.

```
env_chemgen_wildenhain2015: PASS
  [ok] L0 structural: 428206 records validated
  [ok] L1 count: observed 428206, expected 428206
  [ok] L1 pair_uniqueness: 428206 unique (strain, condition) records, one each
  [ok] L1 provenance_gaps: 1252226 documented provenance gaps over 428206/428206 records; 0 deferred field(s): []; 26734849 undeclared None values over 4 carrier fields (top: Compound.inchi x14559004, Compound.chebi_id x11335629, EnvironmentResponsePhenotype.screen_id x428206, EnvironmentResponsePhenotype.environment_response_uncertainty_type x412010)
  [ok] L1 canonical_gene_names: 242 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 428206 values checked
  [ok] L2 se_nonnegative: 16196 values checked
  [ok] L2 uncertainty_sanity: 16196 labelled uncertainties, none a zero dispersion; 0 records report n_samples >= 2 with no uncertainty
  [ok] L3 measurement_type_consistent: single measurement_type: <MeasurementType.z_score: 'z_score'>
  [ok] L3 reference_zero: numeric rule: reference response == 0 for all 428206 records
  [ok] L3 environment_perturbed: all 428206 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=30.0, media='SC (synthetic complete: YNB + 20 amino acids + uracil + adenine + glucose)')
  [ok] L3 compound_identity: environment edits: 856412 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_compound_identity: medium components: 13702592 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable)
  [ok] L3 media_membership: 428206 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 242 measured genes are S288C reference genes (>= 0.9)
  [ok] L4 current_genome_genes: every one of the 242 measured systematic names is a gene of the current genome
```

Two lines to read carefully. `L2 se_nonnegative: 16196 values checked` is no longer the
vacuous "0 values checked" of the previous build: 16,196 of the 16,205 multi-screen cells
carry a derived SE (the other 9 are cells of the dropped SID-only compounds). And
`L2 uncertainty_sanity` now has something to check at all, because every labelled
dispersion comes from at least two distinct screens.

### Open flags

- The kanMX marker is a property of the Euroscarf MATa collection (Winzeler 1999 /
  Giaever 2002), not something either source names. The `collection` label records what IS
  sourced; the marker assertion is inherited from the leaf class.
- The released `sym == 'wild type'` rows (1,940, plus `wtn01`, `YGL11`, `NNK1` and `TSCII`,
  7,296 non-strain rows in total) are NOT ingested. They are strain measurements, not the
  z-score baseline (their z has mean -2.31, median -0.071, sd 7.82), so ingesting them
  needs a wild-type genotype decision the orchestrator has not made.
- The 195-strain sentinel roster is in Table S3, which cell.com does not serve to a script;
  all 242 released ORFs are served and the `si_expected` manifest entry records the gap.
  Coverage is very unbalanced: 63 strains have fewer than 1,000 cells.
- The `p_value`, `normalized OD average`, `raw OD read 1/2` and `cryptagen` columns are
  still not stored.

## 2026.09.30 - Raw sha256 pin verified at build time

Issue #518 (sweep); the whole sweep is in [[torchcell.data.experiment_dataset]] (2026.09.30). Before: download-only check: yes; PyG skips `download()` when `raw/` is populated, so a file placed or edited in `raw/` built unchecked; copy before check: no; refused deposit leaving a directory: partial.

Now `process()` starts with `verify_raw_files(self.raw_dir, ...)` against `DATA_SHA256`, `AID_SHA256`, before any record is read, and raises `RawSha256MismatchError` ("sha256 mismatch for <file>: expected <pin>, observed <digest>") with no store written. `download()` stages files through the shared `copy_verified` / `write_verified` / `link_verified` helpers, which hash before writing, so a refusal leaves nothing in `raw/`. Records built from a verified raw file are unchanged. Test: `test_a_raw_file_off_the_pin_is_refused_at_build_time` (or the renamed former Finding test) in the paired test file.

## 2026.09.30 - Module constant is the one pin at download (issue #561)

Before this change, `download()` verified the mirror bytes against the digest recorded in the raw-mirror `manifest.json`, and `process()` verified them against the module constant. Because `deposit_raw_mirror` writes the manifest from the constant, the two were equal by construction, but the loader still carried two pins. Now `download()` verifies the bytes against the module constant, and `check_manifest_pin` refuses a manifest that records any other digest, raising `ManifestPinMismatchError` named by path with both digests. The manifest stays the retrieval record. Built records are unchanged. Tests: `test_download_refuses_a_manifest_digest_off_the_module_pin` in [[tests.torchcell.datasets.scerevisiae.test_raw_pins]].

## 2026.10.01 - Datapoint key is the parsed z; missing manifest refuses

Previous behavior: `MatrixCell.screens` was keyed by the z_score STRING, so `-4.0` and `-4.00` in one cell counted as two screens with a sample SD of exactly 0, and `_phenotype` aborted the build. `download()` with no mirror deposited failed in `load_manifest` as a bare `FileNotFoundError`.

Fix (issue #520): `screens` is keyed by the parsed float, so equal values are one datapoint; the SD-0 guard in `_phenotype` is removed because two distinct float keys cannot have a sample SD of 0. `load_manifest` refuses with the manifest path and the `deposit_raw_mirror()` step.

Record-neutral, measured on the pinned `1159580.csv.gz`: 484,830 strain datapoint rows, 428,573 cells, 0 cells holding two z strings of equal value, 0 non-finite z. The issue #504 input-audit items (essential genes, background, z reference) are not touched here.

Tests: `test_two_z_strings_of_equal_value_are_one_screen`, `test_download_without_a_manifest_refuses_naming_the_deposit_step`.

Review follow-up (same day): `_parse_z` refuses, naming the cell and the raw string, for an unparseable z ("is not a number") or a non-finite one ("is not finite"). Two `nan` rows would otherwise be two float keys. The refusal fires in `_collapse_matrix`, before the store opens. Measured on the pinned export: 0 unparseable and 0 non-finite z of 484,830. Test: `test_a_non_finite_or_unparseable_z_refuses_naming_the_cell`.

## 2026.10.02 - Input audit fixes: essential-gene strains, Sci Data 2016 sourcing, strain screens, background, z reference (#504)

Issue #504 (input-definition audit). Built on the #507 schema ([[torchcell.datamodels.strain-background]]). Measurement script: `experiments/036-dataset-fixes-before-kg-build/scripts/wildenhain2015_inputs.py`, output `experiments/036-dataset-fixes-before-kg-build/results/wildenhain2015_inputs.json` (+ `_labels.csv`); inputs: the pinned `1159580.csv.gz`, a full scratch build of this branch, and the pre-fix dev store.

### What was wrong

- 33 of the 242 released ORFs are SGD-essential (CDC28, TOR2, IPL1, RAD53, ...) and were served as haploid kanMX nulls, which are not viable.
- The release is the 2016 Sci Data extended CGM (doi:10.1038/sdata.2016.95, mirrored as `wildenhainSystematicChemicalgeneticChemicalchemical2016`), not the 2015 195-strain CGM; it was cited nowhere, and the "195 vs 242" text called the 47 extra strains unexplained.
- The `NA` / `NULL` rows (7,296) were dropped as "non-strain control rows"; they are strain screens.
- The BY4741 background was a bare string; the z reference claimed BY4741 in the same compound well; `Z_SCORE_DEFINITION` quoted the AID's "kernel density" caption; the DMSO fraction, culture format and endpoint rule were not recorded.

### What the loader does now

- Records are `StrainEnvironmentResponseExperiment` on a `CultureEnvironment`; one `StrainEnvironmentResponseExperimentReference` for every record.
- Background: `StrainReferenceGenome(strain="BY4741", background=standard_background("BY4741", provenance=[BY4741_GENOTYPE]))`, the string quoted from Sci Data paper.md line 50 ("isogenic to BY4741, which has the genotype MATa his3Δ1 leu2Δ0 met15Δ0 ura3Δ0"); each allele also carries a `deferred_pending_source_review` gap on `deleted_span` naming Brachmann 1998 (the edit kind of each designation is not in a mirrored source).
- Genotype: `BarcodedKanMxDeletionPerturbation(collection="Euroscarf deletion collection", cassette="kanMX4")` (Giaever 2014) with a pending gap on `barcode` (Giaever 2002) for 209 ORFs; `ConditionalAllelePerturbation(allele_class=None, collection=None)` with pending gaps naming Sci Data Table 1 / Cell Systems Table S3 for the 33 essential ORFs (`ESSENTIAL_GENE_ORFS`, pinned; the test recomputes it from `gene_essentiality_sgd/preprocess/gene_set.json`). Emit-with-gap was chosen over a drop because the records are real measurements.
- Non-ORF labels (`NON_ORF_STRAIN_LABELS`): `NA/NNK1` is mapped to YKL171W (checked against the genome at build time); `NULL/wild type` is served as the empty genotype; `NA/TSCII`, `NULL/YGL11`, `NULL/wtn01` are held under the ledger rule `strain_label_unresolved`. An unlisted label refuses the build.
- Environment: a Wildenhain-local `WILDENHAIN_SC` (same composition as the shared `SC`, so one `media_identity`; provenance replaced by the Sci Data line 54 medium sentence). The shared `SC` in `media.py` (which still carries the 2015 fungal-species sentence) is imported by five loaders and is untouched. 30 C, ~18 h, `CultureFormat(vessel="96-well plate", working_volume_ul=100, shaking_rpm=0, inoculum_cells=50000, endpoint=until_control_saturation)`, `PreCulture(source=overnight_culture)` with a gap on its medium, gaps on `duration_generations` and `auxotroph_supplements`. `Solvent(DMSO, percent=1.96)`: 100 x 2 uL / (100 uL + 2 uL) = 1.96 % v/v from lines 46 and 54, treating the 1 mM working stock as neat DMSO (hypothesis, untested: the 10 mM library stocks are themselves DMSO; if aqueous, 1.76 %).
- Z: `Z_SCORE_DEFINITION` quotes the N(1, IQR) rule (Sci Data line 66); the units say IQR-scaled, 0 = the strain's own plate median. The reference environment has no compound and its units say the baseline is the SAME strain's screen center. Every phenotype carries a `screen_id` gap: the release has no library column, and normalization differs by library (LOWESS vs DMSO controls, lines 63-64).
- Publication stays the 2015 Cell Systems paper (a record holds ONE `Publication`); the 2016 paper reaches every record through the reference background, the culture format, the medium and the z rule (`SourcedValue`s with its citation key and sha256).

### Measured (script above)

- Release: 492,126 rows, 5,518 SIDs, 242 ORFs (= Sci Data's stated counts). Non-ORF labels (rows = SIDs): NNK1 678, TSCII 678, YGL11 2,000, wild type 1,940, wtn01 2,000. NNK1's SIDs and CIDs share 0 with the 11 rows released under `orf=YKL171W`.
- z vs normalized OD, per strain label with >= 50 rows (246 fitted): median r^2 0.999996, median normalized OD at z = 0 0.999996 (mean 0.990), median slope 43.0.
- Records: before 428,206 (dev store), after 430,820 (scratch build) = 406,477 kanMX deletion + 22,407 conditional allele (all 33 essential ORFs, every one with gaps on `allele_class` and `collection`) + 1,936 wild type. YKL171W: 11 -> 689 records. One distinct reference. Ledger: 435,857 cells; `strain_label_unresolved` 4,669 (TSCII 678, YGL11 1,994, wtn01 1,997); `compound_without_a_structure_identifier` 368 (was 367; one wild-type cell is an SID-only compound).
- z identity: all 428,206 pre-fix (ORF, compound) keys are in the new store and 0 have a different z; the 2,614 new keys are 1,936 wild type + 678 YKL171W.
- Verifier (streaming, `torchcell/verification/runners.py` entry, on the scratch build): PASS, L1 count 430,820 = expected, pair_uniqueness 430,820 unique, deferred gap fields `allele_class`, `barcode`, `collection`, L3 reference_zero for all records, L3 media_membership 430,820 records on a medium deriving from `SC`.

### Open

- Retrieval go-ahead needed: Sci Data 2016 Table 1 and Cell Systems 2015 Table S3 (would resolve the 33 alleles' class and collection, TSCII / YGL11 / wtn01, the Euroscarf accessions); Brachmann 1998 (the four BY allele constructions); Giaever 2002 (barcodes).
- Schema items (not edited, shared classes): `ExperimentReference` has no genotype slot, so "same strain, vehicle environment" is half encoded (environment yes, genotype only in `units`); a record holds one `Publication`, so the Sci Data paper cannot be a second one.
- `aerobicity="aerobic"` stays the default for a static 100 uL culture. Hypothesis (untested): it is oxygen-limited.
- The KG needs a full rebuild (closure moves to the strain-resolved family); the dev store must be rebuilt under slurm.
