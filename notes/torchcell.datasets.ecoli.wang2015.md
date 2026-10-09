---
id: u4116zf6d70znlylxp9jsnk
title: Wang2015
desc: ''
updated: 1791389304038
created: 1791389304038
---

## 2026.10.07 - Loader, sourcing decisions and the first build

`torchcell/datasets/ecoli/wang2015.py`, class `EnvChemgenWang2015Dataset`
(`REFERENCE_STRAIN = "BW25113"`, takes `ecoli_genome`), rank 15 of
[[plan.bacteria-ontology-genome]] (tolerance / robustness, `K-12-KO`). Template:
[[torchcell.datasets.ecoli.fuhrer2017]]. Shared API: [[torchcell.datasets.bacteria_common]].
Every number below is read from the build's own ledgers in
`$DATA_ROOT/data/torchcell/env_chemgen_wang2015/preprocess/` (`dropped_records.json`,
`identifier_reconciliation.json`, `extraction.json`, `table_s3.csv`,
`verification_report.json`), written by the loader and by `wang2015.run_verification`.

### Where the data is

Wang, Yang, Shah, Choi and Kim 2015, Sci Rep 5:16505, doi 10.1038/srep16505, citation key
`wangDynamicInterplayMultidrug2015`, `paper.md` sha256
`9cd90f00599871689fa8e62849b4220fa0801a502160b575bca4d04067fc3cd2`, SI OCR `si/si1.md`
sha256 `d1c4aeeb6edcaf88a715838cc1cca10e995e061af1801cc8e74eba3d0055272a`.

There is no data deposit. The per-strain values are Supplementary Table S3 of the SI PDF:
OD600 mean and SD per strain without and with 0.5% (v/v) isoprenol. The text ties it to
the screen: `cell growth was evaluated after $1 2 \mathrm { ~ h ~ }$ exposure to $0 . 5 \%$ (v/v) of isoprenol in a total of 44 MDT null mutants (Fig.  1C and Supplementary Table S3)`.
Table S2 gives each strain's Keio number and transporter family.

Raw mirror `$DATA_ROOT/torchcell-raw/wangDynamicInterplayMultidrug2015/`, written by
`python -m torchcell.datasets.ecoli.wang2015 deposit --download-dir <dir> --retrieve`
(runs the recorded retriever, then `deposit_raw_mirror`, idempotent by sha256, refuses a
differing file):

| file | source | sha256 | read for |
|---|---|---|---|
| `data/si1.pdf` (`srep16505-s1.pdf`) | PMC Open Access bucket, `pmc_cloud`, key `PMC4643228.1/srep16505-s1.pdf` | `ed2a029a...` | Tables S2 and S3 |

The retrieval was re-run on 2026-10-07 and returned bytes identical to the literature
mirror's `si/si1.pdf` (same sha256). The manifest's `processing` record names the
extraction: `pdftotext version 21.01.0`, args `-layout -enc UTF-8`.

**Why the PDF text layer and not the OCR.** The SI PDF is born-digital, so its text layer
is the publisher's own characters (it has `Δ` U+0394 and `±` U+00B1 throughout). The
MinerU OCR of the same tables misreads Table S2 strain names (`BWAmacB` for `BWΔmacB`) and
writes two deltas as U+2206. The OCR stays the anchor of every quote. The two
extractions of Table S3 agree on all 47 rows and 188 numbers
(`test_real_text_layer_agrees_with_the_independent_ocr`).

**Extraction checks, enforced at build time.** Every `BWΔ` label of each table parses to a
row; Table S3 has one BW25113 row and 46 deletion rows; the 46 names equal Table S2's 46
Keio entries; the parsed values hash to `TABLE_S3_SHA256`
(`ce6d952c1058695fe5912353de563faa0224ad5ce8c16c5a8812e62da3b41093`); and the table
reproduces four counts the paper states:

| stated in the paper | measured from Table S3 / S2 |
|---|---|
| `growth inhibition over $5 2 . 5 \%$ was observed in 17 mutants` | 17 |
| `seven null mutants ... showed growth inhibition of more than $5 7 . 5 \%$` | 7 |
| `eleven MDT mutants exhibited phenotypes with higher tolerance` (read as below 47.5%, Fig. 1C's resistant bins, tolC excluded as OMP) | 11 |
| `a library of 45 null mutants (Supplementary Table S2)` (Keio entries of an MDT family) | 45 |

### Strain: BW25113

`E. coli BW25113 $( \mathrm { F ^ { - } } , \ \mathsf { \lambda } ^ { - } , \mathsf { \lambda } ^ { - }$ , rrnB-3, ΔlacZ4787`,
`... rph-1) and its isogenic deletion mutants were obtained from Keio collection (National Institute of Genetics, Shizuoka, Japan) for screening of transporter candidates associated with isoprenol tolerance.`
Baba 2006 (`babaConstructionEscherichiaColi2006/paper.md`, `ca71475b...`) corroborates:
`The Keio collection is comprised of 3985 deletions in duplicate (7970 total) of E. coli K-12 strain BW25113 (Datsenko and Wanner, 2000)`.
Records pin `assembly_reference("BW25113")` = `ecoli_K12_BW25113_ASM75055v1`,
`GCA_000750555.1`. Stored per deletion: `collection = "Keio collection"` (Wang's
wording), `cassette` from Baba (`kanamycin cassette flanked by FLP recognition target sites`,
the same string the Fuhrer loader stores).

### Identifiers: the strain name, cross-checked against Table S2

Table S3 keys its rows by gene name only (`BWΔacrA`). The names go through
`reconcile_locus_tags` on the BW25113 GenBank annotation:

| status | count | layer |
|---|---|---|
| renamed (to a current locus) | 46 | gene symbol 44, gene synonym 2 (`cmr` to `mdfA` BW25113_0842, `yjiO` to `mdtM` BW25113_4337) |
| retired / ambiguous / collision / case-insensitive | 0 | |

Resolved fraction 46 / 46, above the stop threshold `MIN_RESOLVED_FRACTION = 0.95` (more
than 2 of 46 unresolved would mean a parse error or the wrong genome). No record dropped.

The Table S2 Keio numbers go through the same reconciler as a cross-check: 45 resolve at
the synonym layer, 1 is retired (`JJW0452`). **41 name the locus their strain's name
names; 5 do not**:

| strain (name) | name's locus | Table S2 number | that number's locus on BW25113 | Keio list (Fuhrer Table EV1A) |
|---|---|---|---|---|
| acrA | BW25113_0463 | JW0451 | BW25113_0462 (`acrB`) | JW0451 = acrB; acrA = JW0452 |
| acrB | BW25113_0462 | JJW0452 | none (not a Keio number) | JW0452 = acrA; acrB = JW0451 |
| emrK | BW25113_2368 | JW2364 | BW25113_2367 (`emrY`) | JW2364 = emrY; emrK = JW2365 |
| emrY | BW25113_2367 | JW2365 | BW25113_2368 (`emrK`) | JW2365 = emrK; emrY = JW2364 |
| mdtD | BW25113_2077 | JW2077 | BW25113_2093 (`gatB`) | JW2077 = gatB; mdtD = JW2062 |

`JJW0452` is in the publisher's text layer, not an OCR artifact. The last column is an
independent check: the Keio strain list Fuhrer 2017 publishes (its Table EV1A, in its raw
mirror) agrees with the BW25113 annotation on all five
(`test_real_keio_list_confirms_the_five_table_s2_disagreements`).

**Decision: the record's locus is the strain NAME's.** Evidence, from the paper's own
Table S4 (qPCR in the mutants, columns BWΔacrA, BWΔacrB, BWΔtolC):
`<tr><td>acrB</td><td>2.66 ± 0.36</td><td>1.97 ± 0.09</td>` (an acrB transcript in the
strain called BWΔacrA) and `<tr><td>acrA</td><td>3.16 ± 0.87</td><td></td><td>1.59 ± 0.08</td>`
(an acrA transcript in BWΔacrB). A strain transcribing acrB is not the acrB deletion that
JW0451 is, so for acrA and acrB Table S2's number is the error and the name is right.
For emrK/emrY (the same swapped-pair pattern) and mdtD (`JW2077` repeats the digits of
mdtD's Blattner number b2077) there is no such internal evidence; the name is used
uniformly because it is the data table's own key, used by every figure and the text, and
Table S2 is shown to be wrong in the two cases that can be checked. **This is a judgment
call flagged for review**: for emrK, emrY and mdtD it is possible the authors used the
strain their Table S2 number names. The alternative, dropping those three records, is one
line in `resolve_strains`.

Stored per record: `systematic_gene_name` = the name's BW25113 locus tag,
`perturbed_gene_name` = that locus's own GenBank symbol (so `cmr` is stored as `mdfA`),
and `construction.strain_accession` = the Table S2 number only for the 41 that agree. The
5 carry `construction = None`: storing a number that names another locus would make the
record contradict itself, and storing the annotation's number for the name would assert
a strain the paper does not name.

### Records and drops

47 Table S3 rows = BW25113 (the reference) + 46 deletion strains (45 multidrug
transporter genes in 5 families + `tolC`, OMP). Families: MFS 21, RND 11, ABC 8, SMR 4,
MATE 1, OMP 1.

| rule | records |
|---|---|
| `strain_name_not_on_a_bw25113_locus` | 0 |
| kept | **46** |

### Phenotype, n and the uncertainty type

`BacterialEnvironmentResponseExperiment` / `BacterialEnvironmentResponseExperimentReference`
(the assembly-pinned pair), phenotype `EnvironmentResponsePhenotype`,
`measurement_type = log2_ratio`, `assay_type = liquid_od_growth`.

**What the number is.** The paper defines
`Growth inhibition $( \% )$ is defined as $( 1 - \mathrm { O D } _ { 6 0 0 }$ with ... / ... without ...`
and `relative tolerance capacity $( \% )$ by [ $( 1 -$ growth inhibition of Strain A)/ $1 -$ growth inhibition of Strain $\mathbf { B } ) ] \times 1 0 0$ .`
The stored value is the log2 of that ratio with B = BW25113:
`log2[(OD_iso / OD_0)_strain / (OD_iso / OD_0)_BW25113]` from the Table S3 means. The
reference is BW25113 in the same environment, log2(1) = 0. Range -0.453 (emrA, 63.7%
inhibition) to +0.633 (acrA, 23.0%); 25 of 46 below 0, 21 above. Parent inhibition
50.36% (`The growth inhibition of the wild type strain is at approximate $5 0 \%$ as a reference.`).

**Why not `FitnessPhenotype`.** The readout is each strain's growth under isoprenol
normalized by its own isoprenol-free growth, then compared with the parent: a response to
an environmental edit relative to a control, which is what `EnvironmentResponsePhenotype`
models and what section 3c of the plan assigns to the tolerance rows. The value is
signed. `FitnessPhenotype` is a growth ratio in one environment and clamps non-positive
values.

**n_samples = 2, `biological_replicate`.** Fig. 1C caption, the figure Table S3
tabulates: `Results are the means of two biological replicates.` Table S3 states no n. No
range, so no resolution rule was needed.

**Uncertainty: a typed gap.** Table S3: `Note: The results are presented as means $\pm$ standard divisions.`
("standard divisions" is in the publisher's text layer; the same lab writes
`Error bars represent the standard deviations of two biological replicates.` for Figs. 3C,
4 and 5, so the type of each column's number is a sample SD). Those SDs belong to the four
OD600 means a record is computed from, not to the ratio, and the per-replicate ODs are not
released, so the ratio's SD is not derivable without assuming the four cultures
independent. `environment_response_uncertainty` and `environment_response_se` are
`ProvenanceGap(not_reported_by_primary)` on every phenotype. The column SDs are kept in
`table_s3.csv`.

### Environment

`Environment(media=YT_2X, temperature=30 C, aerobicity="aerobic", duration_hours=12)` with
two perturbations:

- `EnvironmentPhysicalPerturbation(factor=ph, magnitude=7.0 pH)` from the recipe sentence
  `adjusted to $\mathrm { p H } 7 . 0 $ ) with agitation.`, as the [[torchcell.datamodels.media]]
  `YT_2X` entry directs; `agent` is a typed gap (no acid or base named).
- `SmallMoleculePerturbation(compound=isoprenol, concentration=0.5 percent_v/v)`;
  `solvent` is a typed gap (no vehicle named; added neat is the likely reading, unstated).

Temperature: `Cultivations were carried out at $3 0 ^ { \circ } \mathrm { C }$`. Medium,
dose, temperature and time together: `MDT mutants were grown in 2YT medium with $0 . 5 \%$ (v/v) of isoprenol at $3 0 ^ { \circ } \mathrm { C }$ for $1 2 \mathrm { { h } }$ .`
`aerobic` is a reading of `with agitation` (4 mL in a 55 mL tube); the paper names no
oxygen regime. The culture format (OD600 0.1 inoculum, 4 mL, 55 mL tube) is quoted in
`SOURCED_VALUES["culture_format"]` and not stored: the bacterial pair declares
`environment: Environment`, so a `CultureEnvironment` would serialize as its base class.

**Isoprenol and the compound-identity layer.** `compound_identity_table.json` has no
isoprenol row, so `resolved_compound("isoprenol")` returns `Compound(name="isoprenol")`
with an `inchikey` gap (`deferred_pending_source_review`), the same object the Carruthers
2025 loader stores, so the two datasets meet on one compound. The identity is recorded in
the module: `ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"`, the RDKit key of
`C=C(C)CCO`, which is the paper's `3M-3=C4OH` (3-methyl-3-buten-1-ol); PubChem's name
lookup returns CID 12988 with the same key. `isoprenol_compound()` stops the build if the
table ever resolves isoprenol to another key. **Dependency:** the table row is added in a
separate branch; after it lands, this dataset's dev LMDB is rebuilt so its records carry
the key.

### Recorded inconsistencies in the paper (measured, not resolved)

- The text says Table S3 holds `44 MDT null mutants`; Table S3 and Fig. 1C both carry 46
  strains (45 MDT genes + tolC). The text also says `a library of 45 null mutants`, which
  matches Table S2's 45 MDT-family entries.
- `presenting $1 3 3 . 5 \%$ of the relative tolerance` for tolC: Table S3's means give
  133.07%. `BWΔtolC exhibits $8 8 . 3 \%$ of the tolerance capacity of E. coli BWΔacrB`:
  the means give 86.68%. The tolC growth inhibition (33.9%) and acrA/acrB (23.0%, 23.8%,
  `around $2 3 \%$`) do match. Hypothesis (untested): the two ratios were computed from
  the separate Fig. 2 time-course experiment, or as a mean of per-replicate ratios.
- `standard divisions` for standard deviations (above).

### Build and verification

`PYTHONPATH=<wt> python -m torchcell.database.build_dataset_lmdb --dataset EnvChemgenWang2015Dataset`:
46 records, 1 reference, about 1 s. `torchcell.provenance.build_manifest.check_all` reads
`env_chemgen_wang2015` as `fresh`. A first build that differed only in the `table_s3.csv`
record column's dtype was retired to `/scratch/projects/torchcell-deprecated/`.

`python -m torchcell.datasets.ecoli.wang2015 verify` (the environment-response family
verifier of `torchcell/verification/environment_response.py` with the BW25113 resolver,
plus a BW25113 L4 and the provenance audits): PASS. L0 46 records validate; L1 count 46,
one record per strain and condition, canonical names current in BW25113; L2 46 finite
values; L3 one measurement type, reference 0 for all 46, every record carries an
environmental edit, isoprenol gapped (not name-only), all 46 on the shared `YT_2X`, 24 of
24 sourced values verbatim at their pinned sha256; L4 46 of 46 deleted loci are BW25113
GenBank gene rows.

Tests: `tests/torchcell/datasets/ecoli/test_wang2015.py`, 30 hermetic (synthetic layout
text over the synthetic BW25113 assembly; `pdftotext` faked) and 30 data-gated
(`--data`). Diff coverage 98% against `origin/main` without `--data`.

### Not loaded, and why

- Figure-only: the Fig. 2 time courses; the transporter overexpression strains on pTrc99A
  in BW25113 and BWΔacrAB (Fig. 3C with IPTG, Fig. S3 without); BWΔacrAB, BWΔABC and the
  0.75% isoprenol arm (Fig. 4); the other alcohols (Fig. 5); pT-tolC in BWΔABC (Fig. S5).
  If digitized, an overexpression strain is a second, plasmid-borne copy of a native gene
  (`pT-acrD pTrc99A containing acrD gene`): `HeterologousPathwayPerturbation` admits a
  host locus tag for exactly that case, with the pTrc99A context as its cassette. Not
  built here.
- Table S4 is tabulated (9 transcripts, RT-qPCR fold change against `cysG`, n = 3) but no
  phenotype class models a targeted qPCR panel; it is used here only as identity evidence.
- The no-isoprenol OD600 column is consumed only as the normalizer; a fitness record of
  each strain in plain 2YT would be a second phenotype family in one dataset.
- Kanamycin: `Kanamycin $( 5 0 \mu \mathrm { g } / \mathrm { m L } )$ ... were added as required`;
  whether the screen cultures carried it is not stated, so no kanamycin is recorded.
- Not registered in `torchcell/verification/runners.py` (outside this branch's files);
  `run_verification` here runs the same verifier.

## 2026.10.09 - The other Table S3 column is now its own dataset

Table S3 releases two OD600 columns and this loader consumes both: the stored log2
relative tolerance is `log2[(with/without)_strain / (with/without)_BW25113]`, so
`od600_without` already enters every record as the per-strain normalizer. The column in
its own right is now served separately, as 46 `FitnessPhenotype` ratios of each deletion
to BW25113 in PLAIN 2YT, by `GrowthWang2015Dataset`
([[torchcell.datasets.ecoli.wang2015_growth]]).

It is not a second copy of a stored number. Measured over the 46 mutants: Pearson
r(`od600_without`, the stored log2) is 0.5454 against 0.9774 for `od600_with`; 0 of 46
stored values equal any no-isoprenol value; and
`max|log2(od600_without ratio) - stored| = 0.6632`. The stored log2 is a function of both
columns, so neither column is recoverable from it.

It is a SEPARATE MODULE, and the reason is measured rather than stylistic: this module's
schema closure is 69 symbols and equals the 69 the served `env_chemgen_wang2015` store
records, and adding the three fitness symbols to the import above raises it to 72, which
would mark that store stale for a change touching none of its 46 records. The new module
imports the pins, the SI parsers, `resolve_strains`, `deletion_genotype`, the media object
and `SOURCED_VALUES` from here, so there is still one copy of each.
