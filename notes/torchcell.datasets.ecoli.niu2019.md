---
id: me62zth05ge5xyuo5iil1b2
title: Niu2019
desc: ''
updated: 1791539218001
created: 1791539218001
---

## 2026.10.09 - Loader added: the CRISPRa/CRISPRi pinene-tolerance arms (issue #799)

`torchcell/datasets/ecoli/niu2019.py` -> `CrisprPineneToleranceNiu2019Dataset`.
Adapter: `torchcell/adapters/niu2019_adapter.py`, conf
`torchcell/adapters/conf/crispr_pinene_tolerance_niu2019_adapter.yaml`.
Tests: `tests/torchcell/datasets/ecoli/test_niu2019.py`,
`tests/torchcell/adapters/test_niu2019_adapter.py`.

Niu et al. 2019, *Synth Syst Biotechnol* 4(2):113-119, doi
`10.1016/j.synbio.2019.05.001`, PMC6556621, citation key
`niuGenomicTranscriptionalChanges2019`. The paper resequences an evolved
pinene-tolerant isolate (`YZFP`), reads its transcriptome against the designed parent
`BW25113(PT5-dxs)`, and then activates or represses the differentially expressed genes
one at a time in that parent with a `dCas9*-MCPSoxS` effector. What this dataset stores
is the CRISPR half.

### What a record is

One `BacterialEnvironmentResponseExperiment` per stored strain:

- **Genotype**: one `BacterialCrisprActivationPerturbation` (the #799 leaf) or
  `BacterialCrisprInterferencePerturbation` per targeted gene, keyed by its BW25113
  locus tag, carrying the shared `dCas9*-MCPSoxS` construct and, when Suppl. Table 1
  releases the matching `N20-<gene>` oligo, its spacer. The six-target interference
  combination strain is one record with six leaves.
- **Environment**: LB plus 0.5% pinene at 37 C for 12 h. The 200 nM
  anhydrotetracycline that induces the effector is a medium COMPONENT, where the
  paper's own recipe puts it and where Rousset 2018 puts the same inducer; pinene is a
  `SmallMoleculePerturbation` on the environment, because it is the challenge the
  phenotype is about.
- **Phenotype**: `EnvironmentResponsePhenotype` with
  `measurement_type=log2_ratio`, `assay_type=liquid_od_growth`,
  `n_samples=3`, `sample_unit=biological_replicate`. The reference is the no-guide
  control at a log2 ratio of 0.
- **Background**: `PT5-dxs` is constant across the dataset AND shared with the control,
  so it is a `BacterialStrainBackground` on the reference, not a genotype perturbation
  (the #507 rule). Its construction is a typed gap: the paper names the parent as
  `BW25113(PT5-dxs)` throughout and never says how the cassette was built.

### Measured counts

Built with `python -m torchcell.database.build_dataset_lmdb --dataset
CrisprPineneToleranceNiu2019Dataset --retire-existing` into
`$DATA_ROOT/data/torchcell/crispr_pinene_tolerance_niu2019`.

| quantity | measured |
|---|---|
| records | 51 |
| `gene_set` | 50 loci |
| experiment references | 2 (one per arm, `screen_id` names the arm) |
| released numeric cells in the whole paper | 94 |
| censored (dashed) cells | 64 |
| guide oligo rows read from Suppl. Table 1 | 91 over 85 single-spacer genes |
| spacer lengths | 20 nt x 90, 22 nt x 1 (`N20-prpR`) |

The 51 are 41 activation growth ratios (43 numeric minus the two operon labels), 9
interference growth ratios, and the six-target interference combination strain.

### L0 to L4 on the built store

`python -m torchcell.datasets.ecoli.niu2019 verify` -> **PASS**.

| level | check | result |
|---|---|---|
| L0 | structural | 51 records validated |
| L1 | count | observed 51, expected 51 |
| L1 | pair_uniqueness | 51 unique (study, strain, condition) records |
| L1 | provenance_gaps | 255 documented gaps over 51/51 records |
| L1 | canonical_gene_names | 50 systematic names, one spelling each, each current |
| L2 | value_fidelity | 51 values checked |
| L2 | uncertainty_sanity | 0 labeled uncertainties; 51 records report n_samples >= 2 with no uncertainty |
| L3 | measurement_type_consistent | single `log2_ratio` |
| L3 | reference_zero | reference response == 0 for all 51 |
| L3 | environment_perturbed | all 51 carry an environmental edit |
| L3 | compound_identity | 0 environment compounds carry a structure identifier; 51 declare a typed gap (pinene) |
| L3 | media_compound_identity | 102 medium components carry one, 0 gapped |
| L4 | gene_containment_sgd | 1.000 of 50 measured genes are reference genes |
| L4 | current_genome_genes | all 50 names are genes of the current genome |

### Decisions taken, each with the measurement behind it

**The stored number is a log2 and the released dispersion is a typed gap.** The release
gives a RATIO with a sample SD over three replicates ("a The ratio of $\mathrm { O D } _
{ 6 0 0 }$ with CRISPRa and without"). `MeasurementType` has no fold-change member, and
`log2_ratio` is the member whose definition the transformed number satisfies exactly,
which is also the encoding Lim 2025 landed for the same kind of quantity. An SD does not
transform with its statistic, so the released ratio and SD go verbatim to
`preprocess/released_growth_ratios.json` and
`environment_response_uncertainty` / `environment_response_se` carry typed gaps rather
than a first-order propagation the source never made. `n_samples=3` IS stored, because
it is a fact about the measurement rather than about the scale ("All experiments were
conducted in triplicate, and data were averaged and presented as the means $\pm$
standard deviation").

**A `fold_change` member is the honest fix, and it is proposed rather than taken here.**
Two things are blocked on it: the growth ratio's released SD, and all 40 pinene cells.
A fold-change member with its own unit-free scale would let both be stored as released.

**Pinene has no curated compound-identity row, so the assay's central environmental
variable is a typed gap.** The pinned compound-identity table resolves
`anhydrotetracycline` and no spelling of pinene (`pinene`, `alpha-pinene`,
`(-)-alpha-pinene`, `(+)-alpha-pinene` all `UNRESOLVED_PUBLIC`; measured in
`experiments/036-dataset-fixes-before-kg-build/results/niu2019_release_loadability.json`).
That table is a sha256-pinned shared artifact whose curator re-queries PubChem for every
row, so adding the row is issue #726's work, not a dataset branch's. L3
`compound_identity` reports the gap rather than passing silently.

**The dose is stored in the basis-free `percent` member.** The paper writes "0.5%
pinene" with no w/v or v/v marker, and pinene is a liquid terpene whose percent could
honestly be either.

**Guide spacers are taken at the length the release gives them.** Every released oligo is
`CGGGGTACC` + spacer + `gttttagagctagaaatag`; 90 of 91 spacers are 20 nt and the
`N20-prpR` row is 22 nt, stored as released rather than trimmed to the name's "N20".
Three stored records carry `guide_sequence=None` with the reason in
`preprocess/guide_spacers.json`: `marA` and `sspA` each get two DIFFERENT spacers with no
rule for choosing, and `opgB` has no oligo row at all.

### What is refused, with the counts that refuse it

`preprocess/dropped_records.json` and `preprocess/not_loaded.json`.

| rule | cells | why |
|---|---|---|
| `pinene_ratio_is_a_dimensionless_product_ratio` | 40 | a dimensionless product ratio with no absolute titer anywhere in this arm. `ProductTiterPhenotype` requires a `ConcentrationUnit` and the enum has no dimensionless member; `EnvironmentResponsePhenotype` is a growth readout. Issue #770 |
| `target_label_is_a_multi_gene_operon` | 2 | `sufBCDS` and `flgFGH` name operons, and a gene-keyed leaf cannot state a label that is not one gene |
| `combination_strain_contains_a_multi_gene_operon` | 1 | the six-target ACTIVATION strain includes both operons, so it cannot be six gene-keyed leaves. The interference one is all single genes and IS stored |
| `target_label_is_not_in_the_bw25113_annotation` | 0 | no released label failed to resolve |

Two further parts of the release are not cells of that grid:

- **The 64 dashed cells.** The footnote defines a dash ("-: means no change or negative
  effect."), and classifying each cell as numeric or dashed reproduces the main text's
  own both / growth-only / pinene-only tallies exactly for both arms (23/20/9 and
  6/3/0), so a dash is a MEASURED non-positive outcome rather than an absent
  measurement. It still has neither a number nor a single `ResponseCategory`, because
  the footnote conflates "no change" with "negative effect", so it is recorded as
  left-censored rather than encoded.
- **Suppl. Table 2's 374 called variants of `YZFP`.** See the next section.

### The evolved clone: a writable genotype with no measured phenotype

Since the #835 leaves landed, these rows are writable: 322 carry a b-number and take
`BacterialSequenceVariantPerturbation`, 48 are intergenic and take
`BacterialSiteVariantPerturbation`, which is 373 of 374; the one blank-mutation-site row
refuses on its missing coordinate, and 4 rows are neither b-numbered nor intergenic
(counts from `niu2019_release_loadability.json`).

**What refuses is the RECORD, not the genotype.** This release measures no number of
`YZFP`:

- Its four SI tables are the primers, that variant list and the two CRISPRa/i target
  tables, and every number in the latter two is measured in the UNEVOLVED parent (both
  table titles say "in E. coli BW25113(PT5-dxs)", and Methods 2.4 says the guides were
  co-transferred into that strain).
- The evolved strain's own tolerance and titer are attributed to reference [8], verbatim:
  "we first improved pinene tolerance to $2 . 0 \%$ and pinene production to $9 . 9
  \mathrm { m g / L }$ from $5 . 6 \mathrm { m g / L }$ through adaptive laboratory
  evolution after atmospheric and room temperature plasma (ARTP) mutagenesis and
  overexpression of the efflux pump to obtain the pinene tolerant strain Escherichia coli
  YZFP [8]". The Caglar 2017 rule (#771) attributes a value to the study that first
  reported it, so those two numbers are that paper's records, not this one's.
- Fig. 2's qRT-PCR `YZFP`-over-parent transcript ratios (182 genes measured, 116
  significant) are released as bar panels. No table carries them, and the SI holds four
  tables, none of which is a transcription table.

A genotype with no measured phenotype is not an `Experiment`, so the 373 writable leaves
have nothing to hang on. Recorded in `preprocess/not_loaded.json` with those counts.
This is a statement about THIS release; a mirrored release that measures `YZFP` would
make the genotype storable immediately, because the leaves now exist.

Separately, and unchanged by the leaves: the 374 calls are `YZFP` against MG1655
(Methods 2.2, "The paired-end reads from *E. coli* YZFP were aligned to the reference
genome of *E. coli* MG1655"), while the strain is a BW25113 derivative and no parent
library was sequenced, so nothing separates evolution-acquired from strain-background
mutations. `call.reference_sequence` makes that visible; it does not solve it.

### Provenance

One supplementary file, `mmc1.docx`, from the PMC Article Datasets bucket
(`pmc_cloud`, key `PMC6556621.1/mmc1.docx`, sha256
`72099acb46abce07983265e559ec49596209b031869f081f40ae1e0bc0ce6fb9`), deposited at
`$DATA_ROOT/torchcell-raw/niuGenomicTranscriptionalChanges2019/data/mmc1.docx` with a
`manifest.json`. Measured: the bucket carries exactly one supplementary object and the
article's JATS declares exactly one `<supplementary-material>` element, so one SI file is
the COMPLETE publisher deposit rather than a partial capture. The two combination strains
are released in the article's MAIN tables only and are read from the sha256-pinned
`paper.md` of the literature mirror. Eleven module-level `SourcedValue`s bind every
sourced number and sentence to a verbatim quote in that OCR; the `--data` test audits all
eleven.

### Adapter

`crispr_pinene_tolerance_niu2019_adapter.yaml` enables `bacterial perturbation`,
`crispr construct`, the `environment perturbation` pair (pinene is on the experiment AND
the reference), and the `environment response phenotype` pair. No record carries a phage,
so that lane is off, and the aTc inducer is in the medium rather than the environment,
which is what leaves pinene as the only environment perturbation.

### Release discrepancies this loader resolved, and how

Recorded first in
[[experiments.036-dataset-fixes-before-kg-build.scripts.niu2019_release_loadability]];
what the loader did with each:

- **The growth timepoint is 10 h in the SI footnote and 12 h in Methods 2.4.** The loader
  stores 12 h, on Methods ("The cultures were incubated at 37 C (200 rpm) for 12 h") plus
  an arithmetic check the release supplies itself: the main text's absolute ODs at 12 h,
  0.451 +- 0.013 against 0.127 +- 0.009, divide to 3.5512, which is the 3.55 printed for
  the activation combination strain. The ratio and the 12 h reading agree, so 12 h is the
  reading the numbers support.
- **`muts` in Suppl. Table 3 is `mutS` in the main table.** `spacer_of` matches
  case-insensitively, because a case difference in one symbol is not a second gene, and
  `perturbed_gene_name` is the pinned annotation's own spelling rather than either
  released one. That is also what keeps one locus from acquiring two common names, which
  the verifier's canonical-name rule flags.
- **`paper.md` lines 186 to 222 are a DIFFERENT article** (DeMott and Dedon on a
  phosphorothioate antiviral mechanism), bundled into the same PDF by the 2020 erratum
  list. The loader never parses `paper.md`; it reads it only through `SourcedValue`
  quotes, each of which is audited for presence, so the foreign tail cannot reach a
  record. The two combination strains' numbers are module constants with their quotes
  beside them rather than a parse of that file.
- **Suppl. Table 2's footnote says the variants "were identified with a frequency of 1.0"
  while ten rows are below 1.00 and one is blank**, and main Table 2's synonymous /
  nonsynonymous tallies do not match the SI rows. Both belong to the variant table, which
  this dataset does not store; they are #731's to carry when a release measures `YZFP`.
- **The erratum** (doi `10.1016/j.synbio.2020.10.004`) corrects a missing
  competing-interest statement and no data, so nothing here is affected.

## 2026.10.11 - Pinene measured against PubChem: it stays a recorded gap (#726, #799)

The #726 curator pass queried both spellings. PubChem's name index answers
`alpha-pinene` with CID 6654, titled (+-)-alpha-Pinene, and bare `pinene` with CID
15837102, titled (-)-Pinene, whose connectivity `CC1=C2CC(C2(C)C)CC1` is neither alpha- nor
beta-pinene. So the two do not collapse onto one row, which #726's comment had left as a
hypothesis. The table now carries an `alpha-pinene` row (for a source that states it) and
a `pinene` row with status UNRESOLVED_PUBLIC and that reason; this paper doses "0.5%
pinene" and names alpha-pinene only as the standard its production assay is quantified
against, so the dosed compound keeps its typed gap and no record changes. The two module
paragraphs that called this "issue #726's work" now record the measurement. The store was
not rebuilt (the store-reach scan reads it fresh: the gap's reason is unchanged); it was
re-verified at this branch by slurm array 3882.

### L0-L4 at this branch

#### `CrisprPineneToleranceNiu2019Dataset`

51 records; store built 2026-10-11 02:17 UTC; 1 report(s); verdict **PASS**.

Report `verification_report.json` (written 2026-10-11 08:32 UTC):

| level | rule | result | message |
|---|---|---|---|
| L0 | `structural` | pass | 51 records validated |
| L1 | `count` | pass | observed 51, expected 51 |
| L1 | `pair_uniqueness` | pass | 51 unique (study, strain, condition) records, one each |
| L2 | `value_fidelity` | pass | 51 values checked |
| L2 | `se_nonnegative` | pass | 0 values checked |
| L2 | `interval_orientation` | pass | 0 of 0 stored intervals do not bracket their value, as declared |
| L3 | `measurement_type_consistent` | pass | single measurement_type: 'log2_ratio' |
| L3 | `reference_zero` | pass | numeric rule: reference response == 0 for all 51 records |
| L3 | `environment_perturbed` | pass | all 51 experiments carry an environmental edit (perturbation, non-baseline temperature, or non-baseline media; baseline temp=37.0, media='LB with 200 nM anhydrotetracycline, formulation not stated (Niu 2019 tolerance assay), liquid') |
| L1 | `provenance_gaps` | pass | 255 documented provenance gaps over 51/51 records; 1 deferred field(s): ['inchikey']; 1428 undeclared None values over 17 carrier fields (top: Compound.inchi x255, Compound.chebi_id x204, Compound.pubchem_cid x153, Compound.smiles x153, Environment.dilution_rate_per_hour x51) |
| L1 | `canonical_gene_names` | pass | 50 systematic names, one canonical spelling each, each current in the genome |
| L2 | `uncertainty_sanity` | pass | 0 labeled uncertainties, none a zero dispersion; 51 records report n_samples >= 2 with no uncertainty |
| L3 | `compound_identity` | pass | environment edits: 0 compound references carry a structure identifier; 51 declare a typed gap (1 distinct compounds, unencodable) |
| L3 | `media_compound_identity` | pass | medium components: 102 compound references carry a structure identifier; 0 declare a typed gap (0 distinct compounds, unencodable) |
| L3 | `media_membership` | pass | 0 records on a shared MEDIA_LIBRARY medium, 51 on a medium deriving from one (1 distinct media) |
| L4 | `gene_containment_sgd` | pass | 1.000 of 50 measured genes are reference genes (>= 0.9) |
| L4 | `current_genome_genes` | pass | every one of the 50 measured systematic names is a gene of the current genome |

## 2026.10.12 - Every released target label is a record identity

The owner's rule of 2026-10-12 ([[database.genome-gating-sweep.2026.10.12]], row 243): the loader reads no annotation. Both arms now take the one released-construct leaf, `BacterialReleasedCrisprPerturbation` (#1077), whose `expression_direction` is the released fact of the arm's effector (`increased` for the dCas9*-MCPSoxS activator, `decreased` for the interference arm) rather than a class default; `systematic_gene_name` and `perturbed_gene_name` are the target label Suppl. Table 3 or 4 writes, `strain_label` is the strain's whole released label (the row label, or the hyphenated combination label shared by its six leaves) and `identifier_kind` says the label is a gene symbol or an operon label. `_resolve`, `canonical_symbol`, the `ecoli_genome` constructor parameter and the three drop rules `target_label_is_a_multi_gene_operon`, `combination_strain_contains_a_multi_gene_operon` and `target_label_is_not_in_the_bw25113_annotation` are gone.

The 3 cells they removed are records now: the two operon labels `sufBCDS` and `flgFGH` (an operon label is a record identity on this leaf, which is what the leaf's docstring states), and the activation combination strain `flgFGH-sufBCDS-dusB-rpoA-yehA-hslU`, which names two of them and is stored as one record with one leaf per listed label. `EXPECTED_RECORDS` 51 -> 54: 44 activation records (43 released growth rows + the combination strain) and 10 interference (9 + its combination), 64 leaves in all (49 `increased`, 15 `decreased`). The L1 and L4 genome-reference rows report what the pinned BW25113 annotation makes of the stored labels (#1069).

The one remaining rule is `pinene_ratio_is_a_dimensionless_product_ratio` (40 cells). **Not done here:** `ConcentrationUnit.fold_change` landed with #1077 and makes those 40 ratios storable as a `ProductTiterPhenotype`, but they are a different record type (`ProductTiterExperiment`), so they need a SECOND dataset class with its own adapter, `dataset_adapter_map` entry, KG mapping, verification-runner family entry and candidate verdict. That is a new registration rather than a gate reversal, and it is left for the follow-up that owns it.

Released vs stored: 94 released numeric cells = 54 records + 40 pinene ratios; 64 censored dashes are counted separately and stored nowhere, as before.

## 2026.10.12 - The 40 pinene ratios are a store of their own (#770 closed)

The follow-up the section above left: `PineneFoldChangeNiu2019Dataset`, 40
`ProductTiterExperiment` records on `ConcentrationUnit.fold_change`, store
`data/torchcell/pinene_fold_change_niu2019`. Every released numeric cell of the paper is
now stored, 54 in the growth store plus 40 here out of the 94 the release prints.

**The 40 cells.** 32 numeric pinene cells of Suppl. Table 3 (of 57 rows; 25 are the
footnote's dash) + 6 of Suppl. Table 4 (of 20; 14 dashed) + the two combination strains'
pinene cells, which the article's MAIN Tables 3 and 4 carry and the SI does not:
`flgFGH-sufBCDS-dusB-rpoA-yehA-hslU` at 1.71 +- 0.02 and
`ydiJ-yjbQ-prpR-marR-fabR-cedA` at 1.13 +- 0.02, each quoted verbatim off the OCR'd
table row. A dashed growth cell does not censor the pinene cell beside it: `hokC` has no
growth number and a pinene ratio of 1.19, so it is a record here and not in the sibling.

**A SEPARATE store, not a second phenotype on the growth records**, because the two
columns are two cultures. The growth ratio is OD600 after 12 h in LB plus 0.5% pinene;
the pinene ratio is a 72 h shake-flask production run in SBMSN under a 20% dodecane
overlay, inoculated from an overnight LB seed, induced at OD600 0.8 with 1 mM IPTG and
200 nM anhydrotetracycline, then held at 30 C and 130 rpm, and read by GC-FID against an
alpha-pinene standard curve. One record carrying both numbers would state that one
culture produced them.

**What the record stores.** `titer` is the released ratio and `titer_unit` is
`fold_change`, so the released dispersion rides along as released: `titer_uncertainty`
is the published sample SD over three replicates, `titer_uncertainty_type=sample_sd`,
`n_samples=3`, and `titer_se` is the derived SD/sqrt(3). This is what the growth store
cannot do, since `MeasurementType` has no fold-change member and its stored number is a
log2 of the ratio. The reference is the empty-vector control at `titer=1.0`, which is
what the ratio's denominator IS; its dispersion, replicate count and sample unit are
four typed gaps rather than numbers the paper never printed.

**The production chassis is a background, not a genotype.** `pBbB2K-dCas9*-MCPSoxS` and
the pathway plasmid `pMEVIGPS` are in every stored strain AND in the control, so they
are a `BacterialStrainBackground` whose `genotype_statement` is
`BW25113(PT5-dxs) (pBbB2K-dCas9*-MCPSoxS, pMEVIGPS)`. Only the guide plasmid
`pTargetA-X` differs between strains, which is why it is the perturbation and the
control carries the empty `pTargetA` in its place. Consequence: the family's
`heterologous_pathway_gene_counts` row reads `[0]`, declared.

**The K2HPO4 amount is stored as published.** The SBMSN recipe reads "K2HPO4 211.42"
(g/L) in both the OCR and the publisher's own PDF text layer. 211.42 g/L is implausibly
high for a buffer, and nothing is corrected: a plausible-looking 11.42 would be our
number rather than theirs, so the released value is stored with the component note
saying so. `Tween-80 2 mL` is stored as 0.2% v/v, the one-step reading of the list's own
per-litre basis, since `ConcentrationUnit` has no volume-per-volume amount member.

**Verification.** The store passes the product-titer family gate plus two rows of its
own, `titers_match_the_released_pinene_cells` (every stored titer and SD re-parsed from
the deposited `mmc1.docx` bytes and the main-table cells) and
`the_reference_is_the_empty_vector_control_at_one`:

| level | rule | result |
|---|---|---|
| L0 | `structural` | pass -- 40 records validated |
| L1 | `count` | pass -- observed 40, expected 40 |
| L2 | `value_fidelity` | pass -- 40 titers finite and non-negative |
| L2 | `uncertainty_nonnegative` | pass -- 40 released SDs |
| L2 | `se_is_the_uncertainty_over_sqrt_n` | pass -- 40 pairs within 1e-12 |
| L2 | `titers_match_the_released_pinene_cells` | pass -- 40 cells, ratio and SD |
| L3 | `titer_unit_is_the_pinned_unit` | pass -- `fold_change` |
| L3 | `uncertainty_is_typed_or_gapped` | pass |
| L3 | `replicate_design_is_sourced_or_gapped` | pass |
| L3 | `heterologous_pathway_gene_counts` | pass -- `[0]`, declared |
| L3 | `product_is_the_declared_one` | pass -- `pinene` |
| L3 | `the_reference_is_the_empty_vector_control_at_one` | pass |
| L3 | `the_pathway_plasmid_is_background_not_a_perturbation` | pass |
| L4 | `perturbed_gene_containment_assembly` | reported, never gated (#1069): 0.000 of 38 measured labels are loci of `ecoli_K12_BW25113_ASM75055v1`, since the release keys every row by a gene symbol (two of them, `flgFGH` and `sufBCDS`, by an operon label) and not by a locus tag |

Candidate verdict: `admissible_with_gaps`, the one gap being the compound `pinene`
itself, whose row stays UNRESOLVED_PUBLIC for the reason the 2026.10.11 section records
(#726).

**Released vs stored for this store:** 40 released numeric pinene cells, 40 records, 0
dropped; 39 dashed pinene cells counted as censored and stored nowhere; the 54 growth
cells belong to the sibling store.

**Built under slurm** with `--retire-existing --verify`: job 4613 wrote 40 records,
gene_set 38, 1 reference, `pinene_fold_change_niu2019: PASS (1 genome finding reported,
not gated)`. The sibling store was rebuilt in job 4614 because this change rewrote its
ledger's rule description (its 40 pinene cells are now stored by this dataset rather
than dropped for want of a unit): 54 records and 64 censored dashes unchanged,
`crispr_pinene_tolerance_niu2019: PASS (3 genome findings reported, not gated)`.
