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
