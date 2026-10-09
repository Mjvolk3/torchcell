---
id: uy5dciuzn3229v8cp6no8kg
title: Rousset2018
desc: ''
updated: 1791405805899
created: 1791405805899
---

## 2026.10.07 - Loader, sourcing, record-type decision and the first build

Row 26 of the fifty bacterial datasets. Rousset et al. 2018, PLoS Genetics 14:e1007749,
`roussetGenomewideCRISPRdCas9Screens2018`, DOI 10.1371/journal.pgen.1007749, PMID
30403660, PMC6242692. Loader: `torchcell/datasets/ecoli/rousset2018.py`, class
`CrispriScreenRousset2018Dataset`, dev store
`$DATA_ROOT/data/torchcell/ecoli_crispri_rousset2018`.

### What the paper released, and what the loader consumes

Five independent screens of ONE pooled sgRNA library in two MG1655 derivatives carrying
an aTc-inducible dCas9. Three of the ten SI tables carry per-guide values and are the
three the loader consumes; S2, S5 and S7 Tables are gene-level medians and model
estimates derived from them, and S3 plus S8 to S10 Tables carry no per-guide value.

| SI table | file | rows | screen(s) | column(s) consumed |
|---|---|---|---|---|
| S1 Table | `pgen.1007749.s011.csv` | 59,246 | growth, 17 generations, strain LC-E75 | `log2FC` |
| S4 Table | `pgen.1007749.s014.csv` | 17,220 | phage lambda, T4, 186cIts at MOI 1, strain FR-E01 | `log2FC_lambda`, `log2FC_T4`, `log2FC_186` |
| S6 Table | `pgen.1007749.s016.csv` | 17,220 | lambda transduction assay, strain FR-E01 | `log2FC` |

S4 and S6 name the same 17,220 spacers; 17,216 of them also appear in S1's
coding-strand set and 4 appear in no S1 row at all, which is the two screens' separate
read-count filters.

### Record type, and why

`BacterialEnvironmentResponseExperiment` with an `EnvironmentResponsePhenotype`,
`measurement_type=log2_ratio`. The released value is a signed DESeq2 `log2FoldChange` of
guide abundance, and it is routinely negative. Measured over the 91,609 STORED records by
`test_the_stored_records_carry_the_measured_sign_distribution` in
`tests/torchcell/datasets/ecoli/test_rousset2018.py`, which streams the built LMDB and
counts by `screen_id` (run with `--data`):

| screen | records | negative | fraction | minimum |
|---|---|---|---|---|
| `growth_17_generations` | 23,209 | 21,471 | 0.9251 | -11.9475 |
| `phage_lambda` | 17,100 | 4,737 | 0.2770 | -2.5409 |
| `phage_T4` | 17,100 | 14,352 | 0.8393 | -2.5929 |
| `phage_186cIts` | 17,100 | 7,665 | 0.4482 | -3.3441 |
| `lambda_transduction` | 17,100 | 15,246 | 0.8916 | -10.9705 |

`FitnessPhenotype.validate_fitness` clamps every non-positive value to 0.0, so that class
would erase that signal, and no other `MeasurementType` member describes a log2 abundance
ratio. The reference carries 0.0, which is also what the environment-response verifier's
numeric L3 rule requires.

A second data test, `test_the_release_has_the_measured_shape_and_sign_distribution`,
asserts 0.9249 and -11.95 for the growth screen instead. It is not a contradiction: that
test measures the RELEASED coding-strand in-gene rows, 23,372 of them, before the 163
unstorable-symbol records are dropped, while the table above measures the 23,209 stored.
Two sets, two numbers, both pinned.

Genotype: one `BacterialCrisprInterferencePerturbation` per record, keyed by the target
gene's MG1655 b-number, with the 20-nt spacer on the shared `crispr` construct
(`effector="dCas9"`, `n_guides=1`) and the released gene symbol kept on
`identifier_mapping` (`route="gene_symbol"`). Guide spacers join the verifier's genotype
signature, so the 6 guides of one gene are 6 strains and one gene.

Environment: the three phage challenges and the transduction assay each carry one
`PhagePerturbation` at `multiplicity_of_infection=1.0`, `host_of_propagation="MG1655"`.
This is the first loader to use that leaf; `mutalik2020` is the module that recorded its
absence. The growth screen carries no environment perturbation at all.

**aTc is a COMPONENT of the two media, not an `Environment.perturbation`**, and this was
a correction. The first version made it a `SmallMoleculePerturbation`, which the adapter
layer then showed to be wrong twice over. The paper puts it in the medium ("diluted
100-fold in LB containing 1 microM aTc, 0.2% Maltose and 5 mM CaCl2" lists aTc beside the
two components the phage medium already carried), it is constant across the dataset rather
than the varied condition, and the served `_environment_perturbation_node` does not filter
phages out, so a conf enabling both `environment perturbation` and `phage perturbation`
would emit each phage twice under two labels on one content id. Moving aTc into the media
keeps both doses (1 nM growth, 1 microM phage, and the dose is what distinguishes the two
media objects) and leaves the phage as the only environment perturbation any record
carries. `role` is `other`, because aTc switches the perturbation on rather than feeding
the cell.

Because the growth screen then has no perturbation, L3 `environment_perturbed` passes it
on that rule's own base-medium clause: its 23,209 records sit on `ROUSSET2018_LB` while
the dataset's modal medium is the phage screens' `ROUSSET2018_LB_MALTOSE_CACL2` (68,400
records). The 2.9x margin is fixed by the release, so the tie cannot drift.

The lambda challenge and the lambda transduction assay have IDENTICAL cultures (the cells
experienced lambda at MOI 1 for 2 h either way); only the readout differs. They are split
by `screen_id`, which joins the verifier's condition signature, and the transduction arm
carries `assay_type=other` because its readout is the cosmid lambda packaged, not the
surviving cell pool. The other four arms carry
`assay_type=pooled_competitive_growth_barcode`, the same choice `mutalik2020` recorded for
a pooled phage challenge read out by barcode.

### Sourcing table

Quotes are verbatim substrings of two sha256-pinned OCR mirrors. Rousset 2018 `paper.md`
sha256 `46ea72979c7f11855477b557824fb62baa7a4937ae4787d930e706cd2b93db3f`; Cui 2018
(`cuiCRISPRiScreenColi2018`) `paper.md` sha256
`b8e28e16f0cbb4f8d4061f79505409c1a9aa03ed4ba8349ba4dca2327d26d22d`. Twenty-eight
`SourcedValue`s in all; `tests/torchcell/datasets/ecoli/test_rousset2018.py` audits every
one against its mirror under `--data`.

| value | source | section | verbatim quote (abridged) |
|---|---|---|---|
| reference strain MG1655 | Rousset | Methods, CRISPRi library design | "These sgRNAs target 20-nt regions adjacent to NGG sites in E. coli K-12 MG1655 (NC_000913.2)" |
| host LC-E75 | Rousset | Methods, strain construction | "This strain expresses an optimized dcas9 cassette under the control of an aTc-inducible pTet promoter integrated at the phage 186 attB site." |
| host FR-E01 | Rousset | Methods, strain construction | "a new strain FR-E01 was constructed with the same cassette integrated at the HK022 attB site to avoid any interference" |
| FR-E01 is built in MG1655 | Rousset | Methods, strain construction | "The resulting vector was electroporated into strain MG1655." |
| effector dCas9 | Rousset | Methods, strain construction | "A fragment containing dcas9 under the control of pTet promoter was amplified by Phusion PCR" |
| growth screen: 17 generations, 3 replicates, deferral to ref [26] | Rousset | Methods, High-throughput screens | "The data for the screen performed with strain LC-E75 grown in rich medium was obtained from our previous study [26]. This screen was performed over 17 generations in triplicates from independent aliquots of the library generated from 3 independent transformations into strain LC-E75." |
| growth medium LB | Cui 2018 | Methods, Bacterial strains and media | "Cells were grown in Luria-Bertani (LB) broth." |
| growth inducer 1 nM aTc | Cui 2018 | Methods, dCas9 knockdown assay | "The expression of dCas9 was then induced by addition of aTc to a final concentration of 1 nM." |
| growth replicate unit = biological | Cui 2018 | Methods, dCas9 knockdown assay | "The experiment was performed in triplicates starting from independent aliquots of the library generated from independent electroporation assays." |
| phage screen: 3 replicates, 37 C | Rousset | Methods, High-throughput screens | "The phage screen was performed in triplicates as follows: FR-E01 was grown at 37 C ... into 500 mL LB." |
| phage inducer 1 microM aTc | Rousset | Methods, High-throughput screens | "dCas9 expression was induced by addition of 1 microM aTc (Acros Organics) to trigger the silencing of the target genes." |
| phage medium LB + 1 microM aTc + 0.2% maltose + 5 mM CaCl2 | Rousset | Methods, High-throughput screens | "diluted 100-fold in LB containing 1 microM aTc, 0.2% Maltose and 5 mM CaCl2" |
| MOI 1, 2 h | Rousset | Methods, High-throughput screens | "to reach a MOI of 1 ensuring a high infection rate while limiting double infections. After 2 h at 37 C, the cultures were harvested" |
| phage names | Rousset | Results, phage host factors | "followed by infection with phage lambda, T4 or 186cIts at a multiplicity of infection (MOI) of 1" |
| stocks propagated on MG1655 | Rousset | Methods, Phage strains and stocks | "All the liquid stocks were further propagated in MG1655 grown in LB supplemented with maltose 0.2% (Sigma) and CaCl2 5 mM (Sigma) at a multiplicity of infection (MOI) of 1." |
| the statistic | Rousset | Results, essential genes | "computed the log2-transformed fold change (log2FC) from DESeq2 R package [32] as a measure of relative sgRNA fitness" |
| normalization | Rousset | Methods, Data analysis | "A paired analysis was performed to compare each sample to its initial condition. The log2FoldChange (log2FC) value represents the enrichment or depletion of each sgRNA." |
| coding-strand rule | Rousset | Methods, Data analysis | "For each gene, the median log2FC value of the sgRNAs targeting the coding strand was used for ranking (S2 and S5 Tables)." |
| phage-screen filter | Rousset | Methods, Data analysis | "For the phage screen, guides targeting the template strand of genes or outside of genes were excluded from the analysis as well as well guides with insufficient number of reads (BaseMean < 10), yielding a library of ~ 17,200 sgRNAs." |
| which tables | Rousset | Methods, Data analysis | "The lists of all sgRNAs with computed log2FC values after the growth-based screen, after phage screens and after transduction assay are provided as S1, S4 and S6 Tables respectively." |
| raw reads accession | Rousset | Methods, Data analysis | "Raw sequencing files are available on the European Nucleotide Archive ... with the accession number PRJEB28256." |

`n_samples = 3`, `sample_unit=biological_replicate` on every record: the growth screen's
three replicates are three independent library transformations and the phage screens'
three are one triplicate series of which the transduction assay is an arm.

### What could not be sourced

- **The growth screen's incubation temperature.** Rousset states 37 C for the PHAGE
  screen only, and the Cui 2018 "dCas9 knockdown assay" paragraph the growth screen defers
  to states none. Both papers state 37 C for OTHER assays, which is not this one. Typed as
  a `ProvenanceGap(field="temperature", reason=not_reported_by_primary)` with `looked_in`
  naming the Cui mirror. Guessing 37 C would be inventing the one number a reader would
  most want.
- **Per-guide uncertainty.** Each table releases one log2FC per guide and screen; DESeq2's
  `lfcSE` is not a released column. Typed as a gap on
  `environment_response_uncertainty`.
- **`gamma`.** The S1 Table caption names it ("computed fold change (log2FC, padj and
  gamma)") and neither mirrored paper defines it. Measured, not sourced: solving
  `log2FC / log2(gamma)` over the 20,540 S1 rows with `|log2FC| > 0.5` gives an implied
  generation count between 11.41 and 15.58, median 12.07, so it is not log2FC rescaled by
  the stated 17 generations either. Left out rather than stored under a field whose
  meaning it would misstate. The analysis code is at
  `gitlab.pasteur.fr/dbikard/dCas9_genome_wide_screen`, which would settle it.
- **`padj`** is defined (the DESeq2 adjusted p-value) but the schema has no p-value slot,
  so it is not stored either. This is a schema observation, not a request: a
  `p_value`/`fdr` field on `EnvironmentResponsePhenotype` would let three of the fifty
  bacterial rows keep a released statistic they currently drop.
- **In-culture phage titer.** The stocks are 10^7 pfu/microL and the infected culture's
  volume at MOI 1 is not stated, so `titer_pfu_per_ml` stays None rather than being
  back-computed.
- **LB formulation.** Neither paper states the amounts, and the project's own "LB" is
  Miller for some rows and Lennox for others, so the loader-local media list LB's three
  ingredients with `concentration=None` instead of reusing the library `LB`'s Miller
  amounts. `base_medium="LB"`, so both media still join there.

### Identifier reconciliation

The release names genes by SYMBOL, not by b-number, and both host strains are MG1655
derivatives, so no ECK crosswalk is needed: one `reconcile_locus_tags` call against
`GCA_000005845.2` over the UNION of every symbol the three tables name, so a gene has one
identity across the five screens. Reconciling per table would let a symbol that collides
in one table and not in another be stored two different ways.

| quantity | count |
|---|---|
| released distinct symbols (coding-strand, in-gene rows of all three tables) | 3,944 |
| resolved to one locus (`renamed` + `non_gene_feature`) | 3,912 (0.9919) |
| stored as a b-number | 3,896 |
| resolver layer: gene symbol | 3,734 |
| resolver layer: gene synonym (ECK/JW) | 179 |
| resolver layer: not found | 31 |
| status `renamed` | 3,853 |
| status `non_gene_feature` (pseudogene locus) | 59 |
| status `retired` | 31 |
| status `ambiguous` | 1 (`rffT` -> b3793, b4481) |
| resolved only by a case-insensitive match | 0 |

The threshold is `MIN_RESOLVED_FRACTION = 0.95` and the release clears it at 0.9919.

### Retention ledger

Source cells: 59,246 (S1) + 3 x 17,220 (S4) + 17,220 (S6) = **128,126**.

| rule | scope | items | records dropped |
|---|---|---|---|
| `guide_targets_no_gene` | guide | 5,063 S1 rows | 5,063 |
| `guide_targets_the_template_strand` | guide | 30,811 S1 rows | 30,811 |
| `gene_symbol_is_not_in_the_mg1655_annotation` | gene | 31 symbols | 307 |
| `gene_symbol_collides_with_another_symbol_on_one_mg1655_locus` | gene | 16 symbols | 321 |
| `gene_symbol_is_ambiguous_in_mg1655` | gene | 1 symbol (`rffT`) | 15 |
| | | **total dropped** | **36,517** |

128,126 - 36,517 = **91,609 records**, and the loader raises if the per-rule totals do not
sum to the shortfall.

The template-strand drop is the one worth arguing. Those 30,811 rows ARE released values,
but the only leaf available is `BacterialCrisprInterferencePerturbation`, whose
`expression_direction` is fixed to `"decreased"` with no slot for the target strand.
dCas9 bound to the template strand does not block elongation: in S2 Table the median
`median_template` is the no-effect baseline (-0.41 overall, -0.46 for essential genes)
against a median `median_coding` of -5.12 for essential genes. Storing a template-strand
guide as a knockdown would assert a perturbation the paper's own data says did not happen,
and the paper's own gene scores and its phage screens both use coding-strand guides only.
If a target-strand field is ever added to `CrisprConstruct`, those rows are recoverable
from the same pinned tables.

The candidate table's **236,000** instances is 59,000 guides x 4 conditions, which counts
every guide in every condition. The phage and transduction screens were released on the
filtered 17,220-guide library, and the growth screen's template-strand and intergenic
guides are not gene perturbations, so the released count at this grain is 91,609.

### Raw mirror

`$DATA_ROOT/torchcell-raw/roussetGenomewideCRISPRdCas9Screens2018/` with
`manifest.json`, three files under `data/`. Retrieval method `pmc_cloud`
(`torchcell.literature.retrieve.pmc_cloud_object`), prefix `PMC6242692.1`, retrieved
2026-10-07. Re-fetching reproduced the sha256 the literature mirror recorded for all
three, so the retrieval is live and the pin is the anchor.

| file | sha256 | bytes |
|---|---|---|
| `data/pgen.1007749.s011.csv` | `015ebda56925fee97cd2c44557b146799299f883b2255ed22836ebab61875ebb` | 7,092,023 |
| `data/pgen.1007749.s014.csv` | `3ee887e7596d2917f1fde1ce41dfcea0b79538ece0f0d654e01dad31ec1686a2` | 1,989,407 |
| `data/pgen.1007749.s016.csv` | `c960a214350944f6c98e945410ee0a7d09b18e587f3ee9cf16a7b6e4cd360b2f` | 1,353,365 |

`si_expected` records what was deliberately NOT mirrored: S2, S5 and S7 Tables (derived
gene-level summaries) and ENA BioProject PRJEB28256 (the raw reads).

### Build

`PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb --dataset CrispriScreenRousset2018Dataset`

```
BUILT CrispriScreenRousset2018Dataset: 91609 records at
/scratch/projects/torchcell-scratch/data/torchcell/ecoli_crispri_rousset2018 in 50s;
gene_set size 3896; references 5
```

Five experiment references: one per screen, each pinning the host strain's background
(`LC-E75` for the growth screen, `FR-E01` for the four phage-derived screens) on the one
MG1655 assembly. `python -m torchcell.provenance.build_manifest` reads
`ecoli_crispri_rousset2018` as `fresh` with no drift.

The store was built three times and the first two were retired through
`scripts/deprecate.sh` rather than deleted: once on the original aTc-as-perturbation
model, once on the corrected model from a tree that still had uncommitted edits
(`build_manifest` recorded `torchcell_dirty: true`), and finally on the clean commit, so
the served manifest pins a committed tree. Record count, gene set and reference count are
identical across the second and third builds; only the manifest's commit pin differs.

### Follow-up worth recording

Rousset's S1 Table IS the LC-E75 screen of Cui 2018, re-analyzed and released per guide
("The data for the screen performed with strain LC-E75 grown in rich medium was obtained
from our previous study [26]"). If a Cui 2018 loader is added, the two must be
de-duplicated on the superset rule rather than both serving the LC-E75 guide-level values:
Cui's own release covers LC-E18 as well as LC-E75, so which one is the superset depends on
what Cui released per guide. Not resolved here; flagged so it is not discovered after both
are served.

### Verification, L0 to L4

`rousset2018.verify_build(root)` runs the shared environment-response verifier against
the MG1655 genome the records pin (resolver plus the L4 universe of all 4,651 GenBank
loci, pseudogenes and RNA tags included) and appends one SUPPLEMENTARY row, the
per-screen census. Report at
`$DATA_ROOT/data/torchcell/ecoli_crispri_rousset2018/preprocess/verification_report.json`.

It uses the STREAMING verifier and streams the LMDB twice, once for the verifier and once
for the census, so the 91,609 records are never materialized. That is the choice Price
2018 and Borchert 2024 make for the other two large bacterial stores, and the difference
is measured, not assumed: an eager `load_records` run of this store took about 25 minutes
against 8 min 48 s for two streaming passes, on identical data and with an identical
verdict.

`ecoli_crispri_rousset2018: PASS`, 17 rows, every one ok:

| level | rule | result |
|---|---|---|
| L0 | structural | 91,609 records validated |
| L1 | count | observed 91,609, expected 91,609 |
| L1 | pair_uniqueness | 91,609 unique (study, strain, condition) records, one each |
| L1 | provenance_gaps | 298,036 documented gaps over 91,609/91,609 records, 0 deferred |
| L1 | canonical_gene_names | 3,896 systematic names, one spelling each; 43 common names the resolver cannot place, all 43 pseudogene loci that resolve to themselves |
| L1 | screen_census (SUPPLEMENTARY) | 5 screens, 23,209 / 17,100 / 17,100 / 17,100 / 17,100 |
| L2 | value_fidelity | 91,609 values checked |
| L2 | uncertainty_sanity | 0 labeled uncertainties; 91,609 records report n_samples >= 2 with no uncertainty |
| L3 | measurement_type_consistent | single measurement_type `log2_ratio` |
| L3 | reference_zero | numeric rule: reference response == 0 for all 91,609 |
| L3 | environment_perturbed | all 91,609 carry an environmental edit (68,400 a phage, 23,209 the non-modal medium; the rule's reported baseline is the phage medium) |
| L3 | compound_identity | 0 environment-edit compound references (a phage has no compound), 0 gaps |
| L3 | media_compound_identity | 320,018 medium-component references carry a structure identifier, 0 gaps |
| L3 | media_membership | 91,609 records on a medium deriving from a MEDIA_LIBRARY key, 2 distinct media |
| L4 | gene_containment | 1.000 of 3,896 measured genes are in the pinned assembly |
| L4 | current_genome_genes | every one of the 3,896 names is a gene of the current genome |

Two labels read oddly and are harmless: `pair_uniqueness` passing at 91,609 unique keys is
what `screen_id` in the condition signature buys (without it the lambda challenge and the
lambda transduction assay would collide on 17,100 keys), and `gene_containment_sgd` is the
shared row's name for the universe it was handed, which here is the MG1655 locus set, not
S288C.

The `uncertainty_sanity` row stating "91,609 records report n_samples >= 2 with no
uncertainty" is the released state, not an omission: the triplicate design is sourced and
no per-guide dispersion was released.

## 2026.10.07 - Adapter, and the conf rule the phage leaf imposes

The bacterial-adapter tranche landed on `main` while this branch was open
(`FEAT(adapters): BioCypher adapters for the 20 E. coli and P. putida dataset classes`).
Its completeness tests assert that every registered bacterial dataset is in
`dataset_adapter_map` and in `kg_bacteria.yaml`, so a new loader without an adapter is a
red branch by that tranche's design. This note's row therefore ships one.

### What was added

| file | what |
|---|---|
| `torchcell/adapters/rousset2018_adapter.py` | `CrispriScreenRousset2018Adapter` |
| `torchcell/adapters/conf/ecoli_crispri_rousset2018_adapter.yaml` | the enable-list |
| `torchcell/knowledge_graphs/dataset_adapter_map.py` | the dataset-to-adapter pair |
| `torchcell/knowledge_graphs/conf/kg_bacteria.yaml` | one more rehearsal dataset |
| `tests/torchcell/adapters/test_rousset2018_adapter.py` | the paired test |

### The conf rule, and why it forced the aTc correction

`phage perturbation` is its own graph node class with its own node method, added because
`environment perturbation` is a SERVED class and giving it an MOI column would be a full
rebuild. The served `_environment_perturbation_node` iterates every perturbation with no
type filter, so it emits a phage under the `environment perturbation` label; and
`_phage_perturbation_node` content-addresses a phage with the SAME projection
(`_environment_perturbation_node_id`). A conf enabling both classes therefore writes each
phage twice, once under each label, on one node id. The two classes are mutually
exclusive per conf, which `cell_adapter` states in a comment and this conf is the first to
have to obey.

That is what settled aTc. With aTc on the environment axis the records carried a small
molecule AND a phage, so neither single class sufficed: `environment perturbation` alone
mislabels the phage, `phage perturbation` alone drops aTc (and the shared data-gated check
`assert_dev_store_graph` fails a conf that leaves off a family the records carry). Moving
aTc into the media, which is where the paper puts it, makes the phage the only
environment perturbation and the conf legal. The alternative was editing a served adapter
method to filter phages out, which the plan's section 5 says is NOT additive and forces a
full rebuild of every dataset importing it; that is not this row's call to make.

### Conf shape

`bacterial perturbation` + `crispr construct` (the guide spacer), `phage perturbation`
and NOT `environment perturbation`, `environment response phenotype`, and the usual
environment, media, temperature, dataset and publication nodes. The phage nodes ride the
existing `environment perturbation to environment` edges, whose graph class already
declares `source: [environment perturbation, phage perturbation]`. Both phage node
methods and both temperature node methods emit nothing for the growth screen's 23,209
records (no phage, temperature a typed gap), which the data-gated graph check exercises.

### Shared harness, extended additively

`tests/torchcell/adapters/_adapter_init_harness.py` gained `Shape.phage`, the two phage
node names in registration order, the phage alternatives on the two
`environment perturbation to environment` edge endpoints and the phage node's entry in
`NODE_LINK`. `_bacterial_adapter_cases.py` gained a `phage` flag on `_case`, and its
blanket `assert "phage perturbation (chunked)" not in names` became per-case: a conf
serves a phage exactly when its case says so, and never both environment-side classes.
The hardcoded counts moved with it: the three bacterial-dataset pins in
`test_bacterial_adapters.py` and `test_build_time_projection.py`'s `dataset_adapter_map`
pin. Those pins collided twice in one session as other branches landed their own bacterial
rows (first the tranche itself, then the Gupta 2024 protein-turnover row), which is what
they are for; the final values are 22 bacterial datasets and 73 mapped datasets. No served
adapter method was touched.

### Not run

No knowledge-graph build, no `kg_bacteria` rehearsal generation and no slurm job. The
adapter is wired and its graph is checked against the dev store by
`assert_dev_store_graph` under `--data`; generating CSVs is plan step 10's call.

## 2026.10.09 - Where the paper puts the aTc, read again from the bytes (#756)

Issue #756 recorded that this loader moved anhydrotetracycline from
`Environment.perturbations` into the two media, and that the move was prompted by an
adapter constraint rather than by the source. The constraint is now gone (the served
`_environment_perturbation_node` filters phages, so a conf may enable both
environment-side lanes), so the placement was re-derived from the paper alone, with no
adapter consideration in it.

Read from the mirror,
`/scratch/projects/torchcell-scratch/torchcell-library/roussetGenomewideCRISPRdCas9Screens2018/paper.md`,
sha256 `46ea72979c7f11855477b557824fb62baa7a4937ae4787d930e706cd2b93db3f` (role
`paper_ocr`, MinerU 2.7.6, 200 dpi). The OCR renders numerals as inline LaTeX, so the
quotes below are the mirror bytes, not the printed glyphs.

**The paper writes aTc in recipe form twice, as an item of a medium's composition.**

Methods, `High-throughput screens`, line 196, the culture the sgRNA distributions are
sampled from:

> The culture was grown to stationary phase $\mathrm { \Delta } \mathrm { \langle { O D } } _ { 6 0 0 } = 2$ ) and diluted 100-fold in LB containing $1 \mu \mathrm { M }$ aTc, $0 . 2 \%$ Maltose and $5 \mathrm { m M } \mathrm { C a C l } _ { 2 }$ .

Methods, `Infection dynamics`, line 210, the same form with the kanamycin added:

> Strains were grown overnight and diluted 100-fold in LB medium containing $0 . 2 \%$ maltose, $1 \mu \mathrm { M }$ aTc, 5 mM $\mathrm { C a C l } _ { 2 }$ and kanamycin.

Methods, `RT-qPCR`, line 218:

> Overnight cultures were diluted 1:100 in $3 \mathrm { m L }$ LB containing $1 \mu \mathrm { M }$ aTc.

**The paper also writes it once as an event**, in the sentence immediately before the
first of those, which is why the reading is a judgment and not a transcription. Methods,
`High-throughput screens`, line 196:

> At $\mathrm { O D } _ { 6 0 0 } = 0 . 2$ , dCas9 expression was induced by addition of $1 \mu \mathrm { M }$ aTc (Acros Organics) to trigger the silencing of the target genes.

and the S6 Fig legend, line 235, uses aTc as a two-level variable in a control experiment
this dataset does not store:

> (A) Relative lexA or rho expression was measured in presence of a lexA- or rho-targeted sgRNA respectively, with or without dCas9 repression (± aTc), showing a low repression activity $6 6 . 4 \%$ and $8 5 . 2 \%$ respectively).

**Decision: aTc stays a `MediaComponent`, on the paper's own recipes.** The treatment
sentence is the initiating addition in the pre-infection outgrowth; every culture
downstream of it, including the one that is sampled and the one `Infection dynamics`
describes, is written as a medium containing aTc at 1 uM beside the maltose and the
CaCl2, both of which this loader already carries as components. The recipe reading is the
one that covers the stored records. Both quotes are in the loader, as
`PHAGE_SCREEN_MEDIUM` and `PHAGE_SCREEN_INDUCTION`, so the competing reading is on the
record rather than argued away.

Supporting, and NOT the reason: S4 Table (`si/si14.csv`, sha256
`3ee887e7596d2917f1fde1ce41dfcea0b79538ece0f0d654e01dad31ec1686a2`) has the columns
`target,position,ori,gene,essential,gene_left,gene_right,gene_ori,log2FC_lambda,log2FC_T4,log2FC_186`,
so the released condition axis is the phage alone and aTc is constant at 1 uM across
every scored arm and the reference. On the environment axis it would be a perturbation
nothing in the dataset contrasts.

A genuine gap, stated rather than filled: the growth-screen arm's aTc protocol is deferred
to reference [26] (Cui 2018) and does not appear in this paper. That arm is not stored
here, so nothing in this build depends on it.

### What changed in the tree

No record changed, so the dev store was not rebuilt: the loader edit is confined to
docstrings (`_atc` and the ENVIRONMENT bullet), which no fingerprint reads. The conf
header no longer gives the adapter as a reason for the phage-only enable-list; the reason
is that no record carries a non-phage environment perturbation, so the served lane would
emit nothing. Verified on the dev store with `--data`: the phage lane emits the phage and
the served lane emits nothing over the same records.
