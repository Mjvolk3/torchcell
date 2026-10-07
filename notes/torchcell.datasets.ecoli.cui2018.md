---
id: qbh95cppilzw41bvz306wrq
title: Cui2018
desc: ''
updated: 1791405939755
created: 1791405939755
---

## 2026.10.07 - Cui 2018 CRISPRi screen loader: sourcing, record type, retention and build

Row 31 of the fifty bacterial datasets. Cui et al. 2018, *A CRISPRi screen in E. coli
reveals sequence-specific toxicity of dCas9*, Nat Commun 9:1912,
doi:10.1038/s41467-018-04209-5, PMID 29765036, PMCID PMC5954155, citation key
`cuiCRISPRiScreenColi2018`.

Loader: `torchcell/datasets/ecoli/cui2018.py`, class `CrispriKnockdownCui2018Dataset`.
Tests: `tests/torchcell/datasets/ecoli/test_cui2018.py`.

### What the release reports, and the one thing it does not separate

The paper's headline finding is the **bad-seed effect**: guides sharing certain five
PAM-proximal bases produce strong fitness defects or kill the cell "regardless of the
other 15 nucleotides of guide sequence". That makes the obvious sourcing question the
important one, and the answer is explicit.

**The release does NOT separate a guide's own sequence effect from its target's
phenotype.** Supplementary Data 5 carries exactly ten columns, `guide, gene, essential,
pos, ori, coding, fit18, fit75, ntargets, seq`, and `fit18` / `fit75` are the RAW
per-guide log2 fold change in each strain. The decomposition into on-target repression,
off-target binding at 9-or-more-nt seed matches, and sequence-intrinsic toxicity lives in
the paper's model (a locally connected network over the one-hot 20-nt spacer, Pearson r
0.56, RMSE 0.81, trained only on guides in regions a regression tree called neutral).
Neither the predictions nor any corrected per-guide value is a released column. So what
the loader stores is the measured log2FC, and it confounds those three causes. The
paper's own design rule (v) is the honest reading of one record:

> For the reasons described above, the effects of genes on a given phenotype should
> ideally not be inferred from the effect of a single guide but rather from the
> statistical analyses of several guides.

A consumer that wants a gene-level call must aggregate over that gene's guides (the
library averages 19 targets per gene), and should prefer the LC-E75 screen, where the
bad-seed effect is largely alleviated.

**Supplementary Data 3 is the bad-seed quantification, and it is deliberately NOT
loaded.** It gives `N_targets`, `mean`, `pval`, `pval.adj` and `std` for each of 1,022
five-base seeds in each strain. Its unit of observation is a five-nucleotide SEQUENCE,
not a strain, so it is not a `genotype x environment -> phenotype` record and no
phenotype in the schema can carry it. The raw mirror's `si_expected` says so.

### Record type, and why

| decision | value | why |
|---|---|---|
| phenotype | `EnvironmentResponsePhenotype`, `measurement_type=log2_ratio` | the readout is a SIGNED log2 fold change (measured range `fit18` -11.93 to +1.71, `fit75` -12.48 to +0.89, with 6,732 and 6,110 positive values). `FitnessPhenotype` clamps non-positive values and its verifier requires a 1.0 reference, so it cannot hold this number |
| assay | `AssayType.pooled_competitive_growth_barcode` | guide abundance in one pooled culture read by deep sequencing at the start and the end of a 17-generation competition |
| perturbation | `BacterialCrisprInterferencePerturbation`, `gene_namespace="ecoli_k12_mg1655_bnumber"` | a dCas9 knockdown of a present gene named by a b-number |
| guide spacer | stored on `CrisprConstruct.guide_sequence` | the spacer IS the perturbation's identity: it determines the target and the strand, and it is what makes two guides of one gene two strains rather than two duplicates |
| experiment | `BacterialEnvironmentResponseExperiment` / `...Reference` | the bacterial environment-response pair |

`n_guides=1` on the construct is a property of the construct, not of the library: one
psgRNA plasmid carries one spacer. The "average of 19 targets per gene" is the number of
RECORDS a gene has.

### Sourcing table

Quotes are verbatim substrings of the pinned mirror bytes, and a `data`-marked test
audits every one of them through `audit_sourced_value`. `paper.md`
sha256 `b8e28e16f0cbb4f8d4061f79505409c1a9aa03ed4ba8349ba4dca2327d26d22d`; `si/si1.md`
sha256 `fd8cbfc076133e745eb5f31e46626869e1baf368d1e41e78fdfe853370f56f95`.

| value | stored | source | verbatim quote (abridged where long) |
|---|---|---|---|
| reference strain | MG1655, GenBank ASM584v2 `GCA_000005845.2` | paper.md, Methods, Library construction | "The library was designed by randomly choosing targets with a proper NGG PAM around the genome of $E$ . coli strain MG1655 (NC_000913.2)." |
| library size | 92,919 oligonucleotides | paper.md, Methods, Library construction | same sentence, "A pool of 92,919 oligonucleotides (synthesized by CustomArray) was amplified" |
| effector | `dCas9` | paper.md, Results | "electroporated in strain LC-E18 carrying the dCas9 gene under the control of a Ptet promoter in the chromosome" |
| readout | log2 fold change in guide abundance | paper.md, Results | "The effect of each guide on the cell fitness can be measured as the fold change in abundance (log2FC) of the guide RNA in the library during the course of the experiment, as measured through deep sequencing of the library." |
| `n_samples` = **3** | 3 | paper.md, Methods, dCas9 knockdown assay | "The experiment was performed in triplicates starting from independent aliquots of the library generated from independent electroporation assays." |
| `sample_unit` = `biological_replicate` | | same quote | independent aliquots from independent electroporations |
| normalization | control guide `TGAGACCAGTCTAGGTCTCG` | paper.md, Methods, Fold-change computation | "The fold change in abundance of each guide RNA was computed from read counts using DESeq2 ... using data from the three replicates and normalized to the control guide 5'-TGAGACCAGTCTAGGTCTCG-3'." |
| read filter | 20 reads, already applied by the authors | paper.md, same section | "Guides with a total number of reads across samples <20 were discarded from the analysis." |
| uncertainty TYPE | **none released** | paper.md, same section | the ten released columns carry no SE. DESeq2 fits an `lfcSE` per guide and the release does not carry it, so `environment_response_se`, `environment_response_uncertainty` and `environment_response_uncertainty_type` are typed `ProvenanceGap`s |
| medium | `MEDIA_LIBRARY["LB"]`, formulation UNSTATED | paper.md, Methods, Bacterial strains and media | "Cells were grown in Luria-Bertani (LB) broth. LB agar 1.5% was used as solid medium." |
| temperature | 37 C | si/si1.md, Supplementary Figure 1 legend | "Cells were grown at 37 C in LB supplemented with aTc and the psgRNA library was extracted and sequenced at the beginning and at the end of the experiment." |
| inducer | anhydrotetracycline, 1 nM | paper.md, Methods, dCas9 knockdown assay | "The expression of dCas9 was then induced by addition of aTc to a final concentration of 1 nM ." |
| duration | 17 generations | paper.md, same section | "Cells were grown for 17 generations by diluting the culture 100-fold once it reached OD600 2.2-2.5." |
| LC-E18 background | Supplementary Table 7 row | si/si1.md | the row's cells are `LC-E18`, `MG1655`, `pOSIP-KL-sulA-GFP`, `N/A`, `pOSIP-KH-RBS2-dCas9`; the stored quote is the OCR's HTML row verbatim, so it is a real substring of the pinned bytes |
| LC-E75 background | Supplementary Table 7 row | si/si1.md | the row's cells are `LC-E75`, `MG1655`, `pOSIP-KL-mcherry`, `pOSIP-CO-RBS-library-dCas9 (2-3)*`, `N/A` |
| dCas9 dose difference | 2.6-fold | paper.md, Results | "The expression cassette selected in this manner displayed an expression level 2.6-time lower than the original strain LC-E18 and was integrated in strain LC-E75." |

**The uncertainty type was combed for, not assumed absent.** Searched in paper.md and in
all three OCR'd SI documents for `standard deviation`, `SD`, `SE`, `standard error`,
`bootstrap`, `variance`, `confidence`, `n =`, `replicate`, and in the released CSV's own
header. The only dispersion the paper releases anywhere is `std_E18` / `std_E75` in
Supplementary Data 3, which is the SD of the log2FC ACROSS the guides sharing one seed
sequence, not an uncertainty of any guide's own measurement. Using it would attach a
between-guide spread to a within-guide estimate. The column descriptions are also a
documented deferral: the rebuttal answers Reviewer #1's "it is not clear what the column
labels in the files mean" with "We have now made all code and corresponding data tables
available as jupyter notebooks at the following address, where the content of the tables
is also extensively described: <https://gitlab.pasteur.fr/dbikard/badSeed_public>". That
notebook is not mirrored and nothing is loaded from it; what the loader needs about the
columns is established from the Methods and from the released values themselves (see
"The released table's structure is checked, not assumed" below).

### Two defects in the paper's own bookkeeping, recorded

1. **The Data availability sentence misnumbers its own file.** It says "The screen
   results are provided as Supplementary Data 4", while the Description of Additional
   Supplementary Files names Supplementary Data 4 "Plasmid sequences" and Supplementary
   Data 5 "Screen results". The bytes the loader consumes are MOESM8, whose columns are
   the screen results, so Supplementary Data 5 is the file and the sentence is off by one.
2. **The library size is quoted three ways.** "~92,000" unique guides in Results, 92,919
   oligonucleotides in Methods, and "84,215" in the Discussion. The released table holds
   85,381 rows over **78,137** distinct guides after the authors' own 20-read filter.
   None of the three paper figures is the record count, so the loader pins the measured
   78,137 and quotes the 92,919 pool as the library size.

### Reference, strain backgrounds and the control guide

The two screens differ only in dCas9 expression level, which is a chromosomal cassette,
not an environment. So they are two `BacterialStrainBackground` objects on the reference:

- `LC-E18`: MG1655 with `pOSIP-KH-RBS2-dCas9` at the HK022 attB site (the original
  Ptet-dCas9 cassette) and `pOSIP-KL-sulA-GFP` at the lambda attB site.
- `LC-E75`: MG1655 with `pOSIP-CO-RBS-library-dCas9 (2-3)` at the primary 186 attB site
  (dCas9 2.6-fold lower) and `pOSIP-KL-mcherry` at the lambda attB site.

`alleles=[]` on both, and that is a statement rather than an omission: both cassettes are
pOSIP integrations at phage attachment sites and the paper names no disrupted MG1655
locus, so there is no allele of a b-numbered gene to type. The strain table's row is kept
verbatim in `genotype_statement`. Note the main text says LC-E75 was built "by
integrating plasmid pIT5-KL-mcherry into strain LC-E69" while Supplementary Table 7 and
the Supplementary Figure 9 legend both say the lambda-attB integration is
`pOSIP-KL-mcherry`; the table is taken as the construction record and the discrepancy is
recorded here rather than resolved.

**The non-targeting control guide is a real design element and it is typed.** Every fold
change is normalized to `TGAGACCAGTCTAGGTCTCG`, and that spacer appears in **0** of the
85,381 released rows (measured). It is therefore the `phenotype_reference`: the same
environment, the same strain, `environment_response=0.0`, which is what the
normalization makes the control guide's log2FC by construction. No record claims it as a
genotype, because no genotype it perturbs was released.

### Positions are not stored, and the reason is a genome version

The library was designed on **NC_000913.2** while the pinned assembly set is GenBank
**ASM584v2 (U00096.3)**, so the released `pos` column is in coordinates one annotation
release older than the records' own assembly. There is no field for a guide target
position on the CRISPRi leaf, and a liftover is not something the release supports, so
positions are not stored. Nothing is lost: the 20-nt spacer is on every record and
determines the target and the strand against whatever assembly a consumer pins. The
`pos`, `ori`, `coding` and `essential` cells are kept verbatim in
`preprocess/guide_retention.csv`.

The same version gap is what the identifier reconciliation measures, below.

### The released table's structure is checked, not assumed

A released row is a **(guide, target position)** pair and a record is a **(guide,
screen)**. Measured on the pinned bytes and asserted by `collapse_to_guides`:

- 85,381 rows over 78,137 distinct guides;
- each guide appears exactly `ntargets` times, once per perfect chromosomal match
  (checked for every guide: 0 mismatches);
- `fit18` and `fit75` are CONSTANT across a guide's rows (0 guides carry two values of
  either), so the measurement belongs to the guide, not the position;
- the 60-nt `seq` window is populated only for single-target guides (9,056 of the rows,
  all multi-target, carry an empty `seq`), which is how the authors themselves treat a
  multi-match guide as positionally ambiguous.

Writing one record per row would therefore replicate one measurement across up to 47
positions. A multi-target guide instead becomes ONE record carrying one knockdown
perturbation per distinct targeted gene. Those guides are genuine multi-locus
knockdowns. The 451 guides with 7 rows are almost all the seven rRNA operons (the
most frequent gene cells among those rows are `rrlA`-`rrlH` at 268 rows each and
`rrsA`-`rrsH` at 126 each, then `insCD-6` at 45); the smaller multiplicities are the
`insL-1/2/3` and `insH-1..11` IS copies, the `rhsA/rhsB/rhsC` paralogs and the
`glnU`/`glnW` tRNA pair.

### Identifier reconciliation: gene symbols to b-numbers

The release names **no b-numbers** (checked: 0 of its 4,360 distinct gene cells match
`^b\d{4}$`); it names gene symbols and EcoCyc-style ids. So the ECK crosswalk is not used
here, and every stored tag is DERIVED from a symbol and carries a
`DerivedIdentifierMapping(source_identifier=<released symbol>, route="gene_symbol")`.

`reconcile_locus_tags` against `ecoli_K12_MG1655_ASM584v2`, over the 4,360 distinct
source gene cells:

| status | count |
|---|---|
| `current` | 0 |
| `renamed` (a symbol or synonym of exactly one locus) | 4,217 |
| `non_gene_feature` (a pseudogene locus) | 69 |
| `retired` | 73 |
| `ambiguous` | 1 (`rffT` -> `b3793`, `b4481`) |

| resolver layer | count |
|---|---|
| locus tag | 0 |
| old locus tag | 0 |
| RefSeq locus tag | 0 |
| gene symbol | 4,093 |
| gene synonym (ECK) | 194 |
| not found | 73 |

Resolved 4,286 / 4,360 = **0.983**; `MIN_RESOLVED_FRACTION` is stated at 0.95. 4,266
names are remapped to a different string than they came in as. **94 names end up outside
the b-number namespace** and the perturbation leaf refuses them: the 73 retired, the 20
kept-on-collision, and the 1 ambiguous.

All three groups are the cost of reading a NC_000913.2-designed library against
ASM584v2. The retired names are mostly IS-element genes (`insAB-1..6`, `insCD-1..6`,
`insEF-1..5`, `insH-1..11`, `insI-1..3`, `insL-1..3`, `insN-1/2`, `insO-1/2`), small
RNAs and EcoCyc ids (`C0299`, `C0362`, `G0-10697..G0-10706`, `sroG`, `tpke11`, `tpke70`,
`nc1`), and split-locus spellings (`efeU_1`, `gatR_1/2`, `phnE_1`, `lomR_1/2`,
`ypjM_1/2/3`). The collisions are 20 symbols that resolve to a locus another source
symbol also resolves to (`mdtQ`, `yaiF`, `yaiT`, `yaiU`, `yaiX`, `ycgI`, `ydeK`, `ydeU`,
`yfcT`, `yfcU`, `ygaX`, `ygaY`, `yjiP`, `yjiQ`, `ymgH`, `yncI`, `yncM`, `yoeA`, `yoeE`,
`yohH`), i.e. genes that were separate in .2 and merged in .3; `reconcile_locus_tags`
keeps both as given so two source genes never collapse into one record identity, which
leaves neither as a b-number.

### Retention ledger, with the arithmetic

A guide is retained when EVERY perfect chromosomal match lies in an annotated gene whose
source symbol resolves to a b-number of the pinned assembly. Rules in application order:

| drop reason | guides | measurements (x2 screens) |
|---|---|---|
| `no_target_in_a_gene` | 6,268 | 12,536 |
| `target_outside_a_gene` (some matches in a gene, some not) | 173 | 346 |
| `gene_retired_in_the_pinned_assembly` | 619 | 1,238 |
| `gene_symbol_shared_with_another_source_name` | 283 | 566 |
| `gene_symbol_ambiguous_in_the_pinned_assembly` | 23 | 46 |
| **dropped total** | **7,366** | **14,732** |
| **retained** | **70,771** | **141,542** |
| sum | 78,137 | 156,274 |

`78,137 guides x 2 screens = 156,274 candidate measurements`;
`141,542 kept + 14,732 dropped = 156,274`. `BuildAccounting.check()` asserts both
identities and refuses any per-reason count other than the pinned ones.

`target_outside_a_gene` is the rule worth naming: a guide with four perfect matches, three
in `rhsA/rhsB/rhsC` and one intergenic, would become a record naming three of the four
loci dCas9 actually binds. That is a genotype the release does not support, so the
measurement is dropped and ledgered rather than partly asserted. The three `gene_*`
reasons are not a modeling choice at all: the leaf's `systematic_gene_name` validator
refuses anything but `^b\d{4}$`, and `torchcell/datamodels/schema.py` is not touched by
this work.

### Build numbers

```
PYTHONPATH=$PWD python -m torchcell.database.build_dataset_lmdb \
  --dataset CrispriKnockdownCui2018Dataset
```

| quantity | value |
|---|---|
| records | **141,542** |
| gene-set size | **4,263** b-numbers |
| experiment references | **2** (one per screened strain) |
| perturbations written | 148,516 |
| wall time | **81 s** |
| build manifest | `fresh` under `python -m torchcell.provenance.build_manifest` |

Perturbations per record: 69,825 records with 1, then 266 with 2, 164 with 3, 45 with 4,
16 with 5, 46 with 6, 399 with 7 (mostly the rRNA operons) and 10 with 8, per screen.

`preprocess/` holds `build_accounting.json` (the arithmetic above plus the
reconciliation report and nine notes) and `guide_retention.csv` (one row per distinct
guide: its spacer, `ntargets`, released gene cells, released positions, stored b-numbers,
both fold changes and its verdict).

### Raw mirror

`$DATA_ROOT/torchcell-raw/cuiCRISPRiScreenColi2018/data/41467_2018_4209_MOESM8_ESM.csv`,
12,080,844 bytes, sha256
`95ebaa5a0c92c63849617f48889e2d28b7805fdffdd960527f8a501381143c1e`. Retrieved
2026-10-07 through `torchcell.literature.retrieve.pmc_cloud_object` with key
`PMC5954155.1/41467_2018_4209_MOESM8_ESM.csv` (`RetrievalMethod.pmc_cloud`), which is
directly scriptable and reproduced both the digest and the byte count. Exactly the one
file the loader consumes is kept; the manifest's `si_expected` names the other five
released supplementary files and why each is not deposited, plus the two things the paper
does not release at all: any sequence-read accession (the Data availability statement
defers to "the corresponding author upon request", so the fold changes cannot be
recomputed from reads) and the resequenced bad-seed-suppressor genomes.

### Medium: the one value the source does not pin

The Methods say only "Cells were grown in Luria-Bertani (LB) broth" and print no amounts,
so the formulation is genuinely UNSTATED and neither `MEDIA_LIBRARY["LB"]` (Miller, 10
g/L NaCl) nor `MEDIA_LIBRARY["LB_LENNOX"]` (5 g/L) is excluded by the text. The loader
takes the shared unqualified-LB object, which is what the rows that name "LB Miller"
without stating amounts also take, so this screen joins them. **No media entry was
added**: `torchcell/datamodels/media.py` is one of the fingerprinted shared value files
`kg_manifest` watches, and adding a `LB_DEFERRED` object there would move a value surface
for every served dataset's admission check, which is not worth doing for a formulation
the source never stated. The choice is stated in the loader docstring, in
`build_accounting.json`, in the verification provenance and here, and it is the one
environment value in this dataset that is not pinned by a quote.

### Verification

`verify_build()` runs `verify_environment_response_dataset_streaming` (the memory-bounded
single pass; the store is 141,542 records) with the MG1655 resolver and the MG1655
GenBank locus set as the L4 universe, then appends three SUPPLEMENTARY rows computed
from a second streaming pass through `summarize_store`:

- L1 `stored_targets_are_loci_of_the_pinned_assembly`: a pseudogene locus resolves to
  itself as `non_gene_feature`, which the shared `canonical_gene_names` rule (which wants
  `current`) will not accept, so this row accepts a gene or a pseudogene locus.
- L2 `guide_spacers_are_twenty_nt_acgt`: the spacer is the only thing on the record that
  carries the target, so a malformed one is a record that cannot be re-mapped.
- L3 `both_screens_are_balanced_and_strain_pinned`: no drop rule fires per screen, so the
  two `screen_id` groups must be the same size and each must pin MG1655's GenBank
  assembly with its own strain background.

The built-in L1 `pair_uniqueness` row passes without a custom replacement, and that is a
design consequence worth naming: `_genotype_signature` folds `crispr.guide_sequence` into
the strain key, so the ~19 guides of one gene are 19 strains rather than 19 duplicates,
and `_study_key` folds `screen_id` in, so the LC-E18 and LC-E75 measurements of one guide
are two contexts rather than a duplicate pair.

**Result on the built store: `CrispriKnockdownCui2018Dataset: PASS`**, all 17 rows, in
800 s wall. The rows worth quoting:

| row | message |
|---|---|
| L0 `structural` | 141542 records validated |
| L1 `count` | observed 141542, expected 141542 |
| L1 `pair_uniqueness` | 141542 unique (study, strain, condition) records, one each |
| L1 `canonical_gene_names` | 4263 systematic names, one canonical spelling each, each current in the genome; 48 are pseudogene loci the genome resolves to themselves |
| L1 `provenance_gaps` | 424626 documented provenance gaps over 141542/141542 records; 0 deferred fields |
| L2 `value_fidelity` | 141542 values checked |
| L2 `uncertainty_sanity` | 0 labeled uncertainties, none a zero dispersion; 141542 records report n_samples >= 2 with no uncertainty |
| L3 `measurement_type_consistent` | single measurement_type: `log2_ratio` |
| L3 `reference_zero` | reference response == 0 for all 141542 records |
| L3 `environment_perturbed` | all 141542 experiments carry an environmental edit |
| L3 `media_membership` | 141542 records on a shared MEDIA_LIBRARY medium (1 distinct medium) |
| L4 `gene_containment_sgd` | 1.000 of 4263 measured genes are reference genes (>= 0.9) |
| L4 `current_genome_genes` | every one of the 4263 measured systematic names is a gene of the current genome |
| L1 `stored_targets_are_loci_of_the_pinned_assembly` | SUPPLEMENTARY: 4263 stored knockdown targets; 0 do not resolve to themselves |
| L2 `guide_spacers_are_twenty_nt_acgt` | SUPPLEMENTARY: 70771 distinct spacers; 0 malformed |
| L3 `both_screens_are_balanced_and_strain_pinned` | SUPPLEMENTARY: records per screen `{'LC-E18': 70771, 'LC-E75': 70771}`; 2 distinct (screen, assembly, background) pins |

`L1 canonical_gene_names` counting 48 pseudogene loci is why the supplementary
`stored_targets_are_loci_of_the_pinned_assembly` row exists: the shared rule wants
status `current`, which a pseudogene locus never has. The `gene_containment_sgd` row's
name is the shared rule's label; the universe passed to it is MG1655's own GenBank locus
set, not S288C.
