---
id: pfvqajukbrziyi2d7gftbsl
title: Menasalvas2025
desc: ''
updated: 1791391972701
created: 1791391972701
---

## 2026.10.07 - The second P. putida loader: a biosensor-coupled CRISPRi selection, and the titer that is not released

Menasalvas et al. 2025 (*Sci Adv* 11, eady2677; doi:10.1126/sciadv.ady2677; PMID
41134890) is row 20 of the bacterial expansion list, classed there as a production
campaign. It is one, and its engineering arm reaches ~900 mg/liter of isoprenol in a
*P. putida* KT2440 chassis. The loader nonetheless stores no titer, because the paper
releases none per strain. What it does release per gene is the output of a
growth-coupled biosensor selection, and that is what landed.

### What each record is

| class | family | records | references | gene set |
|---|---|---|---|---|
| `IsoprenolSelectionMenasalvas2025Dataset` | `BacterialEnvironmentResponseExperiment` | 58 | 2 (one per selection round) | 67 |

One record is one target the authors selected from a round of the pooled dCpf1 CRISPRi
selection: 28 from Supplementary Table 1 (the lower-threshold first round) and 30 from
Supplementary Table 2 (the higher-threshold second round). The gene set is the 60
distinct knockdown targets plus the 7 constant host-background identifiers.

Module: `torchcell/datasets/pputida/menasalvas2025.py`. Tests:
`tests/torchcell/datasets/pputida/test_menasalvas2025.py`. The closest template is
[[torchcell.datasets.pputida.carruthers2025]], whose chassis-as-background,
pathway-as-perturbation and raw-mirror pattern this module copies.

### The titer is figure-only, and that is the finding

The paper's isoprenol titers appear in Figs. 4D, 4G, 6A and 6B and in figs. S13, S14,
S19 and S20, all as plotted points. They appear in no table and in no data file.
Measured on the deposit, Dryad `10.5061/dryad.sbcc2frjq` carries Supplementary Data 1
(metabolite concentrations, the designed gRNA library, its read distribution, the
lost-guide list, the WGS polymorphisms, a ShinyGO enrichment) and Supplementary Data 2
(five proteomics sheets); Zenodo `10.5281/zenodo.17155686` carries AlphaFold3 output;
PRIDE `PXD061547` carries 97 raw proteomics files; BioProject `PRJNA1226229` carries
three resequencing runs. None is a titer table.

The only per-strain titers in the mirrored bytes are prose, and all but one are
approximations or ranges spanning several strains:

| statement (verbatim from `paper.md`) | strain | why it is not a datum |
|---|---|---|
| "TEAM-2595 grown in 5-ml culture tubes produced up to $2 0 0 ~ \mathrm { m g / }$ liter of isoprenol" | TEAM-2595 | "up to" is a bound, not a value |
| "the same strain grown in the deep-well plate format only produced $\mathrm { \sim 2 5 ~ m g }$ /liter" | TEAM-2595 | approximate |
| "this strain increased titers to $2 2 0 \mathrm { m g / }$ liter in the deep-well plate format" | TEAM-2777 | a clean number, but with no stated time point and n = 1 strain |
| "increased the isoprenol titer to $\sim 5 0 0 \mathrm { m g / }$ liter in the stacked deletion strain" | TEAM-2914 | approximate |
| "we identified four different deletion strains that increased titers from 150 to $2 5 0 \mathrm { m g }$ /liter" | four, unnamed | a range over unnamed strains |
| "observed titers now exceeded 850 to $9 0 0 ~ \mathrm { m g } ,$ 'liter ... in isolates TEAM-3185 and TEAM-3174 sampled in two production phase time points" | two strains, two time points | a range over four (strain, time) cells |

Reading a per-strain number off those sentences would be fabrication, so no
`ProductTiterPhenotype` is written. The gap is carried in the raw mirror's
`si_expected` (two all-caps entries), in the module docstring and in
`preprocess/build_accounting.json`.

### The phenotype class, and why it is not a titer class

The released per-gene readout is a CALL: the gene's guide was enriched in a pooled
selection where growth is coupled to isoprenol through the `PpedF`-`pyrF` biosensor.
Growth, not product, is what was measured, and no enrichment score was released. So the
record is an `EnvironmentResponsePhenotype` with:

| field | value | why |
|---|---|---|
| `measurement_type` | `categorical` | the readout is a call, not a quantity |
| `assay_type` | `biosensor_readout` | the typed method axis already has the member |
| `category` | `enhanced` | its definition is "measurably better than the reference ... a biosensor signal above control" |
| `category_label` | `enriched` | the source's own word, kept verbatim |
| `environment_response` | `None` + `ProvenanceGap` | no per-guide read count was released |
| `environment_response_uncertainty` / `_type` | `None` + `ProvenanceGap` | no dispersion was released for this readout |
| `n_samples` | 4 | "with four replicates from each conjugation" |
| `sample_unit` | `biological_replicate` | the criterion's own words, below |
| `screen_id` | `round-1-lower-isoprenol-threshold`, `round-2-higher-isoprenol-threshold` | the two rounds are independently thresholded screens |
| `units` | the call's definition in words | so a reader of the record knows what "enhanced" meant here |

This is the one-sided hit-list shape `ResponseCategory` already serves for Auesukaree
2009's listed stress-sensitive mutants, used with the opposite sign. The reference
record is the control arm of the same selection, "M9 medium kanamycin with or without
$1 \mu \mathrm { M }$ crystal violet": the inducer omitted, so no clone can be enriched
by isoprenol. Its category is the baseline `no_change`, which no record reports, so the
L3 categorical-reference rule runs rather than passing vacuously.

Two phenotype classes were considered and rejected. `ProductTiterPhenotype` has no
value to carry (above). `VisualScorePhenotype` is an ordinal visual-inspection score on
a declared scale, and the selection scored nothing: a clone either grew enough to be
sequenced or it did not.

### Sourcing: the replicate count and the uncertainty type

`n_samples = 4`, from the selection Methods:

> used to inoculate $1 . 5 \mathrm { m l }$ of M9 medium kanamycin with or without $1 \mu \mathrm { M }$ crystal violet in 24-deep-well plates with four replicates from each conjugation

`sample_unit = biological_replicate`, because the enrichment criterion in the same
section calls one replicate exactly that:

> implicat genes were elected on the basis o thefollowing criteri: ( particular gRNA was enriched ${ > } 5$ reads in one biological replicate), (ii) if there are multiple gRNAs targeting the same gene, iii) gRNAs target genes functionally related (i.e., generation of a specific process) or targets in the same operon, and (iv) the repeated occurrence of gRNAs or gene targets across multiple replicates

(The OCR of that sentence is lossy; it is kept verbatim, as the rules require.)

**Uncertainty type: none exists for this readout.** The selection releases a hit list
and no dispersion, so `environment_response_uncertainty` and its type are typed
`ProvenanceGap`s rather than a guessed SD. The paper's only stated uncertainty type
belongs to the TITER readout, which is not released per strain:

> the error bars indicating the SD from the mean for isoprenol titer reflect are calculated using all data points shown in the figure panel

so had a titer loader been possible, its `UncertaintyType` would have been
`sample_sd`. That is recorded in the module as `TITER_UNCERTAINTY` so the next revision
does not have to re-derive it, and `TITER_METHOD` records the instrument (`GC-FID`).

### Sourcing: the parent strain and the chassis genotype

The parent strain is *P. putida* KT2440, pinned to `pputida_KT2440_ASM756v2` /
`GCA_000007565.2`. **`AssemblyReferenceGenome.background` is `None`**, and that is the
honest answer rather than a shortcut, because the paper never names the selection host:

> The pooled CRISPRi-∆pyrF selection regime was applied in two sequential rounds using four $P _ { P } { _ { e d F } }$ -RBSpyrF variants and four different strains, each with varied base isoprenol titers and isoprenol activation thresholds (Materials and Methods)

Four unnamed producer strains with four reporter variants is not a genotype, and none of
them is in Supplementary Table 4, which lists `TEAM-862` (a non-producing ΔpyrF strain)
and the whole producer lineage but no producing ΔpyrF strain. So every edit the paper
DOES state for every selection host is a perturbation in `Genotype` instead:

| perturbation | leaf | identifier | source |
|---|---|---|---|
| native `pyrF` deletion | `BacterialDeletionPerturbation` | `PP_1815` | Table S4, "Pp KT2440 ΔPP_1815/pyrF" |
| `mvaS` | `HeterologousPathwayPerturbation` | `mvaS` | Table S5, pTE744 |
| `mvaE` | `HeterologousPathwayPerturbation` | `mvaE` | Table S5, pTE744 |
| `MKmm` | `HeterologousPathwayPerturbation` | `MKmm` | Table S5, pTE745 |
| `PMDHKQ` | `HeterologousPathwayPerturbation` | `PMDHKQ` | Table S5, pTE745 |
| `aphA` | `HeterologousPathwayPerturbation` | `aphA` | Table S5, pTE745 |
| the growth-coupled reporter | `HeterologousPathwayPerturbation` | `TERTU_1389` | Supplementary Methods |
| the selected knockdown(s) | `BacterialCrisprInterferencePerturbation` | one or three `PP_` tags | Tables S1 and S2 |

`PP_1815` is confirmed by the annotation itself: `GCA_000007565.2` names it `pyrF`, so
the strain table's own locus and the assembly agree.

The reporter is a heterologous addition, not the native gene restored:

> The open reading frame for pyrF homolog purF (orotidine 5'phosphate decarboxylase, TERTU_1389, referred to simply as pyrF) and 300 bp downstream sequence from Teredinibacter turnerae T7901 was synthesized by Genewiz Ltd and assembled into a RSF1010 plasmid backbone immediately downstream of the pedF promoter sequence for cross-species pyrF complementation

**Chassis edits the records deliberately do NOT carry**, because they distinguish
TEAM-2595 from TEAM-2777 and the paper does not say which of the four hosts each round
used: `PJ23100-PP_2666,PP_2665`, `PP_5402intergenic::PpedF-RBS-mCherry`,
`PP_3159intergenic::PpedF-RBS-mCherry`, `ΔPP_2664`, `ΔPP_2675` and
`PJ23119-PP_1697`. These are missing ROWS, not missing fields, so they are not a
`ProvenanceGap` (a gap must name a field that is `None`); they are listed in the module
docstring and in `preprocess/build_accounting.json`.

### Which perturbation leaf each engineered change maps to

The campaign uses all four modes the bacterial ontology types, and Supplementary Table
4 states them as one string per strain. The mapping is settled once, in the module
docstring, so a later revision of this paper's deletion panel does not have to re-decide
it:

| change in the strain table | leaf | example |
|---|---|---|
| a knockout | `BacterialDeletionPerturbation` | `ΔPP_2675`, `ΔPP_2664`, `ΔPP_1815/pyrF`, the 15 validation deletions |
| a knockdown guide | `BacterialCrisprInterferencePerturbation` | a dCpf1 target of the pooled library |
| a promoter change | `PromoterReplacementPerturbation` | `PJ23100-PP_2666,PP_2665`, `PJ23119-PP_1697` |
| an integrated pathway | `HeterologousPathwayPerturbation` | `PP_5322intergenic::Pcv-mvaS,mvaE`, `PP_0871intergenic::Ptrc-MKmm,PMDHKQ,aphA` |

Only the first, second and fourth appear in this dataset's records.

### The pathway is the pIY670 cassette, integrated rather than plasmid-borne

Supplementary Table 5 states pIY670 as "araC-pBAD-mvaS,mvaE ptrc-MKmm,PMDHKQ,aphARK2
kanR" and the two integration vectors as "PP_5322intergenic::Pcv-mvaS,mvaE kanR
sacB(integration allelic exchange vector)" and "PP_0871intergenic::Ptrc-MKmm,PMDHKQ,aphA
gntRsacB (integration allelic exchange vector)". That is the same five-gene IPP-bypass
cassette [[torchcell.datasets.pputida.carruthers2025]] records as
`MvaSEf`/`MvaEEf`/`MKMm`/`PMDScHKQ`/`AphA`, so the two P. putida isoprenol papers share
a pathway and differ in where it sits: `localization="chromosomal_integration"` here
against `episomal_plasmid` there. The identifiers stored are Menasalvas' own tokens, and
the build refuses any token that is not a substring of the quoted plasmid description.

`source_organism` is `"unreported"` for all five. Menasalvas defers the pathway's origin
to reference 22 (Banerjee et al., isoprenol in *P. putida*) and reference 24 (Kang et
al., the IPP-bypass pathway in *E. coli*), neither mirrored, and the three `mvaS`
HOMOLOGS whose organisms it does name (*Enterococcus faecalis*, *Silicibacter
pomeroyi*, *Staphylococcus aureus*) are EXTRA copies in the final producer strains, not
the integrated pathway's own `mvaS`. Reading "Mm" off `MKmm` is exactly the suffix
inference Carruthers refused without independent evidence.

`GeneAdditionPerturbation.source_organism` is a required `str` on a leaf with no
`provenance_gaps` field, so the absence cannot be typed. Making it nullable with the gap
mixin is raised in the PR, not taken here.

### Environment

| field | value | source |
|---|---|---|
| `media` | `MEDIA_LIBRARY["M9_NREL_MOPS_MENASALVAS2025"]` | the Methods' NREL M9, already in [[torchcell.datamodels.media]] |
| `temperature` | `None` + `ProvenanceGap` | see below |
| `duration_hours` | 24.0 | "Samples were grown for 24 hours at which point we examined the cultures for growth" |
| `aerobicity` | `aerobic` | 24-deep-well plate culture |
| kanamycin | 50 µg/mL `SmallMoleculePerturbation` | the Supplementary Methods' plasmid-selection dose |
| pH | 7.0 as `EnvironmentPhysicalPerturbation(factor=ph)` | pH is not a `Media` field, per the media note |
| crystal violet | 1 µM, on the RECORD arm only | the integrated pathway's inducer; omitted from the reference |

**Temperature is a typed absence, not 30 C.** The paper states 30 C for the conjugation
spot on LB agar ("spotted onto solid LB agar media and allowed to incubate overnight at
$3 0 ^ { \circ } \mathrm { C }$"), for petri-dish culture and for the production assays,
and does not state it for the 24-deep-well M9 selection plate. The value 30.0 is kept
in the module as `CONJUGATION_TEMPERATURE` with the reason, so a future revision can see
what was looked at.

The Teknova T1001 trace-metal amount remains an open gap on the medium itself, recorded
in [[torchcell.datamodels.media]], not re-opened here.

### Identifier reconciliation

60 distinct selected targets, reconciled against `pputida_KT2440_ASM756v2`:

| statuses | count |
|---|---|
| `current` | 60 |
| `renamed` | 0 |
| `non_gene_feature` | 0 |
| `retired` | 0 |
| `ambiguous` | 0 |

Layers: locus tag 60, old locus tag 0, RefSeq locus tag 0, gene symbol 0, gene synonym
0, not found 0. Remapped 0, kept on collision 0, outside namespace 0. Resolved fraction
1.000, and `MIN_RESOLVED_FRACTION = 1.0`, so any movement in the annotation or in the
released tables stops the build rather than dropping a record.

**`perturbed_gene_name` comes from the annotation, not from the table.** The tables
spell a gene several ways ("sotB / PP_2428", "cmpX PP_2087", "hisQ |PP_4485", "relA
PP_1656"), and one spelling disagrees with the assembly: the source's "phaAZC-II"
implies `PP_5004` is `phaZ` while `GCA_000007565.2` annotates it `phaB`. The table's
verbatim `Gene`, `Function` and `FunctionalCategory` cells are kept in
`preprocess/selected_targets.csv` beside the record, so nothing the source said is lost.

**One row is an operon.** Supplementary Table 1's "phaAZC-II / PP_5003-PP_5005" is one
guide repressing a polycistron, expanded by integer enumeration of the stated endpoints
to `PP_5003`, `PP_5004`, `PP_5005` and then checked against the annotation. That record
carries three knockdown perturbations, which is why 58 records cover 60 targets.

Functional categories of the 58 records, verbatim from the tables: signaltransduction
15, enzyme 14, efflux pump 13, unknown 9, carbon storage 4, DNA repair 3.

### The released tables are a picked subset, by the authors' own account

> Candidate genes from the gRNA enrichment were first grouped by function using HMMer and COG to identify nonredundant cellular processes. t random, we picked several from ach category to design new gRNA plasmids and recombineering oligos, choosing 28 targets for the first enrichment analysis and 30 for the second analysis

Enrichment is the measurement; the picking is curation. Every record is a gene whose
guide met the stated criterion, and no record claims the list is exhaustive. The pooled
library is ~16,500 guides over 5,591 coding sequences and the deposit holds the designed
sequences, the baseline read distribution and the missing-variant list but no
guide-by-sample abundance, so the unpicked enriched guides are unrecoverable. The build
asserts the two rounds share no target, which the paper states ("Analysis of gRNAs by
sequencing showed no overlap with the first set, as expected").

Note the paper's own cross-reference is off by one: the Methods say "All selected targets
from both rounds are described in tables S2 and S3", while the tables carrying them are
S1 and S2. The loader reads S1 and S2, whose titles name the two thresholds.

### Drops

None. Every row of both tables becomes a record; `preprocess/build_accounting.json`
checks the arithmetic and refuses a drop, because this dataset declares no drop rule.

### Raw mirror

`$DATA_ROOT/torchcell-raw/menasalvasBiosensordrivenStrainEngineering2025/`

| path | role | bytes | sha256 |
|---|---|---|---|
| `si/sciadv.ady2677_sm.pdf` | `si_pdf` | 7,630,709 | `6d82d568b307655878306c236c3d6c04d4776ef73d9b8551d1b8181f2f0be02a` |
| `si/si1.md` | `si_ocr` | 150,870 | `2afa42609d20500f80d5edbddf0bb41b88b4e7f1ea0abefb40fa4b969e4e5e8e` |

The PDF carries a `RetrievalRecord` with
`method=pmc_cloud`, `retriever=torchcell.literature.retrieve.pmc_cloud_object` and
`params={"key": "PMC12551699.1/sciadv.ady2677_sm.pdf"}`. Re-running that retrieval on
2026-10-07 returned 7,630,709 bytes whose sha256 matched the pin, so the recorded
retrieval reproduces as written. The markdown is a DERIVED artifact of exactly those
bytes, so it carries a `ProcessingRecord` (`mineru` 2.7.6, pipeline backend, DPI 200,
`input_sha256=[<the PDF's digest>]`) instead of a retrieval. The loader parses the
markdown, which is why the OCR is pinned in its own right rather than only quoted.

Dryad is not deposited and cannot be scripted: `datadryad.org` serves an Anubis
JavaScript proof-of-work challenge, measured 2026-10-07 (`/downloads/file_stream/<id>`
returns the challenge page with HTTP 200, `/api/v2/files/<id>/download` returns HTTP 401
"must have current bearer token"), the same finding
[[torchcell.datasets.pputida.carruthers2025]] recorded for its own Dryad deposit. The
manual recipe is in `si_expected`. Nothing a loader reads lives there.

### Verification, L0 to L4

Run from the module's own runner,
`python -m torchcell.datasets.pputida.menasalvas2025`, which builds, prints the
accounting and then calls `verify_build`. The constant host background is passed as
`background_genes` so the L1 strain key and the L4 gene universe see only the SCREENED
knockdown targets: the five pathway tokens and `TERTU_1389` are heterologous identifiers
with no locus in any assembly, and `PP_1815` is the same lesion in all 58 records.

```
IsoprenolSelectionMenasalvas2025Dataset: PASS
  [PASS] L0 structural: 58 records validated
  [PASS] L1 count: observed 58, expected 58
  [PASS] L1 pair_uniqueness: 58 unique (study, strain, condition) records, one each
  [PASS] L1 provenance_gaps: 290 documented provenance gaps over 58/58 records; 1 deferred field(s): ['inchikey']
  [PASS] L1 canonical_gene_names: 60 systematic names, one canonical spelling each, each current in the genome
  [PASS] L1 stored_targets_are_loci_of_the_pinned_assembly: SUPPLEMENTARY: 60 stored knockdown targets; 0 do not resolve to themselves
  [PASS] L2 value_fidelity: 0 values checked
  [PASS] L2 se_nonnegative: 0 values checked
  [PASS] L2 uncertainty_sanity: 0 labeled uncertainties, none a zero dispersion; 58 records report n_samples >= 2 with no uncertainty
  [PASS] L3 measurement_type_consistent: single measurement_type: 'categorical'
  [PASS] L3 reference_zero: categorical rule: all 58 references carry the baseline category 'no_change', which no experiment record reports
  [PASS] L3 environment_perturbed: all 58 experiments carry an environmental edit
  [PASS] L3 compound_identity: environment edits: 58 compound references carry a structure identifier; 58 declare a typed gap (1 distinct compounds, unencodable)
  [PASS] L3 media_compound_identity: medium components: 464 compound references carry a structure identifier; 0 declare a typed gap
  [PASS] L3 media_membership: 58 records on a shared MEDIA_LIBRARY medium, 0 on a medium deriving from one (1 distinct media)
  [PASS] L3 assembly_pin_is_kt2440_genbank_with_no_asserted_background: SUPPLEMENTARY: 1 distinct assembly pin(s): [('pputida_KT2440_ASM756v2', 'GCA_000007565.2', 'None')]
  [PASS] L3 every_genotype_carries_the_stated_host_background: SUPPLEMENTARY: 58 genotypes checked against ['MKmm', 'PMDHKQ', 'PP_1815', 'TERTU_1389', 'aphA', 'mvaE', 'mvaS']; 0 incomplete
  [PASS] L4 gene_containment_sgd: 1.000 of 60 measured genes are S288C reference genes (>= 0.9)
  [PASS] L4 current_genome_genes: every one of the 60 measured systematic names is a gene of the current genome
```

The L2 rows read "0 values checked" because this readout has no number at all, which is
the honest state and not a vacuous pass: the L3 categorical-reference rule is what
checks the call, and it names the rule it ran. The L4 row keeps the verifier's own
`gene_containment_sgd` slot name while the universe it checked against is KT2440's
5,786 GenBank loci, passed in that slot the way
[[torchcell.datasets.ecoli.tong2020]] does. `run_environment_response` in
`torchcell/verification/runners.py` is wired to the yeast genome and the yeast gene set,
so a bacterial environment-response dataset cannot be registered there yet; that is
raised in the PR.

### Build

```bash
python -m torchcell.database.build_dataset_lmdb --dataset IsoprenolSelectionMenasalvas2025Dataset
```

58 records, 2 references, gene set 67, built in 1 s. It wrote
`preprocess/build_manifest.json` with a 49-symbol closure, and the dataset reads fresh
under `python -m torchcell.provenance.build_manifest` (it is not among the 12 stale
datasets that run reports).

### Open items for the owner

1. **Isoprenol has no compound-identity row.** `resolved_compound("isoprenol")` is not
   called by this loader (there is no product to name), so the gap does not surface in
   these records, but it still blocks a future titer loader for this paper. The InChIKey
   is `CPJRRXSHAYUTGL-UHFFFAOYSA-N`, recorded in the module for the curation step;
   adding the row is a human PubChem act against
   `torchcell/datamodels/compound_identity_inputs/` plus a re-pin of `_TABLE_SHA256`.
2. **`ConcentrationUnit` has no `mg_per_l`.** Not hit here, but every released isoprenol
   number in this literature is mg/liter, and the current workaround is to store it as
   the numerically identical `ug_per_ml`.
3. **`ProductTiterExperiment.environment` is not narrowed** to `CultureEnvironment`, so
   a vessel, working volume and shaking rate are dumped away by declared-type
   serialization.
4. **`GeneAdditionPerturbation.source_organism` cannot be gapped** (required `str`, no
   gap mixin on the leaf), which is why this loader stores the sentinel `"unreported"`.
5. **`run_environment_response` is yeast-wired**, so this dataset verifies from its own
   module rather than from the family runner.
6. **The deletion-validation panel of Supplementary Table 4** (15 isogenic deletion
   strains in TEAM-2777 and the stacked combinations) is a real genotype set with no
   released phenotype. If the authors ever release the figure source data, those strains
   plus the titers become a second loader; until then they are not records.

## 2026.10.09 - The #731 leaves do not unblock this row, and the measurement that says so

Issue #731 landed `BacterialSequenceVariantPerturbation`,
`BacterialSiteVariantPerturbation` and `BacterialSpanDeletionPerturbation`, and this
loader was on the list of rows they were expected to extend. It does not extend, and the
reason is the release rather than the schema: **the per-variant table is not in the
mirror and cannot be scripted into it.**

What this paper DOES state about its evolved producers, verbatim from
`paper.md` (sha256 `d14536948af5ba67d52362ae71fb804a2fac03a1f7ab152817a01df3aa92d080`):

> "WGS identified 74 additional nonsynonymous single-nucleotide polymorphisms (SNPs) of
> unknown function in both improved producer strains, and a 26.1-kb deletion surrounding
> the fleQ/PP_4373 locus containing 24 flagellarelated genes. A comprehensive tabulation
> of all polymorphisms is described in data S1-5."

And Supplementary Table 3's footnote, verbatim from `si/si1.md` (sha256
`2afa42609d20500f80d5edbddf0bb41b88b4e7f1ea0abefb40fa4b969e4e5e8e`):

> "^candidate identified from gRNA selection. \*candidate identified from literature. #
> Spontaneous polymorphisms characterized by WGS are described in Supplementary Data
> 1-5."

So the aggregate is sourceable (74 nonsynonymous SNPs, one 26.1 kb deletion at
`fleQ`/`PP_4373` covering 24 flagellar genes) and **not one per-SNP position, reference
base, alternate base or frequency is**. Every leaf field a call needs is in Dryad
`data S1-5`, which this loader's `manifest.json` already declares NOT deposited with a
manual recipe, because `datadryad.org` serves an Anubis JavaScript proof-of-work
challenge (measured 2026-10-07: `/downloads/file_stream/<id>` returns the challenge page
with HTTP 200 and `/api/v2/files/<id>/download` returns HTTP 401). The resequencing reads
are BioProject `PRJNA1226229`, and no loader consumes reads.

Two further blockages that survive the leaves even if the Dryad bytes were deposited:

- **No per-strain phenotype exists for the resequenced isolates.** The GC-FID isoprenol
  titers are figure-only; the accounting already records that "this paper releases no
  per-strain isoprenol titer anywhere, only plotted figure panels". A genotype with no
  released measurement is not a record.
- **A strain-name disagreement in the data-availability statement.** It names BioSample
  `SAMN46924003` as strain `TEAM-3175`, and `TEAM-3175` occurs twice in `paper.md` (both
  in that one sentence) and zero times in the SI, while `TEAM-3174` occurs 12 times in
  `paper.md`, 8 times in the SI, and is the strain name in Supplementary Table 4.
  *Hypothesis (untested):* the data-availability `TEAM-3175` is a typo for `TEAM-3174`.
  This needs the BioSample record itself and must not be resolved by preference.

The record count is therefore unchanged at 58, and the correct next step for this row is
the manual-browser Dryad retrieval already written into `si_expected`, not a schema
change.

**One gap of this row that #731 did NOT close.** `GeneAdditionPerturbation.source_organism`
is a required `str`, so the five integrated pathway genes whose source organism this
paper never states carry the sentinel `"unreported"` (`SOURCE_ORGANISM_UNREPORTED`). A
sentinel standing in for an unknown is what the strain-background contract forbids
elsewhere. The fix is the same shape as the variant leaves and is additive: make
`source_organism` optional so a `ProvenanceGap` can cover it, then backfill those five
records. It is recorded on #731 and left open there deliberately, because it is an
addition-axis gap rather than a variant-representation one.
