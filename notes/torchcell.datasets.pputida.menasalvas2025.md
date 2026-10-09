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

## 2026.10.09 - The Dryad deposit is mirrored and three of its arms are loaded

Issue #788 item 1. The deposit the 2026.10.07 row above declared NOT deposited was
retrieved by hand and is now in the raw mirror, and three new datasets read it. The
2026.10.07 conclusion that no per-SNP leaf field was reachable is **corrected**: every
leaf field is in `data S1-5` and `data S1-5` is now mirrored. The conclusion that no
per-strain titer exists anywhere still stands.

### What was deposited, and the recipe that reproduces it

`/scratch/projects/torchcell-scratch/torchcell-raw/menasalvasBiosensordrivenStrainEngineering2025/data/dryad/`

| file | bytes | sha256 |
|---|---|---|
| `doi_10_5061_dryad_sbcc2frjq__v20250919.zip` (the arrival bytes) | 78,995,492 | `67ae73c6f20058513daee837389aace1fc1fd5973840de0d0d886e9da6e4c19c` |
| `Data_Dryad_Supplementary_Data_Updated_2025-9-18_2.zip` (member) | 78,976,184 | `ca5c9a1b7d5aca6851df886b6c5b27c56884e7fde68f4f63880363e023038463` |
| `README.md` (member) | 18,978 | `07f1e8ed9981276341a67a7843eff4c0d408c1c4ee976e5fd08c1d214ebb9271` |

`SHA256SUMS.txt` pins **three** files, not four: this deposit's arrival zip holds two
members where Carruthers 2025's held three. Each of the three is an
`ArtifactRecord` with `RetrievalMethod.manual_browser` whose `retrieval.sha256` equals
the record's own, carrying the recipe from `DEPOSIT.md` verbatim as its
`retrieval_command` (`DRYAD_MANUAL_RECIPE`):

```text
"open https://doi.org/10.5061/dryad.sbcc2frjq in a browser, solve the challenge,
click "Download dataset", save the zip unchanged, then unzip it into this directory
beside the zip."
```

plus `retrieved_by` ("the owner (mjvolk3), browser download"), `deposit_note`,
`deposit_record` (`data/dryad/DEPOSIT.md`) and `checksums`
(`data/dryad/SHA256SUMS.txt`). The two now-false `si_expected` entries were rewritten:
the Dryad entry reads DEPOSITED and names the three files and the junk a loader skips,
and the titer entry keeps the titer claim (still true) while naming the arms that are
now loaded.

### What the inner zip actually holds

170 members. Measured 2026-10-09 with `unzip -l`: **three** are data this module can
read, and the rest are `__MACOSX/` resource forks, `.DS_Store` files, a
`~$Menasalvas et al Supplementary Data 1.xlsx` Excel lock file, 150-odd `.fcs` flow
files, `.fastq` reads, three AlphaFold `.mp4` movies and Supplementary Data 4's
`fast.genomics` TSV.

| member | bytes | read by |
|---|---|---|
| `.../Menasalvas et al Supplementary Data 1.xlsx` | 1,747,748 | the two metabolite datasets (sheet 1) and all three (sheet 5) |
| `.../Menasalvas et al Supplementary Data 2.xlsx` | 7,618,312 | the proteome dataset (sheet 4) |
| `.../Menasalvas et al Supplementary Data 3-5/Menasalvas et al Supplementary Data 4.tsv` | 1,518,084 | nothing |

`extract_dryad_members` reads the two workbooks by their exact zip paths and writes
them under basenames the module chose, into `<dataset root>/preprocess/dryad/`. Nothing
calls `extractall`, no released member name reaches the filesystem, and the extraction
is idempotent by sha256.

### The sourced genotypes

`paper.md` (sha256 `d14536948af5ba67d52362ae71fb804a2fac03a1f7ab152817a01df3aa92d080`),
Results, opening paragraph of "Functional genomics analyses reveal metabolic shifts in
high isoprenol producers". The MinerU OCR breaks this sentence across a blank line after
"PP_3540/mvaB, and"; the quote joins the two fragments with one space and changes
nothing else:

> "Our two highest producer strains TEAM-3185 and TEAM-3174 contain gene deletions in
> PP_2428, PP_4622, PP_3540/mvaB, and PP_4373/fleQ. TEAM-3174 also overexpresses mvaS
> and includes ΔPP_2710. TEAM-3185 lacks the mvaS overexpression, and ΔPP_2710 but
> PP_2074 is deleted."

Supplementary Table 4 (`si/si1.md`, sha256
`2afa42609d20500f80d5edbddf0bb41b88b4e7f1ea0abefb40fa4b969e4e5e8e`) states the same
sets independently and adds what the Results do not: TEAM-3185 is
`"Pp TEAM-2777 ΔPP_ 2428 ΔPP_ 4622 ΔPP_ 3540 ΔPP_4373ΔPP_2074"`, so both improved strains
inherit TEAM-2777's ΔPP_2664 and ΔPP_2675, and TEAM-3174's two mvaS copies sit at
`PP_1117intergenic` and `PP_5464intergenic` under `Pcv`. The organism of those copies is
sourced from the Results ("identified overexpression of Enterococcus faecalis mvaS as
the most successful"); the INTEGRATED pathway's own mvaS keeps
`SOURCE_ORGANISM_UNREPORTED`, because the paper defers that pathway's origin to two
unmirrored references. Table 4's TEAM-3174 row is OCR-merged with its rowspan
neighbors, which is recorded in `TABLE4_TEAM3174.note` rather than silently cleaned.

The metabolomics replicate count came out of the deposit README, line 25, under
"Sheet 1. Metabolite concentrations from selected isoprenol producer strains.":

> "Average value from 3 biological replicates. Strain names are indicated with the
> TEAM-XXXX format. GP = growth phase samples. PP = production phase samples. Fold
> Change was calculated by the determining the ratio of concentrations from the
> indicated strain IDs in during growth phase (GP)"

The article states it too, in the Fig. 7E caption ("Mean values from three biological
replicates for each sample are reported."), but the MinerU OCR of the article DROPPED
that sentence: `grep -cE "three biological|Mean values from"` on `paper.md` returns 0
while the PDF text layer returns 2. Fig. 7's caption carries TWO different replicate
counts, `n = 4` for the panel-A growth curve and three for the panel-E metabolomics, so
taking the wrong one was a live risk. The same README states the replicon
(`Genomic Coordinates in P. putida AE015451`) and the caller
(`breseq v 0.38.1`).

### Arms built, with their measured record counts

| dataset | dev-tree root | records | per record |
|---|---|---|---|
| `ProteomeMenasalvas2025Dataset` | `data/torchcell/proteome_menasalvas2025` | 6 | 2,090 loci, 3 replicates |
| `MetaboliteGrowthPhaseMenasalvas2025Dataset` | `data/torchcell/metabolite_growth_menasalvas2025` | 3 | 46 metabolites, n = 3 |
| `MetaboliteProductionPhaseMenasalvas2025Dataset` | `data/torchcell/metabolite_production_menasalvas2025` | 3 | 46, 46 and 47 metabolites, n = 3 |

The proteome records are `BacterialProteinAbundanceExperiment` and the metabolite ones
`BacterialMetaboliteExperiment`, not the yeast-shaped `MetaboliteExperiment`: only the
bacterial reference classes type `genome_reference` as `AssemblyReferenceGenome`, so the
yeast-shaped one DROPS the KT2440 assembly pin on dump and
`torchcell.verification.runners._dataset_assembly_sets` then refuses the record with "a
genome reference without assembly_set must be 'Saccharomyces cerevisiae'". Caught by
running the runner's own L4 containment against the first build.
| `IsoprenolSelectionMenasalvas2025Dataset` (rebuilt) | `data/torchcell/isoprenol_selection_menasalvas2025` | 58 | unchanged |

The proteome arm reads `Sheet 4. GrowthProduction phase`: 39,438 rows, six samples
`{2595,3174,3185}_{growth,production}` at 6,573 rows each, three replicates `R1 R2 R3`.
One record per sample, `measurement_type = "dia_protein_counts_sum_mean"` (the
per-sample mean of the released `Counts_sum`, with SE = SD / sqrt(3)), reference = the
matching-phase TEAM-2595 profile. The two TEAM-2595 records are therefore their own
reference, which is what keeping every released sample as a record costs; dropping two
real samples to avoid it would cost more, and it is also where the called variants land.

Growth phase and production phase are different environments, sourced from the Methods:

> "The log-phase samples were harvested when each strain reached an $\mathrm { O D } _
> { 6 0 0 }$ of 0.7 (roughly 8 to 14 hours postback dilution and induction) as monitored
> by a spectrophotometer. The production-phase samples were harvested at the 24-hour
> time point."

The production phase carries `duration_hours = 24.0`; the growth phase carries `None`
with a typed gap, because its "roughly 8 to 14 hours" is a 6 h range no single duration
holds and it differs per strain. `Environment.temperature` is a typed absence in both:
30 C is stated for the overnight LB culture, both M9 adaptation steps and the
conjugation spot, and is not restated for the production run.

Both metabolite datasets store the `Average Concentration (µM)` block, not the
`Specific Concentration (µM/OD600)` one: the README says the second is the first
"normalized against the OD~600~ at the time of sample harvest", so storing both would
store one measurement twice, and the harvest OD600 it divides by is not released per
sample. `measurement_type = "lc_ms_intracellular_concentration_uM"`, following
Mulleder 2016's `intracellular_concentration_mM`. `metabolite_level_se` is `None` with
a typed gap: the sheet releases 18 columns and none is an SD, SE, CV or n, there are no
hidden rows or columns and no cell comments.

### Drop rules, with counts

Proteome (`n_records = 0` for all three: no SAMPLE is dropped, only measurement keys):

| rule | items |
|---|---|
| `protein_key_is_not_a_host_protein` | 9: `Q9FD71` `Q9FD70` `Q8PW39` `P32377` `P0AE22` (the five mevalonate-pathway enzymes) and `P04264` `P13645` `P35527` `P00761` (three human keratins and pig trypsin) |
| `protein_key_merges_two_protein_groups` | 4: `Aroe` `Asd` `Dapa` `Dapf` |
| `protein_key_is_not_a_locus_of_the_pinned_assembly` | 86: 85 retired symbols plus the ambiguous `Asd` |

2,178 host keys resolve to 2,092 loci (96.05%), 2,090 of which survive the merged-key
rule. Metabolite (both phases): `relative_concentration_is_on_another_scale` drops the
10 `Relative` rows, quoting the sheet's own footnote row 66 (which the README repeats
verbatim); `metabolite_cell_is_blank_for_this_strain_and_phase` drops 6 cells in the
growth phase (`Glycolate` and `Pyruvate` in all three strains) and 5 in the production
phase. A released 0 is a present measurement and is stored.

### Arms and claims REFUSED, with the measurement that refused them

- **`TEAM-3175`'s 254 calls and `TEAM-3184`'s 255 calls.** Four distinct names for at
  most two improved strains: the Results and every data sheet say `TEAM-3174` /
  `TEAM-3185`, the shotgun-proteomics Methods say `TEAM-3175` / `TEAM-3184`, the
  data-availability statement says `TEAM-3175` / `TEAM-3185`, and `data S1-5` says
  `TEAM-2595` / `TEAM-3175` / `TEAM-3184`. No mirrored byte states any identity between
  them, so attaching those calls to the 3174 and 3185 records would be an unstated
  identity. `TEAM-2595` is spelled identically in `data S1-5` and in every phenotype
  sheet, so its 75 calls attach with no assumption: 26 in-locus
  `BacterialSequenceVariantPerturbation` and 49 intergenic
  `BacterialSiteVariantPerturbation`.
  *Hypothesis (untested, and deliberately NOT acted on):* `TEAM-3184` is `TEAM-3185`
  and `TEAM-3175` is `TEAM-3174`. The circumstantial evidence is one-sided: `TEAM-3184`
  carries a `Δ924 bp` deletion of the whole `PP_2074` CDS, which Table 4 states only
  for `TEAM-3185`, but `TEAM-3175` carries no `PP_2710` call, which Table 4 states only
  for `TEAM-3174`. Resolving it needs the BioSample records, not a preference.
- **`BacterialSpanDeletionPerturbation` for the 26.1 kb `fleQ` deletion.** The Results
  state "a 26.1-kb deletion surrounding the fleQ/PP_4373 locus containing 24
  flagellarelated genes. A comprehensive tabulation of all polymorphisms is described in
  data S1-5." Measured on `data S1-5`: **0** of 584 rows name `fleQ` or `PP_4373`, and
  the largest released deletion is `Δ1,867 bp` on `PP_2664`. The same sentence says "74
  additional" SNPs while the table holds 584 rows over 284 distinct positions for three
  clones, so the prose aggregate and the released table are not the same statement. The
  refusal of 2026.10.07 stands, now with the table in hand rather than inferred.
- **The two promoter replacements** `PJ23100-PP_2666,PP_2665` (all three strains) and
  `PJ23119-PP_1697` (the two improved strains, via TEAM-2777).
  `PromoterReplacementPerturbation.expression_direction` is required with no default and
  no mirrored byte states a direction for either swap; the Results say only "the best
  signal-to-noise isoprenol response was with the constitutive J23100 promoter for the
  PP_2665, PP_2666 operon". Recorded in `UNASSERTED_DESIGNED_CHASSIS`.
- **Supplementary Data 2 sheet 5** (`Sheet 5. Isoprenol pathway over`). Loadable with no
  schema change: 55,080 rows, six samples `TEAM_{2595,3174,3185}_{pIY670,pTE554}` at
  four replicates each over 2,290 proteins, with the environment stated by the
  Supplementary Figure 18 caption. Deferred as its own dataset rather than folded in,
  because it is the SAME three strains carrying an ADDITIONAL episomal copy of the five
  pathway genes the chromosome already holds, so every genotype would carry `mvaS`,
  `mvaE`, `MKmm`, `PMDHKQ` and `aphA` twice under two localizations, and the inducer
  that caption states ("2% of arabinose") names no w/v or v/v basis where Supplementary
  Figure 20's caption does state "0.1% w/v".
- **Supplementary Data 2 sheets 1-3** (the yiaY/yiaZ complementation, the PJ23119-yiaYZ
  isoprenol dose response and the culture-format comparison), **Supplementary Data 1
  sheets 2-4 and the ShinyGO sheet**, and **Supplementary Data 4**. No loader reads
  them; sheets 1-3 also carry non-*P. putida* entries and sheet 1 releases
  `Counts_mean` with no `Replicate` column at all.

### `data S1-5`, as measured

584 rows: `TEAM-3184` 255, `TEAM-3175` 254, `TEAM-2595` 75, over 284 distinct positions.
Evidence `RA` 569, `MC JC` 13, `JC` 2 (the README's legend defines `MJ`, which the sheet
never writes, so the released cell is kept verbatim and never mapped to an enum).
Encodings 395 in-locus and 189 intergenic, and the split is exactly 1:1 with the
`annotation` cell's own `intergenic` prefix. Variant types `snv` 405, `insertion` 122,
`deletion` 40, `substitution` 17, read from five released `mutation` forms with no
default branch: `G→A`, `+C`, `Δ1,227 bp`, `(C)6→7` (an insertion when the copy count
rises, a deletion when it falls) and `2 bp→CT` / `48 bp→33 bp`. The `→` is U+2192 and is
written as that codepoint in every pattern; the cells also carry U+00A0 and U+2011, which
`_norm` normalizes for matching while `type_statement` and `sequence_change` keep the
cell verbatim. All 202 distinct single-locus gene names and all 95 distinct flanking
names resolve to locus tags of `GCA_000007565.2`.

`position_end` repeats `position_start` for every call, uniformly. `data S1-5` releases
one coordinate per call and no end coordinate; `BacterialVariantCall.position_end` is a
required `int`, so the absence cannot be a `ProvenanceGap`, and deriving an end would
assert breseq's coordinate convention for each of the five change forms, which no
mirrored byte states. The released span stays verbatim in `sequence_change` and
`annotation`, and an L3 row (`one_coordinate_per_call`) asserts the convention so a
consumer cannot mistake it for a measurement.

### L0-L4

`verify_build(dataset_root, data_root, family=...)` dispatches on
`Family = "selection" | "proteome" | "metabolite_growth" | "metabolite_production"`.
All four reports PASS.

`proteome` (`verify_protein_dataset` + 6 own rows):

| level | row | result |
|---|---|---|
| L0 | `structural` | ok, 6 records validated |
| L1 | `count` | ok, observed 6, expected 6 |
| L1 | `orf_uniqueness` | ok, 89 ORFs, duplicates expected |
| L1 | `one_record_per_strain_and_growth_phase` | ok, 6 keys over 6 records |
| L2 | `value_fidelity` / `se_nonnegative` | ok, 12,540 values each |
| L3 | `reference_finite` | ok, 12,540 values |
| L3 | `measurement_type_consistent` | ok, `dia_protein_counts_sum_mean` |
| L3 | `every_record_measures_the_same_kt2440_locus_set` | ok, 1 key set, 2,090 loci |
| L3 | `called_variants_attach_only_to_the_identically_named_clone` | ok, TEAM-2595 carries 75, no other strain any |
| L3 | `one_coordinate_per_call` | ok, 150 stored calls |
| L4 | `stored_deletion_sets_are_supplementary_table_4s` | ok, 0 of 6 disagree |
| L4 | `supplementary_note_2_loci_are_measured_in_every_record` | ok, 10 of 10 loci |
| L4 | `pp_2088_production_fold_is_supplementary_note_2s` | ok, stored 31.895 and 18.555 against Note 2's 34 and 20 |

The last two join the built store to `si/si1.md`, a DIFFERENT released file from the
Dryad workbook the loader read. The fold row names its statistic (the ratio of the
per-sample means of the released `Counts_sum`) and holds to a declared relative
tolerance of 0.10, because the SI states two rounded integers computed on the authors'
own normalized intensities.

`metabolite_growth` and `metabolite_production` (`verify_metabolite_dataset` with
`reference_centered=False`, plus 5 own rows each):

| level | row | growth | production |
|---|---|---|---|
| L0 | `structural` | ok, 3 records | ok, 3 records |
| L1 | `count` | ok, 3 | ok, 3 |
| L1 | `genotype_uniqueness` | ok, 3 strains | ok, 3 strains |
| L1 | `one_record_per_released_strain` | ok | ok |
| L2 | `value_fidelity` | ok, 138 values | ok, 139 values |
| L3 | `reference_finite` | ok, 138 | ok, 137 |
| L3 | `measurement_type_consistent` | ok | ok |
| L3 | `every_stored_metabolite_is_an_absolute_row` | ok, 46 of 48, 0 Relative | ok, 48 of 48, 0 Relative |
| L3 | `called_variants_attach_only_to_the_identically_named_clone` | ok, 75 | ok, 75 |
| L3 | `one_coordinate_per_call` | ok, 75 calls | ok, 75 calls |
| L4 | `results_prose_amino_acids_are_not_in_data_s1_1` | ok, 0 of 3 stored | ok |
| L4 | `specific_block_recovers_one_harvest_od600_per_strain` | ok, recovered 2.4 / 1.2 / 2.7 | ok, 15.511 / 12.920 / 13.995 |

Two findings fell out of those L4 rows.

- **The Results cite `data S1-1` for three metabolites it does not release.** "We
  observed a 2-fold increase in phenylalanine and a 15-fold increase in leucine
  concentrations comparing TEAM-3174 and TEAM-3185 to the base strain (Fig. 7E and data
  S1-1). In contrast, tryptophan concentrations showed inconsistent changes between the
  two producers as TEAM-3185 had no change whereas TEAM-3174 showed a $7 \mathbf { x }$
  decrease (Fig. 7E and data S1-1)." Measured: of the 58 released rows, the only
  amino-acid-adjacent ones are `4-Aminobutyric acid` and `Glutamate`. None of the three
  is a released row, so the claim is not sourceable from the deposit. The L4 row asserts
  exactly that, pinned: a released row for any of the three would fail it.
- **The growth-phase `Specific` block recovers the harvest OD600 exactly; the
  production-phase one does not.** Dividing each stored growth-phase level by its
  released specific value gives one number per strain to within the 0.01 rounding of the
  average column: **2.400** for TEAM-2595, **1.200** for TEAM-3174, **2.700** for
  TEAM-3185. That number is the harvest OD600 the deposit nowhere states as such. In the
  production phase the ratio is not constant, and the metabolites that break it are
  pinned exactly (`SPECIFIC_BLOCK_DISAGREEMENTS`): six for TEAM-2595 (`2-Methylcitrate`,
  `ADP`, `Citrate`, `Malonate`, `NADH`, `Pyruvate`), one for TEAM-3174
  (`Methylmalonate`) and two for TEAM-3185 (`NAD`, `NADH`). That is a property of the
  released sheet, not of the build.

`selection` is unchanged and still PASSES all 19 rows, with 58 records over 60 distinct
targets.

### A finding recorded and NOT acted on

`Sheet 4` names nine non-*P. putida* proteins, and five of them are the mevalonate
pathway with their source organisms in the UniProt entry name: `HMGCS_ENTFL` (Q9FD71),
`Q9FD70_ENTFL` (Q9FD70), `Q8PW39_METMA` (Q8PW39, mevalonate kinase), `MVD1_YEAST`
(P32377, diphosphomevalonate decarboxylase) and `APHA_ECOLI` (P0AE22). That is the first
independent evidence in the mirror for the organisms behind the integrated pathway's
`mvaS`, `mvaE`, `MKmm`, `PMDHKQ` and `aphA` tokens, and `MKmm` beside a
*Methanosarcina mazei* mevalonate kinase is suggestive of the suffix read Carruthers
2025 refused. **The sheet never states which pIY670 part token each enzyme is**, so
`SOURCE_ORGANISM_UNREPORTED` stays on all five and the mapping is raised for review
rather than taken. Acting on it would also change every `IsoprenolSelectionMenasalvas2025Dataset`
record's content, which is a full-rebuild case, not an additive one.

### Adapters, written and not yet wired

Three `CellAdapter` subclasses and their enable-lists, each shaped on the precedent that
already serves the same node classes:

| adapter module | class | conf |
|---|---|---|
| `torchcell/adapters/menasalvas2025_proteome_adapter.py` | `ProteomeMenasalvas2025Adapter` | `conf/proteome_menasalvas2025_adapter.yaml` |
| `torchcell/adapters/menasalvas2025_metabolite_growth_adapter.py` | `MetaboliteGrowthPhaseMenasalvas2025Adapter` | `conf/metabolite_growth_menasalvas2025_adapter.yaml` |
| `torchcell/adapters/menasalvas2025_metabolite_production_adapter.py` | `MetaboliteProductionPhaseMenasalvas2025Adapter` | `conf/metabolite_production_menasalvas2025_adapter.yaml` |

All three enable `bacterial perturbation (chunked)` **and**
`bacterial sequence variant perturbation (chunked)`, because the two bacterial node
methods partition the leaves and the TEAM-2595 records carry called variants (the same
pairing `proteome_desiqueira2025_adapter.yaml` uses). They enable the
environment-perturbation pair (crystal violet and pH) and no `crispr construct` pair,
because no record of these three families carries a CRISPRi leaf. `LANE_OF_LABEL` in
`torchcell/database/browser_style.py` needs **no** change: every variant leaf is served
under the one `bacterial sequence variant perturbation` label it already carries, and
`ProteinAbundancePhenotype` and `MetabolitePhenotype` are already served classes.

`torchcell/adapters/__init__.py`, `dataset_adapter_map.py`, `conf/kg_bacteria.yaml`,
`torchcell/verification/runners.py` and
`tests/torchcell/adapters/_bacterial_adapter_cases.py` are NOT touched by this change
and still have to be edited for the three datasets to reach a build or the shared
runner. One thing to get right there: `METABOLITE_DATASETS` in `runners.py` has no
per-dataset `verify` hook and its L4 keys on `metabolite_gene_set`, which counts
heterologous pathway tokens and site-keyed variant ids as host loci; a
`BACTERIAL_METABOLITE_DATASETS` registry driven by `_run_bacterial_family` with
`measured_set=host_perturbed_gene_set` is the shape that works, and it was verified by
hand against the built stores (1.000 of 73 measured genes are loci of
`pputida_KT2440_ASM756v2`, and 1.000 of 2,137 for the proteome).

### Open decisions

1. Whether to map the five pathway tokens onto the five UniProt accessions above. It
   needs an explicit go-ahead and a full KG rebuild, because it changes a served
   dataset's records.
2. Whether `Sheet 5` becomes its own dataset. It is loadable; the arabinose basis and
   the double pathway copy are the two things to settle first.
3. `verify_metabolite_dataset` has no `allow_duplicate_orfs` relaxation, which is why
   the metabolite arm is two datasets rather than one of six records. A one-line
   addition mirroring `verify_protein_dataset` would allow the single-dataset shape; it
   is NOT made here, because `torchcell/verification/` is outside this change.
4. `GeneAdditionPerturbation.source_organism` is still a required `str`, so the
   sentinel of the 2026.10.07 row above is still in place, now on the mCherry biosensor
   copies as well as the five pathway genes.

## 2026.10.09 - The three Dryad arms are wired into the served graph

The 2026.10.09 section above ends with "Adapters, written and not yet wired". They are
wired now, and nothing in the loaders changed to do it.

| place | change |
|---|---|
| `torchcell/adapters/__init__.py` | three exports, three names in `pputida_adapters` (23) |
| `torchcell/knowledge_graphs/dataset_adapter_map.py` | three dataset-to-adapter entries; the map now holds 114 |
| `torchcell/knowledge_graphs/conf/kg_bacteria.yaml` | three dataset names; the bacterial list is now 63, all distinct |
| `torchcell/verification/runners.py` | three verify hooks, `proteome_menasalvas2025` into `BACTERIAL_PROTEIN_ABUNDANCE_DATASETS`, a new `BACTERIAL_METABOLITE_DATASETS` registry, `run_bacterial_metabolite` in `run_all` |
| `tests/torchcell/adapters/_bacterial_adapter_cases.py` | three `_case` rows, all `variant=True` |
| `tests/torchcell/adapters/test_menasalvas2025_{proteome,metabolite_growth,metabolite_production}_adapter.py` | three new modules, the shape the other bacterial adapter tests use |

### The metabolite registry, and why it is its own

The note's recommendation is what landed. `METABOLITE_DATASETS` has no per-dataset
`verify` hook and its L4 keys on `metabolite_gene_set`, which counts a heterologous
pathway token (`MvaSEf`) and a site-keyed variant id (`<replicon>:<position>`) as a locus
of the host assembly, so a production host's records fail it for the wrong reason.
`BACTERIAL_METABOLITE_DATASETS` is driven by `_run_bacterial_family` with
`measured_set=host_perturbed_gene_set`, the set that excludes both, exactly as
`BACTERIAL_PROTEIN_ABUNDANCE_DATASETS` is. The E. coli metabolome rows stay where they
are; nothing about them changes.

### Measured through the new registries, 2026-10-09

| store | records | verdict |
|---|---|---|
| `proteome_menasalvas2025` | 6 | PASS, L0 to L4 |
| `metabolite_growth_menasalvas2025` | 3 | PASS, L0 to L4 |
| `metabolite_production_menasalvas2025` | 3 | PASS, L0 to L4 |

Both metabolite phases report
`L4 perturbed_gene_containment_assembly: 1.000 of 73 measured genes are loci of pputida_KT2440_ASM756v2`,
which is the number the earlier section verified by hand. The growth phase stores 138
values over 46 metabolites, the production phase 139 over 48.

Two stores in `BACTERIAL_PROTEIN_ABUNDANCE_DATASETS` FAIL on the shared dev tree and are
not this branch's: `proteome_caglar2017` (105/105 records fail L0 schema validation) and
`proteome_ishii2007` (28/28 fail L0, and L1 observed 28 against expected 24). Both are
E. coli stores whose loaders this branch does not touch, built by another branch under an
older closure. Measured 2026-10-09 by running
`torchcell.verification.runners.run_bacterial_protein_abundance` over the dev tree.
