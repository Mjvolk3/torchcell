---
id: 6gw9mvzdphly95kxguhu2qv
title: Lim2025
desc: ''
updated: 1791394449001
created: 1791394449001
---

## 2026.10.07 - Lim 2025: two loaders, four records, and the forty-six clones no class can hold

`torchcell/datasets/pputida/lim2025.py` serves row 16 of the bacterial expansion list:
Lim et al. 2025, "Evolution-guided tolerance engineering of Pseudomonas putida KT2440
for production of the aviation fuel precursor isoprenol"
(doi:10.1016/j.ymben.2025.05.007, `citation_key`
`limEvolutionguidedToleranceEngineering2025`, `paper.md` sha256
`26b88d819d429f49cad4ebe85047cfd7354d532d84d584b0f00b222355772642`). Templates:
[[torchcell.datasets.pputida.carruthers2025]] and [[torchcell.datasets.pputida.lim2022]];
skeleton [[torchcell.datasets.bacteria_common]]; schema layer
[[torchcell.datamodels.bacterial-perturbation-ontology]]; media
[[torchcell.datamodels.media]]; plan [[plan.bacteria-ontology-genome]] section 4.

### The two dataset classes

| class | experiment class | records | what one record is |
|---|---|---|---|
| `IsoprenolToleranceLim2025Dataset` | `BacterialEnvironmentResponseExperiment` | 3 | one writable strain's log2 growth-rate ratio to KT2440 at one isoprenol dose |
| `ProteomeLim2025Dataset` | `BacterialProteinAbundanceExperiment` | 1 | the parent IPL400's Top3 log2 proteome under 4 g/L isoprenol |

### The genotype finding: matched to de Siqueira 2025, reached on these bytes

This study tolerizes KT2440 and two stacked deletion strains against isoprenol and
resequences the endpoint and intermediate clones, so an evolved clone's genotype IS its
parent plus the variants breseq 0.33.1 called. Supplementary Data 1
(`si/si2.xlsx`, sha256 `a3cfd601...`, sheet `Fig 2B_Mutation List`) releases exactly
that: **159 rows** on **159 distinct `AE015451` positions** -- the single replicon of the
pinned `pputida_KT2440_ASM756v2` assembly -- crossed against **49 clone columns** as
**443 per-clone calls** (431 at frequency 1, 12 at 0.9). The 49 columns are the 46
resequenced evolved isolates plus the three F0 founder clones (A1 F0 = WT with 0 calls,
A5 F0 = IPL300 with 6, A9 F0 = IPL400 with 8). Mutation types: 100 SNP, 47 DEL, 11 INS,
1 SUB.

No class in `schema.py` can hold one of those calls honestly. These are the same reasons
the de Siqueira 2025 loader found for row 14's 173 calls
([[torchcell.datasets.pputida.desiqueira2025]]), measured independently here, and this
loader **matches that treatment** rather than inventing a second representation:

1. `SequenceVariantPerturbation` is the only variant-level leaf and its base validator
   admits only S288C ORF names:
   `SequenceVariantPerturbation(systematic_gene_name="PP_3415", ...)` raises
   `Invalid systematic gene name format`. It has no `gene_namespace` field.
2. Its contract is a dereferenceable allele SEQUENCE (`sequence_source` +
   `sequence_ref`). This paper deposited raw reads (BioProject PRJNA1187681) and no
   per-gene allele store, so even a widened class would carry two `None` pointers.
3. No class has a slot for what a row RELEASES: replicon, position, reference base,
   alternate base, mutation type, amino-acid change, call frequency.
   `BacterialBackgroundAllele.deleted_span` is deletion-only and would lose all of it.
4. `BacterialBackgroundAllele.functional` is a required `bool`, so the unknown
   functional consequence of a missense SNP cannot be a `ProvenanceGap` -- a gap must
   name a field that is `None`.
5. `BacterialStrainBackground` permits ONE allele entry per locus, which refuses the **8
   rows** whose locus carries more than one call in a single clone. `PP_3415` alone
   carries three distinct variants across the campaign and TWO inside A12_F53_I1 (P293S
   at 3,866,001 and V46I at 3,866,742, Supplementary Table 5).
6. `Genotype.__eq__` compares the perturbation SET, so a clone written with its parent's
   perturbations only would be genotype-identical to its parent, collapsing 46 strains
   onto three identities.

**Two shapes this row adds beyond row 14's.** The 159 rows partition cleanly into **123
on one locus, 17 intergenic and 19 spanning several loci**. An intergenic row's `Gene`
field names the two FLANKING loci, so there is no locus to key to and inventing a
neighbor is exactly what must not be done. A multi-locus row is a large deletion (up to
the 53 genes of `PP_3024-PP_5558`) that no gene-keyed perturbation can state at all; it
needs a span-level carrier, which row 14's point-variant-only table never required.

Every one of the 159 rows is typed into `preprocess/called_variants.json` by
`read_variant_calls`, each with its own `blocking_reasons`, in the same file name and the
same vocabulary row 14 writes.

**A cross-source disagreement kept as a finding.** The Results state "a total of 158
unique mutations in 73 genetic regions"; the released matrix holds 159 rows on 159
distinct positions under 78 distinct region labels. Reported, not reconciled.

### The additive proposal that would make the evolved clones loadable

Four additions, no edit to an existing class, so no served dataset's closure moves. The
first three are row 14's proposal verbatim; the fourth is this row's own.

1. **`BacterialSequenceVariantPerturbation(SequencePerturbation)`**, the bacterial
   sibling of `SequenceVariantPerturbation`:
   `perturbation_type: Literal["bacterial_sequence_variant"]`, `provenance="natural"`,
   `mechanism_so_id="SO:0001483"` (`SNV`) with an `indel` variant for a length change,
   plus `gene_namespace: BacterialGeneNamespace`, the locus-tag validator the other
   bacterial leaves use, and the released call fields: `chromosome`, `position` (1-based
   on the pinned assembly), `reference_allele`, `alternate_allele`, `polymorphism_type`,
   `amino_acid_change: str | None`, `codon_number: int | None`,
   `variant_frequency: float | None`, `caller: str | None`. `functional` is deliberately
   absent: a called variant does not state a functional consequence.
2. **An intergenic carrier.** `GenePerturbation` requires a gene and 17 of these calls
   have none. A sibling `IntergenicVariant` record hung off the strain background, with
   `nearest_gene` + `offset_bp` left `None` and typed gaps where the source gives
   neither, is preferred over a `systematic_gene_name=None` form: it keeps
   `GenePerturbation`'s invariant that a perturbation names a gene.
3. **`BacterialStrainBackground` must admit several entries at one locus**, or the
   variant leaves must live on the genotype rather than the background. "A haploid
   background carries one allele entry per locus" is right for designed alleles and
   wrong for called variants, three of which stack at `PP_3415`.
4. **A span-level deletion carrier** (this row's addition). 19 rows are one deletion
   over a run of loci, up to 53 genes. Writing 53 `BacterialDeletionPerturbation`s
   would assert 53 independent events; the honest form is one record carrying a
   `GenomicSpan` plus the locus list the source gives. The Carruthers 2025 chassis has
   the same shape and solves it the same way it is solved here: as a stated gap for 54
   unnamed loci.

Until all four land, the honest record set is the writable strains below.

### What IS writable, and why only three carry a number

| strain | expressible as | released number |
|---|---|---|
| KT2440 (WT) | empty `Genotype` on `assembly_reference("KT2440")` | it is the reference arm (log2(1) = 0) |
| IPL300 | 6 `BacterialDeletionPerturbation` | Supplementary Table 3 initial growth rate |
| IPL400 | those 6 + `PP_1385` (`ttgB`) | Supplementary Table 3 initial growth rate; the proteome arms |
| KT2440 `dPP_3024` | 1 `BacterialDeletionPerturbation` + `StrainConstruction(JBEI-235868)` | the Fig. 3A text's 1.6-fold |
| 6 other reverse-engineered deletions | `BacterialDeletionPerturbation` | none: Fig. 3A releases no numbers |
| 8 pIY670 strains | + `HeterologousPathwayPerturbation` | none: Figs 3B, 5A, 5D release no numbers |

The parent genotypes are the DESIGNED deletions only, which is a floor on their content:
"the starting strains IPL300 and IPL400 also contained pre-existing mutations missing
from the reference sequence: PP_4986, gacS, yhjE in IPL300, and then IPL400 had the same
mutations along with a mutation in PP_4398." Those calls are in the same
`called_variants.json` and in `preprocess/genotype_gaps.json`; they are missing ROWS,
not missing fields, so they are not a `ProvenanceGap`.

#### PP_2676's truncation: typed where the framing admits it, gapped where it does not

Supplementary Table 1 writes the parent lesion as "KT2440 with the complete internal
in-frame deletion of PP_2675 and partial truncation of the first fourteen amino acids of
PP_2676" -- one deletion event, since `PP_2676`'s ORF overlaps `PP_2675`'s CDS, and the
same physical lesion row 14's `PT` strain carries from the same Thompson 2020 strain
(`thompsonFattyAcidAlcohol2020`, `paper.md` sha256 `389d0d6c...`, Table 1: "Strain with
complete internal in-frame deletion of PP_2675"). The bacterial PERTURBATION axis has no
partial-deletion leaf, so:

- `ProteomeLim2025Dataset`, whose comparison is IPL400-stressed against
  IPL400-unstressed, states the whole strain on a `BacterialStrainBackground`: seven
  `full_deletion` alleles plus `PP_2676` as a `partial_deletion`, each with a typed
  `deleted_span` gap (no source gives coordinates). That is how row 14 types its `PT`
  background, and it is where the truncation CAN be said.
- `IsoprenolToleranceLim2025Dataset`, whose comparison is strain-against-KT2440, carries
  the reference-strain genome with NO background (the Wang 2015 pattern, because the
  reference ARM is the wild type) and the whole lesion set as perturbations. There the
  truncation has no home and stays a stated gap.

`functional=False` on the full-deletion alleles rests on "Both IPL300 and IPL400 were
unable to grow with isoprenol as the sole carbon source in M9 media as they contained
deletions of dPP_2675 and dPP_4064-dPP_4067, which were both shown to abolish growth
individually on isoprenol when deleted". The source never states `PP_2676`'s functional
status on its own; its allele rests on the `d14` designation and on the ORF overlap, and
the allele's own note says so.

### Family 1: isoprenol tolerance

Source: Supplementary Table 3 of `si/si1.docx` (sha256 `7bc74885...`), read from the
WordprocessingML body with a stdlib `zipfile` + `ElementTree` walk. The table is located
by its header cells, the group label and the starting concentration are forward-filled as
the source writes them, and the parsed 16 lineages hash to `TABLE_S3_SHA256`
`07e309b919552ee925bdc1c7d57b0e8fa0085da247b1fb425014db1b75a6c12c`.

**Why a log2 ratio and not an absolute growth rate.** `MeasurementType.growth_rate`
exists, but `torchcell.verification.environment_response`'s L3 `reference_zero` requires
a numeric reference of exactly 0, which an absolute rate in h^-1 cannot be. Both released
quantities are already comparisons of one strain to KT2440 in the SAME medium at the SAME
isoprenol dose by the same liquid-OD assay, so `log2_ratio` is the faithful encoding and
the reference is KT2440 at log2(1) = 0, the Wang 2015 pattern for this same compound
([[torchcell.datasets.ecoli.wang2015]]). Raising the verifier to accept a declared
non-zero numeric baseline is in the PR body.

**`n_samples` = 4 biological replicates, and the lineages therefore AGGREGATE.** "Each
TALE experiment was conducted with four independent biological replicates in one
high-throughput evolution campaign", so ALE 1-4 (WT), 5-8 (IPL300) and 9-12 (IPL400) are
four measurements of one (strain, condition) and L1 requires them in one record. Each
arm's value is the mean of its four lineages; the per-lineage numbers stay in
`preprocess/table_s3.csv` and the whole table in `preprocess/tale_lineages.csv`.

**The uncertainty is a typed gap.** Supplementary Table 3 reports no dispersion for any
lineage, and the stored value divides by the WT arm's mean, so a sample SD over the four
numerators would propagate one arm only. Both `environment_response_uncertainty` and
`environment_response_se` carry `ProvenanceGap`s, as in Wang 2015. What one number IS is
quoted from the unmarked footnote `a` of the same table: "Average growth rate, observed
in the three first flasks of each experiment." Footnote `b` is marked on "Final growth
rate" and names the three LAST flasks, so `a` is the initial-rate footnote by its own
text; the marker is absent from the header cell in the released XML.

#### The three records

| record | genotype | environment | value | source |
|---|---|---|---|---|
| 0 | IPL300, 6 deletions | M9 + 4 g/L glucose + 4 g/L isoprenol | `log2(0.2525 / 0.1535)` = 0.71804 | Table 3, ALE 5-8 |
| 1 | IPL400, 7 deletions | the same | `log2(0.30075 / 0.1535)` = 0.97033 | Table 3, ALE 9-12 |
| 2 | KT2440 `dPP_3024` | M9 + 4 g/L glucose + 6 g/L isoprenol | `log2(1.6)` = 0.67807 | Fig. 3A text |

The `dPP_3024` reading: "The smaller dPP_3024 deletion essentially accounts for most of
the tolerance improvement (1.6-fold) seen in the larger region mutant." Read as that
mutant's own fold improvement, corroborated by the companion sentence -- the two PP_3024
mutants average 1.7-fold, so 1.6 is BELOW the pair mean, which is what "most of" rather
than all of the improvement means. The assertion `PP3024_FOLD < PP3024_PAIR_MEAN_FOLD`
is enforced at build time. The larger mutant's implied 1.8-fold is arithmetic on rounded
numbers and is NOT stored.

#### Dropped, with counts (`preprocess/dropped_records.json`)

| scope | rule | n |
|---|---|---|
| record | `hcho_arm_has_no_strain_other_than_the_reference` | 1 |
| record | `final_growth_rate_is_an_evolved_population` | 1 (16 released values) |
| strain | `evolved_clone_genotype_is_not_representable` | 46 |
| readout | `released_only_as_a_figure` | 14 |
| readout | `no_titer_is_released_per_writable_strain` | 0 |

The HCHO arm (ALE 13-16) tolerized only the WT, which is this dataset's reference, so its
aggregated initial growth rate would be the reference against itself. The figure-only
readouts are Fig. 2A (16 strains x 2 doses), Fig. 3A's other six strains, Figs 3B / 5A /
5D (titers), Figs 5B / 5E (residual glucose), Figs 4D / S3B (isoprenol degradation) and
Figs S1C-F (growth curves). No titer record exists for a writable strain: the Results
give titers only as ranges over several strains (60-70 mg/L, 280-400 mg/L) or for
derivatives of the evolved isolate A10_F63_I1 (337 mg/L `dfleQ`, 370 mg/L `dmvaB`).

#### Three cross-source assertions, all enforced at build time

1. Supplementary Table 3's `Generations` and `CCD` columns reproduce the Results' stated
   spans exactly: 143-353 generations and 1.33-3.32e12 CCD for the isoprenol arm,
   346-387 and 3.11-3.56e12 for the HCHO arm.
2. The HCHO arm's doses agree across units: Table 3's 1 mM and 9 mM are the Results'
   0.03 g/L and 0.27 g/L at formaldehyde's 30.026 g/mol. The molar mass is used only for
   this check and never to produce a stored number.
3. Table 3's own isoprenol starting-concentration column is 4 g/L for all 12 lineages,
   which is the dose the Results state for the TALE start.

### Family 2: the parent's proteome

Source: `si/si2.xlsx` sheets `Proteome_A10F63I1vsIPL400_M9G` (2,367 rows) and
`Proteome_A10F63I1vsIPL400_G+4IP` (2,374 rows), columns `log2_mean_IPL400_*` and
`log2_std_IPL400_*`. The record is IPL400 under 4 g/L isoprenol; its
`phenotype_reference` is the same strain's unstressed arm, which is where the M9-glucose
abundances are stored.

**`n_replicates` and the uncertainty type are BACK-SOLVED, not assumed.** Reading each
`log2_std` as the SAMPLE SD of three replicates reproduces the released `t-test_stat` for
every kept row (worst residual **9.2e-13** on the M9G sheet and **1.0e-12** on the
`G+4IP` sheet); reading it as a population SD misses by a median of 0.45 t units. The
released `log2_Fold_change_A/B` equals the difference of the two means to 9.3e-15. So
`protein_abundance_se = log2_std / sqrt(3)` and `n_replicates = 3`, which agrees with the
`ProteomeXchange Sample Key` sheet's own `R1,R2,R3` for both loaded arms and with
"Samples were analyzed as three independent biological replicates". The Welch identity is
an enforced build-time assertion, which is also what finds the dropped paralogs.

**One IPL400 measurement, exported four times.** `Proteome_A12F53I1vsIPL400_M9G`'s IPL400
columns are bit-identical to the A10 sheet's over all 2,361 kept loci, and the two
`G+4IP` sheets agree exactly on their 2,332 shared loci (the A10 sheet carries 36 more).
Asserted at build time; it becomes one record, never averaged.

#### Identifier histogram

`reconcile_locus_tags` over the 2,361 released locus tags, against
`pputida_KT2440_ASM756v2`: `current` 2,361, everything else 0; resolver layer `locus tag`
2,361; 0 remapped, 0 kept on a collision, 0 outside the namespace. `MIN_RESOLVED_FRACTION
= 1.0`, because the sheets' `Locus tag` column is already `PP_` tags -- the DIA-NN search
database was "the latest Uniprot P. putida KT2440 proteome FASTA sequence". The nine
genotype loci resolve the same way, all `current` through the `locus tag` layer.

#### Dropped protein keys: 13 of 2,374, leaving 2,361

- **6 by `gene_symbol_filed_under_two_paralogous_loci`.** The sheets give one protein
  group's mean and SD to BOTH loci of three symbols -- `Ubid` (PP_0548, PP_5213), `Pyrc`
  (PP_1086, PP_4999), `Dapa` (PP_1237, PP_2639) -- under distinct UniProt accessions, and
  those are exactly the rows whose released t does not reproduce at n = 3. One group's
  statistics cannot be attributed to either paralog. Dropping them is what makes the
  Welch identity hold everywhere else.
- **7 by `no_key_matched_reference_abundance`.** PP_0002, PP_0416, PP_0985, PP_2271,
  PP_3610, PP_3699 and PP_5287 are quantified under isoprenol but not in the unstressed
  reference arm. These are exactly the seven the `G+4IP` sheet title-cases as `Pp_0002`;
  upper-casing the locus-tag column is the one normalization applied, and the verbatim
  spelling is kept on the parsed row.

#### Dropped arms, with counts

| rule | n | what |
|---|---|---|
| `arm_is_an_evolved_isolate` | 4 | A10_F63_I1 and A12_F53_I1 in both media |
| `production_medium_has_no_media_library_entry` | 6 | both strains at 12, 24 and 48 h |
| `unstressed_arm_is_the_reference_not_a_record` | 1 | IPL400 M9G, stored as the reference |

The three pIY670 production-culture timepoints are at **20 g/L glucose** with kanamycin
and 2 g/L arabinose. `MEDIA_LIBRARY` holds Lim 2025's M9 at its default 4 g/L glucose
(`M9_NREL_LIM2025`) only, and `media.py` is a value-surface file this branch does not
edit, so those arms are not loaded and the addition is proposed in the PR.

### Environment, media and compound

Every loaded record is on `M9_NREL_LIM2025`, this paper's own library entry, with
isoprenol as a `SmallMoleculePerturbation` in `g/L`. `temperature` is `None` with a typed
gap: no incubation temperature is stated for the TALE flasks, the tolerance assay or the
proteome cultures (30 C is stated only for the LB pre-culture of the production runs).
`duration_hours` is `None` with a per-family gap: the TALE passaged "once or twice a day",
Fig. 3A states no duration for a maximum-growth-rate readout, and the proteome cultures
were harvested at a growth STATE ("Cultures at the exponential growth phase were
harvested").

`isoprenol` resolves through `compound_identity_table.json`; the row landed with the
bacterial schema follow-ups (PR #729) while this branch was in flight, and the dev store was
rebuilt against it. `ISOPRENOL_INCHIKEY` = `CPJRRXSHAYUTGL-UHFFFAOYSA-N` (SMILES
`C=C(C)CCO`, PubChem CID 12988) pins the identity in the module and the build stops if the
row ever disagrees. This is the same object the Carruthers 2025 and Wang 2015 loaders store.

### Raw mirror

`$DATA_ROOT/torchcell-raw/limEvolutionguidedToleranceEngineering2025/` holds
`data/si1.docx` (5,492,159 B, sha256 `7bc748853a07d23e2bc0712a33cfe74c68b7009ad91ff52b1d98e1f875fbc2d3`)
and `data/si2.xlsx` (2,769,528 B, sha256
`a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612`), each with a
`RetrievalMethod.direct_url` record naming
`torchcell.literature.retrieve.elsevier_mmc` and the PII `S1096717625000837`. The
recorded retrieval was re-run on 2026-10-07 and reproduced both digests byte for byte.

Three sequencing deposits are RECORDED and not downloaded, because no loader consumes
reads or spectra: SRA BioProject **PRJNA1187681** (genome resequencing; the campaign's
own mutation analysis is also public as ALEdb project `Pputida_isoprenol_TALE`), GEO
**GSE281392** (the WT isoprenol-vs-glucose transcriptome) and PRIDE **PXD054609** (raw
DIA spectra).

**Why SI quotes are asserted at build time rather than audited.** A `.docx` and an
`.xlsx` are zip archives, so `audit_sourced_value` cannot find a quote in their bytes.
Every `SourcedValue` therefore anchors to the plain-text `paper.md` (or to Thompson
2020's), and the eight statements that live only in `si1.docx` are asserted present in
the EXTRACTED text inside `process()`, which refuses the build on drift. That is a
stronger check than a post-hoc audit, and it is the pattern
[[torchcell.datasets.ecoli.goodall2018]] uses.

### Build and verification

```bash
python -m torchcell.datasets.pputida.lim2025 deposit --download-dir <dir> --retrieve
python -m torchcell.database.build_dataset_lmdb --dataset IsoprenolToleranceLim2025Dataset
python -m torchcell.database.build_dataset_lmdb --dataset ProteomeLim2025Dataset
python -m torchcell.datasets.pputida.lim2025 verify
```

Both dev-tree stores build (3 and 1 records, gene sets of 8 and 7) and read `fresh` under
`python -m torchcell.provenance.build_manifest`. Both pass L0 to L4 from the module's own
runner, including the 29 `provenance_audit` rows:

- tolerance: L0 structural 3/3, L1 count 3, L1 pair uniqueness 3 unique triples, L1
  provenance gaps 15 over 3/3 records, L1 canonical gene names 8, L2 value fidelity, L2
  uncertainty sanity, L3 measurement type `log2_ratio`, L3 reference zero, L3 environment
  perturbed, L3 compound identity, L3 media membership, L4
  `gene_containment_kt2440_locus_tags` 8 of 8.
- proteome: L0 structural 1/1, L1 count 1, L1 ORF uniqueness 7, L2 value fidelity 2,361,
  L2 SE non-negative 2,361, L3 reference finite + key-matched 2,361, L3 measurement type
  `dia_nn_top3_log2_mean`, L4 deleted loci 7 of 7 and quantified loci 2,361 of 2,361.

`sgd_genes` is deliberately NOT passed to the environment-response verifier: it turns on
a row whose message reads "of N measured genes are S288C reference genes" over what are
KT2440 locus tags, which would be a mislabelled claim in the report. The containment is
asserted by this module's own L4 row instead, and raising that message to name the
reference it was given is in the PR body.

### Tests

`tests/torchcell/datasets/pputida/test_lim2025.py`, 105 synthetic tests plus 5
`@pytest.mark.data`. The synthetic path writes a WordprocessingML `si1.docx` and an
`si2.xlsx` from scratch, deposits them through the module's own `deposit_raw_mirror`, and
builds BOTH classes end to end against a synthetic KT2440 assembly with the network
refused, so the whole pipeline runs with no network call and no read of the real
`$DATA_ROOT`. `diff-cover` reports **98.8%** on `lim2025.py`.

## 2026.10.09 - Called variants are writable: the #731 leaves applied

The 2026.10.07 section above concluded that no perturbation class holds a called base
change at a coordinate, so no clone column became a record. That conclusion no longer
holds: issue #731 added three leaves to `torchcell/datamodels/schema.py`, and this loader
now writes the calls of four of the matrix's 49 clone columns. Everything the earlier
section says about the matrix's SHAPE still stands; what changed is where those rows go.

### The three leaves, and which released row shape maps to each

The partition is decided by the row's own cells (`released_row_representation`), measured
on the pinned `si2.xlsx` (sha256
`a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612`, sheet
`Fig 2B_Mutation List`, 159 data rows):

| released row shape | n rows | leaf | identifier |
|---|---|---|---|
| `Details` starts `intergenic` | 17 | `BacterialSiteVariantPerturbation` (`site_kind=intergenic`) | `AE015451:<position>`, the derived site id |
| `Mutation Type == DEL` and `Details` empty | 25 | `BacterialSpanDeletionPerturbation`, ONE per covered locus | that locus's `PP_` tag |
| everything else | 117 | `BacterialSequenceVariantPerturbation` | that locus's `PP_` tag |

The span-deletion bucket is 25, not the 19 the multi-locus count gives: 19 rows name more
than one locus (up to the 53 of `PP_3024`-`PP_5558`) and 6 name exactly one
(`1578244 DEL ttgB`, `1831944 DEL PP_1635`, three `PP_4063` rows, `5076838 DEL PP_4470`).
Both are the same shape, because breseq omits the coding offset exactly when the named
loci lie wholly inside the deleted interval, so a single-locus row of that shape is a
deletion that removes the whole locus. Every row left after the first two tests names
exactly one locus; the loader asserts that rather than assuming it.

`call_mode=VariantCallMode.clone` and `frequency_basis=fraction` on every call: the
matrix cell is a within-clone fraction, measured as 431 cells reading `1` and 12 reading
`0.9` over its 443 calls, so no cell exceeds 1 and none is a percent. `caller` is the
`breseq 0.33.1` the loader already sourced, used verbatim.

### The `Gene` cell is usually a symbol, and resolving it changed the dedup count

Measured over all 241 distinct `Gene` tokens of the sheet against the pinned
`pputida_KT2440_ASM756v2` assembly: 151 are already locus tags, 90 resolve through the
gene-symbol layer, 1 is ambiguous (`asd` to `PP_1989` or `PP_1992`) and 0 are not found.
Every stored tag that differs from the released token carries a
`DerivedIdentifierMapping(route="gene_symbol")`. A token that does not land on exactly
one locus is a hard error naming it; no mapping is invented. A clone column carrying the
`asd` row would therefore be refused, which is the honest behavior, and none of the four
loaded columns carries it.

This resolution is what corrected the dedup count. Before resolving the symbols, only two
of the IPL400 founder's calls look like restatements of its designed lesions. After
resolving, six do: `adhP` is `PP_3839` and `ivd,mccB,liuC,mccA` are `PP_4064`-`PP_4067`,
all four of them designed deletions of IPL300.

### No end coordinate is asserted

The release gives a 1-based `Position` and a LENGTH fused into `Sequence Change`
(`Δ5,553 bp`, `(CCAC)2→1`, `2 bp→CG`), never an end. `position_end` therefore equals
`position_start` on every call, including the multi-base ones; the released extent stays
verbatim in `sequence_change`, and `deleted_span` is left None on the span leaf.
Synthesizing an end would mean parsing that string and choosing which side of the position
the length runs to, and the source states neither. `reference_allele`,
`alternate_allele`, `amino_acid_change`, `codon_change` and `codon_number` stay None for
the same reason: the sheet fuses them into `Sequence Change` and `Details` (`C→G`,
`G476A (GGT→GCT)`) instead of giving them their own columns, and both cells are stored
verbatim. The `Details` offsets use U+2011 non-breaking hyphens and keep them.

### The clone-column-to-strain assignment is proven, not assumed

`assert_founder_columns` refuses the build unless three measured facts hold together:

- `A1 F0 I1 R1` carries **0** calls. The matrix is called against KT2440 WT, so its own
  founder column must carry none.
- `A9 F0 I1 R1`'s 8 call positions are a strict superset of `A5 F0 I1 R1`'s 6.
- What `A9 F0` adds is exactly two rows, `ttgB` (1578244) and `PP_4398` (4989633), which
  is the Results' own "IPL400 had the same mutations along with a mutation in PP_4398"
  plus IPL400's designed ΔttgB.

The two proteome isolates' parent is quoted, not inferred: "we chose two representative
evolved end-point isolates (i.e., A10_F63_I1 and A12_F53_I1) which were derived from the
same starting strain (IPL400) but contained mutations in different genes (Supplementary
Table 5)." Supplementary Table 3's row groups put ALE 9-12 on IPL400, so ALE 10 and ALE
12 agree independently.

### Measured before and after, read from the rebuilt LMDBs

| dataset | records before | records after | what changed |
|---|---|---|---|
| `IsoprenolToleranceLim2025Dataset` | 3 | 3 | the same three records, with the parents' calls now on the IPL300 and IPL400 genotypes; `gene_set` 8 to 12 |
| `ProteomeLim2025Dataset` | 1 | 3 | A10_F63_I1 and A12_F53_I1 added, each against its own unstressed arm; `gene_set` 7 to 24 |

Per-record called-variant accounting, from
`preprocess/called_variant_perturbations.json` of each build:

| record | clone column | called rows | candidate perturbations | written | in a locus | site-keyed | span loci | restated |
|---|---|---|---|---|---|---|---|---|
| IPL300 | `A5 F0 I1 R1` | 6 | 9 | 3 | 3 | 0 | 0 | 6 |
| IPL400 | `A9 F0 I1 R1` | 8 | 11 | 4 | 4 | 0 | 0 | 7 |
| A10_F63_I1 | `A10 F63 I1 R1` | 16 | 19 | 12 | 8 | 4 | 0 | 7 |
| A12_F53_I1 | `A12 F53 I1 R1` | 18 | 21 | 14 | 11 | 3 | 0 | 7 |

Total genotype sizes: IPL300 9 perturbations (6 designed + 3 called), IPL400 11 (7 + 4),
A10_F63_I1 19 (7 + 12), A12_F53_I1 21 (7 + 14). `KT2440 dPP_3024` gets no calls: it is a
reverse-engineered strain and not one of the 49 clone columns.

### The dedup rule, and the 27 perturbations it dropped

A call whose resolved locus the record's own strain already carries as a designed
`BacterialDeletionPerturbation` is DROPPED, so absence has one encoding. The drop is per
LOCUS, so a span event keeps its full `span_systematic_gene_names` (the event does remove
them all) and only stops writing a second perturbation for the designed one.

Across the four loaded records that is **27** dropped perturbations: 6 on IPL300 and 7 on
each of the other three. The restated loci are `PP_2675` (3063719), `PP_3839`/`adhP`
(4362918) and `PP_4064`-`PP_4067`/`ivd,mccB,liuC,mccA` (4588139) for all four, plus
`PP_1385`/`ttgB` (1578244) for the three IPL400-background records.

One consequence worth stating, because it is easy to misread the leaf table above:
**zero `BacterialSpanDeletionPerturbation` instances survive onto a loaded record.** Every
span-deletion row the four clone columns carry (`1578244 DEL ttgB` and
`4588139 DEL ivd,mccB,liuC,mccA`) is entirely restated by the designed deletions, so the
leaf's code path runs on 5 loci per IPL400-background record and all 5 are dropped. The
span leaf is exercised but currently unpopulated in the built stores; a record of one of
the other 45 clone columns would populate it.

### What still refuses, with counts

- **The 16 lineage final growth rates of Supplementary Table 3.** Each is the average
  over a lineage's three LAST flasks, so its strain is the evolving POPULATION in that
  flask, not one of the 46 sequenced isolates. The matrix calls CLONES
  (`VariantCallMode.clone`) and releases no population allele frequency for a flask, so a
  population genotype would need a threshold the release never gives for one. The #731
  leaves do not change this, because what they hold is a clone's calls; the rule
  `final_growth_rate_is_an_evolved_population` stays, with its reason restated.
- **The 46 evolved isolates as TOLERANCE records.** Writable now, but Supplementary
  Table 3 releases a growth rate per lineage and not per isolate, and Fig. 2A's
  per-isolate rates are a figure with no numbers. The drop rule was renamed from
  `evolved_clone_genotype_is_not_representable` to
  `evolved_isolate_has_no_released_per_isolate_number`, because the old name is now a
  false claim. The two whose proteome IS released are the new `ProteomeLim2025Dataset`
  records.
- **`PP_2676`'s 14-codon N-terminal truncation on the perturbation axis.** Typing it as a
  `BacterialSequenceVariantPerturbation` of deletion type was considered and refused: the
  leaf needs a 1-based `position_start` on `AE015451` and a `sequence_change`, and neither
  Lim's Supplementary Table 1 nor Thompson 2020's strain table gives a coordinate, a
  coding range or a length for it. The `d14` designation does not even say whether 14
  counts base pairs or codons. It stays a `BacterialBackgroundAllele(partial_deletion)` on
  the IPL400 background and a stated entry in `preprocess/genotype_gaps.json`. One entry,
  unchanged.
- **A called variant on the strain-BACKGROUND axis.** `BacterialBackgroundAllele` requires
  a non-optional `functional: bool` that a call's consequence is unknown for, and
  `BacterialStrainBackground` permits one allele entry per locus, which refuses
  A12_F53_I1's two `PP_3415` calls (P293S at 3,866,001 and V46I at 3,866,742). Each
  proteome record's `genome_reference` therefore carries IPL400's background, which for
  the two evolved isolates is a FLOOR on the genomic content of their own unstressed
  reference arm. This is the second `genotype_gaps.json` entry, NEW, and it replaces the
  one the leaves closed. `genotype_gaps.json` therefore still holds 2 entries: the
  `PP_2676` truncation and this one.
- **29 protein keys of A12_F53_I1.** Its `G+4IP` sheet carries 2,338 rows against the
  2,367 of its `M9G` sheet, so 29 loci are quantified unstressed but not under isoprenol
  and the record would have a reference value with no measurement. That is the new drop
  rule `no_key_matched_isoprenol_abundance` (0 for IPL400 and A10_F63_I1, 29 for
  A12_F53_I1), the mirror of the existing `no_key_matched_reference_abundance` (7 keys
  each for IPL400 and A10_F63_I1, 0 for A12_F53_I1).
- **The 6 paralogous protein keys** (`PP_0548`, `PP_1086`, `PP_1237`, `PP_2639`,
  `PP_4999`, `PP_5213`) and the 6 pIY670 production arms, both unchanged. Measured: the
  same six loci on all four consumed sheets.

### What the four proteome sheets now give, and the statistics re-check

All four comparison sheets are consumed, not two, because each record needs the same
column of two sheets: the strain's own `M9G` arm and its own `M9G+4IP` arm. The Welch
back-solve at n = 3 with the released `log2_std` as a sample SD reproduces the released
`t-test_stat` on all four, worst residual 1.05e-12 (`A10 M9G` 9.24e-13, `A10 G+4IP`
1.05e-12, `A12 M9G` 7.46e-13, `A12 G+4IP` 6.07e-13), and the Sample Key sheet gives
`R1,R2,R3` for all six consumed sample names. The IPL400 columns remain bit-identical
across the sheets (2,361 shared loci on the `M9G` pair, 2,332 on the `G+4IP` pair, zero
differing), which is what still licenses ONE IPL400 record rather than an average of four.

The per-locus CSV moved from `preprocess/ipl400_proteome.csv` to
`preprocess/lim2025_proteome.csv` and gained a `strain` column, since it now carries three
strains.

### L0 to L4, both rebuilt stores

`python -m torchcell.datasets.pputida.lim2025 verify`, exit 0, both PASS. The 30
`provenance_audit` rows, one per `SourcedValue` (29 before `evolved_isolate_parent` was
added), are omitted from the table.

| dataset | level | row | verdict |
|---|---|---|---|
| tolerance | L0 | `structural` | ok, 3 records validated |
| tolerance | L1 | `count` | ok, observed 3, expected 3 |
| tolerance | L1 | `pair_uniqueness` | ok, 3 unique (study, strain, condition) |
| tolerance | L1 | `provenance_gaps` | ok, 12 gaps over 3/3 records, 0 deferred |
| tolerance | L1 | `canonical_gene_names` | ok, 12 systematic names, each current |
| tolerance | L2 | `value_fidelity` | ok, 3 values |
| tolerance | L2 | `se_nonnegative` | ok, 0 values |
| tolerance | L2 | `uncertainty_sanity` | ok, 0 labeled uncertainties |
| tolerance | L3 | `measurement_type_consistent` | ok, `log2_ratio` |
| tolerance | L3 | `reference_zero` | ok, reference response == 0 for all 3 |
| tolerance | L3 | `environment_perturbed` | ok, all 3 carry an environmental edit |
| tolerance | L3 | `compound_identity` | ok, 3 compound references |
| tolerance | L3 | `media_compound_identity` | ok, 21 medium components |
| tolerance | L3 | `media_membership` | ok, 3 records on one MEDIA_LIBRARY medium |
| tolerance | L4 | `site_identifiers_deleted_loci` | ok, 0 of 0 |
| tolerance | L4 | `gene_containment_kt2440_locus_tags` | ok, 12 of 12 |
| proteome | L0 | `structural` | ok, 3 records validated |
| proteome | L1 | `count` | ok, observed 3, expected 3 |
| proteome | L1 | `orf_uniqueness` | ok, 24 ORFs, 15 with multiple strains (expected) |
| proteome | L2 | `value_fidelity` | ok, 7,054 values |
| proteome | L2 | `se_nonnegative` | ok, 7,054 values |
| proteome | L3 | `reference_finite` | ok, finite + key-matched for all 7,054 |
| proteome | L3 | `measurement_type_consistent` | ok, `dia_nn_top3_log2_mean` |
| proteome | L4 | `site_identifiers_deleted_loci` | ok, 4 of 4 |
| proteome | L4 | `gene_containment_kt2440_deleted_loci` | ok, 20 of 20 |
| proteome | L4 | `gene_containment_kt2440_quantified_loci` | ok, 2,361 of 2,361 |

Two verifier rows needed the loader to say something it had not had to say before.

- **L4 `site_identifiers_*` is new.** `protein_gene_set` collects every
  `systematic_gene_name` on a genotype, which now includes the 4 site ids
  (`AE015451:1386816`, `:1812462`, `:4586057`, `:4808180`). Asking the gene universe to
  contain one fails by design, since the whole point of the site-keyed leaf is that no
  locus tag holds the call. The identifiers are now partitioned: site ids get their own L4
  row (replicon and 1-based position checked) and only the locus tags go to the
  gene-containment row, which is why the proteome row reads 20 of 20 rather than 20 of 24.
- **`allow_duplicate_orfs=True` is now passed to the protein verifier.** Its
  `orf_uniqueness` rule means "one record per knocked-out ORF", which was never this
  dataset's shape: the three records are three STRAINS of one lineage, each carrying its
  parent IPL400's seven designed deletions by construction, and the two isolates
  additionally share the founder calls IPL400 already had. The flag is the verifier's own
  supported form for that case (Messner 2023 uses it); record identity here is the
  genotype as a whole, which L1 `count` plus each record's distinct `Genotype` carry.

### Two disagreements reported, not reconciled

- The Results say "In A10_F63_I1, three (gnuR, PP_4063 and frmA) out of the total nine
  mutations, and in A12_F53_I1, four (gnuR, PP_4063, PP_3024 and frmA) out of the total
  nine mutations". Measured against the IPL400 founder column, A10 F63 I1 carries 8 call
  positions its founder does not and A12 F53 I1 carries 10 (of which two are the same
  `PP_3415` gene, giving 9 distinct genes). Neither reproduces the stated nine exactly.
  The loader stores the matrix's own rows and reports the difference.
- The 159-rows-against-158-unique-mutations disagreement from the 2026.10.07 section is
  unchanged.

### The `called_variants.json` artifact changed meaning

`read_variant_calls` no longer writes `blocking_reasons`. Each of the 159 rows now carries
`representation` (the leaf its shape maps to, one of the three `perturbation_type`
literals, so the recorded mapping cannot drift from the class written) and
`locus_seen_twice_in_a_clone` (measured: 8 rows, which the perturbation axis admits and
the background axis does not). The `BLOCK_*` constants are gone, because every one of
them asserted something that is no longer true. `preprocess/called_variant_perturbations.json`
is new and holds the founder check, the called-variant locus reconciliation and the
per-record accounting table above.

### The tests this changed, and how they were rewritten

`tests/torchcell/datasets/pputida/test_lim2025.py` asserted the old conclusion in five
places, each of which was replaced by the test of the new behavior at the same strength
rather than loosened:

- the `blocking_reasons` tests, which referenced the deleted `BLOCK_*` constants, now
  assert the `representation` each of the 159 rows carries and that it is one of the
  three `perturbation_type` literals, so the recorded mapping cannot drift from the class
  written;
- `test_the_genotype_gaps_name_the_parents_unwritable_content` now asserts the entry the
  leaves CLOSED is gone and the `PP_2676` entry is still there;
- the `strain_genotype("A10_F63_I1", {})` case, which asserted a `ValueError`, now
  asserts the genotype an evolved isolate gets;
- the build-artifact test now asserts the artifact's new shape;
- the proteome record count is 3 and the renamed drop rule is asserted by its new name.

`pytest tests/torchcell/datasets/pputida/test_lim2025.py` and its de Siqueira sibling run
200 passed, 11 skipped; the anti-padding lint reports 485 files clean.

## 2026.10.09 - The seven proteome sheets stay refused after #770, because every released contrast has an evolved isolate on one side

Issue #770 names Lim 2025's proteome sheets as one of five places a protein-level fold
change plus a p-value is released with no phenotype class to hold it.
`ProteinFoldChangePhenotype` landed in this wave and holds the numbers. The sheets are
still NOT loaded, and the reason is the GENOTYPE axis, not the phenotype one.

Measured by
[[protein_fold_change_refusals_kang_lim|experiments.036-dataset-fixes-before-kg-build.scripts.protein_fold_change_refusals_kang_lim]]
over `si/si2.xlsx` (sha256
`a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612`). Every one of the
seven proteome sheets carries exactly one `log2_Fold_change_A/B` column, and the two arms
it is built from are:

| sheet | rows | arm A | arm B |
|---|---|---|---|
| `Proteome_A10F63I1vsIPL400_M9G` | 2,367 | `log2_mean_A10_F63_I1_M9G` | `log2_mean_IPL400_M9G` |
| `Proteome_A10F63I1vsIPL400_G+4IP` | 2,374 | `log2_mean_A10F63I1_M9G+4IP` | `log2_mean_IPL400_M9G+4IP` |
| `Proteome_A12F53I1vsIPL400_M9G` | 2,367 | `log2_mean_A12_F53_I1_M9G` | `log2_mean_IPL400_M9G` |
| `Proteome_A12F53I1vsIPL400_G+4IP` | 2,338 | `log2_mean_A12F53I1_M9G+4IP` | `log2_mean_IPL400_M9G+4IP` |
| `IPL400vsA10F63I1_pIY670_M9G_12h` | 2,350 | `log2_mean_IPL400_pIY670_M9G_12hr` | `log2_mean_A10F63I1_pIY670_M9G_12hr` |
| `IPL400vsA10F63I1_pIY670_M9G_24h` | 2,350 | `log2_mean_IPL400_pIY670_M9G_24hr` | `log2_mean_A10F63I1_pIY670_M9G_24hr` |
| `IPL400vsA10F63I1_pIY670_M9G_48h` | 2,378 | `log2_mean_IPL400_pIY670_M9G_48hr` | `log2_mean_A10F63I1_pIY670_M9G_48hr` |

`A10F63I1` and `A12F53I1` are evolved isolates, and one of them is an arm of all seven
released fold changes. Writing a record for a fold change means writing the genotype on
each side of it, and an evolved clone's genotype cannot be written with the existing
classes (#731, open: "Bacterial sequence-variant representation: evolved clones cannot be
written with existing classes"). Three of the seven are additionally blocked on the
`pIY670` arm's medium, which `MEDIA_LIBRARY` has no entry for. Note also the direction
flip, which any later loader must handle: in the four `Proteome_*` sheets arm A is the
evolved isolate, and in the three `IPL400vs*` sheets arm A is the parent, so the sign of
`log2_Fold_change_A/B` means the opposite thing between the two families.

So the new phenotype class is NOT what was blocking this paper, and the honest statement is
that #770 closes none of its rows. The existing `ProteomeLim2025Dataset` keeps its one
absolute-abundance record and its refusals stand as written. When #731 lands, the four
`Proteome_*` sheets become four fold-change records (two isolates, two media) with their
`p-value` and `p_adjusted(BH)` columns, which is what `ProteinFoldChangePhenotype` was
shaped for.

## 2026.10.09 - Correction: #731 landed, so the four isolate-vs-parent sheets are no longer refused

The section above was written while #731 was open and says in so many words that an
evolved clone's genotype cannot be written. #731 landed on main the same day
(`e1acbb28b`, `9db9b2fcb`), and `ProteomeLim2025Dataset` now stores three records whose
two evolved isolates carry their called variants on top of IPL400's designed deletions.
The refusal above therefore no longer holds, and this correction supersedes it rather than
being merged into it.

What that changes, and what it does not, measured on `si/si2.xlsx` (sha256
`a3cfd6014cc611b206af9c7e7c8770c23189ae96986d23981da0b11236721612`) by the same script:

| sheet family | sheets | rows each | status after #731 |
|---|---|---|---|
| `Proteome_<isolate>vsIPL400_<medium>` | 4 | 2,338 to 2,374 | **unblocked**: both arms are strains the loader already writes, in M9G and M9G+4IP, both media it already serves |
| `IPL400vsA10F63I1_pIY670_M9G_<t>` | 3 | 2,350 to 2,378 | still blocked: the `pIY670` production arm's medium has no `MEDIA_LIBRARY` entry (20 g/L glucose with kanamycin) |

Two things a loader for the four must still settle, and neither is settled here:

1. **The direction flips between the families.** In `Proteome_*` arm A is the evolved
   isolate; in `IPL400vs*` arm A is the parent. `log2_Fold_change_A/B` means the opposite
   thing in the two, so `reference_basis` has to be read per sheet from its own arm
   labels, never from the family name.
2. **A strain-versus-strain contrast has no reference genotype slot.**
   `BacterialProteinFoldChangeExperimentReference` carries `genome_reference` and
   `phenotype_reference` and NO genotype, because the class was shaped for a perturbation
   against the reference strain. The denominator here is IPL400, which is itself a
   deletion strain, so writing these four records either puts both genotypes in the
   experiment's `Genotype | list[Genotype]` field (which the class permits) or adds a
   genotype to the reference class (which is another schema change).

There IS a precedent for the second point inside this same wave, which is why it is a
decision and not a blocker: `ProteomeFoldChangeCarruthers2025Dataset` writes a
strain-versus-strain contrast by putting the numerator strain in the experiment's
`genotype` and naming the denominator strain in the phenotype's `reference_basis`, with
the reference carrying only `genome_reference` and `phenotype_reference`. The same shape
would write Lim's four. The reason to ask first is that Carruthers' denominator is a
non-targeting CRISPRi control, one step from the reference strain, while Lim's is IPL400,
a seven-deletion production strain, so prose in `reference_basis` is carrying much more
of the genotype there.

Flagged for the owner: this is a loader plus a representation decision, not a pin, so
#770's Lim rows are left open rather than closed in the same pass that corrected the
reason they were refused.

## 2026.10.09 - The four isolate-over-IPL400 contrasts are loaded, and the pIY670 three are refused for the genotype rather than the medium

The correction above left two things for the owner: how a strain-versus-strain contrast
states its denominator, and whether the pIY670 arm's medium is a real blocker. Both are
settled here, and the four `Proteome_*` sheets are loaded by a third dataset class,
`ProteomeFoldChangeLim2025Dataset` (root `data/torchcell/proteome_fold_change_lim2025`).

### The denominator is a stated genotype, not a sentence

The question was whether `reference_basis` plus the existing reference-experiment
machinery can say "denominator = genotype X" with X written out. It can, and in two places
at once:

- **typed**: each record's `genome_reference` carries IPL400's own
  `BacterialStrainBackground` -- the seven `full_deletion` alleles plus `PP_2676` as a
  `partial_deletion`, each with a typed `deleted_span` gap, with Supplementary Table 1's
  genotype string as the background's `genotype_statement`. This is the SAME object
  `ProteomeLim2025Dataset` already builds, and here it is EXACT rather than a floor: the
  denominator arm of these four contrasts literally is IPL400, while an absolute
  evolved-isolate record's reference arm is that isolate unstressed, whose called variants
  no background allele can carry.
- **named, in the source's own terms**: `reference_basis` names IPL400, quotes that
  genotype string, and names the arm it came from, e.g.

  > IPL400, the parent starting strain, in the same medium and the same export: the 'log2_mean_IPL400_M9G' arm of Supplementary Data 1's Proteome_A10F63I1vsIPL400_M9G sheet. Its genotype is 'KT2440 ΔPP_2675 Δ14-PP_2676 ΔPP_3839 ΔPP_4064-∆PP_4067 ΔttgB (PP_1385)' (Supplementary Table 1), carried typed on this record's genome_reference as a BacterialStrainBackground of seven full_deletion alleles plus PP_2676 as a partial_deletion

  The sheet name is in there because two sheets carry the same denominator column
  (`log2_mean_IPL400_M9G`), so the (column, sheet) pair is what locates a stored record
  back in the workbook; the L2 verifier uses exactly that to re-read each record's bytes.

The numerator rides the experiment's `genotype` as IPL400's seven designed deletions plus
that isolate's own called variants, which is the shape
`ProteomeFoldChangeCarruthers2025Dataset` already uses (its numerator genotype carries the
pathway its non-targeting control also has). Nothing goes into `list[Genotype]`: order
would be the only signal of which side is which, and the background says it with a type.

Both arms of a contrast sit in ONE condition, so `environment_reference` is the record's
own environment. That is the opposite of the absolute family, whose reference arm is the
unstressed condition, and it is checked (`both_arms_of_the_contrast_share_one_condition`).

### The four records, measured

| sheet | numerator | condition | protein keys |
|---|---|---|---|
| `Proteome_A10F63I1vsIPL400_M9G` | A10_F63_I1 | M9 + 4 g/L glucose | 2,361 |
| `Proteome_A10F63I1vsIPL400_G+4IP` | A10_F63_I1 | + 4 g/L isoprenol | **2,365** |
| `Proteome_A12F53I1vsIPL400_M9G` | A12_F53_I1 | M9 + 4 g/L glucose | 2,361 |
| `Proteome_A12F53I1vsIPL400_G+4IP` | A12_F53_I1 | + 4 g/L isoprenol | 2,332 |

Each phenotype carries the released `log2_Fold_change_A/B` on the `log2` scale, its
`p-value`, its `p_adjusted(BH)` under `p_value_adjustment_method="benjamini_hochberg"`,
`n_replicates = 3` per key, and `measurement_type =
dia_nn_top3_log2_fold_change_welch_two_sample_t_test`. The reference phenotype is the
scale's neutral value per key, so experiment over reference reproduces the released number
and nothing is imputed.

`protein_fold_change_se` is `sqrt((sd_A^2 + sd_B^2) / 3)`. That is not a free derivation:
it is the exact denominator the released `t-test_stat` divides by, and the build asserts
that identity for every stored row with the module's existing
`assert_sheet_statistics` (worst residual below the 1e-6 tolerance on all four sheets, as
the earlier back-solve measured at 1.1e-12).

### The direction is read off each sheet, never off its name

`assert_fold_change_direction` asserts, per sheet, that the arm columns the module
declares are the ones the header carries AND that the numerator's mean column comes
BEFORE the denominator's, which is what `A/B` means. Measured on the pinned workbook
(sha256 `a3cfd601...`): all four loaded sheets put `log2_mean_A1*` first and
`log2_mean_IPL400*` second, and `IPL400vsA10F63I1_pIY670_M9G_12h` puts
`log2_mean_IPL400_pIY670_M9G_12hr` first. A loader that reused the loaded shape on those
three would store an inverted sign on every key.

### Why the pIY670 three stay refused, and the correction to the stated reason

The earlier section said they were blocked on the medium. **They are not: the Methods
state the production culture verbatim** (`paper.md`, sha256 `26b88d81...`):

> test tubes containing $5 ~ \mathrm { m L }$ NREL M9 minimal medium as described in Section 2.2 with $2 0 g / \mathrm { L }$ glucose as carbon source and $5 0 ~ \mu \mathrm { g / m L }$ kanamycin. The isoprenol pathway was induced by adding arabinose at ${ 2 } \ g / \mathrm { L }$ at $^ { 0 \mathrm { h } }$ .

So 20 g/L glucose, 50 ug/mL kanamycin and 2 g/L arabinose at 0 h are sourced values, not
gaps; `si1.docx` does not state them, which is why the first pass read the gap there. The
operative refusal is the GENOTYPE, and it is the direction flip that creates it: those
sheets put the evolved isolate in the DENOMINATOR, so a record would have to state an
evolved clone's genomic content on the strain-BACKGROUND axis, and a called variant is no
`BacterialBackgroundAllele` (`BacterialStrainBackground` also permits one allele per
locus, which refuses A12_F53_I1's two `PP_3415` calls). The medium remains a second,
weaker obstacle: `MEDIA_LIBRARY` carries no M9 at 20 g/L glucose with kanamycin, which is
a value-surface addition rather than a provenance gap. Both reasons are stored in
`FOLD_CHANGE_REFUSALS` and in `preprocess/dropped_records.json`.

### Three protein keys dropped that the absolute family never meets

| key | status on `pputida_KT2440_ASM756v2` |
|---|---|
| `PP_0985` | `non_gene_feature` |
| `PP_2271` | `retired` |
| `PP_5287` | `retired` |

All three sit on `Proteome_A10F63I1vsIPL400_G+4IP` alone, and all three are among the
seven loci released under isoprenol but absent from the unstressed arm. The absolute
family never meets them because its key set is the INTERSECTION of each strain's two arms;
a contrast's key set is one sheet's own rows. A key with no current gene has no gene node
to key its ratio to, so it is dropped and ledgered. The other four of those seven ARE
stored here, which is why the A10 isoprenol record carries 2,365 keys rather than 2,361.

The six paralogous keys (`Ubid`, `Pyrc`, `Dapa` under two loci each) are dropped for the
same measured reason as in the absolute family: one protein group's statistics cannot be
attributed to either paralog, and those rows are exactly the ones whose released t does
not reproduce at n = 3.

### L0 to L4 on the built store, PASS

Built with `build_dataset_lmdb --dataset ProteomeFoldChangeLim2025Dataset` (4 records,
gene_set 28, 4 references), verified by `lim2025.run_fold_change_verification`, which is
also registered as `proteome_fold_change_lim2025` in
`runners.BACTERIAL_PROTEIN_FOLD_CHANGE_DATASETS`.

| level | check | result |
|---|---|---|
| L0 | `structural` | 4 records validated |
| L1 | `count` | observed 4, expected 4 |
| L1 | `contrast_uniqueness` | 4 distinct contrasts, one record each |
| L2 | `value_fidelity` | 9,419 values checked |
| L2 | `p_values_are_probabilities` | all 9,419 p-values in (0, 1] |
| L2 | `fold_change_equals_arm_a_minus_arm_b` | 9,419 of 9,419, re-read from the workbook |
| L3 | `reference_is_the_scales_neutral_value` | all 9,419 reference values neutral and key-matched |
| L3 | `fold_change_scale_consistent` | single scale `log2` |
| L3 | `measurement_type_consistent` | single measurement type |
| L3 | `reference_genome_is_the_named_denominator_strain` | 4 of 4, 8 alleles each |
| L3 | `both_arms_of_the_contrast_share_one_condition` | 4 of 4 |
| L4 | `gene_containment_kt2440_tested_and_perturbed_loci` | 2,373 of 2,373 |
| L3 | `provenance_audit` | every sourced value's quote re-read from its pinned bytes |

Schema-impact verdict: `scripts/schema_impact_check.py --base origin/main` reports **no
schema contract changes**. `ProteinFoldChangePhenotype`,
`BacterialProteinFoldChangeExperiment` and its reference class all landed with #770; this
branch only adds a loader on them. The adapter gate's four pin places were re-derived from
the merged files (`kg_bacteria.yaml` position, `_bacterial_adapter_cases.py` order, the
two `test_bacterial_adapters.py` addends, the `dataset_adapter_map` total and the
`BACTERIAL_DATASETS` set literal), and the new dataset ships its adapter module, its conf
and its adapter test.

### What is still refused after this, with the measurement

- the three `IPL400vsA10F63I1_pIY670_M9G_<t>` sheets (above): the denominator arm is an
  evolved clone, 2,350 / 2,350 / 2,378 released rows each.
- the 16 LINEAGE final growth rates and the 46 isolates as tolerance records: unchanged,
  and for the reasons the first section states.
- `PP_2676`'s truncation on the perturbation axis: unchanged; it stays a
  `BacterialBackgroundAllele(partial_deletion)` and a stated gap, which the fold-change
  records inherit through IPL400's background.

Related: [[torchcell.datasets.pputida.carruthers2025]] (the strain-versus-strain
precedent), [[torchcell.datamodels.bacterial-perturbation-ontology]].
