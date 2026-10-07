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
neighbour is exactly what must not be done. A multi-locus row is a large deletion (up to
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

`isoprenol` still has no row in `compound_identity_table.json`, so `resolved_compound`
returns the name with an `inchikey` gap. Its identity, `ISOPRENOL_INCHIKEY` =
`CPJRRXSHAYUTGL-UHFFFAOYSA-N` (SMILES `C=C(C)CCO`, PubChem CID 12988) is recorded in the
module and checked against any row the table gains; curating that row is a separate,
human change. This is the same object the Carruthers 2025 and Wang 2015 loaders store.

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
