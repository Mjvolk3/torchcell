---
id: h1xo7sxur1rd97bufeemygu
title: Desiqueira2025
desc: ''
updated: 1791392336843
created: 1791392336843
---

## 2026.10.07 - de Siqueira 2025: two loaders, and the genotype that stopped five strains

`torchcell/datasets/pputida/desiqueira2025.py` serves row 14 of the bacterial expansion
list: de Siqueira et al. 2025, "Alternate routes to acetate tolerance lead to varied
isoprenol production from mixed carbon sources in Pseudomonas putida"
(doi:10.1128/aem.02123-24, `citation_key` `desiqueiraAlternateRoutesAcetate2025`,
`paper.md` sha256 `a3ea14adcbe77144f02fb92ae6b344e4dec637a4211c702aa29fa121eda2de9c`).
Templates: [[torchcell.datasets.pputida.carruthers2025]] and
[[torchcell.datasets.pputida.lim2022]]; skeleton
[[torchcell.datasets.bacteria_common]]; schema layer
[[torchcell.datamodels.bacterial-perturbation-ontology]]; media
[[torchcell.datamodels.media]].

### The two dataset classes

| class | experiment class | records | what one record is |
|---|---|---|---|
| `ProteomeDeSiqueira2025Dataset` | `BacterialProteinAbundanceExperiment` | 5 | one released Data Set S1 Top3 sample of a writable strain |
| `IsoprenolTiterDeSiqueira2025Dataset` | `ProductTiterExperiment` | 2 | one Table S2 maximum isoprenol titer of a writable strain in a medium the library holds |

### The genotype finding

This study evolves KT2440 on acetate and resequences the clones, so an evolved clone's
genotype IS its parent plus the variants Geneious called. Data Set S2 releases **173
calls over 5 sequenced clones** (PT 33, Sigma1 34, Sigma2 28, Sigma4 43, Sigma5 35) at
**83 distinct sites**, 33 of them shared by more than one clone. **Sigma3 was never
sequenced at all.** Measured on the pinned workbook, no class in `schema.py` can hold
one of those calls honestly:

1. `SequenceVariantPerturbation` is the only variant-level perturbation leaf, and its
   base validator admits only S288C ORF names, so a `PP_` tag is refused. It also
   requires `strain_id` plus an off-graph sequence pointer; this paper released SRA
   reads (PRJNA1153078), never per-strain allele sequences.
2. `BacterialBackgroundAllele(edit=sequence_variant)` takes the tag but fails on four
   counts:
   - `functional: bool` is required and not optional, so the unknown functional status
     of a missense SNP cannot be covered by a `ProvenanceGap`, which must name a field
     that is `None`;
   - it has no slot for the five things the released table gives per call (position,
     reference and alternate base, amino-acid change, polymorphism type, variant
     frequency), so they would be lost or crammed into `allele_name`;
   - its container `BacterialStrainBackground` permits ONE allele entry per locus,
     which refuses Sigma1 (`PP_1656` x2, `PP_3827` x3), Sigma2 (`PP_1656` x2) and
     Sigma4 (`PP_1656` x2, `PP_3827` x2);
   - **105 of 173 calls carry no locus at all** and 10 more carry only a RefSeq `PP_RS`
     tag with no GenBank `old_locus_tag`, so `systematic_gene_name` cannot be filled.
     Inventing a neighboring locus for an intergenic call is exactly what must not be
     done, so those calls get a typed gap instead.
3. If a Sigma clone were written anyway, `Genotype.__eq__` compares the perturbation
   SET, so every Sigma record would be genotype-identical to the PT record it descends
   from and five distinct strains would collapse onto one identity.

So the five tolerized isolates are not loaded. Every one of the 173 calls is typed into
`preprocess/called_variants.json` by `read_variant_calls`, each with its own
`blocking_reasons`, so the additive proposal below rests on the real rows.

### The additive proposal that would make the tolerized isolates loadable

Exactly as it appears in the PR body. Three additions, no edit to an existing class, so
no served dataset's closure moves:

1. **`BacterialSequenceVariantPerturbation(SequencePerturbation)`**, the bacterial
   sibling of `SequenceVariantPerturbation`:
   `perturbation_type: Literal["bacterial_sequence_variant"]`, `provenance="natural"`,
   `mechanism_so_id="SO:0001483"` (`SNV`) with an `indel` variant for a length change,
   plus `gene_namespace: BacterialGeneNamespace`, the locus-tag validator the other
   bacterial leaves use, and the released call fields: `chromosome`, `position` (1-based
   on the pinned assembly), `reference_allele`, `alternate_allele`,
   `polymorphism_type`, `amino_acid_change: str | None`, `codon_number: int | None`,
   `variant_frequency: str | None` (released as a range for a multi-base call),
   `read_depth: str | None`, `caller: str | None`. `functional` is deliberately absent:
   a called variant does not state a functional consequence.
2. **An intergenic carrier.** `GenePerturbation` requires a gene, and 105 of the calls
   have none, so an intergenic call needs either a `systematic_gene_name=None` form or
   a sibling `IntergenicVariant` record hung off the strain background with
   `nearest_gene` + `offset_bp` left `None` and a typed `ProvenanceGap` when the source
   does not give them. The second is preferred: it keeps `GenePerturbation`'s invariant
   that a perturbation names a gene.
3. **`BacterialStrainBackground` must admit several entries at one locus**, or the
   variant leaves must live on the genotype rather than the background. Today's "a
   haploid background carries one allele entry per locus" is right for designed alleles
   and wrong for called variants, three of which stack at `PP_1656`.

Until all three land, the honest record set is the two writable strains.

### What IS writable, and how PT is typed

- **WT** is P. putida KT2440 itself ("`<td>WT</td><td>P. putida KT2440</td><td>Wild type
  strain</td><td>ATCC 47054 (JBEI-18711)</td>`"), so its records carry
  `assembly_reference("KT2440")` with no background and an empty `Genotype`.
- **PT** is a defined-deletion strain: "A mutant strain of Pseudomonas putida KT2440
  (DeltaPP_2675 Delta14-PP_2676, referred to as "pre-tolerized" or "PT" throughout this
  manuscript) was used as the base strain for the tolerization described in this work."

PT's two designations are **one deletion event**. The Results explain the renaming:
"This updated nomenclature accounts for the PP_2676 ORF overlapping with the coding
sequence of PP_2675 ... with its predicted start codon residing within the CDS of
PP_2675 (45)". The deferral for the PP_2675 half goes to Thompson 2020 (reference 30,
mirrored, `paper.md` sha256
`389d0d6cd196f9f159d0485f6b8ed09743dfbe3b6682369db489d7356a469b45`), whose Table 1 reads
"Strain with complete internal in-frame deletion of PP_2675". So:

| designation | where it is typed | class | why |
|---|---|---|---|
| `DeltaPP_2675` | `Genotype.perturbations` AND the PT background | `BacterialDeletionPerturbation` + `BacterialBackgroundAllele(full_deletion)` | the perturbation is the ML-facing edit and the gene node a record links to; the allele is part of the background's own definition |
| `Delta14-PP_2676` | the PT background only | `BacterialBackgroundAllele(partial_deletion)` | the bacterial perturbation axis has NO partial-deletion leaf (`BacterialDeletionPerturbation` is `state="absent"`), so a perturbation there would overstate the edit |

`PP_2675` therefore appears twice, which is deliberate:
`BacterialStrainBackground.alleles` is documented as "every allele of the background", so
listing one of the two would be an omission. The `PP_2676` allele carries a
`ProvenanceGap` on `deleted_span`: the source writes "Delta14" without saying whether 14
counts base pairs or codons and without coordinates, and the deferral target names only
the PP_2675 half. `functional=False` on both rests on the Abstract's "its native
isoprenol catabolism pathway is deleted" plus the Delta designation; the source never
states PP_2676's functional status on its own, and the `SourcedValue.note` says so.

**PT's own 33 called variants are a stated gap, not a `ProvenanceGap`.** They are
missing ROWS, not missing fields (the same reading the Carruthers chassis span uses for
its 54 unnamed loci), so they live in `preprocess/called_variants.json` and in the build
accounting.

### Family 1: the proteome

Source: Data Set S1 (`aem.02123-24-s0001.xlsx`, sha256
`6bd1889df343646fdcd972fb6b7d3c6c014fcd428a4a78d8363908cccfbddeae`), one sheet, 34,600
cells = 1,730 protein groups x 20 samples. 20 and not 21 because PT was never sampled in
the mixed-carbon condition, which is consistent with the paper's "often fails to grow in
glucose-acetate medium".

- **Column consumed:** `Top_3pep_counts_rep_mean`, with `Top_3pep_counts_rep_std`.
  `measurement_type` is `dia_nn_top3_peptide_signal_replicate_mean`.
- **`n_samples` and the uncertainty type, with the quote.** "Each of these initial cell
  suspensions was used to inoculate three cultures of either M9 acetate, M9 glucose, or
  M9 glucose-acetate at an initial $\mathsf { O D } _ { 6 0 0 } \ : 0 . 1 5$ in $5 ~
  \mathrm { m L }$ cultures." So **n = 3 biological replicates** and the released `_std`
  is a **sample SD** over them. Stored SE = SD / sqrt(3). The released
  `%_of protein_abundance_Top3_rep_mean_sem` column is NOT used: it is constant across
  every sample of a protein, so it is not that sample's SE.
- **Reference:** the wild type in M9 glucose, per the Fig. 4 caption "For both panels,
  protein levels in the glucose as a sole carbon source medium were used as the
  reference condition." The baseline is COPIED into each record's reference and the WT
  glucose sample stays a record of its own, because a measured sample is not spent by
  being used as a baseline. `genome_reference` is the RECORD's own strain, so a PT
  record carries the PT background and a WT record carries none.
- **Environment:** `M9_NREL_DESIQUEIRA2025` plus the carbon regime as
  `EnvironmentPhysicalPerturbation(factor=carbon_source)`, which is what that media
  object's own provenance note says the loader must do. Acetate 50 mM and glucose 1%
  w/v come from the Methods' sole-carbon default; the mixture is glucose 2% + acetate
  0.65% w/v from the Fig. 1 caption, since the proteomics section names the medium
  without restating its concentrations. `duration_hours` is `None` with a gap: the
  cultures were harvested at a growth STATE ("Once cultures reached exponential growth
  (OD600 of 0.6-0.8)"), not a clock time.

#### Identifier histogram

`reconcile_locus_tags` over the 1,729 distinct `Protein` keys, against
`pputida_KT2440_ASM756v2`:

| status | n |
|---|---|
| `current` | 804 |
| `renamed` | 729 |
| `non_gene_feature` | 1 |
| `retired` (kept as given) | 194 |
| `ambiguous` (kept as given) | 1 |

| resolver layer | n |
|---|---|
| locus tag | 805 |
| old locus tag | 0 |
| RefSeq locus tag | 0 |
| gene symbol | 730 |
| gene synonym | 0 |
| not found | 194 |

`resolved_fraction` 0.8872, so `MIN_RESOLVED_FRACTION = 0.88`. 1,532 names were
remapped; 2 were kept on a collision (`Aspc` and `Pp_3786` both reach `PP_3786`) and 1
is ambiguous (`Asd` -> `PP_1989` or `PP_1992`).

#### Dropped protein keys: 198 of 1,729, leaving 1,531

- **197 outside the namespace.** 192 are PSEPK proteins whose title-cased UniProt gene
  symbol the GenBank annotation of this assembly carries no symbol for, and 5 are the
  contaminants the DIA-NN database was built to include ("The databases used in the
  DIA-NN search (library-free mode) were $P .$ putida KT2440 latest Uniprot proteome
  FASTA sequences (generated in March 2024) and common proteomic contaminants"): 4 human
  keratins (`Krt1`, `Krt2`, `Krt9`, `Krt10`) and pig trypsin (`Tryp_pig`). Measured
  organism mnemonics over the released cells: PSEPK 34,500, HUMAN 80, PIG 20.
- **1 merged key.** `Pyrc` is filed under two protein groups (`Q88D29` and `Q88NW7`), so
  the key names two proteins and its abundance cannot be attributed to either. It
  appears twice per sample in the release.

Every dropped key is listed in `preprocess/dropped_protein_keys.csv`. **The 192 host
proteins are a real loss** and the fix is a UniProt-accession-to-locus-tag crosswalk in
the genomes tier; the deposited GenBank file carries `/protein_id` but no UniProt
`db_xref`, so there is no mirrored crosswalk to use today. Raised in the PR.

### Family 2: the isoprenol titers

Source: Table S2 of the Supplemental Material, read from the mirrored OCR
(`si/si3.md`, sha256 `10c875184be296016ba9416bf9fda2e817a2f80dc2c5eccd4a9f2ba4c57981fc`)
and checked by eye against `si/si3.pdf` page 3. The values are hardcoded `SourcedValue`s
quoting those bytes, not parsed at build time.

**Units.** Table S2 releases millimolar and `ConcentrationUnit` has `millimolar`, so the
number is stored verbatim with no arithmetic. (The Carruthers loader's `mg/L`-as-`ug/mL`
workaround is not needed here, and `ConcentrationUnit` still has no `mg_per_l` member.)

**The statistic is a MAXIMUM.** "Table S2. Maximum isoprenol titers (mM) detected for
Sigma-class and PT strains grown in different growth media." It is taken over the 24, 48
and 72 h samples, so it is an upward-biased order statistic, not a single-time-point
titer. `Environment.duration_hours` is `None` with a gap for exactly that reason, and a
supplementary L3 row names the statistic in the verification report.

**`n_samples` = 3, no uncertainty.** "Data points shown represent three independent
biological replicates, and the error bars indicate standard deviation from the mean"
(Fig. 3 caption) gives the replicate design; Table S2 releases no uncertainty for the
maximum, and the per-time-point means with SD were never released as a table, so
`titer_uncertainty` and `titer_uncertainty_type` are both `None` with typed gaps.

#### Table S2 census: 24 cells

| bucket | n |
|---|---|
| loaded (PT, M9 glucose and M9 glucose+acetate) | 2 |
| `n.d.` (not determined; Sigma2, Sigma4, Sigma5 in both acetate columns) | 6 |
| unwritable strain (every numeric Sigma cell) | 14 |
| medium not in `MEDIA_LIBRARY` (PT in both acetate columns) | 2 |

The two acetate columns are Fig. S3's media A and C: "Medium A contained 75mM acetate
and $7 5 . 5 ~ \mathsf { m M }$ ammonium sulfate. Medium C contained 100mM acetate and
1mM of ammonium sulfate." Both differ from the 2 g/L (15.1 mM) ammonium sulfate the
served `M9_NREL_DESIQUEIRA2025` states, so they are a different medium and the library
holds no object for them. Both PT cells read 0, so no non-zero measurement is lost. The
two `media.py` entries are proposed in the PR.

#### Two cross-source assertions, both enforced at build time

1. Table S2's 3.29 mM for PT in M9 glucose equals the Results text's "In M9 glucose
   medium, the base PT strain produced $2 8 3 \pm 2 6 \mathrm { { \ m g / L } }$
   isoprenol by the $4 8 \mathrm { - h }$ time point": 283 mg/L over isoprenol's 86.13
   g/mol is 3.2858 mM, so **|diff| = 0.0043 mM**, inside Table S2's own two-decimal
   rounding. The molar mass is used only for this check and never to produce a stored
   number.
2. The Table 1 pIY670 part string
   (`pRK2-Kan-araC-PBAD-MvaSef-MvaEef-TrpoH-Ptrc1-O-MKmm-PMDHKQ-AphA`) agrees token for
   token, case-folded, with the Carruthers 2025 Supplementary Data 3 description of the
   same plasmid, **except that this paper's OCR dropped `Sc` from `PMDScHKQ`**. That is
   what licenses reusing the Carruthers-sourced `source_organism` for the five
   heterologous genes, which this paper never states, and why the stored identifiers are
   the workbook spellings rather than the OCR spellings: an OCR case loss must not fork
   a gene node. Note the OCR also merged Table 1's pIY670 and pTE452 rows, so the part
   string sits in pTE452's genotype cell while pIY670's is empty.

#### One cross-source disagreement, kept

For PT in the mixed feed the Results say "In this production regime, the baseline PT
strain failed to grow and therefore did not produce any detectable isoprenol", while
Table S2 releases **0.05 mM** (about 4.3 mg/L) for that cell. The released number is
stored, because the table is the explicit numeric release and the text's claim is
qualitative, and the disagreement is a counted finding in `build_accounting.json` and a
named L3 row in the verification report. It is NOT resolved by preference.

### Compound identity

`resolved_compound("isoprenol")` returns the canonical name with a `ProvenanceGap` on
`inchikey`; `compound_identity_table.json` has no isoprenol row. The key is recorded in
the module as `ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL-UHFFFAOYSA-N"` and the gap is carried,
exactly as [[torchcell.datasets.pputida.carruthers2025]] carries it. `L-arabinose` and
`kanamycin` have the same open gap.

### Raw mirror

`$DATA_ROOT/torchcell-raw/desiqueiraAlternateRoutesAcetate2025/` holds the two consumed
workbooks, both retrieved from the PMC Article Datasets bucket (`PMC12016510.1`), which
is directly scriptable:

| path | role | bytes | sha256 |
|---|---|---|---|
| `si/aem.02123-24-s0001.xlsx` | `si_data` | 4,608,396 | `6bd1889df343646fdcd972fb6b7d3c6c014fcd428a4a78d8363908cccfbddeae` |
| `si/aem.02123-24-s0002.xlsx` | `si_data` | 22,508 | `7ba609170ec7233e09e0ffbbfdb59a190353180ad48b50c45ab60850cd713633` |

`deposit_raw_mirror` is idempotent by sha256 and refuses a differing file rather than
overwriting it; both files are verified before anything is written. `si_expected` names
what is deliberately NOT deposited and why: the Supplemental Material PDF (already
mirrored and OCR'd in torchcell-library), SRA BioProject PRJNA1153078 (raw reads, no
loader consumes them), PRIDE PXD055153 (raw DIA spectra, likewise), and the fact that
the per-time-point titers plotted in Fig. 3B and 3C were never released as a table.

### Build and verification

```bash
python -m torchcell.database.build_dataset_lmdb --dataset ProteomeDeSiqueira2025Dataset
python -m torchcell.database.build_dataset_lmdb --dataset IsoprenolTiterDeSiqueira2025Dataset
```

- `proteome_desiqueira2025`: 5 records, gene_set 1, 2 references; manifest fresh.
- `isoprenol_titer_desiqueira2025`: 2 records, gene_set 6, 1 reference; manifest fresh.

`verify_build(root, data_root, family=...)` is the module's own runner. The proteome
family runs the shared `verify_protein_dataset` with `allow_duplicate_orfs=True` (PT
appears once per condition by design) plus three supplementary rows; the titer family has
no shared verifier, so `titer_levels` builds its own battery. Both reports land in
`preprocess/verification_report.json` and both PASS L0 to L4:

| level | proteome | titer |
|---|---|---|
| L0 | structural: 5 records validated | structural: 2 records validated |
| L1 | count 5/5; orf_uniqueness; strain_condition_uniqueness (5 pairs) | count 2/2; strain_condition_uniqueness (2 pairs) |
| L2 | value_fidelity 7,655; se_nonnegative 7,655 | value_fidelity 2 |
| L3 | reference_finite; measurement_type_consistent; assembly_pin | titer_unit; stored_statistic_is_a_maximum; cross-source disagreement declared; assembly_pin; cross-source titer agrees |
| L4 | gene_containment_kt2440: 1,531 measured + 1 perturbed, 0 outside | gene_containment_kt2440: 1 perturbed, 0 outside |

### Open gaps

1. **The tolerized isolates** (5 strains, 15 proteome samples, 14 numeric Table S2
   cells) wait on the three additive schema changes above.
2. **192 host protein keys** are dropped for want of a UniProt-to-locus-tag crosswalk in
   the genomes tier.
3. **The two Fig. S3 acetate media** are absent from `MEDIA_LIBRARY`, which costs 2 PT
   titer cells (both zero) and would cost 4 Sigma cells once the isolates are loadable.
4. **Isoprenol has no compound-identity row**, so every titer record carries a gap on
   the product's `inchikey`.
5. **`ProductTiterExperiment.environment` is annotated `Environment`**, so the vessel,
   the 5 mL working volume and the OD600 0.2 inoculum are carried in the module's
   `CULTURE_FORMAT` rather than on the record.
6. **`ConcentrationUnit` still has no `mg_per_l`**; this loader did not need it, but the
   Carruthers loader still stores mg/L as the numerically identical `ug/mL`.

## 2026.10.08 - The two further released normalizations, and the measurement that justifies them

Data Set S1 releases 15 columns. The first build asserted the first NINE
(`header[: len(PROTEOME_HEADER)]` against a nine-name tuple) and read `row[7]` and
`row[8]`, so columns 9 to 14 were neither asserted nor read: two further mean + SD pairs
and two derived columns. `PROTEOME_HEADER` now names all 15 and every one of them reaches
a row field or a build oracle.

| released column | what it is | where it goes now |
|---|---|---|
| `Top_3pep_counts_rep_mean` / `_rep_std` | the Top3 peptide signal | `ProteomeDeSiqueira2025Dataset`, unchanged |
| `%_of protein_abundance_Top3_rep_mean` / `%_of protein_abundance_Top3-rep_std` | percent of the sample's total abundance | `ProteomePercentDeSiqueira2025Dataset`, 5 new records |
| `log10_%_abundance_rep_mean` / `log10_%_abundance_rep_std` | the mean of the three replicates' log10 percent | `ProteomeLog10PercentDeSiqueira2025Dataset`, 5 new records |
| `CV%_of_%_protein_abundance` | derived | build oracle, `assert_percent_cv_is_derived` |
| `%_of protein_abundance_Top3_rep_mean_sem` | one value per protein | build oracle, `assert_sem_is_constant_per_protein` |

### Why three dataset classes and not one with three measurement types

`torchcell/verification/protein.py`'s `_l3_measurement_type_consistent` is a shared level
every protein-abundance dataset runs: "L3: all records share a single measurement_type (no
silent cross-assay mixing)". Three normalizations in one dataset class would fail it, and
weakening a level every bacterial and yeast proteome dataset depends on to admit one
paper's extra columns would be the wrong trade. The module's own convention already splits
on the released family, so the split is per normalization. The supplementary level
`stored_scale_is_the_released_column_this_class_reads` was added beside it: the shared level
proves the records agree with each other, the new one proves they agree with the column the
class actually reads, which is what keeps three classes over one workbook from swapping
scales.

### The non-recoverability measurement, over all 34,600 released cells

Storing a second and third normalization of one proteome is only worth doing if neither can
be re-derived from the first. Measured over every released cell of the sha256-pinned
`si/si1.xlsx` (`6bd1889df343646fdcd972fb6b7d3c6c014fcd428a4a78d8363908cccfbddeae`), not a
sample of them. Pinned in
`tests/torchcell/datasets/pputida/test_desiqueira2025.py::test_the_two_new_normalizations_are_not_recoverable_from_the_stored_top3`.

**Finding 1, the percent column is not percent of the stored mean.** Against
`100 * top3_mean / sum(top3_mean)` within each of the 20 released samples:

| statistic | value |
|---|---|
| cells compared | 34,600 |
| cells agreeing exactly (1e-9 relative) | **0** |
| cells agreeing to 5e-4 relative | 9,322 |
| per-cell ratio, min | 0.914506 |
| per-cell ratio, median | 1.000007 |
| per-cell ratio, max | 1.083554 |

Each sample's released percents sum to 100.000 (one sample reads 100.00038), so this is not
a scaling error with a missing denominator. The ratio straddling 1 in both directions with
a median of 1.000007 is the signature of a replicate-wise mean of per-replicate
percentages, each replicate normalized by its OWN total: averaging after normalizing is not
the same as normalizing after averaging, and the gap is whichever direction a protein's
replicate-to-replicate total variation pushes it. There is no route back from a mean of
counts.

The audit note reported this as a per-sample ratio range of 0.959 to 0.997. That range is
narrower and one-sided; the per-cell range measured here is 0.914506 to 1.083554 with a
median at 1.000007. The conclusion is the same and stronger: zero of 34,600 cells agree.

**Finding 2, the log10 column is the mean of logs.** Against `log10(released percent)`:

| statistic | value |
|---|---|
| cells with pct > 0 | 34,600 |
| released log10 STRICTLY BELOW log10 of the percent (beyond 5e-5) | **33,914** |
| agreeing within 5e-5 | 686 |
| released log10 ABOVE log10 of the percent | **0** |
| largest gap, log10 units | 1.378950 |

Zero violations in 34,600 one-sided comparisons is Jensen's inequality for a mean of
logarithms, with equality only where the three replicates coincide. A log10-of-the-mean
column would agree on every cell. The first released row (protein `Csda`) reads
-2.23385590492606 against a log10 of the mean of -2.19547, which matches the audit exactly.

Its SD is not recoverable either: the delta-method transform of the percent SD,
`pct_sd / (pct_mean * ln 10)`, disagrees on 34,496 of 34,600 cells.

**The two derived columns, measured and therefore NOT stored.** The CV is exactly
`100 * pct_sd / pct_mean` on every one of the 34,600 cells, so it carries nothing the
stored percent pair does not. The SEM is single-valued for 1,728 of 1,729 proteins, so it
does not vary with the sample and a per-sample record has nothing to put in it. Both are
asserted at build time instead, which is what keeps every asserted header cell from being
parsed past a second time.

### Why the percent and log10 scales fit `ProteinAbundancePhenotype`'s docstring

The docstring asks for "absolute per-strain quantity on a log signal scale, NOT a ratio --
the WT/parent strain supplies the reference". A percent of the sample's own total is a
compositional quantity of that one strain, not a ratio against another strain, and its log10
is literally a log signal scale. So these two land inside the docstring, unlike the
ratio-to-control the Yunus 2026 loader stores, which the audit's gap R flags.

The log10 means are NEGATIVE (a percent below 1 has a negative log10). The schema permits
it: `protein_abundance` is `dict[str, float]` with no sign validator, only the SE is
required non-negative, and `l2_value_fidelity` for abundance applies no minimum. The
`measurement_type` is what tells a consumer the scale.

### Record counts and the stores

| class | root slug | records | protein keys per record |
|---|---|---|---|
| `ProteomeDeSiqueira2025Dataset` | `proteome_desiqueira2025` | 5 | 1,531 |
| `ProteomePercentDeSiqueira2025Dataset` | `proteome_percent_desiqueira2025` | 5 | 1,531 |
| `ProteomeLog10PercentDeSiqueira2025Dataset` | `proteome_log10_percent_desiqueira2025` | 5 | 1,531 |

Ten new records, 15,310 new values plus their SEs. The same five writable samples, the same
genotypes, the same environments, the same 1,531 surviving protein keys and the same
WT/glucose reference sample on each scale, so every difference between the three stores is
the released column each class reads. All three verified: L0 to L4 pass, 11 levels each.

The two blocked items are unchanged. The 192 host protein keys outside the namespace still
need a UniProt-to-locus-tag crosswalk (open gap 2), and the Sigma-class strains still need
the variant-level perturbation leaf (#731) which would take each class from 5 records to
20.

## 2026.10.09 - The four sequenced isolates are loaded: the #731 leaves applied

The first build of this loader measured that no class in `schema.py` could hold one of
Data Set S2's 173 Geneious calls, typed every one of them into
`preprocess/called_variants.json` with its reason, and loaded WT and PT only. The
called-variant leaves of issue #731 close all four reasons, so the four tolerized
isolates Data Set S2 released calls for are now records.

### Measured before and after, read from the rebuilt dev LMDBs

| dataset | records before | records after | gene_set before | gene_set after |
|---|---|---|---|---|
| `ProteomeDeSiqueira2025Dataset` | 5 | **17** | 1 | 30 |
| `ProteomePercentDeSiqueira2025Dataset` | 5 | **17** | 1 | 30 |
| `ProteomeLog10PercentDeSiqueira2025Dataset` | 5 | **17** | 1 | 30 |
| `IsoprenolTiterDeSiqueira2025Dataset` | 2 | **10** | 6 | 35 |

17 of the 20 released proteome samples and 10 of Table S2's 24 cells. The three and two
left out are Sigma3's, the one isolate Data Set S2 never sequenced.

### How the 173 calls were encoded, measured on the pinned workbook

`preprocess/called_variants.json`, which now records the leaf each call became rather
than the reason none could hold it:

| encoding | calls | leaf |
|---|---|---|
| `bacterial_sequence_variant_in_locus` | 53 | `BacterialSequenceVariantPerturbation` |
| `bacterial_site_variant_intergenic` | 105 | `BacterialSiteVariantPerturbation`, `site_kind=intergenic` |
| `bacterial_site_variant_locus_not_in_assembly` | 10 | `BacterialSiteVariantPerturbation`, `site_kind=locus_not_in_assembly` |
| `restates_the_designed_pt_deletion` | 5 | none; dropped and counted |

Per-strain perturbations written, plus the one `BacterialDeletionPerturbation` every
written strain carries:

| strain | released calls | variants written | genotype perturbations |
|---|---|---|---|
| PT | 33 | 32 | 33 |
| Sigma1 | 34 | 33 | 34 |
| Sigma2 | 28 | 27 | 28 |
| Sigma4 | 43 | 42 | 43 |
| Sigma5 | 35 | 34 | 35 |

No two of those perturbation sets are equal, which is the collapse the first build
refused to risk.

### Four measurements that decided the encoding

- **The RefSeq-only calls cannot be keyed to a GenBank locus, and that is measured, not
  assumed.** The 10 rows name `PP_RS21780` (9) and `PP_RS19075` (1), and
  `resolve_gene_name` on the pinned KT2440 GenBank assembly returns `retired`, "not
  found in GCA_000007565.2_ASM756v2; retained as given", for both. They are written as
  site variants with `site_kind=locus_not_in_assembly` and the released RefSeq tag kept
  verbatim in `released_locus_statement`, so the identifier the source gave is not lost
  and no neighboring locus is invented.
- **One call per sequenced strain restates the designed deletion.** All five carry a
  `Deletion` at 3,063,718..3,064,173 on `PP_2675`, 456 bp of the `cytochrome c-550 PedF`
  CDS at frequency 1, which IS the markerless in-frame deletion the strain was built
  with. Absence has one encoding, so the call is dropped with its own encoding rather
  than written beside the `BacterialDeletionPerturbation` that already states it.
- **The frequency column is not one scale.** 157 of the 173 cells are bare fractions in
  (0, 1] and 16 are percent RANGES (`95.1% -> 97.6%` through `98.5% -> 98.6%`). A range
  has no single value, so those calls keep the verbatim cell and carry `frequency=None`;
  `released_frequency` checks every bare number into (0, 1] rather than trusting it.
- **21 insertions are released with `Maximum == Minimum - 1` and `Length` 0**, the
  zero-length interval between two reference bases. The coordinates are stored as
  released, never normalized to a one-base span the source never wrote. One of those
  rows also has an empty `Sequence` cell, which is correct for an insertion.

### A sequenced isolate's reference is PT's, and its calls are its genotype

`strain_reference` returns PT's `BacterialStrainBackground` for each of the four
isolates: PT is the base strain they were evolved from, so PT's deletion event is what
they hold CONSTANT, and the calls are what varies. That is the division
`BacterialStrainBackground` is documented for, and it keeps the record's genome reference
one interned object across all five strains (the titer store holds 1 reference for 10
records).

### What still refuses, with counts

- **Sigma3, in both families** (3 proteome samples, 2 determined Table S2 cells). Data
  Set S2 releases no row for it, so its genotype is unknown rather than untyped; writing
  it would assert it is genotypically the PT it was evolved from. The drop rule is
  renamed `strain_was_never_sequenced` and no schema change reaches it.
- **The two acetate media** (4 determined cells, all reading 0): Fig. S3's media A and C
  state 75.5 mM and 1 mM ammonium sulfate against the 2 g/L the served
  `M9_NREL_DESIQUEIRA2025` carries, so they are a different medium and `MEDIA_LIBRARY`
  holds no object for them.
- **6 `n.d.` cells**, which the Table S2 caption defines as not determined.
- **`Delta14-PP_2676`** stays a `BacterialBackgroundAllele(partial_deletion)` with a
  `ProvenanceGap` on `deleted_span`. The sequence-variant leaf could take it only with a
  released coordinate, and the source writes `Delta14` with no coordinate and without
  saying whether 14 counts base pairs or codons.
- **197 protein keys outside the namespace and 1 merged key**, unchanged.

### L0 to L4, measured after the rebuild

`proteome_desiqueira2025` **PASS** (the other two normalizations likewise):

| level | row | result |
|---|---|---|
| L0 | structural | 17 records validated |
| L1 | count | observed 17, expected 17 |
| L1 | orf_uniqueness | 71 ORFs, 71 with multiple strains (expected) |
| L1 | strain_condition_uniqueness | 17 distinct (strain, environment) pairs |
| L2 | value_fidelity | 26,027 values checked |
| L2 | se_nonnegative | 26,027 values checked |
| L3 | reference_finite | finite + key-matched for all 26,027 |
| L3 | measurement_type_consistent | one `dia_nn_top3_peptide_signal_replicate_mean` |
| L3 | assembly_pin | `('pputida_KT2440_ASM756v2', 'GCA_000007565.2')` |
| L3 | stored_scale_is_the_released_column_this_class_reads | 17 records on the class's own column pair |
| L4 | gene_containment_kt2440 | 1,531 measured and 30 perturbed host genes; 0 outside |
| L4 | protein_and_perturbed_locus_containment_assembly | 1.000 of 1,549 |

`isoprenol_titer_desiqueira2025` **PASS**:

| level | row | result |
|---|---|---|
| L0 | structural | 10 records validated |
| L1 | count | observed 10, expected 10 |
| L1 | strain_condition_uniqueness | 10 distinct (strain, environment) pairs |
| L2 | value_fidelity | 10 values checked |
| L3 | titer_unit_is_the_released_unit | mM, as Table S2 releases |
| L3 | stored_statistic_is_a_maximum | the maximum over the released sampling times |
| L3 | mixed_feed_cross_source_disagreement_is_declared | released 0.05 mM stored, text's claim reported |
| L3 | assembly_pin | `('pputida_KT2440_ASM756v2', 'GCA_000007565.2')` |
| L3 | cross_source_titer_agrees_with_the_results_text | 0.0043 mM, within Table S2's rounding |
| L4 | gene_containment_kt2440 | 0 measured and 30 perturbed host genes; 0 outside |
| L4 | perturbed_gene_containment_assembly | 1.000 of 30 |

The L4 gene-containment rows exempt `bacterial_site_variant` along with
`heterologous_pathway`: a site id names no gene of the assembly BY CONSTRUCTION, which is
the claim the leaf exists to make, so checking it against the locus universe would fail
the record for stating something true. The flanking loci of an intergenic call ARE
checked, which is what keeps the exemption from hiding a bad identifier.
