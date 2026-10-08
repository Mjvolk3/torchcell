---
id: cdbcwlgh7qnhagkaamtr6hq
title: Foo2014
desc: ''
updated: 1791450412916
created: 1791450412916
---

## 2026.10.08 - Sourcing the nine released isopentenol titers

`IsopentenolTiterFoo2014Dataset` serves nine `ProductTiterExperiment` records: the eight
tolerance strains of mBio Table 1 and the PS+RFP control of its footnote b. Isopentenol is
isoprenol under its older name, so this is the only E. coli campaign on the priority-product
axis that releases numeric titers.

Mirror pins, both verified on disk at build time:

| artifact | sha256 | what it supplies |
| --- | --- | --- |
| `paper.md` | `b24baad46bdf488cf93a7e59c51eceb2af9ba4ad565459130897a6201cd62407` | Table 1, its two footnotes, and every Methods value |
| `si/si9.docx` (publisher `mbo005142049st3.docx`, Table S3) | `f8196b5ad05ce7520694adeb0238181ffc9773c57843a1400e2f9f07320bb4d1` | the nine genotypes and the chassis composition |

The build re-reads all 28 article quotes out of the hashed `paper.md` bytes before a value is
used, and reads the two Table S3 blocks (`Production strains`, `Production plasmids`) out of
the deposited document, so the one SI quote no article quote covers is audited too.

### The replicate design is sourced; the uncertainty TYPE is not released

Table 1 footnote a is the whole statement of the design: ``a Averages from triplicates after
48 h.`` The Results restate it for the same measurement, ``The production of isopentenol was
quantified in triplicate for all PS strains at 24, 48, and 72 h after 500 uM IPTG
induction``. So `n_samples=3` and `sample_unit=biological_replicate` are sourced.

What the plus-minus IS is stated nowhere. Measured over the whole mirror (`paper.md`, the
`pdftotext -layout` reading of `paper.pdf`, Text S1 `si/si1.docx`, and Tables S1 to S4
`si/si7.pdf`, `si/si8.docx`, `si/si9.docx`, `si/si10.docx`), exactly one sentence names an
error type, and it belongs to a different measurement: the Fig. 3 caption's ``Error bars
represent standard errors from at least 4 replicates``, which is the maximum-growth-rate
figure, at a different replicate count. The candidate-table row for this paper says "mean and
SD over 3 replicates"; the SD half of that is the row author's reading, not a sourced value,
and it is not what the loader stores.

The resolution ladder's back-solve was run and does not discriminate. The paper makes eight
calls on this table (PS+SoxS and PS+NrdH ``statistically similar`` to PS+RFP, the other six
increased). Under a one-way ANOVA with Tukey HSD over the nine strains at n=3, both readings
reproduce all eight:

| reading | pooled MSE | q(0.05, 9, 18) | HSD | calls reproduced |
| --- | --- | --- | --- | --- |
| spread = sample SD | 258.9 | 4.955 | 46.0 | 8 of 8 |
| spread = SE, so SD = spread x sqrt(3) | 776.7 | 4.955 | 79.7 | 8 of 8 |

Pooling the nine spreads scales the critical comparison by the same sqrt(3) that separates the
two readings, which is why the test cannot tell them apart. Welch per-pair without a
multiplicity correction reproduces NEITHER reading's nrdH call (p 0.0033 and p 0.017, both
significant where the paper says similar), so it is not the paper's analysis. `UncertaintyType`
has no `unknown` member, so `titer_uncertainty` and `titer_uncertainty_type` are typed
`ProvenanceGap`s on every record and the released plus-minus numbers are kept per record in
`preprocess/titer_rows.csv`.

### The reference titer IS released

Footnote b states it: ``b Percent improvement in titer is in comparison to result for PS+RFP
at 48 h (834 5 mg liter-1).`` (the plus-minus glyph reads as `\x05` in this one OCR line;
`pdftotext` reads the same cell as ``834 ⫾ 5``). That control is a number measured in the same
experiment, so unlike the sibling Yunus 2026 and Kang 2026 isoprenol families nothing is
refused for want of a denominator: `phenotype_reference` carries 834 ug/mL, and PS+RFP is also
a record of its own, as Table S3 lists it among the nine production strains.

### What was declined, and why

- **GEO GSE53138** (GSM1282891 to GSM1282896, platform GPL14649) is six arrays of ONE strain,
  MevT\*, three with and three without 0.2% isopentenol. All six share a genotype, so it is an
  environment-response transcriptome rather than a gene-perturbation record: a different
  experiment class, and a separate dataset if it is ever built. No raw reads are consumed here
  and no loader in this repo consumes raw reads. Table S1's microarray log2 and z scores are
  that series' processed summary on the same single strain, declined for the same reason.
- **Table S2's 40-candidate tolerance screen** releases no number: the growth-lag phenotype
  exists only as the Fig. 4 and Fig. S2 curves, and Table S2 is a gene list.
- **The production pathway is background, not a genotype.** pJBEI-6830 (`pBbA5c-MevTsa-PMK-MK`)
  and pJBEI-6833 (`pTrc99A-NudB-PMD`) are in all nine strains and in the control, so they are
  the `BacterialStrainBackground`, carried verbatim on its `construction` and
  `genotype_statement`. Their five gene tokens are NOT typed as
  `HeterologousPathwayPerturbation`: that leaf requires `source_organism` and
  `is_heterologous`, which this paper defers whole to reference 6 (George 2014, not mirrored),
  and the tokens do not share an answer (MevT with its `sa` suffix is not an E. coli pathway
  while NudB is an E. coli gene).
- **DH1's own lesions.** The paper names the host once, ``For growth assays and isopentenol
  production, E. coli DH1 (ATCC 33849) was used``, and never writes its K-12 marker genotype,
  so `alleles` is empty. MG1655 is the assembly pin because it is the genome the overexpressed
  genes were amplified from (``Gene candidates ... were PCR amplified using E. coli MG1655
  genomic DNA``) and the namespace their b-numbers belong to.
- **Two of the three antibiotics.** Only kanamycin is scoped to this experiment by the source
  (``kanamycin 15 ug ml-1 for isopentenol production or 30 ug ml-1 otherwise``); chloramphenicol
  and carbenicillin appear in the same sentence under ``Where appropriate`` with no statement of
  which plasmid carries which marker.

### The genotype axis and the eight loci

Table S3's `Production strains` block is the axis: `pJBEI-6830 + pJBEI-6833 + pBbS5k-<gene>`.
Reconciled against the pinned `ecoli_K12_MG1655_ASM584v2` annotation, all eight symbols
resolve, seven on the gene symbol and `gidB` on a synonym of its current name `rsmG`:

| strain | gene | locus | 48 h titer (mg/L) | released improvement (%) |
| --- | --- | --- | --- | --- |
| PS+SoxS | soxS | b4062 | 838 | 0 |
| PS+NrdH | nrdH | b2673 | 860 | 3 |
| PS+MdlB | mdlB | b0449 | 931 | 12 |
| PS+GidB | gidB | b3740 | 965 | 16 |
| PS+IbpA | ibpA | b3687 | 967 | 16 |
| PS+Fpr | fpr | b3924 | 989 | 19 |
| PS+YqhD | yqhD | b3011 | 994 | 19 |
| PS+MetR | metR | b3828 | 1,290 | 55 |
| PS+RFP | rfp | (none) | 834 | reference |

The titers are stored in `ug/mL`, because `ConcentrationUnit` has no `mg/L` member and 1 mg/L
is exactly 1 ug/mL, so no arithmetic touches a source value. The product resolves through the
paper's own IUPAC name ``(3-methyl-3-buten-1-ol)``, which `compound_identity_table.json`
carries as a synonym of `isoprenol` (`CPJRRXSHAYUTGL-UHFFFAOYSA-N`, PubChem CID 12988), so
these records join the isoprenol axis rather than an unresolved `isopentenol` stub.
`resolved_compound("isopentenol")` returns an unresolved compound today; adding that string to
the synonym list is a curation act and was not taken here.

### Verification, 2026.10.08

Twelve levels pass over the built dev store
(`$DATA_ROOT/data/torchcell/isopentenol_titer_foo2014`), eleven from the module's own
`verify_build` plus the family containment the product-titer runner appends:

| level | check | result |
| --- | --- | --- |
| L0 | structural | 9 records validated |
| L1 | count | observed 9, expected 9 |
| L2 | value_fidelity | 9 values checked |
| L2 | cross_method (one perturbation per record) | 9 pairs agree within 0.0 |
| L3 | titer_unit_is_the_sources_mg_per_l_as_ug_per_ml | pass |
| L3 | every_uncertainty_is_a_typed_gap_not_a_guess | pass |
| L3 | the_product_is_the_canonical_isoprenol_entity | pass |
| L3 | the_reference_is_the_released_PS_RFP_control_titer | pass |
| L4 | titer_against_released_improvement_percent | 9 entities agree within 0.0 |
| L4 | construct_against_deposited_table_s3 | 9 entities agree within 0.0 |
| L4 | perturbed_gene_containment_assembly | 1.000 of 8 genes are ASM584v2 loci |

The first L4 is a genuine second reading: each stored titer is re-derived from the released
control and Table 1's own `Improvement in titer (%)` column, a different column of the same
table from the one the values were parsed out of. The second re-reads each record's construct
from the deposited Table S3 bytes. The containment counts 8, not 9, because `rfp` carries the
`unreported` source organism and so is correctly excluded from the host locus universe.
