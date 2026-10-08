---
id: k62wtmrmap3xffhq5uprwkf
title: Niu2019_release_loadability
desc: ''
updated: 1791453200472
created: 1791453200472
---

## 2026.10.08 - Settling schedule row 48, Niu 2019 pinene evolved

Script: `experiments/036-dataset-fixes-before-kg-build/scripts/niu2019_release_loadability.py`
Results: `results/niu2019_release_loadability.json`, `results/niu2019_release_loadability_targets.csv`

Everything below is measured by that script on the sha256-pinned mirror, or quoted
verbatim from the pinned `paper.md`
(`43c733ed632ce1501fc734ac687a525b242d2d702e2c39e37adc0ac89c98dc80`) and `si/si1.docx`
(`72099acb46abce07983265e559ec49596209b031869f081f40ae1e0bc0ce6fb9`).

### The mirror is not missing a file

The triage flagged that only one SI file was captured. It is the whole deposit. The PMC
Article Datasets bucket prefix `PMC6556621.1` holds 8 objects, of which exactly one is a
supplementary file (`mmc1.docx`), and the article's JATS declares exactly one
`<supplementary-material>` element, also `mmc1.docx`. The mirror's `si/si1.docx` is those
bytes, sha256-identical, retrieved by `pmc_cloud`. Nothing released is absent.

All four Suppl. Tables and both Suppl. Figures are inside that one file.

### What the release gives per sample

**Suppl. Table 2 is a called variant list of ONE strain**, not an allele-frequency table
and not a clone genotype matrix. 391 rows = 1 header + 16 merged category headings + 374
data rows; the row kind is structural (a heading carries two `w:tc` cells, a data row
four), and 374 reconciles with the paper's own tally:

> A total of 349 single nucleotide variants (SNV) and 25 insertion/deletion (InDel) were identified

Per data row: a gene cell (322 of 374 carry a b-number), a protein description, a
mutation site with an absolute coordinate and reference and alternate base (344 of 374),
and a `Frequence` cell. Frequence is 1.00 in 362 rows, `100` in one (a typo for 1.00),
blank in one, and below 1.00 in ten. 48 rows are intergenic. One row has a blank mutation
site.

The caveat that has to travel with it is in Methods 2.2, verbatim:

> The paired-end reads from *E. coli* YZFP were aligned to the reference genome of *E. coli* MG1655 using the BWA software (version 0.7.12). Potential mutations including point mutation and insertion/deletions were identified using the Samtools software (version 1.1) and GATK module (Unified Genotyper).

No parent library exists and no subtraction is described, and the parent is BW25113, so
the 374 calls are the YZFP genotype against MG1655 and necessarily include every
BW25113-versus-MG1655 background difference plus the engineered insertions. The reads were
never deposited: no SRA, BioProject, ENA or GEO accession appears anywhere, and the
article has no data-availability section, so this docx is the only published form of the
variants.

**Suppl. Tables 3 and 4 are on the UNEVOLVED designed parent.** This is the finding that
decides the row, and it reverses what the schedule's `modality` field says. Methods 2.4:

> We applied this CRISPR-Cas-SoxS system [17] for activating and repressing target genes in E. coli BW25113(PT5-dxs).

and both main table titles name the same host, and the SI footnotes say it twice:

> a: the ratio of optical density (OD600) at 10 h with 0.5% pinene of target genes activation vs control (all the same but no target scRNA) in E. coli BW25113(PT5-dxs).
>
> b: the ratio of pinene production in Shake Flasks of target genes activation vs control (all the same but no target scRNA) in E. coli BW25113(PT5-dxs, pMEVIGPS).

So nothing about the CRISPRa/i values depends on the evolved genotype. CRISPRi *was* also
run in a YZFP-derived background, but only as the Fig. 3B whole-cell biocatalysis
experiment, which releases no numbers.

Counts: Suppl. Table 3 holds 57 activation targets (43 numeric growth ratios, 32 numeric
pinene ratios), Suppl. Table 4 holds 20 interference targets (9 and 6). Both match the
paper's own statements of 57 and 20. Two further strains, one six-target activation and
one six-target interference, are numeric in the MAIN tables only.

The replicate design is one blanket sentence, Methods 2.8:

> All experiments were conducted in triplicate, and data were averaged and presented as the means ± standard deviation.

so the `±` is a sample standard deviation over n = 3. Whether a replicate is biological or
technical is never stated for these assays; the paper distinguishes them only for qRT-PCR,
which is a different experiment, so that cannot be carried over.

### A dash is a measured non-positive, and the measurement proves it

The footnote under both tables reads, verbatim:

> -: means no change or negative effect.

Wording is not proof, so the script tests it: classifying every cell as numeric or dashed
must reproduce the paper's own tallies if the authors counted dashed cells as measured
non-improvements. It does, exactly, for both arms. Activation, stated 23 both / 20
growth-only / 9 pinene-only against observed 23 / 20 / 9; interference, stated 6 / 3 / 0
against observed 6 / 3 / 0. An unmeasured cell could not enter that arithmetic.

So a dash is left-censored, not null. It still cannot be encoded: the footnote conflates
"no change" with "negative effect", so the cell has neither a number nor a single
`ResponseCategory`. 64 cells are dashed.

### Which existing class can carry what

Every candidate was instantiated with the identifiers and values the release holds, and
the result recorded rather than inferred from a docstring.

| candidate | verdict |
|---|---|
| `BacterialCrisprInterferencePerturbation` on a b-number | ACCEPTED |
| `CrisprActivationPerturbation` on a b-number | ValidationError, `Invalid systematic gene name format` |
| `CrisprActivationPerturbation` on a gene symbol | ValidationError, same |
| `SequenceVariantPerturbation` on a b-number | ValidationError, same |
| `AllelePerturbation` on a b-number | ValidationError, same (issue #749) |
| `BacterialBackgroundAllele` on a variant row | takes it only by asserting `functional`, which the release never states, and has no slot for position, reference or alternate base, or Frequence (issue #731) |
| `EnvironmentResponsePhenotype` log2_ratio for the growth ratio | ACCEPTED |
| `ProductTiterPhenotype` for the pinene ratio | takes the number only by calling a dimensionless ratio a g/L concentration |

The CRISPRa gap is a new finding and is structural, not incidental: of the five bacterial
perturbation leaves, only CRISPRi carries an expression direction and it is fixed to
`decreased`, and the only other leaf that can state `increased` is
`PromoterReplacementPerturbation`, which would assert an edit this design never made.

`ConcentrationUnit` has no dimensionless member, so there is no unit under which a product
ratio is honest in `ProductTiterPhenotype`. This is the shape issue #770 already records
for protein fold changes.

### What is loadable, and why it is not loaded here

The only slice existing classes can carry is the interference arm's 9 numeric growth
ratios plus its one combination strain, 10 records, 10.6% of the 94 released numeric
cells. Those are honest to encode: the host is the designed parent, every one of the nine
targets resolves to exactly one BW25113 locus through the gene-symbol layer (`ydiJ ->
BW25113_1687`, `yjbQ -> BW25113_4056`, `prpR -> BW25113_0330`, `marR -> BW25113_1530`,
`fabR -> BW25113_3963`, `cedA -> BW25113_1731`, `narG -> BW25113_1224`, `marA ->
BW25113_1531`, `ychF -> BW25113_1203`), none is a multi-gene label, and every guide spacer
is released in Suppl. Table 1 as an `N20-<gene>` oligo. The encoding would be
`EnvironmentResponsePhenotype` log2_ratio against the no-guide control at log2(1) = 0,
which is the landed Lim 2025 encoding for the same kind of quantity.

They are not loaded in this branch because all three of their prerequisites sit outside
it:

1. The paper has no `torchcell-raw` mirror, so a new citation key and its provenance
   record would have to be deposited.
2. No spelling of pinene is in the pinned compound-identity table, while
   `anhydrotetracycline` is. Pinene is the dose the growth ratio is read against, so a
   record would carry the assay's central environmental variable as a typed gap. Adding
   the row re-queries PubChem for all 5,532 records and re-pins
   `compound_identity._TABLE_SHA256`, which is issue #726's work.
3. The activation arm carries four fifths of the release and waits on a new perturbation
   leaf, which changes a served schema closure and so belongs in the owner's full-rebuild
   batch.

A later superset admission is the clean path: the store can gain the activation arm and
the pinene readout once their classes land, without a rebuild of its own.

### Discrepancies recorded for whoever loads it

- The growth timepoint is 10 h in the SI footnote and 12 h in Methods 2.4. The main
  text's absolute ODs at 12 h, `0.451 ± 0.013` against `0.127 ± 0.009`, divide to 3.5512,
  which is the 3.55 printed for the combination strain, so 12 h is the reading the numbers
  support.
- Table 2's footnote says the variants "were identified with a frequency of 1.0" while ten
  SI rows are below 1.00 and one is blank.
- Main Table 2's own synonymous and nonsynonymous tallies do not match the SI rows.
- `muts` in the SI is `mutS`; one Frequence cell reads `100` for 1.00; `BW2F113(T5)` in a
  SI Table 1 heading is `BW25113(PT5-dxs)`.
- `paper.md` lines 186 to 222 are a DIFFERENT article (DeMott and Dedon on a
  phosphorothioate antiviral mechanism), bundled into the same PDF by the 2020 erratum
  list. A parser must stop at line 185.
- The erratum (doi 10.1016/j.synbio.2020.10.004) corrects a missing competing-interest
  statement and no data.
