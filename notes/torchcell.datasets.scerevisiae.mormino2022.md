---
id: fh5h3nghpjiozvmzt7ccd5n
title: Mormino2022
desc: ''
updated: 1783922212215
created: 1783922212215
---

## 2026.07.13 - Mormino 2022 CRISPRi acetic-acid biosensor screen (12 strains)

`CrispriMormino2022Dataset` -- first CRISPR **interference** dataset (Lian was the first
CRISPR dataset overall). 12 records = Table 1 "Properties of isolated strains".

- **Genotype**: `CrisprInterferencePerturbation(target, effector="dCas9-Mxi1", guide=None)`
  (guides live upstream in the Smith 2017 CRISPRi library; not released here), BY4742.
- **Environment**: acetic acid 50 mM, pH 3.5, 30 C, aerobic (screen/biosensor condition).
- **Phenotype**: `EnvironmentResponsePhenotype` categorical -- Haa1-biosensor RFP `+`
  (enhanced -> more sensitive) -> `sensitive` (QCR8/TIF34/MSN5/PAP1/COX10/TRA1); `=` ->
  `no_effect` (NDC1/CBP2/UBA2/RPS30B/HSH49/LCB1). Reference = CC23 control (`no_effect`).

**Deliberately NOT built** (documented, not guessed): genome-wide enrichment is figure-only
(bar charts) -> not ingested; Table 1 Growth column ambiguous -> not stored; `n_samples` None
(qualitative summary call); temperature 30 C = standard (not stated for these isolates) --
flagged for review. Source = sha256-pinned mirror `paper.pdf` (388f8e92...), Table 1 embedded
literal (no SI data file released). L0-L4 all pass (12 records, containment 1.000). Scout that
established figure-only status: memory `[[remaining-datasets-blocked-status]]`.

## 2026.09.12 - Serve-50 rebuild: SC not SD, the ATc inducer, the CBL comparator

Rebuilt for admission. The 2026-07-12 build failed L0 structural on all 12 records (`Media`
gained a required `is_synthetic` after they were serialized) and carried four incorrect or
missing metadata facts. The 12 VALUES were never in doubt: the Table 1 calls are a verbatim
transcription, and this build now checks all 12 rows against the OCR'd table at build time.
Nothing was served, so nothing is updated in place.

### Four corrections, each sourced

**1. The medium is SC, not SD.** The old build stored `Media(name="SD, pH 3.5",
state="liquid")`. Methods:

> "yeast cells were cultivated in synthetic complete medium (SC) (0.77 g L-1 complete
> supplement mix drop out (CSM), 6.9 g L-1 yeast nitrogen base without amino acids (YNB w/o
> AA), 20 g L-1 glucose, pH 5.5, 4.5 or 3.5)"

SD (minimal) and SC (complete) are different media, so the old string asserted the wrong
nutrient content AND defeated the join to `media.SC`. The build now uses the shared `SC`
object -- whose 0.77 / 6.9 / 20 g/L concentrations are themselves sourced from THIS
sentence, making Mormino the origin of the shipped constant's only quantified components.

**2. pH 3.5 is a typed edit, not part of a free-text name.** It now rides as
`EnvironmentPhysicalPerturbation(factor=ph, magnitude=Concentration(3.5, ph))`, so it can be
compared with any other dataset's pH. `Media` has no `ph` field and cannot gain one (it sits
in 36 served closures).

**3. ATc is in the record.** 2 ug/mL anhydrotetracycline, with its DMSO vehicle as a typed
`Solvent`, is now a `SmallMoleculePerturbation`:

> "containing 2 ug mL-1 ATc (SC-ATc)" ... "The ATc stock solution was prepared by dissolving
> ATc into DMSO to a concentration of 125 ug mL-1" ... "Acetic acid, as well as ATc, were
> added to the media at the beginning of the cultivation."

ATc is the CRISPRi inducer. Without it the dCas9-Mxi1 genotype is not repressed, so a record
that omits it describes a different strain state. The acetic acid dose is unchanged at
50 mM, and the NaOH pH-titration of the acid stock is recorded in its SourcedValue note --
it is what makes the 50 mM dose and the pH 3.5 factor independent edits rather than one
acidification.

**4. The comparator is the CBL pool, not CC23.** Table 1's own footnote:

> "*Reporter expression 30% higher (+) or similar (=) compared to the CBL"

CC23 is the comparator for the Growth column and for the separate 150 mM / sfpHluorin
experiments. The `units` string and the reference now name the CBL (the CRISPRi Biosensor
Library, ~100,000 CFU pooled) and carry the released 30% threshold.

Also fixed: `n_samples` was None although "Screening of the pooled and single cell cultures
sorted by FACS was performed in two biological replicates"; `assay_type` was None where
`biosensor_readout` is exactly this assay; the docstring claimed 30 C was unsourced although
"Plates with 200 uL-cultures were cultivated at 30 C and 85% humidity, shaking at 995 rpm";
and `n_guides` was None where an isolated strain carries exactly one gRNA (the "1-16 gRNAs
each" is a per-GENE property of the library).

### Category mapping

| Table 1 symbol | `ResponseCategory` | `category_label` | n |
|---|---|---|---|
| `+` (reporter 30% higher than the CBL) | `enhanced` | `+` | 6 |
| `=` (similar to the CBL) | `no_change` | `=` | 6 |

The reference (the CBL pool itself) is `no_change` / `=`. `category_label` keeps the source's
own symbol rather than the old loader's invented words `sensitive` / `no_effect`, which were
also mis-stated: `+` marks an enhanced BIOSENSOR SIGNAL, and sensitivity is the paper's
interpretation of it, not the measurement.

### The biosensor cassette is now in the genotype

Every screened strain carries it, verbatim:

> "The biosensor with the sTF encoding construct BM3R1-HAA1-mTurquoise2 expressed under the
> RET2 promoter together with the sfpHluorin expression cassette (pMM4_14L) was integrated
> into the HO locus of the CRISPRi library strains [17] and to the control strain (CC23)"

so the genotype gains two `GeneAdditionPerturbation`s (`localization=chromosomal_integration`,
`integration_locus=HO`, `construct_name=pMM4_14L`, `is_heterologous=True`). They are a
CONSTANT background -- present in every strain AND in the comparator -- so the verifier entry
must pass
`background_genes = frozenset({"BM3R1-HAA1-mTurquoise2", "sfpHluorin"})`,
which keeps them out of the L1 strain identity, the canonical-gene-name rule and the L4 gene
set. Without that parameter L4 `current_genome_genes` fails, because a heterologous cassette
has no R64 systematic name by construction.

Lian 2019's bAID cassettes are deliberately NOT treated this way: its integration locus is
never stated, so there is nothing to source.

### Table 1 is now auditable, not merely transcribed

The 12 rows stay a module literal (the build re-derives nothing), but `audit_table_1` checks
each one VERBATIM against the machine-readable HTML table in the sha256-pinned `paper.md`
before any record is written, and raises if the OCR ever changes underneath. The test file
runs the same audit against the mirror.

### Records dropped: none

Acetic acid, anhydrotetracycline and DMSO all resolve to a structure identifier, all 12
targets resolve to current R64 genes, and every row verifies. `preprocess/dropped_records.json`
records the empty rule set explicitly.

### Provenance

Raw mirror `$DATA_ROOT/torchcell-raw/morminoIdentificationAceticAcid2022/` holds `paper.pdf`
(BMC counter URL, `direct_url`, re-retrieved and sha256-verified `388f8e92...` on
2026-09-12) and `paper.md` (the OCR the Table 1 audit reads, with a `ProcessingRecord`
naming `paper.pdf` as its input). The manifest is `provenance_complete=False` on purpose:
the MinerU version and DPI of the 2026-07-12 OCR run were not recorded at the time, and a
version is not fabricated to make the field green.

### Verifier result (verbatim, scratch runner with this entry's parameters)

```
crispri_mormino2022: FAIL
  [ok] L0 structural: 12 records validated
  [ok] L1 count: observed 12, expected 12
  [ok] L1 pair_uniqueness: 12 unique (study, strain, condition) records, one each
  [ok] L1 provenance_gaps: 0 documented provenance gaps over 0/12 records
  [ok] L1 canonical_gene_names: 12 systematic names, one canonical spelling each, each current in the genome
  [ok] L2 value_fidelity: 0 values checked
  [ok] L2 se_nonnegative: 0 values checked
  [ok] L2 uncertainty_sanity: 0 labelled uncertainties, none a zero dispersion
  [ok] L3 measurement_type_consistent: single measurement_type: MeasurementType.categorical
  [XX] L3 reference_zero: categorical rule: 0 references carry no category; 1 distinct
       reference categories ['no_change']; baseline also used as a measured call in
       {'no_change': 6}
  [ok] L3 environment_perturbed: all 12 experiments carry an environmental edit
  [ok] L3 compound_identity: 36 compound references carry a structure identifier
  [ok] L3 media_compound_identity: 384 compound references carry a structure identifier
  [ok] L3 media_membership: 12 records on a shared MEDIA_LIBRARY medium (1 distinct media)
  [ok] L4 gene_containment_sgd: 1.000 of 12 measured genes are S288C reference genes
  [ok] L4 current_genome_genes: every one of the 12 measured systematic names is current
```

The single failure is the same rule problem Smith 2006 hits, and it is structural rather
than fixable in the data. `_reference_baseline_result` requires the reference's baseline
category to be used by NO experiment record. That holds for a hits-only screen; it cannot
hold here, because the comparator IS the CBL and half the isolated strains are reported as
similar to it, so `no_change` is necessarily both the baseline and a measured call. The
requested shared-layer change is in
[[torchcell.datasets.scerevisiae.smith2006]]: drop `and not collisions` from the categorical
branch's `passed` expression and keep it as a reported detail.

### Is 12 records worth serving

Yes, for entity coverage rather than statistical weight. It is the only dataset exercising
`AssayType.biosensor_readout`, it brings 12 essential / respiratory-growth-essential genes
under CRISPRi knockdown, and its acetic acid + SC + pH 3.5 edit is the join partner for the
acid-stress conditions in the other incoming chemogenomic sets. It carries no guessed VALUE;
what it used to carry was guessed METADATA, and that is what this rebuild fixed.

### Open flags

- Table 1's `Growth` column (`ns` / `-` / `ND`) is a second readout against a DIFFERENT
  comparator (CC23) and its `E or RE` essentiality column is an annotation; neither is
  ingested. The Growth values stay in the literal only so each row can be checked verbatim
  against the OCR table.
- The genome-wide enrichment is figure-only (Figs 2-6); the guide-level screen data lives
  upstream in Smith JD et al. 2017, which is the target if a CRISPRi x fitness matrix is
  wanted.

## 2026.09.13 - Categorical reference rule applied; the dataset PASSES

The shared-layer request from the 2026.09.12 section was applied: the categorical branch of
`_reference_baseline_result` no longer fails when the reference's baseline category is also a
measured call, it reports the collision in `details`. That is structural here, because the
comparator IS the CBL pool and half the isolated strains are reported as similar to it.
Nothing in the loader or the build changed. The verifier line is now:

```
[ok] L3 reference_zero: categorical rule: all 12 references carry the baseline category
     'no_change', which no experiment record reports
```

`crispri_mormino2022: PASS` on all 16 levels, run with
`background_genes = frozenset({"BM3R1-HAA1-mTurquoise2", "sfpHluorin"})`.
