---
id: 492k6601jk4aetp2ojge0cj
title: Yang2019
desc: ''
updated: 1791557416713
created: 1791557416713
---

## 2026.10.09 - Row 49: refused, for want of a RELEASED reference titer (#826)

The umbrella issue #826 asks one question of row 49: whether a two-record
engineered-chassis titer dataset on the Foo 2014 pattern is honest. **It is not, and the
reason is a single missing released number.** No loader is written and no dataset is
registered. This is the dated refusal.

Yang et al. 2019, "Mevalonate production from ethanol by direct conversion through
acetyl-CoA using recombinant Pseudomonas putida, a novel biocatalyst for terpenoid
production", Microb Cell Fact 18:168, doi:10.1186/s12934-019-1213-y, citation key
`yangMevalonateProductionEthanol2019`. Pinned `paper.md` sha256
`693be53e3a9b10697723df15d04f487e2959c93cb795d6af869c9f10f1acb2da`; the two
supplementary files are `829419a065e2b6f59a8a3493131f912212bd767547c882a71e173d97f8d93014`
(`si/si1.docx`) and `862b6df03ed2f80ea68199e3b17ab26fc80653902dd17851ae746e5757128560`
(`si/si2.docx`). Measurement script and committed results:
`experiments/036-dataset-fixes-before-kg-build/scripts/yang2019_release_loadability.py`
(commit `d639bf4ad`).

### What IS stated verbatim, and it is more than the issue's "two prose titers"

Six figure-free mevalonate titers exist, not two; four carry a triplicate SD and two are
batch-fermenter values without one. Every one is in `yang2019_prose_titers.csv`.

| strain | condition | mevalonate (g/L) | SD |
|---|---|---|---|
| ELPP010 | flask, modified M9 + 10 g/L ethanol, 27 h | 1.70 | 0.55 |
| ELPP110 | flask, modified M9 + 10 g/L ethanol, 27 h | 2.43 | 1.34 |
| ELPP111 | flask, modified M9 + 10 g/L ethanol, 27 h | 2.88 | 1.16 |
| ELPP211 | flask, modified M9 + 10 g/L ethanol, 27 h | 4.07 | 0.29 |
| ELPP311 | 2.5 L batch fermenter, 300 mM ethanol, pH 7.0, 24 h | 4.18 | |
| ELPP311 | 2.5 L batch fermenter, 300 mM ethanol, pH 6.75 | 4.60 | |

Of the four fields the issue asks about, three are stated:

- **unit**: `g/L`, in each titer's own sentence.
- **timepoint**: `27 h` for the four flask values, `24 h` for the pH 7.0 batch value.
- **genotype**: the strain table gives each strain its plasmids and deletions verbatim,
  e.g. "ELPP211 | P. putida KT2440 ΔendA ΔendX ΔqedH-IΔqedH-II harboring pSGP11,
  pAWP89-1". The replicate design is stated too: "All strains were cultured under the
  above conditions and in triplet for reproducibility confrmation." and "All the
  experiments were performed in triplicates and standard deviations of triplet culture
  were shown in the form of error bars".

### What is NOT stated, and why that one absence refuses the dataset

**reference**: the base strain is ELPP000 and the release carries NO figure-free
mevalonate titer for it. Re-read on the pinned `paper.md`: every ELPP000 sentence is
qualitative. It "acts as a negative control", it "showed a typical form of aerobic
metabolism, where the cells rapidly grew to certain cell concentration without any
metabolite production and then they oxidized ethanol to acetate", and its mevalonate
curve exists only in Fig. 2a, whose numbers are figure-only along with all 60 time-course
readouts.

The schema makes that terminal, checked against the live classes:
`ProductTiterExperimentReference.phenotype_reference` is a required
`ProductTiterPhenotype`, and `ProductTiterPhenotype.titer` is a required `float` with no
`None` branch. So a reference cannot be constructed at all without a released reference
titer. Writing `titer = 0.0` would be the guess: "without any metabolite production"
describes the first growth phase of a strain that went on to accumulate acetate, so it is
not a released mevalonate measurement, and CLAUDE.md's rule is that a value the paper does
not state is never asserted as sourced.

This is the FOURTH titer-family refusal for exactly this reason, which is what makes it a
pattern rather than a one-off judgment: the same missing released parent-strain titer
blocks MCF2Chem 2023 ([[torchcell.datasets.ecoli.cai2023]], schema blocker 1 of 4) and
two others of the isoprenol panel. The Foo 2014 pattern the issue points at works
*because* that release states its parent-strain titer; Yang's does not.

### Three further measured facts about the release, recorded

- **No deposit exists to reopen it from.** "Availability of data and materials Not
  applicable." is the whole availability statement, and the words `deposit`, `repositor`,
  `accession`, `supplementary data`, `raw data` and `dataset` occur ZERO times in
  `paper.md`. Neither supplementary file carries a measurement: `si1.docx` is 2
  paragraphs and 1 table, `si2.docx` is 12 paragraphs and 0 tables, and `paper.md`'s
  single table is the strain and plasmid inventory.
- **The sequence claim holds exactly.** The released 9,283 bases are confirmed by summing
  the six sequences (mvaE 2,430, mvaS 1,185, atoB 1,216, acs 1,994, eutE 1,432, nphT7
  1,026), which splits as 181 leader plus 9,102 coding bases, every CDS in frame and
  stop-terminated and every leader carrying a Shine-Dalgarno motif.
- **The five `PP_` locus tags of the schedule row are DERIVED and unverifiable here.**
  `PP_` occurs zero times in `paper.md`, in the layout JSON and in both SI files; the
  release names its five deleted genes by GenBank gene ID instead (`endA` 1047019,
  `endX` 1045620, `qedH-I` 1046117, `qedH-II` 1046129, `phaG` 1046114).

### What would reopen it

A released ELPP000 mevalonate titer, in any figure-free form: a number in the text, a
supplementary table, or a deposit of Fig. 2's underlying values. Everything else the six
records need is already stated verbatim, so the reopening is one number.
