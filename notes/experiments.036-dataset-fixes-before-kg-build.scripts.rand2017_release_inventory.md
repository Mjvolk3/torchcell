---
id: f88z0fltde714r072znahh2
title: Rand2017_release_inventory
desc: ''
updated: 1791614263475
created: 1791614263475
---

## 2026.10.10 - Rand 2017 Putida_ML5 library: subsumed by content, held out by a medium

Schedule row "Rand 2017 Putida_ML5 library" (class "Modality / backbone"). Script:
`experiments/036-dataset-fixes-before-kg-build/scripts/rand2017_release_inventory.py`;
results: `experiments/036-dataset-fixes-before-kg-build/results/rand2017_release_inventory.json`
(the same payload is the raw-mirror provenance record `subsumption_record.json`).

### Mirror route

Not in the Zotero-backed literature mirror. Deposited from the PMC Article Datasets bucket
(`PMC5705400.1`, the NIH author manuscript, `pmc_cloud` retrievals with sha256) into
`$DATA_ROOT/torchcell-raw/randMetabolicPathwayCatabolizing2017/`:

| path | role | bytes |
|---|---|---|
| `paper/PMC5705400.1.txt` | paper_text (every quote is from here) | 79,674 |
| `paper/PMC5705400.1.xml` | paper_text | 164,132 |
| `paper/PMC5705400.1.pdf` | paper_pdf | 1,688,033 |
| `si/NIHMS900347-supplement-1.pdf` | si_pdf (Supplementary Note, Figs, Tables 1-10) | 1,754,068 |
| `si/NIHMS900347-supplement-2.pdf` | si_pdf (Life Sciences Reporting Summary) | 69,230 |
| `si/NIHMS900347-supplement-1.txt` | si_text, `pdftotext -layout` (poppler 21.01.0) | 43,325 |

The "Supplementary Files" of plasmid sequences and the Jupyter notebook the Methods name are
not among the PMC objects and were not retrieved.

### What the paper released

No per-record data file. The Data Availability statement points only at the Fitness Browser
("All data from the P. putida transposon sequencing experiments is available through the
fitness browser at http://fit.genomics.lbl.gov/cgi-bin/exps.cgi?orgId=Putida&expGroup=carbon%20source.").
Probed 2026-10-10: HTTP 403 with a Cloudflare "Just a moment" challenge, so nothing is
retrievable from it by script. The only printed values are Supplementary Table 1: 60 genes,
two contrast columns (LA/Gluc and 4HV/Gluc), 59 values each (lvaB, PP_2792, is `NA` in both),
each "the average of 2 replicates" and defined as Fitness(LA) minus Fitness(Glucose).

The Methods' RB-TnSeq design: "The carbon sources tested were 40mM 4HV (pH adjusted to 7 with
NaOH), 40mM LA (pH adjusted to 7 with NaOH), 20mM potassium acetate, and 40mM glucose, each
with two replicates." and "The 4HV and acetate experiments were performed one day and the LA
experiments were performed on a different day, each day with its own 40 mM Glucose control."
That is ten samples.

### (a) The fitness experiments are inside the Borchert 2024 compendium (measured by content)

The compendium's `mutantLibrary` column names `Putida_ML5` on 25 samples (set1 10, set5 10,
set6 5). Each SI column was compared against every compendium sample as treatment and every
glucose-only sample as control, over the 59 printed genes present in the compendium:

| SI column | treatment | control | r | median abs diff | max abs diff | within half-unit | runner-up treatment (median) | runner-up control (median) | other day's glucose (median) |
|---|---|---|---|---|---|---|---|---|---|
| LA/Gluc | set5IT081/082 (LA 40 mM) | set5IT075/076 | 0.99952 | 0.041 | 0.264 | 38/59 | set5IT083, LA 20 mM (0.264) | set100IT009 (0.135) | set1IT078/079 (0.159) |
| 4HV/Gluc | set1IT082/083 (4HV 40 mM) | set1IT078/079 (glucose 40 mM) | 0.99984 | 0.025 | 0.195 | 44/59 | set15IT057 (1.959) | set1IT080, glucose 20 mM (0.149) | set5IT075/076 (0.165) |

So set1 is the 4HV/acetate day and set5 the LA day, each with its own glucose pair, which is
the Methods' design. The agreement is close but NOT bit-identical (38 and 44 of 59 values
within the half-unit of the SI's printed precision). Hypothesis (untested): the Fitness
Browser recomputed the gene fitness after 2017, so the compendium's values are a later
normalization of the same reads. The acetate pair (set1IT084/085, 20 mM) has no printed value
(the SI dropped acetate-shared phenotypes) and is attributed by the Methods quote only.

One discrepancy is carried, not resolved: the Methods say the LA day had a 40 mM glucose
control, and the compendium labels set5IT075/076 as 20 mM glucose. Content picks set5 as the
LA-day control over set1's 40 mM pair (median 0.041 against 0.159).

### Served: zero records, held out by the medium

All ten Rand samples sit on `RCH2_defined_noCarbon`, which `RbTnseqBorchert2024Dataset` drops
(`medium_not_in_media_library`, 20 samples, 94,640 records, read from the dev store's
`preprocess/dropped_records.json`). So the served store holds 0 of Rand 2017's 47,320
gene-level records (10 samples x 4,732 loci). The recipe is in Price 2018 Table S18; the
blocker is compound-identity rows for its salts. Issue #860.

### (b) The library description is a provenance record, not stored genotypes

`TransposonInsertionPerturbation` has `barcode` and `insertion_position` slots, so a released
per-strain table WOULD be stored. Rand 2017 releases none: the Methods name the library ("We
named the final, sequenced mapped transposon mutant library Putida_ML5."), the pKMW3 mariner
vector with random 20mer barcodes, and say only "thousands" of colonies were pooled; the SI
prints no strain table; the Fitness Browser is the only pointer and answers 403. The library
size the schedule row cites (185,401 insertions) comes from Borchert 2022, which is not
mirrored and was not checked here. So the records stay gene-level insertion genotypes with
`barcode` and `insertion_position` None, as Borchert 2024 already stores them, and this
record is what they reference.

### Decision

Subsumed, no loader. `RbTnseqBorchert2024Dataset` now attributes the ten samples to a
`rand2017` source study (basis `source_study_quote`, quotes audited against `torchcell-raw`),
so when #860 lands the 47,320 records carry Rand 2017's publication instead of the
compendium's. Net-new content outside this row (main-text Table 1: 10 mapped Tn5 isolates of a
separate library; Table 2: lva deletions with complementation, ordinal growth calls) is issue
#861.
