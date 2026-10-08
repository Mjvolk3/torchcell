---
id: zmlrczf0yw4lhki45pabblb
title: Mohiuddin2022
desc: ''
updated: 1791496363996
created: 1791496363996
---

## 2026.10.08 - Row 40 of the fifty bacterial datasets

Mohiuddin SG, Massahi A, Orman MA. "High-Throughput Screening of a Promoter Library
Reveals New Persister Mechanisms in Escherichia coli." Microbiology Spectrum 2022
10(1):e02253-21,
[doi:10.1128/spectrum.02253-21](https://doi.org/10.1128/spectrum.02253-21). Citation key
`mohiuddinHighthroughputScreeningPromoter2022`, PMC8865558, PMID 35196813.

Loader: `torchcell/datasets/ecoli/mohiuddin2022.py`, class
`PromoterReporterMohiuddin2022Dataset`, dev store
`$DATA_ROOT/data/torchcell/promoter_reporter_mohiuddin2022`. Adapter:
`torchcell/adapters/mohiuddin2022_adapter.py`, conf
`torchcell/adapters/conf/promoter_reporter_mohiuddin2022_adapter.yaml`. Verifier:
`torchcell/verification/promoter_activity.py`. Tests:
`tests/torchcell/datasets/ecoli/test_mohiuddin2022.py`,
`tests/torchcell/adapters/test_mohiuddin2022_adapter.py`,
`tests/torchcell/verification/test_promoter_activity.py`.

### The paper is in neither Zotero library

The sanctioned capture path does not reach this paper. `scripts/lit_sync.py` diffs Zotero
against the mirror, and a read-only scan of both libraries found no item with this DOI or
title: 1,216 items in group 6582362 and 15,714 in user library 9025336, and a `q`
quick-search for `persister` returns zero in both. The only `Mohiuddin` in the served
`library.bib` is Afroz Mohiuddin, a coauthor of the Performer paper; the only `Orman`
matches are the substring inside `performance`.

This is one step EARLIER than the missing-PDF-attachment failure that blocks Nichols 2011
(issue #691): there is no item to attach a PDF to. Adding the item is the library owner's
to do, so there is no `torchcell-library/<key>/` for this citation key and no `paper.md`
to quote.

Every quote is anchored instead inside the RAW mirror, which the loader writes and which
needs no Zotero item:

| mirror path | role | bytes | what it is |
| --- | --- | --- | --- |
| `data/spectrum02253-21_supp_2_seq5.xlsx` | `raw_data` | 1,311,598 | Supplemental File 2, the build input |
| `si/spectrum02253-21_supp_1_seq6.pdf` | `si_pdf` | 1,045,385 | Supplemental File 1, Figures S1-S8 and Tables S1-S4 |
| `paper/PMC8865558.1.pdf` | `paper_pdf` | 4,300,189 | the article PDF as PMC serves it |
| `paper/PMC8865558.1.txt` | `paper_text` | 73,346 | the publisher's own article text, the quote anchor |
| `si/spectrum02253-21_supp_2_seq5.legends.txt` | `si_text` | 579 | the workbook's four legend rows, rendered to text |

The first four come off the PMC Article Datasets bucket prefix `PMC8865558.1`
(`pmc_cloud`, scriptable, re-retrieved bit-identically). The fifth is DERIVED: the
workbook is binary, so a quote inside it is not auditable, and
`extract_sheet_legends` renders rows 1 and 2 of both sheets to text with a
`ProcessingRecord` naming the reader and pinning the workbook's own sha256. Two new
artifact roles came with this, `paper_text` and `si_text`, both documented in
`torchcell/literature/manifest.py`.

### The record

RECORD = one (arm, plate, well, read hour) `PromoterActivityExperiment`. The grid is
1,930 wells x 4 arms x 9 hourly reads = **69,480 records**, nothing dropped, every
reading a finite positive number (minimum 8.629, median 12.88, maximum 692.3).

The triage row counts 7,720 instances at dim 9, which is the same grid counted as (well
x arm). 69,480 is the row's own note for the released readings, and it is what one record
per time point gives.

- GENOTYPE: one `HeterologousPathwayPerturbation`, the episomal promoter-GFP reporter,
  whose `promoter_name` is the well's promoter verbatim. The chromosome is untouched, so
  `AssemblyReferenceGenome.background` is `None`.
- ENVIRONMENT: `LB` Miller at 37 C, aerobic, `duration_hours` = the read hour, plus the
  arm's antibiotic as one `SmallMoleculePerturbation` ON HOUR FIVE ONWARD ONLY.
- PHENOTYPE: `PromoterActivityPhenotype`, `plate_reader_fluorescence`, `n_samples = 1`.
- REFERENCE: the same well in the untreated arm at the same hour.

### Why a new phenotype family

Checked for an existing-but-unused class first, because three agents this week each
predicted a gap that was already filled. The two phenotype families that still had no
consumer are `FluxPhenotype` (a fitted net-flux map) and `BacterialVisualScorePhenotype`
(an ordinal colony score). Neither is a reporter readout.

`EnvironmentResponsePhenotype` was the near miss, and the decisive reason against it is
structural rather than nominal. It is documented as a fitness/growth response and this
number is transcription, but what settles it is that its verifier requires an
environmental edit on EVERY record: that rule refuses the untreated arm and each treated
well's three pre-dose reads, which is 34,740 of the 69,480 readings, and the control arm
is not an artifact of the design but the denominator the paper publishes.

Its `MeasurementType` enum also has no member for an absolute fluorescence intensity, but
that is a weaker argument than it first looks: `relative_growth_rate` was added to that
same enum on main while this branch was open, so a member CAN be added with the rebuild
consequence declared. The untreated arm is the reason, not the enum.

So the family is new and additive: `PromoterActivityPhenotype` with the
`ReporterReadout` axis, its `PromoterActivityExperiment` / `...Reference` pair pinning
`AssemblyReferenceGenome`, the `promoter activity phenotype` node class under the Biolink
3.2.1 `phenotypic feature` parent, and the `CellAdapter` node method and reference
collector. No served class changed, so this is admissible rather than a rebuild.

### The fold change sheet is exactly derivable, so it is not stored

The sheet's own legend states the rule: "Fold changes were calculated by taking the ratio
of GFP values of antibiotic treated cultures to those of untreated cultures." Measured
over all 52,110 cells, treated / untreated reproduces the released value to a maximum
relative error of **2.19e-16**, which is double-precision round-off; 52,110 of 52,110 are
within 1e-9.

It is recoverable from ONE stored record, not two, because a record's reference carries
the untreated reading of the same well at the same hour. An L3 row re-checks that against
the sheet on the built store.

The join is positional. The sheet keys only by promoter name and nine labels repeat
(`lacZ` 22 times), so the three promoter columns agreeing with the raw sheet's own column
row for row is what makes any cell joinable at all; the build raises if they do not.

### One record per time point, not a nine-vector

No phenotype in the schema carries an ordered sequence. Every vector-valued phenotype is
a dict keyed by a biological entity (gene, protein, metabolite, reaction), never by time.
The repo's convention for repeated measurements over time is one record per time point
with the time on `Environment.duration_hours`, which is what Caglar 2017 does with its
`growthTime_hr` column.

Here it is also the only shape that can state the dose: three of a treated well's nine
reads were taken before the drug went in, so they have a DIFFERENT environment from the
other six, which one record carrying a nine-vector could not say.

### The dose goes on hour five, and the fold changes corroborate it only weakly

The paper says the dose went in at hour five and ran five hours to hour ten. The released
per-arm median fold changes are 1.009/1.015/1.006 at hour two, 1.004/1.004/1.010 at hour
three and 0.992/0.994/0.995 at hour four, all within 1.5% of 1.00, and the first clear
movement is ampicillin at hour seven (1.062, rising to 1.579 at hour ten). So the
pre-dose reads look like baseline, but the placement comes from the paper's statement
rather than from the data. The paper does not say whether the hour-five read was taken
immediately before or immediately after the dose.

### Sourced values

`n_samples = 1` is the one that matters most, and it is stated outright in Statistical
analysis: "High-throughput screening of the promoter library and the Keio collection was
performed only once." There is no replicate, no SD and no SE, so every uncertainty field
is a typed `ProvenanceGap`. The `N=3` and `N=4` in the SI figure legends belong to the
single-strain validation assays, not to this screen.

Not asserted: the general Methods say kanamycin at 50 ug/mL maintained plasmids, but the
screening section names only LB for both the preculture and the assay plate, so no
selection agent is on the medium. The paper calls the backbone "a low-copy-number
plasmid" and never says which of the collection's two backbones carries which promoter,
so `construct_name` is unset and the reporter's origin reads `unreported`. A perturbation
leaf has no gap field, so both absences live in the loader docstring.

### Promoter identity, and where the gene linkage is NOT

`reconcile_locus_tags` resolves the 1,809 distinct labels against MG1655
GCA_000005845.2 (24 as a locus tag, 1,481 as a gene symbol, 266 as a gene synonym, 38 not
found) with the retain-all policy that drops nothing for a naming reason. **1,761 name
one locus; 48 do not** and are stored with `promoter_gene = None` and their label
verbatim: the eight rRNA operon names `rrnA` to `rrnH`, the `insA`/`insB` IS copies,
retired symbols and b-numbers, two ambiguous (`spr`, `ygaD`), six kept as given because
another label reached the same locus, and `Empty`, `U66` and `U139`.

Those last three occupy one well on each of 20 plates and name no E. coli gene. The
library's two backbones are pUA66 and pUA139 in Zaslaver 2006, the collection's source
paper, so they read as the empty-vector and empty-well background controls, but the paper
defines none of the three labels and no record asserts it.

The 48 labels sit in 106 of the 1,930 wells, which is 3,816 of the records.

The only gene the genotype names is the reporter, so the dataset's `gene_set` is
`{"gfp"}` -- the same shape Foo 2014's `rfp` and Menasalvas 2025's reporter take. The
measured gene identity is the phenotype's `promoter_gene` property, which is a property
match and NOT a graph edge to the gene node. That is deliberate: making it an edge would
mean writing the promoter's gene as the perturbation's `systematic_gene_name`, which
asserts the gene was added to the strain, and what the plasmid carries is a copy of its
PROMOTER driving GFP. A promoter or regulatory-element node class is what would turn the
property into an edge, and this dataset is the first that would use one.

### Verification, row by row

`python -m torchcell.datasets.ecoli.mohiuddin2022 verify` on the dev store, all 24 rows
PASS:

| level | row | result |
| --- | --- | --- |
| L0 | `structural` | 69,480 records validated |
| L1 | `count` | observed 69,480, expected 69,480 |
| L1 | `reading_uniqueness` | 69,480 records over 69,480 distinct (well, read time) keys |
| L2 | `value_fidelity` | 69,480 values finite and non-negative |
| L3 | `units_declared` | all 69,480 name units, reporter and readout |
| L3 | `promoter_gene_is_a_locus` | 1,761 distinct genes resolve to themselves; 3,816 records carry no gene |
| L3 | `reference_is_the_control_reading` | all 69,480 references read the same promoter at the same time |
| L3 | `fold_change_recoverable_from_one_record` | 52,110 records reproduce the released fold change, worst relative error 2.19e-16 |
| L3 | `the_drug_is_on_the_reads_from_hour_five_only` | 17,370 untreated, 17,370 pre-dose, 34,740 dosed |
| L3 | `provenance_audit` x 14 | every quote found in its pinned artifact |
| L4 | `promoter_genes_in_the_host_gene_universe` | 1,761 genes of 4,651 MG1655 loci |

The report is at
`$DATA_ROOT/data/torchcell/promoter_reporter_mohiuddin2022/preprocess/verification_report.json`.

### Cost of the per-record reference

The reference is distinct per (well, read hour), so the store holds 17,370 distinct
references and `preprocess/experiment_reference_index.json` is 284 MB, which takes 30.2 s
to load (measured). That is well inside precedent (`proteome_messner2023` is 1.2 GB), but
it is paid by every consumer:

- the verifier streams the store eleven times and reloads the interned set on each pass:
  **7 min** on an idle machine for the whole L0-L4 run (measured 16:45 to 16:52:21);
- `assert_dev_store_graph` runs past the suite's 300 s default, because six reference
  collectors each walk all 17,370 entries and sha256 a `model_dump` carrying the whole LB
  media object. That test carries `@pytest.mark.timeout(1800)` and says why.

The cardinality is irreducible under this schema: `phenotype_reference` requires a real
`PromoterActivityPhenotype` with a float, the only honest float is the matched untreated
reading, and that reading varies per well and per hour. The alternative is to drop the
untreated reading from the reference, which is exactly what makes the fold change
recoverable in-record, so the cost is bought deliberately rather than overlooked.

### Pins

103 mapped datasets, 52 bacterial, both re-derived by importing the merged tree rather
than incremented. The `kg_bacteria.yaml` entry sits in the same position as the
`_bacterial_adapter_cases.py` row, because the gate asserts positional equality.
