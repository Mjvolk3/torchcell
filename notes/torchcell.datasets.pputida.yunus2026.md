---
id: wg87m3rkjkeyluovo3r89ac
title: Yunus2026
desc: ''
updated: 1791394027718
created: 1791394027718
---

## 2026.10.07 - Loader, sourcing decisions and the titer gap

`torchcell/datasets/pputida/yunus2026.py` serves Yunus et al. 2026, "Predictive
CRISPR-mediated gene downregulation for enhanced production of sustainable aviation fuel
precursor in *Pseudomonas putida*" (Metab. Eng., doi:10.1016/j.ymben.2025.11.007), row 13
of [[experiments.database.expansion-bacteria]]. Citation key
`yunusPredictiveCRISPRmediatedGene2026`.

### What the paper is, and what it releases

FluxRETAP picked 57 downregulation targets and intuition picked 51 more; VAMMPIRE built
the sgRNA arrays; every construct went into the engineered isoprenol producer `IY1452`,
and the screen was read out by isoprenol titer. The campaign's single supplementary
component is `mmc1.docx`
(sha256 `daa2c91d0ec7b4560e086517bbdbcbf845c060f0f201c294c5ddef2399b963a9`, 14 tables).
`mmc1..mmc4` crossed with `docx/xlsx/pdf/zip/csv` were probed on `ars.els-cdn.com` for PII
`S1096717625001740` on 2026-10-07 and only `mmc1.docx` answered (HTTP 206); every other
combination answered 404. So the mirror is complete.

### The titers are the campaign's point and they are NOT released

This is the finding that shaped the loader. Every isoprenol number in the paper is a bar
chart: Fig. 4B-J (FluxRETAP group), Supplementary Fig. S6 (intuition group),
Supplementary Fig. S7 (OD600) and Supplementary Fig. S9 (multiplexed strains). The text
states exactly two titers, `1469 mg/L` for the `PP_4188` knockdown and `958 mg/L` for
`PP_0168`, and it never states the control strain's titer. A
`ProductTiterExperimentReference` requires a reference titer, so **the titer family
cannot be built without inventing the denominator, and it is not built.**

Supplementary Note 1 links the per-strain input table of its Pearson analysis
(`strain, isoprenol_production, <protein columns>`, read by the Note's own python
snippet) as a Benchling share page. Measured 2026-10-07: the share page answers HTTP 200
but is a JavaScript single-page app with no data in the HTML, and the internal API behind
it answers HTTP 401 ("permission denied - not logged in") on `/1/api/entries/...`,
`/1/api/shares/...` and `/api/v2/entries/...`, with or without the cookies the share page
hands out. So it is the Dryad-style un-scriptable case: the manual recipe is in the raw
mirror's `si_expected` and repeated below. Closing it is the one thing that would turn
this row into the production-campaign dataset the plan expects.

Both links, for the manual once:

- <https://benchling.com/s/etr-l20YX8nWcCM66vIvFUZf?m=slm-b4jTtjqlVo4ElPg5bjn3> (the input table)
- <https://benchling.com/s/etr-mta4DgBAatF0hFYzy275?m=slm-s52wQQYHQWxFJGHGC79z> (the analysis)

Recipe: open the input-table link in a browser, export the notebook table to CSV, deposit
it under `data/` in the raw mirror with `RetrievalMethod.manual_browser` and the sha256 of
the bytes that arrive, then add an `IsoprenolTiterYunus2026Dataset`. A control row in that
table is what supplies the reference titer.

### What IS built: two relative-expression families

Every released per-strain number in this paper is a RATIO of the target protein's DIA-NN
Top3 abundance in the CRISPRi strain to its abundance in the control strain. Both
families are `BacterialProteinAbundanceExperiment` / `ProteinAbundancePhenotype` with
`measurement_type = "dia_nn_top3_relative_to_control_strain"`; the record carries the
strain's own number and the reference carries `1.0`, which is the ratio's denominator by
definition, so experiment over reference reproduces the released value exactly and nothing
is imputed. An L4 rule asserts every reference value is exactly `1.0`, because a reference
that drifted off 1 would silently rescale the whole dataset.

| dataset | source | records | n | uncertainty |
|---|---|---|---|---|
| `CrispriKnockdownYunus2026Dataset` | Supplementary Table S3 | 102 | 1 | typed gap |
| `CrispriArrayYunus2026Dataset` | Supplementary Tables S8-S12 | 25 | 3 | sample SD -> SE = SD/sqrt(3) |

Dev-tree roots `data/torchcell/crispri_knockdown_yunus2026` and
`data/torchcell/crispri_array_yunus2026`; both read `fresh` under
`python -m torchcell.provenance.build_manifest`, and both pass L0 to L4 from the module's
own runner (`python -m torchcell.datasets.pputida.yunus2026 verify`).

**The schema stretch, stated plainly.** `ProteinAbundancePhenotype`'s docstring asks for
"absolute per-strain quantity on a log signal scale, NOT a ratio -- the WT/parent strain
supplies the reference". These records are ratios, which is a deliberate and documented
stretch of that class: no phenotype class today holds a per-protein abundance expressed
against a control strain. The PR asks for a typed `abundance_basis` axis
(`absolute | ratio_to_reference`) on that phenotype, which would make this exact instead
of documented. `measurement_type` is what keeps these numbers from ever being pooled with
the absolute Top3 signals of
[[torchcell.datasets.pputida.carruthers2025]].

### The chassis: a deferral this paper does not close

The background is `IY1452`, and the paper's ONLY statement of it is "we transformed the
CRISPRi plasmid into a highly genetically engineered isoprenol-producing strain (IY1452)
(Banerjee et al., 2024)". Banerjee 2024 (Metab. Eng. 82:157-170,
doi:10.1016/j.ymben.2024.02.004) is not in the literature mirror, so
`BacterialStrainBackground` carries `alleles=[]` and two typed
`deferred_pending_source_review` gaps on `genotype_statement` and `construction`, each
with `looked_in` = this paper's `paper.md` and `resolve_with` = that DOI. `parents` is
`["KT2440"]`, which the paper does state ("a highly engineered *Pseudomonas putida*
KT2440 strain developed by our group"), and the record pins
`assembly_reference("KT2440")` -> `GCA_000007565.2` / `pputida_KT2440_ASM756v2`.

Carruthers 2025 IS mirrored and states a genotype for `IY1449b` / `IY1452b`. That genotype
is deliberately NOT borrowed: `IY1452` and `IY1452b` are different designations and no
mirrored byte says they are the same strain. Borrowing it would be exactly the
substitution the sourcing rules forbid.

### Every guide, and the oligos that check each other

Supplementary Table S7 releases 204 forward/reverse sgRNA oligo pairs. A forward oligo is
`TCTGGGTCTCTTAGC` + spacer + `GTTTGGAGACCATCG` and a reverse oligo is
`CGATGGTCTCCAAAC` + spacer + `GCTAAGAGACCCAGA`. The build asserts both flank pairs and
that the reverse spacer is the reverse complement of the forward one: **204 of 204 hold**,
which is an independent check that neither sequence was mis-transcribed. 203 spacers are
22 nt, as the Methods state; `PP4650_sgRNA_NT1` is 21 nt and is stored verbatim rather
than corrected.

`NT<n>` is a guide VARIANT number, not a non-targeting filler (the reading Carruthers 2025
had for its own construct names). The evidence: `PP0339_NT1` and `PP0339_NT2` carry
distinct spacers, and Table S3 screens `PP_1607_NT2` and `PP_1607_NT4` as two separate
strains. So the library is keyed on `(locus tag, variant)` and the four variant-labeled
Table S3 strains each get their own spacer.

Assignment outcomes over the 102 kept records: **98 carry a sourced 22 nt spacer, 4 do
not**, each for a measured reason in `preprocess/guide_assignment.csv`:

| locus | reason |
|---|---|
| `PP_5064` (betA-II) | no Table S7 oligo names this locus |
| `PP_4678` (ilvC) | no Table S7 oligo names this locus |
| `PP_1444` (gcd) | two variant oligos (`NT1`, `NT2`) with distinct spacers, and the strain carries no variant label |
| `PP_1319` (petC) | two oligos BOTH labeled `NT1` (`PP1319_NT1_sgRNA`, `PP1319_sgRNA_NT1`) with distinct spacers |

Oligo labels that name no gene of the assembly, each documented rather than remapped:
`RFP`, `BFP` (reporters), `nontarget` (the non-targeting control guide), and `glgC`.

### Two cross-source disagreements, kept as findings

**`glgC` against `GlcC`.** Table S7's only oligo for the Table S1 target `PP_3744` is
labeled `glgC`, while Table S1 names that gene's enzyme `GlcC` ("transcriptional dual
regulator GlcC-Glycolate"). Measured on the pinned annotation: `glcC` resolves to
`PP_3744` and `glgC` resolves to no locus. The loader does not remap it. `PP_3744`'s
strain (IY1594) is an `n.d.` row and is dropped anyway, so nothing turns on the reading.

**`PP_4118` against `PP_4188`.** The Abstract names the best knockdown `PP_4118`; the
Results, the Discussion and Supplementary Fig. S8 all name `PP_4188`. Table S2 lists
`PP_4188` as `SucB`, "2-oxoglutarate dehydrogenase dihydrolipoyltranssuccinylase subunit",
which agrees with the Abstract's own gloss "a gene encoding alpha-ketoglutarate
dehydrogenase"; `PP_4118` appears nowhere else in the paper and in no SI table. Both
spellings are recorded in `SOURCED_VALUES` and no record is written from the Abstract.
Note that `experiments/database/scripts/build_bacteria_candidate_datasets_table.py`
carries the Abstract's `PP_4118` in its `why` field, sourced from the paywalled abstract;
that is now known to be the paper's own typo.

### n_samples and the uncertainty type, with their quotes

**Table S3, `n_replicates = 1`, no uncertainty.** "To show the effectiveness of
downregulation of different genes, we performed shotgun proteomics on 125 samples carrying
different sgRNAs (Supplementary Table S3)." The table holds exactly 125 rows and 125
distinct strain names, which the build asserts. 125 samples over 125 strains is one sample
per strain, so `n_replicates = 1` is arithmetic on the source, not an assumption. No
uncertainty is released for those rows and the replicate DESIGN behind one proteomics
sample is not stated anywhere in the Methods, so `protein_abundance_se` is a typed
`not_reported_by_primary` gap rather than a derived number.

**Tables S8-S12, `n_replicates = 3`, sample SD.** "Error bars represent standard deviation
from three biological replicates" (Fig. 3 caption, panels J-N, which ARE Tables S8-S12).
Those tables release the three per-replicate values themselves (`R1`, `R2`, `R3`), so the
mean and the sample SD are computed from the released values and the stored SE is
`SD / sqrt(3)`.

**Why the two families are separate datasets.** For a genotype both hold, they disagree:
Table S3 gives `PP_4188` 0.2213 while Table S8's three replicates mean 0.2509. They are
separate proteomics runs and are never pooled into one record.

### Records dropped

One rule, 23 records, sourced: "The expression levels of twenty-three genes could not be
determined as their gene expression was not detected in the control strain." An `n.d.` row
has no denominator, so there is no ratio to store. 125 rows - 23 = 102 records, and the
drop log's arithmetic is asserted at build time.

`PP_4090` is worth a line: it resolves `non_gene_feature` because the GenBank annotation
marks it `pseudo=True`. It is one of the 23 `n.d.` rows, so it never reaches a record;
if a future release gives it a value, a pseudogene locus is still a locus of the pinned
assembly and `reconcile_locus_tags` keeps it.

### Identifier histogram

Table S3's 123 distinct `PP_` tags against `pputida_KT2440_ASM756v2`: **122 `current`, 1
`non_gene_feature` (`PP_4090`, a pseudogene), 0 renamed, 0 retired, 0 ambiguous, 0
outside the namespace**; all 123 resolve through the locus-tag layer, so no symbol or
synonym layer was needed and nothing was remapped. The array family's 12 tags (11 guide
targets plus the five measured proteins, overlapping) are 12 of 12 `current`.
`MIN_RESOLVED_FRACTION` is 1.0 for both.

Oligo labels needed the symbol layer: `accA` -> `PP_1607`, `gltA` -> `PP_4194`, `bioB` ->
`PP_0362`, `birA` -> `PP_0437`, `pta` -> `PP_0774`, `edd` -> `PP_1010`, `hsdR` ->
`PP_4740`.

### Environment, and the media dependency

"For isoprenol production, cultures were inoculated at an OD600 of 0.2 in 5 mL M9 medium
with 2 % glucose and antibiotics (kanamycin 50 mg/L, gentamicin 10 mg/L) and induced with
0.2 % L-arabinose 4 h after inoculation", at 30 C and 180 rpm ("incubated under the same
conditions"), sampled at 48 h.

As stored: `media = MEDIA_LIBRARY["M9"]` (the salts, which is what the Methods name),
`temperature = 30 C`, `duration_hours = 48`, `aerobicity = "aerobic"`, and four
perturbations: glucose at 2 `percent_w/v` as `PhysicalFactor.carbon_source`, L-arabinose
at 0.2 `percent_w/v`, kanamycin at 50 `ug/mL` and gentamicin at 10 `ug/mL`.

Three things recorded here rather than typed:

- **The medium is an approximation and the PR asks for the fix.** The culture is M9 plus
  2 % glucose, which should be an `M9_GLUCOSE_2PCT_YUNUS2026` entry with glucose as a
  component. `media.py` is in `VALUE_SURFACE_RELPATHS`, so adding one blocks an
  incremental admission; the library addition is raised in the PR instead of taken here,
  and until then the carbon source rides on the environment as a physical factor.
- **The percentages have no basis in the source.** The Methods write "2 %" and "0.2 %"
  with no w/v or v/v. Both are stored `percent_w_v`, the convention for a solid solute,
  and the inference is on the `SOURCED_VALUES` note.
- **180 rpm and the 5 mL tube are not stored.** They are `CultureEnvironment` fields and
  `Experiment.environment` is annotated `Environment`, so pydantic would dump them away
  by declared type. Same slot Carruthers 2025 reports.

`mg/L` is stored as the numerically identical `ug/mL`, because `ConcentrationUnit` has no
`mg_per_l` member; that enum addition is also in the PR.

### Not loaded, each with its reason

- the per-strain isoprenol titers (above): bar charts, and no control titer
- Supplementary Tables S4 and S5 (145 downregulated and 193 upregulated proteins of the
  `PP_4188` strain, as fold change, log2 fold change and a t-test p-value): a derived
  differential statistic for a single strain with no per-replicate values released, and
  no phenotype class models a per-protein differential with its own test
- Supplementary Table S6 (plasmids) and Table S7's three sequencing primers (IY77, IY169,
  IY425): genotype and method metadata, not measurements
- Fig. 5A (TCA metabolite concentrations at 24, 48 and 72 h), Fig. 3C/D (terminal OD600
  and RFP fluorescence), Fig. 3F (the `PP_1607` growth curve): figures only
- PRIDE `PXD062697`: raw DIA mass spectra, which no loader here consumes
- Tables S1 and S2 (the target lists) are not records; they are carried in
  `preprocess/target_lists.csv` with the selection method (`intuition` / `fluxretap`) on
  each row

### Extraction, and what the build asserts

`mmc1.docx` is read with the stdlib (`zipfile` + `xml.etree.ElementTree`) over
`word/document.xml`, so the extraction adds no dependency and is byte-deterministic for a
pinned file. The build asserts the table COUNT (14), each consumed table's HEADER, and a
pinned sha256 of each consumed table's PARSED rows (`TABLE_DIGESTS`), so a re-released
docx or a changed parser stops the build instead of shifting a column. A construct name
must round-trip through its digit groups, so a name that does not re-serialize cannot
silently lose a guide.

### Compound identity

`isoprenol` (3-methyl-3-buten-1-ol) has a row in `compound_identity_table.json` since the
bacterial schema follow-ups (PR #729), which landed while this branch was in flight, so
`resolved_compound("isoprenol")` returns the curated identity. Its key,
`CPJRRXSHAYUTGL-UHFFFAOYSA-N`, is pinned as `ISOPRENOL_INCHIKEY` and
`check_isoprenol_identity()` stops the build if the row ever disagrees. No released titer
means no record stores the compound; the same row serves
[[torchcell.datasets.ecoli.wang2015]] and [[torchcell.datasets.pputida.carruthers2025]].

### Running it

```bash
export PYTHONPATH=<worktree>
python -m torchcell.datasets.pputida.yunus2026 deposit                  # or --retrieve-into <dir>
python -m torchcell.database.build_dataset_lmdb --dataset CrispriKnockdownYunus2026Dataset
python -m torchcell.database.build_dataset_lmdb --dataset CrispriArrayYunus2026Dataset
python -m torchcell.datasets.pputida.yunus2026 verify
python -m torchcell.datasets.pputida.yunus2026 digests                  # re-pin TABLE_DIGESTS
```

## 2026.10.08 - Tables S4 and S5: the Fold Change column is loaded, the test columns are not

The first build recorded Supplementary Tables S4 and S5 as not loaded, on the grounds
that they are "a derived differential statistic for a single strain with no per-replicate
values released, and no phenotype class models a per-protein differential with its own
test". Half of that is right and half is not. There is no home for the TEST, but the
`Fold Change` column is a ratio to the control strain from the same DIA-NN Top3
quantification, which is exactly the quantity `CrispriKnockdownYunus2026Dataset` already
stores, so it is loadable with no schema change.

`CrispriDifferentialProteomeYunus2026Dataset` is the third family: **one record**, the
`PP_4188` knockdown strain, carrying the 305 of 338 released protein keys that resolve to
a locus of the pinned assembly.

### Why a third dataset class and not a 103rd record of the Table S3 family

`torchcell/verification/protein.py`'s `_l3_measurement_type_consistent` asserts a single
`measurement_type` per protein-abundance dataset, and these two numbers are on different
scales:

| | Table S3 | Tables S4 + S5 |
|---|---|---|
| released column | `Relative expression level` | `Fold Change` |
| `measurement_type` | `dia_nn_top3_relative_to_control_strain` | `dia_nn_top3_fold_change_relative_to_control_strain` |
| replicate design | 1 sample per strain, sourced arithmetically from "shotgun proteomics on 125 samples" over 125 strains | 3 biological replicates (Supplementary Fig. S8's caption) |
| what one record is | one strain, its OWN target protein | one strain, 305 proteins |
| uncertainty | typed gap, no design stated behind one sample | typed gap, design stated and no spread released |

### The sourcing, verbatim

The replicate count comes from the Supplementary Fig. S8 caption, read out of the parsed
`mmc1.docx` (sha256 `daa2c91d0ec7b4560e086517bbdbcbf845c060f0f201c294c5ddef2399b963a9`):

> Supplementary Figure S8. Relative expression level of PP_4188 gene in the control and
> PP_4188 strains. Proteins were extracted at 48 h. Error bars represent standard
> deviation from three biological replicates.

and the two table captions name the strain and the direction:

> Supplementary Table S4. List of downregulated genes from PP_4188 strain
>
> Supplementary Table S5. List of upregulated proteins from PP_4188 strain

`SE` is still a typed `ProvenanceGap`. The caption states the replicate COUNT and that the
spread is a standard deviation, but the number exists only as that figure's error bars:
neither table releases a per-replicate value or a dispersion column, so there is nothing
to divide by sqrt(3). The `P-Value (Equal Variance)` column needs a per-group replicate
set, which corroborates n = 3, and is not itself a dispersion.

These three values are in a separate `SI_SOURCED_VALUES` dict, not in `SOURCED_VALUES`.
`audit_sourced_value` reads its artifact as TEXT to find the quote, and a `.docx` is a zip
of deflated XML, so a quote inside one can never be found that way. They are audited
instead against the paragraphs the WordprocessingML reader returns from the same
sha256-pinned bytes, in a data-marked test.

### What is read and deliberately not stored

| column | why not stored |
|---|---|
| `P-Value (Equal Variance)` | gap R: `ProteinAbundancePhenotype` carries `protein_abundance_se` and no per-protein p-value field, and the only p-value in all of `schema.py` is `gene_interaction_p_value` on `GeneInteractionPhenotype` |
| `(-Log10(P-Value))` | the same, and a presentation transform of the column above |
| `Rank` | a presentation index of the released sort order |

All three are read into `DifferentialRow`, used as build oracles and written to
`preprocess/differential.csv`, so no asserted column is parsed past. Four oracles,
each measured on the pinned bytes before it was written:

- `log2(Fold Change)` reproduces the released `Log2(Fold Change)` on **338 of 338** rows
  at a tolerance of 1e-6;
- `10 ** -(-Log10(P-Value))` reproduces the printed p to its own precision. The worst
  disagreement is **3.2488e-3** relative, on Table S4's `1.30E-06` against a released
  -log10 of 5.884647992, i.e. p = 1.3042e-6. That is the rounding of the **23 of 338**
  cells the docx prints in three-significant-figure scientific notation, which is why the
  tolerance is 5e-3 and not 1e-9;
- the `Rank` column is `1..145` in Table S4's row order and `1..193` in Table S5's, with
  the fold change ascending in the first and descending in the second;
- every row clears its direction's thresholds: fold change below 1 throughout Table S4
  and above 1 throughout Table S5, every absolute log2 at least 1, every p below 0.05.

### A correction to the audit: PP_4188 IS in Table S4, under another name

The audit note states "PP_4188 itself is absent from both tables so there is no clash
with its Table S3 row". The first half is true of the `Protein` column and the build
asserts it: no row of either table carries `PP_4188`, and nothing resolves to `PP_4188`
either, so the stored profile and the Table S3 record for the same strain share no
protein. The reason, though, is not that the protein is missing.

Table S4 row 40 is `Kgdb`, accession `Q88FB0`, described as "Dihydrolipoyllysine-residue
succinyltransferase component of 2-oxoglutarate dehydrogenase complex", at a fold change
of **0.238537433**. That is the enzyme this paper's own Table S2 names for `PP_4188`
("2-oxoglutarate dehydrogenase dihydrolipoyltranssuccinylase subunit"), and the pinned
annotation resolves the symbol `sucB` to `PP_4188`. Row 39 is `Kgda` (`Q88FA9`,
"2-oxoglutarate dehydrogenase, E1 component"), which is `PP_4189` by the same route
(`sucA`). Measured on the pinned annotation: `sucB` -> `PP_4188`, `sucA` -> `PP_4189`,
and `kgdB` resolves to nothing.

So the knocked-down protein's own fold change IS released; it is keyed by a title-cased
UniProt gene symbol this assembly carries no gene row for, which puts it in the 33 dropped
keys. The practical consequence: nothing clashes today, and a UniProt-to-locus-tag
crosswalk in the genomes tier (the same gap
[[torchcell.datasets.pputida.desiqueira2025]] records) would put 0.238537433 beside Table
S3's 0.2213 for the same strain, on a DIFFERENT `measurement_type`. That is the two
independent runs this module already documents (Table S3 gives `PP_4188` 0.2213 while
Table S8's three replicates mean 0.2509), not a contradiction.

### Key resolution, measured

| | count |
|---|---|
| released protein keys | 338 (145 down + 193 up, disjoint) |
| resolve to a current locus tag | 196 |
| resolve through a gene symbol | 109 |
| **stored** | **305** (0.9024) |
| retired, no gene row on this assembly | 33 |
| keys resolving to one locus twice | 0 |

`MIN_RESOLVED_FRACTION` is 0.90 on this class, overriding the base `_Yunus2026Dataset`'s
1.0: that 1.0 is right for SCREENED TARGETS, which are released as `PP_` tags, and wrong
for MEASURED keys, which are DIA-NN protein names. The 33 unresolved keys are in
`preprocess/differential.csv` with `stored=False`.

### Still not loaded, unchanged

The per-strain isoprenol titers (bar charts only, no control titer stated, so a
`ProductTiterExperimentReference` has no reference titer), the two Benchling pages, Table
S6, the three sequencing primers, Fig. 5A's TCA metabolites, Fig. 3C/D/F and PRIDE
PXD062697.

## 2026.10.09 - The two manually deposited Benchling tables become two loaded datasets

Issue #788 item 2 and issue #699. The titer gap the 2026.10.07 section records is closed
by a manual deposit, and the same deposited table carries a protein panel beside it. Two
new dataset classes:

- `IsoprenolTiterYunus2026Dataset` (`data/torchcell/isoprenol_titer_yunus2026`), 125
  records, `ProductTiterExperiment`.
- `CrispriPanelProteomeYunus2026Dataset`
  (`data/torchcell/crispri_panel_proteome_yunus2026`), 125 records,
  `BacterialProteinAbundanceExperiment`.

### The deposit, and its paste caveat

Supplementary Note 1 links two Benchling share pages: the Pearson analysis and its
per-strain input table. Both are a JavaScript single-page app whose data loads through an
authenticated internal API (HTTP 401 to scripts, measured 2026-10-07), so the owner
opened them in a browser and could not export a CSV. Both tables are therefore a
COPY-PASTE of the rendered page, deposited under
`$DATA_ROOT/torchcell-raw/yunusPredictiveCRISPRmediatedGene2026/data/benchling/` with
`RetrievalMethod.manual_browser`:

| file | bytes | sha256 | role |
|---|---|---|---|
| `strain_isoprenol_production_protein_abundance.tsv` | 269,377 | `4b5d70de2b858b93a5359a3d5f3c0c26db4c1b9387187cf39ec97456906c0838` | raw_data, loaded |
| `protein_correlation_with_isoprenol_production.tsv` | 80,862 | `0e05f26b2686a2039dae70e2e87368f23e4c7a7f54fc9048c67adc5b4c986212` | si_data, recorded only |

The caveat, verbatim from the deposit's own `DEPOSIT.md` and carried into
`Provenance.method` and into the `protein_abundance_se` gap of the panel family:

> These are therefore the page's DISPLAYED values, not an export: numeric precision is
> whatever the page rendered (e.g. p-values appear as `1.30e-18`, correlations to 9
> decimals, abundances as integers or with 2 decimals).

The manual recipe, the page-to-file mapping (including the owner's own note that the
mapping is not independently verified), `retrieved_by`, the deposit record and the
checksum file are all in each record's `retrieval.params`, and
`retrieval.sha256 == record.sha256` for both. The mirror manifest now carries three files:
the publisher's `mmc1.docx` plus these two.

### The correlation table is recorded, never stored

Its three columns are `Protein`, `Correlation_with_isoprenol_production` and `p_value`,
one row per protein. That is a derived statistic over a strain panel rather than a
measurement of any one strain, and no phenotype class models a per-protein correlation
with its own test, so it is in the manifest's `si_expected` and in `NOT_LOADED` and no
record stores it. Measured on the pinned bytes: 2,659 rows with p up to 0.9989, so it is
the UNFILTERED result frame of Supplementary Note 1's script rather than its p < 0.05
output, and it covers a wider protein set than the panel (252 of the panel's 253
accessions appear in it, one does not). Neither deposited file completes the other.

### The sourced unit

The deposited `isoprenol_production` column carries no unit. The unit is the paper's,
quoted from `paper.md` (sha256
`32ab4cd3753a930c6ad983809e7083a06159b7b0feb6b5252ace180273bbe563`) under the heading
`# 3.3. Predictive CRISPRi downregulation for improved isoprenol production`:

> The highest recorded isoprenol titer of $1 4 6 9 \mathrm { m g / L }$ (Fig. 4E) was
> achieved by downregulating PP_4188, a gene identified by FluxRETAP

`ConcentrationUnit` has no mg/L member and 1 mg/L is exactly 1 ug/mL, so the deposited
number is stored verbatim under `ug_per_ml` and no arithmetic touches a source value.

### Row-label reconciliation, measured 2026-10-09

The 132 row labels are CRISPRi TARGET labels, not strain names. Reconciled against Table
S3's `CRISPRi target gene` labels and tags plus Tables S1 and S2's locus tags:

| route | count |
|---|---|
| matches a released target exactly | 123 |
| matches after stripping the trailing `" (S)"` | 6 |
| matches after also stripping a trailing `_NT<digit>` | 2 (`PP_1607_NT1`, `PP_1607_NT3`) |
| matches nothing | 1 (`Control`, the reference row) |
| **total** | **132** |

The 121 distinct locus tags behind the 125 kept rows all resolve to current locus tags of
`pputida_KT2440_ASM756v2` (121 of 121, `MIN_RESOLVED_FRACTION` 1.0); 105 of them are in
Tables S1 or S2 and all 121 are in Table S3. Per-label routes are in
`preprocess/label_reconciliation.csv`.

`_NT<digit>` is read as the guide VARIANT number Table S3 already uses, not as a
non-targeting filler. That is this paper's own measured meaning: Table S7 gives
`PP_0339_NT1` and `PP_0339_NT2` distinct spacers, and Table S3 screens `PP_1607_NT2` and
`PP_1607_NT4` as two separate strains. Either reading names the same single perturbed
locus, so only the spacer lookup differs. 5 of the 125 kept records carry no sourced
spacer, for the reasons already in `preprocess/guide_assignment.csv`.

### Drop rules

| rule | scope | count |
|---|---|---|
| `row_label_carries_an_undefined_marker` | record | 6 |
| `no_locus_tag_in_the_goa_proteome_file` | protein_accession | 44 (panel family only) |
| `accession_names_several_loci` | protein_accession | 2 (panel family only) |

The `" (S)"` marker is dropped because NO mirrored byte defines it. The search, recorded
in `BENCHLING_MARKER_SEARCH` and in the rule's own description: `paper.md` contains the
string `(S)` zero times and names none of the six tags; the pinned `mmc1.docx` contains it
exactly once, inside the chemical name "ADP-dependent (S)-NAD(P)H-hydrate dehydratase" in
Supplementary Table S5, which defines nothing about a strain; Supplementary Note 1 states
only the two Benchling links and the Pearson script; and Supplementary Tables S6 and S7
carry each of the six tags as a plasmid or oligo name with no marker beside it. Every one
of the six marked labels (`PP_0751 (S)`, `PP_0815 (S)`, `PP_1240 (S)`, `PP_1769 (S)`,
`PP_4191 (S)`, `PP_4635 (S)`) also appears WITHOUT the marker as its own row, so the
marker separates two rows for one target and dropping the marked one loses no gene. The
alternative would be inventing a second strain identity this release never states.

### Accession resolution, measured 2026-10-09

The deposited columns are UniProt accessions and `ProteinAbundancePhenotype` keys by
locus tag, so each goes through `uniprot_locus_crosswalk`, which reads the assembly set's
GOA proteome file, the same sha256-pinned member the genome reads its GO from.

| | count |
|---|---|
| accession columns | 253 |
| reach exactly one KT2440 locus (**stored**) | **207** (0.8182) |
| reach several | 2 (`Q877U6`, `Q877V8`) |
| reach none | 44 |
| locus tags reached by two accessions | 0 |

`PANEL_MIN_RESOLVED_FRACTION` is pinned at 0.81, just under the measured 0.8182. Every
dropped accession is in `preprocess/dropped_accessions.csv`. 0.82 is high enough for the
family to fit, so the dataset is built rather than refused.

Blank and zero policy: 5,978 of 33,396 released abundance cells are exactly 0 and NONE is
blank. A released 0 is a present measurement and is kept verbatim, never imputed and never
dropped; the reader refuses a blank cell outright, because a blank is an absence this
loader has no sourced rule for.

### Replicate design, and the range rule

`ProductTiterPhenotype` requires neither an uncertainty nor a replicate count, but the
Fig. 4 caption states the design as a RANGE:

> All samples were extracted at $^ { 4 8 \mathrm { ~ h ~ } }$ . $\mathrm { O D } _ { 6 0
> 0 }$ at $^ { 4 8 \mathrm { ~ h ~ } }$ is shown in Supplementary Fig. S7. Error bars
> represent standard deviation from 3 to 6 biological replicates.

The deposit releases one number per strain with no SD and no companion statistic, and the
SD exists only as the figure's error bars, so a back-solve is precluded. CLAUDE.md's range
rule then takes the **conservative lower end, 3**, with
`sample_unit = biological_replicate`; `titer_uncertainty`, `titer_uncertainty_type` and
`titer_se` are typed gaps. The panel family stores `n_replicates = 1` per key, which is
the arithmetic of the deposit: one displayed number per (strain, protein), no replicate
column, no uncertainty, so one sample is the support of one stored abundance.

### measurement_type

`dia_nn_top3_signal_benchling_displayed`, from the paper's own words in Methods 2.6:

> Protein quantities were plotted using the Top 3 method, which averages the MS signal of
> the three most intense tryptic peptides.

It is an ABSOLUTE per-strain signal, distinct from this module's two ratio scales
(`dia_nn_top3_relative_to_control_strain`,
`dia_nn_top3_fold_change_relative_to_control_strain`) and from Carruthers 2025's
`dia_nn_top3_peptide_signal_mean` and `dia_nn_top3_percent_of_proteome_mean`, so
heterogeneous proteomics is never pooled. `benchling_displayed` is the half of the name
that carries the paste caveat into the scale itself.

### Cross-source proof: one agreement asserted, one disagreement recorded

Written into `preprocess/benchling_proofs.json` at build time by
`assert_benchling_titers_match_the_results_text`. The deposit is an independent retrieval
of the same campaign, so the two titers the Results print are the available join.

| strain | Results print | deposit | difference | treatment |
|---|---|---|---|---|
| `PP_0168` | 958 mg/L | 957.246595 | 0.7534 mg/L | ASSERTED, inside the 1 mg/L the paper prints to |
| `PP_4188` | 1469 mg/L | 1494.98874 | 25.98874 mg/L (1.7691 %) | RECORDED, not reconciled |

No mirrored byte says which number Fig. 4E was drawn from, so the deposited per-strain
column is what every record stores and the difference is recorded exactly rather than
repaired by preferring either source.

### References are REAL released controls, not denominators

The `Control` row of the deposited table is a genuine released control strain: isoprenol
845.73 mg/L and a full protein profile. The titer family's
`ProductTiterExperimentReference` carries that titer and the panel family's
`BacterialProteinAbundanceExperimentReference` carries that profile. This is the thing the
2026.10.07 section said was missing, and it is why neither family invents a denominator.
Both are different in kind from this module's two ratio families, whose reference is 1.0
by definition.

### L0-L4 verification, on the built dev stores

`verify_build(dataset_root, data_root, family=...)` dispatches `"titer"` and
`"panel_proteome"`; the registry entries in `torchcell/verification/runners.py` add the
family's own L4 containment.

`isoprenol_titer_yunus2026`, 125 records, PASS:

| level | rule | result |
|---|---|---|
| L0 | structural | 125 records validated |
| L1 | count | observed 125, expected 125 |
| L2 | value_fidelity | 125 values checked |
| L2 | uncertainty_nonnegative | 0 values (none released) |
| L2 | se_is_the_uncertainty_over_sqrt_n | 0 pairs; 125 records release no uncertainty and store no titer_se |
| L3 | titer_unit_is_the_pinned_unit | `ug/mL` |
| L3 | uncertainty_is_typed_or_gapped | pass |
| L3 | replicate_design_is_sourced_or_gapped | pass |
| L3 | heterologous_pathway_gene_counts | [0]; the dataset declares [0] |
| L3 | product_is_the_declared_one | isoprenol |
| L3 | titer_reference_is_one_released_control | all 125 reference 845.73 |
| L4 | titers_are_the_deposited_column | 125 stored titers are the deposited column verbatim |
| L4 | perturbed_locus_containment_assembly | 1.000 of 121 |

`crispri_panel_proteome_yunus2026`, 125 records, PASS:

| level | rule | result |
|---|---|---|
| L0 | structural | 125 records validated |
| L1 | count | observed 125, expected 125 |
| L1 | orf_uniqueness | 121 ORFs, 2 with multiple strains (expected) |
| L1 | panel_key_set_is_shared | all 125 records carry the same 207 protein keys |
| L2 | value_fidelity | 25,875 values checked |
| L2 | se_nonnegative | 0 values (none released) |
| L3 | reference_finite | finite and key-matched for all 25,875 |
| L3 | measurement_type_consistent | `dia_nn_top3_signal_benchling_displayed` |
| L3 | panel_reference_is_a_measured_control | one control profile of 188 distinct measured values |
| L4 | panel_profiles_are_the_deposited_columns | 125 profiles over 207 of 253 accessions verbatim, 3,703 released zeros kept |
| L4 | protein_and_perturbed_locus_containment_assembly | 1.000 of 317 |

The three existing yunus stores were also rebuilt: they could no longer be read by
`torchcell.verification.runners.load_records`, which raised
`AttributeError: Can't get attribute 'FoldChangeScale'` on a class the loader no longer
has. After the rebuild all three PASS too, at 102, 25 and 1 records.

### Decisions taken by recommendation

- `_NT<digit>` is read as this paper's guide VARIANT token rather than as the
  non-targeting filler `carruthers2025.parse_construct` reads it as. The two readings name
  the same single perturbed locus, and this module's own measured finding (distinct Table
  S7 spacers; two separately screened `PP_1607` variants) is about these exact bytes.
- `n_samples = 3` from the stated 3-to-6 range, by the conservative lower-end rule,
  because no companion statistic exists to back-solve a per-strain count from.
- `n_replicates = 1` per protein key on the panel family, the arithmetic of a deposit with
  no replicate column.
- `titer_environment()` is a separate `CultureEnvironment` builder, because
  `ProductTiterExperiment.environment` is annotated as one and pydantic serializes by the
  declared type; it finally carries the 5 mL working volume, 180 rpm and OD600 0.2 the
  Methods state and that `production_environment()` has to leave in `SOURCED_VALUES`.
  `vessel` stays a typed gap: the Methods say "5 mL M9 medium" and name no container.
- `DropRule` gained a `scope` field defaulting to `"record"`, so an accession-scope rule
  can sit in the same ledger with `n_records = 0` without breaking `DropLog.check()`'s
  record arithmetic. This mirrors `carruthers2025.DropRule`.
- `tests/torchcell/adapters/test_bacterial_adapters.py`'s three `== 57` pins were
  re-derived from the merged tree as `== 60`. They were already stale by one before this
  change.

## 2026.10.09 - The three CRISPRi ratio classes moved onto ProteinFoldChangePhenotype, and the S4/S5 p-value is now stored

Issue #770 records the mislabeling this section closes: every CRISPRi number this paper
releases in Tables S3 to S5 is a ratio to a control strain, and
`ProteinAbundancePhenotype`'s own docstring forbids exactly that ("absolute per-strain
quantity on a log signal scale, NOT a ratio"). The three CRISPRi dataset classes now
produce `BacterialProteinFoldChangeExperiment` / `...Reference` with a
`ProteinFoldChangePhenotype`, and the differential family stores the per-protein p-value
the old class had no field for. `CrispriPanelProteomeYunus2026Dataset`, added by the
Benchling deposit above, is NOT moved: its number is an absolute Top3 signal
(`PANEL_PROTEOME_MEASUREMENT_TYPE`), so `ProteinAbundancePhenotype` is its right home.

### What moved

| dataset | class before | class after | records before | records after |
|---|---|---|---|---|
| `CrispriKnockdownYunus2026Dataset` | `BacterialProteinAbundanceExperiment` | `BacterialProteinFoldChangeExperiment` | 102 | 102 |
| `CrispriArrayYunus2026Dataset` | `BacterialProteinAbundanceExperiment` | `BacterialProteinFoldChangeExperiment` | 25 | 25 |
| `CrispriDifferentialProteomeYunus2026Dataset` | `BacterialProteinAbundanceExperiment` | `BacterialProteinFoldChangeExperiment` | 1 (305 keys) | 1 (305 keys) |

The move changes the CLASS, not the retention: the same 23 `n.d.` rows are dropped from
Table S3 and the same 33 unresolvable DIA-NN keys are dropped from the differential
profile. Builds:
`python -m torchcell.database.build_dataset_lmdb --dataset <Class> --retire-existing`,
all three rebuilt 2026.10.09; `--list-stale --include-private` names none of them
afterwards. The five builders moved with the classes
(`relative_expression_phenotype`, `array_phenotype`, `differential_phenotype`,
`differential_reference_phenotype`, `reference_phenotype`), the three adapter confs under
`torchcell/adapters/conf/` now enable `protein fold change phenotype (chunked)` plus
`protein fold change phenotype reference` and neither protein-abundance method, and
`verify_protein_dataset` gained `label_key` / `se_key` keyword arguments (defaulting to
the absolute family, so every other caller is unchanged) because each of its rules is
about the SHAPE of a per-protein map, which both classes share.

### `fold_change_scale` is linear, measured two ways rather than assumed

`FOLD_CHANGE_SCALE = FoldChangeScale.linear` for all three families, and
`REFERENCE_RELATIVE_EXPRESSION` is now read off `FOLD_CHANGE_SCALE.neutral_value` instead
of being written down a second time.

1. Tables S4 and S5 release both columns side by side, verbatim headers `Fold Change` and
   `Log2(Fold Change)` (`si1.docx`, sha256
   `daa2c91d0ec7b4560e086517bbdbcbf845c060f0f201c294c5ddef2399b963a9`, Supplementary
   Tables S4 and S5 header row), and `parse_differential` asserts
   `log2(Fold Change) == Log2(Fold Change)` within 1e-6 on 338 of 338 rows. So the first
   column is the linear ratio and the second is its log2.
2. Table S3's census reproduces exactly on LINEAR thresholds. Verbatim, `paper.md`
   sha256 `32ab4cd3753a930c6ad983809e7083a06159b7b0feb6b5252ace180273bbe563`, Results
   3.2:

   > "Proteomics analysis revealed that 68 genes were downregulated by at least $5 0 \%$
   > , 51 of which were downregulated by more than $9 5 ~ \%$ (Fig. 3). Sixteen genes were
   > downregulated by $1 0 { - } 5 0 ~ \%$ . Thirteen genes were downregulated only by
   > $1 0 ~ \%$ . Five genes were upregulated."

   Measured over the 102 numeric rows: 68 at `<= 0.5`, 51 at `< 0.05`, 16 in
   `(0.5, 0.9]`, 13 in `(0.9, 1.0)`, 5 above `1.0`, and `68 + 16 + 13 + 5 = 102`. On a
   log2 scale 0.5 would be a 1.41-fold increase, not "downregulated by 50 %".

### `reference_basis` comes from the source's own clause

Table S3 and Tables S8-S12 (`REFERENCE_BASIS`), `paper.md`, Fig. 3 caption panel I:

> "(I) Summary of relative expression levels of target genes in comparison to the control
> strains."

What that control strain IS, Fig. 3 caption panel G:

> "(G) Protein counts of PP_1607 in both nontarget (control) and PP_1607 strains."

Tables S4 and S5 (`DIFFERENTIAL_REFERENCE_BASIS`), Fig. 5 caption panel B:

> "(B) Volcano plot representing the results of shotgun proteomic analysis from strain
> PP_4188 strain in comparison to the control strain. Horizontal dashed line represents
> the applied significance threshold of a student's t-test $p$ -value $= 0 . 0 5$ .
> Vertical dashed lines represent the applied thresholds of an absolute fold change
> ${ \geq } 1$ ."

### The S4/S5 p-value is stored, UNADJUSTED, and the paper's one FDR is not a correction

The released `P-Value (Equal Variance)` column now goes into
`protein_fold_change_p_value` for all 305 stored keys. The Fig. 5B caption above names
the test (a Student's t-test against the control strain) and the 0.05 threshold
`parse_differential` already asserted. Nothing in `paper.md` or `si1.docx` adjusts those
p-values: a search of both for `benjamini`, `hochberg`, `fdr`, `false discovery`,
`adjusted`, `multiple test`, `q-value` and `bonferroni` returns exactly one FDR, and it is
an IDENTIFICATION filter applied before any contrast was computed (`paper.md`, Methods
2.6):

> "The main DIA-NN reports were filtered with a global $\mathrm { F D R } = 0 . 0 1$ at
> both the precursor and protein group levels."

So `protein_fold_change_p_value_adjusted` is `None` with a typed `ProvenanceGap` and
`p_value_adjustment_method` is `None` (`DIFFERENTIAL_P_VALUE_ADJUSTMENT`), which is what
the class requires of a record with no adjusted map. `measurement_type` grew the test it
now carries:
`dia_nn_top3_fold_change_relative_to_control_strain_equal_variance_t_test`.

`DIFFERENTIAL_NOT_STORED` narrowed from the p-value columns to two entries: the
`(-Log10(P-Value))` column, because `10 ** -(-Log10(P-Value))` reproduces the stored p on
338 of 338 rows (worst relative disagreement 3.2489e-3, Table S4's `1.30E-06` against a
released `-log10` of `5.884647992`, which is the printed precision of the p column), and
`Rank`, a presentation index of the released sort order. Both stay build oracles.

### A released `0` is a measurement, which needed a one-line schema correction

Measured on the pinned docx: Table S3 writes the verbatim cell `0` on 37 of its 102
numeric rows, and Tables S9/S10/S11/S12 write it on 49 of their 93 replicate cells, which
makes 13 of the 51 (construct, protein) means exactly `0.0`. Tables S4 and S5 have no
such value (min fold change `0.003286423`, 0 of 338 non-positive). Those zeros are
complete knockdowns: the protein was detected in the CONTROL strain, so the ratio has a
denominator, and was not detected in the CRISPRi strain. The census above counts them,
and the 37 sit inside its 51 "more than 95 %" bucket. They are NOT the `n.d.` case, which
the paper states is missing in the control strain ("could not be determined as their gene
expression was not detected in the control strain") and which stays dropped.

`ProteinFoldChangePhenotype` first refused them: its linear branch required
`value > 0.0`. It now refuses `value < 0.0` only, so a negative linear ratio is still
impossible and a measured zero survives. `0.0 / 1.0 = 0.0`, so experiment over reference
still reproduces the released number exactly and `neutral_reference()` is unchanged.

### L0-L4, all three, on the rebuilt dev stores

`python -m torchcell.datasets.pputida.yunus2026 verify`, 2026.10.09, 114 `[ok]` results
and no failures. Two new rules joined the gate: L3 `scale_and_basis_are_one_contrast`
(one `(fold_change_scale, reference_basis)` pair per dataset, since the pair is what makes
two columns comparable) and L3 `p_values_are_unadjusted_probabilities` (a stored p is a
probability and names no correction).

| level | rule | knockdown | array | differential |
|---|---|---|---|---|
| L0 | structural | ok, 102 records | ok, 25 records | ok, 1 record |
| L1 | count | ok, 102 = 102 | ok, 25 = 25 | ok, 1 = 1 |
| L1 | orf_uniqueness | ok, 100 ORFs, 2 multi-strain | ok, 12 ORFs, 8 multi-strain | ok, 1 ORF |
| L2 | value_fidelity | ok, 102 values | ok, 51 values | ok, 305 values |
| L2 | se_nonnegative | ok, 0 values (typed gap) | ok, 51 values | ok, 0 values (typed gap) |
| L3 | reference_finite | ok, 102 values | ok, 51 values | ok, 305 values |
| L3 | measurement_type_consistent | ok, `dia_nn_top3_relative_to_control_strain` | ok, same | ok, `..._equal_variance_t_test` |
| L3 | scale_and_basis_are_one_contrast | ok, 102 records linear | ok, 25 records linear | ok, 1 record linear |
| L3 | p_values_are_unadjusted_probabilities | ok, 0 stored | ok, 0 stored | ok, 305 stored |
| L3 | provenance_audit | ok, 27 of 27 quotes verbatim | ok, 27 of 27 | ok, 27 of 27 |
| L4 | gene_containment_kt2440_locus_tags | ok, 100 of 100 | ok, 12 of 12 | ok, 306 of 306 |
| L4 | reference_is_the_ratio_denominator | ok, 102 values = 1.0 | ok, 51 values = 1.0 | ok, 305 values = 1.0 |

The provenance audit grew from 22 to 27 entries: `relative_expression_basis`,
`control_strain_is_nontarget`, `linear_scale_census`, `differential_basis_and_test` and
`identification_fdr` are the five new `SOURCED_VALUES`, each quoting `paper.md` verbatim.
