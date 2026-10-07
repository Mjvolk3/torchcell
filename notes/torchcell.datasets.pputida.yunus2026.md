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
