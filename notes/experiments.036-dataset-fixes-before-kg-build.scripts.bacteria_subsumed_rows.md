---
id: uf2k85z7q5y574belbmcb2c
title: Bacteria_subsumed_rows
desc: ''
updated: 1791494645599
created: 1791494645599
---

## 2026.10.08 - Butland 2008 (row 33 -> 25) is NOT subsumed, and it was never blocked

Measured by `experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py`;
results in `experiments/036-dataset-fixes-before-kg-build/results/bacteria_subsumed_rows.json`
and the per-pair table `butland2008_high_confidence_pairs.csv`. Provenance record:
`$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008/subsumption_record.json`. Issue #794.

### What the row claimed

Row 33 of the bacterial schedule carried `status="blocked"`, `confidence="recall"` and
`instances_basis="estimate"`, on the grounds that "the article is closed access with no PMC
deposit, and nature.com, Springer static content and mirror sites all refused fetching, so
the number of query genes, array strains and screened pairs are all unconfirmed." Issue #794
then proposed the opposite error: that the data "sits inside Babu's authoritative interaction
table with 727 records carrying `screen_id="Butland et al."`", so the row should become a
provenance record rather than a loader.

Both are wrong, and the second is the one worth stating loudly.

### The release, measured from the mirror

Every file is sha256-pinned in `$DATA_ROOT/torchcell-library/butlandESGAColiSynthetic2008/`
and the four workbooks are now also in the raw mirror as `data/si2.xls` .. `data/si5.xls`.

| Source | Measured |
|---|---|
| Supplementary Table 2 (`si3.xls`) | 39 distinct query strains |
| Supplementary Table 1 (`si2.xls`) | 8,073 recipient rows: 7,924 Keio deletion strains (Isolate 1 = 3,968, Isolate 2 = 3,956) plus 149 SPA-tag essential; 4,117 distinct recipient genes |
| Supplementary Table 4 (`si5.xls`) | 4 sheets (raw colony sizes, normalized median colony sizes, absolute Z scores, S scores), each 39 x 8,073 = **314,847 cells, every one populated**; 160,563 distinct gene pairs |
| Supplementary Table 3 (`si4.xls`) | 1,379 rows, **1,288 distinct ordered pairs** = 799 non-essential + 489 SPA-tag essential; the 799 split 730 aggravating / 69 alleviating |

The 1,288 / 799 / 730 / 69 decomposition reproduces the Results sentence exactly, which is the
check that the parse is right: "a high-confidence dataset of 1,288 genetic interactions ...
Among these, we detected 799 genetic interactions for nonessential genes, with 730 categorized
as aggravating interactions and only 69 as alleviating interactions."

Supplementary Table 4's own banner says what it is: "Raw colony sizes (see Sheet 1), normalized
median colony sizes (see Sheet 2), |Z scores| (see Sheet 3) and interaction (S) scores (see
Sheet 4) of each mutant gene pair from 39 genome-wide screens without any filtering parameters."
That sentence is the whole finding. The paper released the unfiltered matrix, not just its tail.

### What the served store holds of it

`GeneInteractionBabu2014Dataset` at `$DATA_ROOT/data/torchcell/gene_interaction_babu2014`:
38,579 records, screen histogram `{"This Study": 37,852, "Butland et al.": 727}`.

- **727 records**, over 697 distinct unordered gene pairs and 38 of the 39 donors. `b0130`
  (`yadE`) contributes no served record at all: it is one of the 39 donors Babu's Table S1
  attributes to Butland, yet no served pair names it.
- 727 of 314,847 released S scores is **0.23 percent**.
- Babu's Table S1 credits these 39 donors 151,027 tested pairs and releases 1,129 rows over
  them (1,045 aggravating, 84 alleviating); the loader keeps 727 and drops 402 whose recipient
  is a SPA-tag hypomorph.
- Of this paper's 1,270 unordered high-confidence pairs, **780 appear in Babu's Table S2 and
  490 do not** (321 non-essential, 169 SPA-tag essential). 460 reach the served store and 451
  of those carry the Butland tag.

### The scores are the same number, which is what makes the gap loadable

Of the 1,379 Table S3 rows, 873 match a Table S2 row in the same (donor, recipient)
orientation. **793 of those 873 carry the identical S score**; 80 are re-scored (max absolute
difference 9.93). A further 65 match only with donor and recipient swapped, and none of those
is identical, which is correct: a swapped pair is a different conjugant, not the same
measurement.

So Babu's `GI score` IS this paper's S score for the pairs the two releases share. Loading
Butland's release therefore ADDS records; it does not introduce a second score definition over
the same strains, which was the stated reason to prefer a provenance record.

### Conclusion and what changed

`decision = "partly_subsumed_loader_warranted"`. The row's status went from `blocked` to
`candidate` (the `Status` enum has no member for "unblocked and partly served"), `confidence`
from `recall` to `sourced`, `instances_basis` from `estimate` to `reported`, and the counts
from the 155,415 guess to the measured 314,847. That re-ranking moves the row from 33 to 25
and pushes rows 25 to 32 down one place each.

Loadable, with no schema change, against the existing `GeneInteractionPhenotype`:

- the 321 non-essential high-confidence pairs Babu's release omits, with Butland's S score,
  absolute Z score and `Log2(Q/R)` (a second readout Babu never released);
- at the limit, all 160,563 non-hypomorph gene pairs of the unfiltered S-score sheet.

Still blocked, and already filed:

- the 489 SPA-tag essential pairs, on the hypomorph bacterial perturbation leaf the Babu
  loader filed;
- the raw and normalized colony sizes, which need a colony-size phenotype class that does not
  exist (issue #776 item 3 is the neighboring gap).

## 2026.10.08 - Schmidt 2022 nitrogen (row 24) is subsumed condition for condition, and the gap is not loadable

Measured by the same script and recorded in the same results JSON. Per-compound table:
`experiments/036-dataset-fixes-before-kg-build/results/schmidt2022_nitrogen_conditions.csv`.
Provenance record:
`$DATA_ROOT/torchcell-raw/schmidtNitrogenMetabolismPseudomonas2022/subsumption_record.json`.

### The paper's own enumeration

Table 1 of the OCR is a real table, not an image, so the condition list comes out of the
released bytes rather than a hand count. It marks a sole-nitrogen source `(N)` and an
amino-acid drop-out condition `(-)`: **52 `(N)` and 19 `(-)`**, which is the abstract's
"we identified genes and proteins involved in the assimilation of 52 different nitrogen
containing compounds. To assay amino acid biosynthesis, 19 amino acid drop-out conditions
were also tested. From these 71 conditions ...". The script asserts the parse against the
compound map and raises on any disagreement, so the two can never drift.

### The mapping onto the served compendium is a bijection

The Borchert 2024 compendium's `nitrogen source` experiment group holds **104 samples over
52 distinct `condition_1` values**. Mapping Table 1's 52 names onto those 52 conditions
needs a synonym table (salt forms, spelled-out acids, and three genuine chemistry synonyms:
5-oxoproline = L-pyroglutamic acid, valerolactam = 2-piperidinone, gamma-aminobutyric =
4-aminobutyric acid), and with it:

- 52 of 52 compounds match a condition;
- 0 compounds are unmatched;
- 0 group conditions are left over.

Two samples per condition, which is the Methods' "Experiments were conducted in biological
duplicates".

### What the served store holds

`RbTnseqBorchert2024Dataset` at `$DATA_ROOT/data/torchcell/rbtnseq_borchert2024`: 1,372,280
records over 290 samples. All **104** nitrogen-source samples are served, **4,732 loci
each**, for **492,128 records**. A loader for this row would store every one of those values
a second time, so there is nothing to load.

### What the compendium does NOT have, and why it still is not loadable

This is the part the Borchert 2023 precedent warns about, and it is real here too:

- the **19 amino-acid drop-out conditions** have no condition in the compendium's
  nitrogen-source group at all;
- the paper's t-SNE Methods name "129 sole-nitrogen source growth assays in 51 different
  conditions" (2-ABA excluded), against the **102** samples the compendium carries over
  those same 51 conditions, so **27 replicate assays** are absent as well;
- and separately, the compendium's matrix is 4,732 loci, the set carrying a value in every
  one of its 332 samples, so any locus this paper measured that some other sample lacks is
  outside it.

None of it is recoverable. Unlike Borchert 2023, whose Supplementary File 1 released
per-replicate fitness and therefore yielded a real 10,824-value loader, this paper released
**no per-gene data file**: its supplemental material is one figure PDF
("SUPPLEMENTAL FILE 1, PDF file, 3.1 MB"), it references no Table S, and its Methods publish
the fitness data only at `http://fit.genomics.lbl.gov`. That endpoint is probed by the
script rather than assumed Cloudflare-blocked: **HTTP 403 on 2026-10-08**, recorded in the
record's `release_probes`.

So: `decision = "subsumed_no_loader"`, with the absent slice named and its
unrecoverability measured rather than waved at.

## 2026.10.08 - Thompson 2020 fatty acid and alcohol (row 28 -> 29) is subsumed with no gap at all

Measured by the same script. Per-compound table:
`experiments/036-dataset-fixes-before-kg-build/results/thompson2020_carbon_conditions.csv`.
Provenance record:
`$DATA_ROOT/torchcell-raw/thompsonFattyAcidAlcohol2020/subsumption_record.json`.

### The paper's own enumeration

The abstract states the scale ("Global fitness analyses of transposon libraries grown on 13
fatty acids and 10 alcohols produced strong phenotypes for hundreds of genes") and the
Results name every compound: straight-chain C3 to C10 plus C12 and C14, the esters Tween 20
and butyl stearate, and oleic acid; then ethanol, butanol, pentanol, the four diols, and the
three branched-chain alcohols. 13 + 10 = 23, and the script raises if the compound map and
the abstract's two counts ever disagree.

### Every one of the 23 is a served compendium condition

The Borchert 2024 compendium's `carbon source` group holds 206 samples over 54 conditions.
Mapping the paper's 23 names onto it needs the same kind of synonym table as Schmidt (the
paper names an anion, the compendium names the acid or its sodium salt; isopentanol =
3-methyl-1-butanol, isoprenol = 3-methyl-3-butenol), and with it:

- 23 of 23 compounds match a condition;
- 0 compounds are unmatched;
- **47 samples** sit at those 23 conditions, 22 of the 23 in exactly the biological duplicate
  the Methods state. Butyrate is the exception with three: two in the fatty-acid set and one
  from an earlier library set and date.

The samples spread over three compendium library sets, which is why no single metadata column
attributes them: the fatty acids are one set, six alcohols another, and four alcohols plus
propionate sit in a set shared with another study's aromatics. Condition identity, not the
`person` or `set` column, is what makes the match.

### What the served store holds

All **47** of those samples are served, **4,732 loci each**, for **222,404 records** out of
the store's 1,372,280. A loader for this row would store every one of those values a second
time.

### Nothing is left over, and that is measured

Unlike Schmidt 2022, there is not even a condition-level gap: no condition this paper reports
is missing from the compendium's carbon-source group. The schedule row's earlier claim that
"not every published condition survived the compendium's completeness filter" was **wrong**:
that filter eliminated LOCI (the matrix is the 4,732 loci carrying a value in all 332
samples), not conditions.

The eliminated loci are the only slice the compendium is known to drop, and this paper
released nothing to recover them from: its supplemental material is one figure PDF
("SUPPLEMENTAL FILE 1, PDF file, 3 MB"), it references no Table S, and the Methods publish
its fitness data only at `http://fit.genomics.lbl.gov`, which answered **HTTP 403 on
2026-10-08** when the script probed it. Borchert 2023's own SI carried per-replicate fitness,
which is what made that row a real 10,824-value loader; this one has no such bytes.

`decision = "subsumed_no_loader"`.

### What changed on the three rows, together

| Row | Before | After | Loader? |
|---|---|---|---|
| 24 Schmidt 2022 nitrogen | `candidate`, "must be deduplicated against that row" | `candidate`, bijection measured, 492,128 served records | no |
| 29 Thompson 2020 (was 28) | `candidate`, "not every published condition survived" | `candidate`, 23 of 23 conditions served, 222,404 records | no |
| 25 Butland 2008 (was 33) | `blocked`, `recall`, 155,415 estimate | `candidate`, `sourced`, 314,847 measured | YES, 321 pairs at least |

The `Status` enum is `Literal["candidate", "blocked", "aggregation"]` and has no member for
"served elsewhere, no loader needed", so all three keep `candidate` and the measurement with
its date lives in `why`. A `subsumed` member would fit rows 24 and 29 exactly; it was not
invented here.
