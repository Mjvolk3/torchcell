---
id: phjgebl1ch3bqhmvpytqpfz
title: Butland2008
desc: ''
updated: 1791500236344
created: 1791500236345
---

## 2026.10.08 - Loader added: the UNFILTERED eSGA interaction matrix

Row 25 of the fifty bacterial datasets, tier 1. Butland, Babu, Diaz-Mejia et al. 2008,
Nature Methods 5:789-795, doi:10.1038/nmeth.1239, citation key
`butlandESGAColiSynthetic2008`. 39 genome-wide conjugation screens, one Hfr Cavalli query
deletion each crossed into an arrayed F- recipient collection of 8,073 strains.

`GeneInteractionButland2008Dataset` stores **296,390** `BacterialGeneInteractionExperiment`
records, one per cell of Supplementary Table 4's S-score sheet. Sibling of
[[torchcell.datasets.ecoli.babu2014]], which is the second consumer of
`BacterialGeneInteractionExperiment` after this one lands.

### Two earlier conclusions about this row, and why both were wrong

The row carried `status="blocked"` with the prose "the article is closed access with no
PMC deposit ... the row is marked blocked pending a paywalled retrieval". Then the agent
writing the Babu 2014 loader concluded that Babu's Table S2 subsumes these 39 screens and
filed issue #794 recommending a provenance record instead of a loader.

Both were measured and overturned by
`experiments/036-dataset-fixes-before-kg-build/scripts/bacteria_subsumed_rows.py`, whose
record is `$DATA_ROOT/torchcell-raw/butlandESGAColiSynthetic2008/subsumption_record.json`.
This loader reproduces both measurements at build time rather than citing them:

| Measurement | Value | Where the build re-derives it |
|---|---|---|
| Released S scores | 8,073 x 39 = 314,847, all populated | `read_s_scores` + `score_block` |
| Records the served Babu store carries of them | 727, **0.23 percent** | `preprocess/served_partition.json` |
| Babu rows over these same 39 donors | 1,129 | the subsumption record |
| Butland high-confidence pairs Babu omits | 490, 321 of them non-essential | the subsumption record |
| Same-orientation rows with an identical score | 793 of 873 | the subsumption record |

The retrieval block was simply stale: the paper and all five SI files are mirrored. And
Babu publishes the high-confidence tail of a re-analysis, not a superset, so loading this
release adds records rather than a competing score definition.

### Scope: the whole matrix, and why the narrower options lose

The cell is the release's own unit of observation. Supplementary Table 4's banner reads,
verbatim, "Raw colony sizes (see Sheet 1), normalized median colony sizes (see Sheet 2),
|Z scores| (see Sheet 3) and interaction (S) scores (see Sheet 4) of each mutant gene pair
from 39 genome-wide screens without any filtering parameters."

Supplementary Table 3, the |Z|>=4 high-confidence set, is a **measured** proper subset: all
1,379 of its rows are located at their own (query, recipient, isolate) cell of sheet 4 and
all 1,379 S scores are identical, which `high_confidence_ledger` asserts. So:

- loading Table 3 instead stores a selection-biased tail (730 of its 799 non-essential
  pairs are aggravating) and discards the null distribution, which no other bacterial
  interaction store in torchcell carries;
- loading both duplicates 1,379 records;
- loading "the 321 pairs Babu omits" defines torchcell's content by another paper's
  editorial filter rather than by anything this release states.

The restriction to non-essential recipients is forced, not chosen: the 149 SPA-tag rows are
hypomorphs with no bacterial perturbation leaf (issue #792).

### Retention, in rule order

| Rule | Cells | Why |
|---|---|---|
| `spa_tag_recipient_has_no_bacterial_perturbation_leaf` | 5,811 | 149 SPA-tag rows x 39; issue #792 |
| `b_number_is_not_a_locus_tag_of_the_pinned_annotation` | 6,318 | 78 JW ids, `CSCR`, ambiguous ids |
| `b_number_remapped_by_the_annotation` | 4,407 | 57 b-numbers the assembly carries as a synonym of another locus; issue #753 |
| `self_pair_is_not_a_digenic_genotype` | 78 | each of the 39 queries is itself in the array, x 2 isolates |
| `released_score_contradicts_its_own_raw_colonies` | 395 | every raw colony reads 0 yet S is exactly 0.0 |
| `already_served_by_gene_interaction_babu2014` | 1,448 | the oriented pair is a served Babu record |

314,847 - 18,457 = **296,390** records: 143,651 aggravating, 144,953 alleviating, 7,786 at
exactly zero, over 39 query genes and 3,829 recipient genes.

### The partition against the served Babu store, proved both ways

The unit of the proof is the ORIENTED gene pair, because swapping query and recipient swaps
the `cat` and `kan` cassettes, which the Babu loader already relies on to keep its 102
reciprocal pairs as two records each.

- FORWARD: not one stored cell shares an oriented pair with a served record. Dropping a
  pair drops BOTH its isolate cells, since Babu's single score comes from the same colonies.
- REVERSE: 725 of the 727 served `Butland et al.` records ARE cells of this release. The 2
  exceptions are pinned by name, `b2528 -> b4486` and `b2531 -> b4486`, which this release
  names `b2528 -> b4344` and `b2531 -> b4344` (gene name `*`, absent from Genobase ver. 6)
  and rule 3 drops. No record of Babu's `This Study` screen set collides with any cell.

`SERVED_BUTLAND_PAIRS`, `SERVED_OVERLAP_CELLS` and `SERVED_PAIRS_NOT_IN_THIS_RELEASE` are
constants, so a drift in either store stops the build.

### The two Keio isolates are two strains

3,956 of the 3,968 non-essential genes are arrayed as two independently constructed
isolates, each with its own S score against each query. Collapsing them would mean
averaging (a derivation) or picking one (unsourced), so both are stored and the verbatim
"Strain Versions" token goes to the recipient leaf's `StrainConstruction.batch` -- the
field whose docstring cites Hillenmeyer's "some gene deletions were constructed more than
once, in different batches" and states that two records of one ORF whose construction
differs are two strains. Supplementary Table 1's footnote d calls the column "an in house
ID given for the two independent isolates", and the Results report the two isolates
"occasionally" disagree markedly, "suggesting a defect in one strain".

### What has no home on the schema, and what was refused

- **`n_samples` / `sample_unit`** (issue #793). The replicate count is NOT uniform:
  footnote g documents four colonies, two of this isolate in each of two replicate screens,
  but the raw sheet prints each cell's own colony measurements and they run 4 to 182 over
  2, 8 or 13 screens. The build counts them per record (258,364 at the documented four) and
  writes the distribution to `preprocess/replicate_structure.json`.
  `GeneInteractionPhenotype` is a SERVED class, so adding the field would force a full
  rebuild.
- **the per-cell |Z| score**, sheet 3. The class's only statistic field is
  `gene_interaction_p_value` and a |Z| is not a p-value; converting one through a normal CDF
  would be a derivation this release never published. Same #793 constraint.
- **the colony sizes**, sheets 1 and 2. No colony-size phenotype class exists. The raw
  sheet IS read, for the replicate counts and rule 5, but no value of it is stored. The
  normalized sheet also carries undocumented sentinels (`-100000` and `0` among otherwise
  positive sizes).
- **the S-score formula.** Both Table 3's footnote g and Table 4's footnote f say the score
  is "calculated ... based on an algorithm implemented for yeast SGA (Collins et al.,
  2006)". Collins 2006 (Genome Biol 7:R63) is NOT in the literature mirror, so the
  operational definition in `SOURCED_VALUES["score_definition"]` is as far as the chain
  reaches and no other source is substituted for it.
- **no PubMed id.** The mirror carries none for this DOI and the schedule row prints none,
  so `Publication.pubmed_id` is `None`; a DOI plus a URL satisfies the class and no id was
  invented or fetched live.
- **no dose for either marker drug.** The SI prints kanamycin at 50 ug/ml for the recipient
  pre-culture plate and 25 ug/ml "throughout" the mini-array protocol, and nothing at all
  for the genome-wide selection plates. A disagreement within one mirrored source is not a
  value, so every component carries `concentration=None` with both numbers in its note.

### Exact zeros are released values, not a sentinel

8,559 of the 309,036 non-essential cells carry an S of exactly 0 and the release documents
no zero convention, so the question was settled on the bytes:

- the zero-S cells average |Z| 0.1116, against 0.1214 for the cells whose |S| is nonzero
  but under 0.05, so a zero sits exactly where a tiny nonzero score sits;
- 8,071 of the 8,559 carry at least one non-zero raw colony measurement;
- an absent colony does NOT map to zero: 310 of the 798 cells whose every raw colony reads
  0 carry a nonzero S, down to -17.8.

So zeros are stored as released, and only the 395 cells that are BOTH all-zero-colony AND
exactly-zero-score are dropped (rule 5), because there the release contradicts itself.

### One locus, two spellings

The matrix's query header names `b1922` `rpoF` while the array roster names it `fliA`. Both
resolve to `b1922` in the pinned annotation, so neither is wrong, but storing both would
split one locus into two perturbation identities and fail the shared
`canonical_gene_names` rule, which reported it over the 7,663 stored records that touch
`b1922`. The ROSTER spelling wins, because the
roster is the release's own per-strain name column and the matrix's recipient block IS that
roster (asserted at build time), while the header is one cell per screen. The disagreement
is recorded in `preprocess/gene_name_disagreements.json`, where
`n_records_renamed` is **7,587**: the records whose QUERY is `b1922`, the other 76 of the
7,663 already carrying the roster name because `b1922` is their recipient. The 155 rows the
roster leaves unnamed (`*`) are asserted to be dropped by an identifier rule, so no record
stores `*`.

### Verification, measured on the dev store

`python -m torchcell.datasets.ecoli.butland2008 verify` -> PASS, 19 of 19 sourced values
audited. Seven of them are quoted from spreadsheet cells, which `audit_sourced_value` reads
as UTF-8 text and an OLE2 `.xls` is not, so `audit_workbook_quote` runs the same two checks
(the pinned sha256, then the quote's presence) against the cells.

| Level | Rule | Result |
|---|---|---|
| L0 | structural | 296,390 records validated |
| L1 | count | observed 296,390, expected 296,390 |
| L1 | `digenic_pair_of_one_query_and_one_recipient` | 296,390 of 296,390 |
| L1 | `recipient_leaf_carries_its_keio_isolate` | 148,408 Isolate 1, 147,982 Isolate 2, 0 wrong |
| L1 | `provenance_gaps` | 1,185,560 documented gaps over 296,390 of 296,390 records |
| L1 | `canonical_gene_names` | 3,829 systematic names, one spelling each |
| L2 | `gene_interaction_equals_its_matrix_cell` | 296,390 of 296,390 |
| L2 | `uncertainty_sanity` | 0 labeled uncertainties, none a zero dispersion |
| L3 | `signed_unclamped_interaction_score_with_zero_reference` | 143,651 / 144,953 / 7,786 zero, reference 0.0 |
| L3 | `partitioned_from_the_served_babu2014_store` | 0 of 296,390 share a served oriented pair |
| L3 | `compound_identity` / `media_compound_identity` / `media_membership` | pass |
| L3 | `provenance_audit` x 19 | 12 against OCR text, 7 against workbook cells |
| L4 | `gene_containment_mg1655_locus_tags` | 3,829 of 3,829 |

Dev store: `$DATA_ROOT/data/torchcell/gene_interaction_butland2008`, 1.2 GB processed.

### What is NOT done here

- The bacterial schedule row is left alone. Its `status="blocked"` and its sourcing prose
  are now both stale, but issue #794's own scope note reserves that edit as a curation
  decision, and Babu 2014's row still reads `status="candidate"` after its loader landed,
  so the table is a discovery artifact rather than a build-state record.
- No knowledge-graph build and no live rebuild. Adding this dataset is additive, so it is
  an incremental-admission candidate (see [[torchcell.knowledge_graphs.incremental-admission]]),
  but the admission check and the increment are a separate step.

## 2026.10.09 - Every record stores its OWN measured colony count (#793)

`GeneInteractionPhenotype` gained `n_samples`, `sample_unit`,
`gene_interaction_uncertainty` and its type ([[torchcell.datamodels.schema]], section
"2026.10.09"), so the per-cell colony count this release prints is a field of every
record instead of a distribution beside the build.

The count is MEASURED per record off the raw colony-size sheet (`colony_counts`), never
the documented four. The documented design is four colonies and it is the MODE rather
than the rule, which is why a constant would be wrong in one direction only: 258,364 of
296,390 records rest on four colonies and the other 38,026 on 8 to 182, so storing 4
everywhere would overstate the precision of 38,026 records and understate it for none.
The measured count carries no such bias, because the release states it per cell.
`phenotype(score, n_samples)` takes the count and never defaults it.

Sourced design (`SOURCED_VALUES["replicate_design"]`, `si/si5.xls` footnote g, sha256
`74a6ea3a0373fa6e1776b0becb4212f29b9876647b96327db5222dcd68a8822b`), verbatim:

> g In the genome-wide screen, each recipient deletion mutant is pinned twice leaving
> four replicate recipient colonies representing two "Isolate 1" and two "Isolate 2"
> versions of the strain. The number "1" represent the first replicate of the genome-wide
> screen, while the number "2" represent the second replicate of the same screen.

### Measured, before and after

The change is additive, so no record count moves:

| | records | `n_samples` | `sample_unit` |
|---|---|---|---|
| before (origin/main) | 296,390 | absent from the class | absent from the class |
| after | 296,390 | measured, 20 distinct values | `colony` on 296,390 of 296,390 |

The stored histogram, read back off the built LMDB, equals
`preprocess/replicate_structure.json`'s `measured_n_samples_counts` exactly:

| colonies | records | | colonies | records |
|---|---|---|---|---|
| 4 | 258,364 | | 48 | 38 |
| 8 | 16,533 | | 52 | 445 |
| 12 | 1,394 | | 64 | 106 |
| 16 | 10,824 | | 78 | 37 |
| 20 | 954 | | 80 | 26 |
| 24 | 74 | | 96 | 2 |
| 26 | 6,989 | | 104 | 101 |
| 28 | 35 | | 112 | 1 |
| 32 | 440 | | 130 | 24 |
| | | | 156 | 2 |
| | | | 182 | 1 |

A builder that defaulted to 4 would read 296,390 at 4 and 0 everywhere else, which is
what the new test asserts against.

### What stays in the ledger, and why

`preprocess/replicate_structure.json` keeps the documented design, the full measured
distribution, the per-cell REPLICATE-SCREEN counts (2 for 281,231 records, 8 for 7,560,
13 for 7,599) and the verbatim footnote with its sha256. The phenotype has ONE replicate
axis and the colony is the unit the score is an average over, so the screen count has no
field; recording it beside the build is the honest place for it, and the file is also
what makes the stored property auditable rather than asserted.

Sheet 3's per-cell `|Z|` score still has no slot. The quartet #793 added carries a
DISPERSION of the score plus its replicate design, and a `|Z|` is a TEST of the score, so
storing it under `gene_interaction_uncertainty` would name it something it is not. It
stays in `preprocess/high_confidence_pairs.json` for the high-confidence rows.

The reference phenotype carries no replicate design: it is the unperturbed chassis scoring
0 by construction, not a measured cell.

### L0 to L4

Verified on a build of this branch's loader, PASS, 33 rows, 0 failures. The rows this
change touches or that carry the counts:

| level | rule | result |
|---|---|---|
| L0 | `structural` | 296,390 records validated |
| L1 | `count` | observed 296,390, expected 296,390 |
| L1 | `digenic_pair_of_one_query_and_one_recipient` | 296,390 of 296,390 are two distinct loci, one `cat` query and one `kan` recipient |
| L1 | `recipient_leaf_carries_its_keio_isolate` | Isolate 1 148,408, Isolate 2 147,982; 0 wrong |
| L1 | `provenance_gaps` | 1,185,560 documented gaps over 296,390/296,390 records |
| L2 | `gene_interaction_equals_its_matrix_cell` | 296,390 of 296,390 stored scores equal their Supplementary Table 4 cell |
| L2 | `uncertainty_sanity` | 0 labeled uncertainties, none a zero dispersion; 296,390 records report `n_samples >= 2` with no uncertainty |
| L3 | `signed_unclamped_interaction_score_with_zero_reference` | 143,651 aggravating, 144,953 alleviating, 7,786 at exactly zero, reference scores [0.0] |
| L3 | `partitioned_from_the_served_babu2014_store` | 296,390 stored against 38,579 served oriented pairs; 0 share one |
| L3 | `provenance_audit` x 19 | every sourced value backed by a verbatim quote or cell |
| L4 | `gene_containment_mg1655_locus_tags` | 3,829 of 3,829 perturbed loci are MG1655 GenBank loci |

The `uncertainty_sanity` row reads differently now for the same reason it does in Babu:
no record declared `n_samples` before, so the "reports `n_samples >= 2` with no
uncertainty" count was 0 by absence and is 296,390 by measurement. Butland releases a
colony count and no per-cell dispersion, and that is the honest reading of it.

### Why the canonical dev store was NOT rebuilt by this branch

The L0-to-L4 run above is from an isolated root whose `served_root` is a Babu build of
THIS branch (38,579 records), so the partition proof is internally consistent. The
canonical `$DATA_ROOT/data/torchcell/gene_interaction_babu2014` currently holds the
parallel #792 branch's build (41,988 records with the marked-allele leaf), and a Butland
rebuild reads that store for its partition proof, so a canonical Butland rebuild belongs
after both branches are on `main`. The KG 4.0 full build remakes both regardless.

`build_manifest` on the shared tree reports `gene_interaction_butland2008` STALE on
exactly one symbol, `GeneInteractionPhenotype`, which is this change.

Related: [[torchcell.datasets.ecoli.babu2014]], [[torchcell.datamodels.schema]].
