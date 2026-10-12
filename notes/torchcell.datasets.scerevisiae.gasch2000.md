---
id: rfo1jbn49ekwipzocmp1j12
title: Gasch2000
desc: ''
updated: 1791806977566
created: 1791806977566
---

## 2026.10.12 - The release, its schema leaf, and why the loader is blocked (#989)

Gasch et al. 2000, Mol Biol Cell 11:4241 (doi 10.1091/mbc.11.12.4241, `gaschGenomicExpressionPrograms2000`, Zotero `GNF43TXA`), yeast schedule row 75. The environmental stress response arrays: every column of the authors' release is a log2 ratio of a sample (Cy5) against a reference (Cy3) on one spotted array, with no channel signal, SE or replicate count.

### Status

- **Schema: landed.** `MicroarrayExpressionRatioPhenotype`, a leaf of `TwoChannelMicroarrayExpressionPhenotype` beside the served `MicroarrayExpressionPhenotype` and Hu 2007's `MicroarrayExpressionScorePhenotype` ([[torchcell.datamodels.schema]], 2026.10.12).
- **Raw deposit: recorded** (below), retrieved by the registered `direct_url` retriever.
- **Loader: BLOCKED on G2.** The paper's Table 1 gives the host as `DBY7286` "MATa ura3-52 GAL2" with source "Ferea et al., 1999" and states no background; no paper in the library mirror states one (searched every `paper.md` for `DBY`: Gasch, Hackett 2020, Yoshida 2012, Singh 2023). The deferral is Ferea et al. 1999, PNAS 96:9721 (doi 10.1073/pnas.96.17.9721, PMID 10449761, PMC22277). It is not scriptable: `https://www.pnas.org/doi/pdf/10.1073/pnas.96.17.9721` answers 403 to a script, and PMC efetch returns no body ("The publisher of this article does not allow downloading of the full text in XML form."). `StrainBackground.reference_strain` is pinned to S288C, so writing the loader would assert a background no source states. The verdict stays `blocked: provision` (`database/candidates/gaschGenomicExpressionPrograms2000.json`, composed by the yeast Zotero audit); `candidate-gate gate` at this branch refuses the row at G2 as naming no host token at all.
- **Owner action:** a manual-once deposit of the Ferea 1999 PDF into the library mirror; if it names the background (and the GAL2 allele, which S288C lacks), a `DBY7286` host token for G2 and the loader can follow.

### Raw mirror deposit (`$DATA_ROOT/torchcell-raw/gaschGenomicExpressionPrograms2000/`)

| path | role | bytes | sha256 | source |
|---|---|---|---|---|
| `data/complete_dataset.txt` | raw_data | 6,306,835 | `3050664e0deba976b70b954676675cce74d3288198d850abb6fb123bac58daa7` | Wayback `20060110024849id_` of `genome-www.stanford.edu/yeast_stress/data/rawdata/complete_dataset.txt` |
| `si/materials.pdf` | si_pdf | 27,908 | `2fb093bb1a9875652ddfaee4274dda70de745106ab6095ea331a0b76361369b4` | Wayback `20030608072144id_` of `.../yeast_stress/materials.pdf` |
| `si/materials.txt` | si_text | 17,648 | `5402523a710262509ab1cc7bfbf84b09f4a74f145b41fd3c2c715aed73ecaf73` | pypdf text layer of `si/materials.pdf` |
| `si/data.shtml` | si_data | 4,988 | `61e6e743e1ec757f7fa755feb21f746956c525fd29c9c7db56cb040f0d340c47` | Wayback `20010210184407id_` of `.../yeast_stress/data.shtml` |
| `si/arraylist.html` | si_data | 25,739 | `594157fd008117de3a77897d59811ef3b62cd7300b621752bcf5255dec08389b` | Wayback `20010211101344id_` of `.../yeast_stress/arraylist.html` |

The paper defers its per-experiment detail to that supplement: "(Additional details, including descriptions of duplicated experiments and appropriate reference citations, can be found on the web supplement, at the address given above.)" (`paper.md`, sha256 `e22b3e01f5664a00b9a6fdcd4c78bacd5803e5c31f285f45b98c47a871a98fbb`). The data page states the released quantity: "The normalized, relative transcript levels (R/G ratios)" ... "represented in log2 space (RAT2N)" (`si/data.shtml`, the sentence spans a line break). The per-array ScanAlyze `*.out.gz` files are on the Wayback capture and were not deposited: recovering channel intensities from them (#989 option 3) is a separate future row.

### The release by reference

From `experiments/036-dataset-fixes-before-kg-build/scripts/gasch2000_release_inventory.py` ([[experiments.036-dataset-fixes-before-kg-build.scripts.gasch2000_release_inventory]]), which re-reads every quoted sentence from the pinned bytes; results in `experiments/036-dataset-fixes-before-kg-build/results/gasch2000_release_inventory.json`. 173 array columns over 6,152 genes, 1,032,274 non-null log2 ratios (the issue's count).

| reference | columns | values |
|---|---|---|
| the same culture before the change | 121 | 717,347 |
| another culture of known identity | 7 | 42,845 |
| an equal-mass pool of the series' samples | 35 | 212,118 |
| another publication's arrays (DeRisi 1997 diauxic shift) | 7 | 41,898 |
| not resolved by the methods | 3 | 18,066 |

| series | columns | values | reference |
|---|---|---|---|
| hs-1 | 8 | 42,141 | same culture before change |
| hs-2 | 7 | 42,797 | reference pool |
| 37C to 25C | 5 | 30,534 | same culture before change |
| variable heat to 37C | 5 | 30,463 | same culture before change |
| 29C to 33C | 3 | 17,974 | same culture before change |
| 33C vs 30C | 1 | 6,117 | unresolved |
| 29C sorbitol to 33C sorbitol | 3 | 18,393 | same culture before change |
| 29C sorbitol to 33C no sorbitol | 3 | 18,333 | same culture before change |
| H2O2 constant | 10 | 58,942 | same culture before change |
| menadione | 9 | 53,569 | same culture before change |
| dtt-1 | 8 | 42,720 | same culture before change |
| dtt-2 | 7 | 42,195 | reference pool |
| diamide | 8 | 48,287 | same culture before change |
| hyper-osmotic | 7 | 42,531 | same culture before change |
| hypo-osmotic | 5 | 30,577 | same culture before change |
| hypo-osmotic steady state | 1 | 5,936 | unresolved |
| amino acid starvation | 5 | 30,159 | same culture before change |
| nitrogen depletion | 10 | 60,493 | same culture before change |
| diauxic shift | 7 | 41,898 | other publication |
| ypd-2 | 10 | 60,127 | same culture before change |
| ypd-1 | 12 | 70,918 | same culture before change |
| mutant heat | 5 | 30,568 | same culture before change |
| mutant H2O2 | 5 | 30,618 | same culture before change |
| MSN2 MSN4 overexpression | 2 | 12,278 | other culture (DBY7286 with pRS416) |
| YAP1 overexpression | 1 | 6,013 | unresolved |
| car-1 | 6 | 36,189 | reference pool |
| car-2 | 7 | 42,822 | reference pool |
| ct-1 | 5 | 30,567 | other culture (the 33 C culture) |
| ct-2 | 8 | 48,115 | reference pool |

The issue's "every release carries log2 ratios against the same culture at time zero" holds for 121 of 173 columns. The pool series are released against the pool, not "zero transformed" against their time-zero array: the arrays that would read 0 everywhere under that transform carry 6,039 to 6,131 values with 91 to 170 zeros each (`Heat Shock 000 minutes hs-2`, `dtt 000 min dtt-2`, `YP glucose vs reference pool car-2`, `steady state 33 dec C ct-2`).

### Decisions the loader will need (open, not taken here)

- **Pool-referenced arrays (35).** A pool of every sample in a series has no single `Environment`, so `ExperimentReference.environment_reference` cannot name it honestly. Refuse them with a count, or decide a typed pooled-reference representation (a schema question for the owner). Zero-transforming them against the time-zero array is the authors' figure convention, a derived ratio of two arrays, not the release.
- **Labels the methods contradict.** hs-1: the release labels 5, 10, 15, 20, 30, 40, 60, 80 min, the methods "5, 10, 15, 20, 30, 40, 50, and 60 minutes". `33C vs. 30C - 90 minutes` sits in a series that starts at 29 C and is sampled at 5, 15 and 30 minutes. ypd-1 runs to 28 d where the methods list 2 h to 5 d. One msn2 msn4 heat array is labeled "(real strain)" and its partner is not.
- **Refusals by source.** The 7 diauxic-shift columns are DeRisi et al. 1997's arrays ("along with previously-published data (DeRisi et al., 1997)"), so they belong to a DeRisi loader, not to this citation. The YAP1 overexpression column has no construct in the methods or Table 1.
- **Vocabulary.** H2O2, menadione sodium bisulfite, DTT, diamide and sorbitol as compounds; the MSN2 and MSN4 plasmid overexpression (the promoter is not stated in the methods: a typed gap); `yap1::KANMX4` and `msn2::KANMX4 msn4::URA3` deletions; the media shifts (YPD, SCD, YNB without amino acids, limiting ammonium sulfate).
- **`n_replicates`** is a typed gap: one column per array, no statement of spots averaged per gene.
- **Keys, under the owner rule of 2026-10-12** (the data is the data; no released row dropped on genome or annotation grounds; no resolution status): every one of the 6,152 rows is keyed by its released `UID`, verbatim, with no resolution against R64. The rule covers rows and genes; whether it also lifts the G2 host gate (storing `DBY7286` as released, with no typed background) is an owner question, not taken here.

No store exists, so there is no L0-L4 table.

### What unblocks it (owner decision, relayed 2026-10-12)

Gasch 2000 stays blocked at G2: the host strain's background is unstated and the one source that would state it is walled. The pooled-reference design waits with it. Two things unblock the loader:

1. **The host background.** A manual-once deposit of Ferea et al. 1999, PNAS 96:9721, https://www.pnas.org/doi/pdf/10.1073/pnas.96.17.9721 (PMC22277; the script gets 403 at pnas.org and PMC withholds the XML), if it states DBY7286's background.
2. **The 35 pool-referenced arrays** (hs-2 7, dtt-2 7, car-1 6, car-2 7, ct-2 8; 212,118 values), which need a typed pooled-reference representation before they can be stored.

Nothing is refused on a count, because nothing is loaded yet. The parent placement (no graph class for `TwoChannelMicroarrayExpressionPhenotype`) and Hackett 2020's leaf under it are owner-approved; Hackett's and Kemmeren's load-time re-keying belongs to the yeast group-3 reversal branch (#1076).
