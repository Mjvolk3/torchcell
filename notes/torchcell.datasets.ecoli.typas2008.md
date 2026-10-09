---
id: ilkunie8epv8atfzyobgj0y
title: Typas2008
desc: ''
updated: 1791557368341
created: 1791557368341
---

## 2026.10.09 - Row 37 settled as a provenance record (#826)

Row 37 of the fifty bacterial rows is settled, not loaded:
`torchcell/datasets/ecoli/typas2008.py` registers no dataset and is the record of why.
The closer the umbrella issue asked for was "a provenance record like Butland's, or
close"; this is the record, on the MCF2Chem pattern
([[torchcell.datasets.ecoli.cai2023]]). Status stays `blocked`, loaded records 0,
`accession_confirmed` flips to True.

Measurement script and committed results:
`experiments/036-dataset-fixes-before-kg-build/scripts/typas2008_release_loadability.py`
(commit `bc18d6b9c`).

### The accession half was stale, and that is the Butland lesson repeated

Enumerating the PMC Article Datasets bucket for `PMC2700713` returns NINE objects of
which exactly ONE is supplementary, `NIHMS95293-supplement-Supp_Info.pdf`, and the
article's own declared supplementary list names exactly that one href.
Published-not-mirrored and mirrored-not-published are both empty, so the deposit is
mirrored in full and nothing sits behind a paywall. The retrieval is the scriptable
`pmc_cloud` route, run 2026-10-07.

| artifact | sha256 | bytes |
|---|---|---|
| `si/si1.pdf` | `bd531ec2ee865506b9a1c5decb3dd7d3f057896dd65f2b766ae8090b8c0c39da` | 1,197,068 |
| `si/si1.md` (MinerU 2.7.6, pipeline, 200 dpi) | `6e41b6083e483f90cf95f5dbf5fccaebfcdcb503b801a271e7b9bd44f944e85c` | 38,193 |

### The blocking half survives: the release prints no interaction score

The one deposited PDF holds ten tables, and exactly ONE carries a number at all.

| table | what it holds | numeric cells |
|---|---|---|
| 4 panel tables (OCR indices 0 to 3) | Supplementary Figure 4's axis gene labels | 0 |
| index 4 | Supplementary Table 1: 4 synthetic-lethal pairs with a marker co-inheritance percentage | 8 |
| index 5 | Supplementary Table 2A: 23 `pal` partners as the terms `neg (sick)`, `neg (lethal)`, `pos` | 0 |
| index 6 | Supplementary Table 2B: 15 `yraP` suppressors over gene name, ECK nr, Location, Function | 0 |
| index 7 | a prose supplementary table with no value column | 0 |
| index 8 | the strain table | 0 |
| index 9 | the primer table | 0 |

42 released pairs: 23 + 15 + 4. Four carry a number and its statistic is
`% Co-inheritance of both markerswhen host is:`, a marker co-transduction check of
Supplementary Table 1 and not a colony-size interaction score. The Table 2 caption is the
release's own value surface, verbatim: "Neg stands for negative interactions and pos for
positives."

### `GeneInteractionPhenotype` has no categorical mode, probed not assumed

`gene_interaction` is annotated `float` and required. The probe is re-run in the paired
test, so a future categorical mode makes the test fail rather than leaving a stale
refusal on the record:

| attempt | verdict |
|---|---|
| the released term as the value | REFUSED, `float_parsing`, "Input should be a valid number, unable to parse string as a number" |
| no value | REFUSED, `missing`, "Field required" |
| a signed float | ACCEPTED |

So the blocker is the release's content, not the leaf.

### The one quantitative block is a figure

The 12 by 12 cross is four heat-map panels of Supplementary Figure 4 (LB-384, LB-1536,
M9-384, M9-1536). Its 12 axis genes ARE recoverable from the panel labels (`surA`,
`ybaY`, `ycbS`, `ompC`, `yraI`, `cpxR`, `degP`, `pal`, `ompA`, `yfgL`, `yraP`, `basR`)
and its 66 distinct pairwise doubles are colour cells that no table of the deposit
carries. A stored score would be read off a colour.

### Not subsumed, which is the opposite of what Butland turned out to be

Measured against Babu 2014's pinned Table S2
(`0789563ada0db3e349bb7e5396311f9d705ea165090733803c009acfa7c95b96`, 42,705 rows, 42,592
unordered pairs, 163 donors): 0 of the 38 screen pairs are in it, NEITHER `pal` NOR
`yraP` is among Babu's donors, and only 2 of the 42 released pairs overlap, the
verification pairs `degP`/`surA` and `pal`/`ompA`. Babu does hold other partners of both
query genes as a recipient (`pal`: ddlA, efp, malQ, ompA, oppA, thiM, tig, ycbX, ygcI,
yghD; `yraP`: csdA, recC), which is why the overlap is reported pair-level rather than
gene-level. Closing this row therefore loses records no other store holds, and that cost
is stated rather than hidden.

### What would reopen it

A released per-pair colony-size score: the numbers behind Supplementary Figure 4's four
panels, or a deposit of the genome-wide M9-glycerol screen the Supplementary Table 2A
caption describes. The PMC deposit is complete, so this is an author release, not a
retrieval this project can re-run.
