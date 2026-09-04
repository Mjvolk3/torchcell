---
id: drejdu595id0gjdw5ptkn8n
title: Pcl6_deletion_paths
desc: ''
updated: 1788495763999
created: 1788495763999
---

## 2026.09.03 - Every measured 1 to 2 to 3 deletion path through the pcl6 pcl7 screen

Script: `experiments/010-kuzmin-tmi/scripts/pcl6_deletion_paths.py`

### Why this screen is the one that can be walked

`YER059W` (PCL6) is the hub behind the positive-interaction signal in `inference_1`,
and every `YER059W` record in the 010 build comes from one Kuzmin 2020 query double,
`YER059W` + `YIL050W` (PCL7), its whole genome duplication paralog. So every triple in
that screen is `{PCL6, PCL7, X}` for an array gene X, and the whole 1 to 2 to 3
lattice is measured rather than predicted. That is the difference between this and the
`inference_1` triples, whose endpoints exist only as model output.

The screen holds 1,097 trigenic records, 919 with a deletion array and 178 with a
temperature-sensitive allele. Deletions are the primary set here, because the question
is about removing genes. After requiring all three singles and all three doubles,
917 triples survive with a complete lattice, giving 5,502 paths at six orderings each.

### Where each rung comes from

Mixing normalizations is how a ladder gets corrupted, so the source is recorded per
rung rather than assumed.

| Rung | Source | Note |
|---|---|---|
| triple `{PCL6,PCL7,X}` | `TmfKuzmin2020` combined-mutant fitness | in screen |
| double `{PCL6,PCL7}` | `TmfKuzmin2020` query fitness, 0.9063 | one value for the whole screen |
| doubles `{PCL6,X}`, `{PCL7,X}` | `DmfKuzmin2020` (3,668 paths), `DmfCostanzo2016` at 30 C otherwise | |
| singles | `SmfKuzmin2020`, `SmfCostanzo2016` at 30 C otherwise | |

The two single-mutant sources disagree on the query genes, and the disagreement is
load bearing. Kuzmin's own table gives pcl6 0.9153 and pcl7 0.8956; Costanzo gives
1.0245 and 0.9950. The paralog double's digenic term is +0.0866 on the first pair and
-0.1131 on the second, so the sign of that interaction depends on which screen you
believe. Kuzmin's singles are used because they share a normalization with the double
and the triple that sit above them in the same ladder.

### The ladder does not climb

A path is scored as having no backwards move when every deletion leaves fitness at
least as high as the strain before it, starting from wild type at 1.0.

| Criterion | Paths of 5,502 | Triples of 917 |
|---|---|---|
| no backwards move | 3 | 3 |
| no step below 1 SE | 228 | 188 |
| no backwards move after the first deletion | 428 | |

The first deletion alone ends 90.3 percent of paths, because the median array single
sits at 0.973 and both query singles sit near 0.90. Of the survivors the second
deletion removes another 54 percent. Three paths out of 5,502 clear all three steps,
and all three begin with the array gene rather than a query gene, since pcl6 and pcl7
both cost fitness on their own.

The three are `kar9` (`YPL269W`) then `pcl7` then `pcl6` reaching 1.0587, `nuc1`
(`YJL208C`) then `pcl6` then `pcl7` reaching 1.0258, and `chs7` (`YHR142W`) then
`pcl7` then `pcl6` reaching 1.0260. None of the three is a called interaction at
P < 0.05; the closest, `kar9`, sits at P = 0.0502.

### A positive interaction is not a fitter strain

43 of 917 triples finish above wild type, the best at 1.0834. Twelve triples carry a
called positive interaction at tau > +0.08 with P < 0.05, and exactly one of those
twelve also finishes above wild type: `puf4` (`YGL014W`), tau +0.086 at P = 0.022,
endpoint 1.0834. It is both the best endpoint in the screen and the only called
positive interaction that beats wild type, and two of its six routes stay within 1 SE
of never stepping back.

The rest of the called set runs the other way. `pho80` (`YOL001W`) carries the second
largest tau in the screen at +0.324 with P = 0.0004 and finishes at 0.2852. PHO80 is
the cyclin partner of Pho85p, of which PCL6 and PCL7 are two other cyclins, so the
interaction is exactly where a mechanism would put it. Positive tau there means the
triple is less sick than the multiplicative expectation, not that it is healthy.
Across all 917 triples tau and endpoint fitness correlate at r = +0.14.

This is the part that matters for panel design. Selecting on predicted trigenic
interaction and selecting on fitness are close to independent in this screen, so a
panel picked on interaction alone will mostly return sick strains.

### Figures

![](./assets/images/010-kuzmin-tmi/pcl6_deletion_paths_all.svg)

All 5,502 deletion paths, where the ladder breaks, and interaction against endpoint.

![](./assets/images/010-kuzmin-tmi/pcl6_deletion_path_panels.svg)

The six highest-fitness triples, all six deletion orders each.

### Outputs

- `experiments/010-kuzmin-tmi/results/pcl6_deletion_paths.csv`
- `experiments/010-kuzmin-tmi/results/pcl6_deletion_paths_triples.csv`
- `experiments/010-kuzmin-tmi/results/pcl6_deletion_paths_summary.json`

### One incidental defect worth carrying forward

Constructing any `torchcell.datasets.scerevisiae` dataset resets matplotlib's
rcParams to 16 pt DejaVu Sans. A script that sets the figure style at module level,
which is the usual pattern, silently produces a figure with 16 pt ticks. The style is
therefore applied inside each plotting function here. This is the same class of defect
as commit 31d6fcc7.
