---
id: zez13lu9evxk0ho1zhfhlcj
title: Kuzmin_ladder_census
desc: ''
updated: 1788499824115
created: 1788499824115
---

## 2026.09.04 - Does the ladder exist anywhere in Kuzmin? Yes, but it carries no interaction

Script: `experiments/010-kuzmin-tmi/scripts/kuzmin_ladder_census.py`

Written before committing GPU-days to a new inference space. The pcl6 screen gave
3 monotone paths out of 5,502, which is one query of roughly four hundred and
cannot distinguish "rare everywhere" from "concentrated somewhere we have not
looked." This asks the whole corpus from measurements.

### Scope

329,462 Kuzmin trigenic records with a deletion array over 421 query doubles.
286,002 have a complete four-rung lattice, which is 1,716,012 orderings. Every
rung is published: triple and query double in screen, mixed doubles from
`DmfKuzmin2018/2020` then `DmfCostanzo2016` at 30 C, singles from the Kuzmin SMF
tables then Costanzo.

### The ladder exists

| | Count | Share |
|---|---|---|
| Complete lattices | 286,002 | |
| Beating wild type | 44,462 | 15.5% |
| With a monotone route | 12,267 | 4.3% |
| Monotone routes | 22,218 | 1.3% of 1,716,012 |
| Query doubles hosting any | 185 | of 372 |
| Monotone AND called positive | 27 | |
| Monotone AND strong positive | **0** | |

So it is not rare. It is also not epistatic. Across 286,002 measured triples not
one both climbs monotonically and clears tau > +0.16 at P < 0.05, and only 27
clear the ordinary +0.08 call. Median tau of a monotone triple is +0.0071 against
-0.0089 for the corpus.

What the ladders ride is a **beneficial query double**. 83.9 percent of monotone
triples sit under a query double fitter than wild type against 23.4 percent of
all triples; median query fitness 0.992 for a hosting query against 0.859 for a
non-hosting one. An ordering that never steps back is mostly a report that the
first two deletions were already good. That is the greedy result.

### The demonstration is the valley, not the ladder

A monotone route is exactly what greedy stack-and-keep can find, since it never
accepts a loss. Read the same census that way:

| Triples beating wild type | 44,462 | |
|---|---|---|
| Reachable by greedy stacking | 12,267 | 27.6% |
| Behind a non-monotone step | **32,195** | **72.4%** |
| ...with a called positive tau | 97 | |
| ...with a strong positive tau | 24 | |

**72.4 percent of the fitness wins in the corpus are invisible to a search that
accepts only improvements.** That is the claim a predictor is actually needed
for, because the endpoint has to be predicted; no sequence of local decisions
arrives at it.

Caveat that must travel with the number: the barrier is usually shallow. On the
best available route the median dip sits 0.004 **above** wild type and the 90th
percentile only 0.033 below it. The obstacle is a step backwards, not a lethal
intermediate. That is good for construction cost and it means the claim has to be
stated as non-monotonicity, not as a deep fitness valley.

### Why this matters before more inference

Everything above is measured and none of it depends on the transformer. It is a
specification for the model rather than a result of it: score a predictor on
whether it recovers the 32,195 valley wins, which greedy provably cannot, and the
demo no longer needs a strong trigenic interaction to exist in a monotone triple,
which corpus-wide it does not.

### Figure

![](./assets/images/010-kuzmin-tmi/kuzmin_ladder_census.svg)

### Outputs

- `experiments/010-kuzmin-tmi/results/kuzmin_ladder_census_triples.csv`
- `experiments/010-kuzmin-tmi/results/kuzmin_ladder_census_queries.csv`
- `experiments/010-kuzmin-tmi/results/kuzmin_ladder_census.json`
