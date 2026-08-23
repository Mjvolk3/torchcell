---
id: r05evltwxxw8vx79gonihea
title: triple_design_rank_sampling
desc: ''
updated: 1787527417254
created: 1787527417254
---

## 2026.08.23 - Six selection strategies over the 39 constructible triples

Picks the triples for the first trigenic round and shows where on the ranked target list
each candidate design samples. `capped` is the design chosen; the bench list it produces is
[[experiments.W019-echo-crispr-array.build-list]] and the rationale is
[[experiments.W019-echo-crispr-array.next-strains-to-construct]]. Audited independently by
[[experiments.W019-echo-crispr-array.scripts.verify_triple_build_list]].

Run from repo root:

```bash
PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
  experiments/W019-echo-crispr-array/scripts/triple_design_rank_sampling.py
```

Deterministic -- candidate iteration is sorted, so repeated runs are byte-identical. Verified
2026.08.23 by md5 of all four output CSVs across a re-run.

### The basis is 11 genes, not 10

`optimized_doubles_setcover_constructed_10.py` froze a TEN set = panel-12 minus YIL174W and
YLR104W, because the plan then was to swap LCL2 (YLR104W) for LCL1. Run 4 did not do that:
its `s7` is YLR104W, so YLR104W is both built and a prediction node. The constructible basis
is 11 genes and the target set is **39** triples, not 31. YLR104W adds 8, one at prediction
0.7012. YLR313C (SPH1) is also built but was never a panel node, so it contributes no
predicted triple.

`YKL033W-A x YJR060W` failed to construct and is excluded from every candidate set. It
blocks 0 of the 39 targets, so the failure costs no tau, but it must never be proposed.

### The six strategies

| mode | objective |
|---|---|
| `rank` | walk the ranked targets top-down |
| `count` | greedy on targets made scorable per new double |
| `balanced` | greedy on a concave gene-coverage objective, sum_g sqrt(count) |
| `uniform` | balance gene participation, penalised by marginal double cost |
| `capped` | the `cap` best-predicted triples touching a zero-trigenic-data gene, then fill with clean triples that add the fewest new doubles |
| `no_ylr` | count-greedy with YLR312C-B excluded |

`capped` is the only one that holds exposure to the two zero-trigenic-data genes (YER079W,
YLR312C-B) to a chosen level, 6 of 20, against 15 for `balanced` and 17 for `rank`, while
still covering all eleven genes. It costs the most strains, 45 against 40, because
constraining the flagged genes forces it down the ranking.

Two structural limits set the size: only 16 of 39 targets are clean, and only 14 of those 16
have an existing double to build from. A cap of 5 therefore tops out at 19 parallel triples;
cap 6 reaches 20 with every strain still parallel.

### Outputs

- `results/triple_design_rank_sampling_summary.csv` -- per-strategy summary
- `results/triple_design_rank_sampling_selection.csv` -- the picked triples per strategy
- `results/triple_design_rank_sampling_gene_frequency.csv` -- per-gene participation
- `results/triple_build_construction_check.csv` -- parent routes per selected triple

![](assets/images/W019-echo-crispr-array/triple_design_rank_sampling.svg)

![](assets/images/W019-echo-crispr-array/triple_design_gene_frequency.svg)
