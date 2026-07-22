---
id: j1u5s9mnfh63e8cgpvn76hd
title: Query_fig6
desc: ''
updated: 1784701290415
created: 1784701290415
---

## 2026.07.22 - WS4 Fig-6 production/metabolite build + WS11b co-location census

Fig-6 is the "deletion recommendations for a production target" panel, paired
with WS11b ("does adding a metabolome label improve production prediction"). This
workstream mirrors the WS2 Fig-3 build pattern:

- `experiments/019-simb-multimodal/scripts/generate_fig6_cql.py` -> writes
  `queries/fig6_build.cql` (UNION ALL of the 7 datasets).
- `experiments/019-simb-multimodal/scripts/query_fig6.py` -> live-DB census +
  `Neo4jCellDataset` build (converter=None, `MeanExperimentDeduplicator`,
  gene-set-keyed `GenotypeAggregator` (WS2b), `Perturbation` processor (WS1)) +
  post-aggregation LMDB census + a HeteroData label-shape check.
- `results/fig6_overlap_census.json` -> the full census artifact.

### Datasets (all confirmed live, exact spec counts)

| dataset | n | modality | graph_level | media | temp | perturbations |
|---|---|---|---|---|---|---|
| CarotenoidOzaydin2013 | 4474 | beta_carotene_score | global | SC-URA | 30 | 1 kanmx_deletion + 3 gene_addition |
| BetaxanthinCachera2023 | 4735 | betaxanthin | metabolism | SC | 30 | 1 kanmx_deletion + 4 gene_addition |
| AminoAcidMulleder2016 | 4678 | metabolome | metabolism | SM | 30 | 1 kanmx_deletion |
| MetaboliteZelezniak2018 | 95 | metabolome | metabolism | SM | 30 | 1 kanmx_deletion |
| MetaboliteDaSilveira2014 | 127 | metabolome | metabolism | YPD | 30 | 1 kanmx_deletion |
| IsobutanolScreenLopez2024 | 4554 | isobutanol | metabolism | SC | 30 | 1 kanmx_deletion |
| IsobutanolValidatedLopez2024 | 224 | isobutanol | metabolism | SC | 30 | 1 kanmx_deletion |

Media differ (SC-URA / SC / SM / YPD, all 30 C) and are carried, NOT dropped;
downstream WS6 conditions on the env.

### The load-bearing query difference from Fig-3

The two pigment strains carry heterologous biosynthesis cassettes as
`gene_addition` perturbations, whose genes are NOT in the S288C `gene_set`.
Fig-3's `ALL(perts IN $gene_set)` clause would drop all ~9.2k pigment records.
Fig-6 instead requires only that every DELETION perturbation's gene is in
`gene_set` (>= 1 deletion required), leaving cassette additions unrestricted (the
`Perturbation` processor ignores names absent from the gene graph). Union returns
18,753 rows (18,887 total - 134 whose deletion locus is not a graph node).

### THE census (gates WS11b) - the finding

`GenotypeAggregator` keys on the FULL perturbed gene set (deletions + additions),
so a cassette-bearing pigment genotype can NEVER share a bucket with a single-KO
metabolome genotype. Measured:

- **Q2 genotype co-location (full gene-set, aggregator-keyed):**
  - metabolome + isobutanol: **4367** genotypes
  - metabolome + betaxanthin: **0**
  - metabolome + beta_carotene_score: **0**
  - isobutanol + betaxanthin: 0
- **Q2b deletion-keyed (cassettes stripped, hypothetical):** metabolome +
  betaxanthin would be **4439**, metabolome + beta_carotene **4226** - i.e. the
  cassette key-isolation is exactly what zeroes out the pigment co-location.
- **Q1 gene-level deletion overlap** (single genes deleted in BOTH): metabolome
  union (4685 genes) vs isobutanol screen 4366, betaxanthin 4439, beta-carotene
  4226.

Consequence for WS11b: the "metabolome helps production" test is well-powered for
**isobutanol** (thousands of genotypes carry both an isobutanol titer AND a
metabolome label, cassette-free single-KO). For **betaxanthin / beta-carotene**
it is NOT available at the genotype level - relating metabolome to those pigment
titers must go through the shared DELETED gene (gene-level transfer), not a
co-located multi-label genotype. This reshapes the WS11b design and is the primary
Fig-6 finding.

### Rebuild

```bash
env PYTHONPATH=$(pwd) python experiments/019-simb-multimodal/scripts/generate_fig6_cql.py
env PYTHONPATH=$(pwd) FIG6_CENSUS_ONLY=1 python experiments/019-simb-multimodal/scripts/query_fig6.py  # census only
env PYTHONPATH=$(pwd) python experiments/019-simb-multimodal/scripts/query_fig6.py                      # full build + census
```
