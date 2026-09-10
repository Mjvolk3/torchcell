---
id: my1xg525ojxqpotw2crjzam
title: Carotenoid_graph_enrichment
desc: ''
updated: 1789068604208
created: 1789068604208
---

## 2026.09.10 - Are GEA1, VPS1, PCT1, HXK2 near the carotenoid precursor pathway in the 010 graphs?

### Question

The FSEOF scan could not separate the four genes: GEA1 and VPS1 are absent from Yeast9
entirely, and PCT1 and HXK2 are present but none of their reactions scored. That is a
statement about the stoichiometric model, not about the graphs the model actually saw.
The 010 transformer never reads Yeast9. It reads nine gene graphs, and it reads them only
through the layer-1 graph-regularization penalty. So the question worth asking is whether
any of the four sits close to carotenoid-precursor genes in those nine graphs.

### Graphs and gene sets

The graphs are the nine named in
`experiments/010-kuzmin-tmi/conf/equivariant_cell_graph_transformer_cabbi_000.yaml`:
physical, regulatory, tflink, and the six STRING v12.0 channels (neighborhood, fusion,
cooccurence, coexpression, experimental, database). The regulatory and tflink graphs are
directed; adjacency is scored undirected, so a query gene that is a target of a set member
counts as adjacent to it. A tenth row, `union_of_9`, is the edge union.

| graph | nodes | edges | directed |
|---|---|---|---|
| physical | 5721 | 139463 | no |
| regulatory | 6582 | 39636 | yes |
| tflink | 5074 | 200801 | yes |
| string12_0_neighborhood | 2191 | 146713 | no |
| string12_0_fusion | 3081 | 11787 | no |
| string12_0_cooccurence | 2601 | 11085 | no |
| string12_0_coexpression | 6474 | 996199 | no |
| string12_0_experimental | 6016 | 822094 | no |
| string12_0_database | 4027 | 72617 | no |

Yeast has no native carotenoid pathway, so "carotenoid-relevant" means the precursor
supply a heterologous crtE/crtYB/crtI pathway draws on. Three sets:

- `mva_ggpp_backbone`, 10 genes: ERG10, ERG13, HMG1, HMG2, ERG12, ERG8, MVD1, IDI1,
  ERG20, BTS1. Acetyl-CoA through mevalonate to FPP to GGPP.
- `mva_ggpp_plus_squalene_branch`, 12 genes: the backbone plus ERG9 and ERG1, the
  FPP-consuming branch that competes for the same precursor pool.
- `go_GO_0008299_isoprenoid_biosynthetic_process`, 18 genes: the GO term and all its
  descendants, as an annotation-driven set that does not depend on my hand curation.

Enrichment is a hypergeometric test on the query gene's neighbor set against the graph's
own node set, so a hub gene is not credited for touching the pathway by degree alone.

### Answer: one of the four is enriched, and it is HXK2

HXK2 is adjacent to seven of the ten mevalonate-to-GGPP backbone genes in the union of the
nine graphs, against 1.02 expected, a 6.9-fold enrichment at p = 1.0e-5 (hypergeometric).
Adding the squalene branch makes it eight of twelve at p = 4.0e-6. The signal is carried by
STRING coexpression (six of ten, p = 7.1e-5) and STRING fusion (two of ten but on a degree
of only 18, so 34-fold, p = 1.4e-3). The fusion neighbors are HMG1 and HMG2, the two
HMG-CoA reductase paralogs, which is the committed and rate-controlling step of the
pathway.

Across the whole design, 4 genes times 10 graphs times 3 sets is 120 tests, so the
Bonferroni threshold is 4.2e-4. Five rows clear it and all five are HXK2: the union and
STRING coexpression rows for both mevalonate sets, plus the union GO-isoprenoid row at
1.9e-4.

| query gene | best p over all graphs and sets | verdict |
|---|---|---|
| HXK2 | 4.0e-6 | enriched, survives Bonferroni |
| PCT1 | 2.3e-2 | not significant after correction |
| GEA1 | 2.9e-1 | no signal |
| VPS1 | 8.7e-1 | no signal, and below chance in the one graph where it touches the set |

Every neighbor-level hit:

| query gene | graph | gene set | degree | neighbors in set | expected | fold | p | neighbors |
|---|---|---|---|---|---|---|---|---|
| HXK2 | union_of_9 | mva + squalene | 711 | 8 | 1.22 | 6.6 | 4.0e-6 | ERG9, ERG20, HMG1, ERG13, ERG12, MVD1, HMG2, ERG10 |
| HXK2 | union_of_9 | mva_ggpp_backbone | 711 | 7 | 1.02 | 6.9 | 1.0e-5 | ERG20, HMG2, HMG1, ERG13, ERG12, MVD1, ERG10 |
| HXK2 | string12_0_coexpression | mva + squalene | 571 | 7 | 1.06 | 6.6 | 2.1e-5 | ERG9, ERG20, HMG1, ERG13, ERG12, MVD1, ERG10 |
| HXK2 | string12_0_coexpression | mva_ggpp_backbone | 571 | 6 | 0.88 | 6.8 | 7.1e-5 | ERG20, HMG1, ERG13, ERG12, MVD1, ERG10 |
| HXK2 | union_of_9 | go_isoprenoid | 711 | 8 | 1.83 | 4.4 | 1.9e-4 | YBR003W, ERG9, ERG20, HMG2, HMG1, ERG13, ERG12, MVD1 |
| HXK2 | string12_0_coexpression | go_isoprenoid | 571 | 7 | 1.59 | 4.4 | 5.4e-4 | YBR003W, ERG9, ERG20, HMG1, ERG13, ERG12, MVD1 |
| HXK2 | string12_0_fusion | mva_ggpp_backbone | 18 | 2 | 0.06 | 34.2 | 1.4e-3 | HMG2, HMG1 |
| HXK2 | string12_0_fusion | mva + squalene | 18 | 2 | 0.07 | 28.5 | 2.1e-3 | HMG2, HMG1 |
| HXK2 | string12_0_fusion | go_isoprenoid | 18 | 2 | 0.09 | 21.4 | 3.7e-3 | HMG2, HMG1 |
| PCT1 | string12_0_experimental | mva_ggpp_backbone | 144 | 2 | 0.24 | 8.4 | 2.3e-2 | ERG13, BTS1 |
| PCT1 | string12_0_experimental | mva + squalene | 144 | 2 | 0.29 | 7.0 | 3.2e-2 | ERG13, BTS1 |
| PCT1 | string12_0_coexpression | go_isoprenoid | 291 | 3 | 0.81 | 3.7 | 4.4e-2 | YDL193W, ERG13, BTS1 |
| PCT1 | string12_0_experimental | go_isoprenoid | 144 | 2 | 0.41 | 4.9 | 6.1e-2 | ERG13, BTS1 |
| PCT1 | string12_0_coexpression | mva_ggpp_backbone | 291 | 2 | 0.45 | 4.4 | 7.1e-2 | ERG13, BTS1 |
| PCT1 | union_of_9 | go_isoprenoid | 416 | 3 | 1.07 | 2.8 | 8.8e-2 | YDL193W, ERG13, BTS1 |
| PCT1 | string12_0_coexpression | mva + squalene | 291 | 2 | 0.54 | 3.7 | 9.9e-2 | ERG13, BTS1 |
| PCT1 | union_of_9 | mva_ggpp_backbone | 416 | 2 | 0.59 | 3.4 | 1.2e-1 | ERG13, BTS1 |
| PCT1 | union_of_9 | mva + squalene | 416 | 2 | 0.71 | 2.8 | 1.6e-1 | ERG13, BTS1 |
| PCT1 | string12_0_neighborhood | go_isoprenoid | 75 | 1 | 0.55 | 1.8 | 4.3e-1 | YDL193W |
| GEA1 | string12_0_experimental | go_isoprenoid | 121 | 1 | 0.34 | 2.9 | 2.9e-1 | YDL193W |
| GEA1 | union_of_9 | go_isoprenoid | 227 | 1 | 0.58 | 1.7 | 4.5e-1 | YDL193W |
| VPS1 | string12_0_experimental | go_isoprenoid | 685 | 1 | 1.94 | 0.5 | 8.7e-1 | YNR041C |
| VPS1 | union_of_9 | go_isoprenoid | 844 | 1 | 2.17 | 0.5 | 9.0e-1 | YNR041C |

GEA1 and VPS1 have zero neighbors in the mevalonate backbone in every one of the nine
graphs. Their only contact with any carotenoid-relevant set is a single GO-isoprenoid gene
in STRING experimental, which for VPS1 is fewer hits than its degree of 685 predicts.

### Which graph carries the HXK2 signal

Two of the nine, and only one of them clears correction alone. Every graph against the
mevalonate-to-GGPP backbone:

| graph | HXK2 degree | neighbors in backbone | p |
|---|---|---|---|
| string12_0_coexpression | 571 | 6 | 7.1e-5 |
| string12_0_fusion | 18 | 2 | 1.4e-3 |
| string12_0_experimental | 289 | 0 | 1.0 |
| tflink | 69 | 0 | 1.0 |
| physical | 55 | 0 | 1.0 |
| string12_0_database | 49 | 0 | 1.0 |
| regulatory | 12 | 0 | 1.0 |
| string12_0_cooccurence | 3 | 0 | 1.0 |
| string12_0_neighborhood | 3 | 0 | 1.0 |
| union_of_9 | 711 | 7 | 1.0e-5 |

STRING coexpression is the graph. It contributes six of the union's seven backbone
neighbors (ERG10, ERG13, HMG1, ERG12, MVD1, ERG20) and is the only single graph under the
4.2e-4 threshold. STRING fusion contributes the seventh, HMG2, and is the more striking
result per edge at 34-fold on a degree of 18, but two hits do not clear correction.

The zeros matter as much. HXK2 is well connected in STRING experimental (degree 289) and
in the physical graph (degree 55), and has no backbone contact in either. The union row is
those two channels combined, not a tenth source.

### Two cautions on reading this

**Every one of the four is two steps from the pathway, so proximity alone separates
nothing.** The shortest-path distance to the mevalonate backbone in the union graph is 1
for HXK2 and PCT1 and 2 for GEA1 and VPS1, and in the dense STRING channels 22 to 38
percent of all connected genes have at least one neighbor in these sets. Touching the
pathway is common. Only the enrichment test, which conditions on degree, distinguishes
HXK2.

**The enrichment is a coexpression and fusion signal, not a mechanistic one.** HXK2 is a
glycolytic hexokinase and a well-known regulator of glucose repression, so a coexpression
neighborhood shared with the ergosterol and mevalonate genes is what carbon-source
regulation looks like. The fusion channel connecting it to HMG1 and HMG2 is a homology
signal in other genomes, not a yeast interaction. Hypothesis (untested): HXK2's proximity
here reflects shared glucose-repression control of precursor supply rather than any direct
route to GGPP. Nothing in this analysis measures flux, and the FSEOF result stands as the
separate statement that the stoichiometric model gives none of the four a
carotenoid-correlated reaction.

### What the model could have learned from this

The graphs enter the 010 model only through the layer-1 graph-regularization penalty, so
this proximity is available to the model as a weak prior on HXK2's embedding, not as a
feature at prediction time. Whether that prior contributed anything to HXK2's ranking is
not measured here.

### Incidental

VPS1 and PCT1 are direct neighbors in STRING experimental, the only edge among the four
query genes in any of the nine graphs.

### Reproduce

```bash
PYTHONPATH=$PWD python experiments/010-kuzmin-tmi/scripts/carotenoid_graph_enrichment.py
```

Outputs `experiments/010-kuzmin-tmi/results/carotenoid_graph_enrichment.csv` (120 rows,
one per gene by graph by set) and `carotenoid_graph_enrichment_query_pairs.csv`.

## 2026.09.10 - Are GEA1, VPS1, PCT1, HXK2 near the carotenoid precursor supply in the 010 training graphs?

Question: the four genes that FSEOF on Yeast9 cannot separate (GEA1 and VPS1 absent from the GEM; PCT1 and HXK2 present but with no reaction in the flux-correlation scan) -- are they enriched for carotenoid-pathway genes in the nine graphs the 010 model trained on? Yeast has no native carotenoid pathway, so "carotenoid pathway" here means the precursor supply that any crtE/crtYB/crtI cassette draws on. Three gene sets were scored:

- `mva_ggpp_backbone`
