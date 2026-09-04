---
id: fh1c1io6mgyf4lqz6pil2e7
title: Graph_statistics
desc: ''
updated: 1788466930663
created: 1788466930663
---

## 2026.09.03 - Statistics for the nine attention-prior graphs

Script: `experiments/010-kuzmin-tmi/scripts/graph_statistics.py`. Rebuilds the nine graphs every
experiment-010 config regularizes (physical, regulatory, TFLink, six STRING v12.0 channels) from the
cached `SCerevisiaeGraph` pickles under `$DATA_ROOT/data/{sgd,string,tflink}`, plus the STRING v9.1
and v11.0 channels, and writes:

- `experiments/010-kuzmin-tmi/results/graphs/graph_sizes.csv`, `pairwise_overlap.csv`,
  `edge_multiplicity.csv`, `degree_distribution.csv`, `string_version_sizes.csv`, `string_version_drift.csv`
- `paper/nature-biotech/sections/tab-graphs.tex` (replaces the hand-transcribed `tab:databases`) and
  `tab-string-versions.tex`
- true-size panels below (`notes/assets/images/010-kuzmin-tmi/graphs_*.svg`), composed into
  `notes/assets/drawio/FigS-graph-attention-priors.drawio` and `FigS-dango-string-versions.drawio` by
  [[experiments.010-kuzmin-tmi.scripts.compose_graph_si_figures]].

Directed graphs (regulatory, TFLink) are compared as undirected gene pairs; self-loops are dropped.
Node counts are genes with at least one edge.

**The counts match what training used.** Run `3co0xrdg` in wandb project
`torchcell_010-kuzmin-tmi_equivariant_cell_graph_transformer` logs `graph_regularization_info/summary`
at model init with the same nodes/edges per graph (physical 5,721/139,463; regulatory 6,582/39,636;
co-expression 6,474/996,199; ...). The previous SI table, transcribed from
`notes/scratch.2025.05.16.142344-network-sums.md`, was stale: it listed regulatory as 3,632/9,753 and
slightly different STRING/TFLink counts. The regulatory pickle (`G_regulatory.pkl`, 2025.04.29) holds
all 39,636 SGD regulation records: 38,677 transcription, 37,890 high-throughput (24,199 ChIP), 511
regulators, 6,546 targets. I did not find what build produced the 9,753 figure.

Measured (this run, 6,607-gene vocabulary):

| graph | nodes | pairs | mean degree | unique |
|---|---|---|---|---|
| Physical (SGD) | 5,721 | 137,604 | 48.1 | 23% |
| Regulatory (SGD) | 6,582 | 39,546 | 12.0 | 51% |
| TFLink | 5,074 | 199,551 | 78.7 | 80% |
| STRING neighborhood | 2,191 | 146,713 | 133.9 | 10% |
| STRING fusion | 3,081 | 11,787 | 7.7 | 14% |
| STRING co-occurrence | 2,601 | 11,085 | 8.5 | 20% |
| STRING co-expression | 6,474 | 996,199 | 307.8 | 35% |
| STRING experimental | 6,016 | 822,094 | 273.3 | 25% |
| STRING database | 4,027 | 72,617 | 36.1 | 8% |

Union of pairs 1,515,940 (sum 2,437,196): 52.0% of pairs sit in exactly one graph, 37.0% in two, no
pair in more than seven. Largest Jaccard is co-expression vs experimental (0.44); every other pair is
at or below 0.12. Containment is asymmetric: 76% of physical pairs are in STRING experimental, 43% of
regulatory pairs in TFLink, 82% of neighborhood pairs in co-expression.

STRING releases: v9.1 to v12.0 edges grow 2.2x (database) to 8.7x (fusion); Jaccard between
consecutive releases 0.12 to 0.35; only 42% (fusion) to 72% (neighborhood) of a v9.1 channel's
pairs persist in v11.0.

Environment note: on the Mac this ran from a uv venv (Python 3.12) because the base interpreter
cannot parse the package; the `torchcell` conda env rebuilt 2026.09.03 (Python 3.13) should work too.

![](./assets/images/010-kuzmin-tmi/graphs_sizes.svg)

![](./assets/images/010-kuzmin-tmi/graphs_degree_ccdf.svg)

![](./assets/images/010-kuzmin-tmi/graphs_jaccard.svg)

![](./assets/images/010-kuzmin-tmi/graphs_containment.svg)

![](./assets/images/010-kuzmin-tmi/graphs_edge_multiplicity.svg)

The sixth panel of this pass, `graphs_string_versions.svg` (pairwise Jaccard bars between STRING releases), was superseded by `graphs_string_releases.svg` in the section below and is no longer written or kept on disk.

## 2026.09.03 - Structure, hubs, transcription-factor overlap, chronological STRING releases

Author feedback on the first pass: panel b sat low and far right of panel a, the bottom row was
mostly empty, and the pairwise-Jaccard bars for STRING releases looped in time. The script now
computes five more dimensions of the same nine graphs, all from the cached pickles (loads in about
10 s on the Mac; the dense 6,607 x 6,607 matmuls add another 10 s), and lays every panel out with
explicit mm margins so rows and columns align in the composed figure (shared `TOP_MM`, a
`LABEL_LEFT_MM` axes-left for every panel with graph names, one height per row).

New CSVs in `experiments/010-kuzmin-tmi/results/graphs/`:

- `graph_structure.csv`: largest-component fraction, mean clustering (nx.average_clustering
  semantics, via diag(A^3)/2), degree assortativity, mean and median two-hop reach.
- `hub_genes.csv` (top-10 by undirected degree per graph), `hub_recurrence.csv` (how many genes are
  top-10 / top-1% hubs of exactly n graphs), `hub_matrix.csv` (percentile rank of the recurring
  hubs in every graph).
- `tf_overlap.csv`, `tf_overlap_per_regulator.csv`: SGD regulatory vs TFLink as directed graphs.
- `string_version_drift.csv` gained `consecutive`, `added`, `dropped`, `retained_frac_of_b`.

Measured (this run):

| graph | largest component | mean clustering | assortativity | two-hop reach (mean genes) | top hub (degree) |
|---|---|---|---|---|---|
| Physical (SGD) | 1.000 | 0.36 | -0.20 | 4,686 | DHH1 (3,606) |
| Regulatory (SGD) | 1.000 | 0.26 | -0.37 | 3,365 | FKH1 (2,991) |
| TFLink | 1.000 | 0.67 | -0.44 | 5,038 | GCN4 (4,908) |
| STRING neighborhood | 1.000 | 0.27 | 0.04 | 1,646 | DUR12 (936) |
| STRING fusion | 0.962 (43 components) | 0.03 | 0.24 | 78 | CDC28 (84) |
| STRING co-occurrence | 0.432 (486 components) | 0.46 | 0.93 | 22 | SPS1 (97) |
| STRING co-expression | 0.9997 | 0.31 | 0.21 | 4,682 | SSA1 (1,770) |
| STRING experimental | 0.9997 | 0.28 | -0.01 | 4,895 | NAB2 (2,696) |
| STRING database | 0.962 (46 components) | 0.54 | 0.33 | 404 | HFD1 (374) |

Hub recurrence: the nine top-10 lists name 74 genes; 12 are in two lists, 2 (GDE1, ISW1) in three.
At the top 1% (66 genes per graph): 353 genes are hubs of one graph, 79 of two, 22 of three, 3 of
four, 1 (CDC28) of five.

Transcription-factor graphs (directed, self-regulation kept): regulators 511 (regulatory) vs 317
(TFLink), 200 shared; targets 6,546 vs 5,074, 4,963 shared; directed edges 39,636 vs 200,801, 16,817
shared (Jaccard 0.075). Per shared regulator the Jaccard of the two target sets has median 0.03;
FKH1 is the best case at 0.53 with 1,747 shared targets.

Shared pairs (undirected): co-expression and experimental 556,106; neighborhood and co-expression
119,758; physical and experimental 104,771; regulatory and TFLink 16,885.

STRING releases, consecutive steps (retained of previous / added / dropped): neighborhood
32,981 / 87,984 / 12,620 then 65,678 / 81,035 / 55,287; fusion 565 / 3,349 / 793 then
2,001 / 9,786 / 1,913; co-occurrence 1,893 / 2,775 / 766 then 2,654 / 8,431 / 2,014; co-expression
183,849 / 370,984 / 129,839 then 355,122 / 641,077 / 199,711; experimental
121,984 / 270,788 / 97,466 then 295,506 / 526,588 / 97,266; database 19,776 / 27,040 / 13,710 then
26,042 / 46,575 / 20,774. Of a v12.0 channel's pairs, 17% (fusion) to 45% (neighborhood) were
already in v11.0 and 4% (fusion) to 21% (co-expression) in v9.1.

`graphs_string_versions.svg` (pairwise Jaccard bars) is superseded by `graphs_string_releases.svg`
and is no longer written; the DANGO release sweep panel moved to the DANGO note. Panels now
written, in figure order:

![](./assets/images/010-kuzmin-tmi/graphs_sizes.svg)

![](./assets/images/010-kuzmin-tmi/graphs_degree_ccdf.svg)

![](./assets/images/010-kuzmin-tmi/graphs_jaccard.svg)

![](./assets/images/010-kuzmin-tmi/graphs_containment.svg)

![](./assets/images/010-kuzmin-tmi/graphs_edge_multiplicity.svg)

![](./assets/images/010-kuzmin-tmi/graphs_structure.svg)

![](./assets/images/010-kuzmin-tmi/graphs_shared_pairs.svg)

![](./assets/images/010-kuzmin-tmi/graphs_hubs.svg)

![](./assets/images/010-kuzmin-tmi/graphs_tf_overlap.svg)

![](./assets/images/010-kuzmin-tmi/graphs_string_releases.svg)

## 2026.09.03 - Deterministic ordering of the hub and per-regulator tables

Rerunning the script on the Mac changed three committed CSVs without any input change: `hub_genes.csv`
(KSP1 and SPS1 are tied at degree 97 in co-occurrence, and `sort_values` left ties in arbitrary order),
`tf_overlap_per_regulator.csv` (the 200 shared regulators were iterated from a Python `set`, so the row
order among equal `shared_targets` followed the per-process string hash seed), and `graph_structure.csv`
(assortativity and two-hop means differed in the last floating-point digit between BLAS builds). Fixes:
degree sorts now break ties by systematic name (`by_degree`), the per-regulator table is built from
`sorted(shared)` and sorted on `(shared_targets desc, regulator asc)`, and `graph_structure.csv` is written
with `float_format="%.10g"`. Verified stable across two runs with different `PYTHONHASHSEED`. The SI
figure caption now says SPS1 and KSP1 are tied in co-occurrence. No panel or table changed.

The Supplementary Note `note:graphs` prose was compressed on the same day; the hub degrees, shared-pair
counts, two-hop reach, assortativity values, and release growth factors that left the prose are now in
the two figure captions, and everything else is in the sections above.

## 2026.09.03 - Author revision: uncovered genes, third-width matrices, other components, hub descriptions

Author feedback on the second pass, and what changed in the script:

- Panel a now stacks the uncovered remainder (white, hatched) onto the covered-gene bar up to the
  6,607-gene reference and prints the uncovered count right of the dotted line (`uncovered` column
  added to `graph_sizes.csv`, the only change to a previously committed CSV).
- The Jaccard and containment heatmaps share a 0 to 1 colorbar (the diagonal is 1), are third-width
  (57.8 mm) with `SHORT_LABEL` row and column names (STRING prefix dropped), and their cells print two
  decimals without the leading zero at 5 pt. Multiplicity stops at the largest observed bin (7).
- The structure panel (largest component, clustering, assortativity, two-hop reach) moved to figure 2
  as a full-width panel e; its numbers are unchanged (table in the section above, `graph_structure.csv`).
- Panel f is new: `graph_components.csv` + `graph_components_reference.json` from `load_components`,
  `components_table`, and `random_pairs`.
- Hub panel: the rank numbers are gone; each recurring hub gets the opening words of its SGD
  `locus.description` (`hub_descriptions.csv`, source `$DATA_ROOT/data/sgd/genome/genes/<gene>.json`;
  the full first clause is kept in the CSV). The rule in `short_description` keeps whole
  preposition-delimited segments within 34 characters, so a few labels end early (PUF3 "Protein of the
  mitochondrial outer"); hand-tune the rule, not the CSV.
- Every number printed over gridlines (uncovered counts, multiplicity counts, the TF-panel median/FKH1
  text) sits on an opaque white box (`TEXT_BBOX`); the TF-panel outside labels moved from |y| = 0.42 to
  0.62 so "16,817" clears the bar edge.

Sources for panel f, all read offline on the Mac:

- Yeast9 = yeast-GEM 9.0.2 through `torchcell.metabolism.yeast_GEM.YeastGEM` (`$DATA_ROOT/data/torchcell/yeast-GEM`):
  1,161 genes carry a gene-reaction rule (all in the vocabulary), 2,709 of 4,131 reactions have genes,
  107 subsystems. Reaction pairs = gene pairs sharing at least one reaction (1,456 distinct pairs);
  subsystem pairs = gene pairs whose genes appear in reactions of the same subsystem (22,921).
- GO = `SCerevisiaeGraph.go_to_genes` (direct SGD annotations, any evidence code, not propagated) with
  `genome.go_dag` namespaces: biological_process terms, the root excluded, 2 to 500 genes; 1,944 terms
  qualify (only the root exceeds 500 direct genes; the largest term, protein transport, has 396) and
  give 557,180 distinct co-annotated pairs.
- Essential = the `GeneEssentialitySgdDataset` rule applied to `G_raw` phenotype records (null mutant,
  strain S288C, phenotype `inviable`): 1,140 genes, the same count as
  `$DATA_ROOT/data/torchcell/gene_essentiality_sgd/preprocess/gene_set.json`.
- Trigenic panels: Kuzmin 2018 = the 1,400 perturbed genes of the 005 build experiment 010 trains on
  (`experiments/005-kuzmin2018-tmi/001-small-build/processed/is_any_perturbed_gene_index.json`);
  Kuzmin 2020 = `tmi_kuzmin2020/preprocess/gene_set.json`, 4,308 of its 4,367 genes in the vocabulary.
- Random reference: per graph, stub matching on its own degree sequence (configuration model,
  `numpy.random.default_rng(0)`), self-loops and duplicate pairs dropped, the three pair shares
  recomputed on the result.

Measured (this run; random reference in parentheses):

| graph | metabolic genes | essential genes | Kuzmin 2018 covered | Kuzmin 2020 covered | pairs sharing a reaction | a subsystem | a GO process |
|---|---|---|---|---|---|---|---|
| Physical (SGD) | 20.3% | 18.8% | 99.9% | 91.4% | 0.30% (0.005%) | 0.55% (0.05%) | 21.2% (7.5%) |
| Regulatory (SGD) | 17.2% | 16.8% | 98.0% | 95.8% | 0.0025% (0.003%) | 0.0025% (0.06%) | 6.4% (10.6%) |
| TFLink | 22.1% | 21.0% | 96.0% | 81.7% | 0.0005% (0.009%) | 0.0095% (0.10%) | 6.6% (9.8%) |
| STRING neighborhood | 39.4% | 18.9% | 33.6% | 34.6% | 0.10% (0.04%) | 2.8% (0.9%) | 14.9% (5.4%) |
| STRING fusion | 32.7% | 23.2% | 53.4% | 48.9% | 0.27% (0.03%) | 3.1% (0.42%) | 16.1% (5.8%) |
| STRING co-occurrence | 29.8% | 16.5% | 44.3% | 43.6% | 4.2% (0.03%) | 7.5% (0.13%) | 66.9% (17.7%) |
| STRING co-expression | 17.8% | 17.6% | 99.9% | 99.4% | 0.10% (0.011%) | 1.0% (0.17%) | 16.4% (4.9%) |
| STRING experimental | 19.2% | 18.4% | 100% | 95.8% | 0.10% (0.007%) | 0.59% (0.09%) | 16.1% (5.8%) |
| STRING database | 26.1% | 24.0% | 77.4% | 64.7% | 0.88% (0.04%) | 5.5% (0.60%) | 34.6% (4.6%) |

Reference shares over the 6,607 genes: metabolic 17.6%, essential 17.3%. The pair shares span four
orders of magnitude, so panel f draws them on logarithmic axes (1e-6 to 1). Reaction enrichment over
random: physical 62x, database 24x, co-occurrence 148x; GO-process enrichment 2.7x (neighborhood) to
7.5x (database) for the STRING channels and physical, and below 1 for both transcription-factor graphs.

Runtime on the Mac: about 90 s, of which `go_to_genes` takes 26 s and the cobra model load 7 s. A
second run leaves `results/graphs/` unchanged (verified with `git status` after the rerun).

![](./assets/images/010-kuzmin-tmi/graphs_sizes.svg)

![](./assets/images/010-kuzmin-tmi/graphs_jaccard.svg)

![](./assets/images/010-kuzmin-tmi/graphs_containment.svg)

![](./assets/images/010-kuzmin-tmi/graphs_edge_multiplicity.svg)

![](./assets/images/010-kuzmin-tmi/graphs_components.svg)

![](./assets/images/010-kuzmin-tmi/graphs_hubs.svg)

![](./assets/images/010-kuzmin-tmi/graphs_tf_overlap.svg)

![](./assets/images/010-kuzmin-tmi/graphs_structure.svg)
