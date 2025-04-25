---
id: fwcmduwm333j9nkb2cvxqo0
title: 033248
desc: ''
updated: 1744965467376
created: 1744965170271
---
I'll create a markdown table explaining the STRING database columns, focusing especially on how the combined score is computed.

# STRING Database Column Explanations

| Column           | Description                                                                                                                  |
|------------------|------------------------------------------------------------------------------------------------------------------------------|
| **neighborhood** | Score based on genomic proximity of genes across multiple species; indicates potential operons or co-regulated gene clusters |
| **fusion**       | Score reflecting evidence of gene fusion events across species; suggests functional relationship                             |
| **cooccurrence** | Score based on phylogenetic profiles (presence/absence patterns across species)                                              |
| **coexpression** | Score derived from correlated gene expression patterns across multiple experiments                                           |
| **experimental** | Score from direct protein-protein interaction data (Y2H, co-IP, etc.)                                                        |
| **database**     | Score from curated pathway databases and expert knowledge bases                                                              |
| textmining       | Score from automated text analysis of scientific literature co-mentions                                                      |
| combined_score   | Integrated confidence score (0-1000) combining all evidence types                                                            |

## Combined Score Calculation

The combined score in STRING uses a probabilistic approach based on naive Bayesian methods:

$$
S = 1 - \prod_{i} (1 - S_i)
$$

Where:

- $S$ is the combined score
- $S_i$ represents each individual evidence score (normalized between 0 and 1)
- $\prod$ is the product operator

This approach:

1. Treats each evidence channel as independent
2. Ensures that multiple weak signals can still produce a strong combined score
3. Prevents any single evidence type from dominating the final score

The final score is then calibrated against a gold standard of known interactions to ensure that the combined score reflects the probability that the interaction is biologically meaningful.
