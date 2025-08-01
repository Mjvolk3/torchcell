---
id: 362g3xicieoaub7qotvleqt
title: Generate_triple_combinations
desc: ''
updated: 1751854053516
created: 1751854021624
---
## 2025.07.06 - Summary Generated by Claude

### Overview - Triple Combination Filtering

This script generates triple gene combinations and applies two stages of filtering:

1. **DMF Filtering**: Removes triples where any pair of genes has fitness < 0.5 in double mutant fitness (DMF) datasets
2. **TMI Filtering**: Removes triples that already exist in triple mutant interaction (TMI) datasets

### TMI Filtering Implementation

The TMI filtering uses the `is_any_perturbed_gene_index` property from Neo4jCellDataset, which provides a mapping from gene names to dataset indices where that gene is perturbed.

For each triple (gene1, gene2, gene3), we check if there's a common index where all three genes are perturbed together. If such an index exists, the triple is already in the TMI datasets and is excluded.

### Potential Optimizations

If the TMI filtering becomes a bottleneck with larger gene sets:

1. **Pre-compute TMI triples**: Instead of checking each triple individually, pre-compute all existing TMI triples from the dataset and store them in a set for O(1) lookup.

2. **Batch processing**: Process triples in batches to reduce overhead from repeated set operations.

3. **Use the preprocessed dataframes directly**: Similar to how DMF filtering loads preprocessed CSV files, you could extract TMI triples directly from the Kuzmin2018/2020 preprocessed data.

4. **Parallel processing**: The TMI checking could be parallelized since each triple check is independent.

### Output Files

- `triple_combinations_{timestamp}.pkl`: Complete results including all filtering stages
- `triple_combinations_list_{timestamp}.txt`: Simple text file with final filtered triples
- `triple_filtering_summary_{timestamp}.png`: Visualization of the filtering pipeline
