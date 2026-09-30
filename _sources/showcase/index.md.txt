# Showcase

Each showcase page takes one group of datasets from the served knowledge graph and shows what the experiments measured, what one stored record looks like, how the values are distributed, and the supported query that returns them. Every table, figure and record dump on these pages is written by a script under `experiments/034-showcase-datasets/scripts/`, and query results come from a slurm job against the served release named on the page. Caveats that change how a value should be read are stated on the page that shows the value.

```{toctree}
:maxdepth: 1

essentiality-smf
```
