# Exploring the E. coli datasets

Four built *Escherichia coli* K-12 datasets load with one constructor call each. This page reads like a notebook: each cell of code is followed by what it printed or drew, and every cell is a part of one script, {download}`ecoli_download_tutorial.py <../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py>`, which runs top to bottom in an empty directory. Where the archives come from, how large they are and how a download is checked is on {doc}`download`.

```{include} _generated/tutorial/datasets_table.md
```

Terms used below:

- **Hfr Cavalli donor**: the *E. coli* strain that transfers a marked deletion into a recipient by conjugation in the Babu et al. 2014 screen.
- **Keio collection**: the *E. coli* K-12 BW25113 single-gene deletion strains of Baba et al. 2006, in which a kanamycin-resistance cassette replaces each gene.
- **modified z-score**: the value Fuhrer et al. 2017 release for each strain and ion, summarized over the strain's replicates and standardized across strains.
- **record**: one stored measurement, a dict with three keys. `experiment` holds the genotype, the environment, and the phenotype. `reference` holds the genome and the level the value is compared with. `publication` holds the identifiers of the source paper.
- **S-score**: the signed genetic-interaction score Babu et al. 2014 compute from the colony sizes of a double mutant. A negative score is an aggravating interaction and a positive score is an alleviating one.

## Setup

Install torchcell into a Python 3.13 environment ({doc}`../../guide/installation`), then import the loaders. Setting `TC_DATA_URL` makes each loader download its built store from the public endpoint.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [setup]"
:end-before: "# [/setup]"
```

The plotting cell sets the figure style once. Colors come from the torchcell palette, and each figure is one row of three panels.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [style]"
:end-before: "# [/style]"
```

## Growth rate (Campos 2018)

`GrowthRateCampos2018Dataset` holds one record per Keio strain. The label is the strain's maximal growth rate in M9 medium with casamino acids and glucose at 30 °C, stored as a ratio to the parent strain BW25113, so the reference level is 1.0.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth_load]"
:end-before: "# [/growth_load]"
```

```{include} _generated/tutorial/output_growth_load.md
```

One record, field by field:

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth_record]"
:end-before: "# [/growth_record]"
```

```{include} _generated/tutorial/output_growth_record.md
```

All records as a table:

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth_table]"
:end-before: "# [/growth_table]"
```

```{include} _generated/tutorial/output_growth_table.md
```

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [growth_figure]"
:end-before: "# [/growth_figure]"
```

```{figure} _generated/tutorial/ecoli_growth_rate_campos2018.svg
:name: fig-ecoli-growth-rate
:width: 100%

Growth rate of the Keio strains in `GrowthRateCampos2018Dataset`. **a**, strains per 0.02-wide bin of growth rate relative to the parent, with the parent level of 1.0 dashed. **b**, the strains ranked from slowest to fastest, with the five slowest named. **c**, the median (dot) and the interquartile range (bar) of the strains on each Keio plate.
```

## The same record as typed objects

`dataset[i]` returns plain dicts. `transform_item` validates one record through the pydantic classes of the schema and returns the typed objects.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [typed]"
:end-before: "# [/typed]"
```

```{include} _generated/tutorial/output_typed_record.md
```

## Cell shape and cell cycle (Campos 2018)

`MorphologyCampos2018Dataset` covers the same strains from the same imaging screen. Each record stores the cell-shape means and the cell-cycle features under the symbols the paper gives them, and the coefficients of variation of the shape features in `morphology_coefficient_of_variation`.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [morphology_load]"
:end-before: "# [/morphology_load]"
```

```{include} _generated/tutorial/output_morphology_load.md
```

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [morphology_table]"
:end-before: "# [/morphology_table]"
```

```{include} _generated/tutorial/output_morphology_table.md
```

Some records hold 9 features and not 15: a strain with no nucleoid channel has no nucleoid-derived features, and the table holds `NaN` there.

The growth-rate dataset holds the same strains in the same order, so the two join by position:

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [morphology_growth]"
:end-before: "# [/morphology_growth]"
```

```{include} _generated/tutorial/output_morphology_growth.md
```

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [morphology_figure]"
:end-before: "# [/morphology_figure]"
```

```{figure} _generated/tutorial/ecoli_morphology_campos2018.svg
:name: fig-ecoli-morphology
:width: 100%

Cell shape of the Keio strains in `MorphologyCampos2018Dataset`, one point or count per strain. **a**, strains per 0.04 µm bin of mean cell length. **b**, mean cell width against mean cell length. **c**, mean cell area against the growth rate of the same strain from `GrowthRateCampos2018Dataset`.
```

## Genetic interactions (Babu 2014)

`GeneInteractionBabu2014Dataset` holds one record per double mutant of the released high-confidence set. The genotype of a record lists two perturbations, one from an Hfr Cavalli donor and one from the recipient array, and the label is the pair's S-score.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [interaction_load]"
:end-before: "# [/interaction_load]"
```

```{include} _generated/tutorial/output_interaction_load.md
```

The donor is the first perturbation in some records and the second in others, so the `collection` field of each perturbation, not its position, says which side of the cross it came from.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [interaction_table]"
:end-before: "# [/interaction_table]"
```

```{include} _generated/tutorial/output_interaction_table.md
```

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [interaction_figure]"
:end-before: "# [/interaction_figure]"
```

```{figure} _generated/tutorial/gene_interaction_babu2014.svg
:name: fig-ecoli-genetic-interaction
:width: 100%

S-scores in `GeneInteractionBabu2014Dataset`. **a**, gene pairs per unit-wide bin of S-score, negative and positive scores colored apart. The release holds only the high-confidence pairs, which leaves the gap around zero. **b**, genes by the number of scored pairs each takes part in, on logarithmically spaced bins. **c**, positive against negative pairs per gene, one point per gene, on axes that are linear from 0 to 1 and logarithmic above.
```

## Metabolome (Fuhrer 2017)

`MetabolomeFuhrer2017Dataset` holds one record per Keio strain. The label is a dict from ion key to modified z-score, with the same ions in the same order in every record, and `preprocess/ions.csv` gives the mass-to-charge ratio of each key.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [metabolome_load]"
:end-before: "# [/metabolome_load]"
```

```{include} _generated/tutorial/output_metabolome_load.md
```

Because every record holds the same ions in the same order, the labels stack into one strains-by-ions matrix:

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [metabolome_matrix]"
:end-before: "# [/metabolome_matrix]"
```

```{include} _generated/tutorial/output_metabolome_matrix.md
```

Some records pool two or three Keio strains. The deposit labels such a column with one name that matches more than one Keio entry, the record's genotype lists every one of them, and the strain label above joins their gene names with `+`.

```{literalinclude} ../../../../experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py
:language: python
:start-after: "# [metabolome_figure]"
:end-before: "# [/metabolome_figure]"
```

```{figure} _generated/tutorial/metabolome_fuhrer2017.svg
:name: fig-ecoli-metabolome
:width: 100%

Modified z-scores in `MetabolomeFuhrer2017Dataset`. **a**, all strain and ion values per 0.5-wide bin, on a logarithmic count axis. **b**, strains by the number of ions whose z-score is beyond 5 in absolute value, in bins of 10 ions. **c**, for each ion, the number of strains beyond 5 against the ion's mass-to-charge ratio, colored by ionization mode.
```

## Provenance

- Tutorial code: `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial.py`.
- Runs, figures and page fragments: `experiments/034-showcase-datasets/scripts/ecoli_download_tutorial_page.py`, which writes `experiments/034-showcase-datasets/results/ecoli_download_tutorial.<host>.json` per host, and the printed output and figures under `docs/source/datasets/ecoli/_generated/tutorial/`.
- The figures are 179 mm wide with 7 pt Arial text and the first colors of `torchcell.utils.PLOT_PALETTE`.
- The printed output was byte-identical on a repository checkout and on a new environment installed from PyPI ({doc}`../../guide/install-timings`).
