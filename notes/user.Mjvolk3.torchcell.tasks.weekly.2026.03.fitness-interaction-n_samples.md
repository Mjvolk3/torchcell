---
id: eydttxka75k6eoxuxf7w35a
title: fitness-interaction-n_samples
desc: ''
updated: 1768705625126
created: 1768522786449
---
## Dataset Files For Fitness and Interactions

torchcell/datasets/scerevisiae/costanzo2016.py
torchcell/datasets/scerevisiae/kuzmin2018.py
torchcell/datasets/scerevisiae/kuzmin2020.py

torchcell/datamodels/schema.py

## Papers

```bash
michaelvolk@M1-MV-6 torchcell % tree /Users/michaelvolk/Documents/projects/torchcell/papers                                      18:22
/Users/michaelvolk/Documents/projects/torchcell/papers
├── costanzoGlobalGeneticInteraction2016
│   ├── costanzoGlobalGeneticInteraction2016.mmd
│   ├── costanzoGlobalGeneticInteraction2016.pdf
│   ├── SI-costanzoGlobalGeneticInteraction2016.mmd
│   └── SI-costanzoGlobalGeneticInteraction2016.pdf
├── kuzminExploringWholegenomeDuplicate2020
│   ├── kuzminExploringWholegenomeDuplicate2020.mmd
│   ├── kuzminExploringWholegenomeDuplicate2020.pdf
│   ├── SI-kuzminExploringWholegenomeDuplicate2020.mmd
│   └── SI-kuzminExploringWholegenomeDuplicate2020.pdf
└── kuzminSystematicAnalysisComplex2018
    ├── kuzminSystematicAnalysisComplex2018.mmd
    ├── kuzminSystematicAnalysisComplex2018.pdf
    ├── SI-kuzminSystematicAnalysisComplex2018.mmd
    └── SI-kuzminSystematicAnalysisComplex2018.pdf

4 directories, 12 files
```

## Issue

We have developed models that can predict triple mutant fitness and interactions. $\hat{f}(g_1,g_2, g_3)= (\text{fitness},\text{gene interaction})$. This is done after querying over these datasets and reducing the data with `deduplicator`, `aggregator`, and potentially `converter`. This means that we only choose some of the data.

torchcell/data/mean_experiment_deduplicate.py

```python
dataset = Neo4jCellDataset(
    root=dataset_root,
    query=query,
    gene_set=genome.gene_set,
    graphs=None,
    incidence_graphs={"metabolism_bipartite": YeastGEM().bipartite_graph},
    node_embeddings={
        "codon_frequency": codon_frequency,
        "fudt_3prime": fudt_3prime_dataset,
        "fudt_5prime": fudt_5prime_dataset,
    },
    converter=None, # HERE
    deduplicator=MeanExperimentDeduplicator, # HERE
    aggregator=GenotypeAggregator, # HERE
    graph_processor=SubgraphRepresentation(),
)
```

When we do this we might collect say different single mutants than were used to compute the double mutant interaction. Refer to definition of gene interaction here [[Gene_interaction|phenotype.gene_interaction]]. This means that $\epsilon = f_{ij}-f_{i}f_{j}$ after all data aggregation could give us $f_i$, $f_j$, or $f_{ij}$ that were not the original data points used to construct $\epsilon$ and then we also have $\epsilon$ in dataset.

On query we typically group instances into plausible mutants that could be constructed with associated phenotypes. This means that single mutant fitness is on one instance. Double mutant fitness and double interaction could be on another instance. Triple mutant fitness and triple gene interaction could be on another instance. Now you could see that if the datasets have duplicates, then we deduplicate, there could be an issue where if we were to recompute $\epsilon$ we could get a different answer. This is not inherently an issue if say the noise on duplicates was low enough such that $\Delta_{\epsilon}$ was small enough to be negligible. We are going to test this.

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_std"
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_std: float | None = Field(description="fitness standard deviation")

    @field_validator("fitness")
    def validate_fitness(cls, v):
        if math.isnan(v):
            raise ValueError("Fitness cannot be NaN")
        if v <= 0:
            return 0.0
        return v

    @model_validator(mode="after")
    def validate_label_fields(cls, values):
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values
```

```python
class GeneInteractionPhenotype(Phenotype, ModelStrict):
    graph_level: str = "hyperedge"
    label_name: str = "gene_interaction"
    label_statistic_name: str = "gene_interaction_p_value"
    gene_interaction: float = Field(
        description="""epsilon, tau, or analogous gene interaction value.
        Computed from composite fitness phenotypes."""
    )
    gene_interaction_p_value: float | None = Field(
        default=None, description="p-value of gene interaction"
    )

    @field_validator("gene_interaction")
    def validate_fitness(cls, v):
        if math.isnan(v):
            raise ValueError("Gene interaction cannot be NaN")
        return v

    # IDEA
    # This is going to be standard for all child classes of Phenotype
    # This could alternatively be moved to testing
    @model_validator(mode="after")
    def validate_label_fields(cls, values):
        # Check if label_name is a class attribute
        if values.label_name not in cls.__annotations__:
            raise ValueError(
                f"label_name '{values.label_name}' must be a class attribute"
            )

        # Check if label_statistic_name is a class attribute (if it's not None)
        if (
            values.label_statistic_name is not None
            and values.label_statistic_name not in cls.__annotations__
        ):
            raise ValueError(
                f"""label_statistic_name '{values.label_statistic_name}'
                must be a class attribute"""
            )

        return values
```

I believe that to make such a comparison it is probably preferred to compare standard error (SE) and not standard deviation (SD). But to do this our data objects would need to record the number of samples. We could do this but the raw data itself does not record the number of samples.
