---
id: eydttxka75k6eoxuxf7w35a
title: fitness-interaction-n_samples
desc: ''
updated: 1768866905247
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

From debugging `torchcell/datasets/scerevisiae/costanzo2016.py#L960-968`

```python
# Double mutant fitness
dataset = DmfCostanzo2016Dataset(
    root=osp.join(DATA_ROOT, "data/torchcell/dmf_costanzo2016_1e5"),
    io_workers=10,
    batch_size=int(1e4),
    subset_n=int(1e5),
)
print(len(dataset))
print(dataset[0])
```

We can see that we don't track `n`.

```bash
Backend macosx is interactive backend. Turning interactive mode on.
dataset[0]
{'experiment': {'experiment_type': 'fitness', 'dataset_name': 'DmfCostanzo2016Dataset', 'genotype': {...}, 'environment': {...}, 'phenotype': {...}}, 'reference': {'experiment_reference_type': 'fitness', 'dataset_name': 'DmfCostanzo2016Dataset', 'genome_reference': {...}, 'environment_reference': {...}, 'phenotype_reference': {...}}, 'publication': {'pubmed_id': '27708008', 'pubmed_url': 'https://pubmed.ncbi.nlm.nih.gov/27708008/', 'doi': '10.1126/science.aaf1420', 'doi_url': 'https://www.science.org/doi/10.1126/science.aaf1420'}}
special variables
function variables
'experiment' =
{'experiment_type': 'fitness', 'dataset_name': 'DmfCostanzo2016Dataset', 'genotype': {'perturbations': [...]}, 'environment': {'media': {...}, 'temperature': {...}}, 'phenotype': {'graph_level': 'global', 'label_name': 'fitness', 'label_statistic_name': 'fitness_std', 'fitness': 0.5779, 'fitness_std': 0.2388}}
special variables
function variables
'experiment_type' =
'fitness'
'dataset_name' =
'DmfCostanzo2016Dataset'
'genotype' =
{'perturbations': [{...}, {...}]}
special variables
function variables
'perturbations' =
[{'systematic_gene_name': 'YBR188C', 'perturbed_gene_name': 'ntc20', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via NatMX gene replacement.', 'deletion_type': 'NatMX', 'nat_mx_description': 'NatMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YBR188C_sn1898', 'natmx_deletion_type': 'SGA'}, {'systematic_gene_name': 'YMR296C', 'perturbed_gene_name': 'lcb1-10', 'description': 'Temperature sensitive allele compromised by amino acid substitution.', 'perturbation_type': 'temperature_sensitive_allele', 'ts_allele_description': 'Ts Allele Perturbation information specific to SGA experiments.', 'strain_id': 'YMR296C_tsa606', 'temperature_sensitive_allele_perturbation_type': 'SGA'}]
special variables
function variables
0 =
{'systematic_gene_name': 'YBR188C', 'perturbed_gene_name': 'ntc20', 'description': 'Deletion via KanMX or NatMX gene replacement', 'perturbation_type': 'deletion', 'deletion_description': 'Deletion via NatMX gene replacement.', 'deletion_type': 'NatMX', 'nat_mx_description': 'NatMX Deletion Perturbation information specific to SGA experiments.', 'strain_id': 'YBR188C_sn1898', 'natmx_deletion_type': 'SGA'}
1 =
{'systematic_gene_name': 'YMR296C', 'perturbed_gene_name': 'lcb1-10', 'description': 'Temperature sensitive allele compromised by amino acid substitution.', 'perturbation_type': 'temperature_sensitive_allele', 'ts_allele_description': 'Ts Allele Perturbation information specific to SGA experiments.', 'strain_id': 'YMR296C_tsa606', 'temperature_sensitive_allele_perturbation_type': 'SGA'}
len() =
2
len() =
1
'environment' =
{'media': {'name': 'YEPD', 'state': 'solid'}, 'temperature': {'value': 26.0, 'unit': 'Celsius'}}
'phenotype' =
{'graph_level': 'global', 'label_name': 'fitness', 'label_statistic_name': 'fitness_std', 'fitness': 0.5779, 'fitness_std': 0.2388}
special variables
function variables
'graph_level' =
'global'
'label_name' =
'fitness'
'label_statistic_name' =
'fitness_std'
'fitness' =
0.5779
'fitness_std' =
0.2388
len() =
5
len() =
5
'reference' =
{'experiment_reference_type': 'fitness', 'dataset_name': 'DmfCostanzo2016Dataset', 'genome_reference': {'species': 'Saccharomyces cerevisiae', 'strain': 'S288C'}, 'environment_reference': {'media': {...}, 'temperature': {...}}, 'phenotype_reference': {'graph_level': 'global', 'label_name': 'fitness', 'label_statistic_name': 'fitness_std', 'fitness': 1.0, 'fitness_std': 0.04745124871057731}}
'publication' =
{'pubmed_id': '27708008', 'pubmed_url': 'https://pubmed.ncbi.nlm.nih.gov/27708008/', 'doi': '10.1126/science.aaf1420', 'doi_url': 'https://www.science.org/doi/10.1126/science.aaf1420'}
len() =
```

nor is it tracked from processing the raw dataset `tsv` or `xlsx` file.

```
dataset.df
       Query Strain ID  ... array_perturbation_type
0       YBR188C_sn1898  ...   temperature_sensitive
1       YOL126C_sn2847  ...   temperature_sensitive
2        YDR265W_sn300  ...          KanMX_deletion
3      YDR062W_tsq2767  ...          KanMX_deletion
4      YJR017C_tsq2789  ...          KanMX_deletion
...                ...  ...                     ...
99995  YMR270C_tsq2214  ...          KanMX_deletion
99996    YBR169C_sn342  ...          KanMX_deletion
99997  YML023C_tsq1007  ...          KanMX_deletion
99998    YLR319C_sn568  ...   temperature_sensitive
99999  YBR088C_tsq1480  ...          KanMX_deletion

[100000 rows x 16 columns]
dataset.df.iloc[0]
Query Strain ID                                    YBR188C_sn1898
Query allele name                                           ntc20
Array Strain ID                                    YMR296C_tsa606
Array allele name                                         lcb1-10
Arraytype/Temp                                              TSA26
Genetic interaction score (ε)                             -0.1255
P-value                                                    0.1631
Query single mutant fitness (SMF)                          1.0252
Array SMF                                                  0.6861
Double mutant fitness                                      0.5779
Double mutant fitness standard deviation                   0.2388
Query Systematic Name                                     YBR188C
Array Systematic Name                                     YMR296C
Temperature                                                    26
query_perturbation_type                            NatMX_deletion
array_perturbation_type                     temperature_sensitive
Name: 0, dtype: object
```

I think that providing `SE` for all estimates is more appropriate than `SD`, yet we can still record these as they are provided by the original dataset. We basically want to add `n_samples` as an additional field to

`torchcell/datamodels/schema.py#L305-336`

```python
class FitnessPhenotype(Phenotype, ModelStrict):
    graph_level: str = "global"
    label_name: str = "fitness"
    label_statistic_name: str = "fitness_std" # We want this changed to se
    fitness: float = Field(description="wt_growth_rate/ko_growth_rate")
    fitness_std: float | None = Field(description="fitness standard deviation") # we can keep this as it is provided but need se
    # also want n_samples so we can try to compute se

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

Number of samples is contained within the papers. There is a lot of nuance within the paper files so we must be very careful that we are properly representing `n` for the different experimental instances.

The two relevant papers for costanzo are under `/Users/michaelvolk/Documents/projects/torchcell/papers/costanzoGlobalGeneticInteraction2016` ... the mmd file is `mathpix` ocr conversion of pdf to mmd and might be easier to search over. We should be able to process these papers with a language model to extract out a program for assigning `n` to the different rows. I think in the case of `costanzo` it is relatively simple. But there might be some difficult cases for instance, what `n_samples` is for the reference (wildtype).

Now that we are using language models to do such an extraction from text to include in the data object I am concerned about reproducibility. It is as if we need the exact lines from the paper that can be used as the evidence or justification for a programmatic mapping of e.g. `n_samples=2` for this set of experiments and `n_samples_3` for this other set of experiments distinguished by such and such. I am not the right way of doing this because it introduces a potentially non deterministic component of experiment creation.

My first thought is that we should just leave comments in code to give reason for the assignment `n_samples`, but there might be something better for the iterative improvement of dataset representation where schema is in development and subject to change.

To give you a look forward into what am planning for read this:

We also have some permissions issues. Since I have university access to papers I've downloaded them and put them on my zotero. We could orchestrate download from my zotero, and then I can provide permissions. This function would only be possible given access to the so called `torchcell` zotero library which hold paper assets other than raw data.

- s$s - \text{schema code}$
- $s - \text{dataset code}$
- $p -\text{paper artifacts}$
- $o -\text{script output - This can be very precise depending on imports etc.}$
- $f(s, d, p) = o - \text{map to new code}$
  - Here you have some intermediate data object that is constructed according to the LLM and hardcoded as a property verbatim in the source code.
  - After $o$ is produced classify whether there is any distinction between the data extracted from previous attempts. We can also compare against schema to see if this textual reasons for making change to the dataset imply anything about the schema.
j- $\text{build dataset} \rightarrow \text{test dataset}$

That basic outline gives us a mechanism for including data from text. For now I don't think we need or want this much detail as we are doing this for the first time. I think for now it will be enough to leave comments with a quote from the text for the justification of the hardcoded `n_samples.` Until we have thought this through deeper this should be enough.

The first thing to do is write the wip for dealing with adding `n_samples` to `torchcell/datasets/scerevisiae/costanzo2016.py`. We want this to be general so we can then apply the same procedure to `/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/torchcell/datasets/scerevisiae/kuzmin2018.py` and `/Users/michaelvolk/Documents/projects/torchcell.worktrees/fitness-interaction-n_samples_0/torchcell/datasets/scerevisiae/kuzmin2020.py`.

Main objectives:

1. Update fitness to have `se` and  `n_samples`, `fitness_se` is primary statistic.
2. Add to the objects in costanzo2016
3. Once checked move into kuzmin2018 and kuzmin2020

## Concerns

- Fitness is a ratio of growths so does num samples make sense with the wt number (denominator) likely has a different number of samples? I think we could just record `n_samples` of numerator and add this as comment. For reference numerator and denominator are same and we just apply same definition so it is `n_samples` of wt.
