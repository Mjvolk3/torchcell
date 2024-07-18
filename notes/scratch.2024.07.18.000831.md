---
id: hhkjq2r17a15godian8ajz1
title: 000831
desc: ''
updated: 1721280036908
created: 1721279314177
---
Nodes as it stands

```python
def get_nodes(self):
  print("Running: self._get_experiment_reference_nodes()")
  yield from self._get_experiment_reference_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self._get_genome_nodes()")
  yield from self._get_genome_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._experiment_node)")
  yield from self.get_data_by_type(self._experiment_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._genotype_node)")
  yield from self.get_data_by_type(self._genotype_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._perturbation_node)")
  yield from self.get_data_by_type(self._perturbation_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._environment_node)")
  yield from self.get_data_by_type(self._environment_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self._get_reference_environment_nodes()")
  yield from self._get_reference_environment_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._media_node)")
  yield from self.get_data_by_type(self._media_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self._get_reference_media_nodes()")
  yield from self._get_reference_media_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._temperature_node)")
  yield from self.get_data_by_type(self._temperature_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self._get_reference_temperature_nodes()")
  yield from self._get_reference_temperature_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_data_by_type(self._phenotype_node)")
  yield from self.get_data_by_type(self._phenotype_node)
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self._get_reference_phenotype_nodes()")
  yield from self._get_reference_phenotype_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Running: self.get_dataset_nodes()")
  yield from self.get_dataset_nodes()
  self.event += 1
  wandb.log({"event": self.event})
  print("Finished: get_nodes")
```

Simplified without the logging, although we do want to implement the logging.

```python
def get_nodes(self):
  yield from self._get_experiment_reference_nodes()
  yield from self._get_genome_nodes()
  yield from self.get_data_by_type(self._experiment_node)
  yield from self.get_data_by_type(self._genotype_node)
  yield from self.get_data_by_type(self._perturbation_node)
  yield from self.get_data_by_type(self._environment_node)
  yield from self._get_reference_environment_nodes()
  yield from self.get_data_by_type(self._media_node)
  yield from self._get_reference_media_nodes()
  yield from self.get_data_by_type(self._temperature_node)
  yield from self._get_reference_temperature_nodes()
  yield from self.get_data_by_type(self._phenotype_node)
  yield from self._get_reference_phenotype_nodes()
  yield from self.get_dataset_nodes()
```

A `dict` idea

```python
node_dict = {"experiment_reference_nodes", yield from self._get_experiment_reference_nodes(),
  "genome_nodes", yield from self._get_genome_nodes(),
  "experiment_nodes", yield from self.get_data_by_type(self._experiment_node),
  "genotype_nodes", yield from self.get_data_by_type(self._genotype_node),
  "perturbation_nodes", yield from self.get_data_by_type(self._perturbation_node),
  "environment_nodes", yield from self.get_data_by_type(self._environment_node),
  "reference_environment_nodes", yield from self._get_reference_environment_nodes(),
  "media_nodes", yield from self.get_data_by_type(self._media_node),
  "reference_media_nodes", yield from self._get_reference_media_nodes(),
  "temperature_node", yield from self.get_data_by_type(self._temperature_node),
  "reference_temperature_nodes", yield from self._get_reference_temperature_nodes(),
  "phenotype_node", yield from self.get_data_by_type(self._phenotype_node),
  "reference_phenotype_nodes", yield from self._get_reference_phenotype_nodes(),
  "dataset_nodes", yield from self.get_dataset_nodes()}
```

Now I still need to split up phenotype so we would have something like this... omitting repeats.

```python
node_dict = {...
  "fitness_phenotype_nodes", yield from self.get_data_by_type(self._fitness_phenotype_node),
  "reference_fitness_phenotype_nodes", yield from self._get_reference_fitness_phenotype_nodes(),
  "gene_interaction_phenotype_nodes", yield from self.get_data_by_type(self._gene_interaction_phenotype_node),
  "reference_gene_interaction_phenotype_nodes", yield from self._get_reference_gene_interaction_phenotype_nodes(),
  ...
```

This way we can keep all methods as part of the `CellAdapter` class.

Then in `class SmfCostanzo2016Adapter(CellAdapter):` we could use the hydra yaml like so.

```yaml
# for SMF
cell_adapter:
  node_methods:
    - genome_nodes
    - experiment_nodes
    - genotype_nodes
    - perturbation_nodes
    - environment_nodes
    - reference_environment_nodes
    - media_nodes
    - reference_media_nodes
    - temperature_node
    - reference_temperature_nodes
    - fitness_phenotype_node
    - reference_fitness_phenotype_nodes
    - dataset_nodes
  edge_methods:
    ...
```

And for `class SmiCostanzo2016Adapter(CellAdapter):` meaning `i` for interactions we could have.

```yaml
# for SMI
cell_adapter:
  node_methods:
    - genome_nodes
    - experiment_nodes
    - genotype_nodes
    - perturbation_nodes
    - environment_nodes
    - reference_environment_nodes
    - media_nodes
    - reference_media_nodes
    - temperature_node
    - reference_temperature_nodes
    - gene_interaction_phenotype_node
    - reference_gene_interaction_phenotype_nodes
    - dataset_nodes
  edge_methods:
    ...
```
