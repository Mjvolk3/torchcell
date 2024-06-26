---
id: gfslzeb44yjjawurq7h780e
title: Costanzo2016_adapter
desc: ''
updated: 1707156929027
created: 1705537951301
---
## Useful Functions for Debugging Adapter and Printing Ontology

`bc.show_ontology_structure` can be used to print the ontology before processing any of the node data.

```python
if __name__ == "__main__":
  from biocypher import BioCypher

  #Simple Testing
  dataset = SmfCostanzo2016Dataset()
  adapter = SmfCostanzo2016Adapter(dataset=dataset)
  [i for i in adapter.get_nodes()]
  [i for i in adapter.get_edges()]
  
  ## Advanced Testing
  bc = BioCypher()
  dataset = SmfCostanzo2016Dataset()
  adapter = SmfCostanzo2016Adapter(dataset=dataset)
  print(bc.show_ontology_structure())
  bc.write_nodes(adapter.get_nodes())
  bc.write_edges(adapter.get_edges())

  # # Write admin import statement and schema information (for biochatter)
  bc.write_import_call()
  bc.write_schema_info(as_node=True)

  # # Print summary
  bc.summary()
  print()
```

### TypeError NoneType object is not iterable

This error typically indicates that we are not returning the nodes properly. Just check for `return nodes` or `return edges`.

```bash
File "/Users/michaelvolk/Documents/projects/torchcell/torchcell/adapters/costanzo2016_adapter.py", line 701, in get_nodes
    for node in self._get_phenotype_nodes():
TypeError: 'NoneType' object is not iterable
```

### Path Issue Indicated by KeyError 'experiment reference'

If the `schema_config.yaml` is properly written this error indicates that there was some issue in specifying the correct `schema_config.yaml` path.

```bash
File "/Users/michaelvolk/opt/miniconda3/envs/torchcell/lib/python3.11/site-packages/networkx/classes/coreviews.py", line 81, in __getitem__
return AtlasView(self._atlas[name])
                    ~~~~~~~~~~~^^^^^^
KeyError: 'experiment reference'
```

## Warning - The Ontology Contains Multiple Inheritance

According to @Sebastian-Lobentanzer this shouldn't be an issue for the knowledge graph, but it there to indicate that they cannot display a cycle with the command line output.

![](./assets/images/torchcell.adapters.costanzo2016_adapter.md.warning-the-ontology-contains-multiple-inheritance.png)

## Using Static Methods like in get_perturbation

This has the luxury of avoiding iterating over the entire dataset, but it doesn't have the ability to check for duplicates. When you iterate over the entire dataset with a regular method it is easier to create a set of data within the scope that that can be used for removing duplicates. It is so cumbersome with a `@staticmethod` pushing the duplicate tracking a function above making things difficult to read.

I think think the `@staticmethod` is justified when the data has some guarantee that there are no duplicates. This is why it can be safely used on perturbation. With the hourglass strategy we known that `strain_id` is always unique. This comes from knowledge about the preprocessing of the data and probably should be used with some care. We have been using the hourglass strategy with some discretion, for instance it is not used on media. There isn't any additional information that would allow us to used this design pattern. Besides we would end up with weird things like the two different nodes at a specific temperature but with some additional differentiating attribute. We don't want this property for graph querying.

Everything below the red line classes should used Liskov substitution.

 ![](./assets/drawio/ontology_pydantic_hourglass_data_model.drawio.png)

```python
@staticmethod
def _get_perturbation(
    genotype: BaseGenotype,
) -> Generator[BioCypherNode, None, None]:
    if genotype.perturbation:
        i = 1
        perturbation_id = hashlib.md5(
            json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
        ).hexdigest()

        yield BioCypherNode(
            node_id=perturbation_id,
            preferred_id=f"perturbation_{i}",
            node_label="perturbation",
            properties={
                "systematic_gene_name": [
                    genotype.perturbation.systematic_gene_name
                ],
                "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                "description": genotype.perturbation.description,
                "perturbation_type": genotype.perturbation.perturbation_type,
                "strain_id": genotype.perturbation.strain_id,
                "serialized_data": json.dumps(genotype.perturbation.model_dump()),
            },
        )
```

## MVP Costanzo2016 Adapter Without Multiprocessing

```python
# torchcell/adapters/costanzo2016_adapter.py
# [[torchcell.adapters.costanzo2016_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/costanzo2016_adapter.py
# Test file: tests/torchcell/adapters/test_costanzo2016_adapter.py

from tqdm import tqdm
import hashlib
import random
import string
from enum import Enum, auto
from functools import lru_cache
from itertools import chain
from typing import Optional
import json
import pandas as pd
from biocypher._create import BioCypherEdge, BioCypherNode
from biocypher._logger import logger
from typing import Generator, Set
import torch
from torchcell.datasets.scerevisiae import (
    SmfCostanzo2016Dataset,
    DmfCostanzo2016Dataset,
)
from torchcell.datamodels import BaseGenotype, InterferenceGenotype, DeletionGenotype
from sortedcontainers import SortedList
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

logger.debug(f"Loading module {__name__}.")

dataset = SmfCostanzo2016Dataset()


class SmfCostanzo2016Adapter:
    def __init__(self, dataset: SmfCostanzo2016Dataset):
        self.dataset = dataset

    def get_nodes(self) -> None:
        logger.info("Getting nodes.")
        logger.info("Get experiment reference nodes.")
        yield from self._get_experiment_reference_nodes()
        logger.info("Get genome nodes.")
        yield from self._get_genome_nodes()
        logger.info("Get experiment nodes.")
        yield from self._get_experiment_nodes()
        logger.info("Get dataset nodes.")
        yield from self._get_dataset_nodes()
        logger.info("Get genotype nodes.")
        logger.info("--- perturbation nodes.")
        yield from self._get_genotype_nodes()
        logger.info("Get environment nodes.")
        yield from self._get_environment_nodes()
        logger.info("Get media nodes.")
        yield from self._get_media_nodes()
        logger.info("Get temperature nodes.")
        yield from self._get_temperature_nodes()
        logger.info("Get phenotype nodes.")
        yield from self._get_phenotype_nodes()

    def _get_experiment_reference_nodes(self) -> None:
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"CostanzoSmf2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )

    def _get_genome_nodes(self) -> None:
        seen_node_ids: Set[str] = set()

        for i, data in enumerate(self.dataset.experiment_reference_index):
            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                yield BioCypherNode(
                    node_id=genome_id,
                    preferred_id=f"reference_genome_{i}",
                    node_label="genome",
                    properties={
                        "species": data.reference.reference_genome.species,
                        "strain": data.reference.reference_genome.strain,
                        "serialized_data": json.dumps(
                            data.reference.reference_genome.model_dump()
                        ),
                    },
                )

    def _get_experiment_nodes(self) -> None:
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"CostanzoSmf2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            genotype_id = hashlib.md5(
                json.dumps(data["experiment"].genotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if genotype_id not in seen_node_ids:
                seen_node_ids.add(genotype_id)
                systematic_gene_name = data[
                    "experiment"
                ].genotype.perturbation.systematic_gene_name
                perturbed_gene_name = data[
                    "experiment"
                ].genotype.perturbation.perturbed_gene_name
                description = data["experiment"].genotype.perturbation.description
                perturbation_type = data[
                    "experiment"
                ].genotype.perturbation.perturbation_type

                self._get_perturbation(data["experiment"].genotype)

                yield BioCypherNode(
                    node_id=genotype_id,
                    preferred_id=f"genotype_{i}",
                    node_label="genotype",
                    properties={
                        "systematic_gene_names": [systematic_gene_name],
                        "perturbed_gene_names": [perturbed_gene_name],
                        "is_deletion_genotype": isinstance(
                            data["experiment"].genotype, DeletionGenotype
                        ),
                        "is_interference_genotype": isinstance(
                            data["experiment"].genotype, InterferenceGenotype
                        ),
                        "description": description,
                        "perturbation_types": [perturbation_type],
                        "serialized_data": json.dumps(
                            data["experiment"].genotype.model_dump()
                        ),
                    },
                )

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        if genotype.perturbation:
            i = 1
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=perturbation_id,
                preferred_id=f"perturbation_{i}",
                node_label="perturbation",
                properties={
                    "systematic_gene_name": [
                        genotype.perturbation.systematic_gene_name
                    ],
                    "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                    "description": genotype.perturbation.description,
                    "perturbation_type": genotype.perturbation.perturbation_type,
                    "strain_id": genotype.perturbation.strain_id,
                    "serialized_data": json.dumps(genotype.perturbation.model_dump()),
                },
            )

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data["experiment"].environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.model_dump()
                        ),
                    },
                )
        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["reference"].reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                )

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.model_dump()
                        ),
                    },
                )

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.media.model_dump()
                        ),
                    },
                )
        for i, data in enumerate(self.dataset):
            media_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.media.model_dump()
                        ),
                    },
                )

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data["experiment"].environment.temperature.value,
                        "unit": data["experiment"].environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.temperature.model_dump()
                        ),
                    },
                )

        for i, data in enumerate(self.dataset):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "description": data[
                            "reference"
                        ].reference_environment.temperature.description,
                        "serialized_data": json.dumps(
                            data[
                                "reference"
                            ].reference_environment.temperature.model_dump()
                        ),
                    },
                )

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in enumerate(self.dataset):
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["experiment"].phenotype.model_dump()
                        ),
                    },
                )

        # References
        for i, data in enumerate(self.dataset):
            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["reference"].reference_phenotype.graph_level
                label = data["reference"].reference_phenotype.label
                label_error = data["reference"].reference_phenotype.label_error
                fitness = data["reference"].reference_phenotype.fitness
                fitness_std = data["reference"].reference_phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["reference"].reference_phenotype.model_dump()
                        ),
                    },
                )

    def _get_dataset_nodes(self) -> None:
        yield BioCypherNode(
            node_id="CostanzoSmf2016",
            preferred_id="CostanzoSmf2016",
            node_label="dataset",
        )

    def get_edges(self) -> None:
        logger.info("Generating edges.")
        logger.info("Get dataset experiment reference edges.")
        yield from self._get_dataset_experiment_ref_edges()
        logger.info("Get experiment dataset edges.")
        yield from self._get_experiment_dataset_edges()
        logger.info("Get experiment reference experiment edges.")
        yield from self._get_experiment_ref_experiment_edges()
        logger.info("Get genotype experiment edges.")
        logger.info("--- perturbation genotype edges.")
        yield from self._get_genotype_experiment_edges()
        logger.info("Get environment experiment edges.")
        yield from self._get_environment_experiment_edges()
        logger.info("Get environment experiment reference edges.")
        yield from self._get_environment_experiment_ref_edges()
        logger.info("Get phenotype experiment edges.")
        yield from self._get_phenotype_experiment_edges()
        logger.info("Get phenotype experiment reference edges.")
        yield from self._get_phenotype_experiment_ref_edges()
        logger.info("Get media environment edges.")
        yield from self._get_media_environment_edges()
        logger.info("Get temperature environment edges.")
        yield from self._get_temperature_environment_edges()
        logger.info("Get genome experiment reference edges.")
        yield from self._get_genome_edges()

    def _get_dataset_experiment_ref_edges(self):
        # concept level
        for data in self.dataset:
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="CostanzoSmf2016",
                relationship_label="experiment reference member of",
            )

    def _get_experiment_dataset_edges(self):
        # concept level
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_id,
                target_id="CostanzoSmf2016",
                relationship_label="experiment member of",
            )

    def _get_experiment_ref_experiment_edges(self):
        # instance level
        for data in self.dataset.experiment_reference_index:
            dataset_subset = self.dataset[torch.tensor(data.index)]
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.md5(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                yield BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # CHECK if needed - don't think needed since exp ref index
        # seen_genotype_experiment_pairs: Set[tuple] = set()
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            genotype_id = hashlib.md5(
                json.dumps(data["experiment"].genotype.model_dump()).encode("utf-8")
            ).hexdigest()

            self._get_perturbation_genotype_edges(
                genotype=data["experiment"].genotype, genotype_id=genotype_id
            )

            # CHECK if needed - don't think needed since exp ref index
            # genotype_experiment_pair = (genotype_id, experiment_id)
            # if genotype_experiment_pair not in seen_genotype_experiment_pairs:
            #     seen_genotype_experiment_pairs.add(genotype_experiment_pair)

            yield BioCypherEdge(
                source_id=genotype_id,
                target_id=experiment_id,
                relationship_label="genotype member of",
            )

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        if genotype.perturbation:
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherEdge(
                source_id=perturbation_id,
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_pairs: Set[tuple] = set()

        # Linking environments to experiments
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_ref_pairs: Set[tuple] = set()

        # Linking environments to experiment references
        for i, data in enumerate(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.md5(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_pairs: Set[tuple] = set()

        # Linking phenotypes to experiments
        for i, data in enumerate(self.dataset):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()

        # Linking phenotypes to experiment references
        for i, data in enumerate(self.dataset):
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_media_environment_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)

                yield BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )

    def _get_temperature_environment_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_temperature_environment_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)

                yield BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )

    def _get_genome_edges(self) -> None:
        seen_genome_experiment_ref_pairs: Set[tuple] = set()

        for i, data in enumerate(self.dataset):
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.md5(
                json.dumps(data["reference"].reference_genome.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )


class DmfCostanzo2016Adapter:
    def __init__(self, dataset: DmfCostanzo2016Dataset):
        self.dataset = dataset

    def get_nodes(self) -> None:
        logger.info("Getting nodes.")
        logger.info("Get experiment reference nodes.")
        yield from self._get_experiment_reference_nodes()
        logger.info("Get genome nodes.")
        yield from self._get_genome_nodes()
        logger.info("Get experiment nodes.")
        yield from self._get_experiment_nodes()
        logger.info("Get dataset nodes.")
        yield from self._get_dataset_nodes()
        logger.info("Get genotype nodes.")
        logger.info("--- perturbation nodes.")
        yield from self._get_genotype_nodes()
        logger.info("Get environment nodes.")
        yield from self._get_environment_nodes()
        logger.info("Get media nodes.")
        yield from self._get_media_nodes()
        logger.info("Get temperature nodes.")
        yield from self._get_temperature_nodes()
        logger.info("Get phenotype nodes.")
        yield from self._get_phenotype_nodes()

    def _get_experiment_reference_nodes(self) -> None:
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"CostanzoSmf2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )

    def _get_genome_nodes(self) -> None:
        seen_node_ids: Set[str] = set()

        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                yield BioCypherNode(
                    node_id=genome_id,
                    preferred_id=f"reference_genome_{i}",
                    node_label="genome",
                    properties={
                        "species": data.reference.reference_genome.species,
                        "strain": data.reference.reference_genome.strain,
                        "serialized_data": json.dumps(
                            data.reference.reference_genome.model_dump()
                        ),
                    },
                )

    def _get_experiment_nodes(self) -> None:
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"CostanzoSmf2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                if genotype_id not in seen_node_ids:
                    seen_node_ids.add(genotype_id)
                    systematic_gene_name = genotype.perturbation.systematic_gene_name
                    perturbed_gene_name = genotype.perturbation.perturbed_gene_name
                    description = genotype.perturbation.description
                    perturbation_type = genotype.perturbation.perturbation_type
                    self._get_perturbation(genotype)

                    yield BioCypherNode(
                        node_id=genotype_id,
                        preferred_id=f"genotype_{i}",
                        node_label="genotype",
                        properties={
                            "systematic_gene_names": [systematic_gene_name],
                            "perturbed_gene_names": [perturbed_gene_name],
                            "is_deletion_genotype": isinstance(
                                data["experiment"].genotype, DeletionGenotype
                            ),
                            "is_interference_genotype": isinstance(
                                data["experiment"].genotype, InterferenceGenotype
                            ),
                            "description": description,
                            "perturbation_types": [perturbation_type],
                            "serialized_data": json.dumps(genotype.model_dump()),
                        },
                    )

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        if genotype.perturbation:
            i = 1
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=perturbation_id,
                preferred_id=f"perturbation_{i}",
                node_label="perturbation",
                properties={
                    "systematic_gene_name": [
                        genotype.perturbation.systematic_gene_name
                    ],
                    "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                    "description": genotype.perturbation.description,
                    "perturbation_type": genotype.perturbation.perturbation_type,
                    "strain_id": genotype.perturbation.strain_id,
                    "serialized_data": json.dumps(genotype.perturbation.model_dump()),
                },
            )

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data["experiment"].environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.model_dump()
                        ),
                    },
                )
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["reference"].reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                )

                yield BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.model_dump()
                        ),
                    },
                )

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.media.model_dump()
                        ),
                    },
                )
        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                yield BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.media.model_dump()
                        ),
                    },
                )

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data["experiment"].environment.temperature.value,
                        "unit": data["experiment"].environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.temperature.model_dump()
                        ),
                    },
                )

        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                yield BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "description": data[
                            "reference"
                        ].reference_environment.temperature.description,
                        "serialized_data": json.dumps(
                            data[
                                "reference"
                            ].reference_environment.temperature.model_dump()
                        ),
                    },
                )

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["experiment"].phenotype.model_dump()
                        ),
                    },
                )

        # References
        for i, data in tqdm(enumerate(self.dataset)):
            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["reference"].reference_phenotype.graph_level
                label = data["reference"].reference_phenotype.label
                label_error = data["reference"].reference_phenotype.label_error
                fitness = data["reference"].reference_phenotype.fitness
                fitness_std = data["reference"].reference_phenotype.fitness_std

                yield BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["reference"].reference_phenotype.model_dump()
                        ),
                    },
                )

    def _get_dataset_nodes(self) -> None:
        yield BioCypherNode(
            node_id="CostanzoSmf2016",
            preferred_id="CostanzoSmf2016",
            node_label="dataset",
        )

    def get_edges(self) -> None:
        logger.info("Generating edges.")
        logger.info("Get dataset experiment reference edges.")
        yield from self._get_dataset_experiment_ref_edges()
        logger.info("Get experiment dataset edges.")
        yield from self._get_experiment_dataset_edges()
        logger.info("Get experiment reference experiment edges.")
        yield from self._get_experiment_ref_experiment_edges()
        logger.info("Get genotype experiment edges.")
        logger.info("--- perturbation genotype edges.")
        yield from self._get_genotype_experiment_edges()
        logger.info("Get environment experiment edges.")
        yield from self._get_environment_experiment_edges()
        logger.info("Get environment experiment reference edges.")
        yield from self._get_environment_experiment_ref_edges()
        logger.info("Get phenotype experiment edges.")
        yield from self._get_phenotype_experiment_edges()
        logger.info("Get phenotype experiment reference edges.")
        yield from self._get_phenotype_experiment_ref_edges()
        logger.info("Get media environment edges.")
        yield from self._get_media_environment_edges()
        logger.info("Get temperature environment edges.")
        yield from self._get_temperature_environment_edges()
        logger.info("Get genome experiment reference edges.")
        yield from self._get_genome_edges()

    def _get_dataset_experiment_ref_edges(self):
        # concept level
        for data in tqdm(self.dataset):
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="CostanzoSmf2016",
                relationship_label="experiment reference member of",
            )

    def _get_experiment_dataset_edges(self):
        # concept level
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_id,
                target_id="CostanzoSmf2016",
                relationship_label="experiment member of",
            )

    def _get_experiment_ref_experiment_edges(self):
        # instance level
        print()
        for data in tqdm(self.dataset.experiment_reference_index):
            dataset_subset = self.dataset[torch.tensor(data.index)]
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.md5(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                yield BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )

    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # CHECK if needed - don't think needed since exp ref index
        # seen_genotype_experiment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                self._get_perturbation_genotype_edges(
                    genotype=genotype, genotype_id=genotype_id
                )

                # CHECK if needed - don't think needed since exp ref index
                # genotype_experiment_pair = (genotype_id, experiment_id)
                # if genotype_experiment_pair not in seen_genotype_experiment_pairs:
                #     seen_genotype_experiment_pairs.add(genotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=genotype_id,
                    target_id=experiment_id,
                    relationship_label="genotype member of",
                )

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        if genotype.perturbation:
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherEdge(
                source_id=perturbation_id,
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_pairs: Set[tuple] = set()

        # Linking environments to experiments
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_environment_experiment_ref_pairs: Set[tuple] = set()

        # Linking environments to experiment references
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.md5(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_pairs: Set[tuple] = set()

        # Linking phenotypes to experiments
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()

        # Linking phenotypes to experiment references
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            # Get the phenotype ID associated with the experiment reference
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        seen_media_environment_pairs: Set[tuple] = set()

        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)

                yield BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )

    def _get_temperature_environment_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        seen_temperature_environment_pairs: Set[tuple] = set()

        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)

                yield BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )

    def _get_genome_edges(self) -> None:
        seen_genome_experiment_ref_pairs: Set[tuple] = set()

        for i, data in tqdm(enumerate(self.dataset)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.md5(
                json.dumps(data["reference"].reference_genome.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                yield BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )




if __name__ == "__main__":
    from biocypher import BioCypher

    # # # Simple Testing
    # dataset = SmfCostanzo2016Dataset()
    # adapter = SmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    # ## Advanced Testing
    # bc = BioCypher()
    # dataset = SmfCostanzo2016Dataset()
    # adapter = SmfCostanzo2016Adapter(dataset=dataset)
    # bc.write_nodes(adapter.get_nodes())
    # bc.write_edges(adapter.get_edges())

    # # # Write admin import statement and schema information (for biochatter)
    # bc.write_import_call()
    # bc.write_schema_info(as_node=True)

    # # # Print summary
    # bc.summary()
    # print()

    ## Dmf
    # Simple Testing
    # dataset = DmfCostanzo2016Dataset()
    # adapter = DmfCostanzo2016Adapter(dataset=dataset)
    # [i for i in adapter.get_nodes()]
    # [i for i in adapter.get_edges()]

    # Advanced Testing
    bc = BioCypher()
    dataset = DmfCostanzo2016Dataset()
    # dataset = DmfCostanzo2016Dataset(
    #         root="data/torchcell/dmf_costanzo2016_subset_n_1e6",
    #         subset_n=int(1e6),
    #         preprocess=None,
    #     )
    num_workers = multiprocessing.cpu_count()
    adapter = DmfCostanzo2016Adapter(dataset=dataset, num_workers=num_workers)
    bc.show_ontology_structure()
    bc.write_nodes(adapter.get_nodes())
    bc.write_edges(adapter.get_edges())
    bc.write_import_call()
    bc.write_schema_info(as_node=True)
    bc.summary()

```

## Costanzo2016 Adapter with Multiprocessing

```python
class DmfCostanzo2016Adapter:
    def __init__(self, dataset: DmfCostanzo2016Dataset, num_workers: int = 1):
        self.dataset = dataset
        self.num_workers = num_workers

    def get_nodes(self):
        methods = [
            self._get_experiment_reference_nodes,
            self._get_genome_nodes,
            self._get_experiment_nodes,
            self._get_dataset_nodes,
            self._get_genotype_nodes,
            self._get_environment_nodes,
            self._get_media_nodes,
            self._get_temperature_nodes,
            self._get_phenotype_nodes,
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(method) for method in methods]
            for future in as_completed(futures):
                try:
                    node_generator = future.result()
                    for node in node_generator:
                        yield node
                except Exception as exc:
                    logger.error(
                        f"Node generation method generated an exception: {exc}"
                    )

    def _get_experiment_reference_nodes(self) -> None:
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id=f"DmfCostanzo2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.reference.model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_genome_nodes(self) -> None:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
                    node_id=genome_id,
                    preferred_id=f"reference_genome_{i}",
                    node_label="genome",
                    properties={
                        "species": data.reference.reference_genome.species,
                        "strain": data.reference.reference_genome.strain,
                        "serialized_data": json.dumps(
                            data.reference.reference_genome.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_experiment_nodes(self) -> None:
        nodes = []
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()

            node = BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"DmfCostanzo2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data["experiment"].model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_genotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                if genotype_id not in seen_node_ids:
                    seen_node_ids.add(genotype_id)
                    systematic_gene_name = genotype.perturbation.systematic_gene_name
                    perturbed_gene_name = genotype.perturbation.perturbed_gene_name
                    description = genotype.perturbation.description
                    perturbation_type = genotype.perturbation.perturbation_type
                    self._get_perturbation(genotype)

                    node = BioCypherNode(
                        node_id=genotype_id,
                        preferred_id=f"genotype_{i}",
                        node_label="genotype",
                        properties={
                            "systematic_gene_names": [systematic_gene_name],
                            "perturbed_gene_names": [perturbed_gene_name],
                            "is_deletion_genotype": isinstance(
                                data["experiment"].genotype, DeletionGenotype
                            ),
                            "is_interference_genotype": isinstance(
                                data["experiment"].genotype, InterferenceGenotype
                            ),
                            "description": description,
                            "perturbation_types": [perturbation_type],
                            "serialized_data": json.dumps(genotype.model_dump()),
                        },
                    )
                    nodes.append(node)
        return nodes

    @staticmethod
    def _get_perturbation(
        genotype: BaseGenotype,
    ) -> Generator[BioCypherNode, None, None]:
        nodes = []
        if genotype.perturbation:
            i = 1
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            node = BioCypherNode(
                node_id=perturbation_id,
                preferred_id=f"perturbation_{i}",
                node_label="perturbation",
                properties={
                    "systematic_gene_name": [
                        genotype.perturbation.systematic_gene_name
                    ],
                    "perturbed_gene_name": [genotype.perturbation.perturbed_gene_name],
                    "description": genotype.perturbation.description,
                    "perturbation_type": genotype.perturbation.perturbation_type,
                    "strain_id": genotype.perturbation.strain_id,
                    "serialized_data": json.dumps(genotype.perturbation.model_dump()),
                },
            )
            nodes.append(node)
        return nodes

    def _get_environment_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(data["experiment"].environment.media.model_dump())

                node = BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data["experiment"].environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.model_dump()
                        ),
                    },
                )
                nodes.append(node)

        for i, data in tqdm(enumerate(self.dataset)):
            environment_id = hashlib.md5(
                json.dumps(data["reference"].reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            node_id = environment_id

            if node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                media = json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                )

                node = BioCypherNode(
                    node_id=node_id,
                    preferred_id=f"environment_{i}",
                    node_label="environment",
                    properties={
                        "temperature": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "media": media,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_media_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(data["experiment"].environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["experiment"].environment.media.name
                state = data["experiment"].environment.media.state

                node = BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.media.model_dump()
                        ),
                    },
                )
                nodes.append(node)

        for i, data in tqdm(enumerate(self.dataset)):
            media_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.media.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data["reference"].reference_environment.media.name
                state = data["reference"].reference_environment.media.state

                node = BioCypherNode(
                    node_id=media_id,
                    preferred_id=f"media_{media_id}",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data["reference"].reference_environment.media.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_temperature_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["experiment"].environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                node = BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data["experiment"].environment.temperature.value,
                        "unit": data["experiment"].environment.temperature.unit,
                        "serialized_data": json.dumps(
                            data["experiment"].environment.temperature.model_dump()
                        ),
                    },
                )
                nodes.append(node)

        for i, data in tqdm(enumerate(self.dataset)):
            temperature_id = hashlib.md5(
                json.dumps(
                    data["reference"].reference_environment.temperature.model_dump()
                ).encode("utf-8")
            ).hexdigest()

            if temperature_id not in seen_node_ids:
                seen_node_ids.add(temperature_id)

                node = BioCypherNode(
                    node_id=temperature_id,
                    preferred_id=f"temperature_{temperature_id}",
                    node_label="temperature",
                    properties={
                        "value": data[
                            "reference"
                        ].reference_environment.temperature.value,
                        "description": data[
                            "reference"
                        ].reference_environment.temperature.description,
                        "serialized_data": json.dumps(
                            data[
                                "reference"
                            ].reference_environment.temperature.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_phenotype_nodes(self) -> Generator[BioCypherNode, None, None]:
        nodes = []
        seen_node_ids: Set[str] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["experiment"].phenotype.graph_level
                label = data["experiment"].phenotype.label
                label_error = data["experiment"].phenotype.label_error
                fitness = data["experiment"].phenotype.fitness
                fitness_std = data["experiment"].phenotype.fitness_std

                node = BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["experiment"].phenotype.model_dump()
                        ),
                    },
                )
                nodes.append(node)

        for i, data in tqdm(enumerate(self.dataset)):
            phenotype_id = hashlib.md5(
                json.dumps(data["reference"].reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            if phenotype_id not in seen_node_ids:
                seen_node_ids.add(phenotype_id)
                graph_level = data["reference"].reference_phenotype.graph_level
                label = data["reference"].reference_phenotype.label
                label_error = data["reference"].reference_phenotype.label_error
                fitness = data["reference"].reference_phenotype.fitness
                fitness_std = data["reference"].reference_phenotype.fitness_std

                node = BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id=f"phenotype_{phenotype_id}",
                    node_label="phenotype",
                    properties={
                        "graph_level": graph_level,
                        "label": label,
                        "label_error": label_error,
                        "fitness": fitness,
                        "fitness_std": fitness_std,
                        "serialized_data": json.dumps(
                            data["reference"].reference_phenotype.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    def _get_dataset_nodes(self) -> None:
        nodes = [
            BioCypherNode(
                node_id="DmfCostanzo2016",
                preferred_id="DmfCostanzo2016",
                node_label="dataset",
            )
        ]
        return nodes

    def get_edges(self):
        methods = [
            self._get_dataset_experiment_ref_edges,
            self._get_experiment_dataset_edges,
            self._get_experiment_ref_experiment_edges,
            self._get_genotype_experiment_edges,
            self._get_environment_experiment_edges,
            self._get_environment_experiment_ref_edges,
            self._get_phenotype_experiment_edges,
            self._get_phenotype_experiment_ref_edges,
            self._get_media_environment_edges,
            self._get_temperature_environment_edges,
            self._get_genome_edges,
        ]

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Divide methods into smaller chunks
            method_chunks = [
                methods[i :: self.num_workers] for i in range(self.num_workers)
            ]

            futures = []
            for chunk in method_chunks:
                for method in chunk:
                    futures.append(executor.submit(method))

            # Process futures as they complete and yield edges
            for future in as_completed(futures):
                try:
                    edge_generator = future.result()
                    for edge in edge_generator:
                        yield edge
                except Exception as exc:
                    logger.error(
                        f"Edge generation method generated an exception: {exc}"
                    )

    def _get_dataset_experiment_ref_edges(self):
        edges = []
        for data in tqdm(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_ref_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_dataset_edges(self):
        edges = []
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=experiment_id,
                target_id="DmfCostanzo2016",
                relationship_label="experiment member of",
            )
            edges.append(edge)
        return edges

    def _get_experiment_ref_experiment_edges(self):
        edges = []
        for data in tqdm(self.dataset.experiment_reference_index):
            dataset_subset = self.dataset[torch.tensor(data.index)]
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, data in enumerate(dataset_subset):
                experiment_id = hashlib.md5(
                    json.dumps(data["experiment"].model_dump()).encode("utf-8")
                ).hexdigest()
                edge = BioCypherEdge(
                    source_id=experiment_ref_id,
                    target_id=experiment_id,
                    relationship_label="experiment reference of",
                )
                edges.append(edge)
        return edges


    def _get_genotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            for genotype in data["experiment"].genotype:
                genotype_id = hashlib.md5(
                    json.dumps(genotype.model_dump()).encode("utf-8")
                ).hexdigest()

                self._get_perturbation_genotype_edges(
                    genotype=genotype, genotype_id=genotype_id
                )
                edge = BioCypherEdge(
                    source_id=genotype_id,
                    target_id=experiment_id,
                    relationship_label="genotype member of",
                )
                edges.append(edge)
        return edges

    @staticmethod
    def _get_perturbation_genotype_edges(
        genotype: BaseGenotype, genotype_id: str
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        if genotype.perturbation:
            perturbation_id = hashlib.md5(
                json.dumps(genotype.perturbation.model_dump()).encode("utf-8")
            ).hexdigest()

            edge = BioCypherEdge(
                source_id=perturbation_id,
                target_id=genotype_id,
                relationship_label="perturbation member of",
            )
            edges.append(edge)
        return edges

    def _get_environment_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_environment_experiment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = hashlib.md5(
                json.dumps(data["experiment"].environment.model_dump()).encode("utf-8")
            ).hexdigest()

            env_experiment_pair = (environment_id, experiment_id)
            if env_experiment_pair not in seen_environment_experiment_pairs:
                seen_environment_experiment_pairs.add(env_experiment_pair)

                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    def _get_environment_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_environment_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            environment_id = hashlib.md5(
                json.dumps(data.reference.reference_environment.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    def _get_phenotype_experiment_edges(self) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_phenotype_experiment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset)):
            experiment_id = hashlib.md5(
                json.dumps(data["experiment"].model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.md5(
                json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_experiment_pair = (phenotype_id, experiment_id)
            if phenotype_experiment_pair not in seen_phenotype_experiment_pairs:
                seen_phenotype_experiment_pairs.add(phenotype_experiment_pair)

                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    def _get_phenotype_experiment_ref_edges(
        self,
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_phenotype_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            phenotype_id = hashlib.md5(
                json.dumps(data.reference.reference_phenotype.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()

            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    def _get_media_environment_edges(self) -> Generator[BioCypherEdge, None, None]:
        # Optimized by using reference
        # We know reference contains all media and envs
        edges = []
        seen_media_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            environment_id = hashlib.md5(
                json.dumps(data.reference.environment.model_dump()).encode("utf-8")
            ).hexdigest()
            media_id = hashlib.md5(
                json.dumps(data.reference.environment.media.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            media_environment_pair = (media_id, environment_id)
            if media_environment_pair not in seen_media_environment_pairs:
                seen_media_environment_pairs.add(media_environment_pair)
                edge = BioCypherEdge(
                    source_id=media_id,
                    target_id=environment_id,
                    relationship_label="media member of",
                )
                edges.append(edge)
        return edges

    def _get_temperature_environment_edges(
        self,
        # Optimized by using reference
        # We know reference contain all envs and temps
    ) -> Generator[BioCypherEdge, None, None]:
        edges = []
        seen_temperature_environment_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            environment_id = hashlib.md5(
                json.dumps(data.reference.environment.model_dump()).encode("utf-8")
            ).hexdigest()
            temperature_id = hashlib.md5(
                json.dumps(data.reference.environment.temperature.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            temperature_environment_pair = (temperature_id, environment_id)
            if temperature_environment_pair not in seen_temperature_environment_pairs:
                seen_temperature_environment_pairs.add(temperature_environment_pair)
                edge = BioCypherEdge(
                    source_id=temperature_id,
                    target_id=environment_id,
                    relationship_label="temperature member of",
                )
                edges.append(edge)
        return edges

    def _get_genome_edges(self) -> None:
        edges = []
        seen_genome_experiment_ref_pairs: Set[tuple] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.md5(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_id = hashlib.md5(
                json.dumps(data.reference.reference_genome.model_dump()).encode("utf-8")
            ).hexdigest()

            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )
                edges.append(edge)
        return edges
```
