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

from torchcell.datasets.scerevisiae import NeoSmfCostanzo2016Dataset

logger.debug(f"Loading module {__name__}.")

dataset = NeoSmfCostanzo2016Dataset()


class CostanzoSmfAdapter:
    def __init__(
        self,
        dataset: NeoSmfCostanzo2016Dataset,
        node_types: Optional[list] = None,
        node_fields: Optional[list] = None,
        edge_types: Optional[list] = None,
        edge_fields: Optional[list] = None,
    ):
        self.dataset = dataset
        # self._set_types_and_fields(
        #     node_types,
        #     node_fields,
        #     edge_types,
        #     edge_fields,
        # )
        self._preprocess_data()

    def _preprocess_data(self) -> None:
        logger.info("Preprocessing data.")

        self.experiments = dataset

    def get_nodes(self) -> None:
        logger.info("Getting nodes.")
        logger.info("Get experiment reference nodes.")
        yield from self._get_experiment_reference_nodes()
        logger.info("Get experiment nodes.")
        yield from self._get_experiment_nodes()
        logger.info("Get dataset nodes.")
        yield from self._get_dataset_nodes()
        logger.info("Get genotype nodes.")
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
        for i, data in enumerate(self.dataset.reference_index):
            study_ref_id = hashlib.md5(
                json.dumps(data.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherNode(
                node_id=study_ref_id,
                preferred_id=f"CostanzoSmf2016_reference_{i}",
                node_label="experiment reference",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(data.model_dump()),
                },
            )
        
    def _get_experiment_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            experiment_id = hashlib.md5(
                json.dumps(experiment.model_dump()).encode("utf-8")
            ).hexdigest()

            yield BioCypherNode(
                node_id=experiment_id,
                preferred_id=f"CostanzoSmf2016_{i}",
                node_label="experiment",
                properties={
                    "dataset_index": i,
                    "serialized_data": json.dumps(experiment.model_dump()),
                },
            )
    
    def _get_genotype_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            experiment_id = hashlib.md5(
                json.dumps(experiment.model_dump()).encode("utf-8")
            ).hexdigest()
            # CHECK should we be using something  
            genotype_id = hashlib.md5(
                json.dumps(experiment.genotype.model_dump()).encode("utf-8")
            ).hexdigest()
            # HACK if we don't do this for node id, then we get deduplicated genotypes that belong to different experiments.
            # Maybe we want this. Not sure.
            node_id = ("_").join([experiment_id, genotype_id])
            node_id = genotype_id
            # collect data
            sys_gene_name = experiment.genotype.perturbation.sys_gene_name.name
            perturbed_gene_name = experiment.genotype.perturbation.perturbed_gene_name
            description = experiment.genotype.perturbation.description
            perturbation_type = experiment.genotype.perturbation.perturbation_type
            # HACK casting to lists bc there is typically a list of perturbations
            yield BioCypherNode(
                node_id=node_id,
                preferred_id=f"genotype_{i}",
                node_label="genotype",
                properties={
                    "sys_gene_name": [sys_gene_name],
                    "perturbed_gene_name": [perturbed_gene_name],
                    "description": description,
                    "perturbation_type": [perturbation_type],
                    "serialized_data": json.dumps(experiment.genotype.model_dump()),
                },
            )
    
    def _get_environment_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            experiment_id = hashlib.md5(
                json.dumps(experiment.model_dump()).encode("utf-8")
            ).hexdigest()
            # CHECK should we be using something  
            environment_id = hashlib.md5(
                json.dumps(experiment.environment.model_dump()).encode("utf-8")
            ).hexdigest()
            # HACK if we don't do this for node id, then we get deduplicated genotypes that belong to different experiments.
            # Maybe we want this. Not sure.
            node_id = ("_").join([experiment_id, environment_id])
            # collect data
            
            # HACK casting not matching depth of class
            # TODO ask about difference 
            media = json.dumps(experiment.environment.media.model_dump())
            temperature = experiment.environment.temperature
            
            yield BioCypherNode(
                node_id=node_id,
                preferred_id=f"environment_{i}",
                node_label="environment",
                properties={
                    "temperature": temperature,
                    "media": media,
                    "serialized_data": json.dumps(experiment.genotype.model_dump()),
                },
            )
    
    def _get_media_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            media_id = hashlib.md5(
                json.dumps(experiment.environment.media.model_dump()).encode("utf-8")
            ).hexdigest()
            
            # TODO best to do media id as separate?
            node_id = media_id
            # collect data
            name = experiment.environment.media.name
            state = experiment.environment.media.state
            
            yield BioCypherNode(
                node_id=node_id,
                preferred_id=f"media_{node_id}",
                node_label="media",
                properties={
                    "name": name,
                    "state": state,
                    "serialized_data": json.dumps(experiment.environment.media.model_dump()),
                },
            )
        
    def _get_temperature_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            temperature_id = hashlib.md5(
                json.dumps(experiment.environment.temperature.model_dump()).encode("utf-8")
            ).hexdigest()
            node_id = temperature_id
            
            # TODO change this capitalization on celsius
            celsius = experiment.environment.temperature.Celsius
            yield BioCypherNode(
                node_id=node_id,
                preferred_id=f"temperature_{node_id}",
                node_label="temperature",
                properties={
                    "celsius": celsius,
                    "serialized_data": json.dumps(experiment.environment.temperature.model_dump()),
                },
            )
    
    def _get_phenotype_nodes(self) -> None:
        for i, experiment in enumerate(self.experiments):
            node_id = hashlib.md5(
                json.dumps(experiment.phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            # collect data
            graph_level = experiment.phenotype.graph_level
            label = experiment.phenotype.label
            label_error = experiment.phenotype.label_error
            fitness = experiment.phenotype.fitness
            fitness_std = experiment.phenotype.fitness_std
            
            yield BioCypherNode(
                node_id=node_id,
                preferred_id=f"phenotype_{node_id}",
                node_label="phenotype",
                properties={
                    "graph_level" :graph_level,
                    "label" :label,
                    "label_error" :label_error,
                    "fitness" :fitness,
                    "fitness_std" :fitness_std,
                    "serialized_data": json.dumps(experiment.phenotype.model_dump()),
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

    
    def _get_dataset_experiment_ref_edges(self):
        # concept level
        for exp_ref_index in self.dataset.reference_index:
            study_ref_id = hashlib.md5(
                json.dumps(exp_ref_index .model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=study_ref_id,
                target_id="CostanzoSmf2016",
                relationship_label = "experiment reference member of",
            )
             
    def _get_experiment_dataset_edges(self):         
        # concept level
        for i, experiment in enumerate(self.experiments):
            experiment_id = hashlib.md5(
                json.dumps(experiment.model_dump()).encode("utf-8")
            ).hexdigest()
            yield BioCypherEdge(
                source_id=experiment_id,
                target_id="CostanzoSmf2016",
                relationship_label = "experiment member of",
            )
       
    def _get_experiment_ref_experiment_edges(self):
        # instance level
        for exp_ref_index in self.dataset.reference_index:
            true_indices = [i for i, value in enumerate(exp_ref_index.index) if value]
            study_ref_id = hashlib.md5(
                json.dumps(exp_ref_index.model_dump()).encode("utf-8")
            ).hexdigest()
            for i, experiment in enumerate(self.experiments):
                if i in true_indices:
                    experiment_id = hashlib.md5(
                        json.dumps(experiment.model_dump()).encode("utf-8")
                    ).hexdigest()
                    yield BioCypherEdge(
                        source_id=study_ref_id,
                        target_id=experiment_id,
                        relationship_label = "experiment reference of",
                    )
        
                    
    def _set_types_and_fields():
        pass


if __name__ == "__main__":
    dataset = NeoSmfCostanzo2016Dataset()
    adapter = CostanzoSmfAdapter(dataset=dataset)
    [i for i in adapter.get_nodes()]
    [i for i in adapter.get_edges()]
    # print()
