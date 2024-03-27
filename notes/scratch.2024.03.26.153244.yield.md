---
id: vp82b5pcfgwlxf58g3btd1m
title: Yield
desc: ''
updated: 1711486959332
created: 1711485167662
---
29 in dmf

```python
def get_nodes(self):
  log.info("Running: self._get_experiment_reference_nodes()")
  yield from self._get_experiment_reference_nodes()
  log.info("Running: self._get_genome_nodes()")
  yield from self._get_genome_nodes()
  log.info("Running: self.get_data_by_type(self._experiment_node)")
  yield from self.get_data_by_type(self._experiment_node)
  log.info("Running: self.get_data_by_type(self._genotype_node)")
  yield from self.get_data_by_type(self._genotype_node)
  log.info("Running: self.get_data_by_type(self._perturbation_node)")
  yield from self.get_data_by_type(self._perturbation_node)
  log.info("Running: self.get_data_by_type(self._environment_node)")
  yield from self.get_data_by_type(self._environment_node)
  log.info("Running: self._get_reference_environment_nodes()")
  yield from self._get_reference_environment_nodes()
  log.info("Running: self.get_data_by_type(self._media_node)")
  yield from self.get_data_by_type(self._media_node)
  log.info("Running: self._get_reference_media_nodes()")
  yield from self._get_reference_media_nodes()
  log.info("Running: self.get_data_by_type(self._temperature_node)")
  yield from self.get_data_by_type(self._temperature_node)
  log.info("Running: self._get_reference_temperature_nodes()")
  yield from self._get_reference_temperature_nodes()
  log.info("Running: self.get_data_by_type(self._phenotype_node)")
  yield from self.get_data_by_type(self._phenotype_node)
  log.info("Running: self._get_reference_phenotype_nodes()")
  yield from self._get_reference_phenotype_nodes()
  log.info("Running: self.get_dataset_nodes()")
  yield from self.get_dataset_nodes()
  log.info("Finished: get_nodes")

def get_edges(self):
  log.info("Running: self.get_reference_dataset_edges()")
  yield from self.get_reference_dataset_edges()
  log.info("Running: self.get_data_by_type(self._experiment_dataset_edge)")
  yield from self.get_data_by_type(self._experiment_dataset_edge)
  log.info("Running: self._get_reference_experiment_edges()")
  yield from self._get_reference_experiment_edges()
  log.info("Running: self.get_data_by_type(self._genotype_experiment_edge)")
  yield from self.get_data_by_type(self._genotype_experiment_edge)
  log.info("Running: self.get_data_by_type(self._perturbation_genotype_edges)")
  yield from self.get_data_by_type(self._perturbation_genotype_edges)
  log.info("Running: self.get_data_by_type(self._environment_experiment_edges)")
  yield from self.get_data_by_type(self._environment_experiment_edges)
  log.info("Running: self._get_environment_experiment_ref_edges()")
  yield from self._get_environment_experiment_ref_edges()
  log.info("Running: self.get_data_by_type(self._phenotype_experiment_edges)")
  yield from self.get_data_by_type(self._phenotype_experiment_edges)
  log.info("Running: self.get_data_by_type(self._media_environment_edge)")
  yield from self.get_data_by_type(self._media_environment_edge)
  log.info("Running: self.get_data_by_type(self._temperature_environment_edge)")
  yield from self.get_data_by_type(self._temperature_environment_edge)
  log.info("Running: self._get_genome_edges()")
  yield from self._get_genome_edges()
  log.info("Finished: get_edges")
```
