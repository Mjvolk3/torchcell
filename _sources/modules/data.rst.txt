torchcell.data
==============

.. module:: torchcell.data

.. currentmodule:: torchcell.data

The data layer between the knowledge graph and model training. :class:`~torchcell.data.ExperimentDataset` is the abstract LMDB-backed base for experiment datasets, :class:`~torchcell.data.Neo4jCellDataset` queries Neo4j, caches the raw records in LMDB and turns each experiment into a perturbed cell graph, and the graph processors (for example :class:`~torchcell.data.SubgraphRepresentation` and :class:`~torchcell.data.LazySubgraphRepresentation`) define how a genotype perturbation is applied to that graph. Deduplicators and aggregators merge repeated measurements of the same genotype, and the reference-index helpers group experiments by their shared reference state.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   ExperimentReferenceIndex
   ReferenceIndex
   Deduplicator
   MeanExperimentDeduplicator
   Aggregator
   DeletionKeyedGenotypeAggregator
   GenotypeAggregator
   ExperimentDataset
   Neo4jCellDataset
   RawSha256MismatchError
   ManifestPinMismatchError
   SubgraphRepresentation
   LazySubgraphRepresentation
   IncidenceSubgraphRepresentation
   Perturbation
   DCellGraphProcessor
   Unperturbed

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   compute_sha256_hash
   compute_experiment_reference_index_sequential
   compute_experiment_reference_index_parallel
   post_process
   check_manifest_pin
   file_sha256
   verify_sha256
   verify_raw_files
   copy_verified
   write_verified
   link_verified
