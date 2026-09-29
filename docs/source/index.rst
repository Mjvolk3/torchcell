:github_url: https://github.com/Mjvolk3/torchcell

TorchCell
=========

TorchCell is a Python library for modeling the relationship between a cell's genotype,
its environment and its phenotype, starting with the yeast *Saccharomyces cerevisiae*.
It records each published measurement as a typed experiment, a pydantic
``genotype x environment -> phenotype`` object whose genotype is a list of gene
perturbations (deletions, alleles, gene additions, copy-number changes, CRISPR
activation and interference). Dataset loaders
convert the supplementary data of each source paper into these objects, a BioCypher
build loads them into a Neo4j knowledge graph, and the modeling code queries that
graph to train graph neural networks and transformers on the resulting cell graphs.

The project's aim is the experimental-data layer that a virtual yeast cell would be
trained on, so provenance is part of the data model: retrieved source files are recorded
with their retrieval command and sha256 (:mod:`torchcell.literature`), extracted
constants carry their source quote (:mod:`torchcell.verification`), built datasets are
checked at five record-level verification levels (L0 to L4), and a schema change flags
every built dataset it makes stale (:mod:`torchcell.provenance`).
The served knowledge graph can be browsed at
`torchcell-database.ncsa.illinois.edu <https://torchcell-database.ncsa.illinois.edu:7473/browser/>`_,
and an interactive map of the schema is at `/ontology/ <ontology/index.html>`_.

.. toctree::
   :maxdepth: 2
   :caption: Guide

   guide/index

.. toctree::
   :maxdepth: 2
   :caption: Database

   database/index

.. toctree::
   :maxdepth: 1
   :caption: API reference

   modules/adapters
   modules/data
   modules/database
   modules/datamodels
   modules/datamodules
   modules/dataset_readers
   modules/datasets
   modules/graph
   modules/knowledge_graphs
   modules/literature
   modules/loader
   modules/losses
   modules/metabolism
   modules/metrics
   modules/models
   modules/nn
   modules/ontology
   modules/paper
   modules/profiling
   modules/provenance
   modules/scheduler
   modules/sequence
   modules/sga
   modules/trainers
   modules/transforms
   modules/utils
   modules/verification
   modules/viz

Not documented
--------------

The API reference leaves out these parts of the ``torchcell`` package:

- ``torchcell.cell``, ``torchcell.dataset_preprocess`` and ``torchcell.go``: the first
  two contain only an ``__init__.py`` with a docstring, and the one module in
  ``torchcell.go`` is a script that reads ``data/go/go.obo`` when imported.
- ``torchcell.pypy_adapters`` and ``torchcell.profilers``: legacy code kept for
  reference.
- ``torchcell.scratch`` and ``torchcell.experiments``: working scripts, not library
  code.
- Top-level script modules such as ``torchcell/neo4j_fitness_lmdb.py``.
