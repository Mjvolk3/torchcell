torchcell.tasks
===============

.. module:: torchcell.tasks

.. currentmodule:: torchcell.tasks

Standardized benchmark tasks. A task pins one dataset, the genome its records are written against, one label and a set of pinned train, validation and test splits. :func:`~torchcell.tasks.ecoli.load_task` is the one import a benchmark script needs; the test part of every split is locked until :meth:`~torchcell.tasks.task.Split.unlock_test` is called. The tasks, their splits, example scripts and baseline results are on the benchmark website.

.. contents:: Contents
    :local:

Task and split objects
----------------------

.. currentmodule:: torchcell.tasks.task

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   TaskSpec
   BenchmarkTask
   Split
   TestSplitLocked
   TestSplitWarning

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   regression_metrics

E. coli tasks
-------------

.. currentmodule:: torchcell.tasks.ecoli

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   load_task
   build_task
   open_dataset
   match_records
   read_labels
   write_splits
   read_splits

Split functions
---------------

.. currentmodule:: torchcell.tasks.splits

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   assign_groups
   random_split
   block_groups
   pair_parts
   pair_novelty
   part_sizes
