torchcell.loader
================

.. module:: torchcell.loader

.. currentmodule:: torchcell.loader

Batch loaders for experiment datasets. :class:`~torchcell.loader.cpu_experiment_loader.CpuExperimentLoaderMultiprocessing` prefetches batches in worker processes on the CPU side; the ``dense_padding_data_loader`` module pads batches to a fixed node count for models that need dense inputs.

.. contents:: Contents
    :local:

``cpu_experiment_loader``
-------------------------

CPU-side prefetching batch loaders backed by worker threads/processes.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cpu_experiment_loader.CpuExperimentLoaderMultiprocessing

``dense_padding_data_loader``
-----------------------------

Data loader and collater that pad batches to a dense, fixed node count.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dense_padding_data_loader.DensePaddingCollater
   dense_padding_data_loader.DensePaddingDataLoader

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   dense_padding_data_loader.dense_padded_collate
   dense_padding_data_loader.dense_padded_from_data_list
