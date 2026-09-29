torchcell.datamodules
=====================

.. module:: torchcell.datamodules

.. currentmodule:: torchcell.datamodules

PyTorch Lightning data modules for training on cell datasets. :class:`~torchcell.datamodules.CellDataModule` builds train, validation and test splits of a cell dataset and caches the split indices, and the ``perturbation_subset`` module's data module draws a size-limited subset of those splits. The split-index records (:class:`~torchcell.datamodules.DataModuleIndex` and related models) are pydantic objects, cached as JSON in the data module's cache directory.

.. contents:: Contents
    :local:

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   CellDataModule
   IndexSplit
   DatasetSplit
   DataModuleIndex
   DataModuleIndexDetails
