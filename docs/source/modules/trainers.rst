torchcell.trainers
==================

.. module:: torchcell.trainers

.. currentmodule:: torchcell.trainers

PyTorch Lightning training tasks. The package exports the regression task (:class:`~torchcell.trainers.neo_regression.RegressionTask`), a simple linear baseline, and two DCell tasks. The ``fit_int_*`` and ``int_*`` modules hold the tasks used by individual experiments and are imported by module path from the experiment scripts. Every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

``cell``
--------

Minimal LightningModule scaffold for cell-data regression training.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   cell.SimpleModel

``dcell_regression``
--------------------

Lightning training task for DCell graph-based fitness regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dcell_regression.DCellRegressionTask

``dcell_regression_slim``
-------------------------

Slim Lightning trainer for DCell regression with subsystem and root metrics.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dcell_regression_slim.DCellRegressionSlimTask

``fit_int_cell_diffpool_dense_regression``
------------------------------------------

Lightning regression task for the DiffPool dense cell-integration model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_cell_diffpool_dense_regression.RegressionTask

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   fit_int_cell_diffpool_dense_regression.log_error_information

``fit_int_cell_gin_diffpool_dense_binary``
------------------------------------------

Lightning trainer for the dense GIN+DiffPool binary fitness/interaction model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_cell_gin_diffpool_dense_binary.RegressionTask

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   fit_int_cell_gin_diffpool_dense_binary.log_error_information

``fit_int_cell_sagpool_regression``
-----------------------------------

Lightning regression trainer for the SAGPool cell-graph interaction model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_cell_sagpool_regression.RegressionTask

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   fit_int_cell_sagpool_regression.log_error_information

``fit_int_deep_set_regression``
-------------------------------

Lightning trainer and NaN-tolerant losses/metrics for deep-set GI regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_deep_set_regression.NaNTolerantCorrelation
   fit_int_deep_set_regression.NaNTolerantPearsonCorrCoef
   fit_int_deep_set_regression.NaNTolerantSpearmanCorrCoef
   fit_int_deep_set_regression.MultiDimNaNTolerantMSELoss
   fit_int_deep_set_regression.CombinedMSELoss
   fit_int_deep_set_regression.RegressionTask

``fit_int_gat_diffpool_regression``
-----------------------------------

Lightning regression trainer and NaN-tolerant losses for GAT DiffPool models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_gat_diffpool_regression.NaNTolerantCorrelation
   fit_int_gat_diffpool_regression.NaNTolerantPearsonCorrCoef
   fit_int_gat_diffpool_regression.NaNTolerantSpearmanCorrCoef
   fit_int_gat_diffpool_regression.MultiDimNaNTolerantL1Loss
   fit_int_gat_diffpool_regression.MultiDimNaNTolerantMSELoss
   fit_int_gat_diffpool_regression.CombinedRegressionLoss
   fit_int_gat_diffpool_regression.RegressionTask

``fit_int_hetero_cell``
-----------------------

Lightning regression trainer for heterogeneous cell-graph interaction models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_hetero_cell.RegressionTask

``fit_int_hetero_cell_multiset_decomposition``
----------------------------------------------

Lightning trainer for hetero-cell gene-interaction regression with multiset decomposition.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_hetero_cell_multiset_decomposition.RegressionTask

``fit_int_hetero_gnn_pool_binary_classification``
-------------------------------------------------

Lightning task training a hetero GNN pool model for binary fitness classification.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_hetero_gnn_pool_binary_classification.ClassificationTask

``fit_int_hetero_gnn_pool_reg_categorical_entropy``
---------------------------------------------------

Lightning trainer for binned categorical fitness/interaction regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_hetero_gnn_pool_reg_categorical_entropy.RegCategoricalEntropyTask

``fit_int_hetero_gnn_pool_regression``
--------------------------------------

Lightning task for fitness and gene-interaction regression on hetero GNN pools.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_hetero_gnn_pool_regression.RegressionTask

``fit_int_isomorphic_cell_attentional``
---------------------------------------

Lightning regression trainer for the isomorphic attentional cell model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   fit_int_isomorphic_cell_attentional.RegressionTask

``int_dango``
-------------

Lightning training task for the DANGO genetic-interaction regression model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   int_dango.RegressionTask

``int_dcell``
-------------

Lightning training module for DCell gene-interaction regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   int_dcell.RegressionTask

``int_hetero_cell``
-------------------

Lightning regression tasks for heterogeneous-cell gene interaction models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   int_hetero_cell.RegressionTask
   int_hetero_cell.DiffusionRegressionTask

``int_hetero_cell_nsa``
-----------------------

Lightning training tasks for hetero cell gene-interaction regression models.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   int_hetero_cell_nsa.RegressionTask
   int_hetero_cell_nsa.DiffusionRegressionTask

``int_transformer_cell``
------------------------

Lightning training module for the transformer cell gene-interaction model.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   int_transformer_cell.RegressionTask

``neo_regression``
------------------

Lightning regression trainer with MSE/ListMLE losses and correlation metrics.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   neo_regression.ListMLEMetric
   neo_regression.MSEListMLELoss
   neo_regression.RegressionTask

``simple_linear_regression``
----------------------------

Lightning task for simple linear regression on graph perturbation data.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   simple_linear_regression.SimpleLinearRegressionTask

Not documented
--------------

Submodules left out of this page:

- ``fit_int_gat_diffpool_inception_regression`` (import fails: ImportError)
- ``graph_convolution_regression`` (import fails: ImportError)
- ``regression`` (import fails: ImportError)
- ``regression_deep_set_transformer`` (import fails: ImportError)
- ``utils`` (import fails: PydanticImportError)
