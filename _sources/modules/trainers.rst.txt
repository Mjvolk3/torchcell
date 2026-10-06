torchcell.trainers
==================

.. module:: torchcell.trainers

.. currentmodule:: torchcell.trainers

PyTorch Lightning training tasks. The package exports the regression task (:class:`~torchcell.trainers.neo_regression.RegressionTask`), a simple linear baseline, and two DCell tasks. The ``fit_int_*`` and ``int_*`` modules hold the tasks used by individual experiments and are imported by module path from the experiment scripts. Every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

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

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   int_hetero_cell.scheduler_type
   int_hetero_cell.normalize_accumulation_schedule
   int_hetero_cell.accumulation_steps

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
