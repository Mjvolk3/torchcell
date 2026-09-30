torchcell.viz
=============

.. module:: torchcell.viz

.. currentmodule:: torchcell.viz

Plotting helpers used during training and analysis: predicted-versus-measured fitness and genetic-interaction plots, dataset split visualizations, graph-regularization and edge-recovery plots, transformer diagnostics, oversmoothing and oversquashing measures, and regression diagnostics logged to Weights & Biases.

.. contents:: Contents
    :local:

``datamodules``
---------------

Visualization helpers for dataset index splits.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   datamodules.plot_dataset_index_split

``fitness``
-----------

Box-plot visualization of predicted vs measured growth/fitness.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   fitness.box_plot
   fitness.generate_simulated_data
   fitness.generate_simulated_data_with_nan

``genetic_interaction_score``
-----------------------------

Visualizations of genetic interaction score predictions versus truth.

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   genetic_interaction_score.box_plot
   genetic_interaction_score.generate_simulated_data
   genetic_interaction_score.generate_simulated_data_with_nan

``graph_recovery``
------------------

Plotting utilities for graph regularization and edge-recovery metrics.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   graph_recovery.GraphRecoveryVisualization

``transformer_diagnostics``
---------------------------

Visualization helpers for transformer-specific diagnostic metrics.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   transformer_diagnostics.TransformerDiagnostics

``visual_graph_degen``
----------------------

Graph degeneration diagnostics: oversmoothing and oversquashing metrics.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   visual_graph_degen.VisGraphDegen

``visual_regression``
---------------------

Visualization helpers for logging regression diagnostics to Weights & Biases.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   visual_regression.Visualization
