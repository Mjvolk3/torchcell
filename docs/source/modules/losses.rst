torchcell.losses
================

.. module:: torchcell.losses

.. currentmodule:: torchcell.losses

Loss functions. The package exports :class:`~torchcell.losses.list_mle.ListMLELoss`, a listwise ranking loss, and :class:`~torchcell.losses.dcell.DCellLoss`, the root plus subsystem loss of the DCell model. The remaining modules (``multi_dim_nan_tolerant``, ``distributional``, ``mle_wasserstein``, ``point_dist_graph_reg`` and others) are imported by module path from the experiment training scripts and are not re-exported. Every class below is listed under the module that defines it.

.. contents:: Contents
    :local:

``dango``
---------

DANGO loss with epoch-scheduled weighting of reconstruction vs interaction loss.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dango.DangoLossSched
   dango.PreThenPost
   dango.LinearUntilUniform
   dango.LinearUntilFlipped
   dango.DangoLoss

``dcell``
---------

Loss for the DCell model: root MSE plus optional subsystem auxiliary losses.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   dcell.DCellLoss

``diffusion_loss``
------------------

Strict diffusion loss with optional explicit x0 supervision.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   diffusion_loss.DiffusionLoss

``distributional``
------------------

Distributional decoder heads + proper-scoring-rule losses for the multitask CGT.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   distributional.DistHead

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   distributional.gaussian_crps
   distributional.laplace_crps
   distributional.gaussian_nll
   distributional.energy_score
   distributional.pinball
   distributional.masked_mean
   distributional.pit_values
   distributional.coverage
   distributional.pit_ks
   distributional.dist_param_dim
   distributional.make_dist_head

``isomorphic_cell_loss``
------------------------

Isomorphic cell losses combining MSE, distribution, and SupCR terms.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   isomorphic_cell_loss.ICLoss
   isomorphic_cell_loss.ICLossStd

``list_mle``
------------

ListMLE listwise ranking loss for ordered prediction tasks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   list_mle.ListMLELoss

``logcosh``
-----------

Log-cosh regression loss module.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   logcosh.LogCoshLoss

``mle_dist_supcr``
------------------

Combined MLE, distribution, and supervised contrastive loss with buffering.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   mle_dist_supcr.AdaptiveWeighting
   mle_dist_supcr.TemperatureScheduler
   mle_dist_supcr.BufferedWeightedDistLoss
   mle_dist_supcr.BufferedWeightedSupCRCell
   mle_dist_supcr.MleDistSupCR

``mle_wasserstein``
-------------------

Composite MSE + Wasserstein + SupCR loss with buffers and DDP gathering.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   mle_wasserstein.AdaptiveWeighting
   mle_wasserstein.TemperatureScheduler
   mle_wasserstein.WeightedWassersteinLoss
   mle_wasserstein.BufferedWeightedWassersteinLoss
   mle_wasserstein.BufferedWeightedSupCRCell
   mle_wasserstein.MleWassSupCR

``multi_dim_nan_tolerant``
--------------------------

NaN-tolerant multi-dimensional loss functions for cell-state regression.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   multi_dim_nan_tolerant.SupCR
   multi_dim_nan_tolerant.WeightedSupCRCell
   multi_dim_nan_tolerant.NaNTolerantL1Loss
   multi_dim_nan_tolerant.NaNTolerantHuberLoss
   multi_dim_nan_tolerant.NaNTolerantLogCoshLoss
   multi_dim_nan_tolerant.WeightedMSELoss
   multi_dim_nan_tolerant.NaNTolerantMSELoss
   multi_dim_nan_tolerant.NaNTolerantQuantileLoss
   multi_dim_nan_tolerant.FastSoftSort
   multi_dim_nan_tolerant.WeightedDistLoss
   multi_dim_nan_tolerant.CombinedRegressionLoss
   multi_dim_nan_tolerant.MultiDimNaNTolerantCELoss
   multi_dim_nan_tolerant.CombinedCELoss
   multi_dim_nan_tolerant.MonotonicParameter
   multi_dim_nan_tolerant.MultiDimNaNTolerantOrdinalCELoss
   multi_dim_nan_tolerant.CombinedOrdinalCELoss
   multi_dim_nan_tolerant.CategoricalEntropyRegLoss
   multi_dim_nan_tolerant.MseCategoricalEntropyRegLoss

Functions
~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   multi_dim_nan_tolerant.isotonic_l2_pav
   multi_dim_nan_tolerant.fast_soft_sort

``point_dist_graph_reg``
------------------------

Composite point + distribution + graph-regularization loss for the cell transformer.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   point_dist_graph_reg.PointDistGraphReg

Not documented
--------------

Submodules left out of this page:

- ``SupCr`` (import fails: ModuleNotFoundError)
- ``dcell_DEPRECATED`` (scratch, demo or deprecated)
