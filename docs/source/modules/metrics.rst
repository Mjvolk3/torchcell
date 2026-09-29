torchcell.metrics
=================

.. module:: torchcell.metrics

.. currentmodule:: torchcell.metrics

TorchMetrics metrics that ignore missing labels. Each metric masks ``NaN`` targets before updating its state, so a batch with partially labeled multi-task targets can be scored under distributed training. ``nan_tolerant_metrics`` covers regression and correlation (RMSE, MAE, MSE, Pearson, Spearman, R2) and ``nan_tolerant_classification_metrics`` covers classification (accuracy, F1, AUROC, precision, recall).

.. contents:: Contents
    :local:

``nan_tolerant_classification_metrics``
---------------------------------------

NaN-tolerant TorchMetrics classification metrics for DDP training.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   nan_tolerant_classification_metrics.NaNTolerantMetricBase
   nan_tolerant_classification_metrics.NaNTolerantAccuracy
   nan_tolerant_classification_metrics.NaNTolerantF1Score
   nan_tolerant_classification_metrics.NaNTolerantAUROC
   nan_tolerant_classification_metrics.NaNTolerantPrecision
   nan_tolerant_classification_metrics.NaNTolerantRecall

``nan_tolerant_metrics``
------------------------

NaN-tolerant TorchMetrics for regression and correlation under missing labels.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   nan_tolerant_metrics.NaNTolerantRMSE
   nan_tolerant_metrics.NaNTolerantMAE
   nan_tolerant_metrics.NaNTolerantMetricBase
   nan_tolerant_metrics.NaNTolerantPearsonCorrCoef
   nan_tolerant_metrics.NaNTolerantSpearmanCorrCoef
   nan_tolerant_metrics.NaNTolerantMSE
   nan_tolerant_metrics.NaNTolerantR2Score
