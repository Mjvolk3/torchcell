torchcell.transforms
====================

.. module:: torchcell.transforms

.. currentmodule:: torchcell.transforms

PyG transforms applied to cell graphs. ``regression_to_classification`` and its COO variants normalize regression labels, bin them into classification targets, and invert the binning; ``hetero_to_dense`` and ``hetero_to_dense_mask`` convert the sparse adjacencies of a ``HeteroData`` graph into dense matrices or boolean masks.

.. contents:: Contents
    :local:

``coo_regression_to_classification``
------------------------------------

Transforms converting COO-format regression labels to classification targets.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   coo_regression_to_classification.COOLabelNormalizationTransform
   coo_regression_to_classification.BaseBinningStrategy
   coo_regression_to_classification.EqualWidthStrategy
   coo_regression_to_classification.EqualFrequencyStrategy
   coo_regression_to_classification.AutoBinStrategy
   coo_regression_to_classification.COOLabelBinningTransform
   coo_regression_to_classification.COOInverseCompose

``hetero_to_dense``
-------------------

PyG transform converting heterogeneous sparse adjacencies to dense matrices.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_to_dense.HeteroToDense

``hetero_to_dense_mask``
------------------------

Transform converting sparse hetero adjacencies to dense boolean masks.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   hetero_to_dense_mask.HeteroToDenseMask

``regression_to_classification``
--------------------------------

Transforms that bin regression labels into classification targets and invert them.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   regression_to_classification.LabelNormalizationTransform
   regression_to_classification.BaseBinningStrategy
   regression_to_classification.EqualWidthStrategy
   regression_to_classification.EqualFrequencyStrategy
   regression_to_classification.AutoBinStrategy
   regression_to_classification.LabelBinningTransform
   regression_to_classification.InverseCompose

``regression_to_classification_coo``
------------------------------------

COO-format transforms to normalize, bin, and invert regression labels.

Classes
~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: ../generated
   :template: autosummary/class.rst

   regression_to_classification_coo.COOLabelNormalizationTransform
   regression_to_classification_coo.BaseBinningStrategy
   regression_to_classification_coo.EqualWidthStrategy
   regression_to_classification_coo.EqualFrequencyStrategy
   regression_to_classification_coo.AutoBinStrategy
   regression_to_classification_coo.COOLabelBinningTransform
   regression_to_classification_coo.COOInverseCompose
