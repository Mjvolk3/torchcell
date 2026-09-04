# torchcell/datasets/kinetics/boost_km.py
# [[torchcell.datasets.kinetics.boost_km]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/boost_km.py
# Test file: tests/torchcell/datasets/kinetics/test_boost_km.py

r"""Boost_KM: a DMPNN substrate fingerprint plus a UniRep enzyme vector into XGBoost.

Kroll, Engqvist, Heckmann and Lercher (2021), *PLOS Biology*, "Deep learning allows
genome-scale prediction of Michaelis constants from structural features". A 52-dim
task-specific fingerprint, read out of a directed message-passing graph network trained on
the :math:`K_M` regression itself, is concatenated with the 1,900-dim average hidden state
of the UniRep mLSTM and fed to a gradient-boosting regressor. The regressor predicts
:math:`\log_{10} K_M` with :math:`K_M` in **mM**, which is the unit BRENDA reports and the
unit the authors log-transformed without rescaling. This module raises to the power of ten
at the edge, so what leaves here is mM like every other predictor's :math:`K_M`.

WHY THE FITTED REGRESSOR IS THE ONE IN THE MIRROR
--------------------------------------------------
It is easy to look at this repository and conclude no fitted model ships: the largest
block of weight files is the UniRep featurizer, which is not a :math:`K_M` model at all.
The fitted model is ``datasets/model_weights/xgboost_model.dat`` -- 1,381 boosting rounds
over 1,952 features, exactly :math:`52 + 1900`. Scored on the authors' own held-out split
it returns MSE 0.6532 and :math:`R^2` 0.5274 on the :math:`\log_{10}` scale, which is the
paper's headline test-set result. Nothing here is refit.

The sibling file ``xgboost_model_full.dat`` takes the same 1,952 features but scores
**worse** on that split (MSE 0.9153, :math:`R^2` 0.3377). Whatever it was trained on, it
was not trained on data containing the test split, and it is not the paper's model, so it
is not used. Reproduce both numbers with ``run_boost_km_pinned.py --self-test``.

WHY THIS CLASS SHELLS OUT
--------------------------
Two stages want incompatible TensorFlow modes in one process. UniRep is TensorFlow 1 graph
code -- ``tf.placeholder``, ``tf.get_variable``, ``tf.nn.rnn_cell`` -- and needs
``disable_v2_behavior``; the fingerprint network is a TensorFlow 2 functional model whose
layer list only comes out right with v2 behavior left on. Neither is a bug to patch
around, and rewriting either would mean the numbers are no longer the authors'. So the
predictor lives in an environment pinned to TensorFlow 2.15, driven as a subprocess that
splits the two stages into their own processes, with parquet as the interface. This is the
same shape as :mod:`torchcell.datasets.kinetics.unikp`.
"""

from __future__ import annotations

import os.path as osp
import subprocess
import tempfile
from typing import Any

import pandas as pd

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest, sha256_file

PINNED_PYTHON = "/scratch/projects/torchcell-scratch/envs/boost_km/bin/python"
# The fitted regressor, not the featurizer. See the module docstring.
REGRESSOR = "datasets/model_weights/xgboost_model.dat"


class BoostKmDataset(BaseKineticsDataset):
    """Predicted :math:`K_M` in mM for every (enzyme, substrate) pair."""

    EMITS = (KineticParameter.K_M,)
    MIRROR_NAME = "boost_km"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the model loads in the subprocess."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(f"boost_km mirror failed verification: {broken[:3]}")
        # A value is attributable to the gradient-boosting regressor that emitted it. The
        # DMPNN checkpoint and the UniRep weights shape the features and are pinned in the
        # same manifest, so the whole chain is covered by verifying it.
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.rel_path == REGRESSOR
        )
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned Boost_KM process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_boost_km_pinned.py",
        )
        with tempfile.TemporaryDirectory() as work:
            inputs = osp.join(work, "inputs.parquet")
            output = osp.join(work, "predictions.parquet")
            rows.to_parquet(inputs, index=False)
            command = [
                PINNED_PYTHON,
                runner,
                "--inputs",
                inputs,
                "--output",
                output,
                # UniRep is an mLSTM run one residue at a time and the fingerprint network
                # is a 70x70x42 message pass, so both are cached by content. Without the
                # cache a block-wise build re-embeds the same enzyme once per block.
                "--cache-dir",
                osp.join(self.root, self.MIRROR_NAME, "features"),
            ]
            # No capture: the subprocess reports its own progress, and a failure has to
            # surface as a failure rather than as an empty column.
            subprocess.run(command, check=True)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["K_M", "failure"]]


# The regressor Wu et al. actually ran, vendored inside their release rather than mirrored
# separately. Their Methods cite the UniRep repository above; their notebook loads this.
ESM1B_REGRESSOR = (
    "/scratch/projects/torchcell-scratch/data/enzyme_kinetics/yeast_metatwin/repo/"
    "Code/kcatkm_prediction/KM_prediction/data/saved_models/xgboost/"
    "xgboost_model_new_KM_esm1b.dat"
)


class BoostKmEsm1bDataset(BoostKmDataset):
    """Boost_KM as Wu et al. ran it: the ESM-1b enzyme representation, not UniRep.

    Kroll's ``KM_prediction_function`` swaps the paper's UniRep 1900 vector for a 1,280-dim
    mean-pooled ESM-1b embedding and refits the regressor, giving 1,332 features instead of
    1,952. The substrate half is unchanged, and the DMPNN checkpoint is byte-identical
    between the two repositories, so only the enzyme encoder and the regressor differ.

    This is the variant registered as ``boost_km`` because reproducing Fig. 3f means
    running the code the figure came from. The paper-faithful UniRep variant remains
    available as :class:`BoostKmDataset`.
    """

    def initialize_model(self) -> Any:
        """Verify the shared mirror and pin the ESM-1b regressor as the attributed weight."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(f"boost_km mirror failed verification: {broken[:3]}")
        # The substrate checkpoint is covered by the mirror; the value itself is emitted by
        # the vendored ESM-1b regressor, so that file is what a K_M is attributable to.
        self._weights_sha256 = sha256_file(ESM1B_REGRESSOR)
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned ESM-1b Boost_KM process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_boost_km_esm1b_pinned.py",
        )
        with tempfile.TemporaryDirectory() as work:
            inputs = osp.join(work, "inputs.parquet")
            output = osp.join(work, "predictions.parquet")
            rows.to_parquet(inputs, index=False)
            command = [
                PINNED_PYTHON,
                runner,
                "--inputs",
                inputs,
                "--output",
                output,
                "--cache-dir",
                osp.join(self.root, self.MIRROR_NAME, "features"),
            ]
            subprocess.run(command, check=True)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["K_M", "failure"]]
