# torchcell/datasets/kinetics/catpred.py
# [[torchcell.datasets.kinetics.catpred]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/catpred.py
# Test file: tests/torchcell/datasets/kinetics/test_catpred.py

r"""CatPred: a chemprop fork with an ESM-2 enzyme branch and a mean-variance head.

Boorla and Maranas (2025), *Nature Communications* 16:2072,
``boorlaCatPredComprehensiveFramework2025``. The substrate is a directed message-passing
network over its 2D graph; the enzyme is a learned residue embedding carrying rotary
positional information through multi-head self-attention, concatenated with ESM-2 650M
features and reduced by attentive pooling. No EC number is an input, and the structure
branch belongs to the released :math:`K_i` model, which is not mirrored.

WHY THIS ONE IS DIFFERENT FROM THE OTHER SEVEN
------------------------------------------------
Every other predictor in this package returns a point value. CatPred's head returns a mean
AND a variance, trained under a negative log-likelihood loss, and the release is an
ensemble of ten differently seeded copies. One pass therefore yields the prediction plus
two uncertainties, both standard deviations of a :math:`\log_{10}` quantity and so both in
decades:

* ``<parameter>_log10_sd_aleatoric`` -- the noise the training labels carry, which for
  BRENDA and SABIO-RK is real and large;
* ``<parameter>_log10_sd_epistemic`` -- how far the ensemble members disagree, which is
  how far the input sits from what the model was trained on;
* ``<parameter>_log10_sd_total`` -- the two in quadrature, the quantity the authors report.

Those three columns ride along in the table rather than being recomputed downstream,
because they are properties of the prediction and are lost the moment the value is
median-ed with six other predictors.

WHY THIS CLASS SHELLS OUT INSTEAD OF PREDICTING IN PROCESS
-----------------------------------------------------------
The same arrangement :class:`~torchcell.datasets.kinetics.eitlem.EitlemDataset` uses, for
the same reason. CatPred needs ``typed-argument-parser``, ``descriptastorus``,
``pandas-flavor``, ``tensorboardX``, ``ipdb`` and a pinned
``rotary_embedding_torch==0.6.5``, none of which torchcell depends on, and the pin is
load-bearing: a later release changes the cached-frequency shape and the checkpoints
refuse to run. Isolating that in a subprocess keeps the dependency set out of the project
environment, and parquet is the interface.

ONE DEVIATION FROM THE MODEL'S TRAINING CONVENTION, STATED
------------------------------------------------------------
CatPred's :math:`k_{cat}` model was trained on the concatenated SMILES of ALL reactants,
not on one substrate. The shared pair table carries one substrate per row, which is what
the other seven predictors were shown, so that is what CatPred is shown: the comparison
stays like-for-like and the deviation is recorded rather than hidden. :math:`K_M` has no
such gap, since the authors use the single relevant substrate for it.
"""

from __future__ import annotations

import os.path as osp
import subprocess
import tempfile
from typing import Any

import pandas as pd

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest

PINNED_PYTHON = "/home/michaelvolk/miniconda3/envs/torchcell/bin/python"

#: The ensemble member whose sha256 anchors a value's provenance. The mirror manifest
#: pins the other nineteen checkpoints; one hash names the build.
KCAT_CHECKPOINT = "pretrained/production/kcat/fold_0/model_0/model.pt"

#: The uncertainty columns the runner emits per parameter, carried into the table.
UNCERTAINTY_SUFFIXES = ("log10_sd_total", "log10_sd_aleatoric", "log10_sd_epistemic")


class CatPredDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` and :math:`K_M` with a per-prediction variance."""

    EMITS = (KineticParameter.k_cat, KineticParameter.K_M)
    MIRROR_NAME = "catpred"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the model loads in the subprocess."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(f"catpred mirror failed verification: {broken[:3]}")
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.rel_path == KCAT_CHECKPOINT
        )
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned CatPred process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_catpred_pinned.py",
        )
        with tempfile.TemporaryDirectory() as work:
            inputs = osp.join(work, "inputs.parquet")
            output = osp.join(work, "predictions.parquet")
            rows.to_parquet(inputs, index=False)
            # No capture: the subprocess reports its own progress, and a failure has to
            # surface as a failure rather than as an empty column.
            subprocess.run(
                [
                    PINNED_PYTHON,
                    runner,
                    "--inputs",
                    inputs,
                    "--output",
                    output,
                    "--device",
                    self.device,
                ],
                check=True,
            )
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        wanted = [p.value for p in self.EMITS]
        wanted += [f"{p}_{s}" for p in wanted for s in UNCERTAINTY_SUFFIXES]
        return predictions[[*wanted, "failure"]]
