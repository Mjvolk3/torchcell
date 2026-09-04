# torchcell/datasets/kinetics/eitlem.py
# [[torchcell.datasets.kinetics.eitlem]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/eitlem.py
# Test file: tests/torchcell/datasets/kinetics/test_eitlem.py

r"""EITLEM-Kinetics: ESM-1v residues attended by a MACCS fingerprint, emitting both parameters.

Shen et al. (2024), *Chem Catalysis*. A per-residue ESM-1v representation of the enzyme is
attended over by the substrate's 167-bit MACCS fingerprint through ten attention blocks,
and a two-branch aggregator maps the pair to :math:`\log_{10}` of one parameter. What
distinguishes EITLEM from UniKP is not the architecture but the training schedule: the
:math:`k_{cat}` and :math:`K_M` heads are trained alternately over eight iterations, each
warm-started from a joint model, so the two sparse tables inform one another. That makes it
the natural second opinion on :math:`K_M`, where the measured data is thinnest.

WHY THIS CLASS SHELLS OUT INSTEAD OF PREDICTING IN PROCESS
-----------------------------------------------------------
The same reason as :class:`~torchcell.datasets.kinetics.unikp.UniKPDataset`, arrived at
from a different direction. UniKP's regressors cannot be unpickled by a modern
scikit-learn; EITLEM instead needs ``fair-esm``, which torchcell does not depend on, and it
holds a 650M-parameter encoder plus two predictor checkpoints on the GPU. Isolating that in
a subprocess keeps both the dependency and the allocation out of the calling process, and
parquet is the interface. One environment per released predictor is the shape this has to
take, so the pattern is reused rather than special-cased.

TWO WAYS THIS DIFFERS FROM UniKP ON THE SAME INPUT TABLE
---------------------------------------------------------
Both are consequences of the model's own training set, not choices made here.

* **Multi-component SMILES are kept.** UniKP's authors filtered a ``.`` out of their
  training data, so that pipeline reproduces the filter and 105 pairs go unpredicted.
  EITLEM's authors did not, and 2.4% of its :math:`k_{cat}` table and 4.3% of its
  :math:`K_M` table contain one, so all 7,456 pairs are predicted here.
* **Sequences are truncated to 1,022 residues.** ESM-1v's learned positional embedding
  admits 1,024 positions, and the authors' training FASTA tops out at exactly 1,022, so no
  released weight ever saw a longer protein. The 70 GEM enzymes above that length are
  embedded from their first 1,022 residues; ``sequence_length`` in the output table is the
  full length, so those rows stay identifiable.
"""

from __future__ import annotations

import os
import os.path as osp
import subprocess
import tempfile
from typing import Any

import pandas as pd

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest

PINNED_PYTHON = "/home/michaelvolk/miniconda3/envs/torchcell/bin/python"
# ``fair-esm`` is installed to its own prefix rather than into the project environment: it
# is a dependency of this one released model, not of torchcell.
PINNED_SITE_PACKAGES = "/scratch/projects/torchcell-scratch/envs/eitlem/site-packages"

# The final iteration of the eight-round transfer schedule, named by the mirror's README.
KCAT_CHECKPOINT = "Weights/KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787"


class EitlemDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` and :math:`K_M` for every (enzyme, substrate) pair."""

    EMITS = (KineticParameter.k_cat, KineticParameter.K_M)
    MIRROR_NAME = "eitlem"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the model loads in the subprocess."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(f"eitlem mirror failed verification: {broken[:3]}")
        # The k_cat checkpoint is the artifact a k_cat value is attributable to. The K_M
        # checkpoint is pinned by the same manifest; one hash names the build, and the
        # manifest resolves the rest.
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.rel_path == KCAT_CHECKPOINT
        )
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned EITLEM process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_eitlem_pinned.py",
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
                "--device",
                self.device,
            ]
            environment = {
                **os.environ,
                "PYTHONPATH": PINNED_SITE_PACKAGES,
            }
            # No capture: the subprocess reports its own progress, and a failure has to
            # surface as a failure rather than as an empty column.
            subprocess.run(command, check=True, env=environment)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["k_cat", "K_M", "failure"]]
