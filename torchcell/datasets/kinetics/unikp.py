# torchcell/datasets/kinetics/unikp.py
# [[torchcell.datasets.kinetics.unikp]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/unikp.py
# Test file: tests/torchcell/datasets/kinetics/test_unikp.py

r"""UniKP: ProtT5 + SMILES-transformer features into ExtraTrees, emitting both parameters.

Yu et al. (2023), *Nature Communications*. A 1,024-dim protein representation from
ProtT5-XL-UniRef50 is concatenated with a 1,024-dim substrate representation from a SMILES
transformer, and a released ExtraTrees regressor maps the 2,048-dim vector to
:math:`\log_{10}` of the parameter. Separate regressors were released for :math:`k_{cat}`
and :math:`K_M`, which is what makes UniKP the model that closes the :math:`K_M` gap.

WHY THIS CLASS SHELLS OUT INSTEAD OF PREDICTING IN PROCESS
-----------------------------------------------------------
The released regressors were pickled before scikit-learn 1.3 added ``missing_go_to_left``
to the decision-tree node dtype, and scikit-learn 1.7 refuses to load them. Refusing is
correct: reinterpreting a tree's node array under a changed dtype would yield numbers that
are not UniKP's predictions. So inference runs in an environment pinned to scikit-learn
1.2.2, invoked as a subprocess, with parquet as the interface.

Treating that as an accident to be patched around would be a mistake. A released predictor
is a frozen artifact, and several of them pin dependencies that cannot coexist in one
environment: DeepEnzyme additionally needs NVIDIA apex and an old numpy. One environment
per predictor, recorded next to its mirror, is the shape this has to take.
"""

from __future__ import annotations

import os.path as osp
import subprocess
import tempfile
from typing import Any

import pandas as pd

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest

PINNED_PYTHON = "/scratch/projects/torchcell-scratch/envs/unikp/bin/python"
PROTEIN_EMBEDDINGS = (
    "/scratch/projects/torchcell-scratch/data/torchcell/kinetics/_features/"
    "prot_t5_yeast_gem.npz"
)


class UniKPDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` and :math:`K_M` for every (enzyme, substrate) pair."""

    EMITS = (KineticParameter.k_cat, KineticParameter.K_M)
    MIRROR_NAME = "unikp"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the model loads in the subprocess."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(f"unikp mirror failed verification: {broken[:3]}")
        # The kcat regressor is the artifact a k_cat value is attributable to. The SMILES
        # transformer is shared by both parameters and is pinned in the manifest too.
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.rel_path == "UniKP for kcat.pkl"
        )
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned UniKP process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_unikp_pinned.py",
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
                "--protein-embeddings",
                PROTEIN_EMBEDDINGS,
            ]
            # No capture: the subprocess reports its own progress, and a failure has to
            # surface as a failure rather than as an empty column.
            subprocess.run(command, check=True)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["k_cat", "K_M", "failure"]]
