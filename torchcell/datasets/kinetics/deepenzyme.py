# torchcell/datasets/kinetics/deepenzyme.py
# [[torchcell.datasets.kinetics.deepenzyme]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/deepenzyme.py
# Test file: tests/torchcell/datasets/kinetics/test_deepenzyme.py

r"""DeepEnzyme: the structure-informed :math:`k_{cat}` predictor of the set.

Wang et al. (2024), *Briefings in Bioinformatics*. The substrate is encoded as
Weisfeiler-Lehman subgraph fingerprints over its molecular graph; the enzyme is encoded
twice from overlapping 4-grams of its sequence, once through a transformer encoder and
once through a GCN over the residue contact map of an AlphaFold structure at a 10 Angstrom
C-alpha cutoff. The three pooled vectors are concatenated and mapped to a scalar. Adding
the contact map is the whole point of the model, and it is what makes this the only
predictor here that can distinguish two enzymes whose sequences are similar but whose
folds are not.

The model predicts :math:`\log_2 k_{cat}`, exponentiated at the edge of this module so
nothing downstream has to remember the convention. The base was determined from the
authors' own conversion, their training script, and the distribution of their released
training targets rather than assumed; ``run_deepenzyme_pinned.py`` records the evidence.

WHAT THE EARLIER AUDIT GOT WRONG, AND WHAT IT GOT RIGHT
--------------------------------------------------------
The mirror was recorded as ``runnable=False``, blocked by "requires NVIDIA apex plus
pinned old numpy/rdkit; no weights shipped." Two thirds of that is wrong and one third is
right in a way that had a cheap fix. ``apex`` is listed in the authors'
``requirements.txt`` but imported nowhere in ``Code/``; it was a training-time
mixed-precision dependency that never reaches the forward pass, and the network runs
unmodified on this project's torch, numpy and rdkit. The weights genuinely are not in the
repository, but the authors' example points at a figshare record in a comment, and the
110 MB checkpoint there loads into their network with every one of its 137 tensors
matched. What the audit had right is that nothing in the repository alone can predict.

WHY THIS CLASS SHELLS OUT
--------------------------
Not for dependency isolation, unlike ``UniKPDataset``. The authors' network hardcodes
``.cuda()`` in four places, so a GPU can only be selected through ``CUDA_VISIBLE_DEVICES``
set before torch initializes, which an in-process caller that has already imported torch
cannot do. The mirror's module is also called ``example_model`` at the top of
``sys.path``, a name not worth introducing into a long-lived process. Parquet on each side
of the boundary is the interface.
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
STRUCTURES = (
    "/scratch/projects/torchcell-scratch/data/enzyme_kinetics/alphafold/scerevisiae"
)
# The released checkpoint, retrieved from figshare 10.6084/m9.figshare.25771062.v2 (file
# "example", md5 68f83e5d90937e66a6f62b85e8695f38) and pinned in the mirror manifest.
WEIGHTS_REL_PATH = "Weights/example"


class DeepEnzymeDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` for every (enzyme, substrate, structure) triple."""

    EMITS = (KineticParameter.k_cat,)
    MIRROR_NAME = "deepenzyme"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the network loads in the child."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(
                f"deepenzyme mirror failed verification: {broken[:3]}. The weights "
                "changed since they were pinned, so predictions would not match earlier "
                "builds."
            )
        # Matched by path rather than by role: the mirror's role map predates the
        # checkpoint being added, so Weights/example is currently recorded as 'source'.
        # The path is what identifies the artifact either way.
        weights = [f for f in mirror.files if f.rel_path == WEIGHTS_REL_PATH]
        if not weights:
            raise RuntimeError(
                f"{WEIGHTS_REL_PATH} is not in the deepenzyme manifest. Retrieve the "
                "checkpoint from figshare 10.6084/m9.figshare.25771062 and re-pin the "
                "mirror; without it there is no model, only an architecture."
            )
        self._weights_sha256 = weights[0].sha256
        return mirror.path(self.data_root)

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run DeepEnzyme in a child process over a block of pairs."""
        self.model  # verifies the mirror and sets the weights hash

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_deepenzyme_pinned.py",
        )
        # The network selects its GPU by masking, not by an argument, so the device this
        # dataset was constructed with is translated into a mask for the child.
        index = self.device.split(":")[-1] if self.device.startswith("cuda") else "0"
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=index)

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
                "--structures",
                STRUCTURES,
            ]
            # No capture: the child reports its own progress, and a failure has to surface
            # as a failure rather than as an empty column.
            subprocess.run(command, check=True, env=environment)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["k_cat", "failure", "unseen_token_frac", "n_ca"]]
