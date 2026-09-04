# torchcell/datasets/kinetics/turnup.py
# [[torchcell.datasets.kinetics.turnup]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/turnup.py
# Test file: tests/torchcell/datasets/kinetics/test_turnup.py

r"""TurNuP: a reaction fingerprint plus a task-specific ESM-1b vector into xgboost.

Kroll et al. (2023), *Nature Communications*. Every substrate and product of the reaction
is turned into SMARTS and joined into a reaction SMARTS, from which rdkit's difference
fingerprint gives 2,048 dimensions; the enzyme contributes 1,280 dimensions from an
ESM-1b fine-tuned on a task-specific objective, read off the BOS token of layer 33. A
gradient boosted tree maps the concatenation to :math:`\log_{10} k_{cat}`, exponentiated
base 10 at the edge of this module.

THE INPUT IS A REACTION, WHICH THE PAIR TABLE IS NOT
------------------------------------------------------
DLKcat and UniKP both score one enzyme against one substrate, so the shared pair table --
one row per (catalytic unit, substrate) -- is exactly their input. TurNuP has no notion of
a distinguished substrate: its fingerprint is the difference between the whole left side
and the whole right side. So this module resolves each row's REACTION out of yeast-GEM,
assembles both sides from SMILES, and predicts once per distinct reaction. Every pair row
sharing that reaction and enzyme receives the same value, which is a property of the model
rather than an aggregation choice: TurNuP simply does not resolve substrates within a
reaction.

WHY A REACTION WITH ONE UNRESOLVED METABOLITE IS A FAILURE, NOT A SHORTER REACTION
------------------------------------------------------------------------------------
The authors' code marks a reaction side invalid as soon as any one metabolite fails to
parse. Dropping the offending metabolite instead would still produce a fingerprint, and
that fingerprint would be the fingerprint of a DIFFERENT reaction -- an unbalanced one the
model was never fit on. yeast-GEM's acyl-chain-resolved lipids carry no structure in
either ``smilesDB`` or MetaNetX, so this costs real coverage, and the coverage is reported
rather than recovered.

WHY THIS CLASS SHELLS OUT INSTEAD OF PREDICTING IN PROCESS
-----------------------------------------------------------
The released booster is a pickle from xgboost 1.6 wrapping a pre-1.6 JSON model, and
xgboost drops that format at 2.3; the project environment carries no xgboost at all, and
fair-esm pins an older torch surface besides. Inference therefore runs in an environment
pinned to xgboost 1.6.1 / fair-esm 2.0.0 / torch 2.5.1, invoked as a subprocess with
parquet as the interface -- the same arrangement UniKP needs for its scikit-learn pin.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any

import pandas as pd

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest

PINNED_PYTHON = "/scratch/projects/torchcell-scratch/envs/turnup/bin/python"
# The booster is what a k_cat value is attributable to; the ESM-1b fine-tune that produces
# its enzyme half is pinned by the same manifest.
WEIGHTS_REL_PATH = "data/saved_models/xgboost/xgboost_train_and_test.pkl"


class TurnupDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` for every (enzyme, reaction) the GEM can express."""

    EMITS = (KineticParameter.k_cat,)
    MIRROR_NAME = "turnup"

    def initialize_model(self) -> Any:
        """Verify the mirror and pin the weights hash; the model loads in the subprocess."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(
                f"turnup mirror failed verification: {broken[:3]}. The weights changed "
                "since they were pinned, so predictions would not match earlier builds."
            )
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.rel_path == WEIGHTS_REL_PATH
        )
        return mirror.path(self.data_root)

    @property
    def gem_dir(self) -> str:
        """The yeast-GEM release the pair table was built from."""
        return osp.join(self.data_root, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")

    @property
    def embedding_cache(self) -> str:
        """Where the ESM1b_ts vectors live between blocks."""
        return osp.join(self.root, self.MIRROR_NAME, "features", "esm1b_ts_yeast_gem.npz")

    def _smiles_by_name(self) -> dict[str, str]:
        """Metabolite name -> SMILES from yeast-GEM's own table, lowercased for the join."""
        path = osp.join(self.gem_dir, "data", "databases", "smilesDB.tsv")
        out: dict[str, str] = {}
        with open(path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                name, smiles = parts[0].strip(), parts[1].strip()
                if name and smiles:
                    out[name.lower()] = smiles
        return out

    def _smiles_by_metanetx(self, wanted: set[str]) -> dict[str, str]:
        """MNXref id -> SMILES, restricted to the ids the model actually annotates.

        ``chem_prop.tsv`` is 810 MB and holds 1.4 million compounds; keeping only the few
        thousand the GEM references is the difference between a dictionary that fits in
        memory comfortably and one that does not.
        """
        path = osp.join(
            self.data_root, "data", "enzyme_kinetics", "metanetx", "chem_prop.tsv"
        )
        out: dict[str, str] = {}
        with open(path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 9 or parts[0] not in wanted:
                    continue
                if parts[8].strip():
                    out[parts[0]] = parts[8].strip()
        return out

    def _reaction_sides(self) -> dict[str, tuple[str | None, str | None, str | None]]:
        """Reaction id -> (substrate SMILES, product SMILES, failure reason).

        Both SMILES fields are ';'-joined in the form the authors' prediction function
        takes. A reaction that cannot be fully resolved carries ``None`` for both and a
        reason instead.
        """
        import cobra

        model = cobra.io.read_sbml_model(osp.join(self.gem_dir, "model", "yeast-GEM.xml"))
        by_name = self._smiles_by_name()

        wanted: set[str] = set()
        for metabolite in model.metabolites:
            annotation = metabolite.annotation.get("metanetx.chemical")
            if not annotation:
                continue
            for candidate in (
                annotation if isinstance(annotation, list) else [annotation]
            ):
                wanted.add(str(candidate))
        by_metanetx = self._smiles_by_metanetx(wanted)

        def resolve(metabolite: Any) -> str | None:
            smiles = by_name.get((metabolite.name or "").lower())
            if smiles:
                return smiles
            annotation = metabolite.annotation.get("metanetx.chemical")
            if not annotation:
                return None
            for candidate in (
                annotation if isinstance(annotation, list) else [annotation]
            ):
                smiles = by_metanetx.get(str(candidate))
                if smiles:
                    return smiles
            return None

        sides: dict[str, tuple[str | None, str | None, str | None]] = {}
        for reaction in model.reactions:
            reactants = [resolve(m) for m in reaction.reactants]
            products = [resolve(m) for m in reaction.products]
            if not reactants or not products:
                sides[reaction.id] = (None, None, "reaction_has_empty_side")
            elif any(s is None for s in reactants + products):
                sides[reaction.id] = (None, None, "metabolite_without_smiles")
            else:
                sides[reaction.id] = (";".join(reactants), ";".join(products), None)
        return sides

    @property
    def reaction_sides(self) -> dict[str, tuple[str | None, str | None, str | None]]:
        """The reaction table, built once per process."""
        if not hasattr(self, "_reaction_sides_cache"):
            self._reaction_sides_cache = self._reaction_sides()
        return self._reaction_sides_cache

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Run the pinned TurNuP process over a block of pair rows."""
        self.model  # verifies the mirror and sets the weights hash

        sides = self.reaction_sides
        work = rows.copy()
        resolved = [sides.get(r, (None, None, "reaction_not_in_gem")) for r in work["reaction_id"]]
        work["substrate_smiles"] = [s for s, _, _ in resolved]
        work["product_smiles"] = [p for _, p, _ in resolved]
        work["reaction_failure"] = [f for _, _, f in resolved]

        runner = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            "experiments",
            "026-metabolism-flux",
            "scripts",
            "run_turnup_pinned.py",
        )
        os.makedirs(osp.dirname(self.embedding_cache), exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            inputs = osp.join(scratch, "inputs.parquet")
            output = osp.join(scratch, "predictions.parquet")
            sequences = osp.join(scratch, "sequences.json")
            work.to_parquet(inputs, index=False)
            # The superset of sequences, not this block's: loading the ESM-1b encoder
            # reads 10 GB of checkpoint, so it happens on the first block and never again.
            with open(sequences, "w") as handle:
                json.dump(list(self.inputs["sequence"].dropna().unique()), handle)
            command = [
                PINNED_PYTHON,
                runner,
                "--inputs",
                inputs,
                "--output",
                output,
                "--device",
                self.device,
                "--embedding-cache",
                self.embedding_cache,
                "--sequences",
                sequences,
            ]
            # No capture: the subprocess reports its own progress, and a failure has to
            # surface as a failure rather than as an empty column.
            subprocess.run(command, check=True)
            predictions = pd.read_parquet(output)

        predictions.index = rows.index
        return predictions[["k_cat", "failure"]]
