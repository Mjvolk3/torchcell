# torchcell/datasets/kinetics/dlkcat.py
# [[torchcell.datasets.kinetics.dlkcat]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datasets/kinetics/dlkcat.py
# Test file: tests/torchcell/datasets/kinetics/test_dlkcat.py

r"""DLKcat: a graph-neural / CNN model predicting :math:`k_{cat}` from sequence + SMILES.

Li et al. (2022), *Nature Catalysis*. The substrate is encoded as Weisfeiler-Lehman
subgraph fingerprints over its molecular graph, the enzyme as overlapping 3-grams of its
amino-acid sequence through a CNN, and the two are combined by attention into a single
scalar. The model predicts :math:`\log_2 k_{cat}`, which is exponentiated at the edge of
this module so nothing downstream has to remember the convention.

THE AUTHORS' CODE IS RUN, NOT REIMPLEMENTED
--------------------------------------------
``KcatPrediction`` is imported from the mirror rather than transcribed. A reimplementation
that is subtly wrong -- a mean where the authors wrote a sum, a fingerprint radius off by
one -- would still produce plausible numbers, and nothing downstream could detect it. The
cost is a ``sys.path`` insertion; the benefit is that the values are DLKcat's values.

THE TWO DICTIONARIES ARE THE MODEL, AND THEY ARE CLOSED
--------------------------------------------------------
The fingerprint and 3-gram vocabularies were frozen at training time, and the embedding
tables are sized to them. The authors' own script maps an unseen key to index 0 and
continues. That is a real limitation rather than a bug: index 0 is a legitimate learned
embedding, so an unseen substructure is silently predicted as whatever index 0 means. This
module counts those hits instead of hiding them, because the fraction of a molecule's
fingerprints that were never seen in training is the honest confidence signal DLKcat
otherwise does not expose.
"""

from __future__ import annotations

import os.path as osp
import pickle
import sys
from collections import defaultdict
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from torchcell.data.kinetics import BaseKineticsDataset, KineticParameter
from torchcell.data.model_mirror import read_manifest

# Hyperparameters are fixed by the released checkpoint, not free choices: the filename
# records radius 2, 3-grams, dim 10 (doubled to 20 inside the model), 3 GNN layers, a
# window of 11 and 3 CNN / 3 output layers. Changing any of them fails to load.
RADIUS = 2
NGRAM = 3
DIM = 10
LAYER_GNN = 3
WINDOW = 11
LAYER_CNN = 3
LAYER_OUTPUT = 3
CHECKPOINT = (
    "all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3"
    "--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration50"
)


class DlkcatDataset(BaseKineticsDataset):
    """Predicted :math:`k_{cat}` for every (enzyme, substrate) pair in a GEM."""

    EMITS = (KineticParameter.k_cat,)
    MIRROR_NAME = "dlkcat"

    def _mirror_root(self) -> str:
        """The mirror directory, verified before anything is read out of it."""
        mirror = read_manifest(self.data_root, self.MIRROR_NAME)
        broken = mirror.verify(self.data_root)
        if broken:
            raise RuntimeError(
                f"dlkcat mirror failed verification: {broken[:3]}. The weights changed "
                "since they were pinned, so predictions would not match earlier builds."
            )
        self._weights_sha256 = next(
            f.sha256 for f in mirror.files if f.role == "weights"
        )
        return mirror.path(self.data_root)

    def initialize_model(self) -> Any:
        """Load the authors' network and its frozen vocabularies from the mirror."""
        root = self._mirror_root()
        code_dir = osp.join(root, "DeeplearningApproach", "Code", "example")
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        import model as dlkcat_model  # the authors' module, from the mirror

        data_dir = osp.join(root, "DeeplearningApproach", "Data", "input")
        with open(osp.join(data_dir, "fingerprint_dict.pickle"), "rb") as handle:
            self.fingerprint_dict = pickle.load(handle)
        with open(osp.join(data_dir, "atom_dict.pickle"), "rb") as handle:
            self.atom_dict = pickle.load(handle)
        with open(osp.join(data_dir, "bond_dict.pickle"), "rb") as handle:
            self.bond_dict = pickle.load(handle)
        with open(osp.join(data_dir, "edge_dict.pickle"), "rb") as handle:
            self.edge_dict = pickle.load(handle)
        with open(osp.join(data_dir, "sequence_dict.pickle"), "rb") as handle:
            self.word_dict = pickle.load(handle)

        device = torch.device(self.device)
        net = dlkcat_model.KcatPrediction(
            device,
            len(self.fingerprint_dict),
            len(self.word_dict),
            2 * DIM,
            LAYER_GNN,
            WINDOW,
            LAYER_CNN,
            LAYER_OUTPUT,
        ).to(device)
        weights = osp.join(
            root, "DeeplearningApproach", "Results", "output", CHECKPOINT
        )
        net.load_state_dict(torch.load(weights, map_location=device))
        net.eval()
        return net

    def _split_sequence(self, sequence: str) -> tuple[np.ndarray, int]:
        """Overlapping 3-grams as vocabulary indices, plus how many were unseen."""
        padded = "-" + sequence + "="
        words, unseen = [], 0
        for i in range(len(padded) - NGRAM + 1):
            key = padded[i : i + NGRAM]
            if key in self.word_dict:
                words.append(self.word_dict[key])
            else:
                words.append(0)
                unseen += 1
        return np.array(words), unseen

    def _fingerprints(self, mol: Any) -> tuple[np.ndarray, int]:
        """Weisfeiler-Lehman subgraph fingerprints, plus how many were unseen.

        This is the authors' ``extract_fingerprints`` with the silent-miss branch counted
        rather than swallowed.
        """
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        for a in mol.GetAromaticAtoms():
            atoms[a.GetIdx()] = (atoms[a.GetIdx()], "aromatic")
        atom_ids = [self.atom_dict.get(a, 0) for a in atoms]

        i_jbond: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            b = self.bond_dict[str(bond.GetBondType())]
            i_jbond[i].append((j, b))
            i_jbond[j].append((i, b))

        unseen = 0
        if len(atom_ids) == 1 or RADIUS == 0:
            fingerprints = []
            for a in atom_ids:
                if a in self.fingerprint_dict:
                    fingerprints.append(self.fingerprint_dict[a])
                else:
                    fingerprints.append(0)
                    unseen += 1
            return np.array(fingerprints), unseen

        nodes, i_jedge = atom_ids, i_jbond
        fingerprints = []
        for _ in range(RADIUS):
            fingerprints = []
            for i, j_edge in i_jedge.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                key = (nodes[i], tuple(sorted(neighbors)))
                if key in self.fingerprint_dict:
                    fingerprints.append(self.fingerprint_dict[key])
                else:
                    fingerprints.append(0)
                    unseen += 1
            nodes = fingerprints
            next_edge: dict[int, list[tuple[int, int]]] = defaultdict(list)
            for i, j_edge in i_jedge.items():
                for j, edge in j_edge:
                    both = tuple(sorted((nodes[i], nodes[j])))
                    next_edge[i].append((j, self.edge_dict.get((both, edge), 0)))
            i_jedge = next_edge
        return np.array(fingerprints), unseen

    def predict_batch(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Predict :math:`k_{cat}` for each (sequence, SMILES) row.

        DLKcat's forward pass takes one pair at a time: the molecular graph and the
        sequence both have per-example shapes, and the attention step is defined over a
        single pair. Batching would require padding and masking the authors' model does
        not implement, so this loops. At 7,456 pairs on CPU that is minutes.
        """
        from rdkit import Chem, RDLogger

        # rdkit narrates every valence complaint on stderr; the failures that matter are
        # captured per row below.
        RDLogger.DisableLog("rdApp.*")

        net = self.model
        device = torch.device(self.device)
        values: list[float] = []
        failures: list[str | None] = []
        unseen_fracs: list[float] = []

        for row in rows.itertuples():
            smiles = cast(str, row.smiles)
            sequence = cast(str, row.sequence)
            if not isinstance(smiles, str) or not smiles:
                values.append(np.nan)
                failures.append("no_smiles")
                unseen_fracs.append(np.nan)
                continue
            if "." in smiles:
                # DLKcat's own prediction script refuses any multi-component SMILES. Its
                # fingerprint pass indexes ``nodes`` by atom index while building that
                # list only from bonded atoms, so a disconnected fragment -- the lone
                # ``[Fe+2]`` in the heme SMILES yeast-GEM gives for ferricytochrome c --
                # walks off the end of the list. Excluding these is what the model does,
                # so the exclusion is recorded rather than worked around.
                values.append(np.nan)
                failures.append("multi_component_smiles")
                unseen_fracs.append(np.nan)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                values.append(np.nan)
                failures.append("rdkit_parse_failed")
                unseen_fracs.append(np.nan)
                continue
            # Hydrogens are added because the training featurization did; a graph built
            # without them yields different fingerprints for the same molecule.
            mol = Chem.AddHs(mol)
            fingerprints, fp_unseen = self._fingerprints(mol)
            if fingerprints.size == 0:
                values.append(np.nan)
                failures.append("empty_fingerprint")
                unseen_fracs.append(np.nan)
                continue
            adjacency = Chem.GetAdjacencyMatrix(mol)
            words, word_unseen = self._split_sequence(sequence)

            inputs = (
                torch.LongTensor(fingerprints).to(device),
                torch.FloatTensor(adjacency).to(device),
                torch.LongTensor(words).to(device),
            )
            with torch.no_grad():
                log2_kcat = net.forward(inputs).item()
            # BASE 2, not 10. The authors' script exponentiates with ``math.pow(2, ...)``,
            # and the paper's own figures are log2. Reading it as log10 inflates every
            # value by a power of about 3.3 and still looks like a plausible k_cat.
            values.append(float(2.0**log2_kcat))
            failures.append(None)
            total = fingerprints.size + words.size
            unseen_fracs.append((fp_unseen + word_unseen) / total if total else np.nan)

        return pd.DataFrame(
            {"k_cat": values, "failure": failures, "unseen_token_frac": unseen_fracs},
            index=rows.index,
        )
