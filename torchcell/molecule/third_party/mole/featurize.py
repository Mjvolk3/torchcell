# torchcell/molecule/third_party/mole/featurize.py
# [[torchcell.molecule.third_party.mole.featurize]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/molecule/third_party/mole/featurize.py
"""MolE graph featurization, vendored from
https://github.com/rolayoalarcon/MolE (``dataset/dataset_representation.py``,
``MoleculeDataset.__getitem__``; MIT License, see ``LICENSE`` beside this file).

The atom features are ``(atomic-number index, chirality-tag index)`` and the bond
features ``(bond-type index, bond-direction index)``, both as long tensors, on the
molecule WITH explicit hydrogens (``Chem.AddHs``), exactly as at pre-training. The one
change from upstream: a molecule with no bonds (a bare ion such as ``[Cl-]``) gets an
empty ``(0, 2)`` edge-attribute tensor instead of a shape-``(0,)`` one that the
convolution cannot index.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from torch_geometric.data import Data

ATOM_LIST = list(range(1, 119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
]
BOND_LIST = [BT.SINGLE, BT.DOUBLE, BT.TRIPLE, BT.AROMATIC]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT,
]


def mol_to_mole_graph(mol: Chem.Mol) -> Data:
    """One PyG ``Data`` in MolE's feature vocabulary for a sanitized RDKit mol.

    Raises ``ValueError`` (from ``list.index``) for an atom, bond type or chirality
    tag outside the vocabulary the checkpoint was trained on, rather than mapping it
    to a stand-in token.
    """
    mol = Chem.AddHs(mol)

    type_idx = []
    chirality_idx = []
    for atom in mol.GetAtoms():
        type_idx.append(ATOM_LIST.index(atom.GetAtomicNum()))
        chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))

    x1 = torch.tensor(type_idx, dtype=torch.long).view(-1, 1)
    x2 = torch.tensor(chirality_idx, dtype=torch.long).view(-1, 1)
    x = torch.cat([x1, x2], dim=-1)

    row, col, edge_feat = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        feat = [
            BOND_LIST.index(bond.GetBondType()),
            BONDDIR_LIST.index(bond.GetBondDir()),
        ]
        edge_feat.append(feat)
        edge_feat.append(feat)

    edge_index = torch.tensor([row, col], dtype=torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_feat, dtype=torch.long).view(-1, 2)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
