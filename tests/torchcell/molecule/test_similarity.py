# tests/torchcell/molecule/test_similarity.py
# [[tests.torchcell.molecule.test_similarity]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/molecule/test_similarity.py
"""Tanimoto (bit and generalized count) and cosine matrices against RDKit and a
brute-force reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from torchcell.molecule import ENCODERS, cosine_matrix, tanimoto_matrix

SMILES = ["CCO", "c1ccccc1O", "CC(=O)O", "O=Cc1ccc(O)c(OC)c1", "CCCCCCCC"]


def test_bit_tanimoto_matches_rdkit():
    x = ENCODERS["ecfp4_bit"]().encode(SMILES)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    ref = np.array([[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps])
    got = tanimoto_matrix(x, x)
    assert got.shape == (5, 5)
    assert np.allclose(got, ref)
    assert np.allclose(np.diag(got), 1.0)


def test_count_tanimoto_matches_bruteforce_and_rdkit():
    x = ENCODERS["ecfp4_count"]().encode(SMILES)
    got = tanimoto_matrix(x, x)
    ref = np.array(
        [[np.minimum(a, b).sum() / np.maximum(a, b).sum() for b in x] for a in x]
    )
    assert np.allclose(got, ref)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetCountFingerprint(Chem.MolFromSmiles(s)) for s in SMILES]
    rd = np.array([[DataStructs.TanimotoSimilarity(a, b) for b in fps] for a in fps])
    assert np.allclose(got, rd)


def test_tanimoto_rectangular_and_zero_rows():
    a = np.array([[1, 0, 2], [0, 0, 0]], dtype=float)
    b = np.array([[1, 0, 2], [0, 1, 0], [0, 0, 0]], dtype=float)
    got = tanimoto_matrix(a, b)
    assert got.shape == (2, 3)
    assert got[0, 0] == pytest.approx(1.0)
    assert got[0, 1] == pytest.approx(0.0)
    assert got[1, 2] == 0.0  # 0/0 reported as 0
    with pytest.raises(ValueError):
        tanimoto_matrix(-a, b)
    with pytest.raises(ValueError):
        tanimoto_matrix(a, b[:, :2])


def test_cosine_matches_bruteforce():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 7))
    b = rng.normal(size=(3, 7))
    got = cosine_matrix(a, b)
    ref = np.array(
        [[x @ y / (np.linalg.norm(x) * np.linalg.norm(y)) for y in b] for x in a]
    )
    assert got.shape == (4, 3)
    assert np.allclose(got, ref)
    assert np.allclose(np.diag(cosine_matrix(a, a)), 1.0)
    zero = np.zeros((1, 7))
    assert cosine_matrix(zero, b).tolist() == [[0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        cosine_matrix(a, b[:, :2])
    with pytest.raises(ValueError):
        cosine_matrix(a[0], b)
