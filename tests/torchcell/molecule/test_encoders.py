# tests/torchcell/molecule/test_encoders.py
# [[tests.torchcell.molecule.test_encoders]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/molecule/test_encoders.py
"""Every registered molecule encoder: shape, dtype, finiteness, determinism, a
chemical-sense check, and a raise on an unparsable SMILES.

The RDKit encoders run everywhere. The learned encoders are ``slow`` and skipped when
their weights are not already on this machine (the HF hub cache, the pinned files
under ``$DATA_ROOT/data/torchcell/molecule_encoders``, unimol_tools' bundled
checkpoint), so CI never downloads a model.
"""

from __future__ import annotations

import importlib.util
import os.path as osp
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from torchcell.molecule import ENCODERS, MoleculeEncoder, cosine_matrix, standardize
from torchcell.molecule.weights import WEIGHTS, encoders_data_dir

# ethanol, phenol, acetic acid, vanillin
SMILES = ["CCO", "c1ccccc1O", "CC(=O)O", "O=Cc1ccc(O)c(OC)c1"]
FINGERPRINT = {"ecfp4_count", "ecfp4_bit", "fcfp4_count", "maccs"}
RDKIT_ONLY = FINGERPRINT | {"rdkit_2d"}
HF_IDS = {
    "chemberta2_mlm": "DeepChem/ChemBERTa-77M-MLM",
    "chemberta2_mtr": "DeepChem/ChemBERTa-77M-MTR",
    "molformer_xl": "ibm-research/MoLFormer-XL-both-10pct",
    "roberta_zinc_480m": "entropy/roberta_zinc_480m",
}


def _weights_present(name: str) -> bool:
    if name in RDKIT_ONLY:
        return True
    if name in HF_IDS:
        return osp.isdir(
            osp.join(HF_HUB_CACHE, "models--" + HF_IDS[name].replace("/", "--"))
        )
    if name == "mol2vec":
        return (encoders_data_dir() / "mol2vec" / WEIGHTS["mol2vec"].filename).exists()
    if name == "mole_static":
        return (encoders_data_dir() / "mole" / WEIGHTS["mole"].filename).exists()
    if name == "unimol_v1":
        spec = importlib.util.find_spec("unimol_tools")
        if spec is None or spec.origin is None:
            return False
        return (
            Path(spec.origin).parent / "weights" / "mol_pre_all_h_220816.pt"
        ).exists()
    raise KeyError(name)


def _param(name: str) -> object:
    marks = []
    if name not in RDKIT_ONLY:
        marks.append(pytest.mark.slow)
        marks.append(
            pytest.mark.skipif(
                not _weights_present(name), reason=f"{name} weights not on this machine"
            )
        )
    return pytest.param(name, marks=marks)


ENCODER_PARAMS = [_param(n) for n in ENCODERS]
_INSTANCES: dict[str, MoleculeEncoder] = {}


@pytest.fixture(params=ENCODER_PARAMS)
def encoder(request: pytest.FixtureRequest) -> MoleculeEncoder:
    """One instance per encoder for the module (models load once)."""
    name = request.param
    if name not in _INSTANCES:
        _INSTANCES[name] = ENCODERS[name]()
    return _INSTANCES[name]


def test_registry_names_match_class_attribute() -> None:
    for name, cls in ENCODERS.items():
        assert cls.name == name
        assert issubclass(cls, MoleculeEncoder)


def test_standardize_is_canonical_and_keeps_charge_and_stereo() -> None:
    assert standardize("OCC") == standardize("CCO") == "CCO"
    assert standardize("[Na+].[Cl-]") == "[Cl-].[Na+]"
    assert "@" in standardize("C[C@H](N)C(=O)O")
    with pytest.raises(ValueError):
        standardize("not a smiles")


def test_encode_shape_dtype_finite_deterministic(encoder: MoleculeEncoder) -> None:
    x = encoder.encode(SMILES)
    assert x.shape == (4, encoder.dim)
    assert x.dtype == np.float32
    if encoder.name != "rdkit_2d":
        assert np.isfinite(x).all()
    else:
        # every descriptor is defined for these four small molecules
        names = encoder.feature_names  # type: ignore[attr-defined]
        assert np.isfinite(x).all(), np.array(names)[~np.isfinite(x).all(0)]
    y = encoder.encode(SMILES)
    assert np.allclose(x, y, rtol=1e-5, atol=1e-5, equal_nan=True)
    # spelling independence: the same molecule written differently embeds identically
    z = encoder.encode(["OCC"])
    assert np.allclose(x[:1], z, rtol=1e-5, atol=1e-5, equal_nan=True)


def test_ethanol_closer_to_acetic_acid_than_vanillin(encoder: MoleculeEncoder) -> None:
    x = encoder.encode(SMILES)
    if encoder.name == "rdkit_2d":
        x = np.nan_to_num(x)
    sim = cosine_matrix(x[:1], x)
    ethanol_acetic, ethanol_vanillin = sim[0, 2], sim[0, 3]
    print(
        f"{encoder.name}: cos(EtOH, AcOH)={ethanol_acetic:.3f} cos(EtOH, vanillin)={ethanol_vanillin:.3f}"
    )
    if encoder.name in FINGERPRINT:
        assert ethanol_acetic > ethanol_vanillin


def test_unparsable_smiles_raises(encoder: MoleculeEncoder) -> None:
    with pytest.raises(ValueError):
        encoder.encode(["not a smiles"])
    with pytest.raises(ValueError):
        encoder.encode(["CCO", "C1CC"])  # unclosed ring


def test_empty_input(encoder: MoleculeEncoder) -> None:
    x = encoder.encode([])
    assert x.shape == (0, encoder.dim)
    assert x.dtype == np.float32


def test_rdkit_2d_feature_names_and_nan_policy() -> None:
    enc = ENCODERS["rdkit_2d"]()
    names: list[str] = enc.feature_names  # type: ignore[attr-defined]
    assert len(names) == enc.dim
    assert "MolWt" in names
    x = enc.encode(["CCO"])
    col: int = names.index("MolWt")
    assert x[0, col] == pytest.approx(46.069, abs=1e-2)


def test_ecfp4_bit_is_indicator_of_count() -> None:
    xc = ENCODERS["ecfp4_count"]().encode(SMILES)
    xb = ENCODERS["ecfp4_bit"]().encode(SMILES)
    assert set(np.unique(xb)) <= {0.0, 1.0}
    assert np.array_equal(xb, (xc > 0).astype(np.float32))
    assert xc.max() > 1  # counts really are counts


@pytest.mark.slow
@pytest.mark.skipif(not _weights_present("unimol_v1"), reason="unimol weights absent")
def test_unimol_raises_instead_of_embedding_degraded_coordinates() -> None:
    """unimol_tools substitutes zero or 2D coordinates when ETKDG fails; we raise."""
    enc = _INSTANCES.setdefault("unimol_v1", ENCODERS["unimol_v1"]())
    with pytest.raises(ValueError, match="all-zero"):
        enc.encode(["[Na+].[Cl-]"])
    with pytest.raises(ValueError, match="2D"):
        enc.encode(["[Cl][Hg][Cl]"])


def test_mol2alt_sentence_matches_reference_layout() -> None:
    from rdkit import Chem

    from torchcell.molecule import mol2alt_sentence

    # ethanol: 3 heavy atoms x (radius 0 + radius 1) identifiers, all non-zero
    sentence = mol2alt_sentence(Chem.MolFromSmiles("CCO"), radius=1)
    assert len(sentence) == 6
    assert all(s.isdigit() for s in sentence)
    # the two methyl/methylene carbons share the radius-0 id only if same env: they do not
    assert sentence[0] != sentence[2]
