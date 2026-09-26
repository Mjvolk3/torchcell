# torchcell/molecule/encoders.py
# [[torchcell.molecule.encoders]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/molecule/encoders.py
"""Small-molecule encoders: SMILES in, one fixed-width float32 vector per molecule out.

Every encoder is a :class:`MoleculeEncoder` with ``name``, ``dim`` and
``encode(smiles) -> (n, dim) float32``, registered in :data:`ENCODERS`. Contract:

- **Unparsable SMILES raise** (``ValueError`` from :func:`parse_smiles`), before any
  model runs; an encoder never returns a zero row for a molecule it could not read.
  Callers own coverage accounting.
- **Deterministic.** Learned encoders run in eval mode under ``inference_mode``; the
  same SMILES list yields the same array on repeated calls.
- **Spelling-independent.** Fingerprints act on the parsed mol; learned encoders that
  read text are fed :func:`standardize`'s canonical isomeric SMILES, so two spellings
  of one molecule embed identically.
- ``rdkit_2d`` is the one encoder whose output may hold NaN (a descriptor that is
  undefined or overflowed for that molecule); the caller imputes.

Fingerprints (``ecfp4_count``, ``ecfp4_bit``, ``fcfp4_count``, ``maccs``,
``rdkit_2d``) need only RDKit. ``mol2vec`` needs ``gensim`` plus the pinned pickle from
:mod:`torchcell.molecule.weights`; the Hugging Face encoders download to the default
HF cache on first use; ``unimol_v1`` needs ``unimol_tools``; ``mole_static`` needs the
Zenodo checkpoint plus the vendored model under ``third_party/mole``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]

ENCODERS: dict[str, type[MoleculeEncoder]] = {}


def parse_smiles(smiles: str) -> Chem.Mol:
    """Sanitized RDKit mol for ``smiles``; ``ValueError`` when RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparsable SMILES: {smiles!r}")
    return mol


def standardize(smiles: str) -> str:
    """Canonical isomeric SMILES (charges, stereo and every fragment kept).

    No neutralization, salt stripping or tautomer canonicalization: the string names
    the molecule as the source released it, only spelled canonically.
    """
    return Chem.MolToSmiles(parse_smiles(smiles), isomericSmiles=True, canonical=True)


def resolve_device(device: str | None) -> str:
    """``device`` as given, else ``cuda`` when available, else ``cpu``."""
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class MoleculeEncoder(ABC):
    """Base class. Subclasses set ``name``/``dim`` and implement :meth:`_encode`."""

    name: ClassVar[str]
    dim: int

    def check(self, smiles: str) -> Chem.Mol:
        """The parsed mol when this encoder can embed ``smiles``; ``ValueError``
        otherwise. The base check is parsability; an encoder with a further
        per-molecule precondition (Uni-Mol's 3D conformer) extends it, so a caller
        can account coverage molecule by molecule and then batch the survivors.
        """
        return parse_smiles(smiles)

    def encode(self, smiles: list[str]) -> NDArray[np.float32]:
        """``(len(smiles), dim)`` float32; raises on the first SMILES :meth:`check`
        rejects, before any model runs.
        """
        mols = [self.check(s) for s in smiles]
        if not mols:
            return np.zeros((0, self.dim), dtype=np.float32)
        canonical = [
            Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) for m in mols
        ]
        x = np.asarray(self._encode(mols, canonical), dtype=np.float32)
        if x.shape != (len(mols), self.dim):
            raise RuntimeError(
                f"{self.name}: expected shape {(len(mols), self.dim)}, got {x.shape}"
            )
        return x

    @abstractmethod
    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        """Encode already-parsed mols (``canonical`` is their canonical SMILES)."""


def register(cls: type[MoleculeEncoder]) -> type[MoleculeEncoder]:
    """Add ``cls`` to :data:`ENCODERS` under ``cls.name``."""
    if cls.name in ENCODERS:
        raise KeyError(f"encoder {cls.name!r} registered twice")
    ENCODERS[cls.name] = cls
    return cls


# --------------------------------------------------------------------------- RDKit


class _MorganEncoder(MoleculeEncoder):
    """Morgan fingerprint of radius 2 (ECFP4-like) hashed to ``fp_size`` bits."""

    fp_size: ClassVar[int] = 2048
    radius: ClassVar[int] = 2
    feature_invariants: ClassVar[bool] = False
    count: ClassVar[bool] = True

    def __init__(self) -> None:
        self.dim = self.fp_size
        kwargs = {}
        if self.feature_invariants:
            kwargs["atomInvariantsGenerator"] = (
                rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            )
        self._gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.radius, fpSize=self.fp_size, **kwargs
        )

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        if self.count:
            rows = [self._gen.GetCountFingerprintAsNumPy(m) for m in mols]
        else:
            rows = [self._gen.GetFingerprintAsNumPy(m) for m in mols]
        return np.stack(rows).astype(np.float32)


@register
class ECFP4Count(_MorganEncoder):
    """Morgan radius-2 count fingerprint, 2048 bits."""

    name = "ecfp4_count"


@register
class ECFP4Bit(_MorganEncoder):
    """Morgan radius-2 bit fingerprint, 2048 bits."""

    name = "ecfp4_bit"
    count = False


@register
class FCFP4Count(_MorganEncoder):
    """Morgan radius-2 count fingerprint with pharmacophoric feature invariants."""

    name = "fcfp4_count"
    feature_invariants = True


@register
class MACCS(MoleculeEncoder):
    """RDKit MACCS keys, 167 bits (bit 0 is always unset)."""

    name = "maccs"

    def __init__(self) -> None:
        """RDKit-only; nothing to load."""
        self.dim = 167

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        out = np.zeros((len(mols), self.dim), dtype=np.float32)
        for i, m in enumerate(mols):
            arr = np.zeros((self.dim,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(m), arr)  # type: ignore[attr-defined]
            out[i] = arr
        return out


@register
class RDKit2D(MoleculeEncoder):
    """Every RDKit 2D descriptor in ``Descriptors.descList`` (217 in RDKit 2026.03).

    Values that are undefined for a molecule or overflow float32 become NaN. The
    caller must impute before any distance or model; ``feature_names`` lists the
    columns in order.
    """

    name = "rdkit_2d"

    def __init__(self) -> None:
        """Column order is ``Descriptors.descList`` order for this RDKit version."""
        self.feature_names: list[str] = [n for n, _ in Descriptors.descList]
        self.dim = len(self.feature_names)

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        out = np.full((len(mols), self.dim), np.nan, dtype=np.float64)
        for i, m in enumerate(mols):
            d = Descriptors.CalcMolDescriptors(m, missingVal=None, silent=True)  # type: ignore[no-untyped-call]
            out[i] = [
                np.nan if d[n] is None else float(d[n]) for n in self.feature_names
            ]
        x = out.astype(np.float32)
        x[~np.isfinite(x)] = np.nan
        return x


# --------------------------------------------------------------------------- Mol2Vec


def mol2alt_sentence(mol: Chem.Mol, radius: int = 1) -> list[str]:
    """Mol2Vec's "alternating sentence": for each atom in index order, its Morgan
    identifier at radius 0, then radius 1, ... up to ``radius`` (reimplementation of
    ``mol2vec.features.mol2alt_sentence``; identifiers are the unhashed Morgan
    environment ids, which the 300-dim model's vocabulary is keyed on).
    """
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    gen.GetSparseCountFingerprint(mol, additionalOutput=ao)
    per_atom: dict[int, dict[int, int | None]] = {
        a.GetIdx(): dict.fromkeys(range(radius + 1)) for a in mol.GetAtoms()
    }
    for identifier, envs in ao.GetBitInfoMap().items():
        for atom_idx, r in envs:
            per_atom[atom_idx][r] = identifier
    sentence = [
        str(per_atom[atom][r])
        for atom in per_atom
        for r in range(radius + 1)
        if per_atom[atom][r]
    ]
    return sentence


@register
class Mol2Vec(MoleculeEncoder):
    """Sum of Mol2Vec word vectors over the radius-1 alternating sentence, 300-dim,
    with the model's ``UNK`` vector for identifiers outside its vocabulary
    (``mol2vec.features.sentences2vec(..., unseen="UNK")``).
    """

    name = "mol2vec"

    def __init__(self, radius: int = 1) -> None:
        """Load the pinned gensim model (downloaded once via ``ensure_weights``)."""
        from gensim.models import word2vec  # type: ignore[import-untyped]

        from torchcell.molecule.weights import ensure_weights

        self.radius = radius
        self._model = word2vec.Word2Vec.load(str(ensure_weights("mol2vec")))
        self._vocab = self._model.wv.key_to_index
        self._unk = np.asarray(self._model.wv["UNK"], dtype=np.float32)
        self.dim = int(self._model.wv.vector_size)

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        out = np.zeros((len(mols), self.dim), dtype=np.float32)
        for i, m in enumerate(mols):
            for word in mol2alt_sentence(m, self.radius):
                out[i] += self._model.wv[word] if word in self._vocab else self._unk
        return out


# --------------------------------------------------------------------------- HF


class _HFMeanPool(MoleculeEncoder):
    """Mean of the last hidden state over non-pad tokens of a HF encoder."""

    hf_id: ClassVar[str]
    revision: ClassVar[str | None] = None
    trust_remote_code: ClassVar[bool] = False
    model_kwargs: ClassVar[dict[str, object]] = {}

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        from transformers import AutoModel, AutoTokenizer

        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._tok = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            self.hf_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self._model = (
            AutoModel.from_pretrained(
                self.hf_id,
                revision=self.revision,
                trust_remote_code=self.trust_remote_code,
                **self.model_kwargs,
            )
            .to(self.device)
            .eval()
        )
        self.dim = int(self._model.config.hidden_size)

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        import torch

        chunks = []
        for start in range(0, len(canonical), self.batch_size):
            batch = canonical[start : start + self.batch_size]
            enc = self._tok(
                batch, padding=True, truncation=False, return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                h = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1)
            chunks.append(pooled.float().cpu().numpy())
        return np.concatenate(chunks)


@register
class ChemBERTa2MLM(_HFMeanPool):
    """ChemBERTa-2 77M, masked-language-model pretraining, 384-dim."""

    name = "chemberta2_mlm"
    hf_id = "DeepChem/ChemBERTa-77M-MLM"


@register
class ChemBERTa2MTR(_HFMeanPool):
    """ChemBERTa-2 77M, multi-task-regression pretraining, 384-dim."""

    name = "chemberta2_mtr"
    hf_id = "DeepChem/ChemBERTa-77M-MTR"


@register
class MoLFormerXL(_HFMeanPool):
    """MoLFormer-XL (both-10pct), 768-dim.

    Pinned to HF revision ``7b12d946`` (2024-03-31, the safetensors upload): the
    2026-07 revisions of the repo's remote code import
    ``transformers.masking_utils.create_bidirectional_mask``, absent in transformers
    4.57. ``deterministic_eval=True`` turns the linear-attention feature map's random
    projection deterministic so repeated calls agree.
    """

    name = "molformer_xl"
    hf_id = "ibm-research/MoLFormer-XL-both-10pct"
    revision = "7b12d946c181a37f6012b9dc3b002275de070314"
    trust_remote_code = True
    model_kwargs = {"deterministic_eval": True}


@register
class RobertaZinc480M(_HFMeanPool):
    """RoBERTa pretrained on 480M ZINC SMILES (``entropy/roberta_zinc_480m``); the
    width is read from the config (768).
    """

    name = "roberta_zinc_480m"
    hf_id = "entropy/roberta_zinc_480m"


# --------------------------------------------------------------------------- Uni-Mol


@register
class UniMolV1(MoleculeEncoder):
    """Uni-Mol v1 (84M, all-hydrogen checkpoint) CLS representation, 512-dim, via
    ``unimol_tools.UniMolRepr``; conformers are generated by unimol_tools (RDKit
    ETKDG with its fixed seed 42, so repeated calls agree).

    unimol_tools quietly substitutes flat 2D coordinates when ETKDG fails and
    all-zero coordinates for isolated ions, then embeds them as if they were 3D. A
    molecule that hits either path (its own log lines "Failed conformers" and
    "Failed 3d conformers") raises ``ValueError`` here instead, so the caller counts
    it as not embedded.
    """

    name = "unimol_v1"
    _SEED: ClassVar[int] = 42  # UniMolRepr's default conformer seed
    _MODE: ClassVar[str] = "fast"  # UniMolRepr's default ETKDG mode

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        """Build ``UniMolRepr`` (fetches its checkpoint into the HF cache once)."""
        try:
            from unimol_tools import UniMolRepr  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "unimol_v1 needs `pip install unimol_tools` (it pulls its checkpoint "
                "from the dptech/Uni-Mol-Models HF repo on first use)"
            ) from e

        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._repr = UniMolRepr(
            data_type="molecule",
            remove_hs=False,
            batch_size=batch_size,
            use_cuda=self.device.startswith("cuda"),
        )
        self.dim = 512

    def check(self, smiles: str) -> Chem.Mol:
        """Parsable AND a real 3D conformer under unimol_tools' own ETKDG call
        (seed 42, fast mode); raises when it would fall back to 2D or zero
        coordinates.
        """
        from unimol_tools.data.conformer import (  # type: ignore[import-untyped]
            inner_smi2coords,
        )

        mol = parse_smiles(smiles)
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        _atoms, coords, _mol = inner_smi2coords(
            canonical, seed=self._SEED, mode=self._MODE, remove_hs=False
        )
        coords = np.asarray(coords)
        if (coords == 0).all():
            raise ValueError(
                f"unimol_v1: no conformer (all-zero coordinates) for {canonical!r}"
            )
        if (coords[:, 2] == 0).all():
            raise ValueError(
                f"unimol_v1: ETKDG failed, only 2D coordinates for {canonical!r}"
            )
        return mol

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        reps = self._repr.get_repr(canonical, return_atomic_reprs=False)
        if len(reps) != len(canonical):
            raise RuntimeError(
                f"unimol_v1 returned {len(reps)} representations for "
                f"{len(canonical)} molecules"
            )
        return np.stack([np.asarray(r, dtype=np.float32) for r in reps])


# --------------------------------------------------------------------------- MolE


@register
class MolEStatic(MoleculeEncoder):
    """MolE static representation (Olayo-Alarcon et al. 2025): the 1000-dim
    concatenation of the five GIN layers' add-pooled node embeddings from the
    ``gin_concat_R1000_E8000_lambda0.0001`` checkpoint (Zenodo 10803099), before the
    8000-dim Barlow-Twins projection.
    """

    name = "mole_static"

    def __init__(self, device: str | None = None, batch_size: int = 64) -> None:
        """Load the verified Zenodo checkpoint into the vendored ``GINet``."""
        import torch

        from torchcell.molecule.third_party.mole import GINet
        from torchcell.molecule.weights import ensure_weights

        self.device = resolve_device(device)
        self.batch_size = batch_size
        # Hyperparameters of ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints/config.yaml
        self._model = GINet(
            num_layer=5, emb_dim=200, feat_dim=8000, drop_ratio=0.0, pool="add"
        )
        state = torch.load(
            ensure_weights("mole"), map_location="cpu", weights_only=True
        )
        missing, unexpected = self._model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"mole_static checkpoint lacks {sorted(missing)}")
        if unexpected:
            raise RuntimeError(
                f"mole_static checkpoint has unknown keys {sorted(unexpected)}"
            )
        self._model = self._model.to(self.device).eval()
        self.dim = self._model.concat_dim

    def _encode(
        self, mols: list[Chem.Mol], canonical: list[str]
    ) -> NDArray[np.float32]:
        import torch
        from torch_geometric.data import Batch

        from torchcell.molecule.third_party.mole import mol_to_mole_graph

        graphs = [mol_to_mole_graph(m) for m in mols]
        chunks = []
        for start in range(0, len(graphs), self.batch_size):
            batch = Batch.from_data_list(graphs[start : start + self.batch_size]).to(
                self.device
            )
            with torch.inference_mode():
                h, _ = self._model(batch)
            chunks.append(h.float().cpu().numpy())
        return np.concatenate(chunks)
