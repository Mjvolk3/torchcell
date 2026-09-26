# torchcell/molecule/weights.py
# [[torchcell.molecule.weights]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/molecule/weights.py
"""Pinned, sha256-verified weight files for the molecule encoders that are not on the
Hugging Face hub (Mol2Vec's gensim pickle, MolE's Zenodo checkpoint).

Each file lives at ``$DATA_ROOT/data/torchcell/molecule_encoders/<name>/<filename>``
beside a ``manifest.json`` (:class:`WeightsManifest`) recording where the bytes came
from and their sha256. The sha256 pinned in :data:`WEIGHTS` is the canonical anchor:
a fresh download whose hash differs is upstream drift and raises; a file on disk whose
hash differs from its manifest raises; a file with no manifest raises, because a file
nobody recorded retrieving is not a provenance record. Hugging Face models are not
handled here; they cache in the default HF hub cache under their revision hash.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import RetrievalMethod


class WeightsSpec(BaseModel):
    """Where a weight file comes from and the sha256 it must have."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Subdirectory under molecule_encoders/.")
    filename: str
    source_url: str
    retrieval_method: RetrievalMethod
    sha256: str = Field(description="Pinned sha256 of the bytes we validated against.")


class WeightsManifest(BaseModel):
    """Provenance record written beside a downloaded weight file."""

    model_config = ConfigDict(extra="forbid")

    name: str
    source_url: str
    retrieval_method: RetrievalMethod
    retrieval_command: str
    sha256: str
    retrieved_at: str
    bytes: int


WEIGHTS: dict[str, WeightsSpec] = {
    "mol2vec": WeightsSpec(
        name="mol2vec",
        filename="model_300dim.pkl",
        source_url=(
            "https://github.com/samoturk/mol2vec/raw/master/examples/models/model_300dim.pkl"
        ),
        retrieval_method=RetrievalMethod.direct_url,
        sha256="62934b4ec245716c1e97ca95483434e0b66bfe2efc499f5123e17e6eb0c2331f",
    ),
    "mole": WeightsSpec(
        name="mole",
        filename="model.pth",
        source_url="https://zenodo.org/records/10803099/files/model.pth?download=1",
        retrieval_method=RetrievalMethod.zenodo,
        # md5 3d084daa5f75a0bde23caf7a24a43f8d matches the Zenodo file listing.
        sha256="2d324644c5f43e7be6734a9cd7a7966f975bfcc113610c13be897d11674defd8",
    ),
}


def sha256_of(path: Path) -> str:
    """Streamed sha256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def encoders_data_dir() -> Path:
    """``$DATA_ROOT/data/torchcell/molecule_encoders`` from the repo ``.env``."""
    load_dotenv()
    return Path(os.environ["DATA_ROOT"]) / "data" / "torchcell" / "molecule_encoders"


def retrieval_command(spec: WeightsSpec) -> str:
    """The exact shell command that fetches the file (recorded in the manifest)."""
    return f'curl -L -sS -o {spec.filename} "{spec.source_url}"'


def write_manifest(spec: WeightsSpec, path: Path) -> WeightsManifest:
    """Hash ``path``, check it against the pin, and write ``manifest.json`` beside it."""
    digest = sha256_of(path)
    if digest != spec.sha256:
        raise ValueError(
            f"{spec.name}: {path} has sha256 {digest}, pinned {spec.sha256}; upstream "
            "content changed or the download is corrupt"
        )
    manifest = WeightsManifest(
        name=spec.name,
        source_url=spec.source_url,
        retrieval_method=spec.retrieval_method,
        retrieval_command=retrieval_command(spec),
        sha256=digest,
        retrieved_at=datetime.now(UTC).date().isoformat(),
        bytes=path.stat().st_size,
    )
    (path.parent / "manifest.json").write_text(
        manifest.model_dump_json(indent=2) + "\n"
    )
    return manifest


def ensure_weights(name: str) -> Path:
    """Path to a verified weight file, downloading it once with its manifest.

    A present file is re-hashed against its manifest and the pin on every call
    (77 MB to 800 MB, well under a second per GB on local disk).
    """
    spec = WEIGHTS[name]
    target_dir = encoders_data_dir() / spec.name
    path = target_dir / spec.filename
    manifest_path = target_dir / "manifest.json"
    if not path.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(retrieval_command(spec), shell=True, check=True, cwd=target_dir)
        write_manifest(spec, path)
        return path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{path} exists without a manifest.json; it was not retrieved by "
            "ensure_weights. Move it away and let ensure_weights fetch it, or call "
            "write_manifest(spec, path) if the bytes are known to be the pinned ones."
        )
    manifest = WeightsManifest.model_validate_json(manifest_path.read_text())
    digest = sha256_of(path)
    if digest != manifest.sha256 or digest != spec.sha256:
        raise ValueError(
            f"{spec.name}: on-disk sha256 {digest} != manifest {manifest.sha256} "
            f"or pin {spec.sha256}"
        )
    return path
