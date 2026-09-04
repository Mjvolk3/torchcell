# torchcell/data/model_mirror.py
# [[torchcell.data.model_mirror]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/model_mirror.py
# Test file: tests/torchcell/data/test_model_mirror.py

r"""A sha256-pinned local mirror of a third-party model, and the record of how it got here.

WHY A MIRROR RATHER THAN A DOWNLOAD AT USE TIME
------------------------------------------------
A kinetic predictor is a dependency of every value it emits. If a run pulls weights from a
Zenodo record or a git clone at the moment it needs them, the numbers in a dataset cannot
be traced to particular bytes: the record can be revised, a repository can be
force-pushed, and neither event leaves a mark on the values already written. Pinning the
bytes here is what lets a `k_cat` table say which weights produced it.

This is the same contract the paper mirror uses, applied to model weights instead of
PDFs: the stored artifact plus its ``sha256`` is canonical, and the URL is historical
retrieval metadata. On rebuild the recorded command is re-run and the hash re-verified. A
mismatch is a detection, not a fallback -- it means upstream changed, and it creates a new
versioned mirror rather than overwriting the one a dataset was built against.

WHERE MIRRORS LIVE
------------------
``$DATA_ROOT/models/<family>/<name>/``, alongside the existing ``mineru`` mirror, which is
the established location for a third-party model on this machine. Two things are
deliberately kept apart:

* ``$DATA_ROOT/models/...`` -- **the model.** Weights, vocabularies, dictionaries. Written
  once, verified thereafter, never written by a training run.
* ``$DATA_ROOT/data/torchcell/...`` -- **what the model produced.** Dataset builds, which
  are regenerable from a mirror plus code and carry the mirror's hash to say which model
  they came from.

Deleting a build is routine. Deleting a mirror invalidates the provenance of every build
made from it, so a mirror is only ever added.
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import RetrievalMethod

MIRROR_MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"

# Read in 1 MiB blocks: large enough that the syscall overhead disappears against a
# multi-GB weight file, small enough that hashing never holds one in memory.
_HASH_BLOCK_BYTES = 1024 * 1024


def sha256_file(path: str) -> str:
    """Return the hex sha256 of a file, streamed so a 12 GB archive never lands in RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(_HASH_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


class MirroredFile(BaseModel):
    """One file inside a mirror, pinned by hash.

    ``role`` separates the files that ARE the model from the files that merely came in the
    same archive. Only ``weights`` and ``vocabulary`` roles are required to reproduce a
    prediction; a ``training_data`` file is kept when it arrived with the release but is
    not an input to inference, and saying so here stops a later reader from assuming a
    600 MB BRENDA dump is a weight file.
    """

    model_config = ConfigDict(extra="forbid")

    rel_path: str = Field(description="Path relative to the mirror root.")
    sha256: str
    size_bytes: int
    role: str = Field(
        description="weights | vocabulary | dictionary | training_data | source | archive"
    )
    note: str | None = None


class ModelMirror(BaseModel):
    """Provenance and integrity record for one mirrored third-party model.

    Written as ``manifest.json`` at the mirror root. ``emits`` is here because it is the
    question a caller actually asks -- a predictor that returns only ``k_cat`` cannot be
    used to fill a ``K_M`` column, and discovering that by reading inference code is how a
    silently empty column happens.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = Field(default=MIRROR_MANIFEST_VERSION)
    name: str = Field(description="Mirror key, e.g. 'dlkcat'. Stable; used in paths.")
    family: str = Field(default="kinetics", description="Mirror group, the parent dir.")
    display_name: str = Field(description="The name the authors use, e.g. 'DLKcat'.")
    citation_key: str | None = Field(
        default=None,
        description="Join key into the paper mirror, when the paper is held.",
    )
    emits: list[str] = Field(
        default_factory=list, description="Parameters this model predicts: k_cat, K_M."
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="What one prediction consumes: sequence, substrate_smiles, "
        "reaction_smiles, structure.",
    )
    method: RetrievalMethod
    source_url: str
    retrieval_command: str = Field(
        description="The exact command that reproduces this mirror from the source."
    )
    retrieved_at: str
    revision: str | None = Field(
        default=None,
        description="Git commit or Zenodo record version, when there is one.",
    )
    files: list[MirroredFile] = Field(default_factory=list)
    runnable: bool = Field(
        default=True,
        description="False when the mirror is complete but inference is still blocked "
        "(a missing dependency, weights the authors never released). The blocker goes in "
        "``blocked_by`` so a coverage gap is never mistaken for a modeling result.",
    )
    blocked_by: str | None = None
    notes: str | None = None

    def path(self, data_root: str) -> str:
        """Absolute path to this mirror's root directory."""
        return osp.join(data_root, "models", self.family, self.name)

    def weight_files(self) -> list[MirroredFile]:
        """The files that actually parameterize the model."""
        return [
            f for f in self.files if f.role in ("weights", "vocabulary", "dictionary")
        ]

    def verify(self, data_root: str) -> list[str]:
        """Re-hash every pinned file and return the paths that no longer match.

        An empty list means the mirror is intact. A non-empty one is a hard stop, not a
        warning: a changed weight file means predictions made now do not match
        predictions already written, and the dataset built from it is no longer
        attributable.
        """
        root = self.path(data_root)
        broken: list[str] = []
        for record in self.files:
            target = osp.join(root, record.rel_path)
            if not osp.exists(target):
                broken.append(f"{record.rel_path}: absent")
                continue
            actual = sha256_file(target)
            if actual != record.sha256:
                broken.append(
                    f"{record.rel_path}: sha256 {actual[:12]} != pinned {record.sha256[:12]}"
                )
        return broken


def scan_files(
    root: str, roles: dict[str, str], skip_dirs: frozenset[str] = frozenset({".git"})
) -> list[MirroredFile]:
    """Hash every file under ``root`` and label it via a ``rel_path`` prefix to role map.

    ``roles`` is matched by longest prefix so a specific file can override the directory
    it sits in. An unmatched path is ``source``, which is the honest default: it is in the
    mirror, and it is not claimed to be a weight.
    """
    records: list[MirroredFile] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        for filename in sorted(filenames):
            absolute = osp.join(dirpath, filename)
            rel_path = osp.relpath(absolute, root)
            if rel_path == MANIFEST_NAME:
                continue
            matches = [prefix for prefix in roles if rel_path.startswith(prefix)]
            role = roles[max(matches, key=len)] if matches else "source"
            records.append(
                MirroredFile(
                    rel_path=rel_path,
                    sha256=sha256_file(absolute),
                    size_bytes=osp.getsize(absolute),
                    role=role,
                )
            )
    return records


def write_manifest(mirror: ModelMirror, data_root: str) -> str:
    """Serialize a mirror manifest to its root and return the path written."""
    root = mirror.path(data_root)
    os.makedirs(root, exist_ok=True)
    target = osp.join(root, MANIFEST_NAME)
    with open(target, "w") as handle:
        handle.write(mirror.model_dump_json(indent=2))
    return target


def read_manifest(data_root: str, name: str, family: str = "kinetics") -> ModelMirror:
    """Load a mirror manifest by name."""
    target = osp.join(data_root, "models", family, name, MANIFEST_NAME)
    with open(target) as handle:
        return ModelMirror.model_validate_json(handle.read())


def list_mirrors(data_root: str, family: str = "kinetics") -> list[ModelMirror]:
    """Every mirror manifest under a family, sorted by name."""
    base = osp.join(data_root, "models", family)
    if not osp.isdir(base):
        return []
    found: list[ModelMirror] = []
    for name in sorted(os.listdir(base)):
        if osp.exists(osp.join(base, name, MANIFEST_NAME)):
            found.append(read_manifest(data_root, name, family))
    return found


def utc_now() -> str:
    """ISO-8601 UTC timestamp, the format every provenance record in the repo uses."""
    return datetime.now(UTC).isoformat()


def mirror_summary(mirrors: list[ModelMirror]) -> list[dict[str, Any]]:
    """A compact table of what is mirrored and what each one can actually produce."""
    return [
        {
            "name": m.name,
            "emits": ",".join(m.emits) or "-",
            "inputs": ",".join(m.inputs) or "-",
            "runnable": m.runnable,
            "blocked_by": m.blocked_by or "",
            "n_files": len(m.files),
            "weight_bytes": sum(f.size_bytes for f in m.weight_files()),
        }
        for m in mirrors
    ]
