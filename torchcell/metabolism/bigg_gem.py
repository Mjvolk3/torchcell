# torchcell/metabolism/bigg_gem
# [[torchcell.metabolism.bigg_gem]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/bigg_gem
# Test file: tests/torchcell/metabolism/test_bigg_gem.py
"""A BiGG genome-scale model, pinned by sha256, with the graph views the CGT reads.

``YeastGEM`` is the same idea for one organism: it downloads a GitHub release zip
of the yeast-GEM repository and reads the SBML inside. A BiGG model is a single
JSON at a stable URL, and there is more than one of them per organism once a
heterologous pathway is added, so this class is parameterized by model id and
reads from a pinned mirror rather than downloading on construction.

The mirror is ``{gem_root}/{model_id}/`` holding the model file, a
``manifest.json`` in the shape the genome tier uses, and ``SHA256SUMS``. The
model is verified against the manifest every time it is loaded, so a build
cannot silently consume different bytes than the one it records. That is the
pinned-bytes rule applied to the metabolic model, and it matters here more than
usual because one of the two P. putida models is not published anywhere: the
isoprenol-augmented model came from the authors directly, so our deposit is the
only copy and its hash is the only identity it has.

What this exposes, matching ``YeastGEM`` so the dataset code is organism-blind:

* ``model``        the cobra model
* ``bipartite_graph``  the reaction-metabolite ``nx.DiGraph``, built by
  :func:`torchcell.metabolism.bipartite.cobra_bipartite_graph`
* ``gene_set``     every gene id in the model's GPRs

``growth_reaction_ids`` defaults to the reactions carrying a nonzero objective
coefficient, which is how a GEM declares its own biomass reaction, so the
``Growth`` subsystem label that ``to_cell_data`` tests for is populated without
anybody naming ``BIOMASS_KT2440_WT3`` in a config. A BiGG model's own subsystem
string for that reaction is something else entirely ("Biomass and maintenance
functions" in iJN1463), which is why reading the objective rather than trusting
the subsystem is the portable choice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import os.path as osp
from datetime import UTC, datetime
from typing import Any, Final, cast

import cobra
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.bipartite import cobra_bipartite_graph
from torchcell.sequence import GeneSet

# Loaded at import, as ``yeast_GEM`` does, so ``DEFAULT_GEM_ROOT`` resolves to the
# machine's data root rather than to a relative path under the caller's cwd.
load_dotenv()

log = logging.getLogger(__name__)

BIGG_MODEL_BASE: Final = "https://bigg.ucsd.edu/static/models"

#: Default mirror root, anchored to DATA_ROOT the way the genome tier is. With
#: DATA_ROOT unset this becomes a relative path rather than a wrong absolute one.
DEFAULT_GEM_ROOT: Final = osp.join(os.environ.get("DATA_ROOT", ""), "torchcell-gems")


class GemFileRecord(BaseModel):
    """One mirrored model file, pinned by sha256."""

    path: str
    role: str
    bytes: int
    sha256: str
    source: str
    retrieval_method: str
    retrieval_command: str
    retrieved_at: str


class GemManifest(BaseModel):
    """The manifest written beside a mirrored genome-scale model.

    ``extra="forbid"`` because the provenance fields a caller supplies through
    ``deposit(extra=...)`` are exactly the ones worth misspelling:
    ``gene_namespace``, ``genome_ref``, ``published_filename``. Ignoring an
    unknown key loses provenance with no signal, which is the opposite of what a
    manifest is for.
    """

    model_config = ConfigDict(extra="forbid")

    version: int = 1
    model_id: str
    organism: str
    source: str
    source_url: str
    published_filename: str | None = None
    published_model_name: str | None = None
    genome_ref: str | None = None
    last_updated: str | None = None
    gene_namespace: str | None = None
    reaction_count: int | None = None
    metabolite_count: int | None = None
    gene_count: int | None = None
    files: list[GemFileRecord] = Field(default_factory=list)
    provenance_complete: bool = True
    created_at: str
    notes: str = ""


def _sha256_file(path: str) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def bigg_model_url(model_id: str) -> str:
    """The canonical BiGG download URL for a model id."""
    return f"{BIGG_MODEL_BASE}/{model_id}.json"


class ModelHashMismatchError(ValueError):
    """The model file on disk no longer hashes to what its manifest pins."""


class BiggGEM:
    """A pinned BiGG-format genome-scale model and its graph views."""

    def __init__(
        self,
        model_id: str,
        gem_root: str = DEFAULT_GEM_ROOT,
        induced_gene_set: GeneSet | None = None,
        growth_reaction_ids: tuple[str, ...] | None = None,
        verify: bool = True,
    ) -> None:
        """Bind to a mirrored model without loading it.

        Args:
            model_id: The mirror directory and file stem, for example
                ``iJN1463`` or ``pputida_isoprenol``.
            gem_root: The mirror root holding one directory per model.
            induced_gene_set: Restrict GPR terms to these genes when building the
                bipartite graph. A term with no gene in the set is dropped.
            growth_reaction_ids: Reactions to label with the ``Growth``
                subsystem. ``None`` reads the model's objective instead.
            verify: Re-hash the model file against the manifest on load.
        """
        self.model_id = model_id
        self.gem_root = gem_root
        self.induced_gene_set = induced_gene_set
        self._growth_reaction_ids = growth_reaction_ids
        self.verify = verify
        self._model: cobra.Model | None = None
        self._bipartite_graph: nx.DiGraph | None = None
        self._gene_set: GeneSet | None = None

    def __repr__(self) -> str:
        """Show the model id and its mirror directory."""
        return f"BiggGEM(model_id={self.model_id!r}, dir={self.model_dir!r})"

    @property
    def model_dir(self) -> str:
        """The directory holding this model and its manifest."""
        return osp.join(self.gem_root, self.model_id)

    @property
    def model_path(self) -> str:
        """The mirrored model file."""
        return osp.join(self.model_dir, f"{self.model_id}.json")

    @property
    def manifest_path(self) -> str:
        """Where the manifest is written."""
        return osp.join(self.model_dir, "manifest.json")

    def manifest(self) -> GemManifest:
        """Read the model's manifest."""
        with open(self.manifest_path) as handle:
            return GemManifest.model_validate_json(handle.read())

    def _verify_model_file(self) -> None:
        """Raise if the model file does not hash to what the manifest pins."""
        manifest = self.manifest()
        expected = {
            record.path: record.sha256
            for record in manifest.files
            if record.role == "model"
        }
        name = osp.basename(self.model_path)
        if name not in expected:
            raise ModelHashMismatchError(
                f"{self.manifest_path} carries no model record for {name}"
            )
        observed = _sha256_file(self.model_path)
        if observed != expected[name]:
            raise ModelHashMismatchError(
                f"{self.model_path} hashes to {observed}, manifest pins "
                f"{expected[name]}"
            )

    @property
    def model(self) -> cobra.Model:
        """The cobra model, loaded once and verified against the manifest."""
        if self._model is None:
            if self.verify:
                self._verify_model_file()
            self._model = cobra.io.load_json_model(self.model_path)
        return self._model

    @property
    def growth_reaction_ids(self) -> tuple[str, ...]:
        """The reactions treated as growth, from the objective when unset."""
        if self._growth_reaction_ids is not None:
            return self._growth_reaction_ids
        return tuple(
            reaction.id
            for reaction in self.model.reactions
            if reaction.objective_coefficient != 0
        )

    @property
    def bipartite_graph(self) -> nx.DiGraph:
        """The reaction-metabolite bipartite graph ``to_cell_data`` consumes."""
        if self._bipartite_graph is None:
            self._bipartite_graph = cobra_bipartite_graph(
                self.model,
                induced_gene_set=(
                    set(self.induced_gene_set)
                    if self.induced_gene_set is not None
                    else None
                ),
                growth_reaction_ids=self.growth_reaction_ids,
            )
        return self._bipartite_graph

    @property
    def gene_set(self) -> GeneSet:
        """Every gene id the model's GPRs reference."""
        if self._gene_set is None:
            self._gene_set = GeneSet({gene.id for gene in self.model.genes})
        return self._gene_set

    def reaction_ids_producing(self, metabolite_id: str) -> list[str]:
        """Reaction ids that produce a metabolite, for naming a product route."""
        metabolite = self.model.metabolites.get_by_id(metabolite_id)
        return [
            reaction.id
            for reaction in metabolite.reactions
            if reaction.metabolites[metabolite] > 0
        ]

    def deposit(
        self,
        source_path: str,
        organism: str,
        source: str,
        source_url: str,
        retrieval_method: str,
        retrieval_command: str,
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> GemManifest:
        """Copy a model file into the mirror and write its manifest.

        The file is read, hashed and written rather than moved, so the source is
        left untouched. An existing deposit with the same hash is left alone; a
        different hash raises, because overwriting a pinned artifact in place is
        how a build silently changes what it consumed.
        """
        os.makedirs(self.model_dir, exist_ok=True)
        with open(source_path, "rb") as handle:
            payload = handle.read()
        digest = hashlib.sha256(payload).hexdigest()

        if osp.exists(self.model_path):
            existing = _sha256_file(self.model_path)
            if existing != digest:
                raise ModelHashMismatchError(
                    f"{self.model_path} already holds different bytes "
                    f"({existing} on disk, {digest} offered); deposit a new "
                    f"model_id rather than overwriting a pinned artifact"
                )
        else:
            with open(self.model_path, "wb") as handle:
                handle.write(payload)

        model = cobra.io.load_json_model(self.model_path)
        manifest = GemManifest(
            model_id=self.model_id,
            organism=organism,
            source=source,
            source_url=source_url,
            reaction_count=len(model.reactions),
            metabolite_count=len(model.metabolites),
            gene_count=len(model.genes),
            files=[
                GemFileRecord(
                    path=osp.basename(self.model_path),
                    role="model",
                    bytes=len(payload),
                    sha256=digest,
                    source=source_url,
                    retrieval_method=retrieval_method,
                    retrieval_command=retrieval_command,
                    retrieved_at=_now(),
                )
            ],
            created_at=_now(),
            notes=notes,
            **(extra or {}),
        )
        with open(self.manifest_path, "w") as handle:
            handle.write(manifest.model_dump_json(indent=1))
        with open(osp.join(self.model_dir, "SHA256SUMS"), "w") as handle:
            handle.write(f"{digest}  {osp.basename(self.model_path)}\n")
        return manifest

    def summary(self) -> dict[str, Any]:
        """Counts and growth reactions, for a build record."""
        return {
            "model_id": self.model_id,
            "model_path": self.model_path,
            "sha256": _sha256_file(self.model_path),
            "reactions": len(self.model.reactions),
            "metabolites": len(self.model.metabolites),
            "genes": len(self.model.genes),
            "growth_reaction_ids": list(self.growth_reaction_ids),
        }


def load_json_manifest(path: str) -> dict[str, Any]:
    """Read a manifest file as a plain dict, for reporting."""
    with open(path) as handle:
        return cast("dict[str, Any]", json.load(handle))
