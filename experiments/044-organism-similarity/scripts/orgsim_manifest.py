# experiments/044-organism-similarity/scripts/orgsim_manifest.py
# [[experiments.044-organism-similarity.scripts.orgsim_manifest]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/044-organism-similarity/scripts/orgsim_manifest.py

"""Pydantic records for the 044 organism manifest (``results/organisms.json``).

One ``OrganismRecord`` per reference organism of the plan table, including the ones read
from the genomes tier, the SD108 clone and the ones with no annotated proteome (the last
carry ``proteome=None`` and an ``absence_reason``). ``01_organisms.py`` writes the
retrieval half; ``02_proteome_prep.py`` fills ``prep`` (one protein per gene).
"""

from __future__ import annotations

import hashlib
import os
import os.path as osp
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
EXP_DIR = osp.join(EXPERIMENT_ROOT, "044-organism-similarity")
RESULTS_DIR = osp.join(EXP_DIR, "results")
CONF_PATH = osp.join(EXP_DIR, "conf", "organisms.yaml")
MANIFEST_PATH = osp.join(RESULTS_DIR, "organisms.json")
WORK_DIR = osp.join(DATA_ROOT, "data", "organism_similarity")
PROTEOME_DIR = osp.join(WORK_DIR, "proteomes")
ORGSIM_BIN = "/scratch/projects/torchcell-scratch/envs/orgsim/bin"
DATASETS_BIN = osp.join(ORGSIM_BIN, "datasets")


def sha256_file(path: str) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class FileRecord(BaseModel):
    """One file the pipeline read, pinned by sha256."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    bytes: int


class ProteomeSource(BaseModel):
    """Where an organism's protein FASTA came from and how it was retrieved."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["tier", "ncbi"]
    accession: str | None = None
    assembly_name: str | None = None
    annotation_provider: str | None = None
    tier_set: str | None = None
    source_url: str | None = None
    retrieval_method: str
    retrieval_command: str
    datasets_version: str | None = None
    retrieved_at: str
    protein_fasta: FileRecord
    gff: FileRecord | None = None
    n_proteins_raw: int


class ProteomePrep(BaseModel):
    """The one-protein-per-gene proteome every comparison reads."""

    model_config = ConfigDict(extra="forbid")

    rule: str
    n_proteins_raw: int
    n_proteins_prepped: int
    n_proteins_without_gene: int
    output: FileRecord


class OrganismRecord(BaseModel):
    """One plan-table organism."""

    model_config = ConfigDict(extra="forbid")

    key: str
    row: int
    display: str
    organism_name: str = Field(description="NCBI organism name of the chosen assembly.")
    taxid: int
    strain: str | None
    preferred_strain: str
    groups: list[str]
    justification: str
    reference_accession: str | None = Field(
        default=None,
        description="What `datasets summary genome taxon --reference` returned.",
    )
    accession_choice: str | None = None
    proteome: ProteomeSource | None
    absence_reason: str | None = None
    prep: ProteomePrep | None = None


class OrganismManifest(BaseModel):
    """``results/organisms.json``."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    datasets_version: str
    organisms: list[OrganismRecord]

    def present(self) -> list[OrganismRecord]:
        """Organisms with a proteome, in table order."""
        return [o for o in self.organisms if o.proteome is not None]


def load_manifest() -> OrganismManifest:
    """Validate and return ``results/organisms.json``."""
    return OrganismManifest.model_validate_json(Path(MANIFEST_PATH).read_text())


def write_manifest(manifest: OrganismManifest) -> None:
    """Write ``results/organisms.json``."""
    Path(MANIFEST_PATH).write_text(manifest.model_dump_json(indent=2) + "\n")
