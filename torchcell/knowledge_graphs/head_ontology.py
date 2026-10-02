# torchcell/knowledge_graphs/head_ontology.py
# [[torchcell.knowledge_graphs.head_ontology]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/head_ontology.py
# Test file: tests/torchcell/knowledge_graphs/test_head_ontology.py
"""The sha256-pinned local mirror of BioCypher's head ontology (Biolink 3.2.1).

BioCypher places every schema class under a head ontology. Its built-in default is
``https://github.com/biolink/biolink-model/raw/v3.2.1/biolink-model.owl.ttl``, fetched
by ``rdflib`` at every BioCypher construction, so a KG generation run depended on GitHub
answering (live rebuild job 3182 died on an HTTP 504 there, issue #619). The ontology
cannot be dropped (headless mode): the served nodes carry Biolink ancestor labels
(``torchcell.database.browser_style.ANCESTOR_LABELS``) and the schema's ``is_a`` targets
are Biolink classes. So the file is mirrored in the repo at
``biocypher/ontology/biolink-model-v3.2.1.owl.ttl`` with a provenance record beside it,
every BioCypher config points ``head_ontology.url`` at the local copy, and the build
scripts call :func:`verify_head_ontology` before constructing BioCypher.

``directory_setup`` copies the whole ``biocypher/`` directory into
``$BUILD_ROOT/database/biocypher``, which the generation container mounts at
``/var/lib/neo4j/biocypher``; the container configs therefore name
``/var/lib/neo4j/biocypher/ontology/...``. Host-side configs name the repo-relative
``biocypher/ontology/...``, which ``rdflib`` (and this check) resolve against the working
directory, the repo root.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from torchcell.literature.manifest import (
    ArtifactRecord,
    RetrievalMethod,
    RetrievalRecord,
    sha256_file,
)
from torchcell.literature.provenance import check_source

BIOLINK_VERSION = "3.2.1"
BIOLINK_SOURCE_URL = (
    "https://github.com/biolink/biolink-model/raw/v3.2.1/biolink-model.owl.ttl"
)
BIOLINK_ROOT_NODE = "entity"
ONTOLOGY_FILENAME = "biolink-model-v3.2.1.owl.ttl"
#: Repo-relative location of the mirrored file (and of every host-side config's url).
REPO_ONTOLOGY_PATH = f"biocypher/ontology/{ONTOLOGY_FILENAME}"
#: Where the generation container sees the repo's ``biocypher/`` directory.
CONTAINER_BIOCYPHER_DIR = "/var/lib/neo4j/biocypher"
CONTAINER_ONTOLOGY_PATH = f"{CONTAINER_BIOCYPHER_DIR}/ontology/{ONTOLOGY_FILENAME}"
RECORD_SUFFIX = ".provenance.json"
ROLE_ONTOLOGY = "ontology"


class HeadOntologyError(RuntimeError):
    """The configured head ontology is remote, missing, or not the pinned bytes."""


class HeadOntologyMirror(BaseModel):
    """Provenance of one mirrored head-ontology file, written beside it as JSON.

    ``file`` is the shared per-file ``ArtifactRecord`` (the record the literature and
    raw-data mirrors use); its ``retrieval`` names the versioned retriever
    (``torchcell.literature.retrieve.direct_url``) that ``check_source`` re-runs.
    ``retrieval_command`` is the exact shell command that produced the stored bytes.
    """

    model_config = ConfigDict(extra="forbid")

    ontology: str = Field(
        description='Ontology name and release, e.g. "biolink 3.2.1".'
    )
    root_node: str = Field(description="BioCypher head_ontology.root_node.")
    retrieval_command: str = Field(description="Exact command that fetched the bytes.")
    file: ArtifactRecord


def record_path(ontology_path: Path) -> Path:
    """The provenance JSON that sits beside an ontology file."""
    return ontology_path.with_name(ontology_path.name + RECORD_SUFFIX)


def load_mirror_record(ontology_path: Path) -> HeadOntologyMirror:
    """Read the provenance record beside ``ontology_path``; a missing record raises."""
    rec_path = record_path(ontology_path)
    if not rec_path.is_file():
        raise HeadOntologyError(f"no provenance record at {rec_path}")
    return HeadOntologyMirror.model_validate_json(rec_path.read_text())


def configured_head_ontology(biocypher_config_path: str | Path) -> dict[str, str]:
    """The ``biocypher.head_ontology`` mapping of a BioCypher config YAML.

    An absent or null entry raises: BioCypher would then fall back to its packaged
    default, the live GitHub URL.
    """
    config = yaml.safe_load(Path(biocypher_config_path).read_text())
    head = (config.get("biocypher") or {}).get("head_ontology")
    if not isinstance(head, dict) or "url" not in head:
        raise HeadOntologyError(
            f"{biocypher_config_path}: biocypher.head_ontology.url is not set, so "
            f"BioCypher would fetch {BIOLINK_SOURCE_URL} at run time; point it at "
            f"the mirrored {REPO_ONTOLOGY_PATH}"
        )
    return {str(k): str(v) for k, v in head.items()}


def verify_head_ontology(biocypher_config_path: str | Path) -> Path:
    """Check the config's head ontology is a local file with its pinned sha256.

    The url must be a filesystem path (no ``scheme://``); a relative path resolves
    against the working directory, as ``rdflib`` resolves it. The file's sha256 and
    size must equal the provenance record beside it. Returns the resolved path.
    """
    head = configured_head_ontology(biocypher_config_path)
    url = head["url"]
    if "://" in url:
        raise HeadOntologyError(
            f"{biocypher_config_path}: head_ontology.url {url!r} is a URL, not a "
            "local path; the build must not fetch the ontology at run time"
        )
    path = Path(url).resolve()
    if not path.is_file():
        raise HeadOntologyError(
            f"{biocypher_config_path}: head ontology {url!r} not found "
            f"(resolved to {path})"
        )
    record = load_mirror_record(path)
    if head.get("root_node") != record.root_node:
        raise HeadOntologyError(
            f"{biocypher_config_path}: head_ontology.root_node "
            f"{head.get('root_node')!r} != recorded {record.root_node!r}"
        )
    got = sha256_file(path)
    if got != record.file.sha256 or path.stat().st_size != record.file.bytes:
        raise HeadOntologyError(
            f"{path}: sha256 {got} ({path.stat().st_size} bytes) does not match the "
            f"pinned {record.file.sha256} ({record.file.bytes} bytes) in "
            f"{record_path(path)}"
        )
    return path


def build_mirror_record(
    ontology_path: Path, retrieved_at: str, retrieval_command: str, check: bool
) -> HeadOntologyMirror:
    """Record a freshly fetched Biolink file; ``check`` re-runs the retriever once."""
    sha = sha256_file(ontology_path)
    retrieval = RetrievalRecord(
        method=RetrievalMethod.direct_url,
        source_url=BIOLINK_SOURCE_URL,
        retriever="torchcell.literature.retrieve.direct_url",
        params={"url": BIOLINK_SOURCE_URL},
        sha256=sha,
        retrieved_at=retrieved_at,
    )
    if check:
        retrieval.last_check = check_source(
            retrieval, now=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        )
    return HeadOntologyMirror(
        ontology=f"biolink {BIOLINK_VERSION}",
        root_node=BIOLINK_ROOT_NODE,
        retrieval_command=retrieval_command,
        file=ArtifactRecord(
            path=ontology_path.name,
            role=ROLE_ONTOLOGY,
            bytes=ontology_path.stat().st_size,
            sha256=sha,
            source=BIOLINK_SOURCE_URL,
            retrieval=retrieval,
        ),
    )


def main(argv: list[str] | None = None) -> None:
    """CLI: ``record`` writes the provenance JSON, ``verify`` checks a config."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.knowledge_graphs.head_ontology"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record", help="write <file>.provenance.json beside the file")
    rec.add_argument("--file", default=REPO_ONTOLOGY_PATH)
    rec.add_argument("--retrieved-at", required=True, help="ISO UTC time of the fetch")
    rec.add_argument("--retrieval-command", required=True)
    rec.add_argument(
        "--check", action="store_true", help="re-run the retriever and record it"
    )
    ver = sub.add_parser("verify", help="check a BioCypher config's head ontology")
    ver.add_argument("config")
    args = parser.parse_args(argv)
    if args.cmd == "record":
        path = Path(args.file)
        mirror = build_mirror_record(
            path, args.retrieved_at, args.retrieval_command, args.check
        )
        record_path(path).write_text(mirror.model_dump_json(indent=2) + "\n")
        print(f"wrote {record_path(path)} sha256={mirror.file.sha256}")
    else:
        print(f"ok {verify_head_ontology(args.config)}")


if __name__ == "__main__":
    main()
