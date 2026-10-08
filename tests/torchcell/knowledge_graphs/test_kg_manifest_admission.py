# tests/torchcell/knowledge_graphs/test_kg_manifest_admission.py
# [[tests.torchcell.knowledge_graphs.test_kg_manifest_admission]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/knowledge_graphs/test_kg_manifest_admission.py
"""End-to-end admission, bootstrap, record and CLI paths of ``kg_manifest`` on a toy repo.

Fixture (``toy``): a miniature torchcell checkout under ``tmp_path`` holding exactly the
files the manifest reads.

- Schema surface: ``schema.py`` defines ``Media``, ``Solvent`` and ``Experiment`` (with
  a ``Media`` field) on ``pydant.ModelStrict``, so the reference graph is
  Experiment -> {Media, ModelStrict}, Media -> {ModelStrict}, Solvent -> {ModelStrict}.
- Two loaders: ``served_loader.py`` (``ServedDataset``) imports ``Media``, so its
  closure is {Media, ModelStrict}; ``toy_loader.py`` (``ToyDataset``) imports
  ``Experiment`` and ``Solvent``, so its closure is {Experiment, Media, ModelStrict,
  Solvent}: two symbols shared with the served dataset, two novel.
- One adapter module + conf per dataset (the adapter class names its conf in its own
  body, which is where the gate reads it), a toy ``cell_adapter.py`` whose table maps
  ``experiment (chunked)``, ``fitness phenotype (chunked)`` and ``genotype to experiment
  (chunked)``, a toy graph schema, the three value-surface files, and
  ``torchcell/__version__.py`` at ``1.2.0``; the stub git says the commit is tagged
  ``v1.2.0`` (``tag --points-at``, ``describe --tags --exact-match``).
- Dev LMDBs under ``data_root``: an empty ``processed/lmdb`` directory, the records the
  toy dataset classes read (``records.json``), and a fresh ``build_manifest.json``.

The two classes are registered in the real ``dataset_registry`` and
``dataset_adapter_map`` for the test only. Git is a PATH-stubbed script answering
``rev-parse`` (commit ``c0ffee12``), ``status`` (clean), ``show <ref>:<path>`` and
``ls-tree`` from the toy repo itself, so "the build commit" is the toy tree as first
written; edits made afterwards are drift. Neo4j is a scripted fake installed at
``neo4j.GraphDatabase.driver``; the clock is ``kg_manifest._now`` pinned to ``NOW``.

The baseline manifest is ``bootstrap_manifest`` of ``ServedDataset`` at that commit, so
every drift test starts from a manifest that matches the tree and changes one thing.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import torchcell.knowledge_graphs.kg_manifest as km
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.knowledge_graphs.dataset_adapter_map import dataset_adapter_map
from torchcell.provenance.build_manifest import BuildManifest
from torchcell.provenance.schema_deps import load_surface_from_sources

NOW = "2026-09-27T00:00:00+00:00"
COMMIT = "c0ffee12"
BUILT_AT = "2026-09-16T00:44:53+00:00"
BIOCYPHER_OUT = "2026-09-16_00-44-53"

SCHEMA_PY = """from torchcell.datamodels.pydant import ModelStrict


class Media(ModelStrict):
    name: str


class Solvent(ModelStrict):
    name: str


class Experiment(ModelStrict):
    media: Media
    fitness: float
"""
PYDANT_PY = """from pydantic import BaseModel


class ModelStrict(BaseModel):
    pass
"""
SCHEMA_YAML = """experiment:
    represented_as: node
    properties:
        serialized_data: str

fitness phenotype:
    represented_as: node
    properties:
        fitness: float
        serialized_data: str

phenotype member of:
    represented_as: edge
    source: [fitness phenotype]
    target: experiment
"""
CELL_ADAPTER_PY = '''class CellAdapter:
    def __init__(self, config):
        """doc"""
        self.config = config
        self.node_methods = [
            ("experiment (chunked)", self._experiment_node),
            ("fitness phenotype (chunked)", self._fitness_phenotype_node),
        ]
        self.edge_methods = [
            ("genotype to experiment (chunked)", self._genotype_to_experiment_edge),
        ]

    def get_nodes(self):
        return 1

    def _experiment_node(self, data, method_name):
        return data

    def _fitness_phenotype_node(self, data, method_name):
        return data["fitness"]

    def _genotype_to_experiment_edge(self, data, method_name):
        return (data, data)
'''
LOADER_TEMPLATE = """import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchcell.datamodels.schema import {imports}


class _Record:
    def __init__(self, payload):
        self.payload = payload

    def model_dump(self):
        return self.payload


class {cls}:
    def __init__(self, root: str = "data/torchcell/{slug}"):
        with open(os.path.join(root, "records.json")) as handle:
            self.items = json.load(handle)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]

    def transform_item(self, item):
        return {{"experiment": _Record(item)}}

    def close_lmdb(self):
        pass
"""
ADAPTER_TEMPLATE = """class {cls}:
    CONF = "{slug}_adapter.yaml"
"""
CONF_YAML = """cell_adapter:
  node_methods:
    - method_name: experiment (chunked)
    - method_name: fitness phenotype (chunked)
  edge_methods:
    - method_name: genotype to experiment (chunked)
"""
VALUE_FILES = {
    "torchcell/datamodels/media.py": "YPD = 'yeast extract peptone dextrose'\n",
    "torchcell/datamodels/compound_identity.py": "def resolved_compound():\n    pass\n",
    "torchcell/datamodels/compound_identity_table.json": "{}\n",
}
GIT_STUB = """#!/bin/bash
shift 2
case "$1" in
  rev-parse) echo "$STUB_COMMIT" ;;
  status) printf '%s' "$STUB_STATUS" ;;
  tag) [ -n "$STUB_TAG" ] && echo "$STUB_TAG"; exit 0 ;;
  describe) if [ -n "$STUB_TAG" ]; then echo "$STUB_TAG"; else exit 128; fi ;;
  show)
    rel="${2#*:}"
    if [ -f "$STUB_REF_DIR/$rel" ]; then
      cat "$STUB_REF_DIR/$rel"
    else
      echo "fatal: path '$rel' does not exist in '${2%%:*}'" >&2
      exit 128
    fi ;;
  ls-tree)
    cd "$STUB_REF_DIR" && find "$6" -type f 2>/dev/null | sort
    exit 0 ;;
esac
"""
SERVED_RECORDS = [{"s": 1}, {"s": 2}, {"s": 3}]
TOY_RECORDS = [{"t": 1}, {"t": 2}]


def _id(payload: dict[str, Any]) -> str:
    """``experiment_node_id`` of a toy record: sha256 of its json dump."""
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _Toy:
    """The toy checkout, its dev tree, and the two registered classes."""

    def __init__(self, repo: Path, data_root: Path) -> None:
        self.repo = repo
        self.data_root = data_root
        self.classes: dict[str, type] = {}

    def write(self, rel: str, text: str) -> None:
        path = self.repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def read(self, rel: str) -> str:
        return (self.repo / rel).read_text(encoding="utf-8")

    def replace(self, rel: str, old: str, new: str) -> None:
        text = self.read(rel)
        assert old in text, (rel, old)
        self.write(rel, text.replace(old, new))

    def surface_fingerprints(self) -> dict[str, str]:
        """Contract fingerprints of the CURRENT toy surface (schema_deps, not km)."""
        return load_surface_from_sources(
            {
                "schema.py": self.read("torchcell/datamodels/schema.py"),
                "pydant.py": self.read("torchcell/datamodels/pydant.py"),
            }
        ).fingerprints

    def lmdb_root(self, slug: str) -> Path:
        return self.data_root / "data" / "torchcell" / slug

    def build_dev_lmdb(
        self, slug: str, cls: str, closure: dict[str, str], records: list[Any]
    ) -> None:
        root = self.lmdb_root(slug)
        (root / "processed" / "lmdb").mkdir(parents=True, exist_ok=True)
        (root / "preprocess").mkdir(parents=True, exist_ok=True)
        (root / "records.json").write_text(json.dumps(records), encoding="utf-8")
        manifest = BuildManifest(
            dataset_name=slug,
            loader_class=cls,
            loader_module=f"toy_{slug}_loader",
            surface_modules=["pydant.py", "schema.py"],
            closure=closure,
            built_at=BUILT_AT,
            hostname="gilahyper",
        )
        (root / "preprocess" / "build_manifest.json").write_text(
            manifest.model_dump_json(), encoding="utf-8"
        )


def _import(module_name: str, path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Toy:
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    t = _Toy(tmp_path.resolve() / "repo", tmp_path.resolve() / "dev")
    t.write("torchcell/datamodels/schema.py", SCHEMA_PY)
    t.write("torchcell/datamodels/pydant.py", PYDANT_PY)
    t.write(km.VERSION_RELPATH, '__version__ = "1.2.0"\n')
    t.write(km.SCHEMA_CONFIG_RELPATH, SCHEMA_YAML)
    t.write(km.CELL_ADAPTER_RELPATH, CELL_ADAPTER_PY)
    for rel, text in VALUE_FILES.items():
        t.write(rel, text)
    for slug, cls, imports in (
        ("served", "ServedDataset", "Media"),
        ("toy", "ToyDataset", "Experiment, Solvent"),
    ):
        t.write(
            f"torchcell/datasets/{slug}_loader.py",
            LOADER_TEMPLATE.format(imports=imports, cls=cls, slug=slug),
        )
        adapter_cls = cls.replace("Dataset", "Adapter")
        t.write(
            f"torchcell/adapters/{slug}_adapter.py",
            ADAPTER_TEMPLATE.format(slug=slug, cls=adapter_cls),
        )
        t.write(f"torchcell/adapters/conf/{slug}_adapter.yaml", CONF_YAML)
        loader = _import(
            f"toy_{slug}_loader",
            t.repo / f"torchcell/datasets/{slug}_loader.py",
            monkeypatch,
        )
        adapter = _import(
            f"toy_{slug}_adapter",
            t.repo / f"torchcell/adapters/{slug}_adapter.py",
            monkeypatch,
        )
        dataset_cls = getattr(loader, cls)
        t.classes[cls] = dataset_cls
        monkeypatch.setitem(dataset_registry, cls, dataset_cls)
        monkeypatch.setitem(
            dataset_adapter_map, dataset_cls, getattr(adapter, adapter_cls)
        )

    fp = t.surface_fingerprints()
    t.build_dev_lmdb(
        "served",
        "ServedDataset",
        {k: fp[k] for k in ("Media", "ModelStrict")},
        SERVED_RECORDS,
    )
    t.build_dev_lmdb(
        "toy",
        "ToyDataset",
        {k: fp[k] for k in ("Experiment", "Media", "ModelStrict", "Solvent")},
        TOY_RECORDS,
    )

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    git = bin_dir / "git"
    git.write_text(GIT_STUB, encoding="utf-8")
    git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("STUB_REF_DIR", str(t.repo))
    monkeypatch.setenv("STUB_COMMIT", COMMIT)
    monkeypatch.setenv("STUB_STATUS", "")
    monkeypatch.setenv("STUB_TAG", "v1.2.0")
    monkeypatch.setattr(km, "_now", lambda: NOW)
    return t


def _bootstrap(toy: _Toy) -> km.KgBuildManifest:
    return km.bootstrap_manifest(
        repo_root=toy.repo,
        commit=COMMIT,
        dataset_classes=["ServedDataset"],
        n_experiments={"ServedDataset": 2},
        database="torchcell",
        store_host="gilahyper",
        neo4j_version="5.26.28",
        biocypher_version="0.5.43",
        biocypher_out=BIOCYPHER_OUT,
        built_at=BUILT_AT,
    )


def _admit(
    toy: _Toy, manifest: km.KgBuildManifest, name: str = "ToyDataset", **kwargs: Any
) -> km.AdmissionReport:
    return km.check_admission(manifest, toy.repo, name, toy.data_root, **kwargs)


# ------------------------------------------------------------------------ bootstrap


def test_bootstrap_reconstructs_the_full_build_from_the_commit(toy: _Toy) -> None:
    """Finding: the bootstrap's ``adapter_files`` include ``cell_adapter.py``
    (kg_manifest.py line 714 keeps every ``*_adapter.py``), while the recorder's
    ``adapter_file_relpaths`` excludes it (line 522); ``_adopt_current_surfaces``
    therefore drops that key at the first admission.

    One method fingerprint is recomputed independently of ``cell_adapter_surface``: the
    module hashes ``ast.unparse`` of the function with its docstring stripped, which for
    ``_fitness_phenotype_node`` is the two lines
    ``def _fitness_phenotype_node(self, data, method_name):`` and
    ``    return data['fitness']`` joined by a newline (``ast.unparse`` writes single
    quotes), sha256 14a4aec5...73a5ab.
    """
    fp = toy.surface_fingerprints()
    manifest = _bootstrap(toy)
    unparsed = (
        "def _fitness_phenotype_node(self, data, method_name):\n"
        "    return data['fitness']"
    )
    assert manifest.cell_adapter_methods["_fitness_phenotype_node"] == (
        hashlib.sha256(unparsed.encode("utf-8")).hexdigest()
    )
    assert manifest.cell_adapter_methods["_fitness_phenotype_node"] == (
        "14a4aec580c5ff101b9e48b5e645ecf4d67b6fbfe9ca01f6266363337973a5ab"
    )
    expected = km.KgBuildManifest(
        database="torchcell",
        store_host="gilahyper",
        neo4j_version="5.26.28",
        biocypher_version="0.5.43",
        torchcell_commit=COMMIT,
        torchcell_version="1.2.0",
        torchcell_tag="v1.2.0",
        graph_schema={
            "experiment": km.GraphSchemaEntry(
                kind="node", properties=["serialized_data"]
            ),
            "fitness phenotype": km.GraphSchemaEntry(
                kind="node", properties=["fitness", "serialized_data"]
            ),
            "phenotype member of": km.GraphSchemaEntry(
                kind="edge", source=["fitness phenotype"], target=["experiment"]
            ),
        },
        cell_adapter_methods=km.cell_adapter_surface(CELL_ADAPTER_PY)[0],
        cell_adapter_table={
            "experiment (chunked)": "_experiment_node",
            "fitness phenotype (chunked)": "_fitness_phenotype_node",
            "genotype to experiment (chunked)": "_genotype_to_experiment_edge",
        },
        adapter_files={
            "torchcell/adapters/cell_adapter.py": _sha(CELL_ADAPTER_PY),
            "torchcell/adapters/conf/served_adapter.yaml": _sha(CONF_YAML),
            "torchcell/adapters/conf/toy_adapter.yaml": _sha(CONF_YAML),
            "torchcell/adapters/served_adapter.py": _sha(
                ADAPTER_TEMPLATE.format(slug="served", cls="ServedAdapter")
            ),
            "torchcell/adapters/toy_adapter.py": _sha(
                ADAPTER_TEMPLATE.format(slug="toy", cls="ToyAdapter")
            ),
        },
        value_surface={rel: _sha(text) for rel, text in sorted(VALUE_FILES.items())},
        datasets={
            "ServedDataset": km.KgDatasetEntry(
                dataset_class="ServedDataset",
                loader_relpath="torchcell/datasets/served_loader.py",
                adapter_files=[
                    "torchcell/adapters/served_adapter.py",
                    "torchcell/adapters/conf/served_adapter.yaml",
                ],
                closure={"Media": fp["Media"], "ModelStrict": fp["ModelStrict"]},
                n_experiments=2,
                biocypher_out=BIOCYPHER_OUT,
                import_mode="full",
                admitted_at=BUILT_AT,
                torchcell_commit=COMMIT,
            )
        },
        events=[
            km.KgEvent(
                kind="bootstrap",
                at=NOW,
                torchcell_commit=COMMIT,
                torchcell_version="1.2.0",
                datasets=["ServedDataset"],
                biocypher_out=BIOCYPHER_OUT,
                note=f"reconstructed from the full build at {COMMIT}; built_at {BUILT_AT}",
            )
        ],
        created_at=NOW,
        updated_at=NOW,
    )
    assert manifest.model_dump() == expected.model_dump()
    assert set(km.cell_adapter_surface(CELL_ADAPTER_PY)[0]) == {
        "__init__",
        "get_nodes",
        "_experiment_node",
        "_fitness_phenotype_node",
        "_genotype_to_experiment_edge",
    }


def test_git_show_names_the_path_and_ref_it_could_not_read(toy: _Toy) -> None:
    """The stub's stderr travels into the FileNotFoundError."""
    message = (
        f"torchcell/missing.py does not exist at {COMMIT}: "
        f"fatal: path 'torchcell/missing.py' does not exist in '{COMMIT}'\n"
    )
    with pytest.raises(FileNotFoundError, match=f"^{re.escape(message)}$"):
        km._git_show(toy.repo, COMMIT, "torchcell/missing.py")


def test_value_surface_at_ref_skips_files_absent_at_the_commit(toy: _Toy) -> None:
    """With the JSON table deleted from the tree the stub serves, the ref surface has
    the two remaining files; the working tree agrees.
    """
    (toy.repo / "torchcell/datamodels/compound_identity_table.json").unlink()
    expected = {
        "torchcell/datamodels/compound_identity.py": _sha(
            VALUE_FILES["torchcell/datamodels/compound_identity.py"]
        ),
        "torchcell/datamodels/media.py": _sha(
            VALUE_FILES["torchcell/datamodels/media.py"]
        ),
    }
    assert km.value_surface_at_ref(toy.repo, COMMIT) == expected
    assert km.value_surface_in_worktree(toy.repo) == expected


def test_save_then_load_round_trips_and_stamps_updated_at(
    toy: _Toy, tmp_path: Path
) -> None:
    """``save_manifest`` creates the parent directory and rewrites ``updated_at``."""
    manifest = _bootstrap(toy)
    manifest.updated_at = "stale"
    path = tmp_path / "store" / "kg_manifest.json"
    km.save_manifest(manifest, path)
    assert manifest.updated_at == NOW
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(
        manifest.model_dump_json()
    )
    assert km.load_manifest(path) == manifest


# ------------------------------------------------------------------------ admission


def test_a_new_dataset_on_an_unchanged_tree_is_admissible(toy: _Toy) -> None:
    """The new closure is 4 symbols: Media and ModelStrict are shared with the served
    dataset, Experiment and Solvent are novel.
    """
    fp = toy.surface_fingerprints()
    report = _admit(toy, _bootstrap(toy))
    expected = km.AdmissionReport(
        dataset_class="ToyDataset",
        checked_at=NOW,
        torchcell_commit=COMMIT,
        torchcell_dirty=False,
        torchcell_version="1.2.0",
        served_commit=COMMIT,
        verdict="admissible",
        reasons=[],
        new_dataset_closure={
            k: fp[k] for k in ("Experiment", "Media", "ModelStrict", "Solvent")
        },
        novel_symbols=["Experiment", "Solvent"],
        shared_symbols={"Media": ["ServedDataset"], "ModelStrict": ["ServedDataset"]},
        stale_served=[],
        graph_schema_changed=[],
        graph_schema_added=[],
        adapter_drift=km.AdapterDrift(),
        adapter_methods_added=[],
        adapter_drift_acknowledged=None,
        value_surface_changed=[],
        value_surface_added=[],
        value_surface_recorded=True,
        value_drift_acknowledged=None,
        dev_lmdb_status="fresh",
        dev_lmdb_root=str(toy.lmdb_root("toy")),
        in_adapter_map=True,
        undeclared_phenotype_methods=[],
        served=False,
        superset=None,
    )
    assert report.model_dump() == expected.model_dump()
    assert km.format_report(report) == "\n".join(
        [
            "Admission check: ToyDataset  ->  ADMISSIBLE",
            f"  working tree {COMMIT} (dirty=False); served store built at {COMMIT}",
            f"  dev LMDB: fresh ({toy.lmdb_root('toy')})",
            "  closure: 4 symbols, 2 novel (Experiment, Solvent), 2 shared with "
            "served datasets",
            "  served datasets with schema drift: 0",
            "  graph schema: 0 changed, 0 added (-)",
            "  adapter methods added: -",
            "  adapter drift touching served datasets: none",
            "  value surface: unchanged (3 files)",
            "  served: no (new dataset)",
        ]
    )


def test_served_schema_drift_blocks_and_names_changed_and_new_symbols(
    toy: _Toy,
) -> None:
    """Media gains a field (fingerprint changes) and the served loader starts importing
    Solvent (a symbol new to its closure, appended after the changed ones).
    """
    manifest = _bootstrap(toy)
    toy.replace(
        "torchcell/datamodels/schema.py",
        "class Media(ModelStrict):\n    name: str\n",
        "class Media(ModelStrict):\n    name: str\n    state: str\n",
    )
    toy.replace(
        "torchcell/datasets/served_loader.py", "import Media", "import Media, Solvent"
    )
    report = _admit(toy, manifest)
    assert report.verdict == "blocked"
    assert report.stale_served == [
        km.ServedDrift(
            dataset_class="ServedDataset", changed_symbols=["Media", "Solvent"]
        )
    ]
    assert report.reasons[0] == (
        "served datasets whose schema closure changed since the build "
        "(full rebuild required): ServedDataset[Media, Solvent]"
    )
    # the toy dataset's own LMDB was built against the old Media: stale too
    assert report.dev_lmdb_status == "stale"
    assert report.reasons[1] == (
        f"dev-tree LMDB for ToyDataset is stale at {toy.lmdb_root('toy')}; "
        "build it with python -m torchcell.database.build_dataset_lmdb"
    )
    assert len(report.reasons) == 2
    assert "  served datasets with schema drift: 1" in km.format_report(report)


def test_graph_schema_change_blocks_but_an_added_class_does_not(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    toy.write(
        km.SCHEMA_CONFIG_RELPATH,
        SCHEMA_YAML.replace(
            "        fitness: float\n",
            "        fitness: float\n        fitness_se: float\n",
        )
        + "\ncalmorph phenotype:\n    represented_as: node\n",
    )
    report = _admit(toy, manifest)
    assert report.graph_schema_changed == ["fitness phenotype"]
    assert report.graph_schema_added == ["calmorph phenotype"]
    assert report.reasons == [
        "graph schema classes present in the served store changed "
        "(full rebuild required): fitness phenotype"
    ]
    assert (
        "  graph schema: 1 changed, 1 added (calmorph phenotype)"
        in km.format_report(report)
    )
    # an added class alone is additive
    toy.write(
        km.SCHEMA_CONFIG_RELPATH,
        SCHEMA_YAML + "\ncalmorph phenotype:\n    represented_as: node\n",
    )
    assert _admit(toy, manifest).verdict == "admissible"


def test_adapter_drift_blocks_per_kind_and_an_acknowledgment_admits(toy: _Toy) -> None:
    """A served method's body, a plumbing method, a served adapter file, a re-pointed
    table entry, and a newly added method (additive, only reported).
    """
    manifest = _bootstrap(toy)
    toy.write(
        km.CELL_ADAPTER_RELPATH,
        CELL_ADAPTER_PY.replace('return data["fitness"]', 'return data["fitness"] * 2')
        .replace("return 1", "return 2")
        .replace(
            '("genotype to experiment (chunked)", self._genotype_to_experiment_edge)',
            '("genotype to experiment (chunked)", self._experiment_node)',
        )
        + "\n    def _new_node(self, data, method_name):\n        return 0\n",
    )
    toy.write("torchcell/adapters/conf/served_adapter.yaml", CONF_YAML + "# edited\n")
    report = _admit(toy, manifest)
    assert report.adapter_drift == km.AdapterDrift(
        plumbing_methods=["get_nodes"],
        served_methods={
            "_fitness_phenotype_node": ["ServedDataset"],
            "table[genotype to experiment (chunked)]": ["ServedDataset"],
        },
        served_files={"torchcell/adapters/conf/served_adapter.yaml": ["ServedDataset"]},
    )
    assert report.adapter_methods_added == ["_new_node"]
    drift = (
        "plumbing: get_nodes; _fitness_phenotype_node (used by 1 served datasets); "
        "table[genotype to experiment (chunked)] (used by 1 served datasets); "
        "torchcell/adapters/conf/served_adapter.yaml (ServedDataset)"
    )
    assert report.reasons == [
        "adapter code serving existing datasets changed since the build; their node "
        "ids may have moved. Review the diff, then re-run with --ack-adapter-drift "
        f"'<why served ids are unchanged>'. Drift: {drift}"
    ]
    assert report.adapter_drift_acknowledged is None

    acked = _admit(toy, manifest, ack_adapter_drift="reviewed: ids unchanged")
    assert acked.verdict == "admissible"
    assert acked.adapter_drift_acknowledged == "reviewed: ids unchanged"
    text = km.format_report(acked)
    assert (
        f"  adapter drift touching served datasets: {drift} "
        "(acknowledged: reviewed: ids unchanged)"
    ) in text
    assert "  adapter methods added: _new_node" in text
    assert km._acknowledged_drift(acked) == [f"{drift}: reviewed: ids unchanged"]


def test_a_recorded_adapter_file_that_vanished_is_drift(toy: _Toy) -> None:
    """A recorded adapter file missing from the tree hashes to None, never its stored
    hash, so the served dataset that listed it is drifted on that path.
    """
    manifest = _bootstrap(toy)
    retired = "torchcell/adapters/conf/retired_adapter.yaml"
    manifest.datasets["ServedDataset"].adapter_files.append(retired)
    manifest.adapter_files[retired] = _sha("retired")
    drift, added = km.adapter_drift_against(manifest, toy.repo)
    assert drift == km.AdapterDrift(served_files={retired: ["ServedDataset"]})
    assert added == []


def test_a_mis_recorded_conf_still_leaves_the_bound_conf_watched(toy: _Toy) -> None:
    """Issue #743 against a manifest written before the fix.

    Such a manifest recorded another class's conf for the dataset. An edit to the conf
    the dataset's adapter class really binds must still block (it was missed before the
    fix: a wrong ADMISSIBLE), and the recorded conf stays watched as well, so nothing the
    old manifest promised to watch is dropped.
    """
    manifest = _bootstrap(toy)
    entry = manifest.datasets["ServedDataset"]
    assert entry.adapter_files == [
        "torchcell/adapters/served_adapter.py",
        "torchcell/adapters/conf/served_adapter.yaml",
    ]
    entry.adapter_files[1] = "torchcell/adapters/conf/toy_adapter.yaml"
    toy.write("torchcell/adapters/conf/served_adapter.yaml", CONF_YAML + "# edited\n")
    drift, _ = km.adapter_drift_against(manifest, toy.repo)
    assert drift == km.AdapterDrift(
        served_files={"torchcell/adapters/conf/served_adapter.yaml": ["ServedDataset"]}
    )
    toy.write("torchcell/adapters/conf/toy_adapter.yaml", CONF_YAML + "# edited\n")
    drift, _ = km.adapter_drift_against(manifest, toy.repo)
    assert drift == km.AdapterDrift(
        served_files={
            "torchcell/adapters/conf/served_adapter.yaml": ["ServedDataset"],
            "torchcell/adapters/conf/toy_adapter.yaml": ["ServedDataset"],
        }
    )


def test_value_surface_change_blocks_and_the_ack_admits(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    toy.write(
        "torchcell/datamodels/media.py", "YPD = 'yeast extract peptone dextrose agar'\n"
    )
    report = _admit(toy, manifest)
    assert report.value_surface_changed == ["torchcell/datamodels/media.py"]
    assert report.reasons == [
        "VALUE SURFACE CHANGED: ['torchcell/datamodels/media.py']. These files hold the "
        "shared VALUES served media and compound node ids are content-addressed from, so "
        "a served node and a node the new dataset writes for the same substance or "
        "medium will not share an id. Review the diff, then re-run with "
        "--ack-value-drift '<why served ids are unchanged>'."
    ]
    acked = _admit(toy, manifest, ack_value_drift="only a comment")
    assert acked.verdict == "admissible"
    assert acked.value_drift_acknowledged == "only a comment"
    # a manifest that never recorded the surface compares against nothing
    manifest.value_surface = {}
    unrecorded = _admit(toy, manifest)
    assert unrecorded.verdict == "admissible"
    assert unrecorded.value_surface_recorded is False
    assert unrecorded.value_surface_added == sorted(VALUE_FILES)
    assert unrecorded.value_drift_acknowledged is None


def test_value_surface_line_lists_files_added_since_the_build(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    del manifest.value_surface["torchcell/datamodels/compound_identity_table.json"]
    report = _admit(toy, manifest)
    assert report.verdict == "admissible"
    assert (
        "  value surface: unchanged (3 files); added since the build (additive): "
        "torchcell/datamodels/compound_identity_table.json"
    ) in km.format_report(report)


@pytest.mark.parametrize(
    ("damage", "status"),
    [("processed/lmdb", "missing"), ("preprocess/build_manifest.json", "unmanifested")],
)
def test_a_dev_lmdb_that_is_not_built_blocks(
    toy: _Toy, damage: str, status: str
) -> None:
    target = toy.lmdb_root("toy") / damage
    if target.is_dir():
        target.rmdir()
    else:
        target.unlink()
    report = _admit(toy, _bootstrap(toy))
    assert report.dev_lmdb_status == status
    assert report.reasons == [
        f"dev-tree LMDB for ToyDataset is {status} at {toy.lmdb_root('toy')}; "
        "build it with python -m torchcell.database.build_dataset_lmdb"
    ]


def test_a_dataset_missing_from_the_adapter_map_blocks(
    toy: _Toy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Out of the map, the conf is never read, so no undeclared-method check runs."""
    adapter_map: dict[Any, Any] = dataset_adapter_map
    monkeypatch.delitem(adapter_map, toy.classes["ToyDataset"])
    report = _admit(toy, _bootstrap(toy))
    assert report.in_adapter_map is False
    assert report.reasons == ["ToyDataset is not in dataset_adapter_map"]


def test_undeclared_phenotype_methods_block(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    toy.write(
        "torchcell/adapters/conf/toy_adapter.yaml",
        CONF_YAML.replace(
            "    - method_name: fitness phenotype (chunked)\n",
            "    - method_name: fitness phenotype (chunked)\n"
            "    - method_name: calmorph phenotype (chunked)\n",
        ),
    )
    report = _admit(toy, manifest)
    assert report.undeclared_phenotype_methods == ["calmorph phenotype"]
    assert report.reasons == [
        "adapter enables phenotype node methods the graph schema does not declare "
        "(BioCypher would drop them silently): calmorph phenotype"
    ]


# ---------------------------------------------------------------- superset admission


def _served_ids(ids: list[str]) -> Callable[[str], list[str]]:
    def read(name: str) -> list[str]:
        assert name == "ServedDataset"
        return ids

    return read


def test_a_served_dataset_without_the_live_reader_blocks(toy: _Toy) -> None:
    report = _admit(toy, _bootstrap(toy), "ServedDataset")
    assert (report.served, report.superset) == (True, None)
    assert report.reasons == [
        "ServedDataset is already in the served store; re-admitting it needs the "
        "superset proof, which reads the served experiment ids from the live store "
        "(admit --neo4j-uri)"
    ]
    assert "  served: yes; superset proof not run" in km.format_report(report)


def test_superset_readmission_admits_a_strict_superset(toy: _Toy) -> None:
    """Store holds records 1 and 2; the dev LMDB has 1, 2, 3 -> 0 missing, 1 to add."""
    served = [_id(SERVED_RECORDS[0]), _id(SERVED_RECORDS[1])]
    report = _admit(
        toy,
        _bootstrap(toy),
        "ServedDataset",
        served_experiment_ids=_served_ids(served),
        served_source="bolt://x:7687",
    )
    assert report.verdict == "admissible"
    assert report.superset == km.SupersetCheck(
        n_served=2, n_dev=3, n_missing=0, n_added=1, served_source="bolt://x:7687"
    )
    assert report.novel_symbols == []
    assert report.shared_symbols == {
        "Media": ["ServedDataset"],
        "ModelStrict": ["ServedDataset"],
    }


def test_superset_readmission_blocks_on_a_missing_id_or_nothing_new(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    gone = "f" * 64
    missing = _admit(
        toy,
        manifest,
        "ServedDataset",
        served_experiment_ids=_served_ids([_id(SERVED_RECORDS[0]), gone]),
    )
    assert missing.reasons == [
        "ServedDataset is already in the served store and its dev LMDB no longer "
        "produces 1 of the 2 served experiment ids (full rebuild required: "
        "incremental import would leave those nodes beside their replacements). "
        f"First missing ids: {gone}"
    ]
    same = _admit(
        toy,
        manifest,
        "ServedDataset",
        served_experiment_ids=_served_ids([_id(r) for r in SERVED_RECORDS]),
    )
    assert same.reasons == [
        "ServedDataset is already in the served store and its dev LMDB produces "
        "exactly the 3 served experiment ids; nothing to add"
    ]


def test_dev_experiment_ids_walk_transform_item_in_record_order(toy: _Toy) -> None:
    assert km.dev_experiment_ids(toy.classes["ToyDataset"], toy.data_root) == [
        _id(TOY_RECORDS[0]),
        _id(TOY_RECORDS[1]),
    ]


# ------------------------------------------------------------------------ recording


def test_record_admission_adopts_the_current_surfaces_and_logs_the_event(
    toy: _Toy,
) -> None:
    """After recording, the manifest's reference is the tree now in force, so the same
    admission re-checked afterwards sees no drift even though the tree was edited.
    """
    manifest = _bootstrap(toy)
    toy.write(
        km.SCHEMA_CONFIG_RELPATH,
        SCHEMA_YAML + "\ncalmorph phenotype:\n    represented_as: node\n",
    )
    report = _admit(toy, manifest)
    km.record_admission(
        manifest,
        report,
        biocypher_out="2026-09-27_00-00-00",
        n_experiments=2,
        repo_root=toy.repo,
    )
    entry = manifest.datasets["ToyDataset"]
    assert entry == km.KgDatasetEntry(
        dataset_class="ToyDataset",
        loader_relpath="torchcell/datasets/toy_loader.py",
        adapter_files=[
            "torchcell/adapters/toy_adapter.py",
            "torchcell/adapters/conf/toy_adapter.yaml",
        ],
        closure=report.new_dataset_closure,
        n_experiments=2,
        biocypher_out="2026-09-27_00-00-00",
        import_mode="incremental",
        admitted_at=NOW,
        torchcell_commit=COMMIT,
    )
    assert "calmorph phenotype" in manifest.graph_schema
    assert "torchcell/adapters/cell_adapter.py" not in manifest.adapter_files
    assert sorted(manifest.adapter_files) == [
        "torchcell/adapters/conf/served_adapter.yaml",
        "torchcell/adapters/conf/toy_adapter.yaml",
        "torchcell/adapters/served_adapter.py",
        "torchcell/adapters/toy_adapter.py",
    ]
    assert manifest.events[-1] == km.KgEvent(
        kind="incremental_admission",
        at=NOW,
        torchcell_commit=COMMIT,
        torchcell_version="1.2.0",
        datasets=["ToyDataset"],
        biocypher_out="2026-09-27_00-00-00",
    )
    blocked = report.model_copy(update={"verdict": "blocked", "reasons": ["x"]})
    with pytest.raises(
        ValueError, match=re.escape("cannot record a blocked admission: ['x']")
    ):
        km.record_admission(
            manifest, blocked, biocypher_out="b", n_experiments=1, repo_root=toy.repo
        )


def test_record_batch_admission_validates_counts_and_dedupes_acknowledgments(
    toy: _Toy,
) -> None:
    """Both members measure the same adapter drift, so the acknowledgment is recorded
    once; the ServedDataset member grew by 1, making the event a superset admission.
    """
    manifest = _bootstrap(toy)
    toy.write(km.CELL_ADAPTER_RELPATH, CELL_ADAPTER_PY.replace("return 1", "return 3"))
    toy.write("torchcell/datamodels/media.py", "YPD = 'changed'\n")
    batch = km.check_batch_admission(
        manifest,
        toy.repo,
        ["ToyDataset", "ServedDataset"],
        toy.data_root,
        ack_adapter_drift="plumbing only",
        ack_value_drift="rename only",
        served_experiment_ids=_served_ids([_id(r) for r in SERVED_RECORDS[:2]]),
        served_source="bolt://x:7687",
    )
    assert batch.verdict == "admissible"
    with pytest.raises(
        ValueError, match=re.escape("no experiment count given for ['ToyDataset']")
    ):
        km.record_batch_admission(
            manifest,
            batch,
            biocypher_out="b",
            n_experiments={"ServedDataset": 3},
            repo_root=toy.repo,
        )
    with pytest.raises(
        ValueError,
        match=re.escape("experiment counts for datasets not in the batch: ['Other']"),
    ):
        km.record_batch_admission(
            manifest,
            batch,
            biocypher_out="b",
            n_experiments={"ServedDataset": 3, "ToyDataset": 2, "Other": 1},
            repo_root=toy.repo,
        )
    blocked = batch.model_copy(update={"verdict": "blocked", "reasons": ["y"]})
    with pytest.raises(
        ValueError, match=re.escape("cannot record a blocked batch admission: ['y']")
    ):
        km.record_batch_admission(
            manifest, blocked, biocypher_out="b", n_experiments={}, repo_root=toy.repo
        )
    km.record_batch_admission(
        manifest,
        batch,
        biocypher_out="2026-09-27_00-00-00",
        n_experiments={"ServedDataset": 3, "ToyDataset": 2},
        repo_root=toy.repo,
    )
    assert manifest.events[-1] == km.KgEvent(
        kind="superset_admission",
        at=NOW,
        torchcell_commit=COMMIT,
        torchcell_version="1.2.0",
        datasets=["ToyDataset", "ServedDataset"],
        biocypher_out="2026-09-27_00-00-00",
        note="superset of served: ServedDataset +1 (served 2)",
        acknowledged_adapter_drift=["plumbing: get_nodes: plumbing only"],
        acknowledged_value_drift=["torchcell/datamodels/media.py: rename only"],
    )
    grown = manifest.datasets["ServedDataset"]
    assert grown.superset_of == km.SupersetLineage(
        biocypher_out=BIOCYPHER_OUT,
        import_mode="full",
        admitted_at=BUILT_AT,
        n_experiments=2,
        n_added=1,
    )
    assert grown.n_experiments == 3
    assert manifest.value_surface["torchcell/datamodels/media.py"] == _sha(
        "YPD = 'changed'\n"
    )


# ------------------------------------------------------------------------ live store


class _Result:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __iter__(self) -> Any:
        return iter(self.rows)

    def values(self) -> list[list[Any]]:
        return [list(row.values()) for row in self.rows]

    def single(self) -> dict[str, Any]:
        return self.rows[0]


class _Session:
    def __init__(self, driver: _Driver, database: str) -> None:
        self.driver = driver
        self.database = database

    def __enter__(self) -> _Session:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def run(self, query: str, **params: Any) -> _Result:
        self.driver.calls.append((self.database, query, params))
        return self.driver.answer(query, params)


class _Driver:
    """Answers the three queries ``kg_manifest`` sends, and nothing else."""

    served_ids = [_id(SERVED_RECORDS[0]), _id(SERVED_RECORDS[1])]

    def __init__(self, uri: str, auth: tuple[str, str]) -> None:
        self.uri = uri
        self.auth = auth
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self.closed = 0

    def session(self, database: str) -> _Session:
        return _Session(self, database)

    def close(self) -> None:
        self.closed += 1

    def answer(self, query: str, params: dict[str, Any]) -> _Result:
        if query.startswith("MATCH (d:Dataset {id: $name})"):
            assert params == {"name": "ServedDataset"}
            return _Result([{"id": i} for i in self.served_ids])
        if query.startswith("MATCH (d:Dataset) OPTIONAL MATCH"):
            return _Result([{"id": "ServedDataset", "n": 2}])
        if query.startswith("CALL dbms.components()"):
            return _Result([{"versions": ["5.26.28"]}])
        raise AssertionError(f"unscripted query: {query}")


@pytest.fixture
def drivers(monkeypatch: pytest.MonkeyPatch) -> list[_Driver]:
    made: list[_Driver] = []

    def driver(uri: str, auth: tuple[str, str]) -> _Driver:
        made.append(_Driver(uri, auth))
        return made[-1]

    monkeypatch.setattr("neo4j.GraphDatabase.driver", driver)
    monkeypatch.setenv("NEO4J_USER", "reader")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    return made


def test_live_readers_send_their_query_and_close_the_driver(
    drivers: list[_Driver],
) -> None:
    assert km.live_experiment_ids(
        "bolt://x:7687", "u", "p", "torchcell", "ServedDataset"
    ) == [_id(SERVED_RECORDS[0]), _id(SERVED_RECORDS[1])]
    assert km.live_dataset_counts("bolt://x:7687", "u", "p", "torchcell") == {
        "ServedDataset": 2
    }
    assert km.live_neo4j_version("bolt://x:7687", "u", "p") == "5.26.28"
    assert [(d.uri, d.auth, d.closed) for d in drivers] == [
        ("bolt://x:7687", ("u", "p"), 1)
    ] * 3
    assert [call[0] for d in drivers for call in d.calls] == [
        "torchcell",
        "torchcell",
        "system",
    ]
    assert drivers[0].calls[0][1] == (
        "MATCH (d:Dataset {id: $name})<-[:ExperimentMemberOf]-(e:Experiment) "
        "RETURN e.id AS id"
    )


# ------------------------------------------------------------------------------ CLI


def _cli(toy: _Toy, manifest_path: Path, *args: str) -> int:
    return km.main(["--manifest", str(manifest_path), "--repo", str(toy.repo), *args])


def test_cli_bootstrap_then_show(
    toy: _Toy,
    drivers: list[_Driver],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bootstrap reads the datasets and their counts live (1 dataset, 2 experiments)."""
    path = tmp_path / "kg_manifest.json"
    code = _cli(
        toy,
        path,
        "bootstrap",
        "--commit",
        COMMIT,
        "--biocypher-out",
        BIOCYPHER_OUT,
        "--built-at",
        BUILT_AT,
        "--biocypher-version",
        "0.5.43",
        "--store-host",
        "gilahyper",
        "--neo4j-uri",
        "bolt://x:7687",
    )
    assert code == 0
    assert capsys.readouterr().out == f"bootstrapped 1 datasets -> {path}\n"
    assert km.load_manifest(path).model_dump() == _bootstrap(toy).model_dump()
    assert [d.auth for d in drivers] == [("reader", "secret")] * 2

    assert _cli(toy, path, "show") == 0
    assert capsys.readouterr().out == (
        f"torchcell on gilahyper: neo4j 5.26.28, biocypher 0.5.43, full build {COMMIT}, "
        "1 datasets, 1 events\n"
        f"  ServedDataset: 2 experiments (full, {BIOCYPHER_OUT})\n"
    )


def test_cli_admit_single_writes_the_report_and_exits_by_verdict(
    toy: _Toy,
    drivers: list[_Driver],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "kg_manifest.json"
    km.save_manifest(_bootstrap(toy), path)
    report_path = tmp_path / "report.json"
    code = _cli(
        toy,
        path,
        "admit",
        "--dataset",
        "ToyDataset",
        "--data-root",
        str(toy.data_root),
        "--report",
        str(report_path),
        "--neo4j-uri",
        "bolt://x:7687",
    )
    assert code == 0
    report = km.load_report(report_path)
    assert isinstance(report, km.AdmissionReport)
    assert capsys.readouterr().out == km.format_report(report) + "\n"
    assert drivers == []  # a new dataset never reads the live store

    (toy.lmdb_root("toy") / "processed" / "lmdb").rmdir()
    assert (
        _cli(
            toy,
            path,
            "admit",
            "--dataset",
            "ToyDataset",
            "--data-root",
            str(toy.data_root),
        )
        == 1
    )
    assert (
        "  [BLOCK] dev-tree LMDB for ToyDataset is missing" in capsys.readouterr().out
    )


def test_cli_admit_batch_then_record_it(
    toy: _Toy,
    drivers: list[_Driver],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A batch of the new dataset and the served one (a superset read live via the fake:
    2 served ids, 3 in the dev LMDB), then ``record`` with one NAME=COUNT per member.
    """
    path = tmp_path / "kg_manifest.json"
    km.save_manifest(_bootstrap(toy), path)
    report_path = tmp_path / "batch.json"
    code = _cli(
        toy,
        path,
        "admit",
        "--dataset",
        "ToyDataset,ServedDataset",
        "--data-root",
        str(toy.data_root),
        "--report",
        str(report_path),
        "--neo4j-uri",
        "bolt://x:7687",
    )
    assert code == 0
    batch = km.load_report(report_path)
    assert isinstance(batch, km.BatchAdmissionReport)
    out = capsys.readouterr().out
    assert out == km.format_batch_report(batch) + "\n"
    assert out.startswith(
        "Batch admission check: ToyDataset, ServedDataset  ->  ADMISSIBLE\n"
    )
    assert (
        "    served: yes; superset proof from bolt://x:7687: 2 served ids, 3 in the dev "
        "LMDB, 0 served ids missing from it, 1 to add"
    ) in out
    assert [d.auth for d in drivers] == [("reader", "secret")]

    assert (
        _cli(
            toy,
            path,
            "record",
            "--report",
            str(report_path),
            "--biocypher-out",
            "2026-09-27_00-00-00",
            "--n-experiments",
            "ToyDataset=2",
            "--n-experiments",
            "ServedDataset=3",
        )
        == 0
    )
    assert capsys.readouterr().out == f"recorded ToyDataset, ServedDataset -> {path}\n"
    saved = km.load_manifest(path)
    assert saved.events[-1].kind == "superset_admission"
    assert {name: e.n_experiments for name, e in saved.datasets.items()} == {
        "ServedDataset": 3,
        "ToyDataset": 2,
    }


def test_cli_admit_batch_blocks_and_record_single(
    toy: _Toy,
    drivers: list[_Driver],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A blocked batch exits 1; a single report is recorded with a bare count."""
    path = tmp_path / "kg_manifest.json"
    km.save_manifest(_bootstrap(toy), path)
    monkeypatch.setattr(_Driver, "served_ids", [_id(r) for r in SERVED_RECORDS])
    code = _cli(
        toy,
        path,
        "admit",
        "--dataset",
        "ToyDataset",
        "--dataset",
        "ServedDataset",
        "--data-root",
        str(toy.data_root),
        "--neo4j-uri",
        "bolt://x:7687",
    )
    assert code == 1
    assert (
        "  [BLOCK] ServedDataset: ServedDataset is already in the served store and its "
        "dev LMDB produces exactly the 3 served experiment ids; nothing to add"
    ) in capsys.readouterr().out

    report_path = tmp_path / "single.json"
    report_path.write_text(
        _admit(toy, km.load_manifest(path)).model_dump_json(), encoding="utf-8"
    )
    assert (
        _cli(
            toy,
            path,
            "record",
            "--report",
            str(report_path),
            "--biocypher-out",
            "2026-09-27_00-00-00",
            "--n-experiments",
            "2",
        )
        == 0
    )
    assert capsys.readouterr().out == f"recorded ToyDataset -> {path}\n"
    saved = km.load_manifest(path)
    assert saved.datasets["ToyDataset"].n_experiments == 2
    assert saved.events[-1].kind == "incremental_admission"


def test_cli_drift_names_added_classes_widened_edges_and_new_methods(
    toy: _Toy, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``drift`` needs no dataset: it runs the graph, adapter and value checks alone.

    Unchanged tree: ADDITIVE, exit 0. A new node class that joins ``phenotype member of``
    plus a new ``CellAdapter`` method: still ADDITIVE, exit 0, and the class, the widened
    edge and the method are each named. A served class whose property set moved: CHANGES
    SERVED, exit 1.
    """
    path = tmp_path / "kg_manifest.json"
    km.save_manifest(_bootstrap(toy), path)
    assert _cli(toy, path, "drift") == 0
    assert capsys.readouterr().out == (
        "Served-surface drift  ->  ADDITIVE\n"
        f"  working tree {COMMIT} (dirty=False); served store built at {COMMIT}\n"
        "  graph schema CHANGED: -\n"
        "  graph schema ADDED: -\n"
        "  served edge classes widened (additive): -\n"
        "  adapter drift touching served datasets: none\n"
        "  adapter methods ADDED: -\n"
        "  value surface: unchanged; ADDED: -\n"
    )

    toy.write(
        km.SCHEMA_CONFIG_RELPATH,
        SCHEMA_YAML.replace(
            "    source: [fitness phenotype]\n",
            "    source: [fitness phenotype, flux phenotype]\n",
        )
        + "\nflux phenotype:\n    represented_as: node\n    properties:\n"
        "        net_flux: str\n",
    )
    toy.write(
        km.CELL_ADAPTER_RELPATH,
        CELL_ADAPTER_PY + "\n    def _flux_phenotype_node(self, data, method_name):\n"
        "        return data\n",
    )
    report_path = tmp_path / "drift.json"
    assert _cli(toy, path, "drift", "--report", str(report_path)) == 0
    out = capsys.readouterr().out
    assert out.startswith("Served-surface drift  ->  ADDITIVE\n")
    assert "  graph schema CHANGED: -\n" in out
    assert "  graph schema ADDED: flux phenotype\n" in out
    assert (
        "  served edge classes widened (additive): phenotype member of: +flux phenotype\n"
        in out
    )
    assert "  adapter methods ADDED: _flux_phenotype_node\n" in out
    report = km.ServedSurfaceDrift.model_validate_json(
        report_path.read_text(encoding="utf-8")
    )
    assert report.graph_schema_added == ["flux phenotype"]
    assert report.graph_schema_widened == {"phenotype member of": ["flux phenotype"]}
    assert report.adapter_methods_added == ["_flux_phenotype_node"]
    assert report.adapter_drift.is_empty
    assert not report.changes_served

    toy.replace(
        km.SCHEMA_CONFIG_RELPATH,
        "        fitness: float\n",
        "        fitness: float\n        fitness_se: float\n",
    )
    assert _cli(toy, path, "drift") == 1
    out = capsys.readouterr().out
    assert out.startswith("Served-surface drift  ->  CHANGES SERVED\n")
    assert "  graph schema CHANGED: fitness phenotype\n" in out


def test_an_edge_that_loses_a_source_is_changed_not_widened(toy: _Toy) -> None:
    manifest = _bootstrap(toy)
    toy.replace(
        km.SCHEMA_CONFIG_RELPATH,
        "    source: [fitness phenotype]\n",
        "    source: [calmorph phenotype]\n",
    )
    assert km.graph_schema_drift(manifest, toy.repo) == (["phenotype member of"], [])
    assert km.graph_schema_widened(manifest, toy.repo) == {}


# ------------------------------------------------------------------ parsing guards


def test_graph_schema_skips_entries_that_are_not_classes() -> None:
    """A scalar and a mapping without ``represented_as`` are not BioCypher classes."""
    text = "version: 3\nmeta:\n    is_a: thing\nmedia:\n    represented_as: node\n"
    assert km.graph_schema_from_yaml(text) == {
        "media": km.GraphSchemaEntry(kind="node", properties=[])
    }


def test_conf_names_are_scoped_to_one_class_and_must_be_whole_literals() -> None:
    """Only literals that ARE a conf file name, inside the named class, count.

    ``toy.yaml`` lacks the ``_adapter.yaml`` suffix, a docstring that mentions a conf in
    a sentence is not a whole literal, and a module-level constant belongs to no class.
    """
    source = (
        'MODULE_CONF = "module_adapter.yaml"\n\n\n'
        "class ToyAdapter:\n"
        '    """Loads toy_adapter.yaml from conf/."""\n\n'
        "    CONF = 'toy.yaml'\n\n\n"
        "class OtherAdapter:\n"
        "    def __init__(self):\n"
        '        self.path = ("conf", "other_adapter.yaml")\n'
    )
    assert km.class_conf_names(source, "ToyAdapter") == []
    assert km.class_conf_names(source, "OtherAdapter") == ["other_adapter.yaml"]
    with pytest.raises(
        ValueError,
        match="^expected exactly one module-level class MissingAdapter, found 0$",
    ):
        km.class_conf_names(source, "MissingAdapter")


MULTI_CLASS_ADAPTER_PY = '''"""Two dataset classes served by one module, as Costanzo 2016 is served by three.

The module names ``first_multi_adapter.yaml`` first, so resolving by module text would
hand the second class the first class's enable-list.
"""

import os.path as osp

FIRST = "first_multi_adapter.yaml"


class FirstMultiAdapter:
    def __init__(self):
        self.config_path = osp.join("conf", "first_multi_adapter.yaml")


class SecondMultiAdapter:
    def __init__(self):
        self.config_path = osp.join("conf", "second_multi_adapter.yaml")


class ConflessMultiAdapter:
    def __init__(self):
        self.config_path = osp.join("conf", FIRST)


class TwoConfMultiAdapter:
    def __init__(self, het):
        name = "first_multi_adapter.yaml" if het else "second_multi_adapter.yaml"
        self.config_path = osp.join("conf", name)
'''


@pytest.fixture
def multi_class_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, type]]:
    """A repo whose one adapter module serves four datasets, each mapped in the map.

    Returns the repo root and ``dataset class name -> dataset class``. Each dataset is a
    bare class registered in ``dataset_adapter_map`` for the test only.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    repo = tmp_path.resolve() / "repo"
    module_path = repo / "torchcell/adapters/multi_adapter.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(MULTI_CLASS_ADAPTER_PY, encoding="utf-8")
    module = _import("toy_multi_adapter", module_path, monkeypatch)
    datasets: dict[str, type] = {}
    for stem in ("First", "Second", "Confless", "TwoConf"):
        dataset_cls = type(f"{stem}MultiDataset", (), {})
        datasets[dataset_cls.__name__] = dataset_cls
        monkeypatch.setitem(
            dataset_adapter_map, dataset_cls, getattr(module, f"{stem}MultiAdapter")
        )
    return repo, datasets


def test_each_class_of_a_multi_class_module_resolves_to_its_own_conf(
    multi_class_module: tuple[Path, dict[str, type]],
) -> None:
    """Issue #743: the second class gets ITS conf, not the module's first."""
    repo, datasets = multi_class_module
    assert km.dataset_adapter_files(datasets["FirstMultiDataset"], repo) == [
        "torchcell/adapters/multi_adapter.py",
        "torchcell/adapters/conf/first_multi_adapter.yaml",
    ]
    assert km.dataset_adapter_files(datasets["SecondMultiDataset"], repo) == [
        "torchcell/adapters/multi_adapter.py",
        "torchcell/adapters/conf/second_multi_adapter.yaml",
    ]


def test_an_adapter_class_that_binds_no_conf_is_refused(
    multi_class_module: tuple[Path, dict[str, type]],
) -> None:
    """A conf reached only through a module constant is not bound by the class body.

    Refused rather than resolved to the constant's value: the gate reads the class body
    and nothing else, so there is exactly one place a binding can be and no guess.
    """
    repo, datasets = multi_class_module
    with pytest.raises(
        ValueError,
        match=re.escape(
            "ConflessMultiAdapter (multi_adapter.py) binds no conf yaml in its own "
            "class body"
        ),
    ):
        km.dataset_adapter_files(datasets["ConflessMultiDataset"], repo)


def test_an_adapter_class_that_names_two_confs_is_refused(
    multi_class_module: tuple[Path, dict[str, type]],
) -> None:
    repo, datasets = multi_class_module
    with pytest.raises(
        ValueError,
        match=re.escape(
            "TwoConfMultiAdapter (multi_adapter.py) names more than one conf yaml: "
            "['first_multi_adapter.yaml', 'second_multi_adapter.yaml']"
        ),
    ):
        km.dataset_adapter_files(datasets["TwoConfMultiDataset"], repo)


def test_n_experiments_named_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="^--n-experiments given twice for ADataset$"):
        km.parse_n_experiments(["ADataset=1", "ADataset=2"], ["ADataset"])


def test_a_private_dataset_blocks_without_include_private(
    toy: _Toy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-house data enters a store only by an admission that opts into it.

    The reason names the dataset and the flag; the same check with
    ``include_private=True`` (``admit --include-private``, which the GilaHyper
    increment script passes for the in-house store) lets it through.
    """
    from torchcell.data.experiment_dataset import Visibility

    monkeypatch.setattr(
        toy.classes["ToyDataset"], "visibility", Visibility.private, raising=False
    )
    manifest = _bootstrap(toy)
    report = _admit(toy, manifest)
    assert report.verdict == "blocked"
    assert report.reasons == [
        "ToyDataset is PRIVATE (visibility=private): in-house data is admitted only "
        "to an in-house store, by an admission run with --include-private"
    ]
    admitted = km.check_admission(
        manifest, toy.repo, "ToyDataset", toy.data_root, include_private=True
    )
    assert admitted.verdict == "admissible", admitted.reasons


def test_a_recorded_entry_carries_the_loader_visibility(toy: _Toy) -> None:
    """Both the bootstrap and the admission entry stamp the class's visibility."""
    manifest = _bootstrap(toy)
    assert manifest.datasets["ServedDataset"].visibility == "public"
    report = _admit(toy, manifest)
    assert report.verdict == "admissible"
    entry = km._dataset_entry(
        report,
        n_experiments=3,
        biocypher_out=BIOCYPHER_OUT,
        repo_root=toy.repo,
        at=BUILT_AT,
        previous=None,
    )
    assert entry.visibility == "public"
