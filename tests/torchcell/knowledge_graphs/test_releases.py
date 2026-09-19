"""Release identity, version bumps, content hashes, diffs, and the node round trip."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from torchcell.knowledge_graphs.kg_manifest import (
    KgBuildManifest,
    KgDatasetEntry,
    KgEvent,
)
from torchcell.knowledge_graphs.releases import (
    KgRelease,
    ReleaseDataset,
    ServedDatabase,
    content_hashes_from_csv,
    content_sha256,
    diff,
    format_table,
    next_version,
    release_from_manifest,
    release_id,
    stamp_manifest,
    status_rows,
)


def _manifest(
    datasets: dict[str, int], commit: str = "7715ee35d95c"
) -> KgBuildManifest:
    return KgBuildManifest(
        database="torchcell",
        store_host="gilahyper",
        neo4j_version="5.26.28",
        biocypher_version="0.15.2",
        torchcell_commit=commit,
        graph_schema={},
        cell_adapter_methods={},
        cell_adapter_table={},
        adapter_files={},
        datasets={
            name: KgDatasetEntry(
                dataset_class=name,
                loader_relpath=f"torchcell/datasets/{name}.py",
                adapter_files=[],
                closure={"Experiment": "aa", "Genotype": "bb"},
                n_experiments=n,
                biocypher_out="2026-09-16_00-44-53",
                import_mode="full",
                admitted_at="2026-09-17T20:36:32-05:00",
                torchcell_commit=commit,
            )
            for name, n in datasets.items()
        },
        events=[
            KgEvent(
                kind="bootstrap",
                at="2026-09-18T01:36:59+00:00",
                torchcell_commit=commit,
                datasets=sorted(datasets),
                biocypher_out="2026-09-16_00-44-53",
            )
        ],
        created_at="2026-09-18T01:36:59+00:00",
        updated_at="2026-09-18T01:36:59+00:00",
    )


def test_release_id_is_build_date_and_commit_prefix() -> None:
    assert release_id("2026-09-17T20:36:32-05:00", "7715ee35d95c535620b9") == (
        "2026.09.17-7715ee35"
    )


def test_next_version_bumps_major_on_full_and_minor_on_increment() -> None:
    assert next_version(None, "full") == "1.0"
    assert next_version("1.0", "incremental") == "1.1"
    assert next_version("1.3", "full") == "2.0"


def test_content_sha256_is_order_independent_and_matches_definition() -> None:
    ids = ["c", "a", "b"]
    expected = hashlib.sha256(b"a\nb\nc\n").hexdigest()
    assert content_sha256(ids) == expected
    assert content_sha256(reversed(ids)) == expected
    assert content_sha256(["a", "b"]) != expected


def test_content_hashes_from_csv_groups_experiment_ids_by_dataset(
    tmp_path: Path,
) -> None:
    (tmp_path / "ExperimentMemberOf-header.csv").write_text(
        ":START_ID\t:END_ID\t:TYPE\n", encoding="utf-8"
    )
    (tmp_path / "ExperimentMemberOf-part000.csv").write_text(
        "'e1'\t'DsA'\t'ExperimentMemberOf'\n'e2'\t'DsA'\t'ExperimentMemberOf'\n"
        "'e3'\t'DsB'\t'ExperimentMemberOf'\n",
        encoding="utf-8",
    )
    hashes = content_hashes_from_csv(tmp_path)
    assert hashes == {
        "DsA": content_sha256(["e1", "e2"]),
        "DsB": content_sha256(["e3"]),
    }


def test_stamp_manifest_then_release_round_trips_through_node_properties() -> None:
    manifest = _manifest({"DsA": 10, "DsB": 20})
    hashes = {"DsA": "a" * 64, "DsB": "b" * 64}
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes=hashes,
        previous_version=None,
    )
    assert manifest.version == "1.0"
    assert manifest.release == "2026.09.17-7715ee35"
    assert manifest.datasets["DsA"].content_sha256 == "a" * 64

    release = release_from_manifest(
        manifest,
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes=hashes,
        n_nodes=99_723_455,
    )
    assert release.n_datasets == 2
    assert release.closures["DsB"] == {"Experiment": "aa", "Genotype": "bb"}
    props = release.to_properties()
    assert isinstance(props["datasets_json"], str)
    assert KgRelease.from_properties(props) == release


def test_incremental_stamp_keeps_hashes_of_untouched_datasets() -> None:
    manifest = _manifest({"DsA": 10, "DsB": 20})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64, "DsB": "b" * 64},
        previous_version=None,
    )
    manifest.datasets["DsC"] = manifest.datasets["DsA"].model_copy(
        update={"dataset_class": "DsC", "content_sha256": None}
    )
    manifest.events.append(
        KgEvent(
            kind="incremental_admission",
            at="2026-09-30T00:00:00+00:00",
            torchcell_commit="abcdef0123456789",
            datasets=["DsC"],
            biocypher_out="2026-09-30_00-00-00",
        )
    )
    stamp_manifest(
        manifest,
        kind="incremental",
        built_at="2026-09-30T01:00:00+00:00",
        content_hashes={"DsC": "c" * 64},
        previous_version=manifest.version,
    )
    assert manifest.version == "1.1"
    assert manifest.release == "2026.09.30-abcdef01"
    assert manifest.datasets["DsA"].content_sha256 == "a" * 64
    assert manifest.datasets["DsC"].content_sha256 == "c" * 64
    manifest.datasets["DsD"] = manifest.datasets["DsC"].model_copy(
        update={"dataset_class": "DsD", "content_sha256": None}
    )
    with pytest.raises(ValueError, match="DsD"):
        stamp_manifest(
            manifest,
            kind="incremental",
            built_at="2026-10-01T00:00:00+00:00",
            content_hashes={},
            previous_version=manifest.version,
        )


def test_release_from_manifest_refuses_a_missing_hash() -> None:
    manifest = _manifest({"DsA": 10})
    stamp_manifest(
        manifest,
        kind="full",
        built_at="2026-09-17T20:36:32-05:00",
        content_hashes={"DsA": "a" * 64},
        previous_version="1.2",
    )
    assert manifest.version == "2.0"
    with pytest.raises(ValueError, match="no content hash"):
        release_from_manifest(manifest, built_at="x", content_hashes={}, n_nodes=None)


def _release(tag: str, hashes: dict[str, str]) -> KgRelease:
    return KgRelease(
        release=tag,
        version="1.0",
        torchcell_commit="7715ee35",
        built_at="2026-09-17",
        biocypher_out="2026-09-16_00-44-53",
        datasets={
            name: ReleaseDataset(dataset_class=name, n_experiments=1, content_sha256=h)
            for name, h in hashes.items()
        },
    )


def test_diff_names_unchanged_changed_added_removed() -> None:
    a = _release("2026.09.17-7715ee35", {"DsA": "1", "DsB": "2", "DsC": "3"})
    b = _release("2026.09.30-abcdef01", {"DsA": "1", "DsB": "9", "DsD": "4"})
    result = diff(a, b)
    assert result.unchanged == ["DsA"]
    assert result.changed == ["DsB"]
    assert result.added == ["DsD"]
    assert result.removed == ["DsC"]


def test_status_rows_report_faults_and_missing_release_nodes() -> None:
    healthy = ServedDatabase(
        name="torchcell",
        aliases=["latest", "pinned"],
        default=True,
        status="online",
        release=_release("2026.09.17-7715ee35", {"DsA": "1"}),
        n_nodes=99_723_455,
        n_datasets=51,
    )
    faulting = ServedDatabase(
        name="torchcell",
        aliases=[],
        default=True,
        status="online",
        n_datasets=None,
        fault="java.io.IOException: Input/output error",
    )
    bare = ServedDatabase(
        name="neo4j", aliases=[], default=False, status="online", n_datasets=35
    )
    rows = status_rows("gilahyper", [healthy], None) + status_rows(
        "radiant", [faulting, bare], None
    )
    assert rows[0][:4] == [
        "gilahyper",
        "torchcell [default]",
        "1.0",
        "2026.09.17-7715ee35",
    ]
    assert rows[0][7] == "99,723,455" and rows[0][8] == "latest,pinned"
    assert rows[1][9].startswith("faulting (java.io.IOException")
    assert rows[2][9] == "online (no release node)"
    text = format_table(rows)
    assert text.splitlines()[0].startswith("HOST")
    assert json.dumps(rows)  # plain strings only
