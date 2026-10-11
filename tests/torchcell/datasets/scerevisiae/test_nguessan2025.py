# tests/torchcell/datasets/scerevisiae/test_nguessan2025.py
# [[tests.torchcell.datasets.scerevisiae.test_nguessan2025]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_nguessan2025.py
"""N'Guessan 2025 loader built end to end on a synthetic release.

The release, its expected records and why each value is what it is are in
``_nguessan2025_release.py``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae import _nguessan2025_release as rel
from tests.torchcell.datasets.scerevisiae._boocock2025_release import (
    _write_genomes_tier,
)
from torchcell.data.experiment_dataset import RawSha256MismatchError
from torchcell.datamodels.identity import media_identity
from torchcell.datamodels.media import YPD_LIQUID
from torchcell.datamodels.schema import (
    SegregantExpressionExperiment,
    SegregantExpressionExperimentReference,
)
from torchcell.datasets.scerevisiae import nguessan2025 as m
from torchcell.datasets.scerevisiae.bloom2019 import Marker, ParentAssemblyRefs
from torchcell.literature.manifest import Manifest, RetrievalMethod


@pytest.fixture
def dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> m.Nguessan2025Dataset:
    return rel.build(tmp_path, monkeypatch)


def _exp(dataset: m.Nguessan2025Dataset, i: int) -> SegregantExpressionExperiment:
    return SegregantExpressionExperiment.model_validate(dataset[i]["experiment"])


def _blocks(exp: SegregantExpressionExperiment) -> list[tuple[object, ...]]:
    return [
        (b.chromosome, b.start, b.end, b.parent, b.posterior, b.n_markers)
        for b in exp.genotype.blocks
    ]


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def test_one_record_per_assigned_strain_in_batch1_index_order(
    dataset: m.Nguessan2025Dataset,
) -> None:
    assert len(dataset) == 3
    ids = [dataset[i]["experiment"]["genotype"]["segregant_id"] for i in range(3)]
    assert ids == ["10", "11", "12"]
    assert {_exp(dataset, i).experiment_type for i in range(3)} == {
        "segregant_expression"
    }


def test_a_record_sums_its_significantly_assigned_cells(
    dataset: m.Nguessan2025Dataset,
) -> None:
    s10, s11, s12 = (_exp(dataset, i).phenotype for i in range(3))
    assert (dict(s10.umi_count), s10.total_umis, s10.n_cells) == (
        {"YAL002W": 3, "YFL026W": 2},
        5,
        1,
    )
    assert dict(s11.umi_count) == {
        "YAL001C": 3,
        "YAL002W": 1,
        "YCL001W": 1,
        "YFL026W": 2,
    }
    assert (s11.total_umis, s11.n_cells, s11.library_cell_counts) == (7, 2, {"1": 2})
    assert (dict(s12.umi_count), s12.total_umis) == ({"YCL001W": 4, "YFL026W": 1}, 5)
    assert s11.n_features == 4
    assert (
        s11.feature_universe_sha256
        == hashlib.sha256(b"YAL001C\nYAL002W\nYCL001W\nYFL026W").hexdigest()
    )


def test_the_stage_partition_is_a_typed_gap(dataset: m.Nguessan2025Dataset) -> None:
    phenotype = _exp(dataset, 0).phenotype
    assert phenotype.cell_cycle_stage_counts is None
    assert phenotype.gapped_fields() == {"cell_cycle_stage_counts"}
    assert phenotype.provenance_gaps[0].reason == "not_reported_by_primary"


def test_genotype_blocks_carry_the_panel_posterior_on_the_grid(
    dataset: m.Nguessan2025Dataset,
) -> None:
    assert _blocks(_exp(dataset, 0)) == [
        ("chrI", 100, 100, 2, 1.0, 1),
        ("chrI", 200, 200, 2, 0.95, 1),
        ("chrI", 300, 300, 1, 0.8, 1),
        ("chrIII", 150, 150, 2, 1.0, 1),
    ]
    # strain 11 holds an exact 0.5 at chrI:300 -> parent 1 at posterior 0.5
    assert _blocks(_exp(dataset, 1)) == [
        ("chrI", 100, 200, 1, 1.0, 2),
        ("chrI", 300, 300, 1, 0.5, 1),
        ("chrIII", 150, 150, 2, 1.0, 1),
    ]
    assert _blocks(_exp(dataset, 2)) == [
        ("chrI", 100, 300, 1, 1.0, 3),
        ("chrIII", 150, 150, 2, 1.0, 1),
    ]


def test_genotype_parents_and_pins(dataset: m.Nguessan2025Dataset) -> None:
    genotype = _exp(dataset, 0).genotype
    assert (genotype.cross, genotype.call_method) == (m.CROSS, m.CALL_METHOD)
    assert genotype.marker_matrix_sha256 == rel.sha256(
        Path(dataset.raw_dir) / m.GENO_NAME
    )
    by, rm = genotype.parent_1, genotype.parent_2
    assert (by.name, by.peter_strain_id, by.assembly_ref.member) == (
        "BY4741",
        None,
        None,
    )
    assert by.assembly_ref.path == "S288C_reference_sequence_R64-4-1_20230830.fsa"
    assert by.engineered_background == "the yeast laboratory strain BY4741"
    assert (rm.name, rm.peter_strain_id) == ("RM11-1a", "AAA")
    assert rm.assembly_ref.member == "GENOMES_ASSEMBLED/AAA_6.re.fa"
    assert rm.engineered_background == "the vineyard strain RM11-1a"


def test_environment_is_ypd_for_four_hours_with_a_temperature_gap(
    dataset: m.Nguessan2025Dataset,
) -> None:
    environment = _exp(dataset, 0).environment
    assert environment.temperature is None
    assert environment.gapped_fields() == {"temperature"}
    assert environment.duration_hours == 4.0
    assert media_identity(environment.media) == media_identity(YPD_LIQUID)
    assert [sv.quote for sv in environment.media.provenance[-2:]] == [
        m.SOURCED["medium"].quote,
        m.SOURCED["passage"].quote,
    ]


def test_reference_is_every_assigned_cell_summed(
    dataset: m.Nguessan2025Dataset,
) -> None:
    reference = SegregantExpressionExperimentReference.model_validate(
        dataset[0]["reference"]
    ).phenotype_reference
    assert dict(reference.umi_count) == {
        "YAL001C": 3,
        "YAL002W": 4,
        "YCL001W": 5,
        "YFL026W": 5,
    }
    assert (reference.total_umis, reference.n_cells) == (17, 4)
    assert reference.cell_cycle_stage_counts is None


def test_preprocess_index_and_gene_set(dataset: m.Nguessan2025Dataset) -> None:
    index = (Path(dataset.preprocess_dir) / "data.csv").read_text().splitlines()
    assert index == ["segregant_id,n_cells,n_blocks", "10,1,4", "11,2,3", "12,1,2"]
    assert sorted(dataset.gene_set) == ["YAL001C", "YCL001W"]


def test_process_refuses_a_release_with_other_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rel.patch_sizes(monkeypatch)
    monkeypatch.setattr(m, "N_ASSIGNED_CELLS", 5)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "nguessan2025"
    rel.write_release(root / "raw")
    rel.write_mirror_manifest(data_root, root / "raw")
    _write_genomes_tier(data_root)
    with pytest.raises(
        ValueError, match="4 assigned cells over 3 strains, expected 5 over 3"
    ):
        m.Nguessan2025Dataset(root=str(root), genome=rel.genome())


def test_process_refuses_a_raw_file_edited_after_the_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    real_verify_raw_files: Callable[[str, Mapping[str, str]], None],
) -> None:
    """The conftest swaps in a presence-only recorder; the real byte check refuses."""
    monkeypatch.setattr(m, "verify_raw_files", real_verify_raw_files)
    rel.patch_sizes(monkeypatch)
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    root = tmp_path / "nguessan2025"
    rel.write_release(root / "raw")
    rel.write_mirror_manifest(data_root, root / "raw")
    _write_genomes_tier(data_root)
    edited = root / "raw" / m.GENE_TABLE
    edited.write_text(rel.gene_table() + "\n")
    with pytest.raises(RawSha256MismatchError, match=f"sha256 mismatch for {edited}"):
        m.Nguessan2025Dataset(root=str(root), genome=rel.genome())
    assert not (root / "processed" / "lmdb").exists()


def test_gene_set_needs_a_genome(dataset: m.Nguessan2025Dataset) -> None:
    dataset.genome = None
    with pytest.raises(RuntimeError, match="requires a genome"):
        dataset.compute_gene_set()


def test_records_are_built_in_process_only(dataset: m.Nguessan2025Dataset) -> None:
    with pytest.raises(NotImplementedError, match="process"):
        dataset.create_experiment()
    with pytest.raises(NotImplementedError, match="no gene perturbations"):
        m.Nguessan2025Dataset.extract_systematic_gene_names({})
    frame = pd.DataFrame()
    assert dataset.preprocess_raw(frame) is frame


# --------------------------------------------------------------------------- #
# GitHub release readers
# --------------------------------------------------------------------------- #
def test_read_expression_joins_the_parts_in_release_order(tmp_path: Path) -> None:
    rel.write_release(tmp_path)
    release = m.read_expression([tmp_path / n for n in m.EXPRESSION_PARTS])
    assert release.barcodes == ["AAAC-1", "AAAG-1", "AAAT-1", "AACA-1", "AACC-1"]
    assert release.genes == rel.GENES
    assert release.counts.toarray().tolist() == [
        c for _, c in [*rel.PART_1, *rel.PART_2]
    ]
    assert release.cell_totals.tolist() == [4, 5, 3, 5, 5]


def test_read_expression_part_refuses_a_bad_layout(tmp_path: Path) -> None:
    two = tmp_path / "two.tar.gz"
    rel.write_tar_gz(two, {"a.csv": rel.matrix_csv(rel.PART_1), "b.csv": b"x"})
    with pytest.raises(ValueError, match="expected one CSV member, found 2"):
        m.read_expression_part(two)
    real = tmp_path / "real.tar.gz"
    rel.write_tar_gz(real, {"a.csv": b'"YAL001C"\n"AAAC-1",1.5\n'})
    with pytest.raises(ValueError, match="not integers"):
        m.read_expression_part(real)
    negative = tmp_path / "negative.tar.gz"
    rel.write_tar_gz(negative, {"a.csv": b'"YAL001C"\n"AAAC-1",-1\n'})
    with pytest.raises(ValueError, match="negative UMI counts"):
        m.read_expression_part(negative)


def test_read_expression_refuses_mismatched_or_repeated_parts(tmp_path: Path) -> None:
    first = tmp_path / "1.tar.gz"
    rel.write_tar_gz(first, {"a.csv": rel.matrix_csv(rel.PART_1)})
    other_genes = tmp_path / "2.tar.gz"
    rel.write_tar_gz(
        other_genes,
        {"b.csv": rel.matrix_csv(rel.PART_2, ["YAL001C", "YAL002W", "YCL001W", "X"])},
    )
    with pytest.raises(ValueError, match="gene columns differ"):
        m.read_expression([first, other_genes])
    with pytest.raises(ValueError, match="more than one row"):
        m.read_expression([first, first])
    with pytest.raises(ValueError, match="no expression parts"):
        m.read_expression([])


def test_read_gene_table_strips_the_cds_suffix_and_checks_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "genes.tsv"
    path.write_text(rel.gene_table())
    assert m.read_gene_table(path) == rel.GENES
    path.write_text("a\tb\tc\n0\t0\tYAL001C_CDS\n")
    with pytest.raises(ValueError, match="columns"):
        m.read_gene_table(path)
    path.write_text(
        "Expressed_gene_original_index\tsc_eQTL_gene_id\tGene_name\n0\t1\tYAL001C_CDS\n"
    )
    with pytest.raises(ValueError, match="is not 0..0"):
        m.read_gene_table(path)
    path.write_text(
        "Expressed_gene_original_index\tsc_eQTL_gene_id\tGene_name\n0\t0\tYAL001C\n"
    )
    with pytest.raises(ValueError, match="lacks the _CDS suffix"):
        m.read_gene_table(path)


def test_read_assignment_keeps_na_as_none_and_refuses_a_disagreement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "assign.csv"
    path.write_text(rel.assignment_csv())
    cells = m.read_assignment(path)
    assert [(c.barcode, c.best_match, c.significant_best_match) for c in cells] == [
        ("AAAC-1", 1, 1),
        ("AAAG-1", 0, 0),
        ("AAAT-1", 1, 1),
        ("AACA-1", 2, None),
        ("AACC-1", 2, 2),
    ]
    assert m.assigned_cells(cells) == {1: [0, 2], 0: [1], 2: [4]}
    path.write_text(rel.assignment_csv([("AAAC-1", 1, "2")]))
    with pytest.raises(ValueError, match="significant match 2 != best 1"):
        m.read_assignment(path)
    path.write_text('"barcode","best"\n"AAAC-1",1\n')
    with pytest.raises(ValueError, match="columns"):
        m.read_assignment(path)


def test_library_of_reads_the_gem_group_suffix() -> None:
    assert m.library_of("AAACCCAAGGAGAATG-1") == "1"
    assert m.library_of("ACGT-12") == "12"
    with pytest.raises(ValueError, match="not <ACGT...>-<GEM group>"):
        m.library_of("AAAC")


# --------------------------------------------------------------------------- #
# Dryad readers
# --------------------------------------------------------------------------- #
def test_read_batch_numbers_keeps_file_order_within_the_batch(tmp_path: Path) -> None:
    path = tmp_path / "info.txt"
    path.write_text(rel.segregant_info())
    assert m.read_batch_numbers(path) == [10, 11, 12]
    assert m.read_batch_numbers(path, batch=2) == [20]
    path.write_text("Number\tSet\n1\t1\n")
    with pytest.raises(ValueError, match=r"no \['Batch'\] column"):
        m.read_batch_numbers(path)
    path.write_text(rel.segregant_info([(10, 1), (10, 1)]))
    with pytest.raises(ValueError, match="a Number repeats within batch 1"):
        m.read_batch_numbers(path)


def test_read_snp_markers_sorts_and_maps_back_to_geno_columns(tmp_path: Path) -> None:
    path = tmp_path / "snps.txt"
    path.write_text(rel.snp_list())
    markers, order = m.read_snp_markers(path)
    assert [(mk.chromosome, mk.position, mk.ref, mk.alt) for mk in markers] == [
        ("chrI", 100, "A", "T"),
        ("chrI", 200, "C", "G"),
        ("chrI", 300, "G", "C"),
        ("chrIII", 150, "A", "G"),
    ]
    assert order == [1, 3, 0, 2]
    assert [mk.column for mk in markers] == order


def test_read_snp_markers_refuses_an_unconfirmed_layout(tmp_path: Path) -> None:
    path = tmp_path / "snps.txt"
    path.write_text("SNP\tchrom\tpos\nx\tchr01\t1\n")
    with pytest.raises(ValueError, match="update SNP_LIST_COLUMNS"):
        m.read_snp_markers(path)
    path.write_text(rel.snp_list([(1, "chr01", 100, "A", "T")]))
    with pytest.raises(ValueError, match="index is not 0..0"):
        m.read_snp_markers(path)
    path.write_text(rel.snp_list([(0, "chrI", 100, "A", "T")]))
    with pytest.raises(ValueError, match=r"\['chrI'\] are not chr01..chr16"):
        m.read_snp_markers(path)


def test_read_reference_genotypes_keeps_the_requested_rows(tmp_path: Path) -> None:
    path = tmp_path / "geno.txt.gz"
    rel.geno_gz(path)
    rows = m.read_reference_genotypes(path, {10, 12}, 4)
    assert sorted(rows) == [10, 12]
    assert rows[10].tolist() == [0.20, 0.99, 0.98, 0.97]
    with pytest.raises(ValueError, match="1 requested strains are not in this file"):
        m.read_reference_genotypes(path, {10, 99}, 4)
    with pytest.raises(ValueError, match="strain 10 has 4 posteriors, expected 5"):
        m.read_reference_genotypes(path, {10}, 5)


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_read_reference_genotypes_refuses_a_value_outside_the_unit_interval(
    tmp_path: Path, bad: float
) -> None:
    path = tmp_path / "geno.txt.gz"
    rel.geno_gz(path, {10: [0.1, bad, 0.2, 0.3]})
    with pytest.raises(ValueError, match="outside"):
        m.read_reference_genotypes(path, {10}, 4)


def test_read_reference_genotypes_parses_a_float_written_number(tmp_path: Path) -> None:
    path = tmp_path / "geno.txt.gz"
    with gzip.open(path, "wt") as handle:
        handle.write("10.0\t0.1\t0.9\n")
    assert m.read_reference_genotypes(path, {10}, 2)[10].tolist() == [0.1, 0.9]


# --------------------------------------------------------------------------- #
# Posterior blocks
# --------------------------------------------------------------------------- #
def _markers(*spec: tuple[str, int]) -> list[Marker]:
    return [
        Marker(chromosome=c, position=p, ref="A", alt="G", index=i, column=i)
        for i, (c, p) in enumerate(spec)
    ]


def test_encode_reference_posterior_splits_on_parent_level_and_chromosome() -> None:
    markers = _markers(
        ("chrI", 1), ("chrI", 2), ("chrI", 3), ("chrII", 1), ("chrII", 2)
    )
    posterior = np.array([0.99, 0.991, 0.6, 0.6, 0.4])
    blocks = m.encode_reference_posterior(posterior, markers)
    assert [(b.chromosome, b.start, b.end, b.parent, b.posterior) for b in blocks] == [
        ("chrI", 1, 2, 2, 1.0),
        ("chrI", 3, 3, 2, 0.6),
        ("chrII", 1, 1, 2, 0.6),
        ("chrII", 2, 2, 1, 0.6),
    ]
    ok, error = m.round_trip_error(blocks, markers, posterior)
    assert ok and error == pytest.approx(0.01)


def test_an_exact_half_is_parent_one_at_half_and_a_tie_on_the_round_trip() -> None:
    markers = _markers(("chrI", 1), ("chrI", 2))
    posterior = np.array([0.5, 0.5])
    blocks = m.encode_reference_posterior(posterior, markers)
    assert [(b.parent, b.posterior, b.n_markers) for b in blocks] == [(1, 0.5, 2)]
    assert m.round_trip_error(blocks, markers, posterior) == (True, 0.0)


def test_round_trip_error_reports_a_called_parent_that_flipped() -> None:
    markers = _markers(("chrI", 1))
    blocks = m.encode_reference_posterior(np.array([0.9]), markers)
    ok, error = m.round_trip_error(blocks, markers, np.array([0.1]))
    assert not ok and error == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Parents, environment, raw mirror
# --------------------------------------------------------------------------- #
def test_build_parents_refuses_an_index_without_rm(tmp_path: Path) -> None:
    _write_genomes_tier(tmp_path)
    assemblies = ParentAssemblyRefs.from_tier(str(tmp_path))
    assert m.build_parents(assemblies)[1].peter_strain_id == "AAA"
    stripped = assemblies.model_copy(update={"index": {}})
    with pytest.raises(RuntimeError, match="AAA is not in the member index"):
        m.build_parents(stripped)


def _deposit_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    release = tmp_path / "release"
    rel.write_release(release)
    code = tmp_path / "code"
    for repo_path in m.CODE_FILES:
        (code / repo_path).parent.mkdir(parents=True, exist_ok=True)
        (code / repo_path).write_text(f"code {repo_path}\n")
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / m.DRYAD_METADATA).write_text("{}\n")
    (meta / m.DRYAD_LISTING).write_text("{}\n")
    return release, code, meta


def test_deposit_records_the_walled_files_as_pending_manual_browser(
    tmp_path: Path,
) -> None:
    release, code, meta = _deposit_inputs(tmp_path)
    data_root = tmp_path / "root"
    mirror = m.deposit_raw_mirror(
        release_dir=release,
        code_dir=code,
        dryad_metadata_dir=meta,
        retrieved_at="2026-10-11",
        data_root=str(data_root),
    )
    manifest = Manifest.model_validate_json((mirror / "manifest.json").read_text())
    by_path = {f.path: f for f in manifest.files}
    assert sorted(by_path) == sorted(
        [*m.raw_relpaths().values()]
        + [f"code/{p}" for p in m.CODE_FILES]
        + [f"data/dryad/{m.DRYAD_METADATA}", f"data/dryad/{m.DRYAD_LISTING}"]
    )
    geno = by_path[f"data/dryad/{m.GENO_NAME}"]
    assert geno.retrieval is not None
    assert geno.retrieval.method == RetrievalMethod.manual_browser
    assert (geno.sha256, geno.bytes) == (m.DRYAD_FILES[m.GENO_NAME].sha256, 5078870207)
    assert geno.retrieval.params["dryad_download_url"].endswith("/1474423/download")
    assert not (mirror / geno.path).exists()
    assert manifest.provenance_complete is False
    table = by_path[f"data/github/{m.GENE_TABLE}"]
    assert table.retrieval is not None
    assert table.retrieval.source_url == m.github_url(m.GENE_TABLE)
    assert table.sha256 == rel.sha256(release / m.GENE_TABLE)
    assert m.missing_dryad_files(str(data_root)) == sorted(m.DRYAD_FILES)
    assert (
        m.manifest_sha256(m.load_manifest(str(data_root)), table.path) == table.sha256
    )
    with pytest.raises(KeyError, match="not in the raw-mirror manifest"):
        m.manifest_sha256(manifest, "nope")


def test_deposit_is_idempotent_and_refuses_a_drifted_file(tmp_path: Path) -> None:
    release, code, meta = _deposit_inputs(tmp_path)
    kwargs = dict(
        release_dir=release,
        code_dir=code,
        dryad_metadata_dir=meta,
        retrieved_at="2026-10-11",
        data_root=str(tmp_path / "root"),
    )
    mirror = m.deposit_raw_mirror(**kwargs)  # type: ignore[arg-type]
    m.deposit_raw_mirror(**kwargs)  # type: ignore[arg-type]
    (mirror / "data/github" / m.GENE_TABLE).write_text("edited\n")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        m.deposit_raw_mirror(**kwargs)  # type: ignore[arg-type]


def test_download_refuses_while_the_dryad_files_are_walled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release, code, meta = _deposit_inputs(tmp_path)
    data_root = tmp_path / "root"
    m.deposit_raw_mirror(
        release_dir=release,
        code_dir=code,
        dryad_metadata_dir=meta,
        retrieved_at="2026-10-11",
        data_root=str(data_root),
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    with pytest.raises(RuntimeError, match="issue #1014"):
        m.Nguessan2025Dataset(root=str(tmp_path / "store"), genome=rel.genome())


def test_a_store_builds_from_the_mirror_once_the_dryad_files_arrive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The by-hand deposit, simulated: download() links every file, process() builds."""
    rel.patch_sizes(monkeypatch)
    release, code, meta = _deposit_inputs(tmp_path)
    data_root = tmp_path / "root"
    mirror = m.deposit_raw_mirror(
        release_dir=release,
        code_dir=code,
        dryad_metadata_dir=meta,
        retrieved_at="2026-10-11",
        data_root=str(data_root),
    )
    manifest = json.loads((mirror / "manifest.json").read_text())
    for entry in manifest["files"]:
        name = Path(entry["path"]).name
        if entry["path"].startswith("data/dryad/") and name in m.DRYAD_FILES:
            (mirror / entry["path"]).write_bytes((release / name).read_bytes())
            entry["sha256"] = rel.sha256(release / name)
    (mirror / "manifest.json").write_text(json.dumps(manifest))
    _write_genomes_tier(data_root)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    store = tmp_path / "store"
    dataset = m.Nguessan2025Dataset(root=str(store), genome=rel.genome())
    assert len(dataset) == 3
    linked = sorted(p.name for p in (store / "raw").iterdir())
    assert linked == sorted(m.raw_relpaths())
    assert all((store / "raw" / n).is_symlink() for n in linked)


# --------------------------------------------------------------------------- #
# Sourcing
# --------------------------------------------------------------------------- #
def test_every_quote_names_its_pinned_file() -> None:
    assert {sv.provenance.source_uri for sv in m.SOURCED.values()} == {"paper.md"}
    assert {sv.provenance.sha256 for sv in m.SOURCED.values()} == {m.PAPER_MD_SHA256}
    assert {sv.provenance.source_uri for sv in m.RELEASE_SOURCED.values()} == {
        m.README_REL,
        m.LINEAGE_SCRIPT_REL,
        m.DRYAD_METADATA_REL,
    }
    assert m.sourced_values() == list(m.SOURCED.values())
    assert m.VERIFICATION_SPEC["expected_count"] == m.N_SEGREGANTS == 2951


_LIBRARY = Path("/scratch/projects/torchcell-scratch")


@pytest.mark.skipif(
    not (_LIBRARY / "torchcell-library" / m.CITATION_KEY / "paper.md").exists(),
    reason="literature mirror not mounted",
)
def test_every_quote_is_verbatim_in_the_mirror_bytes() -> None:
    from torchcell.verification.sourced import audit_sourced_value

    paper = [
        audit_sourced_value(sv, _LIBRARY / "torchcell-library")
        for sv in m.SOURCED.values()
    ]
    release = [
        audit_sourced_value(sv, _LIBRARY / "torchcell-raw")
        for sv in m.RELEASE_SOURCED.values()
    ]
    assert [a.details for a in paper + release if not a.passed] == []
