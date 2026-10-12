# tests/torchcell/datasets/scerevisiae/test_hale2024.py
# [[tests.torchcell.datasets.scerevisiae.test_hale2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_hale2024.py
"""Hale 2024 loader, built end to end on the hand-derived synthetic release.

The release and every expected value are documented in ``_hale2024_synthetic``: three
backgrounds served (segA with two barcodes, segB, the BY parent), segC never served,
guides gA (YAL001C, standard name AAA1) and gB (YBL001C, no standard name) plus three
control gRNAs that are ledgered, never served. Records: ATC1 rows b1..b5 (indices 0-4),
ATC2 b1, b2, b4, b5 (5-8), CON b1..b5 (9-13).

Also pinned: the run-length encoding of the NA-bearing columns and its inverse, the
release-shape refusals (unordered markers, a call outside 0/1), and both deposits (raw
mirror and the genomes-tier variant set) from a synthetic literature mirror.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import zipfile
from pathlib import Path
from typing import Any

import lmdb
import pandas as pd
import pytest

from tests.torchcell.datasets.scerevisiae import _hale2024_synthetic as syn
from torchcell.datamodels.schema import (
    CrisprInterferencePerturbation,
    HaplotypeBlock,
    MeasurementType,
    PerturbedSegregantGenotype,
    SampleUnit,
    SegregantPerturbationExperiment,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import hale2024 as h
from torchcell.literature.manifest import (
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.registry import (
    HALE2024_3S_VARIANTS,
    load_genome_manifest,
)


def _block(chrom: str, start: int, end: int, parent: int, n: int) -> HaplotypeBlock:
    return HaplotypeBlock(
        chromosome=chrom,
        start=start,
        end=end,
        parent=parent,  # type: ignore[arg-type]
        n_markers=n,
    )


def _matrix() -> pd.DataFrame:
    frame = pd.DataFrame(syn.MARKERS, columns=["c", "p"])
    frame.insert(2, "gp", frame["p"])
    for name, calls in syn.CALLS.items():
        frame[name] = calls
    return frame


def test_blocks_skip_uncalled_markers_and_round_trip() -> None:
    matrix = _matrix()
    seg_a = h.called_markers(matrix, "segA")
    assert seg_a.positions == [100, 200, 300, 50, 10]
    assert h.encode_blocks(seg_a) == [
        _block("chrI", 100, 100, 1, 1),
        _block("chrI", 200, 300, 2, 2),
        _block("chrII", 50, 50, 2, 1),
        _block("chrmt", 10, 10, 2, 1),
    ]
    seg_b = h.called_markers(matrix, "segB")
    assert h.encode_blocks(seg_b) == [
        _block("chrI", 100, 100, 2, 1),
        _block("chrI", 300, 300, 1, 1),
        _block("chrII", 50, 60, 1, 2),
        _block("chrmt", 10, 10, 2, 1),
    ]
    by = h.called_markers(matrix, "x_BY_BYx")
    assert h.encode_blocks(by) == [
        _block("chrI", 100, 300, 1, 3),
        _block("chrII", 50, 60, 1, 2),
        _block("chrmt", 10, 10, 1, 1),
    ]
    for markers in (seg_a, seg_b, by):
        assert h.expand_blocks(h.encode_blocks(markers), markers) == markers.calls


def test_expand_refuses_blocks_that_do_not_cover_the_markers() -> None:
    markers = h.called_markers(_matrix(), "segB")
    blocks = h.encode_blocks(markers)
    with pytest.raises(ValueError, match="blocks cover 4 of 5 called markers"):
        h.expand_blocks(blocks[:-1], markers)
    with pytest.raises(ValueError, match="does not align"):
        h.expand_blocks(blocks[1:], markers)


def test_release_shape_refusals(tmp_path: Path) -> None:
    frame = _matrix()
    path = tmp_path / "m.txt"
    frame.iloc[[1, 0, 2, 3, 4, 5]].to_csv(path, sep="\t", index=False, na_rep="NA")
    with pytest.raises(ValueError, match="not in \\(chromosome, position\\) order"):
        h.read_genotype_matrix(path)
    bad = frame.copy()
    bad.loc[0, "segC"] = 2
    with pytest.raises(ValueError, match="calls outside"):
        h.called_markers(bad, "segC")
    sd1 = tmp_path / "sd1.txt"
    pd.DataFrame({"seg_name": ["a"], "genotype": ["b"]}).to_csv(sd1, index=False)
    with pytest.raises(ValueError, match="columns"):
        h.read_segregant_table(sd1)
    assert h.background_role("x_BY_BYx") == "parent_1"
    assert h.background_role("BYa_1C_A10") == "segregant"


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease]:
    release = syn.write_release(tmp_path)
    syn.patch(monkeypatch, release)
    dataset = h.CrispriSegregantHale2024Dataset(
        root=str(tmp_path / "store"),
        genome=syn.StubGenome(),  # type: ignore[arg-type]
    )
    return dataset, release


def _typed(dataset: h.CrispriSegregantHale2024Dataset, idx: int) -> Any:
    """One stored record rebuilt into its pydantic classes (interned constants spliced)."""
    return dataset.transform_item(dataset[idx])


def test_build_writes_one_record_per_gene_targeting_lineage_per_assay(
    built: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    dataset, _ = built
    assert len(dataset) == 14
    offsets = syn.offsets()
    first = _typed(dataset, 0)["experiment"]
    assert isinstance(first, SegregantPerturbationExperiment)
    genotype = first.genotype
    assert isinstance(genotype, PerturbedSegregantGenotype)
    assert genotype.background.segregant_id == "segA"
    assert genotype.background_role == "segregant"
    assert [b.n_markers for b in genotype.background.blocks] == [1, 2, 1, 1]
    (pert,) = genotype.perturbations
    assert isinstance(pert, CrisprInterferencePerturbation)
    # The released gene_orf and gene_name cells, verbatim; no genome is read.
    assert (pert.systematic_gene_name, pert.perturbed_gene_name) == ("YAL001C", "AAA1")
    assert pert.crispr.guide_sequence == "gA"
    assert pert.crispr.effector == "dCas9-Mxi1"
    phenotype = first.phenotype
    assert phenotype.environment_response == 0.40 - offsets["ATC1"]
    assert phenotype.environment_response_uncertainty == 0.05
    assert phenotype.environment_response_se == 0.05
    assert (
        phenotype.environment_response_uncertainty_type
        is UncertaintyType.standard_error
    )
    assert phenotype.measurement_type is MeasurementType.relative_fitness
    assert phenotype.sample_unit is SampleUnit.biological_replicate
    assert (phenotype.screen_id, phenotype.replicate_id) == ("ATC1", "GA1_b1")
    (atc,) = first.environment.perturbations
    assert atc.compound.name == "anhydrotetracycline"  # type: ignore[union-attr]
    assert atc.concentration.value == 0.25  # type: ignore[union-attr]
    assert first.environment.temperature is not None
    assert first.environment.temperature.value == 21.0
    assert first.environment.duration_generations == 10.0

    by = _typed(dataset, 4)["experiment"].genotype
    assert (by.background.segregant_id, by.background_role) == ("x_BY_BYx", "parent_1")
    # gB's released gene_name cell is NA, so the ORF stands in for it.
    assert by.perturbations[0].perturbed_gene_name == "YBL001C"
    atc2 = _typed(dataset, 7)["experiment"].phenotype
    assert (atc2.screen_id, atc2.replicate_id) == ("ATC2", "GB1_b4")
    assert atc2.environment_response == -0.08 - offsets["ATC2"]
    con = _typed(dataset, 13)["experiment"]
    assert con.environment.perturbations == []
    assert (con.phenotype.screen_id, con.phenotype.replicate_id) == ("CON", "GBY_b5")
    assert con.phenotype.environment_response == 0.30 - offsets["CON"]
    reference = _typed(dataset, 13)["reference"]
    assert reference.phenotype_reference.environment_response == 0.0
    assert reference.phenotype_reference.screen_id == "CON"
    assert reference.genome_reference.strain == "S288C"


def test_build_writes_the_ledger_summary_index_and_gene_set(
    built: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    dataset, _ = built
    pre = Path(dataset.preprocess_dir)
    (control,) = syn.read_json(pre / "dropped_records.json")
    assert control["rule"] == "control_guide_targets_no_gene"
    assert (control["n_records"], control["per_assay"]) == (
        9,
        {"ATC1": 3, "ATC2": 3, "CON": 3},
    )
    assert control["items"] == ["c1", "c2", "c3"]
    summary = syn.read_json(pre / "summary.json")
    assert summary["member_ranges"] == {"ATC1": [0, 5], "ATC2": [5, 9], "CON": [9, 14]}
    assert summary["lineages_per_assay"] == {"ATC1": 5, "ATC2": 4, "CON": 5}
    assert (summary["lineages_in_all_assays"], summary["lineages_in_any_assay"]) == (
        4,
        5,
    )
    assert syn.read_json(pre / "block_counts.json") == {
        "segA": 4,
        "segB": 4,
        "segC": 3,
        "x_BY_BYx": 3,
    }
    assert syn.read_json(pre / "target_genes.json") == ["YAL001C", "YBL001C"]
    assert sorted(dataset.gene_set) == ["YAL001C", "YBL001C"]
    index = dataset.experiment_reference_index
    assert index is not None
    assert [eri.member_indices for eri in index] == [
        [0, 1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12, 13],
    ]
    assert [
        getattr(eri.reference.phenotype_reference, "screen_id") for eri in index
    ] == ["ATC1", "ATC2", "CON"]


def test_constants_are_interned_and_records_stay_small(
    built: tuple[h.CrispriSegregantHale2024Dataset, syn.SyntheticRelease],
) -> None:
    dataset, _ = built
    dataset.close_lmdb()
    env = lmdb.open(
        str(Path(dataset.processed_dir) / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        stored = txn.get(b"0")
        assert stored is not None
        raw: dict[str, Any] = pickle.loads(stored)
    env.close()
    genotype = raw["experiment"]["genotype"]
    assert set(genotype["background"]) == {"$ref", "name"}
    assert genotype["background"]["name"] == "segA"
    assert set(genotype["perturbations"][0]) == {"$ref", "name"}
    assert set(raw["publication"]) == {"$ref", "name"}
    assert set(raw["reference"]) == {"$ref", "name"}
    assert set(raw["experiment"]["environment"]) == {"$ref", "name"}


def test_a_barcode_assigned_to_another_segregant_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = syn.write_release(tmp_path)
    table = pd.read_csv(release.raw_dir / h.SD1)
    table.loc[table["geno_bc"] == "GA2", "genotype"] = "segB"
    table.to_csv(release.raw_dir / h.SD1, index=False)
    manifest = Manifest.model_validate_json(
        (release.data_root / h.RAW_DIR_REL / "manifest.json").read_text()
    )
    sha = hashlib.sha256((release.raw_dir / h.SD1).read_bytes()).hexdigest()
    files = [
        f.model_copy(update={"sha256": sha}) if f.path == f"data/{h.SD1}" else f
        for f in manifest.files
    ]
    (release.data_root / h.RAW_DIR_REL / "manifest.json").write_text(
        manifest.model_copy(update={"files": files}).model_dump_json()
    )
    release.sha[h.SD1] = sha
    syn.patch(monkeypatch, release)
    with pytest.raises(ValueError, match="segregant barcode GA2 is 'segB' in SD1"):
        h.CrispriSegregantHale2024Dataset(
            root=str(tmp_path / "store"),
            genome=syn.StubGenome(),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Deposits
# --------------------------------------------------------------------------- #
def _library(root: Path, release: syn.SyntheticRelease) -> None:
    """A literature mirror holding the released files at their ``si/`` paths."""
    library = root / h.LIBRARY_DIR_REL
    (library / "si").mkdir(parents=True)
    records = []
    sources = {**{n: release.raw_dir / n for n in h.RELEASED_FILES}}
    sources[h.SD13] = release.variants
    specs = {**h.RELEASED_FILES, h.SD13: h.VARIANT_FILE}
    for name, path in sources.items():
        spec = specs[name]
        data = path.read_bytes()
        (library / spec.library_path).write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        records.append(
            ArtifactRecord(
                path=spec.library_path,
                role=ROLE_SI_DATA,
                bytes=len(data),
                sha256=sha,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.pmc_cloud,
                    source_url=h.PMC_BASE + spec.pmc_object,
                    retriever="torchcell.literature.retrieve.pmc_cloud_object",
                    params={"key": spec.pmc_object},
                    sha256=sha,
                    retrieved_at="2026-10-11T10:41:05+00:00",
                ),
            )
        )
    (library / "manifest.json").write_text(
        Manifest(
            citation_key=h.CITATION_KEY,
            files=records,
            provenance_complete=True,
            created_at="2026-10-11T00:00:00+00:00",
        ).model_dump_json()
    )


def test_deposits_copy_the_mirror_bytes_and_record_their_retrievals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = syn.write_release(tmp_path / "src")
    root = tmp_path / "root"
    syn.patch(monkeypatch, release)
    monkeypatch.setattr(
        h,
        "VARIANT_FILE",
        h.VARIANT_FILE.model_copy(
            update={"sha256": hashlib.sha256(release.variants.read_bytes()).hexdigest()}
        ),
    )
    _library(root, release)
    code_zip = tmp_path / "code.zip"
    with zipfile.ZipFile(code_zip, "w") as archive:
        for member in h.CODE_MEMBERS.values():
            archive.writestr(member, f"# {member}\n")
    fitseq = tmp_path / "fitseq.py"
    fitseq.write_text("return -likelihood_log_lineage\n")
    code_sha = {
        rel: hashlib.sha256(f"# {member}\n".encode()).hexdigest()
        for rel, member in h.CODE_MEMBERS.items()
    }
    code_sha[h.FITSEQ_REL] = hashlib.sha256(fitseq.read_bytes()).hexdigest()
    monkeypatch.setattr(h, "CODE_SHA256", code_sha)
    monkeypatch.setattr(
        h, "CODE_ZIP_SHA256", hashlib.sha256(code_zip.read_bytes()).hexdigest()
    )

    for _ in range(2):  # idempotent: the second run finds identical files
        h.deposit_raw_mirror(
            code_zip_path=code_zip,
            fitseq_path=fitseq,
            retrieved_at_code="2026-10-11T10:44:21+00:00",
            retrieved_at_fitseq="2026-10-11T10:48:28+00:00",
            data_root=str(root),
        )
    manifest = h.load_manifest(str(root))
    paths = [f.path for f in manifest.files]
    assert paths == [f"data/{n}" for n in h.RELEASED_FILES] + list(h.CODE_MEMBERS) + [
        h.FITSEQ_REL
    ]
    by_path = {f.path: f for f in manifest.files}
    sd6 = by_path[f"data/{h.SD6}"].retrieval
    assert sd6 is not None and sd6.method is RetrievalMethod.pmc_cloud
    assert sd6.params == {"key": "PMC11102447.1/41467_2024_48626_MOESM9_ESM.csv"}
    job = by_path["code/EIGHT_fitseq.job"].retrieval
    assert (
        job is not None and job.retriever == "torchcell.literature.retrieve.zip_member"
    )
    assert job.params["member"] == "code/data_processing/EIGHT_fitseq.job"
    assert job.params["url"] == "https://ndownloader.figshare.com/files/43268517"
    fit = by_path[h.FITSEQ_REL].retrieval
    assert fit is not None and fit.params["commit"] == h.FITSEQ_COMMIT
    assert h.manifest_sha256(manifest, f"data/{h.SD1}") == release.sha[h.SD1]

    path = h.deposit_variant_set(str(root))
    genome_manifest = load_genome_manifest(HALE2024_3S_VARIANTS, str(root))
    assert path.endswith(f"{HALE2024_3S_VARIANTS}/manifest.json")
    (record,) = genome_manifest.files
    assert (record.path, record.role) == (h.SD13, "variants")
    assert "A VARIANT TABLE, NOT AN ASSEMBLY" in (genome_manifest.notes or "")
    assert h.deposit_variant_set(str(root)) == path  # a deposited set is left alone

    (root / h.RAW_DIR_REL / "data" / h.SD1).write_text("tampered")
    with pytest.raises(RuntimeError, match="exists with a different sha256"):
        h.deposit_raw_mirror(
            code_zip_path=code_zip,
            fitseq_path=fitseq,
            retrieved_at_code="x",
            retrieved_at_fitseq="x",
            data_root=str(root),
        )
    assert json.loads((root / h.RAW_DIR_REL / "manifest.json").read_text())


def test_the_module_pins_match_the_release_shape() -> None:
    assert h.EXPECTED_RECORDS == 2923436 + 2857376 + 2911823 == 8692635
    assert [a.screen_id for a in h.ASSAYS] == ["ATC1", "ATC2", "CON"]
    assert [a.released_rows - a.gene_rows for a in h.ASSAYS] == [36064, 35257, 35475]
    assert h.RELEASED_FILES[h.SD6].pmc_object == (
        "PMC11102447.1/41467_2024_48626_MOESM9_ESM.csv"
    )
    assert h.ATC_DOSE_UG_PER_ML.value == 0.25
    assert h.TEMPERATURE_C.value == 21.0
    assert h.CHROMOSOMES[17] == "chrmt"
