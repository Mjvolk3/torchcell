# tests/torchcell/datasets/scerevisiae/test_ohnuki2022.py
# [[tests.torchcell.datasets.scerevisiae.test_ohnuki2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ohnuki2022.py
"""Hermetic build of the Ohnuki 2022 quadruple-deletion CalMorph loader.

Both TSVs are written into ``<root>/raw/`` so PyG never calls ``download()``. The genome
stub implements only ``resolve_gene_name``: YAL001C, YBR001C and the background gene
YGL013C are current; YDR012W is renamed to the background gene YDR011W; YCR001W is retired.

Features (4 of the 501): base ``A101_A``, ``C103_A1B``; CV ``ACV103_A1B``, ``CCV103_A1B``
(split by membership in ``CALMORPH_STATISTICS``, not by prefix).

``quad1982data.tsv`` (ORF, A101_A, C103_A1B, ACV103_A1B, CCV103_A1B):

    YAL001C    1.0  2.0    0.5    0.25
    ygl013c    2.0  3.0    0.5    0.5      PDR1, already deleted in 3Delta -> dropped
    YDR012W    2.0  3.0    0.5    0.5      alias of SNQ2 (YDR011W) -> dropped after reconciliation
    YBR001C    3.0  (blank) 0.25  0.5      missing value -> dropped
    YCR001W    5.0  6.0    0.125  0.75     retired name kept verbatim

``wt749data.tsv`` (NAME + the same features): 1.0 3.0 0.25 0.5 and 3.0 5.0 0.75 1.0 ->
means A101_A 2.0, C103_A1B 4.0, ACV103_A1B 0.5, CCV103_A1B 0.75.

2026.09.30 (Phase 15): the same five-row matrix, plus

- the drop ledger as logged (lines 241 to 258): one incomplete row, ``['YBR001C']``, and
  two background collisions in file order, ``['YGL013C', 'YDR011W']``, then "Processing
  ... (2 strains)".
- a clean base-only matrix (``A101_A``, ``C103_A1B``; no CV trait, so
  ``calmorph_coefficient_of_variation`` is None) with YAL001C twice (1.0 2.0 and 3.0 4.0)
  and a reference with a blank cell: wt1 1.0 (blank), wt2 3.0 5.0, so the reference means
  are A101_A (1.0 + 3.0) / 2 = 2.0 and C103_A1B 5.0 / 1 = 5.0 (``mean`` skips NaN). No
  drop line is logged.
- ``download`` with the two module sha256 pins replaced by the digests of the synthetic
  files (they are read at call time, line 188): an empty ``raw/`` is filled from
  ``$DATA_ROOT/torchcell-library/ohnukiHighthroughputPlatformYeast2022/data`` and the
  build proceeds; a second call with the mirror gone skips both verified raw files; a
  copy that lands different bytes is refused by the build-time check in ``process``.
- a non-numeric reference cell ("x" in row 2) refuses with pandas' ``ValueError``
  'Unable to parse string "x" at position 1' (``errors="raise"``, line 313).
- the default-genome path (``genome=None`` calls ``default_genome()`` once, line 233) and
  ``main`` with the dataset class replaced by a recorder.

Findings pinned here: the same ORF listed twice gives two records with identical
genotypes (no duplicate check before line 278); and the reference mean silently skips a
blank reference cell (line 313) while a blank mutant cell drops the whole strain (line
239), so "never impute" holds for the mutants only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MarkerDeletionPerturbation,
    Media,
    NatMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import ohnuki2022 as m
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

_FEATURES = ["A101_A", "C103_A1B", "ACV103_A1B", "CCV103_A1B"]
_MUTANT_ROWS = [
    ["YAL001C", "1.0", "2.0", "0.5", "0.25"],
    ["ygl013c ", "2.0", "3.0", "0.5", "0.5"],
    ["YDR012W", "2.0", "3.0", "0.5", "0.5"],
    ["YBR001C", "3.0", "", "0.25", "0.5"],
    ["YCR001W", "5.0", "6.0", "0.125", "0.75"],
]
_WT_ROWS = [["wt1", "1.0", "3.0", "0.25", "0.5"], ["wt2", "3.0", "5.0", "0.75", "1.0"]]
_CURRENT = {"YAL001C", "YBR001C", "YGL013C"}
_RENAMED = {"YDR012W": "YDR011W"}


class _StubGenome:
    def resolve_gene_name(self, name: str) -> GeneNameResolution:
        upper = name.strip().upper()
        if upper in _CURRENT:
            return GeneNameResolution(
                input_name=name, status=GeneNameStatus.CURRENT, systematic_name=upper
            )
        if upper in _RENAMED:
            return GeneNameResolution(
                input_name=name,
                status=GeneNameStatus.RENAMED,
                systematic_name=_RENAMED[upper],
            )
        return GeneNameResolution(
            input_name=name, status=GeneNameStatus.RETIRED, systematic_name=upper
        )


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _write_tsv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.write_text("\n".join("\t".join(r) for r in [header, *rows]) + "\n")


def _root(tmp_path: Path, slug: str = "scmd_ohnuki2022") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_tsv(root / "raw" / m.MUTANT_FILE, ["ORF", *_FEATURES], _MUTANT_ROWS)
    _write_tsv(root / "raw" / m.WT_FILE, ["NAME", *_FEATURES], _WT_ROWS)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.ScmdOhnuki2022Dataset:
    return m.ScmdOhnuki2022Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(
    media=Media(name="YPD", state="liquid", is_synthetic=False),
    temperature=Temperature(value=25),
)
_WT_PHENOTYPE = CalMorphPhenotype(
    calmorph={"A101_A": 2.0, "C103_A1B": 4.0},
    calmorph_coefficient_of_variation={"ACV103_A1B": 0.5, "CCV103_A1B": 0.75},
)
_REFERENCE = CalMorphExperimentReference(
    dataset_name="ScmdOhnuki2022Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4741"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_WT_PHENOTYPE,
).model_dump()
_PUBLICATION = Publication(
    pubmed_id="35087094",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/35087094/",
    doi="10.1038/s41540-022-00212-1",
    doi_url="https://doi.org/10.1038/s41540-022-00212-1",
).model_dump()


def _experiment(orf: str, values: list[float]) -> dict[str, Any]:
    return CalMorphExperiment(
        dataset_name="ScmdOhnuki2022Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                ),
                NatMxDeletionPerturbation(
                    systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
                ),
                MarkerDeletionPerturbation(
                    systematic_gene_name="YBL005W",
                    perturbed_gene_name="PDR3",
                    marker="KlURA3",
                ),
                MarkerDeletionPerturbation(
                    systematic_gene_name="YDR011W",
                    perturbed_gene_name="SNQ2",
                    marker="KlLEU2",
                ),
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=CalMorphPhenotype(
            calmorph={"A101_A": values[0], "C103_A1B": values[1]},
            calmorph_coefficient_of_variation={
                "ACV103_A1B": values[2],
                "CCV103_A1B": values[3],
            },
        ),
    ).model_dump()


def test_two_quadruple_deletion_records(dataset: m.ScmdOhnuki2022Dataset) -> None:
    """Five rows give two records; three rows drop: the two 3Delta-background targets
    (one caught only after alias reconciliation) and the incomplete row. Each genotype is the target
    KanMX deletion plus the constant PDR1/PDR3/SNQ2 background; the reference genome is
    the BY4741 placeholder with the 3Delta parent's measured means as phenotype.
    """
    assert len(dataset) == 2
    assert dataset[0]["experiment"] == _experiment("YAL001C", [1.0, 2.0, 0.5, 0.25])
    assert dataset[1]["experiment"] == _experiment("YCR001W", [5.0, 6.0, 0.125, 0.75])
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[1]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    genotype = dataset[0]["experiment"]["genotype"]["perturbations"]
    assert [p["perturbation_type"] for p in genotype] == [
        "kanmx_deletion",
        "marker_deletion",
        "marker_deletion",
        "natmx_deletion",
    ]


def test_side_files(dataset: m.ScmdOhnuki2022Dataset) -> None:
    """``data.csv`` is the retained matrix (ORF column already stripped and uppercased)
    with ``systematic_gene_name`` appended; one reference covers both records.

    Finding: the gene set includes the three background genes YBL005W, YDR011W and
    YGL013C next to the two targets, because ``compute_gene_set`` reads every
    perturbation of the genotype.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "ORF,A101_A,C103_A1B,ACV103_A1B,CCV103_A1B,systematic_gene_name\n"
        "YAL001C,1.0,2.0,0.5,0.25,YAL001C\n"
        "YCR001W,5.0,6.0,0.125,0.75,YCR001W\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBL005W",
        "YCR001W",
        "YDR011W",
        "YGL013C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1]]
    assert index[0]["reference"]["phenotype_reference"] == _WT_PHENOTYPE.model_dump()
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "scmd_ohnuki2022"
    assert manifest["loader_class"] == "ScmdOhnuki2022Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.ohnuki2022"


def test_download_checks_raw_then_mirror_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present raw file with the wrong hash is refused by ``process()`` (the build-time
    check, issue #518's sweep) with ``RawSha256MismatchError``; an empty root with no
    mirror file raises "mirror file missing"; a mirror file holding ``b"not the matrix"``
    is refused before the copy with its digest (35fe305f...) and nothing lands in
    ``raw/``.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    dataset = m.ScmdOhnuki2022Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / m.MUTANT_FILE
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected {m.MUTANT_SHA256}, observed {digest}"
    )
    mirror = data_root / m.MIRROR_SUBPATH
    with pytest.raises(
        RuntimeError, match=re.escape(f"mirror file missing: {mirror / m.MUTANT_FILE}")
    ):
        m.ScmdOhnuki2022Dataset(root=str(tmp_path / "empty"), genome=_genome())
    mirror.mkdir(parents=True)
    (mirror / m.MUTANT_FILE).write_bytes(b"not the matrix")
    bad = hashlib.sha256(b"not the matrix").hexdigest()
    assert bad.startswith("35fe305f")
    with pytest.raises(RawSha256MismatchError) as err:
        m.ScmdOhnuki2022Dataset(root=str(tmp_path / "empty2"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / m.MUTANT_FILE}: expected {m.MUTANT_SHA256}, "
        f"observed {bad}"
    )
    assert list((tmp_path / "empty2" / "raw").iterdir()) == []


def _background() -> list[Any]:
    return [
        NatMxDeletionPerturbation(
            systematic_gene_name="YGL013C", perturbed_gene_name="PDR1"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YBL005W", perturbed_gene_name="PDR3", marker="KlURA3"
        ),
        MarkerDeletionPerturbation(
            systematic_gene_name="YDR011W", perturbed_gene_name="SNQ2", marker="KlLEU2"
        ),
    ]


def test_items_retype_through_the_calmorph_classes(
    dataset: m.ScmdOhnuki2022Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    166 to 174): a ``CalMorphExperiment`` and a ``CalMorphExperimentReference`` that dump
    back to exactly the stored dictionaries. A fitness class here would drop the
    ``calmorph`` traits and fail the round trip.
    """
    for index in range(2):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is CalMorphExperiment
        assert type(typed["reference"]) is CalMorphExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION


def test_inline_build_leaves_the_generic_hooks_inert(
    dataset: m.ScmdOhnuki2022Dataset,
) -> None:
    """Records are built inside ``process``: ``preprocess_raw`` hands back the frame it was
    given, unchanged, and ``create_experiment`` raises a bare ``NotImplementedError``.
    """
    frame = pd.DataFrame({"ORF": ["YAL001C"], "A101_A": [1.0]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"ORF": ["YAL001C"], "A101_A": [1.0]}
    with pytest.raises(NotImplementedError) as info:
        dataset.create_experiment()
    assert str(info.value) == ""


def test_drop_ledger_is_logged_in_file_order(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The five-row matrix logs one incomplete row and two background collisions, named
    by their reconciled systematic names in file order, then the retained count.
    """
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.ScmdOhnuki2022Dataset(root=str(_root(tmp_path)), genome=_genome())
    messages = [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert messages == [
        "Ohnuki: dropping 1 mutant row(s) with missing CalMorph values: ['YBR001C']",
        "Ohnuki: dropping 2 target row(s) colliding with the 3Delta background: "
        "['YGL013C', 'YDR011W']",
        "Processing Ohnuki 2022 CalMorph morphology (2 strains)...",
    ]


def test_clean_base_only_matrix_keeps_duplicates_and_skips_blank_reference_cells(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Base-only traits give ``calmorph_coefficient_of_variation=None``; no drop line is
    logged when nothing drops.

    Finding: YAL001C listed twice gives two records with the same quadruple genotype,
    nothing deduplicates the target ORF. Finding: the reference C103_A1B is 5.0, the
    mean of the one non-blank replicate, so a blank reference cell is skipped while a
    blank mutant cell drops the strain. Pinned until the loader refuses both.
    """
    root = tmp_path / "clean"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / m.MUTANT_FILE,
        ["ORF", "A101_A", "C103_A1B"],
        [["YAL001C", "1.0", "2.0"], ["YAL001C", "3.0", "4.0"]],
    )
    _write_tsv(
        root / "raw" / m.WT_FILE,
        ["NAME", "A101_A", "C103_A1B"],
        [["wt1", "1.0", ""], ["wt2", "3.0", "5.0"]],
    )
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhnuki2022Dataset(root=str(root), genome=_genome())
    messages = [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert messages == ["Processing Ohnuki 2022 CalMorph morphology (2 strains)..."]
    assert len(dataset) == 2
    genotype = Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YAL001C", perturbed_gene_name="YAL001C"
            ),
            *_background(),
        ]
    )
    for index, values in enumerate(([1.0, 2.0], [3.0, 4.0])):
        assert (
            dataset[index]["experiment"]
            == CalMorphExperiment(
                dataset_name="ScmdOhnuki2022Dataset",
                genotype=genotype,
                environment=_ENVIRONMENT,
                phenotype=CalMorphPhenotype(
                    calmorph={"A101_A": values[0], "C103_A1B": values[1]},
                    calmorph_coefficient_of_variation=None,
                ),
            ).model_dump()
        )
        assert (
            dataset[index]["reference"]
            == CalMorphExperimentReference(
                dataset_name="ScmdOhnuki2022Dataset",
                genome_reference=ReferenceGenome(
                    species="Saccharomyces cerevisiae", strain="BY4741"
                ),
                environment_reference=_ENVIRONMENT,
                phenotype_reference=CalMorphPhenotype(
                    calmorph={"A101_A": 2.0, "C103_A1B": 5.0},
                    calmorph_coefficient_of_variation=None,
                ),
            ).model_dump()
        )


def test_non_numeric_reference_cell_refuses(tmp_path: Path) -> None:
    """``pd.to_numeric(..., errors="raise")`` refuses a reference cell that is not a
    number, naming the value and its row position.
    """
    root = tmp_path / "bad_wt"
    (root / "raw").mkdir(parents=True)
    _write_tsv(root / "raw" / m.MUTANT_FILE, ["ORF", "A101_A"], [["YAL001C", "1.0"]])
    _write_tsv(
        root / "raw" / m.WT_FILE, ["NAME", "A101_A"], [["wt1", "1.0"], ["wt2", "x"]]
    )
    with pytest.raises(ValueError, match=r'^Unable to parse string "x" at position 1$'):
        m.ScmdOhnuki2022Dataset(root=str(root), genome=_genome())


def test_missing_genome_is_built_once_by_default_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``genome=None`` the build calls ``default_genome()`` exactly once and uses
    what it returns to reconcile the targets, so the records match the stubbed build.
    """
    stub: object = _StubGenome()
    calls: list[None] = []

    def fake_default_genome() -> object:
        calls.append(None)
        return stub

    monkeypatch.setattr(m, "default_genome", fake_default_genome)
    dataset = m.ScmdOhnuki2022Dataset(root=str(_root(tmp_path)))
    assert calls == [None]
    assert dataset.genome is stub
    assert [dataset[i]["experiment"] for i in range(len(dataset))] == [
        _experiment("YAL001C", [1.0, 2.0, 0.5, 0.25]),
        _experiment("YCR001W", [5.0, 6.0, 0.125, 0.75]),
    ]


def _pinned_to_synthetic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, bytes]]:
    """A mirror holding the synthetic tables, with the module pins set to their digests."""
    source = _root(tmp_path, "source")
    data_root = tmp_path / "data_root"
    mirror = data_root / m.MIRROR_SUBPATH
    mirror.mkdir(parents=True)
    content: dict[str, bytes] = {}
    for name in (m.MUTANT_FILE, m.WT_FILE):
        content[name] = (source / "raw" / name).read_bytes()
        (mirror / name).write_bytes(content[name])
    monkeypatch.setattr(
        m, "MUTANT_SHA256", hashlib.sha256(content[m.MUTANT_FILE]).hexdigest()
    )
    monkeypatch.setattr(m, "WT_SHA256", hashlib.sha256(content[m.WT_FILE]).hexdigest())
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return data_root, mirror, content


def test_download_copies_verified_mirror_files_then_skips_verified_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An empty root copies both tables from the mirror (logging each byte count), the
    build then reads them (2 records), and a second ``download`` with the mirror removed
    returns without error because both raw files already match their pins.
    """
    _, mirror, content = _pinned_to_synthetic(tmp_path, monkeypatch)
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhnuki2022Dataset(
            root=str(tmp_path / "fresh"), genome=_genome()
        )
    raw = Path(dataset.raw_dir)
    assert {name: (raw / name).read_bytes() for name in content} == content
    copied = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and r.getMessage().startswith("Copied")
    ]
    assert copied == [
        f"Copied {m.MUTANT_FILE} from mirror ({len(content[m.MUTANT_FILE])} bytes, "
        "sha256 verified)",
        f"Copied {m.WT_FILE} from mirror ({len(content[m.WT_FILE])} bytes, "
        "sha256 verified)",
    ]
    assert len(dataset) == 2
    shutil.rmtree(mirror)
    dataset.download()
    assert {name: (raw / name).read_bytes() for name in content} == content


def test_build_refuses_a_copy_that_lands_different_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror bytes verify, but a copy that writes ``b"torn"`` instead lands in
    ``raw/``; ``process()`` re-hashes ``raw/`` before reading a row and refuses it with
    ``RawSha256MismatchError`` naming the raw file and the torn digest.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    _pinned_to_synthetic(tmp_path, monkeypatch)

    def torn_copy(src: str, dest: str) -> None:
        Path(dest).write_bytes(b"torn")

    monkeypatch.setattr(shutil, "copyfile", torn_copy)
    torn = hashlib.sha256(b"torn").hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.ScmdOhnuki2022Dataset(root=str(tmp_path / "torn"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {tmp_path / 'torn' / 'raw' / m.MUTANT_FILE}: "
        f"expected {m.MUTANT_SHA256}, observed {torn}"
    )
    assert list((tmp_path / "torn" / "processed").iterdir()) == []


def test_main_builds_under_data_root_and_prints_the_first_item(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` constructs the dataset at ``$DATA_ROOT/data/torchcell/scmd_ohnuki2022``
    (no other argument) and prints its length and first item; the class is a recorder.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[dict[str, Any]] = []

    class _Recorder:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)

        def __len__(self) -> int:
            return 7

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(m, "ScmdOhnuki2022Dataset", _Recorder)
    m.main()
    assert calls == [{"root": f"{tmp_path}/data/torchcell/scmd_ohnuki2022"}]
    assert capsys.readouterr().out == "len = 7\nitem[0]\n"
