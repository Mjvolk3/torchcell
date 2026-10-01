# tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py
# [[tests.torchcell.datasets.scerevisiae.test_ohnuki2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py
"""Hermetic build of the Ohnuki 2018 essential-gene heterozygote CalMorph loader.

Both TSVs are written into ``<root>/raw/`` so PyG never calls ``download()``. The genome
stub implements only ``resolve_gene_name``: YAL001C and YDR001C are current, YBR002C is
renamed to YBR001C, YCR001W is retired.

Features (4 of the 501): base ``A101_A``, ``C103_A1B``; CV ``ACV103_A1B``, ``CCV103_A1B``.

``ess1112data.tsv`` (ORF, A101_A, C103_A1B, ACV103_A1B, CCV103_A1B):

    YAL001C  1.0  2.0  0.5    0.25
    YBR002C  3.0  4.0  0.25   0.5
    YCR001W  5.0  6.0  0.125  0.75
    YDR001C  7.0  8.0  (blank) 1.0     kept; the blank CV cell is stored as 0.0

``wt114data.tsv`` (NAME + the same features): wt1 1.0 3.0 0.25 0.5; wt2 3.0 5.0 0.75 1.0;
wt3 5.0 7.0 n.d. 1.5. Means: A101_A 3.0, C103_A1B 5.0, ACV103_A1B 0.5 (``n.d.`` coerced
to NaN and skipped), CCV103_A1B 1.0.

2026.09.30 (Phase 16): the same matrices, plus

- the ledger (lines 236 and 188): "Ohnuki 2018: 4 essential-gene heterozygote strains
  (0 dropped for naming)" then "Processing Ohnuki 2018 CalMorph morphology data...".
- ``transform_item`` round trips through ``CalMorphExperiment`` and
  ``CalMorphExperimentReference``.
- an edge matrix (``A101_A``, ``ACV103_A1B``): YAL001C (1.0, 0.1), ``yal001c `` (2.0,
  0.2), a blank ORF (3.0, 0.3) and a whitespace-only ORF (4.0, 0.4); WT w1 (1.0, 0.3),
  w2 (3.0, 0.5), so the reference is A101_A 2.0 and ACV103_A1B (0.3 + 0.5) / 2 = 0.4.
  Two records, both YAL001C; the count line says 2.
- a WT feature that is "n.d." in every row has a NaN mean and the reference phenotype is
  refused: "CV measurement ACV103_A1B cannot be NaN".
- ``genome=None`` calls ``default_genome()`` once (line 230).
- ``download`` with ``_RAW_FILES`` replaced by pins for the synthetic matrices: both are
  copied from ``$DATA_ROOT/<_MIRROR_DIR>`` and a second call with the mirror removed
  re-verifies the raw copies.

2026.10.01 (issue #537): the Phase 16 findings are retired. Two spellings of one strain
are refused before anything is written; blank or whitespace-only ORF rows are dropped
with a counted warning; ``create_experiment`` raises ``NotImplementedError``; every
record is built before the store is opened, so a refused reference leaves no
``processed/lmdb``; ``TCV`` is no longer a CV prefix. The pinned matrices have 0
duplicates, 0 blank ORFs and 0 TCV columns (1112 x 501).

Finding still pinned here: a blank CalMorph cell is stored as 0.0 (see
``test_missing_calmorph_value_is_stored_as_zero``); the pinned matrix has 0 blank
cells, so a fix would not change the built records, but it is not an item of #537.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pydantic
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    EngineeredCopyNumberPerturbation,
    Environment,
    Genotype,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import ohnuki2018 as m
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)


@pytest.fixture(autouse=True)
def _no_tc_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build here reads ``raw/``; an inherited ``TC_DATA_URL`` would send it to
    the tc-data endpoint instead (``ExperimentDataset._download``).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)


_FEATURES = ["A101_A", "C103_A1B", "ACV103_A1B", "CCV103_A1B"]
_MUTANT_ROWS = [
    ["YAL001C", "1.0", "2.0", "0.5", "0.25"],
    ["YBR002C", "3.0", "4.0", "0.25", "0.5"],
    ["YCR001W", "5.0", "6.0", "0.125", "0.75"],
    ["YDR001C", "7.0", "8.0", "", "1.0"],
]
_WT_ROWS = [
    ["wt1", "1.0", "3.0", "0.25", "0.5"],
    ["wt2", "3.0", "5.0", "0.75", "1.0"],
    ["wt3", "5.0", "7.0", "n.d.", "1.5"],
]
_CURRENT = {"YAL001C", "YDR001C"}
_RENAMED = {"YBR002C": "YBR001C"}


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


def _root(tmp_path: Path, slug: str = "scmd_ohnuki2018") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_tsv(root / "raw" / "ess1112data.tsv", ["ORF", *_FEATURES], _MUTANT_ROWS)
    _write_tsv(root / "raw" / "wt114data.tsv", ["NAME", *_FEATURES], _WT_ROWS)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.ScmdOhnuki2018Dataset:
    return m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(
    media=Media(name="YPD", state="liquid", is_synthetic=False),
    temperature=Temperature(value=25),
)
_WT_PHENOTYPE = CalMorphPhenotype(
    calmorph={"A101_A": 3.0, "C103_A1B": 5.0},
    calmorph_coefficient_of_variation={"ACV103_A1B": 0.5, "CCV103_A1B": 1.0},
)
_REFERENCE = CalMorphExperimentReference(
    dataset_name="ScmdOhnuki2018Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_WT_PHENOTYPE,
).model_dump()
_PUBLICATION = Publication(
    pubmed_id="29768403",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/29768403/",
    doi="10.1371/journal.pbio.2005130",
    doi_url="https://doi.org/10.1371/journal.pbio.2005130",
).model_dump()


def _experiment(orf: str, values: list[float]) -> dict[str, Any]:
    return CalMorphExperiment(
        dataset_name="ScmdOhnuki2018Dataset",
        genotype=Genotype(
            perturbations=[
                EngineeredCopyNumberPerturbation(
                    systematic_gene_name=orf,
                    perturbed_gene_name=orf,
                    copy_number=1,
                    reference_copy_number=2,
                    marker="KanMX",
                )
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


def test_four_heterozygote_records_including_the_renamed_and_retired_names(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """Every row is retained: YBR002C is stored under YBR001C, YCR001W keeps its retired
    name. Each genotype is one 2 -> 1 copy-number perturbation with a KanMX marker on the
    diploid BY4743 reference; the reference phenotype is the WT mean.
    """
    assert len(dataset) == 4
    assert dataset[0]["experiment"] == _experiment("YAL001C", [1.0, 2.0, 0.5, 0.25])
    assert dataset[1]["experiment"] == _experiment("YBR001C", [3.0, 4.0, 0.25, 0.5])
    assert dataset[2]["experiment"] == _experiment("YCR001W", [5.0, 6.0, 0.125, 0.75])
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["reference"]["genome_reference"]["ploidy"] == "diploid"


def test_missing_calmorph_value_is_stored_as_zero(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """Finding: unlike Ohya 2005 and Ohnuki 2022, which drop a strain with any missing
    CalMorph value, this loader keeps the row and ``create_calmorph_experiment`` (source
    line 295) writes ``0.0`` for the blank cell, so YDR001C stores ACV103_A1B = 0.0.
    """
    assert dataset[3]["experiment"] == _experiment("YDR001C", [7.0, 8.0, 0.0, 1.0])


def test_side_files(dataset: m.ScmdOhnuki2018Dataset) -> None:
    """``data.csv`` keeps the blank cell blank (the 0.0 is introduced only in the record);
    one reference covers all four records; the gene set is the four stored names.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "ORF,A101_A,C103_A1B,ACV103_A1B,CCV103_A1B,systematic_gene_name,"
        "perturbed_gene_name\n"
        "YAL001C,1.0,2.0,0.5,0.25,YAL001C,YAL001C\n"
        "YBR002C,3.0,4.0,0.25,0.5,YBR001C,YBR001C\n"
        "YCR001W,5.0,6.0,0.125,0.75,YCR001W,YCR001W\n"
        "YDR001C,7.0,8.0,,1.0,YDR001C,YDR001C\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
        "YDR001C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3]]
    assert index[0]["reference"]["phenotype_reference"] == _WT_PHENOTYPE.model_dump()
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "scmd_ohnuki2018"
    assert manifest["loader_class"] == "ScmdOhnuki2018Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.ohnuki2018"


def test_build_refuses_present_files_off_the_pin_and_download_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #537's sweep): with the matrices already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies them against ``_RAW_FILES`` before a row
    is read; the synthetic ``ess1112data.tsv`` is refused with ``RawSha256MismatchError``
    naming it and both digests. On an empty root ``download()`` names the missing
    mirror path under ``$DATA_ROOT``.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / "ess1112data.tsv"
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected "
        f"{m._RAW_FILES['ess1112data.tsv']['sha256']}, observed {digest}"
    )
    mirror = tmp_path / "data_root" / m._MIRROR_DIR
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"ess1112data.tsv not found in the library mirror {mirror}. The SCMD2 portal "
            "is the historical source; recover the file and deposit it, then rebuild "
            "(sha256 verified)."
        ),
    ):
        m.ScmdOhnuki2018Dataset(root=str(tmp_path / "empty"), genome=_genome())


def test_items_retype_through_the_calmorph_classes(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    136 to 144), dumping back to exactly the stored dictionaries, copy-number
    perturbation and diploid reference included.
    """
    for index in range(4):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is CalMorphExperiment
        assert type(typed["reference"]) is CalMorphExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION


def test_generic_hooks_are_inert_and_create_experiment_raises(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """``preprocess_raw`` returns its frame unchanged. Contract (issue #537):
    ``create_experiment`` raises ``NotImplementedError`` naming the method that builds
    the records, instead of returning None.
    """
    frame = pd.DataFrame({"ORF": ["YAL001C"], "A101_A": [1.0]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"ORF": ["YAL001C"], "A101_A": [1.0]}
    with pytest.raises(NotImplementedError) as info:
        dataset.create_experiment()
    assert str(info.value) == (
        "ScmdOhnuki2018Dataset builds records with create_calmorph_experiment"
    )


def test_build_logs_the_strain_count_then_processing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """All four rows are kept, so the count line says 4."""
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)), genome=_genome())
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == [
        "Ohnuki 2018: 4 essential-gene heterozygote strains (0 dropped for naming)",
        "Processing Ohnuki 2018 CalMorph morphology data...",
    ]


def _edge_root(tmp_path: Path, mutant_rows: list[list[str]]) -> Path:
    root = tmp_path / "edges"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "ess1112data.tsv", ["ORF", "A101_A", "ACV103_A1B"], mutant_rows
    )
    _write_tsv(
        root / "raw" / "wt114data.tsv",
        ["NAME", "A101_A", "ACV103_A1B"],
        [["w1", "1.0", "0.3"], ["w2", "3.0", "0.5"]],
    )
    return root


def test_blank_orf_rows_are_dropped_with_a_counted_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Contract (issue #537): the blank and the whitespace-only ORF rows name no strain;
    both are dropped and counted in a warning ahead of the strain count (they used to
    vanish with no log line). The reference is A101_A 2.0 and ACV103_A1B
    (0.3 + 0.5) / 2 = 0.4. The pinned matrix has 0 blank ORFs of 1112.
    """
    root = _edge_root(
        tmp_path,
        [
            ["YAL001C", "1.0", "0.1"],
            ["", "3.0", "0.3"],
            ["  ", "4.0", "0.4"],
            ["ydr001c ", "2.0", "0.2"],
        ],
    )
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhnuki2018Dataset(root=str(root), genome=_genome())
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == m.log.name
    ] == [
        ("WARNING", "Ohnuki 2018: dropping 2 mutant row(s) with a blank ORF"),
        (
            "INFO",
            "Ohnuki 2018: 2 essential-gene heterozygote strains (0 dropped for naming)",
        ),
        ("INFO", "Processing Ohnuki 2018 CalMorph morphology data..."),
    ]
    reference = CalMorphExperimentReference(
        dataset_name="ScmdOhnuki2018Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4743", ploidy="diploid"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=CalMorphPhenotype(
            calmorph={"A101_A": 2.0},
            calmorph_coefficient_of_variation={"ACV103_A1B": 0.4},
        ),
    ).model_dump()
    assert len(dataset) == 2
    for index, (orf, base, cv) in enumerate(
        (("YAL001C", 1.0, 0.1), ("YDR001C", 2.0, 0.2))
    ):
        experiment = dataset[index]["experiment"]
        assert experiment["genotype"] == _experiment(orf, [0, 0, 0, 0])["genotype"]
        assert experiment["phenotype"] == (
            CalMorphPhenotype(
                calmorph={"A101_A": base},
                calmorph_coefficient_of_variation={"ACV103_A1B": cv},
            ).model_dump()
        )
        assert dataset[index]["reference"] == reference
    dataset.close_lmdb()


def test_two_spellings_of_one_strain_are_refused_before_the_store_opens(
    tmp_path: Path,
) -> None:
    """Contract (issue #537): YAL001C and ``yal001c `` name the same strain once
    stripped and uppercased, so the build refuses them by their raw spellings before
    ``data.csv`` or the store is written; a retry refuses again. The pinned matrix has
    0 such pairs among its 1112 rows.
    """
    root = _edge_root(tmp_path, [["YAL001C", "1.0", "0.1"], ["yal001c ", "2.0", "0.2"]])
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            m.ScmdOhnuki2018Dataset(root=str(root), genome=_genome())
        assert str(info.value) == (
            "Ohnuki 2018: the same strain is listed more than once (ORF stripped and "
            "uppercased): ['YAL001C', 'yal001c ']"
        )
        assert not (root / "preprocess" / "data.csv").exists()
        assert not (root / "processed" / "lmdb").exists()


def test_a_wt_feature_with_no_numeric_cell_refuses_the_reference(
    tmp_path: Path,
) -> None:
    """Every WT cell of ACV103_A1B is "n.d.", so its coerced mean is NaN and the reference
    ``CalMorphPhenotype`` refuses it by name. Contract (issue #537): every record is
    built before the store is opened, so the refusal leaves no ``processed/lmdb`` and a
    retry on the same root refuses again instead of serving 0 records.
    """
    root = tmp_path / "nd"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "ess1112data.tsv",
        ["ORF", "A101_A", "ACV103_A1B"],
        [["YAL001C", "1.0", "0.1"]],
    )
    _write_tsv(
        root / "raw" / "wt114data.tsv",
        ["NAME", "A101_A", "ACV103_A1B"],
        [["w1", "1.0", "n.d."], ["w2", "3.0", "n.d."]],
    )
    for _ in range(2):
        with pytest.raises(pydantic.ValidationError) as info:
            m.ScmdOhnuki2018Dataset(root=str(root), genome=_genome())
        assert [(e["loc"], e["msg"]) for e in info.value.errors()] == [
            (
                ("calmorph_coefficient_of_variation",),
                "Value error, CV measurement ACV103_A1B cannot be NaN",
            )
        ]
        assert not (root / "processed" / "lmdb").exists()


def test_missing_genome_is_built_once_by_default_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``genome=None`` the build calls ``default_genome()`` exactly once (line 230)
    and reconciles with what it returns, so YBR002C is stored as YBR001C as in the
    stubbed build.
    """
    stub: object = _StubGenome()
    calls: list[None] = []

    def fake_default_genome() -> object:
        calls.append(None)
        return stub

    monkeypatch.setattr(m, "default_genome", fake_default_genome)
    dataset = m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)))
    assert calls == [None]
    assert dataset.genome is stub
    assert dataset[1]["experiment"] == _experiment("YBR001C", [3.0, 4.0, 0.25, 0.5])


def test_download_copies_both_pinned_matrices_then_reverifies_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``_RAW_FILES`` pinned to the synthetic matrices, an empty root copies both
    from ``$DATA_ROOT/<_MIRROR_DIR>`` byte for byte and builds four records; a second
    ``download`` with the mirror removed only re-hashes the raw copies.
    """
    source = _root(tmp_path, "source") / "raw"
    data_root = tmp_path / "data_root"
    mirror = data_root / m._MIRROR_DIR
    mirror.mkdir(parents=True)
    content: dict[str, bytes] = {}
    for name in ("ess1112data.tsv", "wt114data.tsv"):
        content[name] = (source / name).read_bytes()
        (mirror / name).write_bytes(content[name])
    monkeypatch.setattr(
        m,
        "_RAW_FILES",
        {
            name: {
                "sha256": hashlib.sha256(data).hexdigest(),
                "id_column": m._RAW_FILES[name]["id_column"],
            }
            for name, data in content.items()
        },
    )
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    dataset = m.ScmdOhnuki2018Dataset(root=str(tmp_path / "fresh"), genome=_genome())
    raw = tmp_path / "fresh" / "raw"
    assert {name: (raw / name).read_bytes() for name in content} == content
    assert len(dataset) == 4
    for name in content:
        (mirror / name).unlink()
    dataset.download()
    assert {name: (raw / name).read_bytes() for name in content} == content
