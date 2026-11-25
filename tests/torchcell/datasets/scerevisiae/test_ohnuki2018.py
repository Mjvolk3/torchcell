# tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py
# [[tests.torchcell.datasets.scerevisiae.test_ohnuki2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ohnuki2018.py
"""Hermetic build of the Ohnuki 2018 essential-gene heterozygote CalMorph loader.

Both TSVs are written into ``<root>/raw/`` so PyG never calls ``download()``. No genome
is consulted (#1045): every ORF is stored as released (stripped, upper-cased).

Features (4 of the 501): base ``A101_A``, ``C103_A1B``; CV ``ACV103_A1B``, ``CCV103_A1B``.

``ess1112data.tsv`` (ORF, A101_A, C103_A1B, ACV103_A1B, CCV103_A1B):

    YAL001C  1.0  2.0  0.5    0.25
    YBR002C  3.0  4.0  0.25   0.5
    YCR001W  5.0  6.0  0.125  0.75
    YDR001C  7.0  8.0  (blank) 1.0     kept; the blank CV cell is key-absent

``wt114data.tsv`` (NAME + the same features): wt1 1.0 3.0 0.25 0.5; wt2 3.0 5.0 0.75 1.0;
wt3 5.0 7.0 (blank) 1.5. Means: A101_A 3.0, C103_A1B 5.0, ACV103_A1B 0.5 (the blank cell
skipped), CCV103_A1B 1.0.

2026.11.24 (per-replicate policy, #1178 / #1171): each wild-type row is also its own
record, after the four heterozygote records (indices 4, 5, 6): an EMPTY genotype on the
BY4743 reference genome with ``replicate_id`` = its ``NAME``; wt3's blank ACV103_A1B is
key-absent in its record. The heterozygote records and the shared reference are
unchanged. A blank or repeated ``NAME`` and a non-numeric wild-type cell are refused
before the store opens.

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
import os
import re
import shutil
from pathlib import Path
from typing import Any

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
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import ohnuki2018 as m
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
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
    ["wt3", "5.0", "7.0", "", "1.5"],
]


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
    return m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)))


_ENVIRONMENT = Environment(media=m.OHNUKI_YPD, temperature=Temperature(value=25))
_WT_PHENOTYPE = CalMorphPhenotype(
    calmorph={"A101_A": 3.0, "C103_A1B": 5.0},
    calmorph_coefficient_of_variation={"ACV103_A1B": 0.5, "CCV103_A1B": 1.0},
)
_REFERENCE = CalMorphExperimentReference(
    dataset_name="ScmdOhnuki2018Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae",
        strain="BY4743",
        ploidy="diploid",
        assembly_set=SGD_S288C_R64,
        assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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


def _experiment(orf: str, values: list[float | None]) -> dict[str, Any]:
    """A blank (None) cell is key-absent in the stored record."""
    cv = {
        key: value
        for key, value in zip(("ACV103_A1B", "CCV103_A1B"), values[2:], strict=True)
        if value is not None
    }
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
            calmorph={
                key: value
                for key, value in zip(("A101_A", "C103_A1B"), values[:2], strict=True)
                if value is not None
            },
            calmorph_coefficient_of_variation=cv,
        ),
    ).model_dump()


def test_four_heterozygote_records_including_the_renamed_and_retired_names(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """Every row is retained under its released name: YBR002C (an R64 rename) and
    YCR001W (retired) keep their names. Each genotype is one 2 -> 1 copy-number
    perturbation with a KanMX marker on the diploid BY4743 reference; the reference
    phenotype is the WT mean.
    """
    assert len(dataset) == 7
    assert dataset[0]["experiment"] == _experiment("YAL001C", [1.0, 2.0, 0.5, 0.25])
    assert dataset[1]["experiment"] == _experiment("YBR002C", [3.0, 4.0, 0.25, 0.5])
    assert dataset[2]["experiment"] == _experiment("YCR001W", [5.0, 6.0, 0.125, 0.75])
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["reference"]["genome_reference"]["ploidy"] == "diploid"
    assert all(
        dataset[i]["experiment"]["phenotype"]["replicate_id"] is None for i in range(4)
    )


def _wt_experiment(name: str, values: list[float | None]) -> dict[str, Any]:
    """One BY4743 wild-type replicate record: empty genotype, ``replicate_id`` = NAME."""
    experiment = _experiment("YAL001C", values)
    experiment["genotype"] = Genotype(perturbations=[]).model_dump()
    experiment["phenotype"]["replicate_id"] = name
    return CalMorphExperiment.model_validate(experiment).model_dump()


def test_each_wild_type_replicate_is_an_empty_genotype_record(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """#1178: wt1, wt2 and wt3 are records 4, 5 and 6 in release order, each an empty
    genotype on the BY4743 reference with ``replicate_id`` its NAME and its own released
    values (wt3's blank ACV103_A1B key-absent); they carry the same summary reference
    (``replicate_id`` None), and the per-trait mean over the three records is it.
    """
    expected: list[tuple[str, list[float | None]]] = [
        ("wt1", [1.0, 3.0, 0.25, 0.5]),
        ("wt2", [3.0, 5.0, 0.75, 1.0]),
        ("wt3", [5.0, 7.0, None, 1.5]),
    ]
    traits: dict[str, list[float]] = {}
    for offset, (name, values) in enumerate(expected):
        record = dataset[4 + offset]
        assert record["experiment"] == _wt_experiment(name, values)
        assert record["experiment"]["genotype"]["perturbations"] == []
        assert record["reference"] == _REFERENCE
        assert record["publication"] == _PUBLICATION
        phenotype = record["experiment"]["phenotype"]
        for key, value in {
            **phenotype["calmorph"],
            **phenotype["calmorph_coefficient_of_variation"],
        }.items():
            traits.setdefault(key, []).append(value)
    assert _REFERENCE["phenotype_reference"]["replicate_id"] is None
    assert {k: sum(v) / len(v) for k, v in traits.items()} == {
        **_WT_PHENOTYPE.calmorph,
        **(_WT_PHENOTYPE.calmorph_coefficient_of_variation or {}),
    }


def test_missing_calmorph_value_is_key_absent_never_zero(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """A blank cell is key-absent in the record (it used to be written as 0.0), so
    YDR001C stores no ACV103_A1B and its other three traits.
    """
    assert dataset[3]["experiment"] == _experiment("YDR001C", [7.0, 8.0, None, 1.0])
    assert (
        "ACV103_A1B"
        not in dataset[3]["experiment"]["phenotype"][
            "calmorph_coefficient_of_variation"
        ]
    )


def test_side_files(dataset: m.ScmdOhnuki2018Dataset) -> None:
    """``data.csv`` keeps the blank cell blank; one reference covers all seven records;
    the gene set is the four released names (a wild-type record perturbs no gene).
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "ORF,A101_A,C103_A1B,ACV103_A1B,CCV103_A1B,systematic_gene_name,"
        "perturbed_gene_name\n"
        "YAL001C,1.0,2.0,0.5,0.25,YAL001C,YAL001C\n"
        "YBR002C,3.0,4.0,0.25,0.5,YBR002C,YBR002C\n"
        "YCR001W,5.0,6.0,0.125,0.75,YCR001W,YCR001W\n"
        "YDR001C,7.0,8.0,,1.0,YDR001C,YDR001C\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR002C",
        "YCR001W",
        "YDR001C",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [list(range(7))]
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
    dataset = m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)))
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
        m.ScmdOhnuki2018Dataset(root=str(tmp_path / "empty"))


def test_items_retype_through_the_calmorph_classes(
    dataset: m.ScmdOhnuki2018Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    136 to 144), dumping back to exactly the stored dictionaries, copy-number
    perturbation and diploid reference included.
    """
    for index in range(7):
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
    """All four rows are kept, so the count line says 4; the three WT rows follow."""
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)))
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == [
        "Ohnuki 2018: 4 essential-gene heterozygote strains (0 dropped for naming)",
        "Ohnuki 2018: 3 wild-type replicates, one record each",
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
        dataset = m.ScmdOhnuki2018Dataset(root=str(root))
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == m.log.name
    ] == [
        ("WARNING", "Ohnuki 2018: dropping 2 mutant row(s) with a blank ORF"),
        (
            "INFO",
            "Ohnuki 2018: 2 essential-gene heterozygote strains (0 dropped for naming)",
        ),
        ("INFO", "Ohnuki 2018: 2 wild-type replicates, one record each"),
        ("INFO", "Processing Ohnuki 2018 CalMorph morphology data..."),
    ]
    reference = CalMorphExperimentReference(
        dataset_name="ScmdOhnuki2018Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="BY4743",
            ploidy="diploid",
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=CalMorphPhenotype(
            calmorph={"A101_A": 2.0},
            calmorph_coefficient_of_variation={"ACV103_A1B": 0.4},
        ),
    ).model_dump()
    assert len(dataset) == 4
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
            m.ScmdOhnuki2018Dataset(root=str(root))
        assert str(info.value) == (
            "Ohnuki 2018: the same strain is listed more than once (ORF stripped and "
            "uppercased): ['YAL001C', 'yal001c ']"
        )
        assert not (root / "preprocess" / "data.csv").exists()
        assert not (root / "processed" / "lmdb").exists()


def test_a_wt_feature_with_every_cell_blank_refuses_the_reference(
    tmp_path: Path,
) -> None:
    """Every WT cell of ACV103_A1B is blank, so its mean is NaN and the reference
    ``CalMorphPhenotype`` refuses it by name (each WT record alone would store the
    other traits with that key absent). Contract (issue #537): every record is
    built before the store is opened, so the refusal leaves no ``processed/lmdb`` and a
    retry on the same root refuses again instead of serving 0 records; ``data.csv`` is
    written only after every record is built, so it is absent too.
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
        [["w1", "1.0", ""], ["w2", "3.0", ""]],
    )
    for _ in range(2):
        with pytest.raises(pydantic.ValidationError) as info:
            m.ScmdOhnuki2018Dataset(root=str(root))
        assert [(e["loc"], e["msg"]) for e in info.value.errors()] == [
            (
                ("calmorph_coefficient_of_variation",),
                "Value error, CV measurement ACV103_A1B cannot be NaN",
            )
        ]
        assert not (root / "processed" / "lmdb").exists()
        assert not (root / "preprocess" / "data.csv").exists()


def test_a_tcv_column_is_a_base_parameter_and_refused_by_the_schema(
    tmp_path: Path,
) -> None:
    """Contract (issue #537, #494): CV parameters are prefixed CCV, ACV and DCV only
    (60 + 33 + 127 = 220 in the pinned matrices; no TCV column exists in either), so a
    ``TCV101_X`` column is not classed as a CV trait. It falls to the base traits, where
    the phenotype validator refuses it by name, and no store is written.
    """
    assert m._CV_PREFIXES == ("CCV", "ACV", "DCV")
    root = tmp_path / "tcv"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "ess1112data.tsv",
        ["ORF", "A101_A", "TCV101_X"],
        [["YAL001C", "1.0", "0.1"]],
    )
    _write_tsv(
        root / "raw" / "wt114data.tsv",
        ["NAME", "A101_A", "TCV101_X"],
        [["w1", "1.0", "0.3"]],
    )
    with pytest.raises(pydantic.ValidationError) as info:
        m.ScmdOhnuki2018Dataset(root=str(root))
    assert [(e["loc"], e["msg"]) for e in info.value.errors()] == [
        (
            ("calmorph",),
            "Value error, Invalid CalMorph base parameter: TCV101_X. Must be one of "
            "the 281 base parameters in CALMORPH_LABELS.",
        )
    ]
    assert not (root / "processed" / "lmdb").exists()


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
    dataset = m.ScmdOhnuki2018Dataset(root=str(tmp_path / "fresh"))
    raw = tmp_path / "fresh" / "raw"
    assert {name: (raw / name).read_bytes() for name in content} == content
    assert len(dataset) == 7
    for name in content:
        (mirror / name).unlink()
    dataset.download()
    assert {name: (raw / name).read_bytes() for name in content} == content


# --------------------------------------------------------------------------- #
# 2026.11.24: one record per released wild-type replicate (#1178 / #1171)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("wt_rows", "message"),
    [
        (
            [["w1", "1.0", "0.3"], ["", "2.0", "0.3"]],
            "Ohnuki 2018: 1 wild-type row(s) with a blank NAME; the NAME is the "
            "replicate's replicate_id",
        ),
        (
            [["w1", "1.0", "0.3"], ["w1", "2.0", "0.3"]],
            "Ohnuki 2018: wild-type replicate NAME(s) listed more than once: "
            "['w1', 'w1']",
        ),
    ],
)
def test_a_blank_or_repeated_wild_type_name_is_refused_before_the_store_opens(
    tmp_path: Path, wt_rows: list[list[str]], message: str
) -> None:
    """The NAME is the replicate's ``replicate_id``, so a blank or repeated one refuses
    (0 blank and 0 repeated among the 114 pinned rows) and nothing is written.
    """
    root = _edge_root(tmp_path, [["YAL001C", "1.0", "0.1"]])
    _write_tsv(
        root / "raw" / "wt114data.tsv", ["NAME", "A101_A", "ACV103_A1B"], wt_rows
    )
    with pytest.raises(RuntimeError) as info:
        m.ScmdOhnuki2018Dataset(root=str(root))
    assert str(info.value) == message
    assert not (root / "processed" / "lmdb").exists()
    assert not (root / "preprocess" / "data.csv").exists()


def test_a_non_numeric_wild_type_cell_refuses_before_the_store_opens(
    tmp_path: Path,
) -> None:
    """A wild-type cell ``n.d.`` is no released value for that replicate's record, so it
    refuses (the earlier per-strain build coerced it to NaN inside the reference mean;
    the pinned matrix has 0 such cells).
    """
    root = _edge_root(tmp_path, [["YAL001C", "1.0", "0.1"]])
    _write_tsv(
        root / "raw" / "wt114data.tsv",
        ["NAME", "A101_A", "ACV103_A1B"],
        [["w1", "1.0", "0.3"], ["w2", "3.0", "n.d."]],
    )
    with pytest.raises(
        ValueError, match=r'^Unable to parse string "n.d." at position 1$'
    ):
        m.ScmdOhnuki2018Dataset(root=str(root))
    assert not (root / "processed" / "lmdb").exists()


_MIRROR_DATA = Path(os.environ.get("DATA_ROOT", "/nonexistent"), m._MIRROR_DIR)


@pytest.mark.data
@pytest.mark.skipif(
    not (_MIRROR_DATA / "wt114data.tsv").exists(),
    reason=f"requires the Ohnuki 2018 mirror matrices under {_MIRROR_DATA}",
)
def test_the_pinned_release_holds_1112_heterozygotes_and_114_named_wild_types() -> None:
    """The pinned matrices hold 1112 heterozygote rows and 114 BY4743 wild-type rows
    whose 114 NAMEs are distinct ``WBD<nnn><letter>`` labels (43 K, 40 S, 25 M, 6 N), 0
    blank cells in either.
    """
    for name, spec in m._RAW_FILES.items():
        digest = hashlib.sha256((_MIRROR_DATA / name).read_bytes()).hexdigest()
        assert digest == spec["sha256"]
    mutant = pd.read_csv(_MIRROR_DATA / "ess1112data.tsv", sep="\t")
    wt = pd.read_csv(_MIRROR_DATA / "wt114data.tsv", sep="\t")
    assert mutant.shape == (1112, 502)
    assert wt.shape == (114, 502)
    assert wt["NAME"].is_unique
    assert wt["NAME"].str.fullmatch(r"WBD\d{3}[KSMN]").all()
    assert wt["NAME"].str[-1].value_counts().to_dict() == {
        "K": 43,
        "S": 40,
        "M": 25,
        "N": 6,
    }
    assert int(wt.isna().sum().sum()) == 0
    assert int(mutant.isna().sum().sum()) == 0


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    not (_MIRROR_DATA / "wt114data.tsv").exists(),
    reason=f"requires the Ohnuki 2018 mirror matrices under {_MIRROR_DATA}",
)
def test_the_full_build_stores_1226_records_and_the_wild_type_mean_is_the_reference(
    tmp_path: Path,
) -> None:
    """The pinned release builds 1226 records: 1112 heterozygotes (``replicate_id``
    None) then 114 empty-genotype wild-type replicates with their NAME as
    ``replicate_id``; the per-trait mean over the 114 equals the shared reference to
    1e-12 relative (summation order only).
    """
    root = tmp_path / "scmd_ohnuki2018"
    (root / "raw").mkdir(parents=True)
    for name in m._RAW_FILES:
        shutil.copyfile(_MIRROR_DATA / name, root / "raw" / name)
    built = m.ScmdOhnuki2018Dataset(root=str(root))
    assert len(built) == 1226
    names = pd.read_csv(root / "raw" / "wt114data.tsv", sep="\t")["NAME"].tolist()
    assert all(
        built[i]["experiment"]["phenotype"]["replicate_id"] is None for i in range(1112)
    )
    sums: dict[str, float] = {}
    for offset in range(114):
        experiment = built[1112 + offset]["experiment"]
        assert experiment["genotype"]["perturbations"] == []
        assert experiment["phenotype"]["replicate_id"] == names[offset]
        for key, value in {
            **experiment["phenotype"]["calmorph"],
            **experiment["phenotype"]["calmorph_coefficient_of_variation"],
        }.items():
            sums[key] = sums.get(key, 0.0) + value
    reference = built[0]["reference"]["phenotype_reference"]
    assert reference["replicate_id"] is None
    summary = {
        **reference["calmorph"],
        **reference["calmorph_coefficient_of_variation"],
    }
    assert set(sums) == set(summary)
    for key, value in summary.items():
        assert sums[key] / 114 == pytest.approx(value, rel=1e-12)
    built.close_lmdb()
