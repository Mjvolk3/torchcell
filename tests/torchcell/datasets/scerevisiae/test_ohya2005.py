# tests/torchcell/datasets/scerevisiae/test_ohya2005.py
# [[tests.torchcell.datasets.scerevisiae.test_ohya2005]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ohya2005.py
"""Hermetic build of the Ohya 2005 CalMorph loader on two synthetic SCMD matrices.

Both TSVs are written into ``<root>/raw/`` so PyG never calls ``download()``. The genome
stub implements only ``resolve_gene_name``, returning real ``GeneNameResolution`` objects:
YAL001C, YDR001C, YER001W, YFR001W are current; YBR002C is renamed to YBR001C; YER002W is
renamed to YER001W (which is also a strain, so both keep their names); YCR001W is retired.

Features (4 of the 501): base ``A101_A``, ``C103_A1B``; CV ``ACV103_A1B``, ``CCV103_A1B``.

``mt4718data.tsv`` (ORF, A101_A, C103_A1B, ACV103_A1B, CCV103_A1B):

    YAL001C  1.0   2.0   0.5    0.25
    YBR002C  3.0   4.0   0.25   0.5
    YCR001W  5.0   6.0   0.125  0.75
    YDR001C  7.0   8.0   (blank) 1.0     missing value -> dropped whole
    YER001W  9.0   10.0  0.75   1.25
    YER002W  11.0  12.0  0.875  1.5
    yfr001w  13.0  14.0  1.0    1.75     stripped and uppercased

``wt122data.tsv`` (NAME + the same features): wt1 1.0 3.0 0.25 0.5; wt2 3.0 5.0 0.75 1.0;
wt3 5.0 7.0 n.d. 1.5. Means: A101_A 3.0, C103_A1B 5.0, ACV103_A1B (0.25 + 0.75) / 2 = 0.5
with ``n.d.`` coerced to NaN and skipped, CCV103_A1B 1.0.

2026.09.30 (Phase 16): the same matrices, plus

- the ledger (lines 260 and 274): "Ohya 2005: dropping 1 mutant row(s) with missing
  CalMorph values: ['YDR001C']", "Ohya 2005: 6 non-essential deletion strains (0 dropped
  for naming)", then "Processing Ohya 2005 CalMorph morphology data..."; with 21
  incomplete rows the count is 21 and the list stops at the first 20 (``[:20]``).
- ``transform_item`` round trips through ``CalMorphExperiment`` and
  ``CalMorphExperimentReference``.
- a clean two-feature matrix (``A101_A`` and ``ACV103_A1B``) with YAL001C (1.0, 0.1) and
  ``yal001c`` (2.0, 0.2); WT w1 (1.0, 0.3) and w2 (3.0, 0.5), so the reference is
  A101_A (1.0 + 3.0) / 2 = 2.0 and ACV103_A1B (0.3 + 0.5) / 2 = 0.4. No drop warning.
- a ``TCV101_X`` column is routed to the CV traits and refused by the phenotype
  validator: "Invalid CalMorph CV parameter: TCV101_X. Must be one of the 220 CV
  parameters in CALMORPH_STATISTICS."
- a non-numeric mutant cell ("x") refuses with "could not convert string to float: 'x'"
  (``float(row[col])``, line 330).
- ``genome=None`` calls ``default_genome()`` once (line 268).
- ``download`` with ``_RAW_FILES`` replaced by pins for the synthetic matrices: both are
  copied from ``$DATA_ROOT/<_MIRROR_DIR>`` and a second call with the mirror removed
  re-verifies the raw copies without error.

2026.10.01 (issues #537 and #546): two spellings of one strain are refused before
anything is written; blank ORF rows are dropped with a counted warning; ``TCV`` is no
longer a CV prefix, so a ``TCV*`` column is refused as a base parameter;
``create_experiment`` raises ``NotImplementedError``; every record is built before the
store is opened, so a refusal (a non-numeric cell included) leaves no ``processed/lmdb``;
the store is written through ``_intern_record`` and resolves to exactly the inline
records. The pinned matrices have 0 duplicates, 0 blank ORFs, 0 TCV columns and 0
non-numeric cells (4718 x 501).

Finding still pinned here: the publication is Ohya 2005 (PMID 16365294) while the same
verification records that the distributed matrices are the Suzuki 2018 CalMorph 1.2
re-analysis of the 2005 images (#491).
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, cast

import lmdb
import pandas as pd
import pydantic
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
    Publication,
    ReferenceGenome,
    SourceType,
    Temperature,
)
from torchcell.datasets.scerevisiae import ohya2005 as m
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
    ["YER001W", "9.0", "10.0", "0.75", "1.25"],
    ["YER002W", "11.0", "12.0", "0.875", "1.5"],
    ["yfr001w", "13.0", "14.0", "1.0", "1.75"],
]
_WT_ROWS = [
    ["wt1", "1.0", "3.0", "0.25", "0.5"],
    ["wt2", "3.0", "5.0", "0.75", "1.0"],
    ["wt3", "5.0", "7.0", "n.d.", "1.5"],
]
_CURRENT = {"YAL001C", "YDR001C", "YER001W", "YFR001W"}
_RENAMED = {"YBR002C": "YBR001C", "YER002W": "YER001W"}


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


def _root(tmp_path: Path, slug: str = "scmd_ohya2005") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_tsv(root / "raw" / "mt4718data.tsv", ["ORF", *_FEATURES], _MUTANT_ROWS)
    _write_tsv(root / "raw" / "wt122data.tsv", ["NAME", *_FEATURES], _WT_ROWS)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.ScmdOhya2005Dataset:
    return m.ScmdOhya2005Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(media=m.OHYA_YPD, temperature=Temperature(value=25))
_WT_PHENOTYPE = CalMorphPhenotype(
    calmorph={"A101_A": 3.0, "C103_A1B": 5.0},
    calmorph_coefficient_of_variation={"ACV103_A1B": 0.5, "CCV103_A1B": 1.0},
)
_REFERENCE = CalMorphExperimentReference(
    dataset_name="ScmdOhya2005Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4741"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_WT_PHENOTYPE,
).model_dump()
_PUBLICATION = Publication(
    pubmed_id="16365294",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/16365294/",
    doi="10.1073/pnas.0509436102",
    doi_url="https://www.pnas.org/doi/10.1073/pnas.0509436102",
).model_dump()


def _experiment(orf: str, values: list[float]) -> dict[str, Any]:
    return CalMorphExperiment(
        dataset_name="ScmdOhya2005Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
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


def test_six_records_retain_every_name_and_drop_the_incomplete_row(
    dataset: m.ScmdOhya2005Dataset,
) -> None:
    """YDR001C (blank ACV103_A1B) is the only drop. YBR002C is stored under YBR001C;
    YER002W keeps its legacy name because remapping it to YER001W would collide with
    that strain; YCR001W (retired) and yfr001w (uppercased) pass through. Base and CV
    traits are split by prefix into the two phenotype dicts.
    """
    assert len(dataset) == 6
    expected = [
        ("YAL001C", [1.0, 2.0, 0.5, 0.25]),
        ("YBR001C", [3.0, 4.0, 0.25, 0.5]),
        ("YCR001W", [5.0, 6.0, 0.125, 0.75]),
        ("YER001W", [9.0, 10.0, 0.75, 1.25]),
        ("YER002W", [11.0, 12.0, 0.875, 1.5]),
        ("YFR001W", [13.0, 14.0, 1.0, 1.75]),
    ]
    for i, (orf, values) in enumerate(expected):
        assert dataset[i]["experiment"] == _experiment(orf, values)
        assert dataset[i]["reference"] == _REFERENCE
        assert dataset[i]["publication"] == _PUBLICATION


def test_reference_is_the_wt_mean_with_non_numeric_cells_skipped(
    dataset: m.ScmdOhya2005Dataset,
) -> None:
    """One shared reference (index entry [0..5]) whose ACV103_A1B mean is 0.5 over the two
    numeric WT cells, the ``n.d.`` cell having been coerced to NaN.
    """
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3, 4, 5]]
    assert index[0]["reference"]["phenotype_reference"] == _WT_PHENOTYPE.model_dump()


def test_side_files(dataset: m.ScmdOhya2005Dataset) -> None:
    """``data.csv`` is the retained mutant matrix with the source ORF spelling kept and
    the two resolved-name columns appended; the gene set is the six stored names.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "ORF,A101_A,C103_A1B,ACV103_A1B,CCV103_A1B,systematic_gene_name,"
        "perturbed_gene_name\n"
        "YAL001C,1.0,2.0,0.5,0.25,YAL001C,YAL001C\n"
        "YBR002C,3.0,4.0,0.25,0.5,YBR001C,YBR001C\n"
        "YCR001W,5.0,6.0,0.125,0.75,YCR001W,YCR001W\n"
        "YER001W,9.0,10.0,0.75,1.25,YER001W,YER001W\n"
        "YER002W,11.0,12.0,0.875,1.5,YER002W,YER002W\n"
        "yfr001w,13.0,14.0,1.0,1.75,YFR001W,YFR001W\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
        "YER001W",
        "YER002W",
        "YFR001W",
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "scmd_ohya2005"
    assert manifest["loader_class"] == "ScmdOhya2005Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.ohya2005"
    assert {"CalMorphExperiment", "CalMorphPhenotype"} <= set(manifest["closure"])


def test_build_refuses_present_files_off_the_pin_and_download_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #537's sweep): with the matrices already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies them against ``_RAW_FILES`` before a row
    is read; the synthetic ``mt4718data.tsv`` is refused with ``RawSha256MismatchError``
    naming it and both digests. On an empty root ``download()`` names the missing
    mirror path under ``$DATA_ROOT``.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.ScmdOhya2005Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / "mt4718data.tsv"
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected "
        f"{m._RAW_FILES['mt4718data.tsv']['sha256']}, observed {digest}"
    )
    mirror = tmp_path / "data_root" / m._MIRROR_DIR
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"mt4718data.tsv not found in the library mirror {mirror}. The SCMD portal "
            "is the historical source; recover the file and deposit it, then rebuild "
            "(sha256 verified)."
        ),
    ):
        m.ScmdOhya2005Dataset(root=str(tmp_path / "empty"), genome=_genome())


def test_items_retype_through_the_calmorph_classes(
    dataset: m.ScmdOhya2005Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    149 to 157), dumping back to exactly the stored dictionaries. A fitness class would
    drop the ``calmorph`` traits and fail the round trip.
    """
    for index in range(6):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is CalMorphExperiment
        assert type(typed["reference"]) is CalMorphExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION


def test_generic_hooks_are_inert_and_create_experiment_raises(
    dataset: m.ScmdOhya2005Dataset,
) -> None:
    """``preprocess_raw`` returns its frame unchanged. Contract (issue #537):
    ``create_experiment`` raises ``NotImplementedError`` naming the method that builds
    the records, as the sibling Ohnuki 2022 loader does, instead of returning None.
    """
    frame = pd.DataFrame({"ORF": ["YAL001C"], "A101_A": [1.0]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"ORF": ["YAL001C"], "A101_A": [1.0]}
    with pytest.raises(NotImplementedError) as info:
        dataset.create_experiment()
    assert str(info.value) == (
        "ScmdOhya2005Dataset builds records with create_calmorph_experiment"
    )


def test_publication_is_ohya_2005_although_the_matrix_is_the_suzuki_reanalysis(
    dataset: m.ScmdOhya2005Dataset,
) -> None:
    """Finding: every record cites Ohya 2005 (PMID 16365294, PNAS doi URL), as the module
    docstring claims ("Suzuki et al. 2018 ... merely REUSED this same dataset"). The
    2026.09.29 verification in ``notes/torchcell.datasets.scerevisiae.ohya2005.md``
    records that the distributed matrices are the Suzuki 2018 CalMorph 1.2 re-analysis
    (PMID 29458326) of the 2005 images (#491). Pinned until the loader records both.
    """
    publication = dataset[0]["publication"]
    assert publication == {
        "pubmed_id": "16365294",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/16365294/",
        "doi": "10.1073/pnas.0509436102",
        "doi_url": "https://www.pnas.org/doi/10.1073/pnas.0509436102",
        # a journal article: the non-journal identity fields stay unset (1.7.0)
        "source_type": SourceType.journal_article,
        "title": None,
        "identifier": None,
        "identifier_url": None,
    }


def test_drop_ledger_is_logged_before_the_strain_count(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The incomplete YDR001C row is named as a warning, then the six kept strains, then
    the processing line.
    """
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.ScmdOhya2005Dataset(root=str(_root(tmp_path)), genome=_genome())
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == m.log.name
    ] == [
        (
            "WARNING",
            "Ohya 2005: dropping 1 mutant row(s) with missing CalMorph values: "
            "['YDR001C']",
        ),
        ("INFO", "Ohya 2005: 6 non-essential deletion strains (0 dropped for naming)"),
        ("INFO", "Processing Ohya 2005 CalMorph morphology data..."),
    ]


def test_drop_warning_counts_every_incomplete_row_but_lists_the_first_20(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """21 incomplete rows (Y00..Y20, blank A101_A) and one complete row: the warning says
    21 and names Y00 to Y19 only (``tolist()[:20]``, line 263); one strain is kept.
    """
    root = tmp_path / "many_blank"
    (root / "raw").mkdir(parents=True)
    rows = [[f"Y{k:02d}", ""] for k in range(21)] + [["YAL001C", "1.0"]]
    _write_tsv(root / "raw" / "mt4718data.tsv", ["ORF", "A101_A"], rows)
    _write_tsv(root / "raw" / "wt122data.tsv", ["NAME", "A101_A"], [["wt1", "2.0"]])
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and r.levelname == "WARNING"
    ]
    assert warnings == [
        "Ohya 2005: dropping 21 mutant row(s) with missing CalMorph values: "
        f"{[f'Y{k:02d}' for k in range(20)]}"
    ]
    assert len(dataset) == 1
    dataset.close_lmdb()


def test_clean_matrix_logs_no_drop_and_shares_the_wt_mean_reference(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No drop warning when every row is complete; the reference is A101_A
    (1.0 + 3.0) / 2 = 2.0 and ACV103_A1B (0.3 + 0.5) / 2 = 0.4 for both strains.
    """
    root = tmp_path / "clean"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", "A101_A", "ACV103_A1B"],
        [["YAL001C", "1.0", "0.1"], ["ydr001c", "2.0", "0.2"]],
    )
    _write_tsv(
        root / "raw" / "wt122data.tsv",
        ["NAME", "A101_A", "ACV103_A1B"],
        [["w1", "1.0", "0.3"], ["w2", "3.0", "0.5"]],
    )
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == [
        "Ohya 2005: 2 non-essential deletion strains (0 dropped for naming)",
        "Processing Ohya 2005 CalMorph morphology data...",
    ]
    reference = CalMorphExperimentReference(
        dataset_name="ScmdOhya2005Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=CalMorphPhenotype(
            calmorph={"A101_A": 2.0},
            calmorph_coefficient_of_variation={"ACV103_A1B": 0.4},
        ),
    ).model_dump()
    for index, (orf, base, cv) in enumerate(
        (("YAL001C", 1.0, 0.1), ("YDR001C", 2.0, 0.2))
    ):
        assert dataset[index]["experiment"] == (
            CalMorphExperiment(
                dataset_name="ScmdOhya2005Dataset",
                genotype=Genotype(
                    perturbations=[
                        KanMxDeletionPerturbation(
                            systematic_gene_name=orf, perturbed_gene_name=orf
                        )
                    ]
                ),
                environment=_ENVIRONMENT,
                phenotype=CalMorphPhenotype(
                    calmorph={"A101_A": base},
                    calmorph_coefficient_of_variation={"ACV103_A1B": cv},
                ),
            ).model_dump()
        )
        assert dataset[index]["reference"] == reference
    dataset.close_lmdb()


def test_two_spellings_of_one_strain_are_refused_before_the_store_opens(
    tmp_path: Path,
) -> None:
    """Contract (issue #537): YAL001C and `` yal001c`` name the same strain once
    stripped and uppercased, so the build refuses them by their raw spellings before
    ``data.csv`` or the store is written; a retry refuses again. The pinned matrix
    has 0 such pairs among its 4718 rows (every ORF is lowercase there, all distinct).
    """
    root = tmp_path / "dups"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", "A101_A"],
        [["YAL001C", "1.0"], [" yal001c", "2.0"], ["YDR001C", "3.0"]],
    )
    _write_tsv(root / "raw" / "wt122data.tsv", ["NAME", "A101_A"], [["w1", "1.0"]])
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
        assert str(info.value) == (
            "Ohya 2005: the same strain is listed more than once (ORF stripped and "
            "uppercased): ['YAL001C', ' yal001c']"
        )
        assert not (root / "preprocess" / "data.csv").exists()
        assert not (root / "processed" / "lmdb").exists()


def test_blank_orf_rows_are_dropped_with_a_counted_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Contract (issue #537): a blank and a whitespace-only ORF name no strain; both are
    dropped and counted in a warning ahead of the strain count. The pinned matrix has
    0 blank ORFs of 4718, so the built records do not change.
    """
    root = tmp_path / "blank_orf"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", "A101_A"],
        [["YAL001C", "1.0"], ["", "2.0"], ["  ", "3.0"]],
    )
    _write_tsv(root / "raw" / "wt122data.tsv", ["NAME", "A101_A"], [["w1", "1.0"]])
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
    assert [
        (r.levelname, r.getMessage()) for r in caplog.records if r.name == m.log.name
    ] == [
        ("WARNING", "Ohya 2005: dropping 2 mutant row(s) with a blank ORF"),
        ("INFO", "Ohya 2005: 1 non-essential deletion strains (0 dropped for naming)"),
        ("INFO", "Processing Ohya 2005 CalMorph morphology data..."),
    ]
    assert len(dataset) == 1
    assert dataset[0]["experiment"]["phenotype"]["calmorph"] == {"A101_A": 1.0}
    dataset.close_lmdb()


def test_a_tcv_column_is_a_base_parameter_and_refused_by_the_schema(
    tmp_path: Path,
) -> None:
    """Contract (issue #537, #494): CV parameters are prefixed CCV, ACV and DCV only
    (60 + 33 + 127 = 220 in the pinned matrices; no TCV column exists in either), so a
    ``TCV101_X`` column is not classed as a CV trait. It falls to the base traits, where
    the phenotype validator refuses it by name.
    """
    assert m._CV_PREFIXES == ("CCV", "ACV", "DCV")
    root = tmp_path / "tcv"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", "A101_A", "TCV101_X"],
        [["YAL001C", "1.0", "0.1"]],
    )
    _write_tsv(
        root / "raw" / "wt122data.tsv",
        ["NAME", "A101_A", "TCV101_X"],
        [["w1", "1.0", "0.3"]],
    )
    with pytest.raises(pydantic.ValidationError) as info:
        m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
    assert [(e["loc"], e["msg"]) for e in info.value.errors()] == [
        (
            ("calmorph",),
            "Value error, Invalid CalMorph base parameter: TCV101_X. Must be one of "
            "the 281 base parameters in CALMORPH_LABELS.",
        )
    ]
    assert not (root / "processed" / "lmdb").exists()


def test_non_numeric_mutant_cell_refuses_before_the_store_opens(tmp_path: Path) -> None:
    """A mutant cell "x" passes the completeness check (it is not NaN) and then fails
    ``float(row[col])`` with Python's own message. Contract (issue #537): every record
    is built before ``data.csv`` or the store is written, so the refusal leaves neither
    and a retry refuses again (the pinned matrix has 0 non-numeric cells).
    """
    root = tmp_path / "bad_cell"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / "mt4718data.tsv",
        ["ORF", "A101_A"],
        [["YAL001C", "1.0"], ["YCR001W", "x"]],
    )
    _write_tsv(root / "raw" / "wt122data.tsv", ["NAME", "A101_A"], [["wt1", "2.0"]])
    for _ in range(2):
        with pytest.raises(
            ValueError, match=r"^could not convert string to float: 'x'$"
        ):
            m.ScmdOhya2005Dataset(root=str(root), genome=_genome())
        assert not (root / "processed" / "lmdb").exists()
        assert not (root / "preprocess" / "data.csv").exists()


def test_missing_genome_is_built_once_by_default_genome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``genome=None`` the build calls ``default_genome()`` exactly once (line 268)
    and reconciles with what it returns, so the records match the stubbed build.
    """
    stub: object = _StubGenome()
    calls: list[None] = []

    def fake_default_genome() -> object:
        calls.append(None)
        return stub

    monkeypatch.setattr(m, "default_genome", fake_default_genome)
    dataset = m.ScmdOhya2005Dataset(root=str(_root(tmp_path)))
    assert calls == [None]
    assert dataset.genome is stub
    assert [
        dataset[i]["experiment"]["genotype"]["perturbations"][0]["systematic_gene_name"]
        for i in range(len(dataset))
    ] == ["YAL001C", "YBR001C", "YCR001W", "YER001W", "YER002W", "YFR001W"]


def test_download_copies_both_pinned_matrices_then_reverifies_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``_RAW_FILES`` pinned to the synthetic matrices, an empty root copies both
    from ``$DATA_ROOT/<_MIRROR_DIR>`` byte for byte and builds six records; a second
    ``download`` with the mirror removed only re-hashes the raw copies.
    """
    source = _root(tmp_path, "source") / "raw"
    data_root = tmp_path / "data_root"
    mirror = data_root / m._MIRROR_DIR
    mirror.mkdir(parents=True)
    content: dict[str, bytes] = {}
    for name in ("mt4718data.tsv", "wt122data.tsv"):
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
    dataset = m.ScmdOhya2005Dataset(root=str(tmp_path / "fresh"), genome=_genome())
    raw = tmp_path / "fresh" / "raw"
    assert {name: (raw / name).read_bytes() for name in content} == content
    assert len(dataset) == 6
    for name in content:
        (mirror / name).unlink()
    dataset.download()
    assert {name: (raw / name).read_bytes() for name in content} == content


def _raw_records(processed: Path) -> list[dict[str, Any]]:
    """Every record of a closed store, unpickled as stored (``$ref`` left in place)."""
    env = lmdb.open(str(processed / "lmdb"), readonly=True, lock=False)
    records: list[dict[str, Any]] = []
    with env.begin() as txn:
        for index in range(txn.stat()["entries"]):
            raw = txn.get(f"{index}".encode())
            assert raw is not None, f"record {index} is missing from the store"
            records.append(pickle.loads(raw))
    env.close()
    return records


def test_the_interned_store_resolves_to_exactly_the_inline_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #546): ``process`` writes through ``_intern_record``. The same
    matrices are built twice: once as shipped, once with ``_intern_record`` replaced by
    the earlier inline writer (``pickle.dumps`` of the three dumps). The shipped store
    has a sibling ``interned`` env holding exactly two objects, the shared reference and
    the shared environment (each >= 512 bytes of canonical JSON), and each record
    carries ``{"$ref": <sha256 of its canonical JSON>, "name": ...}`` in their place;
    the publication stays inline. The inline build's ``interned`` env is empty. Read
    back through ``get_single_item`` the six records of the two stores are exactly
    equal.

    2026.10.02 (issue #622): the environment now carries the sourced ``OHYA_YPD`` (three
    components, five SourcedValues) instead of a name-only stub, so it crosses the
    interning threshold; before, it stayed inline.
    """
    interned = m.ScmdOhya2005Dataset(
        root=str(_root(tmp_path, "interned")), genome=_genome()
    )

    def inline_record(
        self: m.ScmdOhya2005Dataset,
        experiment: CalMorphExperiment,
        reference: CalMorphExperimentReference,
        publication: Publication,
        itxn: Any,
    ) -> bytes:
        return pickle.dumps(
            {
                "experiment": experiment.model_dump(),
                "reference": reference.model_dump(),
                "publication": publication.model_dump(),
            }
        )

    with monkeypatch.context() as patch:
        patch.setattr(m.ScmdOhya2005Dataset, "_intern_record", inline_record)
        inline = m.ScmdOhya2005Dataset(
            root=str(_root(tmp_path, "inline")), genome=_genome()
        )
    assert len(interned) == len(inline) == 6
    assert [interned[i] for i in range(6)] == [inline[i] for i in range(6)]
    interned.close_lmdb()
    inline.close_lmdb()

    digest = hashlib.sha256(
        json.dumps(
            CalMorphExperimentReference.model_validate(_REFERENCE).model_dump(
                mode="json"
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    stored = _raw_records(tmp_path / "interned" / "processed")
    assert [r["reference"] for r in stored] == [
        {"$ref": digest, "name": "ScmdOhya2005Dataset"}
    ] * 6
    assert [r["publication"] for r in stored] == [_PUBLICATION] * 6
    env_digest = hashlib.sha256(
        json.dumps(_ENVIRONMENT.model_dump(mode="json"), sort_keys=True).encode()
    ).hexdigest()
    assert [r["experiment"]["environment"] for r in stored] == [
        {"$ref": env_digest, "name": m.OHYA_YPD.name}
    ] * 6
    ienv = lmdb.open(
        str(tmp_path / "interned" / "processed" / "interned"), readonly=True, lock=False
    )
    with ienv.begin() as txn:
        assert sorted(key.decode() for key, _ in txn.cursor()) == sorted(
            [digest, env_digest]
        )
        raw_env = txn.get(env_digest.encode())
        assert raw_env is not None, "the interned environment is missing"
        assert pickle.loads(raw_env) == _ENVIRONMENT.model_dump()
        raw = txn.get(digest.encode())
        assert raw is not None, "the interned reference is missing"
        assert pickle.loads(raw) == _REFERENCE
    ienv.close()
    ienv = lmdb.open(
        str(tmp_path / "inline" / "processed" / "interned"), readonly=True, lock=False
    )
    with ienv.begin() as txn:
        assert txn.stat()["entries"] == 0
    ienv.close()
    assert _raw_records(tmp_path / "inline" / "processed")[0]["reference"] == (
        _REFERENCE
    )
