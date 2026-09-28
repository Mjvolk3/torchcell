# tests/torchcell/datasets/scerevisiae/test_ozaydin2013_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_ozaydin2013_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_ozaydin2013_synthetic.py
"""Hermetic build of the Ozaydin 2013 carotenoid visual-screen loader on a synthetic SI.

The SI workbook is written with openpyxl into ``<root>/raw/`` so PyG never calls
``download()``; ``process()`` runs for real. No genome is involved (the loader validates
ORF names with a regex only).

Sheet ``Color scores of all deletions`` (ORF name, Strain, Color, Comment):

    YAL001C    BY4741  3      slow grower
    yal001c    BY4741  1      (blank)
    YBR001C    BY4730  -1     petite
    YBR001C    BY4730  pet    (blank)
    YCR001W    (blank) -2     red colony, ade2
    YDR001C    BY4741  0      (blank)
    YDR001C    BY4741  tiny   (blank)
    YER001W    W303    5      incorrect strain; het diploid; does not mate
    YFR001W    BY4741  _      (blank)          text-only, no numeric score -> excluded
    YLR287-A   BY4741  2      (blank)          malformed ORF name -> excluded
    (blank)    BY4741  4      (blank)          no ORF -> skipped

Sheet ``Names and Functions of TOP200`` lists YAL001C (TFC3, "TFIIIC subunit",
"transcription") and YCR001W (no gene name, no function, "unknown").

Derived per ORF (first-sighting order): YAL001C scores [3, 1] -> visual_score = max = 3.0,
visual_score_min = 1.0, n = 2; YBR001C [-1] + text "pet" -> -1.0, min None, n = 1, strain
BY4730; YCR001W [-2] -> strain blank -> BY4741; YDR001C [0] + text "tiny"; YER001W [5]
with strain W303, which ``create_experiment`` maps to BY4741 (only BY4741/BY4730 are kept).
The reference is the same WT-with-plasmid score 0.0 for every record, so the reference
index splits only on the background strain: BY4741 -> records [0, 2, 3, 4], BY4730 -> [1].
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path
from typing import Any

import openpyxl
import pytest

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
    VisualScoreExperiment,
    VisualScoreExperimentReference,
    VisualScorePhenotype,
)
from torchcell.datasets.scerevisiae import ozaydin2013 as m

_SCREEN_ROWS: list[tuple[str | None, str | None, Any, str | None]] = [
    ("YAL001C", "BY4741", 3, "slow grower"),
    ("yal001c", "BY4741", 1, None),
    ("YBR001C", "BY4730", -1, "petite"),
    ("YBR001C", "BY4730", "pet", None),
    ("YCR001W", None, -2, "red colony, ade2"),
    ("YDR001C", "BY4741", 0, None),
    ("YDR001C", "BY4741", "tiny", None),
    ("YER001W", "W303", 5, "incorrect strain; het diploid; does not mate"),
    ("YFR001W", "BY4741", "_", None),
    ("YLR287-A", "BY4741", 2, None),
    (None, "BY4741", 4, None),
]
_TOP200_ROWS: list[tuple[str, str | None, str | None, str]] = [
    ("YAL001C", "TFC3", "TFIIIC subunit", "transcription"),
    ("YCR001W", None, None, "unknown"),
]

_NO_FLAGS = {flag: False for flag in m._FLAG_PATTERNS}
_ENVIRONMENT = Environment(
    media=Media(name="SC-URA", state="solid", is_synthetic=True),
    temperature=Temperature(value=30),
)
_PUBLICATION = Publication(
    pubmed_id="22918085",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/22918085/",
    doi="10.1016/j.ymben.2012.07.010",
    doi_url="https://doi.org/10.1016/j.ymben.2012.07.010",
)


def _write_workbook(raw: Path) -> None:
    workbook = openpyxl.Workbook()
    screen = workbook.active
    screen.title = "Color scores of all deletions"
    screen.append(["ORF name", "Strain", "Color", "Comment"])
    for row in _SCREEN_ROWS:
        screen.append(list(row))
    top200 = workbook.create_sheet("Names and Functions of TOP200")
    top200.append(["ORF name", "Gene name", "Gene Function", "Assigned Category"])
    for top_row in _TOP200_ROWS:
        top200.append(list(top_row))
    workbook.save(raw / m.CarotenoidOzaydin2013Dataset.si_filename)


@pytest.fixture
def dataset(tmp_path: Path) -> m.CarotenoidOzaydin2013Dataset:
    root = tmp_path / "carotenoid_ozaydin2013"
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw")
    return m.CarotenoidOzaydin2013Dataset(root=str(root))


def _phenotype(
    score: float,
    score_min: float | None,
    n: int,
    text: str | None,
    flags: dict[str, bool],
) -> VisualScorePhenotype:
    return VisualScorePhenotype(
        visual_score=score,
        visual_score_min=score_min,
        n_replicates=n,
        score_scale_min=-5,
        score_scale_max=5,
        score_semantics=m.SCORE_SEMANTICS,
        target_product="beta-carotene",
        target_metabolite_id=None,
        score_text=text,
        comment_annotations=flags,
    )


def _experiment(orf: str, phenotype: VisualScorePhenotype) -> dict[str, Any]:
    return VisualScoreExperiment(
        dataset_name="CarotenoidOzaydin2013Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=orf
                ),
                *m._carotenogenic_cassette(),
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=phenotype,
    ).model_dump()


def _reference(strain: str) -> dict[str, Any]:
    return VisualScoreExperimentReference(
        dataset_name="CarotenoidOzaydin2013Dataset",
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain=strain
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=VisualScorePhenotype(
            visual_score=0.0,
            n_replicates=1,
            score_scale_min=-5,
            score_scale_max=5,
            score_semantics=m.SCORE_SEMANTICS,
            target_product="beta-carotene",
        ),
    ).model_dump()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, (3.0, None)),
        ("-2", (-2.0, None)),
        ("pet", (None, "pet")),
        (" tiny ", (None, "tiny")),
        ("", (None, None)),
        (float("nan"), (None, None)),
    ],
)
def test_parse_color(value: Any, expected: tuple[float | None, str | None]) -> None:
    """Numeric cells (int or numeric text) become the score; text is stripped; an empty
    string and NaN both give (None, None) because ``text or None`` collapses "".
    """
    assert m._parse_color(value) == expected


def test_parse_comment_flags_every_pattern_case_insensitively() -> None:
    """One comment hitting all seven patterns and one hitting none.

    ``PET`` matches the word-bounded ``pet`` alternative after lowercasing; ``slow on ``
    needs its trailing space; ``petunia`` contains ``pet`` without a word boundary and is
    not a petite flag.
    """
    text = "PET, tiny, slow on glycerol, QC failure, het diploid, bi-mater, pink"
    assert m._parse_comment(text) == {flag: True for flag in m._FLAG_PATTERNS}
    assert m._parse_comment("petunia grows fine") == _NO_FLAGS
    assert m._parse_comment(float("nan")) == _NO_FLAGS


def test_five_records_in_first_sighting_orf_order(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Eleven SI rows give five records; YFR001W (text-only), YLR287-A (malformed) and
    the blank-ORF row are excluded. Record 0 merges the two YAL001C spellings: max 3.0,
    min 1.0, n 2, slow-growth flag from the first row's comment. Record 3 stores the
    text-only replicate as ``score_text`` next to its numeric score.
    """
    assert len(dataset) == 5
    assert dataset[0]["experiment"] == _experiment(
        "YAL001C",
        _phenotype(3.0, 1.0, 2, None, {**_NO_FLAGS, "flag_slow_growth": True}),
    )
    assert dataset[1]["experiment"] == _experiment(
        "YBR001C", _phenotype(-1.0, None, 1, "pet", {**_NO_FLAGS, "flag_petite": True})
    )
    assert dataset[2]["experiment"] == _experiment(
        "YCR001W",
        _phenotype(-2.0, None, 1, None, {**_NO_FLAGS, "flag_unusual_color": True}),
    )
    assert dataset[3]["experiment"] == _experiment(
        "YDR001C", _phenotype(0.0, None, 1, "tiny", _NO_FLAGS)
    )
    assert dataset[4]["experiment"] == _experiment(
        "YER001W",
        _phenotype(
            5.0,
            None,
            1,
            None,
            {
                **_NO_FLAGS,
                "flag_qc_failure": True,
                "flag_het_diploid": True,
                "flag_sterile": True,
            },
        ),
    )
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()


def test_reference_strain_follows_the_si_strain_column_for_by4730_only(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """Finding: ``create_experiment`` keeps the SI strain only when it is BY4741 or
    BY4730; record 4's ``W303`` becomes BY4741 in the reference while ``data.csv`` still
    says W303. The reference index therefore has two entries, BY4741 for [0, 2, 3, 4]
    and BY4730 for [1]; every reference phenotype is the WT-with-plasmid score 0.0.
    """
    assert dataset[1]["reference"] == _reference("BY4730")
    assert dataset[4]["reference"] == _reference("BY4741")
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 2, 3, 4], [1]]
    assert [entry["reference"]["genome_reference"]["strain"] for entry in index] == [
        "BY4741",
        "BY4730",
    ]


def test_data_csv_gene_set_and_build_manifest(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """``preprocess/data.csv`` is the per-ORF aggregate without the flag dict: the
    TOP200 metadata columns are filled for YAL001C and YCR001W (whose gene name and
    function cells are blank) and empty elsewhere.

    Finding: ``gene_set.json`` is not the five deletion ORFs. ``compute_gene_set`` takes
    every perturbation's ``systematic_gene_name``, so the constant YB/I/BTS1 cassette
    contributes ``YPL069C`` (BTS1) and the heterologous symbols ``crtI`` and ``crtYB``,
    which are not S. cerevisiae ORFs; the JSON is sorted, uppercase before lowercase.
    """
    assert dataset.experiment_class is VisualScoreExperiment
    assert dataset.reference_class is VisualScoreExperimentReference
    preprocess = Path(dataset.root) / "preprocess"
    assert (preprocess / "data.csv").read_text() == (
        "orf,strain,visual_score,visual_score_min,n_replicates,score_text,in_top200,"
        "gene_name,gene_function,assigned_category\n"
        "YAL001C,BY4741,3.0,1.0,2,,True,TFC3,TFIIIC subunit,transcription\n"
        "YBR001C,BY4730,-1.0,,1,pet,False,,,\n"
        "YCR001W,BY4741,-2.0,,1,,True,,,unknown\n"
        "YDR001C,BY4741,0.0,,1,tiny,False,,,\n"
        "YER001W,W303,5.0,,1,,False,,,\n"
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
        "YDR001C",
        "YER001W",
        "YPL069C",
        "crtI",
        "crtYB",
    ]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == "carotenoid_ozaydin2013"
    assert manifest["loader_class"] == "CarotenoidOzaydin2013Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.ozaydin2013"
    assert manifest["hostname"] == socket.gethostname()
    assert {"VisualScoreExperiment", "GeneAdditionPerturbation"} <= set(
        manifest["closure"]
    )


def test_download_rejects_a_present_file_with_the_wrong_sha256(
    dataset: m.CarotenoidOzaydin2013Dataset,
) -> None:
    """PyG skips ``download()`` when the raw file exists; called directly it hashes the
    present file and raises naming the path, the actual digest and the pinned one.
    """
    dest = Path(dataset.root) / "raw" / dataset.si_filename
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"Ozaydin SI sha256 mismatch for {dest}: got {digest}, "
            f"expected {m._SI_SHA256}"
        ),
    ):
        dataset.download()
