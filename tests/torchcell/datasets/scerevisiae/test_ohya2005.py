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
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import pytest

from torchcell.datamodels.schema import (
    CalMorphExperiment,
    CalMorphExperimentReference,
    CalMorphPhenotype,
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import ohya2005 as m
from torchcell.sequence.genome.scerevisiae.s288c import (
    GeneNameResolution,
    GeneNameStatus,
    SCerevisiaeGenome,
)

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


_ENVIRONMENT = Environment(
    media=Media(name="YPD", state="liquid", is_synthetic=False),
    temperature=Temperature(value=25),
)
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
    assert dataset.experiment_class is CalMorphExperiment
    assert dataset.reference_class is CalMorphExperimentReference
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


def test_download_verifies_present_files_and_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Called on a root holding the synthetic matrices, ``download()`` hashes the first
    file and raises with its digest and the pinned one; on an empty root it names the
    missing mirror path under ``$DATA_ROOT``.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.ScmdOhya2005Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / "mt4718data.tsv"
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"mt4718data.tsv sha256 mismatch: got {digest}, "
            f"expected {m._RAW_FILES['mt4718data.tsv']['sha256']}"
        ),
    ):
        dataset.download()
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
