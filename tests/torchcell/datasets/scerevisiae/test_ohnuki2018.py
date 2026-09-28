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
    assert dataset.experiment_class is CalMorphExperiment
    assert dataset.reference_class is CalMorphExperimentReference
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


def test_download_verifies_present_files_and_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.ScmdOhnuki2018Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / "ess1112data.tsv"
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"ess1112data.tsv sha256 mismatch: got {digest}, "
            f"expected {m._RAW_FILES['ess1112data.tsv']['sha256']}"
        ),
    ):
        dataset.download()
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
