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
    perturbation of the genotype. ``create_experiment`` is not this loader's path and
    raises ``NotImplementedError``.
    """
    assert dataset.experiment_class is CalMorphExperiment
    assert dataset.reference_class is CalMorphExperimentReference
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
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
    """Three branches of ``download()``: a present raw file with the wrong hash raises
    "in raw dir"; an empty root with no mirror file raises "mirror file missing"; a
    mirror file holding ``b"not the matrix"`` raises "in mirror" with its digest
    (35fe305f...) and nothing is copied into ``raw/``.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    dataset = m.ScmdOhnuki2022Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / m.MUTANT_FILE
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=f"{m.MUTANT_FILE} sha256 mismatch in raw dir: got {digest}, "
        f"expected {m.MUTANT_SHA256}",
    ):
        dataset.download()
    mirror = data_root / m.MIRROR_SUBPATH
    with pytest.raises(
        RuntimeError, match=re.escape(f"mirror file missing: {mirror / m.MUTANT_FILE}")
    ):
        m.ScmdOhnuki2022Dataset(root=str(tmp_path / "empty"), genome=_genome())
    mirror.mkdir(parents=True)
    (mirror / m.MUTANT_FILE).write_bytes(b"not the matrix")
    bad = hashlib.sha256(b"not the matrix").hexdigest()
    assert bad.startswith("35fe305f")
    with pytest.raises(
        RuntimeError,
        match=f"{m.MUTANT_FILE} sha256 mismatch in mirror: got {bad}, "
        f"expected {m.MUTANT_SHA256}",
    ):
        m.ScmdOhnuki2022Dataset(root=str(tmp_path / "empty2"), genome=_genome())
    assert not (tmp_path / "empty2" / "raw" / m.MUTANT_FILE).exists()
