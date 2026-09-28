# tests/torchcell/datasets/scerevisiae/test_oduibhir2014.py
# [[tests.torchcell.datasets.scerevisiae.test_oduibhir2014]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_oduibhir2014.py
"""Hermetic build of the O'Duibhir 2014 relative-growth-rate fitness loader.

The Dataset S2 text file is written into ``<root>/raw/`` so PyG never calls
``download()``. The loader's row-count self-checksum demands exactly 1312 data rows, so
the five hand-made rows are padded with 1307 ``FILLER<k>`` rows that resolve to nothing and
are dropped. The genome stub carries ``gene_attribute_table`` (IDs YAL001C, YBR001C,
YCR001W) and ``alias_to_systematic`` (YBR002C -> YBR001C; YER001W -> YER999W, a target
that is not an ID).

Rows (orf, commonName, log2relT, similarity), after a leading ``#`` comment line:

    YAL001C  TFC3     1.0   0.1     fitness 2^-1.0 = 0.5
    ybr002c  NTH2     -1.0  0.2     alias -> YBR001C; fitness 2^1.0 = 2.0
    YCR001W  YCR001W  0.0   0.3     fitness 1.0
    YZZ999W  GHOST    0.5   0.4     unresolved -> dropped
    YER001W  OLD      0.25  0.5     alias target not an ID -> dropped

Every record: n_samples = 2 (biological replicates), no uncertainty, SC liquid 30 C,
BY4741 reference at fitness 1.0.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from torchcell.datamodels.media import SC
from torchcell.datamodels.schema import (
    Environment,
    FitnessExperiment,
    FitnessExperimentReference,
    FitnessPhenotype,
    Genotype,
    KanMxDeletionPerturbation,
    Publication,
    ReferenceGenome,
    SampleUnit,
    Temperature,
)
from torchcell.datasets.scerevisiae import oduibhir2014 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

_ROWS = [
    ["YAL001C", "TFC3", "1.0", "0.1"],
    ["ybr002c", "NTH2", "-1.0", "0.2"],
    ["YCR001W", "YCR001W", "0.0", "0.3"],
    ["YZZ999W", "GHOST", "0.5", "0.4"],
    ["YER001W", "OLD", "0.25", "0.5"],
]
_HEADER = ["", "commonName", "log2relT", "similarity"]


class _StubGenome:
    gene_attribute_table = pd.DataFrame({"ID": ["YAL001C", "YBR001C", "YCR001W"]})
    alias_to_systematic: dict[str, list[str]] = {
        "YBR002C": ["YBR001C"],
        "YER001W": ["YER999W"],
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


def _write_s2(
    raw: Path, rows: list[list[str]], header: list[str] = _HEADER, pad: bool = True
) -> None:
    filler = (
        [
            [f"FILLER{k:04d}", "x", "0.0", "0.0"]
            for k in range(m._EXPECTED_ROWS - len(rows))
        ]
        if pad
        else []
    )
    lines = [
        "# Supplementary Dataset S2: relative doubling time of 1312 deletion strains",
        "\t".join(header),
        *("\t".join(r) for r in [*rows, *filler]),
    ]
    (raw / m._RAW_FILENAME).write_text("\n".join(lines) + "\n")


def _root(tmp_path: Path, slug: str = "smf_oduibhir2014", **kwargs: Any) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_s2(root / "raw", _ROWS, **kwargs)
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> m.SmfODuibhir2014Dataset:
    return m.SmfODuibhir2014Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(media=SC, temperature=Temperature(value=30))
_REFERENCE = FitnessExperimentReference(
    dataset_name="SmfODuibhir2014Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4741"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=FitnessPhenotype(
        fitness=1.0, n_samples=2, sample_unit=SampleUnit.biological_replicate
    ),
).model_dump()
_PUBLICATION = Publication(
    pubmed_id="24952590",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/24952590/",
    doi="10.15252/msb.20145172",
    doi_url="https://doi.org/10.15252/msb.20145172",
).model_dump()


def _experiment(orf: str, common: str, fitness: float) -> dict[str, Any]:
    return FitnessExperiment(
        dataset_name="SmfODuibhir2014Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=common
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=FitnessPhenotype(
            fitness=fitness, n_samples=2, sample_unit=SampleUnit.biological_replicate
        ),
    ).model_dump()


def test_three_records_with_fitness_two_to_the_minus_log2relt(
    dataset: m.SmfODuibhir2014Dataset,
) -> None:
    """1312 rows give three records: a slow grower (log2relT 1.0 -> 0.5), a fast grower
    reached through an alias (-1.0 -> 2.0) and a neutral one (0.0 -> 1.0). The two
    unresolvable ORFs and the 1307 filler rows are dropped. All uncertainty fields stay
    None; the reference is BY4741 at fitness 1.0 with the same n = 2.
    """
    assert len(dataset) == 3
    assert dataset[0]["experiment"] == _experiment("YAL001C", "TFC3", 0.5)
    assert dataset[1]["experiment"] == _experiment("YBR001C", "NTH2", 2.0)
    assert dataset[2]["experiment"] == _experiment("YCR001W", "YCR001W", 1.0)
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    phenotype = dataset[0]["experiment"]["phenotype"]
    assert phenotype["fitness_std"] is None
    assert phenotype["sample_unit"] == "biological_replicate"


def test_side_files(dataset: m.SmfODuibhir2014Dataset) -> None:
    """No ``data.csv`` is written by this loader; the gene set is the three resolved ORFs
    and one reference covers every record. ``create_experiment`` is not this loader's
    path and raises ``NotImplementedError``.
    """
    assert dataset.experiment_class is FitnessExperiment
    assert dataset.reference_class is FitnessExperimentReference
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()
    preprocess = Path(dataset.root) / "preprocess"
    assert not (preprocess / "data.csv").exists()
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2]]
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "smf_oduibhir2014"
    assert manifest["loader_class"] == "SmfODuibhir2014Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.oduibhir2014"
    assert {"FitnessExperiment", "FitnessPhenotype"} <= set(manifest["closure"])


def test_requires_a_genome_before_reading_the_file(tmp_path: Path) -> None:
    """The genome check runs before the file is read, so even a valid file raises."""
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "SmfODuibhir2014Dataset requires a genome for ORF resolution; "
            "inject SCerevisiaeGenome(...)"
        ),
    ):
        m.SmfODuibhir2014Dataset(root=str(_root(tmp_path)), genome=None)


def test_unexpected_columns_raise_before_the_row_count(tmp_path: Path) -> None:
    """A fifth column trips the column check even though the row count is right."""
    root = _root(tmp_path, header=[*_HEADER, "extra"])
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "unexpected Dataset S2 columns: ['orf', 'commonName', 'log2relT', "
            "'similarity', 'extra'] != ['orf', 'commonName', 'log2relT', 'similarity']"
        ),
    ):
        m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())


def test_row_count_self_checksum(tmp_path: Path) -> None:
    """Without the filler the five rows fail the 1312-row check."""
    root = _root(tmp_path, pad=False)
    with pytest.raises(
        RuntimeError, match="Dataset S2 row-count self-checksum failed: 5 != 1312"
    ):
        m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())


def test_download_verifies_present_file_and_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.SmfODuibhir2014Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / m._RAW_FILENAME
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{m._RAW_FILENAME} sha256 mismatch: got {digest}, "
            f"expected {m._DATASET_S2_SHA256}"
        ),
    ):
        dataset.download()
    src = (
        tmp_path
        / "data_root"
        / "torchcell-library"
        / m._LIBRARY_CITATION_KEY
        / m._MIRROR_REL_PATH
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"library mirror Dataset S2 not found: {src}. This dataset's source is the "
            "sha256-pinned Supplementary Dataset S2 in the torchcell-library mirror."
        ),
    ):
        m.SmfODuibhir2014Dataset(root=str(tmp_path / "empty"), genome=_genome())
