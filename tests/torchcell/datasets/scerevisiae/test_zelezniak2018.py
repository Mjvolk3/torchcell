# tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py
# [[tests.torchcell.datasets.scerevisiae.test_zelezniak2018]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_zelezniak2018.py
"""Hermetic build tests for the two Zelezniak 2018 loaders (proteome + metabolome).

Each test writes a hand-sized Zenodo-shaped TSV under ``<root>/raw/`` so PyG skips
``download()``, builds the LMDB once through ``process()``, and compares the stored
records against hand-built ``ProteinAbundanceExperiment`` /
``MetaboliteExperiment`` objects. No network, no ``$DATA_ROOT``, no YeastGEM: the
metabolome's ``build_metabolite_s_id_map`` (which loads the Yeast9 model) is replaced by
a recording stub that returns a fixed ``s_NNNN`` map.

Proteome fixture (``proteins_dataset.data_prep.tsv``; ``groupby("ORF")`` mean / sample SD /
count, SE = SD / sqrt(n)):

    WT       YAL001C 10, 12        -> mean 11.0, SD sqrt(2), SE 1.0, n 2
    WT       YBR002C 5, 7, 9       -> mean 7.0,  SD 2.0,     SE 2/sqrt(3), n 3
    YDR003W  YAL001C 8, 9          -> mean 8.5,  SD sqrt(.5), SE 0.5, n 2
    YDR003W  YBR002C 6, 6          -> mean 6.0,  SD 0.0,     SE 0.0, n 2
    YER004W  YAL001C 1, 3          -> mean 2.0,  SD sqrt(2), SE 1.0, n 2
    YER004W  YBR002C 20, 24        -> mean 22.0, SD sqrt(8), SE 2.0, n 2
    bad_ko   YAL001C 99            -> KO_ORF fails the nuclear-ORF regex: skipped

Metabolome fixture (``metabolites_dataset.data_prep.tsv``; rows pooled across the
``dataset`` protocol column per metabolite):

    WT       pyr      2, 4 (protocols 1, 2) -> mean 3.0, SE 1.0, n 2
    WT       3pg;2pg  10                    -> n 1, SE NaN
    YDR003W  pyr      1, 3                  -> mean 2.0, SE 1.0, n 2
    YDR003W  atp      5, 5                  -> mean 5.0, SE 0.0, n 2 (WT never measured atp)
    YER004W  3pg;2pg  7                     -> n 1, SE NaN -> all-NaN SE collapses to None
"""

import json
import math
import socket
from pathlib import Path
from typing import Any

import pytest

from torchcell.datamodels.media import SM_DEFERRED
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import zelezniak2018
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    DATA_FILENAME as PROTEOME_FILENAME,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    MEASUREMENT_TYPE as PROTEOME_MEASUREMENT_TYPE,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    METABOLITE_DATA_FILENAME,
    METABOLITE_MEASUREMENT_TYPE,
    MetaboliteZelezniak2018Dataset,
    ProteomeZelezniak2018Dataset,
)

# --------------------------------------------------------------------------- #
# Shared expected pieces
# --------------------------------------------------------------------------- #

BY4741 = ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")
ENVIRONMENT = Environment(media=SM_DEFERRED, temperature=Temperature(value=30))
PUBLICATION = Publication(
    pubmed_id="30195436",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/30195436/",
    doi="10.1016/j.cels.2018.08.001",
    doi_url="https://doi.org/10.1016/j.cels.2018.08.001",
)

PROTEOME_ROWS = [
    # ORF, KO_ORF, KO_gene_name, replicate, value
    ("YAL001C", "WT", "WT", 1, 10.0),
    ("YAL001C", "WT", "WT", 2, 12.0),
    ("YBR002C", "WT", "WT", 1, 5.0),
    ("YBR002C", "WT", "WT", 2, 7.0),
    ("YBR002C", "WT", "WT", 3, 9.0),
    ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
    ("YAL001C", "YDR003W", "KIN3", 2, 9.0),
    ("YBR002C", "YDR003W", "KIN3", 1, 6.0),
    ("YBR002C", "YDR003W", "KIN3", 2, 6.0),
    ("YAL001C", "YER004W", "KIN4", 1, 1.0),
    ("YAL001C", "YER004W", "KIN4", 2, 3.0),
    ("YBR002C", "YER004W", "KIN4", 1, 20.0),
    ("YBR002C", "YER004W", "KIN4", 2, 24.0),
    ("YAL001C", "bad_ko", "BAD", 1, 99.0),
]

METABOLITE_ROWS = [
    # metabolite_id, kegg_id, official_name, dataset, genotype, replicate, value
    ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
    ("pyr", "C00022", "Pyruvate", 2, "WT", 1, 4.0),
    ("3pg;2pg", "C00197;C00631", "3-Phospho-D-glycerate", 1, "WT", 1, 10.0),
    ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 1, 1.0),
    ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 2, 3.0),
    ("atp", "C00002", "ATP", 1, "YDR003W", 1, 5.0),
    ("atp", "C00002", "ATP", 1, "YDR003W", 2, 5.0),
    ("3pg;2pg", "C00197;C00631", "3-Phospho-D-glycerate", 1, "YER004W", 1, 7.0),
]

S_ID_MAP = {"pyr": "s_1399", "3pg;2pg": "s_0188", "atp": "s_0434"}


def _write_tsv(path: Path, header: list[str], rows: list[tuple[Any, ...]]) -> None:
    lines = ["\t".join(header)]
    lines += ["\t".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def _proteome_root(tmp_path: Path, rows: list[tuple[Any, ...]]) -> Path:
    root = tmp_path / "proteome_zelezniak2018"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / PROTEOME_FILENAME,
        ["ORF", "KO_ORF", "KO_gene_name", "replicate", "value"],
        rows,
    )
    return root


def _metabolite_root(tmp_path: Path, rows: list[tuple[Any, ...]]) -> Path:
    root = tmp_path / "metabolite_zelezniak2018"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / METABOLITE_DATA_FILENAME,
        [
            "metabolite_id",
            "kegg_id",
            "official_name",
            "dataset",
            "genotype",
            "replicate",
            "value",
        ],
        rows,
    )
    return root


def _assert_side_files(root: Path, cls: type, gene_set: list[str]) -> None:
    """``preprocess/gene_set.json``, ``experiment_reference_index.json``, and the manifest."""
    pre = root / "preprocess"
    assert json.loads((pre / "gene_set.json").read_text()) == gene_set
    manifest = json.loads((pre / "build_manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == root.name
    assert manifest["loader_class"] == cls.__name__
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.zelezniak2018"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["surface_modules"] == ["pydant.py", "schema.py"]


# --------------------------------------------------------------------------- #
# Proteome
# --------------------------------------------------------------------------- #


@pytest.fixture
def proteome(tmp_path: Path) -> tuple[ProteomeZelezniak2018Dataset, Path]:
    root = _proteome_root(tmp_path, PROTEOME_ROWS)
    return ProteomeZelezniak2018Dataset(root=str(root)), root


def _proteome_reference() -> ProteinAbundanceExperimentReference:
    return ProteinAbundanceExperimentReference(
        dataset_name="ProteomeZelezniak2018Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 11.0, "YBR002C": 7.0},
            protein_abundance_se={"YAL001C": 1.0, "YBR002C": 2.0 / math.sqrt(3)},
            n_replicates={"YAL001C": 2, "YBR002C": 3},
            measurement_type=PROTEOME_MEASUREMENT_TYPE,
        ),
    )


def test_proteome_len_and_gene_set_skip_non_systematic_ko(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Two knockout strains survive: the ``bad_ko`` strain is skipped, not raised.

    ``KO_ORF`` values are grouped alphabetically (``groupby`` sorts), so YDR003W is
    record 0 and YER004W is record 1; the WT rows are the reference and never a record.
    Finding: a non-systematic KO_ORF is silently dropped (counted in a log line) while a
    non-systematic protein ORF raises; the fixture pins both sides of that asymmetry.
    ``preprocess/data.csv`` carries the two kept strains with their gene names.
    """
    ds, root = proteome
    assert len(ds) == 2
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,gene\nYDR003W,KIN3\nYER004W,KIN4\n"
    )
    _assert_side_files(root, ProteomeZelezniak2018Dataset, ["YDR003W", "YER004W"])


def test_proteome_record_0_is_ydr003w_with_hand_computed_se(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Record 0 = YDR003W/KIN3: means 8.5 and 6.0, SEs 0.5 and 0.0, n 2 and 2.

    SE for YAL001C: SD of (8, 9) is sqrt(0.5); / sqrt(2) = 0.5. SE for YBR002C: SD of
    (6, 6) is 0.0. Environment is ``SM_DEFERRED`` at 30 C; genotype a single KanMX
    deletion carrying the KO_gene_name; the reference is the WT aggregate over ALL
    proteins (not restricted, unlike the metabolome loader). The whole dumps are
    compared exactly: pandas ``std() / sqrt(n)`` on these fixtures gives exactly 0.5,
    0.0, 1.0 and ``2.0 / math.sqrt(3)``.
    """
    ds, _ = proteome
    record = ds[0]
    expected = ProteinAbundanceExperiment(
        dataset_name="ProteomeZelezniak2018Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YDR003W", perturbed_gene_name="KIN3"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={"YAL001C": 8.5, "YBR002C": 6.0},
            protein_abundance_se={"YAL001C": 0.5, "YBR002C": 0.0},
            n_replicates={"YAL001C": 2, "YBR002C": 2},
            measurement_type=PROTEOME_MEASUREMENT_TYPE,
        ),
    )
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == _proteome_reference().model_dump()
    assert record["publication"] == PUBLICATION.model_dump()
    # the stored dicts validate back into the schema classes unchanged
    assert (
        ProteinAbundanceExperiment.model_validate(record["experiment"]).model_dump()
        == record["experiment"]
    )


def test_proteome_record_1_is_yer004w(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Record 1 = YER004W/KIN4: means 2.0 and 22.0, SEs 1.0 and 2.0.

    SD of (1, 3) is sqrt(2), / sqrt(2) = 1.0; SD of (20, 24) is sqrt(8), / sqrt(2) = 2.0.
    """
    ds, _ = proteome
    experiment = ds[1]["experiment"]
    assert experiment["genotype"]["perturbations"][0]["systematic_gene_name"] == (
        "YER004W"
    )
    assert experiment["genotype"]["perturbations"][0]["perturbed_gene_name"] == "KIN4"
    phenotype = experiment["phenotype"]
    assert phenotype["protein_abundance"] == {"YAL001C": 2.0, "YBR002C": 22.0}
    assert phenotype["n_replicates"] == {"YAL001C": 2, "YBR002C": 2}
    assert phenotype["protein_abundance_se"] == {"YAL001C": 1.0, "YBR002C": 2.0}


def test_proteome_reference_index_has_one_shared_reference(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Both strains share the single WT reference -> one index entry covering [0, 1]."""
    ds, root = proteome
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert len(index) == 1
    assert index[0]["member_indices"] == [0, 1]
    assert index[0]["reference"]["dataset_name"] == "ProteomeZelezniak2018Dataset"
    assert index[0]["reference"]["experiment_reference_type"] == "protein_abundance"
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == [[0, 1]]


def test_proteome_single_replicate_protein_gets_nan_se(tmp_path: Path) -> None:
    """A protein measured once in a strain stores SE NaN (not None, not 0).

    Finding: the loader writes ``float("nan")`` into ``protein_abundance_se`` for n = 1
    rather than omitting the key; the module docstring says every protein has >= 2
    replicates in the real file, so this is the guarded branch of ``_aggregate``.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "WT", "WT", 2, 12.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
    ]
    ds = ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    phenotype = ds[0]["experiment"]["phenotype"]
    assert phenotype["protein_abundance"] == {"YAL001C": 8.0}
    assert phenotype["n_replicates"] == {"YAL001C": 1}
    assert list(phenotype["protein_abundance_se"]) == ["YAL001C"]
    assert math.isnan(phenotype["protein_abundance_se"]["YAL001C"])


def test_proteome_non_systematic_protein_orf_raises(tmp_path: Path) -> None:
    """A protein ORF outside the nuclear systematic-name pattern aborts the build.

    ``Q0250`` (a mitochondrial ORF) is the offending id here: the proteome regex
    ``_SYSTEMATIC_RE`` has NO Q-plus-four-digits alternative, unlike the Messner and
    YeastPhenome loaders. The error message reports the count of bad rows (1).
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("Q0250", "WT", "WT", 1, 1.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
    ]
    with pytest.raises(RuntimeError, match="non-systematic protein ORF ids present: 1"):
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))


def test_proteome_missing_wt_raises(tmp_path: Path) -> None:
    """No ``KO_ORF == "WT"`` rows -> RuntimeError naming the missing reference strain."""
    rows = [("YAL001C", "YDR003W", "KIN3", 1, 8.0)]
    with pytest.raises(RuntimeError, match="missing the WT reference strain"):
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))


# --------------------------------------------------------------------------- #
# Metabolome
# --------------------------------------------------------------------------- #


@pytest.fixture
def s_id_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    """Replace the YeastGEM-backed mapper with a stub that records its argument."""
    calls: list[dict[str, str]] = []

    def fake_map(kegg_by_metabolite: dict[str, str]) -> dict[str, str]:
        calls.append(dict(kegg_by_metabolite))
        return dict(S_ID_MAP)

    monkeypatch.setattr(zelezniak2018, "build_metabolite_s_id_map", fake_map)
    return calls


@pytest.fixture
def metabolome(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]]:
    root = _metabolite_root(tmp_path, METABOLITE_ROWS)
    return MetaboliteZelezniak2018Dataset(root=str(root)), root, s_id_calls


def test_metabolome_len_side_files_and_s_id_map_argument(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Two strains; the mapper is called once with the deduplicated metabolite -> KEGG map.

    ``preprocess/data.csv`` reports ``n_metabolites`` per strain (2 for YDR003W, 1 for
    YER004W); the gene set is the two KO ORFs.
    """
    ds, root, calls = metabolome
    assert len(ds) == 2
    assert calls == [{"pyr": "C00022", "3pg;2pg": "C00197;C00631", "atp": "C00002"}]
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,n_metabolites\nYDR003W,2\nYER004W,1\n"
    )
    _assert_side_files(root, MetaboliteZelezniak2018Dataset, ["YDR003W", "YER004W"])


def test_metabolome_record_0_reference_restricted_to_measured_keys(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Record 0 = YDR003W: levels pyr 2.0 / atp 5.0, SEs 1.0 / 0.0; reference has pyr ONLY.

    WT never measured atp, so the reference phenotype is restricted to {pyr} with the WT
    aggregate (mean 3.0, SE 1.0, n 2). ``target_metabolite_ids`` is the stub map filtered
    to the phenotype's keys on both sides. Genotype uses the ORF as both names (no gene
    name column in the metabolome file). The dumps are compared exactly: the SEs 1.0,
    0.0 and 1.0 are exact under pandas ``std() / sqrt(n)``.
    """
    ds, _, _ = metabolome
    record = ds[0]
    expected = MetaboliteExperiment(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YDR003W", perturbed_gene_name="YDR003W"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level={"pyr": 2.0, "atp": 5.0},
            metabolite_level_se={"pyr": 1.0, "atp": 0.0},
            n_replicates={"pyr": 2, "atp": 2},
            measurement_type=METABOLITE_MEASUREMENT_TYPE,
            target_metabolite_ids={"pyr": "s_1399", "atp": "s_0434"},
        ),
    )
    expected_reference = MetaboliteExperimentReference(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level={"pyr": 3.0},
            metabolite_level_se={"pyr": 1.0},
            n_replicates={"pyr": 2},
            measurement_type=METABOLITE_MEASUREMENT_TYPE,
            target_metabolite_ids={"pyr": "s_1399"},
        ),
    )
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == PUBLICATION.model_dump()
    assert (
        MetaboliteExperiment.model_validate(record["experiment"]).model_dump()
        == record["experiment"]
    )


def test_metabolome_record_1_all_nan_se_collapses_to_none(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Record 1 = YER004W: one single-row metabolite -> ``metabolite_level_se`` is None.

    Both the strain (7.0, n 1) and the WT baseline (10.0, n 1) for ``3pg;2pg`` have a
    NaN SE, and an all-NaN SE dict is stored as None on both the experiment and the
    reference. The ``;``-merged key is kept verbatim and maps to the stub's ``s_0188``.

    The literal dict also pins the schema envelope fields (``provenance_gaps``,
    ``graph_level``, ``label_name``, ``label_statistic_name``), so adding a field to
    ``MetabolitePhenotype`` fails this test with the loader unchanged. That is intended:
    a stored-record shape change should be noticed here.
    """
    ds, _, _ = metabolome
    record = ds[1]
    phenotype = record["experiment"]["phenotype"]
    assert phenotype == {
        "provenance_gaps": [],
        "graph_level": "metabolism",
        "label_name": "metabolite_level",
        "label_statistic_name": "metabolite_level_se",
        "metabolite_level": {"3pg;2pg": 7.0},
        "metabolite_level_se": None,
        "n_replicates": {"3pg;2pg": 1},
        "measurement_type": METABOLITE_MEASUREMENT_TYPE,
        "target_metabolite_ids": {"3pg;2pg": "s_0188"},
    }
    reference = record["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"3pg;2pg": 10.0}
    assert reference["metabolite_level_se"] is None
    assert reference["n_replicates"] == {"3pg;2pg": 1}
    assert reference["target_metabolite_ids"] == {"3pg;2pg": "s_0188"}


def test_metabolome_reference_index_splits_on_restricted_reference(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Two distinct references (pyr-restricted vs 3pg;2pg-restricted) -> two entries."""
    _, root, _ = metabolome
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert [e["member_indices"] for e in index] == [[0], [1]]
    assert [
        list(e["reference"]["phenotype_reference"]["metabolite_level"]) for e in index
    ] == [["pyr"], ["3pg;2pg"]]


def test_metabolome_non_systematic_genotype_raises_before_mapping(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """A non-WT genotype outside the nuclear-ORF regex aborts BEFORE the mapper runs.

    The regex check precedes ``build_metabolite_s_id_map``, so the stub records zero
    calls; the message reports the bad-row count (2 rows of ``kin3``).
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 1, "kin3", 1, 1.0),
        ("pyr", "C00022", "Pyruvate", 1, "kin3", 2, 3.0),
    ]
    with pytest.raises(
        RuntimeError, match="non-systematic strain genotype ids present: 2"
    ):
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert s_id_calls == []


def test_metabolome_missing_wt_raises_after_mapping(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """No ``genotype == "WT"`` rows -> RuntimeError, raised AFTER the mapper was called once."""
    rows = [("pyr", "C00022", "Pyruvate", 1, "YDR003W", 1, 1.0)]
    with pytest.raises(
        RuntimeError, match="metabolome missing the WT reference strain"
    ):
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert s_id_calls == [{"pyr": "C00022"}]
