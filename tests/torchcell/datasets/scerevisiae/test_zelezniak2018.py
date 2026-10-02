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

Metabolome fixture (``metabolites_dataset.data_prep.tsv``; one record per (protocol,
strain), aggregated over that protocol's rows only, issue #595):

    WT       pyr      protocol 1: 2         -> protocol-1 reference, n 1, SE NaN
    WT       pyr      protocol 2: 4         -> protocol-2 reference, n 1, SE NaN
    WT       3pg;2pg  protocol 1: 10        -> protocol-1 reference, n 1, SE NaN
    YDR003W  pyr      protocol 1: 1, 3      -> record 0: mean 2.0, SE 1.0, n 2
    YDR003W  atp      protocol 1: 5, 5      -> record 0: mean 5.0, SE 0.0, n 2 (no WT atp)
    YER004W  3pg;2pg  protocol 1: 7         -> record 1: n 1, all-NaN SE collapses to None
    YDR003W  pyr      protocol 2: 6, 8      -> record 2: mean 7.0, SE 1.0, n 2

Records are ordered by (protocol, strain). Before issue #595 the WT pyr baseline was the
pooled 3.0 (n 2) and protocol-2 rows joined the protocol-1 strain mean.

2026.09.30 (Phase 12): the uncovered paths. ``download()`` for both loaders runs against a
fake ``urllib.request.urlopen`` that records the ``Request`` (URL, User-Agent, timeout
300) and returns synthetic bytes: a digest mismatch refuses with both digests and writes
nothing, a present raw file short-circuits without a request, and a proteome build with
no raw file downloads (pin set to the synthetic TSV's digest) and then builds.
``build_metabolite_s_id_map`` runs on a fake ``YeastGEM`` whose model carries six
metabolites: KEGG wins over BiGG, a cytosolic form wins over the first-listed one, a
metabolite with no cytosolic form takes the first listed compartment, a ``;``-merged id
resolves through its first token, a list-valued annotation indexes every token, a
``nan`` KEGG id falls back to BiGG, and an unmatched id refuses with both tokens. A
proteome edge fixture pins the replicate handling: the first ``KO_gene_name`` of a
strain wins.

2026.10.01 (issue #520): ``n`` is the row count, so the proteome loader now refuses a
repeated (ORF, strain, replicate) row, a blank value and a non-finite value, naming the
strain (WT included);
an all-blank protein refuses there too instead of in schema validation. Exact messages are
asserted. The pinned release has 0 of each (264,264 rows), so stored records are unchanged.

2026.10.02 (issue #595): the metabolome keeps its three LC-SRM protocols apart. One record
per (protocol, strain), the reference is the same-protocol WT, the measurement_type names
the protocol, and the issue's two example cells (WT ``3pg;2pg``, YIL042C ``r5p``) are
pinned on their verbatim raw values. An unknown protocol, a protocol without WT rows, a
(strain, protocol) sharing no metabolite with its WT, and a repeated (metabolite,
replicate) id inside one protocol each refuse with an exact message. The protocol quotes
are audited against the mirrored ``paper.md`` when the mirror is mounted.
"""

import hashlib
import json
import math
import os
import socket
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from torchcell.data import RawSha256MismatchError
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
    CALIBRATED_UNIT_GAP,
    METABOLITE_DATA_FILENAME,
    METABOLITE_SOURCED_VALUES,
    ZELEZNIAK_METABOLITE_PROTOCOLS,
    MetaboliteZelezniak2018Dataset,
    ProteomeZelezniak2018Dataset,
    ZelezniakMetaboliteProtocol,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    DATA_FILENAME as PROTEOME_FILENAME,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    MEASUREMENT_TYPE as PROTEOME_MEASUREMENT_TYPE,
)
from torchcell.verification import runners
from torchcell.verification.sourced import audit_sourced_value

MTYPE_1 = ZELEZNIAK_METABOLITE_PROTOCOLS[1].measurement_type
MTYPE_2 = ZELEZNIAK_METABOLITE_PROTOCOLS[2].measurement_type

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
    ("pyr", "C00022", "Pyruvate", 2, "YDR003W", 1, 6.0),
    ("pyr", "C00022", "Pyruvate", 2, "YDR003W", 2, 8.0),
]

S_ID_MAP = {"pyr": "s_1399", "3pg;2pg": "s_0188", "atp": "s_0434", "r5p": "s_stub_r5p"}


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
    """Three (protocol, strain) records; the mapper is called once with the deduplicated
    metabolite -> KEGG map.

    ``preprocess/data.csv`` reports the protocol and ``n_metabolites`` per record, ordered
    by (protocol, strain): YDR003W twice (protocols 1 and 2). The gene set is still the
    two KO ORFs.
    """
    ds, root, calls = metabolome
    assert len(ds) == 3
    assert calls == [{"pyr": "C00022", "3pg;2pg": "C00197;C00631", "atp": "C00002"}]
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,dataset,n_metabolites\nYDR003W,1,2\nYER004W,1,1\nYDR003W,2,1\n"
    )
    _assert_side_files(root, MetaboliteZelezniak2018Dataset, ["YDR003W", "YER004W"])


def _ydr003w() -> Genotype:
    return Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YDR003W", perturbed_gene_name="YDR003W"
            )
        ]
    )


def test_metabolome_record_0_reference_is_the_same_protocol_wt(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Record 0 = YDR003W on protocol 1: pyr 2.0 / atp 5.0, SEs 1.0 / 0.0; reference has
    pyr ONLY, at the protocol-1 WT value 2.0 (n 1), never the pooled 3.0 of protocols 1
    and 2. WT never measured atp, so the reference is restricted to {pyr}; its single
    replicate gives an all-NaN SE, stored as None. Genotype uses the ORF as both names
    (no gene name column in the metabolome file). The dumps are compared exactly.
    """
    ds, _, _ = metabolome
    record = ds[0]
    expected = MetaboliteExperiment(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genotype=_ydr003w(),
        environment=ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level={"pyr": 2.0, "atp": 5.0},
            metabolite_level_se={"pyr": 1.0, "atp": 0.0},
            n_replicates={"pyr": 2, "atp": 2},
            measurement_type=MTYPE_1,
            target_metabolite_ids={"pyr": "s_1399", "atp": "s_0434"},
        ),
    )
    expected_reference = MetaboliteExperimentReference(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=MetabolitePhenotype(
            metabolite_level={"pyr": 2.0},
            metabolite_level_se=None,
            n_replicates={"pyr": 1},
            measurement_type=MTYPE_1,
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


def test_metabolome_record_2_is_the_protocol_2_record_of_the_same_strain(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Record 2 = YDR003W on protocol 2: pyr 7.0 (6, 8), SE 1.0, n 2, against the
    protocol-2 WT pyr 4.0. The protocol-1 rows (1, 3) are not in it, and it carries the
    protocol-2 measurement_type on both sides.
    """
    ds, _, _ = metabolome
    record = ds[2]
    expected = MetaboliteExperiment(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genotype=_ydr003w(),
        environment=ENVIRONMENT,
        phenotype=MetabolitePhenotype(
            metabolite_level={"pyr": 7.0},
            metabolite_level_se={"pyr": 1.0},
            n_replicates={"pyr": 2},
            measurement_type=MTYPE_2,
            target_metabolite_ids={"pyr": "s_1399"},
        ),
    )
    reference = record["reference"]["phenotype_reference"]
    assert record["experiment"] == expected.model_dump()
    assert reference["metabolite_level"] == {"pyr": 4.0}
    assert reference["n_replicates"] == {"pyr": 1}
    assert reference["measurement_type"] == MTYPE_2


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
        "measurement_type": MTYPE_1,
        "target_metabolite_ids": {"3pg;2pg": "s_0188"},
    }
    reference = record["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"3pg;2pg": 10.0}
    assert reference["metabolite_level_se"] is None
    assert reference["n_replicates"] == {"3pg;2pg": 1}
    assert reference["target_metabolite_ids"] == {"3pg;2pg": "s_0188"}


def test_metabolome_reference_index_splits_on_protocol_and_restriction(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Three distinct references: protocol-1 {pyr}, protocol-1 {3pg;2pg}, protocol-2 {pyr}."""
    _, root, _ = metabolome
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert [e["member_indices"] for e in index] == [[0], [1], [2]]
    assert [
        (
            e["reference"]["phenotype_reference"]["measurement_type"],
            e["reference"]["phenotype_reference"]["metabolite_level"],
        )
        for e in index
    ] == [
        (MTYPE_1, {"pyr": 2.0}),
        (MTYPE_1, {"3pg;2pg": 10.0}),
        (MTYPE_2, {"pyr": 4.0}),
    ]


# Issue #595's two example cells, rows copied verbatim from the pinned release
# (sha256 c4429fd8...): WT 3pg;2pg and YIL042C r5p, each measured by protocols 1 and 2.
ISSUE_595_ROWS = [
    ("3pg;2pg", "C00197;C00631", "3PG", 1, "WT", 1, 850.2700803858073),
    ("3pg;2pg", "C00197;C00631", "3PG", 2, "WT", 1, 0.387544549099187),
    ("r5p", "C00117", "R5P", 1, "WT", 1, 900.0),
    ("r5p", "C00117", "R5P", 2, "WT", 1, 0.7),
    ("r5p", "C00117", "R5P", 1, "YIL042C", 1, 1130.5742167158426),
    ("r5p", "C00117", "R5P", 2, "YIL042C", 1, 0.8147119220654774),
    ("r5p", "C00117", "R5P", 2, "YIL042C", 2, 0.8534622605378177),
    ("r5p", "C00117", "R5P", 2, "YIL042C", 3, 0.5775148013063441),
    ("3pg;2pg", "C00197;C00631", "3PG", 1, "YIL042C", 1, 800.0),
    ("3pg;2pg", "C00197;C00631", "3PG", 2, "YIL042C", 1, 0.4),
]


def test_issue_595_example_cells_are_kept_per_protocol(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """Contract (issue #595): YIL042C ``r5p`` was stored as mean 283.205, SE 282.456,
    n 4 (protocol 1's 1130.574 pooled with protocol 2's 0.815, 0.853, 0.578), and the WT
    ``3pg;2pg`` reference as mean 425.329, SE 424.941, n 2. Now YIL042C is two records:
    protocol 1 stores r5p 1130.574 (n 1) against WT 3pg;2pg 850.270, protocol 2 stores
    the mean of its three replicates (n 3) against WT 3pg;2pg 0.388.
    """
    ds = MetaboliteZelezniak2018Dataset(
        root=str(_metabolite_root(tmp_path, ISSUE_595_ROWS))
    )
    assert len(ds) == 2
    p1, p2 = ds[0], ds[1]
    assert p1["experiment"]["phenotype"]["measurement_type"] == MTYPE_1
    assert p1["experiment"]["phenotype"]["metabolite_level"]["r5p"] == (
        1130.5742167158426
    )
    assert p1["experiment"]["phenotype"]["n_replicates"]["r5p"] == 1
    assert p1["reference"]["phenotype_reference"]["metabolite_level"]["3pg;2pg"] == (
        850.2700803858073
    )
    assert p1["reference"]["phenotype_reference"]["n_replicates"]["3pg;2pg"] == 1

    reps = [0.8147119220654774, 0.8534622605378177, 0.5775148013063441]
    mean = sum(reps) / 3
    sd = math.sqrt(sum((v - mean) ** 2 for v in reps) / 2)
    phenotype = p2["experiment"]["phenotype"]
    assert phenotype["measurement_type"] == MTYPE_2
    assert phenotype["metabolite_level"]["r5p"] == pytest.approx(mean, rel=1e-12)
    assert phenotype["metabolite_level_se"]["r5p"] == pytest.approx(
        sd / math.sqrt(3), rel=1e-12
    )
    assert phenotype["n_replicates"]["r5p"] == 3
    reference = p2["reference"]["phenotype_reference"]
    assert reference["measurement_type"] == MTYPE_2
    assert reference["metabolite_level"]["3pg;2pg"] == 0.387544549099187
    assert reference["n_replicates"]["3pg;2pg"] == 1
    assert round(mean, 6) == 0.748563


def test_metabolome_unknown_protocol_refuses(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """A ``dataset`` value with no sourced protocol refuses: its unit is unknown."""
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 4, "YDR003W", 1, 1.0),
    ]
    with pytest.raises(RuntimeError) as info:
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak metabolome protocol(s) [4] have no sourced "
        "ZelezniakMetaboliteProtocol; their unit and calibration are unknown"
    )


def test_metabolome_protocol_without_wt_refuses(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """Strain rows on protocol 2 with WT only on protocol 1 refuse: never a
    cross-protocol reference.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 2, "YDR003W", 1, 1.0),
    ]
    with pytest.raises(RuntimeError) as info:
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak metabolome protocol 2 has strain rows but no WT rows; its records "
        "would have no same-protocol reference"
    )


def test_metabolome_no_shared_metabolite_with_the_protocol_wt_refuses(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """A (strain, protocol) whose metabolites the same-protocol WT never measured
    refuses before an empty reference reaches the schema.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("atp", "C00002", "ATP", 1, "YDR003W", 1, 5.0),
    ]
    with pytest.raises(RuntimeError) as info:
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak metabolome strain YDR003W protocol 1 shares no metabolite with "
        "that protocol's WT; its reference would be empty"
    )


def test_metabolome_repeated_replicate_within_a_protocol_refuses(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """``n`` is the row count, so a repeated (metabolite, replicate) id inside one
    protocol refuses; the same replicate id under two protocols is not a repeat.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 2, "WT", 1, 4.0),
        ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 1, 1.0),
        ("pyr", "C00022", "Pyruvate", 2, "YDR003W", 1, 1.5),
        ("pyr", "C00022", "Pyruvate", 2, "YDR003W", 1, 1.5),
    ]
    with pytest.raises(RuntimeError) as info:
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak metabolome strain YDR003W protocol 2: 2 rows share a (metabolite, "
        "replicate) id, first pyr replicate 1; a repeated replicate would count as an "
        "extra replicate"
    )


def test_protocols_record_unit_and_calibration_per_protocol() -> None:
    """Protocol 1 is the only uncalibrated one and keeps the "NOT a concentration" unit;
    protocols 2 and 3 are calibrated with a typed unit gap. The verifier registry
    declares exactly these measurement_types.
    """
    p1, p2, p3 = (ZELEZNIAK_METABOLITE_PROTOCOLS[d] for d in (1, 2, 3))
    assert sorted(ZELEZNIAK_METABOLITE_PROTOCOLS) == [1, 2, 3]
    assert (p1.calibrated, p2.calibrated, p3.calibrated) == (False, True, True)
    assert p1.unit is not None and "NOT a concentration" in p1.unit
    assert p1.unit_gap is None
    for p in (p2, p3):
        assert p.unit is None
        assert p.unit_gap == CALIBRATED_UNIT_GAP
        assert "NOT a concentration" not in p.measurement_type
    assert len({p.measurement_type for p in (p1, p2, p3)}) == 3
    assert runners.METABOLITE_DATASETS["metabolite_zelezniak2018"][
        "protocol_measurement_types"
    ] == frozenset(p.measurement_type for p in (p1, p2, p3))


@pytest.mark.parametrize("unit", ["mM", None])
def test_protocol_requires_exactly_one_of_unit_and_gap(unit: str | None) -> None:
    """A unit with a gap, and neither, both refuse."""
    with pytest.raises(ValueError, match="exactly one of unit and unit_gap"):
        ZelezniakMetaboliteProtocol(
            dataset=9,
            measurement_type="x",
            calibrated=True,
            unit=unit,
            unit_gap=CALIBRATED_UNIT_GAP if unit else None,
            sources=(),
        )


def test_protocol_quotes_are_verbatim_in_the_mirrored_paper() -> None:
    """Every protocol quote is still a substring of the pinned ``paper.md`` bytes."""
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    root = None if data_root is None else Path(data_root) / "torchcell-library"
    if root is None or not root.is_dir():
        pytest.skip("torchcell-library mirror not mounted")
    for key, value in METABOLITE_SOURCED_VALUES.items():
        result = audit_sourced_value(value, root)
        assert result.passed, f"{key}: {result.message}"


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


# --------------------------------------------------------------------------- #
# Downloads (2026.09.30)
# --------------------------------------------------------------------------- #


class _Response:
    """The context-manager face of an ``urlopen`` response."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _fake_urlopen(
    monkeypatch: pytest.MonkeyPatch, payload: bytes
) -> list[tuple[str, str | None, int]]:
    """Record (url, User-Agent, timeout) per request and answer with ``payload``."""
    calls: list[tuple[str, str | None, int]] = []

    def urlopen(req: Any, timeout: int) -> _Response:
        calls.append((req.full_url, req.get_header("User-agent"), timeout))
        return _Response(payload)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return calls


def _bare(cls: Any, root: Path) -> Any:
    """A loader instance without the PyG init, rooted at ``root``."""
    dataset = cls.__new__(cls)
    dataset.root = str(root)
    return dataset


@pytest.mark.parametrize(
    ("cls", "filename", "url", "pinned", "label"),
    [
        (
            ProteomeZelezniak2018Dataset,
            PROTEOME_FILENAME,
            "https://zenodo.org/records/1320289/files/"
            "proteins_dataset.data_prep.tsv?download=1",
            zelezniak2018.DATA_SHA256,
            "proteome",
        ),
        (
            MetaboliteZelezniak2018Dataset,
            METABOLITE_DATA_FILENAME,
            "https://zenodo.org/api/records/1320289/files/"
            "metabolites_dataset.data_prep.tsv/content",
            zelezniak2018.METABOLITE_DATA_SHA256,
            "metabolome",
        ),
    ],
)
def test_download_refuses_a_digest_mismatch_and_writes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[Any],
    filename: str,
    url: str,
    pinned: str,
    label: str,
) -> None:
    """The proteome uses the ``?download=1`` URL, the metabolome the API content
    endpoint (the other 403s); both send a browser User-Agent with a 300 s timeout. A
    body off the pin raises ``RawSha256MismatchError`` naming the URL and both digests,
    and ``raw/`` stays empty (no file, no ``.partial``).
    """
    calls = _fake_urlopen(monkeypatch, b"tampered")
    got = hashlib.sha256(b"tampered").hexdigest()
    with pytest.raises(RawSha256MismatchError) as info:
        _bare(cls, tmp_path / "ds").download()
    assert (
        str(info.value)
        == f"sha256 mismatch for {url}: expected {pinned}, observed {got}"
    )
    assert calls == [(url, "Mozilla/5.0", 300)]
    assert list((tmp_path / "ds" / "raw").iterdir()) == []


@pytest.mark.parametrize(
    ("cls", "filename", "pinned"),
    [
        (
            ProteomeZelezniak2018Dataset,
            PROTEOME_FILENAME,
            "9ff81ecb1e2dd44d2f6e072ce5b628f0be1abdf57cdbd90d645db4d1fb64bfeb",
        ),
        (
            MetaboliteZelezniak2018Dataset,
            METABOLITE_DATA_FILENAME,
            "c4429fd8cef675d96ffacba1ed51e52ea483fd72d6978a22c04fa405f4e1b07d",
        ),
    ],
)
def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    cls: type[Any],
    filename: str,
    pinned: str,
    off_pin_raw: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract (issue #518's sweep): with the matrix already in ``raw/`` PyG skips
    ``download()``, so each loader's ``process()`` verifies it against its pin first and
    raises ``RawSha256MismatchError`` naming it and both digests before a row is read;
    no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(zelezniak2018, [filename])
    raw = staged.root / "raw" / filename
    with pytest.raises(RawSha256MismatchError) as info:
        cls(root=str(staged.root))
    assert str(info.value) == (
        f"sha256 mismatch for {raw}: expected {pinned}, observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


@pytest.mark.parametrize(
    ("cls", "filename", "pin_name"),
    [
        (ProteomeZelezniak2018Dataset, PROTEOME_FILENAME, "DATA_SHA256"),
        (
            MetaboliteZelezniak2018Dataset,
            METABOLITE_DATA_FILENAME,
            "METABOLITE_DATA_SHA256",
        ),
    ],
)
def test_download_writes_verified_bytes_and_skips_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type[Any],
    filename: str,
    pin_name: str,
) -> None:
    """With the pin set to the payload's digest the bytes land in ``raw/``; a second
    call finds the file and makes no request.
    """
    calls = _fake_urlopen(monkeypatch, b"payload bytes")
    monkeypatch.setattr(
        zelezniak2018, pin_name, hashlib.sha256(b"payload bytes").hexdigest()
    )
    dataset = _bare(cls, tmp_path / "ds")
    dataset.download()
    assert (tmp_path / "ds" / "raw" / filename).read_bytes() == b"payload bytes"
    dataset.download()
    assert len(calls) == 1


def test_proteome_build_without_raw_downloads_then_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``raw/`` file: PyG calls ``download()``, which fetches the (synthetic) matrix,
    verifies it against the pin (set to its digest) and writes it; ``process()`` then
    builds the two records of the main fixture.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    source = _proteome_root(tmp_path / "source", PROTEOME_ROWS)
    payload = (source / "raw" / PROTEOME_FILENAME).read_bytes()
    calls = _fake_urlopen(monkeypatch, payload)
    monkeypatch.setattr(
        zelezniak2018, "DATA_SHA256", hashlib.sha256(payload).hexdigest()
    )
    root = tmp_path / "proteome_zelezniak2018"
    ds = ProteomeZelezniak2018Dataset(root=str(root))
    assert len(calls) == 1
    assert (root / "raw" / PROTEOME_FILENAME).read_bytes() == payload
    assert len(ds) == 2
    assert ds[1]["experiment"]["phenotype"]["protein_abundance"] == {
        "YAL001C": 2.0,
        "YBR002C": 22.0,
    }
    assert ds.experiment_class is ProteinAbundanceExperiment
    assert ds.reference_class is ProteinAbundanceExperimentReference


# --------------------------------------------------------------------------- #
# Proteome replicate edges (2026.09.30)
# --------------------------------------------------------------------------- #


def test_proteome_first_gene_name_wins_and_n_counts_distinct_replicates(
    tmp_path: Path,
) -> None:
    """Strain YDR003W: YAL001C rows (rep 1: 8), (rep 2: 10); YBR002C (rep 1: 6). The
    first row's ``KO_gene_name`` (KIN3) is stored over the later ``kin3x``; YAL001C is
    mean 9.0, SD sqrt(2), SE 1.0, n 2; YBR002C is n 1 with SE NaN.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "WT", "WT", 2, 12.0),
        ("YBR002C", "WT", "WT", 1, 5.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "kin3x", 2, 10.0),
        ("YBR002C", "YDR003W", "KIN3", 1, 6.0),
    ]
    root = _proteome_root(tmp_path, rows)
    ds = ProteomeZelezniak2018Dataset(root=str(root))
    assert len(ds) == 1
    (perturbation,) = ds[0]["experiment"]["genotype"]["perturbations"]
    assert perturbation["perturbed_gene_name"] == "KIN3"
    phenotype = ds[0]["experiment"]["phenotype"]
    assert phenotype["protein_abundance"] == {"YAL001C": 9.0, "YBR002C": 6.0}
    assert phenotype["n_replicates"] == {"YAL001C": 2, "YBR002C": 1}
    se = phenotype["protein_abundance_se"]
    assert list(se) == ["YAL001C", "YBR002C"]
    assert se["YAL001C"] == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(se["YBR002C"])
    assert (root / "preprocess" / "data.csv").read_text() == "orf,gene\nYDR003W,KIN3\n"
    reference = ds[0]["reference"]["phenotype_reference"]
    assert reference["n_replicates"] == {"YAL001C": 2, "YBR002C": 1}


def test_proteome_repeated_replicate_id_refuses_naming_the_strain(
    tmp_path: Path,
) -> None:
    """Contract (issue #520): ``n`` is the row count, so a repeated (ORF, strain,
    replicate) row would be counted as a second replicate (n 2, SE 0.0 for two copies of
    8.0). The loader refuses instead, naming the strain, the number of rows in repeated
    groups, and the first such (protein, replicate). The pinned release has 0 such rows.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "WT", "WT", 2, 12.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "KIN3", 2, 9.0),
    ]
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak proteome strain YDR003W: 2 rows share a (protein, replicate) id, "
        "first YAL001C replicate 1; a repeated replicate would count as an extra "
        "replicate"
    )


def test_proteome_repeated_replicate_id_in_the_wt_reference_refuses(
    tmp_path: Path,
) -> None:
    """The WT reference goes through the same aggregation, so its repeat refuses with
    the strain named ``WT`` before any knockout strain is read.
    """
    rows = [
        ("YAL001C", "WT", "WT", 3, 10.0),
        ("YAL001C", "WT", "WT", 3, 12.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
    ]
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak proteome strain WT: 2 rows share a (protein, replicate) id, "
        "first YAL001C replicate 3; a repeated replicate would count as an extra "
        "replicate"
    )


def test_proteome_non_finite_value_refuses(tmp_path: Path) -> None:
    """Contract (issue #520 review): ``inf`` is not blank, so it passed the blank check
    and would give an infinite mean and a NaN SE. It refuses, naming the strain and the
    first non-finite cell, before the store is opened. The pinned release has 0
    non-finite values of 264,264.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "WT", "WT", 2, 12.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "KIN3", 2, "inf"),
    ]
    root = _proteome_root(tmp_path, rows)
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(root))
    assert str(info.value) == (
        "Zelezniak proteome strain YDR003W: 1 non-finite protein value(s), first "
        "YAL001C replicate 2; a non-finite value has no mean or SE"
    )
    assert not (root / "processed" / "lmdb").exists()


def test_proteome_blank_value_refuses_instead_of_shrinking_n(tmp_path: Path) -> None:
    """Contract (issue #520): pandas leaves a blank value out of ``count``, so YAL001C
    (rep 1: 8, rep 2: blank) would be stored as n 1 with nothing recording the missing
    replicate. The loader refuses, naming the strain and the first blank cell. The
    pinned release has 0 blank values.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "WT", "WT", 2, 12.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "KIN3", 2, ""),
    ]
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak proteome strain YDR003W: 1 blank protein value(s), first YAL001C "
        "replicate 2; a blank would drop out of n_replicates unrecorded"
    )


def test_proteome_all_blank_protein_refuses_with_a_loader_message(
    tmp_path: Path,
) -> None:
    """Contract (issue #520): a protein whose every value in one strain is blank used to
    reach ``ProteinAbundancePhenotype`` validation ("n_replicates for YBR002C must be
    >= 1"); it now refuses in the loader, naming the strain, before any schema object.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YBR002C", "WT", "WT", 1, 5.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YBR002C", "YDR003W", "KIN3", 1, ""),
        ("YBR002C", "YDR003W", "KIN3", 2, ""),
    ]
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak proteome strain YDR003W: 2 blank protein value(s), first YBR002C "
        "replicate 1; a blank would drop out of n_replicates unrecorded"
    )


# --------------------------------------------------------------------------- #
# build_metabolite_s_id_map on a fake YeastGEM (2026.09.30)
# --------------------------------------------------------------------------- #


def _met(met_id: str, compartment: str, **annotation: Any) -> SimpleNamespace:
    keys = {"kegg": "kegg.compound", "bigg": "bigg.metabolite"}
    return SimpleNamespace(
        id=met_id,
        compartment=compartment,
        annotation={keys[k]: v for k, v in annotation.items()},
    )


_FAKE_METABOLITES = [
    _met("s_1400", "m", kegg="C00022", bigg="pyr"),  # mitochondrial pyruvate, first
    _met("s_1399", "c", kegg="C00022", bigg="pyr"),  # cytosolic pyruvate
    _met("s_0454", "m", kegg="C04236", bigg="3c3hmp"),  # no cytosolic form
    _met("s_0188", "c", kegg=["C00197", "C99999"], bigg="3pg"),  # list annotation
    _met("s_0434", "c", bigg="atp"),  # BiGG only
    _met("s_9999", "c"),  # no annotation at all
]


@pytest.fixture
def fake_gem(monkeypatch: pytest.MonkeyPatch) -> None:
    """``YeastGEM().model.metabolites`` is the six-metabolite list above."""

    class _FakeGEM:
        def __init__(self) -> None:
            self.model = SimpleNamespace(metabolites=_FAKE_METABOLITES)

    monkeypatch.setattr(zelezniak2018, "YeastGEM", _FakeGEM)


def test_s_id_map_prefers_kegg_then_cytosol_then_first_listed(fake_gem: None) -> None:
    """``pyr`` (C00022) has m and c forms: the cytosolic s_1399 wins over the first
    listed s_1400. ``3c3hmp`` (C04236) has only the m form: s_0454. ``3pg;2pg`` resolves
    through its first KEGG token C00197, which sits in a list-valued annotation. ``atp``
    carries the string ``nan`` as its KEGG id (a blank cell after ``str()``), so the
    BiGG token ``atp`` decides. ``g6p`` (KEGG C99999, the second token of s_0188's
    list) maps to s_0188, so every token of a list annotation is indexed.
    """
    assert zelezniak2018.build_metabolite_s_id_map(
        {
            "pyr": "C00022",
            "3c3hmp": "C04236",
            "3pg;2pg": "C00197;C00631",
            "atp": "nan",
            "g6p": "C99999",
        }
    ) == {
        "pyr": "s_1399",
        "3c3hmp": "s_0454",
        "3pg;2pg": "s_0188",
        "atp": "s_0434",
        "g6p": "s_0188",
    }


def test_s_id_map_bigg_fallback_uses_the_first_merged_token(fake_gem: None) -> None:
    """An empty KEGG id skips the KEGG index; ``3pg;2pg`` then matches BiGG ``3pg``."""
    assert zelezniak2018.build_metabolite_s_id_map({"3pg;2pg": ""}) == {
        "3pg;2pg": "s_0188"
    }


def test_s_id_map_refuses_an_unmatched_metabolite(fake_gem: None) -> None:
    with pytest.raises(RuntimeError) as info:
        zelezniak2018.build_metabolite_s_id_map({"xyz;abc": "C00001;C00002"})
    assert str(info.value) == (
        "no Yeast9 s_NNNN found for metabolite 'xyz;abc' (kegg 'C00001', bigg 'xyz')"
    )


def test_metabolome_build_runs_the_real_mapper_on_the_fake_model(
    tmp_path: Path, fake_gem: None
) -> None:
    """Without the mapper stub, ``process()`` feeds the deduplicated metabolite -> KEGG
    map to ``build_metabolite_s_id_map``; the fake model maps pyr to the cytosolic
    s_1399, so record 0's targets are exactly {pyr: s_1399}.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 2, "WT", 1, 4.0),
        ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 1, 1.0),
        ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 2, 3.0),
    ]
    ds = MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert len(ds) == 1
    phenotype = ds[0]["experiment"]["phenotype"]
    assert phenotype["target_metabolite_ids"] == {"pyr": "s_1399"}
    assert phenotype["metabolite_level"] == {"pyr": 2.0}
    assert ds.experiment_class is MetaboliteExperiment
    assert ds.reference_class is MetaboliteExperimentReference


def test_preprocess_raw_is_identity_for_both_loaders(tmp_path: Path) -> None:
    frame = object()
    for cls in (ProteomeZelezniak2018Dataset, MetaboliteZelezniak2018Dataset):
        assert _bare(cls, tmp_path).preprocess_raw(frame) is frame
