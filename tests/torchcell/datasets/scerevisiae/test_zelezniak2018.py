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

Proteome fixture (``proteins_dataset.data_prep.tsv``; one record per released sample
since #1174, ``sample`` column ``S_<KO_ORF>_<replicate>``; the reference is the WT
summary, ``groupby("ORF")`` mean / sample SD / count, SE = SD / sqrt(n)):

    WT       YAL001C 10, 12        -> records 0-1; reference mean 11.0, SE 1.0, n 2
    WT       YBR002C 5, 7, 9       -> records 0-2; reference mean 7.0, SE 2/sqrt(3), n 3
    YDR003W  YAL001C 8, 9          -> records 3-4 (their mean 8.5 is the old record)
    YDR003W  YBR002C 6, 6          -> records 3-4 (mean 6.0)
    YER004W  YAL001C 1, 3          -> records 5-6 (mean 2.0)
    YER004W  YBR002C 20, 24        -> records 5-6 (mean 22.0)
    bad_ko   YAL001C 99            -> record 7: KO_ORF has no ORF form, the
                                      released-gene-name leaf

Metabolome fixture (``metabolites_dataset.data_prep.tsv``; one record per (protocol,
strain, replicate) since #1173, the WT included; the reference is the same-protocol WT
summary restricted to the record's metabolites, issue #595):

    WT       pyr, 3pg;2pg protocol 1: 2, 10   -> record 0 (WT, empty genotype)
    YDR003W  pyr, atp     protocol 1: 1/3, 5/5 -> records 1-2 (no WT atp)
    YER004W  3pg;2pg      protocol 1: 7        -> record 3
    WT       pyr          protocol 2: 4        -> record 4 (WT)
    YDR003W  pyr          protocol 2: 6, 8     -> records 5-6

Records are ordered by (protocol, strain, replicate). Before issue #595 the WT pyr
baseline was the pooled 3.0 (n 2) and protocol-2 rows joined the protocol-1 strain mean;
before #1173 / #1174 each (strain, protocol) or strain was one mean + SE record.

2026.11.24 (issues #1173, #1174; policy #1178): every record is one released replicate,
verbatim, n 1, no SE map and a typed ``not_applicable_single_measurement`` gap; WT
replicates are records with the empty genotype; record means reproduce the old
aggregates; a replicate label naming two samples and a sample under two labels refuse.
Two ``slow`` + ``data`` tests build the pinned release from the raw mirror (364 and 192
records) and check the per-strain means against the raw rows. The README column quotes
are audited against the raw mirror.

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
    _SINGLE_REFERENCE_SE_NOTE as SINGLE_REFERENCE_SE_NOTE,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    _SINGLE_REPLICATE_SE_NOTE as SINGLE_REPLICATE_SE_NOTE,
)
from torchcell.datasets.scerevisiae.zelezniak2018 import (
    CALIBRATED_UNIT_GAP,
    METABOLITE_DATA_FILENAME,
    METABOLITE_SOURCED_VALUES,
    README_SOURCED_VALUES,
    REPLICATE_SOURCED_VALUES,
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
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification import runners
from torchcell.verification.sourced import (
    ProvenanceGap,
    ProvenanceGapReason,
    audit_sourced_value,
)

MTYPE_1 = ZELEZNIAK_METABOLITE_PROTOCOLS[1].measurement_type
MTYPE_2 = ZELEZNIAK_METABOLITE_PROTOCOLS[2].measurement_type

# --------------------------------------------------------------------------- #
# Shared expected pieces
# --------------------------------------------------------------------------- #

BY4741 = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="BY4741",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)
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


def _proteome_root(
    tmp_path: Path,
    rows: list[tuple[Any, ...]],
    samples: dict[tuple[str, int], str] | None = None,
) -> Path:
    """Write ``rows`` (ORF, KO_ORF, KO_gene_name, replicate, value) in the release's
    column order, with the ``sample`` column named ``S_<KO_ORF>_<replicate>`` unless
    ``samples`` names a (KO_ORF, replicate) otherwise.
    """
    root = tmp_path / "proteome_zelezniak2018"
    (root / "raw").mkdir(parents=True)
    names = samples or {}
    _write_tsv(
        root / "raw" / PROTEOME_FILENAME,
        ["ORF", "sample", "replicate", "KO_ORF", "KO_gene_name", "value"],
        [
            (orf, names.get((ko, rep), f"S_{ko}_{rep}"), rep, ko, gene, value)
            for orf, ko, gene, rep, value in rows
        ],
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
    assert manifest["manifest_schema_version"] == 3
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


def _single_se_gap(field: str) -> list[ProvenanceGap]:
    return [
        ProvenanceGap(
            field=field,
            reason=ProvenanceGapReason.not_applicable_single_measurement,
            note=SINGLE_REPLICATE_SE_NOTE,
        )
    ]


def _kanmx(orf: str, gene: str) -> Genotype:
    return Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name=orf, perturbed_gene_name=gene
            )
        ]
    )


def _sample_phenotype(
    abundance: dict[str, float], replicate_id: str
) -> ProteinAbundancePhenotype:
    return ProteinAbundancePhenotype(
        protein_abundance=abundance,
        protein_abundance_se=None,
        n_replicates=dict.fromkeys(abundance, 1),
        measurement_type=PROTEOME_MEASUREMENT_TYPE,
        replicate_id=replicate_id,
        provenance_gaps=_single_se_gap("protein_abundance_se"),
    )


def test_proteome_one_record_per_released_sample_wt_included(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Contract (#1174): every released sample is a record, the WT samples included.

    The fixture releases 3 WT samples, 2 for YDR003W, 2 for YER004W and 1 for ``bad_ko``,
    so 8 records. ``KO_ORF`` groups sort (``WT`` < ``YDR003W`` < ``YER004W`` <
    ``bad_ko``), and samples inside a strain sort by ``replicate``. The ``bad_ko`` strain
    is stored on the released-gene-name leaf (#1045), not skipped and not raised; the WT
    samples carry the empty genotype, so the gene set is still the three knockouts.
    ``preprocess/data.csv`` lists every record with its replicate label and sample name.
    """
    ds, root = proteome
    assert len(ds) == 8
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,gene,replicate_id,sample\n"
        "WT,WT,1,S_WT_1\nWT,WT,2,S_WT_2\nWT,WT,3,S_WT_3\n"
        "YDR003W,KIN3,1,S_YDR003W_1\nYDR003W,KIN3,2,S_YDR003W_2\n"
        "YER004W,KIN4,1,S_YER004W_1\nYER004W,KIN4,2,S_YER004W_2\n"
        "bad_ko,BAD,1,S_bad_ko_1\n"
    )
    assert [ds[i]["experiment"]["genotype"]["perturbations"] for i in range(3)] == [
        [],
        [],
        [],
    ]
    assert [ds[i]["experiment"]["phenotype"]["replicate_id"] for i in range(8)] == [
        "1",
        "2",
        "3",
        "1",
        "2",
        "1",
        "2",
        "1",
    ]
    (released,) = ds[7]["experiment"]["genotype"]["perturbations"]
    assert (
        released["perturbation_type"],
        released["systematic_gene_name"],
        released["perturbed_gene_name"],
    ) == ("released_identifier", "bad_ko", "BAD")
    _assert_side_files(
        root, ProteomeZelezniak2018Dataset, ["YDR003W", "YER004W", "bad_ko"]
    )


def test_proteome_sample_records_are_verbatim_with_a_typed_se_gap(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Records 3 and 4 = the two YDR003W/KIN3 samples, each value verbatim.

    Replicate 1 is {YAL001C 8, YBR002C 6}, replicate 2 {YAL001C 9, YBR002C 6}: n 1 per
    protein, no SE map and a ``not_applicable_single_measurement`` gap on it. The
    reference is the WT summary over ALL proteins (not restricted, unlike the
    metabolome loader), unchanged from the per-strain loader. The whole dumps are
    compared exactly and validate back into the schema classes unchanged.
    """
    ds, _ = proteome
    for index, replicate_id, abundance in (
        (3, "1", {"YAL001C": 8.0, "YBR002C": 6.0}),
        (4, "2", {"YAL001C": 9.0, "YBR002C": 6.0}),
    ):
        record = ds[index]
        expected = ProteinAbundanceExperiment(
            dataset_name="ProteomeZelezniak2018Dataset",
            genotype=_kanmx("YDR003W", "KIN3"),
            environment=ENVIRONMENT,
            phenotype=_sample_phenotype(abundance, replicate_id),
        )
        assert record["experiment"] == expected.model_dump()
        assert record["reference"] == _proteome_reference().model_dump()
        assert record["publication"] == PUBLICATION.model_dump()
        assert (
            ProteinAbundanceExperiment.model_validate(record["experiment"]).model_dump()
            == record["experiment"]
        )


def test_proteome_wt_samples_are_records_with_the_empty_genotype(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Contract (#1174, policy 7): the 3 WT samples are records of the parent strain.

    Each is ``Genotype(perturbations=[])`` with its own values; WT replicate 3 released
    only YBR002C, so its record holds that one protein. The reference is still the WT
    summary with ``replicate_id=None``.
    """
    ds, _ = proteome
    expected = [
        ("1", {"YAL001C": 10.0, "YBR002C": 5.0}),
        ("2", {"YAL001C": 12.0, "YBR002C": 7.0}),
        ("3", {"YBR002C": 9.0}),
    ]
    for index, (replicate_id, abundance) in enumerate(expected):
        record = ds[index]
        assert record["experiment"] == (
            ProteinAbundanceExperiment(
                dataset_name="ProteomeZelezniak2018Dataset",
                genotype=Genotype(perturbations=[]),
                environment=ENVIRONMENT,
                phenotype=_sample_phenotype(abundance, replicate_id),
            ).model_dump()
        )
        assert record["reference"]["phenotype_reference"]["replicate_id"] is None


def test_proteome_record_means_reproduce_the_per_strain_aggregate(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """Grouping the records of one strain and averaging gives the values the per-strain
    loader stored before #1174: YDR003W means 8.5 and 6.0, YER004W 2.0 and 22.0, and the
    WT records average to the reference (11.0, 7.0). Nothing is lost or reweighted.
    """
    ds, _ = proteome
    by_strain: dict[str, dict[str, list[float]]] = {}
    for i in range(len(ds)):
        experiment = ds[i]["experiment"]
        perturbations = experiment["genotype"]["perturbations"]
        strain = perturbations[0]["systematic_gene_name"] if perturbations else "WT"
        for orf, value in experiment["phenotype"]["protein_abundance"].items():
            by_strain.setdefault(strain, {}).setdefault(orf, []).append(value)
    means = {
        strain: {orf: sum(v) / len(v) for orf, v in cells.items()}
        for strain, cells in by_strain.items()
    }
    assert means == {
        "WT": {"YAL001C": 11.0, "YBR002C": 7.0},
        "YDR003W": {"YAL001C": 8.5, "YBR002C": 6.0},
        "YER004W": {"YAL001C": 2.0, "YBR002C": 22.0},
        "bad_ko": {"YAL001C": 99.0},
    }
    reference = ds[0]["reference"]["phenotype_reference"]
    assert reference["protein_abundance"] == means["WT"]


def test_proteome_reference_index_has_one_shared_reference(
    proteome: tuple[ProteomeZelezniak2018Dataset, Path],
) -> None:
    """All eight records share the single WT reference -> one index entry, [0..7]."""
    ds, root = proteome
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert len(index) == 1
    assert index[0]["member_indices"] == list(range(8))
    assert index[0]["reference"]["dataset_name"] == "ProteomeZelezniak2018Dataset"
    assert index[0]["reference"]["experiment_reference_type"] == "protein_abundance"
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == [list(range(8))]


def test_proteome_single_replicate_wt_protein_gets_nan_reference_se(
    tmp_path: Path,
) -> None:
    """A protein the WT released once has SE NaN in the reference summary (not None,
    not 0), while the knockout sample record itself has no SE map at all.
    """
    rows = [("YAL001C", "WT", "WT", 1, 10.0), ("YAL001C", "YDR003W", "KIN3", 1, 8.0)]
    ds = ProteomeZelezniak2018Dataset(root=str(_proteome_root(tmp_path, rows)))
    assert len(ds) == 2
    reference = ds[1]["reference"]["phenotype_reference"]
    assert reference["protein_abundance"] == {"YAL001C": 10.0}
    assert reference["n_replicates"] == {"YAL001C": 1}
    assert list(reference["protein_abundance_se"]) == ["YAL001C"]
    assert math.isnan(reference["protein_abundance_se"]["YAL001C"])
    assert ds[1]["experiment"]["phenotype"]["protein_abundance_se"] is None


def test_proteome_replicate_label_naming_two_samples_refuses(tmp_path: Path) -> None:
    """One (strain, replicate) label released under two sample names would make one
    record out of two samples, so it refuses naming the strain and the label.
    """
    root = tmp_path / "proteome_zelezniak2018"
    (root / "raw").mkdir(parents=True)
    _write_tsv(
        root / "raw" / PROTEOME_FILENAME,
        ["ORF", "sample", "replicate", "KO_ORF", "KO_gene_name", "value"],
        [
            ("YAL001C", "w1", 1, "WT", "WT", 10.0),
            ("YAL001C", "k1", 1, "YDR003W", "KIN3", 8.0),
            ("YBR002C", "k1_other", 1, "YDR003W", "KIN3", 6.0),
        ],
    )
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(root))
    assert str(info.value) == (
        "Zelezniak proteome strain YDR003W: replicate 1 names 2 samples; one "
        "replicate record would mix them"
    )


def test_proteome_sample_with_two_labels_refuses(tmp_path: Path) -> None:
    """One sample name under two (KO_ORF, replicate) labels would be split across two
    records, so it refuses before any record is built.
    """
    rows = [
        ("YAL001C", "WT", "WT", 1, 10.0),
        ("YAL001C", "YDR003W", "KIN3", 1, 8.0),
        ("YAL001C", "YDR003W", "KIN3", 2, 9.0),
    ]
    root = _proteome_root(
        tmp_path, rows, samples={("YDR003W", 1): "k", ("YDR003W", 2): "k"}
    )
    with pytest.raises(RuntimeError) as info:
        ProteomeZelezniak2018Dataset(root=str(root))
    assert str(info.value) == (
        "Zelezniak proteome sample k carries more than one (KO_ORF, replicate) label; "
        "it would be split across records"
    )


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
    """Seven (protocol, strain, replicate) records, the two WT replicates included; the
    mapper is called once with the deduplicated metabolite -> KEGG map.

    ``preprocess/data.csv`` reports the protocol, replicate label and ``n_metabolites``
    per record, ordered by (protocol, strain, replicate): ``WT`` sorts before the ORFs.
    The gene set is still the two KO ORFs (a WT record deletes nothing).
    """
    ds, root, calls = metabolome
    assert len(ds) == 7
    assert calls == [{"pyr": "C00022", "3pg;2pg": "C00197;C00631", "atp": "C00002"}]
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,dataset,replicate_id,n_metabolites\n"
        "WT,1,1,2\nYDR003W,1,1,2\nYDR003W,1,2,2\nYER004W,1,1,1\n"
        "WT,2,1,1\nYDR003W,2,1,1\nYDR003W,2,2,1\n"
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


def _replicate_metabolites(
    level: dict[str, float], mtype: str, replicate_id: str
) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=None,
        n_replicates=dict.fromkeys(level, 1),
        measurement_type=mtype,
        target_metabolite_ids={m: S_ID_MAP[m] for m in level},
        replicate_id=replicate_id,
        provenance_gaps=_single_se_gap("metabolite_level_se"),
    )


def _single_wt_reference(level: dict[str, float], mtype: str) -> MetabolitePhenotype:
    """The reference summary of a protocol whose WT released one replicate."""
    return MetabolitePhenotype(
        metabolite_level=level,
        metabolite_level_se=None,
        n_replicates=dict.fromkeys(level, 1),
        measurement_type=mtype,
        target_metabolite_ids={m: S_ID_MAP[m] for m in level},
        provenance_gaps=[
            ProvenanceGap(
                field="metabolite_level_se",
                reason=ProvenanceGapReason.not_applicable_single_measurement,
                note=SINGLE_REFERENCE_SE_NOTE,
            )
        ],
    )


def test_metabolome_replicate_records_are_verbatim_against_the_same_protocol_wt(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Records 1 and 2 = YDR003W protocol 1, replicates 1 and 2, values verbatim (pyr 1
    and 3, atp 5 and 5), n 1, no SE map and a typed gap. The reference has pyr ONLY, at
    the protocol-1 WT value 2.0 (n 1), never the pooled 3.0 of protocols 1 and 2: WT
    never measured atp, so the reference is restricted to {pyr}; its single replicate
    has no dispersion, stored as None with a gap. Genotype uses the ORF as both names
    (no gene name column in the metabolome file). The dumps are compared exactly.
    """
    ds, _, _ = metabolome
    expected_reference = MetaboliteExperimentReference(
        dataset_name="MetaboliteZelezniak2018Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=_single_wt_reference({"pyr": 2.0}, MTYPE_1),
    )
    for index, replicate_id, level in (
        (1, "1", {"pyr": 1.0, "atp": 5.0}),
        (2, "2", {"pyr": 3.0, "atp": 5.0}),
    ):
        record = ds[index]
        expected = MetaboliteExperiment(
            dataset_name="MetaboliteZelezniak2018Dataset",
            genotype=_ydr003w(),
            environment=ENVIRONMENT,
            phenotype=_replicate_metabolites(level, MTYPE_1, replicate_id),
        )
        assert record["experiment"] == expected.model_dump()
        assert record["reference"] == expected_reference.model_dump()
        assert record["publication"] == PUBLICATION.model_dump()
        assert (
            MetaboliteExperiment.model_validate(record["experiment"]).model_dump()
            == record["experiment"]
        )


def test_metabolome_protocol_2_records_of_the_same_strain(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Records 5 and 6 = YDR003W on protocol 2: pyr 6.0 and 8.0, against the protocol-2
    WT pyr 4.0. The protocol-1 rows (1, 3) are in neither, and both sides carry the
    protocol-2 measurement_type.
    """
    ds, _, _ = metabolome
    for index, replicate_id, value in ((5, "1", 6.0), (6, "2", 8.0)):
        record = ds[index]
        assert record["experiment"]["phenotype"] == (
            _replicate_metabolites({"pyr": value}, MTYPE_2, replicate_id).model_dump()
        )
        reference = record["reference"]["phenotype_reference"]
        assert reference == _single_wt_reference({"pyr": 4.0}, MTYPE_2).model_dump()


def test_metabolome_wt_replicates_are_records_with_the_empty_genotype(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Contract (#1173, policy 7): each WT replicate is a record of the parent strain,
    ``Genotype(perturbations=[])``, against the summary of its own protocol's WT
    restricted to its own metabolites.
    """
    ds, _, _ = metabolome
    for index, mtype, level in (
        (0, MTYPE_1, {"pyr": 2.0, "3pg;2pg": 10.0}),
        (4, MTYPE_2, {"pyr": 4.0}),
    ):
        record = ds[index]
        assert record["experiment"] == (
            MetaboliteExperiment(
                dataset_name="MetaboliteZelezniak2018Dataset",
                genotype=Genotype(perturbations=[]),
                environment=ENVIRONMENT,
                phenotype=_replicate_metabolites(level, mtype, "1"),
            ).model_dump()
        )
        reference = record["reference"]["phenotype_reference"]
        assert reference == _single_wt_reference(level, mtype).model_dump()


def test_metabolome_single_replicate_record_shape_is_pinned(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Record 3 = YER004W protocol 1, replicate 1: ``3pg;2pg`` 7.0, kept verbatim as the
    ``;``-merged key and mapped to the stub's ``s_0188``.

    The literal dict also pins the schema envelope fields (``provenance_gaps``,
    ``graph_level``, ``label_name``, ``label_statistic_name``), so adding a field to
    ``MetabolitePhenotype`` fails this test with the loader unchanged. That is intended:
    a stored-record shape change should be noticed here.
    """
    ds, _, _ = metabolome
    record = ds[3]
    phenotype = record["experiment"]["phenotype"]
    assert phenotype == {
        "provenance_gaps": [
            gap.model_dump() for gap in _single_se_gap("metabolite_level_se")
        ],
        "graph_level": "metabolism",
        "label_name": "metabolite_level",
        "label_statistic_name": "metabolite_level_se",
        "metabolite_level": {"3pg;2pg": 7.0},
        "metabolite_level_se": None,
        "n_replicates": {"3pg;2pg": 1},
        "measurement_type": MTYPE_1,
        "target_metabolite_ids": {"3pg;2pg": "s_0188"},
        "replicate_id": "1",
    }
    reference = record["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"3pg;2pg": 10.0}
    assert reference["metabolite_level_se"] is None
    assert reference["n_replicates"] == {"3pg;2pg": 1}
    assert reference["replicate_id"] is None
    assert reference["target_metabolite_ids"] == {"3pg;2pg": "s_0188"}


def test_metabolome_reference_summary_with_several_wt_replicates_keeps_its_se(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    """A protocol whose WT released three replicates (protocol 3 in the real file) gets
    a reference with mean, SD / sqrt(n) and n 3, no gap, ``replicate_id=None``; the three
    WT replicates are records too.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 3, "WT", 1, 1.0),
        ("pyr", "C00022", "Pyruvate", 3, "WT", 2, 2.0),
        ("pyr", "C00022", "Pyruvate", 3, "WT", 3, 6.0),
        ("pyr", "C00022", "Pyruvate", 3, "YDR003W", 1, 5.0),
    ]
    ds = MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert len(ds) == 4
    assert [ds[i]["experiment"]["phenotype"]["replicate_id"] for i in range(4)] == [
        "1",
        "2",
        "3",
        "1",
    ]
    reference = ds[3]["reference"]["phenotype_reference"]
    assert reference["metabolite_level"] == {"pyr": 3.0}
    assert reference["n_replicates"] == {"pyr": 3}
    assert reference["metabolite_level_se"]["pyr"] == pytest.approx(
        math.sqrt(7.0) / math.sqrt(3), rel=1e-12
    )
    assert reference["provenance_gaps"] == []
    assert reference["replicate_id"] is None


def test_metabolome_reference_index_splits_on_protocol_and_restriction(
    metabolome: tuple[MetaboliteZelezniak2018Dataset, Path, list[dict[str, str]]],
) -> None:
    """Four distinct references: protocol-1 {pyr, 3pg;2pg} (the WT record),
    protocol-1 {pyr} (both YDR003W replicates), protocol-1 {3pg;2pg}, protocol-2 {pyr}.
    """
    _, root, _ = metabolome
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert sorted(
        (
            e["member_indices"],
            e["reference"]["phenotype_reference"]["measurement_type"],
            e["reference"]["phenotype_reference"]["metabolite_level"],
        )
        for e in index
    ) == [
        ([0], MTYPE_1, {"pyr": 2.0, "3pg;2pg": 10.0}),
        ([1, 2], MTYPE_1, {"pyr": 2.0}),
        ([3], MTYPE_1, {"3pg;2pg": 10.0}),
        ([4, 5, 6], MTYPE_2, {"pyr": 4.0}),
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
    """Contract (issue #595, then #1173): YIL042C ``r5p`` was stored as mean 283.205,
    SE 282.456, n 4 (protocol 1's 1130.574 pooled with protocol 2's 0.815, 0.853,
    0.578), and the WT ``3pg;2pg`` reference as mean 425.329, SE 424.941, n 2. Since
    #595 the protocols are apart, and since #1173 every replicate is its own record:
    YIL042C protocol 1 is record 1 (r5p 1130.574 against WT 3pg;2pg 850.270), protocol
    2 is records 3, 4 and 5 holding the three replicate values verbatim against the
    protocol-2 WT (r5p 0.7, 3pg;2pg 0.388). Their mean, 0.748563, is the value the per-(strain, protocol) loader
    stored; it is now a modeling-time derivation. The WT replicates are records 0 and 2.
    """
    ds = MetaboliteZelezniak2018Dataset(
        root=str(_metabolite_root(tmp_path, ISSUE_595_ROWS))
    )
    assert len(ds) == 6
    assert [ds[i]["experiment"]["genotype"]["perturbations"] for i in (0, 2)] == [
        [],
        [],
    ]
    p1 = ds[1]
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
    stored = []
    for index, replicate_id in ((3, "1"), (4, "2"), (5, "3")):
        phenotype = ds[index]["experiment"]["phenotype"]
        assert phenotype["measurement_type"] == MTYPE_2
        assert phenotype["replicate_id"] == replicate_id
        assert phenotype["n_replicates"]["r5p"] == 1
        assert phenotype["metabolite_level_se"] is None
        stored.append(phenotype["metabolite_level"]["r5p"])
        reference = ds[index]["reference"]["phenotype_reference"]
        assert reference["measurement_type"] == MTYPE_2
        assert reference["n_replicates"] == dict.fromkeys(
            ds[index]["experiment"]["phenotype"]["metabolite_level"], 1
        )
    # replicate 1 also measured 3pg;2pg, replicates 2 and 3 only r5p, so each reference
    # is the protocol-2 WT restricted to that replicate's own metabolites
    assert ds[3]["reference"]["phenotype_reference"]["metabolite_level"] == {
        "r5p": 0.7,
        "3pg;2pg": 0.387544549099187,
    }
    for index in (4, 5):
        assert ds[index]["reference"]["phenotype_reference"]["metabolite_level"] == {
            "r5p": 0.7
        }
    assert stored == reps
    assert round(sum(stored) / 3, 6) == 0.748563


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
    """A replicate whose metabolites the same-protocol WT never measured refuses
    before an empty reference reaches the schema.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("atp", "C00002", "ATP", 1, "YDR003W", 1, 5.0),
    ]
    with pytest.raises(RuntimeError) as info:
        MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert str(info.value) == (
        "Zelezniak metabolome strain YDR003W protocol 1 replicate 1 shares no "
        "metabolite with that protocol's WT; its reference would be empty"
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
    """Every protocol and replicate quote is still a substring of the pinned
    ``paper.md`` bytes.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    root = None if data_root is None else Path(data_root) / "torchcell-library"
    if root is None or not root.is_dir():
        pytest.skip("torchcell-library mirror not mounted")
    for key, value in {**METABOLITE_SOURCED_VALUES, **REPLICATE_SOURCED_VALUES}.items():
        result = audit_sourced_value(value, root)
        assert result.passed, f"{key}: {result.message}"


def test_replicate_quotes_are_verbatim_in_the_raw_mirrored_readme() -> None:
    """Contract (#1173, #1174): the Zenodo README's column definitions that say what a
    released replicate is are still substrings of the pinned raw-mirror bytes.
    """
    from dotenv import load_dotenv

    load_dotenv()
    data_root = os.environ.get("DATA_ROOT")
    root = None if data_root is None else Path(data_root) / "torchcell-raw"
    if root is None or not (root / zelezniak2018.CITATION_KEY).is_dir():
        pytest.skip("torchcell-raw mirror not mounted")
    for key, value in README_SOURCED_VALUES.items():
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
    builds the eight sample records of the main fixture.
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
    assert len(ds) == 8
    assert ds[5]["experiment"]["phenotype"]["protein_abundance"] == {
        "YAL001C": 1.0,
        "YBR002C": 20.0,
    }
    assert ds.experiment_class is ProteinAbundanceExperiment
    assert ds.reference_class is ProteinAbundanceExperimentReference


# --------------------------------------------------------------------------- #
# Proteome replicate edges (2026.09.30)
# --------------------------------------------------------------------------- #


def test_proteome_first_gene_name_wins_and_reference_counts_distinct_replicates(
    tmp_path: Path,
) -> None:
    """Strain YDR003W: YAL001C rows (rep 1: 8), (rep 2: 10); YBR002C (rep 1: 6). The
    first row's ``KO_gene_name`` (KIN3) is stored over the later ``kin3x`` on BOTH
    sample records; replicate 1 is {YAL001C 8, YBR002C 6} and replicate 2 {YAL001C
    10}. The WT reference counts its distinct replicates: YAL001C n 2, YBR002C n 1.
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
    assert len(ds) == 4
    for index in (2, 3):
        (perturbation,) = ds[index]["experiment"]["genotype"]["perturbations"]
        assert perturbation["perturbed_gene_name"] == "KIN3"
    assert ds[2]["experiment"]["phenotype"]["protein_abundance"] == {
        "YAL001C": 8.0,
        "YBR002C": 6.0,
    }
    assert ds[3]["experiment"]["phenotype"]["protein_abundance"] == {"YAL001C": 10.0}
    assert (root / "preprocess" / "data.csv").read_text() == (
        "orf,gene,replicate_id,sample\n"
        "WT,WT,1,S_WT_1\nWT,WT,2,S_WT_2\n"
        "YDR003W,KIN3,1,S_YDR003W_1\nYDR003W,KIN3,2,S_YDR003W_2\n"
    )
    reference = ds[2]["reference"]["phenotype_reference"]
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
    s_1399, so the targets of record 1 (YDR003W protocol 1, replicate 1) are exactly
    {pyr: s_1399}. The four records are the two WT replicates and the two YDR003W ones.
    """
    rows = [
        ("pyr", "C00022", "Pyruvate", 1, "WT", 1, 2.0),
        ("pyr", "C00022", "Pyruvate", 2, "WT", 1, 4.0),
        ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 1, 1.0),
        ("pyr", "C00022", "Pyruvate", 1, "YDR003W", 2, 3.0),
    ]
    ds = MetaboliteZelezniak2018Dataset(root=str(_metabolite_root(tmp_path, rows)))
    assert len(ds) == 4
    phenotype = ds[1]["experiment"]["phenotype"]
    assert phenotype["target_metabolite_ids"] == {"pyr": "s_1399"}
    assert phenotype["metabolite_level"] == {"pyr": 1.0}
    assert ds.experiment_class is MetaboliteExperiment
    assert ds.reference_class is MetaboliteExperimentReference


def test_preprocess_raw_is_identity_for_both_loaders(tmp_path: Path) -> None:
    frame = object()
    for cls in (ProteomeZelezniak2018Dataset, MetaboliteZelezniak2018Dataset):
        assert _bare(cls, tmp_path).preprocess_raw(frame) is frame


# --------------------------------------------------------------------------- #
# The pinned release, built for real (2026.11.24, issues #1173 and #1174)
# --------------------------------------------------------------------------- #

_RAW_MIRROR = Path(
    os.environ["DATA_ROOT"],
    "torchcell-raw",
    zelezniak2018.CITATION_KEY,
    "data",
    "zenodo",
)


def _copy_release(tmp_path: Path, name: str, filename: str) -> Path:
    root = tmp_path / name
    (root / "raw").mkdir(parents=True)
    (root / "raw" / filename).write_bytes((_RAW_MIRROR / filename).read_bytes())
    return root


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    not (_RAW_MIRROR / PROTEOME_FILENAME).exists(),
    reason=f"requires the Zelezniak raw mirror at {_RAW_MIRROR}",
)
def test_the_released_proteome_keeps_all_364_samples(tmp_path: Path) -> None:
    """The pinned matrix (sha256 9ff81ecb...) builds 364 records, one per released
    sample: 352 over 97 knockout strains (samples per strain {2: 3, 3: 72, 4: 3, 6: 15,
    7: 4}) and the 12 WT samples. Grouping the records of one strain and averaging
    reproduces the per-strain means the loader stored before #1174 (recomputed here
    from the raw rows) to float rounding.
    """
    import pandas as pd

    root = _copy_release(tmp_path, "proteome_zelezniak2018", PROTEOME_FILENAME)
    built = ProteomeZelezniak2018Dataset(root=str(root))
    assert len(built) == 364
    summary = pd.read_csv(root / "preprocess" / "data.csv")
    per_strain = summary.groupby("orf").size()
    assert int(per_strain["WT"]) == 12
    assert per_strain.drop("WT").value_counts().to_dict() == {
        3: 72,
        6: 15,
        7: 4,
        2: 3,
        4: 3,
    }
    assert summary["sample"].nunique() == 364
    assert summary["replicate_id"].value_counts().sort_index().to_dict() == {
        1: 98,
        2: 98,
        3: 95,
        4: 23,
        5: 20,
        6: 20,
        7: 5,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
    }
    raw = pd.read_csv(root / "raw" / PROTEOME_FILENAME, sep="\t")
    expected = raw.groupby(["KO_ORF", "ORF"])["value"].mean()
    sums: dict[tuple[str, str], list[float]] = {}
    for i in range(len(built)):
        experiment = built[i]["experiment"]
        perturbations = experiment["genotype"]["perturbations"]
        strain = perturbations[0]["systematic_gene_name"] if perturbations else "WT"
        for orf, value in experiment["phenotype"]["protein_abundance"].items():
            sums.setdefault((strain, orf), []).append(value)
    assert len(sums) == len(expected) == 98 * 726
    worst = max(abs(sum(v) / len(v) - expected[key]) for key, v in sums.items())
    assert worst < 1e-12


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    not (_RAW_MIRROR / METABOLITE_DATA_FILENAME).exists(),
    reason=f"requires the Zelezniak raw mirror at {_RAW_MIRROR}",
)
def test_the_released_metabolome_keeps_all_192_replicates(tmp_path: Path) -> None:
    """The pinned table (sha256 c4429fd8...) builds 192 records, one per (protocol,
    strain, released biological replicate): protocol 1 95 knockouts + 1 WT (one
    replicate each), protocol 2 51 + 1 WT, protocol 3 41 + 3 WT. The real YeastGEM
    mapper runs (50 metabolite ids). Grouping the records of one (strain, protocol) and
    averaging reproduces the means of the raw rows to float rounding.
    """
    import pandas as pd

    root = _copy_release(tmp_path, "metabolite_zelezniak2018", METABOLITE_DATA_FILENAME)
    built = MetaboliteZelezniak2018Dataset(root=str(root))
    assert len(built) == 192
    summary = pd.read_csv(root / "preprocess" / "data.csv")
    wt = summary["orf"] == "WT"
    assert summary[~wt].groupby("dataset").size().to_dict() == {1: 95, 2: 51, 3: 41}
    assert summary[wt].groupby("dataset").size().to_dict() == {1: 1, 2: 1, 3: 3}
    assert summary.groupby(["dataset", "replicate_id"]).size().to_dict() == {
        (1, 1): 96,
        (2, 1): 19,
        (2, 2): 18,
        (2, 3): 15,
        (3, 1): 17,
        (3, 2): 14,
        (3, 3): 13,
    }
    raw = pd.read_csv(root / "raw" / METABOLITE_DATA_FILENAME, sep="\t")
    expected = raw.groupby(["dataset", "genotype", "metabolite_id"])["value"].mean()
    protocol_of = {
        p.measurement_type: d for d, p in ZELEZNIAK_METABOLITE_PROTOCOLS.items()
    }
    cells: dict[tuple[int, str, str], list[float]] = {}
    for i in range(len(built)):
        experiment = built[i]["experiment"]
        phenotype = experiment["phenotype"]
        perturbations = experiment["genotype"]["perturbations"]
        strain = perturbations[0]["systematic_gene_name"] if perturbations else "WT"
        protocol = protocol_of[phenotype["measurement_type"]]
        for metabolite, value in phenotype["metabolite_level"].items():
            cells.setdefault((protocol, strain, metabolite), []).append(value)
    assert len(cells) == len(expected)
    worst = max(abs(sum(v) / len(v) - expected[key]) for key, v in cells.items())
    assert worst < 1e-9
