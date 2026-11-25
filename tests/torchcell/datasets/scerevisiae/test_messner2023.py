# tests/torchcell/datasets/scerevisiae/test_messner2023.py
# [[tests.torchcell.datasets.scerevisiae.test_messner2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_messner2023.py
"""Hermetic build tests for the Messner 2023 genome-wide KO proteome loader.

The loader reads two raw CSVs (a proteins x samples matrix keyed by UniProt accession
and a per-sample metadata table) and stores both as released (#1045): the protein keys
are the accessions and the deleted gene is the metadata ORF, upper-cased; no annotation
is consulted. The GFF helpers (``build_uniprot_to_orf_map``,
``build_orf_to_gene_name_map``) stay in the module as downstream utilities for the
Messner 2021 loader and are tested on a fake GFF under ``tmp_path``; no mirror, no
network.

Fixture (``yeast5k_noimpute_wide.csv``, blank cell = not measured):

    Protein.Group  wt_a  wt_b  10_9_hpr1_ko_YAL059W_ECM1_0.47  3_4_ko_YML009c_yml009c_0.9  qc_1  bad_ko
    P00001         10    12    8                                                           1     1
    P00002         4           (blank)                          5.0                        1     1
    P00410         2     2     3                                                           1     1

Metadata: ``wt_a``/``wt_b`` are ``HIS3`` (WT); the two ``ko`` samples delete YAL059W
and ``YML009c`` (lowercase in metadata, uppercased by the loader); ``qc_1`` is ``qc``
(ignored); ``bad_ko`` is a ``ko`` sample whose ORF ``YOR202W-not`` has no ORF form
(stored on the released-identifier leaf as ``YOR202W-NOT``, #1045); ``absent_ko`` is a
``ko`` row with
no matrix column (ignored).

WT reference (mean, sample SD / sqrt(n), n over non-blank WT cells), keyed by accession:
    P00001: 11.0, 1.0, 2      P00002: 4.0, NaN, 1      P00410: 2.0, 0.0, 2

2026.09.30 (Phase 14): a second build (``_numeric_root``) on one protein (P00001 ->
YBR001C) with two WT samples (1 and 1001) and two KO samples of the same ORF YBL007C
whose filenames are the real shapes ``10_9_hpr1_ko_YBL007C_SLA1_0.47`` and
``10_9_hpr57_ko_YBL007C_2824_0.49`` (the second is quoted in
[[datasets.showcase-verification.2026.09.29]], C19), with KO values 393221 and 0.0313
(the released matrix's max and min). Expected: two records (one per strain, not per
ORF), both named by the released ORF ``YBL007C`` (never the filename token, never a
GFF name); the values stored verbatim (linear, no log2);
the one shared reference is the arithmetic WT mean (1 + 1001) / 2 = 501.0 (a
geometric or log2 mean would give 31.64 or 4.98), SE = sqrt(((1 - 501)^2 + (1001 -
501)^2) / 1) / sqrt(2) = 500.0, n 2; the index is [[0, 1]]. Also pinned: the
summary log line on the first fixture (2 strains, 3 reference proteins, 1 deferred
ORF), a KO protein that no WT sample measured (refused by name since 2026.10.01), a GFF
line whose ninth column has no ORF token and a line with two accessions, the three
``download`` outcomes against a mirror under ``tmp_path`` (missing, off the pin, copied)
and the partial-raw case, and ``main``. The sha256 contract (issue #528, fixed):
``download`` hashes a mirror file before copying it, and ``process`` verifies both raw
files against their pins before reading a row, so a stale file left in ``raw/`` is
refused at build time.

Findings pinned: ``duration_hours`` is None although the 8 h culture is
sourceable (issue #486).

2026.10.01 (issue #528): a protein a KO measured but no WT sample did raises
``MissingWildTypeReferenceError`` naming the sample, its ORF and the protein (it used
to raise a bare ``KeyError`` from ``create_experiment``). The pinned matrix has no
such protein (0 of 1,850), so no stored record changes.

2026.10.02 (issue #485): ``perturbed_gene_name`` comes from the SGD GFF by systematic
name (``gene=`` standard name, else the uppercase ORF), never from the Filename token
after the ORF; ``_gene_from_filename`` is gone. Pinned on the real numeric filename
``10_9_hpr57_ko_YBL007C_2824_0.49`` (now SLA1) and the real lowercase one
``10_9_hpr10_ko_YBR174C_YBR174c_0.23`` (now YBR174C, no standard name in R64-4-1).
"""

import hashlib
import json
import logging
import math
import os
import shutil
import socket
from pathlib import Path

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.media import SM
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    ProteinAbundanceExperiment,
    ProteinAbundanceExperimentReference,
    ProteinAbundancePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import messner2023 as m
from torchcell.datasets.scerevisiae.messner2023 import (
    MATRIX_FILENAME,
    MEASUREMENT_TYPE,
    METADATA_FILENAME,
    MissingWildTypeReferenceError,
    ProteomeMessner2023Dataset,
    build_orf_to_gene_name_map,
    build_uniprot_to_orf_map,
)
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

KO_A = "10_9_hpr1_ko_YAL059W_ECM1_0.47"
KO_B = "3_4_ko_YML009c_yml009c_0.9"

MATRIX_HEADER = ["Protein.Group", "wt_a", "wt_b", KO_A, KO_B, "qc_1", "bad_ko"]
MATRIX_ROWS = [
    ["P00001", "10", "12", "8", "", "1", "1"],
    ["P00002", "4", "", "", "5.0", "1", "1"],
    ["P00410", "2", "2", "3", "", "1", "1"],
]
METADATA_ROWS = [
    ["Filename", "sampletype", "ORF", "plate"],
    ["wt_a", "HIS3", "YOR202W", "1"],
    ["wt_b", "HIS3", "YOR202W", "2"],
    [KO_A, "ko", "YAL059W", "10"],
    [KO_B, "ko", "YML009c", "3"],
    ["qc_1", "qc", "", "1"],
    ["bad_ko", "ko", "YOR202W-not", "4"],
    ["absent_ko", "ko", "YDR003W", "5"],
]

GFF_TEXT = (
    "##gff-version 3\n"
    "chrII\tSGD\tgene\t1\t100\t.\t-\t.\tID=YBR001C;Name=YBR001C;protein_id=UniProtKB:P00001\n"
    "chrIII\tSGD\tgene\t1\t100\t.\t+\t.\tID=YCR002W_mRNA;Parent=YCR002W;protein_id=UniProtKB:P00002\n"
    "chrmt\tSGD\tgene\t1\t100\t.\t+\t.\tID=Q0250;Name=COX2;protein_id=UniProtKB:P00410\n"
    "chrIV\tSGD\tgene\t1\t100\t.\t+\t.\tID=YDR003W;Name=YDR003W\n"
    "chrI\tSGD\tgene\t1\t100\t.\t+\t.\tID=YAL059W;Name=YAL059W;gene=ECM1;Alias=ECM1\n"
    "chrI\tSGD\tmRNA\t1\t100\t.\t+\t.\tID=YAL059W_mRNA;Name=YAL059W_mRNA;Parent=YAL059W\n"
    "chrII\tSGD\tgene\t1\t100\t.\t-\t.\tID=YBL007C;Name=YBL007C;gene=SLA1\n"
    "short\tline\tUniProtKB:P99999\n"
)

BY4741 = ReferenceGenome(
    species="Saccharomyces cerevisiae",
    strain="BY4741",
    assembly_set=SGD_S288C_R64,
    assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
)
ENVIRONMENT = Environment(media=SM, temperature=Temperature(value=30))
PUBLICATION = Publication(
    pubmed_id="37080200",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37080200/",
    doi="10.1016/j.cell.2023.03.026",
    doi_url="https://doi.org/10.1016/j.cell.2023.03.026",
)


def _sample_phenotype(
    abundance: dict[str, float], replicate_id: str
) -> ProteinAbundancePhenotype:
    """One released sample: verbatim values, n = 1, no SE map and its typed absence."""
    return ProteinAbundancePhenotype(
        protein_abundance=abundance,
        protein_abundance_se=None,
        n_replicates=dict.fromkeys(abundance, 1),
        measurement_type=MEASUREMENT_TYPE,
        replicate_id=replicate_id,
        provenance_gaps=[
            ProvenanceGap(
                field="protein_abundance_se",
                reason=ProvenanceGapReason.not_applicable_single_measurement,
                note=m._SINGLE_SAMPLE_SE_NOTE,
            )
        ],
    )


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.write_text("\n".join(",".join(r) for r in rows) + "\n")


def _write_gff(data_root: Path, text: str = GFF_TEXT) -> Path:
    gff_dir = data_root / "data" / "sgd" / "genome" / "S288C_R64-5-1"
    gff_dir.mkdir(parents=True)
    gff = gff_dir / "saccharomyces_cerevisiae_R64-5-1.gff"
    gff.write_text(text)
    return gff


def _root(
    tmp_path: Path,
    matrix_rows: list[list[str]] = MATRIX_ROWS,
    metadata_rows: list[list[str]] = METADATA_ROWS,
) -> Path:
    root = tmp_path / "proteome_messner2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(root / "raw" / MATRIX_FILENAME, [MATRIX_HEADER, *matrix_rows])
    _write_csv(root / "raw" / METADATA_FILENAME, metadata_rows)
    return root


@pytest.fixture
def dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProteomeMessner2023Dataset, Path]:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    root = _root(tmp_path)
    return ProteomeMessner2023Dataset(root=str(root)), root


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_build_uniprot_to_orf_map_reads_first_orf_token(tmp_path: Path) -> None:
    """Three accessions map; the ORF is the FIRST ORF-shaped token of column 9.

    ``ID=YCR002W_mRNA`` still yields ``YCR002W`` (the regex is unanchored), the
    mitochondrial ``Q0250`` matches the Q-plus-four-digits alternative, a line without
    ``UniProtKB:`` contributes nothing, and a line with fewer than nine tab columns is
    skipped even though it mentions an accession.
    """
    _write_gff(tmp_path)
    assert build_uniprot_to_orf_map(str(tmp_path)) == {
        "P00001": "YBR001C",
        "P00002": "YCR002W",
        "P00410": "Q0250",
    }


def test_build_uniprot_to_orf_map_first_sighting_wins(tmp_path: Path) -> None:
    """A second line carrying an already-seen accession does not overwrite (setdefault)."""
    text = (
        "chrII\tSGD\tgene\t1\t100\t.\t-\t.\tID=YBR001C;protein_id=UniProtKB:P00001\n"
        "chrII\tSGD\tCDS\t1\t100\t.\t-\t.\tID=YBR002C;protein_id=UniProtKB:P00001\n"
    )
    _write_gff(tmp_path, text)
    assert build_uniprot_to_orf_map(str(tmp_path)) == {"P00001": "YBR001C"}


def test_build_uniprot_to_orf_map_missing_gff_raises(tmp_path: Path) -> None:
    """No GFF under the glob -> FileNotFoundError naming the pattern searched."""
    with pytest.raises(FileNotFoundError, match="saccharomyces_cerevisiae_\\*.gff"):
        build_uniprot_to_orf_map(str(tmp_path))


def test_build_uniprot_to_orf_map_reads_data_root_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no argument the map is read from ``$DATA_ROOT``."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    assert build_uniprot_to_orf_map()["P00410"] == "Q0250"


def test_build_orf_to_gene_name_map_reads_gene_attribute_by_feature_id(
    tmp_path: Path,
) -> None:
    """Every feature whose ``ID`` is a nuclear ORF maps to its ``gene=`` (percent-decoded),
    else to the ORF; child lines (``YAL059W_mRNA``), mitochondrial ``Q0250`` (not a
    nuclear ORF), comments and short lines add nothing; the first feature per ORF wins.
    """
    text = (
        "##gff-version 3\n"
        "chrI\tSGD\tgene\t1\t9\t.\t+\t.\tID=YAL059W;Name=YAL059W;gene=ECM1\n"
        "chrI\tSGD\tmRNA\t1\t9\t.\t+\t.\tID=YAL059W_mRNA;Parent=YAL059W;gene=XXX1\n"
        "chrI\tSGD\tCDS\t1\t9\t.\t+\t.\tID=YAL059W;gene=LATER1\n"
        "chrII\tSGD\tgene\t1\t9\t.\t-\t.\tID=YBR174C;Name=YBR174C\n"
        "chrXVI\tSGD\tgene\t1\t9\t.\t-\t.\tID=YPL187W;gene=MF%28ALPHA%291\n"
        "chrmt\tSGD\tgene\t1\t9\t.\t+\t.\tID=Q0250;gene=COX2\n"
        "short\tline\tID=YCR001W;gene=NOPE1\n"
    )
    _write_gff(tmp_path, text)
    assert build_orf_to_gene_name_map(str(tmp_path)) == {
        "YAL059W": "ECM1",
        "YBR174C": "YBR174C",
        "YPL187W": "MF(ALPHA)1",
    }


def test_build_orf_to_gene_name_map_missing_gff_raises(tmp_path: Path) -> None:
    """No GFF under the glob -> FileNotFoundError naming the pattern searched."""
    with pytest.raises(FileNotFoundError, match="saccharomyces_cerevisiae_\\*.gff"):
        build_orf_to_gene_name_map(str(tmp_path))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def test_len_data_csv_and_side_files(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Five records, one per released sample: the three KOs (``bad_ko``, with no ORF
    form, on the released-identifier leaf), then the two HIS3 controls; ``qc_1`` (a
    pooled digest) and ``absent_ko`` (no column) are not records.

    ``preprocess/data.csv`` lists filename, sampletype, ORF (uppercased), gene (the ORF
    again: no name is looked up), and the count of measured proteins (2 for KO_A, 1 for
    KO_B, 3 for bad_ko, 3 for wt_a, 2 for wt_b). The gene set is the three released KO
    identifiers plus YOR202W, the controls' his3D::kanMX4; the manifest records this
    loader's module and class.
    """
    ds, root = dataset
    assert len(ds) == 5
    assert (root / "preprocess" / "data.csv").read_text() == (
        "filename,sampletype,orf,gene,n_proteins\n"
        f"{KO_A},ko,YAL059W,YAL059W,2\n"
        f"{KO_B},ko,YML009C,YML009C,1\n"
        "bad_ko,ko,YOR202W-NOT,YOR202W-NOT,3\n"
        "wt_a,HIS3,YOR202W,YOR202W,3\n"
        "wt_b,HIS3,YOR202W,YOR202W,2\n"
    )
    (released,) = ds[2]["experiment"]["genotype"]["perturbations"]
    assert (released["perturbation_type"], released["systematic_gene_name"]) == (
        "released_identifier",
        "YOR202W-NOT",
    )
    assert [ds[i]["experiment"]["phenotype"]["replicate_id"] for i in range(5)] == [
        KO_A,
        KO_B,
        "bad_ko",
        "wt_a",
        "wt_b",
    ]
    pre = root / "preprocess"
    assert json.loads((pre / "gene_set.json").read_text()) == [
        "YAL059W",
        "YML009C",
        "YOR202W",
        "YOR202W-NOT",
    ]
    manifest = json.loads((pre / "build_manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 3
    assert manifest["dataset_name"] == "proteome_messner2023"
    assert manifest["loader_class"] == "ProteomeMessner2023Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.messner2023"
    assert manifest["hostname"] == socket.gethostname()
    assert set(manifest["closure"]) >= {
        "ProteinAbundanceExperiment",
        "ProteinAbundanceExperimentReference",
        "ProteinAbundancePhenotype",
        "KanMxDeletionPerturbation",
        "Environment",
    }


def test_record_0_single_replicate_ko_with_restricted_wt_reference(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Record 0 = KO_A (YAL059W): abundances P00001 8.0 and P00410 3.0, n = 1, SE None
    with a ``not_applicable_single_measurement`` gap, ``replicate_id`` the Filename.

    The blank P00002 cell is dropped (no imputation), so the reference is the WT
    aggregate restricted to {P00001, P00410}: means 11.0 / 2.0, SEs 1.0 / 0.0 (SD of
    (10, 12) is sqrt(2), / sqrt(2) = 1.0; SD of (2, 2) is 0), n 2 / 2, and no
    ``replicate_id`` (it summarizes the controls). Environment is the fully specified
    ``SM`` medium at 30 C; background BY4741.
    """
    ds, _ = dataset
    record = ds[0]
    expected = ProteinAbundanceExperiment(
        dataset_name="ProteomeMessner2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YAL059W", perturbed_gene_name="YAL059W"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=_sample_phenotype({"P00001": 8.0, "P00410": 3.0}, KO_A),
    )
    expected_reference = ProteinAbundanceExperimentReference(
        dataset_name="ProteomeMessner2023Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance={"P00001": 11.0, "P00410": 2.0},
            protein_abundance_se={"P00001": 1.0, "P00410": 0.0},
            n_replicates={"P00001": 2, "P00410": 2},
            measurement_type=MEASUREMENT_TYPE,
        ),
    )
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["reference"]["phenotype_reference"]["replicate_id"] is None
    assert record["publication"] == PUBLICATION.model_dump()
    assert (
        ProteinAbundanceExperiment.model_validate(record["experiment"]).model_dump()
        == record["experiment"]
    )


def test_each_his3_control_culture_is_its_own_record(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Records 3 and 4 are the HIS3 controls ``wt_a`` and ``wt_b``, after the KOs.

    Each is the library's his3D::kanMX4 strain (metadata ORF YOR202W, the same KanMX
    leaf as a KO), its own values verbatim (wt_a 10 / 4 / 2; wt_b 12 / 2 with the blank
    P00002 dropped), n = 1, the SE gap, ``replicate_id`` its Filename. Its reference is
    the same restricted summary every KO gets, so wt_a's covers the three proteins with
    P00002 at n = 1 and a NaN SE.
    """
    ds, _ = dataset
    his3 = Genotype(
        perturbations=[
            KanMxDeletionPerturbation(
                systematic_gene_name="YOR202W", perturbed_gene_name="YOR202W"
            )
        ]
    )
    for index, values, name in (
        (3, {"P00001": 10.0, "P00002": 4.0, "P00410": 2.0}, "wt_a"),
        (4, {"P00001": 12.0, "P00410": 2.0}, "wt_b"),
    ):
        expected = ProteinAbundanceExperiment(
            dataset_name="ProteomeMessner2023Dataset",
            genotype=his3,
            environment=ENVIRONMENT,
            phenotype=_sample_phenotype(values, name),
        )
        assert ds[index]["experiment"] == expected.model_dump()
    reference = ds[3]["reference"]["phenotype_reference"]
    assert reference["protein_abundance"] == {
        "P00001": 11.0,
        "P00002": 4.0,
        "P00410": 2.0,
    }
    assert reference["n_replicates"] == {"P00001": 2, "P00002": 1, "P00410": 2}
    assert math.isnan(reference["protein_abundance_se"]["P00002"])
    assert ds[4]["reference"] == ds[0]["reference"]


def test_grouping_the_control_records_reproduces_the_reference_summary(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """The mean over the control records of each protein is the reference abundance and
    the count is its n: the summary is computed from exactly the records stored.
    """
    ds, _ = dataset
    controls = [ds[i]["experiment"]["phenotype"]["protein_abundance"] for i in (3, 4)]
    reference = ds[3]["reference"]["phenotype_reference"]
    for protein, level in reference["protein_abundance"].items():
        values = [c[protein] for c in controls if protein in c]
        assert sum(values) / len(values) == level
        assert len(values) == reference["n_replicates"][protein]


def test_record_1_lowercase_orf_uppercased_and_single_wt_measurement_nan_se(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Record 1 = KO_B: metadata ``YML009c`` becomes ``YML009C``, as both names.

    Only P00002 is measured (5.0). WT measured P00002 once (4.0), so the restricted
    reference carries n = 1 and a NaN SE for that protein.
    """
    ds, _ = dataset
    record = ds[1]
    perturbation = record["experiment"]["genotype"]["perturbations"][0]
    assert perturbation["systematic_gene_name"] == "YML009C"
    assert perturbation["perturbed_gene_name"] == "YML009C"
    phenotype = record["experiment"]["phenotype"]
    assert phenotype["protein_abundance"] == {"P00002": 5.0}
    assert phenotype["protein_abundance_se"] is None
    assert phenotype["n_replicates"] == {"P00002": 1}
    reference = record["reference"]["phenotype_reference"]
    assert reference["protein_abundance"] == {"P00002": 4.0}
    assert reference["n_replicates"] == {"P00002": 1}
    assert list(reference["protein_abundance_se"]) == ["P00002"]
    assert math.isnan(reference["protein_abundance_se"]["P00002"])


def test_reference_index_splits_on_restricted_reference(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """The restricted reference follows the measured protein set: KO_A and wt_b
    measured {P00001, P00410}, KO_B {P00002}, bad_ko and wt_a all three, so three
    references with members [0, 4], [1], [2, 3].
    """
    ds, root = dataset
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert [e["member_indices"] for e in index] == [[0, 4], [1], [2, 3]]
    assert [
        sorted(e["reference"]["phenotype_reference"]["protein_abundance"])
        for e in index
    ] == [["P00001", "P00410"], ["P00002"], ["P00001", "P00002", "P00410"]]
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == [[0, 4], [1], [2, 3]]


def test_missing_wt_columns_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata with no ``HIS3`` sample present in the matrix -> RuntimeError."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    metadata = [r for r in METADATA_ROWS if r[1] != "HIS3"]
    root = _root(tmp_path, metadata_rows=metadata)
    with pytest.raises(RuntimeError, match="missing the HIS3 \\(WT\\) control columns"):
        ProteomeMessner2023Dataset(root=str(root))


# --------------------------------------------------------------------------- #
# Phase 14: numeric gene tokens, linear values, refusals, download, main
# --------------------------------------------------------------------------- #

KO_SLA1 = "10_9_hpr1_ko_YBL007C_SLA1_0.47"
KO_NUM = "10_9_hpr57_ko_YBL007C_2824_0.49"


def _numeric_root(tmp_path: Path, wt: tuple[str, str] = ("1", "1001")) -> Path:
    """One protein, two WT samples, two KO strains of YBL007C (module docstring)."""
    root = tmp_path / "proteome_messner2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(
        root / "raw" / MATRIX_FILENAME,
        [
            ["Protein.Group", "wt_a", "wt_b", KO_SLA1, KO_NUM],
            ["P00001", wt[0], wt[1], "393221", "0.0313"],
        ],
    )
    _write_csv(
        root / "raw" / METADATA_FILENAME,
        [
            ["Filename", "sampletype", "ORF", "plate"],
            ["wt_a", "HIS3", "YOR202W", "1"],
            ["wt_b", "HIS3", "YOR202W", "2"],
            [KO_SLA1, "ko", "YBL007C", "10"],
            [KO_NUM, "ko", "YBL007C", "10"],
        ],
    )
    return root


@pytest.fixture
def numeric(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> ProteomeMessner2023Dataset:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    return ProteomeMessner2023Dataset(root=str(_numeric_root(tmp_path)))


def _shared_reference() -> dict[str, object]:
    return ProteinAbundanceExperimentReference(
        dataset_name="ProteomeMessner2023Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance={"P00001": 501.0},
            protein_abundance_se={"P00001": 500.0},
            n_replicates={"P00001": 2},
            measurement_type=MEASUREMENT_TYPE,
        ),
    ).model_dump()


def test_numeric_filename_token_is_not_the_gene_name_issue_485(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """``10_9_hpr57_ko_YBL007C_2824_0.49`` stores the deletion of the released ORF
    YBL007C as both names (never "2824", the filename token, and never a GFF name).
    The whole record is pinned, value 0.0313 stored verbatim.
    """
    expected = ProteinAbundanceExperiment(
        dataset_name="ProteomeMessner2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YBL007C", perturbed_gene_name="YBL007C"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=_sample_phenotype({"P00001": 0.0313}, KO_NUM),
    )
    assert numeric[1]["experiment"] == expected.model_dump()
    assert numeric[1]["reference"] == _shared_reference()
    first = numeric[0]["experiment"]["genotype"]["perturbations"][0]
    assert (first["systematic_gene_name"], first["perturbed_gene_name"]) == (
        "YBL007C",
        "YBL007C",
    )


KO_LOWER = "10_9_hpr10_ko_YBR174C_YBR174c_0.23"


def test_lowercase_orf_filename_token_is_not_the_gene_name_issue_485(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real filename ``10_9_hpr10_ko_YBR174C_YBR174c_0.23`` used to store
    ``perturbed_gene_name`` "YBR174c"; the name is the uppercase released ORF "YBR174C".
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(
        tmp_path,
        GFF_TEXT + "chrII\tSGD\tgene\t1\t100\t.\t-\t.\tID=YBR174C;Name=YBR174C\n",
    )
    root = tmp_path / "proteome_messner2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(
        root / "raw" / MATRIX_FILENAME,
        [["Protein.Group", "wt_a", KO_LOWER], ["P00001", "1", "2"]],
    )
    _write_csv(
        root / "raw" / METADATA_FILENAME,
        [
            ["Filename", "sampletype", "ORF", "plate"],
            ["wt_a", "HIS3", "YOR202W", "1"],
            [KO_LOWER, "ko", "YBR174C", "10"],
        ],
    )
    ds = ProteomeMessner2023Dataset(root=str(root))
    perturbation = ds[0]["experiment"]["genotype"]["perturbations"][0]
    assert (
        perturbation["systematic_gene_name"],
        perturbation["perturbed_gene_name"],
    ) == ("YBR174C", "YBR174C")


def test_values_are_linear_and_the_reference_is_the_arithmetic_wt_mean(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """The KO value 393221 is stored as 393221.0 (no log2); the WT reference over 1 and
    1001 is 501.0 with SE 500.0 (a log2 mean would be 4.98), shared by all four records,
    so one reference covers [0, 1, 2, 3]. The two strains of the ORF stay two records,
    told apart by ``replicate_id`` (their Filenames); the controls 1 and 1001 are
    records 2 and 3, stored linear.
    """
    assert len(numeric) == 4
    assert numeric[0]["experiment"]["phenotype"]["protein_abundance"] == {
        "P00001": 393221.0
    }
    assert numeric[0]["reference"] == _shared_reference()
    assert [
        numeric[i]["experiment"]["phenotype"]["replicate_id"] for i in range(4)
    ] == [KO_SLA1, KO_NUM, "wt_a", "wt_b"]
    assert numeric[0]["experiment"]["genotype"] == numeric[1]["experiment"]["genotype"]
    assert [
        numeric[i]["experiment"]["phenotype"]["protein_abundance"] for i in (2, 3)
    ] == [{"P00001": 1.0}, {"P00001": 1001.0}]
    index = numeric.experiment_reference_index
    assert index is not None
    assert [e.member_indices for e in index] == [[0, 1, 2, 3]]
    pre = Path(numeric.preprocess_dir)
    assert json.loads((pre / "gene_set.json").read_text()) == ["YBL007C", "YOR202W"]
    assert (pre / "data.csv").read_text() == (
        "filename,sampletype,orf,gene,n_proteins\n"
        f"{KO_SLA1},ko,YBL007C,YBL007C,1\n"
        f"{KO_NUM},ko,YBL007C,YBL007C,1\n"
        "wt_a,HIS3,YOR202W,YOR202W,1\n"
        "wt_b,HIS3,YOR202W,YOR202W,1\n"
    )


def test_culture_duration_is_not_recorded_issue_486(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """Finding (issue #486): the environment is SM at 30 C with ``duration_hours`` None,
    although the paper states the 8 h post-dilution culture. Pinned until #486 records it.
    """
    environment = numeric[0]["experiment"]["environment"]
    assert environment == ENVIRONMENT.model_dump()
    assert environment["duration_hours"] is None
    assert environment["temperature"]["value"] == 30.0
    assert environment["media"] == SM.model_dump()


def test_a_ko_protein_no_wt_sample_measured_is_refused_by_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #528): with both WT cells blank, YBR001C has no reference value,
    so the first KO sample that measured it raises ``MissingWildTypeReferenceError``
    naming the sample, its ORF and the protein, before any LMDB is written (it used to
    fail in ``create_experiment`` with a bare ``KeyError('YBR001C')``).
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    root = _numeric_root(tmp_path, wt=("", ""))
    with pytest.raises(MissingWildTypeReferenceError) as excinfo:
        ProteomeMessner2023Dataset(root=str(root))
    assert str(excinfo.value) == (
        f"Messner: KO sample {KO_SLA1} (YBL007C) measured 1 protein(s) no WT sample "
        "measured (e.g. ['P00001']); the reference has no value for them"
    )
    assert not (root / "processed" / "lmdb").exists()


def test_the_build_summary_is_logged_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """First fixture: 3 KO strains and 2 HIS3 controls, 3 WT proteins, 1 ORF on the
    released-identifier leaf (``bad_ko``; the ``absent_ko`` row has no matrix column and
    is not counted), 5 records written.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    root = _root(tmp_path)
    with caplog.at_level(logging.INFO, logger=m.log.name):
        ProteomeMessner2023Dataset(root=str(root))
    messages = [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert messages == [
        "Messner: 3 KO strains and 2 HIS3 control samples, WT reference with 3 "
        "proteins, 1 ORFs on the released-identifier leaf",
        "Wrote 5 Messner proteome sample records to LMDB",
    ]


def test_gff_line_without_an_orf_token_is_skipped_and_two_accessions_share_one(
    tmp_path: Path,
) -> None:
    """Line 139: a nine-column line with an accession but no ORF-shaped token adds
    nothing; a line with two accessions maps both to its first ORF token.
    """
    text = (
        "chrI\tSGD\tgene\t1\t9\t.\t+\t.\tID=tRNA-1;protein_id=UniProtKB:P77777\n"
        "chrI\tSGD\tgene\t1\t9\t.\t+\t.\tID=YAL001C;Parent=YAL002W;"
        "protein_id=UniProtKB:P11111,UniProtKB:P22222\n"
    )
    _write_gff(tmp_path, text)
    assert build_uniprot_to_orf_map(str(tmp_path)) == {
        "P11111": "YAL001C",
        "P22222": "YAL001C",
    }


def _bare(root: Path) -> ProteomeMessner2023Dataset:
    """An uninitialized instance whose ``raw_dir`` is ``root/raw``."""
    dataset = ProteomeMessner2023Dataset.__new__(ProteomeMessner2023Dataset)
    dataset.root = str(root)
    return dataset


def _mirror(data_root: Path) -> Path:
    mirror = data_root / "torchcell-library" / m._CITATION_KEY / "data"
    mirror.mkdir(parents=True)
    return mirror


def test_download_refuses_a_missing_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    dataset = _bare(tmp_path / "ds")
    src = tmp_path / "torchcell-library" / m._CITATION_KEY / "data" / MATRIX_FILENAME
    with pytest.raises(FileNotFoundError) as excinfo:
        dataset.download()
    assert str(excinfo.value) == (
        f"Messner mirror file missing: {src}. The mirror is canonical; "
        "restore it from backup (fetched once from Mendeley 10.17632/w8jtmnszd9.1)."
    )


def test_download_refuses_a_mirror_file_off_the_pin_and_copies_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    mirror = _mirror(tmp_path)
    (mirror / MATRIX_FILENAME).write_bytes(b"not the matrix")
    dataset = _bare(tmp_path / "ds")
    with pytest.raises(RawSha256MismatchError) as excinfo:
        dataset.download()
    got = hashlib.sha256(b"not the matrix").hexdigest()
    assert str(excinfo.value) == (
        f"sha256 mismatch for {mirror / MATRIX_FILENAME}: expected {m.MATRIX_SHA256}, "
        f"observed {got}"
    )
    assert list((tmp_path / "ds" / "raw").iterdir()) == []


def test_download_copies_both_files_on_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    mirror = _mirror(tmp_path)
    (mirror / MATRIX_FILENAME).write_bytes(b"matrix bytes")
    (mirror / METADATA_FILENAME).write_bytes(b"metadata bytes")
    monkeypatch.setattr(m, "MATRIX_SHA256", hashlib.sha256(b"matrix bytes").hexdigest())
    monkeypatch.setattr(
        m, "METADATA_SHA256", hashlib.sha256(b"metadata bytes").hexdigest()
    )
    dataset = _bare(tmp_path / "ds")
    dataset.download()
    raw = tmp_path / "ds" / "raw"
    assert (raw / MATRIX_FILENAME).read_bytes() == b"matrix bytes"
    assert (raw / METADATA_FILENAME).read_bytes() == b"metadata bytes"


def test_a_present_raw_file_off_the_pin_is_refused_at_build_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #528): ``download`` leaves a matrix already in ``raw/`` in place
    and fetches only the absent metadata (verified before the copy); the build then
    verifies both raw files in ``process`` and refuses the stale matrix with
    ``RawSha256MismatchError`` naming it and both digests, before any store exists.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    mirror = _mirror(tmp_path)
    (mirror / METADATA_FILENAME).write_bytes(b"metadata bytes")
    monkeypatch.setattr(
        m, "METADATA_SHA256", hashlib.sha256(b"metadata bytes").hexdigest()
    )
    raw = tmp_path / "ds" / "raw"
    raw.mkdir(parents=True)
    (raw / MATRIX_FILENAME).write_bytes(b"stale unverified matrix")
    _bare(tmp_path / "ds").download()
    assert (raw / MATRIX_FILENAME).read_bytes() == b"stale unverified matrix"
    assert (raw / METADATA_FILENAME).read_bytes() == b"metadata bytes"
    with pytest.raises(RawSha256MismatchError) as excinfo:
        ProteomeMessner2023Dataset(root=str(tmp_path / "ds"))
    stale = hashlib.sha256(b"stale unverified matrix").hexdigest()
    assert str(excinfo.value) == (
        f"sha256 mismatch for {raw / MATRIX_FILENAME}: expected {m.MATRIX_SHA256}, "
        f"observed {stale}"
    )
    assert list((tmp_path / "ds" / "processed").iterdir()) == []


def test_main_builds_from_data_root_and_prints_the_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds ``$DATA_ROOT/data/torchcell/proteome_messner2023`` and prints its
    length, then record 0.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    _write_gff(tmp_path)
    root = tmp_path / "data" / "torchcell" / "proteome_messner2023"
    (root / "raw").mkdir(parents=True)
    _write_csv(root / "raw" / MATRIX_FILENAME, [MATRIX_HEADER, *MATRIX_ROWS])
    _write_csv(root / "raw" / METADATA_FILENAME, METADATA_ROWS)
    m.main()
    # the build's own progress line comes first; main's two prints are the last two
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "proteome len = 5"
    assert lines[-1] == str(ProteomeMessner2023Dataset(root=str(root))[0])


def test_items_retype_through_the_abundance_classes_and_the_hooks_are_inert(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """``transform_item`` rebuilds a stored item as a ``ProteinAbundanceExperiment`` with
    its reference, dumping to exactly the stored dictionaries (``experiment_dataset.py``
    lines 638 to 641); the raw file list is the two mirror files, and ``preprocess_raw``
    is the documented identity.
    """
    item = numeric[0]
    typed = numeric.transform_item(item)
    assert type(typed["experiment"]) is ProteinAbundanceExperiment
    assert type(typed["reference"]) is ProteinAbundanceExperimentReference
    assert typed["experiment"].model_dump() == item["experiment"]
    assert typed["reference"].model_dump() == item["reference"]
    assert numeric.raw_file_names == [MATRIX_FILENAME, METADATA_FILENAME]
    frame = pd.DataFrame({"a": [1]})
    assert numeric.preprocess_raw(frame) is frame


# --------------------------------------------------------------------------- #
# 2026.11.24: one record per released sample, counted on the pinned release
# --------------------------------------------------------------------------- #

_MIRROR_DATA = Path(
    os.environ.get("DATA_ROOT", "/nonexistent"),
    "torchcell-library",
    m._CITATION_KEY,
    "data",
)


@pytest.mark.data
@pytest.mark.skipif(
    not (_MIRROR_DATA / METADATA_FILENAME).exists(),
    reason=f"requires the Messner 2023 mirror files under {_MIRROR_DATA}",
)
def test_the_pinned_release_holds_4699_ko_and_388_his3_samples_one_culture_each() -> (
    None
):
    """The pinned metadata (sha256 48864282...) lists 4,699 ``ko``, 388 ``HIS3`` and
    389 ``qc`` samples, every one a column of the pinned matrix header. The 5,087 KO and
    HIS3 samples occupy 5,087 distinct (plate, well) positions, so each is one culture;
    the ``qc`` pooled digests repeat their position (repeat injections). The HIS3
    controls sit on all 57 plates in the paper's wells (row-wise 1-based: 11, 20, 29,
    38, 71, 80, 89 = A11, B8, C5, D2, F11, G8, H5) plus 46 on one plate, and no ``ko``
    sample deletes YOR202W. KO ORFs, upper-cased: 4,549 distinct, multiplicity
    {1: 4403, 2: 142, 3: 4}.
    """
    for name, pin in (
        (METADATA_FILENAME, m.METADATA_SHA256),
        (MATRIX_FILENAME, m.MATRIX_SHA256),
    ):
        digest = hashlib.sha256((_MIRROR_DATA / name).read_bytes()).hexdigest()
        assert digest == pin
    meta = pd.read_csv(_MIRROR_DATA / METADATA_FILENAME)
    header = pd.read_csv(_MIRROR_DATA / MATRIX_FILENAME, nrows=0).columns[1:]
    assert meta["sampletype"].value_counts().to_dict() == {
        "ko": 4699,
        "qc": 389,
        "HIS3": 388,
    }
    assert set(meta["Filename"]) == set(header)
    assert meta["Filename"].is_unique
    cultures = meta[meta["sampletype"].isin(["ko", "HIS3"])]
    positions = list(
        zip(
            cultures["Plate (batch) nr"],
            cultures["Well nr (counted row-wise)"],
            strict=True,
        )
    )
    assert len(set(positions)) == len(positions) == 5087
    his3 = meta[meta["sampletype"] == "HIS3"]
    assert set(his3["ORF"]) == {"YOR202W"}
    assert his3["Plate (batch) nr"].nunique() == 57
    assert sorted(set(his3["Well nr (counted row-wise)"])) == [
        11,
        20,
        29,
        38,
        46,
        71,
        80,
        89,
    ]
    ko_orfs = meta.loc[meta["sampletype"] == "ko", "ORF"].str.upper()
    assert "YOR202W" not in set(ko_orfs)
    assert ko_orfs.nunique() == 4549
    assert ko_orfs.value_counts().value_counts().to_dict() == {1: 4403, 2: 142, 3: 4}


@pytest.mark.slow
@pytest.mark.data
@pytest.mark.skipif(
    not (_MIRROR_DATA / MATRIX_FILENAME).exists(),
    reason=f"requires the Messner 2023 mirror files under {_MIRROR_DATA}",
)
def test_the_full_build_stores_5087_sample_records(tmp_path: Path) -> None:
    """The pinned release builds 5,087 records: 4,699 KO strains then 388 HIS3
    cultures, each ``replicate_id`` a distinct released Filename, every control on the
    YOR202W KanMX leaf, and the mean over the 388 control records of each protein equal
    to the shared reference summary (to 1e-9 relative).
    """
    root = tmp_path / "proteome_messner2023"
    (root / "raw").mkdir(parents=True)
    for name in (MATRIX_FILENAME, METADATA_FILENAME):
        shutil.copyfile(_MIRROR_DATA / name, root / "raw" / name)
    built = ProteomeMessner2023Dataset(root=str(root))
    assert len(built) == 5087
    summary = pd.read_csv(root / "preprocess" / "data.csv")
    assert summary["sampletype"].tolist() == ["ko"] * 4699 + ["HIS3"] * 388
    assert summary["filename"].is_unique
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for i in range(4699, 5087):
        experiment = built[i]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        assert perturbation["systematic_gene_name"] == "YOR202W"
        assert perturbation["perturbation_type"] == "kanmx_deletion"
        assert experiment["phenotype"]["replicate_id"] == summary["filename"][i]
        for protein, value in experiment["phenotype"]["protein_abundance"].items():
            sums[protein] = sums.get(protein, 0.0) + value
            counts[protein] = counts.get(protein, 0) + 1
    reference = built[4699]["reference"]["phenotype_reference"]
    for protein, level in reference["protein_abundance"].items():
        assert sums[protein] / counts[protein] == pytest.approx(level, rel=1e-9)
        assert counts[protein] == reference["n_replicates"][protein]
