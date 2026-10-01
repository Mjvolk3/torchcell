# tests/torchcell/datasets/scerevisiae/test_messner2023.py
# [[tests.torchcell.datasets.scerevisiae.test_messner2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_messner2023.py
"""Hermetic build tests for the Messner 2023 genome-wide KO proteome loader.

The loader reads two raw CSVs (a proteins x samples matrix keyed by UniProt accession
and a per-sample metadata table) and maps accessions to systematic ORFs through the
SGD GFF found under ``$DATA_ROOT/data/sgd/genome/*/saccharomyces_cerevisiae_*.gff``.
Every test points ``DATA_ROOT`` at ``tmp_path`` and writes a five-line fake GFF, three
of whose lines carry ``protein_id=UniProtKB:<acc>`` in the ninth column; no mirror, no
network.

Fixture (``yeast5k_noimpute_wide.csv``, blank cell = not measured):

    Protein.Group  wt_a  wt_b  10_9_hpr1_ko_YAL059W_ECM1_0.47  3_4_ko_YML009c_yml009c_0.9  qc_1  bad_ko
    P00001         10    12    8                                                           1     1
    P00002         4           (blank)                          5.0                        1     1
    P00410         2     2     3                                                           1     1

GFF: P00001 -> YBR001C, P00002 -> YCR002W, P00410 -> Q0250 (mitochondrial COX2).
Metadata: ``wt_a``/``wt_b`` are ``HIS3`` (WT); the two ``ko`` samples delete YAL059W
and ``YML009c`` (lowercase in metadata, uppercased by the loader); ``qc_1`` is ``qc``
(ignored); ``bad_ko`` is a ``ko`` sample whose ORF ``YOR202W-not`` fails the nuclear
regex (skipped); ``absent_ko`` is a ``ko`` row with no matrix column (ignored).

WT reference (mean, sample SD / sqrt(n), n over non-blank WT cells):
    YBR001C: 11.0, 1.0, 2      YCR002W: 4.0, NaN, 1      Q0250: 2.0, 0.0, 2

2026.09.30 (Phase 14): a second build (``_numeric_root``) on one protein (P00001 ->
YBR001C) with two WT samples (1 and 1001) and two KO samples of the same ORF YBL007C
whose filenames are the real shapes ``10_9_hpr1_ko_YBL007C_SLA1_0.47`` and
``10_9_hpr57_ko_YBL007C_2824_0.49`` (the second is quoted in
[[datasets.showcase-verification.2026.09.29]], C19), with KO values 393221 and 0.0313
(the released matrix's max and min). Expected: two records (one per strain, not per
ORF), gene names ``SLA1`` and ``2824``; the values stored verbatim (linear, no log2);
the one shared reference is the arithmetic WT mean (1 + 1001) / 2 = 501.0 (a
geometric or log2 mean would give 31.64 or 4.98), SE = sqrt(((1 - 501)^2 + (1001 -
501)^2) / 1) / sqrt(2) = 500.0, n 2; the index is [[0, 1]]. Also pinned: the
drop summary log line on the first fixture (2 strains, 3 reference proteins, 1 skipped
ORF), a KO protein that no WT sample measured (a bare ``KeyError``, a Finding), a GFF
line whose ninth column has no ORF token and a line with two accessions, the three
``download`` outcomes against a mirror under ``tmp_path`` (missing, off the pin, copied)
and the partial-raw case, and ``main``. The sha256 contract (issue #528, fixed):
``download`` hashes a mirror file before copying it, and ``process`` verifies both raw
files against their pins before reading a row, so a stale file left in ``raw/`` is
refused at build time.

Findings pinned: a numeric filename token is stored as ``perturbed_gene_name`` (issue
#485, 156 served records); ``duration_hours`` is None although the 8 h culture is
sourceable (issue #486); a protein a KO measured but no WT sample did raises
``KeyError`` from ``create_experiment`` (line 360) rather than a named refusal.
"""

import hashlib
import json
import logging
import math
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
    ProteomeMessner2023Dataset,
    _gene_from_filename,
    build_uniprot_to_orf_map,
)

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
    "short\tline\tUniProtKB:P99999\n"
)

BY4741 = ReferenceGenome(species="Saccharomyces cerevisiae", strain="BY4741")
ENVIRONMENT = Environment(media=SM, temperature=Temperature(value=30))
PUBLICATION = Publication(
    pubmed_id="37080200",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/37080200/",
    doi="10.1016/j.cell.2023.03.026",
    doi_url="https://doi.org/10.1016/j.cell.2023.03.026",
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


@pytest.mark.parametrize(
    ("filename", "orf", "expected"),
    [
        ("10_9_hpr1_ko_YAL059W_ECM1_0.47", "YAL059W", "ECM1"),
        ("3_4_ko_YML009c_yml009c_0.9", "YML009C", "yml009c"),
        ("3_4_ko_YAL059W", "YAL059W", "YAL059W"),
        ("3_4_ko_YAL059W_", "YAL059W", "YAL059W"),
        ("3_4_ko_something_else", "YAL059W", "YAL059W"),
        ("3_4_ko_YAL059W_0.47", "YAL059W", "0.47"),
    ],
)
def test_gene_from_filename(filename: str, orf: str, expected: str) -> None:
    """The token after the (case-insensitive) ORF token is the gene; else the ORF.

    Rows 1 and 2 are the two documented shapes (standard name; lowercased ORF when no
    standard name exists). Rows 3 to 5 pin the fallback: ORF last, ORF followed by an
    empty token, and no ORF token at all each return the ORF. Finding (row 6): when the
    ORF is followed directly by the numeric suffix, that suffix (``"0.47"``) is returned
    as the gene name; the parser does not validate the token it returns.
    """
    assert _gene_from_filename(filename, orf) == expected


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def test_len_data_csv_and_side_files(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Two KO records: ``bad_ko`` (non-systematic ORF) and ``absent_ko`` (no column) drop.

    ``preprocess/data.csv`` lists filename, ORF (uppercased), gene, and the count of
    measured proteins (2 for KO_A, 1 for KO_B). The gene set is the two deletion ORFs;
    the manifest records this loader's module and class.
    """
    ds, root = dataset
    assert len(ds) == 2
    assert (root / "preprocess" / "data.csv").read_text() == (
        "filename,orf,gene,n_proteins\n"
        f"{KO_A},YAL059W,ECM1,2\n"
        f"{KO_B},YML009C,yml009c,1\n"
    )
    pre = root / "preprocess"
    assert json.loads((pre / "gene_set.json").read_text()) == ["YAL059W", "YML009C"]
    manifest = json.loads((pre / "build_manifest.json").read_text())
    assert manifest["manifest_schema_version"] == 1
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
    """Record 0 = KO_A (YAL059W/ECM1): abundances YBR001C 8.0 and Q0250 3.0, n = 1, SE None.

    The blank P00002 cell is dropped (no imputation), so the reference is the WT
    aggregate restricted to {YBR001C, Q0250}: means 11.0 / 2.0, SEs 1.0 / 0.0 (SD of
    (10, 12) is sqrt(2), / sqrt(2) = 1.0; SD of (2, 2) is 0), n 2 / 2. Environment is the
    fully specified ``SM`` medium at 30 C; background BY4741.
    """
    ds, _ = dataset
    record = ds[0]
    expected = ProteinAbundanceExperiment(
        dataset_name="ProteomeMessner2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YAL059W", perturbed_gene_name="ECM1"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={"YBR001C": 8.0, "Q0250": 3.0},
            protein_abundance_se=None,
            n_replicates={"YBR001C": 1, "Q0250": 1},
            measurement_type=MEASUREMENT_TYPE,
        ),
    )
    expected_reference = ProteinAbundanceExperimentReference(
        dataset_name="ProteomeMessner2023Dataset",
        genome_reference=BY4741,
        environment_reference=ENVIRONMENT,
        phenotype_reference=ProteinAbundancePhenotype(
            protein_abundance={"YBR001C": 11.0, "Q0250": 2.0},
            protein_abundance_se={"YBR001C": 1.0, "Q0250": 0.0},
            n_replicates={"YBR001C": 2, "Q0250": 2},
            measurement_type=MEASUREMENT_TYPE,
        ),
    )
    assert record["experiment"] == expected.model_dump()
    assert record["reference"] == expected_reference.model_dump()
    assert record["publication"] == PUBLICATION.model_dump()
    assert (
        ProteinAbundanceExperiment.model_validate(record["experiment"]).model_dump()
        == record["experiment"]
    )


def test_record_1_lowercase_orf_uppercased_and_single_wt_measurement_nan_se(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """Record 1 = KO_B: metadata ``YML009c`` becomes ``YML009C``; gene ``yml009c``.

    Only P00002 -> YCR002W is measured (5.0). WT measured YCR002W once (4.0), so the
    restricted reference carries n = 1 and a NaN SE for that protein.
    """
    ds, _ = dataset
    record = ds[1]
    perturbation = record["experiment"]["genotype"]["perturbations"][0]
    assert perturbation["systematic_gene_name"] == "YML009C"
    assert perturbation["perturbed_gene_name"] == "yml009c"
    phenotype = record["experiment"]["phenotype"]
    assert phenotype["protein_abundance"] == {"YCR002W": 5.0}
    assert phenotype["protein_abundance_se"] is None
    assert phenotype["n_replicates"] == {"YCR002W": 1}
    reference = record["reference"]["phenotype_reference"]
    assert reference["protein_abundance"] == {"YCR002W": 4.0}
    assert reference["n_replicates"] == {"YCR002W": 1}
    assert list(reference["protein_abundance_se"]) == ["YCR002W"]
    assert math.isnan(reference["protein_abundance_se"]["YCR002W"])


def test_reference_index_splits_on_restricted_reference(
    dataset: tuple[ProteomeMessner2023Dataset, Path],
) -> None:
    """The two KOs measured different protein sets -> two references, one member each."""
    ds, root = dataset
    index = json.loads(
        (root / "preprocess" / "experiment_reference_index.json").read_text()
    )
    assert [e["member_indices"] for e in index] == [[0], [1]]
    assert [
        sorted(e["reference"]["phenotype_reference"]["protein_abundance"])
        for e in index
    ] == [["Q0250", "YBR001C"], ["YCR002W"]]
    loaded = ds.experiment_reference_index
    assert loaded is not None
    assert [eri.member_indices for eri in loaded] == [[0], [1]]


def test_unmapped_uniprot_accession_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matrix row whose accession is absent from the GFF aborts the build.

    ``P99999`` appears only on the short (7-column) GFF line, which the parser skips,
    so it is unmapped; the message counts it and lists it.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    rows = [*MATRIX_ROWS, ["P99999", "1", "1", "1", "1", "1", "1"]]
    root = _root(tmp_path, matrix_rows=rows)
    with pytest.raises(
        RuntimeError,
        match=r"1 Messner proteins have no UniProt->ORF mapping \(e.g. \['P99999'\]\)",
    ):
        ProteomeMessner2023Dataset(root=str(root))


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


def test_missing_gff_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the SGD GFF under ``$DATA_ROOT`` the build fails with FileNotFoundError.

    Only the exception type and message are asserted. That ``process()`` reads the matrix
    (source line 217), then the GFF map (222), then the metadata (231) is read from the
    source, not observed by this test.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    root = _root(tmp_path)
    with pytest.raises(FileNotFoundError, match="SGD GFF not found under"):
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
            protein_abundance={"YBR001C": 501.0},
            protein_abundance_se={"YBR001C": 500.0},
            n_replicates={"YBR001C": 2},
            measurement_type=MEASUREMENT_TYPE,
        ),
    ).model_dump()


def test_numeric_filename_token_is_stored_as_the_gene_name_issue_485(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """Finding (issue #485): ``10_9_hpr57_ko_YBL007C_2824_0.49`` stores SLA1's deletion
    with ``perturbed_gene_name`` "2824", beside a second strain of the same ORF named
    "SLA1"; one ORF gets two spellings. The whole record is pinned, value 0.0313 stored
    verbatim. Pinned until the gene name comes from the SGD GFF, not the filename.
    """
    expected = ProteinAbundanceExperiment(
        dataset_name="ProteomeMessner2023Dataset",
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name="YBL007C", perturbed_gene_name="2824"
                )
            ]
        ),
        environment=ENVIRONMENT,
        phenotype=ProteinAbundancePhenotype(
            protein_abundance={"YBR001C": 0.0313},
            protein_abundance_se=None,
            n_replicates={"YBR001C": 1},
            measurement_type=MEASUREMENT_TYPE,
        ),
    )
    assert numeric[1]["experiment"] == expected.model_dump()
    assert numeric[1]["reference"] == _shared_reference()
    first = numeric[0]["experiment"]["genotype"]["perturbations"][0]
    assert (first["systematic_gene_name"], first["perturbed_gene_name"]) == (
        "YBL007C",
        "SLA1",
    )


def test_values_are_linear_and_the_reference_is_the_arithmetic_wt_mean(
    numeric: ProteomeMessner2023Dataset,
) -> None:
    """The KO value 393221 is stored as 393221.0 (no log2); the WT reference over 1 and
    1001 is 501.0 with SE 500.0 (a log2 mean would be 4.98), shared by both strains of
    the ORF, so one reference covers [0, 1] and the two strains stay two records.
    """
    assert len(numeric) == 2
    assert numeric[0]["experiment"]["phenotype"]["protein_abundance"] == {
        "YBR001C": 393221.0
    }
    assert numeric[0]["reference"] == _shared_reference()
    index = numeric.experiment_reference_index
    assert index is not None
    assert [e.member_indices for e in index] == [[0, 1]]
    pre = Path(numeric.preprocess_dir)
    assert json.loads((pre / "gene_set.json").read_text()) == ["YBL007C"]
    assert (pre / "data.csv").read_text() == (
        "filename,orf,gene,n_proteins\n"
        f"{KO_SLA1},YBL007C,SLA1,1\n"
        f"{KO_NUM},YBL007C,2824,1\n"
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


def test_a_ko_protein_no_wt_sample_measured_raises_a_bare_key_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding: with both WT cells blank, YBR001C is dropped from the reference (``keep``
    needs n >= 1, line 245) and the first KO that measured it fails the lookup at line
    360 with ``KeyError('YBR001C')``, not a message naming the strain. Pinned until the
    build refuses such a protein with a named error.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    root = _numeric_root(tmp_path, wt=("", ""))
    with pytest.raises(KeyError) as excinfo:
        ProteomeMessner2023Dataset(root=str(root))
    assert excinfo.value.args == ("YBR001C",)


def test_the_build_summary_is_logged_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """First fixture: 2 KO strains, 3 WT proteins, 1 skipped ORF (``bad_ko``; the
    ``absent_ko`` row has no matrix column and is not counted).
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    _write_gff(tmp_path)
    root = _root(tmp_path)
    with caplog.at_level(logging.INFO, logger=m.log.name):
        ProteomeMessner2023Dataset(root=str(root))
    messages = [r.getMessage() for r in caplog.records if r.name == m.log.name]
    assert messages == [
        "Messner: 2 KO strains, WT reference with 3 proteins, 1 non-systematic "
        "KO ORF skipped",
        "Wrote 2 Messner proteome experiments to LMDB",
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
    assert lines[-2] == "proteome len = 2"
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
