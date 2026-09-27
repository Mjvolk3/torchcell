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
"""

import json
import math
import socket
from pathlib import Path

import pytest

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
