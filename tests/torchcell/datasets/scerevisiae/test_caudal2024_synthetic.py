# tests/torchcell/datasets/scerevisiae/test_caudal2024_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_caudal2024_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_caudal2024_synthetic.py
"""Caudal 2024 pan-transcriptome loader built end to end on synthetic raw files.

The four ``raw_file_names`` are written under ``<root>/raw/`` so PyG never calls
``download()``. ``process()`` also reads the SGD S288C chromosome FASTA through the
genomes registry (``resolve(SGD_S288C_R64, SGD_FSA_NAME)``). ``DATA_ROOT`` is pointed
into ``tmp_path`` and a real ``GenomeManifest`` is written for that tier rather than
monkeypatching ``resolve``: the registry is cheap to satisfy, and doing so keeps its
sha256 verification on the path the build takes.

SGD FASTA: chromosome I ``AAAACCCCGGGGTTTT`` (two lines), II ``ACCCTTTGGA``, the
mitochondrion ``TTAACCGG``, and a plasmid entry with neither tag (ignored).

Peter gene tarball (member order): a directory (skipped), ``YAL001C.fasta`` whose
reference slice is chromosome1:1-8 + = ``AAAACCCC`` (AAA equal, SACE_YAU ``AAAACCCT``
differs, ZZZ is not a Caudal strain and is skipped), an empty ``EMPTY.fasta`` (no
records), ``YBR001W.fasta`` on chromosome2:2-5 - (slice ``CCCT`` reverse-complemented
to ``AGGG``; AAA's lowercase ``aggg`` is equal, SACE_YAU's ``AGGA`` differs, and its
token ``SACE_YAU_YBR001W_`` carries an empty symbol) and ``Q0010.fasta`` on
chromosome17:3-6 + (mitochondrial slice ``AACC``; AAA ``AACG`` differs).

Presence matrix columns (pangenome id -> S288C name): X1.YAL001C -> YAL001C,
X2.EC1118_1F14_0012g -> none, X3.YAL005C_NumOfGenes_3 -> YAL005C, X4.YBR001W -> YBR001W,
X5.contig_7 -> none, X6.YCR001W -> YCR001W. AAA = 1 1 0 1 0 1, SACE_YAU = 1 0 1 1 1 0,
ZZZ = 1 0 1 0 0 1, and XTRA_ABC = FY4-6 = 1 0 0 0 0 0, so X1 is the only core ORF
(present in all five rows). The copy-number file lists the same columns in reverse order (the loader reindexes);
AAA's accessory X2 has copy number 2.5, SACE_YAU's accessory X5 is ``NA`` (so 1.0).

Caudal expression rows (Strain, systematic_name, count, tpm; every row named and
``pan_absence`` ``present``, so the blank-name ledger of issue #598 is empty): AAA YAL001C 10/1.5 and
5/0.5 (summed to 15/2.0), AAA YBR001W 20/4.0, SACE_YAU YAL001C 30/3.0, plus XTRA_ABC,
FY4-6 and QQQ, which are all excluded. XTRA_ABC and FY4-6 ARE in the Peter matrix, so
what removes them is the loader's explicit exclusions (caudal2024.py lines 457-458: the
``^XTRA_`` pattern and ``!= "FY4-6"``); QQQ is not in Peter and drops at the
intersection. Reference = per-gene mean over the
two built isolates: YAL001C tpm (2.0 + 3.0)/2 = 2.5, count round(22.5) = 22 (banker's
rounding); YBR001W 4.0 / 20.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.schema import (
    Environment,
    GenePerturbationType,
    Genotype,
    Media,
    NaturalGeneAbsencePerturbation,
    NaturalGenePresencePerturbation,
    Publication,
    ReferenceGenome,
    RNASeqExpressionExperiment,
    RNASeqExpressionExperimentReference,
    RNASeqExpressionPhenotype,
    SequenceVariantPerturbation,
    Temperature,
)
from torchcell.datasets.scerevisiae import caudal2024 as m
from torchcell.literature.manifest import ArtifactRecord
from torchcell.sequence.genome.registry import (
    PETER2018_1011,
    SGD_S288C_R64,
    GenomeManifest,
)

_DATASET = "CaudalPanTranscriptome2024Dataset"

_SGD_FSA = (
    ">ref|NC_001133| [org=Saccharomyces cerevisiae] [chromosome=I]\n"
    "AAAACCCC\nGGGGTTTT\n"
    ">ref|NC_001134| [org=Saccharomyces cerevisiae] [chromosome=II]\n"
    "ACCCTTTGGA\n"
    ">ref|NC_001224| [org=Saccharomyces cerevisiae] [location=mitochondrion]\n"
    "TTAACCGG\n"
    ">ref|NC_001398| [org=Saccharomyces cerevisiae] [location=plasmid]\n"
    "GGGG\n"
)

_GENE_FILES: list[tuple[str, str]] = [
    (
        "YAL001C.fasta",
        ">AAA_YAL001C_TFC3 chromosome1:1-8 +\nAAAA\nCCCC\n"
        ">SACE_YAU_YAL001C_TFC3 chromosome1:1-8 +\nAAAACCCT\n"
        ">ZZZ_YAL001C_TFC3 chromosome1:1-8 +\nGGGGGGGG\n",
    ),
    ("EMPTY.fasta", ""),
    (
        "YBR001W.fasta",
        ">AAA_YBR001W_ chromosome2:2-5 -\naggg\n"
        ">SACE_YAU_YBR001W_ chromosome2:2-5 -\nAGGA\n",
    ),
    (
        "Q0010.fasta",
        ">AAA_Q0010_COX1 chromosome17:3-6 +\nAACG\n"
        ">SACE_YAU_Q0010_COX1 chromosome17:3-6 +\nAACC\n",
    ),
]

_COLUMNS = [
    "X1.YAL001C",
    "X2.EC1118_1F14_0012g",
    "X3.YAL005C_NumOfGenes_3",
    "X4.YBR001W",
    "X5.contig_7",
    "X6.YCR001W",
]
_PRESENCE = {
    "AAA": ["1", "1", "0", "1", "0", "1"],
    "SACE_YAU": ["1", "0", "1", "1", "1", "0"],
    "ZZZ": ["1", "0", "1", "0", "0", "1"],
    "XTRA_ABC": ["1", "0", "0", "0", "0", "0"],
    "FY4-6": ["1", "0", "0", "0", "0", "0"],
}
_COPYNUMBER = {
    "AAA": ["1", "2.5", "0", "1", "0", "1"],
    "SACE_YAU": ["1", "0", "1", "1", "NA", "0"],
    "ZZZ": ["1", "0", "1", "0", "0", "1"],
    "XTRA_ABC": ["1", "0", "0", "0", "0", "0"],
    "FY4-6": ["1", "0", "0", "0", "0", "0"],
}

_CAUDAL_CSV = (
    "Strain,systematic_name,ORF,Ortholog_in_SGD_2010,pan_absence,gene,count,tpm\n"
    "AAA,YAL001C,YAL001C,,present,TFC3,10,1.5\n"
    "AAA,YAL001C,YAL001C,,present,TFC3,5,0.5\n"
    "AAA,YBR001W,YBR001W,,present,,20,4.0\n"
    "SACE_YAU,YAL001C,YAL001C,,present,TFC3,30,3.0\n"
    "XTRA_ABC,YAL001C,YAL001C,,present,TFC3,99,9.0\n"
    "FY4-6,YAL001C,YAL001C,,present,TFC3,99,9.0\n"
    "QQQ,YAL001C,YAL001C,,present,TFC3,99,9.0\n"
)


def _matrix_gz(rows: dict[str, list[str]], columns: list[str]) -> bytes:
    lines = ["\t".join(["strain", *columns])]
    lines += ["\t".join([name, *values]) for name, values in rows.items()]
    return gzip.compress(("\n".join(lines) + "\n").encode())


def _reversed(rows: dict[str, list[str]]) -> dict[str, list[str]]:
    return {name: values[::-1] for name, values in rows.items()}


def _refgene_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        directory = tarfile.TarInfo("subdir")
        directory.type = tarfile.DIRTYPE
        tar.addfile(directory)
        for name, text in _GENE_FILES:
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _caudal_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("README.txt", "not the table")
        archive.writestr("final_data_annotated_merged_04052022.tab", _CAUDAL_CSV)
    return buffer.getvalue()


def _raw_bytes() -> dict[str, bytes]:
    return {
        m.CAUDAL_ZIP_BASENAME: _caudal_zip(),
        m.REFGENE_TAR_NAME: _refgene_tar(),
        m.PRESENCE_NAME: _matrix_gz(_PRESENCE, _COLUMNS),
        m.COPYNUMBER_NAME: _matrix_gz(_reversed(_COPYNUMBER), _COLUMNS[::-1]),
    }


def _write_tier(data_root: Path, assembly_set: str, files: dict[str, bytes]) -> None:
    tier = data_root / "torchcell-genomes" / assembly_set
    tier.mkdir(parents=True)
    records = []
    for name, data in files.items():
        (tier / name).write_bytes(data)
        records.append(
            ArtifactRecord(
                path=name,
                role="sequence",
                bytes=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
    manifest = GenomeManifest(
        assembly_set=assembly_set,
        organism="Saccharomyces cerevisiae",
        strain_or_population="synthetic",
        source="test",
        release="test",
        files=records,
        provenance_complete=True,
        created_at="2026-09-27T00:00:00+00:00",
    )
    (tier / "manifest.json").write_text(manifest.model_dump_json(indent=2))


def _write_sgd_tier(data_root: Path) -> None:
    _write_tier(data_root, SGD_S288C_R64, {m.SGD_FSA_NAME: _SGD_FSA.encode()})


def _write_raw(raw: Path) -> None:
    raw.mkdir(parents=True)
    for name, data in _raw_bytes().items():
        (raw / name).write_bytes(data)


def _write_peter_tier(data_root: Path) -> None:
    """The Peter tier listing the synthetic tarball and matrices (their build-time pins)."""
    raw = _raw_bytes()
    _write_tier(
        data_root,
        PETER2018_1011,
        {
            name: raw[name]
            for name in (m.REFGENE_TAR_NAME, m.PRESENCE_NAME, m.COPYNUMBER_NAME)
        },
    )


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    _write_peter_tier(data_root)
    root = tmp_path / "caudal"
    _write_raw(root / "raw")
    return root


@pytest.fixture
def dataset(root: Path) -> m.CaudalPanTranscriptome2024Dataset:
    return m.CaudalPanTranscriptome2024Dataset(root=str(root))


_ENVIRONMENT = Environment(
    media=Media(name="SC", state="liquid", is_synthetic=True),
    temperature=Temperature(value=30),
)
_REFERENCE_PHENOTYPE = RNASeqExpressionPhenotype(
    expression_tpm={"YAL001C": 2.5, "YBR001W": 4.0},
    expression_count={"YAL001C": 22, "YBR001W": 20},
    measurement_type="rnaseq_tpm",
    n_mapped_reads=None,
)


def _variant(
    strain: str, sys_name: str, name: str, token: str
) -> SequenceVariantPerturbation:
    return SequenceVariantPerturbation(
        systematic_gene_name=sys_name,
        perturbed_gene_name=name,
        strain_id=strain,
        sequence_source="peterGenomeEvolution10112018",
        sequence_uri=f"{sys_name}.fasta#{token}",
        sequence_sha256=m.REFGENE_TAR_SHA256,
    )


def _present(
    strain: str, orf: str, copy_number: float
) -> NaturalGenePresencePerturbation:
    return NaturalGenePresencePerturbation(
        systematic_gene_name=orf,
        perturbed_gene_name=orf,
        copy_number=copy_number,
        strain_id=strain,
        pangenome_orf_id=orf,
        origin=None,
        sequence_source="peterGenomeEvolution10112018",
    )


def _absent(strain: str, name: str, orf: str) -> NaturalGeneAbsencePerturbation:
    return NaturalGeneAbsencePerturbation(
        systematic_gene_name=name,
        perturbed_gene_name=name,
        strain_id=strain,
        pangenome_orf_id=orf,
        sequence_source="peterGenomeEvolution10112018",
    )


def _experiment(
    perturbations: list[GenePerturbationType],
    tpm: dict[str, float],
    count: dict[str, int],
) -> RNASeqExpressionExperiment:
    return RNASeqExpressionExperiment(
        dataset_name=_DATASET,
        genotype=Genotype(perturbations=perturbations),
        environment=_ENVIRONMENT,
        phenotype=RNASeqExpressionPhenotype(
            expression_tpm=tpm,
            expression_count=count,
            measurement_type="rnaseq_tpm",
            n_mapped_reads=None,
        ),
    )


_REFERENCE = RNASeqExpressionExperimentReference(
    dataset_name=_DATASET,
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="S288C"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_REFERENCE_PHENOTYPE,
)

_EXPECTED = [
    _experiment(
        [
            _variant("AAA", "Q0010", "COX1", "AAA_Q0010_COX1"),
            _present("AAA", "2-EC1118_1F14_0012g", 2.5),
            _absent("AAA", "YAL005C", "3-YAL005C_NumOfGenes_3"),
        ],
        {"YAL001C": 2.0, "YBR001W": 4.0},
        {"YAL001C": 15, "YBR001W": 20},
    ),
    _experiment(
        [
            _variant("SACE_YAU", "YAL001C", "TFC3", "SACE_YAU_YAL001C_TFC3"),
            _variant("SACE_YAU", "YBR001W", "YBR001W", "SACE_YAU_YBR001W_"),
            _present("SACE_YAU", "5-contig_7", 1.0),
            _absent("SACE_YAU", "YCR001W", "6-YCR001W"),
        ],
        {"YAL001C": 3.0},
        {"YAL001C": 30},
    ),
]

_PUBLICATION = Publication(
    pubmed_id="38778243",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/38862621/",
    doi="10.1038/s41588-024-01769-9",
    doi_url="https://doi.org/10.1038/s41588-024-01769-9",
)


def test_two_matched_isolates_build_in_sorted_strain_order(
    dataset: m.CaudalPanTranscriptome2024Dataset,
) -> None:
    """Caudal strains AAA, SACE_YAU, XTRA_ABC, FY4-6, QQQ against Peter's AAA,
    SACE_YAU, ZZZ, XTRA_ABC, FY4-6: the explicit exclusions drop XTRA_ABC and FY4-6,
    so only AAA and SACE_YAU build (the ``SACE_`` prefix is kept). The loader
    passes sequence variants, then presences, then absences, and ``Genotype`` stores
    them sorted by systematic name.
    """
    assert len(dataset) == 2
    assert [
        [
            (p["systematic_gene_name"], p["perturbation_type"])
            for p in dataset[i]["experiment"]["genotype"]["perturbations"]
        ]
        for i in range(2)
    ] == [
        [
            ("2-EC1118_1F14_0012g", "natural_gene_presence"),
            ("Q0010", "sequence_variant"),
            ("YAL005C", "natural_gene_absence"),
        ],
        [
            ("5-contig_7", "natural_gene_presence"),
            ("YAL001C", "sequence_variant"),
            ("YBR001W", "sequence_variant"),
            ("YCR001W", "natural_gene_absence"),
        ],
    ]
    for i, expected in enumerate(_EXPECTED):
        assert dataset[i]["experiment"] == expected.model_dump()
        assert dataset[i]["reference"] == _REFERENCE.model_dump()


def test_publication_pubmed_id_and_url_name_different_articles(
    dataset: m.CaudalPanTranscriptome2024Dataset,
) -> None:
    """Finding: ``create_experiment`` (caudal2024.py lines 684-685) stores
    ``pubmed_id="38778243"`` beside ``pubmed_url`` ending in ``38862621``, two different
    PubMed ids for one publication record.
    """
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()
    publication = dataset[0]["publication"]
    assert publication["pubmed_url"].rstrip("/").rsplit("/", 1)[1] == "38862621"
    assert publication["pubmed_id"] == "38778243"


def test_side_files_gene_set_reference_index_and_strain_list(
    dataset: m.CaudalPanTranscriptome2024Dataset,
) -> None:
    """The gene set is every perturbation's ``systematic_gene_name`` (pangenome ids for
    accessory ORFs), sorted; one shared reference covers both records.
    """
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "2-EC1118_1F14_0012g",
        "5-contig_7",
        "Q0010",
        "YAL001C",
        "YAL005C",
        "YBR001W",
        "YCR001W",
    ]
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [[0, 1]]
    assert [entry["reference"] for entry in index] == [_REFERENCE.model_dump()]
    assert (preprocess / "data.csv").read_text() == "strain_id\nAAA\nSACE_YAU\n"
    variants = pd.read_parquet(preprocess / "sequence_variants.parquet")
    assert variants.to_dict(orient="list") == {
        "strain_id": ["SACE_YAU", "SACE_YAU", "AAA"],
        "systematic_gene_name": ["YAL001C", "YBR001W", "Q0010"],
        "symbol": ["TFC3", "", "COX1"],
        "header_token": [
            "SACE_YAU_YAL001C_TFC3",
            "SACE_YAU_YBR001W_",
            "AAA_Q0010_COX1",
        ],
    }
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "caudal"
    assert manifest["loader_class"] == _DATASET
    assert sorted(manifest["closure"]) == [
        "ComponentDefinition",
        "Compound",
        "Concentration",
        "ConcentrationUnit",
        "DoseBasis",
        "Environment",
        "Experiment",
        "ExperimentReference",
        "GenePerturbation",
        "Genotype",
        "Media",
        "MediaComponent",
        "MediaComponentRole",
        "ModelStrict",
        "NaturalGeneAbsencePerturbation",
        "NaturalGenePresencePerturbation",
        "Phenotype",
        "PresenceAbsencePerturbation",
        "ProvenanceGapMixin",
        "Publication",
        "RNASeqExpressionExperiment",
        "RNASeqExpressionExperimentReference",
        "RNASeqExpressionPhenotype",
        "ReferenceGenome",
        "SequencePerturbation",
        "SequenceVariantPerturbation",
        "Temperature",
        "TemperatureUnit",
    ]
    assert dataset.experiment_class is RNASeqExpressionExperiment
    assert dataset.reference_class is RNASeqExpressionExperimentReference
    assert dataset.raw_file_names == [
        "final_data_annotated_merged_04052022.tab.zip",
        "allReferenceGenesWithSNPsAndIndelsInferred.tar.gz",
        "genesMatrix_PresenceAbsence.tab.gz",
        "genesMatrix_CopyNumber.tab.gz",
    ]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame


def test_rebuild_reads_the_cached_variant_table_not_the_tarball(root: Path) -> None:
    """With ``processed/`` removed and the tarball replaced by bytes that are not a
    tarball, the rebuild still yields the same two records: ``_sequence_variants``
    loads ``preprocess/sequence_variants.parquet`` and never opens the tar.
    """
    first = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    first.close_lmdb()
    shutil.rmtree(root / "processed")
    (root / "raw" / m.REFGENE_TAR_NAME).write_bytes(b"not a tarball")
    rebuilt = m.CaudalPanTranscriptome2024Dataset(root=str(root))
    assert [rebuilt[i]["experiment"] for i in range(len(rebuilt))] == [
        e.model_dump() for e in _EXPECTED
    ]


def test_a_matched_isolate_missing_from_every_gene_header_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding Peter isolate BBB and a Caudal row for it makes BBB a matched strain that
    no gene FASTA names, so ``_assert_all_isolates_seen`` refuses the build.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    _write_peter_tier(data_root)
    raw = tmp_path / "caudal" / "raw"
    _write_raw(raw)
    presence = {**_PRESENCE, "BBB": ["1"] * 6}
    (raw / m.PRESENCE_NAME).write_bytes(_matrix_gz(presence, _COLUMNS))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("t.tab", _CAUDAL_CSV + "BBB,YAL001C,TFC3,1,1.0\n")
    (raw / m.CAUDAL_ZIP_BASENAME).write_bytes(buffer.getvalue())
    with pytest.raises(
        ValueError,
        match=(
            r"1 of 3 matched isolates never appeared in a Peter gene-FASTA header, "
            r"so their genotype would be silently empty: \['BBB'\]"
        ),
    ):
        m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))


def test_reference_slice_rejects_a_header_without_coordinates() -> None:
    with pytest.raises(
        ValueError, match=r"unparseable coordinate header: 'AAA_YAL001C_TFC3 nowhere'"
    ):
        m._reference_slice("AAA_YAL001C_TFC3 nowhere", {"I": "ACGT"})


def test_sgd_chromosomes_keys_roman_numerals_and_the_mitochondrion(
    tmp_path: Path,
) -> None:
    """Multi-line records are joined and upper-cased; the untagged plasmid is dropped."""
    path = tmp_path / "sgd.fsa"
    path.write_text(_SGD_FSA.replace("TTAACCGG", "ttaaccgg"))
    assert m._sgd_chromosomes(str(path)) == {
        "I": "AAAACCCCGGGGTTTT",
        "II": "ACCCTTTGGA",
        "MT": "TTAACCGG",
    }


def _write_mirror(data_root: Path) -> dict[str, bytes]:
    """Library-mirror Caudal zip plus the Peter tier with its three files."""
    raw = _raw_bytes()
    zip_path = data_root / m.CAUDAL_ZIP_REL
    zip_path.parent.mkdir(parents=True)
    zip_path.write_bytes(raw[m.CAUDAL_ZIP_BASENAME])
    _write_peter_tier(data_root)
    return raw


def test_download_refuses_a_mirror_zip_whose_sha256_is_not_the_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture zip is not the released bytes, so the pinned Caudal digest fails
    with ``RawSha256MismatchError`` naming the mirror zip and both digests, and nothing
    is linked for it.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    raw = _write_mirror(data_root)
    got = hashlib.sha256(raw[m.CAUDAL_ZIP_BASENAME]).hexdigest()
    with pytest.raises(RawSha256MismatchError) as err:
        m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))
    assert str(err.value) == (
        f"sha256 mismatch for {data_root / m.CAUDAL_ZIP_REL}: expected "
        f"{m.CAUDAL_ZIP_SHA256}, observed {got}"
    )
    assert not (tmp_path / "caudal" / "raw" / m.CAUDAL_ZIP_BASENAME).exists()


def test_download_refuses_a_missing_mirror_zip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    _write_mirror(data_root)
    (data_root / m.CAUDAL_ZIP_REL).unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"required raw artifact missing from mirror: {data_root / m.CAUDAL_ZIP_REL}"
        ),
    ):
        m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))


def test_download_symlinks_every_file_then_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the two pins repointed at the fixture's digests, each raw file becomes a
    symlink to its mirror path (the three Peter files resolved through the registry),
    and the build yields the same two records.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _write_sgd_tier(data_root)
    raw = _write_mirror(data_root)
    monkeypatch.setattr(
        m, "CAUDAL_ZIP_SHA256", hashlib.sha256(raw[m.CAUDAL_ZIP_BASENAME]).hexdigest()
    )
    monkeypatch.setattr(
        m, "REFGENE_TAR_SHA256", hashlib.sha256(raw[m.REFGENE_TAR_NAME]).hexdigest()
    )
    dataset = m.CaudalPanTranscriptome2024Dataset(root=str(tmp_path / "caudal"))
    tier = data_root / "torchcell-genomes" / PETER2018_1011
    links = {
        name: os.readlink(tmp_path / "caudal" / "raw" / name) for name in m.RAW_FILES
    }
    assert links == {
        m.CAUDAL_ZIP_BASENAME: str(data_root / m.CAUDAL_ZIP_REL),
        m.REFGENE_TAR_NAME: str(tier / m.REFGENE_TAR_NAME),
        m.PRESENCE_NAME: str(tier / m.PRESENCE_NAME),
        m.COPYNUMBER_NAME: str(tier / m.COPYNUMBER_NAME),
    }
    assert len(dataset) == 2
    pinned = hashlib.sha256(raw[m.REFGENE_TAR_NAME]).hexdigest()
    assert [
        p["sequence_sha256"]
        for p in dataset[1]["experiment"]["genotype"]["perturbations"]
        if p["perturbation_type"] == "sequence_variant"
    ] == [pinned, pinned]
