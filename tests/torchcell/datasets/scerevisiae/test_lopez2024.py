# tests/torchcell/datasets/scerevisiae/test_lopez2024.py
# [[tests.torchcell.datasets.scerevisiae.test_lopez2024]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_lopez2024.py
"""Montaño López 2024 loaders: Table S2 aggregation, Table S3 two-block read, resolver rules.

The supplementary workbook is written synthetically with openpyxl into ``<root>/raw/`` so
PyG never calls ``download()``; ``process()`` runs for real against a duck-typed genome stub
(``gene_attribute_table`` with ``ID``/``gene`` columns and ``alias_to_systematic``), which
is everything the two loaders read from a genome. The stub is the authority for every
name-to-ORF mapping asserted here.

Stub genome: current ORFs YAL001C (TFC3), YBR001C (NTH2), YCR001W (no standard name),
YDR001C (NTH1); old names YAL002W -> YAL001C, YBR002C -> YBR001C, YDR002W -> YDR001C,
YER001W -> YER999W (a target that is not a current ORF).
"""

from __future__ import annotations

import hashlib
import json
import math
import socket
import statistics
import subprocess
from pathlib import Path
from typing import cast

import openpyxl
import pandas as pd
import pytest

from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    Media,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import lopez2024 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


class _StubGenome:
    """The two genome attributes the loaders read: the ID/gene table and the alias map."""

    gene_attribute_table = pd.DataFrame(
        {
            "ID": ["YAL001C", "YBR001C", "YCR001W", "YDR001C"],
            "gene": ["TFC3", "NTH2", None, "NTH1"],
        }
    )
    alias_to_systematic: dict[str, list[str]] = {
        "YAL002W": ["YAL001C"],
        "YBR002C": ["YBR001C"],
        "YDR002W": ["YDR001C"],
        "YER001W": ["YER999W"],
    }


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


# Table S2 rows (Gene-knockout strain, Fold change). Two rows for YAL001C (one lowercase),
# one alias whose target is directly present (YBR002C -> YBR001C, must be DROPPED, not
# merged), one alias that resolves (YDR002W -> YDR001C), one alias to a non-current target
# (YER001W), one systematic name with no alias entry (YFR001W), one non-systematic token.
_SCREEN_ROWS: list[tuple[str, float]] = [
    ("yal001c", 1.5),
    ("YAL001C", 2.5),
    ("YBR001C", 0.5),
    ("YCR001W", 4.0),
    ("YBR002C", 0.8),
    ("YDR002W", 0.25),
    ("YER001W", 9.0),
    ("YFR001W", 1.1),
    ("ABC1", 1.2),
]

# Table S3 UP block (FC>=2) and DOWN block (FC<=0.5): (gene, FC average, STD). YBL071W-A is
# in both blocks and is dropped before resolution; the DOWN block is one row longer so the
# UP block's trailing gene cell is blank (exercises the dropna on the gene column).
_UP_ROWS: list[tuple[str, float, float]] = [
    ("YAL001C", 3.0, 0.3),
    ("YBL071W-A", 3.469, 0.1),
    ("YDR002W", 2.2, 0.6),
]
_DOWN_ROWS: list[tuple[str, float, float]] = [
    ("ybr001c", 0.25, 0.06),
    ("YBL071W-A", 0.0757, 0.01),
    ("YCR001W", 0.4, 0.09),
    ("YER001W", 0.3, 0.05),
]


def _write_workbook(
    raw: Path,
    screen_rows: list[tuple[str, float]],
    up_rows: list[tuple[str, float, float]],
    down_rows: list[tuple[str, float, float]],
) -> None:
    """Write ``supplementary_tables.xlsx`` with the loaders' header offsets.

    ``Table S2``: title row, then the header on 0-based row 1. ``Table S3``: title row,
    block-label row, then the header on 0-based row 2 with the UP block in columns A-C and
    the DOWN block in columns E-G (pandas suffixes the repeated names with ``.1``).
    """
    workbook = openpyxl.Workbook()
    s2 = workbook.active
    s2.title = "Table S2"
    s2.append(["Table S2. First genome-wide biosensor screen"])
    s2.append(["Gene-knockout strain", "Fold change"])
    for gene, fc in screen_rows:
        s2.append([gene, fc])
    s3 = workbook.create_sheet("Table S3")
    s3.append(["Table S3. Validated hits"])
    s3.append(["FC >= 2", None, None, None, "FC <= 0.5"])
    s3.append(
        [
            "Gene knockout-strain",
            "FC (average)",
            "STD",
            None,
            "Gene knockout-strain",
            "FC (average)",
            "STD",
        ]
    )
    for i in range(max(len(up_rows), len(down_rows))):
        up = list(up_rows[i]) if i < len(up_rows) else [None, None, None]
        down = list(down_rows[i]) if i < len(down_rows) else [None, None, None]
        s3.append([*up, None, *down])
    workbook.save(raw / m._XLSX_FILENAME)


def _root(
    tmp_path: Path,
    slug: str,
    screen_rows: list[tuple[str, float]] = _SCREEN_ROWS,
    up_rows: list[tuple[str, float, float]] = _UP_ROWS,
    down_rows: list[tuple[str, float, float]] = _DOWN_ROWS,
) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_workbook(root / "raw", screen_rows, up_rows, down_rows)
    return root


@pytest.fixture
def screen(tmp_path: Path) -> m.IsobutanolScreenLopez2024Dataset:
    root = _root(tmp_path, "isobutanol_screen_lopez2024")
    return m.IsobutanolScreenLopez2024Dataset(root=str(root), genome=_genome())


@pytest.fixture
def validated(tmp_path: Path) -> m.IsobutanolValidatedLopez2024Dataset:
    root = _root(tmp_path, "isobutanol_validated_lopez2024")
    return m.IsobutanolValidatedLopez2024Dataset(root=str(root), genome=_genome())


_ENVIRONMENT = Environment(
    media=Media(name="SC", state="liquid", is_synthetic=True),
    temperature=Temperature(value=30.0),
    aerobicity="aerobic",
)
_PUBLICATION = Publication(
    pubmed_id="35022416",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/35022416/",
    doi="10.1038/s41467-021-27852-x",
    doi_url="https://doi.org/10.1038/s41467-021-27852-x",
)


def _phenotype(fc: float, se: float | None, n: int) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level={"isobutanol": fc},
        metabolite_level_se=None if se is None else {"isobutanol": se},
        n_replicates={"isobutanol": n},
        measurement_type="biosensor_gfp_fluorescence_fold_change",
        target_metabolite_ids=None,
    )


def _experiment(
    dataset_name: str, orf: str, gene: str, fc: float, se: float | None, n: int
) -> dict[str, object]:
    return MetaboliteExperiment(
        dataset_name=dataset_name,
        genotype=Genotype(
            perturbations=[
                KanMxDeletionPerturbation(
                    systematic_gene_name=orf, perturbed_gene_name=gene
                )
            ]
        ),
        environment=_ENVIRONMENT,
        phenotype=_phenotype(fc, se, n),
    ).model_dump()


def _reference(dataset_name: str, n: int) -> dict[str, object]:
    return MetaboliteExperimentReference(
        dataset_name=dataset_name,
        genome_reference=ReferenceGenome(
            species="Saccharomyces cerevisiae", strain="BY4741"
        ),
        environment_reference=_ENVIRONMENT,
        phenotype_reference=_phenotype(1.0, None, n),
    ).model_dump()


def _records(
    dataset: m._IsobutanolLopez2024Base,
) -> list[tuple[str, str, float, float | None, int]]:
    """(orf, perturbed name, fold change, se, n) per stored record, in LMDB order."""
    out = []
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        (perturbation,) = experiment["genotype"]["perturbations"]
        phenotype = experiment["phenotype"]
        se = phenotype["metabolite_level_se"]
        out.append(
            (
                perturbation["systematic_gene_name"],
                perturbation["perturbed_gene_name"],
                phenotype["metabolite_level"]["isobutanol"],
                None if se is None else se["isobutanol"],
                phenotype["n_replicates"]["isobutanol"],
            )
        )
    return out


def test_screen_aggregates_rows_per_resolved_orf_and_drops_by_rule(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """Nine Table S2 rows become four records in first-sighting ORF order.

    YAL001C: rows 1.5 and 2.5 (one typed lowercase) -> mean 2.0, n = 2, SE = sample SD /
    sqrt(2) = sqrt(0.5) / sqrt(2) = 0.5. YBR001C 0.5, YCR001W 4.0 (unnamed, so the
    perturbed name is the ORF) and YDR002W -> YDR001C (NTH1) 0.25 are single rows with
    SE None. Dropped: YBR002C (its alias target YBR001C is directly present, so it is not
    merged into YBR001C, which keeps n = 1), YER001W (alias target is not a current ORF),
    YFR001W (systematic form, no alias entry), ABC1 (not systematic).
    """
    assert len(screen) == 4
    assert _records(screen) == [
        ("YAL001C", "TFC3", 2.0, statistics.stdev([1.5, 2.5]) / math.sqrt(2), 2),
        ("YBR001C", "NTH2", 0.5, None, 1),
        ("YCR001W", "YCR001W", 4.0, None, 1),
        ("YDR001C", "NTH1", 0.25, None, 1),
    ]
    assert statistics.stdev([1.5, 2.5]) / math.sqrt(2) == 0.5
    assert screen[0]["experiment"] == _experiment(
        "IsobutanolScreenLopez2024Dataset", "YAL001C", "TFC3", 2.0, 0.5, 2
    )
    assert screen[0]["publication"] == _PUBLICATION.model_dump()


def test_screen_reference_replicate_count_mirrors_each_record(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """Finding: the WT reference is rebuilt per record with ``n_replicates`` = that record's
    row count, so the screen carries one reference per distinct n rather than one shared
    WT reference. Here record 0 (n = 2) and records 1-3 (n = 1) give two entries in
    ``experiment_reference_index.json`` with member indices [0] and [1, 2, 3]; both have
    FC 1.0, SE None, BY4741, and the SC liquid 30 C aerobic environment.
    """
    assert screen[0]["reference"] == _reference("IsobutanolScreenLopez2024Dataset", 2)
    assert screen[3]["reference"] == _reference("IsobutanolScreenLopez2024Dataset", 1)
    index = json.loads(
        (
            Path(screen.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0], [1, 2, 3]]
    assert [
        entry["reference"]["phenotype_reference"]["n_replicates"] for entry in index
    ] == [{"isobutanol": 2}, {"isobutanol": 1}]
    assert index[0]["reference"]["genome_reference"] == {
        "species": "Saccharomyces cerevisiae",
        "strain": "BY4741",
        "ploidy": "haploid",
    }


def test_screen_writes_gene_set_and_build_manifest(
    screen: m.IsobutanolScreenLopez2024Dataset,
) -> None:
    """``preprocess/gene_set.json`` is the four resolved ORFs sorted; the build manifest
    names the root slug, the loader class/module, this host, and the worktree HEAD.

    The HEAD comparison shells out to ``git rev-parse HEAD`` in the loader's directory,
    so this test fails in a non-git checkout, where ``torchcell_commit`` is None by design.
    """
    preprocess = Path(screen.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
        "YCR001W",
        "YDR001C",
    ]
    assert screen.gene_set == {"YAL001C", "YBR001C", "YCR001W", "YDR001C"}
    assert screen.experiment_class is MetaboliteExperiment
    assert screen.reference_class is MetaboliteExperimentReference
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(m.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == "isobutanol_screen_lopez2024"
    assert manifest["loader_class"] == "IsobutanolScreenLopez2024Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.lopez2024"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {"MetaboliteExperiment", "KanMxDeletionPerturbation"} <= set(
        manifest["closure"]
    )


def test_validated_reads_both_blocks_and_drops_the_contradictory_strain(
    validated: m.IsobutanolValidatedLopez2024Dataset,
) -> None:
    """UP rows then DOWN rows, each a triplicate record with SE = STD / sqrt(3).

    YAL001C 3.0 (SE 0.3 / sqrt(3) = 0.1732), YDR002W -> YDR001C 2.2 (0.6 / sqrt(3)),
    ybr001c -> YBR001C 0.25 (0.06 / sqrt(3)), YCR001W 0.4 (0.09 / sqrt(3)); YBL071W-A is
    dropped from both blocks and YER001W is unresolved, so 7 source rows give 4 records.
    The reference is one shared object with n = 3, so the index has one entry [0, 1, 2, 3].
    """
    assert len(validated) == 4
    assert _records(validated) == [
        ("YAL001C", "TFC3", 3.0, 0.3 / math.sqrt(3), 3),
        ("YDR001C", "NTH1", 2.2, 0.6 / math.sqrt(3), 3),
        ("YBR001C", "NTH2", 0.25, 0.06 / math.sqrt(3), 3),
        ("YCR001W", "YCR001W", 0.4, 0.09 / math.sqrt(3), 3),
    ]
    assert validated[1]["experiment"] == _experiment(
        "IsobutanolValidatedLopez2024Dataset",
        "YDR001C",
        "NTH1",
        2.2,
        0.6 / math.sqrt(3),
        3,
    )
    assert validated[2]["reference"] == _reference(
        "IsobutanolValidatedLopez2024Dataset", 3
    )
    assert validated[2]["publication"] == _PUBLICATION.model_dump()
    index = json.loads(
        (
            Path(validated.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [[0, 1, 2, 3]]
    assert json.loads(
        (Path(validated.root) / "preprocess" / "gene_set.json").read_text()
    ) == ["YAL001C", "YBR001C", "YCR001W", "YDR001C"]
    manifest = json.loads(
        (Path(validated.root) / "preprocess" / "build_manifest.json").read_text()
    )
    assert manifest["dataset_name"] == "isobutanol_validated_lopez2024"
    assert manifest["loader_class"] == "IsobutanolValidatedLopez2024Dataset"


def test_validated_raises_when_an_orf_repeats_after_resolution(tmp_path: Path) -> None:
    """YAL001C in the UP block and again in the DOWN block is a second contradictory strain
    and raises rather than overwriting; the message names the ORF and the source name.
    """
    root = _root(
        tmp_path,
        "dup",
        up_rows=[("YAL001C", 3.0, 0.3)],
        down_rows=[("YAL001C", 0.25, 0.06)],
    )
    with pytest.raises(
        RuntimeError,
        match=r"ORF YAL001C \(from YAL001C\) appears twice after resolution",
    ):
        m.IsobutanolValidatedLopez2024Dataset(root=str(root), genome=_genome())


def test_both_loaders_require_an_injected_genome(tmp_path: Path) -> None:
    """With the raw workbook present and ``genome=None`` the first resolver call raises;
    the message names the loader class.
    """
    with pytest.raises(
        RuntimeError,
        match="IsobutanolScreenLopez2024Dataset requires an injected SCerevisiaeGenome",
    ):
        m.IsobutanolScreenLopez2024Dataset(root=str(_root(tmp_path, "s")), genome=None)
    with pytest.raises(
        RuntimeError,
        match="IsobutanolValidatedLopez2024Dataset requires an injected SCerevisiaeGenome",
    ):
        m.IsobutanolValidatedLopez2024Dataset(
            root=str(_root(tmp_path, "v")), genome=None
        )


def test_download_reads_the_library_mirror_and_verifies_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw file, ``download()`` copies from ``$DATA_ROOT/torchcell-library/
    lopezSystemsMetabolicEngineering2024/data/`` and checks the pinned digest. A missing
    mirror file raises naming the path; a mirror file holding ``b"not the real workbook"``
    is copied and then rejected with its actual sha256 (55b4751d...) and the pinned one.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / "data"
    with pytest.raises(
        RuntimeError, match=f"library mirror data file not found: {mirror}"
    ):
        m.IsobutanolScreenLopez2024Dataset(root=str(tmp_path / "a"), genome=_genome())
    mirror.mkdir(parents=True)
    (mirror / m._XLSX_FILENAME).write_bytes(b"not the real workbook")
    digest = hashlib.sha256(b"not the real workbook").hexdigest()
    assert digest.startswith("55b4751d")
    with pytest.raises(
        RuntimeError, match=f"sha256 mismatch: got {digest}, expected {m._XLSX_SHA256}"
    ):
        m.IsobutanolScreenLopez2024Dataset(root=str(tmp_path / "b"), genome=_genome())
    assert (tmp_path / "b" / "raw" / m._XLSX_FILENAME).read_bytes() == (
        b"not the real workbook"
    )
