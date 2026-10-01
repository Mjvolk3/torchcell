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

2026.09.30 (Phase 16): the same five rows, plus

- the ledger line (line 316): "Wrote 3 ODuibhir2014 fitness experiments to LMDB (1309
  ORFs dropped: ...)", 1309 = 1307 fillers + YZZ999W + YER001W, the raw tokens sorted.
- ``transform_item`` round trips through ``FitnessExperiment`` and
  ``FitnessExperimentReference`` (a gene-interaction class would drop ``fitness``).
- a duplicate and edge matrix (padded to 1312): YAL001C at log2relT 1.0, again at 0.5
  (fitness 2^-0.5 = 0.7071067811865476), lowercase ``yal001c`` / ``tfc3`` at 2
  (fitness 0.25), and YBR001C with a blank ``commonName`` at 0.25 (fitness 2^-0.25 =
  0.8408964152537145).
- a blank ``log2relT`` (row 1 of the matrix): ``2.0 ** -nan`` is NaN and the
  ``FitnessPhenotype`` validator refuses "Fitness cannot be NaN".
- ``download`` with ``_DATASET_S2_SHA256`` replaced by the digest of the synthetic file
  (read at call time): the mirror copy lands byte-identical and logs "Verified <dest>
  (sha256 <digest>)"; a mirror file whose digest differs from the pin raises
  ``RawSha256MismatchError`` before the copy, so ``raw/`` stays empty (issue #537).
- ``process`` verifies the file in ``raw/`` against the pin before reading a row, so a
  file placed there by hand is refused with no store written (issue #537).
- ``main`` with ``load_dotenv`` stubbed and the genome and dataset classes as recorders.

2026.10.01 (issue #537): the Phase 16 findings are retired. An ORF listed twice (verbatim
or in lowercase), a blank ``commonName`` and a blank ``log2relT`` are each refused with a
named ``RuntimeError`` before the store is opened, so the refused root holds no
``processed/lmdb`` and a retry refuses again (each 0 of the 1312 rows of the pinned file).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
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


@pytest.fixture(autouse=True)
def _no_tc_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build here reads ``raw/``; an inherited ``TC_DATA_URL`` would send it to
    the tc-data endpoint instead (``ExperimentDataset._download``).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)


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
    and one reference covers every record.
    """
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


def test_download_leaves_a_present_file_to_the_build_check_and_needs_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file already in ``raw/`` is not copied over (``process`` verifies it at build
    time); with no raw file and no mirror the constructor names the missing mirror path.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data_root"))
    dataset = m.SmfODuibhir2014Dataset(root=str(_root(tmp_path)), genome=_genome())
    dest = Path(dataset.root) / "raw" / m._RAW_FILENAME
    before = dest.read_bytes()
    dataset.download()
    assert dest.read_bytes() == before
    assert sorted(p.name for p in dest.parent.iterdir()) == [m._RAW_FILENAME]
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


def test_items_retype_through_the_fitness_classes(
    dataset: m.SmfODuibhir2014Dataset,
) -> None:
    """``transform_item`` rebuilds each stored item through the declared classes (lines
    146 to 153): a ``FitnessExperiment`` and a ``FitnessExperimentReference`` that dump
    back to exactly the stored dictionaries.
    """
    for index in range(3):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is FitnessExperiment
        assert type(typed["reference"]) is FitnessExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION


def test_inline_build_leaves_the_generic_hooks_inert(
    dataset: m.SmfODuibhir2014Dataset,
) -> None:
    """Records are built inside ``process``: ``preprocess_raw`` hands back the frame it was
    given, unchanged, and ``create_experiment`` raises a bare ``NotImplementedError``.
    """
    frame = pd.DataFrame({"orf": ["YAL001C"], "log2relT": [1.0]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"orf": ["YAL001C"], "log2relT": [1.0]}
    with pytest.raises(NotImplementedError) as info:
        dataset.create_experiment()
    assert str(info.value) == ""


def test_drop_ledger_names_every_unresolved_raw_token(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The ledger line counts the three written records and the 1309 dropped rows (1307
    fillers plus YZZ999W and YER001W, whose alias target YER999W is not an ID), listing
    the raw tokens in sorted order.
    """
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.SmfODuibhir2014Dataset(root=str(_root(tmp_path)), genome=_genome())
    fillers = [f"FILLER{k:04d}" for k in range(1307)]
    wrote = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and r.getMessage().startswith("Wrote")
    ]
    assert wrote == [
        "Wrote 3 ODuibhir2014 fitness experiments to LMDB (1309 ORFs dropped: "
        f"{[*fillers, 'YER001W', 'YZZ999W']})"
    ]


def _refused_twice(root: Path, message: str) -> None:
    """Two constructors on ``root`` both raise ``message`` and leave no store behind."""
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
        assert str(info.value) == message
        assert not (root / "processed" / "lmdb").exists()
        assert not (root / "preprocess" / "gene_set.json").exists()


def _edge_root(tmp_path: Path, slug: str, rows: list[list[str]]) -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    _write_s2(root / "raw", rows)
    return root


@pytest.mark.parametrize(
    ("second", "token"),
    [
        (["YAL001C", "TFC3", "0.5", "0.1"], "YAL001C"),
        (["yal001c", "tfc3", "2", "0.0"], "yal001c"),
    ],
)
def test_an_orf_listed_twice_is_refused_before_the_store_opens(
    tmp_path: Path, second: list[str], token: str
) -> None:
    """Contract (issue #537): one record per deletion strain. A second row resolving to
    an ORF already seen, verbatim or in lowercase, is refused naming both raw tokens,
    before ``processed/lmdb`` is opened, so a retry on the same root refuses again
    instead of serving an empty store. The pinned Dataset S2 has 0 such rows of 1312.
    """
    root = _edge_root(
        tmp_path, f"dup_{token}", [["YAL001C", "TFC3", "1.0", "0.1"], second]
    )
    _refused_twice(
        root,
        f"Dataset S2 lists YAL001C twice (rows 'YAL001C' and '{token}'); one record "
        "per deletion strain is required",
    )


def test_a_blank_common_name_is_refused_before_the_store_opens(tmp_path: Path) -> None:
    """Contract (issue #537): a blank ``commonName`` is refused by row instead of being
    stored as the string "nan"; nothing is written, and a retry refuses again. The
    pinned Dataset S2 has 0 blank common names of 1312.
    """
    root = _edge_root(
        tmp_path,
        "blank_name",
        [["YAL001C", "TFC3", "1.0", "0.1"], ["YBR001C", "", "0.25", "0.2"]],
    )
    _refused_twice(root, "Dataset S2 row 'YBR001C' has a blank commonName")


def test_a_blank_log2relt_is_refused_before_the_store_opens(tmp_path: Path) -> None:
    """Contract (issue #537): a blank ``log2relT`` is refused by row with a named
    message (it used to reach the phenotype validator as NaN fitness after the write
    env was open, leaving an empty ``processed/lmdb`` that a retry served as 0
    records). Now no store exists after the refusal and a retry refuses again. The
    pinned Dataset S2 has 0 blank ``log2relT`` of 1312.
    """
    root = _edge_root(
        tmp_path,
        "blank",
        [["YAL001C", "TFC3", "1.0", "0.1"], ["YCR001W", "YCR001W", "", "0.3"]],
    )
    _refused_twice(root, "Dataset S2 row 'YCR001W' has a blank log2relT")


def _mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: bytes) -> Path:
    """Stage ``content`` at the library-mirror path under a temporary ``DATA_ROOT``."""
    data_root = tmp_path / "data_root"
    src = data_root / "torchcell-library" / m._LIBRARY_CITATION_KEY / m._MIRROR_REL_PATH
    src.parent.mkdir(parents=True)
    src.write_bytes(content)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    return src


def _synthetic_bytes(tmp_path: Path) -> bytes:
    source = _root(tmp_path, "source")
    return (source / "raw" / m._RAW_FILENAME).read_bytes()


def test_download_copies_the_pinned_mirror_file_then_verifies_in_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """With the pin set to the synthetic file's digest, an empty root copies the mirror
    file byte for byte, logs "Verified <dest> (sha256 <pin>)" and builds three records
    under the real build-time check; a second ``download`` with the mirror removed finds
    the raw copy present and neither copies nor logs.
    """
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    content = _synthetic_bytes(tmp_path)
    digest = hashlib.sha256(content).hexdigest()
    src = _mirror(tmp_path, monkeypatch, content)
    monkeypatch.setattr(m, "_DATASET_S2_SHA256", digest)
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.SmfODuibhir2014Dataset(
            root=str(tmp_path / "fresh"), genome=_genome()
        )
    dest = tmp_path / "fresh" / "raw" / m._RAW_FILENAME
    assert dest.read_bytes() == content
    assert len(dataset) == 3
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name][0] == (
        f"Verified {dest} (sha256 {digest})"
    )
    src.unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset.download()
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == []
    assert dest.read_bytes() == content


def test_a_mirror_file_off_the_pin_is_refused_and_nothing_lands_in_raw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #537): the mirror holds a well-formed Dataset S2 whose digest is
    not the pin. ``download`` hashes it before copying and raises
    ``RawSha256MismatchError`` naming the mirror file and both digests; ``raw/`` stays
    empty, so a second constructor runs ``download`` again and refuses again.
    """
    content = _synthetic_bytes(tmp_path)
    digest = hashlib.sha256(content).hexdigest()
    src = _mirror(tmp_path, monkeypatch, content)
    root = tmp_path / "unverified"
    for _ in range(2):
        with pytest.raises(RawSha256MismatchError) as err:
            m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
        assert str(err.value) == (
            f"sha256 mismatch for {src}: expected {m._DATASET_S2_SHA256}, "
            f"observed {digest}"
        )
        assert list((root / "raw").iterdir()) == []
        assert not (root / "processed" / "lmdb").exists()


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #537): with Dataset S2 already in ``raw/`` PyG skips
    ``download``, so ``process`` verifies it first and raises ``RawSha256MismatchError``
    before any row is read; no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(m, [m._RAW_FILENAME])
    raw = staged.root / "raw" / m._RAW_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        m.SmfODuibhir2014Dataset(root=str(staged.root), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "37ef19ee249c64c0557c84870e59b2fd7a8bbaf14371fd355775e650f2a39f1c, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_main_builds_genome_and_dataset_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds the genome from ``$DATA_ROOT/data/sgd/genome`` and ``data/go`` with
    ``overwrite=False``, then the dataset at ``$DATA_ROOT/data/torchcell/smf_oduibhir2014``
    with that genome, and prints its length and first item. Both classes are recorders.
    """
    monkeypatch.setenv("DATA_ROOT", str(tmp_path))
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    calls: list[tuple[str, dict[str, Any]]] = []

    class _Genome:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("genome", kwargs))

    class _Dataset:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(("dataset", kwargs))

        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(m, "SCerevisiaeGenome", _Genome)
    monkeypatch.setattr(m, "SmfODuibhir2014Dataset", _Dataset)
    m.main()
    assert [name for name, _ in calls] == ["genome", "dataset"]
    assert calls[0][1] == {
        "genome_root": f"{tmp_path}/data/sgd/genome",
        "go_root": f"{tmp_path}/data/go",
        "overwrite": False,
    }
    assert list(calls[1][1]) == ["root", "genome"]
    assert calls[1][1]["root"] == f"{tmp_path}/data/torchcell/smf_oduibhir2014"
    assert type(calls[1][1]["genome"]) is _Genome
    assert capsys.readouterr().out == "len = 4\nitem[0]\n"
