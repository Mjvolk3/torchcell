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
  (read at call time, line 181): the mirror copy lands byte-identical and logs
  "Verified <dest> (sha256 <digest>)"; a mirror file whose digest differs from the pin
  raises "sha256 mismatch" after the copy.
- ``main`` with ``load_dotenv`` stubbed and the genome and dataset classes as recorders.

Findings pinned here: the same ORF listed twice (or once in lowercase) gives one record
per row with identical genotypes (no duplicate check in lines 292 to 314); a blank
``commonName`` is stored as the string "nan" (``str(row["commonName"])``, line 301);
a blank ``log2relT`` raises pydantic's ``ValidationError`` instead of a named refusal,
and because the write env is opened first (line 289) the aborted build leaves an empty
``processed/lmdb`` that a retry serves as 0 records; a mirror file that fails the pin is
left in ``raw/`` after the refusal (the copy at line 179 precedes the check at 180), so
the next constructor finds the raw file present, skips ``download`` and builds from the
unverified bytes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pydantic
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


def test_duplicate_rows_are_all_kept_and_a_blank_common_name_becomes_nan(
    tmp_path: Path,
) -> None:
    """Finding: YAL001C twice and ``yal001c`` once give three records with the same
    systematic name (the lowercase row keeps its lowercase ``tfc3``), because nothing
    deduplicates the resolved ORF (lines 292 to 314). Finding: a blank ``commonName`` is
    read as NaN and stored as the string "nan" (line 301). Pinned until the loader
    refuses a duplicated ORF and a missing common name.
    """
    root = tmp_path / "dups"
    (root / "raw").mkdir(parents=True)
    _write_s2(
        root / "raw",
        [
            ["YAL001C", "TFC3", "1.0", "0.1"],
            ["YAL001C", "TFC3", "0.5", "0.1"],
            ["YBR001C", "", "0.25", "0.2"],
            ["yal001c", "tfc3", "2", "0.0"],
        ],
    )
    dataset = m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
    assert [dataset[i]["experiment"] for i in range(len(dataset))] == [
        _experiment("YAL001C", "TFC3", 0.5),
        _experiment("YAL001C", "TFC3", 0.7071067811865476),
        _experiment("YBR001C", "nan", 0.8408964152537145),
        _experiment("YAL001C", "tfc3", 0.25),
    ]
    assert json.loads((root / "preprocess" / "gene_set.json").read_text()) == [
        "YAL001C",
        "YBR001C",
    ]
    dataset.close_lmdb()


def test_blank_log2relt_fails_validation_and_leaves_a_store_a_retry_serves_empty(
    tmp_path: Path,
) -> None:
    """Finding: a blank ``log2relT`` gives ``2.0 ** -nan`` = NaN and the phenotype
    validator raises pydantic's ``ValidationError`` ("Fitness cannot be NaN"), not a named
    refusal. The write env was already opened at line 289, so the failed build leaves an
    empty ``processed/lmdb`` and no ``gene_set.json``; a retry on the same root finds the
    store, skips ``process()`` and serves 0 records. Pinned until the loader refuses a
    blank cell before opening the store.
    """
    root = tmp_path / "blank"
    (root / "raw").mkdir(parents=True)
    _write_s2(
        root / "raw",
        [["YAL001C", "TFC3", "1.0", "0.1"], ["YCR001W", "YCR001W", "", "0.3"]],
    )
    with pytest.raises(pydantic.ValidationError) as info:
        m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
    assert [e["msg"] for e in info.value.errors()] == [
        "Value error, Fitness cannot be NaN"
    ]
    assert (root / "processed" / "lmdb").is_dir()
    assert not (root / "preprocess" / "gene_set.json").exists()
    retry = m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
    assert len(retry) == 0
    retry.close_lmdb()


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
    file byte for byte, logs "Verified <dest> (sha256 <pin>)" and builds three records; a
    second ``download`` with the mirror removed re-verifies the raw copy and logs again.
    """
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
    src.unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset.download()
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == [
        f"Verified {dest} (sha256 {digest})"
    ]


def test_a_mirror_file_off_the_pin_is_refused_but_left_in_raw_for_the_next_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror holds a well-formed Dataset S2 whose digest is not the pin: the copy
    lands, then the check raises "sha256 mismatch" with that digest.

    Finding: the refused copy stays in ``raw/`` (the copy at line 179 precedes the check
    at 180), so a second constructor finds every raw file present, PyG skips
    ``download()`` and the build serves three records from the unverified bytes. Pinned
    until a failed check removes the copy.
    """
    content = _synthetic_bytes(tmp_path)
    digest = hashlib.sha256(content).hexdigest()
    _mirror(tmp_path, monkeypatch, content)
    root = tmp_path / "unverified"
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{m._RAW_FILENAME} sha256 mismatch: got {digest}, "
            f"expected {m._DATASET_S2_SHA256}"
        ),
    ):
        m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
    assert (root / "raw" / m._RAW_FILENAME).read_bytes() == content
    rebuilt = m.SmfODuibhir2014Dataset(root=str(root), genome=_genome())
    assert rebuilt[0]["experiment"] == _experiment("YAL001C", "TFC3", 0.5)
    assert len(rebuilt) == 3
    rebuilt.close_lmdb()


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
