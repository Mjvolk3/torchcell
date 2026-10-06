# tests/torchcell/datasets/scerevisiae/test_yoshida2012.py
# [[tests.torchcell.datasets.scerevisiae.test_yoshida2012]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_yoshida2012.py
"""Hermetic build of the Yoshida 2012 organic-acid loader from its embedded Table 3.

The only raw file is ``paper.pdf``, whose presence is all PyG checks; a placeholder is
written so ``download()`` is never called, and the values come from the module-level
``TABLE_3`` literal. ``build_metabolite_s_id_map`` (which loads Yeast9) is replaced by a
stub that records its argument and returns ``s_0001`` to ``s_0005`` for the five acids.
The genome stub carries ``alias_to_systematic`` for the 16 common names and a
``gene_set`` of their 16 ORFs plus ``YDR379C-A``, which matches the systematic-name regex
and is checked against that set.

Record 0 = ASM4 (Table 3 mean, SD; n = 3):
    acetate 4.02 +/- 0.30, citrate 0.10 +/- 0.03, malate 0.15 +/- 0.02,
    pyruvate 0.13 +/- 0.04, succinate 1.31 +/- 0.06, phosphate 0.78 +/- 0.07
SE = SD / sqrt(3) per analyte; ``target_metabolite_ids`` covers the five acids only.
Reference = the WT row for every record: acetate 4.21 +/- 0.30, citrate 0.10 +/- 0.04,
malate 0.15 +/- 0.01, pyruvate 0.18 +/- 0.05, succinate 1.29 +/- 0.05, phosphate 0.69
+/- 0.11, same SE rule; BY4742, static liquid YPD at 25 C.

2026.09.30 (Phase 16): the full build as above, plus

- the two ledger lines (lines 387 and 415): "Yoshida2012: 17 deletion strains, WT
  reference over 6 analytes, 5 organic acids mapped to Yeast9 s_NNNN" and "Wrote 17
  Yoshida2012 organic-acid experiments to LMDB".
- ``transform_item`` round trips through ``MetaboliteExperiment`` and
  ``MetaboliteExperimentReference`` for all 17 records.
- ``create_experiment`` on a built dataset with a strain row measuring only acetate
  (4.0 +/- 0.3) and phosphate (0.5 +/- 0.06): the reference is the WT row restricted to
  those two analytes (4.21 / 0.30 and 0.69 / 0.11), SE = SD / sqrt(3), and the target
  ids cover acetate only; a phosphate-only row has ``target_metabolite_ids`` None
  (``targets or None``, line 441).
- a synthetic ``TABLE_3`` (module literal replaced; ``process`` reads it at call time,
  line 382): WT as released; ASM4 as released; ``YDL088C`` (the ASM4 ORF written
  systematically) with citrate (nan, nan); ``yal999w`` (systematic-shaped, not in the
  genome stub) and ``AMB`` whose alias maps to two ORFs, [YBR001C, YCR001W]. Four
  records in that order, with ORFs YDL088C, YDL088C, YAL999W and YBR001C.
- a negative SD (acetate (4.0, -0.3)) refuses in the phenotype validator with "SE for
  acetate must be non-negative" (-0.3 / sqrt(3) < 0).
- ``download`` with ``PDF_SHA256`` replaced by the digest of a synthetic PDF: the
  mirror file is copied and "Staged <dest> (<n> bytes, sha256 verified)" is logged.
- ``main`` with ``load_dotenv`` stubbed and the genome and dataset classes as recorders.

2026.10.01 (issue #537): the Phase 16 findings are retired. A systematic-shaped name
absent from ``genome.gene_set`` (``yal999w``), an alias of two ORFs (``AMB``), two rows
resolving to one ORF (ASM4 and ``YDL088C``) and a NaN mean or SD are each refused with a
named ``RuntimeError`` before ``data.csv`` or the store is written; a retry refuses
again. The released Table 3 has 0 of each among its 17 strains (``YDR379C-A`` is a
gene of the R64 genome; every common name has exactly one candidate ORF).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pydantic
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.data.experiment_dataset import verify_raw_files
from torchcell.datamodels.schema import (
    Environment,
    Genotype,
    KanMxDeletionPerturbation,
    MetaboliteExperiment,
    MetaboliteExperimentReference,
    MetabolitePhenotype,
    Publication,
    ReferenceGenome,
    Temperature,
)
from torchcell.datasets.scerevisiae import yoshida2012 as m
from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome


@pytest.fixture(autouse=True)
def _no_tc_data_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every build here reads ``raw/``; an inherited ``TC_DATA_URL`` would send it to
    the tc-data endpoint instead (``ExperimentDataset._download``).
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)


_ORF_BY_NAME = {
    "ASM4": "YDL088C",
    "EMI5": "YOL071W",
    "GTR1": "YML121W",
    "GTR2": "YGR163W",
    "LIP5": "YOR196C",
    "LSM1": "YJL124C",
    "MKS1": "YNL076W",
    "NFU1": "YKL040C",
    "PCK1": "YKR097W",
    "PHO85": "YPL031C",
    "PLM2": "YDR501W",
    "RTG1": "YOL067C",
    "RTG2": "YGL252C",
    "TIF3": "YPR163C",
    "UBA3": "YPR066W",
    "UBP3": "YER151C",
}
_S_IDS = {
    "acetate": "s_0001",
    "citrate": "s_0002",
    "malate": "s_0003",
    "pyruvate": "s_0004",
    "succinate": "s_0005",
}


class _StubGenome:
    def __init__(self, names: dict[str, str]) -> None:
        self.alias_to_systematic = {k: [v] for k, v in names.items()}
        self.gene_set = {*_ORF_BY_NAME.values(), "YDR379C-A"}


def _genome(names: dict[str, str] = _ORF_BY_NAME) -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome(names))


@pytest.fixture
def s_id_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    calls: list[dict[str, str]] = []

    def fake_map(kegg_by_metabolite: dict[str, str]) -> dict[str, str]:
        calls.append(dict(kegg_by_metabolite))
        return dict(_S_IDS)

    monkeypatch.setattr(m, "build_metabolite_s_id_map", fake_map)
    return calls


def _root(tmp_path: Path, slug: str = "organic_acid_yoshida2012") -> Path:
    root = tmp_path / slug
    (root / "raw").mkdir(parents=True)
    (root / "raw" / m.PDF_FILENAME).write_bytes(b"%PDF-1.4 synthetic placeholder")
    return root


@pytest.fixture
def dataset(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> m.OrganicAcidYoshida2012Dataset:
    return m.OrganicAcidYoshida2012Dataset(root=str(_root(tmp_path)), genome=_genome())


_ENVIRONMENT = Environment(media=m.YOSHIDA_YPD, temperature=Temperature(value=25))
_PUBLICATION = Publication(
    pubmed_id="22277779",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/22277779/",
    doi="10.1016/j.jbiosc.2011.12.017",
    doi_url="https://doi.org/10.1016/j.jbiosc.2011.12.017",
).model_dump()
_ANALYTES = ["acetate", "citrate", "malate", "pyruvate", "succinate", "phosphate"]


def _phenotype(means: list[float], sds: list[float]) -> MetabolitePhenotype:
    return MetabolitePhenotype(
        metabolite_level=dict(zip(_ANALYTES, means, strict=True)),
        metabolite_level_se={
            a: sd / math.sqrt(3) for a, sd in zip(_ANALYTES, sds, strict=True)
        },
        n_replicates=dict.fromkeys(_ANALYTES, 3),
        measurement_type="hplc_organic_acid_titer_mM",
        target_metabolite_ids=dict(_S_IDS),
    )


_WT_PHENOTYPE = _phenotype(
    [4.21, 0.10, 0.15, 0.18, 1.29, 0.69], [0.30, 0.04, 0.01, 0.05, 0.05, 0.11]
)
_REFERENCE = MetaboliteExperimentReference(
    dataset_name="OrganicAcidYoshida2012Dataset",
    genome_reference=ReferenceGenome(
        species="Saccharomyces cerevisiae", strain="BY4742"
    ),
    environment_reference=_ENVIRONMENT,
    phenotype_reference=_WT_PHENOTYPE,
).model_dump()


def test_record_0_is_asm4_with_se_from_sd_over_sqrt_3(
    dataset: m.OrganicAcidYoshida2012Dataset, s_id_calls: list[dict[str, str]]
) -> None:
    """The mapper is called once with ``ACID_KEGG_IDS``; record 0 stores the six ASM4
    analytes (OD dropped), SE = SD / sqrt(3), n = 3, Yeast9 ids for the five acids and
    none for phosphate, against the measured WT reference.
    """
    assert s_id_calls == [m.ACID_KEGG_IDS]
    assert len(dataset) == 17
    assert (
        dataset[0]["experiment"]
        == MetaboliteExperiment(
            dataset_name="OrganicAcidYoshida2012Dataset",
            genotype=Genotype(
                perturbations=[
                    KanMxDeletionPerturbation(
                        systematic_gene_name="YDL088C", perturbed_gene_name="ASM4"
                    )
                ]
            ),
            environment=_ENVIRONMENT,
            phenotype=_phenotype(
                [4.02, 0.10, 0.15, 0.13, 1.31, 0.78],
                [0.30, 0.03, 0.02, 0.04, 0.06, 0.07],
            ),
        ).model_dump()
    )
    assert dataset[0]["reference"] == _REFERENCE
    assert dataset[0]["publication"] == _PUBLICATION
    assert dataset[0]["experiment"]["phenotype"]["metabolite_level_se"]["acetate"] == (
        0.30 / math.sqrt(3)
    )


def test_seventeen_records_in_table_order_with_one_shared_reference(
    dataset: m.OrganicAcidYoshida2012Dataset,
) -> None:
    """Records follow ``TABLE_3`` order minus WT; the last is YDR379C-A, resolved by the
    regex rather than the alias table, with the ORF as its perturbed name. Every strain
    measures all six analytes, so the WT reference is shared: one index entry.
    """
    genes = [g for g in m.TABLE_3 if g != "WT"]
    stored = [
        (
            dataset[i]["experiment"]["genotype"]["perturbations"][0][
                "systematic_gene_name"
            ],
            dataset[i]["experiment"]["genotype"]["perturbations"][0][
                "perturbed_gene_name"
            ],
        )
        for i in range(17)
    ]
    assert stored == [(_ORF_BY_NAME.get(g, g), g) for g in genes]
    assert stored[-1] == ("YDR379C-A", "YDR379C-A")
    preprocess = Path(dataset.root) / "preprocess"
    index = json.loads((preprocess / "experiment_reference_index.json").read_text())
    assert [entry["member_indices"] for entry in index] == [list(range(17))]
    assert (preprocess / "data.csv").read_text() == "orf,gene\n" + "".join(
        f"{_ORF_BY_NAME.get(g, g)},{g}\n" for g in genes
    )
    assert json.loads((preprocess / "gene_set.json").read_text()) == sorted(
        [*_ORF_BY_NAME.values(), "YDR379C-A"]
    )
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    assert manifest["dataset_name"] == "organic_acid_yoshida2012"
    assert manifest["loader_class"] == "OrganicAcidYoshida2012Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.yoshida2012"


def test_unresolvable_common_name_raises(
    tmp_path: Path, s_id_calls: list[dict[str, str]]
) -> None:
    names = {k: v for k, v in _ORF_BY_NAME.items() if k != "ASM4"}
    with pytest.raises(
        RuntimeError, match="Yoshida2012: could not resolve gene name 'ASM4'"
    ):
        m.OrganicAcidYoshida2012Dataset(
            root=str(_root(tmp_path)), genome=_genome(names)
        )


def test_requires_an_injected_genome(tmp_path: Path) -> None:
    """The genome check precedes the Yeast9 mapper call, so no stub is needed here."""
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "OrganicAcidYoshida2012Dataset requires an injected SCerevisiaeGenome to "
            "resolve common gene names to systematic ORF ids (Table 3 uses names)."
        ),
    ):
        m.OrganicAcidYoshida2012Dataset(root=str(_root(tmp_path)), genome=None)


def test_download_stages_the_mirror_pdf_only_after_verifying_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s_id_calls: list[dict[str, str]]
) -> None:
    """An empty root looks for ``$DATA_ROOT/torchcell-library/<key>/paper.pdf`` and names
    it when absent; a mirror PDF holding ``b"wrong pdf"`` is rejected with
    ``RawSha256MismatchError`` naming it, the pin and its digest (43d5ed94...) and not
    copied into ``raw/``. With ``paper.pdf`` already in ``raw/`` the method returns
    without hashing it, and the build-time check in ``process`` (issue #537's sweep)
    refuses the staged copy instead.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = data_root / "torchcell-library" / m.LIBRARY_CITATION_KEY / m.PDF_FILENAME
    with pytest.raises(
        RuntimeError, match=re.escape(f"Yoshida2012 mirror PDF not found: {mirror}")
    ):
        m.OrganicAcidYoshida2012Dataset(root=str(tmp_path / "empty"), genome=_genome())
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(b"wrong pdf")
    digest = hashlib.sha256(b"wrong pdf").hexdigest()
    assert digest.startswith("43d5ed94")
    with pytest.raises(RawSha256MismatchError) as err:
        m.OrganicAcidYoshida2012Dataset(root=str(tmp_path / "empty2"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror}: expected {m.PDF_SHA256}, observed {digest}"
    )
    assert list((tmp_path / "empty2" / "raw").iterdir()) == []
    dataset = m.OrganicAcidYoshida2012Dataset(
        root=str(_root(tmp_path)), genome=_genome()
    )
    dest = Path(dataset.root) / "raw" / m.PDF_FILENAME
    placeholder = hashlib.sha256(dest.read_bytes()).hexdigest()
    assert placeholder != m.PDF_SHA256
    dataset.download()
    assert dest.read_bytes() == b"%PDF-1.4 synthetic placeholder"
    monkeypatch.setattr(m, "verify_raw_files", verify_raw_files)
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.process()
    assert str(err.value) == (
        f"sha256 mismatch for {dest}: expected {m.PDF_SHA256}, observed {placeholder}"
    )


def test_items_retype_through_the_metabolite_classes(
    dataset: m.OrganicAcidYoshida2012Dataset,
) -> None:
    """``transform_item`` rebuilds every stored item through the declared classes (lines
    311 to 319): a ``MetaboliteExperiment`` and a ``MetaboliteExperimentReference`` that
    dump back to exactly the stored dictionaries. A fitness class would drop
    ``metabolite_level`` and fail the round trip.
    """
    for index in range(17):
        item = dataset[index]
        typed = dataset.transform_item(item)
        assert type(typed["experiment"]) is MetaboliteExperiment
        assert type(typed["reference"]) is MetaboliteExperimentReference
        assert typed["experiment"].model_dump() == item["experiment"]
        assert typed["reference"].model_dump() == item["reference"]
        assert typed["publication"].model_dump() == _PUBLICATION


def test_build_logs_the_strain_analyte_and_mapping_counts(
    tmp_path: Path, s_id_calls: list[dict[str, str]], caplog: pytest.LogCaptureFixture
) -> None:
    """17 strains (18 rows minus WT), 6 reference analytes (OD dropped), 5 mapped acids,
    then the written count.
    """
    with caplog.at_level(logging.INFO, logger=m.log.name):
        m.OrganicAcidYoshida2012Dataset(root=str(_root(tmp_path)), genome=_genome())
    assert [r.getMessage() for r in caplog.records if r.name == m.log.name] == [
        "Yoshida2012: 17 deletion strains, WT reference over 6 analytes, "
        "5 organic acids mapped to Yeast9 s_NNNN",
        "Wrote 17 Yoshida2012 organic-acid experiments to LMDB",
    ]


def test_reference_is_restricted_to_the_analytes_the_strain_measured(
    dataset: m.OrganicAcidYoshida2012Dataset,
) -> None:
    """A row measuring acetate and phosphate gets a WT reference over those two analytes
    only, and target ids for acetate only; a phosphate-only row has no target ids at all
    (``targets or None``). ``preprocess_raw`` returns its frame unchanged.
    """
    experiment, reference, publication = dataset.create_experiment(
        {
            "orf": "YDL088C",
            "gene": "ASM4",
            "analytes": {"acetate": (4.0, 0.3), "phosphate": (0.5, 0.06)},
        }
    )
    assert experiment.phenotype.model_dump() == (
        MetabolitePhenotype(
            metabolite_level={"acetate": 4.0, "phosphate": 0.5},
            metabolite_level_se={
                "acetate": 0.3 / math.sqrt(3),
                "phosphate": 0.06 / math.sqrt(3),
            },
            n_replicates={"acetate": 3, "phosphate": 3},
            measurement_type="hplc_organic_acid_titer_mM",
            target_metabolite_ids={"acetate": "s_0001"},
        ).model_dump()
    )
    assert reference.phenotype_reference.model_dump() == (
        MetabolitePhenotype(
            metabolite_level={"acetate": 4.21, "phosphate": 0.69},
            metabolite_level_se={
                "acetate": 0.30 / math.sqrt(3),
                "phosphate": 0.11 / math.sqrt(3),
            },
            n_replicates={"acetate": 3, "phosphate": 3},
            measurement_type="hplc_organic_acid_titer_mM",
            target_metabolite_ids={"acetate": "s_0001"},
        ).model_dump()
    )
    assert publication.model_dump() == _PUBLICATION
    phosphate_only, phosphate_reference, _ = dataset.create_experiment(
        {"orf": "YDL088C", "gene": "ASM4", "analytes": {"phosphate": (0.5, 0.06)}}
    )
    assert phosphate_only.phenotype.target_metabolite_ids is None
    assert phosphate_reference.phenotype_reference.metabolite_level == {
        "phosphate": 0.69
    }
    frame = pd.DataFrame({"orf": ["YDL088C"], "gene": ["ASM4"]})
    returned = dataset.preprocess_raw(frame)
    assert returned is frame
    assert returned.to_dict("list") == {"orf": ["YDL088C"], "gene": ["ASM4"]}


def _row(means: list[float], sds: list[float]) -> list[tuple[float, float]]:
    """A Table 3 row in column order OD, acetate, citrate, malate, phosphate, pyruvate,
    succinate.
    """
    return list(zip(means, sds, strict=True))


_OTHER = _row(
    [3.0, 2.0, 0.2, 0.3, 0.4, 0.5, 0.6], [0.1, 0.2, 0.02, 0.03, 0.04, 0.05, 0.06]
)


@pytest.mark.parametrize(
    ("table", "aliases", "message"),
    [
        (
            {"ASM4": "ASM4", "YDL088C": "ASM4"},
            {},
            "Yoshida2012: Table 3 rows 'ASM4' and 'YDL088C' both resolve to YDL088C",
        ),
        (
            {"yal999w": "OTHER"},
            {},
            "Yoshida2012: systematic name 'YAL999W' is not a gene of the genome",
        ),
        (
            {"AMB": "OTHER"},
            {"AMB": ["YBR001C", "YCR001W"]},
            "Yoshida2012: gene name 'AMB' is an alias of 2 ORFs ['YBR001C', 'YCR001W']",
        ),
        (
            {"ASM4": "NAN_CITRATE"},
            {},
            "Yoshida2012: Table 3 row 'ASM4' has a NaN citrate cell (nan, nan)",
        ),
    ],
    ids=["duplicate_orf", "orf_not_in_genome", "ambiguous_alias", "nan_cell"],
)
def test_a_bad_table_row_is_refused_before_anything_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s_id_calls: list[dict[str, str]],
    table: dict[str, str],
    aliases: dict[str, list[str]],
    message: str,
) -> None:
    """Contract (issue #537): beside the released WT row, a table with ASM4 twice (as
    ``ASM4`` and as its ORF ``YDL088C``), a systematic-shaped ``yal999w`` the genome does
    not have, an alias ``AMB`` of two ORFs, or a citrate (nan, nan) cell is refused with
    a named message. Nothing is written: no ``data.csv``, no ``processed/lmdb``, and a
    second constructor on the same root refuses again.
    """
    asm4 = m.TABLE_3["ASM4"]
    rows = {
        "ASM4": asm4,
        "OTHER": _OTHER,
        "NAN_CITRATE": [asm4[0], asm4[1], (math.nan, math.nan), *asm4[3:]],
    }
    monkeypatch.setattr(
        m,
        "TABLE_3",
        {"WT": m.TABLE_3["WT"], **{key: rows[row] for key, row in table.items()}},
    )
    genome = _StubGenome(_ORF_BY_NAME)
    genome.alias_to_systematic.update(aliases)
    root = _root(tmp_path, "refused")
    for _ in range(2):
        with pytest.raises(RuntimeError) as info:
            m.OrganicAcidYoshida2012Dataset(
                root=str(root), genome=cast(SCerevisiaeGenome, genome)
            )
        assert str(info.value) == message
        assert not (root / "preprocess" / "data.csv").exists()
        assert not (root / "processed" / "lmdb").exists()


def test_negative_sd_is_refused_by_the_phenotype_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s_id_calls: list[dict[str, str]]
) -> None:
    """A negative SD becomes a negative SE (-0.3 / sqrt(3)) and the phenotype validator
    refuses it with "SE for acetate must be non-negative". Contract (issue #537): the
    refusal comes from ``create_experiment``, which runs for every record before
    ``data.csv`` or the store is written, so neither exists afterwards and a second
    constructor on the same root refuses with the same message instead of serving 0
    records.
    """
    wt = m.TABLE_3["WT"]
    bad = [wt[0], (4.0, -0.3), *wt[2:]]
    monkeypatch.setattr(m, "TABLE_3", {"WT": wt, "ASM4": bad})
    root = _root(tmp_path, "negative")
    for _ in range(2):
        with pytest.raises(pydantic.ValidationError) as info:
            m.OrganicAcidYoshida2012Dataset(root=str(root), genome=_genome())
        assert [e["msg"] for e in info.value.errors()] == [
            "Value error, SE for acetate must be non-negative"
        ]
        assert not (root / "preprocess" / "data.csv").exists()
        assert not (root / "processed" / "lmdb").exists()


def test_download_copies_a_verified_mirror_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    s_id_calls: list[dict[str, str]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With the pin set to the synthetic PDF's digest, an empty root copies the mirror
    PDF byte for byte, logs the staged path and byte count, and builds 17 records.
    """
    content = b"%PDF-1.4 synthetic mirror copy"
    data_root = tmp_path / "data_root"
    mirror = data_root / "torchcell-library" / m.LIBRARY_CITATION_KEY / m.PDF_FILENAME
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(content)
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setattr(m, "PDF_SHA256", hashlib.sha256(content).hexdigest())
    with caplog.at_level(logging.INFO, logger=m.log.name):
        dataset = m.OrganicAcidYoshida2012Dataset(
            root=str(tmp_path / "fresh"), genome=_genome()
        )
    dest = tmp_path / "fresh" / "raw" / m.PDF_FILENAME
    assert dest.read_bytes() == content
    staged = [
        r.getMessage()
        for r in caplog.records
        if r.name == m.log.name and r.getMessage().startswith("Staged")
    ]
    assert staged == [f"Staged {dest} ({len(content)} bytes, sha256 verified)"]
    assert len(dataset) == 17


def test_main_builds_genome_and_dataset_under_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` builds the genome from ``$DATA_ROOT/data/sgd/genome`` and ``data/go`` with
    ``overwrite=False``, then the dataset at
    ``$DATA_ROOT/data/torchcell/organic_acid_yoshida2012`` with that genome, and prints
    its length and first item. Both classes are recorders.
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
            return 17

        def __getitem__(self, index: int) -> str:
            return f"item[{index}]"

    monkeypatch.setattr(m, "SCerevisiaeGenome", _Genome)
    monkeypatch.setattr(m, "OrganicAcidYoshida2012Dataset", _Dataset)
    m.main()
    assert [name for name, _ in calls] == ["genome", "dataset"]
    assert calls[0][1] == {
        "genome_root": f"{tmp_path}/data/sgd/genome",
        "go_root": f"{tmp_path}/data/go",
        "overwrite": False,
    }
    assert list(calls[1][1]) == ["root", "genome"]
    assert calls[1][1]["root"] == f"{tmp_path}/data/torchcell/organic_acid_yoshida2012"
    assert type(calls[1][1]["genome"]) is _Genome
    assert capsys.readouterr().out == "len = 17\nitem[0]\n"
