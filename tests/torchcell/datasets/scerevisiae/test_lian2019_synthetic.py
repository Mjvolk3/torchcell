# tests/torchcell/datasets/scerevisiae/test_lian2019_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_lian2019_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_lian2019_synthetic.py
"""Hermetic end-to-end build of ``CrisprMagicLian2019Dataset`` from a TSV and a design xlsx.

``guide_enrichment_final.tsv`` (``gene, mod, spacer, is_control, corrupted_gene,
r{1,2,3}_log2fc_mean, r{1,2,3}_log2fc_sd``) and Supplementary Data 3
(``41467_2019_13621_MOESM5_ESM.xlsx``, one ``Sequence`` row per CRISPRd row of the TSV,
in order) are written under ``<root>/raw/`` so PyG never calls ``download()``;
``process()`` runs once against a duck-typed genome stub carrying ``gene_set``,
``feature_index["standard_to_ids"]`` and ``resolve_gene_name``. The stub is the
authority for every mapping: PDR1 = YGL013C, SIZ1 = YDR409W, SAP30 = YMR263W,
NAT1 = YDL040C.

TSV rows, in file order (means/SDs as r1, r2, r3):

===  =============  ===  ===============  =====  ========  ============  ============  ============
row  gene           mod  spacer           ctrl   corrupt   r1            r2            r3
===  =============  ===  ===============  =====  ========  ============  ============  ============
0    PDR1           i    20 nt            no     no        0.5 / 0.1     1.2 / 0.2     2.5 / 0.3
1    SIZ1           i    20 nt            no     no        1.0 / 0.1     0.4 / 0.05    0.2 / 0.01
2    sap30          d    44 nt (lower)    no     no        0.9 / inf     blank         -0.3 / blank
3    ctrl_random_1  a    23 nt            yes    no        0.0 / 0.1     0.0 / 0.1     0.0 / 0.1
4    2-Sep          a    23 nt            no     yes       0.1 / 0.1     0.1 / 0.1     0.1 / 0.1
5    RDN25-1        a    23 nt            no     no        0.2 / 0.1     0.2 / 0.1     0.2 / 0.1
6    NAT1           a    23 nt            no     no        0.1 / 0.02    0.7 / 0.1     blank
===  =============  ===  ===============  =====  ========  ============  ============  ============

Source records = 7 rows x 3 rounds = 21. Kept, in write order (row-major, round-minor):
PDR1 r1/r2/r3 (records 0-2), SIZ1 r1 (3), SAP30 r1 and r3 (4, 5), NAT1 r1 and r2 (6, 7)
= 8. Dropped 13 = control 1 x 3 + corrupted 1 x 3 + unresolved 1 x 3 (RDN25-1) + no
enrichment value 2 (SAP30 r2, NAT1 r3) + own-background 2 (SIZ1 r2 and r3: SIZ1 is the
integrated background of rounds 2 and 3).

The CRISPRd cassette is 121 nt: the upper-cased 44 nt barcode + 56 nt of donor tail +
a 21 nt spacer; the stored guide is the last 21 nt and the donor the first 100 nt. An
``inf`` or blank SD stores ``environment_response_uncertainty`` None with no type, so no
SE; a finite SD is ``sample_sd`` with n = 3, SE = SD / sqrt(3) (0.1 -> 0.057735...).
Round 2 records carry the SIZ1 CRISPRi background, round 3 SIZ1 CRISPRi + NAT1 CRISPRa,
each with no guide; the host is the one typed bAID ``StrainBackground`` on every round's
reference. Furfural 5 / 10 / 15 mM by round in SED-URA/G418 at 30 C, 50 mL in a shaken
250 mL baffled flask.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import lmdb
import pandas as pd
import pytest

from torchcell.data import RawSha256MismatchError
from torchcell.datamodels.compound_identity import resolved_compound
from torchcell.datamodels.media import SED_URA_G418
from torchcell.datamodels.schema import (
    AssayType,
    Concentration,
    ConcentrationUnit,
    CrisprActivationPerturbation,
    CrisprConstruct,
    CrisprDeletionPerturbation,
    CrisprInterferencePerturbation,
    CultureEnvironment,
    CultureFormat,
    EnvironmentResponsePhenotype,
    Genotype,
    MeasurementType,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    StrainReferenceGenome,
    Temperature,
    UncertaintyType,
)
from torchcell.datamodels.strain_background import baid_background
from torchcell.datasets.scerevisiae import lian2019 as ln
from torchcell.literature.manifest import (
    ROLE_RAW_DATA,
    ROLE_SI_DATA,
    ArtifactRecord,
    Manifest,
    ProcessingRecord,
    RetrievalMethod,
    RetrievalRecord,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome

_GENES = {"YGL013C", "YDR409W", "YMR263W", "YDL040C"}
_STANDARD = {
    "PDR1": ["YGL013C"],
    "SIZ1": ["YDR409W"],
    "SAP30": ["YMR263W"],
    "NAT1": ["YDL040C"],
}


class _Resolution:
    """The two attributes and the one property the loader reads from a resolution."""

    def __init__(self, status: str, systematic: str | None) -> None:
        self.status = status
        self.systematic_name = systematic

    @property
    def is_current_gene(self) -> bool:
        return self.status in ("current", "renamed")


class _StubGenome:
    """``gene_set``, ``feature_index`` and the resolver: all the genome the loader reads."""

    gene_set = _GENES
    feature_index = {"standard_to_ids": _STANDARD}

    def resolve_gene_name(self, name: str) -> _Resolution:
        upper = name.strip().upper()
        if upper in _GENES:
            return _Resolution("current", upper)
        if upper in _STANDARD:
            return _Resolution("renamed", _STANDARD[upper][0])
        return _Resolution("retired", upper)


def _genome() -> SCerevisiaeGenome:
    return cast(SCerevisiaeGenome, _StubGenome())


_BARCODE = "acgtacgtacgtacgtacgtacgtacgtacgtacgtacgtacgt"
_DONOR_TAIL = "GGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCCAAAATTTTGGGGCCCC"
_D_SPACER = "TTTTTTTTTTTTTTTTTTTTT"
_CASSETTE = _BARCODE.upper() + _DONOR_TAIL + _D_SPACER
_PDR1_SPACER = "AAAACCCCGGGGTTTTAAAA"
_SIZ1_SPACER = "CCCCGGGGTTTTAAAACCCC"
_NAT1_SPACER = "CCCCAAAACCCCAAAACCCCAAA"

_COLUMNS = [
    "gene",
    "mod",
    "spacer",
    "is_control",
    "corrupted_gene",
    "r1_log2fc_mean",
    "r1_log2fc_sd",
    "r2_log2fc_mean",
    "r2_log2fc_sd",
    "r3_log2fc_mean",
    "r3_log2fc_sd",
]
_ROWS: list[list[Any]] = [
    ["PDR1", "i", _PDR1_SPACER, False, False, 0.5, 0.1, 1.2, 0.2, 2.5, 0.3],
    ["SIZ1", "i", _SIZ1_SPACER, False, False, 1.0, 0.1, 0.4, 0.05, 0.2, 0.01],
    ["sap30", "d", _BARCODE, False, False, 0.9, "inf", "", "", -0.3, ""],
    [
        "ctrl_random_1",
        "a",
        "GGGGTTTTAAAACCCCGGGGTTT",
        True,
        False,
        0.0,
        0.1,
        0.0,
        0.1,
        0.0,
        0.1,
    ],
    [
        "2-Sep",
        "a",
        "TTTTAAAACCCCGGGGTTTTAAA",
        False,
        True,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    ],
    [
        "RDN25-1",
        "a",
        "AAAATTTTAAAATTTTAAAATTT",
        False,
        False,
        0.2,
        0.1,
        0.2,
        0.1,
        0.2,
        0.1,
    ],
    ["NAT1", "a", _NAT1_SPACER, False, False, 0.1, 0.02, 0.7, 0.1, "", ""],
]


def _write_raw(raw: Path) -> None:
    """Write the enrichment TSV and the one-row CRISPRd design workbook."""
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_ROWS, columns=_COLUMNS).to_csv(
        raw / ln.TSV_FILENAME, sep="\t", index=False
    )
    pd.DataFrame({"Gene": ["SAP30"], "Sequence": [_CASSETTE]}).to_excel(
        raw / ln.DESIGN_D_FILENAME, index=False
    )


def _root(tmp_path: Path, slug: str = "crispr_magic_lian2019") -> Path:
    root = tmp_path / slug
    _write_raw(root / "raw")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> ln.CrisprMagicLian2019Dataset:
    return ln.CrisprMagicLian2019Dataset(root=str(_root(tmp_path)), genome=_genome())


_PUBLICATION = Publication(
    pubmed_id="31857575",
    pubmed_url="https://pubmed.ncbi.nlm.nih.gov/31857575/",
    doi="10.1038/s41467-019-13621-4",
    doi_url="https://doi.org/10.1038/s41467-019-13621-4",
)


def _environment(furfural_mm: float) -> CultureEnvironment:
    return CultureEnvironment(
        media=SED_URA_G418,
        temperature=Temperature(value=30.0),
        perturbations=[
            SmallMoleculePerturbation(
                compound=resolved_compound("furfural"),
                concentration=Concentration(
                    value=furfural_mm, unit=ConcentrationUnit.millimolar
                ),
            )
        ],
        aerobicity="aerobic",
        culture_format=CultureFormat(
            vessel="250 mL baffled flask",
            working_volume_ul=50000.0,
            shaking_rpm=250.0,
            provenance=[ln.SCREEN_VESSEL, ln.SHAKING_RPM, ln.HARVEST],
            provenance_gaps=[ln.ENDPOINT_GAP, ln.INOCULUM_GAP],
        ),
        provenance_gaps=[
            *ln.DURATION_GAPS,
            ln.PRE_CULTURE_GAP,
            ln.AUXOTROPH_SUPPLEMENT_GAP,
        ],
    )


_ENVIRONMENT = {1: _environment(5.0), 2: _environment(10.0), 3: _environment(15.0)}
_SIZ1_BACKGROUND = CrisprInterferencePerturbation(
    systematic_gene_name="YDR409W",
    perturbed_gene_name="SIZ1",
    crispr=CrisprConstruct(
        effector="dSpCas9-RD1152", guide_sequence=None, n_guides=None
    ),
)
_NAT1_BACKGROUND = CrisprActivationPerturbation(
    systematic_gene_name="YDL040C",
    perturbed_gene_name="NAT1",
    crispr=CrisprConstruct(effector="dLbCas12a-VP", guide_sequence=None, n_guides=None),
)
_BACKGROUND: dict[int, list[Any]] = {
    1: [],
    2: [_SIZ1_BACKGROUND],
    3: [_SIZ1_BACKGROUND, _NAT1_BACKGROUND],
}


def _phenotype(mean: float, sd: float | None) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=mean,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=(
            UncertaintyType.sample_sd if sd is not None else None
        ),
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units=ln.UNITS,
    )


def _experiment(
    foreground: Any, rnd: int, mean: float, sd: float | None
) -> StrainEnvironmentResponseExperiment:
    """The record: the screened guide plus the round's accumulated integrated edits.

    The round's edits stay in the genotype, so a round-2/3 record is the 2-/3-
    perturbation strain actually in the tube; the host is the typed bAID background on
    the reference (``_reference``), which every round shares.
    """
    return StrainEnvironmentResponseExperiment(
        dataset_name="CrisprMagicLian2019Dataset",
        genotype=Genotype(perturbations=[foreground, *_BACKGROUND[rnd]]),
        environment=_ENVIRONMENT[rnd],
        phenotype=_phenotype(mean, sd),
    )


def _reference(rnd: int) -> StrainEnvironmentResponseExperimentReference:
    return StrainEnvironmentResponseExperimentReference(
        dataset_name="CrisprMagicLian2019Dataset",
        genome_reference=StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="bAID",
            ploidy="haploid",
            background=baid_background(),
        ),
        environment_reference=_ENVIRONMENT[rnd],
        phenotype_reference=EnvironmentResponsePhenotype(
            measurement_type=MeasurementType.log2_ratio,
            assay_type=AssayType.pooled_competitive_growth_barcode,
            environment_response=0.0,
            n_samples=3,
            sample_unit=SampleUnit.biological_replicate,
            units=ln.UNITS,
        ),
    )


_PDR1 = CrisprInterferencePerturbation(
    systematic_gene_name="YGL013C",
    perturbed_gene_name="PDR1",
    crispr=CrisprConstruct(
        effector="dSpCas9-RD1152", guide_sequence=_PDR1_SPACER, n_guides=1
    ),
)
_SIZ1 = CrisprInterferencePerturbation(
    systematic_gene_name="YDR409W",
    perturbed_gene_name="SIZ1",
    crispr=CrisprConstruct(
        effector="dSpCas9-RD1152", guide_sequence=_SIZ1_SPACER, n_guides=1
    ),
)
_SAP30 = CrisprDeletionPerturbation(
    systematic_gene_name="YMR263W",
    perturbed_gene_name="SAP30",
    crispr=CrisprConstruct(effector="SaCas9", guide_sequence=_D_SPACER, n_guides=1),
    donor_sequence=_BARCODE.upper() + _DONOR_TAIL,
)
_NAT1 = CrisprActivationPerturbation(
    systematic_gene_name="YDL040C",
    perturbed_gene_name="NAT1",
    crispr=CrisprConstruct(
        effector="dLbCas12a-VP", guide_sequence=_NAT1_SPACER, n_guides=1
    ),
)
_EXPECTED = [
    _experiment(_PDR1, 1, 0.5, 0.1),
    _experiment(_PDR1, 2, 1.2, 0.2),
    _experiment(_PDR1, 3, 2.5, 0.3),
    _experiment(_SIZ1, 1, 1.0, 0.1),
    _experiment(_SAP30, 1, 0.9, None),
    _experiment(_SAP30, 3, -0.3, None),
    _experiment(_NAT1, 1, 0.1, 0.02),
    _experiment(_NAT1, 2, 0.7, 0.1),
]
_EXPECTED_ROUNDS = [1, 2, 3, 1, 1, 3, 1, 2]


def _interned_entries(root: Path) -> int:
    env = lmdb.open(str(root / "processed" / "interned"), readonly=True, lock=False)
    with env.begin() as txn:
        entries: int = txn.stat()["entries"]
    env.close()
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_writes_eight_guide_round_records_with_round_backgrounds(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """The eight kept (guide, round) cells, in row-major then round order, each equal to
    a hand-built ``StrainEnvironmentResponseExperiment``: the modality's effector and leaf
    class (i -> ``dSpCas9-RD1152`` CRISPRi, a -> ``dLbCas12a-VP`` CRISPRa, d -> ``SaCas9``
    CRISPRd), the round's integrated background with no guide, furfural 5/10/15 mM, and
    the SE derived from a finite SD with n = 3 (0.1 / sqrt(3) = 0.057735...). The CRISPRd
    row stores the last 21 nt of the 121 nt cassette as the guide and the first 100 nt
    (the upper-cased barcode + 56 nt tail) as the donor; its ``inf`` and blank SDs store
    no uncertainty, no type and no SE. ``sap30`` is written lowercase in the TSV and comes
    back as the genome's ``SAP30``.
    """
    assert len(dataset) == 8
    assert [dataset[i]["experiment"] for i in range(8)] == [
        e.model_dump() for e in _EXPECTED
    ]
    assert [
        dataset[i]["experiment"]["phenotype"]["environment_response_se"]
        for i in range(8)
    ] == [
        0.1 / math.sqrt(3),
        0.2 / math.sqrt(3),
        0.3 / math.sqrt(3),
        0.1 / math.sqrt(3),
        None,
        None,
        0.02 / math.sqrt(3),
        0.1 / math.sqrt(3),
    ]
    deletion = dataset[4]["experiment"]["genotype"]["perturbations"][0]
    assert deletion["perturbation_type"] == "crispr_deletion"
    assert deletion["crispr"]["guide_sequence"] == _D_SPACER
    assert deletion["donor_sequence"] == _CASSETTE[:100]
    assert len(deletion["donor_sequence"]) == 100
    # 1 / 2 / 3 perturbations by round: the round's accumulated integrated edits stay
    # in the genotype beside the screened guide.
    assert [
        len(dataset[i]["experiment"]["genotype"]["perturbations"]) for i in range(8)
    ] == [1, 2, 3, 1, 1, 3, 1, 2]
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()


def test_drop_log_counts_guides_times_three_and_nan_before_own_background(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """Finding: the empty-value test (lian2019.py 777) runs before the own-background test
    (780), so NAT1's blank round-3 cell is counted under
    ``guide_round_has_no_enrichment_value`` although NAT1 is also round 3's integrated
    background; the own-background rule therefore counts 2 (SIZ1 r2, SIZ1 r3), not 3.

    Guide-scoped rules multiply by the three rounds whatever the rows hold: control 3,
    corrupted 3, unresolved 3 (item RDN25-1). 21 source, 8 kept, 13 dropped.
    """
    drop_log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert drop_log == {
        "dataset": "CrisprMagicLian2019Dataset",
        "source_records": 21,
        "kept_records": 8,
        "dropped_records": 13,
        "rules": [
            {
                "rule": "random_negative_control_guide",
                "scope": "guide",
                "description": (
                    "one of the 300 random negative-control guides (100 per library); "
                    "it targets no gene, so there is no genotype to key a record to"
                ),
                "n_records": 3,
                "items": [],
            },
            {
                "rule": "source_corrupted_gene_name",
                "scope": "guide",
                "description": (
                    "the gene name is an Excel date/serial artifact present IDENTICALLY "
                    "in the released reference and the design library, so the target is "
                    "unrecoverable from our inputs"
                ),
                "n_records": 3,
                "items": [],
            },
            {
                "rule": "target_gene_is_not_a_current_genome_gene",
                "scope": "guide",
                "description": (
                    "the guide targets an ncRNA / rDNA feature or a name absent from the "
                    "current R64 ORF genome, so no gene entity exists to key the record "
                    "to"
                ),
                "n_records": 3,
                "items": ["RDN25-1"],
            },
            {
                "rule": "guide_round_has_no_enrichment_value",
                "scope": "guide_round",
                "description": (
                    "the guide was not detected in this round's before/after libraries, "
                    "so the round has no log2 enrichment for it"
                ),
                "n_records": 2,
                "items": [],
            },
            {
                "rule": "guide_targets_its_own_round_background",
                "scope": "guide_round",
                "description": (
                    "in this round the guide's target gene is already an integrated "
                    "background perturbation, so the foreground edit is redundant and "
                    "the strain signature would collapse to empty once the background "
                    "is subtracted"
                ),
                "n_records": 2,
                "items": [],
            },
        ],
    }
    assert drop_log == ln.DropLog.model_validate(drop_log).model_dump()


def test_one_reference_per_round_gene_set_manifest_and_interned_store(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """Three references in first-sighting order (rounds 1, 2, 3) with member indices
    [0, 3, 4, 6], [1, 7], [2, 5], each the typed bAID host with the round's furfural
    environment and a zero log2FC baseline at n = 3. ``gene_set.json`` is the four ORFs
    sorted (the round backgrounds contribute SIZ1 and NAT1 as well); the build manifest
    names the slug, class, module, host and HEAD; ``interned`` holds 6 entries (three
    environments + three references) while the publication stays inline.
    """
    index = json.loads(
        (
            Path(dataset.root) / "preprocess" / "experiment_reference_index.json"
        ).read_text()
    )
    assert [entry["member_indices"] for entry in index] == [
        [0, 3, 4, 6],
        [1, 7],
        [2, 5],
    ]
    assert [entry["reference"] for entry in index] == [
        _reference(rnd).model_dump() for rnd in (1, 2, 3)
    ]
    assert [dataset[i]["reference"] for i in range(8)] == [
        _reference(rnd).model_dump() for rnd in _EXPECTED_ROUNDS
    ]
    preprocess = Path(dataset.root) / "preprocess"
    assert json.loads((preprocess / "gene_set.json").read_text()) == [
        "YDL040C",
        "YDR409W",
        "YGL013C",
        "YMR263W",
    ]
    assert dataset.gene_set == _GENES
    assert dataset.experiment_class is StrainEnvironmentResponseExperiment
    assert dataset.reference_class is StrainEnvironmentResponseExperimentReference
    manifest = json.loads((preprocess / "build_manifest.json").read_text())
    head = subprocess.run(
        ["git", "-C", str(Path(ln.__file__).parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert manifest["manifest_schema_version"] == 1
    assert manifest["dataset_name"] == "crispr_magic_lian2019"
    assert manifest["loader_class"] == "CrisprMagicLian2019Dataset"
    assert manifest["loader_module"] == "torchcell.datasets.scerevisiae.lian2019"
    assert manifest["hostname"] == socket.gethostname()
    assert manifest["torchcell_commit"] == head
    assert {
        "CrisprActivationPerturbation",
        "CrisprInterferencePerturbation",
        "CrisprDeletionPerturbation",
    } <= set(manifest["closure"])
    assert _interned_entries(Path(dataset.root)) == 6


def test_store_reopens_without_a_genome_and_serves_the_getters(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """With ``processed/lmdb`` present a second instance on the same root skips both
    ``download()`` and ``process()`` (no genome, raw files deleted): eight equal records,
    ``get([2, 5])`` as a list, ``dataset[[2, 5]]`` as PyG's two-item subset,
    ``transform_item`` rebuilding the typed round-3 CRISPRd record, ``repr``
    ``CrisprMagicLian2019Dataset(8)``, and the inline hooks.
    """
    records = [dataset[i] for i in range(8)]
    dataset.close_lmdb()
    for name in (ln.TSV_FILENAME, ln.DESIGN_D_FILENAME):
        (Path(dataset.root) / "raw" / name).unlink()
    reopened = ln.CrisprMagicLian2019Dataset(root=dataset.root, genome=None)
    assert len(reopened) == 8
    assert [reopened[i] for i in range(8)] == records
    assert reopened.get([2, 5]) == [records[2], records[5]]
    subset = reopened[[2, 5]]
    assert len(subset) == 2
    assert [subset[0], subset[1]] == [records[2], records[5]]
    # PyG's list index is a shallow copy that opens its own handle on the same store;
    # py-lmdb refuses a second open of one path in a process, so close it first.
    subset.close_lmdb()
    assert reopened.transform_item(reopened[5]) == {
        "experiment": _EXPECTED[5],
        "reference": _reference(3),
        "publication": _PUBLICATION,
    }
    assert repr(reopened) == "CrisprMagicLian2019Dataset(8)"
    assert reopened.gene_set == _GENES
    with pytest.raises(NotImplementedError):
        reopened.create_experiment()
    sentinel = object()
    assert reopened.preprocess_raw(sentinel) is sentinel
    reopened.close_lmdb()


def test_genome_is_required_before_the_store_is_opened(tmp_path: Path) -> None:
    """With the raw files present and ``genome=None`` the resolver (line 696) raises the
    loader's message before ``_open_write_lmdb`` (752), so no ``processed/lmdb`` is left
    behind and a retry with a genome builds the eight records.
    """
    root = _root(tmp_path)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "CrisprMagicLian2019Dataset requires a genome; inject SCerevisiaeGenome(...)"
        ),
    ):
        ln.CrisprMagicLian2019Dataset(root=str(root), genome=None)
    assert not (root / "processed" / "lmdb").exists()
    retry = ln.CrisprMagicLian2019Dataset(root=str(root), genome=_genome())
    assert len(retry) == 8
    retry.close_lmdb()


def _staged_raw(tmp_path: Path, name: str) -> tuple[Path, str, str]:
    """Write the fixture raw files to ``<tmp>/<name>``; return (dir, tsv sha, design sha)."""
    staging = tmp_path / name
    _write_raw(staging)
    return (
        staging,
        _sha256(staging / ln.TSV_FILENAME),
        _sha256(staging / ln.DESIGN_D_FILENAME),
    )


def _expected_manifest(
    tsv_sha: str, tsv_bytes: int, design_sha: str, design_bytes: int
) -> Manifest:
    design_url = f"{ln._ESM_BASE}{ln.DESIGN_D_FILENAME}"
    reference_url = f"{ln._ESM_BASE}{ln.REFERENCE_FILENAME}"
    return Manifest(
        citation_key=ln.CITATION_KEY,
        doi=ln.DOI,
        title=(
            "Multi-functional genome-wide CRISPR system for high throughput "
            "genotype-phenotype mapping"
        ),
        files=[
            ArtifactRecord(
                path=ln.TSV_REL,
                role=ROLE_RAW_DATA,
                bytes=tsv_bytes,
                sha256=tsv_sha,
                source="derived: SRA PRJNA504483 reprocessing",
                processing=ProcessingRecord(
                    processor="experiments/016-lian-magic-reprocess/scripts/reproduce.sh",
                    tool="torchcell lian-magic-reprocess pipeline",
                    version="2026-07-13",
                    params={
                        "sra_project": "PRJNA504483",
                        "n_runs": 21,
                        "barcode_window": "read[27:70] (43 bp activation) | read[27:71] "
                        "(44 bp interference/deletion), forward, exact match",
                        "normalization": "CPM(+1) per library; per round per replicate "
                        "log2(furfural-after / untreated-before); mean +- SD over "
                        "triplicates",
                        "validation": "PDR1i round-3 rank 1, SLX5i round-1 rank 1, SAP30d "
                        "round-1 rank 2 against the paper's reported hits",
                        "reference_url": reference_url,
                    },
                    input_sha256=[ln.REFERENCE_SHA256],
                ),
            ),
            ArtifactRecord(
                path=ln.DESIGN_D_REL,
                role=ROLE_SI_DATA,
                bytes=design_bytes,
                sha256=design_sha,
                source=design_url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=design_url,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": design_url},
                    sha256=design_sha,
                    retrieved_at="2026-09-12",
                ),
            ),
        ],
        si_data_sources=[
            design_url,
            reference_url,
            "https://www.ncbi.nlm.nih.gov/bioproject/PRJNA504483/",
        ],
        si_expected=[
            "Supplementary Data 3 (CRISPRd design library) -- consumed here for the "
            "guide/donor split; its Sequence column is element-for-element identical to "
            "the archived lab design file, and the same holds for Supplementary Data 1 "
            "and 2 against the CRISPRa and CRISPRi lab files",
            "Supplementary Data 4 (100,493-guide reference) -- an INPUT to the derived "
            "enrichment table, recorded in its processing record rather than mirrored "
            "here, since the loader does not read it",
            "the per-guide furfural enrichment itself was NEVER released; it is "
            "reprocessed from SRA PRJNA504483",
        ],
        provenance_complete=True,
    )


def test_deposit_raw_mirror_records_the_derived_table_and_the_esm_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit_raw_mirror`` refuses a source whose sha256 is not the pinned one. With
    the pins patched to the fixture digests it copies the TSV to ``data/`` and the design
    workbook to ``si/si_data/`` and writes a manifest whose first record is a derived
    ``raw_data`` artifact with a ``ProcessingRecord`` (the 2026-07-13 reprocessing
    pipeline, the Supplementary Data 4 sha256 as its input) and no retrieval, and whose
    second is a ``springer_esm`` retrieval; ``provenance_complete`` is True and
    ``created_at`` is UTC. A repeat deposit is a no-op and a different source under a
    re-pinned digest is refused because the destination holds other bytes.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    staging, tsv_sha, design_sha = _staged_raw(tmp_path, "staging")
    tsv_path = staging / ln.TSV_FILENAME
    design_path = staging / ln.DESIGN_D_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        ln.deposit_raw_mirror(enrichment_path=tsv_path, design_d_path=design_path)
    assert str(err.value) == (
        f"sha256 mismatch for {tsv_path}: expected {ln.TSV_SHA256}, observed {tsv_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()
    # A second source off its pin is caught before the first is written either.
    monkeypatch.setattr(ln, "TSV_SHA256", tsv_sha)
    with pytest.raises(RawSha256MismatchError) as err:
        ln.deposit_raw_mirror(enrichment_path=tsv_path, design_d_path=design_path)
    assert str(err.value) == (
        f"sha256 mismatch for {design_path}: expected {ln.DESIGN_D_SHA256}, "
        f"observed {design_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()

    monkeypatch.setattr(ln, "TSV_SHA256", tsv_sha)
    monkeypatch.setattr(ln, "DESIGN_D_SHA256", design_sha)
    mirror = ln.deposit_raw_mirror(enrichment_path=tsv_path, design_d_path=design_path)
    assert (
        mirror == data_root / "torchcell-raw" / ln.CITATION_KEY == ln.raw_mirror_dir()
    )
    assert (mirror / ln.TSV_REL).read_bytes() == tsv_path.read_bytes()
    assert (mirror / ln.DESIGN_D_REL).read_bytes() == design_path.read_bytes()
    manifest = ln.load_manifest()
    expected = _expected_manifest(
        tsv_sha, tsv_path.stat().st_size, design_sha, design_path.stat().st_size
    )
    assert manifest.model_dump(exclude={"created_at"}) == expected.model_dump(
        exclude={"created_at"}
    )
    assert manifest.files[0].retrieval is None
    assert manifest.files[1].processing is None
    assert manifest.created_at is not None
    assert datetime.fromisoformat(manifest.created_at).tzinfo == UTC
    assert ln.manifest_sha256(manifest, ln.TSV_REL) == tsv_sha
    assert ln.manifest_sha256(manifest, ln.DESIGN_D_REL) == design_sha
    with pytest.raises(KeyError, match="nope is not in the raw-mirror manifest"):
        ln.manifest_sha256(manifest, "nope")

    written = (mirror / "manifest.json").read_text()
    assert ln.deposit_raw_mirror(
        enrichment_path=tsv_path, design_d_path=design_path
    ) == (mirror)
    assert ln.load_manifest().files == Manifest.model_validate_json(written).files

    other = tmp_path / "other"
    other.mkdir()
    (other / ln.TSV_FILENAME).write_bytes(b"a different table")
    monkeypatch.setattr(ln, "TSV_SHA256", _sha256(other / ln.TSV_FILENAME))
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{mirror / ln.TSV_REL} exists with a different sha256; refusing"
        ),
    ):
        ln.deposit_raw_mirror(
            enrichment_path=other / ln.TSV_FILENAME, design_d_path=design_path
        )
    assert (mirror / ln.TSV_REL).read_bytes() == tsv_path.read_bytes()


def test_download_links_the_verified_mirror_and_refuses_drift_or_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw files ``download()`` needs the mirror manifest (``FileNotFoundError``
    naming it). After the fixture files are deposited, a root whose ``raw/`` already links
    the TSV keeps that link (``download()`` only creates a missing one), gets the design
    workbook linked beside it and builds the same eight records; a drifted design
    workbook is refused
    naming both digests and a removed one is refused naming its path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    manifest_path = data_root / "torchcell-raw" / ln.CITATION_KEY / "manifest.json"
    with pytest.raises(FileNotFoundError, match=re.escape(str(manifest_path))):
        ln.CrisprMagicLian2019Dataset(root=str(tmp_path / "a"), genome=_genome())

    staging, tsv_sha, design_sha = _staged_raw(tmp_path, "staging")
    monkeypatch.setattr(ln, "TSV_SHA256", tsv_sha)
    monkeypatch.setattr(ln, "DESIGN_D_SHA256", design_sha)
    mirror = ln.deposit_raw_mirror(
        enrichment_path=staging / ln.TSV_FILENAME,
        design_d_path=staging / ln.DESIGN_D_FILENAME,
    )
    root = tmp_path / "b"
    (root / "raw").mkdir(parents=True)
    os.symlink(mirror / ln.TSV_REL, root / "raw" / ln.TSV_FILENAME)
    built = ln.CrisprMagicLian2019Dataset(root=str(root), genome=_genome())
    assert (root / "raw" / ln.TSV_FILENAME).is_symlink()
    assert (root / "raw" / ln.TSV_FILENAME).resolve() == (mirror / ln.TSV_REL).resolve()
    assert (root / "raw" / ln.DESIGN_D_FILENAME).resolve() == (
        mirror / ln.DESIGN_D_REL
    ).resolve()
    assert len(built) == 8
    assert [built[i]["experiment"] for i in range(8)] == [
        e.model_dump() for e in _EXPECTED
    ]
    built.close_lmdb()

    (mirror / ln.DESIGN_D_REL).write_bytes(b"drifted upstream")
    drifted = _sha256(mirror / ln.DESIGN_D_REL)
    with pytest.raises(RawSha256MismatchError) as err:
        ln.CrisprMagicLian2019Dataset(root=str(tmp_path / "c"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / ln.DESIGN_D_REL}: expected {design_sha}, "
        f"observed {drifted}"
    )
    assert not (tmp_path / "c" / "raw" / ln.DESIGN_D_FILENAME).exists()
    (mirror / ln.DESIGN_D_REL).unlink()
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"required raw artifact missing from mirror: {mirror / ln.DESIGN_D_REL}"
        ),
    ):
        ln.CrisprMagicLian2019Dataset(root=str(tmp_path / "d"), genome=_genome())


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #518's sweep): with both files already in ``raw/`` PyG skips
    ``download()``, so ``process()`` verifies them against ``TSV_SHA256`` and
    ``DESIGN_D_SHA256`` first. The TSV off its pin raises ``RawSha256MismatchError``
    naming it and both digests before a row is read; no store is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(ln, [ln.TSV_FILENAME, ln.DESIGN_D_FILENAME])
    raw = staged.root / "raw" / ln.TSV_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        ln.CrisprMagicLian2019Dataset(root=str(staged.root), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "f9af849f97a2d460c3a6d628308491ec3966c6cc2a7f6cad130848d2bad32647, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert _sha256(raw) == staged.observed


# Phase 24: the drop-accounting guard


def test_a_drop_log_that_disagrees_with_its_rules_refuses_after_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``DropLog`` is patched to report one more dropped record than the build lost.

    The rules total 13 (the ``dropped_records`` of
    ``test_drop_log_counts_guides_times_three_and_nan_before_own_background``) while the
    log says 14: the exact mismatch message, and the tampered 14 already on disk.
    """
    real = ln.DropLog

    def inflated(**kwargs: Any) -> Any:
        kwargs["dropped_records"] += 1
        return real(**kwargs)

    monkeypatch.setattr(ln, "DropLog", inflated)
    root = _root(tmp_path)
    with pytest.raises(RuntimeError) as err:
        ln.CrisprMagicLian2019Dataset(root=str(root), genome=_genome())
    assert str(err.value) == (
        "drop accounting mismatch: rules total 13, 14 records missing from the build"
    )
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert log["dropped_records"] == 14
