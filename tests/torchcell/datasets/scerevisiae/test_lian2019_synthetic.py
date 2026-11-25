# tests/torchcell/datasets/scerevisiae/test_lian2019_synthetic.py
# [[tests.torchcell.datasets.scerevisiae.test_lian2019_synthetic]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_lian2019_synthetic.py
"""Hermetic end-to-end build of ``CrisprMagicLian2019Dataset`` from a TSV and a design xlsx.

``guide_enrichment_replicates.tsv`` (``gene, mod, spacer, is_control, corrupted_gene,
r{1,2,3}_rep{1,2,3}_log2fc``) and Supplementary Data 3
(``41467_2019_13621_MOESM5_ESM.xlsx``, one ``Sequence`` row per CRISPRd row of the TSV,
in order) are written under ``<root>/raw/`` so PyG never calls ``download()``;
``process()`` runs once against a duck-typed genome stub carrying ``gene_set``,
``feature_index["standard_to_ids"]`` and ``resolve_gene_name``. The stub is the
authority for every mapping: PDR1 = YGL013C, SIZ1 = YDR409W, SAP30 = YMR263W,
NAT1 = YDL040C.

TSV rows, in file order (each round's three biological replicates):

===  =============  ===  ===============  =====  ========  ===============  ===============  ================
row  gene           mod  spacer           ctrl   corrupt   r1 rep 1/2/3     r2 rep 1/2/3     r3 rep 1/2/3
===  =============  ===  ===============  =====  ========  ===============  ===============  ================
0    PDR1           i    20 nt            no     no        0.4 / 0.5 / 0.6  1.1 / 1.2 / 1.3  2.4 / 2.5 / 2.6
1    SIZ1           i    20 nt            no     no        0.9 / 1.0 / 1.1  0.3 / 0.4 / 0.5  0.1 / 0.2 / 0.3
2    sap30          d    44 nt (lower)    no     no        0.8 / inf / 1.0  blank            -0.3 / -0.4 / blank
3    ctrl_random_1  a    23 nt            yes    no        0.0 each         0.0 each         0.0 each
4    2-Sep          a    23 nt            no     yes       0.1 each         0.1 each         0.1 each
5    RDN25-1        a    23 nt            no     no        0.2 each         0.2 each         0.2 each
6    NAT1           a    23 nt            no     no        0.1 each         0.6 / 0.7 / 0.8  blank
===  =============  ===  ===============  =====  ========  ===============  ===============  ================

Source records = 7 rows x 3 rounds x 3 replicates = 63. Kept, in write order (row, then
round, then replicate): PDR1 all nine (records 0-8), SIZ1 round 1 (9-11), SAP30 round 1
replicates 1 and 3 (12, 13) and round 3 replicates 1 and 2 (14, 15), NAT1 round 1 (16-18)
and round 2 (19-21) = 22. Dropped 41 = control 9 + corrupted 9 + unresolved 9 (RDN25-1)
+ no enrichment value 5 (SAP30's ``inf`` round-1 replicate, its blank round 2, its blank
round-3 replicate) + own-background 9 (SIZ1 rounds 2 and 3, NAT1 round 3: both are
integrated backgrounds of those rounds).

The CRISPRd cassette is 121 nt: the upper-cased 44 nt barcode + 56 nt of donor tail +
a 21 nt spacer; the stored guide is the last 21 nt and the donor the first 100 nt. Every
record is ONE biological replicate: ``n_samples = 1``, no uncertainty, no type and no SE
(three typed ``not_applicable_single_measurement`` gaps), and ``replicate_id`` the
authors' own library pair for that (round, replicate), e.g. ``2Ff1a/2Ff1b``.
Round 2 records carry the SIZ1 CRISPRi background, round 3 SIZ1 CRISPRi + NAT1 CRISPRa,
each with no guide; the host is the one typed bAID ``StrainBackground`` on every round's
reference, whose zero baseline stays the AGGREGATE over the three replicates (n = 3,
``replicate_id`` None). Furfural 5 / 10 / 15 mM by round in SED-URA/G418 at 30 C, 50 mL
in a shaken 250 mL baffled flask.
"""

from __future__ import annotations

import hashlib
import json
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
from torchcell.sequence.genome.registry import (
    SGD_S288C_R64,
    SGD_S288C_R64_GENBANK_ACCESSION,
)
from torchcell.sequence.genome.scerevisiae import SCerevisiaeGenome
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

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
    *(f"r{rnd}_rep{rep}_log2fc" for rnd in (1, 2, 3) for rep in (1, 2, 3)),
]
_ROWS: list[list[Any]] = [
    ["PDR1", "i", _PDR1_SPACER, False, False]
    + [0.4, 0.5, 0.6, 1.1, 1.2, 1.3, 2.4, 2.5, 2.6],
    ["SIZ1", "i", _SIZ1_SPACER, False, False]
    + [0.9, 1.0, 1.1, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3],
    ["sap30", "d", _BARCODE, False, False]
    + [0.8, "inf", 1.0, "", "", "", -0.3, -0.4, ""],
    ["ctrl_random_1", "a", "GGGGTTTTAAAACCCCGGGGTTT", True, False] + [0.0] * 9,
    ["2-Sep", "a", "TTTTAAAACCCCGGGGTTTTAAA", False, True] + [0.1] * 9,
    ["RDN25-1", "a", "AAAATTTTAAAATTTTAAAATTT", False, False] + [0.2] * 9,
    ["NAT1", "a", _NAT1_SPACER, False, False]
    + [0.1, 0.1, 0.1, 0.6, 0.7, 0.8, "", "", ""],
]


def _write_raw(raw: Path) -> None:
    """Write the enrichment TSV and the one-row CRISPRd design workbook."""
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(_ROWS, columns=_COLUMNS).to_csv(
        raw / ln.REPLICATES_TSV_FILENAME, sep="\t", index=False
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


def _phenotype(value: float, rnd: int, replicate: int) -> EnvironmentResponsePhenotype:
    """One replicate's value: n = 1, no dispersion, the authors' library-pair label."""
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=value,
        environment_response_uncertainty=None,
        environment_response_uncertainty_type=None,
        n_samples=1,
        sample_unit=SampleUnit.biological_replicate,
        units=ln.UNITS,
        replicate_id=ln.REPLICATE_LIBRARY[rnd][replicate],
        provenance_gaps=[
            ProvenanceGap(
                field=field,
                reason=ProvenanceGapReason.not_applicable_single_measurement,
                note=ln._SINGLE_REPLICATE_NOTE,
            )
            for field in (
                "environment_response_uncertainty",
                "environment_response_uncertainty_type",
                "environment_response_se",
            )
        ],
    )


def _experiment(
    foreground: Any, rnd: int, replicate: int, value: float
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
        phenotype=_phenotype(value, rnd, replicate),
    )


def _reference(rnd: int) -> StrainEnvironmentResponseExperimentReference:
    return StrainEnvironmentResponseExperimentReference(
        dataset_name="CrisprMagicLian2019Dataset",
        genome_reference=StrainReferenceGenome(
            species="Saccharomyces cerevisiae",
            strain="bAID",
            ploidy="haploid",
            background=baid_background(),
            assembly_set=SGD_S288C_R64,
            assembly_accession=SGD_S288C_R64_GENBANK_ACCESSION,
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
#: ``(foreground, round, replicate, value)`` in the loader's write order.
_EXPECTED_CELLS: list[tuple[Any, int, int, float]] = [
    (_PDR1, 1, 1, 0.4),
    (_PDR1, 1, 2, 0.5),
    (_PDR1, 1, 3, 0.6),
    (_PDR1, 2, 1, 1.1),
    (_PDR1, 2, 2, 1.2),
    (_PDR1, 2, 3, 1.3),
    (_PDR1, 3, 1, 2.4),
    (_PDR1, 3, 2, 2.5),
    (_PDR1, 3, 3, 2.6),
    (_SIZ1, 1, 1, 0.9),
    (_SIZ1, 1, 2, 1.0),
    (_SIZ1, 1, 3, 1.1),
    (_SAP30, 1, 1, 0.8),
    (_SAP30, 1, 3, 1.0),
    (_SAP30, 3, 1, -0.3),
    (_SAP30, 3, 2, -0.4),
    (_NAT1, 1, 1, 0.1),
    (_NAT1, 1, 2, 0.1),
    (_NAT1, 1, 3, 0.1),
    (_NAT1, 2, 1, 0.6),
    (_NAT1, 2, 2, 0.7),
    (_NAT1, 2, 3, 0.8),
]
_EXPECTED = [_experiment(*cell) for cell in _EXPECTED_CELLS]
_N_RECORDS = len(_EXPECTED)
_EXPECTED_ROUNDS = [cell[1] for cell in _EXPECTED_CELLS]


def _interned_entries(root: Path) -> int:
    env = lmdb.open(str(root / "processed" / "interned"), readonly=True, lock=False)
    with env.begin() as txn:
        entries: int = txn.stat()["entries"]
    env.close()
    return entries


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_writes_one_record_per_guide_round_replicate(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """The 22 kept (guide, round, replicate) cells, in row-major then round then replicate
    order, each equal to a hand-built ``StrainEnvironmentResponseExperiment``: the
    modality's effector and leaf class (i -> ``dSpCas9-RD1152`` CRISPRi, a ->
    ``dLbCas12a-VP`` CRISPRa, d -> ``SaCas9`` CRISPRd), the round's integrated background
    with no guide, furfural 5/10/15 mM, and that replicate's value verbatim with
    ``n_samples = 1``, no uncertainty, no type, no derived SE and three typed
    ``not_applicable_single_measurement`` gaps. ``replicate_id`` is the authors' own
    library pair, so the three replicates of one (guide, round) are three distinct
    records. The CRISPRd row stores the last 21 nt of the 121 nt cassette as the guide and
    the first 100 nt (the upper-cased barcode + 56 nt tail) as the donor; ``sap30`` is
    written lowercase in the TSV and comes back as the genome's ``SAP30``.
    """
    assert len(dataset) == _N_RECORDS == 22
    assert [dataset[i]["experiment"] for i in range(_N_RECORDS)] == [
        e.model_dump() for e in _EXPECTED
    ]
    phenotypes = [dataset[i]["experiment"]["phenotype"] for i in range(_N_RECORDS)]
    assert {p["environment_response_se"] for p in phenotypes} == {None}
    assert {p["environment_response_uncertainty"] for p in phenotypes} == {None}
    assert {p["environment_response_uncertainty_type"] for p in phenotypes} == {None}
    assert {p["n_samples"] for p in phenotypes} == {1}
    assert [p["replicate_id"] for p in phenotypes[:3]] == [
        "Ff1a/Ff1b",
        "Ff2a/Ff2b",
        "Ff3a/Ff3b",
    ]
    assert [p["replicate_id"] for p in phenotypes[3:6]] == [
        "2Ff1a/2Ff1b",
        "2Ff2a/2Ff2b",
        "2Ff3a/2Ff3b",
    ]
    assert len({(p["replicate_id"], p["environment_response"]) for p in phenotypes}) > 1
    assert {gap["field"] for gap in phenotypes[0]["provenance_gaps"]} == {
        "environment_response_uncertainty",
        "environment_response_uncertainty_type",
        "environment_response_se",
    }
    assert {gap["reason"] for gap in phenotypes[0]["provenance_gaps"]} == {
        "not_applicable_single_measurement"
    }
    # the three replicates of SAP30 round 1: the inf replicate is the one not stored
    assert [p["environment_response"] for p in phenotypes[12:14]] == [0.8, 1.0]
    assert [p["replicate_id"] for p in phenotypes[12:14]] == ["Ff1a/Ff1b", "Ff3a/Ff3b"]
    deletion = dataset[12]["experiment"]["genotype"]["perturbations"][0]
    assert deletion["perturbation_type"] == "crispr_deletion"
    assert deletion["crispr"]["guide_sequence"] == _D_SPACER
    assert deletion["donor_sequence"] == _CASSETTE[:100]
    assert len(deletion["donor_sequence"]) == 100
    # 1 / 2 / 3 perturbations by round: the round's accumulated integrated edits stay
    # in the genotype beside the screened guide.
    assert [
        len(dataset[i]["experiment"]["genotype"]["perturbations"])
        for i in range(_N_RECORDS)
    ] == [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 2, 2, 2]
    assert dataset[0]["publication"] == _PUBLICATION.model_dump()


def test_the_replicate_mean_reproduces_the_value_the_mean_build_stored(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """Grouping the new records by (genotype, environment) and averaging gives back what
    the previous build stored for that (guide, round): PDR1's rounds average to 0.5, 1.2
    and 2.5, the three numbers the mean table held for those cells. Nothing is
    precomputed for query; the mean is this grouping.
    """
    grouped: dict[tuple[str, float], list[float]] = {}
    for i in range(len(dataset)):
        experiment = dataset[i]["experiment"]
        # the screened guide is the one perturbation carrying a spacer; the round's
        # integrated background edits carry none, and Genotype orders perturbations
        # canonically rather than foreground-first
        guides = [
            p["crispr"]["guide_sequence"]
            for p in experiment["genotype"]["perturbations"]
            if p["crispr"]["guide_sequence"] is not None
        ]
        assert len(guides) == 1
        guide = guides[0]
        furfural = experiment["environment"]["perturbations"][0]["concentration"][
            "value"
        ]
        grouped.setdefault((guide, furfural), []).append(
            experiment["phenotype"]["environment_response"]
        )
    means = {key: sum(v) / len(v) for key, v in grouped.items()}
    assert len(grouped[(_PDR1_SPACER, 5.0)]) == 3
    for furfural, expected in ((5.0, 0.5), (10.0, 1.2), (15.0, 2.5)):
        assert means[(_PDR1_SPACER, furfural)] == pytest.approx(expected)
    # SAP30 round 1 kept two of three replicates, so its mean is over what was released
    assert means[(_D_SPACER, 5.0)] == pytest.approx(0.9)


def test_drop_log_counts_guides_times_nine_and_background_before_the_value(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """The own-background test is round-scoped and now runs BEFORE the per-replicate value
    test, so NAT1's three blank round-3 cells count as own-background (NAT1 is round 3's
    integrated background) rather than as missing values: own-background 9, not 6. The
    earlier mean build counted that cell the other way round, because its value test came
    first.

    Guide-scoped rules multiply by the nine (round, replicate) cells whatever the rows
    hold: control 9, corrupted 9, unresolved 9 (item RDN25-1). 63 source, 22 kept, 41
    dropped.
    """
    drop_log = json.loads(
        (Path(dataset.root) / "preprocess" / "dropped_records.json").read_text()
    )
    assert drop_log == {
        "dataset": "CrisprMagicLian2019Dataset",
        "source_records": 63,
        "kept_records": 22,
        "dropped_records": 41,
        "rules": [
            {
                "rule": "random_negative_control_guide",
                "scope": "guide",
                "description": (
                    "one of the 300 random negative-control guides (100 per library); "
                    "it targets no gene, so there is no genotype to key a record to"
                ),
                "n_records": 9,
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
                "n_records": 9,
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
                "n_records": 9,
                "items": ["RDN25-1"],
            },
            {
                "rule": "guide_round_replicate_has_no_enrichment_value",
                "scope": "guide_round_replicate",
                "description": (
                    "the guide was not detected in this replicate's before/after "
                    "libraries, so that replicate has no finite log2 enrichment for it. "
                    "A guide the count matrix holds at all carries all nine (round, "
                    "replicate) values, so in practice this rule fires for the 9,264 "
                    "reference guides that appear in no run"
                ),
                "n_records": 5,
                "items": [],
            },
            {
                "rule": "guide_targets_its_own_round_background",
                "scope": "guide_round",
                "description": (
                    "in this round the guide's target gene is already an integrated "
                    "background perturbation, so the foreground edit is redundant and "
                    "the strain signature would collapse to empty once the background "
                    "is subtracted. Counted in records, which is three (one per "
                    "replicate) per skipped (guide, round)"
                ),
                "n_records": 9,
                "items": [],
            },
        ],
    }
    assert drop_log == ln.DropLog.model_validate(drop_log).model_dump()


def test_one_reference_per_round_gene_set_manifest_and_interned_store(
    dataset: ln.CrisprMagicLian2019Dataset,
) -> None:
    """Three references in first-sighting order (rounds 1, 2, 3), each the typed bAID
    host with the round's furfural
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
        [0, 1, 2, 9, 10, 11, 12, 13, 16, 17, 18],
        [3, 4, 5, 19, 20, 21],
        [6, 7, 8, 14, 15],
    ]
    assert [entry["reference"] for entry in index] == [
        _reference(rnd).model_dump() for rnd in (1, 2, 3)
    ]
    assert [dataset[i]["reference"] for i in range(_N_RECORDS)] == [
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
    assert manifest["manifest_schema_version"] == 3
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
    ``download()`` and ``process()`` (no genome, raw files deleted): 22 equal records,
    ``get([2, 5])`` as a list, ``dataset[[2, 5]]`` as PyG's two-item subset,
    ``transform_item`` rebuilding the typed round-2 CRISPRi record, ``repr``
    ``CrisprMagicLian2019Dataset(22)``, and the inline hooks.
    """
    records = [dataset[i] for i in range(_N_RECORDS)]
    dataset.close_lmdb()
    for name in (ln.REPLICATES_TSV_FILENAME, ln.DESIGN_D_FILENAME):
        (Path(dataset.root) / "raw" / name).unlink()
    reopened = ln.CrisprMagicLian2019Dataset(root=dataset.root, genome=None)
    assert len(reopened) == _N_RECORDS
    assert [reopened[i] for i in range(_N_RECORDS)] == records
    assert reopened.get([2, 5]) == [records[2], records[5]]
    subset = reopened[[2, 5]]
    assert len(subset) == 2
    assert [subset[0], subset[1]] == [records[2], records[5]]
    # PyG's list index is a shallow copy that opens its own handle on the same store;
    # py-lmdb refuses a second open of one path in a process, so close it first.
    subset.close_lmdb()
    assert reopened.transform_item(reopened[5]) == {
        "experiment": _EXPECTED[5],
        "reference": _reference(2),
        "publication": _PUBLICATION,
    }
    assert repr(reopened) == f"CrisprMagicLian2019Dataset({_N_RECORDS})"
    assert reopened.gene_set == _GENES
    with pytest.raises(NotImplementedError):
        reopened.create_experiment()
    sentinel = object()
    assert reopened.preprocess_raw(sentinel) is sentinel
    reopened.close_lmdb()


def test_genome_is_required_before_the_store_is_opened(tmp_path: Path) -> None:
    """With the raw files present and ``genome=None`` the resolver (line 696) raises the
    loader's message before ``_open_write_lmdb`` (752), so no ``processed/lmdb`` is left
    behind and a retry with a genome builds the 22 records.
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
    assert len(retry) == _N_RECORDS
    retry.close_lmdb()


class _Staged:
    """The four fixture sources ``deposit_raw_mirror`` takes, and their digests."""

    def __init__(self, staging: Path) -> None:
        self.dir = staging
        self.replicates = staging / ln.REPLICATES_TSV_FILENAME
        self.design = staging / ln.DESIGN_D_FILENAME
        self.mean = staging / ln.TSV_FILENAME
        self.runinfo = staging / ln.RUNINFO_FILENAME
        self.mean.write_text("guide_id\tr1_log2fc_mean\n1_i_0PDR1\t0.5\n")
        self.runinfo.write_text("Run,LibraryName\nSRR8293138,Ff1a\n")
        self.replicates_sha = _sha256(self.replicates)
        self.design_sha = _sha256(self.design)
        self.mean_sha = _sha256(self.mean)
        self.runinfo_sha = _sha256(self.runinfo)

    def pin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Re-pin all four module digests to this fixture's bytes."""
        monkeypatch.setattr(ln, "REPLICATES_TSV_SHA256", self.replicates_sha)
        monkeypatch.setattr(ln, "DESIGN_D_SHA256", self.design_sha)
        monkeypatch.setattr(ln, "TSV_SHA256", self.mean_sha)
        monkeypatch.setattr(ln, "RUNINFO_SHA256", self.runinfo_sha)

    def deposit(self) -> Path:
        return ln.deposit_raw_mirror(
            enrichment_path=self.mean,
            replicates_path=self.replicates,
            runinfo_path=self.runinfo,
            design_d_path=self.design,
        )


def _staged_raw(tmp_path: Path, name: str) -> _Staged:
    """Write the fixture raw + mirror sources to ``<tmp>/<name>``."""
    staging = tmp_path / name
    _write_raw(staging)
    return _Staged(staging)


def _expected_manifest(staged: _Staged) -> Manifest:
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
                bytes=staged.mean.stat().st_size,
                sha256=staged.mean_sha,
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
                path=ln.REPLICATES_TSV_REL,
                role=ROLE_RAW_DATA,
                bytes=staged.replicates.stat().st_size,
                sha256=staged.replicates_sha,
                source="derived: SRA PRJNA504483 reprocessing, per replicate",
                processing=ProcessingRecord(
                    processor=(
                        "experiments/016-lian-magic-reprocess/scripts/"
                        "replicate_enrichment.py"
                    ),
                    tool="torchcell lian-magic-reprocess pipeline",
                    version="2026-11-24",
                    params={
                        "sra_project": "PRJNA504483",
                        "n_runs": 21,
                        "barcode_window": "read[27:70] (43 bp activation) | read[27:71] "
                        "(44 bp interference/deletion), forward, exact match",
                        "normalization": "CPM(+1) per library; per round per replicate "
                        "log2(furfural-after / untreated-before), one column per (round, "
                        "replicate); NO mean and NO SD",
                        "replicate_pairing": "the authors' library labels from the SRA "
                        "run report, 'a' stressed over 'b' untreated of the same "
                        "replicate (Ff1a/Ff1b ... 3Ff3a/3Ff3b)",
                        "verification": "the mean and sample SD over the three emitted "
                        "columns reproduce guide_enrichment_final.tsv's "
                        "r{rnd}_log2fc_mean / _sd for all 91,229 detected guides in all "
                        "three rounds; max absolute difference 1.3e-15, asserted at "
                        "<= 1e-9",
                        "flags": "is_control / corrupted_gene carried over verbatim from "
                        "guide_enrichment_final.tsv",
                    },
                    input_sha256=[ln.REFERENCE_SHA256, staged.mean_sha],
                ),
            ),
            ArtifactRecord(
                path=ln.RUNINFO_REL,
                role=ROLE_RAW_DATA,
                bytes=staged.runinfo.stat().st_size,
                sha256=staged.runinfo_sha,
                source=ln.RUNINFO_URL,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.direct_url,
                    source_url=ln.RUNINFO_URL,
                    retriever="curl",
                    params={
                        "command": (
                            f"curl -s '{ln.RUNINFO_URL}' -o {ln.RUNINFO_FILENAME}"
                        ),
                        "why": "the authors' own library labels (LibraryName = "
                        "SampleName) for the 21 runs, which is what replicate_id "
                        "states. Named by the Data availability statement: 'The raw "
                        "reads of the NGS data were deposited into the NCBI Sequence "
                        "Read Archive (SRA) database (accession number: PRJNA504483)'",
                    },
                    sha256=staged.runinfo_sha,
                    retrieved_at=ln.RUNINFO_RETRIEVED_AT,
                ),
            ),
            ArtifactRecord(
                path=ln.DESIGN_D_REL,
                role=ROLE_SI_DATA,
                bytes=staged.design.stat().st_size,
                sha256=staged.design_sha,
                source=design_url,
                retrieval=RetrievalRecord(
                    method=RetrievalMethod.springer_esm,
                    source_url=design_url,
                    retriever="torchcell.literature.retrieve.springer_esm",
                    params={"url": design_url},
                    sha256=staged.design_sha,
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
            "enrichment tables, recorded in their processing records rather than mirrored "
            "here, since the loader does not read it",
            "the per-guide furfural enrichment itself was NEVER released; it is "
            "reprocessed from SRA PRJNA504483, per biological replicate",
            "the Source Data file (41467_2019_13621_MOESM8_ESM.xlsx) holds the "
            "verification measurements of the INDIVIDUALLY constructed strains (relative "
            "biomass accumulation, relative cellulase activity, qPCR Ct, reporter "
            "fluorescence), three or more repeats each. No class reads them; see the "
            "dataset note's 2026.11.24 section",
        ],
        provenance_complete=True,
    )


def test_deposit_raw_mirror_records_both_derived_tables_the_run_report_and_the_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``deposit_raw_mirror`` refuses a source whose sha256 is not the pinned one. With
    the four pins patched to the fixture digests it copies the mean table, the
    per-replicate table and the SRA run report to ``data/`` and the design workbook to
    ``si/si_data/``, and writes a manifest whose first two records are derived
    ``raw_data`` artifacts with a ``ProcessingRecord`` each (the mean table's 2026-07-13
    pipeline; the per-replicate table's 2026-11-24 one, naming the mean table among its
    inputs), whose third is the run report's ``direct_url`` retrieval and whose fourth is
    the ``springer_esm`` design retrieval; ``provenance_complete`` is True and
    ``created_at`` is UTC. A repeat deposit is a no-op and a different source under a
    re-pinned digest is refused because the destination holds other bytes.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    staged = _staged_raw(tmp_path, "staging")
    with pytest.raises(RawSha256MismatchError) as err:
        staged.deposit()
    assert str(err.value) == (
        f"sha256 mismatch for {staged.mean}: expected {ln.TSV_SHA256}, "
        f"observed {staged.mean_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()
    # A later source off its pin is caught before the first is written either.
    monkeypatch.setattr(ln, "TSV_SHA256", staged.mean_sha)
    with pytest.raises(RawSha256MismatchError) as err:
        staged.deposit()
    assert str(err.value) == (
        f"sha256 mismatch for {staged.replicates}: expected "
        f"{ln.REPLICATES_TSV_SHA256}, observed {staged.replicates_sha}"
    )
    assert not (data_root / "torchcell-raw").exists()

    staged.pin(monkeypatch)
    mirror = staged.deposit()
    assert (
        mirror == data_root / "torchcell-raw" / ln.CITATION_KEY == ln.raw_mirror_dir()
    )
    assert (mirror / ln.TSV_REL).read_bytes() == staged.mean.read_bytes()
    assert (
        mirror / ln.REPLICATES_TSV_REL
    ).read_bytes() == staged.replicates.read_bytes()
    assert (mirror / ln.RUNINFO_REL).read_bytes() == staged.runinfo.read_bytes()
    assert (mirror / ln.DESIGN_D_REL).read_bytes() == staged.design.read_bytes()
    manifest = ln.load_manifest()
    expected = _expected_manifest(staged)
    assert manifest.model_dump(exclude={"created_at"}) == expected.model_dump(
        exclude={"created_at"}
    )
    assert [record.retrieval is None for record in manifest.files] == [
        True,
        True,
        False,
        False,
    ]
    assert [record.processing is None for record in manifest.files] == [
        False,
        False,
        True,
        True,
    ]
    assert manifest.created_at is not None
    assert datetime.fromisoformat(manifest.created_at).tzinfo == UTC
    assert ln.manifest_sha256(manifest, ln.REPLICATES_TSV_REL) == staged.replicates_sha
    assert ln.manifest_sha256(manifest, ln.DESIGN_D_REL) == staged.design_sha
    with pytest.raises(KeyError, match="nope is not in the raw-mirror manifest"):
        ln.manifest_sha256(manifest, "nope")

    written = (mirror / "manifest.json").read_text()
    assert staged.deposit() == mirror
    assert ln.load_manifest().files == Manifest.model_validate_json(written).files

    other = tmp_path / "other"
    other.mkdir()
    (other / ln.REPLICATES_TSV_FILENAME).write_bytes(b"a different table")
    monkeypatch.setattr(
        ln, "REPLICATES_TSV_SHA256", _sha256(other / ln.REPLICATES_TSV_FILENAME)
    )
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            f"{mirror / ln.REPLICATES_TSV_REL} exists with a different sha256; refusing"
        ),
    ):
        ln.deposit_raw_mirror(
            enrichment_path=staged.mean,
            replicates_path=other / ln.REPLICATES_TSV_FILENAME,
            runinfo_path=staged.runinfo,
            design_d_path=staged.design,
        )
    assert (
        mirror / ln.REPLICATES_TSV_REL
    ).read_bytes() == staged.replicates.read_bytes()


def test_download_links_the_verified_mirror_and_refuses_drift_or_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no raw files ``download()`` needs the mirror manifest (``FileNotFoundError``
    naming it). After the fixture files are deposited, a root whose ``raw/`` already links
    the per-replicate table keeps that link (``download()`` only creates a missing one),
    gets the design workbook linked beside it and builds the same 22 records; a drifted
    design workbook is refused naming both digests and a removed one is refused naming
    its path.
    """
    data_root = tmp_path / "data_root"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    manifest_path = data_root / "torchcell-raw" / ln.CITATION_KEY / "manifest.json"
    with pytest.raises(FileNotFoundError, match=re.escape(str(manifest_path))):
        ln.CrisprMagicLian2019Dataset(root=str(tmp_path / "a"), genome=_genome())

    staged = _staged_raw(tmp_path, "staging")
    staged.pin(monkeypatch)
    mirror = staged.deposit()
    root = tmp_path / "b"
    (root / "raw").mkdir(parents=True)
    os.symlink(
        mirror / ln.REPLICATES_TSV_REL, root / "raw" / ln.REPLICATES_TSV_FILENAME
    )
    built = ln.CrisprMagicLian2019Dataset(root=str(root), genome=_genome())
    assert (root / "raw" / ln.REPLICATES_TSV_FILENAME).is_symlink()
    assert (root / "raw" / ln.REPLICATES_TSV_FILENAME).resolve() == (
        mirror / ln.REPLICATES_TSV_REL
    ).resolve()
    assert (root / "raw" / ln.DESIGN_D_FILENAME).resolve() == (
        mirror / ln.DESIGN_D_REL
    ).resolve()
    assert len(built) == _N_RECORDS
    assert [built[i]["experiment"] for i in range(_N_RECORDS)] == [
        e.model_dump() for e in _EXPECTED
    ]
    built.close_lmdb()

    (mirror / ln.DESIGN_D_REL).write_bytes(b"drifted upstream")
    drifted = _sha256(mirror / ln.DESIGN_D_REL)
    with pytest.raises(RawSha256MismatchError) as err:
        ln.CrisprMagicLian2019Dataset(root=str(tmp_path / "c"), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {mirror / ln.DESIGN_D_REL}: expected "
        f"{staged.design_sha}, observed {drifted}"
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
    ``download()``, so ``process()`` verifies them against ``REPLICATES_TSV_SHA256`` and
    ``DESIGN_D_SHA256`` first. The per-replicate table off its pin raises
    ``RawSha256MismatchError`` naming it and both digests before a row is read; no store
    is written.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(ln, [ln.REPLICATES_TSV_FILENAME, ln.DESIGN_D_FILENAME])
    raw = staged.root / "raw" / ln.REPLICATES_TSV_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        ln.CrisprMagicLian2019Dataset(root=str(staged.root), genome=_genome())
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "46eaf7957ef410395177acf842ed0d36c12fd7c7f20ac5f5b2ee7dfd47d698ab, "
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

    The rules total 41 (the ``dropped_records`` of
    ``test_drop_log_counts_guides_times_nine_and_background_before_the_value``) while the
    log says 42: the exact mismatch message, and the tampered 42 already on disk.
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
        "drop accounting mismatch: rules total 41, 42 records missing from the build"
    )
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert log["dropped_records"] == 42
