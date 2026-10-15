# tests/torchcell/datasets/ecoli/test_lamoureux2023_public_k12.py
# [[tests.torchcell.datasets.ecoli.test_lamoureux2023_public_k12]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_lamoureux2023_public_k12.py
"""Public K-12 (Lamoureux 2023) loader: settling rules, dedup evidence, and a build.

The synthetic tests run everywhere. The end-to-end build reads the real
``EcoliK12MG1655Genome`` class over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (b0001 thrL, b0002 thrA, b0003
thrW, b0004 yaaP pseudogene, b0005 proB, b0006 proC, b0007 insZ pseudogene), served
through a stubbed ``resolve`` with the network refused, and replaces the module's
``assembly_reference`` with the MG1655 pin.

Synthetic release (``_release_files``): a Public K-12 metadata table of 24 rows, two of
them ``p1k_`` ids that ``public_rows`` must exclude, plus 22 public experiment accessions.
Four are kept (two wild-type LB replicates, one ``del_thrA`` on M9, one ``del_proB
del_proC`` anaerobic LB) and the other eighteen each hit exactly one drop rule, ten at the
genotype stage and eight at the environment stage. Sample ``k``'s count of gene ``i``
(0-based over b0001..b0007) is ``100000 (i + 1) + 7 k``, so no two libraries share a
profile and every library clears the paper's 500,000-read floor; the MultiQC table carries
that same sum as ``Assigned``. ``gene_info.csv`` repeats the synthetic assembly's spans
except on b0003, which diverges so the divergence ledger is exercised.

The data-gated tests (``--data``) audit every ``SourcedValue`` against the pinned paper
mirror, re-hash the raw mirror against the module pins, and pin the measured facts of the
dev-tree build: 240 records of 1,675 public rows (140 wild type, 84 single and 16 double
deletions), the thirteen drop rules, the dedup ledger (1,675 distinct experiment and run
accessions over 1,568 BioSamples, 38 of which carry several rows and 7 of which span
several conditions; zero repeated count profiles within the arm and zero against
PRECISE-1K), the 39 gene-span divergences, and the gene-resolution histogram.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import torchcell.datasets.ecoli.lamoureux2023 as L
import torchcell.datasets.ecoli.lamoureux2023_public_k12 as P
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.schema import (
    ConcentrationUnit,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    PhysicalFactor,
    ReferenceGenome,
    SmallMoleculePerturbation,
    StrainBackground,
    StrainReferenceGenome,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import SourcedValue, audit_sourced_value

G = P.PublicGenotypeRule
E = P.PublicEnvironmentRule

GENES = ["b0001", "b0002", "b0003", "b0004", "b0005", "b0006", "b0007"]
#: The synthetic assembly's own spans, and the one b0003 divergence the build must count.
SPANS = {
    "b0001": (1, 9),
    "b0002": (12, 26),
    "b0003": (30, 48),
    "b0004": (44, 52),
    "b0005": (55, 66),
    "b0006": (70, 81),
    "b0007": (84, 97),
}


# --------------------------------------------------------------------------- #
# Genotype settling
# --------------------------------------------------------------------------- #
def _row(**cells: str) -> dict[str, str]:
    """A public metadata row of wild-type LB cells, overridden by ``cells``."""
    row = {
        P.COL_SAMPLE_LABEL: "rpoB__wt__1",
        L.COL_PROJECT: "rpoB",
        L.COL_CONDITION: "wt",
        P.COL_REP: "1",
        L.COL_DESCRIPTION: "MG1655",
        L.COL_STRAIN: "MG1655",
        L.COL_CULTURE: "batch",
        L.COL_MEDIA: "LB",
        L.COL_TEMPERATURE: "37.0",
        L.COL_PH: "7.0",
        L.COL_CARBON: "",
        L.COL_NITROGEN: "",
        L.COL_SUPPLEMENT: "",
        L.COL_FULL_NAME: "rpoB:wt",
        P.COL_RUN: "SRR000001",
        P.COL_BIOPROJECT: "PRJNA000001",
        P.COL_BIOSAMPLE: "SAMN000001",
        P.COL_SCIENTIFIC_NAME: "Escherichia coli str. K-12 substr. MG1655",
        P.COL_GEO_SERIES: "",
        P.COL_GEO_SAMPLE: "",
        P.COL_PMID: "",
        P.COL_AEROBICITY: "aerobic",
        P.COL_TIME: "",
        P.COL_REFERENCE_CONDITION: "wt",
    }
    row.update(cells)
    return row


@pytest.mark.parametrize(
    "cells,rule,deleted",
    [
        ({}, G.wild_type, ()),
        ({L.COL_DESCRIPTION: "MG1655 del_thrA"}, G.deletion, ("thrA",)),
        ({L.COL_DESCRIPTION: "MG1655 del_proB del_proC"}, G.deletion, ("proB", "proC")),
        ({L.COL_STRAIN: "BW25113"}, G.strain_not_mg1655, ()),
        ({L.COL_STRAIN: "W3110"}, G.strain_not_mg1655, ()),
        ({L.COL_DESCRIPTION: "MG1655 pBR322"}, G.plasmid_borne_construct, ()),
        (
            {L.COL_DESCRIPTION: "MG1655 del_pgaC pBR322_csrA"},
            G.plasmid_borne_construct,
            (),
        ),
        ({L.COL_DESCRIPTION: "MG1655 fusAA608E"}, G.point_mutation_allele, ()),
        ({L.COL_DESCRIPTION: "MG1655 dnaA46"}, G.point_mutation_allele, ()),
        ({L.COL_DESCRIPTION: "MG1655 infect_T7"}, G.phage_infected, ()),
        ({L.COL_DESCRIPTION: "MG1655 Svi3-3 comp."}, G.phage_infected, ()),
        ({L.COL_DESCRIPTION: "MG1655 CHL evolved"}, G.evolved_isolate, ()),
        ({L.COL_DESCRIPTION: "MG1655 del_mutS Mutated"}, G.evolved_isolate, ()),
        (
            {L.COL_DESCRIPTION: "MG1655 with inactive relA"},
            G.allele_described_in_prose,
            (),
        ),
        ({L.COL_DESCRIPTION: "MG1655 del_7prrn"}, G.deletion_not_one_gene, ()),
        ({L.COL_DESCRIPTION: "MG1655 crp_ar1_ar2"}, G.partial_gene_edit, ()),
        ({L.COL_DESCRIPTION: "MG1655 Z1 del_arcZ"}, G.background_label_undefined, ()),
        ({L.COL_DESCRIPTION: "MG1655 lacIq"}, G.background_label_undefined, ()),
        ({L.COL_DESCRIPTION: "MG1655 del_ihf"}, G.deletion_symbol_unresolved, ("ihf",)),
        (
            {L.COL_DESCRIPTION: "MG1655 del_rnhA del_dnaA del_rrnD del_rrnC"},
            G.deletion_symbol_unresolved,
            ("rnhA", "dnaA", "rrnD", "rrnC"),
        ),
    ],
)
def test_every_genotype_rule_settles_its_own_cells(
    cells: dict[str, str], rule: P.PublicGenotypeRule, deleted: tuple[str, ...]
) -> None:
    verdict = P.settle_public_genotype(_row(**cells))
    assert verdict.rule is rule
    assert verdict.deleted_symbols == deleted
    assert verdict.kept is (rule in P.KEPT_GENOTYPE_RULES)


def test_an_unlisted_edit_token_raises_rather_than_being_guessed() -> None:
    with pytest.raises(ValueError, match="are of no listed form"):
        P.settle_public_genotype(_row(**{L.COL_DESCRIPTION: "MG1655 pIMAGINARY77"}))


def test_a_description_that_does_not_name_the_strain_raises() -> None:
    with pytest.raises(ValueError, match="unrecognized Strain Description"):
        P.settle_public_genotype(_row(**{L.COL_DESCRIPTION: "K-12 del_thrA"}))


def test_a_repeated_deletion_raises() -> None:
    with pytest.raises(ValueError, match="repeats a deletion"):
        P.settle_public_genotype(
            _row(**{L.COL_DESCRIPTION: "MG1655 del_thrA del_thrA"})
        )


def test_every_rule_carries_a_reason_except_the_two_kept_ones() -> None:
    assert set(P.PUBLIC_GENOTYPE_RULE_REASONS) | P.KEPT_GENOTYPE_RULES == set(
        P.PublicGenotypeRule
    )
    assert set(P.PUBLIC_ENVIRONMENT_RULE_REASONS) == set(P.PublicEnvironmentRule)
    assert set(P.PUBLIC_RULE_ORDER) == set(P.PUBLIC_EDIT_TOKENS.values())


# --------------------------------------------------------------------------- #
# Environment settling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "culture,named", [("bioreactor", "bioreactor"), ("", "(blank)")]
)
def test_a_non_batch_culture_is_stored_with_a_typed_gap(
    culture: str, named: str
) -> None:
    """#1057: never refused; the gap names the culture cell as released."""
    verdict = P.settle_public_environment(_row(**{L.COL_CULTURE: culture}))
    assert verdict.rule is None and verdict.spec is not None
    assert verdict.spec.culture_without_rate == named
    environment = P.build_public_environment(verdict.spec)
    gap = {g.field: g for g in environment.provenance_gaps}["dilution_rate_per_hour"]
    assert named in str(gap.note)


def test_a_strain_or_label_sample_is_stored_on_its_released_name() -> None:
    """#1057: a strain types its deletions; an untyped description is the name."""
    bw = P.settle_public_background(
        _row(**{L.COL_STRAIN: "BW25113", L.COL_DESCRIPTION: "BW25113 del_thrA"})
    )
    assert bw.kept and bw.deleted_symbols == ("thrA",)
    assert bw.background is not None and bw.background.name == "BW25113"
    w = P.settle_public_background(
        _row(**{L.COL_STRAIN: "W3110", L.COL_DESCRIPTION: "W3110 D13"})
    )
    assert w.background is not None and w.background.name == "W3110 D13"
    assert (w.untyped_edits, w.deleted_symbols) == (True, ())
    plasmid = P.settle_public_background(
        _row(**{L.COL_STRAIN: "W3110", L.COL_DESCRIPTION: "W3110 pBR322"})
    )
    assert (plasmid.kept, plasmid.rule) == (False, G.plasmid_borne_construct)
    # #1045 part 2: the dispatch stores it on a background named as released
    stored = P.settle_public_sample_genotype(
        _row(**{L.COL_STRAIN: "W3110", L.COL_DESCRIPTION: "W3110 pBR322"})
    )
    assert stored.kept and stored.untyped_edits
    assert stored.background is not None and stored.background.name == "W3110 pBR322"
    evolved = P.settle_public_background(
        _row(**{L.COL_DESCRIPTION: "MG1655 CHL evolved"})
    )
    assert evolved.background is not None and evolved.background.evolved


def _spec(verdict: P.PublicEnvironmentVerdict) -> L.ConditionSpec:
    """The parsed condition of a kept verdict."""
    assert verdict.spec is not None, verdict.rule
    return verdict.spec


def test_every_oxygen_cell_is_stored_as_released() -> None:
    """#1045 part 3: a blank cell is a typed gap, 'transition' is its own regime, and
    'aerobic(30% DO)' is aerobic plus a dissolved-oxygen setpoint of 30 percent.
    """
    blank = P.build_public_environment(
        _spec(P.settle_public_environment(_row(**{P.COL_AEROBICITY: ""})))
    )
    assert blank.aerobicity is None
    (gap,) = [g for g in blank.provenance_gaps if g.field == "aerobicity"]
    assert gap.looked_in is not None
    assert gap.looked_in.source_uri.endswith("k12_modulome/metadata_qc.csv")
    transition = P.build_public_environment(
        _spec(P.settle_public_environment(_row(**{P.COL_AEROBICITY: "transition"})))
    )
    assert transition.aerobicity == "transition"
    held = P.build_public_environment(
        _spec(
            P.settle_public_environment(_row(**{P.COL_AEROBICITY: "aerobic(30% DO)"}))
        )
    )
    assert held.aerobicity == "aerobic"
    (setpoint,) = [
        p
        for p in held.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
        and p.factor is PhysicalFactor.dissolved_oxygen
    ]
    assert setpoint.magnitude is not None
    assert (setpoint.magnitude.value, setpoint.magnitude.unit) == (
        30.0,
        ConcentrationUnit.percent,
    )
    assert [g.field for g in setpoint.provenance_gaps] == ["agent"]
    with pytest.raises(KeyError):
        P.settle_public_environment(_row(**{P.COL_AEROBICITY: "hypoxic"}))


def test_a_dose_the_release_writes_as_a_multiple_is_kept_verbatim() -> None:
    """``halicin(0.25x)``: the ``x`` names no unit, so the dose is the verbatim cell."""
    (halicin,) = P.parse_public_supplements("halicin(0.25x)")
    assert (halicin.name, halicin.value, halicin.unitless_value) == (
        "halicin",
        None,
        "0.25x",
    )


@pytest.mark.parametrize(
    "cells,gaps",
    [
        ({L.COL_MEDIA: "NM3"}, ["is_synthetic", "state"]),
        ({L.COL_MEDIA: ""}, ["name", "is_synthetic", "state"]),
    ],
)
def test_a_medium_no_library_carries_is_stored_as_released(
    cells: dict[str, str], gaps: list[str]
) -> None:
    """#1045 part 2: the label names the medium; what it does not state is a gap."""
    verdict = P.settle_public_environment(_row(**cells))
    assert verdict.rule is None and verdict.spec is not None
    media = P.build_public_environment(verdict.spec).media
    assert [g.field for g in media.provenance_gaps] == gaps
    assert (media.is_synthetic, media.base_medium) == (None, None)


def test_a_blank_temperature_is_a_typed_gap() -> None:
    """#1045 part 2: never refused; the environment says the cell is blank."""
    verdict = P.settle_public_environment(_row(**{L.COL_TEMPERATURE: ""}))
    assert verdict.rule is None and verdict.spec is not None
    environment = P.build_public_environment(verdict.spec)
    assert environment.temperature is None
    (gap,) = [g for g in environment.provenance_gaps if g.field == "temperature"]
    assert gap.looked_in is not None
    assert gap.looked_in.source_uri.endswith("k12_modulome/metadata_qc.csv")


def test_a_cell_with_a_parenthesized_name_and_the_public_units_parse() -> None:
    """The cells of the media stored since #1045 part 2, verbatim from the release."""
    sulfate = L.parse_amount(
        "(NH4)2SO4(3.8g/L)",
        default_unit=ConcentrationUnit.g_per_l,
        extra_units=P.EXTRA_UNIT_TOKENS,
    )
    assert (sulfate.name, sulfate.value, sulfate.unit) == (
        "ammonium sulfate",
        3.8,
        ConcentrationUnit.g_per_l,
    )
    (thiamine,) = P.parse_public_supplements("thiamine(500ug/L)")
    assert (thiamine.value, thiamine.unit) == (0.5, ConcentrationUnit.ug_per_ml)
    (dpta,) = P.parse_public_supplements("DPTA(250µM)")
    assert (dpta.value, dpta.unit) == (250.0, ConcentrationUnit.micromolar)


def test_a_blank_ph_or_minimal_source_is_a_gap_and_mops_derives_from_its_key() -> None:
    """#1045 row 24: the cells that used to drop a sample are typed gaps now.

    A blank pH keeps the pH factor with its magnitude gapped; a blank source cell of an
    M9 row keeps that source factor with its compound and amount gapped; and 'MOPS'
    names a library medium, so it derives from ``MOPS_MINIMAL`` with its recipe
    deferred, as 'M9' and 'LB' always did.
    """
    ph = P.settle_public_environment(_row(**{L.COL_PH: ""})).spec
    assert ph is not None and ph.ph is None
    environment = P.build_public_environment(ph)
    (factor,) = [
        p
        for p in environment.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
        and p.factor is PhysicalFactor.ph
    ]
    assert factor.magnitude is None
    assert "magnitude" in {gap.field for gap in factor.provenance_gaps}
    minimal = P.settle_public_environment(
        _row(**{L.COL_MEDIA: "M9", L.COL_CARBON: "glucose(2)", L.COL_NITROGEN: ""})
    ).spec
    assert minimal is not None
    assert minimal.unstated_sources == (PhysicalFactor.nitrogen_source,)
    neither = P.settle_public_environment(
        _row(**{L.COL_MEDIA: "M9", L.COL_CARBON: "", L.COL_NITROGEN: ""})
    ).spec
    assert neither is not None
    assert neither.unstated_sources == (
        PhysicalFactor.carbon_source,
        PhysicalFactor.nitrogen_source,
    )
    mops = P.settle_public_environment(_row(**{L.COL_MEDIA: "MOPS"})).spec
    assert mops is not None
    assert P.build_public_environment(mops).media.base_medium == "MOPS_MINIMAL"


def test_an_unlisted_aerobicity_cell_raises() -> None:
    with pytest.raises(KeyError):
        P.settle_public_environment(_row(**{P.COL_AEROBICITY: "hypoxic"}))


@pytest.mark.parametrize("cell,expected", [("aerobic", "aerobic"), ("O2", "aerobic")])
def test_the_release_aerobic_tokens_both_mean_aerobic(cell: str, expected: str) -> None:
    verdict = P.settle_public_environment(_row(**{P.COL_AEROBICITY: cell}))
    assert verdict.spec is not None
    assert verdict.spec.aerobicity == expected


def test_a_kept_m9_row_carries_both_sources_and_the_supplements() -> None:
    verdict = P.settle_public_environment(
        _row(
            **{
                L.COL_MEDIA: "M9",
                L.COL_CARBON: "glucose(5g/L)",
                L.COL_NITROGEN: "NH4Cl(1)",
                L.COL_SUPPLEMENT: "casamino acids(0.1%), thiamin",
                P.COL_AEROBICITY: "anaerobic",
            }
        )
    )
    spec = verdict.spec
    assert spec is not None
    assert spec.aerobicity == "anaerobic"
    assert spec.electron_acceptor is None
    assert spec.trace is None
    assert spec.antibiotic is None
    assert spec.carbon == L.Amount(
        label="glucose(5g/L)", name="glucose", value=5.0, unit=ConcentrationUnit.g_per_l
    )
    assert spec.nitrogen == L.Amount(
        label="NH4Cl(1)",
        name="ammonium chloride",
        value=1.0,
        unit=ConcentrationUnit.g_per_l,
    )
    assert [a.name for a in spec.supplements] == ["casamino acids", "thiamin"]


@pytest.mark.parametrize(
    "cell,names",
    [
        ("", []),
        ("cytidine", ["cytidine"]),
        ("casamino acids(0.1%), thiamin", ["casamino acids", "thiamin"]),
        ("Ciprofloxacin(1ug/mL);ET2DA(500uM)", ["Ciprofloxacin", "ET2DA"]),
        ("MgSO4(2mM); CaCl2(100uM)", ["MgSO4", "CaCl2"]),
        ("2,2'-Dipyridyl(200uM)", ["2,2'-Dipyridyl"]),
    ],
)
def test_the_public_supplement_separators_split_only_where_they_should(
    cell: str, names: list[str]
) -> None:
    assert [a.name for a in P.parse_public_supplements(cell)] == names


def test_ng_per_ml_is_restated_in_ug_per_ml_exactly() -> None:
    (amount,) = P.parse_public_supplements("doxycycline(100ng/mL)")
    assert amount.value == 0.1
    assert amount.unit is ConcentrationUnit.ug_per_ml


def test_a_supplement_with_no_dose_is_a_fixed_basis_in_the_environment() -> None:
    verdict = P.settle_public_environment(_row(**{L.COL_SUPPLEMENT: "cytidine"}))
    assert verdict.spec is not None
    environment = P.build_public_environment(verdict.spec)
    small = [
        p for p in environment.perturbations if isinstance(p, SmallMoleculePerturbation)
    ]
    assert len(small) == 1
    assert small[0].concentration.basis is DoseBasis.fixed
    assert environment.duration_hours is None
    assert [g.field for g in environment.provenance_gaps] == ["duration_hours"]
    ph = [
        p
        for p in environment.perturbations
        if isinstance(p, EnvironmentPhysicalPerturbation)
        and p.factor is PhysicalFactor.ph
    ]
    assert len(ph) == 1
    assert ph[0].magnitude is not None
    assert ph[0].magnitude.value == 7.0


def test_the_sibling_parser_is_unchanged_without_the_extra_units() -> None:
    """``extra_units`` is additive: an unknown token still raises for PRECISE-1K."""
    with pytest.raises(ValueError, match="unknown unit 'g/L'"):
        L.parse_amount("glucose(5g/L)", default_unit=None)
    assert L.parse_amount("adenine (100mg/L)", default_unit=None).value == 100.0


# --------------------------------------------------------------------------- #
# Values
# --------------------------------------------------------------------------- #
def test_tpm_is_length_normalized_and_sums_to_one_million() -> None:
    counts = np.array([100.0, 200.0], dtype=np.float64)
    lengths = np.array([10.0, 100.0], dtype=np.float64)
    values = P.tpm(counts, lengths)
    assert values.sum() == pytest.approx(1e6)
    # 100/10 = 10 per base against 200/100 = 2 per base, so 5:1 despite 1:2 in counts.
    assert values[0] / values[1] == pytest.approx(5.0)


def test_the_phenotype_carries_the_released_counts_and_the_assigned_total() -> None:
    phenotype = P.rnaseq_phenotype(
        ["b0001", "b0002"],
        np.array([3, 7], dtype=np.int64),
        np.array([10.0, 10.0], dtype=np.float64),
        n_mapped_reads=10,
    )
    assert phenotype.expression_count == {"b0001": 3, "b0002": 7}
    assert phenotype.n_mapped_reads == 10
    assert phenotype.measurement_type == "rnaseq_tpm_from_released_counts"
    assert sum(phenotype.expression_tpm.values()) == pytest.approx(1e6)
    assert phenotype.provenance_gaps == []


def test_public_rows_drops_the_precise1k_arm_and_demands_an_accession() -> None:
    frame = pd.DataFrame(index=["p1k_00001", "SRX1", "ERX2", "DRX3"])
    assert P.public_rows(frame) == ["DRX3", "ERX2", "SRX1"]
    with pytest.raises(RuntimeError, match="public rows with no experiment accession"):
        P.public_rows(pd.DataFrame(index=["GSM1"]))


# --------------------------------------------------------------------------- #
# Raw mirror
# --------------------------------------------------------------------------- #
def _release_files(directory: Path) -> dict[str, bytes]:
    """The six synthetic release members, written under ``directory``; name -> bytes."""
    public = pd.DataFrame.from_dict(_public_rows(), orient="index")
    p1k = pd.DataFrame.from_dict(_p1k_rows(), orient="index")
    samples = list(public.index)
    counts = pd.DataFrame(
        {
            s: [100000 * (i + 1) + 7 * k for i in range(len(GENES))]
            for k, s in enumerate([*samples, "SRX999999"])
        },
        index=GENES,
    )
    counts.index.name = "Geneid"
    p1k_counts = pd.DataFrame(
        {
            s: [90000 * (i + 1) + 11 * k for i in range(len(GENES))]
            for k, s in enumerate(p1k.index)
        },
        index=GENES,
    )
    p1k_counts.index.name = "Geneid"
    multiqc = pd.DataFrame({P.COL_MULTIQC_ASSIGNED: counts[counts.columns].sum(axis=0)})
    multiqc.index.name = P.COL_MULTIQC_SAMPLE
    annotation = pd.DataFrame(
        {
            P.COL_GENE_START: [SPANS[g][0] for g in GENES],
            P.COL_GENE_END: [SPANS[g][1] for g in GENES],
        },
        index=GENES,
    )
    annotation.index.name = "locus_tag"
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for name, frame, sep in [
        (P.PUBLIC_METADATA.name, public, ","),
        (P.PUBLIC_COUNTS.name, counts, ","),
        (P.PUBLIC_MULTIQC.name, multiqc, "\t"),
        (P.GENE_INFO.name, annotation, ","),
        (L.COUNTS.name, p1k_counts, ","),
        (L.METADATA.name, p1k, ","),
    ]:
        frame.to_csv(directory / name, sep=sep)
        out[name] = (directory / name).read_bytes()
    return out


def _p1k_rows() -> dict[str, dict[str, str]]:
    """The two PRECISE-1K ``control:wt_glc`` rows this dataset's reference reads."""
    row = {
        L.COL_SAMPLE: "control__wt_glc__1",
        L.COL_STUDY: "Control",
        L.COL_PROJECT: L.CONTROL_PROJECT,
        L.COL_CONDITION: "wt_glc",
        L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655",
        L.COL_STRAIN: "MG1655",
        L.COL_CULTURE: "Batch",
        L.COL_EVOLVED: "No",
        L.COL_MEDIA: "M9",
        L.COL_TEMPERATURE: "37",
        L.COL_PH: "7.0",
        L.COL_CARBON: "glucose(2)",
        L.COL_NITROGEN: "NH4Cl(1)",
        L.COL_ACCEPTOR: "O2",
        L.COL_TRACE: "sauer trace element mixture",
        L.COL_SUPPLEMENT: "",
        L.COL_ANTIBIOTIC: "",
        L.COL_FULL_NAME: "control:wt_glc",
        L.COL_REP: "1",
        L.COL_REPLICATES: "2.0",
        L.COL_PROJECT_REFERENCE: "p1k_00001;p1k_00002",
    }
    second = dict(row, **{L.COL_SAMPLE: "control__wt_glc__2", L.COL_REP: "2"})
    return {"p1k_00001": row, "p1k_00002": second}


def _public_rows() -> dict[str, dict[str, str]]:
    """Twenty-two public accessions plus the two ``p1k_`` ids the loader excludes."""

    def row(accession: str, k: int, **cells: str) -> dict[str, str]:
        return _row(
            **{
                P.COL_RUN: f"SRR{k:06d}",
                P.COL_BIOSAMPLE: f"SAMN{k:06d}",
                P.COL_SAMPLE_LABEL: f"sample__{k}",
                **cells,
            }
        )

    rows = {
        "SRX000001": row("SRX000001", 1),
        "SRX000002": row("SRX000002", 2, **{P.COL_REP: "2"}),
        "SRX000003": row(
            "SRX000003",
            3,
            **{
                L.COL_DESCRIPTION: "MG1655 del_thrA",
                L.COL_FULL_NAME: "thrA:del_thrA",
                L.COL_MEDIA: "M9",
                L.COL_CARBON: "glucose(5g/L)",
                L.COL_NITROGEN: "NH4Cl(1)",
                L.COL_SUPPLEMENT: "casamino acids(0.1%), thiamin",
                P.COL_AEROBICITY: "O2",
                P.COL_PMID: "29046437",
                P.COL_GEO_SERIES: "GSE102381",
                P.COL_GEO_SAMPLE: "GSM2735461",
            },
        ),
        "SRX000004": row(
            "SRX000004",
            4,
            **{
                L.COL_DESCRIPTION: "MG1655 del_proB del_proC",
                L.COL_FULL_NAME: "pro:double",
                L.COL_TEMPERATURE: "30.0",
                L.COL_SUPPLEMENT: "doxycycline(100ng/mL);cytidine",
                P.COL_AEROBICITY: "anaerobic",
                P.COL_TIME: "12:00:00",
            },
        ),
        "SRX000005": row(
            "SRX000005",
            5,
            **{
                L.COL_STRAIN: "BW25113",
                L.COL_DESCRIPTION: "BW25113 del_thrA",
                L.COL_FULL_NAME: "bw:del_thrA",
            },
        ),
        "SRX000006": row(
            "SRX000006",
            6,
            **{L.COL_DESCRIPTION: "MG1655 pBR322", L.COL_FULL_NAME: "pbr:wt"},
        ),
        "SRX000007": row(
            "SRX000007",
            7,
            **{L.COL_DESCRIPTION: "MG1655 fusAA608E", L.COL_FULL_NAME: "fusA:a608e"},
        ),
        "SRX000008": row(
            "SRX000008",
            8,
            **{L.COL_DESCRIPTION: "MG1655 infect_T7", L.COL_FULL_NAME: "t7:infect"},
        ),
        "SRX000009": row(
            "SRX000009",
            9,
            **{L.COL_DESCRIPTION: "MG1655 CHL evolved", L.COL_FULL_NAME: "chl:ale"},
        ),
        "SRX000010": row(
            "SRX000010",
            10,
            **{
                L.COL_DESCRIPTION: "MG1655 with inactive relA",
                L.COL_FULL_NAME: "rela:inactive",
            },
        ),
        "SRX000011": row(
            "SRX000011",
            11,
            **{L.COL_DESCRIPTION: "MG1655 del_7prrn", L.COL_FULL_NAME: "rrn:del7"},
        ),
        "SRX000012": row(
            "SRX000012",
            12,
            **{L.COL_DESCRIPTION: "MG1655 crp_ar1_ar2", L.COL_FULL_NAME: "crp:ar1ar2"},
        ),
        "SRX000013": row(
            "SRX000013",
            13,
            **{L.COL_DESCRIPTION: "MG1655 Z1 del_thrA", L.COL_FULL_NAME: "z1:del_thrA"},
        ),
        "SRX000014": row(
            "SRX000014",
            14,
            **{L.COL_DESCRIPTION: "MG1655 del_ihf", L.COL_FULL_NAME: "ihf:del"},
        ),
        "SRX000015": row(
            "SRX000015", 15, **{L.COL_MEDIA: "NM3", L.COL_FULL_NAME: "nm3:wt"}
        ),
        "SRX000016": row(
            "SRX000016",
            16,
            **{L.COL_CULTURE: "bioreactor", L.COL_FULL_NAME: "bioreactor:wt"},
        ),
        # #1045 part 3: the three oxygen cells the schema could not hold, now stored
        "SRX000017": row(
            "SRX000017", 17, **{P.COL_AEROBICITY: "", L.COL_FULL_NAME: "o2:blank"}
        ),
        "SRX000018": row(
            "SRX000018",
            18,
            **{P.COL_AEROBICITY: "transition", L.COL_FULL_NAME: "o2:transition"},
        ),
        "SRX000019": row(
            "SRX000019",
            19,
            **{P.COL_AEROBICITY: "aerobic(30% DO)", L.COL_FULL_NAME: "o2:do30"},
        ),
        # #1045 part 2: kept with a gapped temperature, in the group of SRX000001 and
        # SRX000002, from whose rows it differs in that one cell
        "SRX000020": row("SRX000020", 20, **{L.COL_TEMPERATURE: ""}),
        # #1045 row 24: kept, with the blank pH a gapped factor
        "SRX000021": row("SRX000021", 21, **{L.COL_PH: "", L.COL_FULL_NAME: "noph:wt"}),
        # #1045 row 24: kept, with the blank nitrogen source a gapped factor
        "SRX000022": row(
            "SRX000022",
            22,
            **{
                L.COL_MEDIA: "M9",
                L.COL_CARBON: "glucose(2)",
                L.COL_FULL_NAME: "m9:no_nitrogen",
            },
        ),
    }
    for key, value in _p1k_rows().items():
        rows[key] = {column: value.get(column, "") for column in rows["SRX000001"]}
    return rows


def _pin(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    """Point the module pins at the synthetic bytes."""
    pinned = {
        raw.name: raw.model_copy(
            update={"sha256": hashlib.sha256(files[raw.name]).hexdigest()}
        )
        for raw in P.CONSUMED_RAW_FILES
    }
    monkeypatch.setattr(
        P, "PUBLIC_RAW_FILES", tuple(pinned[raw.name] for raw in P.PUBLIC_RAW_FILES)
    )
    monkeypatch.setattr(
        P, "CONSUMED_RAW_FILES", tuple(pinned[raw.name] for raw in P.CONSUMED_RAW_FILES)
    )
    monkeypatch.setattr(P, "P1K_COUNTS", pinned[L.COUNTS.name])
    monkeypatch.setattr(P, "P1K_METADATA", pinned[L.METADATA.name])
    monkeypatch.setattr(P, "PUBLIC_METADATA", pinned[P.PUBLIC_METADATA.name])
    monkeypatch.setattr(P, "PUBLIC_COUNTS", pinned[P.PUBLIC_COUNTS.name])
    monkeypatch.setattr(P, "PUBLIC_MULTIQC", pinned[P.PUBLIC_MULTIQC.name])
    monkeypatch.setattr(P, "GENE_INFO", pinned[P.GENE_INFO.name])
    monkeypatch.setattr(
        P,
        "REFERENCE_RAW_FILES",
        tuple(pinned[raw.name] for raw in P.REFERENCE_RAW_FILES),
    )
    monkeypatch.setattr(P, "GENE_LENGTH_DIVERGENCE_N", 1)


def _seed_reference_manifest(mirror: Path, files: dict[str, bytes]) -> None:
    """The two PRECISE-1K members and their pins, as the sibling deposit leaves them.

    ``deposit_public_raw_mirror`` is additive: it keeps every manifest record it does not
    itself write, so the sibling's PRECISE-1K pins survive a public deposit.
    """
    from torchcell.literature.manifest import ROLE_RAW_DATA, ArtifactRecord, Manifest

    records = []
    for raw in P.REFERENCE_RAW_FILES:
        (mirror / raw.relpath).parent.mkdir(parents=True, exist_ok=True)
        (mirror / raw.relpath).write_bytes(files[raw.name])
        records.append(
            ArtifactRecord(
                path=raw.relpath,
                role=ROLE_RAW_DATA,
                bytes=len(files[raw.name]),
                sha256=raw.sha256,
                source=f"{L.ARCHIVE_URL}#{raw.member}",
                retrieval=L.retrieval_record(raw),
            )
        )
    mirror.mkdir(parents=True, exist_ok=True)
    (mirror / "manifest.json").write_text(
        Manifest(
            citation_key=L.CITATION_KEY,
            doi=L.PAPER_DOI,
            title=L.PAPER_TITLE,
            files=records,
            provenance_complete=True,
            created_at="2026-10-07T00:00:00+00:00",
        ).model_dump_json(indent=2)
    )


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for raw in P.CONSUMED_RAW_FILES:
            archive.writestr(raw.member, files[raw.name])
        archive.writestr(f"{L.ARCHIVE_PREFIX}README.md", b"not consumed")
    return path


def test_deposit_writes_the_public_members_and_pins_both_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "precise1k-v1.0.zip", files)
    monkeypatch.setattr(
        P, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    _seed_reference_manifest(L.raw_mirror_dir(str(tmp_path / "dr")), files)
    root = P.deposit_public_raw_mirror(
        archive_path=archive, data_root=str(tmp_path / "dr")
    )
    assert sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    ) == [
        "data/annotation/gene_info.csv",
        "data/k12_modulome/counts.csv",
        "data/k12_modulome/metadata_qc.csv",
        "data/k12_modulome/multiqc_stats.tsv",
        "data/precise1k/counts.csv",
        "data/precise1k/metadata_qc.csv",
        "manifest.json",
    ]
    manifest = P.load_manifest(str(tmp_path / "dr"))
    assert [record.path for record in manifest.files] == [
        L.COUNTS.relpath,
        L.METADATA.relpath,
        P.PUBLIC_METADATA.relpath,
        P.PUBLIC_COUNTS.relpath,
        P.PUBLIC_MULTIQC.relpath,
        P.GENE_INFO.relpath,
    ]
    assert P.manifest_sha256(manifest, P.GENE_INFO.relpath) == P.GENE_INFO.sha256
    with pytest.raises(KeyError, match="is not in the raw-mirror manifest"):
        P.manifest_sha256(manifest, "data/k12_modulome/log_tpm.csv")
    record = manifest.files[2]
    assert record.retrieval is not None
    assert record.retrieval.params == {
        "url": L.ARCHIVE_URL,
        "member": f"{L.ARCHIVE_PREFIX}data/k12_modulome/metadata_qc.csv",
        "container_sha256": L.ARCHIVE_SHA256,
    }
    # Idempotent: a second deposit leaves the files alone; a tampered file refuses.
    P.deposit_public_raw_mirror(archive_path=archive, data_root=str(tmp_path / "dr"))
    (root / P.PUBLIC_MULTIQC.relpath).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        P.deposit_public_raw_mirror(
            archive_path=archive, data_root=str(tmp_path / "dr")
        )


def test_deposit_refuses_an_archive_off_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "a.zip", files)
    with pytest.raises(RuntimeError, match="sha256 mismatch: got"):
        P.deposit_public_raw_mirror(
            archive_path=archive, data_root=str(tmp_path / "dr")
        )


def _bare_dataset(root: Path) -> P.RnaseqPublicK12Lamoureux2023Dataset:
    dataset = P.RnaseqPublicK12Lamoureux2023Dataset.__new__(
        P.RnaseqPublicK12Lamoureux2023Dataset
    )
    dataset.root = str(root)
    return dataset


def test_download_links_every_member_of_both_arms_and_refuses_an_off_pin_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "a.zip", files)
    monkeypatch.setattr(
        P, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    _seed_reference_manifest(L.raw_mirror_dir(), files)
    mirror = P.deposit_public_raw_mirror(archive_path=archive)
    dataset = _bare_dataset(tmp_path / "build")
    dataset.download()
    for raw in P.CONSUMED_RAW_FILES:
        assert os.readlink(tmp_path / "build" / "raw" / raw.name) == str(
            mirror / raw.relpath
        )
    manifest = json.loads((mirror / "manifest.json").read_text())
    manifest["files"][2]["sha256"] = "ab" * 32
    (mirror / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestPinMismatchError):
        _bare_dataset(tmp_path / "build2").download()


def test_download_refuses_a_member_missing_from_the_mirror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "a.zip", files)
    monkeypatch.setattr(
        P, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    mirror = L.raw_mirror_dir()
    _seed_reference_manifest(mirror, files)
    assert P.deposit_public_raw_mirror(archive_path=archive) == mirror
    (mirror / P.GENE_INFO.relpath).unlink()
    with pytest.raises(
        RuntimeError, match="required raw artifact missing from mirror"
    ) as excinfo:
        _bare_dataset(tmp_path / "build").download()
    assert P.GENE_INFO.relpath in str(excinfo.value)


# --------------------------------------------------------------------------- #
# End-to-end build over the synthetic MG1655 assembly
# --------------------------------------------------------------------------- #
MG1655_PIN = ReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


def _pinned(
    strain: str, *, background: StrainBackground | None = None
) -> ReferenceGenome:
    """The MG1655 pin, on a released strain's background when one is given."""
    if background is None:
        return MG1655_PIN
    return StrainReferenceGenome.model_validate(
        MG1655_PIN.model_dump() | {"strain": background.name, "background": background}
    )


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> P.RnaseqPublicK12Lamoureux2023Dataset:
    """The synthetic release built end to end over the synthetic MG1655 genome."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_public_k12_lamoureux2023"
    files = _release_files(root / "raw")
    _pin(monkeypatch, files)
    monkeypatch.setattr(P, "assembly_reference", _pinned)
    return P.RnaseqPublicK12Lamoureux2023Dataset(root=str(root), ecoli_genome=genome)


def _json(dataset: P.RnaseqPublicK12Lamoureux2023Dataset, name: str) -> Any:
    return json.loads(Path(dataset.preprocess_dir, name).read_text())


#: The synthetic accessions the build stores: every one (#1045 part 3).
KEPT = [f"SRX{k:06d}" for k in range(1, 23)]


def test_the_build_keeps_wild_type_and_deletions_and_counts_every_drop(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    # SRX000021 (blank pH) and SRX000022 (blank nitrogen source on M9) are kept since
    # #1045 row 24, each with a gapped factor; since #1057 the BW25113 strain, the
    # evolved isolate, the undefined label and the bioreactor culture are stored too;
    # since #1045 part 2 the plasmid, the phage, the prose allele, the NM3 medium and
    # the blank temperature as well, and #1045 part 3 the three oxygen cells, so no
    # sample is dropped
    assert len(built) == len(KEPT)
    assert _json(built, "record_samples.json") == KEPT
    drops = _json(built, "dropped_records.json")
    assert drops["source_records"] == 22
    assert drops["kept_records"] == 22
    assert drops["dropped_records"] == 0
    assert drops["rules"] == []
    oxygen = [built[i]["experiment"]["environment"]["aerobicity"] for i in (16, 17, 18)]
    assert oxygen == [None, "transition", "aerobic"]
    assert drops["kept_by_genotype_class"] == {
        "background_allele_described_in_prose": 1,
        "background_background_label_undefined": 1,
        "background_evolved_isolate": 1,
        "background_phage_infected": 1,
        "background_plasmid_borne_construct": 1,
        "background_strain_not_mg1655": 1,
        "deletion_1": 1,
        "deletion_2": 1,
        "released_allele_deletion_not_one_gene": 1,
        "released_allele_deletion_symbol_unresolved": 1,
        "released_allele_partial_gene_edit": 1,
        "released_allele_point_mutation_allele": 1,
        "wild_type": 10,
    }
    assert drops["kept_by_base_media"] == {"LB": 19, "M9": 2, "NM3": 1}
    released = {
        built[i]["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]: (built[i]["experiment"]["genotype"]["perturbations"][0]["released_kind"])
        for i in (6, 10, 11, 13)
    }
    assert released == {
        "fusA": "sequence_allele",
        "7prrn": "deletion",
        "crp": "sequence_allele",
        "ihf": "deletion",
    }
    strains = [built[i]["reference"]["genome_reference"]["strain"] for i in range(22)]
    assert [strains[i] for i in (4, 5, 7, 8, 9, 12)] == [
        "BW25113",
        "MG1655 pBR322",
        "MG1655 infect_T7",
        "MG1655 CHL evolved",
        "MG1655 with inactive relA",
        "MG1655 Z1 del_thrA",
    ]
    for i in (5, 7, 9):
        assert built[i]["experiment"]["genotype"]["perturbations"] == []
        background = built[i]["reference"]["genome_reference"]["background"]
        assert [g["field"] for g in background["provenance_gaps"]] == [
            "genotype_statement"
        ]


def test_the_accession_ledger_is_the_deduplication_evidence(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    ledger = _json(built, "accession_ledger.json")
    assert ledger["source_rows"] == 24
    assert ledger["public_rows"] == 22
    assert ledger["p1k_rows"] == 2
    assert ledger["distinct_experiment_accessions"] == 22
    assert ledger["distinct_run_accessions"] == 22
    assert ledger["distinct_biosamples"] == 22
    assert ledger["biosamples_with_several_rows"] == 0
    assert ledger["rows_sharing_a_biosample"] == 0
    assert ledger["identical_count_profiles_within_public"] == 0
    assert ledger["identical_count_profiles_against_precise1k"] == 0
    assert ledger["distinct_pmids"] == 1
    assert ledger["distinct_geo_series"] == 1
    assert [r["experiment_accession"] for r in ledger["records"]] == KEPT
    assert ledger["records"][2] == {
        "experiment_accession": "SRX000003",
        "run_accession": "SRR000003",
        "biosample": "SAMN000003",
        "bioproject": "PRJNA000001",
        "scientific_name": "Escherichia coli str. K-12 substr. MG1655",
        "sample_label": "sample__3",
        "full_name": "thrA:del_thrA",
        "replicate": "1",
        "geo_series": "GSE102381",
        "geo_sample": "GSM2735461",
        "pmid": "29046437",
        "record_index": 2,
    }


def test_a_repeated_count_profile_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_public_k12_lamoureux2023"
    files = _release_files(root / "raw")
    counts = pd.read_csv(root / "raw" / P.PUBLIC_COUNTS.name, index_col=0)
    counts["SRX000002"] = counts["SRX000001"]
    counts.to_csv(root / "raw" / P.PUBLIC_COUNTS.name)
    files[P.PUBLIC_COUNTS.name] = (root / "raw" / P.PUBLIC_COUNTS.name).read_bytes()
    multiqc = pd.read_csv(root / "raw" / P.PUBLIC_MULTIQC.name, sep="\t", index_col=0)
    multiqc.loc["SRX000002", P.COL_MULTIQC_ASSIGNED] = int(counts["SRX000002"].sum())
    multiqc.to_csv(root / "raw" / P.PUBLIC_MULTIQC.name, sep="\t")
    files[P.PUBLIC_MULTIQC.name] = (root / "raw" / P.PUBLIC_MULTIQC.name).read_bytes()
    _pin(monkeypatch, files)
    monkeypatch.setattr(P, "assembly_reference", _pinned)
    with pytest.raises(RuntimeError, match="public count columns repeat another"):
        P.RnaseqPublicK12Lamoureux2023Dataset(root=str(root), ecoli_genome=genome)


def test_the_build_records_the_gene_span_divergence(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    divergence = _json(built, "gene_length_divergence.json")
    assert divergence == [
        {
            "released_name": "b0003",
            "locus_tag": "b0003",
            "release_span_bp": 19,
            "assembly_span_bp": 12,
        }
    ]


def test_every_record_stores_the_released_counts_and_a_one_million_tpm(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    for index in range(len(built)):
        phenotype = built[index]["experiment"]["phenotype"]
        assert sorted(phenotype["expression_tpm"]) == GENES
        assert sum(phenotype["expression_tpm"].values()) == pytest.approx(1e6)
        assert phenotype["n_mapped_reads"] == sum(
            phenotype["expression_count"].values()
        )
        assert phenotype["measurement_type"] == "rnaseq_tpm_from_released_counts"
    first = built[0]["experiment"]["phenotype"]["expression_count"]
    assert first == {g: 100000 * (i + 1) for i, g in enumerate(GENES)}


def test_the_reference_is_the_precise1k_control_condition(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    reference = built[0]["reference"]
    assert reference["dataset_name"] == "RnaseqPublicK12Lamoureux2023Dataset"
    assert reference["genome_reference"]["assembly_set"] == "ecoli_K12_MG1655_ASM584v2"
    phenotype = reference["phenotype_reference"]
    assert sum(phenotype["expression_tpm"].values()) == pytest.approx(1e6)
    # The mean of the two control columns, rounded half to even.
    assert phenotype["expression_count"] == {
        g: 90000 * (i + 1) + 6 for i, g in enumerate(GENES)
    }
    assert [g["field"] for g in phenotype["provenance_gaps"]] == ["n_mapped_reads"]
    media = reference["environment_reference"]["media"]
    assert media["base_medium"] == "M9"
    assert reference["environment_reference"]["aerobicity"] == "aerobic"


def test_the_replicate_groups_pin_the_conditions_and_their_accessions(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    groups = _json(built, "replicate_groups.json")
    assert [g["full_name"] for g in groups] == [
        "bioreactor:wt",
        "bw:del_thrA",
        "chl:ale",
        "crp:ar1ar2",
        "fusA:a608e",
        "ihf:del",
        "m9:no_nitrogen",
        "nm3:wt",
        "noph:wt",
        "o2:blank",
        "o2:do30",
        "o2:transition",
        "pbr:wt",
        "pro:double",
        "rela:inactive",
        "rpoB:wt",
        "rrn:del7",
        "t7:infect",
        "thrA:del_thrA",
        "z1:del_thrA",
    ]
    by_name = {g["full_name"]: g for g in groups}
    assert by_name["rpoB:wt"]["experiment_accessions"] == [
        "SRX000001",
        "SRX000002",
        "SRX000020",
    ]
    assert by_name["rpoB:wt"]["record_indices"] == [0, 1, 19]
    assert set(by_name["rpoB:wt"]["condition_cells"]) == set(P.CONDITION_COLUMNS)
    # #1045 part 2: a replicate whose row differs is reported, never refused
    assert by_name["rpoB:wt"]["differing_cells"] == {
        "SRX000020": {L.COL_TEMPERATURE: ""}
    }
    assert all(
        g["differing_cells"] == {} for name, g in by_name.items() if name != "rpoB:wt"
    )


def test_the_deleted_genes_are_written_as_released(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    """#1045 part 2: every deletion keys the symbol its token releases; no crosswalk."""
    assert built.gene_set == {"7prrn", "crp", "fusA", "ihf", "proB", "proC", "thrA"}
    assert not Path(built.preprocess_dir, "locus_tag_reconciliation.json").exists()
    (thra,) = built[2]["experiment"]["genotype"]["perturbations"]
    assert (thra["perturbation_type"], thra["released_kind"]) == (
        "released_identifier",
        "deletion",
    )
    assert thra["strain_label"] == "MG1655 del_thrA"


def test_a_divergence_count_off_its_pin_does_not_refuse_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_public_k12_lamoureux2023"
    files = _release_files(root / "raw")
    _pin(monkeypatch, files)
    monkeypatch.setattr(P, "GENE_LENGTH_DIVERGENCE_N", 7)
    monkeypatch.setattr(P, "assembly_reference", _pinned)
    # #1045 part 2: the span comparison is a derived finding; the genome gates nothing
    dataset = P.RnaseqPublicK12Lamoureux2023Dataset(root=str(root), ecoli_genome=genome)
    assert len(dataset) == len(KEPT)
    assert len(_json(dataset, "gene_length_divergence.json")) == 1


def test_a_count_total_off_the_multiqc_assigned_total_refuses_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_public_k12_lamoureux2023"
    files = _release_files(root / "raw")
    multiqc = pd.read_csv(root / "raw" / P.PUBLIC_MULTIQC.name, sep="\t", index_col=0)
    column = multiqc[P.COL_MULTIQC_ASSIGNED].to_numpy(dtype=np.int64)
    column[0] += 1
    multiqc[P.COL_MULTIQC_ASSIGNED] = column
    multiqc.to_csv(root / "raw" / P.PUBLIC_MULTIQC.name, sep="\t")
    files[P.PUBLIC_MULTIQC.name] = (root / "raw" / P.PUBLIC_MULTIQC.name).read_bytes()
    _pin(monkeypatch, files)
    monkeypatch.setattr(P, "assembly_reference", _pinned)
    with pytest.raises(RuntimeError, match="do not sum to the MultiQC Assigned total"):
        P.RnaseqPublicK12Lamoureux2023Dataset(root=str(root), ecoli_genome=genome)


def test_create_experiment_is_not_the_entrypoint(
    built: P.RnaseqPublicK12Lamoureux2023Dataset,
) -> None:
    frame = pd.DataFrame({"a": [1]})
    assert built.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        built.create_experiment()


# --------------------------------------------------------------------------- #
# Data-gated: the real paper mirror and the real dev-tree build
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


def _built_root() -> Path:
    root = Path(_data_root()) / "data/torchcell/rnaseq_public_k12_lamoureux2023"
    if not (root / "processed" / "lmdb").is_dir():
        pytest.skip("the dev-tree LMDB is not built")
    return root


def _sourced_values() -> dict[str, SourcedValue]:
    return {
        name: value
        for name, value in vars(P).items()
        if isinstance(value, SourcedValue)
    }


@pytest.mark.data
def test_every_sourced_value_quotes_the_pinned_mirror() -> None:
    values = _sourced_values()
    assert set(values) == {
        "CURATION",
        "CURATION_STRAIN_FILTER",
        "REPLICATE_RULE",
        "COMBINED",
        "SUBSTRAINS",
        "RELEASED_FILES",
        "CENTERING",
        "READ_DEPTH_FLOOR",
        "COUNTING",
    }
    library = os.path.join(_data_root(), "torchcell-library")
    assert {v.provenance.source_uri for v in values.values()} == {L.PAPER_MD}
    for name, value in values.items():
        result = audit_sourced_value(value, library)
        assert result.passed, (name, value.quote, result.message)


@pytest.mark.data
def test_the_raw_mirror_matches_every_module_pin() -> None:
    mirror = L.raw_mirror_dir(_data_root())
    manifest = P.load_manifest(_data_root())
    for raw in P.CONSUMED_RAW_FILES:
        assert P._sha256(mirror / raw.relpath) == raw.sha256
        assert P.manifest_sha256(manifest, raw.relpath) == raw.sha256


@pytest.mark.data
def test_the_dev_store_pins_the_measured_build() -> None:
    root = _built_root()
    drops = json.loads((root / "preprocess/dropped_records.json").read_text())
    assert drops["source_records"] == 1675
    # #1057: the other substrains, evolved isolates, undefined labels and non-batch
    # cultures are stored on a background named as released (240 before); #1045 part 2
    # stores the plasmid, phage and prose-allele samples, the media no library carries
    # and the blank temperatures (789 before), leaving the oxygen regimes
    # Environment.aerobicity cannot hold
    assert drops["kept_records"] == 1675
    assert drops["kept_by_genotype_class"] == {
        "background_allele_described_in_prose": 27,
        "background_background_label_undefined": 7,
        "background_evolved_isolate": 29,
        "background_phage_infected": 20,
        "background_plasmid_borne_construct": 175,
        "background_point_mutation_allele": 6,
        "background_strain_not_mg1655": 669,
        "deletion_1": 155,
        "deletion_2": 20,
        "released_allele_allele_described_in_prose": 9,
        "released_allele_deletion_symbol_unresolved": 6,
        "released_allele_partial_gene_edit": 3,
        "released_allele_point_mutation_allele": 35,
        "wild_type": 514,
    }
    assert drops["kept_by_base_media"] == {
        "": 180,
        "2xYT": 9,
        "Gutnick": 12,
        "Gutnick media": 4,
        "Kornberg": 30,
        "L-broth": 15,
        "LB": 426,
        "LB,MOPS": 12,
        "LB-Miller": 2,
        "LBK": 25,
        "LBKmal": 32,
        "Lennox LB": 23,
        "M4 minimal": 8,
        "M9": 510,
        "M9 (solid)": 42,
        "M9 RDM": 10,
        "MHBII": 18,
        "MOPS": 68,
        "MOPS EZ Rich": 36,
        "MOPS w/o KH2PO4": 6,
        "Mueller-Hinton": 3,
        "NM3": 30,
        "PBS": 9,
        "Tryptone broth": 12,
        "isosensitest broth": 56,
        "minimal medium A": 12,
        "other_defined_media": 74,
        "tryptone broth": 11,
    }
    # #1045 part 3: the 317 oxygen cells are records, so nothing is dropped
    assert drops["rules"] == []


@pytest.mark.data
def test_the_dev_store_pins_the_deduplication_measurement() -> None:
    root = _built_root()
    ledger = json.loads((root / "preprocess/accession_ledger.json").read_text())
    assert ledger["source_rows"] == 2710
    assert ledger["public_rows"] == 1675
    assert ledger["p1k_rows"] == 1035
    assert ledger["distinct_experiment_accessions"] == 1675
    assert ledger["distinct_run_accessions"] == 1675
    # BioSample is NOT the key: 38 BioSamples carry 145 rows, 7 of them across
    # several conditions, so collapsing on it would merge distinct genotypes.
    assert ledger["distinct_biosamples"] == 1568
    assert ledger["biosamples_with_several_rows"] == 38
    assert ledger["rows_sharing_a_biosample"] == 145
    assert ledger["biosamples_spanning_several_conditions"] == 7
    # The value-level rule, within the arm and against the stored PRECISE-1K counts.
    assert ledger["identical_count_profiles_within_public"] == 0
    assert ledger["identical_count_profiles_against_precise1k"] == 0
    assert ledger["distinct_bioprojects"] == 89
    assert ledger["distinct_pmids"] == 38
    assert ledger["distinct_geo_series"] == 30
    assert len(ledger["records"]) == 1675


@pytest.mark.data
def test_the_dev_store_pins_the_released_keys_and_the_span_divergence() -> None:
    root = _built_root()
    # #1045 part 2: the keys are the release's own row ids; no crosswalk is written
    assert not (root / "preprocess/locus_tag_reconciliation.json").exists()
    divergence = json.loads(
        (root / "preprocess/gene_length_divergence.json").read_text()
    )
    assert len(divergence) == P.GENE_LENGTH_DIVERGENCE_N == 38
    assert sorted(
        d["released_name"]
        for d in divergence
        if d["release_span_bp"] > d["assembly_span_bp"]
    ) == [
        "b0259",
        "b0552",
        "b0656",
        "b1331",
        "b1963",
        "b1978",
        "b1994",
        "b2030",
        "b2192",
        "b2982",
        "b3218",
        "b3505",
        "b4522",
        "b4711",
        "b4737",
        "b4751",
    ]
