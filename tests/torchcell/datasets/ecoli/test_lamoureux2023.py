# tests/torchcell/datasets/ecoli/test_lamoureux2023.py
# [[tests.torchcell.datasets.ecoli.test_lamoureux2023]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/ecoli/test_lamoureux2023.py
"""PRECISE-1K (Lamoureux 2023) loader: settling rules, parsing, mirror, and a full build.

The synthetic tests run everywhere. The end-to-end build reads the real
``EcoliK12MG1655Genome`` class over the synthetic MG1655 assembly of
``tests/torchcell/sequence/genome/_bacterial_fixtures.py`` (b0001 thrL, b0002 thrA, b0003
thrW, b0004 yaaP pseudogene, b0005 proB, b0006 proC, b0007 insZ pseudogene), served
through a stubbed ``resolve`` with the network refused, and replaces the module's
``assembly_reference`` (which reads the tier's assembly report) with the MG1655 pin.

Synthetic release (``_release``): seventeen samples. Kept, in sample order: p1k_00001 and
p1k_00002 (the ``control:wt_glc`` reference pair), p1k_00003 and p1k_00004 (``del_thrA``
on M9 with 50 ug/mL kanamycin), p1k_00005 (``delproB del_proC`` on LB with 5% w/v
ethanol), p1k_00015 (wild type, anaerobic with 20 mM KNO3 and a bare ``glucose``). Each
other sample hits one rule: 6 evolved, 7 BW25113, 8 pColi, 9 crp activating-region
deletion, 10 rpoB substitution, 16 W3110, 17 DGF-298 (genotype stage); 11 CAMHB,
12 chemostat, 13 blank electron acceptor, 14 ``Acetoacetate/LiCl(10mM)`` (environment
stage). Sample ``k`` has TPM ``[1, 2, 3, 0.5, 1.5, 1, 1] * 1e5`` rotated left by ``k mod
7`` over b0001..b0007, so its seven genes sum to exactly 1e6; b0004 and b0007 exist only in
the pre-filter matrix, so a stored sample sums to 1e6 minus their share. Sample ``k``'s
count of gene ``i`` (0-based over b0001..b0007) is ``10 (i + 1) + k``.

The data-gated tests (``--data``) audit every ``SourcedValue`` against the pinned paper
and SI mirror, re-hash the raw mirror against the module pins, and pin the measured facts
of the dev-tree build: 241 records (121 wild type, 104 single, 12 double, 4 triple
deletions), the eleven drop rules, the pseudocount back-solve (921 of 1,035 samples within
1 TPM of 1e6, median 1,000,000.0000000008), and the locus-tag histogram (4,155 current, 99
pseudogene, 3 retired; b3681 stored as b4556); and run the RNA-seq verifier, the shared
record rules and the MG1655 gene containment over the stored records.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import os.path as osp
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import torchcell.datasets.ecoli.lamoureux2023 as L
from tests.torchcell.sequence.genome._bacterial_fixtures import (
    MG1655_GAF,
    MG1655_LOCI,
    forbid_network,
    serve_tier,
    write_assembly,
)
from torchcell.data import ManifestPinMismatchError
from torchcell.datamodels.schema import (
    AssemblyReferenceGenome,
    BacterialDeletionPerturbation,
    ComponentDefinition,
    ConcentrationUnit,
    DoseBasis,
    EnvironmentPhysicalPerturbation,
    MediaComponentRole,
    PhysicalFactor,
    SmallMoleculePerturbation,
)
from torchcell.sequence.genome.ecoli.k12 import MG1655_ASSEMBLY, EcoliK12MG1655Genome
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)

G = L.GenotypeRule
E = L.EnvironmentRule


# --------------------------------------------------------------------------- #
# Genotype settling
# --------------------------------------------------------------------------- #
def _row(**cells: str) -> dict[str, str]:
    """A metadata row with the wild-type M9 control cells, overridden by ``cells``."""
    row = {
        L.COL_SAMPLE: "control__wt_glc__1",
        L.COL_STUDY: "Control",
        L.COL_PROJECT: "control",
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
    }
    row.update(cells)
    return row


@pytest.mark.parametrize(
    "cells,rule,deleted",
    [
        ({}, G.wild_type, ()),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 del_fur"},
            G.deletion,
            ("fur",),
        ),
        (
            {
                L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 del_ndh del_cydB del_appC"
            },
            G.deletion,
            ("ndh", "cydB", "appC"),
        ),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 delyieP"},
            G.deletion,
            ("yieP",),
        ),
        ({L.COL_DESCRIPTION: "Escherichia coli del_pdhR"}, G.deletion, ("pdhR",)),
        ({L.COL_EVOLVED: "Endpoint", L.COL_STRAIN: "BW25113"}, G.evolved_isolate, ()),
        ({L.COL_EVOLVED: "Midpoint"}, G.evolved_isolate, ()),
        ({L.COL_STRAIN: "BW25113"}, G.strain_bw25113, ()),
        ({L.COL_STRAIN: "W3110"}, G.strain_w3110, ()),
        ({L.COL_STRAIN: "DGF-298"}, G.strain_dgf298, ()),
        ({L.COL_STRAIN: "GMOS"}, G.strain_gmos, ()),
        (
            {
                L.COL_STUDY: "pColi",
                L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 hsTPH_E2K",
            },
            G.heterologous_expression_construct,
            (),
        ),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 crp_delAr1delAr2"},
            G.partial_gene_edit,
            (),
        ),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 rpoBE546V"},
            G.point_mutation_allele,
            (),
        ),
    ],
    ids=[
        "wild-type",
        "one-deletion",
        "three-deletions",
        "deletion-without-underscore",
        "deletion-without-strain-words",
        "evolved-before-strain",
        "midpoint-isolate",
        "bw25113",
        "w3110",
        "dgf298",
        "gmos",
        "pcoli",
        "crp-activating-regions",
        "rpob-substitution",
    ],
)
def test_settle_genotype_applies_the_first_matching_rule(
    cells: dict[str, str], rule: L.GenotypeRule, deleted: tuple[str, ...]
) -> None:
    verdict = L.settle_genotype(_row(**cells))
    assert (verdict.rule, verdict.deleted_symbols) == (rule, deleted)
    assert verdict.kept is (rule in {G.wild_type, G.deletion})


@pytest.mark.parametrize(
    "cells,message",
    [
        ({L.COL_STRAIN: "Nissle"}, "unknown Strain cell 'Nissle'"),
        ({L.COL_EVOLVED: "Yes"}, "unknown Evolved Sample cell 'Yes'"),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 BRCA"},
            "edit token 'BRCA' is of no known form",
        ),
        (
            {L.COL_DESCRIPTION: "Escherichia coli K-12 MG1655 del_fur del_fur"},
            "repeats a deletion",
        ),
        ({L.COL_DESCRIPTION: "Salmonella enterica"}, "unrecognized Strain Description"),
    ],
    ids=["unknown-strain", "unknown-evolved", "unknown-token", "repeat", "foreign"],
)
def test_settle_genotype_refuses_what_it_cannot_read(
    cells: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        L.settle_genotype(_row(**cells))


# --------------------------------------------------------------------------- #
# Amounts and environments
# --------------------------------------------------------------------------- #
GL = ConcentrationUnit.g_per_l


@pytest.mark.parametrize(
    "cell,default,expected",
    [
        ("glucose(2)", GL, ("glucose", 2.0, GL, None)),
        ("glucose(.2%)", GL, ("glucose", 0.2, ConcentrationUnit.percent_w_v, None)),
        ("glucose", GL, ("glucose", None, None, None)),
        ("NH4Cl(1)", GL, ("ammonium chloride", 1.0, GL, None)),
        (
            "ceftriaxone(4.1mg/mL)",
            None,
            ("ceftriaxone", 4100.0, ConcentrationUnit.ug_per_ml, None),
        ),
        (
            "adenine (100mg/L)",
            None,
            ("adenine", 100.0, ConcentrationUnit.ug_per_ml, None),
        ),
        ("Ethanol(5%w/v)", None, ("Ethanol", 5.0, ConcentrationUnit.percent_w_v, None)),
        (
            "FeCl3 (20 uM)",
            None,
            ("iron(III) chloride", 20.0, ConcentrationUnit.micromolar, None),
        ),
        ("NaCl (0.3M)", None, ("NaCl", 0.3, ConcentrationUnit.molar, None)),
        ("uracil (1 mM)", None, ("uracil", 1.0, ConcentrationUnit.millimolar, None)),
        ("lactic acid(5)", None, ("lactic acid", None, None, "5")),
        ("adenosine()", None, ("adenosine", None, None, None)),
    ],
)
def test_parse_amount_keeps_the_stated_dose_and_never_invents_a_unit(
    cell: str,
    default: ConcentrationUnit | None,
    expected: tuple[str, float | None, ConcentrationUnit | None, str | None],
) -> None:
    amount = L.parse_amount(cell, default_unit=default)
    assert amount.label == cell
    assert (amount.name, amount.value, amount.unit, amount.unitless_value) == expected


def test_parse_amount_refuses_an_unknown_unit() -> None:
    with pytest.raises(ValueError, match="unknown unit 'mg/g'"):
        L.parse_amount("salt(3mg/g)", default_unit=None)


def test_parse_supplements_splits_a_plus_into_two_compounds() -> None:
    names = [a.name for a in L.parse_supplements("cytidine (1mM) + arginine (5mM)")]
    assert names == ["cytidine", "arginine"]
    assert L.parse_supplements("") == ()


@pytest.mark.parametrize(
    "cells,rule",
    [
        ({L.COL_MEDIA: "CAMHB"}, E.medium_not_in_library),
        ({L.COL_MEDIA: "01xLB"}, E.medium_not_in_library),
        ({L.COL_CULTURE: "Chemostat"}, E.culture_not_batch),
        ({L.COL_CULTURE: "Fed-batch"}, E.culture_not_batch),
        ({L.COL_ACCEPTOR: ""}, E.oxygen_regime_not_stated),
        ({L.COL_ACCEPTOR: "KNO3(20mM)"}, E.oxygen_regime_not_stated),
        ({L.COL_SUPPLEMENT: "Acetoacetate/LiCl(10mM)"}, E.supplement_label_ambiguous),
    ],
)
def test_settle_environment_names_the_rule_that_drops_a_sample(
    cells: dict[str, str], rule: L.EnvironmentRule
) -> None:
    verdict = L.settle_environment(_row(**cells))
    assert (verdict.rule, verdict.spec) == (rule, None)


def test_settle_environment_refuses_an_m9_row_without_a_carbon_source() -> None:
    with pytest.raises(ValueError, match="an M9 row with no carbon or nitrogen source"):
        L.settle_environment(_row(**{L.COL_CARBON: ""}))


def test_an_anaerobic_condition_keeps_its_named_electron_acceptor() -> None:
    spec = L.settle_environment(
        _row(**{L.COL_ACCEPTOR: "KNO3(20mM)", L.COL_CONDITION: "no3_anaero"})
    ).spec
    assert spec is not None
    assert spec.aerobicity == "anaerobic"
    assert spec.electron_acceptor == L.Amount(
        label="KNO3(20mM)",
        name="potassium nitrate",
        value=20.0,
        unit=ConcentrationUnit.millimolar,
    )
    blank = L.settle_environment(
        _row(**{L.COL_ACCEPTOR: "", L.COL_CONDITION: "wt_glc_anaero"})
    ).spec
    assert blank is not None
    assert (blank.aerobicity, blank.electron_acceptor) == ("anaerobic", None)


def test_build_environment_types_every_metadata_cell() -> None:
    row = _row(
        **{
            L.COL_CARBON: "glucose",
            L.COL_SUPPLEMENT: "PQ (250uM) + lactic acid(5)",
            L.COL_ANTIBIOTIC: "Kanamycin (50 ug/mL)",
            L.COL_PH: "5.5",
        }
    )
    spec = L.settle_environment(row).spec
    assert spec is not None
    env = L.build_environment(spec)
    assert env.temperature is not None and env.temperature.value == 37.0
    assert env.aerobicity == "aerobic"
    assert [g.field for g in env.provenance_gaps] == ["duration_hours"]

    carbon, nitrogen, ph, paraquat, lactic = env.perturbations
    assert isinstance(carbon, EnvironmentPhysicalPerturbation)
    assert (carbon.factor, carbon.magnitude) == (PhysicalFactor.carbon_source, None)
    assert carbon.agent is not None and carbon.agent.name == "D-glucose"
    assert [(g.field, g.reason) for g in carbon.provenance_gaps] == [
        ("magnitude", ProvenanceGapReason.not_reported_by_primary)
    ]
    assert isinstance(nitrogen, EnvironmentPhysicalPerturbation)
    assert nitrogen.factor is PhysicalFactor.nitrogen_source
    assert nitrogen.magnitude is not None
    assert (nitrogen.magnitude.value, nitrogen.magnitude.unit) == (1.0, GL)
    assert nitrogen.agent is not None and nitrogen.agent.name == "ammonium chloride"
    assert isinstance(ph, EnvironmentPhysicalPerturbation)
    assert ph.magnitude is not None
    assert (ph.factor, ph.magnitude.value, ph.magnitude.unit) == (
        PhysicalFactor.ph,
        5.5,
        ConcentrationUnit.ph,
    )
    assert [g.field for g in ph.provenance_gaps] == ["agent"]
    assert isinstance(paraquat, SmallMoleculePerturbation)
    assert paraquat.compound.name == "PQ"
    assert (paraquat.concentration.value, paraquat.concentration.unit) == (
        250.0,
        ConcentrationUnit.micromolar,
    )
    assert isinstance(lactic, SmallMoleculePerturbation)
    assert (lactic.concentration.value, lactic.concentration.basis) == (
        None,
        DoseBasis.fixed,
    )

    media = env.media
    assert (media.base_medium, media.is_synthetic, media.state) == (
        "M9",
        True,
        "liquid",
    )
    assert media.name == (
        "M9 (PRECISE-1K; recipe not stated) + sauer trace element mixture + "
        "Kanamycin (50 ug/mL)"
    )
    base, trace, kanamycin = media.components
    assert (base.definition, base.role) == (
        ComponentDefinition.composition_deferred,
        MediaComponentRole.other,
    )
    assert (trace.compound.name, trace.definition, trace.role) == (
        "sauer trace element mixture",
        ComponentDefinition.composition_deferred,
        MediaComponentRole.trace_element,
    )
    assert kanamycin.role is MediaComponentRole.selection_agent
    assert kanamycin.concentration is not None
    assert (kanamycin.concentration.value, kanamycin.concentration.unit) == (
        50.0,
        ConcentrationUnit.ug_per_ml,
    )
    assert [p.quote for p in kanamycin.provenance] == ["Kanamycin (50 ug/mL)"]


def test_identical_cells_give_one_environment_and_different_cells_two() -> None:
    first = L.settle_environment(_row()).spec
    again = L.settle_environment(
        _row(**{L.COL_SAMPLE: "other", L.COL_PROJECT: "x"})
    ).spec
    other = L.settle_environment(_row(**{L.COL_CARBON: "glucose(4)"})).spec
    assert first is not None and again is not None and other is not None
    assert L.build_environment(first) == L.build_environment(again)
    assert L.build_environment(first) != L.build_environment(other)


def test_lb_is_a_complex_medium_deriving_from_the_library_lb() -> None:
    spec = L.settle_environment(
        _row(
            **{L.COL_MEDIA: "LB", L.COL_CARBON: "", L.COL_NITROGEN: "", L.COL_TRACE: ""}
        )
    ).spec
    assert spec is not None
    env = L.build_environment(spec)
    assert (env.media.base_medium, env.media.is_synthetic) == ("LB", False)
    assert [type(p).__name__ for p in env.perturbations] == [
        "EnvironmentPhysicalPerturbation"
    ]


# --------------------------------------------------------------------------- #
# Expression values
# --------------------------------------------------------------------------- #
def test_the_phenotype_inverts_log2_tpm_plus_one_exactly() -> None:
    phenotype = L.rnaseq_phenotype(
        ["b0001", "b0002"],
        np.array([0.0, math.log2(7.0)]),
        np.array([0, 12], dtype=np.int64),
    )
    assert dict(phenotype.expression_tpm) == {"b0001": 0.0, "b0002": 6.0}
    assert dict(phenotype.expression_count) == {"b0001": 0, "b0002": 12}
    assert phenotype.measurement_type == "rnaseq_tpm"
    assert phenotype.n_mapped_reads is None
    assert [g.field for g in phenotype.provenance_gaps] == ["n_mapped_reads"]


def _log_frame(tpm: dict[str, list[float]], genes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {s: np.log2(np.array(v) + 1.0) for s, v in tpm.items()}, index=genes
    )


def test_back_solve_accepts_a_tpm_and_refuses_a_shifted_pseudocount() -> None:
    genes = ["g1", "g2", "g3"]
    prefilter = _log_frame({"s1": [5e5, 4e5, 1e5], "s2": [1e5, 1e5, 8e5]}, genes)
    result = L.back_solve_pseudocount(prefilter.loc[["g1", "g2"]], prefilter)
    assert result.n_samples_within_tolerance == 2
    assert result.median_total_tpm == pytest.approx(1e6, abs=1e-6)
    shifted = prefilter + 1e-3
    with pytest.raises(RuntimeError, match="median per-sample sum"):
        L.back_solve_pseudocount(shifted.loc[["g1"]], shifted)
    altered = prefilter.loc[["g1"]] + 1.0
    with pytest.raises(RuntimeError, match="differ from the pre-filter matrix"):
        L.back_solve_pseudocount(altered, prefilter)


# --------------------------------------------------------------------------- #
# Synthetic release: mirror, download, and a full build
# --------------------------------------------------------------------------- #
RELEASED_GENES = ["b0001", "b0002", "b0003", "b0005", "b0006"]
PREFILTER_GENES = ["b0001", "b0002", "b0003", "b0004", "b0005", "b0006", "b0007"]
TPM_SHAPE = [1.0, 2.0, 3.0, 0.5, 1.5, 1.0, 1.0]  # sums to 10 -> scaled to 1e6


def _metadata_rows() -> dict[str, dict[str, str]]:
    def row(sample: str, full: str, **cells: str) -> dict[str, str]:
        project, condition = full.split(":")
        base = _row(
            **{
                L.COL_SAMPLE: f"{project}__{condition}__x",
                L.COL_PROJECT: project,
                L.COL_CONDITION: condition,
            }
        )
        base.update(
            {
                L.COL_FULL_NAME: full,
                L.COL_REP: "1",
                L.COL_REPLICATES: "2.0",
                L.COL_PROJECT_REFERENCE: "p1k_00001;p1k_00002",
            }
        )
        base.update(cells)
        return base

    mg = "Escherichia coli K-12 MG1655"
    return {
        "p1k_00001": row("p1k_00001", "control:wt_glc"),
        "p1k_00002": row("p1k_00002", "control:wt_glc", **{L.COL_REP: "2"}),
        "p1k_00003": row(
            "p1k_00003",
            "ytf:delthrA",
            **{
                L.COL_DESCRIPTION: f"{mg} del_thrA",
                L.COL_ANTIBIOTIC: "Kanamycin (50 ug/mL)",
            },
        ),
        "p1k_00004": row(
            "p1k_00004",
            "ytf:delthrA",
            **{
                L.COL_DESCRIPTION: f"{mg} del_thrA",
                L.COL_ANTIBIOTIC: "Kanamycin (50 ug/mL)",
                L.COL_REP: "2",
            },
        ),
        "p1k_00005": row(
            "p1k_00005",
            "tcs:delpro_etoh",
            **{
                L.COL_DESCRIPTION: f"{mg} delproB del_proC",
                L.COL_MEDIA: "LB",
                L.COL_CARBON: "",
                L.COL_NITROGEN: "",
                L.COL_TRACE: "",
                L.COL_SUPPLEMENT: "Ethanol(5%w/v)",
                L.COL_REPLICATES: "1.0",
            },
        ),
        "p1k_00006": row("p1k_00006", "glu:ale", **{L.COL_EVOLVED: "Endpoint"}),
        "p1k_00007": row("p1k_00007", "omics:bw", **{L.COL_STRAIN: "BW25113"}),
        "p1k_00008": row(
            "p1k_00008",
            "pcoli:BRCA",
            **{L.COL_STUDY: "pColi", L.COL_DESCRIPTION: f"{mg} BRCA"},
        ),
        "p1k_00009": row(
            "p1k_00009", "crp:ar1", **{L.COL_DESCRIPTION: f"{mg} crp_delAr1"}
        ),
        "p1k_00010": row(
            "p1k_00010", "rpoB:e546v", **{L.COL_DESCRIPTION: f"{mg} rpoBE546V"}
        ),
        "p1k_00011": row("p1k_00011", "abx:camhb", **{L.COL_MEDIA: "CAMHB"}),
        "p1k_00012": row("p1k_00012", "rpoB:wt_031", **{L.COL_CULTURE: "Chemostat"}),
        "p1k_00013": row("p1k_00013", "minicoli:glc", **{L.COL_ACCEPTOR: ""}),
        "p1k_00014": row(
            "p1k_00014",
            "tcs:wt_LiAcet",
            **{L.COL_SUPPLEMENT: "Acetoacetate/LiCl(10mM)"},
        ),
        "p1k_00015": row(
            "p1k_00015",
            "minspan:no3_anaero",
            **{
                L.COL_CONDITION: "no3_anaero",
                L.COL_ACCEPTOR: "KNO3(20mM)",
                L.COL_CARBON: "glucose",
                L.COL_REPLICATES: "1.0",
            },
        ),
        "p1k_00016": row("p1k_00016", "w3110:wt", **{L.COL_STRAIN: "W3110"}),
        "p1k_00017": row("p1k_00017", "minicoli:dgf", **{L.COL_STRAIN: "DGF-298"}),
    }


def _tpm(sample: str) -> list[float]:
    """Sample k's TPM over PREFILTER_GENES: the shape rotated by k, summing to 1e6."""
    k = int(sample[-5:])
    shape = TPM_SHAPE[k % 7 :] + TPM_SHAPE[: k % 7]
    return [v * 1e5 for v in shape]


def _release_files(directory: Path) -> dict[str, bytes]:
    """The four synthetic release files, written under ``directory``; name -> bytes."""
    rows = _metadata_rows()
    samples = sorted(rows)
    metadata = pd.DataFrame.from_dict(rows, orient="index")
    prefilter = pd.DataFrame(
        {s: np.log2(np.array(_tpm(s)) + 1.0) for s in samples}, index=PREFILTER_GENES
    )
    counts = pd.DataFrame(
        {
            s: [10 * (i + 1) + int(s[-2:]) for i in range(len(PREFILTER_GENES))]
            for s in [*samples, "p1k_09999"]
        },
        index=PREFILTER_GENES,
    )
    frames = {
        L.METADATA.name: metadata,
        L.LOG_TPM_PREFILTER.name: prefilter,
        L.LOG_TPM.name: prefilter.loc[RELEASED_GENES],
        L.COUNTS.name: counts,
    }
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, bytes] = {}
    for name, frame in frames.items():
        frame.to_csv(directory / name)
        out[name] = (directory / name).read_bytes()
    return out


def _pin(monkeypatch: pytest.MonkeyPatch, files: dict[str, bytes]) -> None:
    """Point the module pins at the synthetic bytes."""
    pinned = tuple(
        raw.model_copy(update={"sha256": hashlib.sha256(files[raw.name]).hexdigest()})
        for raw in L.RAW_FILES
    )
    monkeypatch.setattr(L, "RAW_FILES", pinned)


def _archive(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for raw in L.RAW_FILES:
            archive.writestr(raw.member, files[raw.name])
        archive.writestr(f"{L.ARCHIVE_PREFIX}README.md", b"not consumed")
    return path


def test_deposit_writes_the_consumed_members_and_their_retrieval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "precise1k-v1.0.zip", files)
    monkeypatch.setattr(
        L, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    root = L.deposit_raw_mirror(archive_path=archive, data_root=str(tmp_path / "dr"))
    assert root == tmp_path / "dr" / "torchcell-raw" / L.CITATION_KEY
    assert sorted(
        p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
    ) == [
        "data/precise1k/counts.csv",
        "data/precise1k/log_tpm_qc.csv",
        "data/precise1k/log_tpm_qc_w_short_low_fpkm.csv",
        "data/precise1k/metadata_qc.csv",
        "manifest.json",
    ]
    manifest = L.load_manifest(str(tmp_path / "dr"))
    record = manifest.files[2]
    assert record.path == L.METADATA.relpath
    assert record.retrieval is not None
    assert record.retrieval.model_dump(exclude={"last_check"}) == {
        "method": "zenodo",
        "source_url": L.ARCHIVE_URL,
        "retriever": "torchcell.literature.retrieve.zip_member",
        "params": {
            "url": L.ARCHIVE_URL,
            "member": "SBRG-precise1k-71e1157/data/precise1k/metadata_qc.csv",
            "container_sha256": L.ARCHIVE_SHA256,
        },
        "sha256": hashlib.sha256(files[L.METADATA.name]).hexdigest(),
        "retrieved_at": "2026-10-07",
    }
    # Idempotent: a second deposit leaves the files alone.
    L.deposit_raw_mirror(archive_path=archive, data_root=str(tmp_path / "dr"))
    (root / L.COUNTS.relpath).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="exists with a different sha256; refusing"):
        L.deposit_raw_mirror(archive_path=archive, data_root=str(tmp_path / "dr"))


def test_deposit_refuses_an_archive_off_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "a.zip", files)
    with pytest.raises(RuntimeError, match="sha256 mismatch: got"):
        L.deposit_raw_mirror(archive_path=archive, data_root=str(tmp_path / "dr"))


def _bare_dataset(root: Path) -> L.RnaseqLamoureux2023Dataset:
    dataset = L.RnaseqLamoureux2023Dataset.__new__(L.RnaseqLamoureux2023Dataset)
    dataset.root = str(root)
    return dataset


def test_download_links_each_verified_member_and_refuses_an_off_pin_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = _release_files(tmp_path / "src")
    _pin(monkeypatch, files)
    archive = _archive(tmp_path / "a.zip", files)
    monkeypatch.setattr(
        L, "ARCHIVE_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    mirror = L.deposit_raw_mirror(archive_path=archive)
    dataset = _bare_dataset(tmp_path / "build")
    dataset.download()
    for raw in L.RAW_FILES:
        assert os.readlink(tmp_path / "build" / "raw" / raw.name) == str(
            mirror / raw.relpath
        )
    manifest = json.loads((mirror / "manifest.json").read_text())
    manifest["files"][0]["sha256"] = "ab" * 32
    (mirror / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ManifestPinMismatchError):
        _bare_dataset(tmp_path / "build2").download()


MG1655_PIN = AssemblyReferenceGenome(
    species="Escherichia coli",
    strain="MG1655",
    assembly_set="ecoli_K12_MG1655_ASM584v2",
    assembly_accession="GCA_000005845.2",
)


@pytest.fixture
def built(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> L.RnaseqLamoureux2023Dataset:
    """The synthetic release built end to end over the synthetic MG1655 genome."""
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_lamoureux2023"
    files = _release_files(root / "raw")
    _pin(monkeypatch, files)
    monkeypatch.setattr(L, "assembly_reference", lambda strain: MG1655_PIN)
    return L.RnaseqLamoureux2023Dataset(root=str(root), ecoli_genome=genome)


def _json(dataset: L.RnaseqLamoureux2023Dataset, name: str) -> Any:
    return json.loads(Path(dataset.preprocess_dir, name).read_text())


def test_the_build_keeps_wild_type_and_deletions_and_counts_every_drop(
    built: L.RnaseqLamoureux2023Dataset,
) -> None:
    assert len(built) == 6
    assert _json(built, "record_samples.json") == [
        "p1k_00001",
        "p1k_00002",
        "p1k_00003",
        "p1k_00004",
        "p1k_00005",
        "p1k_00015",
    ]
    drop_log = _json(built, "dropped_records.json")
    assert {
        k: drop_log[k] for k in ("source_records", "kept_records", "dropped_records")
    } == {"source_records": 17, "kept_records": 6, "dropped_records": 11}
    assert [(r["stage"], r["rule"], r["samples"]) for r in drop_log["rules"]] == [
        ("genotype", "evolved_isolate", ["p1k_00006"]),
        ("genotype", "strain_bw25113", ["p1k_00007"]),
        ("genotype", "heterologous_expression_construct", ["p1k_00008"]),
        ("genotype", "partial_gene_edit", ["p1k_00009"]),
        ("genotype", "point_mutation_allele", ["p1k_00010"]),
        ("genotype", "strain_w3110", ["p1k_00016"]),
        ("genotype", "strain_dgf298", ["p1k_00017"]),
        ("environment", "medium_not_in_library", ["p1k_00011"]),
        ("environment", "culture_not_batch", ["p1k_00012"]),
        ("environment", "oxygen_regime_not_stated", ["p1k_00013"]),
        ("environment", "supplement_label_ambiguous", ["p1k_00014"]),
    ]
    assert drop_log["kept_by_genotype_class"] == {
        "deletion_1": 2,
        "deletion_2": 1,
        "wild_type": 3,
    }
    assert drop_log["kept_by_base_media"] == {"LB": 1, "M9": 5}


def test_a_deletion_record_names_the_b_number_and_the_genome_s_symbol(
    built: L.RnaseqLamoureux2023Dataset,
) -> None:
    thra = built[2]["experiment"]["genotype"]["perturbations"]
    double = built[4]["experiment"]["genotype"]["perturbations"]
    assert built[0]["experiment"]["genotype"]["perturbations"] == []
    assert [(p["systematic_gene_name"], p["perturbed_gene_name"]) for p in thra] == [
        ("b0002", "thrA")
    ]
    assert [(p["systematic_gene_name"], p["perturbed_gene_name"]) for p in double] == [
        ("b0005", "proB"),
        ("b0006", "proC"),
    ]
    perturbation = BacterialDeletionPerturbation.model_validate(thra[0])
    assert perturbation.gene_namespace == "ecoli_k12_mg1655_bnumber"
    assert perturbation.perturbation_type == "bacterial_deletion"


def test_the_phenotype_is_the_inverted_release_value_and_its_count(
    built: L.RnaseqLamoureux2023Dataset,
) -> None:
    phenotype = built[2]["experiment"]["phenotype"]
    expected = dict(zip(PREFILTER_GENES, _tpm("p1k_00003"), strict=True))
    assert list(phenotype["expression_tpm"]) == RELEASED_GENES
    for gene, value in phenotype["expression_tpm"].items():
        assert value == pytest.approx(expected[gene], rel=1e-12)
    assert sum(phenotype["expression_tpm"].values()) == pytest.approx(
        1e6 - expected["b0004"] - expected["b0007"], rel=1e-12
    )
    assert phenotype["expression_count"] == {
        "b0001": 13,
        "b0002": 23,
        "b0003": 33,
        "b0005": 53,
        "b0006": 63,
    }
    back_solve = _json(built, "pseudocount_back_solve.json")
    assert (back_solve["n_genes"], back_solve["n_samples"]) == (7, 17)
    assert back_solve["n_samples_within_tolerance"] == 17


def test_one_reference_holds_the_control_pair_s_centering_vector(
    built: L.RnaseqLamoureux2023Dataset,
) -> None:
    reference = built[3]["reference"]
    assert reference["genome_reference"] == MG1655_PIN.model_dump()
    index = built.experiment_reference_index
    assert index is not None and len(index) == 1
    assert reference["environment_reference"] == built[0]["experiment"]["environment"]
    one = np.log2(np.array(_tpm("p1k_00001")) + 1.0)
    two = np.log2(np.array(_tpm("p1k_00002")) + 1.0)
    centering = dict(zip(PREFILTER_GENES, np.exp2((one + two) / 2) - 1.0, strict=True))
    for gene, value in reference["phenotype_reference"]["expression_tpm"].items():
        assert value == pytest.approx(centering[gene], rel=1e-12)
    assert reference["phenotype_reference"]["expression_count"] == {
        "b0001": 12,  # mean of 11 and 12, rounded half to even
        "b0002": 22,
        "b0003": 32,
        "b0005": 52,
        "b0006": 62,
    }


def test_replicates_share_an_environment_and_are_grouped_by_condition(
    built: L.RnaseqLamoureux2023Dataset,
) -> None:
    groups = {g["full_name"]: g for g in _json(built, "replicate_groups.json")}
    assert {k: g["record_indices"] for k, g in groups.items()} == {
        "control:wt_glc": [0, 1],
        "minspan:no3_anaero": [5],
        "tcs:delpro_etoh": [4],
        "ytf:delthrA": [2, 3],
    }
    assert groups["ytf:delthrA"]["condition_cells"][L.COL_ANTIBIOTIC] == (
        "Kanamycin (50 ug/mL)"
    )
    assert (
        built[2]["experiment"]["environment"] == built[3]["experiment"]["environment"]
    )
    assert (
        built[0]["experiment"]["environment"] != built[2]["experiment"]["environment"]
    )
    anaerobic = built[5]["experiment"]["environment"]
    assert anaerobic["aerobicity"] == "anaerobic"
    assert anaerobic["perturbations"][3]["compound"]["name"] == "potassium nitrate"
    assert anaerobic["perturbations"][3]["concentration"]["value"] == 20.0


def test_the_build_refuses_a_control_pair_that_names_another_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    tier = write_assembly(tmp_path / "tier", MG1655_ASSEMBLY, MG1655_LOCI, MG1655_GAF)
    forbid_network(monkeypatch)
    serve_tier(monkeypatch, tier)
    (tmp_path / "mg1655").mkdir()
    genome = EcoliK12MG1655Genome(genome_root=str(tmp_path / "mg1655"), overwrite=False)
    root = tmp_path / "rnaseq_lamoureux2023"
    files = _release_files(root / "raw")
    metadata = pd.read_csv(root / "raw" / L.METADATA.name, index_col=0, dtype=str)
    metadata.loc["p1k_00002", L.COL_PROJECT_REFERENCE] = "p1k_00002"
    metadata.to_csv(root / "raw" / L.METADATA.name)
    files[L.METADATA.name] = (root / "raw" / L.METADATA.name).read_bytes()
    _pin(monkeypatch, files)
    monkeypatch.setattr(L, "assembly_reference", lambda strain: MG1655_PIN)
    with pytest.raises(RuntimeError, match="control sample p1k_00002 names reference"):
        L.RnaseqLamoureux2023Dataset(root=str(root), ecoli_genome=genome)


# --------------------------------------------------------------------------- #
# The real mirror and the dev-tree build (--data)
# --------------------------------------------------------------------------- #
def _data_root() -> str:
    return os.environ["DATA_ROOT"]


@pytest.mark.data
def test_every_sourced_value_is_a_verbatim_quote_in_the_pinned_mirror() -> None:
    library = osp.join(_data_root(), "torchcell-library")
    values = [
        getattr(L, name)
        for name in dir(L)
        if isinstance(getattr(L, name), SourcedValue)
    ]
    assert len(values) == 14
    assert {v.provenance.source_uri for v in values} == {L.PAPER_MD, L.SI_MD}
    for value in values:
        result = audit_sourced_value(value, library)
        assert result.passed, (value.quote, result.message)


@pytest.mark.data
def test_the_raw_mirror_holds_exactly_the_pinned_members() -> None:
    root = L.raw_mirror_dir(_data_root())
    manifest = L.load_manifest(_data_root())
    assert [(f.path, f.sha256) for f in manifest.files] == [
        (raw.relpath, raw.sha256) for raw in L.RAW_FILES
    ]
    for raw in L.RAW_FILES:
        digest = hashlib.sha256((root / raw.relpath).read_bytes()).hexdigest()
        assert digest == raw.sha256


def _built_root() -> Path:
    root = Path(_data_root()) / "data/torchcell/rnaseq_lamoureux2023"
    if not (root / "processed" / "lmdb").is_dir():
        pytest.skip("the dev-tree LMDB is not built")
    return root


@pytest.mark.data
def test_the_dev_tree_build_pins_the_measured_counts() -> None:
    root = _built_root()
    drop_log = json.loads((root / "preprocess/dropped_records.json").read_text())
    assert (drop_log["source_records"], drop_log["kept_records"]) == (1035, 241)
    assert {r["rule"]: r["n_records"] for r in drop_log["rules"]} == {
        "evolved_isolate": 421,
        "strain_bw25113": 148,
        "heterologous_expression_construct": 94,
        "strain_dgf298": 26,
        "strain_w3110": 26,
        "point_mutation_allele": 9,
        "partial_gene_edit": 8,
        "medium_not_in_library": 53,
        "supplement_label_ambiguous": 4,
        "culture_not_batch": 3,
        "oxygen_regime_not_stated": 2,
    }
    assert drop_log["kept_by_genotype_class"] == {
        "deletion_1": 104,
        "deletion_2": 12,
        "deletion_3": 4,
        "wild_type": 121,
    }
    assert drop_log["kept_by_base_media"] == {"LB": 32, "M9": 209}
    back_solve = json.loads(
        (root / "preprocess/pseudocount_back_solve.json").read_text()
    )
    assert (back_solve["n_genes"], back_solve["n_samples"]) == (4355, 1035)
    assert back_solve["n_samples_within_tolerance"] == 921
    reconciliation = json.loads(
        (root / "preprocess/locus_tag_reconciliation.json").read_text()
    )
    expression = reconciliation["expression_genes"]
    assert expression["status_histogram"] == {
        "current": 4155,
        "renamed": 0,
        "non_gene_feature": 99,
        "retired": 3,
        "ambiguous": 0,
    }
    assert expression["retired_kept"] == ["b3036", "b4223", "b4590"]
    assert reconciliation["deleted_genes"]["unique_names"] == 42
    assert reconciliation["deleted_symbol_to_locus"]["ydhB"] == ["b1659", "punR"]


@pytest.mark.data
def test_the_dev_tree_build_passes_the_record_rules_and_never_merges_two_conditions() -> (
    None
):
    """L0-L4 over the stored records, and the replicate structure the records encode.

    ``strain_uniqueness`` is the one result expected to fail: it was written for one
    record per isolate (Caudal) and requires a ``strain_id`` on a perturbation, which a
    wild-type record (no perturbation) and a bacterial deletion do not carry, and which a
    per-replicate record could not satisfy anyway. Its replacement here is the check that
    matters for replicates: records sharing (genotype, environment) are replicates only
    if their release conditions carry identical cells.
    """
    import lmdb

    from torchcell.data.experiment_dataset import resolve_interned
    from torchcell.verification.common import shared_rule_results
    from torchcell.verification.report import Provenance
    from torchcell.verification.rnaseq import rnaseq_gene_set, verify_rnaseq_dataset
    from torchcell.verification.runners import (
        _gene_set_for_reference,
        _genome_for_reference,
        _load_interned,
    )

    root = _built_root()
    interned = _load_interned(str(root))
    env = lmdb.open(str(root / "processed" / "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        by_index = {
            int(key.decode()): resolve_interned(pickle.loads(value), interned)
            for key, value in txn.cursor()
        }
    env.close()
    records = [by_index[i] for i in range(len(by_index))]
    reference = records[0]["reference"]["genome_reference"]
    genes = _gene_set_for_reference(reference, _data_root())
    genome = _genome_for_reference(reference, _data_root())
    report = verify_rnaseq_dataset(
        records,
        dataset_name="rnaseq_lamoureux2023",
        provenance=Provenance(
            source_uri=L.ZENODO_RECORD_URL, citation_key=L.CITATION_KEY
        ),
        expected_count=241,
    )
    results = report.results + shared_rule_results(
        records, resolve_gene_name=genome.resolve_gene_name, sgd_genes=genes
    )
    assert {r.name for r in results if not r.passed} == {"strain_uniqueness"}
    measured = rnaseq_gene_set(records)
    assert (len(measured), sorted(measured - genes)) == (
        4257,
        ["b3036", "b4223", "b4590"],
    )

    samples = json.loads((root / "preprocess/record_samples.json").read_text())
    groups = json.loads((root / "preprocess/replicate_groups.json").read_text())
    cells = {
        sample: json.dumps(group["condition_cells"], sort_keys=True)
        for group in groups
        for sample in group["samples"]
    }
    merged: dict[str, set[str]] = {}
    for index, sample in enumerate(samples):
        experiment = records[index]["experiment"]
        key = json.dumps(
            [experiment["genotype"], experiment["environment"]],
            sort_keys=True,
            default=str,
        )
        merged.setdefault(key, set()).add(cells[sample])
    assert (len(groups), len(merged)) == (121, 114)
    assert all(len(distinct) == 1 for distinct in merged.values())
