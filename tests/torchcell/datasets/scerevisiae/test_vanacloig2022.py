# tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
# [[tests.torchcell.datasets.scerevisiae.test_vanacloig2022]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datasets/scerevisiae/test_vanacloig2022.py
"""Vanacloig-Pedros 2022 loader: TMM ratio, stored-as-released ledger, DMSO, MBO.

Everything but the mirror audit runs on synthetic count matrices, so no build tree, no
genome and no network are needed. The audit test binds each module-level
``SourcedValue`` to its verbatim quote in the sha256-pinned mirror and skips when the
mirror is not mounted.

The GEO-layout fixture (``_geo_matrix``): a gzipped TSV in the GSE186866 layout (``gene``
= ``<ORF>_<barcode>``, ``std_name``, then ``ControlN_CG00b`` and ``<token>_CG00b_repN``
count columns) read by the real ``_load_matrix``. Two batches with two controls each;
six tokens (Furfural, MMS, SodiumGlyoxylate, DMSO, MBO, QUADRIS1) with three replicate
columns each (batches 001, 002, 001, except MMS 001, 002, 002). Seven library rows:
YAL001C, YAL002W, YAL003W (an ORF a later annotation folds into YAL002W), YPL999C (an
ORF no annotation lists, released std_name ``NOPE``), YGL013C (PDR1, a locus the SGA
background also deleted), YBR001C (one NaN count, in Control1_CG001), YBR002C (no
barcode); 60 non-ORF ``Const`` rows holding 1,000 reads in every column; and a non-ORF
``Filler`` row whose counts bring EVERY column total to 1,000,000.

Why the closed forms survive TMM: every column has the same library size, so a
``Const`` row's log ratio against any reference is exactly 0. Those 60 tied rows sit in
the middle of the log-ratio ranks, inside edgeR's 30% trim, and no nonzero row does, so
every trimmed weighted mean is exactly 0, every TMM factor is exactly 1, and TMM-CPM
equals the raw count (pinned by ``test_geo_fixture_tmm_factors_are_exactly_one``).

Expected values, with ``L(x) = log2(x + 1)``:

- YAL001C / Furfural (paired): controls CG001 (3, 3) -> L = 2, CG002 (7, 7) -> L = 3;
  replicates 15 (CG001), 63 (CG002), 63 (CG001) -> 4 - 2, 6 - 3, 6 - 2 = 2, 3, 4;
  response 3.0, sample SD 1.0, SE 1/sqrt(3).
- YAL001C / MMS (pooled over all four controls, mean 5 -> log2 6): replicates 11, 23,
  47 -> 1, 2, 3; response 2.0, SD 1.0.
- DMSO, MBO and QUADRIS1 hold 5 reads in every replicate of every library row: YAL001C
  -> paired DMSO and MBO L(5) - (2, 3, 2), response log2(6) - 7/3, SD sqrt(1/3);
  QUADRIS1 is pooled (mean 5), response 0, SD 0. YAL002W (controls 0) -> log2(6), SD 0;
  the three rows at 2 reads -> log2(6) - log2(3) = 1, SD 0; YBR002C (controls 1) ->
  log2(6) - 1, SD 0.
- YAL002W: Furfural 0, 1, 3 -> 0, 1, 2; response 1.0, SD 1.0. MMS and SodiumGlyoxylate
  are all-zero cells against zero controls: response L(0) - L(0) = 0.0, SD exactly 0,
  STORED.
- Every other Furfural / MMS / SodiumGlyoxylate cell has replicates equal to its
  controls: response 0.0, SD 0.

Records, condition-major over every released token sorted (DMSO, Furfural, MBO, MMS,
QUADRIS1, SodiumGlyoxylate), then the stored rows in matrix order (YAL001C, YAL002W,
YAL003W, YPL999C, YGL013C, YBR002C): six records per condition, 36 in all. YBR001C is a
stored library row, but its one missing count is a paired or pooled control of every
condition, so all six of its cells are dropped under ``a_count_the_cell_uses_is_missing``.
Ledger: source 68 rows x 6 tokens = 408; 61 non-ORF rows x 6 = 366 under
``gene_column_is_not_a_systematic_orf``; YBR001C's six cells; so 372 dropped and 36
stored; QUADRIS1 is the one ``compound_identity_gaps`` entry.

The genome-gate reversal (2026-10-12): no annotation renames, merges or refuses a row,
so the renamed, retired and background-locus rows are stored under their released ORF
and ``std_name``; an all-zero cell is stored with SD 0.0; the 11 Table-S1-only tokens are
stored at their Table S1 doses. Issue #501 pins kept: TMM equals edgeR's
``calcNormFactors`` (and a compositional takeover leaves no TMM offset); DMSO is a
served condition dosed at Table S1's 2.50% v/v (#764); the background is the SGA MATa
progeny with ONE perturbation per genotype (#500); MBO is 2-methyl-3-buten-2-ol, CID
8257; ammonium sulfate is a SynBase dropout. Issue #524: a missing count drops only the
cells that use it; a row with no barcode is stored with ``barcode None`` and
``BARCODE_ABSENT_GAP``.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import os.path as osp
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from torchcell.data import ManifestPinMismatchError, RawSha256MismatchError
from torchcell.datamodels.compound_identity import (
    resolve_compound_identity,
    resolved_compound,
)
from torchcell.datamodels.media import MEDIA_LIBRARY, SYNBASE
from torchcell.datamodels.schema import (
    AlleleEdit,
    AssayType,
    BarcodedKanMxDeletionPerturbation,
    ConcentrationUnit,
    CultureEnvironment,
    DoseBasis,
    EndpointRule,
    EnvironmentPhysicalPerturbation,
    EnvironmentResponsePhenotype,
    Genotype,
    MatingType,
    MeasurementType,
    PhysicalFactor,
    Publication,
    SampleUnit,
    SmallMoleculePerturbation,
    StrainEnvironmentResponseExperiment,
    StrainEnvironmentResponseExperimentReference,
    UncertaintyType,
)
from torchcell.datasets.scerevisiae import vanacloig2022 as v
from torchcell.verification.report import Provenance
from torchcell.verification.sourced import (
    ProvenanceGapReason,
    SourcedValue,
    audit_sourced_value,
)


def _matrix() -> pd.DataFrame:
    """Four library rows x (2 controls per batch + 3 conditions x 3 reps)."""
    rows = [
        ("YAL001C_AAAACCCCGGGGTTTTACGT", "TFC3"),
        ("YAL002W_CCCCGGGGTTTTAAAACGTA", "VPS8"),
        ("YAL003W_GGGGTTTTAAAACCCCGTAC", "YAL003W"),  # renamed later: stored as is
        ("YPL999C_TTTTAAAACCCCGGGGTACG", "NOPE"),  # in no annotation: stored as is
    ]
    data: dict[str, Any] = {
        "gene": [r[0] for r in rows],
        "std_name": [r[1] for r in rows],
        "Control1_CG003": [100, 200, 300, 400],
        "Control2_CG003": [120, 220, 320, 420],
        "Control1_CG004": [400, 300, 200, 100],
        "Control2_CG004": [420, 320, 220, 120],
        # Benomyl: one replicate per batch (paired), row 1 is an all-zero cell
        "Benomyl_CG003_rep1": [110, 0, 310, 410],
        "Benomyl_CG004_rep2": [410, 0, 210, 110],
        "Benomyl_CG003_rep3": [130, 0, 330, 430],
        # DMSO: a Fig 1B condition, served at Table S1's 2.50% v/v
        "DMSO_CG003_rep1": [90, 190, 290, 390],
        "DMSO_CG004_rep2": [390, 290, 190, 90],
        "DMSO_CG003_rep3": [95, 195, 295, 395],
        # MBO: a Fig 1B condition, adjudicated to 2-methyl-3-buten-2-ol
        "MBO_CG003_rep1": [70, 170, 270, 370],
        "MBO_CG004_rep2": [370, 270, 170, 70],
        "MBO_CG003_rep3": [75, 175, 275, 375],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def built(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build the synthetic dataset end to end into a temporary root."""
    monkeypatch.setattr(v.EnvChemgenVanacloig2022Dataset, "download", lambda self: None)
    monkeypatch.setattr(
        v.EnvChemgenVanacloig2022Dataset, "_load_matrix", lambda self: _matrix()
    )
    root = str(tmp_path / "env_chemgen_vanacloig2022")
    os.makedirs(osp.join(root, "raw"), exist_ok=True)
    with gzip.open(osp.join(root, "raw", v.DATA_FILENAME), "wt") as handle:
        handle.write("placeholder\n")  # presence only; _load_matrix is patched
    return v.EnvChemgenVanacloig2022Dataset(root=root)


def test_every_source_record_is_stored(built: Any) -> None:
    """Four rows x three conditions, the renamed, the unannotated and the all-zero
    Benomyl cell included: nothing is dropped and both rules are at 0.
    """
    log = v.DropLog.model_validate_json(
        open(osp.join(built.root, "preprocess", "dropped_records.json")).read()
    )
    assert [(r.rule, r.n_records) for r in log.rules] == [
        ("gene_column_is_not_a_systematic_orf", 0),
        ("a_count_the_cell_uses_is_missing", 0),
    ]
    assert [(n.rule, n.n_cells) for n in log.not_a_value] == [
        ("every_count_of_the_row_is_missing", 0)
    ]
    assert log.compound_identity_gaps == []
    assert (log.source_records, log.kept_records, log.dropped_records) == (12, 12, 0)
    assert len(built) == log.kept_records
    stored = {
        (
            built[i]["experiment"]["genotype"]["perturbations"][0][
                "systematic_gene_name"
            ],
            built[i]["experiment"]["genotype"]["perturbations"][0][
                "perturbed_gene_name"
            ],
        )
        for i in range(len(built))
    }
    assert stored == {
        ("YAL001C", "TFC3"),
        ("YAL002W", "VPS8"),
        ("YAL003W", "YAL003W"),
        ("YPL999C", "NOPE"),
    }


def test_record_carries_one_screened_deletion_and_the_shared_medium(built: Any) -> None:
    """#500 / #501 finding 4: the genotype is the ONE screened kanMX deletion."""
    record = built[0]
    experiment = record["experiment"]
    assert experiment["experiment_type"] == "strain_environment_response"
    (deletion,) = experiment["genotype"]["perturbations"]
    assert deletion["perturbation_type"] == "barcoded_kanmx_deletion"
    assert deletion["systematic_gene_name"] == "YAL001C"
    assert deletion["perturbed_gene_name"] == "TFC3"  # the release's std_name
    assert deletion["barcode"] == "AAAACCCCGGGGTTTTACGT"
    assert deletion["collection"] == v.LIBRARY_COLLECTION.value
    assert deletion["cassette"] == "kanMX"  # Piotrowski: 'MATa xxxΔ::kanMX'
    assert experiment["environment"]["media"] == MEDIA_LIBRARY["SYNBASE"].model_dump()
    assert experiment["environment"]["aerobicity"] == "anaerobic"
    assert experiment["environment"]["duration_hours"] == 48.0
    assert experiment["environment"]["duration_generations"] == 6.5


def test_environment_carries_the_compound_and_a_typed_ph(built: Any) -> None:
    perturbations = built[0]["experiment"]["environment"]["perturbations"]
    compound = next(
        p for p in perturbations if p["perturbation_type"] == "small_molecule"
    )
    assert compound["compound"]["name"] == "benomyl"
    assert compound["compound"]["inchikey"] == "RIOXQFHNBCKOKP-UHFFFAOYSA-N"
    # Piotrowski 2017's molar statement of Vanacloig's '10 ug/mL as previously published'
    assert compound["concentration"]["value"] == 34.4
    assert compound["concentration"]["unit"] == "uM"
    assert compound["concentration"]["basis"] == "fixed"
    # Table S1 says benomyl was not dissolved in DMSO: no vehicle, and no gap
    assert compound["solvent"] is None
    assert compound["provenance_gaps"] == []
    ph = next(
        p for p in perturbations if p["perturbation_type"] == "environment_physical"
    )
    assert ph["factor"] == "pH"
    assert ph["magnitude"]["value"] == 5.0 and ph["magnitude"]["unit"] == "pH"
    assert ph["agent"]["name"] == "hydrochloric acid"  # the acid the paper names
    assert ph["provenance_gaps"] == []


def test_reference_environment_holds_no_inhibitor_and_the_typed_background(
    built: Any,
) -> None:
    reference = built[0]["reference"]
    types = {
        p["perturbation_type"]
        for p in reference["environment_reference"]["perturbations"]
    }
    assert types == {"environment_physical"}  # pH only: the control is inhibitor-free
    assert reference["phenotype_reference"]["environment_response"] == 0.0
    genome = reference["genome_reference"]
    assert genome["strain"] == v.LIBRARY_STRAIN
    assert genome["ploidy"] == "haploid"
    assert genome["background"]["mating_type"] == "a"
    assert reference["experiment_reference_type"] == "strain_environment_response"


def test_unpaired_compound_keeps_the_pooled_control_in_its_units() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    assert "pooled rather than batch-matched" in dataset._units("MMS")
    assert "SAME CG batch" in dataset._units("Furfural")
    assert "TMM-normalized" in dataset._units("Furfural")
    # MMS's dose was published, not set to an IC30, and Table S1 writes no v/v or w/v
    mms = dataset._concentration("MMS")
    assert (mms.value, mms.unit, mms.basis) == (
        0.01,
        ConcentrationUnit.percent,
        DoseBasis.fixed,
    )
    assert dataset._concentration("Furfural").basis is DoseBasis.IC30
    dmso = dataset._concentration("DMSO")
    assert (dmso.value, dmso.unit, dmso.basis) == (
        2.5,
        ConcentrationUnit.percent_v_v,
        DoseBasis.IC30,
    )


def test_released_library_rows_store_every_orf_verbatim() -> None:
    """The ORF and std_name as released; no annotation is read."""
    frame = pd.DataFrame(
        {
            "gene": [
                "YAL003W_CCCC",  # an ORF a later annotation folds into another
                "YPL999C_GGGG",  # an ORF no annotation lists
                "YEL063C_AAAA",  # CAN1, a locus the SGA background also deleted
                "YBR002C",  # no barcode
                "YBR003W_TTTT",  # every count missing
                "Filler_row",  # not ORF-shaped
            ],
            "std_name": ["YAL003W", "NOPE", "CAN1", "", "COQ1", "none"],
            "a_CG001_rep1": [1.0, 2.0, 3.0, 4.0, float("nan"), 5.0],
        }
    )
    rows = v.released_library_rows(frame, ["a_CG001_rep1"])
    assert rows == v.LibraryRows(
        keep_mask=[True, True, True, True, False, False],
        systematic=["YAL003W", "YPL999C", "YEL063C", "YBR002C"],
        perturbed=["YAL003W", "NOPE", "CAN1", "YBR002C"],
        barcode=["CCCC", "GGGG", "AAAA", None],
        every_count_missing=["YBR003W_TTTT"],
        not_an_orf=["Filler_row"],
    )


def test_every_sourced_value_is_backed_by_a_verbatim_quote_in_the_mirror() -> None:
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    library = osp.join(data_root, "torchcell-library")
    if not osp.isdir(osp.join(library, v.CITATION_KEY)):
        pytest.skip("paper mirror not mounted")
    values = [
        getattr(v, name)
        for name in dir(v)
        if isinstance(getattr(v, name), SourcedValue)
    ] + list(v.TABLE_S1_DOSES.values())
    assert len(values) >= 64
    assert {d.provenance.source_uri for d in v.TABLE_S1_DOSES.values()} == {"si/si2.md"}
    citation_keys = {value.provenance.citation_key for value in values}
    assert citation_keys == {v.CITATION_KEY, v.PIOTROWSKI_KEY, v.OHNUKI_KEY}
    for value in values:
        assert audit_sourced_value(value, library).passed


def test_fig_1b_image_is_the_pinned_artifact() -> None:
    """The 34 Fig 1B bar labels are read from this exact image file."""
    data_root = os.environ.get("DATA_ROOT")
    if data_root is None:
        pytest.skip("DATA_ROOT not set")
    image = Path(data_root) / "torchcell-library" / v.CITATION_KEY / v.FIG_1B_IMAGE
    if not image.exists():
        pytest.skip("paper mirror not mounted")
    assert hashlib.sha256(image.read_bytes()).hexdigest() == v.FIG_1B_IMAGE_SHA256


def test_perturbation_leaves_are_the_typed_ones() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    genotype = dataset._genotype("YAL001C", "TFC3", "ACGT")
    assert len(genotype.perturbations) == 1
    assert isinstance(genotype.perturbations[0], BarcodedKanMxDeletionPerturbation)
    environment = dataset._environment("Furfural")
    assert isinstance(environment, CultureEnvironment)
    assert isinstance(environment.perturbations[0], SmallMoleculePerturbation)
    assert isinstance(environment.perturbations[1], EnvironmentPhysicalPerturbation)
    assert environment.perturbations[1].factor is PhysicalFactor.ph


# --------------------------------------------------------------------------- #
# #501 finding 1: TMM, ported from edgeR
# --------------------------------------------------------------------------- #
#: A 24-gene x 4-sample matrix (numpy default_rng(501): Poisson around one base
#: profile at depths 1.0, 1.3, 0.7, 1.0; gene 0 takes over sample 4 with 6,000 reads;
#: gene 1 is all zero) and edgeR 4.4.2's ``calcNormFactors(x, method="TMM")`` on it,
#: printed with ``sprintf("%.15f")``. edgeR 4.4.2's TMM code path (reference choice,
#: ``.calcFactorTMM``) reads the same as 3.26.8's.
_SMALL = np.array(
    [
        [348, 0, 180, 271, 187, 189, 5, 197, 271, 312, 168, 366]
        + [119, 34, 76, 297, 152, 50, 81, 319, 273, 188, 359, 339],
        [477, 0, 232, 320, 208, 222, 5, 301, 365, 440, 220, 497]
        + [158, 29, 86, 412, 231, 79, 116, 401, 342, 285, 502, 434],
        [251, 0, 109, 169, 117, 116, 3, 150, 222, 249, 120, 283]
        + [80, 17, 58, 255, 116, 46, 56, 203, 173, 130, 276, 247],
        [6000, 0, 171, 244, 160, 156, 9, 214, 324, 383, 164, 417]
        + [136, 32, 63, 325, 151, 51, 70, 258, 280, 168, 400, 326],
    ],
    dtype=np.float64,
).T
_EDGER_SMALL = [
    1.221962662869660,
    1.217307425526262,
    1.188537312695488,
    0.565625489178291,
]


def test_tmm_factors_equal_edger_calc_norm_factors() -> None:
    factors = v.tmm_factors(_SMALL, _SMALL.sum(axis=0), ["a", "b", "c", "d"])
    assert factors.factors == pytest.approx(_EDGER_SMALL, abs=1e-12)
    assert factors.library_sizes == [4781.0, 6362.0, 3446.0, 10502.0]
    assert factors.reference_column == "a"
    assert math.prod(factors.factors) == pytest.approx(1.0)


def test_tmm_refuses_missing_counts() -> None:
    counts = _SMALL.copy()
    counts[2, 1] = np.nan
    with pytest.raises(ValueError, match="NA counts not permitted"):
        v.tmm_factors(counts, np.nansum(counts, axis=0), ["a", "b", "c", "d"])


def test_a_compositional_takeover_leaves_no_tmm_offset() -> None:
    """#501 finding 1, the crystal-violet pattern: one resistant strain takes half of
    every treated library. Library-size CPM then shifts EVERY other strain down by
    log2(2) = 1; the TMM ratio of an unaffected strain stays at 0.
    """
    profile = np.arange(1, 201, dtype=np.float64) * 10.0
    control = profile.copy()
    treated = profile.copy()
    treated[0] = profile.sum()  # strain 0 holds half of the treated reads
    counts = np.column_stack([control, treated])
    sizes = counts.sum(axis=0)
    factors = v.tmm_factors(counts, sizes, ["control", "treated"])
    effective = sizes * np.asarray(factors.factors)
    cpm_ratio = np.log2((treated / sizes[1]) / (control / sizes[0]))[1:]
    tmm_ratio = np.log2((treated / effective[1]) / (control / effective[0]))[1:]
    assert np.median(cpm_ratio) == pytest.approx(-1.0, abs=1e-2)
    assert np.median(tmm_ratio) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# #501 findings 2, 4, 5, 7: the condition list, the background, MBO, the medium
# --------------------------------------------------------------------------- #
#: The 11 matrix tokens Fig 1B does not list; Table S1 doses each of them.
_UNREPORTED = {
    "QUADRIS1",
    "QUADRIS2",
    "24Dimethylimidazole",
    "2Methylimidazole",
    "45Methylimidazole",
    "CaffeicAcid",
    "LevulinicAcid",
    "Mycobutanil",
    "SodiumAcetate",
    "SodiumButyrate",
    "SodiumGlyoxylate",
}


def test_fig_1b_lists_34_conditions_including_dmso_and_mbo() -> None:
    assert len(v.FIG_1B_TOKENS) == v.FIG_1B_CONDITIONS.value == 34
    assert {"DMSO", "MBO", "MMS", "Benomyl"} <= v.FIG_1B_TOKENS
    assert not _UNREPORTED & v.FIG_1B_TOKENS
    assert v.TABLE_S1_ONLY_TOKENS == _UNREPORTED
    # every Fig 1B token resolves to a structure (MBO included)
    assert all(resolve_compound_identity(name=t).identified for t in v.FIG_1B_TOKENS)


def test_mbo_is_adjudicated_to_2_methyl_3_buten_2_ol() -> None:
    """#501 finding 5: the Results definition wins over the Abbreviations line."""
    compound = resolved_compound("MBO")
    assert compound.name == "2-methyl-3-buten-2-ol"
    assert compound.pubchem_cid == 8257
    assert compound.inchikey == "HNVRRHSXBLFLIG-UHFFFAOYSA-N"
    assert compound.smiles == "CC(C)(C=C)O"
    assert compound.provenance_gaps == []
    assert v.MBO_IDENTITY.value == compound.name
    # both conflicting lines stay on record, verbatim
    assert v.MBO_ABBREVIATION.quote == "MBO : 2-Methyl-3-butyn-2-ol"
    assert "2-methyl-3-buten-2-ol (MBO)" in v.MBO_IDENTITY.quote
    assert "line 31" in v.MBO_IDENTITY_RULE and "line 103" in v.MBO_IDENTITY_RULE


def test_library_background_is_the_sga_mata_progeny() -> None:
    """#500: MATa haploid SGA progeny; reporters and sensitizers sourced, BY pending."""
    background = v.library_background()
    assert background.name == v.LIBRARY_STRAIN
    assert background.mating_type is MatingType.a
    assert background.ploidy == "haploid"
    assert background.parents == ["Y13206", "MATa xxxΔ::kanMX deletion array"]
    by_name = {a.allele_name: a for a in background.alleles}
    assert list(by_name) == [
        "can1Δ::STE2pr-Sp_his5",
        "lyp1Δ",
        "pdr1Δ::natMX",
        "pdr3Δ::KlURA3",
        "snq2Δ::KlLEU2",
        "his3Δ1",
        "leu2Δ0",
        "ura3Δ0",
        "met15Δ0",
    ]
    sourced = {name for name, a in by_name.items() if a.is_sourced}
    assert sourced == {
        "can1Δ::STE2pr-Sp_his5",
        "lyp1Δ",
        "pdr1Δ::natMX",
        "pdr3Δ::KlURA3",
        "snq2Δ::KlLEU2",
    }
    for name in ("his3Δ1", "leu2Δ0", "ura3Δ0", "met15Δ0"):
        (gap,) = by_name[name].provenance_gaps
        assert gap.field == "provenance"
        assert gap.reason.value == "deferred_pending_source_review"
        assert gap.resolve_with is not None and "Brachmann" in str(
            gap.resolve_with.method
        )
    assert by_name["can1Δ::STE2pr-Sp_his5"].edit is AlleleEdit.cassette_replacement
    assert by_name["can1Δ::STE2pr-Sp_his5"].provenance == [
        v.QUERY_STRAIN,
        v.Y8835_GENOTYPE,
    ]
    assert by_name["pdr1Δ::natMX"].provenance == [v.QUERY_STRAIN, v.TRIPLE_SELECTION]
    assert background.provenance == [v.MATA_PROGENY, v.QUERY_STRAIN]
    assert background.functional_copies("YGL013C") == 0  # PDR1
    assert background.functional_copies("YAL001C") == 1  # untouched haploid locus
    assert not background.is_fully_sourced


def test_synbase_lists_ammonium_sulfate_as_a_dropout() -> None:
    """#501 finding 7: MSG replaced ammonium sulfate, so the medium omits it."""
    assert [c.name for c in SYNBASE.dropouts] == [
        "acetamide",
        "sodium acetate",
        "cellobiose",
        "ammonium sulfate",
    ]


def test_culture_environment_states_the_protocol_and_gaps_the_rest() -> None:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    environment = dataset._environment("Furfural")
    culture = environment.culture_format
    assert culture is not None
    assert (
        culture.vessel,
        culture.working_volume_ul,
        culture.shaking_rpm,
        culture.inoculum_od600,
        culture.endpoint,
    ) == ("24-well plates (Falcon)", 1500.0, 0.0, 0.1, EndpointRule.fixed_duration)
    assert culture.provenance == [v.CULTURE_VESSEL, v.STATIC_CULTURE]
    assert environment.pre_culture is None and environment.auxotroph_supplements is None
    assert [(g.field, g.resolve_with) for g in environment.provenance_gaps] == [
        ("pre_culture", v.PIOTROWSKI_2015),
        ("auxotroph_supplements", v.ZHANG_2019),
    ]


def _dataset_shell() -> Any:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.name = "EnvChemgenVanacloig2022Dataset"
    return dataset


def test_table_s1_covers_every_matrix_token() -> None:
    assert set(v.TABLE_S1_DOSES) == v.FIG_1B_TOKENS | _UNREPORTED
    assert len(v.TABLE_S1_DOSES) == 45
    assert v.TABLE_S1_DOSES["Furfural"].quote == (
        "<tr><td>Furfural</td><td>8 mM</td><td>No</td><td>98-01-1</td>"
        "<td>Fisher Scientific</td><td>F94-500</td></tr>"
    )
    assert v.TABLE_S1_DOSES["Furfural"].value == v.TableS1Row(
        chemical="Furfural", ic30="8 mM", dissolved_in_dmso=False
    )
    assert v.TABLE_S1.source_uri == "si/si2.md"
    assert v.TABLE_S1.sha256 == v.SI2_MD_SHA256


def test_table_s1_only_conditions_are_stored_at_their_table_s1_doses() -> None:
    """The 11 tokens no figure shows carry Table S1's dose and vehicle."""
    dataset = _dataset_shell()
    got = {
        token: (c.value, c.unit, c.basis)
        for token in sorted(_UNREPORTED)
        for c in [dataset._concentration(token)]
    }
    mm, ug, pct = (
        ConcentrationUnit.millimolar,
        ConcentrationUnit.ug_per_ml,
        ConcentrationUnit.percent,
    )
    assert got == {
        "24Dimethylimidazole": (6.0, mm, DoseBasis.IC30),
        "2Methylimidazole": (3.8, mm, DoseBasis.IC30),
        "45Methylimidazole": (3.3, mm, DoseBasis.IC30),
        "CaffeicAcid": (7.5, mm, DoseBasis.IC30),
        "LevulinicAcid": (10.0, mm, DoseBasis.IC30),
        "Mycobutanil": (13.0, ug, DoseBasis.IC30),
        "QUADRIS1": (0.08, pct, DoseBasis.fixed),
        "QUADRIS2": (0.04, pct, DoseBasis.fixed),
        "SodiumAcetate": (150.0, mm, DoseBasis.IC30),
        "SodiumButyrate": (0.19, mm, DoseBasis.IC30),
        "SodiumGlyoxylate": (0.19, mm, DoseBasis.IC30),
    }
    dmso_dissolved = {
        token for token in _UNREPORTED if v.table_s1_solvent(token) is not None
    }
    assert dmso_dissolved == {"CaffeicAcid", "2Methylimidazole", "45Methylimidazole"}
    assert v.QUADRIS_ARBITRARY_DOSE.value is DoseBasis.fixed
    assert "arbitrarily selected" in v.QUADRIS_ARBITRARY_DOSE.quote


def test_quadris_is_pooled_and_carries_the_inchikey_gap() -> None:
    """The Methods analyzed QUADRIS unpaired; its curated identity row resolves no
    structure, so the compound is stored with the typed ``inchikey`` gap.
    """
    dataset = _dataset_shell()
    assert dataset._units("QUADRIS1") == v._POOLED_UNITS
    assert dataset._units("QUADRIS2") == v._POOLED_UNITS
    quadris = dataset._compound("QUADRIS1")
    assert quadris.compound.name == "QUADRIS1"
    assert quadris.compound.inchikey is None
    assert [g.field for g in quadris.compound.provenance_gaps] == ["inchikey"]
    assert quadris.solvent is None


def test_dmso_dissolved_inhibitor_carries_the_one_percent_dmso_vehicle() -> None:
    """Vanillin's Table S1 row says 'Dissolved in DMSO? Yes'."""
    vanillin = _dataset_shell()._compound("Vanillin")
    assert vanillin.solvent is not None
    assert vanillin.solvent.name == "DMSO"
    assert vanillin.solvent.percent == 1.0
    assert vanillin.solvent.compound == resolved_compound("DMSO")
    assert vanillin.solvent.compound.name == "dimethyl sulfoxide"
    assert vanillin.provenance_gaps == []
    assert (
        vanillin.concentration.value,
        vanillin.concentration.unit,
        vanillin.concentration.basis,
    ) == (5.0, ConcentrationUnit.millimolar, DoseBasis.IC30)


def test_water_dissolved_inhibitor_has_no_vehicle_and_no_gap() -> None:
    """Furfural's Table S1 row says 'Dissolved in DMSO? No'."""
    furfural = _dataset_shell()._compound("Furfural")
    assert furfural.solvent is None
    assert furfural.provenance_gaps == []


def test_furfural_is_dosed_at_8_mm_ic30() -> None:
    concentration = _dataset_shell()._concentration("Furfural")
    assert (concentration.value, concentration.unit, concentration.basis) == (
        8.0,
        ConcentrationUnit.millimolar,
        DoseBasis.IC30,
    )


def test_percent_compounds_store_the_reported_number_in_basis_free_percent() -> None:
    """#764: a bare Table S1 percent is the reported number in ConcentrationUnit.percent."""
    dataset = _dataset_shell()
    got = {
        token: (c.value, c.unit, c.basis)
        for token in ("MBO", "EtOH", "IBA", "GVL", "MMS")
        for c in [dataset._concentration(token)]
    }
    percent = ConcentrationUnit.percent
    assert got == {
        "MBO": (1.5, percent, DoseBasis.IC30),
        "EtOH": (4.0, percent, DoseBasis.IC30),
        "IBA": (0.75, percent, DoseBasis.IC30),
        "GVL": (1.5, percent, DoseBasis.IC30),
        "MMS": (0.01, percent, DoseBasis.fixed),
    }
    assert ConcentrationUnit.percent.value == "percent"
    assert v.TABLE_S1_DOSES["EtOH"].value.ic30 == "4%"
    assert "IC30 '4%'" in str(v.TABLE_S1_DOSES["EtOH"].note)
    assert "ConcentrationUnit.percent" in str(v.TABLE_S1_DOSES["EtOH"].note)
    # the OCR left GVL's cell empty and put its 1.5% on the next row; the dose is
    # parsed from that displaced row's quote
    assert v.TABLE_S1_DOSES["GVL"].value.ic30 == ""
    assert v.GVL_TABLE_S1_DISPLACED.value == "1.5%"
    assert "<td>OTHER COMPOUNDS</td><td>1.5%</td>" in v.GVL_TABLE_S1_DISPLACED.quote
    assert v.table_s1_percent("GVL") == 1.5


def test_dipyridyl_is_dosed_at_18_ug_per_ml() -> None:
    concentration = _dataset_shell()._concentration("22Dipyridyl")
    assert (concentration.value, concentration.unit, concentration.basis) == (
        18.0,
        ConcentrationUnit.ug_per_ml,
        DoseBasis.IC30,
    )


def test_dmso_condition_is_dosed_at_table_s1_and_the_vehicle_stays_one_percent() -> (
    None
):
    """#764: DMSO's own condition is Table S1's 2.50%, v/v by the paper's DMSO
    convention; the 1% v/v sentence is the vehicle of the DMSO-dissolved inhibitors.
    """
    dmso_row = v.TABLE_S1_DOSES["DMSO"]
    assert dmso_row.value.ic30 == "2.50%"
    assert "without v/v or w/v" in str(dmso_row.note)
    assert "1% (v/v)" in str(dmso_row.note)
    assert "final concentration of DMSO" in v.VEHICLE_CONTROL.quote
    assert "v / v" in v.VEHICLE_CONTROL.quote
    dmso = _dataset_shell()._compound("DMSO")
    assert dmso.compound.name == "dimethyl sulfoxide"
    assert dmso.compound.inchikey == "IAZDPXIOMUYVGZ-UHFFFAOYSA-N"
    assert dmso.solvent is None and dmso.provenance_gaps == []
    assert (
        dmso.concentration.value,
        dmso.concentration.unit,
        dmso.concentration.basis,
    ) == (2.5, ConcentrationUnit.percent_v_v, DoseBasis.IC30)
    assert v.DMSO_VEHICLE_PERCENT.value == 1.0
    vehicle = v.table_s1_solvent("Vanillin")
    assert vehicle is not None and vehicle.percent == 1.0
    assert v.TABLE_S1_DOSES["Benomyl"].value.ic30 == "10 ug/mL"
    benomyl = _dataset_shell()._concentration("Benomyl")
    assert (benomyl.value, benomyl.unit, benomyl.basis) == (
        34.4,
        ConcentrationUnit.micromolar,
        DoseBasis.fixed,
    )


def test_an_unreadable_table_s1_cell_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    row = v.TABLE_S1_DOSES["Furfural"]
    odd = row.model_copy(update={"value": row.value.model_copy(update={"ic30": "8"})})
    monkeypatch.setitem(v.TABLE_S1_DOSES, "Furfural", odd)
    with pytest.raises(
        ValueError, match="^Furfural: unreadable Table S1 IC30 cell '8'$"
    ):
        v.table_s1_ic30("Furfural")
    with pytest.raises(ValueError, match="not a Table S1 row"):
        v._row("<tr><td>x</td></tr>")
    # an empty IC30 cell is read only through a recorded displaced row (GVL's)
    empty = row.model_copy(update={"value": row.value.model_copy(update={"ic30": ""})})
    monkeypatch.setitem(v.TABLE_S1_DOSES, "Furfural", empty)
    with pytest.raises(
        ValueError, match="^Furfural: unreadable Table S1 IC30 cell ''$"
    ):
        v.table_s1_ic30("Furfural")
    with pytest.raises(ValueError, match="^Furfural: Table S1 IC30 cell '' is not a"):
        v.table_s1_percent("Furfural")
    with pytest.raises(ValueError, match="not a Table S1 section-header row"):
        v._section_row_cell("<tr><td>x</td></tr>", note="")


# --------------------------------------------------------------------------- #
# The GEO-layout matrix, the exact records, the ledger, the refusals
# --------------------------------------------------------------------------- #
_CONTROLS = ["Control1_CG001", "Control2_CG001", "Control1_CG002", "Control2_CG002"]
_BATCHES = {"MMS": ("001", "002", "002")}
_TOKENS = ["Furfural", "MMS", "SodiumGlyoxylate", "DMSO", "MBO", "QUADRIS1"]
_N_CONST = 60


def _replicates(token: str) -> list[str]:
    batches = _BATCHES.get(token, ("001", "002", "001"))
    return [f"{token}_CG{b}_rep{i}" for i, b in enumerate(batches, start=1)]


_COLUMNS = _CONTROLS + [c for token in _TOKENS for c in _replicates(token)]
_FILLED = 5.0  # DMSO, MBO and QUADRIS1 counts for every library row


def _row(
    gene: str, std: str, counts: dict[str, list[float]], default: float
) -> dict[str, Any]:
    """One matrix row: named groups of columns, every other column at ``default``."""
    row: dict[str, Any] = {"gene": gene, "std_name": std}
    row.update(dict.fromkeys(_COLUMNS, default))
    for group, values in counts.items():
        names = _CONTROLS if group == "controls" else _replicates(group)
        row.update(zip(names, values, strict=True))
    for token in ("DMSO", "MBO", "QUADRIS1"):
        row.update(dict.fromkeys(_replicates(token), _FILLED))
    return row


def _geo_matrix() -> pd.DataFrame:
    rows = [
        _row(
            "YAL001C_AAAACCCC",
            "TFC3",
            {
                "controls": [3, 3, 7, 7],
                "Furfural": [15, 63, 63],
                "MMS": [11, 23, 47],
                "SodiumGlyoxylate": [3, 7, 3],
            },
            0,
        ),
        _row(
            "YAL002W_CCCCGGGG",
            "VPS8",
            {
                "controls": [0, 0, 0, 0],
                "Furfural": [0, 1, 3],
                "MMS": [0, 0, 0],
                "SodiumGlyoxylate": [0, 0, 0],
            },
            0,
        ),
        _row("YAL003W_GGGGTTTT", "YAL003W", {}, 2),
        _row("YPL999C_TTTTAAAA", "NOPE", {}, 2),
        _row("YGL013C_ACGTACGT", "PDR1", {}, 2),
        _row("YBR001C_TTAATTAA", "SPO23", {}, 2),
        _row("YBR002C", "YBR002C", {}, 1),
    ]
    rows[5]["Control1_CG001"] = float("nan")
    const = [
        {"gene": f"Const{i}_row", "std_name": "none", **dict.fromkeys(_COLUMNS, 1000)}
        for i in range(_N_CONST)
    ]
    frame = pd.DataFrame(rows + const)
    filler: dict[str, Any] = {"gene": "Filler_row", "std_name": "none"}
    for column in _COLUMNS:
        filler[column] = 1_000_000 - frame[column].sum(skipna=True)
    matrix = pd.concat([frame, pd.DataFrame([filler])], ignore_index=True)
    # The fixture's premise: library size 1e6 in every column, so CPM equals the count.
    assert {float(matrix[c].sum(skipna=True)) for c in _COLUMNS} == {1_000_000.0}
    return matrix


def _write_matrix(path: str, frame: pd.DataFrame) -> None:
    with gzip.open(path, "wt") as handle:
        frame.to_csv(handle, sep="\t", index=False)


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> Any:
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    root = tmp_path / "env_chemgen_vanacloig2022"
    (root / "raw").mkdir(parents=True)
    _write_matrix(str(root / "raw" / v.DATA_FILENAME), frame)
    return v.EnvChemgenVanacloig2022Dataset(root=str(root))


@pytest.fixture()
def geo_built(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """The GEO-layout matrix of the module docstring, built end to end."""
    return _build(tmp_path, monkeypatch, _geo_matrix())


def _phenotype(response: float, sd: float, units: str) -> EnvironmentResponsePhenotype:
    return EnvironmentResponsePhenotype(
        measurement_type=MeasurementType.log2_ratio,
        assay_type=AssayType.pooled_competitive_growth_barcode,
        environment_response=response,
        environment_response_uncertainty=sd,
        environment_response_uncertainty_type=UncertaintyType.sample_sd,
        n_samples=3,
        sample_unit=SampleUnit.biological_replicate,
        units=units,
    )


def _genotype(systematic: str, common: str, barcode: str | None) -> Genotype:
    return Genotype(
        perturbations=[
            BarcodedKanMxDeletionPerturbation(
                systematic_gene_name=systematic,
                perturbed_gene_name=common,
                barcode=barcode,
                collection="3DeltaAlpha drug-sensitive yeast deletion collection of "
                "4309 mutants",
                cassette="kanMX",
                provenance_gaps=[] if barcode is not None else [v.BARCODE_ABSENT_GAP],
            )
        ]
    )


def test_geo_fixture_tmm_factors_are_exactly_one(geo_built: Any) -> None:
    """The fixture premise: every condition's TMM factors are exactly 1.0."""
    factors = json.loads(
        (Path(geo_built.preprocess_dir) / "normalization_factors.json").read_text()
    )
    assert sorted(factors) == [
        "DMSO",
        "Furfural",
        "MBO",
        "MMS",
        "QUADRIS1",
        "SodiumGlyoxylate",
    ]
    for entry in factors.values():
        assert set(entry["factors"]) == {1.0}
        assert set(entry["library_sizes"]) == {1_000_000.0}
    assert factors["MMS"]["columns"] == sorted(_CONTROLS) + _replicates("MMS")
    assert factors["QUADRIS1"]["columns"] == sorted(_CONTROLS) + _replicates("QUADRIS1")
    assert factors["Furfural"]["columns"] == sorted(_CONTROLS) + _replicates("Furfural")


def test_records_are_condition_major_with_the_closed_form_responses(
    geo_built: Any,
) -> None:
    """(screened ORF, condition, response, SD) for all 36 records, in LMDB order."""
    got = []
    for i in range(len(geo_built)):
        experiment = geo_built[i]["experiment"]
        compound = experiment["environment"]["perturbations"][0]["compound"]["name"]
        (screened,) = experiment["genotype"]["perturbations"]
        got.append(
            (
                screened["systematic_gene_name"],
                compound,
                experiment["phenotype"]["environment_response"],
                experiment["phenotype"]["environment_response_uncertainty"],
            )
        )
    l6 = math.log2(6)
    rows = ["YAL001C", "YAL002W", "YAL003W", "YPL999C", "YGL013C", "YBR002C"]
    filled_paired = [l6 - 7 / 3, l6, 1.0, 1.0, 1.0, l6 - 1]
    filled_pooled = [0.0, l6, 1.0, 1.0, 1.0, l6 - 1]
    by_condition = {
        "dimethyl sulfoxide": filled_paired,
        "furfural": [3.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "2-methyl-3-buten-2-ol": filled_paired,
        "methyl methanesulfonate": [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "QUADRIS1": filled_pooled,
        "sodium glyoxylate": [0.0] * 6,
    }
    sds = {
        "dimethyl sulfoxide": [math.sqrt(1 / 3)] + [0.0] * 5,
        "furfural": [1.0, 1.0] + [0.0] * 4,
        "2-methyl-3-buten-2-ol": [math.sqrt(1 / 3)] + [0.0] * 5,
        "methyl methanesulfonate": [1.0] + [0.0] * 5,
        "QUADRIS1": [0.0] * 6,
        "sodium glyoxylate": [0.0] * 6,
    }
    expected = [
        (row, name, by_condition[name][i], sds[name][i])
        for name in by_condition
        for i, row in enumerate(rows)
    ]
    assert [g[:2] for g in got] == [e[:2] for e in expected]
    for (_, _, response, sd), (_, _, want_response, want_sd) in zip(
        got, expected, strict=True
    ):
        assert response == pytest.approx(want_response, abs=1e-12)
        assert sd == pytest.approx(want_sd, abs=1e-12)


def test_full_paired_record_equals_the_hand_built_experiment(geo_built: Any) -> None:
    """Record 6, YAL001C / Furfural: TMM log2 ratio against its OWN batch's controls."""
    record = geo_built[6]
    experiment = record["experiment"]
    assert experiment["experiment_type"] == "strain_environment_response"
    assert (
        experiment["genotype"] == _genotype("YAL001C", "TFC3", "AAAACCCC").model_dump()
    )
    assert experiment["phenotype"] == _phenotype(3.0, 1.0, v._PAIRED_UNITS).model_dump()
    assert experiment["phenotype"]["environment_response_se"] == 1.0 / math.sqrt(3)
    environment = experiment["environment"]
    furfural = environment["perturbations"][0]
    assert furfural["compound"] == resolved_compound("Furfural").model_dump()
    assert furfural["concentration"] == {"value": 8.0, "unit": "mM", "basis": "IC30"}
    assert furfural["solvent"] is None and furfural["provenance_gaps"] == []
    assert environment["culture_format"]["working_volume_ul"] == 1500.0
    reference = record["reference"]
    assert reference["phenotype_reference"]["units"] == v._PAIRED_UNITS
    assert (
        StrainEnvironmentResponseExperimentReference.model_validate(
            reference
        ).genome_reference.background
        == v.library_background()
    )
    assert StrainEnvironmentResponseExperiment.model_validate(experiment)
    assert (
        record["publication"]
        == Publication(
            doi="10.1093/femsyr/foac036",
            doi_url="https://doi.org/10.1093/femsyr/foac036",
        ).model_dump()
    )


def test_dmso_record_is_served_at_the_table_s1_percent(geo_built: Any) -> None:
    """#501 finding 3, revised by #764: DMSO is paired against the SynBase Control
    columns and dosed at Table S1's 2.50%, v/v by the paper's DMSO convention.
    """
    dmso = geo_built[0]["experiment"]["environment"]["perturbations"][0]
    assert dmso["compound"]["name"] == "dimethyl sulfoxide"
    assert dmso["concentration"] == {
        "value": 2.5,
        "unit": "percent_v/v",
        "basis": "IC30",
    }
    assert dmso["solvent"] is None and dmso["provenance_gaps"] == []
    assert (
        geo_built[0]["experiment"]["phenotype"]["units"] == v._PAIRED_UNITS
    )  # Control columns, same batch


def test_pooled_mms_edge_record(geo_built: Any) -> None:
    """Record 18, YAL001C / MMS: the pooled control and the fixed, basis-free percent."""
    experiment = geo_built[18]["experiment"]
    assert experiment["phenotype"] == _phenotype(2.0, 1.0, v._POOLED_UNITS).model_dump()
    mms = experiment["environment"]["perturbations"][0]
    assert mms["compound"]["name"] == "methyl methanesulfonate"
    assert mms["concentration"] == {"value": 0.01, "unit": "percent", "basis": "fixed"}
    assert geo_built[18]["reference"]["phenotype_reference"]["units"] == v._POOLED_UNITS


def test_a_row_without_a_barcode_is_served_with_a_typed_absence(geo_built: Any) -> None:
    """#524: ``YBR002C`` has no ``_<barcode>`` suffix, so the barcode is absent from the
    release: ``barcode None`` with ``BARCODE_ABSENT_GAP``, never an empty string. Its
    released std_name is the ORF itself. Record 11 is YBR002C / Furfural, whose equal
    nonzero counts give SD exactly 0.
    """
    experiment = geo_built[11]["experiment"]
    expected = _genotype("YBR002C", "YBR002C", None)
    (deletion,) = expected.perturbations
    assert isinstance(deletion, BarcodedKanMxDeletionPerturbation)
    assert deletion.provenance_gaps == [v.BARCODE_ABSENT_GAP]
    assert experiment["genotype"] == expected.model_dump()
    assert experiment["genotype"]["perturbations"][0]["barcode"] is None
    gap = v.BARCODE_ABSENT_GAP
    assert (gap.field, gap.reason, gap.looked_in) == (
        "barcode",
        ProvenanceGapReason.not_reported_by_primary,
        Provenance(
            source_uri=f"{v.RAW_DIR_REL}/{v.DATA_REL}",
            citation_key=v.CITATION_KEY,
            sha256=v.DATA_SHA256,
            page="gene column ('<ORF>_<barcode>')",
        ),
    )
    assert experiment["phenotype"] == _phenotype(0.0, 0.0, v._PAIRED_UNITS).model_dump()
    # a barcoded record carries no gap
    (barcoded,) = geo_built[6]["experiment"]["genotype"]["perturbations"]
    assert barcoded["barcode"] == "AAAACCCC" and barcoded["provenance_gaps"] == []


def test_drop_ledger_is_written_exactly(geo_built: Any) -> None:
    """Every released cell accounted for: 36 stored, 366 non-ORF row cells, YBR001C's six
    cells with a missing control count, and no row with every count missing.
    """
    log = json.loads(
        (Path(geo_built.preprocess_dir) / "dropped_records.json").read_text()
    )
    rules = [(r["rule"], r["scope"], r["n_records"], r["items"]) for r in log["rules"]]
    assert (log["dataset"], log["source_records"], log["kept_records"]) == (
        "EnvChemgenVanacloig2022Dataset",
        68 * 6,
        36,
    )
    assert log["dropped_records"] == 372
    non_orf = sorted([f"Const{i}_row" for i in range(_N_CONST)] + ["Filler_row"])
    assert rules == [
        ("gene_column_is_not_a_systematic_orf", "library_row", 366, non_orf),
        (
            "a_count_the_cell_uses_is_missing",
            "cell",
            6,
            [
                "YBR001C:DMSO",
                "YBR001C:Furfural",
                "YBR001C:MBO",
                "YBR001C:MMS",
                "YBR001C:QUADRIS1",
                "YBR001C:SodiumGlyoxylate",
            ],
        ),
    ]
    assert [(n["rule"], n["n_cells"], n["items"]) for n in log["not_a_value"]] == [
        ("every_count_of_the_row_is_missing", 0, [])
    ]
    assert log["compound_identity_gaps"] == [
        {"token": "QUADRIS1", "compound_name": "QUADRIS1", "n_records": 6}
    ]
    assert "released-gene-name" in log["rules"][0]["description"]
    assert log["rules"][1]["description"].startswith(
        "one of the counts this (strain, compound) cell is computed from, a replicate "
        "or one of its paired controls, is a source NaN in the release"
    )
    assert log["source_records"] == (
        log["kept_records"]
        + log["dropped_records"]
        + sum(n["n_cells"] for n in log["not_a_value"])
    )


def _records_of(built: Any, orf: str) -> list[dict[str, Any]]:
    return [
        built[i]["experiment"]
        for i in range(len(built))
        if built[i]["experiment"]["genotype"]["perturbations"][0][
            "systematic_gene_name"
        ]
        == orf
    ]


def test_annotation_renamed_and_unannotated_orfs_are_stored_as_released(
    geo_built: Any,
) -> None:
    """YAL003W (folded into YAL002W by a later annotation) and YPL999C (in no
    annotation) are stored under their released ORF and std_name, one record per
    condition, with no ConstructedOrf anywhere.
    """
    for orf, name, barcode in (
        ("YAL003W", "YAL003W", "GGGGTTTT"),
        ("YPL999C", "NOPE", "TTTTAAAA"),
    ):
        records = _records_of(geo_built, orf)
        assert len(records) == 6
        for experiment in records:
            assert experiment["genotype"] == _genotype(orf, name, barcode).model_dump()
    assert not hasattr(v, "LegacyOrfStrain")
    assert not hasattr(v, "resolve_library_rows")


def test_a_background_locus_row_is_stored_as_released(geo_built: Any) -> None:
    """YGL013C is PDR1, which the SGA background deleted (pdr1Δ::natMX); the release
    carries a kanMX row for it, and that row is stored, the contradiction noted, not
    resolved.
    """
    assert v.library_background().functional_copies("YGL013C") == 0
    records = _records_of(geo_built, "YGL013C")
    assert len(records) == 6
    assert (
        records[0]["genotype"] == _genotype("YGL013C", "PDR1", "ACGTACGT").model_dump()
    )


def test_an_all_zero_cell_is_stored_with_sd_zero(geo_built: Any) -> None:
    """Record 19, YAL002W / MMS: three replicate counts of 0 against zero controls.
    The value is L(0) - L(0) = 0.0 and the sample SD is exactly 0, stored as 0.0.
    """
    experiment = geo_built[19]["experiment"]
    assert experiment["genotype"]["perturbations"][0]["systematic_gene_name"] == (
        "YAL002W"
    )
    assert experiment["environment"]["perturbations"][0]["compound"]["name"] == (
        "methyl methanesulfonate"
    )
    assert experiment["phenotype"] == _phenotype(0.0, 0.0, v._POOLED_UNITS).model_dump()
    assert experiment["phenotype"]["environment_response_uncertainty"] == 0.0


def test_a_missing_replicate_count_drops_only_that_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#524: YBR001C's NaN moved from a control to one Furfural replicate drops ONLY its
    Furfural cell; its other five cells are stored with all three replicates.
    """
    frame = _geo_matrix()
    ybr001c = frame.index[frame["gene"] == "YBR001C_TTAATTAA"][0]
    frame.loc[ybr001c, "Control1_CG001"] = 2.0
    frame.loc[ybr001c, "Furfural_CG001_rep1"] = float("nan")
    filler = frame.index[frame["gene"] == "Filler_row"][0]
    frame.loc[filler, "Control1_CG001"] -= 2.0  # keep every library size at 1e6
    frame.loc[filler, "Furfural_CG001_rep1"] += 2.0
    built = _build(tmp_path, monkeypatch, frame)
    log = v.DropLog.model_validate_json(
        (Path(built.preprocess_dir) / "dropped_records.json").read_text()
    )
    rules = {rule.rule: rule for rule in log.rules}
    assert rules["a_count_the_cell_uses_is_missing"].items == ["YBR001C:Furfural"]
    assert rules["gene_column_is_not_a_systematic_orf"].n_records == 366
    assert (log.kept_records, log.dropped_records) == (41, 367)
    served = []
    for i in range(len(built)):
        experiment = built[i]["experiment"]
        (screened,) = experiment["genotype"]["perturbations"]
        if screened["systematic_gene_name"] == "YBR001C":
            compound = experiment["environment"]["perturbations"][0]["compound"]
            served.append((compound["name"], experiment["phenotype"]["n_samples"]))
    assert served == [
        ("dimethyl sulfoxide", 3),
        ("2-methyl-3-buten-2-ol", 3),
        ("methyl methanesulfonate", 3),
        ("QUADRIS1", 3),
        ("sodium glyoxylate", 3),
    ]


def test_a_row_with_every_count_missing_is_not_a_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row whose EVERY count is missing holds no value: counted, not a record."""
    frame = _geo_matrix()
    ybr001c = frame.index[frame["gene"] == "YBR001C_TTAATTAA"][0]
    frame.loc[ybr001c, _COLUMNS] = float("nan")
    built = _build(tmp_path, monkeypatch, frame)
    log = v.DropLog.model_validate_json(
        (Path(built.preprocess_dir) / "dropped_records.json").read_text()
    )
    rules = {rule.rule: rule for rule in log.rules}
    assert rules["gene_column_is_not_a_systematic_orf"].n_records == 366
    assert rules["a_count_the_cell_uses_is_missing"].n_records == 0
    (not_a_value,) = log.not_a_value
    assert (not_a_value.n_cells, not_a_value.items) == (6, ["YBR001C_TTAATTAA"])
    assert (log.kept_records, log.dropped_records) == (36, 366)


def test_a_barcodeless_row_and_a_barcoded_row_of_one_orf_are_both_stored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two released rows, two strains: the barcodeless YBR002C row keeps ``barcode
    None`` with its gap, the barcoded copy keeps its barcode.
    """
    frame = _geo_matrix()
    barcoded = frame[frame["gene"] == "YBR002C"].copy()
    barcoded["gene"] = "YBR002C_GGGGAAAA"
    frame = pd.concat([frame, barcoded], ignore_index=True)
    filler = frame.index[frame["gene"] == "Filler_row"][0]
    # keep every library size at 1e6 with the copied row's reads added
    frame.loc[filler, _COLUMNS] -= barcoded[_COLUMNS].iloc[0]
    built = _build(tmp_path, monkeypatch, frame)
    log = v.DropLog.model_validate_json(
        (Path(built.preprocess_dir) / "dropped_records.json").read_text()
    )
    assert log.source_records == 69 * 6 and log.kept_records == 42
    barcodes = [
        experiment["genotype"]["perturbations"][0]["barcode"]
        for experiment in _records_of(built, "YBR002C")
    ]
    assert sorted(barcodes, key=str) == ["GGGGAAAA"] * 6 + [None] * 6


def test_reference_index_splits_paired_from_pooled_and_the_gene_set(
    geo_built: Any,
) -> None:
    """The inhibitor-free reference is shared across paired conditions; MMS differs
    and QUADRIS1 differ only in their units string. The gene set is the screened genes
    only: the background loci ride on the reference, not on the genotype.
    """
    index = geo_built.experiment_reference_index
    assert index is not None
    by_units = {e.reference.phenotype_reference.units: e.member_indices for e in index}
    assert by_units == {
        v._PAIRED_UNITS: list(range(18)) + list(range(30, 36)),
        v._POOLED_UNITS: list(range(18, 30)),
    }
    assert json.loads(
        (Path(geo_built.preprocess_dir) / "gene_set.json").read_text()
    ) == ["YAL001C", "YAL002W", "YAL003W", "YBR002C", "YGL013C", "YPL999C"]


def _with_columns(extra: dict[str, float], drop: list[str]) -> pd.DataFrame:
    frame = _geo_matrix().drop(columns=drop)
    for column, value in extra.items():
        frame[column] = value
    return frame


def test_an_unparseable_sample_column_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({"Furfural_rep4": 1.0}, [])
    with pytest.raises(
        RuntimeError, match="^unparseable sample column: 'Furfural_rep4'$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_a_matrix_without_control_columns_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({}, _CONTROLS)
    with pytest.raises(
        RuntimeError, match="^no ControlN_CG\\* columns found in GSE186866 matrix$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_a_kept_compound_with_two_replicates_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({}, ["Furfural_CG001_rep3"])
    with pytest.raises(
        RuntimeError, match="^Furfural: 2 replicate columns, expected 3$"
    ):
        _build(tmp_path, monkeypatch, frame)


def test_two_barcodes_of_one_orf_are_two_stored_strains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second barcode of YAL001C is a second released strain: both are stored."""
    frame = _geo_matrix()
    duplicate = frame.iloc[[0]].copy()
    duplicate["gene"] = "YAL001C_TTTTGGGG"
    frame = pd.concat([duplicate, frame], ignore_index=True)
    built = _build(tmp_path, monkeypatch, frame)
    barcodes = {
        experiment["genotype"]["perturbations"][0]["barcode"]
        for experiment in _records_of(built, "YAL001C")
    }
    assert barcodes == {"AAAACCCC", "TTTTGGGG"}


def test_one_orf_and_barcode_released_twice_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two rows with the same ORF and barcode would store identical records; the build
    refuses and names them rather than collapse them.
    """
    frame = _geo_matrix()
    frame = pd.concat([frame.iloc[[0]].copy(), frame], ignore_index=True)
    with pytest.raises(RuntimeError) as err:
        _build(tmp_path, monkeypatch, frame)
    assert str(err.value) == (
        "two library rows release the same ORF and barcode, so their records would be "
        "identical: ['YAL001C_AAAACCCC']"
    )


def test_a_matrix_token_with_no_table_s1_row_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = _with_columns({"Newtoken_CG001_rep1": 1.0, "Newtoken_CG002_rep2": 1.0}, [])
    with pytest.raises(
        RuntimeError, match="^matrix tokens with no Table S1 row: \\['Newtoken'\\]$"
    ):
        _build(tmp_path, monkeypatch, frame)


def _mirror(data_root: Path, payload: bytes, digest: str | None) -> Path:
    """A raw mirror holding ``payload`` and a manifest recording ``digest``."""
    root = v.raw_mirror_dir(str(data_root))
    (root / "data").mkdir(parents=True)
    (root / v.DATA_REL).write_bytes(payload)
    files = []
    if digest is not None:
        files.append(
            {
                "path": v.DATA_REL,
                "role": "raw_data",
                "bytes": len(payload),
                "sha256": digest,
            }
        )
    (root / "manifest.json").write_text(
        json.dumps({"citation_key": v.CITATION_KEY, "files": files})
    )
    return root / v.DATA_REL


def _bare_dataset(root: Path) -> Any:
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    dataset.root = str(root)
    return dataset


def test_download_links_the_verified_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    monkeypatch.setattr(v, "DATA_SHA256", hashlib.sha256(b"counts").hexdigest())
    src = _mirror(data_root, b"counts", v.DATA_SHA256)
    dataset = _bare_dataset(tmp_path / "build")
    dataset.download()
    dest = tmp_path / "build" / "raw" / v.DATA_FILENAME
    assert os.readlink(dest) == str(src)
    dataset.download()  # an existing link is left alone
    assert os.readlink(dest) == str(src)


def test_download_refuses_a_manifest_digest_off_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #561): ``DATA_SHA256`` is the one pin and the manifest is its
    retrieval record. A manifest recording any other digest is refused by name with
    both digests before the mirror bytes are read, and nothing is linked.
    """
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _mirror(data_root, b"counts", "ab" * 32)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(ManifestPinMismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"raw-mirror manifest records sha256 {'ab' * 32} for {v.DATA_REL}, but the "
        "loader pins e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
    )
    assert not (tmp_path / "build" / "raw").exists()


def test_download_refuses_mirror_bytes_off_the_module_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the manifest agreeing with ``DATA_SHA256``, mirror bytes hashing to anything
    else raise ``RawSha256MismatchError`` naming the mirror file, the pin and the
    observed digest, and nothing is linked.
    """
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    _mirror(data_root, b"tampered", v.DATA_SHA256)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(RawSha256MismatchError) as err:
        dataset.download()
    assert str(err.value) == (
        f"sha256 mismatch for {v.raw_mirror_dir(str(data_root)) / v.DATA_REL}: "
        "expected e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a, "
        f"observed {hashlib.sha256(b'tampered').hexdigest()}"
    )
    assert list((tmp_path / "build" / "raw").iterdir()) == []


def test_a_raw_file_off_the_pin_is_refused_at_build_time(
    off_pin_raw: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contract (issue #524's sweep): with the count matrix already in ``raw/`` PyG
    skips ``download()``, so ``process()`` verifies it against ``DATA_SHA256`` first and
    raises ``RawSha256MismatchError`` naming it and both digests before a row is read;
    no store is written and the file is left as found.
    """
    monkeypatch.delenv("TC_DATA_URL", raising=False)
    staged = off_pin_raw(v, [v.DATA_FILENAME])
    raw = staged.root / "raw" / v.DATA_FILENAME
    with pytest.raises(RawSha256MismatchError) as err:
        v.EnvChemgenVanacloig2022Dataset(root=str(staged.root))
    assert str(err.value) == (
        f"sha256 mismatch for {raw}: expected "
        "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a, "
        f"observed {staged.observed}"
    )
    assert list((staged.root / "processed").iterdir()) == []
    assert not (staged.root / "preprocess").exists()
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == staged.observed


def test_download_refuses_a_mirror_missing_the_file_or_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "dr"
    monkeypatch.setenv("DATA_ROOT", str(data_root))
    src = _mirror(data_root, b"counts", None)
    dataset = _bare_dataset(tmp_path / "build")
    with pytest.raises(KeyError) as missing_record:
        dataset.download()
    assert missing_record.value.args == (
        f"{v.DATA_REL} is not in the raw-mirror manifest",
    )
    manifest = src.parent.parent / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "citation_key": v.CITATION_KEY,
                "files": [
                    {
                        "path": v.DATA_REL,
                        "role": "raw_data",
                        "bytes": 1,
                        "sha256": v.DATA_SHA256,
                    }
                ],
            }
        )
    )
    src.unlink()
    with pytest.raises(RuntimeError) as missing_file:
        dataset.download()
    assert (
        str(missing_file.value) == f"required raw artifact missing from mirror: {src}"
    )


def test_download_without_a_manifest_is_a_bare_file_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``load_manifest`` reads the file directly, so no mirror is a FileNotFoundError."""
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "dr"))
    with pytest.raises(FileNotFoundError) as err:
        _bare_dataset(tmp_path / "build").download()
    assert err.value.filename == str(v.raw_mirror_dir() / "manifest.json")


class _FrozenDatetime:
    @staticmethod
    def now(tz: Any = None) -> datetime:
        return datetime(2026, 9, 30, 12, 0, tzinfo=UTC)


def test_deposit_raw_mirror_writes_the_file_and_an_exact_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"synthetic GEO counts")
    pin = hashlib.sha256(b"synthetic GEO counts").hexdigest()
    monkeypatch.setattr(v, "DATA_SHA256", pin)
    monkeypatch.setattr(v, "datetime", _FrozenDatetime)
    data_root = tmp_path / "dr"

    root = v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))

    assert root == data_root / "torchcell-raw" / v.CITATION_KEY
    assert (root / v.DATA_REL).read_bytes() == b"synthetic GEO counts"
    assert json.loads((root / "manifest.json").read_text()) == {
        "version": 1,
        "citation_key": "vanacloig-pedrosComparativeChemicalGenomic2022",
        "doi": "10.1093/femsyr/foac036",
        "title": "Comparative chemical genomic profiling across plant-based hydrolysate "
        "toxins reveals widespread antagonism in fitness contributions",
        "library_id": None,
        "zotero_item_key": None,
        "collections": [],
        "files": [
            {
                "path": v.DATA_REL,
                "role": "raw_data",
                "bytes": 20,
                "sha256": pin,
                "source": v.DATA_URL,
                "zotero_md5": None,
                "retrieval": {
                    "method": "direct_url",
                    "source_url": v.DATA_URL,
                    "retriever": "torchcell.literature.retrieve.direct_url",
                    "params": {"url": v.DATA_URL},
                    "sha256": pin,
                    "retrieved_at": "2026-09-13",
                    "last_check": None,
                },
                "processing": None,
            }
        ],
        "si_data_sources": [
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186866",
            v.DATA_URL,
        ],
        "si_expected": [
            "Table S1 (per-compound IC30 molar values + DMSO control pairing) -- "
            "academic.oup.com is not scriptable, so it is NOT mirrored",
            "Dataset2_mclust_cdt (clustered edgeR logFC matrix) -- same publisher gate",
        ],
        "provenance_complete": True,
        "created_at": "2026-09-30T12:00:00+00:00",
    }
    assert v.manifest_sha256(v.load_manifest(str(data_root)), v.DATA_REL) == pin
    mtime = (root / v.DATA_REL).stat().st_mtime_ns
    v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))
    assert (root / v.DATA_REL).stat().st_mtime_ns == mtime


def test_deposit_refuses_a_source_off_the_pin_before_touching_the_mirror(
    tmp_path: Path,
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"not the GEO matrix")
    data_root = tmp_path / "dr"
    with pytest.raises(RuntimeError) as err:
        v.deposit_raw_mirror(counts_path=counts, data_root=str(data_root))
    assert str(err.value) == (
        f"{counts} sha256 mismatch: got "
        f"{hashlib.sha256(b'not the GEO matrix').hexdigest()}, expected "
        "e29eb02769ce2180d632020dc612a7f3e14a124fc7f1e0e33f9d41b6f4e4a85a"
    )
    assert not data_root.exists()


def test_deposit_refuses_to_overwrite_a_differing_mirror_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts = tmp_path / "counts.txt.gz"
    counts.write_bytes(b"synthetic GEO counts")
    monkeypatch.setattr(
        v, "DATA_SHA256", hashlib.sha256(b"synthetic GEO counts").hexdigest()
    )
    dest = v.raw_mirror_dir(str(tmp_path / "dr")) / v.DATA_REL
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"other bytes")
    with pytest.raises(RuntimeError) as err:
        v.deposit_raw_mirror(counts_path=counts, data_root=str(tmp_path / "dr"))
    assert str(err.value) == f"{dest} exists with a different sha256; refusing"
    assert dest.read_bytes() == b"other bytes"


def test_schema_classes_raw_file_and_the_inline_stubs() -> None:
    """The class wiring; ``preprocess_raw`` is an identity and ``create_experiment``
    refuses because both steps live inside ``process``.
    """
    dataset = v.EnvChemgenVanacloig2022Dataset.__new__(v.EnvChemgenVanacloig2022Dataset)
    assert dataset.experiment_class is StrainEnvironmentResponseExperiment
    assert dataset.reference_class is StrainEnvironmentResponseExperimentReference
    assert dataset.raw_file_names == ["GSE186866_ChemGenomics_Raw_Counts_matrix.txt.gz"]
    frame = pd.DataFrame({"a": [1]})
    assert dataset.preprocess_raw(frame) is frame
    with pytest.raises(NotImplementedError):
        dataset.create_experiment()


# Phase 24: the drop-accounting guard and the TMM reference fallback


def test_a_drop_log_that_disagrees_with_its_rules_refuses_after_writing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``built`` fixture's setup with ``DropLog`` reporting one extra dropped record.

    That fixture drops 0 records and its rules total 0
    (``test_every_source_record_is_stored``); the log says 1, so the build raises the
    exact mismatch message after the tampered log is on disk.
    """
    real = v.DropLog

    def inflated(**kwargs: Any) -> Any:
        kwargs["dropped_records"] += 1
        return real(**kwargs)

    monkeypatch.setattr(v, "DropLog", inflated)
    monkeypatch.setattr(v.EnvChemgenVanacloig2022Dataset, "download", lambda self: None)
    monkeypatch.setattr(
        v.EnvChemgenVanacloig2022Dataset, "_load_matrix", lambda self: _matrix()
    )
    root = tmp_path / "env_chemgen_vanacloig2022"
    (root / "raw").mkdir(parents=True)
    with gzip.open(root / "raw" / v.DATA_FILENAME, "wt") as handle:
        handle.write("placeholder\n")
    with pytest.raises(RuntimeError) as err:
        v.EnvChemgenVanacloig2022Dataset(root=str(root))
    assert str(err.value) == (
        "drop accounting mismatch: rules total 0, 1 records missing from the build"
    )
    log = json.loads((root / "preprocess" / "dropped_records.json").read_text())
    assert log["dropped_records"] == 1


def test_tmm_reference_falls_back_to_the_largest_sqrt_sum_when_fractions_vanish() -> (
    None
):
    """Library sizes of 1e30 push every 75th-percentile fraction below 1e-20.

    Counts are constant per column (1, 9, 4 over four genes), so the fractions are
    1e-30, 9e-30, 4e-30: the usual rule (closest to the mean 4.67e-30) would pick ``c``.
    The fallback takes ``argmax(sum(sqrt(counts)))`` over (4, 12, 8), i.e. ``b``. The
    same counts with unit library sizes keep the usual rule and pick ``c``.
    """
    counts = np.array([[1.0, 9.0, 4.0]] * 4)
    factors = v.tmm_factors(counts, np.array([1e30, 1e30, 1e30]), ["a", "b", "c"])
    assert factors.reference_column == "b"
    assert factors.library_sizes == [1e30, 1e30, 1e30]
    usual = v.tmm_factors(counts, np.array([1.0, 1.0, 1.0]), ["a", "b", "c"])
    assert usual.reference_column == "c"
