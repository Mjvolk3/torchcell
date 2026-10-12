# torchcell/candidates/store.py
# [[torchcell.candidates.store]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/store.py
# Test file: tests/torchcell/candidates/test_store.py
"""The tracked verdict store: ``database/candidates/<citation_key>.json``, one per key.

The store is authoritative (plan decision 7): the candidate tables render a verdict
column from it and never carry a verdict in a row literal, and the enforcement test reads
it. It sits beside ``database/releases/`` because a verdict is a release-adjacent record,
and outside the supported-queries hook pattern, which anchors on ``database/releases/``.

``GRANDFATHERED`` is every dataset class registered when the gate landed (2026-10-10 at main
ecbd943ac, 134 classes, derived from ``dataset_registry`` after importing every module under
``torchcell/datasets``). It is sorted and frozen and can only shrink: a class leaves it
when its verdict is written (the aggregation backfill starts with SynthLethDB). A class
registered after that date is not in it and must carry ``CITATION_KEY`` and a passing
verdict.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from torchcell.candidates.verdict import CandidateVerdict

REPO: Final = Path(__file__).resolve().parents[2]
STORE_DIR: Final = REPO / "database" / "candidates"
SUFFIX: Final = ".json"

GRANDFATHERED: Final[tuple[str, ...]] = (
    "AminoAcidCooper2010Dataset",
    "AminoAcidMulleder2016Dataset",
    "BetaxanthinCachera2023Dataset",
    "BiofuelTiterBrunk2016Dataset",
    "Bloom2019Dataset",
    "CampaignProteomeCarruthers2025Dataset",
    "CarbonSourceTong2020Dataset",
    "CarotenoidOzaydin2013Dataset",
    "CaudalPanTranscriptome2024Dataset",
    "CrisprMagicLian2019Dataset",
    "CrisprPineneToleranceNiu2019Dataset",
    "CrispriArrayYunus2026Dataset",
    "CrispriChemgenChoe2025Dataset",
    "CrispriChemgenSmith2016Dataset",
    "CrispriCrossRachwalski2024Dataset",
    "CrispriDifferentialProteomeYunus2026Dataset",
    "CrispriGuideFfaEnrichmentFang2025Dataset",
    "CrispriGuideFitnessWang2018Dataset",
    "CrispriKnockdownCui2018Dataset",
    "CrispriKnockdownYunus2026Dataset",
    "CrispriMormino2022Dataset",
    "CrispriPanelProteomeYunus2026Dataset",
    "CrispriScreenRousset2018Dataset",
    "DmMicroarraySameith2015Dataset",
    "DmfCostanzo2016Dataset",
    "DmfKuzmin2018Dataset",
    "DmfKuzmin2020Dataset",
    "DmiCostanzo2016Dataset",
    "DmiKuzmin2018Dataset",
    "DmiKuzmin2020Dataset",
    "DoublingTimeCaglar2017Dataset",
    "EnvChemgenAuesukaree2009Dataset",
    "EnvChemgenCostanzo2021Dataset",
    "EnvChemgenGirgis2009Dataset",
    "EnvChemgenHoepfner2014Dataset",
    "EnvChemgenMota2024Dataset",
    "EnvChemgenShiver2016Dataset",
    "EnvChemgenVanacloig2022Dataset",
    "EnvChemgenWang2015Dataset",
    "EnvChemgenWang2024Dataset",
    "EnvChemgenWildenhain2015Dataset",
    "EnvMetalTnseqRoyet2025Dataset",
    "ExometaboliteBrunk2016Dataset",
    "FattyAcidSmith2006Dataset",
    "FattyAcidXue2025Dataset",
    "FluxIshii2007Dataset",
    "GeneEssentialityGoodall2018Dataset",
    "GeneEssentialityPrice2018EcoliDataset",
    "GeneEssentialitySgdDataset",
    "GeneInteractionBabu2014Dataset",
    "GeneInteractionButland2008Dataset",
    "GrowthAucRapp2026Dataset",
    "GrowthRateCampos2018Dataset",
    "GrowthRateChoe2019Dataset",
    "GrowthRateLamoureux2023Dataset",
    "GrowthRateS23Schmidt2016Dataset",
    "GrowthRateSchmidt2016Dataset",
    "GrowthWang2015Dataset",
    "HetHillenmeyer2008Dataset",
    "HomHillenmeyer2008Dataset",
    "InhibitorBioscreenVolk2021Dataset",
    "IsobutanolScreenLopez2024Dataset",
    "IsobutanolValidatedLopez2024Dataset",
    "IsopentenolTiterFoo2014Dataset",
    "IsoprenolSelectionMenasalvas2025Dataset",
    "IsoprenolTiterCarruthers2025Dataset",
    "IsoprenolTiterDeSiqueira2025Dataset",
    "IsoprenolTiterYunus2026Dataset",
    "IsoprenolToleranceLim2025Dataset",
    "IsoprenylAcetateTiterKang2026Dataset",
    "LactamGrowthRateThompson2019Dataset",
    "MetaboliteDaSilveira2014Dataset",
    "MetaboliteGrowthPhaseMenasalvas2025Dataset",
    "MetaboliteIntensityRapp2026Dataset",
    "MetaboliteProductionPhaseMenasalvas2025Dataset",
    "MetaboliteZelezniak2018Dataset",
    "MetabolomeBrunk2016Dataset",
    "MetabolomeFuhrer2017Dataset",
    "MetabolomeIshii2007Dataset",
    "MetabolomeRapp2026Dataset",
    "MetabolomeSchastnaya2021Dataset",
    "MicroarrayKemmeren2014Dataset",
    "MismatchCrispriFitnessHawkins2020Dataset",
    "MorphologyCampos2018Dataset",
    "NadalRibellesPerturbSeq2025Dataset",
    "OrganicAcidYoshida2012Dataset",
    "PhageRbTnseqMutalik2020Dataset",
    "PromoterReporterMohiuddin2022Dataset",
    "ProteinFoldChangeCaglar2017Dataset",
    "ProteinTurnoverGupta2024Dataset",
    "ProteomeBanerjee2025Dataset",
    "ProteomeBrunk2016Dataset",
    "ProteomeCaglar2017Dataset",
    "ProteomeCarruthers2025Dataset",
    "ProteomeDeSiqueira2025Dataset",
    "ProteomeFoldChangeCarruthers2025Dataset",
    "ProteomeFoldChangeLim2025Dataset",
    "ProteomeIshii2007Dataset",
    "ProteomeLim2025Dataset",
    "ProteomeLog10PercentDeSiqueira2025Dataset",
    "ProteomeMenasalvas2025Dataset",
    "ProteomeMessner2023Dataset",
    "ProteomeMori2021Dataset",
    "ProteomePercentDeSiqueira2025Dataset",
    "ProteomeSchmidt2016Dataset",
    "ProteomeSrmSet1Schmidt2016Dataset",
    "ProteomeSrmSet2Schmidt2016Dataset",
    "ProteomeZelezniak2018Dataset",
    "PutidaPrecise321Lim2022Dataset",
    "RbTnseqBorchert2023Dataset",
    "RbTnseqBorchert2024Dataset",
    "RbTnseqPrice2018EcoliDataset",
    "RnaseqCaglar2017Dataset",
    "RnaseqLamoureux2023Dataset",
    "RnaseqPublicK12Lamoureux2023Dataset",
    "ScmdOhnuki2018Dataset",
    "ScmdOhnuki2022Dataset",
    "ScmdOhya2005Dataset",
    "SmMicroarraySameith2015Dataset",
    "SmfBaryshnikova2010Dataset",
    "SmfCostanzo2016Dataset",
    "SmfKuzmin2018Dataset",
    "SmfKuzmin2020Dataset",
    "SmfODuibhir2014Dataset",
    "SynthLethalityYeastSynthLethDbDataset",
    "SynthRescueYeastSynthLethDbDataset",
    "TargetedMetabolomeRapp2026Dataset",
    "TmfKuzmin2018Dataset",
    "TmfKuzmin2020Dataset",
    "TmiKuzmin2018Dataset",
    "TmiKuzmin2020Dataset",
    "TranscriptionFactorKnockoutChoe2019Dataset",
    "ValerolactamTiterThompson2019Dataset",
    "YeastPhenomeDataset",
)


def verdict_path(citation_key: str, store_dir: Path = STORE_DIR) -> Path:
    """Where a key's verdict lives."""
    if not citation_key or "/" in citation_key or citation_key.startswith("."):
        raise ValueError(f"not a citation key: {citation_key!r}")
    return store_dir / f"{citation_key}{SUFFIX}"


def write_verdict(verdict: CandidateVerdict, store_dir: Path = STORE_DIR) -> Path:
    """Write (or replace) a key's verdict; git keeps the history."""
    path = verdict_path(verdict.citation_key, store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(verdict.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def read_verdict(citation_key: str, store_dir: Path = STORE_DIR) -> CandidateVerdict:
    """A key's verdict; a missing file raises ``FileNotFoundError``."""
    path = verdict_path(citation_key, store_dir)
    verdict = CandidateVerdict.model_validate_json(path.read_text(encoding="utf-8"))
    if verdict.citation_key != citation_key:
        raise ValueError(f"{path} holds the verdict of {verdict.citation_key!r}")
    return verdict


def load_store(store_dir: Path = STORE_DIR) -> dict[str, CandidateVerdict]:
    """Every verdict in the store, by citation key (empty for an absent directory)."""
    if not store_dir.is_dir():
        return {}
    return {
        path.stem: read_verdict(path.stem, store_dir)
        for path in sorted(store_dir.glob(f"*{SUFFIX}"))
    }


def verdicts_by_row(
    table: str, store_dir: Path = STORE_DIR
) -> dict[str, CandidateVerdict]:
    """One table's verdicts keyed by row name; two verdicts for one row raise."""
    out: dict[str, CandidateVerdict] = {}
    for verdict in load_store(store_dir).values():
        if verdict.table != table:
            continue
        if verdict.row_name in out:
            raise ValueError(
                f"row {verdict.row_name!r} has two verdicts: "
                f"{out[verdict.row_name].citation_key!r} and {verdict.citation_key!r}"
            )
        out[verdict.row_name] = verdict
    return out


def citation_key_of(cls: type) -> str | None:
    """A dataset class's ``CITATION_KEY``: the module-level constant of its module."""
    key = getattr(sys.modules[cls.__module__], "CITATION_KEY", None)
    return key if isinstance(key, str) else None


def enforcement_violations(
    registry: Mapping[str, type],
    grandfathered: Sequence[str],
    verdicts: Mapping[str, CandidateVerdict],
) -> list[str]:
    """Every registered class outside ``grandfathered`` lacking a key or a passing verdict.

    Passing is ``admissible`` or ``admissible_with_gaps``
    (:attr:`CandidateVerdict.passing`). The returned lines name the class and the reason,
    sorted by class name.
    """
    exempt = set(grandfathered)
    out: list[str] = []
    for name in sorted(registry):
        if name in exempt:
            continue
        key = citation_key_of(registry[name])
        if key is None:
            out.append(f"{name}: its module declares no CITATION_KEY")
        elif key not in verdicts:
            out.append(f"{name}: no verdict at database/candidates/{key}.json")
        elif not verdicts[key].passing:
            out.append(f"{name}: verdict {key} is {verdicts[key].outcome}")
    return out
