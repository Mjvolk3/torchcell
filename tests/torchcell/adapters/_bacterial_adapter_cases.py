# tests/torchcell/adapters/_bacterial_adapter_cases.py
# [[tests.torchcell.adapters._bacterial_adapter_cases]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/adapters/_bacterial_adapter_cases.py
"""Shared cases and checks for the bacterial adapters (not a test module).

Every E. coli and P. putida dataset class (plan.bacteria-ontology-genome step 9) has
its own adapter module, its own conf and its own paired test file,
``test_<module>.py``; each of those runs the checks here for its one dataset, and
``test_bacterial_adapters.py`` holds the checks across the whole set. The checks:

* ``assert_construction`` / ``assert_missing_conf`` (``_adapter_init_harness``): the
  constructor loads exactly the conf its graph shape implies and refuses a missing conf
  by its exact path, with the perturbation nodes the ``bacterial perturbation`` class
  rather than the served yeast ``perturbation`` class;
* ``assert_conf_registered_and_declared``: every conf method name is registered on
  ``CellAdapter`` and every node class the conf can emit, the phenotype included, is
  declared in ``torchcell_schema_config.yaml`` (BioCypher drops an undeclared class
  silently), read through the admission gate's own ``dataset_conf_methods``;
* ``assert_gate_resolves_own_files``: ``kg_manifest`` resolves the dataset to ITS
  adapter module and ITS conf. A module holding two adapter classes resolves both to the
  first conf it names, which is why every bacterial dataset class has a module of its
  own;
* ``assert_dev_store_graph`` (data-gated, ``--data`` with a real ``DATA_ROOT``): the
  adapter runs every enabled method over the first records of its dev-tree LMDB the way
  the build runs them (one single-pass traversal for the chunked nodes, one for the
  chunked edges); the emitted graph is closed (every edge endpoint is an emitted node),
  every label and property is declared, and every sub-object family the conf leaves off
  is absent from the records. For a phage dataset ``_phage_pair`` additionally runs the
  phage method and the served environment-perturbation method over the same records and
  requires the same node ids from both, which is the record-level form of the one-class
  rule; it runs over a second view built from a phage-bearing reference's own records,
  since the leading records of a store can all be unchallenged controls. A store that is
  absent or mid-rebuild (no ``build_manifest.json``) is skipped, and so is one without a
  cached ``experiment_reference_index.json``, because computing that index would write it
  into the store.
"""

from __future__ import annotations

import importlib
import inspect
import os
import os.path as osp
from pathlib import Path
from types import ModuleType
from typing import Any, NamedTuple

import pytest
import yaml

import torchcell
import torchcell.datasets.ecoli  # noqa: F401  # populates the registry
import torchcell.datasets.pputida  # noqa: F401  # populates the registry
from tests.torchcell.adapters._adapter_init_harness import (
    AdapterCase,
    Shape,
    install_recorder,
)
from torchcell.adapters import (
    CarbonSourceTong2020Adapter,
    CrispriArrayYunus2026Adapter,
    CrispriChemgenChoe2025Adapter,
    CrispriDifferentialProteomeYunus2026Adapter,
    CrispriGuideFitnessWang2018Adapter,
    CrispriKnockdownCui2018Adapter,
    CrispriKnockdownYunus2026Adapter,
    CrispriScreenRousset2018Adapter,
    EnvChemgenGirgis2009Adapter,
    EnvChemgenShiver2016Adapter,
    EnvChemgenWang2015Adapter,
    GeneEssentialityGoodall2018Adapter,
    GeneEssentialityPrice2018EcoliAdapter,
    GeneInteractionBabu2014Adapter,
    GrowthAucRapp2026Adapter,
    GrowthRateCampos2018Adapter,
    GrowthRateChoe2019Adapter,
    GrowthRateSchmidt2016Adapter,
    IsopentenolTiterFoo2014Adapter,
    IsoprenolSelectionMenasalvas2025Adapter,
    IsoprenolTiterCarruthers2025Adapter,
    IsoprenolTiterDeSiqueira2025Adapter,
    IsoprenolToleranceLim2025Adapter,
    IsoprenylAcetateTiterKang2026Adapter,
    MetaboliteIntensityRapp2026Adapter,
    MetabolomeFuhrer2017Adapter,
    MetabolomeRapp2026Adapter,
    MetabolomeSchastnaya2021Adapter,
    PhageRbTnseqMutalik2020Adapter,
    ProteinTurnoverGupta2024Adapter,
    ProteomeCaglar2017Adapter,
    ProteomeCarruthers2025Adapter,
    ProteomeDeSiqueira2025Adapter,
    ProteomeLim2025Adapter,
    ProteomeLog10PercentDeSiqueira2025Adapter,
    ProteomeMori2021Adapter,
    ProteomePercentDeSiqueira2025Adapter,
    ProteomeSchmidt2016Adapter,
    ProteomeSrmSet1Schmidt2016Adapter,
    ProteomeSrmSet2Schmidt2016Adapter,
    PutidaPrecise321Lim2022Adapter,
    RbTnseqBorchert2023Adapter,
    RbTnseqBorchert2024Adapter,
    RbTnseqPrice2018EcoliAdapter,
    RnaseqCaglar2017Adapter,
    RnaseqLamoureux2023Adapter,
    RnaseqPublicK12Lamoureux2023Adapter,
    TargetedMetabolomeRapp2026Adapter,
    TranscriptionFactorKnockoutChoe2019Adapter,
)
from torchcell.adapters.cell_adapter import SINGLE_PASS_EDGES, SINGLE_PASS_NODES
from torchcell.datamodels.schema import PhagePerturbation
from torchcell.datasets.dataset_registry import dataset_registry
from torchcell.datasets.ecoli.babu2014 import GeneInteractionBabu2014Dataset
from torchcell.datasets.ecoli.caglar2017 import (
    ProteomeCaglar2017Dataset,
    RnaseqCaglar2017Dataset,
)
from torchcell.datasets.ecoli.campos2018 import GrowthRateCampos2018Dataset
from torchcell.datasets.ecoli.choe2019_growth_rate import (
    GrowthRateChoe2019Dataset,
    TranscriptionFactorKnockoutChoe2019Dataset,
)
from torchcell.datasets.ecoli.choe2025 import CrispriChemgenChoe2025Dataset
from torchcell.datasets.ecoli.cui2018 import CrispriKnockdownCui2018Dataset
from torchcell.datasets.ecoli.foo2014 import IsopentenolTiterFoo2014Dataset
from torchcell.datasets.ecoli.fuhrer2017 import MetabolomeFuhrer2017Dataset
from torchcell.datasets.ecoli.girgis2009 import EnvChemgenGirgis2009Dataset
from torchcell.datasets.ecoli.goodall2018 import GeneEssentialityGoodall2018Dataset
from torchcell.datasets.ecoli.gupta2024 import ProteinTurnoverGupta2024Dataset
from torchcell.datasets.ecoli.lamoureux2023 import RnaseqLamoureux2023Dataset
from torchcell.datasets.ecoli.lamoureux2023_public_k12 import (
    RnaseqPublicK12Lamoureux2023Dataset,
)
from torchcell.datasets.ecoli.mori2021 import ProteomeMori2021Dataset
from torchcell.datasets.ecoli.mutalik2020 import PhageRbTnseqMutalik2020Dataset
from torchcell.datasets.ecoli.price2018 import (
    GeneEssentialityPrice2018EcoliDataset,
    RbTnseqPrice2018EcoliDataset,
)
from torchcell.datasets.ecoli.rapp2026 import MetabolomeRapp2026Dataset
from torchcell.datasets.ecoli.rapp2026_platforms import (
    GrowthAucRapp2026Dataset,
    MetaboliteIntensityRapp2026Dataset,
    TargetedMetabolomeRapp2026Dataset,
)
from torchcell.datasets.ecoli.rousset2018 import CrispriScreenRousset2018Dataset
from torchcell.datasets.ecoli.schastnaya2021 import MetabolomeSchastnaya2021Dataset
from torchcell.datasets.ecoli.schmidt2016 import ProteomeSchmidt2016Dataset
from torchcell.datasets.ecoli.schmidt2016_growth_rate import (
    GrowthRateSchmidt2016Dataset,
)
from torchcell.datasets.ecoli.schmidt2016_srm import (
    ProteomeSrmSet1Schmidt2016Dataset,
    ProteomeSrmSet2Schmidt2016Dataset,
)
from torchcell.datasets.ecoli.shiver2016 import EnvChemgenShiver2016Dataset
from torchcell.datasets.ecoli.tong2020 import CarbonSourceTong2020Dataset
from torchcell.datasets.ecoli.wang2015 import EnvChemgenWang2015Dataset
from torchcell.datasets.ecoli.wang2018 import CrispriGuideFitnessWang2018Dataset
from torchcell.datasets.pputida.borchert2023 import RbTnseqBorchert2023Dataset
from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset
from torchcell.datasets.pputida.carruthers2025 import (
    IsoprenolTiterCarruthers2025Dataset,
    ProteomeCarruthers2025Dataset,
)
from torchcell.datasets.pputida.desiqueira2025 import (
    IsoprenolTiterDeSiqueira2025Dataset,
    ProteomeDeSiqueira2025Dataset,
    ProteomeLog10PercentDeSiqueira2025Dataset,
    ProteomePercentDeSiqueira2025Dataset,
)
from torchcell.datasets.pputida.kang2026 import IsoprenylAcetateTiterKang2026Dataset
from torchcell.datasets.pputida.lim2022 import PutidaPrecise321Lim2022Dataset
from torchcell.datasets.pputida.lim2025 import (
    IsoprenolToleranceLim2025Dataset,
    ProteomeLim2025Dataset,
)
from torchcell.datasets.pputida.menasalvas2025 import (
    IsoprenolSelectionMenasalvas2025Dataset,
)
from torchcell.datasets.pputida.yunus2026 import (
    CrispriArrayYunus2026Dataset,
    CrispriDifferentialProteomeYunus2026Dataset,
    CrispriKnockdownYunus2026Dataset,
)
from torchcell.knowledge_graphs.kg_manifest import (
    CELL_ADAPTER_RELPATH,
    SCHEMA_CONFIG_RELPATH,
    GraphSchemaEntry,
    cell_adapter_surface,
    dataset_adapter_files,
    dataset_conf_methods,
    graph_schema_from_yaml,
)

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent
KG_BACTERIA = REPO_ROOT / "torchcell/knowledge_graphs/conf/kg_bacteria.yaml"
BACTERIAL_PACKAGES = ("torchcell.datasets.ecoli.", "torchcell.datasets.pputida.")


class Bacterial(NamedTuple):
    """One adapter: its harness case and the module stem that holds it."""

    case: AdapterCase
    module: str


def _case(
    adapter_cls: type[Any],
    module: str,
    slug: str,
    dataset_cls: type[Any],
    phenotype: str,
    *,
    perturbation: bool = True,
    crispr: bool = False,
    env_perturbation: bool = True,
    phage: bool = False,
) -> Bacterial:
    shape = Shape(
        phenotype,
        perturbation=perturbation,
        crispr=crispr,
        env_perturbation=env_perturbation,
        phage=phage,
        bacterial=True,
    )
    return Bacterial(
        AdapterCase(adapter_cls, f"{slug}_adapter.yaml", shape, dataset_cls),
        f"{module}_adapter",
    )


RNASEQ = "rnaseq expression phenotype"
PROTEOME = "protein abundance phenotype"
TITER = "product titer phenotype"
RESPONSE = "environment response phenotype"
TURNOVER = "protein turnover phenotype"
INTERACTION = "gene interaction phenotype"

# The shape of each dataset's records, measured on its dev-tree LMDB on 2026-10-07 (and,
# for the two RB-TnSeq stores rebuilding at the time, read off `build_genotype` /
# `build_environment` in the loader). Caglar 2017 is a wild-type panel with no
# perturbation in any record, and so are Schmidt 2016 and its two SRM arms, whose
# paper's three deletion strains carry no abundance data and are served instead as the
# six Table S24 fitness records, and Mori 2021, whose engineered
# NCM3722 derivatives are all in samples dropped on their medium; Fuhrer 2017, Goodall
# 2018 and Price 2018's Table S1 essentiality calls (one library-selection
# environment, measured 2026-10-08) carry no environment perturbation, and neither
# do Campos 2018, whose
# screen is one medium at one temperature, nor either Choe 2019 arm, whose two
# panels are one M9 glucose medium with no added compound, nor Schastnaya 2021, whose carbon source is
# part of the medium; the CRISPRi leaves of Carruthers, Choe, Cui, Menasalvas, Wang 2018 and
# Yunus carry a CrisprConstruct. Shiver 2016's three temperature-only conditions
# carry no environment perturbation, but its other 54 do, so its pair is enabled.
BACTERIAL: list[Bacterial] = [
    _case(
        GeneInteractionBabu2014Adapter,
        "babu2014",
        "gene_interaction_babu2014",
        GeneInteractionBabu2014Dataset,
        INTERACTION,
        env_perturbation=False,
    ),
    _case(
        RnaseqCaglar2017Adapter,
        "caglar2017_rnaseq",
        "rnaseq_caglar2017",
        RnaseqCaglar2017Dataset,
        RNASEQ,
        perturbation=False,
    ),
    _case(
        ProteomeCaglar2017Adapter,
        "caglar2017_proteome",
        "proteome_caglar2017",
        ProteomeCaglar2017Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        GrowthRateCampos2018Adapter,
        "campos2018",
        "ecoli_growth_rate_campos2018",
        GrowthRateCampos2018Dataset,
        "fitness phenotype",
        env_perturbation=False,
    ),
    _case(
        GrowthRateChoe2019Adapter,
        "choe2019_growth_rate",
        "growth_rate_choe2019",
        GrowthRateChoe2019Dataset,
        "fitness phenotype",
        env_perturbation=False,
    ),
    _case(
        TranscriptionFactorKnockoutChoe2019Adapter,
        "choe2019_tf_knockout",
        "tf_knockout_growth_choe2019",
        TranscriptionFactorKnockoutChoe2019Dataset,
        "fitness phenotype",
        env_perturbation=False,
    ),
    _case(
        CrispriChemgenChoe2025Adapter,
        "choe2025",
        "crispri_chemgen_choe2025",
        CrispriChemgenChoe2025Dataset,
        RESPONSE,
        crispr=True,
    ),
    _case(
        CrispriKnockdownCui2018Adapter,
        "cui2018",
        "crispri_knockdown_cui2018",
        CrispriKnockdownCui2018Dataset,
        RESPONSE,
        crispr=True,
    ),
    _case(
        IsopentenolTiterFoo2014Adapter,
        "foo2014",
        "isopentenol_titer_foo2014",
        IsopentenolTiterFoo2014Dataset,
        TITER,
    ),
    _case(
        MetabolomeFuhrer2017Adapter,
        "fuhrer2017",
        "metabolome_fuhrer2017",
        MetabolomeFuhrer2017Dataset,
        "metabolite phenotype",
        env_perturbation=False,
    ),
    _case(
        EnvChemgenGirgis2009Adapter,
        "girgis2009",
        "ecoli_env_chemgen_girgis2009",
        EnvChemgenGirgis2009Dataset,
        RESPONSE,
    ),
    _case(
        GeneEssentialityGoodall2018Adapter,
        "goodall2018",
        "gene_essentiality_goodall2018",
        GeneEssentialityGoodall2018Dataset,
        "gene essentiality phenotype",
        env_perturbation=False,
    ),
    _case(
        ProteinTurnoverGupta2024Adapter,
        "gupta2024",
        "protein_turnover_gupta2024",
        ProteinTurnoverGupta2024Dataset,
        TURNOVER,
    ),
    _case(
        RnaseqLamoureux2023Adapter,
        "lamoureux2023",
        "rnaseq_lamoureux2023",
        RnaseqLamoureux2023Dataset,
        RNASEQ,
    ),
    _case(
        RnaseqPublicK12Lamoureux2023Adapter,
        "lamoureux2023_public_k12",
        "rnaseq_public_k12_lamoureux2023",
        RnaseqPublicK12Lamoureux2023Dataset,
        RNASEQ,
    ),
    _case(
        ProteomeMori2021Adapter,
        "mori2021",
        "proteome_mori2021",
        ProteomeMori2021Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        PhageRbTnseqMutalik2020Adapter,
        "mutalik2020",
        "phage_rbtnseq_mutalik2020",
        PhageRbTnseqMutalik2020Dataset,
        RESPONSE,
        env_perturbation=False,
        phage=True,
    ),
    _case(
        RbTnseqPrice2018EcoliAdapter,
        "price2018_ecoli",
        "rbtnseq_price2018_ecoli",
        RbTnseqPrice2018EcoliDataset,
        RESPONSE,
    ),
    _case(
        GeneEssentialityPrice2018EcoliAdapter,
        "price2018_ecoli_essentiality",
        "gene_essentiality_price2018_ecoli",
        GeneEssentialityPrice2018EcoliDataset,
        "gene essentiality phenotype",
        env_perturbation=False,
    ),
    _case(
        MetabolomeRapp2026Adapter,
        "rapp2026",
        "metabolome_rapp2026",
        MetabolomeRapp2026Dataset,
        "metabolite phenotype",
        crispr=True,
    ),
    _case(
        GrowthAucRapp2026Adapter,
        "rapp2026_growth",
        "growth_auc_rapp2026",
        GrowthAucRapp2026Dataset,
        "fitness phenotype",
        crispr=True,
    ),
    _case(
        MetaboliteIntensityRapp2026Adapter,
        "rapp2026_intensity",
        "metabolite_intensity_rapp2026",
        MetaboliteIntensityRapp2026Dataset,
        "metabolite phenotype",
        crispr=True,
    ),
    _case(
        TargetedMetabolomeRapp2026Adapter,
        "rapp2026_targeted",
        "targeted_metabolome_rapp2026",
        TargetedMetabolomeRapp2026Dataset,
        "metabolite phenotype",
        crispr=True,
    ),
    _case(
        CrispriScreenRousset2018Adapter,
        "rousset2018",
        "ecoli_crispri_rousset2018",
        CrispriScreenRousset2018Dataset,
        RESPONSE,
        crispr=True,
        env_perturbation=False,
        phage=True,
    ),
    _case(
        MetabolomeSchastnaya2021Adapter,
        "schastnaya2021",
        "metabolome_schastnaya2021",
        MetabolomeSchastnaya2021Dataset,
        "metabolite phenotype",
        env_perturbation=False,
    ),
    _case(
        EnvChemgenShiver2016Adapter,
        "shiver2016",
        "ecoli_env_chemgen_shiver2016",
        EnvChemgenShiver2016Dataset,
        RESPONSE,
    ),
    _case(
        ProteomeSchmidt2016Adapter,
        "schmidt2016",
        "proteome_schmidt2016",
        ProteomeSchmidt2016Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        ProteomeSrmSet1Schmidt2016Adapter,
        "schmidt2016_srm_set1",
        "proteome_srm_set1_schmidt2016",
        ProteomeSrmSet1Schmidt2016Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        ProteomeSrmSet2Schmidt2016Adapter,
        "schmidt2016_srm_set2",
        "proteome_srm_set2_schmidt2016",
        ProteomeSrmSet2Schmidt2016Dataset,
        PROTEOME,
        perturbation=False,
    ),
    _case(
        GrowthRateSchmidt2016Adapter,
        "schmidt2016_growth_rate",
        "growth_rate_schmidt2016",
        GrowthRateSchmidt2016Dataset,
        "fitness phenotype",
    ),
    _case(
        CarbonSourceTong2020Adapter,
        "tong2020",
        "ecoli_carbon_source_tong2020",
        CarbonSourceTong2020Dataset,
        "fitness phenotype",
    ),
    _case(
        EnvChemgenWang2015Adapter,
        "wang2015",
        "env_chemgen_wang2015",
        EnvChemgenWang2015Dataset,
        RESPONSE,
    ),
    _case(
        CrispriGuideFitnessWang2018Adapter,
        "wang2018",
        "crispri_guide_fitness_wang2018",
        CrispriGuideFitnessWang2018Dataset,
        RESPONSE,
        crispr=True,
    ),
    _case(
        RbTnseqBorchert2023Adapter,
        "borchert2023",
        "rbtnseq_borchert2023",
        RbTnseqBorchert2023Dataset,
        RESPONSE,
    ),
    _case(
        RbTnseqBorchert2024Adapter,
        "borchert2024",
        "rbtnseq_borchert2024",
        RbTnseqBorchert2024Dataset,
        RESPONSE,
    ),
    _case(
        IsoprenolTiterCarruthers2025Adapter,
        "carruthers2025_titer",
        "isoprenol_titer_carruthers2025",
        IsoprenolTiterCarruthers2025Dataset,
        TITER,
        crispr=True,
    ),
    _case(
        ProteomeCarruthers2025Adapter,
        "carruthers2025_proteome",
        "proteome_carruthers2025",
        ProteomeCarruthers2025Dataset,
        PROTEOME,
        crispr=True,
    ),
    _case(
        ProteomeDeSiqueira2025Adapter,
        "desiqueira2025_proteome",
        "proteome_desiqueira2025",
        ProteomeDeSiqueira2025Dataset,
        PROTEOME,
    ),
    _case(
        ProteomePercentDeSiqueira2025Adapter,
        "desiqueira2025_proteome_percent",
        "proteome_percent_desiqueira2025",
        ProteomePercentDeSiqueira2025Dataset,
        PROTEOME,
    ),
    _case(
        ProteomeLog10PercentDeSiqueira2025Adapter,
        "desiqueira2025_proteome_log10_percent",
        "proteome_log10_percent_desiqueira2025",
        ProteomeLog10PercentDeSiqueira2025Dataset,
        PROTEOME,
    ),
    _case(
        IsoprenolTiterDeSiqueira2025Adapter,
        "desiqueira2025_titer",
        "isoprenol_titer_desiqueira2025",
        IsoprenolTiterDeSiqueira2025Dataset,
        TITER,
    ),
    _case(
        IsoprenylAcetateTiterKang2026Adapter,
        "kang2026",
        "isoprenyl_acetate_titer_kang2026",
        IsoprenylAcetateTiterKang2026Dataset,
        TITER,
    ),
    _case(
        PutidaPrecise321Lim2022Adapter,
        "lim2022",
        "putida_precise321_lim2022",
        PutidaPrecise321Lim2022Dataset,
        RNASEQ,
    ),
    _case(
        IsoprenolToleranceLim2025Adapter,
        "lim2025_tolerance",
        "isoprenol_tolerance_lim2025",
        IsoprenolToleranceLim2025Dataset,
        RESPONSE,
    ),
    _case(
        ProteomeLim2025Adapter,
        "lim2025_proteome",
        "proteome_lim2025",
        ProteomeLim2025Dataset,
        PROTEOME,
    ),
    _case(
        IsoprenolSelectionMenasalvas2025Adapter,
        "menasalvas2025",
        "isoprenol_selection_menasalvas2025",
        IsoprenolSelectionMenasalvas2025Dataset,
        RESPONSE,
        crispr=True,
    ),
    _case(
        CrispriArrayYunus2026Adapter,
        "yunus2026_array",
        "crispri_array_yunus2026",
        CrispriArrayYunus2026Dataset,
        PROTEOME,
        crispr=True,
    ),
    _case(
        CrispriKnockdownYunus2026Adapter,
        "yunus2026_knockdown",
        "crispri_knockdown_yunus2026",
        CrispriKnockdownYunus2026Dataset,
        PROTEOME,
        crispr=True,
    ),
    _case(
        CrispriDifferentialProteomeYunus2026Adapter,
        "yunus2026_differential",
        "crispri_differential_proteome_yunus2026",
        CrispriDifferentialProteomeYunus2026Dataset,
        PROTEOME,
        crispr=True,
    ),
]
IDS = [b.case.adapter_cls.__name__ for b in BACTERIAL]


def graph_schema() -> dict[str, GraphSchemaEntry]:
    return graph_schema_from_yaml(
        (REPO_ROOT / SCHEMA_CONFIG_RELPATH).read_text(encoding="utf-8")
    )


def node_class(method_name: str) -> str:
    """The graph node class a conf node method writes (``... reference`` included)."""
    label = method_name.removesuffix(" (chunked)")
    if label == "experiment reference":
        return label
    return label.removesuffix(" reference")


def registered_bacterial_classes() -> set[type[Any]]:
    return {
        cls
        for cls in dataset_registry.values()
        if cls.__module__.startswith(BACTERIAL_PACKAGES)
    }


def case_for(dataset_cls: type[Any]) -> Bacterial:
    """The one case whose dataset class is ``dataset_cls``."""
    (match,) = [b for b in BACTERIAL if b.case.dataset_cls is dataset_cls]
    return match


def adapter_module(bacterial: Bacterial) -> ModuleType:
    """The adapter's own module, checked to be the one the case names."""
    module = importlib.import_module(f"torchcell.adapters.{bacterial.module}")
    assert bacterial.case.adapter_cls.__module__ == module.__name__
    return module


def assert_conf_registered_and_declared(bacterial: Bacterial) -> None:
    """Read through the gate's ``dataset_conf_methods``, as ``admit`` reads it."""
    _, table = cell_adapter_surface(
        (REPO_ROOT / CELL_ADAPTER_RELPATH).read_text(encoding="utf-8")
    )
    schema = graph_schema()
    names = dataset_conf_methods(bacterial.case.dataset_cls, REPO_ROOT)
    assert [n for n in names if n not in table] == []
    conf = yaml.safe_load(
        (REPO_ROOT / "torchcell/adapters/conf" / bacterial.case.conf_name).read_text(
            encoding="utf-8"
        )
    )
    node_names = [m["method_name"] for m in conf["cell_adapter"]["node_methods"]]
    classes = {node_class(n) for n in node_names}
    assert sorted(c for c in classes if c not in schema) == []
    assert all(schema[c].kind == "node" for c in classes)
    phenotype = bacterial.case.shape.phenotype
    assert {n for n in names if n.endswith("phenotype (chunked)")} == {
        f"{phenotype} (chunked)"
    }
    # a bacterial leaf is never written under the served yeast class, and no conf here
    # serves a segregant genotype
    assert "perturbation (chunked)" not in names
    assert "segregant genotype (chunked)" not in names
    # a phage challenge is served as `phage perturbation` and NEVER beside
    # `environment perturbation`: the served `_environment_perturbation_node` does not
    # filter phages out, so a conf enabling both writes every phage twice under two
    # labels on one content id (cell_adapter.py, above `_phage_perturbation_node_from`).
    phage = bacterial.case.shape.phage
    assert ("phage perturbation (chunked)" in names) is phage
    assert ("environment perturbation (chunked)" in names) is (
        bacterial.case.shape.env_perturbation
    )
    assert not (phage and bacterial.case.shape.env_perturbation)


def assert_gate_resolves_own_files(bacterial: Bacterial) -> None:
    """``kg_manifest`` fingerprints this dataset's own module and conf."""
    assert dataset_adapter_files(bacterial.case.dataset_cls, REPO_ROOT) == [
        f"torchcell/adapters/{bacterial.module}.py",
        f"torchcell/adapters/conf/{bacterial.case.conf_name}",
    ]


# --------------------------------------------------------------------------- #
# Data-gated: each adapter over its dev-tree LMDB
# --------------------------------------------------------------------------- #
RECORDS = 200
# Chunked node methods a conf enables only when the records carry that sub-object.
ENV_PERTURBATION_NODE = "environment perturbation (chunked)"
ENV_PERTURBATION_NODE_LABEL = "environment perturbation"
PHAGE_NODE = "phage perturbation (chunked)"
PHAGE_NODE_LABEL = "phage perturbation"
OPTIONAL_FAMILIES = (
    "bacterial perturbation (chunked)",
    "crispr construct (chunked)",
    ENV_PERTURBATION_NODE,
    PHAGE_NODE,
)
STORE_FILES = (
    "processed/lmdb",
    "preprocess/build_manifest.json",
    "preprocess/experiment_reference_index.json",
)


def _dev_root(dataset_cls: type[Any]) -> str:
    params = inspect.signature(dataset_cls.__init__).parameters
    return osp.join(os.environ["DATA_ROOT"], params["root"].default)


def _run(adapter: Any, view: Any, kind: str) -> list[Any]:
    """Every enabled method of ``kind``: reference methods, then ONE chunked pass."""
    if kind == "node":
        table, conf, pass_name = (
            adapter.node_methods,
            adapter.config.cell_adapter.node_methods,
            SINGLE_PASS_NODES,
        )
    else:
        table, conf, pass_name = (
            adapter.edge_methods,
            adapter.config.cell_adapter.edge_methods,
            SINGLE_PASS_EDGES,
        )
    enabled = {m["method_name"] for m in conf}
    out: list[Any] = []
    chunked = []
    for name, method in table:
        if name not in enabled:
            continue
        if method.__name__.startswith("_get_"):
            out.extend(method())
        else:
            chunked.append((name, method))
    adapter._single_pass_methods = chunked
    out.extend(adapter._all_chunked(view, pass_name, inprocess=True))
    return out


def _phage_leaves(records: Any) -> int:
    """The phage perturbations the records carry, counted off the records themselves."""
    total = 0
    for i in range(len(records)):
        item = records.transform_item(records[i])
        total += sum(
            isinstance(perturbation, PhagePerturbation)
            for perturbation in item["experiment"].environment.perturbations
        )
    records.close_lmdb()
    return total


def _phage_bearing_view(dataset: Any) -> Any:
    """A view over the records of the first reference whose environment carries a phage.

    The prefix view the other checks run over is NOT guaranteed to hold one: a dataset
    that also measures unchallenged controls can order those records first, and Mutalik
    2020's leading 3,667 records are exactly that, their environment carrying no
    perturbation at all (measured on its dev store, 2026.10.07). ``member_indices`` names
    the records of one reference, so a phage is present in this view by construction, and
    it is read off the reference index the store already caches (``STORE_FILES``) rather
    than found by scanning records.
    """
    phage_references = [
        entry
        for entry in dataset.experiment_reference_index
        if any(
            isinstance(perturbation, PhagePerturbation)
            for perturbation in entry.reference.environment_reference.perturbations
        )
    ]
    assert phage_references, "the shape says phage, but no reference carries one"
    return dataset[phage_references[0].member_indices[:RECORDS]]


def _phage_pair(adapter: Any, records: Any) -> list[Any]:
    """The phage and the environment-perturbation node method over the SAME records.

    A phage conf exempts ``environment perturbation (chunked)`` from the left-off check,
    and this pair is what stands in for it. The two methods must agree node for node by
    id: equal ids prove the served method would emit nothing the phage class does not
    already serve -- one content id written under two labels, which is the double write
    the one-class rule forbids (issue #756) -- and nothing beyond it either, so the
    exemption drops no perturbation the records carry. The phage node count is then
    checked against the phage leaves read straight off the records, so a view that
    carries none asserts that both methods emit nothing instead of passing vacuously.

    Returns the phage nodes.
    """
    by_name = dict(adapter.node_methods)
    ran: dict[str, list[Any]] = {}
    for name in (PHAGE_NODE, ENV_PERTURBATION_NODE):
        adapter._single_pass_methods = [(name, by_name[name])]
        ran[name] = adapter._all_chunked(records, SINGLE_PASS_NODES, inprocess=True)
    assert {n.get_label() for n in ran[PHAGE_NODE]} <= {PHAGE_NODE_LABEL}
    assert {n.get_label() for n in ran[ENV_PERTURBATION_NODE]} <= {
        ENV_PERTURBATION_NODE_LABEL
    }
    assert {n.get_id() for n in ran[ENV_PERTURBATION_NODE]} == {
        n.get_id() for n in ran[PHAGE_NODE]
    }
    assert len(ran[PHAGE_NODE]) == _phage_leaves(records)
    return ran[PHAGE_NODE]


def assert_dev_store_graph(
    bacterial: Bacterial, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter over its dev-tree LMDB emits a closed, declared, lossless graph."""
    root = _dev_root(bacterial.case.dataset_cls)
    missing = [rel for rel in STORE_FILES if not osp.exists(osp.join(root, rel))]
    if missing:
        pytest.skip(f"dev store at {root} is absent or mid-rebuild: no {missing}")
    install_recorder(monkeypatch)
    dataset = bacterial.case.dataset_cls(root=root)
    adapter = bacterial.case.adapter_cls(
        dataset=dataset, process_workers=1, io_workers=0
    )
    view = dataset[0 : min(len(dataset), RECORDS)]

    nodes = _run(adapter, view, "node")
    edges = _run(adapter, view, "edge")

    schema = graph_schema()
    labels = {node.get_label() for node in nodes}
    assert sorted(label for label in labels if label not in schema) == []
    for node in nodes:
        # BioCypherNode.get_properties() also reports the node's id and preferred_id
        declared = set(schema[node.get_label()].properties) | {"id", "preferred_id"}
        assert set(node.get_properties()) <= declared, node.get_label()
    node_ids = {node.get_id() for node in nodes}
    dangling = [
        (edge.get_label(), edge.get_source_id(), edge.get_target_id())
        for edge in edges
        if edge.get_source_id() not in node_ids or edge.get_target_id() not in node_ids
    ]
    assert dangling == []
    shape = bacterial.case.shape
    assert shape.phenotype in labels
    assert "perturbation" not in labels
    assert ("bacterial perturbation" in labels) is shape.perturbation
    assert ("crispr construct" in labels) is shape.crispr
    assert ("environment perturbation" in labels) is shape.env_perturbation
    assert ("phage perturbation" in labels) is shape.phage

    # The converse: a sub-object family the conf leaves OFF is absent from the records,
    # so the enable-list drops nothing they carry. `environment perturbation (chunked)`
    # is exempt for a phage conf, and only there: that method does NOT filter phages
    # out, so running it would re-emit the records' phages under the served label --
    # which is the documented reason a conf enables one of the two classes and never
    # both, not evidence that the phage conf drops anything. What the exemption leaves
    # unproved, `_phage_pair` below proves instead, record for record.
    enabled = {m["method_name"] for m in adapter.config.cell_adapter.node_methods}
    overlapping = {ENV_PERTURBATION_NODE} if bacterial.case.shape.phage else set()
    left_off = [
        (name, method)
        for name, method in adapter.node_methods
        if name in OPTIONAL_FAMILIES and name not in enabled | overlapping
    ]
    adapter._single_pass_methods = left_off
    assert adapter._all_chunked(view, SINGLE_PASS_NODES, inprocess=True) == []

    if not bacterial.case.shape.phage:
        return
    # The phage pair over the first records, which is where the
    # `environment perturbation (chunked)` exemption above applies, and then over records
    # that carry a phage by construction, which is what keeps the pair's property from
    # being vacuous for a dataset whose leading records are unchallenged controls.
    _phage_pair(adapter, view)
    phage_nodes = _phage_pair(adapter, _phage_bearing_view(dataset))
    assert phage_nodes
