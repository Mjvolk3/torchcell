"""Every perturbation leaf a mapped loader builds is served by a node lane of its conf.

The adapter's edge methods walk EVERY perturbation of a genotype and of an environment,
while its node methods partition the leaves by class
(``torchcell.knowledge_graphs.perturbation_lanes``). A loader that builds a leaf under a
conf enabling the edge method and no node lane holding that leaf writes a ``perturbation
member of`` row whose start node does not exist, which the full import's
``skip_bad_relationships: true`` drops in silence: the Carruthers
``GeneAdditionPerturbation`` records under the bacterial-only confs lost their rows that
way (ontology unification audit C 1.3, 2026-10-15).

The rule runs on each loader's DIRECT schema imports (``loader_schema_deps``), not the
closure: the closure reaches every base class a leaf subclasses and flags 55 of 170
mapped datasets for leaves they never build (review 3 finding 6). A leaf counts only when
the matching edge method is enabled, so the seven natural-isolate confs, which omit
``perturbation to genotype (chunked)`` by design, need no allowlist. The one allowlist is
for a multi-class module (the #743 shape): a class whose module imports a leaf for a
SIBLING class in the same file, each entry naming that sibling and why.

Three datasets failed the rule before the ontology unification's adapter stage (one
``_perturbation_node`` and one ``_environment_perturbation_node`` with the label chosen
by leaf, the host-keyed lanes retired): the two live Carruthers stores
(``GeneAdditionPerturbation`` under the bacterial-only confs) and Mutalik 2020
(``SmallMoleculePerturbation`` under the phage-only conf). They were ``xfail(strict=True)``
while the lanes were keyed by conf; with one lane per axis the rule reduces to "the edge
method is enabled only with its node method", and every mapped dataset passes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import torchcell
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
from torchcell.knowledge_graphs.kg_manifest import dataset_conf_methods, loader_relpath
from torchcell.knowledge_graphs.perturbation_lanes import (
    LaneGaps,
    loader_leaf_imports,
    unserved_leaves,
)
from torchcell.provenance.schema_deps import load_default_surface

REPO_ROOT = Path(torchcell.__file__).resolve().parent.parent
SURFACE = load_default_surface()

#: Leaves a multi-class module imports for a SIBLING class, keyed by the class the rule
#: would otherwise flag. Each entry is checked both ways below: the import must still
#: be there (an entry for a module that dropped the import is dead), and the named
#: sibling must be a mapped class of the same module that serves the leaf.
SIBLING_IMPORT_ALLOWLIST: dict[str, dict[str, str]] = {
    # menasalvas2025.py: the selection class (2138-2798) builds no called variant; the
    # variant leaves are imported for ProteomeMenasalvas2025Dataset (3433), whose conf
    # enables the bacterial sequence variant lane (review 3 finding 6).
    "IsoprenolSelectionMenasalvas2025Dataset": {
        "LocusVariantCallPerturbation": "ProteomeMenasalvas2025Dataset",
        "SiteVariantCallPerturbation": "ProteomeMenasalvas2025Dataset",
    },
    # carruthers2025.py: GeneAdditionPerturbation (line 200) is built by
    # ``native_copy_perturbation`` for the titer and proteome classes; the campaign
    # and fold-change classes build 0 of them (audit C 1.3 item 1, dev stores counted).
    "CampaignProteomeCarruthers2025Dataset": {
        "GeneAdditionPerturbation": "IsoprenolTiterCarruthers2025Dataset"
    },
    "ProteomeFoldChangeCarruthers2025Dataset": {
        "GeneAdditionPerturbation": "IsoprenolTiterCarruthers2025Dataset"
    },
}

MAPPED = {cls.__name__: cls for cls in build_adapter_map(include_private=True)}


def _gaps_of(dataset_class: type) -> LaneGaps:
    """The rule for one mapped dataset, with the sibling allowlist applied."""
    imports = loader_leaf_imports(
        REPO_ROOT / loader_relpath(dataset_class, REPO_ROOT), SURFACE
    )
    sibling = SIBLING_IMPORT_ALLOWLIST.get(dataset_class.__name__, {})
    return unserved_leaves(
        [leaf for leaf in imports.genotype if leaf not in sibling],
        [leaf for leaf in imports.environment if leaf not in sibling],
        dataset_conf_methods(dataset_class, REPO_ROOT),
    )


@pytest.mark.parametrize("dataset_name", sorted(MAPPED))
def test_every_imported_perturbation_leaf_is_served_by_an_enabled_lane(
    dataset_name: str,
) -> None:
    gaps = _gaps_of(MAPPED[dataset_name])
    assert gaps.empty, f"{dataset_name}: {gaps.describe()}"


def test_the_known_gaps_are_exactly_the_three_the_unification_closes() -> None:
    """The three gaps of the conf-keyed lanes are closed; the failing set is empty."""
    failing = sorted(name for name, cls in MAPPED.items() if not _gaps_of(cls).empty)
    assert failing == []


@pytest.mark.parametrize("dataset_name", sorted(SIBLING_IMPORT_ALLOWLIST))
def test_each_sibling_allowlist_entry_is_still_needed_and_names_a_serving_sibling(
    dataset_name: str,
) -> None:
    """An allowlist entry is live only while the import exists and names a real sibling.

    If the module stops importing the leaf, the entry is dead and must go. The sibling
    must be a mapped class of the SAME module that is not itself allowlisted for the
    leaf, so the import is attributed to the class that builds it; whether that
    class's conf serves the leaf is that class's own row of the rule above (the
    Carruthers titer store is an xfail there today).
    """
    dataset_class = MAPPED[dataset_name]
    imports = loader_leaf_imports(
        REPO_ROOT / loader_relpath(dataset_class, REPO_ROOT), SURFACE
    )
    for leaf, sibling_name in SIBLING_IMPORT_ALLOWLIST[dataset_name].items():
        assert leaf in imports.genotype + imports.environment, (
            f"{dataset_name}: the allowlisted {leaf} is no longer imported; drop the entry"
        )
        sibling = MAPPED[sibling_name]
        assert sibling is not dataset_class
        assert loader_relpath(sibling, REPO_ROOT) == loader_relpath(
            dataset_class, REPO_ROOT
        )
        assert leaf not in SIBLING_IMPORT_ALLOWLIST.get(sibling_name, {}), (
            f"{sibling_name} is allowlisted for {leaf} too; nobody builds it"
        )
        assert _gaps_of(sibling).empty
