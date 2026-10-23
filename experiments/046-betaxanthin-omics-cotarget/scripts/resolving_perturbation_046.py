# experiments/046-betaxanthin-omics-cotarget/scripts/resolving_perturbation_046.py
# [[experiments.046-betaxanthin-omics-cotarget.scripts.resolving_perturbation_046]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/046-betaxanthin-omics-cotarget/scripts/resolving_perturbation_046
"""A ``Perturbation`` graph processor that resolves released gene symbols at READ time.

Since PR #1085 (#1045) a loader stores a deleted gene under the name its source released.
Cachera 2023 names 3,592 of its 4,759 strains by gene symbol (``ADE2``), and those sit on
``ReleasedIdentifierPerturbation`` with ``systematic_gene_name`` equal to the symbol; Messner
2023 and Mulleder 2016 write the same leaf for a non-systematic ORF column entry. The stock
``Perturbation`` processor maps ``systematic_gene_name`` to a gene node by exact match
against ``cell_graph["gene"].node_ids``, so such a record comes out with NO perturbed gene
and no error, which for a training set is a strain that looks like the wild type.

Resolution of a released name to its R64 ORF is the downstream step the loader deliberately
leaves out (``torchcell/datasets/gene_alias_resolution.py``). This processor is that step for
experiment 046, with two rules and a typed refusal, every outcome counted and the refusals
carried on the processed record rather than dropped:

``strict``
    :func:`resolve_gene_name_strict` with no pins: a live systematic name, or a name with
    exactly one candidate ORF among the alias table and the SGD standard-name owners.
``sgd_standard_name``
    an ambiguous name (several candidates) that exactly one live ORF carries as its SGD
    standard name (``gene=`` in the R64 GFF) resolves to that ORF. This is the
    ``AliasResolutionRule.SGD_STANDARD_NAME`` rule of the alias module applied generically
    rather than through a per-name pin with a source quote, because a deletion collection
    names its strains by the standard name of the deleted ORF (``ADH5`` is the standard name
    of YBR145W and an alias of YDL168W, SFA1; the ADH5 strain deletes YBR145W). It is a
    declared modeling-time rule, not a loader decision, and the census says how many
    records it touched.
``refused``
    anything else: no live gene, alias or standard name matches (``AAD6``, ``CRS5``,
    ``FLO8``, ``YLR287-A`` in Cachera), or an ambiguous name with no standard-name owner or
    several. The record is processed with that perturbation UNRESOLVED: its index is absent
    from ``perturbation_indices`` and the name and reason are listed on
    ``processed_graph["gene"].unresolved_perturbations``, so a trainer can exclude the record
    explicitly instead of training on it as a wild type.

A name that is already a gene node passes through untouched (every KanMX leaf and every
placed deletion), and so does a perturbation that is not a deletion kind (Cachera's
heterologous cassette additions CYP76AD1 and DOD name no yeast gene by design, and the
stock processor leaves them unmapped too), so a record with no released identifier
processes exactly as under the stock processor.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from torch_geometric.data import HeteroData

from torchcell.data.graph_processor import Perturbation
from torchcell.datamodels.schema import ExperimentReferenceType, ExperimentType
from torchcell.datasets.gene_alias_resolution import (
    GeneNameRefused,
    GenomeNameIndex,
    candidate_orfs,
    resolve_gene_name_strict,
)

RULE_STRICT = "strict"
RULE_STANDARD_NAME = "sgd_standard_name"
RULE_REFUSED = "refused"


def _is_deletion_kind(perturbation_type: str) -> bool:
    """A placed deletion leaf or the released-identifier leaf.

    The released kind is deletion in every loader of this query; a gene addition or an
    allele is not resolved.
    """
    return (
        perturbation_type.endswith("deletion")
        or perturbation_type == "released_identifier"
    )


class ReleasedNameResolver:
    """Released gene name -> ``(ORF or None, rule, detail)``, memoized per name."""

    def __init__(self, genome: GenomeNameIndex) -> None:
        """Index the genome once; every lookup after that is a dict hit."""
        self.genome = genome
        self.gene_set = set(genome.gene_set)
        self._cache: dict[str, tuple[str | None, str, str]] = {}
        self.outcomes: Counter[str] = Counter()

    def resolve(self, name: str) -> tuple[str | None, str, str]:
        """Resolve one released name; the rule names how, the detail says why."""
        key = name.strip().upper()
        if key in self._cache:
            return self._cache[key]
        if key in self.gene_set:
            out: tuple[str | None, str, str] = (
                key,
                RULE_STRICT,
                "live systematic name",
            )
        else:
            try:
                orf = resolve_gene_name_strict(self.genome, key, pins={})
                out = (orf, RULE_STRICT, "single candidate ORF")
            except GeneNameRefused as refused:
                reason = refused.refusal.reason.value
                candidates = candidate_orfs(self.genome, key)
                owners = sorted(
                    orf
                    for orf in self.genome.feature_index["standard_to_ids"].get(key, [])
                    if orf in self.gene_set
                )
                if len(candidates) > 1 and len(owners) == 1:
                    out = (
                        owners[0],
                        RULE_STANDARD_NAME,
                        f"ambiguous among {candidates}; SGD standard name of {owners[0]}",
                    )
                else:
                    out = (None, RULE_REFUSED, f"{reason}; candidates {candidates}")
        self._cache[key] = out
        self.outcomes[out[1]] += 1
        return out


class ResolvingPerturbation(Perturbation):
    """``Perturbation`` whose perturbed genes are resolved released names."""

    def __init__(self, genome: GenomeNameIndex) -> None:
        """Build the resolver over ``genome``; the stock processor does the rest."""
        super().__init__()
        self.resolver = ReleasedNameResolver(genome)

    def _resolved(
        self, experiment: ExperimentType
    ) -> tuple[ExperimentType, list[dict[str, str]], Counter[str]]:
        """The experiment with every perturbation name resolved, the refusals, the rules."""
        perturbations = getattr(experiment.genotype, "perturbations", None)
        if not perturbations:
            return experiment, [], Counter()
        new_perts: list[Any] = []
        unresolved: list[dict[str, str]] = []
        rules: Counter[str] = Counter()
        changed = False
        for p in perturbations:
            name = p.systematic_gene_name
            if name in self.resolver.gene_set or not _is_deletion_kind(
                p.perturbation_type
            ):
                # A gene node already, or a perturbation that is not a deletion of a host
                # gene (Cachera's heterologous cassette additions CYP76AD1 and DOD name no
                # yeast gene by design); neither is a name to resolve.
                new_perts.append(p)
                continue
            orf, rule, detail = self.resolver.resolve(name)
            rules[rule] += 1
            if orf is None:
                unresolved.append(
                    {
                        "released_name": name,
                        "perturbation_type": p.perturbation_type,
                        "detail": detail,
                    }
                )
                new_perts.append(p)
                continue
            # A frozen pydantic model: model_copy returns a copy with the ORF in place of
            # the symbol, the stored record is untouched.
            new_perts.append(p.model_copy(update={"systematic_gene_name": orf}))
            changed = True
        if not changed:
            return experiment, unresolved, rules
        genotype = experiment.genotype.model_copy(update={"perturbations": new_perts})
        return experiment.model_copy(update={"genotype": genotype}), unresolved, rules

    def process(
        self,
        cell_graph: HeteroData,
        phenotype_info: list[Any],
        data: list[dict[str, ExperimentType | ExperimentReferenceType]],  # type: ignore[override]
    ) -> HeteroData:
        """Resolve, then process as the stock ``Perturbation`` does; report refusals."""
        if not data:
            raise ValueError("Data list is empty")
        resolved_data: list[dict[str, ExperimentType | ExperimentReferenceType]] = []
        unresolved_all: list[dict[str, str]] = []
        rules_all: Counter[str] = Counter()
        for item in data:
            experiment, unresolved, rules = self._resolved(item["experiment"])  # type: ignore[arg-type]
            resolved_data.append({**item, "experiment": experiment})
            unresolved_all.extend(unresolved)
            rules_all.update(rules)
        graph = super().process(cell_graph, phenotype_info, resolved_data)
        graph["gene"].unresolved_perturbations = unresolved_all
        graph["gene"].resolution_rules = dict(rules_all)
        return graph
