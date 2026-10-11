# tests/torchcell/verification/test_common.py
# [[tests.torchcell.verification.test_common]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/verification/test_common.py
"""Tests for the family-agnostic shared record rules (``verification/common.py``).

2026.09.30 (Phase 17). Every fixture is a hand-built stored MAPPING of the shape an LMDB
yields (``{"experiment": {"genotype", "environment", "phenotype"}}``), because the rules
read mappings, never models. A mapping is a gap carrier exactly when it has a
``provenance_gaps`` key, so a plain dict without one is invisible to the census, and a
dict with one whose key set matches no model is counted under the class ``unknown``.
Model dumps (``Compound``, ``FitnessPhenotype``, ``MEDIA_LIBRARY["YPD"]``) are used only
where the rule matches on the model's exact key set or on library equality.

Expected values, each derived by hand from the source:

- Census: a name-only ``Compound`` dump carries five ``None`` identity fields
  (``inchikey``, ``inchi``, ``smiles``, ``pubchem_cid``, ``chebi_id``); a gap on
  ``inchikey`` leaves four silent. Two records carrying it give 2 gaps and 8 silent
  values; one ``unknown`` carrier with a gapped ``fitness_se`` and two silent fields adds
  1 gap and 2 silent values, so 3 gaps over 3/4 records and 10 silent values over 6
  carrier fields. The "top" list sorts by (-count, name) and keeps 5: the four
  ``Compound`` fields at x2, then ``unknown.n_samples`` (alphabetically before
  ``unknown.sample_unit``) at x1.
- Gene names: ``TOR1`` + ``Tor1`` + ``TOR1`` on ``YJR066W`` is one case-only split over 3
  records; an allele-bearing perturbation contributes no spelling; a background gene is
  skipped everywhere.
- Uncertainty: ``sample_sd`` and ``variance`` are the dispersion kinds, so a 0 there fails
  and a ``standard_error`` of 0 does not; examples stop at 20.
- L4: measured {YA, YB, YC} against SGD {YA, YB} is 2/3 = 0.667; YC sits on two records.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from torchcell.datamodels.media import MEDIA_LIBRARY
from torchcell.datamodels.schema import (
    Compound,
    Environment,
    FitnessPhenotype,
    Media,
    Temperature,
    UncertaintyType,
)
from torchcell.verification.common import (
    SharedRecordRules,
    gap_carrier_fields,
    shared_rule_results,
)
from torchcell.verification.report import Level, LevelResult
from torchcell.verification.sourced import ProvenanceGap, ProvenanceGapReason

FREE_MEDIUM: dict[str, Any] = {"name": "lab broth", "base_medium": None}


def _record(
    *,
    perturbations: list[dict[str, Any]] | None = None,
    environment: dict[str, Any] | None = None,
    phenotype: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One stored record; defaults carry no carrier, no gene and a free-text medium."""
    return {
        "experiment": {
            "genotype": {"perturbations": perturbations or []},
            "environment": environment
            if environment is not None
            else {"media": FREE_MEDIUM, "perturbations": []},
            "phenotype": phenotype if phenotype is not None else {},
        }
    }


def _by_name(results: list[LevelResult]) -> dict[str, LevelResult]:
    return {result.name: result for result in results}


def _run(records: list[dict[str, Any]], **kwargs: Any) -> dict[str, LevelResult]:
    rules = SharedRecordRules(**kwargs)
    rules.add_all(records)
    return _by_name(rules.results())


def _deletion(systematic: str, common: str | None) -> dict[str, Any]:
    return {
        "systematic_gene_name": systematic,
        "perturbed_gene_name": common,
        "perturbation_type": "deletion",
    }


@dataclass
class _Resolution:
    """Structural stand-in for the genome's ``resolve_gene_name`` result."""

    status: Any
    systematic_name: str | None


class _Status(Enum):
    current = "current"
    renamed = "renamed"


def _resolver(table: dict[str, tuple[Any, str | None]]) -> Any:
    """A resolver answering from ``name -> (status, systematic_name)``."""

    def resolve(name: str) -> _Resolution:
        status, systematic = table[name]
        return _Resolution(status=status, systematic_name=systematic)

    return resolve


# --------------------------------------------------------------------------- #
# Carrier index
# --------------------------------------------------------------------------- #
def test_gap_carrier_fields_maps_stored_dumps_back_to_their_class() -> None:
    """A stored dump's key set names its class, grandchildren of the mixin included.

    ``Environment`` subclasses the mixin directly; ``FitnessPhenotype`` subclasses
    ``Phenotype`` (the direct mixin subclass), so it is reached only through the
    recursive ``__subclasses__`` walk. A key set with one extra key matches nothing.
    """
    index = gap_carrier_fields()
    compound_keys = frozenset(Compound(name="x").model_dump())
    fitness = FitnessPhenotype(
        graph_level="global",
        label_name="fitness",
        label_statistic_name="fitness_std",
        fitness=1.0,
        fitness_std=0.1,
    )
    environment = Environment(
        media=Media(name="YPD", state="solid", is_synthetic=False),
        temperature=Temperature(value=30.0),
    )
    assert index[compound_keys] == "Compound"
    assert index[frozenset(fitness.model_dump())] == "FitnessPhenotype"
    assert index[frozenset(environment.model_dump())] == "Environment"
    assert index.get(compound_keys | {"extra"}) is None


# --------------------------------------------------------------------------- #
# L1 provenance_gaps
# --------------------------------------------------------------------------- #
def test_provenance_gaps_fully_sourced_message_counts_unknown_carriers() -> None:
    """A carrier with no None and no gap is fully sourced even when its class is unknown."""
    phenotype = {"label_name": "fitness", "provenance_gaps": []}
    rules = SharedRecordRules()
    rules.add_all([_record(phenotype=phenotype), _record(phenotype=phenotype)])
    result = _by_name(rules.results())["provenance_gaps"]
    assert result.level == Level.L1
    assert result.passed is True
    assert result.message == (
        "no provenance gaps and no undeclared None values across 2 records "
        "(2 gap-capable carriers; fully sourced)"
    )
    census = rules.census()
    assert census.carriers_by_class == {"unknown": 2}
    assert census.unrecognized_carriers == 2
    assert census.n_carriers == 2


def test_provenance_gaps_census_counts_gaps_silent_nones_and_worklist() -> None:
    """Declared gaps, silent Nones and the deferred worklist, with the top-5 truncation.

    Records 1 and 2 dose the same gapped compound (enum-valued reason, as a model dump
    stores it); record 3 has an unknown phenotype carrier gapping ``fitness_se`` with two
    silent fields; record 4 has no carrier. The perturbation mapping itself has no
    ``provenance_gaps`` key, so it is not a carrier.
    """
    compound = Compound(
        name="drug",
        provenance_gaps=[
            ProvenanceGap(
                field="inchikey",
                reason=ProvenanceGapReason.deferred_pending_source_review,
            )
        ],
    ).model_dump()
    dosed = {"media": FREE_MEDIUM, "perturbations": [{"compound": compound}]}
    phenotype = {
        "label_name": "fitness",
        "fitness_se": None,
        "n_samples": None,
        "sample_unit": None,
        "provenance_gaps": [
            {"field": "fitness_se", "reason": "not_reported_by_primary"}
        ],
    }
    rules = SharedRecordRules()
    rules.add_all(
        [
            _record(environment=dosed),
            _record(environment=dosed),
            _record(phenotype=phenotype),
            _record(),
        ]
    )
    result = _by_name(rules.results())["provenance_gaps"]
    assert result.passed is True
    assert result.message == (
        "3 documented provenance gaps over 3/4 records; 1 deferred field(s): "
        "['inchikey']; 10 undeclared None values over 6 carrier fields (top: "
        "Compound.chebi_id x2, Compound.inchi x2, Compound.pubchem_cid x2, "
        "Compound.smiles x2, unknown.n_samples x1)"
    )
    census = rules.census()
    assert census.n_records == 4
    assert census.n_records_with_gaps == 3
    assert census.n_gaps == 3
    assert census.by_reason == {
        "deferred_pending_source_review": 2,
        "not_reported_by_primary": 1,
    }
    assert census.by_field == {"inchikey": 2, "fitness_se": 1}
    assert census.worklist_fields == ["inchikey"]
    assert census.silent_none_by_field == {
        "Compound.inchi": 2,
        "Compound.smiles": 2,
        "Compound.pubchem_cid": 2,
        "Compound.chebi_id": 2,
        "unknown.n_samples": 1,
        "unknown.sample_unit": 1,
    }
    assert census.carriers_by_class == {"Compound": 2, "unknown": 1}
    assert result.details == census.model_dump()


def test_provenance_gaps_message_omits_top_clause_when_nothing_is_silent() -> None:
    """Gaps with no silent None: the message ends without a ``(top: ...)`` clause."""
    phenotype = {
        "x": None,
        "provenance_gaps": [{"field": "x", "reason": "not_carried_by_curation"}],
    }
    result = _run([_record(phenotype=phenotype)])["provenance_gaps"]
    assert result.message == (
        "1 documented provenance gaps over 1/1 records; 0 deferred field(s): []; "
        "0 undeclared None values over 0 carrier fields"
    )


# --------------------------------------------------------------------------- #
# L1 canonical_gene_names
# --------------------------------------------------------------------------- #
def test_gene_names_with_no_perturbations_pass_with_nothing_to_check() -> None:
    result = _run([_record()])["canonical_gene_names"]
    assert result.passed is True
    assert result.message == "no gene perturbations to check"
    assert result.details["n_genes"] == 0


def test_gene_names_skip_background_allele_spelling_and_missing_systematic() -> None:
    """Background genes and unnamed perturbations are skipped; an allele adds no spelling.

    ``tor1-1`` (temperature-sensitive allele of YJR066W) would be a second spelling next
    to ``TOR1`` if the allele sentinel counted, and the background gene ``YBG`` would be
    a third gene.
    """
    records = [
        _record(
            perturbations=[
                _deletion("YJR066W", "TOR1"),
                _deletion("YBG", "bg1"),
                {"systematic_gene_name": None, "perturbed_gene_name": "x"},
            ]
        ),
        _record(
            perturbations=[
                {
                    "systematic_gene_name": "YJR066W",
                    "perturbed_gene_name": "tor1-1",
                    "perturbation_type": "temperature_sensitive_allele",
                },
                _deletion("YFL033C", "RIM15"),
            ]
        ),
    ]
    result = _run(records, background_genes=frozenset({"YBG"}))["canonical_gene_names"]
    assert result.passed is True
    assert result.message == (
        "2 systematic names, one canonical spelling each "
        "(no resolver supplied: spelling checked, annotation not)"
    )
    assert result.details["n_genes"] == 2
    assert result.details["resolver"] is False


def test_gene_names_case_only_split_fails_even_when_the_resolver_accepts_both() -> None:
    """``TOR1`` and ``Tor1`` on one systematic name fail as a case-only split."""
    records = [
        _record(perturbations=[_deletion("YJR066W", name)])
        for name in ("TOR1", "Tor1", "TOR1")
    ]
    resolver = _resolver(
        {
            "YJR066W": ("current", "YJR066W"),
            "TOR1": ("current", "YJR066W"),
            "Tor1": ("current", "YJR066W"),
        }
    )
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is False
    assert result.message == (
        "1 genes carry conflicting common-name spellings (3 records; 1 case-only); "
        "0 systematic names are not the genome's current name; 0 common names "
        "resolve to another gene"
    )
    assert result.details["split_spellings"] == {"YJR066W": ["TOR1", "Tor1"]}
    assert result.details["n_records_with_split_spelling"] == 3


def test_gene_names_distinct_spellings_fail_without_a_resolver() -> None:
    """Two different names on one gene cannot be told apart from a split without SGD."""
    records = [
        _record(perturbations=[_deletion("YPR089W", "YPR089W")]),
        _record(perturbations=[_deletion("YPR089W", "YPR090W")]),
    ]
    result = _run(records)["canonical_gene_names"]
    assert result.passed is False
    assert result.message == (
        "1 genes carry conflicting common-name spellings (2 records; 0 case-only); "
        "0 systematic names are not the genome's current name; 0 common names "
        "resolve to another gene"
    )
    assert result.details["split_spellings"] == {"YPR089W": ["YPR089W", "YPR090W"]}


def test_gene_names_merged_orf_aliases_pass_with_a_resolver() -> None:
    """Two names both resolving (current or renamed) to the gene are a merged-ORF alias.

    The statuses are enum members here, which the rule reads through their ``value``.
    """
    records = [
        _record(perturbations=[_deletion("YPR089W", "YPR089W")]),
        _record(perturbations=[_deletion("YPR089W", "YPR090W")]),
    ]
    resolver = _resolver(
        {
            "YPR089W": (_Status.current, "YPR089W"),
            "YPR090W": (_Status.renamed, "YPR089W"),
        }
    )
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is True
    assert result.message == (
        "1 systematic names, one canonical spelling each (1 merged-ORF aliases), "
        "each current in the genome"
    )
    assert result.details["merged_orf_aliases"] == {"YPR089W": ["YPR089W", "YPR090W"]}
    assert result.details["split_spellings"] == {}


def test_gene_names_common_name_resolving_elsewhere_splits_and_mismatches() -> None:
    """A second name that resolves to ANOTHER gene fails the split and the mismatch."""
    records = [
        _record(perturbations=[_deletion("YSYS", "GOOD1")]),
        _record(perturbations=[_deletion("YSYS", "BAD1")]),
    ]
    resolver = _resolver(
        {
            "YSYS": ("current", "YSYS"),
            "GOOD1": ("current", "YSYS"),
            "BAD1": ("current", "YOTHER"),
        }
    )
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is False
    assert result.message == (
        "1 genes carry conflicting common-name spellings (2 records; 0 case-only); "
        "0 systematic names are not the genome's current name; 1 common names "
        "resolve to another gene"
    )
    assert result.details["common_name_mismatch"] == ["BAD1 -> YOTHER (stored YSYS)"]


def test_gene_names_retired_systematic_name_fails_as_not_current() -> None:
    """A stored systematic name the genome renamed is not its own current name."""
    records = [_record(perturbations=[_deletion("YOLD", "FOO1")])]
    resolver = _resolver({"YOLD": ("renamed", "YNEW"), "FOO1": ("current", "YOLD")})
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is False
    assert result.message == (
        "0 genes carry conflicting common-name spellings (0 records; 0 case-only); "
        "1 systematic names are not the genome's current name; 0 common names "
        "resolve to another gene"
    )
    assert result.details["not_current"] == ["YOLD (renamed -> YNEW)"]


def test_gene_names_a_self_resolving_pseudogene_locus_passes_and_is_counted() -> None:
    """A pseudogene locus the genome answers with its OWN tag is a real deletion target.

    A bacterial annotation's ``gene`` features exclude ``/pseudo`` loci by construction,
    so a pseudogene can never come back ``current``; the Keio collection and the sRNA
    library nonetheless deleted 108 BW25113 and 1 MG1655 pseudogene loci, whose stored
    identifier is exactly right. The rule reports them rather than passing over them.
    """
    records = [
        _record(perturbations=[_deletion("BW25113_0021", "insB1")]),
        _record(perturbations=[_deletion("BW25113_0022", "insA1")]),
    ]
    resolver = _resolver(
        {
            "BW25113_0021": ("non_gene_feature", "BW25113_0021"),
            "BW25113_0022": ("non_gene_feature", "BW25113_0022"),
            "insB1": ("current", "BW25113_0021"),
            "insA1": ("current", "BW25113_0022"),
        }
    )
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is True
    assert result.message == (
        "2 systematic names, one canonical spelling each, each current in the genome; "
        "2 are pseudogene loci the genome resolves to themselves"
    )
    assert result.details["not_current"] == []
    assert result.details["n_self_resolving_non_gene_features"] == 2
    assert result.details["self_resolving_non_gene_features"] == [
        "BW25113_0021 (non_gene_feature, None)",
        "BW25113_0022 (non_gene_feature, None)",
    ]


def test_gene_names_a_non_gene_feature_resolving_elsewhere_still_fails() -> None:
    """The acceptance is only for a SELF-resolving locus; a redirect is still a defect.

    This is the case the rule exists for: the stored tag names one locus and the
    annotation says the biology is at another, so the record's identifier is wrong.
    """
    records = [_record(perturbations=[_deletion("BW25113_0021", "insB1")])]
    resolver = _resolver(
        {
            "BW25113_0021": ("non_gene_feature", "BW25113_4496"),
            "insB1": ("current", "BW25113_0021"),
        }
    )
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is False
    assert result.details["not_current"] == [
        "BW25113_0021 (non_gene_feature -> BW25113_4496)"
    ]
    assert result.details["n_self_resolving_non_gene_features"] == 0


def test_gene_names_unplaceable_common_name_is_reported_not_failed() -> None:
    """A name the resolver cannot place passes, with the count appended to the message."""
    records = [_record(perturbations=[_deletion("YSYS", "MYST1")])]
    resolver = _resolver({"YSYS": ("current", "YSYS"), "MYST1": ("not_found", None)})
    result = _run(records, resolve_gene_name=resolver)["canonical_gene_names"]
    assert result.passed is True
    assert result.message == (
        "1 systematic names, one canonical spelling each, each current in the genome; "
        "1 common names the resolver cannot place"
    )
    assert result.details["unresolved_common_names"] == [
        "MYST1 (not_found; stored YSYS)"
    ]


# --------------------------------------------------------------------------- #
# L2 uncertainty_sanity
# --------------------------------------------------------------------------- #
def _fitness(**fields: Any) -> dict[str, Any]:
    return {"label_name": "fitness", "label_statistic_name": "fitness_std", **fields}


def test_uncertainty_passes_and_counts_replicated_records_without_uncertainty() -> None:
    """A standard-error 0 is not a dispersion; only n_samples >= 2 with no SE is counted.

    Checked: the ``standard_error`` record (1). Unreported: ``n_samples=3`` with neither
    SE nor uncertainty (1). Not counted: ``n_samples=1``, ``n_samples=2`` with an SE, and
    a phenotype with no string ``label_name`` (skipped entirely).
    """
    records = [
        _record(
            phenotype=_fitness(
                fitness_std=0.0,
                fitness_uncertainty=0.0,
                fitness_uncertainty_type="standard_error",
            )
        ),
        _record(phenotype=_fitness(n_samples=3)),
        _record(phenotype=_fitness(n_samples=1)),
        _record(phenotype=_fitness(n_samples=2, fitness_std=0.2)),
        _record(phenotype={"n_samples": 5}),
    ]
    result = _run(records)["uncertainty_sanity"]
    assert result.level == Level.L2
    assert result.passed is True
    assert result.message == (
        "1 labeled uncertainties, none a zero dispersion; 1 records report "
        "n_samples >= 2 with no uncertainty"
    )
    assert result.details == {
        "n_checked": 1,
        "n_zero_dispersion": 0,
        "n_no_uncertainty_with_replicates": 1,
        "examples": [],
    }


def test_uncertainty_fails_on_a_zero_sample_dispersion_with_examples() -> None:
    """``sample_sd`` uncertainty 0 and ``variance`` SE 0 fail; the SE field defaults.

    The variance record has no ``label_statistic_name``, so its SE field is
    ``fitness_se``; its kind is the enum member, read through its value.
    """
    records = [
        _record(
            phenotype=_fitness(
                fitness_std=0.1,
                fitness_uncertainty=0.0,
                fitness_uncertainty_type="sample_sd",
                n_samples=4,
            )
        ),
        _record(
            phenotype={
                "label_name": "fitness",
                "fitness_se": 0.0,
                "fitness_uncertainty_type": UncertaintyType.variance,
            }
        ),
        _record(
            phenotype=_fitness(
                fitness_uncertainty=0.0, fitness_uncertainty_type="bootstrap_se"
            )
        ),
    ]
    result = _run(records)["uncertainty_sanity"]
    assert result.passed is False
    assert result.message == (
        "2/3 labeled uncertainties are a sample dispersion of exactly 0 "
        "(a pseudocount artifact, not perfect precision)"
    )
    assert result.details["examples"] == [
        {
            "uncertainty_type": "sample_sd",
            "uncertainty": 0.0,
            "fitness_std": 0.1,
            "n_samples": 4,
        },
        {
            "uncertainty_type": "variance",
            "uncertainty": None,
            "fitness_se": 0.0,
            "n_samples": None,
        },
    ]


def test_uncertainty_examples_stop_at_twenty() -> None:
    """21 zero dispersions are all counted; only the first 20 are kept as examples."""
    records = [
        _record(
            phenotype=_fitness(
                fitness_uncertainty=0.0,
                fitness_uncertainty_type="sample_sd",
                n_samples=i,
            )
        )
        for i in range(21)
    ]
    result = _run(records)["uncertainty_sanity"]
    assert result.details["n_zero_dispersion"] == 21
    assert [e["n_samples"] for e in result.details["examples"]] == list(range(20))


# --------------------------------------------------------------------------- #
# L3 compound_identity / media_compound_identity
# --------------------------------------------------------------------------- #
def _identified(name: str) -> dict[str, Any]:
    return {"name": name, "chebi_id": "CHEBI:1", "inchikey": None}


def _gapped(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "inchikey": None,
        "provenance_gaps": [{"field": "inchikey", "reason": "not_reported_by_primary"}],
    }


def test_compound_identity_splits_edits_from_medium_and_fails_name_only() -> None:
    """Every compound context is verdicted and reported under the layer that owns it.

    Medium: a defined identified component and a defined name-only one count; a defined
    component with no compound mapping and a ``composition_deferred`` name-only one are
    skipped. Edits: a dosed compound (identified), a physical agent gapped on a field
    that is not an identity field (``roles``, so still name-only), a name-only dose with
    an identified solvent, and a solvent given as a bare string (skipped). The same
    environment object is added twice, so every count doubles.
    """
    environment = {
        "media": {
            "name": "custom",
            "base_medium": "YPD",
            "components": [
                {"definition": "defined", "compound": _identified("D-glucose")},
                {"definition": "defined", "compound": {"name": "mystery salt"}},
                {"definition": "defined", "compound": None},
                {"definition": "composition_deferred", "compound": {"name": "YNB"}},
            ],
        },
        "perturbations": [
            {"compound": _identified("rapamycin")},
            {
                "agent": {
                    "name": "psoralen",
                    "provenance_gaps": [
                        {"field": "roles", "reason": "not_reported_by_primary"}
                    ],
                }
            },
            {
                "compound": {"name": "drug X"},
                "solvent": {"compound": _identified("DMSO")},
            },
            {"solvent": {"compound": "DMSO"}},
        ],
    }
    results = _run([_record(environment=environment), _record(environment=environment)])
    edits = results["compound_identity"]
    assert edits.level == Level.L3
    assert edits.passed is False
    assert edits.message == (
        "environment edits: 2 compounds are name-only (no identifier, no gap) over 4 "
        "references: psoralen (perturbation.agent) x2, drug X (perturbation.compound) x2"
    )
    assert (edits.details["n_identified"], edits.details["n_gapped"]) == (4, 0)
    medium = results["media_compound_identity"]
    assert medium.passed is False
    assert medium.message == (
        "medium components: 1 compounds are name-only (no identifier, no gap) over 2 "
        "references: mystery salt (media.component[custom]) x2"
    )
    assert medium.details["n_identified"] == 2
    assert medium.details["scope"] == "medium components"


def test_compound_identity_passes_on_identified_and_gapped_compounds() -> None:
    """A typed identity gap passes; the distinct-compound count is by name and context."""
    environment = {
        "media": FREE_MEDIUM,
        "perturbations": [
            {"compound": _identified("rapamycin")},
            {"compound": _gapped("CMB123")},
        ],
    }
    other = {"media": FREE_MEDIUM, "perturbations": [{"compound": _gapped("CMB123")}]}
    results = _run([_record(environment=environment), _record(environment=other)])
    edits = results["compound_identity"]
    assert edits.passed is True
    assert edits.message == (
        "environment edits: 1 compound references carry a structure identifier; 2 "
        "declare a typed gap (1 distinct compounds, unencodable)"
    )
    assert edits.details["gapped_records"] == {"CMB123 (perturbation.compound)": 2}
    medium = results["media_compound_identity"]
    assert medium.passed is True
    assert medium.message == (
        "medium components: 0 compound references carry a structure identifier; 0 "
        "declare a typed gap (0 distinct compounds, unencodable)"
    )


# --------------------------------------------------------------------------- #
# L3 media_membership
# --------------------------------------------------------------------------- #
def test_media_membership_accepts_library_and_derived_media() -> None:
    """Equality with a library dump is ``library:<key>``; a library base is ``derived``."""
    ypd = MEDIA_LIBRARY["YPD"].model_dump()
    derived = {"name": "YPD + 1 M sorbitol", "base_medium": "YPD"}
    records = [
        _record(environment={"media": ypd}),
        _record(environment={"media": MEDIA_LIBRARY["YPD"].model_dump()}),
        _record(environment={"media": derived}),
    ]
    result = _run(records)["media_membership"]
    assert result.passed is True
    assert result.message == (
        "2 records on a shared MEDIA_LIBRARY medium, 1 on a medium deriving from one "
        "(2 distinct media)"
    )
    assert result.details["matched_media"] == {
        "YPD (yeast extract / peptone / dextrose)": "library:YPD",
        "YPD + 1 M sorbitol": "derived:YPD",
    }


def test_media_membership_fails_free_text_media_including_a_missing_medium() -> None:
    """A non-library base, no base, and no medium at all each join nothing."""
    records = [
        _record(environment={"media": FREE_MEDIUM}),
        _record(environment={"media": FREE_MEDIUM}),
        _record(environment={"media": {"name": "mystery", "base_medium": "NOT_A_KEY"}}),
        _record(environment={}),
    ]
    result = _run(records)["media_membership"]
    assert result.passed is False
    assert result.message == (
        "3 free-text media over 4 records join nothing: lab broth (base_medium=None) "
        "x2, mystery (base_medium=NOT_A_KEY) x1, None (base_medium=None) x1"
    )


# --------------------------------------------------------------------------- #
# L4 gene containment + result order
# --------------------------------------------------------------------------- #
def _gene_records() -> list[dict[str, Any]]:
    """YA x3, YB x1, YC x2 (on two records), background YBG never counted."""
    return [
        _record(perturbations=[_deletion("YA", None)]),
        _record(perturbations=[_deletion("YA", None), _deletion("YC", None)]),
        _record(perturbations=[_deletion("YB", None)]),
        _record(perturbations=[_deletion("YC", None)]),
        _record(perturbations=[_deletion("YA", None), _deletion("YBG", None)]),
    ]


def test_gene_containment_and_off_genome_records() -> None:
    """2 of 3 measured genes are in SGD (0.667); YC is off-genome on 2 records.

    The record carrying only YA plus the background gene YBG (absent from SGD) is not
    off-genome, because background genes are excluded before the membership test.
    """
    results = _run(
        _gene_records(),
        background_genes=frozenset({"YBG"}),
        sgd_genes={"YA", "YB"},
        min_containment=0.6,
    )
    containment = results["gene_containment_sgd"]
    assert containment.level == Level.L4
    assert containment.passed is True
    # No ``gene_universe_label``, so the row names no reference: this rule serves every
    # host and the caller is the only one that knows which universe it handed over.
    assert containment.message == (
        "0.667 of 3 measured genes are reference genes (>= 0.6)"
    )
    assert containment.details["missing_examples"] == ["YC"]
    off = results["current_genome_genes"]
    assert off.passed is False
    assert off.message == (
        "1 systematic names are absent from the current genome over 2 records: YC x2"
    )
    assert off.details["missing_records"] == {"YC": 2}


def test_a_named_gene_universe_is_what_the_containment_row_claims() -> None:
    """The label is the row's claim, so a bacterial universe never reads as S288C's."""
    results = _run(
        _gene_records(),
        background_genes=frozenset({"YBG"}),
        sgd_genes={"YA", "YB", "YC"},
        gene_universe_label="pputida_KT2440_ASM756v2 locus",
    )
    assert results["gene_containment_sgd"].message == (
        "1.000 of 3 measured genes are pputida_KT2440_ASM756v2 locus genes (>= 0.9)"
    )


def test_gene_containment_passes_when_every_gene_is_on_the_genome() -> None:
    results = _run(
        _gene_records(),
        background_genes=frozenset({"YBG"}),
        sgd_genes={"YA", "YB", "YC"},
    )
    assert results["gene_containment_sgd"].message == (
        "1.000 of 3 measured genes are reference genes (>= 0.9)"
    )
    off = results["current_genome_genes"]
    assert off.passed is True
    assert off.message == (
        "every one of the 3 measured systematic names is a gene of the current genome"
    )


def test_gene_containment_with_no_genes_fails_on_measured_genes_present() -> None:
    """With ``sgd_genes`` given and no gene perturbations, both containment results
    pass vacuously and a failing ``measured_genes_present`` result states why.

    Contract (issue #541 and its review): the empty measured set is vacuously
    contained and on the genome (each message names the empty set, ``overlap`` 1.0 so
    the floor holds at any ``min_containment`` up to 1), and the dataset fails because
    it measures no gene, not because of a fabricated 0.000 overlap.
    """
    rules = SharedRecordRules(sgd_genes={"YA"}, min_containment=1.0)
    rules.add_all([_record(), _record()])
    results = rules.results()
    assert [r.name for r in results][6:] == [
        "measured_genes_present",
        "gene_containment_sgd",
        "current_genome_genes",
    ]
    by_name = _by_name(results)
    absent = by_name["measured_genes_present"]
    assert (absent.level, absent.passed) == (Level.L4, False)
    assert absent.message == (
        "no measured genes: none of the 2 records carries a gene perturbation outside "
        "the background genes, so the SGD gene rules have nothing to check"
    )
    assert absent.details == {"n_records": 2, "n_measured": 0}
    containment = by_name["gene_containment_sgd"]
    assert containment.passed is True
    assert containment.message == (
        "no measured genes (the measured gene set is empty); containment holds "
        "vacuously"
    )
    assert containment.details == {
        "n_measured": 0,
        "n_in_sgd": 0,
        "overlap": 1.0,
        "missing_examples": [],
    }
    off = by_name["current_genome_genes"]
    assert off.passed is True
    assert off.message == (
        "no measured genes (the measured gene set is empty); genome membership holds "
        "vacuously"
    )


def test_background_only_genes_count_as_no_measured_genes() -> None:
    """A record whose only perturbation is a background gene measures nothing.

    The fixture's free-text medium also fails ``media_membership``; the L4 group is
    the failing ``measured_genes_present`` plus the two vacuous passes.
    """
    rules = SharedRecordRules(background_genes=frozenset({"YBG"}), sgd_genes={"YBG"})
    rules.add(_record(perturbations=[_deletion("YBG", "BG1")]))
    assert [(r.name, r.passed) for r in rules.results()][6:] == [
        ("measured_genes_present", False),
        ("gene_containment_sgd", True),
        ("current_genome_genes", True),
    ]


def test_results_order_and_levels_with_and_without_sgd_genes() -> None:
    """The six shared results in level order; the two L4 results only with SGD genes."""
    without = SharedRecordRules()
    without.add(_record())
    assert [(r.name, r.level) for r in without.results()] == [
        ("provenance_gaps", Level.L1),
        ("canonical_gene_names", Level.L1),
        ("uncertainty_sanity", Level.L2),
        ("compound_identity", Level.L3),
        ("media_compound_identity", Level.L3),
        ("media_membership", Level.L3),
    ]
    with_sgd = SharedRecordRules(sgd_genes={"YA"})
    with_sgd.add(_record(perturbations=[_deletion("YA", None)]))
    assert [r.name for r in with_sgd.results()][6:] == [
        "gene_containment_sgd",
        "current_genome_genes",
    ]


def test_shared_rule_results_forwards_every_option() -> None:
    """The one-call form equals the accumulator and forwards ``min_containment``.

    At 0.667 containment the default floor 0.9 fails and 0.6 passes, so a dropped
    keyword would flip the verdict.
    """
    kwargs: dict[str, Any] = {
        "background_genes": frozenset({"YBG"}),
        "sgd_genes": {"YA", "YB"},
        "min_containment": 0.6,
    }
    one_call = shared_rule_results(_gene_records(), **kwargs)
    rules = SharedRecordRules(**kwargs)
    rules.add_all(_gene_records())
    assert [r.model_dump() for r in one_call] == [
        r.model_dump() for r in rules.results()
    ]
    assert _by_name(one_call)["gene_containment_sgd"].passed is True
    default = _by_name(shared_rule_results(_gene_records(), sgd_genes={"YA", "YB"}))
    assert default["gene_containment_sgd"].passed is False


# --------------------------------------------------------------------------- #
# #889: the helpers the streaming family verifiers share
# --------------------------------------------------------------------------- #
def test_declared_member_validator_checks_the_tag_then_the_model() -> None:
    from torchcell.verification.common import declared_member_validator

    validate = declared_member_validator("GeneEssentialityExperiment")
    stored = {
        "experiment_type": "gene essentiality",
        "dataset_name": "d",
        "genotype": {"perturbations": []},
        "environment": {
            "media": {"name": "m", "state": "solid", "is_synthetic": False}
        },
        "phenotype": {"is_essential": True},
    }
    assert type(validate(stored)).__name__ == "GeneEssentialityExperiment"
    import pytest

    with pytest.raises(ValueError, match="'fitness' is not GeneEssentialityExperiment"):
        validate({**stored, "experiment_type": "fitness"})
    with pytest.raises(ValueError):
        validate({**stored, "phenotype": {"is_essential": "maybe"}})
    with pytest.raises(
        ValueError, match="not a member of schema.ExperimentReferenceType"
    ):
        declared_member_validator("FitnessExperiment", union="ExperimentReferenceType")
    reference = declared_member_validator(
        "FitnessExperimentReference", union="ExperimentReferenceType"
    )
    with pytest.raises(ValueError, match="experiment_reference_type None"):
        reference({})


def test_key_digest_is_sixteen_deterministic_bytes() -> None:
    from torchcell.verification.common import key_digest

    key = (("YAL001C", "sga_kanmx_deletion", None), (26.0, "SC"))
    assert key_digest(key) == key_digest(tuple(key))
    assert len(key_digest(key)) == 16
    assert key_digest(key) != key_digest((("YAL001C",), (30.0, "SC")))


def test_l0_validated_row_fails_an_empty_store_and_names_the_class() -> None:
    from torchcell.verification.common import l0_validated_row

    empty = l0_validated_row("structural", 0, [], "X")
    assert empty.passed is False
    assert empty.level is Level.L0
    failed = l0_validated_row("structural", 2, [{"index": 1, "error": "e"}], "X")
    assert failed.message == "1/2 records failed X validation"
    ok = l0_validated_row("structural", 2, [], "X")
    assert (ok.passed, ok.message, ok.details["validated_as"]) == (
        True,
        "2 records validated as X",
        "X",
    )
