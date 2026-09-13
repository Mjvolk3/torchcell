# torchcell/verification/common
# [[torchcell.verification.common]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/common
"""Record-level rules shared by every dataset-family verifier.

The per-family verifiers (``environment_response``, ``fitness``, ``segregant_growth``,
...) each own the checks that only make sense for their readout. The rules HERE are the
ones that hold for any stored experiment record, and they exist because a review of the
chemogenomic block found each of them missing in a way that let an unusable dataset
report PASS:

- L1 ``provenance_gaps`` -- the census now walks EVERY gap-capable carrier in a record
  (phenotype, environment, each perturbation's compound/agent/solvent, each media
  component's compound), and reports, per carrier field, the ``None`` values that carry
  NO gap. Reading only the phenotype and the environment is what made "fully sourced"
  a false pass for a dataset whose compounds were name-only.
- L1 ``canonical_gene_names`` -- a gene is one persistent entity, so one systematic name
  carries one spelling of its common name; ``TOR1`` and ``Tor1`` in one release split the
  perturbation identity in the graph. With a genome resolver the stored systematic name
  must also BE the resolver's current name for itself, and the common name must resolve
  back to it.
- L2 ``uncertainty_sanity`` -- a sample-SD uncertainty of exactly 0 is a pseudocount
  artifact (identical replicates after a +1), not a measurement of perfect precision.
- L3 ``compound_identity`` / ``media_compound_identity`` -- a compound with no structure
  identifier and no typed gap cannot be encoded or joined, so it fails the level by name
  with its record count. This is what makes "drop the unencodable records" enforceable
  rather than aspirational. The dataset's environment EDITS and the base medium's
  components are reported separately because different layers fix them (the shared
  library's own name-only components are the subject of
  ``test_shared_media_compounds_are_identified_or_gapped``).
- L3 ``media_membership`` -- a medium must BE a shared ``MEDIA_LIBRARY`` object or derive
  from one by ``base_medium``; a free-text medium joins nothing.
- L4 ``gene_containment_sgd`` + ``current_genome_genes`` -- the aggregate containment
  floor stays, and a record keyed to a systematic name the current genome does not carry
  fails outright (a 0.993 containment hid seven records on retired ORFs).

Everything reads the stored MAPPINGS an LMDB yields, never the loader, so one
implementation serves the eager and the streaming verifiers: each accumulator takes one
record at a time (``add``) and renders its ``LevelResult`` at the end (``results``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, Field

from torchcell.verification.report import Level, LevelResult
from torchcell.verification.sourced import ProvenanceGapCensus, ProvenanceGapReason

# ``schema.py`` imports this package's ``sourced`` module, so importing the datamodels at
# module level here would close an import cycle. Everything schema-derived is resolved
# once in ``SharedRecordRules.__init__`` (or inside the helper that needs it), which runs
# long after both modules are loaded.

__all__ = [
    "CarrierGapCensus",
    "GeneNameResolution",
    "GeneNameResolver",
    "SharedRecordRules",
    "gap_carrier_fields",
    "media_library_dumps",
    "shared_rule_results",
]

Record = Mapping[str, Any]

# The field name holding a model's typed absences; its presence in a stored mapping is
# what identifies that mapping as a ProvenanceGapMixin carrier.
GAPS_FIELD = "provenance_gaps"


def _dispersion_types() -> frozenset[str]:
    """Uncertainty kinds that are a DISPERSION of observations.

    Exactly 0 of one of these means every replicate came out identical, which for a
    count-based readout is a pseudocount artifact rather than perfect precision. A
    ``standard_error`` or ``bootstrap_se`` of 0 is a different claim and is left alone.
    """
    from torchcell.datamodels.schema import UncertaintyType

    return frozenset({UncertaintyType.sample_sd.value, UncertaintyType.variance.value})


class GeneNameResolution(Protocol):
    """Structural view of ``SCerevisiaeGenome.resolve_gene_name``'s return value."""

    status: Any
    systematic_name: str | None


# A resolver maps a source gene name to the current annotation. The genome's bound
# ``resolve_gene_name`` satisfies this; passing None skips the resolver-backed parts of
# the canonical-name rule (the one-spelling-per-gene part still runs).
GeneNameResolver = Callable[[str], GeneNameResolution]


def _enum_value(value: Any) -> Any:
    """The stored value of an enum member, or the value itself when it is plain."""
    return getattr(value, "value", value)


def gap_carrier_fields() -> dict[frozenset[str], str]:
    """``frozenset(model fields) -> class name`` for every ``ProvenanceGapMixin``.

    Which fields are gap-capable is read from the MODELS, never from a hand list: a
    ``ProvenanceGap`` must name a field of its carrier and that field must be None, so a
    carrier's gap-capable fields are exactly its model fields minus ``provenance_gaps``.
    A stored mapping is matched back to its class by its key set, which is also how an
    unrecognized carrier (a class this build does not know) stays visible in the report.
    """
    from torchcell.datamodels.schema import ProvenanceGapMixin

    index: dict[frozenset[str], str] = {}
    stack: list[type[BaseModel]] = list(ProvenanceGapMixin.__subclasses__())
    while stack:
        model = stack.pop()
        stack.extend(model.__subclasses__())
        index[frozenset(model.model_fields)] = model.__name__
    return index


class CarrierGapCensus(ProvenanceGapCensus):
    """The gap census, extended to every carrier and to undeclared ``None`` values.

    ``ProvenanceGapCensus`` counts the gaps a build DECLARES. A field that is simply
    ``None`` with no gap declares nothing, and is indistinguishable in a report from a
    field that was never applicable, so it is counted here separately, keyed by
    ``Class.field``, alongside how many carrier objects of each class were seen.
    """

    n_carriers: int = 0
    n_silent_none: int = 0
    silent_none_by_field: dict[str, int] = Field(default_factory=dict)
    carriers_by_class: dict[str, int] = Field(default_factory=dict)
    unrecognized_carriers: int = 0


class _IdentityMemo:
    """Cache of a derived verdict, keyed by the identity of the mapping it came from.

    A built LMDB interns its constant sub-objects, so the SAME environment mapping is
    yielded for every record of a condition; deriving its verdict once turns a per-record
    cost into a per-condition one. The key object is held alongside its value, which is
    what makes ``id()`` a safe key (a live object's id cannot be reused).
    """

    def __init__(self) -> None:
        self._store: dict[int, tuple[Any, Any]] = {}

    def get(self, obj: Any, compute: Callable[[Any], Any]) -> Any:
        entry = self._store.get(id(obj))
        if entry is not None:
            return entry[1]
        value = compute(obj)
        self._store[id(obj)] = (obj, value)
        return value


def media_library_dumps() -> list[tuple[str, dict[str, Any]]]:
    """``(library key, model_dump)`` for every shared medium, for equality matching."""
    from torchcell.datamodels.media import MEDIA_LIBRARY

    return [(key, media.model_dump()) for key, media in sorted(MEDIA_LIBRARY.items())]


def _compounds_in(
    environment: Mapping[str, Any], defined: str
) -> list[tuple[str, Mapping[str, Any]]]:
    """``(context, compound mapping)`` for every compound an environment names.

    The three places a compound is an identity key: a small molecule dosed as an edit, the
    agent that realizes a physical factor, and a DEFINED component of the base medium. A
    ``composition_deferred`` or ``intrinsically_undefined`` component is a mixture (YNB,
    peptone), so no structure identifier exists to demand of it. The vehicle a compound was
    delivered in is included when it carries a typed identity.
    """
    out: list[tuple[str, Mapping[str, Any]]] = []
    media = environment.get("media") or {}
    media_context = f"media.component[{media.get('name')}]"
    for component in media.get("components") or []:
        if str(_enum_value(component.get("definition"))) != defined:
            continue
        compound = component.get("compound")
        if isinstance(compound, Mapping):
            out.append((media_context, compound))
    for perturbation in environment.get("perturbations") or []:
        for field, context in (
            ("compound", "perturbation.compound"),
            ("agent", "perturbation.agent"),
        ):
            compound = perturbation.get(field)
            if isinstance(compound, Mapping):
                out.append((context, compound))
        solvent = perturbation.get("solvent")
        if isinstance(solvent, Mapping) and isinstance(
            solvent.get("compound"), Mapping
        ):
            out.append(("perturbation.solvent.compound", solvent["compound"]))
    return out


def _compound_verdict(
    compound: Mapping[str, Any], identity_fields: tuple[str, ...]
) -> str:
    """``identified`` | ``gapped`` | ``name_only`` for one stored compound.

    Mirrors ``ontology_checks.compound_has_identity`` /
    ``compound_has_identity_gap`` on the stored MAPPING (those read the model), so the
    definition of "identified" is the ontology's own ``IDENTITY_FIELDS``.
    """
    if any(compound.get(field) is not None for field in identity_fields):
        return "identified"
    gaps = compound.get(GAPS_FIELD) or []
    if any(str(gap.get("field")) in identity_fields for gap in gaps):
        return "gapped"
    return "name_only"


class SharedRecordRules:
    """Accumulate the family-agnostic rules over a dataset's records.

    One instance consumes the records of ONE dataset (``add`` per record, ``results`` at
    the end), so the eager and streaming verifiers run identical rules with identical
    messages. ``sgd_genes`` is optional: when it is given the L4 gene rules are emitted
    here; when it is not, the caller owns L4 (the segregant verifier derives its gene set
    from the genome, not from perturbations).
    """

    def __init__(
        self,
        *,
        background_genes: frozenset[str] = frozenset(),
        resolve_gene_name: GeneNameResolver | None = None,
        sgd_genes: set[str] | None = None,
        min_containment: float = 0.90,
    ) -> None:
        """Start an empty accumulation for one dataset."""
        from torchcell.datamodels.media import MEDIA_LIBRARY
        from torchcell.datamodels.ontology_checks import IDENTITY_FIELDS
        from torchcell.datamodels.schema import ComponentDefinition

        self.background_genes = background_genes
        self.resolve_gene_name = resolve_gene_name
        self.sgd_genes = sgd_genes
        self.min_containment = min_containment

        self._census = CarrierGapCensus(n_records=0, n_records_with_gaps=0, n_gaps=0)
        self._carrier_index = gap_carrier_fields()
        self._gap_by_reason: Counter[str] = Counter()
        self._gap_by_field: Counter[str] = Counter()
        self._silent_by_field: Counter[str] = Counter()
        self._carriers_by_class: Counter[str] = Counter()
        self._worklist: set[str] = set()

        self._name_records: Counter[tuple[str, str | None]] = Counter()

        self._n_uncertainty_checked = 0
        self._n_zero_dispersion = 0
        self._zero_dispersion_examples: list[dict[str, Any]] = []
        self._n_unreported_uncertainty = 0

        self._compound_records: Counter[tuple[str, str, str]] = Counter()
        self._media_records: Counter[tuple[str, str, str]] = Counter()

        self._gene_records: Counter[str] = Counter()
        self._n_records_off_genome = 0

        self._env_memo = _IdentityMemo()
        self._media_memo = _IdentityMemo()
        self._media_dumps = media_library_dumps()
        self._media_keys = frozenset(MEDIA_LIBRARY)
        self._identity_fields = IDENTITY_FIELDS
        self._defined = ComponentDefinition.defined.value
        self._dispersion_types = _dispersion_types()

    # ---- ingestion ------------------------------------------------------- #
    def add(self, record: Record) -> None:
        """Fold one stored ``{"experiment": ..., "reference": ...}`` record in."""
        experiment = record["experiment"]
        self._add_gaps(experiment)
        self._add_gene_names(experiment)
        self._add_uncertainty(experiment["phenotype"])
        self._add_environment(experiment["environment"])

    def add_all(self, records: Iterable[Record]) -> None:
        """Fold a sequence of records in (the eager verifiers' entrypoint)."""
        for record in records:
            self.add(record)

    def _add_gaps(self, experiment: Mapping[str, Any]) -> None:
        """Walk the record and census every gap carrier it contains.

        The walk is structural: any mapping carrying ``provenance_gaps`` IS a carrier, so
        a carrier added to the ontology later is reached without editing this code. The
        carrier's class is recovered from its key set (the models' own field sets), which
        is what lets the census report ``Compound.inchikey`` rather than a path.
        """
        n_gaps_before = self._census.n_gaps
        for carrier in _walk_carriers(experiment):
            keys = frozenset(carrier)
            class_name = self._carrier_index.get(keys)
            if class_name is None:
                self._census.unrecognized_carriers += 1
                class_name = "unknown"
            self._carriers_by_class[class_name] += 1
            self._census.n_carriers += 1
            gapped: set[str] = set()
            for gap in carrier.get(GAPS_FIELD) or []:
                field = str(gap.get("field"))
                reason = str(_enum_value(gap.get("reason")))
                gapped.add(field)
                self._census.n_gaps += 1
                self._gap_by_reason[reason] += 1
                self._gap_by_field[field] += 1
                if reason == ProvenanceGapReason.deferred_pending_source_review.value:
                    self._worklist.add(field)
            for field, value in carrier.items():
                if field == GAPS_FIELD or value is not None or field in gapped:
                    continue
                self._census.n_silent_none += 1
                self._silent_by_field[f"{class_name}.{field}"] += 1
        self._census.n_records += 1
        if self._census.n_gaps > n_gaps_before:
            self._census.n_records_with_gaps += 1

    def _add_gene_names(self, experiment: Mapping[str, Any]) -> None:
        genotype = experiment.get("genotype") or {}
        for perturbation in genotype.get("perturbations") or []:
            systematic = perturbation.get("systematic_gene_name")
            if systematic is None or systematic in self.background_genes:
                continue
            self._name_records[
                (systematic, perturbation.get("perturbed_gene_name"))
            ] += 1
            self._gene_records[systematic] += 1
        if self.sgd_genes is not None:
            genes = {
                p.get("systematic_gene_name")
                for p in genotype.get("perturbations") or []
                if p.get("systematic_gene_name") is not None
                and p.get("systematic_gene_name") not in self.background_genes
            }
            if any(gene not in self.sgd_genes for gene in genes):
                self._n_records_off_genome += 1

    def _add_uncertainty(self, phenotype: Mapping[str, Any]) -> None:
        """Check the reported dispersion of ONE phenotype against its replicate design.

        The field names follow the phenotype envelope (``label_name`` names the value,
        ``label_statistic_name`` the SE), so this reads a fitness phenotype and an
        environment-response phenotype with the same code.
        """
        label = phenotype.get("label_name")
        if not isinstance(label, str):
            return
        se_field = phenotype.get("label_statistic_name") or f"{label}_se"
        se = phenotype.get(se_field)
        uncertainty = phenotype.get(f"{label}_uncertainty")
        kind = _enum_value(phenotype.get(f"{label}_uncertainty_type"))
        n_samples = phenotype.get("n_samples")
        if kind is not None:
            self._n_uncertainty_checked += 1
            if str(kind) in self._dispersion_types and (
                se == 0.0 or uncertainty == 0.0
            ):
                self._n_zero_dispersion += 1
                if len(self._zero_dispersion_examples) < 20:
                    self._zero_dispersion_examples.append(
                        {
                            "uncertainty_type": str(kind),
                            "uncertainty": uncertainty,
                            se_field: se,
                            "n_samples": n_samples,
                        }
                    )
        elif (
            isinstance(n_samples, int)
            and n_samples >= 2
            and se is None
            and uncertainty is None
        ):
            self._n_unreported_uncertainty += 1

    def _add_environment(self, environment: Mapping[str, Any]) -> None:
        for context, name, verdict in self._env_memo.get(
            environment, self._environment_compounds
        ):
            self._compound_records[(context, name, verdict)] += 1
        media = environment.get("media") or {}
        name, verdict = self._media_memo.get(media, self._media_verdict)
        self._media_records[(name, verdict, str(media.get("base_medium")))] += 1

    def _environment_compounds(
        self, environment: Mapping[str, Any]
    ) -> list[tuple[str, str, str]]:
        return [
            (
                context,
                str(compound.get("name")),
                _compound_verdict(compound, self._identity_fields),
            )
            for context, compound in _compounds_in(environment, self._defined)
        ]

    def _media_verdict(self, media: Mapping[str, Any]) -> tuple[str, str]:
        """``(name, verdict)``: is this medium a shared object, or derived from one?

        Model equality against a ``MEDIA_LIBRARY`` member is the strong form (the same
        components, concentrations and provenance); ``base_medium`` naming a library key is
        the derived form (a dataset-specific variant that still joins at its base). Anything
        else is free text and joins nothing.
        """
        name = str(media.get("name"))
        for key, dump in self._media_dumps:
            if media == dump:
                return name, f"library:{key}"
        base = media.get("base_medium")
        if base in self._media_keys:
            return name, f"derived:{base}"
        return name, "free_text"

    # ---- rendering ------------------------------------------------------- #
    def results(self) -> list[LevelResult]:
        """Every shared level result, in level order."""
        out = [
            self._gap_result(),
            self._gene_name_result(),
            self._uncertainty_result(),
            *self._compound_identity_results(),
            self._media_result(),
        ]
        if self.sgd_genes is not None:
            out.extend(self._gene_containment_results(self.sgd_genes))
        return out

    def census(self) -> CarrierGapCensus:
        """The finished gap + silent-None census."""
        census = self._census.model_copy(deep=True)
        census.by_reason = dict(self._gap_by_reason)
        census.by_field = dict(self._gap_by_field)
        census.worklist_fields = sorted(self._worklist)
        census.silent_none_by_field = dict(self._silent_by_field)
        census.carriers_by_class = dict(self._carriers_by_class)
        return census

    def _gap_result(self) -> LevelResult:
        """L1: the gap + silent-None census over every carrier. Informational.

        A DOCUMENTED gap is honest, so it never fails a build; an undeclared ``None`` is
        not a defect of the record either (a field can be genuinely inapplicable). What the
        census buys is that "fully sourced" is now a statement about every carrier in the
        record rather than about two of them.
        """
        census = self.census()
        top_silent = ", ".join(
            f"{field} x{count}"
            for field, count in sorted(
                self._silent_by_field.items(), key=lambda kv: (-kv[1], kv[0])
            )[:5]
        )
        if census.n_gaps == 0 and census.n_silent_none == 0:
            message = (
                f"no provenance gaps and no undeclared None values across "
                f"{census.n_records} records ({census.n_carriers} gap-capable carriers; "
                "fully sourced)"
            )
        else:
            message = (
                f"{census.n_gaps} documented provenance gaps over "
                f"{census.n_records_with_gaps}/{census.n_records} records; "
                f"{len(census.worklist_fields)} deferred field(s): "
                f"{census.worklist_fields}; {census.n_silent_none} undeclared None "
                f"values over {len(self._silent_by_field)} carrier fields"
                + (f" (top: {top_silent})" if top_silent else "")
            )
        return LevelResult(
            level=Level.L1,
            name="provenance_gaps",
            passed=True,
            message=message,
            details=census.model_dump(),
        )

    def _gene_name_result(self) -> LevelResult:
        """L1: one systematic name, one spelling, and both resolve to each other."""
        spellings: dict[str, set[str | None]] = {}
        for (systematic, common), _ in self._name_records.items():
            spellings.setdefault(systematic, set()).add(common)
        split = {
            systematic: sorted(str(name) for name in names)
            for systematic, names in spellings.items()
            if len(names) > 1
        }
        split_records = sum(
            count
            for (systematic, _), count in self._name_records.items()
            if systematic in split
        )
        not_current: list[str] = []
        mismatched: list[str] = []
        if self.resolve_gene_name is not None:
            for systematic in sorted(spellings):
                resolution = self.resolve_gene_name(systematic)
                if (
                    str(_enum_value(resolution.status)) != "current"
                    or resolution.systematic_name != systematic
                ):
                    not_current.append(
                        f"{systematic} ({_enum_value(resolution.status)}"
                        f" -> {resolution.systematic_name})"
                    )
            for systematic, common in sorted(
                (s, c) for s, c in self._name_records if c is not None
            ):
                resolved = self.resolve_gene_name(str(common)).systematic_name
                if resolved != systematic:
                    mismatched.append(f"{common} -> {resolved} (stored {systematic})")
        passed = not (split or not_current or mismatched)
        if not self._name_records:
            message = "no gene perturbations to check"
        elif passed:
            message = (
                f"{len(spellings)} systematic names, one canonical spelling each"
                + (
                    ", each current in the genome"
                    if self.resolve_gene_name is not None
                    else " (no resolver supplied: spelling checked, annotation not)"
                )
            )
        else:
            message = (
                f"{len(split)} genes carry more than one common-name spelling "
                f"({split_records} records); {len(not_current)} systematic names are not "
                f"the genome's current name; {len(mismatched)} common names resolve to "
                "another gene"
            )
        return LevelResult(
            level=Level.L1,
            name="canonical_gene_names",
            passed=passed,
            message=message,
            details={
                "n_genes": len(spellings),
                "resolver": self.resolve_gene_name is not None,
                "split_spellings": dict(sorted(split.items())[:20]),
                "n_records_with_split_spelling": split_records,
                "not_current": not_current[:20],
                "common_name_mismatch": mismatched[:20],
            },
        )

    def _uncertainty_result(self) -> LevelResult:
        """L2: a labelled dispersion is never exactly 0; unreported ones are counted."""
        passed = self._n_zero_dispersion == 0
        return LevelResult(
            level=Level.L2,
            name="uncertainty_sanity",
            passed=passed,
            message=(
                f"{self._n_uncertainty_checked} labelled uncertainties, none a zero "
                f"dispersion; {self._n_unreported_uncertainty} records report "
                "n_samples >= 2 with no uncertainty"
                if passed
                else f"{self._n_zero_dispersion}/{self._n_uncertainty_checked} labelled "
                "uncertainties are a sample dispersion of exactly 0 (a pseudocount "
                "artifact, not perfect precision)"
            ),
            details={
                "n_checked": self._n_uncertainty_checked,
                "n_zero_dispersion": self._n_zero_dispersion,
                "n_no_uncertainty_with_replicates": self._n_unreported_uncertainty,
                "examples": self._zero_dispersion_examples,
            },
        )

    def _compound_identity_results(self) -> list[LevelResult]:
        """L3: every compound is identified, or its absence is typed. Two results.

        The rule is one rule, but a dataset owns the compounds it DOSES and the shared
        library owns the components of the medium, so an unencodable compound is reported
        under the layer that has to fix it: ``compound_identity`` for the environment edits,
        ``media_compound_identity`` for the base medium's defined components. Both fail on a
        name-only compound; splitting them is what keeps one shared-library gap from hiding
        a chemogenomic set's own 1,700 name-only compounds in the same line.
        """
        return [
            self._identity_result(name, media=is_media)
            for name, is_media in (
                ("compound_identity", False),
                ("media_compound_identity", True),
            )
        ]

    def _identity_result(self, name: str, *, media: bool) -> LevelResult:
        by_verdict: Counter[str] = Counter()
        name_only: Counter[str] = Counter()
        gapped: Counter[str] = Counter()
        for (context, compound, verdict), count in self._compound_records.items():
            if context.startswith("media.component") is not media:
                continue
            by_verdict[verdict] += count
            if verdict == "name_only":
                name_only[f"{compound} ({context})"] += count
            elif verdict == "gapped":
                gapped[f"{compound} ({context})"] += count
        scope = "medium components" if media else "environment edits"
        passed = not name_only
        return LevelResult(
            level=Level.L3,
            name=name,
            passed=passed,
            message=(
                f"{scope}: {by_verdict['identified']} compound references carry a "
                f"structure identifier; {by_verdict['gapped']} declare a typed gap "
                f"({len(gapped)} distinct compounds, unencodable)"
                if passed
                else f"{scope}: {len(name_only)} compounds are name-only (no identifier, "
                f"no gap) over {by_verdict['name_only']} references: "
                + ", ".join(
                    f"{compound} x{count}"
                    for compound, count in name_only.most_common(10)
                )
            ),
            details={
                "scope": scope,
                "n_identified": by_verdict["identified"],
                "n_gapped": by_verdict["gapped"],
                "n_name_only": by_verdict["name_only"],
                "name_only_records": dict(name_only.most_common(50)),
                "gapped_records": dict(gapped.most_common(50)),
            },
        )

    def _media_result(self) -> LevelResult:
        """L3: every medium is a shared library object or derives from one."""
        free_text: Counter[str] = Counter()
        matched: dict[str, str] = {}
        n_library = 0
        n_derived = 0
        for (name, verdict, base), count in self._media_records.items():
            if verdict == "free_text":
                free_text[f"{name} (base_medium={base})"] += count
            elif verdict.startswith("library:"):
                n_library += count
                matched[name] = verdict
            else:
                n_derived += count
                matched[name] = verdict
        passed = not free_text
        return LevelResult(
            level=Level.L3,
            name="media_membership",
            passed=passed,
            message=(
                f"{n_library} records on a shared MEDIA_LIBRARY medium, {n_derived} on a "
                f"medium deriving from one ({len(matched)} distinct media)"
                if passed
                else f"{len(free_text)} free-text media over {sum(free_text.values())} "
                "records join nothing: "
                + ", ".join(
                    f"{name} x{count}" for name, count in free_text.most_common(10)
                )
            ),
            details={
                "n_library_records": n_library,
                "n_derived_records": n_derived,
                "matched_media": dict(sorted(matched.items())),
                "free_text_records": dict(free_text.most_common(50)),
            },
        )

    def _gene_containment_results(self, sgd_genes: set[str]) -> list[LevelResult]:
        """L4: the aggregate containment floor, plus the per-record genome membership."""
        measured = set(self._gene_records)
        missing = sorted(measured - sgd_genes)
        overlap = len(measured & sgd_genes) / len(measured) if measured else 0.0
        containment = LevelResult(
            level=Level.L4,
            name="gene_containment_sgd",
            passed=overlap >= self.min_containment,
            message=(
                f"{overlap:.3f} of {len(measured)} measured genes are S288C reference "
                f"genes (>= {self.min_containment})"
            ),
            details={
                "n_measured": len(measured),
                "n_in_sgd": len(measured & sgd_genes),
                "overlap": overlap,
                "missing_examples": missing[:20],
            },
        )
        off_genome = LevelResult(
            level=Level.L4,
            name="current_genome_genes",
            passed=not missing,
            message=(
                f"every one of the {len(measured)} measured systematic names is a gene of "
                "the current genome"
                if not missing
                else f"{len(missing)} systematic names are absent from the current genome "
                f"over {self._n_records_off_genome} records: "
                + ", ".join(
                    f"{gene} x{self._gene_records[gene]}" for gene in missing[:10]
                )
            ),
            details={
                "n_missing_genes": len(missing),
                "n_records": self._n_records_off_genome,
                "missing_records": {
                    gene: self._gene_records[gene] for gene in missing[:50]
                },
            },
        )
        return [containment, off_genome]


def _walk_carriers(node: Any) -> Iterable[Mapping[str, Any]]:
    """Every gap-capable carrier mapping reachable inside a stored record.

    A carrier is any mapping that carries ``provenance_gaps`` -- that field is exactly what
    ``ProvenanceGapMixin`` adds, so the walk finds the phenotype, the environment, each
    compound in the medium and in each perturbation, and anything a later model gains the
    mixin for, without a traversal list to keep in sync.
    """
    if isinstance(node, Mapping):
        if GAPS_FIELD in node:
            yield node
        for value in node.values():
            yield from _walk_carriers(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _walk_carriers(value)


def shared_rule_results(
    records: Sequence[Record],
    *,
    background_genes: frozenset[str] = frozenset(),
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    min_containment: float = 0.90,
) -> list[LevelResult]:
    """Run every shared rule over a materialized record sequence."""
    rules = SharedRecordRules(
        background_genes=background_genes,
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        min_containment=min_containment,
    )
    rules.add_all(records)
    return rules.results()
