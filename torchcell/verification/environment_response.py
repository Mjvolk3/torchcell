# torchcell/verification/environment_response
# [[torchcell.verification.environment_response]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/environment_response
"""L0-L4 record-level verifier for environment-response datasets (WS15).

For ``EnvironmentResponsePhenotype`` datasets: chemical-genomic / stress screens read
out as ``(deletion genotype x EnvironmentPerturbation -> response)`` records (e.g. the
Vanacloig 2022 anaerobic hydrolysate-toxin bar-seq screen). The schema validator already
enforces numeric-vs-categorical coherence and the uncertainty invariant (subsumed by L0);
this verifier adds:

1. L1 ``count`` -- exact record-count oracle.
2. L1 ``pair_uniqueness`` -- one record per (screened STRAIN x condition) pair, where the
   strain is the full genotype signature (so an allelic series is distinct, not duplicate).
3. L2 ``response_finiteness`` -- numeric responses are finite (SIGNED; negatives allowed).
4. L2 ``se_nonnegative`` -- reported SEs are non-negative.
5. L3 ``measurement_type_consistent`` -- one measurement_type across the dataset.
6. L2 ``interval_orientation`` -- how many stored two-sided intervals do not bracket
   their value, against a count the loader declares (a limit carried through a nonlinear
   transform can land on the wrong side of the estimate; #776).
7. L3 ``reference_zero`` -- the reference (parent-strain) response is 0 for a numeric
   readout; for a purely CATEGORICAL dataset the numeric rule has nothing to look at, so
   the reference instead has to carry the dataset's neutral baseline category; for an
   ABSOLUTE readout (``reference_centered=False``) the reference instead states its own
   finite value on the record's own scale, and that branch refuses any record whose
   ``measurement_type`` is not in ``ABSOLUTE_MEASUREMENT_TYPES``. The result says which
   of the three rules ran.
8. L3 ``environment_perturbed`` -- every experiment carries a genuine environmental edit
   (>= 1 perturbation, or a temperature shift off the dataset baseline, e.g. heat),
   except for a declared count of base-condition records an absolute readout measures.
9. L4 ``gene_containment`` -- screened deletions overlap the deletion collection.

On top of these it runs :class:`torchcell.verification.common.SharedRecordRules`, the
rules that are not specific to this readout: the gap + silent-None census over every
carrier, canonical gene names, uncertainty sanity, compound identity, media membership,
and the L4 gene rules.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from torchcell.verification.common import GeneNameResolver, SharedRecordRules
from torchcell.verification.levels import l0_structural, l1_count
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

Record = dict[str, Any]


def _screened_genes(
    experiment: dict[str, Any], background: frozenset[str]
) -> list[str]:
    """Systematic names deleted in this experiment's genotype, minus the constant
    drug-sensitized background (which is identical in every strain and is not a
    screened deletion, so it must not enter the L1/L4 deleted-ORF key).
    """
    return [
        p["systematic_gene_name"]
        for p in experiment["genotype"]["perturbations"]
        if p.get("systematic_gene_name") is not None
        and p["systematic_gene_name"] not in background
    ]


def _condition_signature(experiment: dict[str, Any]) -> tuple[Any, ...]:
    """Canonical ENVIRONMENT (condition) identity -- the join key on the environment axis.

    A record's condition is its full environment, not just the compound NAME: the same
    compound at two concentrations, or two exposure durations, are DIFFERENT conditions.
    The signature is the sorted set of perturbations -- each keyed by
    ``(perturbation_type, compound/agent name, factor, concentration value, unit, basis)``
    -- plus the scalar environment fields (temperature, media, duration in hours and in
    generations). Temperature-only records (heat on ``Environment.temperature``, no
    perturbation) are distinguished by the temperature scalar.

    Mirrors :func:`_genotype_signature` on the environment axis: identity is derived from
    the environment's own content, so nothing has to be smuggled into a name. Elements are
    stringified for a total order (mixed None/str/float never breaks the sort).

    A study that dosed the SAME compound at the SAME concentration in two separate screens
    measured two conditions, not one: the screens are normalized independently, so the
    phenotype's ``screen_id`` joins the signature when the dataset carries one (Hoepfner
    has 45 such same-compound, same-dose column pairs). ``.get`` keeps the signature
    identical for every dataset that does not carry the field.
    """
    env = experiment["environment"]
    perts: list[tuple[str, ...]] = []
    for p in env.get("perturbations") or []:
        compound = p.get("compound") if isinstance(p.get("compound"), dict) else {}
        agent = p.get("agent") if isinstance(p.get("agent"), dict) else {}
        dose = p.get("concentration") or p.get("magnitude") or {}
        dose = dose if isinstance(dose, dict) else {}
        fields = (
            p.get("perturbation_type"),
            compound.get("name") or agent.get("name"),
            p.get("factor"),
            dose.get("value"),
            dose.get("unit"),
            dose.get("basis"),
        )
        perts.append(tuple("" if v is None else str(v) for v in fields))
    temp = (env.get("temperature") or {}).get("value")
    media = (env.get("media") or {}).get("name")
    return (
        tuple(sorted(perts)),
        temp,
        media,
        env.get("duration_hours"),
        env.get("duration_generations"),
        experiment["phenotype"].get("screen_id"),
    )


def _genotype_signature(
    experiment: dict[str, Any], background: frozenset[str]
) -> tuple[tuple[str | None, ...], ...]:
    """Canonical STRAIN identity: the sorted set of screened perturbations, each keyed by
    ``(systematic_gene_name, perturbation_type, perturbed_gene_name)``, excluding the constant
    drug-sensitized background.

    This is the entity a record measures and the join key across datasets. A deletion
    collection has one strain per gene, so the signature reduces to the gene. A TS-allele
    collection has MANY strains per essential gene: ``act1-101`` and ``act1-3`` differ in
    ``perturbed_gene_name`` and so get distinct signatures -- an allelic series is not a set
    of duplicates. A CRISPR guide library likewise has MANY strains per (gene, mode): six
    guides targeting the same gene under the same effector are distinct strains, so the
    guide spacer (``crispr.guide_sequence``) joins the key when present (None for a
    background/unspecified guide leaves the key unchanged). When a study screens several
    guide LIBRARY POOLS, the SAME spacer measured in two pools is two independent pooled
    measurements (Smith 2016: pool-relative median-centred fitness differs by up to ~8 log2
    units between pools), so the pool (``crispr.library_pool``) also joins the key when
    present (None leaves the key unchanged -> single-pool studies keep their signatures). L4
    gene-containment still keys on the bare systematic name (a gene-level question); L1
    uniqueness keys on this strain identity.
    """

    def _identity(p: dict[str, Any]) -> tuple[str | None, ...]:
        ident: tuple[str | None, ...] = (
            p.get("systematic_gene_name"),
            p.get("perturbation_type"),
            p.get("perturbed_gene_name"),
        )
        crispr = p.get("crispr")
        if isinstance(crispr, dict):
            if crispr.get("guide_sequence") is not None:
                ident = ident + (crispr["guide_sequence"],)
            if crispr.get("library_pool") is not None:
                ident = ident + (crispr["library_pool"],)
        # the donor is a field of the CRISPR deletion perturbation itself, not of
        # its construct: two designs can share a gene and a spacer and differ only
        # in the donor (Lian 2019, 150 groups)
        if p.get("donor_sequence") is not None:
            ident = ident + (p["donor_sequence"],)
        # The strain discriminators of the #507 chemogenomic leaves: two strains that
        # delete one ORF but were constructed separately (Hillenmeyer "constructed more
        # than once, in different batches"), sit in different collections or pools, or
        # were built against different historical ORFs are different strains, never
        # duplicates. Absent on every other leaf, so their keys are unchanged.
        construction = p.get("construction")
        if isinstance(construction, dict):
            ident = ident + tuple(
                construction.get(field)
                for field in ("strain_accession", "lab", "batch", "plate", "well")
            )
        if p.get("collection") is not None:
            ident = ident + (p["collection"],)
        # The strain discriminators of the round-2 bacterial allele leaves (#749, #792).
        # One locus can carry a deletion, an affinity tag, a marked insertion and two
        # DIFFERENT degrons, and the first three already separate on perturbation_type;
        # the two degrons do not. 'imp-DAS' and 'imp-DAS+4' agree on locus, kind and
        # gene name and differ only in the degron, so the allele fields join the key.
        # Adding a field can only SPLIT a group, never merge two, so every dataset that
        # passes uniqueness today still passes. Absent on every other leaf except
        # ``cassette``, which is constant per strain where it is present.
        for field in ("tag", "terminus", "degron", "cassette", "insertion_site"):
            if p.get(field) is not None:
                ident = ident + (str(p[field]),)
        constructed = p.get("constructed_orf")
        if isinstance(constructed, dict):
            ident = ident + (constructed.get("source_systematic_name"),)
        return ident

    return tuple(
        sorted(
            _identity(p)
            for p in experiment["genotype"]["perturbations"]
            if p.get("systematic_gene_name") not in background
        )
    )


def _study_key(record: Record) -> tuple[str, str, str, str]:
    """Measurement-context discriminator: (publication, readout ``units``, ``screen_id``).

    A single-study, single-assay dataset has a constant context, so this does not affect
    uniqueness. A MULTI-study / multi-assay aggregation (e.g. YeastPhenome) legitimately
    measures the SAME (strain, condition) in DIFFERENT screens -- a different study, the
    same study by a different assay (microarray vs barseq, recorded in ``units``), or the
    same study's own repeat screen of one compound (``screen_id``). Each readout is
    normalized within its own screen, so those are independent measurements, NOT
    duplicates -- the context joins the uniqueness key. A true duplicate (same context, same
    strain, same condition) is still caught. ``screen_id`` is read with ``.get`` so a
    dataset without the field keys exactly as before.

    ``replicate_id`` (#776) joins it for the same reason one step finer: a source that
    releases one row PER REPLICATE CURVE, each with its own confidence interval and fit
    quality (Caglar 2017 Table S5: 55 rows over 19 conditions), has measured 55 things,
    not 19 with 36 duplicates. Aggregating them would mean computing a condition mean the
    paper never released. It is read with ``.get`` and an absent or ``None`` value keys as
    ``""``, so every dataset that releases one aggregated measurement per (strain,
    condition) -- every other dataset -- keys exactly as before.
    """
    pub = record.get("publication") or {}
    phenotype = record["experiment"]["phenotype"]
    return (
        str(pub.get("pubmed_id") or pub.get("doi") or ""),
        str(phenotype.get("units") or ""),
        str(phenotype.get("screen_id") or ""),
        str(phenotype.get("replicate_id") or ""),
    )


def _l1_pair_uniqueness(
    records: Sequence[Record], background: frozenset[str]
) -> LevelResult:
    """L1: exactly one record per (study, screened STRAIN, condition) triple.

    The screened unit is the STRAIN (the full genotype signature), not the bare gene: an
    essential gene screened as an allelic series (18 ACT1 ts alleles) contributes 18 distinct
    strains, not 18 duplicate ACT1 records. Genuine replicate measurements of an identical
    (strain, condition) WITHIN one study must aggregate into one record (n_samples); a repeat
    within a study is a real duplicate. The study (source publication) joins the key so that
    independent near-replicate SCREENS across studies (a curated multi-study aggregation) are
    not flagged as duplicates -- for a single-study dataset the study is constant and the key
    reduces to (strain, condition).
    """
    seen = {_pair_key(rec, background) for rec in records}
    return _pair_uniqueness_result(
        n_pairs=len(seen), n_duplicated=len(records) - len(seen)
    )


def _pair_key(record: Record, background: frozenset[str]) -> tuple[Any, ...]:
    """The L1 uniqueness key: (study, strain signature, condition signature)."""
    experiment = record["experiment"]
    return (
        _study_key(record),
        _genotype_signature(experiment, background),
        _condition_signature(experiment),
    )


def _pair_uniqueness_result(*, n_pairs: int, n_duplicated: int) -> LevelResult:
    """L1 ``pair_uniqueness`` row, shared by the eager and streaming verifiers.

    ``n_duplicated`` counts REDUNDANT RECORDS (every record after the first with a given
    key), so three copies of one record are 2 duplicates and
    ``n_pairs + n_duplicated`` equals the ``count`` row's observed record total. This is
    the number of records the loader has to aggregate or drop for L1 to pass, the unit the
    count oracle is stated in, and the one a single streaming pass computes without
    holding a per-key counter.
    """
    return LevelResult(
        level=Level.L1,
        name="pair_uniqueness",
        passed=n_duplicated == 0,
        message=(
            f"{n_pairs} unique (study, strain, condition) records, one each"
            if n_duplicated == 0
            else f"{n_duplicated} records duplicate an earlier (study, strain, "
            f"condition) triple; {n_pairs} unique triples"
        ),
        details={"n_pairs": n_pairs, "n_duplicated": n_duplicated},
    )


def _value_problem(
    index: int, value: float, *, minimum: float | None
) -> dict[str, Any] | None:
    """The L2 entry for one bad value, or None when the value is fine.

    ``index`` is the RECORD's position in the dataset (records without the value are
    skipped but still counted), so an entry points at the record to inspect. The entry
    shape is :func:`torchcell.verification.levels.l2_value_fidelity`'s.
    """
    if math.isnan(value):
        return {"index": index, "value": "nan", "reason": "nan"}
    if math.isinf(value):
        return {"index": index, "value": repr(value), "reason": "inf"}
    if minimum is not None and value < minimum:
        return {"index": index, "value": value, "reason": f"< {minimum}"}
    return None


def _value_result(name: str, n_values: int, bad: list[dict[str, Any]]) -> LevelResult:
    """L2 value row (``value_fidelity`` / ``se_nonnegative``) for both verifiers."""
    return LevelResult(
        level=Level.L2,
        name=name,
        passed=not bad,
        message=(
            f"{n_values} values checked"
            if not bad
            else f"{len(bad)}/{n_values} values invalid"
        ),
        details={"n_values": n_values, "n_bad": len(bad), "bad": bad[:20]},
    )


def _response_value(record: Record) -> float | None:
    """The experiment's numeric response, or None for a categorical record."""
    value = record["experiment"]["phenotype"]["environment_response"]
    return None if value is None else float(value)


def _se_value(record: Record) -> float | None:
    """The reported response SE, or None when it is absent or NaN (not reported)."""
    value = record["experiment"]["phenotype"].get("environment_response_se")
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _measurement_type_result(types: set[str]) -> LevelResult:
    """L3 ``measurement_type_consistent`` row for both verifiers.

    ``types`` holds the enum VALUES (``str`` of a ``MeasurementType`` member), so the
    message reads ``'log2_ratio'`` whether the records carry the enum member
    (``model_dump()``) or its JSON string.
    """
    return LevelResult(
        level=Level.L3,
        name="measurement_type_consistent",
        passed=len(types) <= 1,
        message=(
            f"single measurement_type: {next(iter(types), None)!r}"
            if len(types) <= 1
            else f"{len(types)} distinct measurement_types mixed: {sorted(types)}"
        ),
        details={"measurement_types": sorted(types)},
    )


def _l3_measurement_type_consistent(records: Sequence[Record]) -> LevelResult:
    """L3: all records share a single measurement_type (no silent cross-assay mixing)."""
    return _measurement_type_result(
        {str(rec["experiment"]["phenotype"]["measurement_type"]) for rec in records}
    )


def _reference_baseline_result(
    *,
    n_numeric: int,
    worst: float,
    reference_categories: Counter[str],
    n_reference_missing_category: int,
    experiment_categories: Counter[str],
) -> LevelResult:
    """L3: the reference record carries the readout's baseline. Two rules, one result.

    NUMERIC readout: the reference (parent-strain) response is 0 -- log2(1) = 0, the
    control baseline. CATEGORICAL readout: there is no number to be 0, and the numeric rule
    then passes over zero values, which is how a purely categorical dataset scored a green
    L3 that checked nothing. Its baseline is a CATEGORY, so the rule becomes: every
    reference carries a category, they are all the SAME category (the dataset's declared
    baseline), and no experiment record reports that baseline as a measured call. The
    message names the rule that ran so a reader can tell a real pass from a vacuous one.
    """
    if n_numeric == 0 and (reference_categories or n_reference_missing_category):
        baselines = sorted(reference_categories)
        n_refs = sum(reference_categories.values())
        collisions = {
            category: experiment_categories[category]
            for category in baselines
            if experiment_categories.get(category)
        }
        # A census screen scores every strain, so its neutral baseline is
        # necessarily also a measured call (Smith 2006 scores wild type as the
        # modal grade); only a hits-only screen keeps the baseline out of the
        # measured vocabulary. Collisions are reported, never failed.
        passed = n_reference_missing_category == 0 and len(baselines) == 1
        if passed:
            message = (
                f"categorical rule: all {n_refs} references carry the baseline category "
                f"{baselines[0]!r}, which no experiment record reports"
            )
        else:
            message = (
                f"categorical rule: {n_reference_missing_category} references carry no "
                f"category; {len(baselines)} distinct reference categories {baselines}; "
                f"baseline also used as a measured call in {collisions}"
            )
        return LevelResult(
            level=Level.L3,
            name="reference_zero",
            passed=passed,
            message=message,
            details={
                "rule": "categorical_baseline",
                "n_values": n_refs,
                "reference_categories": dict(reference_categories),
                "n_reference_missing_category": n_reference_missing_category,
                "baseline_used_as_measured_call": collisions,
            },
        )
    holds = worst == 0.0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=holds,
        message=(
            f"numeric rule: reference response == 0 for all {n_numeric} records"
            if holds
            else f"numeric rule: reference response not identically 0: max|v|={worst:.3g}"
        ),
        details={"rule": "numeric_zero", "n_values": n_numeric, "worst_abs": worst},
    )


def _l3_reference_zero(records: Sequence[Record]) -> LevelResult:
    """L3: the reference carries the baseline (numeric 0, or the neutral category)."""
    worst = 0.0
    n = 0
    reference_categories: Counter[str] = Counter()
    n_missing = 0
    experiment_categories: Counter[str] = Counter()
    for rec in records:
        reference = rec["reference"]["phenotype_reference"]
        v = reference["environment_response"]
        if v is None:
            category = reference.get("category")
            if category is None:
                n_missing += 1
            else:
                reference_categories[str(category)] += 1
        else:
            n += 1
            worst = max(worst, abs(float(v)))
        experiment_category = rec["experiment"]["phenotype"].get("category")
        if experiment_category is not None:
            experiment_categories[str(experiment_category)] += 1
    return _reference_baseline_result(
        n_numeric=n,
        worst=worst,
        reference_categories=reference_categories,
        n_reference_missing_category=n_missing,
        experiment_categories=experiment_categories,
    )


# --------------------------------------------------------------------------- #
# ABSOLUTE readouts and two-sided intervals (#776). Two DECLARED reliefs, each
# gated on a measurement the caller has to state, so neither is a blanket
# relaxation: `reference_centered=False` refuses any record whose
# measurement_type is not in ABSOLUTE_MEASUREMENT_TYPES, and the two counts are
# oracles (observed must EQUAL declared), so a regression in either fails.
# --------------------------------------------------------------------------- #
def _reference_absolute_result(
    *,
    n_values: int,
    n_bad_reference: int,
    n_relative_type: int,
    n_type_mismatch: int,
    types: set[str],
) -> LevelResult:
    """L3 ``reference_zero``, the ABSOLUTE branch: the reference states its own value.

    An absolute readout cannot have a reference of 0 and be honest: Caglar 2017's base
    condition doubles in 53.68 min, and a stored 0 would assert instant growth. So the
    rule becomes the one the metabolite verifier's ``reference_finite`` branch uses for
    absolute quantities -- the reference is WELL-DEFINED rather than centered: a finite
    number is present, and it is on the SAME measurement scale as the experiment record
    it references. The relief is gated: every record's ``measurement_type`` must be a
    member of ``ABSOLUTE_MEASUREMENT_TYPES``, so asking for it on a log2-ratio or z-score
    dataset FAILS instead of quietly skipping the zero check. The message names the
    branch that ran, as the categorical branch does.
    """
    passed = n_bad_reference == 0 and n_relative_type == 0 and n_type_mismatch == 0
    return LevelResult(
        level=Level.L3,
        name="reference_zero",
        passed=passed,
        message=(
            f"absolute rule: reference value finite and on the record's own scale for "
            f"all {n_values} records ({sorted(types)})"
            if passed
            else f"absolute rule: {n_bad_reference} references absent or non-finite; "
            f"{n_relative_type} records carry a RELATIVE measurement_type, which is 0 "
            f"at its control by construction and may not take this relief; "
            f"{n_type_mismatch} references on a different scale than their experiment "
            f"({sorted(types)})"
        ),
        details={
            "rule": "absolute_reference",
            "n_values": n_values,
            "n_bad_reference": n_bad_reference,
            "n_relative_measurement_type": n_relative_type,
            "n_reference_type_mismatch": n_type_mismatch,
            "measurement_types": sorted(types),
        },
    )


def _l3_reference_absolute(records: Sequence[Record]) -> LevelResult:
    """L3: the reference of an ABSOLUTE readout states its own measured value."""
    from torchcell.datamodels.schema import ABSOLUTE_MEASUREMENT_TYPES

    absolute = {str(member.value) for member in ABSOLUTE_MEASUREMENT_TYPES}
    accumulator = _AbsoluteReferenceAccumulator(absolute)
    for rec in records:
        accumulator.add(rec)
    return accumulator.result()


class _AbsoluteReferenceAccumulator:
    """Single-pass accumulator for the absolute-reference rule (eager + streaming)."""

    def __init__(self, absolute_types: set[str]) -> None:
        self._absolute = absolute_types
        self.n_values = 0
        self.n_bad_reference = 0
        self.n_relative_type = 0
        self.n_type_mismatch = 0
        self.types: set[str] = set()

    def add(self, record: Record) -> None:
        """Score one record's reference against its experiment's scale."""
        phenotype = record["experiment"]["phenotype"]
        kind = str(phenotype["measurement_type"])
        self.types.add(kind)
        self.n_values += 1
        if kind not in self._absolute:
            self.n_relative_type += 1
        reference = record["reference"]["phenotype_reference"]
        value = reference["environment_response"]
        if value is None or not math.isfinite(float(value)):
            self.n_bad_reference += 1
        if str(reference["measurement_type"]) != kind:
            self.n_type_mismatch += 1

    def result(self) -> LevelResult:
        """The L3 row this accumulator's counts imply."""
        return _reference_absolute_result(
            n_values=self.n_values,
            n_bad_reference=self.n_bad_reference,
            n_relative_type=self.n_relative_type,
            n_type_mismatch=self.n_type_mismatch,
            types=self.types,
        )


def _interval_orientation_result(
    *,
    n_intervals: int,
    n_non_bracketing: int,
    expected: int,
    examples: list[dict[str, Any]],
) -> LevelResult:
    """L2 ``interval_orientation``: how many stored intervals do NOT bracket their value.

    A confidence limit carried through a nonlinear transform can land on the wrong side
    of the estimate -- Caglar 2017 Table S5 releases ``95p = -1027.769034`` against a
    doubling time of 80.95 min, which is what the image of a slope interval straddling
    zero becomes under ``DT = log_e 2 / slope``. The schema stores the released bytes
    rather than repairing them, so this rule is the measurement of how many such rows a
    dataset holds, against a count the loader DECLARES. ``expected=0`` (the default) is
    the ordinary case and makes any inverted interval a failure; a dataset that declares
    a nonzero count passes only while the count holds exactly, so neither a new inverted
    row nor a vanished one goes unnoticed.
    """
    passed = n_non_bracketing == expected
    return LevelResult(
        level=Level.L2,
        name="interval_orientation",
        passed=passed,
        message=(
            f"{n_non_bracketing} of {n_intervals} stored intervals do not bracket their "
            f"value, as declared"
            if passed
            else f"{n_non_bracketing} of {n_intervals} stored intervals do not bracket "
            f"their value; {expected} declared"
        ),
        details={
            "n_intervals": n_intervals,
            "n_non_bracketing": n_non_bracketing,
            "expected_non_bracketing": expected,
            "examples": examples[:10],
        },
    )


class _IntervalAccumulator:
    """Counts stored two-sided intervals and the ones that do not bracket the value."""

    def __init__(self) -> None:
        self.n_intervals = 0
        self.n_non_bracketing = 0
        self.examples: list[dict[str, Any]] = []

    def add(self, index: int, record: Record) -> None:
        """Score one record's stored interval, if it carries one."""
        phenotype = record["experiment"]["phenotype"]
        lower = phenotype.get("environment_response_lower")
        upper = phenotype.get("environment_response_upper")
        if lower is None and upper is None:
            return
        self.n_intervals += 1
        value = phenotype.get("environment_response")
        if value is None:
            return
        value = float(value)
        brackets = (lower is None or float(lower) <= value) and (
            upper is None or float(upper) >= value
        )
        if brackets:
            return
        self.n_non_bracketing += 1
        if len(self.examples) < 10:
            self.examples.append(
                {"index": index, "value": value, "lower": lower, "upper": upper}
            )

    def result(self, expected: int) -> LevelResult:
        """The L2 row this accumulator's counts imply, against the declared count."""
        return _interval_orientation_result(
            n_intervals=self.n_intervals,
            n_non_bracketing=self.n_non_bracketing,
            expected=expected,
            examples=self.examples,
        )


def _environment_perturbed_result(
    *,
    n_records: int,
    n_missing: int,
    expected: int,
    baseline_temp: Any,
    baseline_media: Any,
) -> LevelResult:
    """L3 ``environment_perturbed`` row, shared by the eager and streaming verifiers.

    ``expected`` is how many records the loader DECLARES carry no environmental edit.
    It is 0 for a response dataset, where an unperturbed record is a defect. For an
    ABSOLUTE readout the base condition is itself a measured condition, so the declared
    count is the number of its records (Caglar 2017: the base condition was run in three
    separate experiments, 9 of 55 rows). Declared rather than waived: observed must equal
    declared, so the base-condition rows cannot silently grow or disappear. This is the
    same shape as Bloom 2019's ``conditions_documented``, which requires the no-edit
    columns to be exactly the absolute-readout columns.
    """
    passed = n_missing == expected
    return LevelResult(
        level=Level.L3,
        name="environment_perturbed",
        passed=passed,
        message=(
            f"all {n_records} experiments carry an environmental edit "
            f"(perturbation, non-baseline temperature, or non-baseline media; "
            f"baseline temp={baseline_temp}, media={baseline_media!r})"
            if passed and expected == 0
            else f"{n_missing} of {n_records} experiments carry no environmental edit, "
            f"as declared (the absolute readout's base condition; baseline temp="
            f"{baseline_temp}, media={baseline_media!r})"
            if passed
            else f"{n_missing} experiments have no environmental edit "
            f"(no perturbation, baseline temperature {baseline_temp}, baseline media); "
            f"{expected} declared"
        ),
        details={
            "n_records": n_records,
            "n_missing": n_missing,
            "expected_unperturbed": expected,
            "baseline_temperature": baseline_temp,
            "baseline_media": baseline_media,
        },
    )


# --------------------------------------------------------------------------- #
# end ABSOLUTE readouts and two-sided intervals (#776)
# --------------------------------------------------------------------------- #


def _modal_scalar(
    records: Sequence[Record], getter: Callable[[dict[str, Any]], Any]
) -> Any:
    """The dataset's baseline (most common) value of an environment scalar.

    ``None`` counts as a value: a dataset whose records gap the temperature has an
    UNSTATED baseline (Hillenmeyer 2008, whose SOM never gives the growth
    temperature), and a record that states one differs from it. Skipping ``None``
    would elect a temperature-shift condition as the baseline and then flag that
    condition's own records as unperturbed.
    """
    values = Counter(getter(rec["experiment"]["environment"]) for rec in records)
    return values.most_common(1)[0][0] if values else None


def _l3_environment_perturbed(
    records: Sequence[Record], expected_unperturbed: int = 0
) -> LevelResult:
    """L3: every experiment carries a genuine environmental edit.

    The edit is >= 1 environment perturbation (an added small molecule / physical factor)
    OR a base-environment scalar that differs from the dataset's baseline (modal) value --
    a temperature shift (heat) or a base-medium swap lives canonically on
    ``Environment.temperature`` / ``Environment.media`` with NO perturbation object (M2),
    and is a valid edit. A record is flagged only when it has NO perturbation AND sits at
    the baseline temperature AND the baseline media (a genuinely unperturbed record).

    ``expected_unperturbed`` is the count the loader DECLARES (#776): 0 for a response
    dataset, and the base condition's record count for an absolute readout, whose base
    condition is a measured condition rather than a defect.
    """
    baseline_temp = _modal_scalar(
        records, lambda e: (e.get("temperature") or {}).get("value")
    )
    baseline_media = _modal_scalar(
        records, lambda e: (e.get("media") or {}).get("name")
    )
    n_missing = 0
    for rec in records:
        env = rec["experiment"]["environment"]
        if env.get("perturbations"):
            continue
        temp = (env.get("temperature") or {}).get("value")
        if temp is not None and temp != baseline_temp:
            continue  # temperature-only edit (e.g. heat) -- genuine environmental edit
        media = (env.get("media") or {}).get("name")
        if media is not None and media != baseline_media:
            continue  # base-medium swap (minimal / synthetic complete) -- genuine edit
        n_missing += 1
    return _environment_perturbed_result(
        n_records=len(records),
        n_missing=n_missing,
        expected=expected_unperturbed,
        baseline_temp=baseline_temp,
        baseline_media=baseline_media,
    )


def verify_environment_response_dataset(
    records: Sequence[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    background_genes: frozenset[str] = frozenset(),
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
    reference_centered: bool = True,
    expected_unperturbed: int = 0,
    expected_non_bracketing: int = 0,
) -> VerificationReport:
    """Run the L0-L4 record-level gate for an environment-response dataset.

    ``background_genes`` are the systematic names of the constant drug-sensitized
    background (e.g. Vanacloig 3DeltaAlpha = PDR1/PDR3/SNQ2), excluded from the
    (ORF, compound) uniqueness and gene-set keys. ``sgd_genes`` turns on the L4 gene rules
    (aggregate containment + per-record genome membership) and
    ``gene_universe_label`` is what the containment row calls that universe, since it is
    the host's and not always S288C's; ``resolve_gene_name`` turns on the annotation half
    of the canonical-gene-name rule. Both are optional so this verifier still runs where
    no genome is mounted.

    The three ABSOLUTE-readout declarations (#776), each an oracle rather than a waiver:

    - ``reference_centered`` (default True) asserts the reference response is identically
      0, which is what a relative readout is at its control. Set False for an ABSOLUTE
      quantity (Caglar 2017's doubling time in minutes), where the reference states the
      reference condition's own measured value; the absolute branch then REFUSES any
      record whose ``measurement_type`` is not in ``ABSOLUTE_MEASUREMENT_TYPES``, so it
      cannot be used to skip the zero check on a log2-ratio dataset.
    - ``expected_unperturbed`` (default 0) is how many records carry no environmental
      edit. Nonzero only for an absolute readout, whose base condition is a measured
      condition; observed must EQUAL declared.
    - ``expected_non_bracketing`` (default 0) is how many stored two-sided intervals do
      not bracket their value, which a nonlinear transform of a released interval can
      produce; observed must EQUAL declared.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(l0_structural((rec["experiment"] for rec in records), validate))
    report.add(l1_count(len(records), expected_count))
    report.add(_l1_pair_uniqueness(records, background_genes))

    shared = SharedRecordRules(
        background_genes=background_genes,
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )
    shared.add_all(records)

    responses = [
        (i, v)
        for i, rec in enumerate(records)
        if (v := _response_value(rec)) is not None
    ]
    report.add(
        _value_result(
            "value_fidelity",
            len(responses),
            [
                bad
                for i, v in responses
                if (bad := _value_problem(i, v, minimum=None)) is not None
            ],
        )
    )
    se_values = [
        (i, v) for i, rec in enumerate(records) if (v := _se_value(rec)) is not None
    ]
    report.add(
        _value_result(
            "se_nonnegative",
            len(se_values),
            [
                bad
                for i, v in se_values
                if (bad := _value_problem(i, v, minimum=0.0)) is not None
            ],
        )
    )

    intervals = _IntervalAccumulator()
    for i, rec in enumerate(records):
        intervals.add(i, rec)
    report.add(intervals.result(expected_non_bracketing))

    report.add(_l3_measurement_type_consistent(records))
    report.add(
        _l3_reference_zero(records)
        if reference_centered
        else _l3_reference_absolute(records)
    )
    report.add(_l3_environment_perturbed(records, expected_unperturbed))
    for result in shared.results():
        report.add(result)
    return report


def environment_response_gene_set(
    records: Sequence[Record], background_genes: frozenset[str] = frozenset()
) -> set[str]:
    """Union of screened (non-background) deleted gene names -- the L4 overlap key."""
    genes: set[str] = set()
    for rec in records:
        genes.update(_screened_genes(rec["experiment"], background_genes))
    return genes


def verify_environment_response_dataset_streaming(
    records: Iterable[Record],
    *,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    sgd_genes: set[str],
    background_genes: frozenset[str] = frozenset(),
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
    resolve_gene_name: GeneNameResolver | None = None,
    reference_centered: bool = True,
    expected_unperturbed: int = 0,
    expected_non_bracketing: int = 0,
) -> VerificationReport:
    """Single-pass, memory-bounded L0-L4 gate for LARGE environment-response datasets.

    Semantically identical to ``verify_environment_response_dataset``, but consumes
    ``records`` as a stream so a 30M-record dataset (e.g. the Hoepfner HIP-HOP atlas) never
    has to be materialized in RAM. The dominant accumulator is the (ORF, compound) pair
    set; interning keeps it a few GB, not the ~450 GB a full materialization would cost.
    The shared rules are accumulators for the same reason, and memoize their per-condition
    verdicts on the identity of the interned environment mapping.
    """
    from pydantic import TypeAdapter

    from torchcell.datamodels.schema import ExperimentType

    validate: Callable[[Any], object] = TypeAdapter(ExperimentType).validate_python

    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    pair_seen: set[tuple[Any, ...]] = set()
    n_pairs = 0
    n_pair_dups = 0
    n_responses = 0
    bad_responses: list[dict[str, Any]] = []
    n_se = 0
    bad_se: list[dict[str, Any]] = []
    measurement_types: set[str] = set()
    ref_worst = 0.0
    n_ref = 0
    reference_categories: Counter[str] = Counter()
    n_reference_missing_category = 0
    experiment_categories: Counter[str] = Counter()
    # Environment-edit accounting: a record with no perturbation is a valid edit iff its
    # temperature or media differs from the dataset baseline (modal). Baselines need all
    # records, so accumulate scalar counts + the (temp, media) of no-perturbation records
    # and resolve after the pass (the no-perturbation set is small).
    temp_counts: Counter[Any] = Counter()
    media_counts: Counter[Any] = Counter()
    no_pert_env: list[tuple[Any, Any]] = []
    # #776: the two declared-count rules and the absolute-reference branch, as
    # accumulators, so the streaming report carries exactly the eager one's rows.
    from torchcell.datamodels.schema import ABSOLUTE_MEASUREMENT_TYPES

    intervals = _IntervalAccumulator()
    absolute_reference = _AbsoluteReferenceAccumulator(
        {str(member.value) for member in ABSOLUTE_MEASUREMENT_TYPES}
    )
    shared = SharedRecordRules(
        background_genes=background_genes,
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )

    for i, rec in enumerate(records):
        exp = rec["experiment"]
        n_records += 1
        shared.add(rec)
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})

        # L1 uniqueness keys on the STUDY x the STRAIN (genotype signature) x the full
        # CONDITION signature (environment identity); L4 gene-containment accumulates the
        # bare screened systematic names.
        pkey = _pair_key(rec, background_genes)
        if pkey in pair_seen:
            n_pair_dups += 1
        else:
            pair_seen.add(pkey)
            n_pairs += 1

        response = _response_value(rec)
        if response is not None:
            n_responses += 1
            if (bad := _value_problem(i, response, minimum=None)) is not None:
                bad_responses.append(bad)
        se = _se_value(rec)
        if se is not None:
            n_se += 1
            if (bad := _value_problem(i, se, minimum=0.0)) is not None:
                bad_se.append(bad)
        measurement_types.add(str(exp["phenotype"]["measurement_type"]))

        intervals.add(i, rec)
        absolute_reference.add(rec)

        reference = rec["reference"]["phenotype_reference"]
        ref_val = reference["environment_response"]
        if ref_val is not None:
            n_ref += 1
            ref_worst = max(ref_worst, abs(float(ref_val)))
        else:
            ref_category = reference.get("category")
            if ref_category is None:
                n_reference_missing_category += 1
            else:
                reference_categories[str(ref_category)] += 1
        exp_category = exp["phenotype"].get("category")
        if exp_category is not None:
            experiment_categories[str(exp_category)] += 1
        temp = (exp["environment"].get("temperature") or {}).get("value")
        media = (exp["environment"].get("media") or {}).get("name")
        # None counts as a baseline value, as in _modal_scalar: a gapped baseline
        # temperature is an unstated one, and a stated temperature differs from it.
        temp_counts[temp] += 1
        media_counts[media] += 1
        if not (exp["environment"].get("perturbations") or []):
            no_pert_env.append((temp, media))

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        LevelResult(
            level=Level.L0,
            name="structural",
            passed=not l0_failures,
            message=(
                f"{n_records} records validated"
                if not l0_failures
                else f"{len(l0_failures)}/{n_records} records failed schema validation"
            ),
            details={
                "n_records": n_records,
                "n_failures": len(l0_failures),
                "failures": l0_failures[:10],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L1,
            name="count",
            passed=n_records == expected_count,
            message=f"observed {n_records}, expected {expected_count}",
            details={"observed": n_records, "expected": expected_count},
        )
    )
    report.add(_pair_uniqueness_result(n_pairs=n_pairs, n_duplicated=n_pair_dups))
    report.add(_value_result("value_fidelity", n_responses, bad_responses))
    report.add(_value_result("se_nonnegative", n_se, bad_se))
    report.add(intervals.result(expected_non_bracketing))
    report.add(_measurement_type_result(measurement_types))
    report.add(
        _reference_baseline_result(
            n_numeric=n_ref,
            worst=ref_worst,
            reference_categories=reference_categories,
            n_reference_missing_category=n_reference_missing_category,
            experiment_categories=experiment_categories,
        )
        if reference_centered
        else absolute_reference.result()
    )
    baseline_temp = temp_counts.most_common(1)[0][0] if temp_counts else None
    baseline_media = media_counts.most_common(1)[0][0] if media_counts else None
    n_env_missing = sum(
        1
        for temp, media in no_pert_env
        if not (temp is not None and temp != baseline_temp)
        and not (media is not None and media != baseline_media)
    )
    report.add(
        _environment_perturbed_result(
            n_records=n_records,
            n_missing=n_env_missing,
            expected=expected_unperturbed,
            baseline_temp=baseline_temp,
            baseline_media=baseline_media,
        )
    )
    # L1 census + canonical names, L2 uncertainty, L3 identity/media and the two L4 gene
    # rules all come from the shared accumulator, so the streaming report carries exactly
    # the results the eager one does.
    for result in shared.results():
        report.add(result)
    return report


__all__ = [
    "verify_environment_response_dataset",
    "verify_environment_response_dataset_streaming",
    "environment_response_gene_set",
    "Record",
]
