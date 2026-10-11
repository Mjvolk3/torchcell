# torchcell/verification/synthetic_lethality
# [[torchcell.verification.synthetic_lethality]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/synthetic_lethality
"""L0-L4 record-level verifier for the SynLethDB yeast stores (#889).

``syn_leth_db_yeast`` and ``syn_rescue_db_yeast`` store one record per kept row of
SynLethDB 2.0's ``Yeast_SL.csv`` / ``Yeast_SR.csv``: an unordered pair of yeast genes,
each side resolved by its Entrez id, a boolean label (synthetic lethal / synthetic
rescue) that is True, the curation's statistic score when it states one, and the row's
PubMed id. The two arms differ only in their label field names, which
:class:`PairLabelKind` carries.

1. L0 ``structural`` / ``reference_structural`` -- as the declared experiment and
   reference classes.
2. L1 ``count`` -- the registry oracle; L1 ``two_distinct_genes`` -- every record
   perturbs two different named genes at graph level ``edge``; L1
   ``unordered_pair_uniqueness`` -- no unordered gene pair is stored twice (the loader
   refuses a repeat, so this checks the bytes it left behind).
3. L2 ``positive_label`` -- every experiment's label is True; L2
   ``score_in_unit_interval`` -- every stated score lies in [0, 1], the range of the
   normalized confidence score the release paper defines for SL pairs (for a file the
   paper defines no score for, L2 ``score_defined_by_source`` instead: no record may
   state one); L2 ``released_rows`` -- the stored (unordered ORF pair,
   score, PubMed id) multiset is the released file's kept rows, read from the pinned
   CSV with each side resolved by its Entrez id through the pinned NCBI GFF; L2
   ``release_accounting`` -- released rows = stored records + ledgered drops, and,
   where the release paper states the number, released rows = that number.
4. L3 ``reference_negative`` -- every reference's label is False with no score, in the
   experiment's own environment; L3 ``drop_ledger`` (computed by the caller) -- every
   ledgered drop meets its stated rule on the released bytes; L3 ``provenance_audit``
   rows for the quotes.
5. Shared rules and the two L4 gene rules against the S288C universe.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from torchcell.verification.common import (
    GeneNameResolver,
    SharedRecordRules,
    declared_member_validator,
    l0_validated_row,
)
from torchcell.verification.levels import l1_count
from torchcell.verification.report import (
    Level,
    LevelResult,
    Provenance,
    VerificationReport,
)

__all__ = [
    "PairLabelKind",
    "Record",
    "ReleasedPairs",
    "SYNTHETIC_LETHALITY",
    "SYNTHETIC_RESCUE",
    "pair_row_key",
    "verify_synthetic_pair_dataset",
]

Record = Mapping[str, Any]

#: One stored or released row: the unordered ORF pair, the score (None when the release
#: states none) and the PubMed id string.
PairRow = tuple[tuple[str, str], float | None, str]


class PairLabelKind(BaseModel):
    """The class and field names that tell a lethality record from a rescue record."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    experiment_class: str
    reference_class: str
    label_field: str
    score_field: str


SYNTHETIC_LETHALITY = PairLabelKind(
    experiment_class="SyntheticLethalityExperiment",
    reference_class="SyntheticLethalityExperimentReference",
    label_field="is_synthetic_lethal",
    score_field="synthetic_lethality_statistic_score",
)
SYNTHETIC_RESCUE = PairLabelKind(
    experiment_class="SyntheticRescueExperiment",
    reference_class="SyntheticRescueExperimentReference",
    label_field="is_synthetic_rescue",
    score_field="synthetic_rescue_statistic_score",
)


def pair_row_key(
    orf_a: str, orf_b: str, score: float | None, pubmed_id: str
) -> PairRow:
    """The order-free key one stored record and one released row are compared by."""
    pair = (orf_a, orf_b) if orf_a <= orf_b else (orf_b, orf_a)
    return (pair, score, pubmed_id)


class ReleasedPairs(BaseModel):
    """What the caller read from the pinned release.

    ``drift`` names every pinned file whose sha256 changed; when non-empty, ``rows`` is
    None and the comparison rules fail on the drift alone. ``n_released_rows`` counts
    every data row of the released file; ``n_dropped`` is the ledger's drop count;
    ``stated_count`` is the number the release paper states for this file, None when it
    states none.
    """

    model_config = ConfigDict(extra="forbid")

    files: list[str]
    drift: dict[str, str]
    rows: Counter[PairRow] | None
    n_released_rows: int | None
    n_dropped: int
    stated_count: int | None
    stated_count_quote: str | None


def verify_synthetic_pair_dataset(
    records: Iterable[Record],
    *,
    kind: PairLabelKind,
    dataset_name: str,
    provenance: Provenance,
    expected_count: int,
    released: ReleasedPairs,
    score_definition: str | None,
    extra_results: Sequence[LevelResult] = (),
    resolve_gene_name: GeneNameResolver | None = None,
    sgd_genes: set[str] | None = None,
    gene_universe_label: str = "reference",
    min_containment: float = 0.90,
) -> VerificationReport:
    """Run the L0-L4 gate over one SynLethDB store in a single pass.

    ``score_definition`` is the source's own statement of what the score is (SynLethDB
    2.0 defines its normalized confidence score for SL pairs); with one, every stated
    score must lie in [0, 1]. Without one, a stated score has no sourced meaning, and the
    store passes only if it states none.
    """
    validate = declared_member_validator(kind.experiment_class)
    validate_reference = declared_member_validator(
        kind.reference_class, union="ExperimentReferenceType"
    )
    shared = SharedRecordRules(
        resolve_gene_name=resolve_gene_name,
        sgd_genes=sgd_genes,
        gene_universe_label=gene_universe_label,
        min_containment=min_containment,
    )
    n_records = 0
    l0_failures: list[dict[str, Any]] = []
    ref_failures: list[dict[str, Any]] = []
    not_pair: list[dict[str, Any]] = []
    pairs: Counter[tuple[str, str]] = Counter()
    stored: Counter[PairRow] = Counter()
    not_true: list[int] = []
    n_scores = 0
    bad_scores: list[dict[str, Any]] = []
    reference_problems: Counter[str] = Counter()

    for i, rec in enumerate(records):
        n_records += 1
        exp = rec["experiment"]
        ref = rec["reference"]
        shared.add(rec)
        try:
            validate(exp)
        except (ValueError, TypeError) as err:
            l0_failures.append({"index": i, "error": str(err)[:500]})
        try:
            validate_reference(ref)
        except (ValueError, TypeError) as err:
            ref_failures.append({"index": i, "error": str(err)[:500]})

        perturbations = exp["genotype"]["perturbations"]
        names = [str(p.get("systematic_gene_name")) for p in perturbations]
        phenotype = exp["phenotype"]
        if (
            len(perturbations) != 2
            or len(set(names)) != 2
            or "None" in names
            or phenotype.get("graph_level") != "edge"
        ):
            not_pair.append({"index": i, "genes": names})
            continue
        pair = (names[0], names[1]) if names[0] <= names[1] else (names[1], names[0])
        pairs[pair] += 1
        score = phenotype.get(kind.score_field)
        stored[
            pair_row_key(pair[0], pair[1], score, str(rec["publication"]["pubmed_id"]))
        ] += 1
        if phenotype.get(kind.label_field) is not True:
            not_true.append(i)
        if score is not None:
            n_scores += 1
            if not isinstance(score, (int, float)) or not (
                math.isfinite(score) and 0.0 <= score <= 1.0
            ):
                bad_scores.append({"index": i, "value": repr(score)})
        ref_phenotype = ref["phenotype_reference"]
        if ref_phenotype.get(kind.label_field) is not False:
            reference_problems[f"reference {kind.label_field} is not False"] += 1
        if ref_phenotype.get(kind.score_field) is not None:
            reference_problems["reference carries a score"] += 1
        if ref["environment_reference"] != exp["environment"]:
            reference_problems["reference environment differs"] += 1

    report = VerificationReport(dataset_name=dataset_name, provenance=provenance)
    report.add(
        l0_validated_row("structural", n_records, l0_failures, kind.experiment_class)
    )
    report.add(
        l0_validated_row(
            "reference_structural", n_records, ref_failures, kind.reference_class
        )
    )
    report.add(l1_count(n_records, expected_count))
    report.add(
        LevelResult(
            level=Level.L1,
            name="two_distinct_genes",
            passed=n_records > 0 and not not_pair,
            message=(
                f"all {n_records} records perturb two distinct named genes at graph "
                "level 'edge'"
                if not not_pair
                else f"{len(not_pair)} of {n_records} records are not a pair of two "
                "distinct genes at graph level 'edge'"
            ),
            details={"n_bad": len(not_pair), "examples": not_pair[:20]},
        )
    )
    repeated = {pair: n for pair, n in pairs.items() if n > 1}
    report.add(
        LevelResult(
            level=Level.L1,
            name="unordered_pair_uniqueness",
            passed=n_records > 0 and not repeated,
            message=(
                f"{len(pairs)} unordered gene pairs, one record each"
                if not repeated
                else f"{len(repeated)} unordered gene pairs are stored more than once"
            ),
            details={
                "n_pairs": len(pairs),
                "n_repeated": len(repeated),
                "examples": sorted("+".join(p) for p in repeated)[:20],
            },
        )
    )
    report.add(
        LevelResult(
            level=Level.L2,
            name="positive_label",
            passed=n_records > 0 and not not_true,
            message=(
                f"all {n_records} experiments say {kind.label_field} True"
                if not not_true
                else f"{len(not_true)} of {n_records} experiments do not say "
                f"{kind.label_field} True"
            ),
            details={"n_bad": len(not_true), "example_indices": not_true[:20]},
        )
    )
    report.add(
        _score_result(
            n_records, n_scores, bad_scores, defined=score_definition is not None
        )
    )
    report.add(_released_rows_result(stored, released))
    report.add(_release_accounting_result(n_records, released))
    report.add(
        LevelResult(
            level=Level.L3,
            name="reference_negative",
            passed=n_records > 0 and not reference_problems,
            message=(
                f"every one of the {n_records} references says {kind.label_field} "
                "False with no score, in its experiment's own environment"
                if not reference_problems
                else f"reference problems: {dict(reference_problems)}"
            ),
            details={"problems": dict(reference_problems)},
        )
    )
    for result in extra_results:
        report.add(result)
    for result in shared.results():
        report.add(result)
    return report


def _released_rows_result(
    stored: Counter[PairRow], released: ReleasedPairs
) -> LevelResult:
    """L2 ``released_rows``: the stored rows are the release's kept rows."""
    details: dict[str, Any] = {
        "files": released.files,
        "n_stored": sum(stored.values()),
    }
    if released.drift or released.rows is None:
        return LevelResult(
            level=Level.L2,
            name="released_rows",
            passed=False,
            message=f"sha256 drift in {sorted(released.drift)}: the release was not read",
            details={**details, "drift": released.drift},
        )
    only_stored = stored - released.rows
    only_released = released.rows - stored
    passed = not only_stored and not only_released
    return LevelResult(
        level=Level.L2,
        name="released_rows",
        passed=passed,
        message=(
            f"the {sum(stored.values())} stored (gene pair, score, PubMed id) rows are "
            "the release's kept rows, each side resolved by its Entrez id"
            if passed
            else f"{sum(only_stored.values())} stored rows are not released and "
            f"{sum(only_released.values())} kept released rows are not stored"
        ),
        details={
            **details,
            "n_released_kept": sum(released.rows.values()),
            "only_stored": sorted(map(str, only_stored))[:20],
            "only_released": sorted(map(str, only_released))[:20],
        },
    )


def _release_accounting_result(n_records: int, released: ReleasedPairs) -> LevelResult:
    """L2 ``release_accounting``: released rows = stored + dropped (= the stated count)."""
    n_rows = released.n_released_rows
    problems: list[str] = []
    if n_rows is None:
        problems.append("the release was not read")
    else:
        if n_rows != n_records + released.n_dropped:
            problems.append(
                f"{n_rows} released rows != {n_records} stored + "
                f"{released.n_dropped} dropped"
            )
        if released.stated_count is not None and n_rows != released.stated_count:
            problems.append(
                f"{n_rows} released rows != the {released.stated_count} the release "
                "paper states"
            )
    stated = (
        f"; the release paper states {released.stated_count}"
        if released.stated_count is not None
        else "; the release paper states no count for this file"
    )
    return LevelResult(
        level=Level.L2,
        name="release_accounting",
        passed=not problems,
        message=(
            f"{n_rows} released rows = {n_records} stored + {released.n_dropped} "
            f"ledgered drops{stated}"
            if not problems
            else "; ".join(problems)
        ),
        details={
            "n_released_rows": n_rows,
            "n_stored": n_records,
            "n_dropped": released.n_dropped,
            "stated_count": released.stated_count,
            "stated_count_quote": released.stated_count_quote,
        },
    )


def _score_result(
    n_records: int, n_scores: int, bad_scores: list[dict[str, Any]], *, defined: bool
) -> LevelResult:
    """L2: the stated scores lie in [0, 1], or, with no sourced definition, none exist."""
    if defined:
        return LevelResult(
            level=Level.L2,
            name="score_in_unit_interval",
            passed=not bad_scores,
            message=(
                f"{n_scores} stated scores lie in [0, 1] ({n_records - n_scores} "
                "records state none)"
                if not bad_scores
                else f"{len(bad_scores)} of {n_scores} stated scores lie outside [0, 1]"
            ),
            details={
                "n_scores": n_scores,
                "n_without_score": n_records - n_scores,
                "bad": bad_scores[:20],
            },
        )
    return LevelResult(
        level=Level.L2,
        name="score_defined_by_source",
        passed=n_scores == 0,
        message=(
            f"no record states a score ({n_records} records), and the source defines none"
            if n_scores == 0
            else f"{n_scores} of {n_records} records state a score the source never "
            f"defines ({len(bad_scores)} of them outside [0, 1])"
        ),
        details={
            "n_scores": n_scores,
            "n_outside_unit_interval": len(bad_scores),
            "examples": bad_scores[:20],
        },
    )
