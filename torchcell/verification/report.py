# torchcell/verification/report
# [[torchcell.verification.report]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/verification/report
"""Provenance + L0-L4 verification report models (roadmap WS3).

These are pure data models describing WHERE a dataset came from (``Provenance``)
and WHETHER it passed the universal L0-L4 record-level gate (``VerificationReport``
of ``LevelResult``). The level-check building blocks that produce ``LevelResult``s
live in ``torchcell.verification.levels``.

Every rule result carries a typed :class:`FindingClass`. A ``gate`` rule tests the
data's fidelity to its release and decides the verdict; a ``report`` rule compares the
data with OUR genome reference and is listed under the verdict with its statistic,
never failing a store. The class is a property of the rule, read from one registry
(:data:`GENOME_REFERENCE_RULES`, :data:`GENOME_REFERENCE_RULE_PREFIXES`) by
:func:`finding_class_of`, so a rule rendered by any verifier, a family runner or a
loader's own gate is classified the same way.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class Level(IntEnum):
    """The five universal verification levels (ordered: a gate, not a set)."""

    L0 = 0  # structural: record instantiates its schema model
    L1 = 1  # completeness: count / known-keys / contiguity vs an oracle
    L2 = 2  # value fidelity: cross-method agreement, type/range/NaN
    L3 = 3  # semantic: units/conventions (fitness = ko/wt; log2(sample/ref))
    L4 = 4  # cross-source consistency: overlapping entities agree


class Provenance(BaseModel):
    """Where a dataset's records came from, recorded alongside verification."""

    model_config = ConfigDict(extra="forbid")

    source_uri: str = Field(
        description="URL, DOI, or file path of the source artifact."
    )
    citation_key: str | None = Field(
        default=None, description="Zotero/library identity (e.g. authorTitleYear)."
    )
    sha256: str | None = Field(
        default=None, description="Hex sha256 of the source artifact, if hashed."
    )
    method: str | None = Field(
        default=None, description="Extraction method (e.g. 'pdftotext -layout')."
    )
    page: str | None = Field(
        default=None, description="Page/table/sheet reference within the source."
    )
    retrieved: str | None = Field(
        default=None,
        description="ISO date the artifact was retrieved (caller-supplied).",
    )


class DerivationMethod(StrEnum):
    """How a non-sourced statistic's value was fixed (ordered by preference).

    ``back_solve`` and the range fallbacks apply only when the source does NOT
    state the value per-record. Policy (CLAUDE.md "Adding Datasets"): prefer
    ``back_solve`` from another reported statistic; else ``conservative_low``;
    ``median`` only if a central estimate is explicitly wanted over conservatism.
    """

    sourced = "sourced"  # stated explicitly in the source (constant + citation)
    back_solve = "back_solve"  # inferred from another reported statistic
    conservative_low = "conservative_low"  # lower end of a stated range
    median = "median"  # median of a stated range


class StatDerivation(BaseModel):
    """How a derived (non-per-record-sourced) statistic was fixed, as data.

    For values the source does not give per record (e.g. an effective
    ``n_samples`` that only appears as a range): records the method, chosen value,
    the source range, and diagnostics, so the number and its justification travel
    together and stay re-checkable. Intended to be emitted as a post-process
    artifact alongside ``experiment_reference_index.json`` (roadmap WS3), and to
    back the loader constant (e.g. Kuzmin ``N_SAMPLES_COMBINED_MUTANT``).
    """

    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        description="Statistic derived, e.g. 'combined_mutant_n_samples'."
    )
    method: DerivationMethod
    value: float = Field(description="The chosen value the loader uses.")
    range_low: float | None = Field(
        default=None, description="Low end of the source range (range methods)."
    )
    range_high: float | None = Field(
        default=None, description="High end of the source range (range methods)."
    )
    statistic: str | None = Field(
        default=None,
        description="For back_solve: the reported statistic matched "
        "(e.g. 'reported interaction p-value median').",
    )
    diagnostics: dict[str, float] = Field(
        default_factory=dict,
        description="Supporting numbers (e.g. per-candidate match, spearman).",
    )
    provenance: Provenance | None = Field(
        default=None, description="Source of the range/statistic (citation + sha256)."
    )
    rationale: str = Field(default="", description="Human-readable why.")

    @model_validator(mode="after")
    def _check_method_inputs(self) -> StatDerivation:
        """Enforce that each method carries the inputs that justify it."""
        if self.method in (DerivationMethod.conservative_low, DerivationMethod.median):
            if self.range_low is None or self.range_high is None:
                raise ValueError(f"{self.method} requires range_low and range_high")
            if not (self.range_low <= self.value <= self.range_high):
                raise ValueError("value must lie within [range_low, range_high]")
        if self.method is DerivationMethod.back_solve and self.statistic is None:
            raise ValueError("back_solve requires a `statistic` (what was matched)")
        return self


class FindingClass(StrEnum):
    """What a rule's outcome does to the store's verdict (owner, 2026-10-12).

    The owner's rule, verbatim: "the data is the data!!!! it is what they reported ...
    we don't drop raw data unless they themselves pull it! ... the genome is separate..
    it doesn't gate anything... we do that at modeling time if it is an issue."
    """

    #: The data's fidelity to its release: L0 structure, L1 counts and uniqueness, L2
    #: value fidelity to the released bytes and uncertainty sanity, L3 vocabulary,
    #: reference, compound and media membership, L4 agreement with another release.
    #: Every gate rule must hold for the store to PASS.
    gate = "gate"
    #: The data compared with OUR genome reference: a released name or position
    #: against the annotation (containment, name currency, canonical spellings, locus
    #: resolution, reference alleles at markers, MAT and cis positions, assembly and
    #: parent pins). Reported with its statistic under the verdict; never a FAIL, and
    #: never a reason to retire a store's build manifest.
    report = "report"


#: Rules whose subject is the genome reference rather than the data's fidelity to its
#: release, by exact name. Each one keeps its computation; only what it gates changes.
GENOME_REFERENCE_RULES: Final[frozenset[str]] = frozenset(
    {
        # verification.common (every family that passes a resolver or a gene universe);
        # its ``measured_genes_present`` row reads no annotation (it asks whether any
        # record carries a gene perturbation at all), so it stays a gate
        "canonical_gene_names",
        "current_genome_genes",
        # yeast family verifiers: alleles, markers and positions on the R64 annotation,
        # parent assemblies in the genomes tier, ids against the annotation's gene set
        "marker_reference",
        "mat_fixation",
        "mat_orientation",
        "cis_concordance",
        "variant_reference",
        "parents_pinned",
        "induced_gene_containment",
        "s288c_gene_containment",
        "pangenome_orf_containment",
        "feature_containment",
        "deleted_gene_containment_sgd",
        "promoter_gene_is_a_locus",
        # a stored or released identifier read against OUR annotation of the host. Each
        # of these three says in its own message that it is "a finding, never a gate"
        # and "the genome never gates storage", and each was classified ``gate`` until
        # 2026-10-15 because its name was off this list: the measured instance of the
        # defect the audit predicted (lane R, item 3). The exhaustiveness test over
        # every rule name the fleet emits is what finds the next one.
        "perturbed_locus_in_bw25113_annotation",
        "released_b_numbers_on_mg1655_annotation",
        "released_tags_on_kt2440_annotation",
        # bacterial loaders' own gates: stored identifiers against the pinned assembly
        "promoter_genes_in_the_host_gene_universe",
        "stored_tags_are_loci",
        "stored_tags_are_loci_of_the_pinned_assembly",
        "stored_targets_are_loci_of_the_pinned_assembly",
        "stored_protein_keys_are_loci_of_the_pinned_assembly",
        "stored_loci_are_bw25113_loci_derived_from_the_released_b_number",
        # Choe 2025: the released spacer against the released window read off OUR
        # pinned MG1655 sequence
        "spacer_matches_the_released_window_on_the_pinned_sequence",
        # Choe 2019: where the pinned MG1655 assembly places the released deletion
        # symbols (#1045 part 2)
        "deleted_symbols_are_one_contiguous_run_of_the_pinned_assembly",
        # the construct store: how many of a store's gene additions point at deposited
        # construct sequence, at a named host site with no coordinate, or at nothing but
        # a typed gap (``torchcell.verification.construct_placement``). The subject is
        # the record against OUR sequence references, so the row reports its statistic
        # and never gates a store whose source released no plasmid map
        "construct_placement_states",
    }
)

#: Rule families one helper renders under several names: ``gene_containment_<universe>``
#: (S288C, an assembly, a strain's locus tags, another store's deletion set keyed by
#: resolved names), the protein and perturbed-locus containment rows, every
#: ``assembly_pin*`` row (the record's pinned assembly is the genome reference itself),
#: and ``no_viable_deletion_in_<store>`` (an essentiality annotation judged against
#: another paper's deletion strains through the shared gene key: a gene-set judgment
#: across releases, not this store's fidelity to its own bytes).
GENOME_REFERENCE_RULE_PREFIXES: Final[tuple[str, ...]] = (
    "gene_containment_",
    "perturbed_gene_containment_",
    "protein_and_perturbed_locus_containment_",
    "assembly_pin",
    "no_viable_deletion_in_",
)


#: Words that appear in the NAME of a rule whose subject is our genome reference. Not a
#: classifier: :func:`finding_class_of` never reads this, and a name matching one of them
#: is not thereby a ``report`` rule. It is the vocabulary
#: ``tests/torchcell/verification/test_finding_class.py`` sweeps the fleet with, so a new
#: rule whose name speaks of an annotation, an assembly, a locus or a containment is
#: either classified here or listed in :data:`DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES`
#: with its reason. That is what makes the registry above exhaustive by test rather than
#: by hand: three rules had already slipped past it.
GENOME_REFERENCE_NAME_VOCABULARY: Final[tuple[str, ...]] = (
    "containment",
    "annotation",
    "assembly",
    "pinned",
    "_loci",
    "canonical_gene",
    "current_genome",
)

#: Rules whose NAME matches :data:`GENOME_REFERENCE_NAME_VOCABULARY` and that are
#: nevertheless ``gate``, each because its subject is the RELEASE and not our reference:
#:
#: - ``both_screens_are_balanced_and_strain_pinned`` (Cui 2018), ``table_s1_call_histogram_is_pinned``
#:   (Gerdes 2003), ``per_medium_key_counts_are_the_pinned_ones`` (Li 2014),
#:   ``per_condition_censored_and_interval_key_counts_are_the_pinned_ones`` (Gupta 2024),
#:   ``quantified_sites_per_sample_are_the_pinned_ones`` (Potel 2018): "pinned" here is a
#:   count or a key set this repo pinned from the released file, not an assembly pin.
#: - ``released_annotations`` (SGD essentiality): the annotations are the SOURCE's, and
#:   the rule is that the store holds the ones the release states.
#: - ``stored_loci_are_the_released_b_numbers_on_the_bw25113_pin`` (Hawkins 2020): its own
#:   docstring states "No annotation is read" -- the claim is that the keys are the
#:   released b-numbers verbatim.
#: - ``supplementary_note_2_loci_are_measured_in_every_record`` (Menasalvas 2025): the
#:   loci are the ones the release's own Supplementary Note names.
#: - ``fitness_companion_containment`` (SGA interactions): the companion is the same
#:   paper's fitness release, so the rule is cross-release fidelity, not an annotation.
DATA_FIDELITY_RULES_NAMED_LIKE_GENOME_RULES: Final[frozenset[str]] = frozenset(
    {
        "both_screens_are_balanced_and_strain_pinned",
        "fitness_companion_containment",
        "per_condition_censored_and_interval_key_counts_are_the_pinned_ones",
        "per_medium_key_counts_are_the_pinned_ones",
        "quantified_sites_per_sample_are_the_pinned_ones",
        "released_annotations",
        "stored_loci_are_the_released_b_numbers_on_the_bw25113_pin",
        "supplementary_note_2_loci_are_measured_in_every_record",
        "table_s1_call_histogram_is_pinned",
    }
)


def finding_class_of(rule_name: str) -> FindingClass:
    """``report`` for a genome-reference rule, ``gate`` for every other rule."""
    if rule_name in GENOME_REFERENCE_RULES or rule_name.startswith(
        GENOME_REFERENCE_RULE_PREFIXES
    ):
        return FindingClass.report
    return FindingClass.gate


class LevelResult(BaseModel):
    """The outcome of a single L-level check.

    ``passed`` is the rule's own outcome (does the rule hold). What that outcome DOES is
    :attr:`finding_class`, derived from the rule's name, so a renamed result
    (``model_copy(update={"name": ...})``, which skips validation) is reclassified with
    it and no caller can pass a genome rule off as a gate.
    """

    model_config = ConfigDict(extra="forbid")

    level: Level
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _read_serialized_finding_class(cls, data: Any) -> Any:
        """Accept the ``finding_class`` a written report carries, if it is the rule's.

        A report serialized under a different classification is stale against this
        registry and is refused rather than silently reclassified.
        """
        if not isinstance(data, Mapping) or "finding_class" not in data:
            return data
        stated = FindingClass(data["finding_class"])
        derived = finding_class_of(str(data["name"]))
        if stated is not derived:
            raise ValueError(
                f"rule {data['name']!r} was written as {stated.value}, but the registry "
                f"classifies it {derived.value}; regenerate the report"
            )
        return {key: value for key, value in data.items() if key != "finding_class"}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def finding_class(self) -> FindingClass:
        """``gate`` or ``report``, from the rule's name (:func:`finding_class_of`)."""
        return finding_class_of(self.name)

    @property
    def is_open_finding(self) -> bool:
        """A report-class rule that does not hold: listed under the verdict, not failed."""
        return self.finding_class is FindingClass.report and not self.passed


class VerificationReport(BaseModel):
    """Provenance + the L-level results for one dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    provenance: Provenance
    results: list[LevelResult] = Field(default_factory=list)

    @property
    def gate_results(self) -> list[LevelResult]:
        """The results that decide the verdict (``finding_class`` ``gate``)."""
        return [r for r in self.results if r.finding_class is FindingClass.gate]

    @property
    def findings(self) -> list[LevelResult]:
        """The genome-reference results (``finding_class`` ``report``), held or not."""
        return [r for r in self.results if r.finding_class is FindingClass.report]

    @property
    def open_findings(self) -> list[LevelResult]:
        """The genome-reference results that do not hold, listed under the verdict."""
        return [r for r in self.results if r.is_open_finding]

    @property
    def passed(self) -> bool:
        """True only if every GATE result passed; a report without one is NOT passed.

        A genome-reference finding never enters the verdict, whatever it says: a store
        whose only failing rows are ``report`` rows passes, with those rows listed.
        """
        gates = self.gate_results
        return bool(gates) and all(r.passed for r in gates)

    @property
    def levels_covered(self) -> set[Level]:
        """The set of L-levels that have a recorded result."""
        return {r.level for r in self.results}

    def add(self, result: LevelResult) -> VerificationReport:
        """Append a result and return self (chainable)."""
        self.results.append(result)
        return self

    def summary(self) -> str:
        """The verdict, one line per gate rule, then the genome findings under it.

        A gate row is ``[ok]`` or ``[XX]``; a genome-reference row is ``[ok]`` when the
        reference agrees and ``[rp]`` (reported) when it does not.
        """
        open_findings = self.open_findings
        head = f"{self.dataset_name}: {'PASS' if self.passed else 'FAIL'}" + (
            f" ({len(open_findings)} genome finding(s) reported, not gated)"
            if open_findings
            else ""
        )
        lines = [head]
        for r in sorted(self.gate_results, key=lambda x: x.level):
            mark = "ok" if r.passed else "XX"
            lines.append(f"  [{mark}] {r.level.name} {r.name}: {r.message}")
        findings = sorted(self.findings, key=lambda x: x.level)
        if findings:
            lines.append("  genome-reference findings (reported, never gated):")
            for r in findings:
                mark = "ok" if r.passed else "rp"
                lines.append(f"  [{mark}] {r.level.name} {r.name}: {r.message}")
        return "\n".join(lines)


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks (large artifacts)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
