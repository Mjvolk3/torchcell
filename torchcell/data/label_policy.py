# torchcell/data/label_policy.py
# [[torchcell.data.label_policy]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/data/label_policy
# Test file: tests/torchcell/data/test_label_policy.py
"""Choose, at READ time, which measurement of a genotype each label takes.

A build with no deduplication stage keeps every source entry under a genotype: a single
carries its Costanzo entries at both temperatures and both markers, its Kuzmin query
fitness where one exists, and the converted 0 where SGD lists the gene essential; a
double carries both Costanzo orientations and every Kuzmin digenic screen; a triple
carries one entry per screen that reached it. The trainer reads one fitness and one
interaction per record, so something has to choose, and that choice is a modeling
assumption rather than a property of the data.

``LabelPolicy`` is that chooser, expressed as a hashed pydantic object so it can sit
beside the split artifacts instead of inside the store. The build keeps every
alternative; the policy names the one this run uses; changing the rule costs a new
cached table rather than a rebuild.

Two faces, one policy:

``select``        given one record's entries, pick one value per label, with the entry
                  that supplied it recorded for provenance.
``trigenic_tau``  given a triple and the entries of its three doubles and three singles,
                  reconstruct the interaction score under the same precedence, which is
                  how a policy is validated against the published values.

The trigenic identity, verified against both Kuzmin raw tables, is

    tau_ijk = f_ijk - f_ij f_k - eps_ik - eps_jk

the triple's own fitness minus the double-mutant QUERY strain's fitness times the array
single, minus the two single-mutant control queries' adjusted scores at the same array.
It reproduces the released value on 99.98 percent of Kuzmin 2018 rows and 99.56 percent
of Kuzmin 2020 rows. The query single fitnesses enter as 1: that is what the released
numbers do, and it is NOT what the supplementary methods describe, which weight the two
control terms by the measured query singles. ``source_convention`` names which of the
two a run takes, because they are different numbers by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

__all__ = [
    "CONVERTED_ZERO",
    "LabelChoice",
    "LabelEntry",
    "LabelPolicy",
    "source_key",
    "strain_token",
]

# A Kuzmin query strain is identified by a "tm" number, but the two years write the
# surrounding string differently: 2018 puts the whole pair on both query perturbations
# ("YKL010C+YMR067C_tm2424"), 2020 puts it on one ("YDR003W_tm1501") and leaves the other
# bare. Measured on 5,000 triples of the 029 build, the tm token resolves in all 5,000,
# so it is the join key between a triple and the double-mutant query strain's own fitness
# record, which carries the full "GENE1+GENE2_tmNNNN" form.
_TM_TOKEN = re.compile(r"(tm\d+)")


def strain_token(strain_id: str | None) -> str | None:
    """The tm token identifying a Kuzmin query strain, or None when there is none."""
    match = _TM_TOKEN.search(strain_id or "")
    return match.group(1) if match else None


# A converted 0 is a statement about the gene, not a measurement of a strain: SGD
# essentiality and a SynthLethDB lethal pair both reach the store as fitness 0 through
# CompositeFitnessConverter. It is kept as its own source so a policy can refuse it
# wherever a measurement exists.
CONVERTED_ZERO = "converted_zero"

_CONVERTED_ZERO_DATASETS = ("GeneEssentialitySgd", "SynthLethality")


def source_key(dataset_name: str, temperature: float | None) -> str:
    """Name the screen an entry came from, at the granularity a policy ranks.

    Costanzo is split by temperature because its 26 and 30 degree screens are separate
    experiments; Kuzmin is not, because it screened at one temperature. A converted 0
    collapses to one key whatever produced it.
    """
    if any(tag in dataset_name for tag in _CONVERTED_ZERO_DATASETS):
        return CONVERTED_ZERO
    if "Kuzmin2018" in dataset_name:
        return "kuzmin2018"
    if "Kuzmin2020" in dataset_name:
        return "kuzmin2020"
    if "Costanzo2016" in dataset_name:
        if temperature is None:
            raise ValueError(f"{dataset_name} entry carries no temperature")
        return f"costanzo2016@{int(temperature)}"
    raise ValueError(f"no source key for dataset {dataset_name!r}")


class LabelEntry(BaseModel):
    """One source measurement of one genotype, normalized for the policy to rank."""

    source: str = Field(description="source_key of the screen that produced it")
    label: str = Field(description="'fitness' or 'gene_interaction'")
    value: float
    sd: float | None = None
    n_samples: int | None = None
    p_value: float | None = None
    strain_id: str | None = Field(
        default=None,
        description=(
            "the source strain identifier, e.g. a Kuzmin query strain 'GENE1+GENE2_tmNNNN'. "
            "Carries the query/array role and is what a triple matches its doubles on."
        ),
    )

    @property
    def is_converted_zero(self) -> bool:
        """True when the value is a converted 0 rather than a measured strain."""
        return self.source == CONVERTED_ZERO

    @property
    def standard_error(self) -> float | None:
        """SD over sqrt(n) when both are reported, else the SD, else nothing."""
        if self.sd is None or not math.isfinite(self.sd) or self.sd <= 0.0:
            return None
        if self.n_samples is None or self.n_samples < 1:
            return float(self.sd)
        return float(self.sd) / math.sqrt(float(self.n_samples))


class LabelChoice(BaseModel):
    """What a policy chose for one label of one record, and what it chose from."""

    value: float
    source: str
    sd: float | None = None
    standard_error: float | None = None
    n_samples: int | None = None
    p_value: float | None = None
    n_entries_combined: int = 1
    n_entries_available: int = 1

    model_config = {"frozen": True}


def _stouffer(
    p_values: list[float], signs: list[float], weights: list[float]
) -> float | None:
    """Combine two-sided p-values that share a direction, Stouffer's z with weights.

    A two-sided p carries magnitude but not direction, so the score's sign restores it
    before combining; the result is converted back to two-sided. Entries whose signs
    disagree combine toward a larger p, which is the honest reading of two screens that
    disagree about the direction of an interaction.
    """
    from scipy import stats

    usable = [
        (p, s, w)
        for p, s, w in zip(p_values, signs, weights)
        if p is not None and math.isfinite(p) and 0.0 < p <= 1.0 and w > 0.0
    ]
    if not usable:
        return None
    if len(usable) == 1:
        return float(usable[0][0])
    z = 0.0
    norm = 0.0
    for p, s, w in usable:
        one_sided = p / 2.0
        z_i = float(stats.norm.isf(one_sided)) * (1.0 if s >= 0 else -1.0)
        z += w * z_i
        norm += w * w
    z_combined = z / math.sqrt(norm)
    return float(2.0 * stats.norm.sf(abs(z_combined)))


class LabelPolicy(BaseModel):
    """Which measurement each label takes, as a hashed, versioned object.

    The default is the convention the released Kuzmin scores were computed under, which
    is the only setting that reproduces published trigenic values.
    """

    name: str = Field(
        description="human name, does not enter the hash's meaning but does enter the hash"
    )

    fitness_precedence: list[str] = Field(
        default_factory=lambda: [
            "kuzmin2018",
            "kuzmin2020",
            "costanzo2016@30",
            "costanzo2016@26",
            CONVERTED_ZERO,
        ],
        description="source keys, best first; an entry from an unlisted source is never chosen",
    )
    interaction_precedence: list[str] = Field(
        default_factory=lambda: [
            "kuzmin2018",
            "kuzmin2020",
            "costanzo2016@30",
            "costanzo2016@26",
        ]
    )
    same_year_kuzmin_first: bool = Field(
        default=True,
        description=(
            "when reconstructing a triple, promote the Kuzmin screen of the triple's own "
            "year above the other, since the source scored each screen against its own controls"
        ),
    )
    converted_zero_only_without_measurement: bool = Field(
        default=True,
        description=(
            "a converted 0 is admissible only when the genotype has no measured entry. "
            "Averaging it with a measurement is hazard H1 of the 025 build: it pulled 907 "
            "singles down by a median of 0.16."
        ),
    )
    replicate_combination: Literal["inverse_variance", "mean", "first"] = Field(
        default="inverse_variance",
        description="how entries of the SAME source combine; 'first' takes one and reports the rest as available",
    )
    p_combination: Literal["stouffer", "min", "none"] = "stouffer"
    source_convention: Literal["published", "measured"] = Field(
        default="published",
        description=(
            "'published' sets the query single-mutant fitness to 1 in the trigenic identity, "
            "which is what the released Kuzmin scores do. 'measured' uses the measured query "
            "singles, which the supplementary methods describe and which gives a different "
            "number by construction."
        ),
    )
    prefer_strain_matched_double: bool = Field(
        default=True,
        description=(
            "for a triple's double terms, prefer the entry whose strain_id equals the triple's "
            "query strain id, which is the double-mutant query strain the screen actually used, "
            "before falling back to the pair's own digenic screen"
        ),
    )

    model_config = {"frozen": True}

    @field_validator("fitness_precedence", "interaction_precedence")
    @classmethod
    def _no_duplicates(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"precedence has a repeated source: {v}")
        return v

    # ------------------------------------------------------------------ identity
    @property
    def policy_id(self) -> str:
        """sha256 over the canonical dump, so a table can be cached under the rule."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def precedence_for(self, label: str) -> list[str]:
        """The ordered source keys this policy ranks for one label."""
        if label == "fitness":
            return self.fitness_precedence
        if label == "gene_interaction":
            return self.interaction_precedence
        raise ValueError(f"no precedence for label {label!r}")

    # -------------------------------------------------------------------- select
    def select(self, entries: list[LabelEntry], label: str) -> LabelChoice | None:
        """Pick one value for ``label`` from one record's entries.

        Returns None when nothing admissible is present, which is the honest outcome for
        a record whose only entry is a converted 0 that this policy refuses.
        """
        pool = [e for e in entries if e.label == label]
        if not pool:
            return None
        n_available = len(pool)
        measured = [e for e in pool if not e.is_converted_zero]
        if self.converted_zero_only_without_measurement and measured:
            pool = measured
        order = self.precedence_for(label)
        for source in order:
            same = [e for e in pool if e.source == source]
            if same:
                return self._combine(same, n_available)
        return None

    def _combine(self, same_source: list[LabelEntry], n_available: int) -> LabelChoice:
        """Combine entries that share a source; they are replicates of one quantity."""
        source = same_source[0].source
        if len(same_source) == 1 or self.replicate_combination == "first":
            e = same_source[0]
            return LabelChoice(
                value=float(e.value),
                source=source,
                sd=e.sd,
                standard_error=e.standard_error,
                n_samples=e.n_samples,
                p_value=e.p_value,
                n_entries_combined=1,
                n_entries_available=n_available,
            )
        values = np.array([e.value for e in same_source], dtype=float)
        ses: list[float | None] = [e.standard_error for e in same_source]
        n_total = sum(e.n_samples for e in same_source if e.n_samples) or None
        weighted = self.replicate_combination == "inverse_variance" and all(
            s is not None and s > 0.0 for s in ses
        )
        se: float | None
        if weighted:
            positive = [s for s in ses if s is not None]
            w = np.array([1.0 / (s * s) for s in positive], dtype=float)
            value = float((w * values).sum() / w.sum())
            se = float(1.0 / math.sqrt(float(w.sum())))
        else:
            value = float(values.mean())
            finite = [s for s in ses if s is not None]
            se = (
                float(math.sqrt(sum(s * s for s in finite)) / len(same_source))
                if finite
                else None
            )
        sd = float(se * math.sqrt(n_total)) if (se is not None and n_total) else None

        p = None
        if self.p_combination == "stouffer":
            p = _stouffer(
                [e.p_value for e in same_source],  # type: ignore[misc]
                [float(np.sign(e.value)) for e in same_source],
                [float(e.n_samples or 1) for e in same_source],
            )
        elif self.p_combination == "min":
            candidates = [e.p_value for e in same_source if e.p_value is not None]
            p = float(min(candidates)) if candidates else None

        return LabelChoice(
            value=value,
            source=source,
            sd=sd,
            standard_error=se,
            n_samples=n_total,
            p_value=p,
            n_entries_combined=len(same_source),
            n_entries_available=n_available,
        )

    # ------------------------------------------------------------ term selection
    def select_double(
        self,
        entries: list[LabelEntry],
        query_strain_id: str | None,
        label: str = "fitness",
    ) -> LabelChoice | None:
        """Pick a double's value for use inside a triple's identity.

        The screen that produced the triple measured its own double-mutant query strain,
        and that measurement is a different experiment from the pair's digenic array
        screen: on Kuzmin 2018 the two agree at r 0.777 with a median absolute difference
        of 0.045, which is more than half the 0.08 calling threshold. So a strain match is
        preferred whenever the record carries one.
        """
        if self.prefer_strain_matched_double and query_strain_id is not None:
            token = strain_token(query_strain_id)
            matched = [
                e
                for e in entries
                if e.label == label
                and (
                    e.strain_id == query_strain_id
                    or (token is not None and strain_token(e.strain_id) == token)
                )
            ]
            if matched:
                return self._combine(matched, len(matched))
        return self.select(entries, label)

    def promoted_for_year(self, year_source: str | None) -> LabelPolicy:
        """This policy with the triple's own Kuzmin year moved to the front."""
        if not self.same_year_kuzmin_first or year_source not in (
            "kuzmin2018",
            "kuzmin2020",
        ):
            return self

        def _promote(order: list[str]) -> list[str]:
            if year_source not in order:
                return order
            return [year_source] + [s for s in order if s != year_source]

        return self.model_copy(
            update={
                "fitness_precedence": _promote(self.fitness_precedence),
                "interaction_precedence": _promote(self.interaction_precedence),
            }
        )

    # ------------------------------------------------------------------ identity
    def trigenic_tau(
        self,
        f_triple: float,
        f_query_double: float,
        f_array_single: float,
        eps_ik: float,
        eps_jk: float,
        f_query_single_i: float | None = None,
        f_query_single_j: float | None = None,
    ) -> float:
        """Tau under this policy's convention.

        ``published``: tau = f_ijk - f_ij f_k - eps_ik - eps_jk, the form the released
        Kuzmin scores were computed under.
        ``measured``:  the two control terms are weighted by the measured query singles,
        which is the form the supplementary methods write.
        """
        base = f_triple - f_query_double * f_array_single
        if self.source_convention == "published":
            return float(base - eps_ik - eps_jk)
        if f_query_single_i is None or f_query_single_j is None:
            raise ValueError(
                "the 'measured' convention needs both query single fitnesses"
            )
        return float(base - eps_ik * f_query_single_j - eps_jk * f_query_single_i)


def entries_from_records(rows: list[dict[str, Any]]) -> list[LabelEntry]:
    """Normalize a record's stored entries into ``LabelEntry`` objects.

    ``rows`` are the per-entry dicts a no-merge build stores under one genotype, each
    carrying at least ``dataset``/``dataset_name``, ``exp_type``/``experiment_type``,
    ``value`` and ``temp``/``temperature``.
    """
    out: list[LabelEntry] = []
    for r in rows:
        dataset = r.get("dataset") or r["dataset_name"]
        raw_type = r.get("exp_type") or r["experiment_type"]
        label = "gene_interaction" if "interaction" in raw_type else "fitness"
        temp = r.get("temp", r.get("temperature"))
        value = r["value"]
        if value is None or not math.isfinite(float(value)):
            continue
        sd = r.get("sd")
        n = r.get("n_samples")
        p = r.get("p", r.get("p_value"))
        out.append(
            LabelEntry(
                source=source_key(str(dataset), None if temp is None else float(temp)),
                label=label,
                value=float(value),
                sd=None if sd is None or not math.isfinite(float(sd)) else float(sd),
                n_samples=None if n is None or not math.isfinite(float(n)) else int(n),
                p_value=None if p is None or not math.isfinite(float(p)) else float(p),
                strain_id=r.get("strain_id"),
            )
        )
    return out
