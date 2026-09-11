# experiments/026-metabolism-flux/scripts/thermo_curation_closure_summary.py
# [[experiments.026-metabolism-flux.scripts.thermo_curation_closure_summary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/thermo_curation_closure_summary.py

r"""What the curation round bought, per record, and what is still open.

Reads the curated records, the coverage ceiling the last
``thermo_pipeline_yeast9.py --curation-dir`` run wrote, and a compact BEFORE snapshot, and
writes ``results/thermo_curation_closure.json`` and ``.md``: one row per curated record,
the ladder before and after, and every reaction still unpriced with its reason.

The before snapshot is a real run rather than a remembered number. ``--make-baseline
<pipeline result>`` reduces a full ``thermo_pipeline_yeast9.json`` to
``results/thermo_curation_baseline.json`` -- per reaction whether it was priced, per
metabolite its reason, and the ladder -- which is small enough to keep next to the code and
is what every "before" figure here is read off.

Run from the repository root under the torchcell env::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/thermo_curation_closure_summary.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.metabolism.thermo_curation import (
    CarrierPotentialValue,
    CuratedThermoRecord,
    CurationKind,
    FormationEnergyValue,
    PkaLadderValue,
    PolymerConventionValue,
    StructureValue,
    load_curated_records,
    structure_for,
)
from torchcell.metabolism.thermo_pipeline import read_result, sha256_of

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
CURATION_DIR = osp.join(RESULTS, "curation")
PIPELINE_JSON = osp.join(RESULTS, "thermo_pipeline_yeast9.json")
COVERAGE_JSON = osp.join(RESULTS, "reaction_thermo_coverage_final.json")
PRICED_JSON = osp.join(RESULTS, "thermo_curation_priced_structures.json")
BASELINE_JSON = osp.join(RESULTS, "thermo_curation_baseline.json")
OUT_JSON = osp.join(RESULTS, "thermo_curation_closure.json")
OUT_MD = osp.join(RESULTS, "thermo_curation_closure.md")


class SlugRow(BaseModel):
    """One curated record, and what it did to the table."""

    slug: str
    kind: str
    admission: str
    admitted: bool
    met_ids: list[str] = Field(default_factory=list)
    blocked_reactions: list[str] = Field(default_factory=list)
    value_summary: str = ""
    sigma_kj_per_mol: float | None = None
    n_sources: int = 0
    first_source_url: str = ""
    first_source_sha256: str = ""
    n_searches: int = 0
    decomposition: str = ""
    #: What the structure route said when it was run on this record's structure.
    structure_route_outcome: str = ""
    n_metabolites_priced: int = 0
    #: Reactions this record's own VALUE closed. A reaction that closed because its
    #: unpriced participants cancel is counted separately: the cancellation rule closed it
    #: and would have closed it with no record at all, and crediting curation with it would
    #: overstate what the round bought.
    n_reactions_closed: int = 0
    reactions_closed: list[str] = Field(default_factory=list)
    n_reactions_closed_by_cancellation: int = 0
    reactions_closed_by_cancellation: list[str] = Field(default_factory=list)
    #: The reason each of this record's metabolites now carries, deduplicated.
    reasons_now: dict[str, int] = Field(default_factory=dict)
    what_it_would_take: str = ""


class OpenReaction(BaseModel):
    """One reaction with no standard energy, and why."""

    reaction_id: str
    reason: str
    non_chemical: bool
    blocking_met_ids: list[str] = Field(default_factory=list)
    blocking_reasons: list[str] = Field(default_factory=list)


class ClosureSummary(BaseModel):
    """The curation round, end to end."""

    built_at: str
    curation_dir: str
    coverage_path: str
    coverage_sha256: str
    baseline_path: str
    baseline_built_at: str
    n_records: int
    n_admitted: int
    n_recorded_not_admitted: int
    #: reactions priced, over ALL reactions, before and after.
    n_reactions: int
    n_reactions_priced_before: int
    n_reactions_priced_after: int
    #: The same over the CHEMICAL reactions, which is the honest denominator and exists
    #: only after this round.
    n_reactions_chemical: int
    n_reactions_chemical_priced: int
    n_reactions_non_chemical: int
    n_reactions_cancelled: int
    n_metabolites_priced_before: int
    n_metabolites_priced_after: int
    ladder_before: list[dict[str, Any]] = Field(default_factory=list)
    ladder_after: list[dict[str, Any]] = Field(default_factory=list)
    rows: list[SlugRow] = Field(default_factory=list)
    n_open_reactions: int = 0
    open_by_reason: dict[str, int] = Field(default_factory=dict)
    open_reactions: list[OpenReaction] = Field(default_factory=list)
    carrier_refusals: dict[str, str] = Field(default_factory=dict)
    #: What the accession composition sweep found, and what the guard withdrew.
    accession_sweep: dict[str, Any] = Field(default_factory=dict)
    #: What the conserved-moiety route did, by outcome.
    moiety_outcomes: dict[str, int] = Field(default_factory=dict)
    moiety_rows: list[dict[str, Any]] = Field(default_factory=list)
    formula_overrides: list[dict[str, Any]] = Field(default_factory=list)
    n_reactions_imbalanced: int = 0
    notes: str = ""


def make_baseline(pipeline_path: str, out_path: str) -> None:
    """Reduce a full pipeline result to the compact BEFORE snapshot."""
    result = read_result(pipeline_path)
    payload = {
        "built_at": result.built_at,
        "source_path": pipeline_path,
        "source_sha256": sha256_of(pipeline_path),
        "n_metabolites_priced": result.metabolite_counts.n_priced,
        "n_reactions": result.reaction_counts.n_total,
        "n_reactions_priced": result.reaction_counts.n_priced,
        "reaction_priced": {
            r.reaction_id: r.delta_r_g0_kj_per_mol is not None for r in result.reactions
        },
        "metabolite_reason": {
            p.met_id: (str(p.reason) if p.reason else "") for p in result.metabolites
        },
        "metabolite_priced": {p.met_id: p.priced for p in result.metabolites},
    }
    with open(out_path, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {out_path}: {payload['n_reactions_priced']}/{payload['n_reactions']}")


def _value_summary(record: CuratedThermoRecord) -> str:
    """One line describing what the record's value actually is."""
    value = record.value
    if isinstance(value, StructureValue):
        if value.per_metabolite:
            named = ", ".join(
                f"{e.met_id}={'a structure' if e.smiles else 'none'}"
                for e in value.per_metabolite
            )
            return f"per metabolite: {named}"
        return f"SMILES {value.smiles or 'none'}, formula {value.formula or 'unstated'}"
    if isinstance(value, PolymerConventionValue):
        price = value.curated_price
        if price is not None:
            values = ", ".join(
                f"{k} {v:.1f}" for k, v in sorted(price.dgf_prime_kj_per_mol.items())
            )
            return f"repeat unit {value.repeat_unit_formula}, curated dGf'0 {values}"
        return (
            f"repeat unit {value.repeat_unit_formula or value.formula}, "
            f"{value.linkage[:40] or 'no linkage stated'}"
        )
    if isinstance(value, CarrierPotentialValue):
        return (
            f"E'0 {value.e_prime_mv} mV vs {value.vs or 'unstated'}, n={value.n_electrons}"
            f", {value.organism or 'organism unstated'}"
        )
    if isinstance(value, PkaLadderValue):
        return f"pKa ladder {value.ladder}"
    if isinstance(value, FormationEnergyValue):
        return (
            f"dGf {value.dgf_kj_per_mol} kJ/mol, {value.standard_or_transformed}, "
            f"pH {value.pH}"
        )
    if record.kind is CurationKind.FORMULA_CORRECTION:
        return f"formula -> {getattr(value, 'proposed_formula', '')}"
    return "nothing found"


def _what_it_would_take(record: CuratedThermoRecord, reasons: dict[str, int]) -> str:
    """The one thing that would close this row, from what the run measured."""
    if "group_in_null_space" in reasons:
        return (
            "a microspecies whose groups component contribution has seen: the structure "
            "decomposes and one of its groups is in the null space"
        )
    if not record.admitted:
        return (
            "a measured or tabulated source; the record is reported and never priced at "
            f"admission={record.admission}"
        )
    if "group_decomposition_failed" in reasons:
        return (
            "a group decomposition, or a measured formation energy; not more retrieval"
        )
    if "no_pka_prediction" in reasons:
        return "a pKa ladder as a record field, so the compound bridge can create the compound"
    if "placeholder_structure" in reasons:
        return "a curated value; the structure is a residue and has no decomposition"
    if "priced_at_reaction_level" in reasons:
        return "nothing; this record prices its reactions directly"
    if not reasons:
        return "nothing; every species this record covers is priced"
    return "; ".join(sorted(reasons))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curation-dir", default=CURATION_DIR)
    parser.add_argument("--pipeline-result", default=PIPELINE_JSON)
    parser.add_argument("--coverage", default=COVERAGE_JSON)
    parser.add_argument("--priced-structures", default=PRICED_JSON)
    parser.add_argument("--baseline", default=BASELINE_JSON)
    parser.add_argument(
        "--make-baseline",
        default=None,
        help="a pipeline result JSON to reduce into the BEFORE snapshot, then exit",
    )
    parser.add_argument("--out-json", default=OUT_JSON)
    parser.add_argument("--out-md", default=OUT_MD)
    args = parser.parse_args()

    if args.make_baseline:
        make_baseline(args.make_baseline, args.baseline)
        return

    records = load_curated_records(args.curation_dir)
    result = read_result(args.pipeline_result)
    with open(args.coverage) as handle:
        coverage = json.load(handle)
    with open(args.baseline) as handle:
        baseline = json.load(handle)
    priced_structures: dict[str, str] = {}
    if osp.isfile(args.priced_structures):
        with open(args.priced_structures) as handle:
            for row in json.load(handle)["records"]:
                priced_structures[row["smiles"]] = row["outcome"]

    price_by_met = {p.met_id: p for p in result.metabolites}
    reaction_by_id = {r.reaction_id: r for r in result.reactions}
    was_priced = baseline["reaction_priced"]

    rows: list[SlugRow] = []
    for record in records.records:
        reasons: dict[str, int] = {}
        n_priced = 0
        for met_id in record.met_ids:
            price = price_by_met.get(met_id)
            if price is None:
                continue
            if price.priced:
                n_priced += 1
            elif price.reason is not None:
                reasons[str(price.reason)] = reasons.get(str(price.reason), 0) + 1
        newly_priced = [
            reaction_id
            for reaction_id in record.blocked_reactions
            if reaction_id in reaction_by_id
            and reaction_by_id[reaction_id].delta_r_g0_kj_per_mol is not None
            and not was_priced.get(reaction_id, False)
        ]
        by_cancellation = [
            reaction_id
            for reaction_id in newly_priced
            if set(record.met_ids) & set(reaction_by_id[reaction_id].cancelled_met_ids)
        ]
        closed = [r for r in newly_priced if r not in by_cancellation]
        outcomes = {
            priced_structures[smiles]
            for smiles in (
                structure_for(record, met_id)[0] for met_id in record.met_ids
            )
            if smiles is not None and smiles in priced_structures
        }
        check = record.decomposition_check
        rows.append(
            SlugRow(
                slug=record.slug,
                kind=str(record.kind),
                admission=str(record.admission),
                admitted=record.admitted,
                met_ids=record.met_ids,
                blocked_reactions=record.blocked_reactions,
                value_summary=_value_summary(record),
                sigma_kj_per_mol=record.sigma_kj_per_mol,
                n_sources=len(record.sources),
                first_source_url=record.sources[0].source_url if record.sources else "",
                first_source_sha256=record.sources[0].sha256 if record.sources else "",
                n_searches=len(record.searches),
                decomposition=(
                    "not attempted"
                    if check is None or not check.attempted
                    else (
                        "decomposes"
                        if check.decomposes
                        else ("mixed" if check.decomposes is None else "refused")
                    )
                ),
                structure_route_outcome=", ".join(sorted(outcomes)),
                n_metabolites_priced=n_priced,
                n_reactions_closed=len(closed),
                reactions_closed=closed,
                n_reactions_closed_by_cancellation=len(by_cancellation),
                reactions_closed_by_cancellation=by_cancellation,
                reasons_now=dict(sorted(reasons.items())),
                what_it_would_take=_what_it_would_take(record, reasons),
            )
        )
    rows.sort(key=lambda r: (-r.n_reactions_closed, r.slug))

    open_reactions = [
        OpenReaction(
            reaction_id=r.reaction_id,
            reason=str(r.reason) if r.reason else "",
            non_chemical=r.non_chemical,
            blocking_met_ids=r.blocking_met_ids,
            blocking_reasons=[str(x) for x in r.blocking_reasons],
        )
        for r in result.reactions
        if r.delta_r_g0_kj_per_mol is None
    ]
    open_by_reason: dict[str, int] = {}
    for entry in open_reactions:
        open_by_reason[entry.reason] = open_by_reason.get(entry.reason, 0) + 1

    curation = coverage.get("curation", {})
    sweep: dict[str, Any] = {}
    sweep_path = osp.join(RESULTS, "thermo_accession_composition.json")
    if osp.isfile(sweep_path):
        with open(sweep_path) as handle:
            payload = json.load(handle)
        sweep = {
            "n_checked": payload["n_checked"],
            "n_match": payload["n_match"],
            "n_repeat_unit_multiple": payload["n_repeat_unit_multiple"],
            "n_other": payload["n_other"],
            "repeat_unit": payload["repeat_unit"],
            "repeat_unit_offsets": payload["repeat_unit_offsets"],
            "family_consistent": payload["family_consistent"],
            "other": [
                {
                    "met_id": row["met_id"],
                    "name": row["name"],
                    "gem_formula": row["gem_formula"],
                    "structure_formula": row["structure_formula"],
                    "shipped_dgf_prime_kj_per_mol": row["dgf_prime_kj_per_mol"],
                    "detail": row["detail"],
                }
                for row in payload["rows"]
                if row["classification"] == "other"
            ],
            "dimer_differences": payload["dimer_differences"],
        }
    moiety_outcomes: dict[str, int] = {}
    moiety_rows: list[dict[str, Any]] = []
    moiety_path = osp.join(RESULTS, "thermo_moiety_priced.json")
    if osp.isfile(moiety_path):
        with open(moiety_path) as handle:
            payload = json.load(handle)
        moiety_outcomes = payload["outcomes"]
        moiety_rows = [
            {
                "reaction_id": row["reaction_id"],
                "outcome": row["outcome"],
                "delta_r_g0_kj_per_mol": row["delta_r_g0_kj_per_mol"],
                "sigma_kj_per_mol": row["sigma_kj_per_mol"],
                "detail": row["detail"][:180],
            }
            for row in payload["records"]
        ]
    summary = ClosureSummary(
        built_at=datetime.now(UTC).isoformat(),
        curation_dir=args.curation_dir,
        coverage_path=args.coverage,
        coverage_sha256=sha256_of(args.coverage),
        baseline_path=args.baseline,
        baseline_built_at=baseline["built_at"],
        n_records=len(records.records),
        n_admitted=sum(1 for r in records.records if r.admitted),
        n_recorded_not_admitted=sum(1 for r in records.records if not r.admitted),
        n_reactions=result.reaction_counts.n_total,
        n_reactions_priced_before=baseline["n_reactions_priced"],
        n_reactions_priced_after=result.reaction_counts.n_priced,
        n_reactions_chemical=result.reaction_counts_chemical.n_total,
        n_reactions_chemical_priced=result.reaction_counts_chemical.n_priced,
        n_reactions_non_chemical=result.n_non_chemical_reactions,
        n_reactions_cancelled=sum(1 for r in result.reactions if r.cancelled_met_ids),
        n_metabolites_priced_before=baseline["n_metabolites_priced"],
        n_metabolites_priced_after=result.metabolite_counts.n_priced,
        ladder_before=[],
        ladder_after=coverage["ceiling_ladder"]["rungs"],
        rows=rows,
        n_open_reactions=len(open_reactions),
        open_by_reason=dict(sorted(open_by_reason.items(), key=lambda kv: -kv[1])),
        open_reactions=open_reactions,
        carrier_refusals=cast(dict[str, str], curation.get("carrier_refusals", {})),
        accession_sweep=sweep,
        moiety_outcomes=moiety_outcomes,
        moiety_rows=moiety_rows,
        formula_overrides=cast(
            list[dict[str, Any]], curation.get("formula_overrides", [])
        ),
        n_reactions_imbalanced=coverage.get("n_reactions_imbalanced", 0),
        notes=(
            "Before and after are two runs of thermo_pipeline_yeast9.py, the first without "
            "--curation-dir and under the pricing rules that predate this round. The "
            "chemical denominator exists only in the after run, so the two whole-model "
            "fractions are the comparable pair and the chemical fraction is the honest one."
        ),
    )
    with open(args.out_json, "w") as handle:
        handle.write(summary.model_dump_json(indent=1))
    print(f"wrote {args.out_json}")
    _write_md(summary, args.out_md)


def _write_md(summary: ClosureSummary, path: str) -> None:
    """The table that goes into the note, generated rather than transcribed."""
    lines = [
        "# Thermodynamic curation closure",
        "",
        f"Built {summary.built_at} by "
        "`experiments/026-metabolism-flux/scripts/thermo_curation_closure_summary.py`.",
        "",
        "## Coverage before and after",
        "",
        "| quantity | before | after |",
        "|---|---:|---:|",
        (
            f"| reactions priced, all {summary.n_reactions} | "
            f"{summary.n_reactions_priced_before} | {summary.n_reactions_priced_after} |"
        ),
        (
            "| fraction, all reactions | "
            f"{summary.n_reactions_priced_before / summary.n_reactions:.4f} | "
            f"{summary.n_reactions_priced_after / summary.n_reactions:.4f} |"
        ),
        (
            f"| reactions priced, chemical {summary.n_reactions_chemical} | not reported | "
            f"{summary.n_reactions_chemical_priced} |"
        ),
        (
            "| fraction, chemical only | not reported | "
            f"{summary.n_reactions_chemical_priced / summary.n_reactions_chemical:.4f} |"
        ),
        f"| metabolites priced | {summary.n_metabolites_priced_before} | "
        f"{summary.n_metabolites_priced_after} |",
        f"| non-chemical reactions, outside the denominator | 0 | "
        f"{summary.n_reactions_non_chemical} |",
        f"| reactions priced through cancellation | 0 | {summary.n_reactions_cancelled} |",
        "",
        "## Ceiling ladder after, over the chemical reactions",
        "",
        "| rung | reactions priced | fraction | flux carrying | fraction |",
        "|---|---:|---:|---:|---:|",
    ]
    for rung in summary.ladder_after:
        lines.append(
            f"| {rung['name'].replace('_', ' ')} | {rung['n_reactions_priced']} | "
            f"{rung['fraction_reactions_priced']:.4f} | "
            f"{rung['n_flux_carrying_priced']} | "
            f"{rung['fraction_flux_carrying_priced']:.4f} |"
        )
    lines += [
        "",
        "## One row per curated record",
        "",
        "| slug | kind | admission | species | decomposition | priced | closed by the "
        "record | closed by cancellation | what it would take |",
        "|---|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in summary.rows:
        lines.append(
            f"| {row.slug} | {row.kind} | {row.admission} | {len(row.met_ids)} | "
            f"{row.decomposition} | {row.n_metabolites_priced} | "
            f"{row.n_reactions_closed} | {row.n_reactions_closed_by_cancellation} | "
            f"{row.what_it_would_take} |"
        )
    lines += [
        "",
        "## What each record holds",
        "",
        "| slug | value | sources | source |",
        "|---|---|---:|---|",
    ]
    for row in summary.rows:
        url = row.first_source_url or "none"
        lines.append(f"| {row.slug} | {row.value_summary} | {row.n_sources} | {url} |")
    lines += [
        "",
        "## Reactions still open",
        "",
        f"{summary.n_open_reactions} reactions carry no standard energy. The full list, "
        "with the metabolites blocking each, is in `thermo_curation_closure.json`.",
        "",
        "| reason | reactions |",
        "|---|---:|",
    ]
    for reason, count in summary.open_by_reason.items():
        lines.append(f"| {reason} | {count} |")
    if summary.accession_sweep:
        sweep = summary.accession_sweep
        lines += [
            "",
            "## The accession route had no composition check",
            "",
            f"Of {sweep['n_checked']} accession-priced metabolites, {sweep['n_match']} "
            f"match the GEM's formula, {sweep['n_repeat_unit_multiple']} differ by whole "
            f"{sweep['repeat_unit']} repeat units (offsets {sweep['repeat_unit_offsets']}, "
            f"family consistent {sweep['family_consistent']}), and {sweep['n_other']} were "
            "priced from a different molecule.",
            "",
            "| metabolite | name | GEM formula | structure formula | shipped dGf'0 |",
            "|---|---|---|---|---:|",
        ]
        for row in sweep["other"]:
            value = row["shipped_dgf_prime_kj_per_mol"]
            lines.append(
                f"| {row['met_id']} | {row['name']} | {row['gem_formula']} | "
                f"{row['structure_formula']} | "
                f"{'' if value is None else format(value, '.1f')} |"
            )
    if summary.moiety_rows:
        lines += [
            "",
            "## The conserved-moiety route",
            "",
            "| reaction | outcome | dGr'0 | sigma | why |",
            "|---|---|---:|---:|---|",
        ]
        for row in summary.moiety_rows:
            value = row["delta_r_g0_kj_per_mol"]
            sigma = row["sigma_kj_per_mol"]
            lines.append(
                f"| {row['reaction_id']} | {row['outcome']} | "
                f"{'' if value is None else format(value, '.2f')} | "
                f"{'' if sigma is None else format(sigma, '.2f')} | {row['detail']} |"
            )
    if summary.carrier_refusals:
        lines += [
            "",
            "## Carrier reactions the potential route refused",
            "",
            "| reaction | why |",
            "|---|---|",
        ]
        for reaction_id, why in sorted(summary.carrier_refusals.items()):
            lines.append(f"| {reaction_id} | {why} |")
    lines += [
        "",
        f"Elementally or charge imbalanced chemical reactions, diagnostic only: "
        f"{summary.n_reactions_imbalanced}.",
        "",
    ]
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
