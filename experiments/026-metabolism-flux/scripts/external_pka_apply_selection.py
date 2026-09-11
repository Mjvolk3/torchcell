# experiments/026-metabolism-flux/scripts/external_pka_apply_selection.py
# [[experiments.026-metabolism-flux.scripts.external_pka_apply_selection]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/external_pka_apply_selection.py

r"""Re-price the external-pKa metabolites with the selected method and the pinned convention.

Two things were wrong with the values the external-pKa route shipped on 2026-09-10, and
they are independent.

**1. The protonation convention.** eQuilibrator derives a compound's reference microspecies
from the SMILES it is handed, so submitting the un-ionized neutral parent anchors the
formation energy to a different microstate than the cache's own entry, which is ChemAxon's
major microspecies at pH 7. Measured in ``external_pka_reprice.json``: submitting the pH 7
microspecies instead moves the residual against the cache from a mean of -777.5 kJ/mol to
+0.128 kJ/mol over the 7 targets priced both ways. The 48 lipids priced through this route
still carry the old convention.

**2. The pKa method.** ``external_pka_method_selection.py`` chose one on 2,400 cache
compounds at every working compartment pH. Whatever it chose is read here from
``pka_method_selection.json``; no method is named in this file.

The two are applied together, and every re-priced compound carries its own provenance
record: the method and its version, the sha256 of the weight files where the method is a
local checkpoint, the conditions, the pKa ladder, the submitted SMILES, and the resulting
``dGf'0`` with its sigma.

WHAT IS RE-PRICED
-----------------
* the 48 lipid records in ``lipid_priced.json`` whose route is ``external_pka`` (58
  yeast-GEM metabolite rows); the cache-lookup lipids are NOT touched, since a cache
  compound is already on the cache's convention and carries ChemAxon pKas and magnesium
  constants that no external route can reproduce;
* the tier-a targets in ``external_pka_pilot.json``, the compounds whose structure resolves
  and whose only missing input was a pKa list.

Subcommands, each under the environment that holds its dependencies.

``targets``     assemble the target list with structures. Any env with pydantic.
``ladders``     pKa ladders from the selected method's members. pka-tools (MolGpKa) and
                pkasolver-env (pKasolver-lite); run once per member.
``microspecies`` the pH 7 major microspecies per target. pkasolver-env: the state comes
                from ``States.ph7_mol``, which dimorphite-dl enumerates, so the CONVENTION
                and the pKa VALUES come from different tools on purpose.
``combine``     apply the selection rule to the members' ladders. Any env.
``price``       create the compounds and price them. equilibrator-ext.

Writes ``results/external_pka_applied.json``, which
``thermo_pipeline_yeast9.py`` picks up as its ``external_pka_targets`` stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from pydantic import BaseModel, Field

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
DATA_ROOT = os.environ["DATA_ROOT"]
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
TARGETS_PATH = osp.join(RESULTS, "external_pka_apply_targets.json")
LADDER_PATH = osp.join(RESULTS, "external_pka_apply_ladders_{tool}.json")
MICROSPECIES_PATH = osp.join(RESULTS, "external_pka_apply_microspecies.json")
COMBINED_PATH = osp.join(RESULTS, "external_pka_apply_combined.json")
APPLIED_PATH = osp.join(RESULTS, "external_pka_applied.json")
DECISION_PATH = osp.join(RESULTS, "pka_method_selection.json")

P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0
MIN_PH = 0.0
MAX_PH = 14.0
MAX_PKAS = 6

#: The MolGpKa checkout, mirrored out of the throwaway agent directory it was cloned into.
MOLGPKA_REPO = osp.join(DATA_ROOT, "pka", "MolGpKa")

#: Ionizable centers, as SMARTS. A molecule matching none of these has no proton to lose or
#: gain inside (0, 14), so an empty pKa ladder is the right answer for it rather than a
#: failure. Used to tell those two apart WITHOUT reading a tool's error string, since each
#: tool words "there is nothing here" differently and one of them words it as an error.
IONIZABLE_CENTER_SMARTS: tuple[str, ...] = (
    "[CX3](=O)[OX2H1,OX1-]",                      # carboxylic acid or carboxylate
    "[PX4](=O)([OX2H1,OX1-])",                    # phosphate
    "[SX4](=O)(=O)[OX2H1,OX1-]",                  # sulfonate
    "[OX2H1,OX1-]c",                              # phenol or phenolate
    "[NX3;!$(N[C,S]=[O,S,N]);!$(N=*);!$([N+](=O)[O-])]",  # amine
    "[NX4+]",                                     # quaternary or protonated amine
    "[SX2H]",                                     # thiol
    "[nX3H]",                                     # protonated aromatic nitrogen
    "[NX3][CX3](=[NX2])[NX3]",                    # guanidine
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


class ApplyTarget(BaseModel):
    """One compound to re-price, and where its structure came from."""

    key: str
    name: str
    source: str = Field(description="'lipid_external_pka' or 'tier_a_pilot'.")
    met_ids: list[str]
    smiles: str
    gem_formula: str = ""
    previous_dgf_prime_kj_per_mol: float | None = None
    previous_route: str = ""
    previous_convention: str = "neutral parent submitted to the compound bridge"
    note: str = ""


class ApplyTargetSet(BaseModel):
    """The whole re-pricing target list and where each entry came from."""

    built_at: str
    n_targets: int
    n_metabolite_rows: int
    sources: dict[str, int]
    targets: list[ApplyTarget]


def cmd_targets(args: argparse.Namespace) -> None:
    """Assemble the re-pricing target list from the two artifacts that hold structures."""
    targets: list[ApplyTarget] = []

    lipid_path = osp.join(RESULTS, "lipid_priced.json")
    with open(lipid_path) as handle:
        lipids = json.load(handle)
    for record in lipids["records"]:
        if record.get("route") != "external_pka":
            continue
        if not record.get("smiles"):
            continue
        targets.append(
            ApplyTarget(
                key=f"lipid:{record['name']}",
                name=record["name"],
                source="lipid_external_pka",
                met_ids=list(record.get("met_ids", [])),
                smiles=record["smiles"],
                gem_formula=record.get("gem_formula", ""),
                previous_dgf_prime_kj_per_mol=record.get("dgf_prime_kj_per_mol"),
                previous_route="external_pka",
                note=(
                    "written structure, priced through the bridge on the neutral parent "
                    "with MolGpKa pKas"
                ),
            )
        )

    pilot_path = osp.join(RESULTS, "external_pka_pilot.json")
    with open(pilot_path) as handle:
        pilot = json.load(handle)
    for record in pilot["targets"]:
        if record.get("route") != "external_pka" or not record.get("smiles"):
            continue
        targets.append(
            ApplyTarget(
                key=f"tier_a:{record['name']}",
                name=record["name"],
                source="tier_a_pilot",
                met_ids=[],
                smiles=record["smiles"],
                previous_dgf_prime_kj_per_mol=record.get("dgf_prime_kj_per_mol"),
                previous_route=str(record.get("outcome")),
                note=str(record.get("detail") or ""),
            )
        )

    # Structures composed from a REACTION rather than from a name, by
    # `thermo_acyl_transfer_writer.py`. Only the ones that passed its formula acceptance
    # test are taken, so a composed molecule that is not the GEM's molecule never reaches
    # the bridge.
    acyl_path = osp.join(RESULTS, "thermo_acyl_transfer.json")
    if osp.isfile(acyl_path):
        with open(acyl_path) as handle:
            acyl = json.load(handle)
        for record in acyl["records"]:
            if not record.get("formula_match") or not record.get("product_smiles"):
                continue
            targets.append(
                ApplyTarget(
                    key=f"acyl:{record['name']}",
                    name=record["name"],
                    source="acyl_transfer_writer",
                    met_ids=list(record.get("met_ids", [])),
                    smiles=record["product_smiles"],
                    gem_formula=record.get("gem_formula", ""),
                    previous_route="none",
                    note=(
                        f"composed from {record['reaction_id']} "
                        f"({record['reaction_name']}) by acylating "
                        f"{record['acceptor_name']}"
                    ),
                )
            )

    # The tier-a records carry a NAME, not the metabolite ids, so the ids are joined back
    # from the audit table. Without this the re-priced values would have nothing to attach
    # to and the stage would price zero metabolites while reporting success.
    import csv as csv_module

    audit_path = osp.join(RESULTS, "unresolved_metabolites.csv")
    by_name: dict[str, list[str]] = {}
    with open(audit_path) as handle:
        for row in csv_module.DictReader(handle):
            by_name.setdefault(row["name"], []).append(row["met_id"])
    for target in targets:
        if not target.met_ids:
            target.met_ids = sorted(by_name.get(target.name, []))

    payload = ApplyTargetSet(
        built_at=_timestamp(),
        n_targets=len(targets),
        n_metabolite_rows=sum(len(t.met_ids) for t in targets),
        sources={
            source: sum(1 for t in targets if t.source == source)
            for source in sorted({t.source for t in targets})
        },
        targets=targets,
    )
    with open(TARGETS_PATH, "w") as handle:
        handle.write(payload.model_dump_json(indent=1))
    print(
        f"wrote {TARGETS_PATH}: {payload.n_targets} targets over"
        f" {payload.n_metabolite_rows} metabolite rows, {payload.sources}"
    )
    for target in targets:
        if not target.met_ids:
            print(f"  WARNING no metabolite id joined for {target.key}")


def _load_targets() -> ApplyTargetSet:
    with open(TARGETS_PATH) as handle:
        return ApplyTargetSet.model_validate_json(handle.read())


class LadderRecord(BaseModel):
    """One tool's macroscopic pKa ladder for one target."""

    key: str
    tool: str
    ok: bool
    pkas: list[float] = Field(default_factory=list)
    site_pkas: list[float] = Field(default_factory=list)
    error: str | None = None


class LadderSet(BaseModel):
    """Every ladder one tool produced, with its version and weight hashes."""

    built_at: str
    tool: str
    tool_version: str
    weights_sha256: dict[str, str] = Field(default_factory=dict)
    records: list[LadderRecord]


def cmd_ladders(args: argparse.Namespace) -> None:
    """Run one member tool over the targets and write its macroscopic ladders.

    The aggregation is imported from :mod:`external_pka_benchmark` rather than restated, so
    the ladders here are built exactly the way the benchmark that chose the method built
    them. A method whose ladders were aggregated one way for scoring and another way for
    use would not be the method that was chosen.
    """
    from external_pka_benchmark import _apply_proton_change, _macro_from_sites

    targets = _load_targets()
    records: list[LadderRecord] = []
    weights: dict[str, str] = {}

    if args.tool == "molgpka":
        sys.path.insert(0, osp.join(MOLGPKA_REPO, "src"))
        os.chdir(osp.join(MOLGPKA_REPO, "src"))
        import torch

        _orig_load = torch.load

        def _load(*a, **kw):
            kw["weights_only"] = False
            return _orig_load(*a, **kw)

        torch.load = _load
        import predict_pka  # type: ignore[import-not-found]

        _load_model = predict_pka.load_model
        _cache: dict[tuple[str, str], object] = {}

        def _cached(model_file: str, device: str = "cpu"):
            key = (model_file, device)
            if key not in _cache:
                _cache[key] = _load_model(model_file, device)
            return _cache[key]

        predict_pka.load_model = _cached
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        for name in ("weight_acid.pth", "weight_base.pth"):
            path = osp.join(MOLGPKA_REPO, "models", name)
            if osp.isfile(path):
                weights[name] = _sha256(path)

        for target in targets.targets:
            ladder: list[float] = []
            sites: list[float] = []
            error: str | None = None
            try:
                for acidic in (True, False):
                    state = target.smiles
                    for _ in range(MAX_PKAS + 1):
                        mol = Chem.MolFromSmiles(state)
                        if mol is None:
                            break
                        mol_h = Chem.AddHs(mol)
                        found = (
                            predict_pka.predict_acid(mol_h)
                            if acidic
                            else predict_pka.predict_base(mol_h)
                        )
                        if not found:
                            break
                        values = [float(v) for v in found.values()]
                        sites.extend(values)
                        macro = _macro_from_sites(values, acidic)
                        if not MIN_PH < macro < MAX_PH:
                            break
                        ladder.append(macro)
                        dominant = (
                            min(found, key=lambda k: float(found[k]))
                            if acidic
                            else max(found, key=lambda k: float(found[k]))
                        )
                        if acidic:
                            atom = mol_h.GetAtomWithIdx(int(dominant))
                            neighbors = atom.GetNeighbors()
                            if not neighbors:
                                break
                            heavy = neighbors[0].GetIdx()
                        else:
                            heavy = int(dominant)
                        if heavy >= mol.GetNumAtoms():
                            break
                        nxt = _apply_proton_change(state, heavy, -1 if acidic else 1)
                        if nxt is None:
                            break
                        state = nxt
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:300]
            records.append(
                LadderRecord(
                    key=target.key,
                    tool="molgpka",
                    ok=error is None and bool(ladder),
                    pkas=sorted(ladder, reverse=True),
                    site_pkas=sites,
                    error=error,
                )
            )
            print(f"  {target.key}: {len(ladder)} pKas", flush=True)
    elif args.tool == "pkasolver":
        from pkasolver.query import QueryModel, calculate_microstate_pka_values
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        model = QueryModel()
        for target in targets.targets:
            try:
                mol = Chem.MolFromSmiles(target.smiles)
                states = calculate_microstate_pka_values(mol, query_model=model)
                pkas = sorted(
                    (
                        float(s.pka)
                        for s in states
                        if MIN_PH < float(s.pka) < MAX_PH
                    ),
                    reverse=True,
                )
                records.append(
                    LadderRecord(
                        key=target.key,
                        tool="pkasolver-lite",
                        ok=bool(pkas),
                        pkas=pkas,
                        site_pkas=[float(s.pka) for s in states],
                    )
                )
            except Exception as exc:
                records.append(
                    LadderRecord(
                        key=target.key,
                        tool="pkasolver-lite",
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}"[:300],
                    )
                )
            print(f"  {target.key}: {len(records[-1].pkas)} pKas", flush=True)
    else:
        raise ValueError(f"no ladder runner for {args.tool!r}")

    payload = LadderSet(
        built_at=_timestamp(),
        tool=records[0].tool if records else args.tool,
        tool_version=args.tool_version,
        weights_sha256=weights,
        records=records,
    )
    path = LADDER_PATH.format(tool=args.tool)
    with open(path, "w") as handle:
        handle.write(payload.model_dump_json(indent=1))
    print(f"wrote {path}: {sum(1 for r in records if r.ok)}/{len(records)} ladders")


#: Where the Uni-pKa adapter stages its two files. The runner is
#: ``external_pka_benchmark.py predict-unipka``, which reads a ValidationSet and writes a
#: PredictionSet, so the adapter is a shape conversion in and a key join out.
UNIPKA_INPUT_NAME = "pka_apply_unipka_input.json"
UNIPKA_PRED_TEMPLATE = "pka_apply_unipka_{tool}.json"


def cmd_unipka_input(args: argparse.Namespace) -> None:
    """Write the targets in the shape ``external_pka_benchmark predict-unipka`` reads.

    Uni-pKa is not a site predictor and cannot be walked from here the way MolGpKa is: it
    consumes a protonation ENSEMBLE built by its own enumerator and produces the macroscopic
    constant itself. Rather than restate that pipeline, the targets are written as a
    ValidationSet and the benchmark's own runner is pointed at them through
    ``EXTERNAL_PKA_SAMPLE``. The compound id is the target's index in the target file, which
    is what ``unipka-ladders`` joins back on.

    The ``cache_pkas`` field is required by the schema and is never read by the runner; it
    is written as an empty list so nothing downstream can mistake it for ground truth.
    """
    from external_pka_benchmark import ValidationCompound, ValidationSet

    targets = _load_targets()
    compounds = [
        ValidationCompound(
            compound_id=index,
            inchi_key="",
            major_ms_smiles=target.smiles,
            neutral_smiles=target.smiles,
            formula_atom_bag={},
            cache_pkas=[],
            n_cache_pkas=0,
            groups=[],
            n_heavy_atoms=0,
            has_magnesium_constants=False,
            stratum="apply_target",
        )
        for index, target in enumerate(targets.targets)
    ]
    payload = ValidationSet(
        built_at=_timestamp(),
        cache_path="not a cache sample: the re-pricing targets in ValidationSet shape",
        n_compounds=len(compounds),
        seed=0,
        n_candidates_scanned=0,
        n_candidates_eligible=len(compounds),
        stratum_counts={"apply_target": len(compounds)},
        group_counts={},
        n_with_magnesium_constants=0,
        compounds=compounds,
    )
    path = osp.join(RESULTS, UNIPKA_INPUT_NAME)
    with open(path, "w") as handle:
        handle.write(payload.model_dump_json(indent=1))
    print(f"wrote {path}: {len(compounds)} targets")
    print(
        "now run, under the unipka env:\n"
        f"  EXTERNAL_PKA_SAMPLE={UNIPKA_INPUT_NAME} \\\n"
        f"  EXTERNAL_PKA_PRED_TEMPLATE={UNIPKA_PRED_TEMPLATE} \\\n"
        "  $DATA_ROOT/envs/unipka/bin/python "
        "experiments/026-metabolism-flux/scripts/external_pka_benchmark.py predict-unipka "
        "--enumerator $DATA_ROOT/unipka/Uni-pKa/enumerator "
        "--model-dir $DATA_ROOT/unipka/models --work-dir <a fresh dir> "
        "--tool-name unipka-simple-template --template simple_smarts_pattern.tsv "
        "--folds 0 1 2 3 4 --charge-upper 2 --charge-lower -4 --batch-size 8 --threads 8"
    )


def cmd_unipka_ladders(args: argparse.Namespace) -> None:
    """Join the benchmark's Uni-pKa output back onto the target keys."""
    from external_pka_benchmark import PredictionSet

    targets = _load_targets()
    path = osp.join(RESULTS, UNIPKA_PRED_TEMPLATE.format(tool=args.tool_name))
    with open(path) as handle:
        pset = PredictionSet.model_validate_json(handle.read())
    by_index = {r.compound_id: r for r in pset.records}
    records = []
    for index, target in enumerate(targets.targets):
        record = by_index.get(index)
        records.append(
            LadderRecord(
                key=target.key,
                tool=pset.tool,
                ok=bool(record and record.ok and record.predicted_pkas),
                pkas=sorted(record.predicted_pkas, reverse=True) if record else [],
                site_pkas=list(record.site_pkas) if record else [],
                error=(record.error if record else "no record for this target index"),
            )
        )
    payload = LadderSet(
        built_at=_timestamp(),
        tool=pset.tool,
        tool_version=pset.tool_version,
        weights_sha256={},
        records=records,
    )
    out = LADDER_PATH.format(tool=args.tool_name)
    with open(out, "w") as handle:
        handle.write(payload.model_dump_json(indent=1))
    print(f"wrote {out}: {sum(1 for r in records if r.ok)}/{len(records)} ladders")


class MicrospeciesRecord(BaseModel):
    """One target's pH 7 major microspecies, the convention the cache is built in."""

    key: str
    input_smiles: str
    input_charge: int | None = None
    ph7_smiles: str | None = None
    ph7_charge: int | None = None
    error: str | None = None


def cmd_microspecies(args: argparse.Namespace) -> None:
    """The pH 7 major microspecies per target, the convention the cache is built in.

    The state comes from ``States.ph7_mol``, which pKasolver derives with dimorphite-dl.
    That is a protonation-state enumerator, not a pKa regressor, so using it for the
    CONVENTION while the selected method supplies the pKa VALUES is not mixing two
    predictions: it is the same split ``external_pka_benchmark.reprice-microspecies``
    measured the convention shift with.
    """
    from pkasolver.query import QueryModel, calculate_microstate_pka_values
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    targets = _load_targets()
    model = QueryModel()
    records: list[MicrospeciesRecord] = []
    for target in targets.targets:
        row = MicrospeciesRecord(key=target.key, input_smiles=target.smiles)
        mol = Chem.MolFromSmiles(target.smiles)
        if mol is None:
            row.error = "rdkit could not parse the SMILES"
            records.append(row)
            continue
        row.input_charge = Chem.GetFormalCharge(mol)
        try:
            states = calculate_microstate_pka_values(mol, query_model=model)
        except Exception as exc:
            row.error = f"{type(exc).__name__}: {exc}"[:300]
            records.append(row)
            continue
        if not states or states[0].ph7_mol is None:
            row.error = "pkasolver returned no pH 7 state"
            records.append(row)
            continue
        row.ph7_smiles = Chem.MolToSmiles(states[0].ph7_mol)
        row.ph7_charge = Chem.GetFormalCharge(states[0].ph7_mol)
        records.append(row)
        print(f"  {target.key}: {row.input_charge} -> {row.ph7_charge}", flush=True)

    payload = {
        "built_at": _timestamp(),
        "tool": "pkasolver-lite ph7_mol (dimorphite-dl)",
        "records": [r.model_dump() for r in records],
    }
    with open(MICROSPECIES_PATH, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(
        f"wrote {MICROSPECIES_PATH}: "
        f"{sum(1 for r in records if r.ph7_smiles)}/{len(records)} states"
    )


class CombinedLadder(BaseModel):
    """The ladder the selected rule produces for one target, and which members fed it."""
    key: str
    method: str
    members_used: list[str]
    pkas: list[float]
    ok: bool
    reason: str = ""


def cmd_combine(args: argparse.Namespace) -> None:
    """Apply the selected rule to the member ladders, exactly as the benchmark scored it."""
    import numpy as np

    with open(DECISION_PATH) as handle:
        decision = json.load(handle)
    method = decision["selected"]
    members = decision["selected_members"] or [method]
    # The benchmark's tool keys are the prediction FILE keys; the ladder files here use the
    # same keys, so a member the decision names and this run did not produce is an error
    # rather than a quietly shorter ensemble.
    ladders: dict[str, dict[str, list[float]]] = {}
    for member in members:
        key = "pkasolver" if member.startswith("pkasolver") else member
        path = LADDER_PATH.format(tool=key)
        if not osp.isfile(path):
            raise FileNotFoundError(
                f"the selected method {method!r} needs member {member!r} and {path} is "
                "absent; run `ladders --tool` for it"
            )
        with open(path) as handle:
            payload = LadderSet.model_validate_json(handle.read())
        ladders[member] = {r.key: r.pkas for r in payload.records if r.ok and r.pkas}

    targets = _load_targets()
    out: list[CombinedLadder] = []
    for target in targets.targets:
        used: list[str] = []
        pkas: list[float] = []
        if method.startswith("cascade_"):
            for member in members:
                candidate = ladders[member].get(target.key)
                if candidate:
                    pkas, used = list(candidate), [member]
                    break
        elif method.startswith("ensemble_"):
            present = [ladders[m].get(target.key, []) for m in members]
            used = [m for m, p in zip(members, present) if p]
            need = len(members) / 2.0
            depth = max((len(p) for p in present), default=0)
            for rank in range(depth):
                values = [p[rank] for p in present if len(p) > rank]
                if len(values) > need:
                    pkas.append(float(np.median(values)))
            pkas.sort(reverse=True)
        elif method == "per_group_best":
            # This rule needs each target's functional group, and the target file does not
            # carry one. Applying the first member to everything would silently be a
            # DIFFERENT method from the one that was selected, so it raises.
            raise NotImplementedError(
                "the selected method is per_group_best, which needs a functional-group "
                "label per target; add one to the target file before applying it"
            )
        else:
            candidate = ladders.get(method, {}).get(target.key) or ladders.get(
                members[0], {}
            ).get(target.key)
            if candidate:
                pkas, used = list(candidate), [members[0]]
        out.append(
            CombinedLadder(
                key=target.key,
                method=method,
                members_used=used,
                pkas=pkas,
                ok=bool(pkas),
                reason="" if pkas else "no member produced a ladder in (0, 14)",
            )
        )

    payload = {
        "built_at": _timestamp(),
        "method": method,
        "members": members,
        "decision_path": DECISION_PATH,
        "decision_sha256": _sha256(DECISION_PATH),
        "n_ok": sum(1 for r in out if r.ok),
        "records": [r.model_dump() for r in out],
    }
    with open(COMBINED_PATH, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {COMBINED_PATH}: {payload['n_ok']}/{len(out)} ladders from {method}")


class AppliedRecord(BaseModel):
    """One re-priced compound, with everything needed to reproduce the number."""

    key: str
    name: str
    source: str
    met_ids: list[str]
    method: str
    method_version: str
    weights_sha256: dict[str, str] = Field(default_factory=dict)
    members_used: list[str] = Field(default_factory=list)
    smiles: str
    ph7_smiles: str | None = None
    submitted_smiles: str | None = None
    convention: str = "pH 7 major microspecies submitted to the compound bridge"
    pkas: list[float] = Field(default_factory=list)
    conditions: dict[str, float] = Field(default_factory=dict)
    compound_id: int | None = None
    group_decomposition_ok: bool | None = None
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    previous_dgf_prime_kj_per_mol: float | None = None
    change_kj_per_mol: float | None = None
    outcome: str = ""
    detail: str = ""


def _ionizable_centers(smiles: str) -> list[str]:
    """Which ionizable-center patterns a structure matches, by SMARTS.

    An empty result is a positive statement: the molecule has no proton to lose or gain in
    the window, so an empty pKa ladder is its answer and not a tool failure.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ["unparseable"]
    found = []
    for smarts in IONIZABLE_CENTER_SMARTS:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            found.append(smarts)
    return found


def cmd_price(args: argparse.Namespace) -> None:
    """Create each compound from its pH 7 microspecies and the selected ladder, and price it."""
    import shutil

    from equilibrator_api import Q_, ComponentContribution
    from equilibrator_assets.local_compound_cache import LocalCompoundCache

    targets = {t.key: t for t in _load_targets().targets}
    with open(COMBINED_PATH) as handle:
        combined = json.load(handle)
    with open(MICROSPECIES_PATH) as handle:
        micro = {r["key"]: r for r in json.load(handle)["records"]}
    ladder_versions: dict[str, dict] = {}
    for member in combined["members"]:
        key = "pkasolver" if member.startswith("pkasolver") else member
        path = LADDER_PATH.format(tool=key)
        with open(path) as handle:
            payload = json.load(handle)
        ladder_versions[member] = payload

    if args.seed_cache and not osp.isfile(args.local_cache):
        os.makedirs(osp.dirname(args.local_cache), exist_ok=True)
        shutil.copy2(args.seed_cache, args.local_cache)
        print(f"seeded {args.local_cache} from {args.seed_cache}")

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)
    local = LocalCompoundCache(pkas_provided_externally=True)
    local.load_cache(args.local_cache)
    # A second handle on the SAME cache file without the external-pKa guard, used only for
    # molecules with no ionizable center; ChemAxon stays bypassed on that path too.
    plain = LocalCompoundCache()
    plain.load_cache(args.local_cache)

    records: list[AppliedRecord] = []
    for row in combined["records"]:
        target = targets[row["key"]]
        state = micro.get(row["key"], {})
        submitted = state.get("ph7_smiles") or target.smiles
        record = AppliedRecord(
            key=target.key,
            name=target.name,
            source=target.source,
            met_ids=target.met_ids,
            method=combined["method"],
            method_version=", ".join(
                f"{m}={ladder_versions[m]['tool_version']}" for m in combined["members"]
            ),
            weights_sha256={
                f"{m}:{k}": v
                for m in combined["members"]
                for k, v in (ladder_versions[m].get("weights_sha256") or {}).items()
            },
            members_used=row["members_used"],
            smiles=target.smiles,
            ph7_smiles=state.get("ph7_smiles"),
            submitted_smiles=submitted,
            convention=(
                "pH 7 major microspecies submitted to the compound bridge"
                if state.get("ph7_smiles")
                else "NEUTRAL PARENT submitted: no pH 7 state was available"
            ),
            pkas=row["pkas"],
            conditions={
                "p_h": P_H,
                "ionic_strength_m": IONIC_STRENGTH_M,
                "temperature_k": TEMPERATURE_K,
                "p_mg": P_MG,
            },
            previous_dgf_prime_kj_per_mol=target.previous_dgf_prime_kj_per_mol,
        )
        if not row["pkas"]:
            # An empty ladder has two causes and they are not the same thing. Every member
            # returning an empty ladder with NO error is the right answer for a molecule
            # with no ionizable center inside (0, 14): a neutral ester has none. A member
            # that errored is a failure. The bridge's `pkas_provided_externally` guard
            # refuses an empty list either way, so the no-centre case is created through a
            # cache WITHOUT that guard, with ChemAxon still bypassed, which yields the
            # single-microspecies compound such a molecule actually is. It is recorded as
            # its own outcome so it is never read as an external-pKa value.
            centers = _ionizable_centers(submitted)
            if centers:
                record.outcome = "no_pka_ladder"
                record.detail = (
                    f"{row.get('reason', '')}; the structure DOES carry ionizable centers "
                    f"({', '.join(centers)}), so the empty ladder is a failure rather than "
                    "the answer"
                )
                records.append(record)
                print(f"{target.key}: skipped, {record.outcome}", flush=True)
                continue
            try:
                results = plain.get_compounds(
                    [submitted], mol_format="smiles", bypass_chemaxon=True
                )
            except Exception as exc:
                record.outcome = "compound_creation_raised"
                record.detail = f"{type(exc).__name__}: {exc}"[:300]
                records.append(record)
                print(f"{target.key}: {record.outcome}", flush=True)
                continue
            compound = results[0].compound if results else None
            if compound is None or compound.group_vector is None:
                record.outcome = "failed_group_decomposition"
                records.append(record)
                print(f"{target.key}: {record.outcome}", flush=True)
                continue
            mu, sigma, _ = cc.standard_dg_formation(compound)
            if mu is None:
                record.outcome = "no_formation_energy"
                records.append(record)
                continue
            correction = compound.transform(
                Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
            )
            record.compound_id = int(compound.id)
            record.group_decomposition_ok = True
            record.dgf_prime_kj_per_mol = float(mu) + float(correction.m_as("kJ/mol"))
            record.outcome = "priced_no_ionizable_center"
            record.detail = (
                "every member of the selected method returned an empty ladder and the "
                "structure matches no ionizable-center SMARTS, so the compound was created "
                "as a single microspecies"
            )
            records.append(record)
            print(
                f"{target.key}: {record.dgf_prime_kj_per_mol:.2f} kJ/mol"
                f" ({record.outcome})",
                flush=True,
            )
            continue
        try:
            results = local.get_compounds(
                [submitted],
                mol_format="smiles",
                specified_pkas={submitted: row["pkas"]},
                bypass_chemaxon=True,
            )
        except Exception as exc:
            record.outcome = "compound_creation_raised"
            record.detail = f"{type(exc).__name__}: {exc}"[:300]
            records.append(record)
            print(f"{target.key}: {record.outcome}: {record.detail}", flush=True)
            continue
        compound = results[0].compound if results else None
        if compound is None:
            record.outcome = "compound_creation_failed"
            record.detail = f"method={results[0].method} status={results[0].status}"
            records.append(record)
            print(f"{target.key}: {record.outcome}", flush=True)
            continue
        record.compound_id = int(compound.id)
        record.group_decomposition_ok = compound.group_vector is not None
        if not record.group_decomposition_ok:
            record.outcome = "failed_group_decomposition"
            records.append(record)
            print(f"{target.key}: {record.outcome}", flush=True)
            continue
        mu, sigma, _ = cc.standard_dg_formation(compound)
        if mu is None:
            record.outcome = "no_formation_energy"
            records.append(record)
            print(f"{target.key}: {record.outcome}", flush=True)
            continue
        correction = compound.transform(
            Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
        )
        record.dgf_prime_kj_per_mol = float(mu) + float(correction.m_as("kJ/mol"))
        if sigma is not None:
            import numpy as np

            record.sigma_kj_per_mol = float(np.linalg.norm(np.asarray(sigma)))
        if record.previous_dgf_prime_kj_per_mol is not None:
            record.change_kj_per_mol = (
                record.dgf_prime_kj_per_mol - record.previous_dgf_prime_kj_per_mol
            )
        record.outcome = "priced"
        records.append(record)
        print(
            f"{target.key}: {record.dgf_prime_kj_per_mol:.2f} kJ/mol"
            f" (was {record.previous_dgf_prime_kj_per_mol})",
            flush=True,
        )

    priced = [r for r in records if r.outcome == "priced"]
    payload = {
        "built_at": _timestamp(),
        "method": combined["method"],
        "members": combined["members"],
        "decision_path": DECISION_PATH,
        "decision_sha256": combined["decision_sha256"],
        "local_cache_path": args.local_cache,
        "conditions": {
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        "convention": (
            "the pH 7 major microspecies is submitted to the compound bridge, which is the "
            "microstate the eQuilibrator cache references its own compounds to; submitting "
            "the neutral parent instead moved dGf'0 by a mean of 777.5 kJ/mol on the 7 "
            "targets external_pka_reprice.json priced both ways"
        ),
        "n_targets": len(records),
        "n_priced": len(priced),
        "n_metabolite_rows_priced": sum(len(r.met_ids) for r in priced),
        "outcomes": {
            outcome: sum(1 for r in records if r.outcome == outcome)
            for outcome in sorted({r.outcome for r in records})
        },
        "records": [r.model_dump() for r in records],
    }
    with open(APPLIED_PATH, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"wrote {APPLIED_PATH}: {len(priced)}/{len(records)} priced")
    print(f"  outcomes: {payload['outcomes']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_t = sub.add_parser("targets", help="assemble the re-pricing target list")
    p_t.set_defaults(func=cmd_targets)

    p_l = sub.add_parser("ladders", help="pKa ladders from one member tool")
    p_l.add_argument("--tool", required=True, choices=["molgpka", "pkasolver"])
    p_l.add_argument("--tool-version", default="MolGpKa@4dc8352")
    p_l.set_defaults(func=cmd_ladders)

    p_ui = sub.add_parser(
        "unipka-input", help="write the targets in the shape predict-unipka reads"
    )
    p_ui.set_defaults(func=cmd_unipka_input)

    p_ul = sub.add_parser(
        "unipka-ladders", help="join predict-unipka's output back onto the target keys"
    )
    p_ul.add_argument("--tool-name", default="unipka-simple-template")
    p_ul.set_defaults(func=cmd_unipka_ladders)

    p_m = sub.add_parser("microspecies", help="the pH 7 major microspecies per target")
    p_m.set_defaults(func=cmd_microspecies)

    p_c = sub.add_parser("combine", help="apply the selected rule to the member ladders")
    p_c.set_defaults(func=cmd_combine)

    p_p = sub.add_parser("price", help="create and price the compounds")
    p_p.add_argument("--local-cache", required=True)
    p_p.add_argument("--seed-cache", default=None)
    p_p.set_defaults(func=cmd_price)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
