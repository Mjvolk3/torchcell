# experiments/026-metabolism-flux/scripts/external_pka_benchmark.py
# [[experiments.026-metabolism-flux.scripts.external_pka_benchmark]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/external_pka_benchmark.py

r"""Score open-source pKa predictors against eQuilibrator's own ChemAxon pKas.

eQuilibrator 3.0 cannot create a compound outside its MetaNetX cache without ChemAxon
``cxcalc``, which is licensed. Upstream ``equilibrator-assets`` master added a
``pkas_provided_externally`` switch that accepts a caller-supplied list of macroscopic
pKas instead, so an open-source predictor could take ChemAxon's place. Whether it can is
an empirical question, and the cache answers it for free: every cached compound carries
the ChemAxon dissociation constants that ChemAxon itself produced, so the cache is a
ground-truth set of ~470,000 labeled compounds that no predictor in this comparison was
trained to reproduce.

Seven subcommands, each run under the environment that holds its dependencies.

``sample``
    Draw a stratified validation set from the compound cache: compounds that are
    group-decomposable (so they are actually priceable), organic, and carry between one
    and six pKas inside the pH 0 to 14 window. Stratified by pKa count and by the
    functional groups that dominate yeast-GEM: phosphate, carboxylate, amine, phenol.

``predict-pkasolver`` / ``predict-qupkake``
    Run one predictor over the sample. pKasolver walks a protonation ladder itself and
    returns one pKa per step, which is already the macroscopic shape eQuilibrator wants.
    QupKake returns SITE pKas for one protonation state, so the ladder is built here: at
    each charge level the macroscopic constant is the log-sum-exp over that level's site
    constants, and the next level is entered through the dominant site. Sorting site pKas
    and handing them to eQuilibrator would be wrong for anything polyprotic.

``salvage-qupkake``
    Rebuild QupKake ladders from the per-rung SDF outputs a stopped run left behind.
    ``predict-qupkake`` writes its JSON only after every rung of both directions has
    run, and a QupKake rung costs a full xTB wave over every still-active compound, so
    a run that is killed part way leaves usable predictions with no usable JSON. The
    walk is replayed from those files, and a compound whose walk had not terminated
    when the rungs ran out is recorded as truncated rather than as a short ladder.

``score``
    Compare predicted pKa lists to the cache lists, and propagate the difference to the
    quantity the flux layer consumes: the Legendre transform that turns an untransformed
    formation energy into ``dGf'0`` at pH 7.2. The transform is rebuilt from the predicted
    pKas with :func:`equilibrator_assets.thermodynamics.create_microspecies_mappings`, the
    same function the compound-creation path calls, then re-referenced to the cache's
    major microspecies so that only the pKa contribution is compared.

``check``
    A per-compound plausibility test on the returned ladder that needs no ground truth:
    count the ionizable centers by SMARTS and flag any prediction whose pKa count inside
    (0, 14) exceeds that count, or whose count is zero while centers exist. ChemAxon's own
    lists are scored the same way, so the counter's false-flag rate is measured rather than
    assumed.

``table``
    Print the scored JSON as the markdown tables the note carries: one overall row per
    tool, then the same statistics per pKa-count stratum and per functional group.

Magnesium note: ``specified_pkas`` carries proton dissociation constants only. A compound
built through the external-pKa path has an empty
``magnesium_dissociation_constants`` list, so the transform here is computed from the
proton ladder alone on BOTH sides, and the count of sampled compounds that do carry Mg
constants in the cache is reported as a separate, unclosable gap of the route.

Run::

    # sample and score (equilibrator env, any version with the cache)
    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/external_pka_benchmark.py sample --n 250

    # predictors, each in its own env
    PATH=$HOME/miniconda3/envs/pkasolver-env/bin:$PATH \
        ~/miniconda3/envs/pkasolver-env/bin/python ... predict-pkasolver
    ~/miniconda3/envs/pka-tools/bin/python ... predict-qupkake --workers 32

Outputs ``results/external_pka_benchmark.json`` plus ``results/external_pka_per_compound.csv``,
and ``results/external_pka_check.json`` from ``check``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import random
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

# The validation set and the prediction filenames are overridable through the environment
# so a SECOND, larger sample can be predicted by the same commands without duplicating any
# of them. Each predictor runs under its own conda environment and re-invokes this file as
# a subprocess (Uni-pKa does, one per fold), and an environment variable is the one channel
# that survives both hops. Unset, the names are the 300-compound benchmark's, so every
# command reproduces that run exactly as before.
SAMPLE_PATH = osp.join(
    RESULTS, os.environ.get("EXTERNAL_PKA_SAMPLE", "external_pka_validation_set.json")
)
PRED_PATH_TEMPLATE = osp.join(
    RESULTS,
    os.environ.get(
        "EXTERNAL_PKA_PRED_TEMPLATE", "external_pka_predictions_{tool}.json"
    ),
)
BENCHMARK_PATH = osp.join(RESULTS, "external_pka_benchmark.json")
PER_COMPOUND_CSV = osp.join(RESULTS, "external_pka_per_compound.csv")

MIN_PH = 0.0
MID_PH = 7.0
MAX_PH = 14.0
P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

ORGANIC_ELEMENTS = {"C", "H", "N", "O", "P", "S"}
MAX_HEAVY_ATOMS = 50
MAX_PKAS = 6

GROUP_SMARTS = {
    "phosphate": "[PX4](=O)([OX2H1,OX1-])",
    "carboxylate": "[CX3](=[OX1])[OX2H1,OX1-]",
    "amine": "[NX3;!$(N[C,S]=[O,S,N]);!$(N=*);!$([N+](=O)[O-])]",
    "phenol": "[OX2H1,OX1-]c",
}


class ValidationCompound(BaseModel):
    """One cached compound with its ChemAxon pKas, used as ground truth."""

    compound_id: int
    inchi_key: str
    major_ms_smiles: str
    neutral_smiles: str
    formula_atom_bag: dict[str, int]
    cache_pkas: list[float]
    n_cache_pkas: int
    groups: list[str]
    n_heavy_atoms: int
    has_magnesium_constants: bool
    stratum: str
    # The four axes below are written only by the method-selection sampler
    # (``external_pka_method_selection.py sample``), which stratifies on more than the pKa
    # count. They default so the 300-compound benchmark file still validates unchanged.
    net_charge_ph7: int = Field(
        default=0,
        description=(
            "Formal charge of the cache's major microspecies, which the cache defines at "
            "pH 7. Read off that SMILES rather than derived from the pKa list."
        ),
    )
    molecular_weight: float = Field(
        default=0.0, description="Average molecular weight of the neutral parent, Da."
    )
    lipid_like: bool = Field(
        default=False,
        description=(
            "Long acyl chain or a phosphate headgroup, the regime the yeast-GEM fill-in "
            "actually runs in. Defined by SMARTS in the method-selection sampler."
        ),
    )
    strata: dict[str, str] = Field(
        default_factory=dict,
        description="Stratum label per axis: pka_count, group, charge, mass, lipid.",
    )
    in_seed17_subset: bool = Field(
        default=False,
        description="True for a compound carried over from the 300-compound seed-17 set.",
    )


class ValidationSet(BaseModel):
    """The stratified sample plus how it was drawn."""

    built_at: str
    cache_path: str
    n_compounds: int
    seed: int
    n_candidates_scanned: int
    n_candidates_eligible: int
    stratum_counts: dict[str, int]
    group_counts: dict[str, int]
    n_with_magnesium_constants: int
    compounds: list[ValidationCompound]
    pinned_from: str | None = Field(
        default=None,
        description="Sample file whose compounds were carried over verbatim, if any.",
    )
    n_pinned: int = 0
    stratum_counts_by_axis: dict[str, dict[str, int]] = Field(default_factory=dict)


class PredictionRecord(BaseModel):
    """What one predictor returned for one compound."""

    compound_id: int
    tool: str
    ok: bool
    error: str | None = None
    predicted_pkas: list[float] = Field(default_factory=list)
    site_pkas: list[float] = Field(default_factory=list)
    seconds: float | None = None


class PredictionSet(BaseModel):
    """All predictions from one tool over the validation set."""

    tool: str
    tool_version: str
    built_at: str
    n_requested: int
    n_ok: int
    seconds_total: float
    records: list[PredictionRecord]


class ScoreRow(BaseModel):
    """Per-compound scoring of one tool against the cache."""

    compound_id: int
    tool: str
    inchi_key: str
    stratum: str
    groups: list[str]
    n_cache_pkas: int
    n_pred_pkas: int
    count_matches: bool
    cache_pkas: list[float]
    predicted_pkas: list[float]
    pka_mae: float | None
    pka_max_abs_error: float | None
    ddgf_prime_kj_per_mol: float | None
    ok: bool
    error: str | None = None


class ToolSummary(BaseModel):
    """Aggregate scores for one tool, overall and per pKa-count stratum."""

    tool: str
    n_attempted: int
    n_ok: int
    frac_ok: float
    n_count_match: int
    frac_count_match: float
    pka_mae_paired: float | None
    pka_mae_n: int
    ddgf_mae_kj_per_mol: float | None
    ddgf_median_abs_kj_per_mol: float | None
    ddgf_p90_abs_kj_per_mol: float | None
    ddgf_n: int
    by_stratum: dict[str, dict[str, float | int | None]]
    by_group: dict[str, dict[str, float | int | None]]


class BenchmarkResult(BaseModel):
    """The whole comparison."""

    built_at: str
    validation_set_path: str
    n_validation_compounds: int
    conditions: dict[str, float]
    magnesium_gap: dict[str, int | str]
    tools: list[ToolSummary]


def _timestamp() -> str:
    # `timezone.utc` rather than `datetime.UTC`: pKasolver's environment is Python 3.9,
    # where the alias does not exist, and `predict-pkasolver` and `microspecies-at-ph`
    # both run this file there. Leave it as is (ruff UP017 is silenced for this line).
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


# --------------------------------------------------------------------------- sample


def _neutralize(smiles: str) -> str | None:
    """The neutral parent of a major-microspecies SMILES, or None."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    neutral = rdMolStandardize.Uncharger().uncharge(mol)
    if neutral is None:
        return None
    if Chem.GetFormalCharge(neutral) != 0:
        return None
    return Chem.MolToSmiles(neutral)


def _groups_of(smiles: str) -> list[str]:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    found = []
    for name, smarts in GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            found.append(name)
    return found


def cmd_sample(args: argparse.Namespace) -> None:
    """Draw the stratified validation set from the eQuilibrator compound cache."""
    from equilibrator_cache.api import create_compound_cache_from_zenodo
    from equilibrator_cache.models import Compound
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    ccache = create_compound_cache_from_zenodo()
    query = (
        ccache.session.query(Compound)
        .filter(
            Compound.smiles.isnot(None),
            Compound.group_vector.isnot(None),
            Compound.dissociation_constants.isnot(None),
            Compound.mass < args.max_mass,
        )
        .order_by(Compound.id)
    )

    rng = random.Random(args.seed)
    eligible: dict[str, list[ValidationCompound]] = {}
    n_scanned = 0
    n_eligible = 0
    for compound in query.yield_per(2000):
        n_scanned += 1
        if n_scanned > args.scan_limit:
            break
        pkas = list(compound.dissociation_constants or [])
        pkas = sorted((p for p in pkas if MIN_PH < float(p) < MAX_PH), reverse=True)
        if not 1 <= len(pkas) <= MAX_PKAS:
            continue
        atom_bag = dict(compound.atom_bag or {})
        elements = {k for k in atom_bag if k != "e-"}
        if not elements or not elements <= ORGANIC_ELEMENTS:
            continue
        n_heavy = sum(v for k, v in atom_bag.items() if k not in ("H", "e-"))
        if n_heavy > MAX_HEAVY_ATOMS or n_heavy < 3:
            continue
        neutral = _neutralize(compound.smiles)
        if neutral is None:
            continue
        mol = Chem.MolFromSmiles(neutral)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        groups = _groups_of(neutral)
        stratum = f"n_pka_{len(pkas)}"
        record = ValidationCompound(
            compound_id=compound.id,
            inchi_key=compound.inchi_key or "",
            major_ms_smiles=compound.smiles,
            neutral_smiles=neutral,
            formula_atom_bag=atom_bag,
            cache_pkas=[float(p) for p in pkas],
            n_cache_pkas=len(pkas),
            groups=groups,
            n_heavy_atoms=n_heavy,
            has_magnesium_constants=bool(compound.magnesium_dissociation_constants),
            stratum=stratum,
        )
        eligible.setdefault(stratum, []).append(record)
        n_eligible += 1

    per_stratum = max(1, args.n // MAX_PKAS)
    chosen: list[ValidationCompound] = []
    for k in range(1, MAX_PKAS + 1):
        stratum = f"n_pka_{k}"
        pool = eligible.get(stratum, [])
        rng.shuffle(pool)
        # Take group-bearing compounds first so phosphates, carboxylates, amines and
        # phenols are all represented, then fill the stratum from the rest.
        with_groups = [r for r in pool if r.groups]
        without = [r for r in pool if not r.groups]
        chosen.extend((with_groups + without)[:per_stratum])

    stratum_counts: dict[str, int] = {}
    group_counts: dict[str, int] = {}
    for record in chosen:
        stratum_counts[record.stratum] = stratum_counts.get(record.stratum, 0) + 1
        for g in record.groups:
            group_counts[g] = group_counts.get(g, 0) + 1

    vset = ValidationSet(
        built_at=_timestamp(),
        cache_path=str(ccache.session.bind.url),
        n_compounds=len(chosen),
        seed=args.seed,
        n_candidates_scanned=n_scanned,
        n_candidates_eligible=n_eligible,
        stratum_counts=stratum_counts,
        group_counts=group_counts,
        n_with_magnesium_constants=sum(1 for r in chosen if r.has_magnesium_constants),
        compounds=chosen,
    )
    os.makedirs(RESULTS, exist_ok=True)
    with open(SAMPLE_PATH, "w") as handle:
        handle.write(vset.model_dump_json(indent=1))
    print(f"wrote {SAMPLE_PATH}: n={vset.n_compounds}")
    print("strata:", stratum_counts)
    print("groups:", group_counts)
    print("with Mg constants:", vset.n_with_magnesium_constants)


def _load_sample() -> ValidationSet:
    with open(SAMPLE_PATH) as handle:
        return ValidationSet.model_validate_json(handle.read())


def _shard_of(
    compounds: list[ValidationCompound], shard: int, n_shards: int
) -> list[ValidationCompound]:
    """Every ``n_shards``-th compound starting at ``shard``.

    MolGpKa costs ~39 s per compound in one process and pKasolver-lite ~4 s, so a
    2,000-compound set is 22 h and 2.2 h respectively on one core. Both are pure CPU with
    no shared state, so the set is cut round-robin rather than into contiguous blocks: the
    strata are interleaved in the sample file, and a contiguous cut would hand one worker
    every 6-pKa compound and leave it running long after the rest are done.
    """
    if n_shards <= 1:
        return compounds
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} is outside 0..{n_shards - 1}")
    return [c for i, c in enumerate(compounds) if i % n_shards == shard]


def _prediction_path(tool: str, shard: int, n_shards: int) -> str:
    """Where one worker writes; a sharded run writes one file per worker."""
    if n_shards <= 1:
        return PRED_PATH_TEMPLATE.format(tool=tool)
    return PRED_PATH_TEMPLATE.format(tool=f"{tool}_shard{shard:03d}of{n_shards:03d}")


def cmd_merge_shards(args: argparse.Namespace) -> None:
    """Concatenate one tool's shard files into the single prediction file ``score`` reads.

    A missing shard is an error rather than a silently shorter prediction set: a merged
    file that quietly dropped a worker would be scored as if the tool had simply failed on
    those compounds, which is a different claim from the tool never having seen them.
    """
    records: list[PredictionRecord] = []
    seconds = 0.0
    n_requested = 0
    version = ""
    for shard in range(args.n_shards):
        path = _prediction_path(args.tool, shard, args.n_shards)
        if not osp.isfile(path):
            raise FileNotFoundError(
                f"shard {shard} of {args.n_shards} is missing: {path}"
            )
        with open(path) as handle:
            pset = PredictionSet.model_validate_json(handle.read())
        records.extend(pset.records)
        seconds += pset.seconds_total
        n_requested += pset.n_requested
        version = pset.tool_version
    merged = PredictionSet(
        tool=args.tool,
        tool_version=version,
        built_at=_timestamp(),
        n_requested=n_requested,
        n_ok=sum(1 for r in records if r.ok),
        seconds_total=seconds,
        records=records,
    )
    out = PRED_PATH_TEMPLATE.format(tool=args.tool)
    with open(out, "w") as handle:
        handle.write(merged.model_dump_json(indent=1))
    print(
        f"wrote {out}: {merged.n_ok}/{merged.n_requested} over {args.n_shards} shards,"
        f" {seconds:.0f} CPU-seconds"
    )


# ------------------------------------------------------------------ predict-pkasolver


def cmd_predict_pkasolver(args: argparse.Namespace) -> None:
    """Run pKasolver-lite over the validation set."""
    import time

    from pkasolver.query import QueryModel, calculate_microstate_pka_values
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    vset = _load_sample()
    compounds = _shard_of(vset.compounds, args.shard, args.n_shards)
    model = QueryModel()

    records: list[PredictionRecord] = []
    t_all = time.time()
    for i, compound in enumerate(compounds):
        t0 = time.time()
        try:
            mol = Chem.MolFromSmiles(compound.neutral_smiles)
            states = calculate_microstate_pka_values(mol, query_model=model)
            pkas = sorted(
                (float(s.pka) for s in states if MIN_PH < float(s.pka) < MAX_PH),
                reverse=True,
            )
            records.append(
                PredictionRecord(
                    compound_id=compound.compound_id,
                    tool="pkasolver-lite",
                    ok=True,
                    predicted_pkas=pkas,
                    site_pkas=[float(s.pka) for s in states],
                    seconds=time.time() - t0,
                )
            )
        except Exception as exc:  # a failure is a result; record it and continue
            records.append(
                PredictionRecord(
                    compound_id=compound.compound_id,
                    tool="pkasolver-lite",
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}"[:300],
                    seconds=time.time() - t0,
                )
            )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(compounds)}", flush=True)

    import pkasolver

    pset = PredictionSet(
        tool="pkasolver-lite",
        tool_version=getattr(pkasolver, "__version__", "unknown"),
        built_at=_timestamp(),
        n_requested=len(compounds),
        n_ok=sum(1 for r in records if r.ok),
        seconds_total=time.time() - t_all,
        records=records,
    )
    path = _prediction_path("pkasolver", args.shard, args.n_shards)
    with open(path, "w") as handle:
        handle.write(pset.model_dump_json(indent=1))
    print(
        f"wrote {path}: ok {pset.n_ok}/{pset.n_requested} in {pset.seconds_total:.0f}s"
    )


# -------------------------------------------------------------------- predict-qupkake


def _qupkake_sites(
    smiles_by_name: dict[str, str], root: str, workers: int
) -> dict[str, list[tuple[int, str, float]]]:
    """Site pKas for a batch of molecules: name -> [(atom_idx, type, pka), ...].

    QupKake ships a CLI and writes an SDF with one record per predicted site. It is run as
    a subprocess so that its own dataset caching and multiprocessing are used as intended.

    Conformers are embedded here rather than by QupKake's CSV path, which applies
    ``MMFFOptimizeMolecule`` without catching a failed embedding, so one molecule that
    will not embed aborts the whole batch. Embedding here drops that molecule and keeps
    the rest.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    os.makedirs(root, exist_ok=True)
    in_sdf = osp.join(root, "batch_in.sdf")
    out_name = "batch_out.sdf"

    writer = Chem.SDWriter(in_sdf)
    n_written = 0
    for name, smiles in smiles_by_name.items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 0xF00D
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except ValueError:
            pass
        mol.SetProp("_Name", name)
        mol.SetProp("name", name)
        writer.write(mol)
        n_written += 1
    writer.close()
    if n_written == 0:
        raise RuntimeError("no molecule in the batch could be embedded")

    runner = osp.join(root, "_run_qupkake.py")
    with open(runner, "w") as handle:
        handle.write(
            "import sys, torch\n"
            "_orig = torch.load\n"
            "def _load(*a, **kw):\n"
            "    kw['weights_only'] = False\n"
            "    return _orig(*a, **kw)\n"
            "torch.load = _load\n"
            "from qupkake.cli import main\n"
            "sys.exit(main())\n"
        )
    cmd = [
        sys.executable,
        runner,
        "file",
        in_sdf,
        "-s",
        "smiles",
        "-n",
        "name",
        "-o",
        out_name,
        "-r",
        osp.join(root, "data"),
        "-mp",
        str(workers),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out_sdf = osp.join(root, "data", "output", out_name)
    if not osp.isfile(out_sdf):
        raise RuntimeError(
            f"qupkake produced no output ({proc.returncode}): {proc.stderr[-1500:]}"
        )
    return _parse_qupkake_sdf(out_sdf)


def _parse_qupkake_sdf(out_sdf: str) -> dict[str, list[tuple[int, str, float]]]:
    """QupKake's output SDF as name -> [(atom_idx, type, pka), ...].

    One record per predicted site, carrying ``idx``, ``pka_type`` and ``pka``. Both
    acidic and basic sites of the SAME input state are in one file, which is what makes
    a partial run salvageable in both ladder directions.
    """
    from rdkit import Chem

    sites: dict[str, list[tuple[int, str, float]]] = {}
    supplier = Chem.SDMolSupplier(out_sdf, removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is None:
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else None
        if name is None:
            continue
        entry = sites.setdefault(name, [])
        for key in mol.GetPropNames():
            if not key.startswith("pka_type"):
                continue
            suffix = key[len("pka_type") :]
            idx_key = f"idx{suffix}"
            pka_key = f"pka{suffix}"
            if not (mol.HasProp(idx_key) and mol.HasProp(pka_key)):
                continue
            entry.append(
                (
                    int(mol.GetProp(idx_key)),
                    mol.GetProp(key).strip(),
                    float(mol.GetProp(pka_key)),
                )
            )
    return sites


def _apply_proton_change(smiles: str, atom_idx: int, delta: int) -> str | None:
    """The SMILES with one proton added to or removed from ``atom_idx``."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or atom_idx >= mol.GetNumAtoms():
        return None
    editable = Chem.RWMol(mol)
    atom = editable.GetAtomWithIdx(atom_idx)
    n_h = atom.GetTotalNumHs()
    if delta < 0 and n_h == 0:
        return None
    atom.SetNumExplicitHs(max(0, n_h + delta))
    atom.SetFormalCharge(atom.GetFormalCharge() + delta)
    atom.SetNoImplicit(True)
    out = editable.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Chem.MolToSmiles(out)


def _macro_from_sites(site_pkas: list[float], acidic: bool) -> float:
    """The macroscopic constant for one charge step from its site constants.

    For a deprotonation out of a single reference macrostate, the macroscopic association
    constant is the sum of the site constants, so
    ``pKa = -log10 sum_i 10^(-pKa_i)``; for a protonation the sum runs the other way,
    ``pKa = log10 sum_j 10^(pKa_j)``. Both are the log-sum-exp of the site values, which
    is why a sorted list of site pKas is not a list of macroscopic pKas: N equivalent
    sites shift the macroscopic value by log10(N).
    """
    if acidic:
        return -math.log10(sum(10.0 ** (-p) for p in site_pkas))
    return math.log10(sum(10.0**p for p in site_pkas))


def cmd_predict_qupkake(args: argparse.Namespace) -> None:
    """Run QupKake and build a macroscopic ladder from its site pKas.

    QupKake predicts site pKas for ONE protonation state, so the ladder is walked
    explicitly: from the current state the macroscopic constant is the log-sum-exp over
    the state's sites, and the next state is reached through the dominant site (the most
    acidic site going down, the most basic going up). Holding only the dominant
    microstate at each charge is an approximation to the full microstate graph, and it is
    exact whenever one microstate dominates its charge level.
    """
    import time

    vset = _load_sample()
    work_root = args.work_dir or tempfile.mkdtemp(prefix="qupkake_bench_")
    os.makedirs(work_root, exist_ok=True)
    print(f"qupkake work dir: {work_root}")

    compounds = vset.compounds
    if args.strata:
        compounds = [c for c in compounds if c.stratum in set(args.strata)]
    if args.limit:
        # Each rung is a full xTB wave over every still-active compound, so the run
        # time is set by the compound count. The limit keeps a QupKake run finite.
        per_stratum = max(1, args.limit // max(1, len({c.stratum for c in compounds})))
        kept: dict[str, int] = {}
        subset = []
        for c in compounds:
            if kept.get(c.stratum, 0) < per_stratum:
                kept[c.stratum] = kept.get(c.stratum, 0) + 1
                subset.append(c)
        compounds = subset
    print(f"qupkake over {len(compounds)} compounds, max {args.max_rungs} rungs")
    by_id = {c.compound_id: c for c in compounds}
    ladder: dict[int, list[float]] = {cid: [] for cid in by_id}
    all_sites: dict[int, list[float]] = {cid: [] for cid in by_id}
    errors: dict[int, str] = {}

    t_all = time.time()
    for direction, acidic in (("acidic", True), ("basic", False)):
        # state[cid] is the SMILES of the compound at the current charge level
        state = {cid: c.neutral_smiles for cid, c in by_id.items()}
        active = set(state)
        for step in range(args.max_rungs):
            if not active:
                break
            batch = {f"c{cid}": state[cid] for cid in sorted(active)}
            root = osp.join(work_root, f"{direction}_{step}")
            try:
                sites = _qupkake_sites(batch, root, args.workers)
            except Exception as exc:
                for cid in active:
                    errors.setdefault(cid, f"{type(exc).__name__}: {exc}"[:300])
                break
            still_active = set()
            for cid in sorted(active):
                entry = sites.get(f"c{cid}", [])
                wanted = [
                    (idx, pka)
                    for idx, kind, pka in entry
                    if kind == ("acidic" if acidic else "basic")
                ]
                if not wanted:
                    continue
                all_sites[cid].extend(p for _, p in wanted)
                macro = _macro_from_sites([p for _, p in wanted], acidic)
                if not MIN_PH < macro < MAX_PH:
                    continue
                ladder[cid].append(macro)
                dominant = (
                    min(wanted, key=lambda t: t[1])
                    if acidic
                    else max(wanted, key=lambda t: t[1])
                )
                nxt = _apply_proton_change(state[cid], dominant[0], -1 if acidic else 1)
                if nxt is None:
                    continue
                state[cid] = nxt
                still_active.add(cid)
            active = still_active
            print(
                f"  {direction} step {step}: {len(active)} compounds continue",
                flush=True,
            )

    records = []
    for cid, compound in by_id.items():
        pkas = sorted((p for p in ladder[cid]), reverse=True)
        records.append(
            PredictionRecord(
                compound_id=cid,
                tool="qupkake",
                ok=cid not in errors,
                error=errors.get(cid),
                predicted_pkas=pkas,
                site_pkas=all_sites[cid],
            )
        )

    pset = PredictionSet(
        tool="qupkake",
        tool_version=args.tool_version,
        built_at=_timestamp(),
        n_requested=len(by_id),
        n_ok=sum(1 for r in records if r.ok),
        seconds_total=time.time() - t_all,
        records=records,
    )
    path = PRED_PATH_TEMPLATE.format(tool="qupkake")
    with open(path, "w") as handle:
        handle.write(pset.model_dump_json(indent=1))
    print(
        f"wrote {path}: ok {pset.n_ok}/{pset.n_requested} in {pset.seconds_total:.0f}s"
    )


# -------------------------------------------------------------------- salvage-qupkake


def cmd_salvage_qupkake(args: argparse.Namespace) -> None:
    """Rebuild QupKake ladders from the step outputs a killed run left on disk.

    ``predict-qupkake`` writes its JSON only after every rung of both directions has
    run, so a run that is stopped part way leaves nothing but its per-rung work
    directories. Each of those holds the input SDF for that rung and, if the rung
    finished, QupKake's output SDF. This replays the same walk as
    :func:`cmd_predict_qupkake` against those files, so no model is run again.

    Two properties of the walk make the replay exact rather than approximate.

    1. The rung-0 input of BOTH directions is the neutral molecule, and QupKake reports
       the acidic and the basic sites of a state in one file, so ``acidic_0`` supplies
       the basic rung 0 as well.
    2. A compound leaves the walk when its state has no site of the wanted type, when
       the macroscopic value falls outside (0, 14), or when the proton change does not
       sanitize. A compound that left the walk at or before the last FINISHED rung has
       the ladder the full run would have produced; a compound still walking when the
       rungs ran out is truncated, and is recorded with ``ok=False``, since a truncated
       ladder is short by an unknown number of pKas rather than wrong by a known amount.
    """
    vset = _load_sample()
    by_id = {c.compound_id: c for c in vset.compounds}

    step_sites: dict[tuple[str, int], dict[str, list[tuple[int, str, float]]]] = {}
    for direction in ("acidic", "basic"):
        for step in range(args.max_rungs):
            # The neutral state is shared, so the basic rung 0 is read from acidic_0.
            source = "acidic" if (direction == "basic" and step == 0) else direction
            out_sdf = osp.join(
                args.work_dir, f"{source}_{step}", "data", "output", "batch_out.sdf"
            )
            if not osp.isfile(out_sdf):
                break
            step_sites[(direction, step)] = _parse_qupkake_sdf(out_sdf)

    rung0 = osp.join(args.work_dir, "acidic_0", "batch_in.sdf")
    if not osp.isfile(rung0):
        raise RuntimeError(f"no rung-0 input at {rung0}")
    from rdkit import Chem

    attempted: list[int] = []
    for mol in Chem.SDMolSupplier(rung0, removeHs=False, sanitize=False):
        if mol is None or not mol.HasProp("_Name"):
            continue
        attempted.append(int(mol.GetProp("_Name")[1:]))

    ladder: dict[int, list[float]] = {cid: [] for cid in attempted}
    all_sites: dict[int, list[float]] = {cid: [] for cid in attempted}
    truncated: dict[int, str] = {}

    for direction, acidic in (("acidic", True), ("basic", False)):
        state = {cid: by_id[cid].neutral_smiles for cid in attempted}
        active = set(attempted)
        step = 0
        while active:
            sites = step_sites.get((direction, step))
            if sites is None:
                for cid in active:
                    truncated.setdefault(
                        cid, f"{direction} ladder truncated at rung {step}"
                    )
                break
            still_active = set()
            for cid in sorted(active):
                entry = sites.get(f"c{cid}", [])
                wanted = [
                    (idx, pka)
                    for idx, kind, pka in entry
                    if kind == ("acidic" if acidic else "basic")
                ]
                if not wanted:
                    continue
                all_sites[cid].extend(p for _, p in wanted)
                macro = _macro_from_sites([p for _, p in wanted], acidic)
                if not MIN_PH < macro < MAX_PH:
                    continue
                ladder[cid].append(macro)
                dominant = (
                    min(wanted, key=lambda t: t[1])
                    if acidic
                    else max(wanted, key=lambda t: t[1])
                )
                nxt = _apply_proton_change(state[cid], dominant[0], -1 if acidic else 1)
                if nxt is None:
                    continue
                state[cid] = nxt
                still_active.add(cid)
            active = still_active
            print(
                f"  {direction} rung {step}: {len(active)} compounds continue",
                flush=True,
            )
            step += 1

    # Two files, because they answer two different questions. ``qupkake`` holds only
    # the ladders that terminated, which is the honest accuracy set. ``qupkake-truncated``
    # holds every attempted compound with the rungs that did run, whose ladders are
    # short by an unknown number of pKas, so its scores bound the tool rather than
    # measure it.
    for tool_name, file_key in (
        ("qupkake", "qupkake"),
        ("qupkake-truncated", "qupkake-truncated"),
    ):
        keep_truncated = tool_name.endswith("-truncated")
        records = [
            PredictionRecord(
                compound_id=cid,
                tool=tool_name,
                ok=keep_truncated or cid not in truncated,
                error=truncated.get(cid),
                predicted_pkas=sorted(ladder[cid], reverse=True),
                site_pkas=all_sites[cid],
            )
            for cid in attempted
        ]
        pset = PredictionSet(
            tool=tool_name,
            tool_version=args.tool_version,
            built_at=_timestamp(),
            n_requested=len(attempted),
            n_ok=sum(1 for r in records if r.ok),
            seconds_total=0.0,
            records=records,
        )
        path = PRED_PATH_TEMPLATE.format(tool=file_key)
        with open(path, "w") as handle:
            handle.write(pset.model_dump_json(indent=1))
        print(f"wrote {path}: {pset.n_ok}/{pset.n_requested} ladders kept")
    print(f"rungs read: {sorted(step_sites)}")
    print(f"truncated ladders: {len(truncated)}/{len(attempted)}")


# -------------------------------------------------------------------- predict-molgpka


def cmd_predict_molgpka(args: argparse.Namespace) -> None:
    """Run MolGpKa and build the same macroscopic ladder from its site pKas.

    MolGpKa finds ionizable centers with a SMARTS table and predicts one pKa per center
    with a graph network, so like QupKake it is a site-level predictor and needs the same
    aggregation. Its published entry point neutralizes the input before predicting, which
    would reset the ladder at every step, so ``predict_acid`` and ``predict_base`` are
    called directly on the current protonation state instead. Predicting on a charged
    state is outside the neutral-molecule regime the model was trained on, and the deeper
    rungs of the ladder should be read with that in mind.
    """
    import time

    sys.path.insert(0, osp.join(args.repo, "src"))
    os.chdir(osp.join(args.repo, "src"))

    import torch

    _orig_load = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _load

    import predict_pka  # type: ignore[import-not-found]

    # predict_acid and predict_base each reload the .pth from disk on every call, which
    # dominates the run once the ladder is walked state by state. The weights never
    # change, so they are loaded once.
    _load_model = predict_pka.load_model
    _model_cache: dict[tuple[str, str], object] = {}

    def _cached_load_model(model_file: str, device: str = "cpu"):
        key = (model_file, device)
        if key not in _model_cache:
            _model_cache[key] = _load_model(model_file, device)
        return _model_cache[key]

    predict_pka.load_model = _cached_load_model
    predict_acid = predict_pka.predict_acid
    predict_base = predict_pka.predict_base
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    vset = _load_sample()
    compounds = _shard_of(vset.compounds, args.shard, args.n_shards)
    records: list[PredictionRecord] = []
    t_all = time.time()
    for i, compound in enumerate(compounds):
        t0 = time.time()
        ladder: list[float] = []
        sites_seen: list[float] = []
        error: str | None = None
        try:
            for acidic in (True, False):
                state = compound.neutral_smiles
                for _ in range(MAX_PKAS + 1):
                    mol = Chem.MolFromSmiles(state)
                    if mol is None:
                        break
                    mol_h = Chem.AddHs(mol)
                    found = predict_acid(mol_h) if acidic else predict_base(mol_h)
                    if not found:
                        break
                    values = [float(v) for v in found.values()]
                    sites_seen.extend(values)
                    macro = _macro_from_sites(values, acidic)
                    if not MIN_PH < macro < MAX_PH:
                        break
                    ladder.append(macro)
                    dominant_idx = (
                        min(found, key=lambda k: float(found[k]))
                        if acidic
                        else max(found, key=lambda k: float(found[k]))
                    )
                    if acidic:
                        # An acidic site is reported as the hydrogen's index, so the
                        # proton comes off its heavy neighbor.
                        atom = mol_h.GetAtomWithIdx(int(dominant_idx))
                        neighbors = atom.GetNeighbors()
                        if not neighbors:
                            break
                        heavy_idx = neighbors[0].GetIdx()
                    else:
                        heavy_idx = int(dominant_idx)
                    if heavy_idx >= mol.GetNumAtoms():
                        break
                    nxt = _apply_proton_change(state, heavy_idx, -1 if acidic else 1)
                    if nxt is None:
                        break
                    state = nxt
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        records.append(
            PredictionRecord(
                compound_id=compound.compound_id,
                tool="molgpka",
                ok=error is None,
                error=error,
                predicted_pkas=sorted(ladder, reverse=True),
                site_pkas=sites_seen,
                seconds=time.time() - t0,
            )
        )
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(compounds)}", flush=True)

    pset = PredictionSet(
        tool="molgpka",
        tool_version=args.tool_version,
        built_at=_timestamp(),
        n_requested=len(compounds),
        n_ok=sum(1 for r in records if r.ok),
        seconds_total=time.time() - t_all,
        records=records,
    )
    path = _prediction_path("molgpka", args.shard, args.n_shards)
    with open(path, "w") as handle:
        handle.write(pset.model_dump_json(indent=1))
    print(
        f"wrote {path}: ok {pset.n_ok}/{pset.n_requested} in {pset.seconds_total:.0f}s"
    )


# --------------------------------------------------------------------- predict-unipka


#: Atomic numbers a Uni-pKa microstate filter written with wildcard charges will discard.
#: Anything past the second row that is not a common organic heteroatom counts as a metal
#: for this purpose; the point is to refuse the class rather than to classify chemistry.
_ORGANIC_ELEMENTS = frozenset(
    {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"}
)


def _has_metal(smiles: str) -> bool:
    """Whether a structure carries an atom outside the organic set, a metal in practice."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(a.GetSymbol() not in _ORGANIC_ELEMENTS for a in mol.GetAtoms())


def cmd_predict_unipka(args: argparse.Namespace) -> None:
    """Run Uni-pKa and take its OWN macroscopic pKa, not a site ladder.

    Uni-pKa is not a site predictor and it is not aggregated here. The model consumes a
    protonation ensemble written as ``A1,...,Am>>B1,...,Bn``, the acid macrostate against
    the base macrostate, and its head produces the standard free energy of each microstate;
    the macroscopic constant is formed inside the model's own loss as

    ``pKa = (logsumexp(-G_acid) - logsumexp(-G_base)) / ln 10``

    (``unimol/losses/reg_loss.py``), which is the ensemble average this benchmark's
    ``_macro_from_sites`` only approximates for the site-level tools. So the ladder is
    built by handing the model one adjacent macrostate PAIR per rung and reading its
    output, rather than by aggregating anything afterwards.

    The macrostates come from Uni-pKa's own microstate enumerator (``enumerator/main.py
    ensemble``), which walks a SMARTS protonation template out from the neutral parent to
    a charge ceiling and floor. A rung exists wherever two adjacent charge levels are both
    populated.

    Five folds are run and averaged, which is the protocol the repository documents for
    its own metrics ("The metrics are calculated using the average of the 5-fold model
    predictions").
    """
    import csv as csv_module
    import time

    import pandas as pd
    from unipkainfer import PredictionConfig, predict_pkas

    vset = _load_sample()
    compounds = vset.compounds[: args.limit] if args.limit else vset.compounds
    # Uni-pKa's enumerator drops any microstate matching its FILTER_PATTERNS, and that list
    # contains the wildcard `[*-2]`, which matches `[Fe-2]` in every metal-complex
    # microstate and empties the ensemble. The failure is silent: the model is handed an
    # empty macrostate pair and returns nothing, which reads downstream as "no ladder" and
    # not as "the enumerator threw the molecule away". Metal complexes are therefore
    # refused here, by name, until the enumerator's filter is restricted to non-metals.
    metals = [c for c in compounds if _has_metal(c.neutral_smiles)]
    if metals:
        print(
            f"refusing {len(metals)} metal-containing compounds: Uni-pKa's enumerator "
            "filter discards every microstate of a metal complex. "
            + ", ".join(c.compound_id for c in metals[:8])
        )
        compounds = [c for c in compounds if not _has_metal(c.neutral_smiles)]
    work = args.work_dir
    os.makedirs(work, exist_ok=True)

    # 1. The neutral parents, in the enumerator's input shape.
    enum_in = osp.join(work, "unipka_parents.tsv")
    with open(enum_in, "w", newline="") as handle:
        writer = csv_module.writer(handle, delimiter="\t")
        writer.writerow(["compound_id", "SMILES"])
        for compound in compounds:
            writer.writerow([compound.compound_id, compound.neutral_smiles])

    # 2. The protonation ensemble, one column per total charge.
    enum_out = osp.join(work, "unipka_ensemble.tsv")
    if not osp.isfile(enum_out):
        command = [
            sys.executable,
            "main.py",
            "ensemble",
            "-i",
            enum_in,
            "-o",
            enum_out,
            "-u",
            str(args.charge_upper),
            "-l",
            str(args.charge_lower),
            "-t",
            args.template,
        ]
        print(f"enumerating: {' '.join(command)}", flush=True)
        proc = subprocess.run(
            command, cwd=args.enumerator, capture_output=True, text=True
        )
        if not osp.isfile(enum_out):
            raise RuntimeError(
                f"the enumerator produced no output ({proc.returncode}):"
                f" {proc.stderr[-2000:]}"
            )
    ensemble = pd.read_csv(enum_out, sep="\t", index_col=0)

    # 3. One inference row per rung: the charge-q macrostate against charge q-1.
    charges = sorted(
        (int(c) for c in ensemble.columns if c not in ("SMILES", "compound_id")),
        reverse=True,
    )
    rows: list[dict[str, object]] = []
    for position, compound in enumerate(compounds):
        record = ensemble.iloc[position]
        # The enumerator writes its input SMILES back out, so row order can be checked
        # rather than assumed; a silent misalignment here would attach every ladder to
        # the wrong compound.
        if record["SMILES"] != compound.neutral_smiles:
            raise RuntimeError(
                f"enumerator row {position} is {record['SMILES']!r}, expected"
                f" {compound.neutral_smiles!r}"
            )
        for upper, lower in zip(charges, charges[1:]):
            acid = record.get(str(upper))
            base = record.get(str(lower))
            if not isinstance(acid, str) or not isinstance(base, str):
                continue
            rows.append(
                {
                    "SMILES": f"{acid}>>{base}",
                    # TARGET is carried through inference untouched, so it is the join
                    # key back to (compound, rung). Matching on the reaction string
                    # instead would break on the two compounds that share one.
                    "TARGET": float(len(rows)),
                    "compound_id": compound.compound_id,
                    "charge_from": upper,
                    "charge_to": lower,
                }
            )
    frame = pd.DataFrame(rows)
    infer_in = osp.join(work, "unipka_rungs.tsv")
    frame[["SMILES", "TARGET"]].to_csv(infer_in, sep="\t", index=False)
    print(
        f"{len(frame)} rungs over {len(compounds)} compounds -> {infer_in}", flush=True
    )

    # 4. Inference, one fold per PROCESS over one shared LMDB.
    #
    # Uni-Core's LMDB dataset holds its environment open for the life of the process and
    # lmdb refuses a second open of the same path, so calling ``predict_pkas`` twice in
    # one process dies on the second fold with "The environment ... is already open in
    # this process". Each fold therefore runs as a subprocess of this same command with
    # ``--single-fold-out``; the LMDB is built by whichever fold runs first and read by
    # the rest, since ``overwrite_lmdb`` is False.
    t_all = time.time()
    if args.single_fold_out is not None:
        if len(args.folds) != 1:
            raise ValueError("--single-fold-out takes exactly one --folds value")
        config = PredictionConfig(
            processed_lmdb_dir=osp.join(work, "lmdb"),
            results_dir=osp.join(work, "results"),
            model_dir=args.model_dir,
            fold=args.folds[0],
            batch_size=args.batch_size,
            nthreads=args.threads,
            num_workers=0,
            overwrite_lmdb=False,
        )
        result = predict_pkas(infer_in, task_name="unipka_rungs", config=config)
        with open(args.single_fold_out, "w") as handle:
            json.dump(
                {
                    str(int(round(target))): float(predict)
                    for target, predict in zip(result["target"], result["predict"])
                },
                handle,
            )
        print(f"wrote {args.single_fold_out}: {len(result)} rungs")
        return

    per_fold: dict[int, dict[int, float]] = {}
    for fold in args.folds:
        fold_out = osp.join(work, f"unipka_fold_{fold}.json")
        command = [
            sys.executable,
            osp.abspath(__file__),
            "predict-unipka",
            "--enumerator",
            args.enumerator,
            "--model-dir",
            args.model_dir,
            "--work-dir",
            work,
            "--folds",
            str(fold),
            "--charge-upper",
            str(args.charge_upper),
            "--charge-lower",
            str(args.charge_lower),
            "--template",
            args.template,
            "--batch-size",
            str(args.batch_size),
            "--threads",
            str(args.threads),
            "--single-fold-out",
            fold_out,
        ]
        if args.limit:
            command += ["--limit", str(args.limit)]
        print(f"fold {fold}: {' '.join(command)}", flush=True)
        proc = subprocess.run(command)
        if proc.returncode != 0 or not osp.isfile(fold_out):
            raise RuntimeError(f"fold {fold} failed with code {proc.returncode}")
        with open(fold_out) as handle:
            per_fold[fold] = {int(k): float(v) for k, v in json.load(handle).items()}
        print(f"  fold {fold}: {len(per_fold[fold])} of {len(frame)} rungs", flush=True)

    # 5. The ladder per compound: the fold-averaged macro pKa of every rung that ran.
    ladders: dict[int, list[float]] = {c.compound_id: [] for c in compounds}
    all_values: dict[int, list[float]] = {c.compound_id: [] for c in compounds}
    n_missing = 0
    for key, row in frame.iterrows():
        index = int(row["TARGET"])
        values = [
            per_fold[fold][index] for fold in args.folds if index in per_fold[fold]
        ]
        if len(values) < len(args.folds):
            n_missing += 1
        if not values:
            continue
        macro = float(sum(values) / len(values))
        compound_id = int(row["compound_id"])
        all_values[compound_id].append(macro)
        if MIN_PH < macro < MAX_PH:
            ladders[compound_id].append(macro)
        _ = key

    records = [
        PredictionRecord(
            compound_id=compound.compound_id,
            tool=args.tool_name,
            ok=bool(all_values[compound.compound_id]),
            error=(
                None
                if all_values[compound.compound_id]
                else "no macrostate pair survived enumeration and inference"
            ),
            predicted_pkas=sorted(ladders[compound.compound_id], reverse=True),
            site_pkas=sorted(all_values[compound.compound_id], reverse=True),
        )
        for compound in compounds
    ]
    pset = PredictionSet(
        tool=args.tool_name,
        tool_version=args.tool_version,
        built_at=_timestamp(),
        n_requested=len(compounds),
        n_ok=sum(1 for r in records if r.ok),
        seconds_total=time.time() - t_all,
        records=records,
    )
    path = PRED_PATH_TEMPLATE.format(tool=args.tool_name)
    with open(path, "w") as handle:
        handle.write(pset.model_dump_json(indent=1))
    print(
        f"wrote {path}: ok {pset.n_ok}/{pset.n_requested} in {pset.seconds_total:.0f}s,"
        f" {n_missing} rungs short of a full fold set"
    )


# ---------------------------------------------------------------------------- score


def _transform_kj_per_mol(
    atom_bag: dict[str, int], pkas: list[float], reference_n_h: int, p_h: float = P_H
) -> float | None:
    """The Legendre transform at the study conditions, referenced to ``reference_n_h``.

    ``p_h`` defaults to the study condition, pH 7.2. The method-selection benchmark calls
    the same function at each compartment pH the flux layer actually runs at, because the
    transform is what a pKa error propagates through and its sensitivity to a pKa is
    largest where that pKa sits near the pH.

    Built with the same function the compound-creation path uses, then re-referenced so
    that a predicted pKa set and the cache's pKa set are compared on one energy zero.
    The reference is the cache's major microspecies, whose untransformed formation energy
    the component-contribution model supplies and which the pKas do not change.
    """
    import numpy as np
    from equilibrator_assets.thermodynamics import create_microspecies_mappings
    from equilibrator_cache.models import Compound, CompoundMicrospecies
    from equilibrator_cache.thermodynamic_constants import R, default_T

    compound = Compound(
        id=-1,
        atom_bag=dict(atom_bag),
        dissociation_constants=list(pkas),
        inchi_key="BENCHMARK-STANDIN-N",
    )
    compound.magnesium_dissociation_constants = []
    mappings = create_microspecies_mappings(compound, mid_ph=MID_PH)
    by_n_h = {m["number_protons"]: m for m in mappings}
    if reference_n_h not in by_n_h:
        return None
    offset = by_n_h[reference_n_h]["ddg_over_rt"]

    ddg_over_rt = []
    for mapping in mappings:
        micro = CompoundMicrospecies(
            compound_id=-1,
            charge=mapping["charge"],
            number_protons=mapping["number_protons"],
            number_magnesiums=mapping["number_magnesiums"],
            is_major=mapping["is_major"],
            ddg_over_rt=mapping["ddg_over_rt"] - offset,
        )
        # Compound._get_ms_ddg_over_rt negates each microspecies transform before the
        # log-sum-exp, so the same negation is applied here.
        ddg_over_rt.append(
            -micro.transform(
                pH=p_h,
                pMg=P_MG,
                ionic_strength_M=IONIC_STRENGTH_M,
                T_in_K=TEMPERATURE_K,
            )
        )
    rt = float(R.m_as("kJ/mol/K")) * TEMPERATURE_K
    _ = default_T
    return -rt * float(np.logaddexp.reduce(np.asarray(ddg_over_rt)))


def _paired_errors(cache: list[float], pred: list[float]) -> tuple[float, float] | None:
    """MAE and max error over the pKas paired by rank, on the shorter list's length."""
    n = min(len(cache), len(pred))
    if n == 0:
        return None
    diffs = [abs(cache[i] - pred[i]) for i in range(n)]
    return sum(diffs) / n, max(diffs)


def _summarize(rows: list[ScoreRow], tool: str) -> ToolSummary:
    import numpy as np

    ok_rows = [r for r in rows if r.ok]
    mae_values = [r.pka_mae for r in ok_rows if r.pka_mae is not None]
    ddgf_values = [
        abs(r.ddgf_prime_kj_per_mol)
        for r in ok_rows
        if r.ddgf_prime_kj_per_mol is not None
    ]

    def _block(subset: list[ScoreRow]) -> dict[str, float | int | None]:
        sub_ok = [r for r in subset if r.ok]
        sub_mae = [r.pka_mae for r in sub_ok if r.pka_mae is not None]
        sub_ddgf = [
            abs(r.ddgf_prime_kj_per_mol)
            for r in sub_ok
            if r.ddgf_prime_kj_per_mol is not None
        ]
        return {
            "n": len(subset),
            "n_ok": len(sub_ok),
            "n_count_match": sum(1 for r in sub_ok if r.count_matches),
            "frac_count_match": (
                sum(1 for r in sub_ok if r.count_matches) / len(sub_ok)
                if sub_ok
                else None
            ),
            "pka_mae": float(np.mean(sub_mae)) if sub_mae else None,
            "ddgf_mae_kj_per_mol": float(np.mean(sub_ddgf)) if sub_ddgf else None,
        }

    strata = sorted({r.stratum for r in rows})
    groups = sorted({g for r in rows for g in r.groups})
    return ToolSummary(
        tool=tool,
        n_attempted=len(rows),
        n_ok=len(ok_rows),
        frac_ok=len(ok_rows) / len(rows) if rows else 0.0,
        n_count_match=sum(1 for r in ok_rows if r.count_matches),
        frac_count_match=(
            sum(1 for r in ok_rows if r.count_matches) / len(ok_rows)
            if ok_rows
            else 0.0
        ),
        pka_mae_paired=float(np.mean(mae_values)) if mae_values else None,
        pka_mae_n=len(mae_values),
        ddgf_mae_kj_per_mol=float(np.mean(ddgf_values)) if ddgf_values else None,
        ddgf_median_abs_kj_per_mol=(
            float(np.median(ddgf_values)) if ddgf_values else None
        ),
        ddgf_p90_abs_kj_per_mol=(
            float(np.percentile(ddgf_values, 90)) if ddgf_values else None
        ),
        ddgf_n=len(ddgf_values),
        by_stratum={s: _block([r for r in rows if r.stratum == s]) for s in strata},
        by_group={g: _block([r for r in rows if g in r.groups]) for g in groups},
    )


def cmd_score(args: argparse.Namespace) -> None:
    """Score every prediction file present against the cache pKas."""
    import csv

    vset = _load_sample()
    by_id = {c.compound_id: c for c in vset.compounds}

    all_rows: list[ScoreRow] = []
    summaries: list[ToolSummary] = []
    for tool_key in args.tools:
        path = PRED_PATH_TEMPLATE.format(tool=tool_key)
        if not osp.isfile(path):
            print(f"skip {tool_key}: no {path}")
            continue
        with open(path) as handle:
            pset = PredictionSet.model_validate_json(handle.read())

        rows: list[ScoreRow] = []
        for record in pset.records:
            compound = by_id[record.compound_id]
            reference_n_h = compound.formula_atom_bag.get("H", 0)
            cache_transform = _transform_kj_per_mol(
                compound.formula_atom_bag, compound.cache_pkas, reference_n_h
            )
            pred_pkas = sorted(record.predicted_pkas, reverse=True)
            pred_transform = (
                _transform_kj_per_mol(
                    compound.formula_atom_bag, pred_pkas, reference_n_h
                )
                if record.ok
                else None
            )
            paired = _paired_errors(compound.cache_pkas, pred_pkas)
            ddgf = (
                pred_transform - cache_transform
                if (pred_transform is not None and cache_transform is not None)
                else None
            )
            rows.append(
                ScoreRow(
                    compound_id=record.compound_id,
                    tool=pset.tool,
                    inchi_key=compound.inchi_key,
                    stratum=compound.stratum,
                    groups=compound.groups,
                    n_cache_pkas=compound.n_cache_pkas,
                    n_pred_pkas=len(pred_pkas),
                    count_matches=len(pred_pkas) == compound.n_cache_pkas,
                    cache_pkas=compound.cache_pkas,
                    predicted_pkas=pred_pkas,
                    pka_mae=paired[0] if paired else None,
                    pka_max_abs_error=paired[1] if paired else None,
                    ddgf_prime_kj_per_mol=ddgf,
                    ok=record.ok and bool(pred_pkas),
                    error=record.error,
                )
            )
        all_rows.extend(rows)
        summaries.append(_summarize(rows, pset.tool))

    result = BenchmarkResult(
        built_at=_timestamp(),
        validation_set_path=SAMPLE_PATH,
        n_validation_compounds=vset.n_compounds,
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        magnesium_gap={
            "n_sampled_with_magnesium_constants": vset.n_with_magnesium_constants,
            "note": (
                "specified_pkas carries proton dissociation constants only, so a"
                " compound built through the external-pKa path has no magnesium"
                " microspecies; both transforms here are proton-only"
            ),
        },
        tools=summaries,
    )
    with open(BENCHMARK_PATH, "w") as handle:
        handle.write(result.model_dump_json(indent=1))

    with open(PER_COMPOUND_CSV, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tool",
                "compound_id",
                "inchi_key",
                "stratum",
                "groups",
                "n_cache_pkas",
                "n_pred_pkas",
                "count_matches",
                "cache_pkas",
                "predicted_pkas",
                "pka_mae",
                "pka_max_abs_error",
                "ddgf_prime_kj_per_mol",
                "ok",
                "error",
            ]
        )
        for row in all_rows:
            writer.writerow(
                [
                    row.tool,
                    row.compound_id,
                    row.inchi_key,
                    row.stratum,
                    "|".join(row.groups),
                    row.n_cache_pkas,
                    row.n_pred_pkas,
                    row.count_matches,
                    ";".join(f"{p:.2f}" for p in row.cache_pkas),
                    ";".join(f"{p:.2f}" for p in row.predicted_pkas),
                    "" if row.pka_mae is None else f"{row.pka_mae:.4f}",
                    ""
                    if row.pka_max_abs_error is None
                    else f"{row.pka_max_abs_error:.4f}",
                    ""
                    if row.ddgf_prime_kj_per_mol is None
                    else f"{row.ddgf_prime_kj_per_mol:.4f}",
                    row.ok,
                    row.error or "",
                ]
            )

    print(f"wrote {BENCHMARK_PATH} and {PER_COMPOUND_CSV}")
    for summary in summaries:
        print(
            f"{summary.tool}: ok {summary.n_ok}/{summary.n_attempted},"
            f" count match {summary.frac_count_match:.3f},"
            f" pKa MAE {summary.pka_mae_paired},"
            f" |ddGf'| MAE {summary.ddgf_mae_kj_per_mol} kJ/mol"
        )


# --------------------------------------------------------------- microspecies convention


class MicrospeciesTarget(BaseModel):
    """One structure in two protonation conventions, with the pKas that place it."""

    key: str
    name: str
    source: str
    input_smiles: str
    ph7_smiles: str | None = None
    ph7_charge: int | None = None
    input_charge: int | None = None
    pkas: list[float] = Field(default_factory=list)
    cache_compound_id: int | None = None
    cache_smiles: str | None = None
    error: str | None = None


class MicrospeciesSet(BaseModel):
    """The pH 7 states of every target the external route creates."""

    built_at: str
    p_h: float
    tool: str
    targets: list[MicrospeciesTarget]


def cmd_microspecies_at_ph(args: argparse.Namespace) -> None:
    """Build each target's pH 7 major microspecies, which is the convention the cache uses.

    The bridge takes the caller's SMILES AS the reference microspecies: ``get_atom_bag``
    reads the proton count straight off that structure and the group decomposer runs on it
    too. The cache stores the major microspecies at pH 7, so handing the bridge an
    un-ionized neutral parent creates a compound referenced to a different microstate of
    the same molecule. For a phosphatidylethanolamine the cache holds
    ``COP([O-])(=O)OCC[NH3+]`` and a structure writer naturally emits ``COP(=O)(O)OCCN``:
    same formula, same net charge, different microspecies, and a different group vector.

    The pH 7 state is not constructed here by rule. pKasolver walks its own protonation
    ladder and every ``States`` record it returns carries ``ph7_mol``, the state the tool
    itself places at pH 7, so the state and the pKas come from one source.
    """
    import time

    from pkasolver.query import QueryModel, calculate_microstate_pka_values
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")

    targets: list[MicrospeciesTarget] = []
    if args.pilot and osp.isfile(args.pilot):
        with open(args.pilot) as handle:
            pilot = json.load(handle)
        for row in pilot["targets"]:
            if row.get("route") != "external_pka" or not row.get("smiles"):
                continue
            targets.append(
                MicrospeciesTarget(
                    key=row["key"],
                    name=row["name"],
                    source="pilot",
                    input_smiles=row["smiles"],
                    pkas=[float(p) for p in row.get("predicted_pkas", [])],
                )
            )
    if args.lipids and osp.isfile(args.lipids):
        with open(args.lipids) as handle:
            lipids = json.load(handle)
        for row in lipids["records"]:
            # Only the pairs the lipid run priced BOTH ways carry a residual to close.
            if not row.get("cache_same_protonation_state"):
                continue
            if row.get("dgf_cache_minus_external_kj_per_mol") is None:
                continue
            targets.append(
                MicrospeciesTarget(
                    key=f"lipid:{row['name']}",
                    name=row["name"],
                    source="lipid",
                    input_smiles=row["smiles"],
                    pkas=[float(p) for p in row.get("pkas", [])],
                    cache_compound_id=row.get("cache_compound_id"),
                )
            )

    model = QueryModel()
    t0 = time.time()
    for target in targets:
        mol = Chem.MolFromSmiles(target.input_smiles)
        if mol is None:
            target.error = "rdkit could not parse the input SMILES"
            continue
        target.input_charge = Chem.GetFormalCharge(mol)
        states = calculate_microstate_pka_values(mol, query_model=model)
        if not states:
            target.error = "pkasolver returned no protonation state"
            continue
        ph7 = states[0].ph7_mol
        if ph7 is None:
            target.error = "pkasolver returned no pH 7 state"
            continue
        target.ph7_smiles = Chem.MolToSmiles(ph7)
        target.ph7_charge = Chem.GetFormalCharge(ph7)
        if not target.pkas:
            target.pkas = sorted(
                (float(s.pka) for s in states if MIN_PH < float(s.pka) < MAX_PH),
                reverse=True,
            )
        print(
            f"  {target.key}: charge {target.input_charge} -> {target.ph7_charge}",
            flush=True,
        )

    result = MicrospeciesSet(
        built_at=_timestamp(), p_h=7.0, tool="pkasolver-lite", targets=targets
    )
    path = osp.join(RESULTS, "external_pka_microspecies.json")
    with open(path, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {path}: {len(targets)} targets in {time.time() - t0:.0f}s")


class RepricedTarget(BaseModel):
    """One target priced from both protonation conventions."""

    key: str
    name: str
    source: str
    input_smiles: str
    ph7_smiles: str | None
    pkas: list[float]
    dgf_prime_from_input_kj_per_mol: float | None = None
    dgf_prime_from_ph7_kj_per_mol: float | None = None
    dgf_prime_cache_kj_per_mol: float | None = None
    residual_input_kj_per_mol: float | None = None
    residual_ph7_kj_per_mol: float | None = None
    convention_shift_kj_per_mol: float | None = None
    detail: str = ""


class RepriceResult(BaseModel):
    """What submitting the cache's protonation convention is worth."""

    built_at: str
    conditions: dict[str, float]
    local_cache_path: str
    local_cache_ph7_path: str
    n_targets: int
    n_priced_both_ways: int
    residual_input_summary: dict[str, float]
    residual_ph7_summary: dict[str, float]
    benchmark_reference_sensitivity: dict[str, float]
    targets: list[RepricedTarget]


def cmd_reprice_microspecies(args: argparse.Namespace) -> None:
    """Price every target from both conventions and report what the switch is worth.

    Two numbers per target: ``dGf'0`` when the un-ionized structure is submitted, and
    ``dGf'0`` when the pH 7 major microspecies is. Where the compound is also in the cache,
    the residual against the cache's own value is reported for both, so the closure is
    measured rather than argued.

    The 300-compound accuracy benchmark is a separate question and is answered here too.
    That benchmark never creates a compound: :func:`_transform_kj_per_mol` rebuilds only
    the Legendre transform, re-references both the cache's pKas and the predicted pKas to
    the SAME microspecies, and subtracts. The untransformed group-contribution term, which
    is where the whole lipid offset sits, is not in it on either side. The sensitivity
    block moves the reference proton count and reports how far the scored differences move
    with it.
    """
    import shutil

    import numpy as np
    from equilibrator_api import Q_, ComponentContribution
    from equilibrator_assets.local_compound_cache import LocalCompoundCache
    from equilibrator_cache.models import Compound

    with open(osp.join(RESULTS, "external_pka_microspecies.json")) as handle:
        micro = MicrospeciesSet.model_validate_json(handle.read())

    # Two writable caches, one per convention, and this is not tidiness. The bridge
    # deduplicates by standard InChIKey, and standard InChI normalizes mobile protons, so
    # the un-ionized parent and the zwitterion of a phosphatidylethanolamine hash to the
    # SAME key (CZOSTDZGCCEZTJ-PEZBUJJGSA-N) with the same atom bag. Submit both to one
    # cache and the second call returns the compound the first created, which reports a
    # zero difference that is an artifact of the lookup. Their GROUP VECTORS do differ, in
    # two positions, so the conventions have to be created in caches that cannot see each
    # other.
    for path in (args.local_cache, args.local_cache_ph7):
        if args.seed_cache and not osp.isfile(path):
            os.makedirs(osp.dirname(path), exist_ok=True)
            shutil.copy2(args.seed_cache, path)
            print(f"seeded {path} from {args.seed_cache}")

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    local = LocalCompoundCache(pkas_provided_externally=True)
    local.load_cache(args.local_cache)
    local_ph7 = LocalCompoundCache(pkas_provided_externally=True)
    local_ph7.load_cache(args.local_cache_ph7)

    def _price(compound) -> float | None:
        mu, _sigma, _ = cc.standard_dg_formation(compound)
        if mu is None:
            return None
        correction = compound.transform(
            Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
        )
        return float(mu) + float(correction.m_as("kJ/mol"))

    def _create_and_price(
        smiles: str, pkas: list[float], cache: LocalCompoundCache
    ) -> tuple[float | None, str]:
        results = cache.get_compounds(
            [smiles],
            mol_format="smiles",
            specified_pkas={smiles: pkas},
            bypass_chemaxon=True,
        )
        compound = results[0].compound if results else None
        if compound is None:
            return None, f"method={results[0].method} status={results[0].status}"
        if compound.group_vector is None:
            return None, "failed group decomposition"
        return _price(compound), f"compound_id={compound.id}"

    rows: list[RepricedTarget] = []
    for target in micro.targets:
        row = RepricedTarget(
            key=target.key,
            name=target.name,
            source=target.source,
            input_smiles=target.input_smiles,
            ph7_smiles=target.ph7_smiles,
            pkas=target.pkas,
        )
        if not target.pkas:
            # The interface guard is truthiness, not presence, so an empty list means
            # "no pKa in (0, 14)" and raises instead of creating a single-microspecies
            # compound. Superoxide is that case and it is a property of the bridge.
            row.detail = "no pKa in (0, 14); the bridge cannot express an empty list"
            rows.append(row)
            print(f"{target.key}: skipped, {row.detail}", flush=True)
            continue
        if target.cache_compound_id is not None:
            cached = (
                local.ccache.session.query(Compound)
                .filter(Compound.id == target.cache_compound_id)
                .one_or_none()
            )
            if cached is not None:
                row.dgf_prime_cache_kj_per_mol = _price(cached)
        value, detail = _create_and_price(target.input_smiles, target.pkas, local)
        row.dgf_prime_from_input_kj_per_mol = value
        row.detail = f"input {detail}"
        if target.ph7_smiles:
            value7, detail7 = _create_and_price(
                target.ph7_smiles, target.pkas, local_ph7
            )
            row.dgf_prime_from_ph7_kj_per_mol = value7
            row.detail += f"; ph7 {detail7}"
        if (
            row.dgf_prime_from_input_kj_per_mol is not None
            and row.dgf_prime_from_ph7_kj_per_mol is not None
        ):
            row.convention_shift_kj_per_mol = (
                row.dgf_prime_from_ph7_kj_per_mol - row.dgf_prime_from_input_kj_per_mol
            )
        if row.dgf_prime_cache_kj_per_mol is not None:
            if row.dgf_prime_from_input_kj_per_mol is not None:
                row.residual_input_kj_per_mol = (
                    row.dgf_prime_cache_kj_per_mol - row.dgf_prime_from_input_kj_per_mol
                )
            if row.dgf_prime_from_ph7_kj_per_mol is not None:
                row.residual_ph7_kj_per_mol = (
                    row.dgf_prime_cache_kj_per_mol - row.dgf_prime_from_ph7_kj_per_mol
                )
        rows.append(row)
        print(
            f"{target.key}: input {row.dgf_prime_from_input_kj_per_mol},"
            f" ph7 {row.dgf_prime_from_ph7_kj_per_mol},"
            f" cache {row.dgf_prime_cache_kj_per_mol}",
            flush=True,
        )

    def _summary(values: list[float]) -> dict[str, float]:
        if not values:
            return {"n": 0.0}
        array = np.asarray(values)
        return {
            "n": float(array.size),
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean_abs": float(np.abs(array).mean()),
        }

    # The 300-compound benchmark, answered with a number rather than an argument.
    #
    # That benchmark never creates a compound, so it never re-decomposes a structure and
    # the group-contribution term where the whole lipid offset sits is absent from both
    # sides of every scored difference. What it does have is a reference microspecies, and
    # the analogue of the lipid mistake inside it would be anchoring the ladder to the
    # NEUTRAL PARENT the predictors were fed rather than to the cache's own atom bag. This
    # scores MolGpKa's real predictions both ways and reports how far they move.
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    vset = _load_sample()
    by_id = {c.compound_id: c for c in vset.compounds}
    molgpka_path = PRED_PATH_TEMPLATE.format(tool="molgpka")
    changes: list[float] = []
    n_reference_differs = 0
    if osp.isfile(molgpka_path):
        with open(molgpka_path) as handle:
            pset = PredictionSet.model_validate_json(handle.read())
        for record in pset.records:
            if not record.ok or not record.predicted_pkas:
                continue
            compound = by_id[record.compound_id]
            mol = Chem.MolFromSmiles(compound.neutral_smiles)
            if mol is None:
                continue
            neutral_h = sum(
                atom.GetTotalNumHs() + (1 if atom.GetSymbol() == "H" else 0)
                for atom in mol.GetAtoms()
            )
            cache_h = compound.formula_atom_bag.get("H", 0)
            if neutral_h != cache_h:
                n_reference_differs += 1
            predicted = sorted(record.predicted_pkas, reverse=True)
            scored = []
            for reference in (cache_h, neutral_h):
                cache_transform = _transform_kj_per_mol(
                    compound.formula_atom_bag, compound.cache_pkas, reference
                )
                pred_transform = _transform_kj_per_mol(
                    compound.formula_atom_bag, predicted, reference
                )
                if cache_transform is None or pred_transform is None:
                    scored = []
                    break
                scored.append(pred_transform - cache_transform)
            if len(scored) == 2:
                changes.append(abs(scored[0] - scored[1]))

    result = RepriceResult(
        built_at=_timestamp(),
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        local_cache_path=args.local_cache,
        local_cache_ph7_path=args.local_cache_ph7,
        n_targets=len(rows),
        n_priced_both_ways=sum(
            1
            for r in rows
            if r.dgf_prime_from_input_kj_per_mol is not None
            and r.dgf_prime_from_ph7_kj_per_mol is not None
        ),
        residual_input_summary=_summary(
            [r.residual_input_kj_per_mol for r in rows if r.residual_input_kj_per_mol]
        ),
        residual_ph7_summary=_summary(
            [
                r.residual_ph7_kj_per_mol
                for r in rows
                if r.residual_ph7_kj_per_mol is not None
            ]
        ),
        benchmark_reference_sensitivity={
            "n": float(len(changes)),
            "n_reference_differs": float(n_reference_differs),
            "max_abs_change_kj_per_mol": float(max(changes)) if changes else 0.0,
            "mean_abs_change_kj_per_mol": float(np.mean(changes)) if changes else 0.0,
        },
        targets=rows,
    )
    path = osp.join(RESULTS, "external_pka_reprice.json")
    with open(path, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {path}")
    print()
    print(
        "| target | source | dGf'0 from input | dGf'0 from pH 7 ms |"
        " cache dGf'0 | residual input | residual pH 7 |"
    )
    print("|---|---|---|---|---|---|---|")
    for row in rows:
        print(
            f"| {row.name[:44]} | {row.source} |"
            f" {_fmt(row.dgf_prime_from_input_kj_per_mol)} |"
            f" {_fmt(row.dgf_prime_from_ph7_kj_per_mol)} |"
            f" {_fmt(row.dgf_prime_cache_kj_per_mol)} |"
            f" {_fmt(row.residual_input_kj_per_mol)} |"
            f" {_fmt(row.residual_ph7_kj_per_mol)} |"
        )
    print()
    sensitivity = result.benchmark_reference_sensitivity
    print(
        f"benchmark reference: {sensitivity['n_reference_differs']:.0f} of"
        f" {sensitivity['n']:.0f} scored compounds have a neutral parent whose proton"
        " count differs from the cache atom bag; scoring MolGpKa against the neutral"
        f" parent instead moves dGf'0 by at most"
        f" {sensitivity['max_abs_change_kj_per_mol']:.4f} kJ/mol, mean"
        f" {sensitivity['mean_abs_change_kj_per_mol']:.4f}"
    )


# ------------------------------------------------------------------------ magnesium


class MagnesiumCost(BaseModel):
    """What omitting one compound's Mg constants costs its transformed energy."""

    compound_id: int
    inchi_key: str
    name: str
    stratum: str
    n_mg_constants: int
    mg_constants: list[dict[str, float]]
    n_microspecies_total: int
    n_microspecies_with_mg: int
    dgf_prime_full_kj_per_mol: dict[str, float]
    dgf_prime_no_mg_kj_per_mol: dict[str, float]
    omission_cost_kj_per_mol: dict[str, float]
    wrong_pkd_cost_kj_per_mol: dict[str, float]


class MagnesiumBucket(BaseModel):
    """The spread of Mg pKd within one substructure bucket of the shipped table."""

    n_phosphate_steps: int
    n_constants: int
    n_compounds: int
    pkd_mean: float
    pkd_sd: float
    pkd_min: float
    pkd_max: float


class MagnesiumReport(BaseModel):
    """The magnesium gap of the external-pKa route, measured rather than asserted."""

    built_at: str
    conditions: dict[str, float]
    p_mg_values: list[float]
    n_cache_compounds: int
    n_cache_compounds_with_mg: int
    mg_table_path: str
    mg_table_sha256: str
    mg_table_rows: int
    mg_table_distinct_compounds: int
    mg_table_references: dict[str, int]
    n_sample: int
    n_sample_with_mg: int
    n_sample_with_phosphate: int
    costs: list[MagnesiumCost]
    substructure_buckets: list[MagnesiumBucket]


def cmd_magnesium(args: argparse.Namespace) -> None:
    """Measure what the external-pKa route loses by not carrying Mg constants.

    The route cannot supply them: ``specified_pkas`` is proton-only, and
    ``Compound.transform`` reads the microspecies table rather than the constants, so a
    constant written onto a compound after creation is thermodynamically inert until the
    microspecies are regenerated from it. This measures the size of that gap on the
    sampled compounds that do carry constants, and tests whether the shipped table could
    be extended to new structures by substructure, by asking how tightly the shipped pKds
    cluster within a substructure bucket.
    """
    import hashlib

    import numpy as np
    from equilibrator_cache.api import create_compound_cache_from_zenodo
    from equilibrator_cache.models import Compound, MagnesiumDissociationConstant
    from equilibrator_cache.thermodynamic_constants import R
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    ccache = create_compound_cache_from_zenodo()
    session = ccache.session
    vset = _load_sample()
    p_mg_values = [float(v) for v in args.p_mg]
    rt = float(R.m_as("kJ/mol/K")) * TEMPERATURE_K

    from equilibrator_cache.thermodynamic_constants import LOG10, default_T_in_K

    def _dgf(
        compound: Compound, p_mg: float, with_mg: bool, p_kd_shift: float = 0.0
    ) -> float:
        """``dGf'0`` from the microspecies, optionally without Mg or with shifted pKds.

        A Mg microspecies is built at compound-creation time as
        ``ddg_over_rt = ref + dGf_Mg/RT0 - pKd * ln10``, so moving every pKd by
        ``p_kd_shift`` moves that microspecies' stored ``ddg_over_rt`` by
        ``-p_kd_shift * ln10 * number_magnesiums``, and moves its returned transform by
        that times ``T0/T``. Applying the shift to the returned value rather than to the
        ORM object keeps the session clean.
        """
        species = [
            ms for ms in compound.microspecies if with_mg or ms.number_magnesiums == 0
        ]
        values = [
            -(
                ms.transform(
                    pH=P_H,
                    pMg=p_mg,
                    ionic_strength_M=IONIC_STRENGTH_M,
                    T_in_K=TEMPERATURE_K,
                )
                - p_kd_shift
                * LOG10
                * ms.number_magnesiums
                * default_T_in_K
                / TEMPERATURE_K
            )
            for ms in species
        ]
        return -rt * float(np.logaddexp.reduce(np.asarray(values)))

    sampled_ids = [c.compound_id for c in vset.compounds if c.has_magnesium_constants]
    strata = {c.compound_id: c.stratum for c in vset.compounds}
    costs: list[MagnesiumCost] = []
    for compound_id in sampled_ids + list(args.extra_compound_ids):
        compound = session.query(Compound).filter(Compound.id == compound_id).one()
        constants = sorted(
            compound.magnesium_dissociation_constants,
            key=lambda m: (m.number_magnesiums, m.number_protons),
        )
        names = [
            i.accession
            for i in compound.identifiers
            if i.registry.namespace == "synonyms"
        ]
        full = {f"pmg_{p:g}": _dgf(compound, p, True) for p in p_mg_values}
        no_mg = {f"pmg_{p:g}": _dgf(compound, p, False) for p in p_mg_values}
        costs.append(
            MagnesiumCost(
                compound_id=compound.id,
                inchi_key=compound.inchi_key or "",
                name=names[0] if names else "",
                stratum=strata.get(compound_id, "not_sampled"),
                n_mg_constants=len(constants),
                mg_constants=[
                    {
                        "number_protons": float(m.number_protons),
                        "number_magnesiums": float(m.number_magnesiums),
                        "p_kd": float(m.dissociation_constant),
                    }
                    for m in constants
                ],
                n_microspecies_total=len(compound.microspecies),
                n_microspecies_with_mg=sum(
                    1 for ms in compound.microspecies if ms.number_magnesiums > 0
                ),
                dgf_prime_full_kj_per_mol=full,
                dgf_prime_no_mg_kj_per_mol=no_mg,
                omission_cost_kj_per_mol={k: no_mg[k] - full[k] for k in full},
                # What a substructure-derived pKd would cost if it were off by one
                # within-bucket standard deviation of the shipped table.
                wrong_pkd_cost_kj_per_mol={
                    f"pmg_{p:g}_shift_{sign * args.p_kd_shift:+g}": (
                        _dgf(compound, p, True, sign * args.p_kd_shift)
                        - full[f"pmg_{p:g}"]
                    )
                    for p in p_mg_values
                    for sign in (1, -1)
                },
            )
        )

    # Can the shipped table be carried to a new structure by substructure match? The
    # test is whether the pKd is a property of the substructure at all: bucket every
    # single-Mg constant in the table by the compound's count of ionizable phosphate
    # steps, the substructure any such rule would key on, and report the spread inside
    # each bucket. A rule can only be as accurate as that spread.
    rows = (
        session.query(MagnesiumDissociationConstant, Compound.smiles)
        .join(Compound, Compound.id == MagnesiumDissociationConstant.compound_id)
        .filter(MagnesiumDissociationConstant.number_magnesiums == 1)
        .all()
    )
    buckets: dict[int, list[float]] = {}
    bucket_compounds: dict[int, set[int]] = {}
    for constant, smiles in rows:
        if not smiles:
            continue
        counts, _ = _ionizable_centers(smiles)
        key = counts.get("phosphate", 0)
        buckets.setdefault(key, []).append(float(constant.dissociation_constant))
        bucket_compounds.setdefault(key, set()).add(constant.compound_id)

    substructure = [
        MagnesiumBucket(
            n_phosphate_steps=key,
            n_constants=len(values),
            n_compounds=len(bucket_compounds[key]),
            pkd_mean=float(np.mean(values)),
            pkd_sd=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            pkd_min=float(np.min(values)),
            pkd_max=float(np.max(values)),
        )
        for key, values in sorted(buckets.items())
    ]

    n_cache = session.query(Compound).count()
    n_cache_mg = (
        session.query(MagnesiumDissociationConstant.compound_id).distinct().count()
    )
    n_mg_rows = session.query(MagnesiumDissociationConstant).count()

    import csv as csv_module

    with open(args.mg_table, "rb") as handle:
        table_bytes = handle.read()
    # A real CSV reader, not a split on commas: several compound NAMES in this file
    # contain commas inside quotes, and splitting on the character mis-assigns their
    # reference field and inflates the distinct-compound count.
    table_rows = list(csv_module.DictReader(table_bytes.decode().splitlines()))
    references: dict[str, int] = {}
    for row in table_rows:
        reference = row["reference"]
        references[reference] = references.get(reference, 0) + 1

    report = MagnesiumReport(
        built_at=_timestamp(),
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
        },
        p_mg_values=p_mg_values,
        n_cache_compounds=n_cache,
        n_cache_compounds_with_mg=n_cache_mg,
        mg_table_path=args.mg_table,
        mg_table_sha256=hashlib.sha256(table_bytes).hexdigest(),
        mg_table_rows=len(table_rows),
        mg_table_distinct_compounds=len({row["compound_id"] for row in table_rows}),
        mg_table_references=references,
        n_sample=vset.n_compounds,
        n_sample_with_mg=vset.n_with_magnesium_constants,
        n_sample_with_phosphate=sum(
            1 for c in vset.compounds if "phosphate" in c.groups
        ),
        costs=costs,
        substructure_buckets=substructure,
    )
    path = osp.join(RESULTS, "external_pka_magnesium.json")
    with open(path, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {path}")
    print(
        f"cache: {n_cache_mg} of {n_cache} compounds carry Mg constants"
        f" ({n_mg_rows} constants); sample: {report.n_sample_with_mg} of"
        f" {report.n_sample}, against {report.n_sample_with_phosphate} phosphate-bearing"
    )
    print()
    header_cells = " | ".join(f"omit at pMg {p:g}" for p in p_mg_values)
    print(
        f"| compound | id | n Mg constants | {header_cells} |"
        f" pKd off by {args.p_kd_shift:g} at pMg {P_MG:g} |"
    )
    print("|---|---|---|" + "---|" * (len(p_mg_values) + 1))
    for cost in costs:
        cells = " | ".join(
            f"{cost.omission_cost_kj_per_mol[f'pmg_{p:g}']:.4f}" for p in p_mg_values
        )
        wrong = max(
            abs(v)
            for k, v in cost.wrong_pkd_cost_kj_per_mol.items()
            if k.startswith(f"pmg_{P_MG:g}_")
        )
        print(
            f"| {cost.name or cost.inchi_key} | {cost.compound_id} |"
            f" {cost.n_mg_constants} | {cells} | {wrong:.4f} |"
        )
    print()
    print(
        "| phosphate steps | n constants | n compounds | pKd mean | pKd sd | pKd range |"
    )
    print("|---|---|---|---|---|---|")
    for bucket in substructure:
        print(
            f"| {bucket.n_phosphate_steps} | {bucket.n_constants} |"
            f" {bucket.n_compounds} | {bucket.pkd_mean:.2f} | {bucket.pkd_sd:.2f} |"
            f" {bucket.pkd_min:.2f} to {bucket.pkd_max:.2f} |"
        )


# ---------------------------------------------------------------------------- check

# One entry per ionizable-center class: (name, SMARTS, index of the anchor atom in the
# match). The anchor is the atom that gains or loses the proton, and it is what makes the
# total a count of CENTERS rather than of pattern hits: two classes that describe the same
# atom (an imidazole nitrogen is both an aromatic N and part of an imidazole ring) collapse
# onto one anchor and are counted once. Phosphorus and sulfur oxyacids are not in this
# table because their step count is not one per match; it is read off the terminal-oxygen
# count in :func:`_ionizable_centers`.
IONIZABLE_SMARTS: list[tuple[str, str, int]] = [
    ("carboxylic_acid", "[CX3](=[OX1])[OX2H1,OX1-]", 2),
    (
        "amine",
        "[NX3;!$(N[C,S,P]=[O,S,N]);!$(N=*);!$([N+](=O)[O-]);!$(N[CX3]=[NX2]);!a]",
        0,
    ),
    ("aromatic_n", "[n]", 0),
    ("phenol", "[OX2H1,OX1-]c", 0),
    ("thiol", "[SX2H1,SX1-;!$(S=*)]", 0),
    ("imidazole", "c1cnc[nH]1", 0),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]", 1),
    ("enol", "[OX2H1,OX1-][CX3]=[CX3]", 0),
    ("hydroxamic", "[CX3](=[OX1])[NX3][OX2H1,OX1-]", 3),
    ("amide_nh", "[NX3;H1,H2;$(N[CX3]=[OX1,SX1]),$(N[CX3]=[NX2]);!a]", 0),
    ("sulfonamide_nh", "[NX3;H1,H2][SX4](=[OX1])(=[OX1])", 0),
    (
        "alcohol",
        "[OX2H1,OX1-;!$([OX2H1,OX1-]c);!$([OX2H1,OX1-][CX3]=[OX1]);"
        "!$([OX2H1,OX1-][CX3]=[CX3]);!$([OX2H1,OX1-][P,S,N])]",
        0,
    ),
    ("carbon_acid", "[CX4;H1,H2,H3]([CX3]=[OX1])[CX3]=[OX1]", 0),
    ("imide_nh", "[NX3;H1]([CX3]=[OX1])[CX3]=[OX1]", 0),
]


def _ionizable_centers(smiles: str) -> tuple[dict[str, int], int]:
    """Ionizable centers of a structure as ``(counts by class, total steps)``.

    The total is an UPPER BOUND on how many macroscopic pKas a molecule can have, not an
    estimate of how many it does have: every class that can ionize anywhere in the pH 0 to
    14 window is counted, including the amide and sulfonamide N-H that ChemAxon reports for
    activated amides and the 1,3-dicarbonyl carbon acids it reports for diketones. A
    prediction is only flagged when it exceeds this bound, so a generous bound is the
    conservative choice.

    A phosphorus oxyacid contributes one step per terminal oxygen beyond the first, which
    is what distinguishes a phosphate monoester (two steps) from a phosphodiester bridge
    (one), and a sulfur oxyacid one step per terminal oxygen beyond the doubly bonded pair.
    """
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}, 0
    # Count centers on the uncharged parent, so an input written as its major
    # microspecies is not scored short: an ammonium written [NH3+] is four-connected and
    # matches no amine pattern, though the center is there and titrates. The validation
    # set is already neutralized, so this is a no-op there and matters for the pilot
    # structures, which come from MetaNetX in their charged form. Unlike
    # :func:`_neutralize` this does not require the result to be net neutral, since a
    # metal complex such as ferroheme b never reaches zero.
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    steps: dict[int, int] = {}
    counts: dict[str, int] = {}

    def _add(name: str, anchor: int, n: int) -> None:
        if n <= 0:
            return
        counts[name] = counts.get(name, 0) + n
        steps[anchor] = max(steps.get(anchor, 0), n)

    for name, smarts, anchor_pos in IONIZABLE_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"bad SMARTS for {name}: {smarts}")
        for match in mol.GetSubstructMatches(patt, uniquify=True):
            _add(name, match[anchor_pos], 1)

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in ("P", "S"):
            continue
        oxygens = [nb for nb in atom.GetNeighbors() if nb.GetSymbol() == "O"]
        terminal = sum(1 for nb in oxygens if nb.GetDegree() == 1)
        if symbol == "P":
            _add("phosphate", atom.GetIdx(), max(0, terminal - 1))
        elif len(oxygens) >= 3:
            _add("sulfonate", atom.GetIdx(), max(0, terminal - 2))
        elif len(oxygens) == 2:
            _add("sulfinate", atom.GetIdx(), max(0, terminal - 1))
    return counts, sum(steps.values())


class CheckRow(BaseModel):
    """One structure, one pKa list, and whether the list is possible for the structure."""

    tool: str
    compound_id: int | str
    smiles: str
    n_centers: int
    centers: dict[str, int]
    n_pkas_in_window: int
    flag_over_count: bool
    flag_zero_with_centers: bool


class CheckSummary(BaseModel):
    """Flag rates for one tool over the compounds it returned a ladder for."""

    tool: str
    n_evaluated: int
    n_over_count: int
    n_zero_with_centers: int
    n_flagged: int
    frac_flagged: float
    mean_excess: float | None


class CheckResult(BaseModel):
    """The plausibility check over the validation set and the pilot targets."""

    built_at: str
    ph_window: list[float]
    n_validation_compounds: int
    reference_note: str
    tools: list[CheckSummary]
    pilot_rows: list[CheckRow]
    flagged_rows: list[CheckRow]


def cmd_check(args: argparse.Namespace) -> None:
    """Flag pKa lists that no structure could support, with no ground truth needed.

    The cache's own ChemAxon lists are run through the same test and reported as the
    ``chemaxon-cache`` row. That row is the counter's false-flag rate: ChemAxon is the
    reference the whole benchmark is scored against, so every compound where its list
    exceeds the SMARTS bound is a compound where the bound is too tight, not one where
    ChemAxon is wrong. A tool's flag rate is only interesting above that floor.
    """
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    vset = _load_sample()
    by_id = {c.compound_id: c for c in vset.compounds}
    centers_by_id = {
        c.compound_id: _ionizable_centers(c.neutral_smiles) for c in vset.compounds
    }

    def _row(
        tool: str,
        compound_id: int | str,
        smiles: str,
        centers: tuple[dict[str, int], int],
        pkas: list[float],
    ) -> CheckRow:
        in_window = [p for p in pkas if MIN_PH < p < MAX_PH]
        return CheckRow(
            tool=tool,
            compound_id=compound_id,
            smiles=smiles,
            n_centers=centers[1],
            centers=centers[0],
            n_pkas_in_window=len(in_window),
            flag_over_count=len(in_window) > centers[1],
            flag_zero_with_centers=len(in_window) == 0 and centers[1] > 0,
        )

    summaries: list[CheckSummary] = []
    flagged: list[CheckRow] = []

    cache_rows = [
        _row(
            "chemaxon-cache",
            c.compound_id,
            c.neutral_smiles,
            centers_by_id[c.compound_id],
            c.cache_pkas,
        )
        for c in vset.compounds
    ]
    tool_rows: list[tuple[str, list[CheckRow]]] = [("chemaxon-cache", cache_rows)]

    for tool_key in args.tools:
        path = PRED_PATH_TEMPLATE.format(tool=tool_key)
        if not osp.isfile(path):
            print(f"skip {tool_key}: no {path}")
            continue
        with open(path) as handle:
            pset = PredictionSet.model_validate_json(handle.read())
        rows = [
            _row(
                pset.tool,
                record.compound_id,
                by_id[record.compound_id].neutral_smiles,
                centers_by_id[record.compound_id],
                record.predicted_pkas,
            )
            for record in pset.records
            if record.ok
        ]
        tool_rows.append((pset.tool, rows))

    for tool, rows in tool_rows:
        over = [r for r in rows if r.flag_over_count]
        zero = [r for r in rows if r.flag_zero_with_centers]
        any_flag = [r for r in rows if r.flag_over_count or r.flag_zero_with_centers]
        flagged.extend(any_flag)
        summaries.append(
            CheckSummary(
                tool=tool,
                n_evaluated=len(rows),
                n_over_count=len(over),
                n_zero_with_centers=len(zero),
                n_flagged=len(any_flag),
                frac_flagged=len(any_flag) / len(rows) if rows else 0.0,
                mean_excess=(
                    sum(r.n_pkas_in_window - r.n_centers for r in over) / len(over)
                    if over
                    else None
                ),
            )
        )

    pilot_rows: list[CheckRow] = []
    if osp.isfile(args.pilot):
        with open(args.pilot) as handle:
            pilot = json.load(handle)
        for target in pilot["targets"]:
            smiles = target.get("smiles")
            if not smiles or not target.get("predicted_pkas"):
                continue
            pilot_rows.append(
                _row(
                    f"{pilot['tool']}/pilot",
                    target["key"],
                    smiles,
                    _ionizable_centers(smiles),
                    [float(p) for p in target["predicted_pkas"]],
                )
            )

    result = CheckResult(
        built_at=_timestamp(),
        ph_window=[MIN_PH, MAX_PH],
        n_validation_compounds=vset.n_compounds,
        reference_note=(
            "chemaxon-cache is the same test applied to the cache's own lists, so its"
            " flag rate is the SMARTS bound's false-flag rate rather than an error rate"
            " of ChemAxon"
        ),
        tools=summaries,
        pilot_rows=pilot_rows,
        flagged_rows=flagged,
    )
    path = osp.join(RESULTS, "external_pka_check.json")
    with open(path, "w") as handle:
        handle.write(result.model_dump_json(indent=1))
    print(f"wrote {path}")
    print()
    print(
        "| tool | n evaluated | over count | zero with centers | flagged |"
        " flag rate | mean excess pKas |"
    )
    print("|---|---|---|---|---|---|---|")
    for summary in summaries:
        print(
            f"| {summary.tool} | {summary.n_evaluated} | {summary.n_over_count} |"
            f" {summary.n_zero_with_centers} | {summary.n_flagged} |"
            f" {_fmt(summary.frac_flagged, 3)} | {_fmt(summary.mean_excess)} |"
        )
    print()
    print("**Pilot targets**")
    print()
    print("| target | centers | n centers | n pKas | flagged |")
    print("|---|---|---|---|---|")
    for row in pilot_rows:
        flag = "over count" if row.flag_over_count else ""
        if row.flag_zero_with_centers:
            flag = "zero with centers"
        centers = ", ".join(f"{k} {v}" for k, v in sorted(row.centers.items()))
        print(
            f"| {row.compound_id} | {centers} | {row.n_centers} |"
            f" {row.n_pkas_in_window} | {flag or 'no'} |"
        )


# ---------------------------------------------------------------------------- table


def _fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def cmd_table(args: argparse.Namespace) -> None:
    """Print the accuracy tables as markdown, straight from the scored JSON."""
    with open(BENCHMARK_PATH) as handle:
        result = BenchmarkResult.model_validate_json(handle.read())

    print(
        f"Scored at {result.built_at} over {result.n_validation_compounds} sampled"
        f" compounds at pH {result.conditions['p_h']:g},"
        f" I = {result.conditions['ionic_strength_m']:g} M,"
        f" T = {result.conditions['temperature_k']:g} K,"
        f" pMg {result.conditions['p_mg']:g}."
    )
    print()
    print(
        "| tool | n attempted | n scored | right pKa count | pKa MAE |"
        " dGf'0 MAE (kJ/mol) | dGf'0 median (kJ/mol) | dGf'0 p90 (kJ/mol) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for tool in result.tools:
        print(
            f"| {tool.tool} | {tool.n_attempted} | {tool.n_ok} |"
            f" {_fmt(tool.frac_count_match, 3)} | {_fmt(tool.pka_mae_paired)} |"
            f" {_fmt(tool.ddgf_mae_kj_per_mol)} |"
            f" {_fmt(tool.ddgf_median_abs_kj_per_mol)} |"
            f" {_fmt(tool.ddgf_p90_abs_kj_per_mol)} |"
        )

    for title, attr in (
        ("By number of cache pKas", "by_stratum"),
        ("By functional group", "by_group"),
    ):
        print()
        print(f"**{title}**")
        print()
        print(
            "| key | tool | n attempted | n scored | right pKa count | pKa MAE |"
            " dGf'0 MAE (kJ/mol) |"
        )
        print("|---|---|---|---|---|---|---|")
        keys = sorted({k for tool in result.tools for k in getattr(tool, attr)})
        for key in keys:
            for tool in result.tools:
                block = getattr(tool, attr).get(key)
                if block is None:
                    continue
                print(
                    f"| {key} | {tool.tool} | {block['n']} | {block['n_ok']} |"
                    f" {_fmt(block['frac_count_match'], 3)} |"
                    f" {_fmt(block['pka_mae'])} |"
                    f" {_fmt(block['ddgf_mae_kj_per_mol'])} |"
                )


def _add_shard_args(parser: argparse.ArgumentParser) -> None:
    """The round-robin split shared by the two single-process CPU predictors."""
    parser.add_argument(
        "--shard", type=int, default=0, help="this worker's index, 0-based"
    )
    parser.add_argument(
        "--n-shards",
        type=int,
        default=1,
        help="how many workers split the set; 1 writes the unsharded file",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="draw the validation set from the cache")
    p_sample.add_argument("--n", type=int, default=250)
    p_sample.add_argument("--seed", type=int, default=17)
    p_sample.add_argument("--max-mass", type=float, default=600.0)
    p_sample.add_argument("--scan-limit", type=int, default=400000)
    p_sample.set_defaults(func=cmd_sample)

    p_ps = sub.add_parser("predict-pkasolver", help="run pKasolver-lite")
    _add_shard_args(p_ps)
    p_ps.set_defaults(func=cmd_predict_pkasolver)

    p_merge = sub.add_parser(
        "merge-shards",
        help="concatenate one tool's shard files into one prediction file",
    )
    p_merge.add_argument("--tool", required=True, help="pkasolver or molgpka")
    p_merge.add_argument("--n-shards", type=int, required=True)
    p_merge.set_defaults(func=cmd_merge_shards)

    p_qk = sub.add_parser("predict-qupkake", help="run QupKake and build the ladder")
    p_qk.add_argument("--workers", type=int, default=16)
    p_qk.add_argument("--work-dir", type=str, default=None)
    p_qk.add_argument("--tool-version", type=str, default="qupkake@main")
    p_qk.add_argument(
        "--max-rungs",
        type=int,
        default=MAX_PKAS + 1,
        help="cap the ladder depth per direction; each rung costs a full xTB wave",
    )
    p_qk.add_argument(
        "--limit",
        type=int,
        default=None,
        help="keep at most this many compounds, spread evenly over the strata",
    )
    p_qk.add_argument(
        "--strata",
        nargs="+",
        default=None,
        help="restrict to these strata, e.g. n_pka_1 n_pka_2",
    )
    p_qk.set_defaults(func=cmd_predict_qupkake)

    p_sv = sub.add_parser(
        "salvage-qupkake",
        help="rebuild QupKake ladders from the rungs a stopped run left on disk",
    )
    p_sv.add_argument("--work-dir", required=True)
    p_sv.add_argument("--tool-version", type=str, default="qupkake@main")
    p_sv.add_argument("--max-rungs", type=int, default=MAX_PKAS + 1)
    p_sv.set_defaults(func=cmd_salvage_qupkake)

    p_mg = sub.add_parser("predict-molgpka", help="run MolGpKa and build the ladder")
    p_mg.add_argument(
        "--repo", required=True, help="path to a MolGpKa checkout (holds models/)"
    )
    p_mg.add_argument("--tool-version", type=str, default="MolGpKa@main")
    _add_shard_args(p_mg)
    p_mg.set_defaults(func=cmd_predict_molgpka)

    p_score = sub.add_parser("score", help="score the predictions against the cache")
    p_score.add_argument(
        "--tools",
        nargs="+",
        default=["pkasolver", "molgpka", "unipka", "qupkake", "qupkake-truncated"],
    )
    p_score.set_defaults(func=cmd_score)

    p_up = sub.add_parser(
        "predict-unipka", help="run Uni-pKa over enumerated protonation ensembles"
    )
    p_up.add_argument(
        "--enumerator",
        required=True,
        help="path to the Uni-pKa checkout's enumerator/ directory",
    )
    p_up.add_argument("--model-dir", required=True, help="dir holding fold_N/*.pt")
    p_up.add_argument("--work-dir", required=True)
    p_up.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p_up.add_argument("--charge-upper", type=int, default=2)
    p_up.add_argument("--charge-lower", type=int, default=-4)
    p_up.add_argument("--template", type=str, default="simple_smarts_pattern.tsv")
    p_up.add_argument("--batch-size", type=int, default=16)
    p_up.add_argument("--threads", type=int, default=8)
    p_up.add_argument("--tool-version", type=str, default="unipkainfer 0.2.0")
    p_up.add_argument(
        "--tool-name",
        type=str,
        default="unipka",
        help="tool key of the prediction file; the enumeration template changes it",
    )
    p_up.add_argument(
        "--single-fold-out",
        type=str,
        default=None,
        help="internal: run one fold and dump its rung values, then exit",
    )
    p_up.add_argument(
        "--limit", type=int, default=None, help="smoke-test on the first N compounds"
    )
    p_up.set_defaults(func=cmd_predict_unipka)

    p_ms = sub.add_parser(
        "microspecies-at-ph",
        help="build each external-route target's pH 7 major microspecies",
    )
    p_ms.add_argument(
        "--pilot", type=str, default=osp.join(RESULTS, "external_pka_pilot.json")
    )
    p_ms.add_argument(
        "--lipids", type=str, default=osp.join(RESULTS, "lipid_priced.json")
    )
    p_ms.set_defaults(func=cmd_microspecies_at_ph)

    p_rp = sub.add_parser(
        "reprice-microspecies",
        help="price every target from both protonation conventions",
    )
    p_rp.add_argument(
        "--local-cache",
        required=True,
        help="writable cache for the submitted structure",
    )
    p_rp.add_argument(
        "--local-cache-ph7",
        required=True,
        help="a SECOND writable cache for the pH 7 microspecies; see the docstring",
    )
    p_rp.add_argument(
        "--seed-cache",
        type=str,
        default=None,
        help="copy this sqlite to --local-cache when it does not exist yet",
    )
    p_rp.set_defaults(func=cmd_reprice_microspecies)

    p_magnesium = sub.add_parser(
        "magnesium", help="measure the cost of omitting magnesium constants"
    )
    p_magnesium.add_argument(
        "--p-mg",
        nargs="+",
        type=float,
        default=[2.0, P_MG, 5.0],
        help="pMg values to price the omission at; the study condition is pMg 3",
    )
    p_magnesium.add_argument(
        "--extra-compound-ids",
        nargs="*",
        type=int,
        default=[10, 20],
        help="cache ids to price alongside the sample; 10 is ADP and 20 is UDP",
    )
    p_magnesium.add_argument(
        "--p-kd-shift",
        type=float,
        default=2.0,
        help="pKd error to price, in log units; 2 is the shipped table's within-bucket sd",
    )
    p_magnesium.add_argument(
        "--mg-table",
        type=str,
        required=True,
        help="path to equilibrator-assets' data/magnesium_pkds.csv",
    )
    p_magnesium.set_defaults(func=cmd_magnesium)

    p_check = sub.add_parser(
        "check", help="flag pKa lists that exceed the structure's ionizable centers"
    )
    p_check.add_argument(
        "--tools",
        nargs="+",
        default=["pkasolver", "molgpka", "unipka", "qupkake", "qupkake-truncated"],
    )
    p_check.add_argument(
        "--pilot",
        type=str,
        default=osp.join(RESULTS, "external_pka_pilot.json"),
        help="pilot result file whose targets are checked alongside the sample",
    )
    p_check.set_defaults(func=cmd_check)

    p_table = sub.add_parser("table", help="print the scored JSON as markdown tables")
    p_table.set_defaults(func=cmd_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
