# experiments/027-betaxanthin-metabolic/scripts/compare_thermo_sources.py
# [[experiments.027-betaxanthin-metabolic.wave8]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/027-betaxanthin-metabolic/scripts/compare_thermo_sources.py

r"""What changes in the reaction energies between two thermodynamic routes?

Wave 8 block 1 pairs the concentration head on ``union_external`` against the same head on
``verified``. A paired score difference is only interpretable next to what actually moved in
the table it is a contrast over, so this script diffs the two routes reaction by reaction
and reports, separately:

* the coverage change (how many reactions each route prices);
* every reaction whose ``delta_r_G'0`` differs, with the size of the change;
* the same restricted to the reactions that TOUCH one of the 19 measured amino acids, which
  are the ones the concentration readout is scored on;
* and the same for the betaxanthin route's precursors, for wave 8 block 3.

Run from the repo root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/027-betaxanthin-metabolic/scripts/compare_thermo_sources.py

Writes ``results/thermo_source_diff_union_external_vs_verified.json``.
"""

import json
import os
import os.path as osp
import sys

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

sys.path.insert(
    0, osp.join(os.environ["EXPERIMENT_ROOT"], "026-metabolism-flux", "scripts")
)
from train_flux import (  # noqa: E402
    AMINO_ACID_TO_GEM_METABOLITE,
    BETAXANTHIN_PRECURSORS,
)

from torchcell.metabolism.constraints import build_gem_tensors  # noqa: E402
from torchcell.metabolism.parameters import (  # noqa: E402
    ThermoSource,
    resolve_reaction_delta_g,
)
from torchcell.metabolism.yeast_GEM import YeastGEM  # noqa: E402

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "027-betaxanthin-metabolic", "results")
THERMO_MIRROR = "data/torchcell/thermo_equilibrator"
#: Below this the two routes agree to numerical dust rather than to a decision.
CHANGE_TOLERANCE_KJ = 1e-6


class ReactionChange(BaseModel):
    """One reaction whose standard energy differs between the two routes."""

    model_config = ConfigDict(extra="forbid")

    reaction: str
    delta_g_union_external: float | None
    delta_g_verified: float | None
    change_kj_per_mol: float | None
    kind: str
    touches_amino_acid: list[str] = Field(default_factory=list)
    touches_betaxanthin_precursor: list[str] = Field(default_factory=list)


class ThermoSourceDiff(BaseModel):
    """The whole comparison, written verbatim into the wave-8 note."""

    model_config = ConfigDict(extra="forbid")

    n_reactions: int
    n_priced_union_external: int
    n_priced_verified: int
    sha256_union_external: dict[str, str]
    sha256_verified: dict[str, str]
    counts_union_external: dict[str, int]
    counts_verified: dict[str, int]
    n_changed: int
    n_newly_priced: int
    n_lost: int
    max_abs_change_kj_per_mol: float
    median_abs_change_kj_per_mol: float
    n_amino_acid_reactions: int
    n_amino_acid_reactions_changed: int
    amino_acid_changes: list[ReactionChange] = Field(default_factory=list)
    betaxanthin_precursor_changes: list[ReactionChange] = Field(default_factory=list)
    largest_changes: list[ReactionChange] = Field(default_factory=list)


def main() -> None:
    source = YeastGEM()
    gem = build_gem_tensors(source.model, model_dir=source.model_dir)
    thermo_dir = osp.join(DATA_ROOT, THERMO_MIRROR)
    union = resolve_reaction_delta_g(gem, thermo_dir, ThermoSource.UNION_EXTERNAL)
    verified = resolve_reaction_delta_g(gem, thermo_dir, ThermoSource.VERIFIED)

    # Which species each reaction touches, so a change can be attributed to a phenotype.
    aa_species = {v: k for k, v in AMINO_ACID_TO_GEM_METABOLITE.items()}
    by_id = {m.id: m for m in source.model.metabolites}
    aa_base = {by_id[s].name.split(" [")[0]: k for s, k in aa_species.items()}
    met_to_aa: dict[str, str] = {}
    for met in source.model.metabolites:
        base = met.name.split(" [")[0]
        if base in aa_base:
            met_to_aa[met.id] = aa_base[base]

    u_vals = union.delta_r_g0.numpy()
    v_vals = verified.delta_r_g0.numpy()
    u_mask, v_mask = union.mask.numpy(), verified.mask.numpy()
    changes: list[ReactionChange] = []
    n_changed = n_new = n_lost = 0
    for j, rxn_id in enumerate(gem.rxn_ids):
        u_ok, v_ok = bool(u_mask[j]), bool(v_mask[j])
        if not u_ok and not v_ok:
            continue
        u = float(u_vals[j]) if u_ok else None
        v = float(v_vals[j]) if v_ok else None
        if u_ok and v_ok:
            delta = float(v_vals[j] - u_vals[j])
            if abs(delta) <= CHANGE_TOLERANCE_KJ:
                continue
            kind = "value_changed"
            n_changed += 1
        elif v_ok:
            delta = None
            kind = "priced_only_by_verified"
            n_new += 1
        else:
            delta = None
            kind = "priced_only_by_union_external"
            n_lost += 1
        rxn = source.model.reactions.get_by_id(rxn_id)
        aas = sorted({met_to_aa[m.id] for m in rxn.metabolites if m.id in met_to_aa})
        bxs = sorted(
            {m.id for m in rxn.metabolites if m.id in set(BETAXANTHIN_PRECURSORS)}
        )
        changes.append(
            ReactionChange(
                reaction=rxn_id,
                delta_g_union_external=u,
                delta_g_verified=v,
                change_kj_per_mol=delta,
                kind=kind,
                touches_amino_acid=aas,
                touches_betaxanthin_precursor=bxs,
            )
        )

    valued = [c for c in changes if c.change_kj_per_mol is not None]
    magnitudes = [abs(c.change_kj_per_mol) for c in valued if c.change_kj_per_mol]
    aa_reactions = [
        r.id
        for r in source.model.reactions
        if any(m.id in met_to_aa for m in r.metabolites)
    ]
    aa_changes = [c for c in changes if c.touches_amino_acid]
    diff = ThermoSourceDiff(
        n_reactions=len(gem.rxn_ids),
        n_priced_union_external=int(u_mask.sum()),
        n_priced_verified=int(v_mask.sum()),
        sha256_union_external=dict(union.sha256),
        sha256_verified=dict(verified.sha256),
        counts_union_external={k: int(v) for k, v in union.counts.items()},
        counts_verified={k: int(v) for k, v in verified.counts.items()},
        n_changed=n_changed,
        n_newly_priced=n_new,
        n_lost=n_lost,
        max_abs_change_kj_per_mol=float(max(magnitudes)) if magnitudes else 0.0,
        median_abs_change_kj_per_mol=(
            float(np.median(magnitudes)) if magnitudes else 0.0
        ),
        n_amino_acid_reactions=len(aa_reactions),
        n_amino_acid_reactions_changed=len(aa_changes),
        amino_acid_changes=aa_changes,
        betaxanthin_precursor_changes=[
            c for c in changes if c.touches_betaxanthin_precursor
        ],
        largest_changes=sorted(valued, key=lambda c: -abs(c.change_kj_per_mol or 0.0))[
            :25
        ],
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = osp.join(RESULTS_DIR, "thermo_source_diff_union_external_vs_verified.json")
    with open(path, "w") as fh:
        json.dump(diff.model_dump(), fh, indent=2)

    print(
        f"priced: union_external {diff.n_priced_union_external}, verified "
        f"{diff.n_priced_verified}, of {diff.n_reactions} reactions"
    )
    print(
        f"changed value {diff.n_changed}, priced only by verified {diff.n_newly_priced}, "
        f"priced only by union_external {diff.n_lost}; max |change| "
        f"{diff.max_abs_change_kj_per_mol:.3f} kJ/mol, median "
        f"{diff.median_abs_change_kj_per_mol:.3f}"
    )
    print(
        f"\namino-acid reactions: {diff.n_amino_acid_reactions_changed} of "
        f"{diff.n_amino_acid_reactions} differ between the routes"
    )
    if aa_changes:
        print(
            "\n| reaction | amino acids | union_external | verified | change kJ/mol |"
        )
        print("|---|---|---:|---:|---:|")
        for c in aa_changes[:40]:
            u_txt = (
                "n/a"
                if c.delta_g_union_external is None
                else f"{c.delta_g_union_external:.3f}"
            )
            v_txt = "n/a" if c.delta_g_verified is None else f"{c.delta_g_verified:.3f}"
            d_txt = (
                "n/a" if c.change_kj_per_mol is None else f"{c.change_kj_per_mol:+.3f}"
            )
            print(
                f"| {c.reaction} | {', '.join(c.touches_amino_acid)} | {u_txt} | "
                f"{v_txt} | {d_txt} |"
            )
    print(
        f"\nbetaxanthin precursor reactions changed: {len(diff.betaxanthin_precursor_changes)}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
