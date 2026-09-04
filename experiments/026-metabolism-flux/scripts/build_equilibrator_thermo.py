# experiments/026-metabolism-flux/scripts/build_equilibrator_thermo.py
# [[experiments.026-metabolism-flux.scripts.build_equilibrator_thermo]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/build_equilibrator_thermo.py

r"""Recompute every yeast-GEM formation energy from eQuilibrator, with uncertainty.

WHY NOT THE SHIPPED CSVs
-------------------------
yeast-GEM ships ``model_metDeltaG.csv`` and ``model_rxnDeltaG.csv``, and they are usable:
2,389 metabolites and 3,210 reactions. They are also a dead end for two reasons. They are
**point values with no covariance**, so Thermo-Flux's correlated error term
:math:`\Delta_r G^{\circ,\mathrm{error}} = Qm` cannot be formed at all, and the flux layer
currently records that as an unavailable quantity. And they exist only for this model:
another organism's GEM arrives with a different table or none, so nothing built on them
transfers.

Component contribution replaces both problems with one method. It decomposes a compound
into groups, fits group energies against the measured reaction corpus, and returns a
formation energy **plus the covariance implied by that fit**. Applied to a structure, it
works for any organism, and for a heterologous compound that no database has ever
tabulated.

WHAT THIS PRODUCES
------------------
Per compartment-specific metabolite, a transformed standard formation energy
:math:`\Delta_f G'^\circ` and its uncertainty, plus the square-root covariance factor
:math:`\Sigma` whose product :math:`Q = \Sigma\Sigma^\top` is the correlated error matrix.
Reaction energies are then :math:`\Delta_r G'^\circ_j = \sum_i S_{ij}\,\Delta_f G'^\circ_i`,
which is exactly how the flux layer already consumes the shipped table, so this is a
drop-in replacement carrying strictly more information.

RESOLUTION IS TIERED, AND THE TIER IS RECORDED
------------------------------------------------
A compound is looked up by accession first (MetaNetX, then ChEBI, then KEGG, then BiGG)
and by structure second. Which tier hit is stored per metabolite, because an accession
match and an InChI match are different claims: the first says two databases agree on an
identifier, the second says a structure was handed to the group decomposition directly.

TEMPERATURE, pH, IONIC STRENGTH
---------------------------------
303.15 K, because the SGA screens this model is used against are run at 30 C. pH 7.0 and
ionic strength 0.25 M are eQuilibrator's own defaults for a cytosolic estimate.

**Compartment pH is applied, and the table is an assumption.** Every value in
``COMPARTMENT_PH`` is a textbook organellar pH rather than something sourced from a
mirrored paper, so it is labeled ``assumed`` in the output. It matters: a proton appearing
on both sides of a membrane at different pH is the entire driving force of a transport
reaction, and computing every compartment at 7.0 sets that force to zero by construction.
Running with ``--single-condition`` produces the uniform pH 7.0 table for comparison.

Run under the eQuilibrator environment, which is separate from the torchcell environment
because component-contribution pins numpy and pandas majors:

    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/build_equilibrator_thermo.py
"""

import argparse
import hashlib
import json
import os
import os.path as osp
from datetime import UTC, datetime

import cobra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from equilibrator_api import ComponentContribution, Q_

load_dotenv(
    osp.join(
        "/home/michaelvolk/Documents/projects/torchcell.worktrees",
        "feat/kinetics-equilibrator-datasets",
        ".env",
    )
)
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
OUT_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")

TEMPERATURE_K = 303.15
IONIC_STRENGTH_M = 0.25
P_MG = 3.0
REFERENCE_PH = 7.0

# Textbook organellar pH. ASSUMED, not sourced from a mirrored paper: treat every value as
# an assumption that a later revision should replace with a cited measurement. The
# membrane pseudo-compartments take the pH of the aqueous phase they face.
COMPARTMENT_PH: dict[str, float] = {
    "c": 7.2,  # cytoplasm
    "e": 5.0,  # extracellular, SC medium is acidic
    "m": 7.8,  # mitochondrial matrix, alkaline relative to cytosol
    "n": 7.2,  # nucleus, continuous with cytosol through the pore
    "p": 7.0,  # peroxisome
    "er": 7.1,  # endoplasmic reticulum
    "g": 6.6,  # Golgi, acidified along the secretory path
    "v": 6.2,  # vacuole, the acidic compartment
    "ce": 5.0,  # cell envelope, faces the medium
    "lp": 7.2,  # lipid particle, cytosol-facing
    "erm": 7.1,
    "vm": 6.2,
    "gm": 6.6,
    "mm": 7.8,
}

# Accession namespaces in the order they are tried. MetaNetX first because it is the
# reconciliation layer eQuilibrator itself indexes most densely; BiGG last because its
# identifiers are model-specific rather than chemical.
ACCESSION_TIERS: list[tuple[str, str]] = [
    ("metanetx.chemical", "metanetx.chemical"),
    ("chebi", "chebi"),
    ("kegg.compound", "kegg"),
    ("bigg.metabolite", "bigg.metabolite"),
]


def sha256_file(path: str) -> str:
    """Hex sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def read_smiles_db(path: str) -> dict[str, str]:
    """Metabolite name -> SMILES, lowercased for a case-insensitive join."""
    out: dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            name, smiles = parts[0].strip(), parts[1].strip()
            if name and smiles:
                out[name.lower()] = smiles
    return out


def resolve_compound(cc, metabolite, smiles_db):
    """Find an eQuilibrator compound for a GEM metabolite, and say how it was found.

    Returns ``(compound, tier, accession)``. ``tier`` is the namespace that hit, or
    ``inchi`` when the accession lookup failed and the structure resolved it, or ``None``
    when nothing did.
    """
    for annotation_key, prefix in ACCESSION_TIERS:
        value = metabolite.annotation.get(annotation_key)
        if not value:
            continue
        # cobra gives a list when a metabolite carries several ids in one namespace.
        for candidate in value if isinstance(value, list) else [value]:
            accession = str(candidate)
            # ChEBI ids arrive already prefixed ("CHEBI:12345"); the rest do not.
            query = (
                accession
                if accession.lower().startswith(prefix.lower())
                else f"{prefix}:{accession}"
            )
            compound = cc.get_compound(query)
            if compound is not None:
                return compound, annotation_key, query

    smiles = smiles_db.get((metabolite.name or "").lower())
    if smiles:
        compound = cc.get_compound_by_inchi(smiles_to_inchi(smiles))
        if compound is not None:
            return compound, "inchi", smiles
    return None, None, None


def smiles_to_inchi(smiles: str) -> str:
    """SMILES to InChI, the key eQuilibrator's structure lookup takes."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return Chem.MolToInchi(mol)


def main() -> None:
    """Resolve every metabolite, compute formation energies, and write the tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-condition", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    model_path = osp.join(GEM_DIR, "model", "yeast-GEM.xml")
    model = cobra.io.read_sbml_model(model_path)
    smiles_db = read_smiles_db(osp.join(GEM_DIR, "data", "databases", "smilesDB.tsv"))

    cc = ComponentContribution()
    cc.p_h = Q_(REFERENCE_PH)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    metabolites = model.metabolites[: args.limit] if args.limit else model.metabolites

    # One resolution per distinct CHEMICAL, cached, because the same species appears in up
    # to fourteen compartments and the group decomposition is the expensive step.
    chemical_cache: dict[str, tuple] = {}
    rows = []
    sigma_vectors: dict[str, np.ndarray] = {}
    sigma_inf_vectors: dict[str, np.ndarray] = {}

    for index, metabolite in enumerate(metabolites):
        if index % 250 == 0:
            print(f"  {index}/{len(metabolites)}", flush=True)
        # Strip the compartment suffix so 'ATP in the cytosol' and 'ATP in the matrix'
        # share one resolution.
        chemical_key = (metabolite.name or metabolite.id).lower()
        if chemical_key not in chemical_cache:
            compound, tier, accession = resolve_compound(cc, metabolite, smiles_db)
            mu = sigma_fin = sigma_inf = None
            if compound is not None:
                # Three parts, not two: the estimate, the FINITE uncertainty directions,
                # and the INFINITE ones. A nonzero infinite component means that direction
                # of the group decomposition was never determined by the training corpus.
                # Almost every compound has one, because an absolute formation energy is
                # only defined up to element conservation -- which is precisely why the
                # test has to be applied to a balanced REACTION, where those directions
                # cancel, and never to a compound in isolation.
                mu, sigma_fin, sigma_inf = cc.standard_dg_formation(compound)
            chemical_cache[chemical_key] = (
                compound,
                tier,
                accession,
                mu,
                sigma_fin,
                sigma_inf,
            )
        compound, tier, accession, mu, sigma_fin, sigma_inf = chemical_cache[chemical_key]

        compartment = metabolite.compartment
        p_h = REFERENCE_PH if args.single_condition else COMPARTMENT_PH.get(
            compartment, REFERENCE_PH
        )

        transformed = np.nan
        if compound is not None and mu is not None:
            # The Legendre transform to (pH, I, T, pMg). Adding it to the untransformed
            # formation energy is what makes the value a dfG'o rather than a dfGo.
            correction = compound.transform(
                Q_(p_h),
                Q_(f"{IONIC_STRENGTH_M}M"),
                Q_(f"{TEMPERATURE_K}K"),
                Q_(P_MG),
            )
            transformed = float(mu) + float(correction.m_as("kJ/mol"))
            sigma_vectors[metabolite.id] = np.asarray(sigma_fin).flatten()
            sigma_inf_vectors[metabolite.id] = np.asarray(sigma_inf).flatten()

        rows.append(
            {
                "met_id": metabolite.id,
                "name": metabolite.name,
                "compartment": compartment,
                "formula": metabolite.formula,
                "charge": metabolite.charge,
                "resolution_tier": tier,
                "accession": accession,
                "p_h": p_h,
                "p_h_source": "reference" if args.single_condition else "assumed",
                "dgf_prime_kj_per_mol": transformed,
                "dgf_untransformed_kj_per_mol": float(mu) if mu is not None else np.nan,
                "sigma_kj_per_mol": (
                    float(np.linalg.norm(sigma_fin)) if sigma_fin is not None else np.nan
                ),
            }
        )

    table = pd.DataFrame(rows)
    resolved = table["dgf_prime_kj_per_mol"].notna()

    # The square-root covariance factor. Q = Sigma Sigma^T is the correlated error matrix
    # Thermo-Flux calls for; storing the factor rather than Q keeps it small and keeps the
    # matrix positive semidefinite by construction.
    ordered_ids = [m for m in table.loc[resolved, "met_id"] if m in sigma_vectors]
    if ordered_ids:
        width = max(len(sigma_vectors[m]) for m in ordered_ids)
        sigma_matrix = np.zeros((len(ordered_ids), width))
        for row_index, met_id in enumerate(ordered_ids):
            vector = sigma_vectors[met_id]
            sigma_matrix[row_index, : len(vector)] = vector
        np.savez_compressed(
            osp.join(OUT_DIR, "formation_sigma.npz"),
            met_ids=np.array(ordered_ids),
            sigma=sigma_matrix,
        )

    suffix = "_single_condition" if args.single_condition else ""
    out_path = osp.join(OUT_DIR, f"metabolite_dgf_prime{suffix}.parquet")
    table.to_parquet(out_path, index=False)

    # Reaction energies by S^T dfG'o, with BOTH uncertainty parts propagated the same way.
    #
    # The finite part gives the reaction's standard error, sqrt of the diagonal of
    # Q = (S^T Sigma)(S^T Sigma)^T. The infinite part gives the estimability test: a
    # reaction whose S^T sigma_inf is nonzero depends on a direction the component
    # decomposition never determined, so its energy is not merely uncertain, it is
    # UNDEFINED. Reporting such a reaction with a finite error bar would be the single
    # most misleading thing this script could do, so it is masked instead.
    lookup = dict(zip(table["met_id"], table["dgf_prime_kj_per_mol"]))
    inf_tolerance = 1e-10
    reaction_rows = []
    for reaction in model.reactions:
        participants = list(reaction.metabolites.items())
        values = [lookup.get(m.id, np.nan) for m, _ in participants]
        coefficients = [c for _, c in participants]
        complete = not any(np.isnan(v) for v in values)
        drg = (
            float(sum(c * v for c, v in zip(coefficients, values)))
            if complete
            else np.nan
        )

        sigma_error = np.nan
        estimable = False
        if complete and all(m.id in sigma_vectors for m, _ in participants):
            finite = sum(
                c * sigma_vectors[m.id] for (m, _), c in zip(participants, coefficients)
            )
            infinite = sum(
                c * sigma_inf_vectors[m.id]
                for (m, _), c in zip(participants, coefficients)
            )
            sigma_error = float(np.linalg.norm(finite))
            estimable = bool(np.linalg.norm(infinite) < inf_tolerance)

        reaction_rows.append(
            {
                "reaction_id": reaction.id,
                "name": reaction.name,
                "n_participants": len(values),
                "all_participants_known": complete,
                "drg_prime_kj_per_mol": drg,
                "sigma_kj_per_mol": sigma_error,
                "estimable": estimable,
            }
        )
    reactions = pd.DataFrame(reaction_rows)
    reactions.to_parquet(
        osp.join(OUT_DIR, f"reaction_drg_prime{suffix}.parquet"), index=False
    )

    tiers = table["resolution_tier"].value_counts(dropna=False)
    summary = {
        "gem_model_sha256": sha256_file(model_path),
        "n_metabolites": len(table),
        "n_metabolites_resolved": int(resolved.sum()),
        "frac_metabolites_resolved": float(resolved.mean()),
        "resolution_tiers": {str(k): int(v) for k, v in tiers.items()},
        "n_reactions": len(reactions),
        "n_reactions_all_participants_known": int(
            reactions["all_participants_known"].sum()
        ),
        "frac_reactions_all_participants_known": float(
            reactions["all_participants_known"].mean()
        ),
        "n_reactions_estimable": int(reactions["estimable"].sum()),
        "frac_reactions_estimable": float(reactions["estimable"].mean()),
        "reaction_sigma_median_kj_per_mol": float(
            reactions.loc[reactions["estimable"], "sigma_kj_per_mol"].median()
        )
        if reactions["estimable"].any()
        else float("nan"),
        "has_covariance_factor": bool(ordered_ids),
        "sigma_rank": int(sigma_matrix.shape[1]) if ordered_ids else 0,
        "conditions": {
            "temperature_k": TEMPERATURE_K,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "p_mg": P_MG,
            "p_h": "uniform 7.0" if args.single_condition else "per compartment (assumed)",
        },
        "built_at": datetime.now(UTC).isoformat(),
    }
    with open(osp.join(RESULTS, f"equilibrator_thermo{suffix}.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
