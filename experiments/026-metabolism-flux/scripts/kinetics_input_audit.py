# experiments/026-metabolism-flux/scripts/kinetics_input_audit.py
# [[experiments.026-metabolism-flux.scripts.kinetics_input_audit]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetics_input_audit.py

r"""Assemble the input every k_cat / K_M sequence predictor consumes, and measure it.

WHY THIS COMES BEFORE INSTALLING ANY PREDICTOR
-----------------------------------------------
All eleven predictors in the Wu Figure 3 set plus the three the author named take some
subset of ``(protein sequence, substrate SMILES, reaction SMILES, structure)``. Five of
them take exactly ``(sequence, substrate SMILES)``, which is the pair
``KcatPredictor.predict`` in :mod:`torchcell.metabolism.parameters` already declares.

None of that can run against yeast-GEM until the model's 3,728 catalytic units are
actually resolved to those inputs, and the ceiling on any predictor's coverage is set by
how many units resolve, NOT by the predictor. Installing a model first and discovering
afterwards that only part of the network can be fed is the expensive order to do this in.

WHERE THE INPUTS COME FROM
---------------------------
``swissprot.tsv``, shipped inside yeast-GEM
    ``uniprot, name, gene_id, ec_code, MW, sequence``. The protein sequence for a GEM
    gene, offline. This is the same table the molecular weights already come from, so the
    sequence column was sitting next to a column the layer was reading. It resolves
    1,161 of 1,161 genes, and the genome's own protein FASTA independently resolves the
    same 1,161, so **sequence is not a gap from either direction**.
``smilesDB.tsv``, shipped inside yeast-GEM
    metabolite name to SMILES, by exact name. Matches 1,800 of 2,806 metabolites.
``chem_prop.tsv``, mirrored from MetaNetX by :mod:`fetch_kinetics_assets`
    SMILES keyed by MNXM id, the second route for a metabolite the name table misses.

Substrate SMILES is the binding gap and the only one worth optimizing. Protein 3D
structure is a THIRD input, needed by DeepEnzyme alone, and AlphaFold covers it; it does
not touch the numbers below.

WHAT IT REPORTS
---------------
Coverage at each join, because a single headline percentage hides which join is lossy:

1. units whose genes all resolve to a UniProt accession with a sequence;
2. reactions whose substrates resolve to at least one SMILES, reported twice, from the
   shipped name table alone and with the MetaNetX route added, so the value of the extra
   mirror is visible rather than folded into one number;
3. units that clear BOTH, which is the real ceiling on a
   ``(sequence, substrate SMILES)`` predictor;
4. how much of the k_cat gap that ceiling could close, against the 4.0 % the Open Enzyme
   Database currently supplies.

Measured 2026.09.02: 87.5 % of units from the shipped tables alone, 95.3 % once MetaNetX
is joined, against 4.0 % with a measured turnover number. The residual 176 units are
dominated by acyl-chain-specific lipid species such as
``phosphatidylcholine (1-16:0, 2-16:1)``, which are combinatorial names yeast-GEM
enumerates rather than compounds absent from chemistry, so that tail is a naming problem
and not a structural one.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetics_input_audit.py
"""

import csv
import json
import os
import os.path as osp
from collections import defaultdict
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from torchcell.metabolism.constraints import build_gem_tensors
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")


class PredictorInput(BaseModel):
    """One ``(catalytic unit, gene, substrate)`` row, the unit of predictor inference.

    A unit with three genes and two substrates yields six rows, because a turnover number
    is a property of one enzyme acting on one substrate. Collapsing to one row per unit
    would silently pick a substrate.
    """

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    gene_id: str
    uniprot: str
    sequence_length: int
    substrate_met_id: str
    substrate_name: str
    smiles: str
    smiles_source: str


def read_swissprot(path: str) -> dict[str, tuple[str, str]]:
    """Systematic gene name -> (uniprot accession, protein sequence).

    The ``gene_id`` column holds whitespace-separated synonyms, standard name first when
    one exists (``RMA1 YKL132C``), so every token is indexed rather than only the first.
    Indexing the first token alone loses every gene whose standard name precedes its
    systematic name, which is most of the annotated ones.
    """
    out: dict[str, tuple[str, str]] = {}
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            seq = (row.get("sequence") or "").strip()
            acc = (row.get("uniprot") or "").strip()
            if not seq or not acc:
                continue
            for token in (row.get("gene_id") or "").split():
                out.setdefault(token.strip().upper(), (acc, seq))
    return out


def read_smiles_db(path: str) -> dict[str, str]:
    """Metabolite name -> SMILES, lowercased for a case-insensitive join."""
    out: dict[str, str] = {}
    with open(path) as f:
        for parts in csv.reader(f, delimiter="\t"):
            if len(parts) < 2:
                continue
            name, smiles = parts[0].strip(), parts[1].strip()
            if name and smiles:
                out.setdefault(name.lower(), smiles)
    return out


def read_metanetx_smiles(path: str) -> dict[str, str]:
    """MetaNetX MNXM id -> SMILES, from the mirrored ``chem_prop.tsv``.

    The second identifier route for a metabolite the shipped name table misses. Column 0
    is the MNXM id and column 8 the SMILES; rows are skipped rather than defaulted when
    either is blank, since a metabolite with no structure has none rather than an empty
    one. Returns an empty map when the mirror is absent, so the audit still runs and
    reports the shipped-only coverage instead of failing.
    """
    out: dict[str, str] = {}
    if not osp.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) > 8 and cols[0] and cols[8].strip():
                out[cols[0]] = cols[8].strip()
    return out


def metanetx_id(met: Any) -> str | None:
    """The metabolite's MetaNetX id, which cobra may hold as a string or a list."""
    ref = (met.annotation or {}).get("metanetx.chemical")
    if isinstance(ref, list):
        return ref[0] if ref else None
    return ref


def main() -> None:
    gem_source = YeastGEM()
    model = gem_source.model
    # `model_dir` is the version root (`.../yeast-GEM-9.0.2`), not its `model/` subdir.
    db_dir = osp.join(gem_source.model_dir, "data", "databases")
    swissprot = read_swissprot(osp.join(db_dir, "swissprot.tsv"))
    smiles_db = read_smiles_db(osp.join(db_dir, "smilesDB.tsv"))
    mnx_smiles = read_metanetx_smiles(
        osp.join(os.environ["DATA_ROOT"], "data/enzyme_kinetics/metanetx/chem_prop.tsv")
    )
    print(f"swissprot: {len(swissprot)} gene tokens with a sequence")
    print(f"smilesDB:  {len(smiles_db)} metabolite names with SMILES")
    print(f"metanetx:  {len(mnx_smiles)} MNXM ids with SMILES")

    gem = build_gem_tensors(model, model_dir=gem_source.model_dir)
    units = gem.catalytic_units
    unit_genes: dict[int, list[str]] = defaultdict(list)
    for u, g in units.unit_gene_index.t().tolist():
        unit_genes[int(u)].append(units.gene_ids[int(g)])
    unit_reaction = units.unit_reaction.tolist()

    # Substrates of a reaction are the metabolites it consumes, i.e. negative
    # stoichiometry. A predictor takes the SUBSTRATE, so products are not candidates.
    s = gem.s.coalesce()
    idx, val = s.indices(), s.values()
    rxn_substrates: dict[int, list[int]] = defaultdict(list)
    for (i, j), v in zip(idx.t().tolist(), val.tolist()):
        if v < 0:
            rxn_substrates[int(j)].append(int(i))

    met_name = {k: m.name for k, m in enumerate(model.metabolites)}
    met_id = {k: m.id for k, m in enumerate(model.metabolites)}
    # SMILES per metabolite, shipped name table first and the MetaNetX id second, with
    # the source recorded so the two routes stay separable in the output.
    met_smiles: dict[int, tuple[str, str]] = {}
    for k, met in enumerate(model.metabolites):
        hit = smiles_db.get((met.name or "").lower())
        if hit:
            met_smiles[k] = (hit, "smilesDB")
            continue
        mnx = metanetx_id(met)
        if mnx and mnx in mnx_smiles:
            met_smiles[k] = (mnx_smiles[mnx], "metanetx")

    rows: list[PredictorInput] = []
    units_with_seq: set[int] = set()
    units_with_smiles: set[int] = set()
    units_with_smiles_shipped_only: set[int] = set()
    genes_resolved: set[str] = set()
    genes_unresolved: set[str] = set()

    for u, genes in unit_genes.items():
        j = int(unit_reaction[u])
        rxn = model.reactions[j].id
        resolved = []
        for g in genes:
            hit = swissprot.get(g.upper())
            if hit is None:
                genes_unresolved.add(g)
            else:
                genes_resolved.add(g)
                resolved.append((g, hit))
        # A complex is only runnable if EVERY subunit resolves: its turnover is the min
        # over members, and a missing member makes that min undefined rather than smaller.
        if resolved and len(resolved) == len(genes):
            units_with_seq.add(u)

        subs = []
        for i in rxn_substrates.get(j, []):
            hit = met_smiles.get(i)
            if hit:
                subs.append((met_id[i], met_name[i], hit[0], hit[1]))
        if subs:
            units_with_smiles.add(u)
        if any(src == "smilesDB" for *_, src in subs):
            units_with_smiles_shipped_only.add(u)

        if u in units_with_seq and subs:
            for g, (acc, seq) in resolved:
                for mid, mname, sm, sm_src in subs:
                    rows.append(
                        PredictorInput(
                            unit_id=u,
                            reaction_id=rxn,
                            gene_id=g,
                            uniprot=acc,
                            sequence_length=len(seq),
                            substrate_met_id=mid,
                            substrate_name=mname,
                            smiles=sm,
                            smiles_source=sm_src,
                        )
                    )

    n_units = units.n_units
    both = units_with_seq & units_with_smiles
    report: dict[str, Any] = {
        "n_catalytic_units": n_units,
        "n_gem_genes": len(units.gene_ids),
        "genes_with_sequence": len(genes_resolved),
        "genes_without_sequence": len(genes_unresolved),
        "units_all_genes_have_sequence": len(units_with_seq),
        "units_with_substrate_smiles_shipped_only": len(units_with_smiles_shipped_only),
        "units_with_any_substrate_smiles": len(units_with_smiles),
        "units_ready_for_seq_plus_smiles_predictor": len(both),
        "coverage_ready_frac": len(both) / n_units,
        "coverage_ready_frac_shipped_only": len(
            units_with_seq & units_with_smiles_shipped_only
        )
        / n_units,
        "oed_measured_coverage_frac": 148 / n_units,
        "predictor_input_rows": len(rows),
        "example_unresolved_genes": sorted(genes_unresolved)[:15],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(osp.join(RESULTS_DIR, "kinetics_input_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(osp.join(RESULTS_DIR, "kinetics_predictor_inputs.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(PredictorInput.model_fields))
        w.writeheader()
        for r in rows:
            w.writerow(r.model_dump())

    print()
    for k, v in report.items():
        if k == "example_unresolved_genes":
            continue
        print(f"{k:44} {v:.4f}" if isinstance(v, float) else f"{k:44} {v}")
    print(
        f"\nready {len(both)}/{n_units} units "
        f"({100 * len(both) / n_units:.1f} %) vs Open Enzyme Database "
        f"148/{n_units} ({100 * 148 / n_units:.1f} %)"
    )
    print(f"\n-> {osp.join(RESULTS_DIR, 'kinetics_input_audit.json')}")
    print(f"-> {osp.join(RESULTS_DIR, 'kinetics_predictor_inputs.csv')}")


if __name__ == "__main__":
    main()
