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

BOTH INPUTS ARE ALREADY MIRRORED, WHICH WAS NOT OBVIOUS
--------------------------------------------------------
yeast-GEM ships them in ``data/databases/`` and neither needs the network:

``swissprot.tsv``
    ``uniprot, name, gene_id, ec_code, MW, sequence``. The protein sequence for a GEM
    gene, offline. This is the same table the molecular weights already come from, so
    the sequence column was sitting next to a column the layer was reading.
``smilesDB.tsv``
    metabolite name to SMILES.

So this audit runs entirely off hash-pinnable local files, and its output is the exact
per-unit input table a predictor loop would iterate.

WHAT IT REPORTS
---------------
Coverage at each join, because a single headline percentage hides which join is the
lossy one:

1. units whose genes all resolve to a UniProt accession with a sequence;
2. reactions whose substrates resolve to at least one SMILES;
3. units that clear BOTH, which is the real ceiling on a
   ``(sequence, substrate SMILES)`` predictor;
4. how much of the k_cat gap that ceiling could close, against the 4.0 % the Open Enzyme
   Database currently supplies.

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


def main() -> None:
    gem_source = YeastGEM()
    model = gem_source.model
    # `model_dir` is the version root (`.../yeast-GEM-9.0.2`), not its `model/` subdir.
    db_dir = osp.join(gem_source.model_dir, "data", "databases")
    swissprot = read_swissprot(osp.join(db_dir, "swissprot.tsv"))
    smiles_db = read_smiles_db(osp.join(db_dir, "smilesDB.tsv"))
    print(f"swissprot: {len(swissprot)} gene tokens with a sequence")
    print(f"smilesDB:  {len(smiles_db)} metabolite names with SMILES")

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

    rows: list[PredictorInput] = []
    units_with_seq: set[int] = set()
    units_with_smiles: set[int] = set()
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
            sm = smiles_db.get((met_name[i] or "").lower())
            if sm:
                subs.append((met_id[i], met_name[i], sm))
        if subs:
            units_with_smiles.add(u)

        if u in units_with_seq and subs:
            for g, (acc, seq) in resolved:
                for mid, mname, sm in subs:
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
        "units_with_any_substrate_smiles": len(units_with_smiles),
        "units_ready_for_seq_plus_smiles_predictor": len(both),
        "coverage_ready_frac": len(both) / n_units,
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
