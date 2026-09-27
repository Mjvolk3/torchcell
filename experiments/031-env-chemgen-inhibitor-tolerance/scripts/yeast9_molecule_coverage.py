# experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_molecule_coverage.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.yeast9_molecule_coverage]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/yeast9_molecule_coverage
"""Measure how much of the Yeast9 (yeast-GEM 9.0.2) metabolite set a unified
small-molecule representation can actually cover.

Three questions, one script:

1. **Structure inventory.** How many metabolites does the model hold, how many
   distinct chemical species survive collapsing the compartment copies, and how many
   carry a resolvable structure. Identifiers come from exactly two places, both
   named per row in the output: the SBML annotation block of
   ``model/yeast-GEM.xml`` (``chebi``, ``kegg.compound``, ``metanetx.chemical``,
   ``bigg.metabolite``) and the release's own name-keyed SMILES table
   ``data/databases/smilesDB.tsv``. The repo-side resolver
   ``torchcell/datamodels/compound_identity_table.json`` is checked as a third
   candidate route and its incremental contribution is reported.
2. **Encoder coverage.** Every encoder in ``torchcell.molecule.ENCODERS`` is run over
   every structure-bearing species, one ``check`` per molecule so a rejection is
   counted rather than aborting the batch, and the failures are grouped by the
   structural class of the molecule that failed.
3. **Overlap.** The InChIKeys derived from the Yeast9 SMILES against the dosed
   compound keys in ``results/embeddings/*.npz`` and the media component keys in
   ``results/embeddings_media/*.npz``. Both the full key and the 14-character
   connectivity skeleton are compared, because a curated PubChem key and an
   RDKit-derived key for one molecule measurably disagree in the stereo and
   protonation blocks (``torchcell/datamodels/compound_identity.py`` lines 46-54).

Structural classes are assigned by ordered, first-match rules over the model's OWN
fields (name, formula), never by hand-listing members, so every class count is
reproducible from the model file. A ``formula`` carrying a standalone ``R`` or ``X``
element token is the model's own marker for an unspecified substituent.

Nothing here writes to any file another task owns: outputs are the ``yeast9_*``
names plus the ``yeast9_embeddings/`` directory.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import time

import cobra
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rdkit import Chem, RDLogger
from rdkit.Chem import inchi as rdinchi
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from torchcell.molecule import ENCODERS, MoleculeEncoder

RDLogger.DisableLog("rdApp.*")

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMBED_DIR = osp.join(RESULTS_DIR, "yeast9_embeddings")
DOSED_EMBED_DIR = osp.join(RESULTS_DIR, "embeddings")
MEDIA_EMBED_DIR = osp.join(RESULTS_DIR, "embeddings_media")

GEM_ROOT = osp.join(DATA_ROOT, "data/torchcell/yeast-GEM/yeast-GEM-9.0.2")
GEM_SBML = osp.join(GEM_ROOT, "model", "yeast-GEM.xml")
GEM_XLSX = osp.join(GEM_ROOT, "model", "yeast-GEM.xlsx")
GEM_SMILES_DB = osp.join(GEM_ROOT, "data", "databases", "smilesDB.tsv")
GEM_MET_DG = osp.join(GEM_ROOT, "data", "databases", "model_metDeltaG.csv")
GEM_RXN_DG = osp.join(GEM_ROOT, "data", "databases", "model_rxnDeltaG.csv")

#: yeast-GEM writes this value in the dG tables for "no value"; it is not a dG. The
#: tables carry a second absence marker, a literal empty cell that reads back as NaN,
#: so both are rejected (``torchcell/metabolism/constraints.py`` lines 37-58 makes the
#: same rejection and reports the same coverage).
DG_SENTINEL = 10000000


def _dg_usable(value: float | None) -> bool:
    """Whether a dG table cell is a real value rather than either absence marker."""
    return value is not None and value == value and value != DG_SENTINEL


#: The biomass pseudo-metabolites: names the model uses for a lumped pool rather than
#: for a molecule. Closed set, taken from the species that carry no formula at all.
PSEUDO_METABOLITES = frozenset(
    {
        "biomass",
        "protein",
        "lipid",
        "carbohydrate",
        "ion",
        "DNA",
        "RNA",
        "lipid backbone",
        "lipid chain",
        "fatty acid backbone",
        "Sulfur donor",
    }
)

#: A standalone R or X element token in a molecular formula: the model's own marker
#: for an unspecified substituent or chain. A formula is a run of ``[A-Z][a-z]?``
#: symbols with counts, so the character before an ``R`` is usually another element's
#: letter (``C22H31N7O17P3SR``) and must NOT be required to be a non-letter. The
#: trailing lookahead is what distinguishes the marker from ``Rb``/``Ru`` and ``Xe``,
#: since every element's second letter is lowercase.
_GENERIC_FORMULA = re.compile(r"[RX](?![a-z])")
#: An acyl chain written out as ``16:0`` / ``18:1``: a lipid resolved to chain lengths.
_ACYL_RESOLVED = re.compile(r"\d\d:\d")
#: A KEGG glycan accession used as a metabolite name.
_KEGG_GLYCAN = re.compile(r"^G\d{5}$")


def classify_species(name: str, formula: str | None) -> str:
    """The structural class of one species, by ordered first-match rules.

    Rules read only the model's own ``name`` and ``formula``. The order matters: a
    tRNA-charged amino acid is also formula-generic, and the tRNA class is the more
    informative statement about why no structure exists.
    """
    if name in PSEUDO_METABOLITES:
        return "pseudo_metabolite"
    if "tRNA" in name:
        return "trna"
    if re.search(r"-ACP\b|\[acp\]|^ACP\d*$", name):
        return "acp_thioester"
    if re.search(
        r"protein|cytochrome|histone|TRX1|desulfurase|scaffold|carrier\)|Apo", name
    ):
        return "protein_species"
    if name.endswith(" backbone") or name.endswith(" chain"):
        return "pooled_lipid_class"
    if _ACYL_RESOLVED.search(name):
        return "acyl_resolved_lipid"
    if _KEGG_GLYCAN.match(name):
        return "kegg_glycan"
    if formula is None or formula == "":
        return "no_formula_other"
    if _GENERIC_FORMULA.search(formula):
        return "generic_r_group"
    if not re.search(r"C(?![aoudlrs])", formula):
        return "inorganic_or_ion"
    return "specific_molecule"


def load_smiles_db() -> dict[str, str]:
    """``metabolite name -> SMILES`` from the release's own ``smilesDB.tsv``.

    The file is two tab-separated columns with no header: the metabolite name exactly
    as the model spells it, and a SMILES. Rows with an empty SMILES are absent
    entries, not molecules, and are dropped.
    """
    table: dict[str, str] = {}
    with open(GEM_SMILES_DB) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1].strip():
                table[parts[0].strip()] = parts[1].strip()
    return table


def load_identity_table() -> tuple[dict[str, dict], dict[str, dict]]:
    """The repo-side resolver's rows indexed by lowercased name/synonym and by ChEBI.

    Read straight from the sha256-pinned JSON the resolver module owns, so this
    script measures the same bytes a loader would resolve against.
    """
    from torchcell.datamodels.compound_identity import _TABLE_PATH

    with open(_TABLE_PATH) as f:
        records = json.load(f)["records"]
    by_name: dict[str, dict] = {}
    by_chebi: dict[str, dict] = {}
    for r in records:
        for label in [r["name"], *r.get("synonyms", [])]:
            by_name.setdefault(label.strip().lower(), r)
        if r.get("chebi_id"):
            by_chebi.setdefault(str(r["chebi_id"]), r)
    return by_name, by_chebi


def first_annotation(value: object) -> str | None:
    """One identifier from an SBML annotation entry (cobra yields str or list)."""
    if value is None:
        return None
    if isinstance(value, list):
        return str(value[0]) if value else None
    return str(value)


def build_inventory() -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per distinct chemical species, plus the model-level counts."""
    model = cobra.io.read_sbml_model(GEM_SBML)
    smiles_db = load_smiles_db()
    by_name, by_chebi = load_identity_table()

    met_dg = pd.read_csv(GEM_MET_DG)
    dg_by_met = dict(zip(met_dg["Var1"], met_dg["Var2"], strict=True))
    rxn_dg = pd.read_csv(GEM_RXN_DG)

    xlsx_mets = pd.read_excel(GEM_XLSX, sheet_name="METS")

    species: dict[str, list[cobra.Metabolite]] = {}
    for met in model.metabolites:
        species.setdefault(met.name, []).append(met)

    rows = []
    for name, mets in sorted(species.items()):
        head = mets[0]
        ann = head.annotation
        chebi = first_annotation(ann.get("chebi"))
        smiles = smiles_db.get(name)
        source = "yeast-GEM-9.0.2/data/databases/smilesDB.tsv" if smiles else None

        table_row = by_name.get(name.strip().lower())
        table_by_chebi = by_chebi.get(str(chebi)) if chebi else None
        table_smiles = None
        table_route = None
        if table_row is not None and table_row.get("smiles"):
            table_smiles = table_row["smiles"]
            table_route = "compound_identity_table.json:name"
        elif table_by_chebi is not None and table_by_chebi.get("smiles"):
            table_smiles = table_by_chebi["smiles"]
            table_route = "compound_identity_table.json:chebi_id"
        if smiles is None and table_smiles is not None:
            smiles = table_smiles
            source = table_route

        parsed = Chem.MolFromSmiles(smiles) if smiles else None
        inchikey = (
            rdinchi.MolToInchiKey(parsed) if parsed is not None else None
        ) or None
        smiles_formula = CalcMolFormula(parsed) if parsed is not None else None

        has_dg = any(_dg_usable(dg_by_met.get(m.id)) for m in mets)

        rows.append(
            {
                "species_name": name,
                "n_compartment_copies": len(mets),
                "compartments": ";".join(sorted({m.compartment for m in mets})),
                "met_ids": ";".join(sorted(m.id for m in mets)),
                "formula": head.formula,
                "charge": head.charge,
                "chebi": chebi,
                "kegg_compound": first_annotation(ann.get("kegg.compound")),
                "metanetx_chemical": first_annotation(ann.get("metanetx.chemical")),
                "bigg_metabolite": first_annotation(ann.get("bigg.metabolite")),
                "smiles": smiles,
                "smiles_source": source,
                "in_smiles_db": name in smiles_db,
                "identity_table_smiles_route": table_route,
                "rdkit_parses": parsed is not None,
                "inchikey": inchikey,
                "inchikey_skeleton": inchikey.split("-")[0] if inchikey else None,
                "n_unassigned_stereocenters": (
                    _n_unassigned_stereocenters(parsed) if parsed is not None else None
                ),
                "stereo_free_inchikey": (
                    inchikey.split("-")[1].startswith("UHFFFAOYS") if inchikey else None
                ),
                "smiles_formula": smiles_formula,
                "formula_agrees": (
                    None
                    if smiles_formula is None or not head.formula
                    else _formula_agrees(head.formula, smiles_formula)
                ),
                "structure_class": classify_species(name, head.formula),
                "has_formation_dg": has_dg,
            }
        )

    inventory = pd.DataFrame(rows)

    # A structure-based reaction property (a group-contribution dG, a reaction
    # fingerprint) needs a structure for EVERY participant, so the fraction of
    # reactions all of whose metabolites carry SMILES is the ceiling on that axis.
    smiles_names = {r["species_name"] for r in rows if r["smiles"]}
    n_rxn_all_structured = 0
    n_rxn_any_structured = 0
    for rxn in model.reactions:
        names = {met.name for met in rxn.metabolites}
        if not names:
            continue
        if names <= smiles_names:
            n_rxn_all_structured += 1
        if names & smiles_names:
            n_rxn_any_structured += 1

    counts = {
        "n_metabolite_entries": len(model.metabolites),
        "n_distinct_species": len(species),
        "n_reactions": len(model.reactions),
        "n_genes": len(model.genes),
        "n_compartments": len(model.compartments),
        "n_xlsx_met_rows": len(xlsx_mets),
        "n_xlsx_inchi_populated": int(xlsx_mets["InChI"].notna().sum()),
        "n_met_dg_rows": len(met_dg),
        "n_met_dg_sentinel": int((met_dg["Var2"] == DG_SENTINEL).sum()),
        "n_met_dg_blank": int(met_dg["Var2"].isna().sum()),
        "n_met_dg_usable": int(met_dg["Var2"].map(_dg_usable).sum()),
        "n_rxn_dg_rows": len(rxn_dg),
        "n_rxn_dg_sentinel": int((rxn_dg["Var2"] == DG_SENTINEL).sum()),
        "n_rxn_dg_blank": int(rxn_dg["Var2"].isna().sum()),
        "n_rxn_dg_usable": int(rxn_dg["Var2"].map(_dg_usable).sum()),
        "n_reactions_all_metabolites_structured": n_rxn_all_structured,
        "n_reactions_some_metabolite_structured": n_rxn_any_structured,
    }
    return inventory, counts


def _n_unassigned_stereocenters(mol: Chem.Mol) -> int:
    """Potential stereocenters the SMILES leaves unspecified.

    The shipped SMILES are written flat for most species, so the InChIKey's stereo
    block comes back as the no-stereo ``UHFFFAOYS`` sentinel and L- and D- forms of one
    amino acid land on the same key. This counts how much stereochemistry the string
    declines to state, which is what makes an exact-key join to a curated PubChem key
    fail even when the molecule is the same.
    """
    return len(
        Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
    ) - len(
        Chem.FindMolChiralCenters(
            mol, includeUnassigned=False, useLegacyImplementation=False
        )
    )


def _formula_agrees(model_formula: str, smiles_formula: str) -> bool:
    """Whether the SMILES encodes the same heavy-atom composition the model states.

    Compared on element counts with charge and hydrogens dropped: the model's formula
    is written for the charge state named in ``charge``, while ``CalcMolFormula``
    appends the SMILES' own charge, so a raw string compare would report a
    disagreement that is only a protonation convention. A real disagreement (the
    polysaccharide whose SMILES is its monomer) shows in the carbon count.
    """
    return _elements(model_formula) == _elements(smiles_formula)


def _elements(formula: str) -> dict[str, int]:
    """Element -> count for a formula string, ignoring H and any charge suffix."""
    out: dict[str, int] = {}
    for sym, num in re.findall(
        r"([A-Z][a-z]?)(\d*)", formula.split("+")[0].split("-")[0]
    ):
        if sym == "H":
            continue
        out[sym] = out.get(sym, 0) + (int(num) if num else 1)
    return out


def run_encoders(
    inventory: pd.DataFrame, encoder_names: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, str]], pd.DataFrame]:
    """Every encoder over every structure-bearing species; coverage and failures.

    ``check`` is called once per molecule so a rejection is a counted outcome rather
    than an aborted batch, matching ``embed_compounds.py``'s accounting. The
    survivors are then encoded in one batched call.
    """
    bearing = inventory[inventory["smiles"].notna()].copy()
    smiles_by_species = dict(
        zip(bearing["species_name"], bearing["smiles"], strict=True)
    )
    class_by_species = dict(
        zip(bearing["species_name"], bearing["structure_class"], strict=True)
    )

    os.makedirs(EMBED_DIR, exist_ok=True)
    rows = []
    failures: dict[str, dict[str, str]] = {}
    class_rows = []
    for enc_name in encoder_names:
        t0 = time.time()
        encoder: MoleculeEncoder = ENCODERS[enc_name]()
        load_seconds = time.time() - t0

        t0 = time.time()
        accepted: list[str] = []
        failed: dict[str, str] = {}
        for key, smi in smiles_by_species.items():
            try:
                encoder.check(smi)
            except ValueError as e:
                failed[key] = f"{type(e).__name__}: {e}"
            else:
                accepted.append(key)
        x = encoder.encode([smiles_by_species[k] for k in accepted])
        seconds = time.time() - t0

        np.savez(
            osp.join(EMBED_DIR, f"{enc_name}.npz"),
            species_name=np.array(accepted, dtype=str),
            inchikey=np.array(
                [
                    inventory.set_index("species_name").loc[k, "inchikey"] or ""
                    for k in accepted
                ],
                dtype=str,
            ),
            X=x.astype(np.float32),
        )
        failures[enc_name] = failed
        n_nan = (
            int(np.isnan(x).any(axis=1).sum()) if x.size else 0
        )  # rdkit_2d is the one encoder allowed to emit NaN
        rows.append(
            {
                "encoder": enc_name,
                "dim": encoder.dim,
                "n_attempted": len(smiles_by_species),
                "n_embedded": len(accepted),
                "n_failed": len(failed),
                "n_rows_with_nan": n_nan,
                "load_seconds": round(load_seconds, 2),
                "seconds": round(seconds, 2),
            }
        )
        for key, msg in failed.items():
            class_rows.append(
                {
                    "encoder": enc_name,
                    "species_name": key,
                    "structure_class": class_by_species[key],
                    "reason": _reason_class(msg),
                    "message": msg[:300],
                }
            )
        print(
            f"{enc_name}: dim {encoder.dim}, load {load_seconds:.1f}s, "
            f"embedded {len(accepted)}/{len(smiles_by_species)} in {seconds:.1f}s, "
            f"failed {len(failed)}, rows with NaN {n_nan}"
        )
        del encoder

    return pd.DataFrame(rows), failures, pd.DataFrame(class_rows)


def _reason_class(message: str) -> str:
    """Group an encoder's own rejection message into a reason class."""
    if "unparsable SMILES" in message:
        return "rdkit_unparsable"
    if "all-zero coordinates" in message:
        return "no_3d_conformer_zero_coordinates"
    if "only 2D coordinates" in message:
        return "etkdg_failed_2d_only"
    return "other"


def consensus_keys(path: str) -> tuple[set[str], pd.DataFrame]:
    """The compound set an embedding directory holds, by per-file consensus.

    Every encoder archive in the directory embeds the same compound set, so a key is
    in the set when at least half the archives carry it. The consensus is used rather
    than the union because a directory can be mid-rewrite: another job re-embedding a
    different compound set leaves one archive with a key set that is neither a subset
    nor a superset of the others, and a union would silently absorb it. The per-file
    frame reports each archive's size and whether its keys sit inside the consensus,
    so such a file is visible instead of averaged in.
    """
    per_file = []
    tally: dict[str, int] = {}
    names = [f for f in sorted(os.listdir(path)) if f.endswith(".npz")]
    for fname in names:
        keys = set(np.load(osp.join(path, fname))["inchikey"].tolist())
        per_file.append({"file": fname, "n_keys": len(keys)})
        for k in keys:
            tally[k] = tally.get(k, 0) + 1
    consensus = {k for k, n in tally.items() if n * 2 >= len(names)}
    for row, fname in zip(per_file, names, strict=True):
        keys = set(np.load(osp.join(path, fname))["inchikey"].tolist())
        row["subset_of_consensus"] = keys <= consensus
    return consensus, pd.DataFrame(per_file)


def overlap(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Yeast9 structures against the dosed and media compound sets.

    The dosed and media keys are read from the ``inchikey`` array of the already
    written embedding archives; nothing is recomputed from the parquet records and
    nothing in those directories is written.
    """
    yeast = inventory[inventory["inchikey"].notna()]
    y_full = set(yeast["inchikey"])
    y_skel = set(yeast["inchikey_skeleton"])

    summary = []
    members = []
    audits = []
    for label, path in [("dosed", DOSED_EMBED_DIR), ("media", MEDIA_EMBED_DIR)]:
        if not osp.isdir(path):
            summary.append({"set": label, "present": False})
            continue
        keys, audit = consensus_keys(path)
        audit.insert(0, "set", label)
        audits.append(audit)
        skel = {k.split("-")[0] for k in keys}
        shared_full = sorted(y_full & keys)
        shared_skel = sorted(y_skel & skel)
        summary.append(
            {
                "set": label,
                "present": True,
                "n_other": len(keys),
                "n_yeast9_with_inchikey": len(y_full),
                "n_shared_full_key": len(shared_full),
                "n_shared_skeleton": len(shared_skel),
            }
        )
        # Several Yeast9 species can share one skeleton, because the shipped SMILES are
        # largely stereo-free: L- and D- forms of an amino acid collapse onto the same
        # connectivity block. Every such species is named, not just the first.
        for s in shared_skel:
            hits = yeast[yeast["inchikey_skeleton"] == s]
            members.append(
                {
                    "set": label,
                    "inchikey_skeleton": s,
                    "n_yeast9_species_on_skeleton": len(hits),
                    "yeast9_species_names": ";".join(sorted(hits["species_name"])),
                    "yeast9_inchikeys": ";".join(sorted(set(hits["inchikey"]))),
                    "exact_key_match": bool(set(hits["inchikey"]) & keys),
                    "structure_classes": ";".join(sorted(set(hits["structure_class"]))),
                }
            )
    return (
        pd.DataFrame(summary),
        pd.DataFrame(members),
        pd.concat(audits, ignore_index=True) if audits else pd.DataFrame(),
    )


def markdown_table(df: pd.DataFrame) -> str:
    """Pipe-delimited markdown for a small frame (no tabulate dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS)
    )
    ap.add_argument("--skip-encoders", action="store_true")
    args = ap.parse_args()

    inventory, counts = build_inventory()
    inventory.to_csv(
        osp.join(RESULTS_DIR, "yeast9_structure_inventory.csv"), index=False
    )

    bearing = inventory["smiles"].notna()
    counts["n_species_with_smiles"] = int(bearing.sum())
    counts["n_species_without_smiles"] = int((~bearing).sum())
    counts["n_species_smiles_from_smilesdb"] = int(inventory["in_smiles_db"].sum())
    counts["n_species_smiles_from_identity_table"] = int(
        (inventory["smiles_source"].notna() & ~inventory["in_smiles_db"]).sum()
    )
    counts["n_species_rdkit_parses"] = int(inventory["rdkit_parses"].sum())
    counts["n_distinct_inchikey"] = int(inventory["inchikey"].nunique())
    counts["n_distinct_inchikey_skeleton"] = int(
        inventory["inchikey_skeleton"].nunique()
    )
    counts["n_species_formula_disagrees_with_smiles"] = int(
        (inventory["formula_agrees"] == False).sum()  # noqa: E712
    )
    counts["n_species_stereo_free_inchikey"] = int(
        (inventory["stereo_free_inchikey"] == True).sum()  # noqa: E712
    )
    counts["n_species_with_unassigned_stereocenters"] = int(
        (inventory["n_unassigned_stereocenters"].fillna(0) > 0).sum()
    )
    for col in ["chebi", "kegg_compound", "metanetx_chemical", "bigg_metabolite"]:
        counts[f"n_species_with_{col}"] = int(inventory[col].notna().sum())
    counts["n_species_with_formation_dg"] = int(inventory["has_formation_dg"].sum())

    pd.DataFrame([counts]).T.rename(columns={0: "value"}).to_csv(
        osp.join(RESULTS_DIR, "yeast9_structure_summary.csv"), index_label="statistic"
    )
    print(json.dumps(counts, indent=2))

    cls = (
        inventory.groupby("structure_class")
        .agg(
            n_species=("species_name", "size"),
            n_compartment_copies=("n_compartment_copies", "sum"),
            n_with_smiles=("smiles", lambda s: int(s.notna().sum())),
            n_with_chebi=("chebi", lambda s: int(s.notna().sum())),
            n_with_formation_dg=("has_formation_dg", "sum"),
        )
        .sort_values("n_species", ascending=False)
        .reset_index()
    )
    cls["n_without_smiles"] = cls["n_species"] - cls["n_with_smiles"]
    cls.to_csv(osp.join(RESULTS_DIR, "yeast9_failure_classes.csv"), index=False)
    print(cls.to_string(index=False))

    ov_summary, ov_members, ov_audit = overlap(inventory)
    ov_summary.to_csv(osp.join(RESULTS_DIR, "yeast9_overlap_summary.csv"), index=False)
    ov_members.to_csv(osp.join(RESULTS_DIR, "yeast9_overlap_members.csv"), index=False)
    ov_audit.to_csv(
        osp.join(RESULTS_DIR, "yeast9_overlap_source_audit.csv"), index=False
    )
    print(ov_audit.to_string(index=False))
    print(ov_summary.to_string(index=False))

    if args.skip_encoders:
        return

    cov, failures, fail_rows = run_encoders(inventory, args.encoders)
    cov.to_csv(osp.join(RESULTS_DIR, "yeast9_encoder_coverage.csv"), index=False)
    with open(osp.join(RESULTS_DIR, "yeast9_encoder_failures.json"), "w") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    if not fail_rows.empty:
        fail_rows.to_csv(
            osp.join(RESULTS_DIR, "yeast9_encoder_failure_detail.csv"), index=False
        )
        pivot = (
            fail_rows.groupby(["encoder", "reason", "structure_class"])
            .size()
            .reset_index(name="n")
        )
        pivot.to_csv(
            osp.join(RESULTS_DIR, "yeast9_encoder_failure_classes.csv"), index=False
        )
        print(pivot.to_string(index=False))
    with open(osp.join(RESULTS_DIR, "yeast9_encoder_coverage.md"), "w") as f:
        f.write(markdown_table(cov))
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
