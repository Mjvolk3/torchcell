# experiments/026-metabolism-flux/scripts/audit_unresolved_metabolites.py
# [[experiments.026-metabolism-flux.scripts.audit_unresolved_metabolites]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/audit_unresolved_metabolites.py

r"""Say exactly which yeast-GEM metabolites eQuilibrator cannot price, and why.

``build_equilibrator_thermo.py`` resolves a metabolite through an ordered accession tier
list and reports how many landed in each tier. That histogram hides two different failures
behind one number, and the difference decides whether coverage is fixable:

1. **No accession matched.** Nothing in ``ACCESSION_TIERS`` found a cache record.
2. **A record matched but carries no structure.** ``get_compound`` returns a Compound whose
   ``group_vector`` is ``None``, so ``standard_dg_formation`` returns ``(None, None, None)``
   and no formation energy exists. The tier histogram counts this as a hit.

Reaction coverage needs every participant PRICED, not merely matched, so both failures cost
the same. This script measures both, classifies every failing metabolite by chemical class,
assigns a diagnosis bucket, and then attempts three recoveries, reporting the metabolite and
REACTION coverage after each so the gain is attributable to the step that produced it.

RECOVERY STEPS
--------------
``R1`` namespaces the tier list ignores that the eQuilibrator cache does index
(``seed.compound``, ``hmdb``, ``lipidmaps``, ``reactome``, ``biocyc`` -> ``metacyc.compound``).
``pubchem.compound`` appears in the SBML and is NOT a cache registry, so it is recorded as
unsupported rather than tried.

``R2`` MetaNetX ``chem_xref.tsv``. The SBML's MetaNetX ids come from an older MNXref release,
so a deprecated ``MNXM`` (or a BiGG/ChEBI/KEGG accession) can be re-mapped onto the current
``MNXM`` the cache knows. This is the step that repairs an id the cache has simply renamed.

``R3`` structure. An InChI from MetaNetX ``chem_prop.tsv`` or a SMILES from the GEM's own
``smilesDB.tsv``, converted to an InChIKey and matched against the cache on the connectivity
layer (first 14 characters). The shipped ``inchi`` tier uses ``get_compound_by_inchi``, which
is an exact full-string match on a protonation-state-specific InChI, and that is why it hit
twice in the entire model.

NO FALLBACKS. Every lookup that fails is recorded with the reason it failed; nothing is
silently substituted, and a metabolite that no step recovers stays unpriced in the report.

Run under the eQuilibrator environment:

    PYTHONPATH=<worktree-root> \
    /scratch/projects/torchcell-scratch/envs/equilibrator/bin/python \
        experiments/026-metabolism-flux/scripts/audit_unresolved_metabolites.py
"""

import csv
import importlib.util
import json
import os
import os.path as osp
import re
from datetime import UTC, datetime

import cobra
from dotenv import load_dotenv
from equilibrator_api import ComponentContribution
from pydantic import BaseModel, Field

WORKTREE = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
load_dotenv(osp.join(WORKTREE, ".env"))
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
METANETX_DIR = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "metanetx")
CHEM_XREF = osp.join(METANETX_DIR, "chem_xref.tsv")
CHEM_PROP = osp.join(METANETX_DIR, "chem_prop.tsv")

BUILD_SCRIPT = osp.join(
    EXPERIMENT_ROOT, "026-metabolism-flux", "scripts", "build_equilibrator_thermo.py"
)

# Annotation keys on a species that name a CHEMICAL. Everything else the SBML carries
# (pubmed, kegg.pathway, ec-code, sbo) names a paper, a pathway, or a reaction class, and
# can never resolve a compound.
CHEMICAL_ANNOTATION_KEYS = frozenset(
    {
        "metanetx.chemical",
        "chebi",
        "kegg.compound",
        "bigg.metabolite",
        "seed.compound",
        "pubchem.compound",
        "reactome",
        "biocyc",
        "hmdb",
        "lipidmaps",
        "swisslipid",
        "sabiork.compound",
    }
)

# SBML annotation key -> eQuilibrator cache registry namespace. Only the first four are in
# the shipped ACCESSION_TIERS; the rest are what R1 adds.
CACHE_REGISTRY_FOR_KEY: dict[str, str] = {
    "metanetx.chemical": "metanetx.chemical",
    "chebi": "chebi",
    "kegg.compound": "kegg",
    "bigg.metabolite": "bigg.metabolite",
    "seed.compound": "seed",
    "reactome": "reactome",
    "biocyc": "metacyc.compound",
    "hmdb": "hmdb",
    "lipidmaps": "lipidmaps",
    "swisslipid": "swisslipid",
    "sabiork.compound": "sabiork.compound",
}

# Namespaces the SBML uses that the cache has no registry for. Recorded, never tried.
UNSUPPORTED_KEYS = frozenset({"pubchem.compound"})

# SBML annotation key -> the source prefix MetaNetX chem_xref.tsv uses in its first column.
# A key can have several spellings across MNXref releases, so each maps to a tuple.
XREF_PREFIX_FOR_KEY: dict[str, tuple[str, ...]] = {
    "metanetx.chemical": ("mnx",),
    "chebi": ("chebi", "CHEBI"),
    "kegg.compound": ("kegg.compound", "keggC"),
    "bigg.metabolite": ("bigg.metabolite", "biggM"),
    "seed.compound": ("seed.compound", "seedM"),
    "reactome": ("reactome", "reactomeM"),
    "biocyc": ("metacyc.compound", "metacycM"),
    "hmdb": ("hmdb",),
    "lipidmaps": ("lipidmaps", "lipidmapsM"),
}

# Chemical class from the metabolite name. ORDER MATTERS: the first pattern that matches
# wins, so the specific lipid headgroups come before the generic acyl/chain catch-alls.
CLASS_PATTERNS: list[tuple[str, str]] = [
    ("tRNA_linked", r"tRNA"),
    ("acyl_CoA", r"\bCoA\b|-CoA"),
    ("cardiolipin_CL", r"cardiolipin"),
    (
        "phosphatidylinositol_PI",
        r"phosphatidyl-1D-myo-inositol|phosphatidylinositol|lysophosphatidylinositol"
        r"|glycerophosphoinositol",
    ),
    (
        "phosphatidylethanolamine_PE",
        r"phosphatidylethanolamine|phosphatidyl-N-methylethanolamine"
        r"|phosphatidyl-N,N-dimethylethanolamine|glycerophosphoethanolamine",
    ),
    ("phosphatidylcholine_PC", r"phosphatidylcholine|glycerophosphocholine"),
    ("phosphatidylserine_PS", r"phosphatidylserine|glycerophosphoserine"),
    ("phosphatidylglycerol_PG", r"phosphatidyl.*glycerol|phosphatidylglycerol"),
    (
        "phosphatidate_PA",
        r"phosphatidate|diacylglycerol 3-diphosphate|CDP-diacylglycerol",
    ),
    ("triacylglycerol_TAG", r"triglyceride|triacylglycerol|diglyceride|monoglyceride"),
    (
        "sphingolipid",
        r"ceramide|long-chain base|sphinganine|sphingosine|phytosphingosine"
        r"|inositol phosphoryl",
    ),
    ("sterol_ester", r"steryl ester|sterol ester|ergosteryl|zymosteryl ester"),
    (
        "protein_linked",
        r"^\[|\]\s*$|\[.*\]|carrier\)|\bACP\b|scaffold protein|desulfurase",
    ),
    ("glycan", r"^G\d{5}$|glycosyl|dolichol|chitobiosyl|sugar acceptor|glucan|Starch"),
    (
        "fatty_acid_acyl_chain",
        r"\bchain\b|fatty acid|acyl|acylglycerone|\bC\d+:\d+\b",
    ),
    (
        "generic_pseudo_metabolite",
        r"^biomass$|^RNA$|^DNA$|^protein$|^lipid$|^cofactor$|^ion$|backbone$",
    ),
]

# Elements that are not elements. A formula carrying one of these cannot be decomposed into
# groups, so no component-contribution estimate exists for it no matter which id resolves.
PLACEHOLDER_TOKENS = ("R", "X", "*")


class RecoveryAttempt(BaseModel):
    """One lookup that was actually performed, and what came back."""

    step: str
    query: str
    outcome: str
    compound_id: int | None = None


class MetaboliteAudit(BaseModel):
    """Everything measured about one compartment-specific metabolite."""

    met_id: str
    name: str
    compartment: str
    formula: str | None
    charge: float | None
    annotations: dict[str, list[str]]
    chemical_annotation_keys: list[str]
    chemical_class: str
    has_placeholder_formula: bool
    baseline_tier: str | None
    baseline_compound_id: int | None
    baseline_has_structure: bool
    baseline_priceable: bool
    diagnosis: str = "unassigned"
    recovered_by: str | None = None
    recovered_compound_id: int | None = None
    recovered_priceable: bool = False
    attempts: list[RecoveryAttempt] = Field(default_factory=list)


class StepCoverage(BaseModel):
    """Metabolite and reaction coverage measured after one recovery step."""

    step: str
    n_metabolites_with_compound: int
    n_metabolites_priceable: int
    frac_metabolites_priceable: float
    n_reactions_all_participants_priceable: int
    frac_reactions_all_participants_priceable: float
    delta_metabolites_priceable_vs_previous: int
    delta_reactions_vs_previous: int


def load_build_module():
    """Import the shipped build script by path so the audit uses its exact tier logic."""
    spec = importlib.util.spec_from_file_location("build_equilibrator_thermo", BUILD_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify(name: str) -> str:
    """Chemical class from the metabolite name, first matching pattern wins."""
    for label, pattern in CLASS_PATTERNS:
        if re.search(pattern, name, flags=re.IGNORECASE):
            return label
    return "other"


def has_placeholder_formula(formula: str | None) -> bool:
    """True when the formula is missing or carries a placeholder element (R, X, *)."""
    if not formula:
        return True
    # Split into element tokens: an uppercase letter plus optional lowercase letters.
    tokens = re.findall(r"[A-Z][a-z]?|\*", formula)
    return any(token in PLACEHOLDER_TOKENS for token in tokens)


def normalize_annotations(annotation: dict) -> dict[str, list[str]]:
    """cobra gives a str or a list per key; make every value a list of str."""
    out: dict[str, list[str]] = {}
    for key, value in annotation.items():
        values = value if isinstance(value, list) else [value]
        out[key] = [str(v) for v in values]
    return out


def cache_query(registry: str, accession: str) -> str:
    """The string CompoundCache.get_compound expects for this registry."""
    if registry == "chebi":
        return accession if accession.upper().startswith("CHEBI:") else f"CHEBI:{accession}"
    return f"{registry}:{accession}"


def reaction_coverage(model, priceable: set[str]) -> int:
    """Reactions whose every participant has a formation energy."""
    return sum(
        1
        for reaction in model.reactions
        if all(m.id in priceable for m in reaction.metabolites)
    )


def load_xref_index(wanted: set[str]) -> dict[str, str]:
    """chem_xref source key -> current MNXM id, restricted to the keys we will ask for.

    ``wanted`` holds fully-qualified ``prefix:accession`` strings; keeping the index to
    those avoids holding all 1.4 million xref rows in memory.
    """
    index: dict[str, str] = {}
    with open(CHEM_XREF) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            source, mnx_id = parts[0], parts[1]
            if source in wanted:
                index[source] = mnx_id
    return index


def load_chem_prop(wanted: set[str]) -> dict[str, tuple[str, str, str]]:
    """MNXM id -> (name, InChI, SMILES) for the ids we will ask for."""
    out: dict[str, tuple[str, str, str]] = {}
    with open(CHEM_PROP) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[0] not in wanted:
                continue
            out[parts[0]] = (parts[1], parts[6], parts[8])
    return out


def inchi_key_from_structure(inchi: str, smiles: str) -> str | None:
    """InChIKey from an InChI when there is one, else from a SMILES via RDKit."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if inchi:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            return Chem.MolToInchiKey(mol)
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return Chem.MolToInchiKey(mol)
    return None


def main() -> None:
    """Characterize, diagnose, recover, and write the report."""
    os.makedirs(RESULTS, exist_ok=True)
    build = load_build_module()

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    smiles_db = build.read_smiles_db(
        osp.join(GEM_DIR, "data", "databases", "smilesDB.tsv")
    )
    cc = ComponentContribution()
    ccache = cc.ccache

    # ------------------------------------------------------------------ baseline
    # Re-run the shipped resolution, then ALSO ask whether the matched compound can
    # actually be priced. The build script's tier histogram stops at the match.
    print("baseline resolution", flush=True)
    audits: list[MetaboliteAudit] = []
    chemical_cache: dict[str, tuple] = {}
    for index, metabolite in enumerate(model.metabolites):
        if index % 500 == 0:
            print(f"  {index}/{len(model.metabolites)}", flush=True)
        key = (metabolite.name or metabolite.id).lower()
        if key not in chemical_cache:
            compound, tier, _accession = build.resolve_compound(cc, metabolite, smiles_db)
            priceable = False
            if compound is not None:
                mu, _fin, _inf = cc.standard_dg_formation(compound)
                priceable = mu is not None
            chemical_cache[key] = (compound, tier, priceable)
        compound, tier, priceable = chemical_cache[key]

        annotations = normalize_annotations(metabolite.annotation)
        chem_keys = sorted(set(annotations) & CHEMICAL_ANNOTATION_KEYS)
        audits.append(
            MetaboliteAudit(
                met_id=metabolite.id,
                name=metabolite.name or "",
                compartment=metabolite.compartment,
                formula=metabolite.formula,
                charge=metabolite.charge,
                annotations=annotations,
                chemical_annotation_keys=chem_keys,
                chemical_class=classify(metabolite.name or metabolite.id),
                has_placeholder_formula=has_placeholder_formula(metabolite.formula),
                baseline_tier=tier,
                baseline_compound_id=compound.id if compound is not None else None,
                baseline_has_structure=(
                    compound is not None and compound.inchi_key is not None
                ),
                baseline_priceable=priceable,
            )
        )

    by_id = {a.met_id: a for a in audits}
    priceable_ids = {a.met_id for a in audits if a.baseline_priceable}
    coverage_steps: list[StepCoverage] = [
        StepCoverage(
            step="baseline",
            n_metabolites_with_compound=sum(
                1 for a in audits if a.baseline_compound_id is not None
            ),
            n_metabolites_priceable=len(priceable_ids),
            frac_metabolites_priceable=len(priceable_ids) / len(audits),
            n_reactions_all_participants_priceable=reaction_coverage(model, priceable_ids),
            frac_reactions_all_participants_priceable=(
                reaction_coverage(model, priceable_ids) / len(model.reactions)
            ),
            delta_metabolites_priceable_vs_previous=0,
            delta_reactions_vs_previous=0,
        )
    ]

    # ------------------------------------------------------------------ diagnosis
    # Buckets are mutually exclusive and applied in this order. Precedence is deliberate:
    # a metabolite whose formula cannot be decomposed is a ceiling regardless of which id
    # resolves it, so that test runs first.
    print("diagnosis", flush=True)
    for audit in audits:
        if audit.baseline_priceable:
            audit.diagnosis = "priceable_baseline"
        elif audit.baseline_compound_id is not None and not audit.baseline_has_structure:
            audit.diagnosis = "cache_record_without_structure"
        elif audit.baseline_compound_id is not None:
            audit.diagnosis = "cache_structure_without_group_decomposition"
        elif audit.has_placeholder_formula and not audit.chemical_annotation_keys:
            audit.diagnosis = "unannotated_placeholder_formula"
        elif not audit.chemical_annotation_keys:
            audit.diagnosis = "unannotated_definite_formula"
        else:
            audit.diagnosis = "annotated_but_unmatched"

    # ------------------------------------------------------- R1: extra namespaces
    print("R1 extra namespaces", flush=True)
    for audit in audits:
        if audit.baseline_priceable:
            continue
        for key in audit.chemical_annotation_keys:
            if key in UNSUPPORTED_KEYS:
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R1",
                        query=f"{key}:{audit.annotations[key][0]}",
                        outcome="namespace_not_a_cache_registry",
                    )
                )
                continue
            if key in {t[0] for t in build.ACCESSION_TIERS}:
                continue  # already tried by the shipped tier list
            registry = CACHE_REGISTRY_FOR_KEY[key]
            for accession in audit.annotations[key]:
                query = cache_query(registry, accession)
                compound = ccache.get_compound(query)
                if compound is None:
                    audit.attempts.append(
                        RecoveryAttempt(step="R1", query=query, outcome="not_in_cache")
                    )
                    continue
                mu, _fin, _inf = cc.standard_dg_formation(compound)
                if mu is None:
                    audit.attempts.append(
                        RecoveryAttempt(
                            step="R1",
                            query=query,
                            outcome="matched_but_no_group_decomposition",
                            compound_id=compound.id,
                        )
                    )
                    continue
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R1", query=query, outcome="priceable", compound_id=compound.id
                    )
                )
                audit.recovered_by = "R1_extra_namespace"
                audit.recovered_compound_id = compound.id
                audit.recovered_priceable = True
                break
            if audit.recovered_priceable:
                break

    def record_step(label: str) -> None:
        """Measure metabolite and reaction coverage after a recovery step."""
        current = {
            a.met_id for a in audits if a.baseline_priceable or a.recovered_priceable
        }
        n_reactions = reaction_coverage(model, current)
        previous = coverage_steps[-1]
        coverage_steps.append(
            StepCoverage(
                step=label,
                n_metabolites_with_compound=sum(
                    1
                    for a in audits
                    if a.baseline_compound_id is not None
                    or a.recovered_compound_id is not None
                ),
                n_metabolites_priceable=len(current),
                frac_metabolites_priceable=len(current) / len(audits),
                n_reactions_all_participants_priceable=n_reactions,
                frac_reactions_all_participants_priceable=n_reactions
                / len(model.reactions),
                delta_metabolites_priceable_vs_previous=len(current)
                - previous.n_metabolites_priceable,
                delta_reactions_vs_previous=n_reactions
                - previous.n_reactions_all_participants_priceable,
            )
        )

    record_step("R1_extra_namespaces")

    # --------------------------------------------------- R2: MetaNetX chem_xref
    print("R2 MetaNetX chem_xref", flush=True)
    wanted_xref: set[str] = set()
    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        for key in audit.chemical_annotation_keys:
            if key not in XREF_PREFIX_FOR_KEY:
                continue
            for accession in audit.annotations[key]:
                bare = accession.split(":", 1)[-1] if key == "chebi" else accession
                for prefix in XREF_PREFIX_FOR_KEY[key]:
                    wanted_xref.add(f"{prefix}:{bare}")
                    if key == "chebi":
                        wanted_xref.add(f"{prefix}:CHEBI:{bare}")
    print(f"  xref keys wanted: {len(wanted_xref)}", flush=True)
    xref_index = load_xref_index(wanted_xref)
    print(f"  xref keys found: {len(xref_index)}", flush=True)

    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        for key in audit.chemical_annotation_keys:
            if key not in XREF_PREFIX_FOR_KEY:
                continue
            for accession in audit.annotations[key]:
                bare = accession.split(":", 1)[-1] if key == "chebi" else accession
                candidates = [f"{p}:{bare}" for p in XREF_PREFIX_FOR_KEY[key]]
                if key == "chebi":
                    candidates += [f"{p}:CHEBI:{bare}" for p in XREF_PREFIX_FOR_KEY[key]]
                for candidate in candidates:
                    mnx_id = xref_index.get(candidate)
                    if mnx_id is None:
                        continue
                    query = f"metanetx.chemical:{mnx_id}"
                    compound = ccache.get_compound(query)
                    if compound is None:
                        audit.attempts.append(
                            RecoveryAttempt(
                                step="R2",
                                query=f"{candidate} -> {query}",
                                outcome="remapped_mnx_not_in_cache",
                            )
                        )
                        continue
                    mu, _fin, _inf = cc.standard_dg_formation(compound)
                    if mu is None:
                        audit.attempts.append(
                            RecoveryAttempt(
                                step="R2",
                                query=f"{candidate} -> {query}",
                                outcome="matched_but_no_group_decomposition",
                                compound_id=compound.id,
                            )
                        )
                        continue
                    audit.attempts.append(
                        RecoveryAttempt(
                            step="R2",
                            query=f"{candidate} -> {query}",
                            outcome="priceable",
                            compound_id=compound.id,
                        )
                    )
                    audit.recovered_by = "R2_metanetx_xref"
                    audit.recovered_compound_id = compound.id
                    audit.recovered_priceable = True
                    break
                if audit.recovered_priceable:
                    break
            if audit.recovered_priceable:
                break

    record_step("R2_metanetx_xref")

    # ------------------------------------------------------- R3: structure search
    print("R3 structure search", flush=True)
    # Collect every MNXM id still reachable for a still-unpriced metabolite, so chem_prop
    # is scanned once.
    wanted_mnx: set[str] = set()
    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        for accession in audit.annotations.get("metanetx.chemical", []):
            wanted_mnx.add(accession)
        for key in audit.chemical_annotation_keys:
            if key not in XREF_PREFIX_FOR_KEY:
                continue
            for accession in audit.annotations[key]:
                bare = accession.split(":", 1)[-1] if key == "chebi" else accession
                for prefix in XREF_PREFIX_FOR_KEY[key]:
                    mapped = xref_index.get(f"{prefix}:{bare}")
                    if mapped:
                        wanted_mnx.add(mapped)
    print(f"  chem_prop ids wanted: {len(wanted_mnx)}", flush=True)
    chem_prop = load_chem_prop(wanted_mnx)
    print(f"  chem_prop ids found: {len(chem_prop)}", flush=True)

    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        structures: list[tuple[str, str, str]] = []  # (source, inchi, smiles)
        for accession in audit.annotations.get("metanetx.chemical", []):
            if accession in chem_prop:
                _name, inchi, smiles = chem_prop[accession]
                structures.append((f"chem_prop:{accession}", inchi, smiles))
        for key in audit.chemical_annotation_keys:
            if key not in XREF_PREFIX_FOR_KEY:
                continue
            for accession in audit.annotations[key]:
                bare = accession.split(":", 1)[-1] if key == "chebi" else accession
                for prefix in XREF_PREFIX_FOR_KEY[key]:
                    mapped = xref_index.get(f"{prefix}:{bare}")
                    if mapped and mapped in chem_prop:
                        _name, inchi, smiles = chem_prop[mapped]
                        structures.append((f"chem_prop:{mapped}", inchi, smiles))
        smiles = smiles_db.get(audit.name.lower())
        if smiles:
            structures.append(("smilesDB", "", smiles))

        if not structures:
            audit.attempts.append(
                RecoveryAttempt(step="R3", query=audit.name, outcome="no_structure_source")
            )
            continue

        for source, inchi, smiles_value in structures:
            inchi_key = inchi_key_from_structure(inchi, smiles_value)
            if inchi_key is None:
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R3", query=source, outcome="structure_not_parseable"
                    )
                )
                continue
            hits = ccache.search_compound_by_inchi_key(inchi_key[:14])
            if not hits:
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R3",
                        query=f"{source} -> {inchi_key[:14]}",
                        outcome="inchikey_not_in_cache",
                    )
                )
                continue
            for compound in hits:
                mu, _fin, _inf = cc.standard_dg_formation(compound)
                if mu is None:
                    audit.attempts.append(
                        RecoveryAttempt(
                            step="R3",
                            query=f"{source} -> {inchi_key[:14]}",
                            outcome="matched_but_no_group_decomposition",
                            compound_id=compound.id,
                        )
                    )
                    continue
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R3",
                        query=f"{source} -> {inchi_key[:14]}",
                        outcome="priceable",
                        compound_id=compound.id,
                    )
                )
                audit.recovered_by = "R3_inchikey_structure"
                audit.recovered_compound_id = compound.id
                audit.recovered_priceable = True
                break
            if audit.recovered_priceable:
                break

    record_step("R3_inchikey_structure")

    # ------------------------------------------------------------------- report
    unpriced = [a for a in audits if not (a.baseline_priceable or a.recovered_priceable)]
    baseline_unmatched = [a for a in audits if a.baseline_compound_id is None]

    def counter(records: list[MetaboliteAudit], field: str) -> dict[str, int]:
        out: dict[str, int] = {}
        for record in records:
            value = str(getattr(record, field))
            out[value] = out.get(value, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # A metabolite is un-priceable in principle when its formula cannot be decomposed into
    # groups. That is the ceiling; everything else is a mapping gap.
    ceiling = [a for a in unpriced if a.has_placeholder_formula]
    fixable = [a for a in unpriced if not a.has_placeholder_formula]

    report = {
        "built_at": datetime.now(UTC).isoformat(),
        "gem_model_sha256": build.sha256_file(osp.join(GEM_DIR, "model", "yeast-GEM.xml")),
        "n_metabolites": len(audits),
        "n_reactions": len(model.reactions),
        "n_distinct_chemicals": len(chemical_cache),
        "baseline": {
            "n_unmatched_by_accession": len(baseline_unmatched),
            "n_matched_but_unpriceable": sum(
                1
                for a in audits
                if a.baseline_compound_id is not None and not a.baseline_priceable
            ),
            "n_priceable": sum(1 for a in audits if a.baseline_priceable),
            "unmatched_distinct_names": len({a.name for a in baseline_unmatched}),
            "unmatched_by_class": counter(baseline_unmatched, "chemical_class"),
            "unmatched_annotation_key_profile": counter(
                baseline_unmatched, "chemical_annotation_keys"
            ),
        },
        "diagnosis_buckets_all_unpriced_at_baseline": counter(
            [a for a in audits if not a.baseline_priceable], "diagnosis"
        ),
        "diagnosis_buckets_by_class": {
            cls: counter(
                [
                    a
                    for a in audits
                    if not a.baseline_priceable and a.chemical_class == cls
                ],
                "diagnosis",
            )
            for cls in sorted(
                {a.chemical_class for a in audits if not a.baseline_priceable}
            )
        },
        "coverage_by_step": [s.model_dump() for s in coverage_steps],
        "recovered_by": counter([a for a in audits if a.recovered_priceable], "recovered_by"),
        "recovered_by_class": counter(
            [a for a in audits if a.recovered_priceable], "chemical_class"
        ),
        "remaining_unpriced": {
            "n": len(unpriced),
            "n_distinct_names": len({a.name for a in unpriced}),
            "by_class": counter(unpriced, "chemical_class"),
            "by_diagnosis": counter(unpriced, "diagnosis"),
            "n_placeholder_formula_ceiling": len(ceiling),
            "frac_of_remaining_that_is_ceiling": (
                len(ceiling) / len(unpriced) if unpriced else 0.0
            ),
            "n_definite_formula_still_unpriced": len(fixable),
            "definite_formula_by_class": counter(fixable, "chemical_class"),
        },
        "attempt_outcome_counts": {
            step: {
                outcome: sum(
                    1
                    for a in audits
                    for t in a.attempts
                    if t.step == step and t.outcome == outcome
                )
                for outcome in sorted(
                    {t.outcome for a in audits for t in a.attempts if t.step == step}
                )
            }
            for step in ("R1", "R2", "R3")
        },
        "metabolites_unpriced_at_baseline": [
            a.model_dump()
            for a in audits
            if not a.baseline_priceable
        ],
    }

    json_path = osp.join(RESULTS, "unresolved_metabolites.json")
    with open(json_path, "w") as handle:
        json.dump(report, handle, indent=2)

    csv_path = osp.join(RESULTS, "unresolved_metabolites.csv")
    columns = [
        "met_id",
        "name",
        "compartment",
        "formula",
        "charge",
        "chemical_class",
        "has_placeholder_formula",
        "chemical_annotation_keys",
        "annotations",
        "baseline_tier",
        "baseline_compound_id",
        "baseline_has_structure",
        "diagnosis",
        "recovered_by",
        "recovered_compound_id",
        "attempt_outcomes",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for audit in audits:
            if audit.baseline_priceable:
                continue
            writer.writerow(
                [
                    audit.met_id,
                    audit.name,
                    audit.compartment,
                    audit.formula or "",
                    audit.charge,
                    audit.chemical_class,
                    audit.has_placeholder_formula,
                    ";".join(audit.chemical_annotation_keys),
                    ";".join(
                        f"{k}={'|'.join(v)}"
                        for k, v in sorted(audit.annotations.items())
                        if k in CHEMICAL_ANNOTATION_KEYS
                    ),
                    audit.baseline_tier or "",
                    audit.baseline_compound_id if audit.baseline_compound_id else "",
                    audit.baseline_has_structure,
                    audit.diagnosis,
                    audit.recovered_by or "",
                    audit.recovered_compound_id if audit.recovered_compound_id else "",
                    ";".join(f"{t.step}:{t.outcome}" for t in audit.attempts),
                ]
            )

    printable = {k: v for k, v in report.items() if k != "metabolites_unpriced_at_baseline"}
    print(json.dumps(printable, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
