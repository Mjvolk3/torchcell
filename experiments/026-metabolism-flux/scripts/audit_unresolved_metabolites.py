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

THE COMPOSITION GUARD, AND WHY IT IS NOT OPTIONAL
--------------------------------------------------
Every candidate compound, in every step, must carry the GEM metabolite's heavy-atom
composition before it is accepted. Hydrogen is excluded because the cache stores one
protonation state and the GEM writes another; nothing else is negotiable. A metabolite whose
GEM formula carries a placeholder element (``R``, ``X``, ``*``) has no composition to check
against, so it is refused a structural match outright rather than matched on a partial
structure.

Both halves were written after an unguarded version of this script priced ``Ala-tRNA(Ala)``
as water: MetaNetX gives that species a SMILES containing a dummy atom, RDKit returns an
EMPTY InChIKey for it rather than failing, the empty key reaches the cache as ``LIKE '%'``,
and the first row of the entire compound table comes back as a hit. A structural search
without a composition check does not fail loudly, it succeeds wrongly.

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
import pickle
import re
import sqlite3
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
    ("fatty_acid_acyl_chain", r"\bchain\b|fatty acid|acyl|acylglycerone|\bC\d+:\d+\b"),
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
    baseline_composition_matches: bool
    diagnosis: str = "unassigned"
    recovered_by: str | None = None
    recovered_compound_id: int | None = None
    recovered_priceable: bool = False
    final_blocker: str | None = None
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


def group_decomposition_by_mass(ccache) -> list[dict]:
    """Per mass band, how many cache compounds carry a group vector at all.

    The lipid species this model is short of are large, so the question of whether
    recovering their structures would even help is answerable directly: read the cache and
    ask what fraction of compounds AT THAT SIZE component contribution can decompose. This
    is a property of the method, measured, not an expectation.
    """
    path = ccache.session.get_bind().url.database
    connection = sqlite3.connect(path)
    rows = connection.execute(
        """
        SELECT
          CASE
            WHEN mass IS NULL THEN 'unknown'
            WHEN mass < 200 THEN 'a_lt_200'
            WHEN mass < 400 THEN 'b_200_400'
            WHEN mass < 600 THEN 'c_400_600'
            WHEN mass < 800 THEN 'd_600_800'
            WHEN mass < 1000 THEN 'e_800_1000'
            ELSE 'f_ge_1000'
          END AS band,
          COUNT(*),
          SUM(CASE WHEN group_vector IS NOT NULL THEN 1 ELSE 0 END)
        FROM compounds
        GROUP BY band
        ORDER BY band
        """
    ).fetchall()
    connection.close()
    return [
        {
            "mass_band_da": band,
            "n_compounds": total,
            "n_with_group_vector": decomposed,
            "frac_with_group_vector": decomposed / total if total else 0.0,
        }
        for band, total, decomposed in rows
    ]


def compositions_present_in_cache(
    ccache, wanted: set[tuple[tuple[str, int], ...]]
) -> dict[tuple[tuple[str, int], ...], int]:
    """Which of these heavy-atom compositions the cache holds WITH a group vector.

    Answers the question the structure gap raises: if we built the missing lipid structures
    ourselves, would eQuilibrator have anything to decompose? A composition present here is
    a compound eQuilibrator can already price, so the only thing missing for that metabolite
    is a structure to hand it. One streaming pass over the compound table, because there is
    no index on composition. ``atom_bag`` and ``group_vector`` are SQLAlchemy ``PickleType``
    columns, so a raw row holds a pickle rather than JSON.
    """
    path = ccache.session.get_bind().url.database
    connection = sqlite3.connect(path)
    found: dict[tuple[tuple[str, int], ...], int] = {}
    for atom_bag_blob, group_vector in connection.execute(
        "SELECT atom_bag, group_vector FROM compounds WHERE atom_bag IS NOT NULL"
    ):
        if group_vector is None:
            continue
        bag = pickle.loads(atom_bag_blob)
        key = tuple(
            sorted(
                (element, count)
                for element, count in bag.items()
                if element not in ("H", "e-")
            )
        )
        if key in wanted:
            found[key] = found.get(key, 0) + 1
    connection.close()
    return found


def load_build_module():
    """Import the shipped build script by path so the audit uses its exact tier logic."""
    spec = importlib.util.spec_from_file_location(
        "build_equilibrator_thermo", BUILD_SCRIPT
    )
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
    """Make every annotation value a list of str, since cobra gives a str or a list."""
    out: dict[str, list[str]] = {}
    for key, value in annotation.items():
        values = value if isinstance(value, list) else [value]
        out[key] = [str(v) for v in values]
    return out


def cache_query(registry: str, accession: str) -> str:
    """The string CompoundCache.get_compound expects for this registry."""
    if registry == "chebi":
        return (
            accession
            if accession.upper().startswith("CHEBI:")
            else f"CHEBI:{accession}"
        )
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


def parse_formula(formula: str | None) -> dict[str, int] | None:
    """Element -> count, or None when the formula is missing or has a placeholder.

    Hydrogen is dropped. A cache compound is stored at one protonation state and the GEM
    writes another, so comparing H would reject correct matches; every other element is a
    hard constraint on identity.
    """
    if not formula:
        return None
    tokens = re.findall(r"([A-Z][a-z]?|\*)(\d*)", formula)
    counts: dict[str, int] = {}
    for element, digits in tokens:
        if element in PLACEHOLDER_TOKENS:
            return None
        if element == "H":
            continue
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)
    return counts


def compound_composition(compound) -> dict[str, int] | None:
    """Element -> count for a cache compound, from its atom bag, hydrogen dropped."""
    if not compound.atom_bag:
        return None
    return {
        element: count
        for element, count in compound.atom_bag.items()
        if element not in ("H", "e-")
    }


def composition_matches(gem_formula: str | None, compound) -> bool:
    """True when a candidate compound has the GEM metabolite's heavy-atom composition.

    This is the guard that separates a resolution from a substitution. Without it a
    structure search on a metabolite whose formula carries a placeholder happily returns
    an unrelated molecule, and the pipeline prices the wrong compound with no error.
    """
    wanted = parse_formula(gem_formula)
    if wanted is None:
        return False
    found = compound_composition(compound)
    if found is None:
        return False
    return wanted == found


def inchi_key_from_structure(inchi: str, smiles: str) -> str | None:
    """A full 27-character InChIKey from an InChI or a SMILES, else None.

    RDKit returns an EMPTY string rather than failing for a structure carrying a dummy
    atom (the ``[*]`` MetaNetX writes for a polymer or a protein conjugate). An empty key
    fed to the cache search becomes ``LIKE '%'``, which matches every compound in the
    database, so the caller would price the metabolite as whatever sorts first. Reject
    anything that is not a well-formed key, and reject the dummy atom outright.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    for source in (inchi, smiles):
        if not source:
            continue
        if "*" in source:
            continue  # a dummy atom means there is no definite structure to match
        mol = (
            Chem.MolFromInchi(source)
            if source.startswith("InChI=")
            else Chem.MolFromSmiles(source)
        )
        if mol is None:
            continue
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            continue
        key = Chem.MolToInchiKey(mol)
        if key and len(key) == 27:
            return key
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
            compound, tier, _accession = build.resolve_compound(
                cc, metabolite, smiles_db
            )
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
                baseline_composition_matches=(
                    compound is not None
                    and composition_matches(metabolite.formula, compound)
                ),
            )
        )

    priceable_ids = {a.met_id for a in audits if a.baseline_priceable}
    coverage_steps: list[StepCoverage] = [
        StepCoverage(
            step="baseline",
            n_metabolites_with_compound=sum(
                1 for a in audits if a.baseline_compound_id is not None
            ),
            n_metabolites_priceable=len(priceable_ids),
            frac_metabolites_priceable=len(priceable_ids) / len(audits),
            n_reactions_all_participants_priceable=reaction_coverage(
                model, priceable_ids
            ),
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
        elif (
            audit.baseline_compound_id is not None and not audit.baseline_has_structure
        ):
            audit.diagnosis = "cache_record_without_structure"
        elif audit.baseline_compound_id is not None:
            audit.diagnosis = "cache_structure_without_group_decomposition"
        elif audit.has_placeholder_formula and not audit.chemical_annotation_keys:
            audit.diagnosis = "unannotated_placeholder_formula"
        elif not audit.chemical_annotation_keys:
            audit.diagnosis = "unannotated_definite_formula"
        else:
            audit.diagnosis = "annotated_but_unmatched"

    # Every candidate compound goes through one acceptance test, so the guard cannot be
    # applied in one step and forgotten in another.
    def consider(audit: MetaboliteAudit, step: str, label: str, compound, source: str):
        """Accept a candidate only if it is priceable AND has the GEM's composition."""
        if not composition_matches(audit.formula, compound):
            audit.attempts.append(
                RecoveryAttempt(
                    step=step,
                    query=label,
                    outcome="composition_mismatch",
                    compound_id=compound.id,
                )
            )
            return False
        mu, _fin, _inf = cc.standard_dg_formation(compound)
        if mu is None:
            audit.attempts.append(
                RecoveryAttempt(
                    step=step,
                    query=label,
                    outcome="matched_but_no_group_decomposition",
                    compound_id=compound.id,
                )
            )
            return False
        audit.attempts.append(
            RecoveryAttempt(
                step=step, query=label, outcome="priceable", compound_id=compound.id
            )
        )
        audit.recovered_by = source
        audit.recovered_compound_id = compound.id
        audit.recovered_priceable = True
        return True

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
                if consider(audit, "R1", query, compound, "R1_extra_namespace"):
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
                    candidates += [
                        f"{p}:CHEBI:{bare}" for p in XREF_PREFIX_FOR_KEY[key]
                    ]
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
                    if consider(
                        audit,
                        "R2",
                        f"{candidate} -> {query}",
                        compound,
                        "R2_metanetx_xref",
                    ):
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

    # The same connectivity block is asked for by every compartment copy of a chemical, and
    # the ORM query that answers it is the slowest call in this script, so cache it.
    hits_cache: dict[str, list] = {}

    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        # A metabolite whose GEM formula carries a placeholder has no definite composition
        # to check a structural match against, so a match cannot be verified and must not
        # be made. This is the guard that stops a tRNA conjugate being priced as its free
        # amino acid.
        if parse_formula(audit.formula) is None:
            audit.attempts.append(
                RecoveryAttempt(
                    step="R3",
                    query=audit.formula or "no formula",
                    outcome="placeholder_formula_not_structurally_identifiable",
                )
            )
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
                RecoveryAttempt(
                    step="R3", query=audit.name, outcome="no_structure_source"
                )
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
            block = inchi_key[:14]
            if block not in hits_cache:
                hits_cache[block] = ccache.search_compound_by_inchi_key(block)
            hits = hits_cache[block]
            if not hits:
                audit.attempts.append(
                    RecoveryAttempt(
                        step="R3",
                        query=f"{source} -> {block}",
                        outcome="inchikey_not_in_cache",
                    )
                )
                continue
            for compound in hits:
                if consider(
                    audit,
                    "R3",
                    f"{source} -> {block}",
                    compound,
                    "R3_inchikey_structure",
                ):
                    break
            if audit.recovered_priceable:
                break

    record_step("R3_inchikey_structure")

    # ------------------------------------------------- one blocker per metabolite
    # Which of the three things stopped this metabolite, after every step has run.
    # Precedence runs from the hardest limit to the softest, so a metabolite is reported
    # under the reason that would still hold if every softer one were repaired:
    #   1. no definite structure exists at all (the model wrote a placeholder element)
    #   2. eQuilibrator HAS the molecule and still cannot decompose it into groups
    #   3. the molecule is outside eQuilibrator's structure space
    #   4. only wrong-composition candidates came back
    print("blockers", flush=True)
    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        outcomes = {t.outcome for t in audit.attempts}
        if parse_formula(audit.formula) is None:
            audit.final_blocker = "placeholder_formula_no_definite_structure"
        elif "matched_but_no_group_decomposition" in outcomes:
            audit.final_blocker = "structure_known_but_no_group_decomposition"
        elif "no_structure_source" in outcomes or "structure_not_parseable" in outcomes:
            audit.final_blocker = "no_usable_structure_available"
        elif "inchikey_not_in_cache" in outcomes:
            audit.final_blocker = "structure_absent_from_cache"
        elif "composition_mismatch" in outcomes:
            audit.final_blocker = "only_composition_mismatched_candidates"
        else:
            audit.final_blocker = "no_candidate_found"

    # ------------------------- is a missing structure the ONLY thing missing?
    # For every still-unpriced metabolite that has a definite formula, ask whether the
    # cache already holds a decomposable compound of exactly that composition. Where it
    # does, building the structure is the whole remaining task; where it does not, the
    # compound is outside eQuilibrator's coverage entirely.
    print("composition probe", flush=True)
    probe_targets: dict[str, tuple[tuple[str, int], ...]] = {}
    for audit in audits:
        if audit.baseline_priceable or audit.recovered_priceable:
            continue
        composition = parse_formula(audit.formula)
        if composition is not None:
            probe_targets[audit.met_id] = tuple(sorted(composition.items()))
    present = compositions_present_in_cache(ccache, set(probe_targets.values()))
    composition_probe = {
        "n_unpriced_with_definite_formula": len(probe_targets),
        "n_distinct_compositions": len(set(probe_targets.values())),
        "n_metabolites_whose_composition_is_decomposable_in_cache": sum(
            1 for key in probe_targets.values() if key in present
        ),
        "n_distinct_compositions_found": len(present),
        "by_class": {},
    }
    for audit in audits:
        key = probe_targets.get(audit.met_id)
        if key is None:
            continue
        bucket = composition_probe["by_class"].setdefault(
            audit.chemical_class, {"n": 0, "n_composition_decomposable_in_cache": 0}
        )
        bucket["n"] += 1
        if key in present:
            bucket["n_composition_decomposable_in_cache"] += 1

    # ------------------------------------------------------------------- report
    unpriced = [
        a for a in audits if not (a.baseline_priceable or a.recovered_priceable)
    ]
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
        "gem_model_sha256": build.sha256_file(
            osp.join(GEM_DIR, "model", "yeast-GEM.xml")
        ),
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
            # The shipped script accepts an accession match without checking that the
            # matched compound has the metabolite's composition. This counts how often
            # that acceptance prices a compound whose heavy-atom formula disagrees.
            "n_priced_with_composition_mismatch": sum(
                1
                for a in audits
                if a.baseline_priceable and not a.baseline_composition_matches
            ),
            "priced_with_composition_mismatch_by_class": counter(
                [
                    a
                    for a in audits
                    if a.baseline_priceable and not a.baseline_composition_matches
                ],
                "chemical_class",
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
        "cache_group_decomposition_by_mass": group_decomposition_by_mass(ccache),
        "composition_probe_for_unpriced": composition_probe,
        "recovered_by": counter(
            [a for a in audits if a.recovered_priceable], "recovered_by"
        ),
        "recovered_by_class": counter(
            [a for a in audits if a.recovered_priceable], "chemical_class"
        ),
        "remaining_unpriced": {
            "n": len(unpriced),
            "n_distinct_names": len({a.name for a in unpriced}),
            "by_class": counter(unpriced, "chemical_class"),
            "by_diagnosis": counter(unpriced, "diagnosis"),
            "by_final_blocker": counter(unpriced, "final_blocker"),
            "by_final_blocker_and_class": {
                blocker: counter(
                    [a for a in unpriced if a.final_blocker == blocker],
                    "chemical_class",
                )
                for blocker in sorted({a.final_blocker for a in unpriced})
            },
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
            a.model_dump() for a in audits if not a.baseline_priceable
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
        "baseline_composition_matches",
        "diagnosis",
        "recovered_by",
        "recovered_compound_id",
        "final_blocker",
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
                    audit.baseline_composition_matches,
                    audit.diagnosis,
                    audit.recovered_by or "",
                    audit.recovered_compound_id if audit.recovered_compound_id else "",
                    audit.final_blocker or "",
                    ";".join(f"{t.step}:{t.outcome}" for t in audit.attempts),
                ]
            )

    printable = {
        k: v for k, v in report.items() if k != "metabolites_unpriced_at_baseline"
    }
    print(json.dumps(printable, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
