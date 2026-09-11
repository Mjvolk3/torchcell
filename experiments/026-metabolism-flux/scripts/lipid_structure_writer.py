# experiments/026-metabolism-flux/scripts/lipid_structure_writer.py
# [[experiments.026-metabolism-flux.lipid-structure-writer]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/lipid_structure_writer.py

r"""Write structures for the yeast-GEM lipids whose acyl composition is in their name.

eQuilibrator prices a metabolite only if a structure reaches its group decomposer. 665
yeast-GEM 9.0.2 metabolites are unpriced after every namespace-mapping recovery step, and
:func:`external_pka_pilot.cmd_reach` splits them into four tiers. Tier b, 409 metabolites
over 206 distinct names, has a definite elemental formula and no structure in any
namespace, and is almost entirely lipids whose acyl composition is written into the name:
``phosphatidylethanolamine (1-16:1, 2-16:1)``, ``cardiolipin (1-18:1, 2-16:1, 3-16:1,
4-18:1)``, ``triglyceride (1-18:1, 2-16:1, 3-16:0)``. Those names determine a structure up
to detail that group contribution does not read, so the structure can be written rather
than looked up.

Five subcommands, each run under the environment that holds its dependencies.

``parse``
    Decompose every tier-b name into a headgroup template and a list of acyl chains with
    position, carbon count and double-bond count. Pure python. Writes
    ``results/lipid_names_parsed.json``, which lists every name that does NOT parse and
    why, by class.

``build``
    Assemble the SMILES in RDKit and run the acceptance test: the RDKit formula of the
    written structure against the formula yeast-GEM declares for that metabolite,
    charge-adjusted. A structure passes only on an exact string match. Writes
    ``results/lipid_structures.json`` and ``results/lipid_structures.csv``.

``pkas``
    Run MolGpKa over the passing structures and aggregate its SITE pKas into the
    macroscopic ladder eQuilibrator wants, with the aggregation imported verbatim from
    :mod:`external_pka_benchmark` rather than restated. Writes ``results/lipid_pkas.json``.

``price``
    Two routes per structure, at pH 7.2, I = 0.25 M, T = 303.15 K, pMg 3.0. First the
    written structure is looked up in the shipped compound cache by InChIKey, which the
    audit could not do because it had no structure to key on: once on the skeleton and
    stereochemistry blocks, then on the skeleton alone, since the structures are written
    with unspecified stereocenters and a group vector does not read stereochemistry. A hit
    carries eQuilibrator's own ChemAxon pKas and its magnesium constants, so it wins.
    Second, the compound is created through
    ``LocalCompoundCache(pkas_provided_externally=True)`` on equilibrator-assets master
    from the MolGpKa ladder. ``get_compounds`` resolves an InChIKey against the cache
    before it creates anything, so on a structure the cache already holds the second route
    returns the FIRST route's compound and the supplied pKas go unused; that is recorded
    per record rather than reported as agreement. Writes ``results/lipid_priced.json`` and
    ``results/lipid_priced.csv``.

``coverage``
    What the new compounds buy at the reaction level, reusing
    :mod:`audit_reaction_thermo_coverage` and :mod:`improve_parameter_coverage` for the
    definitions rather than restating them. Writes ``results/lipid_coverage.json``.

**Design decisions, all of them recorded because none is read off the name.**

1. **Double-bond position.** yeast-GEM writes ``16:1`` and never says where the double
   bond is. Every double bond is placed at Delta-9 counting from the carbonyl carbon, and
   at every third carbon after that (Delta-12, Delta-15), in the *cis* configuration. That
   is the yeast pattern: palmitoleoyl is 16:1(9Z) and oleoyl is 18:1(9Z), and the ``build``
   step asserts that the free 16:1 and 18:1 acids it writes carry the InChIKeys of
   palmitoleic and oleic acid. Group contribution reads a group vector that does not carry
   the double-bond position, so the choice does not move the energy; it is recorded because
   the structure is published and a reader cannot recover the convention from the name.
2. **Stereocenters are left unspecified**, including in the coenzyme A moiety taken from
   the cache. The glycerol sn-2 carbon, the inositol ring and the (R)-3-hydroxyacyl carbon
   all carry stereochemistry in the real molecule and none of it is in the yeast-GEM name
   for the glycerolipids. Double-bond geometry IS written, since the name states cis for
   the acyl-CoAs and the Delta-9 convention is cis.
3. **Protonation.** Every structure is written in its neutral form, so the reference
   microspecies eQuilibrator derives from the SMILES is the neutral one. yeast-GEM declares
   a formula at the model's own charge state, so the acceptance test compares against the
   NEUTRAL formula, ``H_neutral = H_gem - charge_gem`` with every other element unchanged.
   For the 165 lipids that pass, ``charge_gem`` is 0 and the adjustment is the identity.
4. **A created compound is NOT on the same energy zero as a cached one, and the size of
   the gap is measured here rather than assumed.** eQuilibrator derives the reference
   microspecies from the SMILES it is handed, and the untransformed formation energy is
   the group-contribution estimate for THAT species. The cache stores ChemAxon's major
   microspecies, which for a phosphoamine lipid is a zwitterion; the structure written
   here is the un-ionized neutral form. Both carry the same atom bag and the Legendre
   transform agrees between them, but the group vectors differ and so do the energies.
   ``price`` reports the difference for every name where a cache compound and a created
   compound both exist, split by whether the two reference microspecies carry the same
   number of protons. Where a cache compound exists it is the value reported, because the
   recomputed table the flux layer already reads is built from cache compounds.
5. **The SLIME backbone pseudo-metabolites are not molecules and no structure is
   attempted.** yeast-GEM's lipid pseudo-reactions carry species named ``<class>
   backbone`` whose formula is a difference formula, not a composition: ``triglyceride
   backbone`` is ``C3H2``, ``ceramide backbone`` is ``HNO2``, ``long-chain base backbone``
   is ``H4N``. They are reported as unparsed with that reason rather than forced.

Run::

    # 1. parse (any env with pydantic)
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/lipid_structure_writer.py parse

    # 2. build + acceptance test (equilibrator-ext: RDKit + the compound cache for CoA)
    ~/miniconda3/envs/equilibrator-ext/bin/python \
        experiments/026-metabolism-flux/scripts/lipid_structure_writer.py build

    # 3. pKas (pka-tools: torch + torch_scatter for MolGpKa)
    OMP_NUM_THREADS=4 ~/miniconda3/envs/pka-tools/bin/python \
        experiments/026-metabolism-flux/scripts/lipid_structure_writer.py pkas \
        --repo ~/.claude/jobs/7a6382f2/tmp/pka-agent/MolGpKa

    # 4. price (equilibrator-ext: assets master with pkas_provided_externally)
    ~/miniconda3/envs/equilibrator-ext/bin/python \
        experiments/026-metabolism-flux/scripts/lipid_structure_writer.py price

    # 5. coverage (torchcell: cobra + pandas)
    ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/lipid_structure_writer.py coverage

Every subcommand is run from the repository root so ``load_dotenv`` resolves
``DATA_ROOT``, ``EXPERIMENT_ROOT`` and ``ASSET_IMAGES_DIR``.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import cast

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

DATA_ROOT = cast(str, os.environ["DATA_ROOT"])
EXPERIMENT_ROOT = cast(str, os.environ["EXPERIMENT_ROOT"])
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
SCRIPTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "scripts")
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
THERMO_DIR = osp.join(DATA_ROOT, "data", "torchcell", "thermo_equilibrator")

AUDIT_PATH = osp.join(RESULTS, "unresolved_metabolites.json")
PARSED_PATH = osp.join(RESULTS, "lipid_names_parsed.json")
STRUCTURES_PATH = osp.join(RESULTS, "lipid_structures.json")
STRUCTURES_CSV = osp.join(RESULTS, "lipid_structures.csv")
PKAS_PATH = osp.join(RESULTS, "lipid_pkas.json")
PRICED_PATH = osp.join(RESULTS, "lipid_priced.json")
PRICED_CSV = osp.join(RESULTS, "lipid_priced.csv")
COVERAGE_PATH = osp.join(RESULTS, "lipid_coverage.json")

#: A writable copy of the compound cache. The shipped cache at
#: ``~/.cache/equilibrator/compounds.sqlite`` is NEVER opened for writing: a compound
#: created here would otherwise be indistinguishable from a MetaNetX one on the next read.
LOCAL_CCACHE = osp.join(DATA_ROOT, "tmp", "lipid_ccache.sqlite")

#: The conditions the flux layer consumes, matching external_pka_benchmark and the pilot.
P_H = 7.2
IONIC_STRENGTH_M = 0.25
TEMPERATURE_K = 303.15
P_MG = 3.0

#: The first double bond of every acyl chain, counting the carbonyl carbon as 1, and the
#: spacing of every further one. Not in the name; see design decision 1 in the docstring.
FIRST_DOUBLE_BOND_POSITION = 9
DOUBLE_BOND_SPACING = 3
DOUBLE_BOND_GEOMETRY = "cis"

#: Free acids the Delta-9 convention has to reproduce, as an assertion rather than a
#: comment. Palmitoleic acid is 16:1(9Z) and oleic acid is 18:1(9Z).
DELTA9_WITNESSES = {
    (16, 1): "SECPZKHBENQXJG-FPLPWBNLSA-N",
    (18, 1): "ZQPPMHVWECSIRJ-KTKRTIGZSA-N",
}

#: Coenzyme A in the cache. The acyl-CoA structures are built onto the cache's own CoA
#: rather than a typed one, so the 48 heavy atoms of the nucleotide are not ours to get
#: wrong. Neutral CoA-SH is C21H36N7O16P3S; the cache holds the -4 major microspecies.
COA_INCHI_KEY_PREFIX = "RGJOEKWQDUBAIZ"
COA_NEUTRAL_FORMULA = "C21H36N7O16P3S"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- templates


#: Headgroup SMILES, written as the substituent that hangs off the sn-3 oxygen of the
#: glycerol backbone. ``{}`` never appears: the chains are attached by the assembler.
HEAD_SMILES: dict[str, str] = {
    # phosphatidate and the amine headgroups
    "PA": "P(=O)(O)O",
    "PE": "P(=O)(O)OCCN",
    "PMME": "P(=O)(O)OCCNC",
    "PDME": "P(=O)(O)OCCN(C)C",
    # choline is a quaternary ammonium, so the neutral species is the zwitterion
    "PC": "P(=O)([O-])OCC[N+](C)(C)C",
    "PS": "P(=O)(O)OCC(N)C(=O)O",
    "PG": "P(=O)(O)OCC(O)CO",
    "PGP": "P(=O)(O)OCC(O)COP(=O)(O)O",
    "DGPP": "P(=O)(O)OP(=O)(O)O",
    # CDP: pyrophosphate to the 5' of cytidine
    "CDPDAG": "P(=O)(O)OP(=O)(O)OCC1OC(n2ccc(N)nc2=O)C(O)C1O",
}

#: Inositol ring phosphorylation, by the ring positions the name gives. The ring is
#: written from the C1 that carries the phosphodiester, so positions 2 to 6 follow in
#: order and a named 3- or 4- or 5-phosphate lands on the carbon the name says.
INOSITOL_PHOSPHATES: dict[str, tuple[int, ...]] = {
    "PI": (),
    "PI3P": (3,),
    "PI4P": (4,),
    "PI35P2": (3, 5),
    "PI45P2": (4, 5),
}

#: Every template, as (assembly kind, headgroup key). The assembly kinds are described on
#: :func:`assemble`.
TEMPLATES: dict[str, tuple[str, str]] = {
    **{k: ("gpl_diacyl", k) for k in HEAD_SMILES},
    **{k: ("gpl_diacyl", k) for k in INOSITOL_PHOSPHATES},
    "LPI2": ("gpl_sn2_lyso", "PI"),
    "TAG": ("triacylglycerol", ""),
    "DAG": ("diacylglycerol", ""),
    "MAG1": ("monoacylglycerol_sn1", ""),
    "CL": ("cardiolipin", ""),
    "FFA": ("free_fatty_acid", ""),
    "ACYLDHAP": ("acyl_dihydroxyacetone_phosphate", ""),
    "ACYLCOA": ("acyl_coa", ""),
}

#: The exact yeast-GEM name that precedes the acyl-composition parenthesis, mapped to the
#: template it selects. Matching is exact: an unrecognized headgroup is reported, never
#: guessed at by substring.
HEADGROUP_NAMES: dict[str, str] = {
    "phosphatidate": "PA",
    "phosphatidylethanolamine": "PE",
    "phosphatidyl-N-methylethanolamine": "PMME",
    "phosphatidyl-N,N-dimethylethanolamine": "PDME",
    "phosphatidylcholine": "PC",
    "phosphatidyl-L-serine": "PS",
    "phosphatidylglycerol": "PG",
    "3-(3-sn-phosphatidyl)-sn-glycerol 1-phosphate": "PGP",
    "1,2-diacylglycerol 3-diphosphate": "DGPP",
    "CDP-diacylglycerol": "CDPDAG",
    "1-phosphatidyl-1D-myo-inositol": "PI",
    "1-phosphatidyl-1D-myo-inositol 3-phosphate": "PI3P",
    "1-phosphatidyl-1D-myo-inositol 4-phosphate": "PI4P",
    "1-phosphatidyl-1D-myo-inositol 3,5-bisphosphate": "PI35P2",
    "1-phosphatidyl-1D-myo-inositol 4,5-bisphosphate": "PI45P2",
    "sn-2-acyl-1-lysophosphatidylinositol": "LPI2",
    "triglyceride": "TAG",
    "diglyceride": "DAG",
    "1-monoglyceride": "MAG1",
    "cardiolipin": "CL",
    "monolysocardiolipin": "CL",
    "acylglycerone phosphate": "ACYLDHAP",
}

#: How many acyl positions each template has, and which ones the name is allowed to name.
TEMPLATE_POSITIONS: dict[str, tuple[int, ...]] = {
    **{k: (1, 2) for k in HEAD_SMILES},
    **{k: (1, 2) for k in INOSITOL_PHOSPHATES},
    "LPI2": (2,),
    "TAG": (1, 2, 3),
    "DAG": (1, 2),
    "MAG1": (1,),
    "CL": (1, 2, 3, 4),
    "ACYLDHAP": (1,),
}

#: Systematic multiplier prefixes for the acyl-CoA names, longest first at match time.
ACYL_MULTIPLIERS: dict[str, int] = {
    "but": 4,
    "pent": 5,
    "hex": 6,
    "hept": 7,
    "oct": 8,
    "non": 9,
    "dec": 10,
    "undec": 11,
    "dodec": 12,
    "tridec": 13,
    "tetradec": 14,
    "pentadec": 15,
    "hexadec": 16,
    "heptadec": 17,
    "octadec": 18,
    "eicos": 20,
}


# ------------------------------------------------------------------------------ models


class AcylChain(BaseModel):
    """One acyl chain of a lipid, as the name gives it."""

    position: int | None = None
    n_carbons: int
    n_double_bonds: int
    #: Double-bond positions, counting the carbonyl carbon as 1. Filled from the name for
    #: the systematic acyl-CoA names and from the Delta-9 convention otherwise.
    double_bond_positions: list[int] = Field(default_factory=list)
    double_bond_geometry: list[str] = Field(default_factory=list)
    double_bond_source: str = "delta9_convention"
    #: Ring substituents the name states, as ``position -> hydroxy|oxo``.
    substituents: dict[int, str] = Field(default_factory=dict)


class ParsedName(BaseModel):
    """One distinct yeast-GEM name, and what the parser made of it."""

    name: str
    chemical_class: str
    formula: str
    charge: float
    met_ids: list[str]
    n_metabolite_rows: int
    parsed: bool
    template: str | None = None
    chains: list[AcylChain] = Field(default_factory=list)
    failure: str | None = None


class ParseReport(BaseModel):
    """Coverage of the name parser over the tier-b names."""

    built_at: str
    source_file: str
    n_metabolite_rows: int
    n_distinct_names: int
    n_parsed: int
    n_unparsed: int
    parsed_rows: int
    unparsed_rows: int
    parsed_by_class: dict[str, int]
    unparsed_by_class: dict[str, int]
    unparsed_by_reason: dict[str, int]
    unparsed_names: list[dict[str, str]]
    names: list[ParsedName]


class WrittenStructure(BaseModel):
    """One written structure and the outcome of the formula acceptance test."""

    name: str
    chemical_class: str
    template: str
    met_ids: list[str]
    n_metabolite_rows: int
    smiles: str | None = None
    inchi_key: str | None = None
    n_heavy_atoms: int | None = None
    gem_formula: str
    gem_charge: float
    #: The GEM formula moved to the neutral species: ``H - charge``, everything else fixed.
    neutral_target_formula: str
    rdkit_formula: str | None = None
    formula_match: bool = False
    failure: str | None = None


class StructureReport(BaseModel):
    """Everything the writer produced, plus the acceptance-test tally."""

    built_at: str
    conventions: dict[str, str]
    n_attempted: int
    n_written: int
    n_formula_pass: int
    n_formula_fail: int
    pass_by_class: dict[str, int]
    fail_by_class: dict[str, int]
    pass_by_template: dict[str, int]
    metabolite_rows_pass: int
    failures: list[dict[str, str]]
    structures: list[WrittenStructure]


class LipidPka(BaseModel):
    """The macroscopic ladder MolGpKa produced for one written structure."""

    name: str
    smiles: str
    ok: bool
    error: str | None = None
    pkas: list[float] = Field(default_factory=list)
    n_acidic_rungs: int = 0
    n_basic_rungs: int = 0
    #: True when a direction used every rung the walk allows, so the ladder stopped on the
    #: rung budget rather than on running out of ionizable sites. Such a list is a lower
    #: bound on the ladder, and its deepest values were predicted on charge states far
    #: outside the neutral regime MolGpKa was trained on.
    at_rung_limit: bool = False
    site_pkas: list[float] = Field(default_factory=list)
    seconds: float = 0.0


class PkaReport(BaseModel):
    """Every ladder the predictor produced, and the two ways a ladder can be unusable."""

    built_at: str
    tool: str
    tool_version: str
    aggregation: str
    n_requested: int
    n_ok: int
    n_empty_ladder: int = 0
    n_at_rung_limit: int = 0
    seconds_total: float
    records: list[LipidPka]


class PricedLipid(BaseModel):
    """One lipid pushed through the external-pKa route, and where it stopped."""

    name: str
    chemical_class: str
    template: str
    met_ids: list[str]
    n_metabolite_rows: int
    smiles: str
    gem_formula: str
    neutral_target_formula: str
    rdkit_formula: str
    formula_match: bool
    inchi_key: str
    pkas: list[float] = Field(default_factory=list)
    pka_ladder_at_rung_limit: bool = False
    #: Which route produced the value that is reported. ``cache_lookup`` means the written
    #: structure turned out to resolve an existing cache compound by InChIKey, so
    #: eQuilibrator's own ChemAxon pKas were used and nothing was created.
    route: str = ""
    cache_compound_id: int | None = None
    cache_compound_inchi_key: str | None = None
    #: ``stereo_and_skeleton`` when the first two InChIKey blocks matched,
    #: ``skeleton_only`` when the stereochemistry block had to be dropped as well. Either
    #: way every element but hydrogen must agree, so a match cannot be a different molecule.
    cache_match_level: str | None = None
    cache_compound_formula: str | None = None
    #: The cache stores the major microspecies, which need not carry the same number of
    #: protons as the neutral structure written here. When this is false the two routes
    #: are referenced to different microspecies and their difference mixes two effects.
    cache_same_protonation_state: bool | None = None
    dgf_prime_cache_kj_per_mol: float | None = None
    sigma_cache_kj_per_mol: float | None = None
    compound_id: int | None = None
    #: ``get_compounds`` resolves an InChIKey against the cache BEFORE it creates anything,
    #: so a structure the cache already holds comes back as that same compound with
    #: ChemAxon's pKas. When this is true the supplied pKas were never used and the
    #: external route measured nothing independent of the cache.
    external_route_reused_existing_compound: bool | None = None
    group_decomposition_ok: bool | None = None
    n_microspecies: int | None = None
    dgf_prime_external_kj_per_mol: float | None = None
    sigma_external_kj_per_mol: float | None = None
    #: ``cache`` minus ``external`` on the same written structure, filled only when the
    #: external route actually created a NEW compound from the supplied pKas.
    dgf_cache_minus_external_kj_per_mol: float | None = None
    dgf_prime_kj_per_mol: float | None = None
    sigma_kj_per_mol: float | None = None
    outcome: str = ""
    detail: str | None = None


class PriceReport(BaseModel):
    """The pricing tally, by route, plus the cost of the pKa substitution where measurable."""

    built_at: str
    cache_path: str
    conditions: dict[str, float]
    n_attempted: int
    n_written_structure_found_in_cache: int
    cache_match_levels: dict[str, int]
    n_created: int
    n_group_decomposed: int
    n_priced: int
    n_priced_by_cache_lookup: int
    n_priced_by_external_pka: int
    #: Structures whose external-route call returned the compound the cache already held
    #: instead of creating one. The supplied pKas were unused on these.
    n_external_route_reused_existing_compound: int
    #: Names where the external route genuinely created a compound AND the cache also had
    #: one, which is the only configuration in which the two numbers are independent.
    n_both_routes_independent: int
    dgf_route_difference_kj_per_mol: dict[str, float]
    #: The same difference restricted to pairs whose reference microspecies carries the
    #: same number of protons, which isolates the drawing convention from the protonation
    #: state. This is the number that says whether a created compound is on the same
    #: energy zero as the table the flux layer already reads.
    dgf_route_difference_same_protonation_kj_per_mol: dict[str, float]
    priced_by_class: dict[str, int]
    outcome_counts: dict[str, int]
    route_counts: dict[str, int]
    metabolite_rows_priced: int
    records: list[PricedLipid]


# ------------------------------------------------------------------------------- parse


ACYL_POSITIONAL = re.compile(r"^(\d+)-(\d+):(\d+)$")
ACYL_BARE = re.compile(r"^(\d+):(\d+)$")
NAME_WITH_SPEC = re.compile(r"^(?P<head>.+?) \((?P<spec>[^()]+)\)$")
CHAIN_NAME = re.compile(r"^C(\d+):(\d+) chain$")


def _double_bonds_by_convention(n_carbons: int, n_double: int) -> tuple[list[int], list[str]]:
    """Delta-9 and every third carbon after it, all cis. See design decision 1."""
    positions = [
        FIRST_DOUBLE_BOND_POSITION + DOUBLE_BOND_SPACING * i for i in range(n_double)
    ]
    if positions and positions[-1] + 1 > n_carbons:
        raise ValueError(
            f"the Delta-{FIRST_DOUBLE_BOND_POSITION} convention needs at least "
            f"{positions[-1] + 1} carbons for {n_double} double bonds, chain has {n_carbons}"
        )
    return positions, [DOUBLE_BOND_GEOMETRY] * len(positions)


def _chain(position: int | None, n_carbons: int, n_double: int) -> AcylChain:
    positions, geometry = _double_bonds_by_convention(n_carbons, n_double)
    return AcylChain(
        position=position,
        n_carbons=n_carbons,
        n_double_bonds=n_double,
        double_bond_positions=positions,
        double_bond_geometry=geometry,
        double_bond_source="delta9_convention",
    )


def parse_acyl_coa(name: str) -> AcylChain:
    """One systematic acyl-CoA name to its chain.

    These names carry their double-bond positions and geometry explicitly, so the
    Delta-9 convention is not used for them: ``trans-2,cis-5-dodecadienoyl-CoA`` is a
    12-carbon chain with a trans bond at 2 and a cis bond at 5. Raises on anything the
    grammar does not cover, rather than guessing.
    """
    stem = name[: -len("-CoA")]
    match = re.match(r"^\((R|S)\)-", stem)
    if match:
        # The stereo descriptor is read and discarded: stereocenters are left
        # unspecified (design decision 2).
        stem = stem[match.end() :]
    substituents: dict[int, str] = {}
    while True:
        match = re.match(r"^(\d+)-(hydroxy|oxo)-", stem)
        if not match:
            break
        substituents[int(match.group(1))] = match.group(2)
        stem = stem[match.end() :]

    double_positions: list[int] = []
    geometry: list[str] = []
    prefixed = re.match(r"^((?:(?:cis|trans)-\d+,)*(?:cis|trans)-\d+)-(.*)$", stem)
    if prefixed:
        # trans-2,cis-5-dodecadienoyl
        for geo, pos in re.findall(r"(cis|trans)-(\d+)", prefixed.group(1)):
            double_positions.append(int(pos))
            geometry.append(geo)
        stem = prefixed.group(2)
    else:
        leading = re.match(r"^(cis|trans)-(.*)$", stem)
        geo = DOUBLE_BOND_GEOMETRY
        if leading:
            geo = leading.group(1)
            stem = leading.group(2)
        positioned = re.match(r"^([a-z]+)-(\d+)-enoyl$", stem)
        if positioned:
            # cis-dodec-5-enoyl
            double_positions.append(int(positioned.group(2)))
            geometry.append(geo)
            stem = positioned.group(1)
        elif not stem.endswith("anoyl"):
            raise ValueError(f"unrecognized acyl-CoA suffix in {name!r}")

    stem = re.sub(r"(a?)(di|tri)?(en|an)oyl$", "", stem)
    n_carbons = None
    for prefix in sorted(ACYL_MULTIPLIERS, key=len, reverse=True):
        if stem.startswith(prefix) and stem[len(prefix) :] in ("", "a"):
            n_carbons = ACYL_MULTIPLIERS[prefix]
            break
    if n_carbons is None:
        raise ValueError(f"unrecognized carbon-count stem {stem!r} in {name!r}")
    order = sorted(range(len(double_positions)), key=lambda i: double_positions[i])
    return AcylChain(
        position=None,
        n_carbons=n_carbons,
        n_double_bonds=len(double_positions),
        double_bond_positions=[double_positions[i] for i in order],
        double_bond_geometry=[geometry[i] for i in order],
        double_bond_source="stated_in_name",
        substituents=substituents,
    )


def parse_name(name: str) -> tuple[str | None, list[AcylChain], str | None]:
    """One yeast-GEM name to (template, chains, failure reason)."""
    if name.endswith(" backbone"):
        return (
            None,
            [],
            "slime_backbone_pseudo_metabolite",
        )
    chain_only = CHAIN_NAME.match(name)
    if chain_only:
        return (
            "FFA",
            [_chain(None, int(chain_only.group(1)), int(chain_only.group(2)))],
            None,
        )
    if name.endswith("-CoA"):
        try:
            return "ACYLCOA", [parse_acyl_coa(name)], None
        except ValueError as exc:
            return None, [], f"acyl_coa_grammar: {exc}"

    spec_match = NAME_WITH_SPEC.match(name)
    if not spec_match:
        return None, [], "no_acyl_composition_in_name"
    head = spec_match.group("head")
    template = HEADGROUP_NAMES.get(head)
    if template is None:
        return None, [], "unknown_headgroup"

    chains: list[AcylChain] = []
    for token in spec_match.group("spec").split(", "):
        positional = ACYL_POSITIONAL.match(token)
        bare = ACYL_BARE.match(token)
        try:
            if positional:
                chains.append(
                    _chain(
                        int(positional.group(1)),
                        int(positional.group(2)),
                        int(positional.group(3)),
                    )
                )
            elif bare:
                chains.append(_chain(None, int(bare.group(1)), int(bare.group(2))))
            else:
                return None, [], f"unparseable_acyl_token: {token}"
        except ValueError as exc:
            return None, [], f"chain_outside_convention: {exc}"
    if not chains:
        return None, [], "no_acyl_composition_in_name"

    allowed = TEMPLATE_POSITIONS[template]
    for chain in chains:
        if chain.position is not None and chain.position not in allowed:
            return None, [], f"acyl_position_outside_template: {chain.position}"
    if len(chains) == 1 and chains[0].position is None:
        chains[0].position = allowed[0]
    if any(chain.position is None for chain in chains):
        return None, [], "mixed_positional_and_bare_acyl_tokens"
    if len(chains) != len({chain.position for chain in chains}):
        return None, [], "repeated_acyl_position"
    return template, chains, None


def _tier_b_rows() -> list[dict]:
    """The tier-b metabolite rows, selected exactly as ``external_pka_pilot reach`` does."""
    with open(AUDIT_PATH) as handle:
        audit = json.load(handle)
    return [
        row
        for row in audit["metabolites_unpriced_at_baseline"]
        if not row.get("recovered_priceable")
        and row.get("final_blocker") == "no_usable_structure_available"
    ]


def cmd_parse(args: argparse.Namespace) -> None:
    rows = _tier_b_rows()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["name"]].append(row)

    names: list[ParsedName] = []
    for name in sorted(grouped):
        members = grouped[name]
        template, chains, failure = parse_name(name)
        names.append(
            ParsedName(
                name=name,
                chemical_class=members[0]["chemical_class"],
                formula=members[0]["formula"],
                charge=float(members[0]["charge"]),
                met_ids=[m["met_id"] for m in members],
                n_metabolite_rows=len(members),
                parsed=template is not None,
                template=template,
                chains=chains,
                failure=failure,
            )
        )

    parsed = [n for n in names if n.parsed]
    unparsed = [n for n in names if not n.parsed]
    report = ParseReport(
        built_at=_timestamp(),
        source_file=AUDIT_PATH,
        n_metabolite_rows=len(rows),
        n_distinct_names=len(names),
        n_parsed=len(parsed),
        n_unparsed=len(unparsed),
        parsed_rows=sum(n.n_metabolite_rows for n in parsed),
        unparsed_rows=sum(n.n_metabolite_rows for n in unparsed),
        parsed_by_class=dict(Counter(n.chemical_class for n in parsed).most_common()),
        unparsed_by_class=dict(Counter(n.chemical_class for n in unparsed).most_common()),
        unparsed_by_reason=dict(
            Counter(cast(str, n.failure) for n in unparsed).most_common()
        ),
        unparsed_names=[
            {
                "name": n.name,
                "chemical_class": n.chemical_class,
                "formula": n.formula,
                "reason": cast(str, n.failure),
            }
            for n in unparsed
        ],
        names=names,
    )
    with open(PARSED_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {PARSED_PATH}")
    print(
        f"{report.n_parsed}/{report.n_distinct_names} distinct names parse "
        f"({report.parsed_rows}/{report.n_metabolite_rows} metabolite rows)"
    )
    for reason, count in report.unparsed_by_reason.items():
        print(f"  unparsed {count:3d}  {reason}")
    for row in report.unparsed_names:
        print(f"    [{row['chemical_class']}] {row['name']}  ({row['formula']})")


# ------------------------------------------------------------------------------- build


def acyl_tail(chain: AcylChain) -> str:
    """Carbons 2..n of an acyl chain as SMILES, carbonyl carbon excluded.

    Position ``k`` in a double-bond list means the bond between carbon ``k`` and carbon
    ``k+1``, which is the standard Delta numbering, and it is written with the two
    directional single bonds that carry the cis or trans geometry.
    """
    doubles = dict(zip(chain.double_bond_positions, chain.double_bond_geometry))
    smiles = ""
    index = 2
    while index <= chain.n_carbons:
        if index in doubles:
            smiles += "/C=C\\" if doubles[index] == "cis" else "/C=C/"
            index += 2
            continue
        substituent = chain.substituents.get(index)
        if substituent == "hydroxy":
            smiles += "C(O)"
        elif substituent == "oxo":
            smiles += "C(=O)"
        elif substituent is not None:
            raise ValueError(f"unsupported substituent {substituent!r}")
        else:
            smiles += "C"
        index += 1
    if 1 in doubles or chain.n_carbons in doubles:
        raise ValueError("a double bond cannot start at the carbonyl or the last carbon")
    return smiles


def ester(chain: AcylChain) -> str:
    """The ``-O-C(=O)R`` ester that hangs an acyl chain off a glycerol carbon."""
    return "OC(=O)" + acyl_tail(chain)


def inositol_head(template: str) -> str:
    """The phosphoinositol headgroup, phosphorylated where the name says."""
    phosphates = INOSITOL_PHOSPHATES[template]
    substituents = [
        "OP(=O)(O)O" if position in phosphates else "O" for position in range(2, 7)
    ]
    ring = "C1" + "".join(f"C({s})" for s in substituents[:4]) + f"C1{substituents[4]}"
    return "P(=O)(O)O" + ring


def head_smiles(template: str) -> str:
    if template in INOSITOL_PHOSPHATES:
        return inositol_head(template)
    return HEAD_SMILES[template]


def _phosphatidyl_unit(chains: dict[int, AcylChain], sn1: int, sn2: int) -> str:
    """One phosphatidyl arm of a cardiolipin, attached through its sn-3 phosphate."""
    first = chains.get(sn1)
    second = chains.get(sn2)
    return (
        "P(=O)(O)OCC("
        + (ester(second) if second is not None else "O")
        + ")C"
        + (ester(first) if first is not None else "O")
    )


def assemble(template: str, chains: list[AcylChain], coa_neutral_smiles: str | None) -> str:
    """The SMILES for one template and its chains.

    Assembly kinds:

    ``gpl_diacyl``
        glycerol with acyl at sn-1 and sn-2 and the headgroup at sn-3
    ``gpl_sn2_lyso``
        the same with sn-1 a free hydroxyl
    ``triacylglycerol`` / ``diacylglycerol`` / ``monoacylglycerol_sn1``
        glycerol with three, two or one acyl and no headgroup
    ``cardiolipin``
        a central glycerol whose sn-1 and sn-3 each carry a phosphatidyl arm, positions
        1 and 2 on the first arm and 3 and 4 on the second; a position the name omits is
        a free hydroxyl, which is what makes a monolysocardiolipin
    ``free_fatty_acid``
        the acid itself
    ``acyl_dihydroxyacetone_phosphate``
        glycerone phosphate with the acyl on its free hydroxymethyl
    ``acyl_coa``
        the cache's coenzyme A, neutralized, thioesterified with the acyl chain
    """
    kind, head_key = TEMPLATES[template]
    by_position = {cast(int, c.position): c for c in chains if c.position is not None}

    if kind == "gpl_diacyl":
        head = head_smiles(template if template in INOSITOL_PHOSPHATES else head_key)
        return (
            f"C(C{ester(by_position[1])})(CO{head}){ester(by_position[2])}"
        )
    if kind == "gpl_sn2_lyso":
        head = head_smiles(head_key)
        return f"C(CO)(CO{head}){ester(by_position[2])}"
    if kind == "triacylglycerol":
        return (
            f"C(C{ester(by_position[1])})(C{ester(by_position[3])}){ester(by_position[2])}"
        )
    if kind == "diacylglycerol":
        return f"C(C{ester(by_position[1])})(CO){ester(by_position[2])}"
    if kind == "monoacylglycerol_sn1":
        return f"OCC(O)C{ester(by_position[1])}"
    if kind == "cardiolipin":
        return (
            "C(O)(CO"
            + _phosphatidyl_unit(by_position, 1, 2)
            + ")CO"
            + _phosphatidyl_unit(by_position, 3, 4)
        )
    if kind == "free_fatty_acid":
        return "OC(=O)" + acyl_tail(chains[0])
    if kind == "acyl_dihydroxyacetone_phosphate":
        return f"OP(=O)(O)OCC(=O)C{ester(by_position[1])}"
    if kind == "acyl_coa":
        if coa_neutral_smiles is None:
            raise ValueError("acyl-CoA assembly needs the cache's CoA structure")
        return _thioesterify(coa_neutral_smiles, chains[0])
    raise ValueError(f"unknown assembly kind {kind!r}")


def _thioesterify(coa_neutral_smiles: str, chain: AcylChain) -> str:
    """Condense the acid onto CoA's thiol, then drop every atom stereocenter."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    reaction = AllChem.ReactionFromSmarts(
        "[SX2H1:1].[CX3:2](=[OX1:3])[OX2H1]>>[S:1][C:2]=[O:3]"
    )
    coa = Chem.MolFromSmiles(coa_neutral_smiles)
    acid = Chem.MolFromSmiles("OC(=O)" + acyl_tail(chain))
    products = reaction.RunReactants((coa, acid))
    if not products:
        raise ValueError("thioesterification of CoA produced no product")
    product = products[0][0]
    Chem.SanitizeMol(product)
    for atom in product.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    Chem.AssignStereochemistry(product, cleanIt=True, force=True)
    return Chem.MolToSmiles(product)


def neutral_target_formula(formula: str, charge: float) -> str:
    """The GEM formula moved to the neutral species: ``H -> H - charge``.

    yeast-GEM writes a formula at the model's own charge state, and every structure here
    is written neutral, so the acceptance test needs one common protonation state. A
    species at charge ``z`` carries ``z`` protons more than its neutral parent, so the
    neutral hydrogen count is ``H - z``.
    """
    if charge != int(charge):
        raise ValueError(f"non-integer charge {charge}")
    counts = _formula_counts(formula)
    counts["H"] = counts.get("H", 0) - int(charge)
    if counts["H"] < 0:
        raise ValueError(f"charge {charge} leaves a negative hydrogen count in {formula}")
    return _formula_string(counts)


def _formula_counts(formula: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if not element:
            continue
        counts[element] = counts.get(element, 0) + (int(number) if number else 1)
    return counts


def _formula_string(counts: dict[str, int]) -> str:
    """Hill order, which is the order RDKit's ``CalcMolFormula`` writes."""
    order = sorted(k for k in counts if k not in ("C", "H"))
    keys = (["C"] if counts.get("C") else []) + (["H"] if counts.get("H") else []) + order
    parts = []
    for key in keys:
        n = counts[key]
        if n == 0:
            continue
        parts.append(key if n == 1 else f"{key}{n}")
    return "".join(parts)


def _coa_neutral_smiles() -> str:
    """Neutral coenzyme A, taken from the compound cache rather than typed here."""
    from equilibrator_api import ComponentContribution
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    cc = ComponentContribution()
    hits = cc.search_compound_by_inchi_key(COA_INCHI_KEY_PREFIX)
    if not hits:
        raise RuntimeError("coenzyme A is not in the compound cache")
    neutral = rdMolStandardize.Uncharger().uncharge(Chem.MolFromSmiles(hits[0].smiles))
    formula = CalcMolFormula(neutral)
    if formula != COA_NEUTRAL_FORMULA:
        raise RuntimeError(
            f"neutralized cache CoA is {formula}, expected {COA_NEUTRAL_FORMULA}"
        )
    return Chem.MolToSmiles(neutral)


def _assert_delta9_witnesses() -> None:
    """The Delta-9 convention must reproduce palmitoleic and oleic acid exactly."""
    from rdkit import Chem

    for (n_carbons, n_double), expected in DELTA9_WITNESSES.items():
        chain = _chain(None, n_carbons, n_double)
        key = Chem.MolToInchiKey(Chem.MolFromSmiles("OC(=O)" + acyl_tail(chain)))
        if key != expected:
            raise RuntimeError(
                f"the Delta-{FIRST_DOUBLE_BOND_POSITION} convention wrote {n_carbons}:"
                f"{n_double} as {key}, expected {expected}"
            )


def cmd_build(args: argparse.Namespace) -> None:
    from rdkit import Chem, RDLogger
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    RDLogger.DisableLog("rdApp.*")
    _assert_delta9_witnesses()
    print("Delta-9 convention reproduces palmitoleic and oleic acid")

    with open(PARSED_PATH) as handle:
        parsed = ParseReport.model_validate_json(handle.read())
    targets = [n for n in parsed.names if n.parsed]
    needs_coa = any(n.template == "ACYLCOA" for n in targets)
    coa = _coa_neutral_smiles() if needs_coa else None
    if coa is not None:
        print(f"coenzyme A from the cache: {coa}")

    structures: list[WrittenStructure] = []
    for target in targets:
        row = WrittenStructure(
            name=target.name,
            chemical_class=target.chemical_class,
            template=cast(str, target.template),
            met_ids=target.met_ids,
            n_metabolite_rows=target.n_metabolite_rows,
            gem_formula=target.formula,
            gem_charge=target.charge,
            neutral_target_formula=neutral_target_formula(target.formula, target.charge),
        )
        try:
            smiles = assemble(cast(str, target.template), target.chains, coa)
        except (ValueError, KeyError) as exc:
            row.failure = f"assembly_failed: {type(exc).__name__}: {exc}"[:200]
            structures.append(row)
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            row.smiles = smiles
            row.failure = "rdkit_could_not_parse_written_smiles"
            structures.append(row)
            continue
        net_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
        if net_charge != 0:
            row.smiles = smiles
            row.failure = f"written_structure_is_not_net_neutral: {net_charge}"
            structures.append(row)
            continue
        row.smiles = Chem.MolToSmiles(molecule)
        row.inchi_key = Chem.MolToInchiKey(molecule)
        row.n_heavy_atoms = molecule.GetNumAtoms()
        row.rdkit_formula = CalcMolFormula(molecule)
        row.formula_match = row.rdkit_formula == row.neutral_target_formula
        if not row.formula_match:
            row.failure = "formula_mismatch"
        structures.append(row)

    passed = [s for s in structures if s.formula_match]
    failed = [s for s in structures if not s.formula_match]
    report = StructureReport(
        built_at=_timestamp(),
        conventions={
            "double_bond_position": (
                f"Delta-{FIRST_DOUBLE_BOND_POSITION} for the first double bond and every "
                f"{DOUBLE_BOND_SPACING} carbons after it, {DOUBLE_BOND_GEOMETRY}, except "
                "for the acyl-CoA names which state their own positions and geometry"
            ),
            "stereocenters": "left unspecified, including in the CoA moiety",
            "protonation": (
                "every structure is written neutral; the acceptance test compares against "
                "the GEM formula moved to the neutral species by H -> H - charge"
            ),
            "coenzyme_a": (
                f"taken from the compound cache by InChIKey prefix {COA_INCHI_KEY_PREFIX} "
                f"and neutralized to {COA_NEUTRAL_FORMULA}, not typed here"
            ),
        },
        n_attempted=len(structures),
        n_written=sum(1 for s in structures if s.smiles is not None),
        n_formula_pass=len(passed),
        n_formula_fail=len(failed),
        pass_by_class=dict(Counter(s.chemical_class for s in passed).most_common()),
        fail_by_class=dict(Counter(s.chemical_class for s in failed).most_common()),
        pass_by_template=dict(Counter(s.template for s in passed).most_common()),
        metabolite_rows_pass=sum(s.n_metabolite_rows for s in passed),
        failures=[
            {
                "name": s.name,
                "chemical_class": s.chemical_class,
                "template": s.template,
                "gem_formula": s.gem_formula,
                "neutral_target_formula": s.neutral_target_formula,
                "rdkit_formula": s.rdkit_formula or "",
                "failure": s.failure or "",
            }
            for s in failed
        ],
        structures=structures,
    )
    with open(STRUCTURES_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    _write_csv(
        STRUCTURES_CSV,
        [
            {
                "name": s.name,
                "chemical_class": s.chemical_class,
                "template": s.template,
                "n_metabolite_rows": s.n_metabolite_rows,
                "gem_formula": s.gem_formula,
                "gem_charge": s.gem_charge,
                "neutral_target_formula": s.neutral_target_formula,
                "rdkit_formula": s.rdkit_formula or "",
                "formula_match": s.formula_match,
                "n_heavy_atoms": s.n_heavy_atoms or "",
                "inchi_key": s.inchi_key or "",
                "smiles": s.smiles or "",
                "failure": s.failure or "",
            }
            for s in structures
        ],
    )
    print(f"wrote {STRUCTURES_PATH}")
    print(f"wrote {STRUCTURES_CSV}")
    print(
        f"formula acceptance test: {report.n_formula_pass}/{report.n_attempted} names pass "
        f"({report.metabolite_rows_pass} metabolite rows)"
    )
    for template, count in report.pass_by_template.items():
        print(f"  pass {count:3d}  {template}")
    for failure in report.failures:
        print(
            f"  FAIL [{failure['chemical_class']}] {failure['name']}: "
            f"wrote {failure['rdkit_formula']} vs GEM neutral "
            f"{failure['neutral_target_formula']} ({failure['failure']})"
        )


def _write_csv(path: str, rows: list[dict]) -> None:
    import csv

    if not rows:
        raise ValueError(f"refusing to write an empty csv to {path}")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# -------------------------------------------------------------------------------- pkas


def cmd_pkas(args: argparse.Namespace) -> None:
    """MolGpKa site pKas aggregated into the ladder, with the benchmark's own code.

    ``_macro_from_sites`` and ``_apply_proton_change`` are imported from
    :mod:`external_pka_benchmark` so the aggregation here is the same function the
    300-compound benchmark scored, not a second implementation of it.
    """
    import time

    sys.path.insert(0, SCRIPTS)
    from external_pka_benchmark import (  # type: ignore[import-not-found]
        MAX_PH,
        MAX_PKAS,
        MIN_PH,
        _apply_proton_change,
        _macro_from_sites,
    )

    sys.path.insert(0, osp.join(args.repo, "src"))
    os.chdir(osp.join(args.repo, "src"))

    import torch

    torch.set_num_threads(args.threads)
    _orig_load = torch.load

    def _load(*a, **kw):
        kw["weights_only"] = False
        return _orig_load(*a, **kw)

    torch.load = _load

    import predict_pka  # type: ignore[import-not-found]

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

    with open(STRUCTURES_PATH) as handle:
        structures = StructureReport.model_validate_json(handle.read())
    targets = [s for s in structures.structures if s.formula_match]

    records: list[LipidPka] = []
    t_all = time.time()
    for index, target in enumerate(targets):
        t0 = time.time()
        ladder: list[float] = []
        sites_seen: list[float] = []
        per_direction = {True: 0, False: 0}
        error: str | None = None
        try:
            for acidic in (True, False):
                state = cast(str, target.smiles)
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
                    per_direction[acidic] += 1
                    dominant_idx = (
                        min(found, key=lambda k: float(found[k]))
                        if acidic
                        else max(found, key=lambda k: float(found[k]))
                    )
                    if acidic:
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
        except Exception as exc:  # the tool's own failures are the measurement
            error = f"{type(exc).__name__}: {exc}"[:300]
        records.append(
            LipidPka(
                name=target.name,
                smiles=cast(str, target.smiles),
                ok=error is None,
                error=error,
                pkas=sorted(ladder, reverse=True),
                n_acidic_rungs=per_direction[True],
                n_basic_rungs=per_direction[False],
                at_rung_limit=max(per_direction.values()) > MAX_PKAS,
                site_pkas=sites_seen,
                seconds=time.time() - t0,
            )
        )
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(targets)}", flush=True)

    report = PkaReport(
        built_at=_timestamp(),
        tool="molgpka",
        tool_version=args.tool_version,
        aggregation=(
            "site pKas aggregated per charge level by log-sum-exp and walked through the "
            "dominant site, imported from external_pka_benchmark"
        ),
        n_requested=len(targets),
        n_ok=sum(1 for r in records if r.ok),
        n_empty_ladder=sum(1 for r in records if r.ok and not r.pkas),
        n_at_rung_limit=sum(1 for r in records if r.at_rung_limit),
        seconds_total=time.time() - t_all,
        records=records,
    )
    with open(PKAS_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {PKAS_PATH}: ok {report.n_ok}/{report.n_requested} in "
          f"{report.seconds_total:.0f}s")
    print(
        f"  {report.n_empty_ladder} structures have no pKa in (0, 14); "
        f"{report.n_at_rung_limit} ladders stopped on the rung budget"
    )


# ------------------------------------------------------------------------------- price


def cmd_price(args: argparse.Namespace) -> None:
    import numpy as np
    from equilibrator_api import ComponentContribution, Q_
    from equilibrator_assets.local_compound_cache import LocalCompoundCache

    if osp.realpath(args.cache) == osp.realpath(
        osp.expanduser("~/.cache/equilibrator/compounds.sqlite")
    ):
        raise RuntimeError("refusing to write compounds into the shipped compound cache")
    if not osp.exists(args.cache):
        raise RuntimeError(
            f"{args.cache} does not exist; copy the shipped cache there first"
        )

    with open(STRUCTURES_PATH) as handle:
        structures = StructureReport.model_validate_json(handle.read())
    by_name = {s.name: s for s in structures.structures if s.formula_match}
    with open(PKAS_PATH) as handle:
        pkas = PkaReport.model_validate_json(handle.read())

    cc = ComponentContribution()
    cc.p_h = Q_(P_H)
    cc.ionic_strength = Q_(f"{IONIC_STRENGTH_M}M")
    cc.temperature = Q_(f"{TEMPERATURE_K}K")
    cc.p_mg = Q_(P_MG)

    local = LocalCompoundCache(pkas_provided_externally=True)
    local.load_cache(args.cache)

    def _price(compound) -> tuple[float | None, float | None, str]:
        mu, sigma_fin, _ = cc.standard_dg_formation(compound)
        if mu is None:
            return None, None, "component contribution returned no formation energy"
        correction = compound.transform(
            Q_(P_H), Q_(f"{IONIC_STRENGTH_M}M"), Q_(f"{TEMPERATURE_K}K"), Q_(P_MG)
        )
        return (
            float(mu) + float(correction.m_as("kJ/mol")),
            float(np.linalg.norm(sigma_fin)),
            "ok",
        )

    records: list[PricedLipid] = []
    for index, prediction in enumerate(pkas.records):
        structure = by_name[prediction.name]
        smiles = cast(str, structure.smiles)
        inchi_key = cast(str, structure.inchi_key)
        row = PricedLipid(
            name=structure.name,
            chemical_class=structure.chemical_class,
            template=structure.template,
            met_ids=structure.met_ids,
            n_metabolite_rows=structure.n_metabolite_rows,
            smiles=smiles,
            gem_formula=structure.gem_formula,
            neutral_target_formula=structure.neutral_target_formula,
            rdkit_formula=cast(str, structure.rdkit_formula),
            formula_match=True,
            inchi_key=inchi_key,
            pkas=prediction.pkas,
            pka_ladder_at_rung_limit=prediction.at_rung_limit,
        )

        # Route 1: the written structure may resolve a compound the cache already holds.
        # The audit could not try this, since it had no structure to key on.
        #
        # Two passes, and the second is a deliberate relaxation. The first drops only the
        # LAST InChIKey block, because the cache stores the MAJOR MICROSPECIES and that
        # block records the protonation state; this is the same fix pathway_thermo.py
        # needed. The second drops the stereochemistry block as well, which is required
        # here because the structures are written with unspecified stereocenters
        # (design decision 2) while the cache's entries often carry them: our
        # phosphatidylcholine (1-16:0, 2-18:1) hashes to WTJKGGKOPKCXLL-MRCUWXFGSA-N and
        # POPC in the cache to WTJKGGKOPKCXLL-VQVTYTSYSA-N, the same constitution written
        # with and without the sn-2 center. A skeleton match is exact for this purpose:
        # the group decomposition that produces the energy does not read stereochemistry,
        # so stereoisomers carry the same group vector. The two passes are reported
        # separately so a reader can discount the relaxed one.
        written_counts = _formula_counts(cast(str, structure.rdkit_formula))

        def _heavy_atoms_match(hit) -> bool:
            """Every element but hydrogen must agree, which protonation cannot change."""
            bag = hit.atom_bag or {}
            elements = (set(bag) | set(written_counts)) - {"H", "e-"}
            return all(bag.get(e, 0) == written_counts.get(e, 0) for e in elements)

        skeleton = inchi_key.split("-", 1)[0]
        hits = [
            hit
            for hit in cc.search_compound_by_inchi_key(inchi_key.rsplit("-", 1)[0])
            if hit.group_vector is not None and _heavy_atoms_match(hit)
        ]
        match_level = "stereo_and_skeleton" if hits else ""
        if not hits:
            hits = [
                hit
                for hit in cc.search_compound_by_inchi_key(skeleton)
                if hit.group_vector is not None and _heavy_atoms_match(hit)
            ]
            match_level = "skeleton_only" if hits else ""
        if hits:
            hit = hits[0]
            row.cache_compound_id = hit.id
            row.cache_compound_inchi_key = hit.inchi_key
            row.cache_match_level = match_level
            row.cache_compound_formula = hit.formula
            row.cache_same_protonation_state = (hit.atom_bag or {}).get(
                "H"
            ) == written_counts.get("H")
            dgf, sigma, detail = _price(hit)
            row.dgf_prime_cache_kj_per_mol = dgf
            row.sigma_cache_kj_per_mol = sigma
            if dgf is None:
                row.detail = detail

        # Route 2: create the compound from the written structure and the predicted pKas.
        if not prediction.ok:
            row.detail = prediction.error
        elif not prediction.pkas:
            # The interface guard is truthiness, so an empty list raises MissingPKasError
            # instead of creating a single-microspecies compound. Recorded, not worked
            # around.
            row.detail = "no pKa in (0, 14); specified_pkas cannot express an empty list"
        else:
            try:
                results = local.get_compounds(
                    [smiles],
                    mol_format="smiles",
                    specified_pkas={smiles: prediction.pkas},
                    bypass_chemaxon=True,
                )
            except Exception as exc:
                results = []
                row.detail = f"{type(exc).__name__}: {exc}"[:300]
            compound = results[0].compound if results else None
            if compound is not None:
                row.compound_id = compound.id
                row.external_route_reused_existing_compound = (
                    compound.id == row.cache_compound_id
                )
                row.group_decomposition_ok = compound.group_vector is not None
                row.n_microspecies = len(compound.microspecies)
                if compound.group_vector is not None:
                    dgf, sigma, detail = _price(compound)
                    row.dgf_prime_external_kj_per_mol = dgf
                    row.sigma_external_kj_per_mol = sigma
                    if dgf is None:
                        row.detail = detail
            elif results:
                row.group_decomposition_ok = False
                row.detail = f"method={results[0].method} status={results[0].status}"[:300]

        if (
            row.dgf_prime_cache_kj_per_mol is not None
            and row.dgf_prime_external_kj_per_mol is not None
            and row.external_route_reused_existing_compound is False
        ):
            row.dgf_cache_minus_external_kj_per_mol = (
                row.dgf_prime_cache_kj_per_mol - row.dgf_prime_external_kj_per_mol
            )
        # The cache value wins where it exists: it carries ChemAxon's own pKas and, unlike
        # a created compound, its magnesium dissociation constants.
        if row.dgf_prime_cache_kj_per_mol is not None:
            row.route = "cache_lookup"
            row.dgf_prime_kj_per_mol = row.dgf_prime_cache_kj_per_mol
            row.sigma_kj_per_mol = row.sigma_cache_kj_per_mol
            row.outcome = "priced"
        elif row.dgf_prime_external_kj_per_mol is not None:
            row.route = "external_pka"
            row.dgf_prime_kj_per_mol = row.dgf_prime_external_kj_per_mol
            row.sigma_kj_per_mol = row.sigma_external_kj_per_mol
            row.outcome = "priced"
        else:
            row.route = "blocked"
            if not prediction.ok:
                row.outcome = "pka_prediction_failed"
            elif not prediction.pkas:
                row.outcome = "no_pka_in_window_interface_cannot_express"
            elif row.group_decomposition_ok is False:
                row.outcome = "failed_group_decomposition"
            elif row.compound_id is None:
                row.outcome = "compound_creation_raised"
            else:
                row.outcome = "failed_other"
        records.append(row)
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(pkas.records)}", flush=True)

    priced = [r for r in records if r.outcome == "priced"]
    compared = [r for r in records if r.dgf_cache_minus_external_kj_per_mol is not None]
    differences = [
        abs(cast(float, r.dgf_cache_minus_external_kj_per_mol)) for r in compared
    ]
    same_protonation = [
        cast(float, r.dgf_cache_minus_external_kj_per_mol)
        for r in compared
        if r.cache_same_protonation_state
    ]

    def _stats(values: list[float], signed: bool = False) -> dict[str, float]:
        if not values:
            return {"n": 0.0}
        ordered = sorted(values)
        return {
            "n": float(len(values)),
            "mean": float(sum(values) / len(values)),
            "median": float(ordered[len(ordered) // 2]),
            "min": float(ordered[0]),
            "max": float(ordered[-1]),
        }
    report = PriceReport(
        built_at=_timestamp(),
        cache_path=args.cache,
        conditions={
            "p_h": P_H,
            "ionic_strength_m": IONIC_STRENGTH_M,
            "temperature_k": TEMPERATURE_K,
            "p_mg": P_MG,
        },
        n_attempted=len(records),
        n_written_structure_found_in_cache=sum(
            1 for r in records if r.cache_compound_id is not None
        ),
        cache_match_levels=dict(
            Counter(
                r.cache_match_level for r in records if r.cache_match_level
            ).most_common()
        ),
        n_created=sum(1 for r in records if r.compound_id is not None),
        n_group_decomposed=sum(1 for r in records if r.group_decomposition_ok),
        n_priced=len(priced),
        n_priced_by_cache_lookup=sum(1 for r in priced if r.route == "cache_lookup"),
        n_priced_by_external_pka=sum(1 for r in priced if r.route == "external_pka"),
        n_external_route_reused_existing_compound=sum(
            1 for r in records if r.external_route_reused_existing_compound
        ),
        n_both_routes_independent=len(differences),
        dgf_route_difference_kj_per_mol=_stats(differences),
        dgf_route_difference_same_protonation_kj_per_mol=_stats(same_protonation),
        priced_by_class=dict(Counter(r.chemical_class for r in priced).most_common()),
        outcome_counts=dict(Counter(r.outcome for r in records).most_common()),
        route_counts=dict(Counter(r.route for r in records).most_common()),
        metabolite_rows_priced=sum(r.n_metabolite_rows for r in priced),
        records=records,
    )
    with open(PRICED_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    _write_csv(
        PRICED_CSV,
        [
            {
                "name": r.name,
                "chemical_class": r.chemical_class,
                "template": r.template,
                "n_metabolite_rows": r.n_metabolite_rows,
                "gem_formula": r.gem_formula,
                "rdkit_formula": r.rdkit_formula,
                "formula_match": r.formula_match,
                "n_pkas": len(r.pkas),
                "pka_ladder_at_rung_limit": r.pka_ladder_at_rung_limit,
                "pkas": ";".join(f"{p:.2f}" for p in r.pkas),
                "route": r.route,
                "inchi_key": r.inchi_key,
                "cache_compound_id": r.cache_compound_id or "",
                "cache_match_level": r.cache_match_level or "",
                "cache_compound_formula": r.cache_compound_formula or "",
                "cache_same_protonation_state": (
                    r.cache_same_protonation_state
                    if r.cache_same_protonation_state is not None
                    else ""
                ),
                "compound_id": r.compound_id or "",
                "external_route_reused_existing_compound": (
                    r.external_route_reused_existing_compound
                    if r.external_route_reused_existing_compound is not None
                    else ""
                ),
                "group_decomposition_ok": r.group_decomposition_ok,
                "dgf_prime_cache_kj_per_mol": r.dgf_prime_cache_kj_per_mol or "",
                "dgf_prime_external_kj_per_mol": r.dgf_prime_external_kj_per_mol or "",
                "dgf_cache_minus_external_kj_per_mol": (
                    r.dgf_cache_minus_external_kj_per_mol
                    if r.dgf_cache_minus_external_kj_per_mol is not None
                    else ""
                ),
                "dgf_prime_kj_per_mol": r.dgf_prime_kj_per_mol or "",
                "sigma_kj_per_mol": r.sigma_kj_per_mol or "",
                "outcome": r.outcome,
                "detail": (r.detail or "")[:120],
                "smiles": r.smiles,
            }
            for r in records
        ],
    )
    print(f"wrote {PRICED_PATH}")
    print(f"wrote {PRICED_CSV}")
    for outcome, count in report.outcome_counts.items():
        print(f"  {count:4d}  {outcome}")
    for route, count in report.route_counts.items():
        print(f"  route {count:4d}  {route}")
    print(
        f"priced {report.n_priced}/{report.n_attempted} names, "
        f"{report.metabolite_rows_priced} metabolite rows"
    )
    print(
        f"{report.n_written_structure_found_in_cache} written structures already in the "
        f"cache {report.cache_match_levels}; on "
        f"{report.n_external_route_reused_existing_compound} of them get_compounds "
        "returned that same compound and the supplied pKas went unused"
    )
    if report.n_both_routes_independent:
        block = report.dgf_route_difference_same_protonation_kj_per_mol
        print(
            f"independent on {report.n_both_routes_independent} names, |cache - external| "
            f"mean {report.dgf_route_difference_kj_per_mol['mean']:.2f} max "
            f"{report.dgf_route_difference_kj_per_mol['max']:.2f} kJ/mol"
        )
        if block["n"]:
            print(
                f"  restricted to the {int(block['n'])} pairs on the SAME protonation "
                f"state: cache - external mean {block['mean']:.2f}, median "
                f"{block['median']:.2f}, range {block['min']:.2f} to {block['max']:.2f} "
                "kJ/mol"
            )


# ---------------------------------------------------------------------------- coverage


class CoverageReport(BaseModel):
    """What the written lipids buy, at the metabolite and the reaction level."""

    built_at: str
    n_reactions: int
    n_metabolites: int
    n_unpriced_after_audit: int
    n_tier_b: int
    n_newly_priced_metabolites: int
    n_newly_priced_metabolites_cache_route: int
    newly_priced_by_class: dict[str, int]
    newly_priced_by_route: dict[str, int]
    #: Reaction counts. ``recomputation`` is the eQuilibrator route on its own, ``union``
    #: adds yeast-GEM's shipped metabolite column, ``recommended`` adds the tight
    #: composition surrogates, matching improve_parameter_coverage's own ladder.
    reactions: dict[str, dict[str, float | int]]
    flux_carrying: dict[str, float | int]
    sigma_newly_priced: list[float]
    sigma_newly_priced_cache_route: list[float]
    sigma_newly_priced_external_pka: list[float]
    sigma_cache_priced_lipids: list[float]
    n_lipid_names_in_gem: int
    notes: dict[str, str]


def cmd_coverage(args: argparse.Namespace) -> None:
    import cobra
    import pandas as pd

    sys.path.insert(0, SCRIPTS)
    from audit_reaction_thermo_coverage import (  # type: ignore[import-not-found]
        shipped_series,
    )
    from improve_parameter_coverage import (  # type: ignore[import-not-found]
        TIGHT_SPREAD_KJ,
    )

    model = cobra.io.read_sbml_model(osp.join(GEM_DIR, "model", "yeast-GEM.xml"))
    reactions = [r.id for r in model.reactions]
    participants = {r.id: {m.id for m in r.metabolites} for r in model.reactions}

    def priceable(priced: set[str]) -> set[str]:
        """Reactions all of whose participants carry a value.

        The same predicate as ``audit_reaction_thermo_coverage.main``'s local
        ``priceable``; it is a closure there, so it cannot be imported.
        """
        return {r for r in reactions if participants[r] and participants[r] <= priced}

    shipped_met = shipped_series("model_metDeltaG.csv")
    shipped_priced = set(shipped_met.index[shipped_met.notna()])
    recomputed = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_prime.parquet"))
    recomputed_priced = set(
        recomputed.loc[recomputed["dgf_prime_kj_per_mol"].notna(), "met_id"]
    )
    surrogates = pd.read_parquet(osp.join(THERMO_DIR, "metabolite_dgf_surrogate.parquet"))
    tight_priced = set(
        surrogates.loc[
            surrogates["surrogate_spread_kj_per_mol"] < TIGHT_SPREAD_KJ, "met_id"
        ]
    )

    with open(PRICED_PATH) as handle:
        price_report = PriceReport.model_validate_json(handle.read())
    lipid_priced_rows = [r for r in price_report.records if r.outcome == "priced"]
    lipid_met_ids = {m for r in lipid_priced_rows for m in r.met_ids}
    # The cache-routed subset is the conservative one: those values come from compounds
    # eQuilibrator itself holds, so they sit on the same energy zero as the recomputed
    # table. The externally created ones do not (see design decision 4), so both counts
    # are reported and the note can quote whichever the claim needs.
    lipid_met_ids_cache_route = {
        m for r in lipid_priced_rows if r.route == "cache_lookup" for m in r.met_ids
    }
    if lipid_met_ids & recomputed_priced:
        raise RuntimeError("a newly priced lipid was already priced by the recomputation")

    routes = {
        "recomputation": recomputed_priced,
        "recomputation_plus_lipids_cache_route": recomputed_priced
        | lipid_met_ids_cache_route,
        "recomputation_plus_lipids": recomputed_priced | lipid_met_ids,
        "union_shipped_recomputed": shipped_priced | recomputed_priced,
        "union_plus_lipids_cache_route": (
            shipped_priced | recomputed_priced | lipid_met_ids_cache_route
        ),
        "union_plus_lipids": shipped_priced | recomputed_priced | lipid_met_ids,
        "recommended_union_plus_tight_surrogate": (
            shipped_priced | recomputed_priced | tight_priced
        ),
        "recommended_plus_lipids_cache_route": (
            shipped_priced | recomputed_priced | tight_priced | lipid_met_ids_cache_route
        ),
        "recommended_plus_lipids": (
            shipped_priced | recomputed_priced | tight_priced | lipid_met_ids
        ),
        "shipped_metabolite_sum": shipped_priced,
    }
    covered = {name: priceable(priced) for name, priced in routes.items()}
    reaction_block = {
        name: {
            "n_metabolites_priced": len(routes[name]),
            "n_reactions": len(rxns),
            "frac_reactions": len(rxns) / len(reactions),
        }
        for name, rxns in covered.items()
    }
    reaction_block["gain_from_lipids"] = {
        "recomputation_only": len(
            covered["recomputation_plus_lipids"] - covered["recomputation"]
        ),
        "recomputation_only_cache_route": len(
            covered["recomputation_plus_lipids_cache_route"] - covered["recomputation"]
        ),
        "union": len(covered["union_plus_lipids"] - covered["union_shipped_recomputed"]),
        "union_cache_route": len(
            covered["union_plus_lipids_cache_route"] - covered["union_shipped_recomputed"]
        ),
        "recommended": len(
            covered["recommended_plus_lipids"]
            - covered["recommended_union_plus_tight_surrogate"]
        ),
        "recommended_cache_route": len(
            covered["recommended_plus_lipids_cache_route"]
            - covered["recommended_union_plus_tight_surrogate"]
        ),
    }

    fva = pd.read_csv(osp.join(RESULTS, "fva_wildtype.csv"), index_col=0)
    carries_flux = set(fva.index[fva["width"] > 1e-9])
    flux_carrying = {
        "n_reactions_with_nonzero_fva_width": len(carries_flux),
        "n_recommended": len(covered["recommended_union_plus_tight_surrogate"] & carries_flux),
        "n_recommended_plus_lipids_cache_route": len(
            covered["recommended_plus_lipids_cache_route"] & carries_flux
        ),
        "n_recommended_plus_lipids": len(covered["recommended_plus_lipids"] & carries_flux),
    }
    total_flux = flux_carrying["n_reactions_with_nonzero_fva_width"]
    flux_carrying["frac_recommended"] = flux_carrying["n_recommended"] / total_flux
    flux_carrying["frac_recommended_plus_lipids_cache_route"] = (
        flux_carrying["n_recommended_plus_lipids_cache_route"] / total_flux
    )
    flux_carrying["frac_recommended_plus_lipids"] = (
        flux_carrying["n_recommended_plus_lipids"] / total_flux
    )

    # Panel c needs a comparison population: the yeast-GEM metabolites this parser also
    # recognizes as lipids and that the cache could already price on its own.
    lipid_names_in_gem = set()
    for metabolite in model.metabolites:
        template, _, _ = parse_name(metabolite.name)
        if template is not None:
            lipid_names_in_gem.add(metabolite.name)
    # One value per distinct NAME, because the newly priced set is one value per name and
    # a per-metabolite count would weight a chemical by how many compartments it sits in.
    cache_lipid_sigma = (
        recomputed.loc[
            recomputed["name"].isin(lipid_names_in_gem)
            & recomputed["sigma_kj_per_mol"].notna(),
            ["name", "sigma_kj_per_mol"],
        ]
        .drop_duplicates("name")["sigma_kj_per_mol"]
    )

    with open(AUDIT_PATH) as handle:
        audit = json.load(handle)
    n_unpriced = sum(
        1
        for row in audit["metabolites_unpriced_at_baseline"]
        if not row.get("recovered_priceable")
    )

    report = CoverageReport(
        built_at=_timestamp(),
        n_reactions=len(reactions),
        n_metabolites=len(model.metabolites),
        n_unpriced_after_audit=n_unpriced,
        n_tier_b=len(_tier_b_rows()),
        n_newly_priced_metabolites=len(lipid_met_ids),
        n_newly_priced_metabolites_cache_route=len(lipid_met_ids_cache_route),
        newly_priced_by_class=dict(
            Counter(
                cls
                for r in lipid_priced_rows
                for cls in [r.chemical_class] * r.n_metabolite_rows
            ).most_common()
        ),
        newly_priced_by_route=dict(
            Counter(
                route
                for r in lipid_priced_rows
                for route in [r.route] * r.n_metabolite_rows
            ).most_common()
        ),
        reactions=cast(dict, reaction_block),
        flux_carrying=cast(dict, flux_carrying),
        sigma_newly_priced=[
            cast(float, r.sigma_kj_per_mol)
            for r in lipid_priced_rows
            if r.sigma_kj_per_mol is not None
        ],
        sigma_newly_priced_cache_route=[
            cast(float, r.sigma_kj_per_mol)
            for r in lipid_priced_rows
            if r.sigma_kj_per_mol is not None and r.route == "cache_lookup"
        ],
        sigma_newly_priced_external_pka=[
            cast(float, r.sigma_kj_per_mol)
            for r in lipid_priced_rows
            if r.sigma_kj_per_mol is not None and r.route == "external_pka"
        ],
        sigma_cache_priced_lipids=[float(v) for v in cache_lipid_sigma],
        n_lipid_names_in_gem=len(lipid_names_in_gem),
        notes={
            "conditions": (
                f"the new values are dGf'0 at pH {P_H}, I = {IONIC_STRENGTH_M} M, "
                f"T = {TEMPERATURE_K} K, pMg {P_MG}, while the recomputed table transforms "
                "each metabolite at its own compartment pH, so the two are not on one "
                "transform and only the reaction COUNTS are comparable here"
            ),
            "priceable": (
                "a reaction is priceable when every one of its participants carries a "
                "formation energy, the same predicate audit_reaction_thermo_coverage uses"
            ),
            "recommended": (
                "shipped metabolite column, the eQuilibrator recomputation, and the tight "
                "composition surrogates, which is improve_parameter_coverage's own "
                "recommended set"
            ),
        },
    )
    with open(COVERAGE_PATH, "w") as handle:
        handle.write(report.model_dump_json(indent=1))
    print(f"wrote {COVERAGE_PATH}")
    print(
        f"{report.n_newly_priced_metabolites} of {report.n_unpriced_after_audit} unpriced "
        f"metabolites are now priced"
    )
    for name, block in reaction_block.items():
        if name == "gain_from_lipids":
            continue
        print(
            f"  {name:42s} {block['n_reactions']:5d}  {block['frac_reactions']:6.2%}"
        )
    print(f"  gain from lipids: {reaction_block['gain_from_lipids']}")
    print(
        f"  flux-carrying recommended {flux_carrying['frac_recommended']:.2%} -> "
        f"{flux_carrying['frac_recommended_plus_lipids_cache_route']:.2%} (cache route) "
        f"-> {flux_carrying['frac_recommended_plus_lipids']:.2%} (all)"
    )


# -------------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="decompose the tier-b names")
    p_parse.set_defaults(func=cmd_parse)

    p_build = sub.add_parser("build", help="write the SMILES and check the formulas")
    p_build.set_defaults(func=cmd_build)

    p_pkas = sub.add_parser("pkas", help="MolGpKa ladders for the passing structures")
    p_pkas.add_argument("--repo", required=True, help="a MolGpKa checkout (holds models/)")
    p_pkas.add_argument("--tool-version", type=str, default="MolGpKa@main")
    p_pkas.add_argument("--threads", type=int, default=4)
    p_pkas.set_defaults(func=cmd_pkas)

    p_price = sub.add_parser("price", help="create the compounds and price them")
    p_price.add_argument("--cache", type=str, default=LOCAL_CCACHE)
    p_price.set_defaults(func=cmd_price)

    p_cov = sub.add_parser("coverage", help="reaction coverage before and after")
    p_cov.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
