# torchcell/metabolism/kinetics_pipeline
# [[torchcell.metabolism.kinetics_pipeline]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/kinetics_pipeline
# Test file: tests/torchcell/metabolism/test_kinetics_pipeline.py

r"""From a GEM plus a proteome to per-catalytic-unit :math:`k_{cat}` and :math:`K_M`.

WHY A PIPELINE AND NOT A YEAST SCRIPT
--------------------------------------
The parameter tables in :mod:`torchcell.metabolism.parameters` are yeast-shaped by
accident of where they grew: they read ``swissprot.tsv`` and ``smilesDB.tsv``, two files
that exist because yeast-GEM ships them. Neither exists for a new bacterial chassis, and
neither is where the information actually lives. What a GEM always carries is a GPR rule
per reaction, an annotation dictionary per metabolite, and a formula plus a charge per
metabolite; what an organism always has is a proteome. This module takes exactly those and
nothing else, so porting to another organism is supplying two files rather than editing
code.

Four stages, each one separately reportable because each one fails differently:

1. **sequence** -- gene to (accession, protein sequence), from a proteome mapping the
   caller supplies. A gene the mapping misses is ``no_uniprot_mapping`` or
   ``sequence_missing``; a catalytic unit with only some subunits resolved is
   ``complex_rule_unhandled``, since the unit's turnover is the min over its members and a
   missing member makes that min undefined rather than smaller.
2. **structure** -- substrate metabolite to SMILES, through the routes in
   :class:`SubstrateStructureResolver`, every one of them accepted only against the GEM's
   own formula and charge.
3. **measured** -- an optional first tier from a kinetics database, keyed either on the
   accession or on EC number plus substrate plus organism. Keying on the accession alone
   is the strictest and the thinnest; the EC key is what lets a non-model organism inherit
   a measurement from a characterized homolog, and it is a DIFFERENT claim, so it is
   tagged differently and never silently merged.
4. **predicted** -- the median over the sequence predictors a within-class band study
   admits. Which predictors those are is a computed stage
   (:func:`~torchcell.metabolism.parameters.PredictorSelection.from_study`), so another
   organism re-decides it from its own statistics rather than inheriting yeast's three.

THE ACCEPTANCE TEST FOR A STRUCTURE, WHICH IS THE WHOLE GAME
-------------------------------------------------------------
Identifier cross-references are dirty and a wrong SMILES is worse than no SMILES: a
predictor will return a confident number for the wrong molecule and nothing downstream can
tell. So a candidate structure is accepted only when the molecular formula RDKit computes
from the SMILES equals the GEM's formula for that metabolite, both moved to the neutral
species by adding one hydrogen per negative charge. Comparing at neutrality is what makes
the test usable at all: a GEM writes the physiological ionization state
(``C6H10O12P2`` at charge -4 for fructose 1,6-bisphosphate) while a structure database
usually writes the neutral acid (``C6H14O12P2``), and those are the same compound.

The test also rejects, for free and for the right reason, every polymer and R-group
placeholder. yeast-GEM writes an attachment point as ``R`` and MetaNetX as ``*``, so
``C4H7O2SR`` never equals ``C4H7O2S*``: a species defined only up to an unspecified
substituent has no structure to give a predictor, and dropping it is correct rather than
lossy.
"""

from __future__ import annotations

import hashlib
import json
import os.path as osp
import re
from collections import Counter, defaultdict
from collections.abc import Callable
from enum import StrEnum
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field

#: One element plus an optional count in a Hill-notation formula. ``R`` and ``*`` are
#: parsed as ordinary tokens on purpose, so a placeholder never silently compares equal to
#: a real composition.
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?|\*|R)(\d*)")


class Blocker(StrEnum):
    """Why one catalytic unit cannot be given a value on the tier being audited."""

    NO_UNIPROT_MAPPING = "no_uniprot_mapping"
    SEQUENCE_MISSING = "sequence_missing"
    COMPLEX_RULE_UNHANDLED = "complex_rule_unhandled"
    REACTION_HAS_NO_SUBSTRATE = "reaction_has_no_substrate"
    SUBSTRATE_STRUCTURE_MISSING = "substrate_structure_missing"
    PREDICTOR_FAILED = "predictor_failed"
    OTHER = "other"


class StructureRoute(StrEnum):
    """How a substrate's SMILES was obtained. Recorded per metabolite, never aggregated
    away, because the routes have different failure modes and different trust.
    """

    #: An exact metabolite-name match against a table the GEM ships.
    NAME_TABLE = "name_table"
    #: The metabolite's own annotation id found directly in the structure table.
    ANNOTATION_ID = "annotation_id"
    #: An annotation id resolved through a cross-reference table to a current id that the
    #: structure table carries. This is what recovers a GEM annotated against an older
    #: release of the structure database.
    ANNOTATION_XREF = "annotation_xref"
    #: A structure written by an external builder for a name the databases do not carry,
    #: such as the acyl-chain-specific lipid species a GEM enumerates combinatorially.
    SUPPLIED = "supplied"


def _disable_rdkit_logs() -> None:
    """Silence rdkit's C++ logger before a parse.

    ``rdkit-stubs`` does not declare ``RDLogger.DisableLog``: the name is re-exported from
    the compiled ``rdBase`` module at runtime, so the attribute is reached through a typed
    accessor rather than read off the stub.
    """
    from rdkit import RDLogger

    disable = cast(Callable[[str], None], getattr(RDLogger, "DisableLog"))
    disable("rdApp.*")


def parse_formula(formula: str) -> Counter[str]:
    """Element counts of a Hill-notation formula, placeholders included as elements."""
    counts: Counter[str] = Counter()
    for element, digits in _FORMULA_TOKEN.findall(formula or ""):
        counts[element] += int(digits) if digits else 1
    return counts


def neutral_formula(formula: str, charge: float) -> Counter[str]:
    """Element counts moved to the neutral species by adding one H per negative charge.

    A GEM and a structure database record the same compound at different ionization
    states, so the composition is only comparable once both are neutralized. A cation
    loses hydrogens; the result is not clamped, and a negative hydrogen count simply fails
    the comparison, which is the honest outcome for a formula that cannot be neutralized.
    """
    counts = parse_formula(formula)
    counts["H"] = counts.get("H", 0) - int(round(charge))
    if counts["H"] == 0:
        del counts["H"]
    return counts


def formula_from_smiles(smiles: str) -> str | None:
    """RDKit's molecular formula for a SMILES, or None when it does not parse.

    The formula is recomputed from the structure rather than read from the database's own
    formula column, because the column and the structure disagree in real releases and the
    structure is what a predictor consumes.
    """
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    _disable_rdkit_logs()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return CalcMolFormula(mol)


def _strip_charge_suffix(formula: str) -> tuple[str, int]:
    """Split RDKit's ``C6H10O12P2-4`` into its formula and its net charge."""
    match = re.match(r"^(.*?)([+-]\d*)$", formula)
    if not match:
        return formula, 0
    body, sign = match.groups()
    magnitude = int(sign[1:]) if len(sign) > 1 else 1
    return body, magnitude if sign[0] == "+" else -magnitude


def structure_matches_metabolite(
    smiles: str, gem_formula: str, gem_charge: float
) -> bool:
    """Does this SMILES describe the metabolite the GEM means?

    Both sides are neutralized before comparison, so an anion in the GEM matches the
    neutral acid in the structure database. A SMILES that does not parse, or one carrying
    an unspecified substituent, fails.
    """
    if not gem_formula:
        return False
    computed = formula_from_smiles(smiles)
    if computed is None:
        return False
    body, charge = _strip_charge_suffix(computed)
    return neutral_formula(body, charge) == neutral_formula(gem_formula, gem_charge)


class ResolvedStructure(BaseModel):
    """One substrate structure, with the route and the key it was reached by."""

    model_config = ConfigDict(extra="forbid")

    met_id: str
    smiles: str
    route: StructureRoute
    #: The annotation key or name the structure was reached through, e.g. ``chebi:16905``.
    key: str
    #: The identifier in the structure table the SMILES was read from, when the route went
    #: through one.
    structure_id: str | None = None
    gem_formula: str
    gem_charge: float
    #: The formula RDKit computed from the accepted SMILES, before neutralization.
    smiles_formula: str


class StructureCandidate(BaseModel):
    """A SMILES offered by one route, before the formula test decides on it."""

    model_config = ConfigDict(extra="forbid")

    smiles: str
    route: StructureRoute
    key: str
    structure_id: str | None = None


class CrossReferenceStructureIndex(BaseModel):
    """SMILES keyed by a canonical compound id, plus a cross-reference into it.

    Built by streaming two tab-separated tables, MetaNetX ``chem_prop.tsv`` and
    ``chem_xref.tsv`` in the reference build, and filtered to the identifiers actually
    asked for, since the pair is 1.5 GB and a GEM needs a few thousand rows of it. Nothing
    about the class is MetaNetX-specific beyond the column positions the loader is given.
    """

    model_config = ConfigDict(extra="forbid")

    #: canonical id -> SMILES
    smiles_by_id: dict[str, str] = Field(default_factory=dict)
    #: external id, verbatim as the xref table spells it -> canonical ids
    ids_by_xref: dict[str, list[str]] = Field(default_factory=dict)
    source_paths: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_metanetx(
        cls,
        chem_prop_path: str,
        chem_xref_path: str,
        wanted_xrefs: set[str],
        smiles_column: int = 8,
    ) -> CrossReferenceStructureIndex:
        """Load only the rows the requested cross-references can reach.

        Two passes, one per file, because the second file's row set is decided by the
        first: the xref table says which canonical ids are reachable, and only those are
        kept from the property table.
        """
        ids_by_xref: dict[str, list[str]] = defaultdict(list)
        with open(chem_xref_path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                columns = line.split("\t", 2)
                if len(columns) > 1 and columns[0] in wanted_xrefs:
                    ids_by_xref[columns[0]].append(columns[1])
        reachable = {i for ids in ids_by_xref.values() for i in ids}
        reachable |= {x for x in wanted_xrefs if ":" not in x}
        smiles_by_id: dict[str, str] = {}
        with open(chem_prop_path) as handle:
            for line in handle:
                if line.startswith("#"):
                    continue
                columns = line.rstrip("\n").split("\t")
                if columns[0] in reachable and len(columns) > smiles_column:
                    smiles = columns[smiles_column].strip()
                    if smiles:
                        smiles_by_id[columns[0]] = smiles
        return cls(
            smiles_by_id=smiles_by_id,
            ids_by_xref=dict(ids_by_xref),
            source_paths={
                chem_prop_path: sha256_of(chem_prop_path),
                chem_xref_path: sha256_of(chem_xref_path),
            },
        )


def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    """sha256 of a file a pipeline stage read, so a result pins its own inputs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


#: How a metabolite annotation namespace is spelled in the cross-reference table, for the
#: namespaces a cobra GEM commonly carries. Extending this is how another structure
#: database is wired in; nothing here is organism-specific.
XREF_SPELLINGS: dict[str, tuple[str, ...]] = {
    "chebi": ("{value_lower}", "{value_upper}"),
    "kegg.compound": ("kegg.compound:{value}", "keggC:{value}", "kegg:{value}"),
    "bigg.metabolite": ("bigg.metabolite:{value}", "biggM:{value}"),
    "metanetx.chemical": ("{value}", "mnx:{value}"),
    "seed.compound": ("seed.compound:{value}", "seedM:{value}"),
    "hmdb": ("hmdb:{value}",),
}

#: Which annotation namespace is tried first. A ChEBI id names one chemical species; a
#: BiGG id names a modeling convention and is the loosest, so it goes last.
ANNOTATION_PRIORITY = (
    "metanetx.chemical",
    "chebi",
    "kegg.compound",
    "seed.compound",
    "hmdb",
    "bigg.metabolite",
)


def annotation_values(metabolite: Any, namespace: str) -> list[str]:
    """The metabolite's ids in one namespace; cobra holds these as a string or a list."""
    reference = (metabolite.annotation or {}).get(namespace)
    if reference is None:
        return []
    values = reference if isinstance(reference, list) else [reference]
    return [str(v) for v in values if v]


def xref_spellings(namespace: str, value: str) -> list[str]:
    """Every way the cross-reference table might spell one annotation id."""
    patterns = XREF_SPELLINGS.get(namespace)
    if patterns is None:
        return []
    return [
        pattern.format(
            value=value, value_lower=value.lower(), value_upper=value.upper()
        )
        for pattern in patterns
    ]


class SubstrateStructureResolver(BaseModel):
    """Substrate SMILES for a GEM's metabolites, by route, accepted on formula.

    The routes are tried in the order they are listed in :meth:`resolve`, and the first
    candidate that passes the formula test wins. Trying them in order rather than
    collecting a consensus is deliberate: the routes are not independent opinions about a
    structure, they are lookups of decreasing specificity, and a candidate that fails the
    formula test is not evidence about the ones that pass.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    #: Lowercased metabolite name -> SMILES, from a table the GEM ships. Optional: a GEM
    #: that ships no name table simply has one route fewer.
    name_table: dict[str, str] = Field(default_factory=dict)
    #: Canonical-id structure table plus its cross-reference.
    index: CrossReferenceStructureIndex | None = None
    #: Metabolite id -> SMILES written by an external builder, already formula-checked by
    #: that builder and checked again here.
    supplied: dict[str, str] = Field(default_factory=dict)

    def candidates(self, metabolite: Any) -> list[StructureCandidate]:
        """Every SMILES any route offers for this metabolite, in route order."""
        out: list[StructureCandidate] = []
        hit = self.name_table.get((metabolite.name or "").lower())
        if hit:
            out.append(
                StructureCandidate(
                    smiles=hit,
                    route=StructureRoute.NAME_TABLE,
                    key=(metabolite.name or "").lower(),
                )
            )
        if metabolite.id in self.supplied:
            out.append(
                StructureCandidate(
                    smiles=self.supplied[metabolite.id],
                    route=StructureRoute.SUPPLIED,
                    key=metabolite.id,
                )
            )
        if self.index is not None:
            for namespace in ANNOTATION_PRIORITY:
                for value in annotation_values(metabolite, namespace):
                    direct = self.index.smiles_by_id.get(value)
                    if direct:
                        out.append(
                            StructureCandidate(
                                smiles=direct,
                                route=StructureRoute.ANNOTATION_ID,
                                key=f"{namespace}:{value}",
                                structure_id=value,
                            )
                        )
                    for spelling in xref_spellings(namespace, value):
                        for canonical in self.index.ids_by_xref.get(spelling, []):
                            smiles = self.index.smiles_by_id.get(canonical)
                            if smiles:
                                out.append(
                                    StructureCandidate(
                                        smiles=smiles,
                                        route=StructureRoute.ANNOTATION_XREF,
                                        key=spelling,
                                        structure_id=canonical,
                                    )
                                )
        return out

    def resolve(self, metabolite: Any) -> ResolvedStructure | None:
        """The first candidate whose composition matches the GEM's, or None."""
        for candidate in self.candidates(metabolite):
            formula = metabolite.formula or ""
            charge = float(metabolite.charge or 0.0)
            if structure_matches_metabolite(candidate.smiles, formula, charge):
                computed = formula_from_smiles(candidate.smiles)
                return ResolvedStructure(
                    met_id=metabolite.id,
                    smiles=candidate.smiles,
                    route=candidate.route,
                    key=candidate.key,
                    structure_id=candidate.structure_id,
                    gem_formula=formula,
                    gem_charge=charge,
                    smiles_formula=computed or "",
                )
        return None

    def wanted_xrefs(self, metabolites: list[Any]) -> set[str]:
        """Every cross-reference spelling these metabolites could be looked up by.

        Used to build the index before the resolver exists, which is why it is a method on
        the class that will consume it rather than free-floating.
        """
        wanted: set[str] = set()
        for metabolite in metabolites:
            for namespace in ANNOTATION_PRIORITY:
                for value in annotation_values(metabolite, namespace):
                    wanted.add(value)
                    wanted.update(xref_spellings(namespace, value))
        return wanted


def wanted_xrefs_for(metabolites: list[Any]) -> set[str]:
    """Cross-reference spellings for a metabolite list, before a resolver is built."""
    return SubstrateStructureResolver().wanted_xrefs(metabolites)


def inchikey_block(smiles: str) -> str | None:
    """The connectivity block of a SMILES' InChIKey, the substrate-identity key.

    The first block of an InChIKey encodes the skeleton and drops protonation and
    stereochemistry, which is exactly the tolerance a substrate comparison needs: a
    kinetics database writes the neutral acid it assayed and a GEM writes the species at
    physiological pH, and those name the same substrate.
    """
    from rdkit import Chem

    _disable_rdkit_logs()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Unannotated in rdkit-stubs; the wrapper in rdkit.Chem.inchi returns the key string.
    to_inchikey = cast(Callable[[Any], str], Chem.MolToInchiKey)
    key = to_inchikey(mol)
    return key.split("-")[0] if key else None


# =====================================================================================
# STAGE 1: SEQUENCE
# =====================================================================================


class ProteinRecord(BaseModel):
    """One gene's protein, as a kinetics predictor needs it."""

    model_config = ConfigDict(extra="forbid")

    gene_id: str
    sequence: str
    #: Database accession, needed only by the accession-keyed measured tier. A proteome
    #: without accessions still runs every other stage.
    accession: str | None = None


def read_proteome(
    path: str, accessions_path: str | None = None
) -> dict[str, ProteinRecord]:
    """Gene id -> protein, from a FASTA or a ``{gene_id: sequence}`` JSON.

    A FASTA header's FIRST whitespace-delimited token is taken as the gene id, which is the
    convention every reference proteome follows and the one a GEM's GPR uses. Accessions
    come from a separate ``{gene_id: accession}`` JSON when there are any: a proteome file
    and an accession mapping are two different artifacts with two different provenances,
    and merging them into one bespoke format is what makes a pipeline unportable.
    """
    records: dict[str, ProteinRecord] = {}
    if path.endswith(".json"):
        with open(path) as handle:
            for gene, sequence in json.load(handle).items():
                records[gene] = ProteinRecord(gene_id=gene, sequence=sequence)
    else:
        gene_id: str | None = None
        chunks: list[str] = []
        with open(path) as handle:
            for line in handle:
                if line.startswith(">"):
                    if gene_id is not None:
                        records[gene_id] = ProteinRecord(
                            gene_id=gene_id, sequence="".join(chunks)
                        )
                    gene_id, chunks = line[1:].split()[0], []
                else:
                    chunks.append(line.strip())
        if gene_id is not None:
            records[gene_id] = ProteinRecord(gene_id=gene_id, sequence="".join(chunks))
    if accessions_path:
        with open(accessions_path) as handle:
            for gene_id, accession in json.load(handle).items():
                if gene_id in records:
                    records[gene_id].accession = accession
    return records


# =====================================================================================
# STAGE 2: PER-UNIT INPUTS
# =====================================================================================


class PairRow(BaseModel):
    """One ``(catalytic unit, gene, substrate)`` row, the unit of predictor inference.

    A turnover number is a property of one enzyme acting on one substrate, so a unit with
    three genes and two substrates is six rows. Collapsing to one row per unit would pick a
    substrate silently.
    """

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    gene_id: str
    accession: str | None
    sequence_length: int
    substrate_met_id: str
    substrate_name: str
    smiles: str
    smiles_route: StructureRoute


class UnitBlock(BaseModel):
    """Why one catalytic unit produced no predictor input."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    gene_ids: list[str]
    blocker: Blocker
    detail: str


class UnitInputs(BaseModel):
    """The predictor input table for a whole GEM, plus the units that produced none."""

    model_config = ConfigDict(extra="forbid")

    rows: list[PairRow]
    blocked: list[UnitBlock]
    n_units: int
    structures: dict[str, ResolvedStructure]

    def blocked_counts(self) -> dict[str, int]:
        """How many units each blocker refused."""
        return dict(Counter(str(b.blocker) for b in self.blocked).most_common())


def substrate_indices(reaction: Any) -> list[Any]:
    """The metabolites a reaction consumes. Products are not predictor inputs."""
    return [m for m, coefficient in reaction.metabolites.items() if coefficient < 0]


def build_unit_inputs(
    model: Any,
    units: Any,
    proteins: dict[str, ProteinRecord],
    resolver: SubstrateStructureResolver,
) -> UnitInputs:
    """Turn a GPR-parsed GEM plus a proteome into predictor inputs, per catalytic unit.

    Args:
        model: any cobra model.
        units: :class:`~torchcell.metabolism.constraints.CatalyticUnits` for that model.
        proteins: gene id -> protein, from :func:`read_proteome`.
        resolver: substrate structures, already built.

    Every unit that yields no row carries a :class:`Blocker` saying which stage refused it,
    so the coverage report is a partition of the units rather than a percentage.
    """
    unit_genes: dict[int, list[str]] = defaultdict(list)
    for unit, gene in units.unit_gene_index.t().tolist():
        unit_genes[int(unit)].append(units.gene_ids[int(gene)])

    structures: dict[str, ResolvedStructure] = {}
    rows: list[PairRow] = []
    blocked: list[UnitBlock] = []
    for unit_id in range(units.n_units):
        reaction = model.reactions[int(units.unit_reaction[unit_id])]
        genes = unit_genes[unit_id]
        present = [g for g in genes if g in proteins]
        if not present:
            blocked.append(
                UnitBlock(
                    unit_id=unit_id,
                    reaction_id=reaction.id,
                    gene_ids=genes,
                    blocker=Blocker.NO_UNIPROT_MAPPING,
                    detail="no member gene is in the proteome",
                )
            )
            continue
        if len(present) < len(genes):
            blocked.append(
                UnitBlock(
                    unit_id=unit_id,
                    reaction_id=reaction.id,
                    gene_ids=genes,
                    blocker=Blocker.COMPLEX_RULE_UNHANDLED,
                    detail=(
                        f"{len(genes) - len(present)} of {len(genes)} subunits are absent "
                        "from the proteome; the unit's value is a min over its members and "
                        "a missing member makes that min undefined"
                    ),
                )
            )
            continue
        empty = [g for g in present if not proteins[g].sequence]
        if empty:
            blocked.append(
                UnitBlock(
                    unit_id=unit_id,
                    reaction_id=reaction.id,
                    gene_ids=genes,
                    blocker=Blocker.SEQUENCE_MISSING,
                    detail=f"empty sequence for {', '.join(empty)}",
                )
            )
            continue

        substrates = substrate_indices(reaction)
        if not substrates:
            blocked.append(
                UnitBlock(
                    unit_id=unit_id,
                    reaction_id=reaction.id,
                    gene_ids=genes,
                    blocker=Blocker.REACTION_HAS_NO_SUBSTRATE,
                    detail=f"{reaction.id} consumes nothing",
                )
            )
            continue
        resolved: list[ResolvedStructure] = []
        for metabolite in substrates:
            if metabolite.id not in structures:
                hit = resolver.resolve(metabolite)
                if hit is not None:
                    structures[metabolite.id] = hit
            if metabolite.id in structures:
                resolved.append(structures[metabolite.id])
        if not resolved:
            blocked.append(
                UnitBlock(
                    unit_id=unit_id,
                    reaction_id=reaction.id,
                    gene_ids=genes,
                    blocker=Blocker.SUBSTRATE_STRUCTURE_MISSING,
                    detail=(
                        f"none of {len(substrates)} substrates resolves to a structure "
                        "that matches the GEM's formula"
                    ),
                )
            )
            continue
        for gene in present:
            for structure in resolved:
                metabolite = model.metabolites.get_by_id(structure.met_id)
                rows.append(
                    PairRow(
                        unit_id=unit_id,
                        reaction_id=reaction.id,
                        gene_id=gene,
                        accession=proteins[gene].accession,
                        sequence_length=len(proteins[gene].sequence),
                        substrate_met_id=structure.met_id,
                        substrate_name=metabolite.name or "",
                        smiles=structure.smiles,
                        smiles_route=structure.route,
                    )
                )
    return UnitInputs(
        rows=rows, blocked=blocked, n_units=units.n_units, structures=structures
    )


# =====================================================================================
# STAGE 3: MEASURED
# =====================================================================================


class MeasuredKey(StrEnum):
    """How a database row is matched to a catalytic unit.

    The keys are not interchangeable and the tier they produce is tagged with the key that
    produced it. An accession match is a measurement OF THIS PROTEIN; an EC match in the
    same organism is a measurement of a protein doing the same chemistry; an EC match in
    another organism is a measurement of a homolog. Only the first is what "measured"
    usually means, and only the third is available for an organism nobody has assayed.
    """

    UNIPROT = "uniprot"
    UNIPROT_SUBSTRATE = "uniprot_substrate"
    EC_ORGANISM = "ec_organism"
    EC_SUBSTRATE_ORGANISM = "ec_substrate_organism"
    EC_SUBSTRATE_ANY_ORGANISM = "ec_substrate_any_organism"

    @property
    def requires_substrate_match(self) -> bool:
        """Does this key require the assayed substrate to be one the reaction consumes?"""
        return self in {
            MeasuredKey.UNIPROT_SUBSTRATE,
            MeasuredKey.EC_SUBSTRATE_ORGANISM,
            MeasuredKey.EC_SUBSTRATE_ANY_ORGANISM,
        }

    @property
    def is_accession_keyed(self) -> bool:
        """Does this key match on the protein rather than on the reaction's chemistry?"""
        return self in {MeasuredKey.UNIPROT, MeasuredKey.UNIPROT_SUBSTRATE}


class Tier(StrEnum):
    """Where one unit's value came from, at the resolution the report needs.

    Coarser reporting merges the three measured tiers into ``measured``; the split is kept
    in the record so a run can be filtered to protein-level measurements alone.
    """

    MEASURED_PROTEIN = "measured_protein"
    MEASURED_EC_SAME_ORGANISM = "measured_ec_same_organism"
    MEASURED_EC_HOMOLOG = "measured_ec_other_organism"
    PREDICTED = "predicted"
    DEFAULT = "organism_default"

    @property
    def is_measured(self) -> bool:
        """Is this tier a database value rather than a prediction or a constant?"""
        return self.value.startswith("measured")


def reaction_ec_codes(reaction: Any) -> list[str]:
    """EC numbers annotated on a reaction, as a list of strings."""
    reference = (reaction.annotation or {}).get("ec-code")
    if reference is None:
        return []
    return [str(r) for r in (reference if isinstance(reference, list) else [reference])]


def read_supplied_structures(path: str) -> dict[str, str]:
    """Metabolite id -> SMILES from an external structure builder's JSON report.

    The expected shape is the one ``lipid_structure_writer.py`` writes: a ``structures``
    list whose entries carry ``met_ids``, ``smiles`` and a ``formula_match`` flag. Only
    formula-matching entries are read, and they are formula-checked again by the resolver
    against the GEM, so a builder's own acceptance is never taken on trust.
    """
    if not osp.exists(path):
        return {}
    with open(path) as handle:
        report = json.load(handle)
    out: dict[str, str] = {}
    for entry in report.get("structures", []):
        if not entry.get("formula_match") or not entry.get("smiles"):
            continue
        for met_id in entry.get("met_ids", []):
            out[met_id] = entry["smiles"]
    return out


class MeasuredValue(BaseModel):
    """One database measurement assigned to one catalytic unit."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    gene_id: str
    value: float
    unit: str | None
    key: MeasuredKey
    tier: Tier
    organism: str | None
    substrate_assayed: str | None
    substrate_matched_by: str | None
    temperature_c: float | None
    pubmed_id: str | None
    n_candidates: int
    selection_rule: str


def match_measured(
    records: list[Any],
    model: Any,
    units: Any,
    proteins: dict[str, ProteinRecord],
    structures: dict[str, ResolvedStructure],
    kind: Any,
    key: MeasuredKey,
    organism: str | None = None,
    target_temperature_c: float = 30.0,
) -> dict[int, MeasuredValue]:
    r"""Assign database measurements to catalytic units under one key.

    The per-unit value is the MINIMUM over member genes for turnover and the MAXIMUM for
    affinity, which is the same rule the flux layer applies: a complex cannot turn over
    faster than its slowest subunit, and its weakest binding step sets where it saturates.
    Within one gene the choice among candidate rows is
    :func:`~torchcell.metabolism.enzyme_kinetics.resolve_parameter`'s documented cascade,
    reused rather than reimplemented.

    ``key`` decides which rows are candidates at all, and the substrate filter, where the
    key has one, is applied BEFORE that cascade. That ordering matters: filtering after
    would let a row assayed on an unrelated substrate win the temperature tie-break and
    then be discarded, leaving the unit unmatched when a valid row existed.
    """
    from torchcell.metabolism.enzyme_kinetics import KineticKind, resolve_parameter

    by_accession: dict[str, list[Any]] = defaultdict(list)
    by_ec: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        if organism is not None and record.organism != organism:
            continue
        if record.uniprot:
            by_accession[record.uniprot].append(record)
        if record.ec:
            by_ec[record.ec].append(record)

    unit_genes: dict[int, list[str]] = defaultdict(list)
    for unit, gene in units.unit_gene_index.t().tolist():
        unit_genes[int(unit)].append(units.gene_ids[int(gene)])

    key_cache: dict[str, str | None] = {}

    def skeleton(smiles: str) -> str | None:
        if smiles not in key_cache:
            key_cache[smiles] = inchikey_block(smiles)
        return key_cache[smiles]

    out: dict[int, MeasuredValue] = {}
    for unit_id in range(units.n_units):
        reaction = model.reactions[int(units.unit_reaction[unit_id])]
        wanted_keys: set[str] = set()
        wanted_names: set[str] = set()
        for metabolite in substrate_indices(reaction):
            wanted_names.add((metabolite.name or "").lower())
            structure = structures.get(metabolite.id)
            if structure is not None:
                block = skeleton(structure.smiles)
                if block:
                    wanted_keys.add(block)

        best: MeasuredValue | None = None
        for gene in unit_genes[unit_id]:
            protein = proteins.get(gene)
            if key.is_accession_keyed:
                accession = protein.accession if protein else None
                candidates = by_accession.get(accession or "", [])
            else:
                candidates = [
                    row
                    for code in reaction_ec_codes(reaction)
                    for row in by_ec.get(code, [])
                ]
            matched_by: str | None = None
            if key.requires_substrate_match:
                filtered = []
                for row in candidates:
                    by_structure = bool(
                        row.smiles and skeleton(row.smiles) in wanted_keys
                    )
                    by_name = bool(
                        row.substrate and row.substrate.lower() in wanted_names
                    )
                    if by_structure or by_name:
                        filtered.append(row)
                        matched_by = (
                            "inchikey_block" if by_structure else "substrate_name"
                        )
                candidates = filtered
            if not candidates:
                continue
            resolved = resolve_parameter(
                candidates,
                kind,
                uniprot=(protein.accession if protein else "") or "",
                target_temperature_c=target_temperature_c,
            )
            if resolved is None:
                continue
            row = next(
                (
                    c
                    for c in candidates
                    if c.substrate == resolved.substrate
                    and c.value_for(kind) is not None
                    and abs(float(c.value_for(kind)) - resolved.value) < 1e-9
                ),
                None,
            )
            tier = (
                Tier.MEASURED_PROTEIN
                if key.is_accession_keyed
                else Tier.MEASURED_EC_SAME_ORGANISM
                if organism is not None
                else Tier.MEASURED_EC_HOMOLOG
            )
            candidate = MeasuredValue(
                unit_id=unit_id,
                gene_id=gene,
                value=float(resolved.value),
                unit=resolved.unit,
                key=key,
                tier=tier,
                organism=row.organism if row is not None else organism,
                substrate_assayed=resolved.substrate,
                substrate_matched_by=matched_by,
                temperature_c=resolved.temperature_c,
                pubmed_id=resolved.pubmed_id,
                n_candidates=resolved.n_candidates,
                selection_rule=resolved.selection_rule,
            )
            if best is None:
                best = candidate
            elif kind is KineticKind.KCAT:
                best = min(best, candidate, key=lambda c: c.value)
            else:
                best = max(best, candidate, key=lambda c: c.value)
        if best is not None:
            out[unit_id] = best
    return out


# =====================================================================================
# STAGE 4: PREDICTED, AND THE WHOLE RUN
# =====================================================================================


class PredictorTable(BaseModel):
    """One predictor's output for one parameter, as a file the run reads."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str


class KineticsPipelineConfig(BaseModel):
    r"""Everything one run of :func:`resolve_kinetics` needs, and nothing organism-specific.

    The only required inputs are a GEM and a proteome. Every other stage is optional and
    reports its own absence: with no kinetics mirror there is no measured tier, with no
    predictor tables there is no predicted tier, and the run still emits a full blocker
    accounting over the catalytic units.
    """

    model_config = ConfigDict(extra="forbid")

    sbml_path: str
    proteome_path: str
    accessions_path: str | None = None

    #: Metabolite-name -> SMILES table, if the GEM ships one.
    name_table_path: str | None = None
    #: Structure table plus cross-reference, MetaNetX in the reference build.
    chem_prop_path: str | None = None
    chem_xref_path: str | None = None
    #: Structures written by an external builder, keyed by metabolite id.
    supplied_structures_path: str | None = None

    #: Kinetics database mirror, read by ``load_mirrored_records``.
    measured_mirror_dir: str | None = None
    measured_key: MeasuredKey = MeasuredKey.UNIPROT_SUBSTRATE
    organism: str | None = None
    target_temperature_c: float = 30.0

    #: A SECOND database stage, applied only where the first finds nothing. This is how the
    #: four-tier cascade is expressed: an accession-keyed measurement of this organism's
    #: protein first, then a cross-organism EC-plus-substrate measurement of the same
    #: chemistry, then predictions, then the constant. The two are tagged apart and never
    #: merged, and the homolog stage reads its own mirror because it is not organism
    #: filtered.
    homolog_key: MeasuredKey | None = None
    homolog_mirror_dir: str | None = None

    #: parameter -> list of (predictor name, parquet path).
    predictor_tables: dict[str, list[PredictorTable]] = Field(default_factory=dict)
    #: Names admitted for each parameter. When a within-class study result is given
    #: instead, the names are read from it, so another organism re-decides the set from
    #: its own statistics.
    selected_predictors: dict[str, list[str]] = Field(default_factory=dict)
    plausibility_json: str | None = None

    parameters: list[str] = Field(default_factory=lambda: ["k_cat", "K_M"])
    #: Used only where a parameter has no measurement anywhere, i.e. an organism with an
    #: empty database slice. Where the database supplies anything, the default is the
    #: median of what it supplied and this value is not read.
    fallback_defaults: dict[str, float] = Field(
        default_factory=lambda: {"k_cat": 13.7, "K_M": 1.0}
    )


class UnitParameter(BaseModel):
    """The resolved value for one catalytic unit and one parameter, with its tier."""

    model_config = ConfigDict(extra="forbid")

    unit_id: int
    reaction_id: str
    gene_ids: list[str]
    value: float
    tier: Tier
    #: Predictors that contributed, when the tier is predicted.
    predictors: list[str] = Field(default_factory=list)
    measured: MeasuredValue | None = None
    blocker: Blocker | None = None
    blocker_detail: str | None = None


class ParameterResult(BaseModel):
    """One parameter's table, its tier counts and its blocker counts."""

    model_config = ConfigDict(extra="forbid")

    parameter: str
    unit: str
    selected_predictors: list[str]
    default_value: float
    values: list[UnitParameter]

    def tier_counts(self) -> dict[str, int]:
        """Units per tier, with the three measured tiers kept apart."""
        return dict(Counter(str(v.tier) for v in self.values).most_common())

    def coarse_tier_counts(self) -> dict[str, int]:
        """The three-way split a coverage report prints, measured tiers merged."""
        counts = {"measured": 0, "predicted": 0, "default": 0}
        for value in self.values:
            if value.tier.is_measured:
                counts["measured"] += 1
            elif value.tier is Tier.PREDICTED:
                counts["predicted"] += 1
            else:
                counts["default"] += 1
        return counts

    def blocker_counts(self) -> dict[str, int]:
        """How many default-tier units each blocker accounts for."""
        return dict(
            Counter(
                str(v.blocker) for v in self.values if v.blocker is not None
            ).most_common()
        )


class KineticsPipelineResult(BaseModel):
    """A whole run: the tables, the accounting, and the hash of every file read."""

    model_config = ConfigDict(extra="forbid")

    built_at: str
    config: KineticsPipelineConfig
    n_catalytic_units: int
    n_reactions: int
    n_metabolites: int
    n_genes_in_proteome: int
    n_substrates_resolved: int
    structures_by_route: dict[str, int]
    input_blockers: dict[str, int]
    sources: dict[str, str]
    parameters: dict[str, ParameterResult]


def resolve_kinetics(config: KineticsPipelineConfig) -> KineticsPipelineResult:
    """Run every stage and return one record: tables, tiers, blockers, provenance.

    This is the whole portable path. Nothing in it names an organism, a gene, a metabolite
    or a predictor: the GEM supplies the GPR and the annotations, the proteome supplies the
    sequences, the config supplies the mirrors, and the admitted predictor set is either
    given or read from a within-class study's own result file.
    """
    import cobra
    import numpy as np

    from torchcell.metabolism.constraints import parse_catalytic_units
    from torchcell.metabolism.enzyme_kinetics import KineticKind, load_mirrored_records
    from torchcell.metabolism.parameters import (
        PredictorSelection,
        _per_unit_from_predictor,
    )

    model = cobra.io.read_sbml_model(config.sbml_path)
    units = parse_catalytic_units(model, [r.id for r in model.reactions])
    proteins = read_proteome(config.proteome_path, config.accessions_path)

    sources: dict[str, str] = {
        config.sbml_path: sha256_of(config.sbml_path),
        config.proteome_path: sha256_of(config.proteome_path),
    }
    name_table: dict[str, str] = {}
    if config.name_table_path:
        sources[config.name_table_path] = sha256_of(config.name_table_path)
        with open(config.name_table_path) as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
                    name_table.setdefault(parts[0].strip().lower(), parts[1].strip())
    index = None
    if config.chem_prop_path and config.chem_xref_path:
        index = CrossReferenceStructureIndex.from_metanetx(
            config.chem_prop_path,
            config.chem_xref_path,
            wanted_xrefs_for(list(model.metabolites)),
        )
        sources.update(index.source_paths)
    supplied: dict[str, str] = {}
    if config.supplied_structures_path:
        supplied = read_supplied_structures(config.supplied_structures_path)
        sources[config.supplied_structures_path] = sha256_of(
            config.supplied_structures_path
        )
    resolver = SubstrateStructureResolver(
        name_table=name_table, index=index, supplied=supplied
    )

    inputs = build_unit_inputs(model, units, proteins, resolver)
    blocked_by_unit = {b.unit_id: b for b in inputs.blocked}
    unit_genes: dict[int, list[str]] = defaultdict(list)
    for unit, gene in units.unit_gene_index.t().tolist():
        unit_genes[int(unit)].append(units.gene_ids[int(gene)])

    records: list[Any] = []
    if config.measured_mirror_dir:
        records = load_mirrored_records(config.measured_mirror_dir)
        sources[osp.join(config.measured_mirror_dir, "oed_records.json")] = sha256_of(
            osp.join(config.measured_mirror_dir, "oed_records.json")
        )
    homolog_records: list[Any] = []
    if config.homolog_mirror_dir:
        homolog_records = load_mirrored_records(config.homolog_mirror_dir)
        sources[osp.join(config.homolog_mirror_dir, "oed_records.json")] = sha256_of(
            osp.join(config.homolog_mirror_dir, "oed_records.json")
        )

    results: dict[str, ParameterResult] = {}
    for parameter in config.parameters:
        kind = KineticKind.KCAT if parameter == "k_cat" else KineticKind.KM
        measured = (
            match_measured(
                records,
                model,
                units,
                proteins,
                inputs.structures,
                kind,
                config.measured_key,
                organism=config.organism,
                target_temperature_c=config.target_temperature_c,
            )
            if records
            else {}
        )
        if config.homolog_key is not None and homolog_records:
            homolog = match_measured(
                homolog_records,
                model,
                units,
                proteins,
                inputs.structures,
                kind,
                config.homolog_key,
                organism=None,
                target_temperature_c=config.target_temperature_c,
            )
            # The primary stage wins wherever it has an answer: a measurement of this
            # organism's own protein is never displaced by one of a homolog.
            measured = {**homolog, **measured}

        selected = list(config.selected_predictors.get(parameter, []))
        if not selected and config.plausibility_json:
            selection = PredictorSelection.from_study(
                parameter, config.plausibility_json
            )
            selected = selection.selected
            sources[config.plausibility_json] = sha256_of(config.plausibility_json)
        tables = {t.name: t.path for t in config.predictor_tables.get(parameter, [])}
        stacked = []
        used: list[str] = []
        for name in selected:
            path = tables.get(name)
            if path is None:
                continue
            per_unit, _ = _per_unit_from_predictor(path, parameter, units.n_units)
            stacked.append(per_unit)
            used.append(name)
            sources[path] = sha256_of(path)
        if stacked:
            matrix = np.vstack(stacked)
            n_predictors = np.isfinite(matrix).sum(axis=0)
            predicted = np.full(units.n_units, np.nan)
            hit = n_predictors > 0
            predicted[hit] = np.power(10.0, np.nanmedian(matrix[:, hit], axis=0))
        else:
            predicted = np.full(units.n_units, np.nan)
            n_predictors = np.zeros(units.n_units, dtype=int)

        # ONE DEFINITION OF THE ORGANISM DEFAULT, shared with
        # `torchcell.metabolism.parameters._organism_default`: the median of the values the
        # DATABASE TIERS placed in the table, per catalytic unit. Per-unit rather than
        # per-gene, because the table's entries are per unit and a per-gene median silently
        # reweights complexes; measured-and-homolog rather than measured alone, because the
        # default's job is to be on the right scale for the reactions that have no number.
        observed = [m.value for m in measured.values()]
        default = (
            float(np.median(observed))
            if observed
            else config.fallback_defaults[parameter]
        )

        values: list[UnitParameter] = []
        for unit_id in range(units.n_units):
            reaction = model.reactions[int(units.unit_reaction[unit_id])]
            block = blocked_by_unit.get(unit_id)
            if unit_id in measured:
                hit = measured[unit_id]
                values.append(
                    UnitParameter(
                        unit_id=unit_id,
                        reaction_id=reaction.id,
                        gene_ids=unit_genes[unit_id],
                        value=hit.value,
                        tier=hit.tier,
                        measured=hit,
                    )
                )
            elif np.isfinite(predicted[unit_id]):
                values.append(
                    UnitParameter(
                        unit_id=unit_id,
                        reaction_id=reaction.id,
                        gene_ids=unit_genes[unit_id],
                        value=float(predicted[unit_id]),
                        tier=Tier.PREDICTED,
                        predictors=used,
                    )
                )
            else:
                values.append(
                    UnitParameter(
                        unit_id=unit_id,
                        reaction_id=reaction.id,
                        gene_ids=unit_genes[unit_id],
                        value=default,
                        tier=Tier.DEFAULT,
                        blocker=block.blocker if block else Blocker.PREDICTOR_FAILED,
                        blocker_detail=(
                            block.detail
                            if block
                            else "inputs resolved but no admitted predictor emitted a value"
                        ),
                    )
                )
        results[parameter] = ParameterResult(
            parameter=parameter,
            unit="1/s" if parameter == "k_cat" else "mM",
            selected_predictors=used,
            default_value=default,
            values=values,
        )

    return KineticsPipelineResult(
        built_at=_now(),
        config=config,
        n_catalytic_units=units.n_units,
        n_reactions=len(model.reactions),
        n_metabolites=len(model.metabolites),
        n_genes_in_proteome=len(proteins),
        n_substrates_resolved=len(inputs.structures),
        structures_by_route=dict(
            Counter(str(s.route) for s in inputs.structures.values()).most_common()
        ),
        input_blockers=inputs.blocked_counts(),
        sources=sources,
        parameters=results,
    )


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def main() -> None:
    """CLI: one JSON config in, one JSON result out.

    ::

        python -m torchcell.metabolism.kinetics_pipeline \
            --config kinetics_pipeline.json --out result.json
    """
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="KineticsPipelineConfig JSON.")
    parser.add_argument("--out", required=True, help="Where to write the result JSON.")
    arguments = parser.parse_args()

    with open(arguments.config) as handle:
        config = KineticsPipelineConfig.model_validate_json(handle.read())
    result = resolve_kinetics(config)
    with open(arguments.out, "w") as handle:
        handle.write(result.model_dump_json(indent=2))
    for parameter, block in result.parameters.items():
        print(parameter, block.coarse_tier_counts(), block.tier_counts())
    print("input blockers", result.input_blockers)
    print(f"-> {arguments.out}")


if __name__ == "__main__":
    main()
