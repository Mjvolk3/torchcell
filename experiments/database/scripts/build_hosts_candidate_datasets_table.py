# experiments/database/scripts/build_hosts_candidate_datasets_table.py
# [[experiments.database.expansion-hosts]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_hosts_candidate_datasets_table
r"""The cross-host candidate list: *P. kudriavzevii*, the pseudomonads and yeast regulatory DNA.

The third sibling of ``build_candidate_datasets_table.py`` (the 200-dataset
*S. cerevisiae* pass) and ``build_bacteria_candidate_datasets_table.py`` (the fifty
*E. coli* and *P. putida* rows). This is the CURATION for the cross-host wave, held as
data: nothing here can be recomputed from a store, so the judgment itself is the
artifact and it lives in a committed script.

Every row traces to one report on disk, ``Pk and next dataset wave triage.md``, and to
the six research notes beside it. The report's own verification labels are carried
through verbatim in ``verified``: **V** read on the source in session, **S** a search
snippet only, **I** inferred, which includes every count that is the product of two
verified numbers. A count is never promoted from one class to a stronger one, and
``instances_basis`` says separately whether the instance figure is reported, a product
of the two axes, or an order-of-magnitude estimate.

ORDERING RULE. The yeast script's rule, unchanged: rows sort by MEASUREMENTS, meaning
instances times phenotype dimensionality, descending, inside a tier. Ranking on
instances alone silently prefers a scalar fitness screen over a vector-valued omics
panel of the same size. A BAND is applied before tier, and the bands here are this
wave's, not the yeast pass's:

  regulatory DNA   sequence-to-readout libraries. The only yeast axis still orders of
                   magnitude from exhausted, and the one place this wave adds
                   measurements rather than rows: three rows carry more measurements
                   than everything else in the triage combined.
  cross-host fitness
                   the Fitness Browser rows, one per organism, which deliver the ROW
                   count. They share one artifact, one schema, one condition vocabulary
                   and one identifier bridge, so the marginal cost of the tenth row is
                   a genome registration.
  host bootstrap   a row whose purpose is to give a low-data host its first labels.
                   *P. kudriavzevii* has no genome-wide screen of any kind, so its
                   largest benchmark is 128 MIC values and scale ranking buries every
                   one of its rows below every pseudomonad screen. The band is the
                   honest mechanism, because the alternative is a hand-tuned weight
                   that hides the same decision inside a score.
  scale            everything else, ranked as before.

PER-HOST FLOOR. ``QUOTA_1`` is a floor inside the first tranche, not a split: without
it the recommended fifty is nearly all yeast and pseudomonad, because the measurement
rule ranks a 32-isolate MIC table below every barcoded screen. It is applied AFTER
ranking and every lift is reported row by row, the way the bacteria script reports its
isoprenol and host lifts, so the printed order can be reconciled against the pure
measurement order.

OWNER PIN. ``OWNER_PINS`` places a named row at a stated rank after the floor has run,
and overrides the ordering rule on purpose. It is the bacterial pass's pin mechanism
with one row in it: a stakeholder priority is a real criterion, so it is applied as a
reported move with its before and after rank, never as a tier or band edit that would
make the row look as though it earned the position by measurements.

Emits, off the same records:
  - notes-tex/database/database-expansion-hosts/tables/final.tex       (the recommended fifty plus ten)
  - notes-tex/database/database-expansion-hosts/tables/sources.tex     (citation, link, reason, data)
  - notes-tex/database/database-expansion-hosts/tables/candidates.tex  (the whole pool, ranked)
  - notes-tex/database/database-expansion-hosts/tables/sources_all.tex (sources for the whole pool)
  - notes-tex/database/database-expansion-hosts/tables/counts.tex      (per class and per host)
  - notes-tex/database/database-expansion-hosts/tables/summary.tex     (the recommended set)
  - notes-tex/database/database-expansion-hosts/tables/excluded.tex    (considered and dropped)
  - notes-tex/database/database-expansion-hosts/tables/asks.tex        (rows gated on a request)
  - notes-tex/database/database-expansion-hosts/tables/namespaces.tex  (gene-id namespaces per host)
  - notes-tex/database/database-expansion-hosts/tables/schema.tex      (what the wave changes)
  - notes-tex/database/database-expansion-hosts/tables/verdicts.tex    (candidate-gate verdicts)
  - <results>/candidates/hosts_candidate_datasets.json                 (machine-readable dump)

Run from the repo root:
  python experiments/database/scripts/build_hosts_candidate_datasets_table.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from torchcell.candidates.store import verdicts_by_row

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
# notes-tex documents live one group below notes-tex/ (notes-tex/README.md).
TEX_DIR = REPO / "notes-tex" / "database" / "database-expansion-hosts" / "tables"
JSON_OUT = RESULTS / "candidates" / "hosts_candidate_datasets.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_hosts_candidate_datasets_table.py"
)

# The name this table is registered under in torchcell/candidates/cli.py::TABLE_SCRIPTS,
# and the name render_verdicts reads the verdict store with. No stored verdict carries
# it: verdict.CandidateTable is Literal["bacteria", "yeast"] and widening it is a policy
# decision, since it is the `table` field of all 52 stored verdicts. So every row here
# prints "no verdict" until that decision is made, which is the honest state and not a
# failure.
TABLE = "hosts"

# ADDITIONS is what this wave adds; it is not a quota that has to be filled. The pool is
# shorter than this, and the shortfall is reported rather than padded: the report's
# finding is that the row count is reachable only through the Fitness Browser and the
# pseudomonad literature, and inventing rows to reach 200 would bury exactly that.
ADDITIONS = 200
TARGET_COUNT = ADDITIONS
CUT = ADDITIONS

# The two nearer lines a build week is planned against, as in the yeast script.
WAVE_1 = 50
WAVE_2 = 70
# Rows rendered in the final table: the recommended fifty plus ten extra, so a row that
# proves unreachable has a named replacement.
FINAL = 60
# The first tranche the per-host floor applies to is the recommended fifty.
TRANCHE_1 = WAVE_1

# Per-host FLOOR inside the first tranche, not a split. Eight rows each for the two
# hosts the wave exists to bootstrap. P. kudriavzevii cannot win a measurement contest
# against a pseudomonad fitness compendium under any ordering, and P. putida's omics and
# fitness rows lose to the yeast MPRA libraries by four orders of magnitude, so without
# a floor the first tranche is neither. A floor rather than an even split because an
# even split would fight the ordering rule: the three literatures are not the same size.
# Every lift is reported with its before and after rank.
QUOTA_1 = {"P. kudriavzevii": 8, "P. putida": 8}

# Owner pins: (row name, rank, reason). Applied last, in the order written, and each
# one is reported as a move. The reason is what the sources tables print in place of
# the row's band reason, so a pinned position is never read as a measurement result.
DOUGLASS_MIC = "Douglass 2018 (32 sequenced isolates, 4-antifungal MIC)"
OWNER_PINS: tuple[tuple[str, int, str], ...] = (
    (
        DOUGLASS_MIC,
        1,
        "Owner pin, 2026-11-24, to rank 1 from the host bootstrap band: it is the one "
        "P. kudriavzevii growth-type panel that can be fetched today with no ask, and "
        "fluconazole is the one compound measured on natural isolates in both yeasts, "
        "against Peter 2018 at 20 ug/ml on 971 S. cerevisiae isolates and Bloom 2019 "
        "at 100 uM on 13,950 segregants, so it is the first cross-host transfer test "
        "that has labels at both ends. The other three drugs are measured in "
        "S. cerevisiae only as deletion or CRISPRi chemogenomics, in Parsons 2006, "
        "Hillenmeyer 2008, and Piotrowski 2017.",
    ),
)
PIN_REASON: dict[str, str] = {name: why for name, _, why in OWNER_PINS}

# ---------------------------------------------------------------------------
# Vocabulary. Defined here so it is defined before use in the document, and so a
# typo becomes a validation error rather than a silently novel category.
# ---------------------------------------------------------------------------

# Only the hosts rows actually use. The Fitness Browser's GTDB-style renames are not
# separate members: a row carries the historical binomial, which is the name the
# released artifact and the published literature use, and the current binomial is a
# per-row alias in the row's own text.
Organism = Literal[
    "P. kudriavzevii",
    "P. putida",
    "S. cerevisiae",
    "E. coli",
    "P. fluorescens",
    "P. simiae",
    "S. stutzeri",
    "P. aeruginosa",
    "P. syringae",
]

Klass = Literal[
    "Regulatory DNA",
    "Fitness / chemical genomics",
    "Transposon fitness",
    "CRISPR library screen",
    "Essentiality",
    "Natural variation",
    "Tolerance / robustness",
    "Transcriptome",
    "Proteome",
    "Metabolome / flux",
    "Deep mutational scan",
    "Modality / backbone",
    "Engineering parts",
    "Production campaign",
]

# How the total genomic content of one strain would be reconstructed. The same hard gate
# both siblings apply: a row with no route to a sequence cannot train a
# genotype-to-phenotype model and is excluded rather than ranked low. The bases are named
# per host because the reference differs, and because a JGI gNNNN model id, an INSDC
# SD108 locus tag, a PP_ tag and a PA number are different namespaces that must not be
# silently merged. No yeast-shaped value is reused for a bacterial or a Pichia row.
SeqBasis = Literal[
    # -- S. cerevisiae ------------------------------------------------------
    "S288C+reporter-locus",  # reference plus a designed regulatory sequence driving a reporter
    "S288C+designed-edit",  # designed SNV/indel library; designed, not per-strain verified
    "S288C+integration-locus",  # reference plus one cassette at a varied, named coordinate
    "S288C-KO",  # reference minus one cataloged ORF
    # A mapped transposon insertion in an S288C-derived background: the saturation-screen
    # basis, where the perturbation is an insertion site rather than a cataloged deletion.
    "S288C+transposon",
    "isolate-WGS",  # per-isolate assembly or VCF published
    # A reporter library carried on a plasmid rather than integrated: the MPRA basis
    # where the measured sequence never enters the chromosome. Distinct from
    # "S288C+reporter-locus" because the genotype is the total genomic content of the
    # cell, and a plasmid that is present is part of it.
    "plasmid-reporter-library",
    # -- P. kudriavzevii ----------------------------------------------------
    "SD108-reference",  # the CABBI/JGI SD108 background, unedited
    "SD108+designed-edit",  # SD108 plus named deletions or integrations
    "SD108+guide",  # SD108 plus a cassette and a guide target; the genome is unedited
    "CBS573-reference",  # the clinical/type-strain background, unedited
    # A named P. kudriavzevii reference strain that the release does not identify. Kept
    # distinct from the two named backgrounds because this species has one name and four
    # live annotations, so an unnamed strain is not the same claim as a named one.
    "Pk-reference",
    "Pk-isolate-WGS",  # per-isolate Pichia assembly, VCF or CNV calls published
    # The SECOND production lineage of this species, plus designed edits. Two groups
    # build on two different parents: SD108 (the PBY, 1AP and 1APX series) and
    # NRRL Y-27441 (the IoDY01 and IoDY01H series), with CNB1 from CD1822 a third in the
    # patents. Separate from "Pk-reference" because the strain IS named; what is missing
    # is a reference assembly in the genomes tier, so the gap is a pin, not an identity.
    "Pk-named+designed-edit",
    # -- E. coli ------------------------------------------------------------
    # A chromosomal reporter library: the measured regulatory sequence sits at a named
    # landing pad in the MG1655 chromosome, so it is part of the cell's genomic content
    # rather than carried on a plasmid.
    "MG1655+reporter-locus",
    # A coding-sequence variant library on a plasmid, under selection on the protein's
    # own function. Separate from "plasmid-reporter-library" because the measured
    # sequence IS the gene being selected, not a regulatory element driving a reporter.
    "plasmid-coding-variant",
    # A designed chromosomal edit in a named E. coli background that is not a deletion:
    # an amino-acid substitution in a chromosomal gene, which no KO basis describes.
    "ecoli+designed-edit",
    # A per-isolate E. coli assembly or reads deposited, the basis every natural-isolate
    # and clinical-surveillance panel needs.
    "ecoli-isolate-WGS",
    # A cataloged single-gene deletion whose collection background the release names
    # (Keio BW25113, Yale CGSC). Deliberately not split per collection: a screen that
    # draws from both names both, and the b-number is the join either way.
    "ecoli-KO",
    # -- pseudomonads -------------------------------------------------------
    "KT2440-reference",
    "KT2440+transposon",  # mapped, barcoded insertion site
    "KT2440+guide",
    "KT2440-KO",  # a named chromosomal deletion in the reference background
    "PAO1-reference",
    "PAO1+transposon",
    "PAO1+guide",
    "PA14+transposon",
    "pseudomonad-isolate-WGS",  # per-isolate pseudomonad assembly or reads deposited
    # The Fitness Browser basis: a barcoded transposon insertion in one of the browser's
    # own organisms, whose locusId resolves to a RefSeq accession through the release's
    # refseq_mapping.tsv. Separate from "KT2440+transposon" because the reference is the
    # browser's assembly pin for that organism, not the KT2440 reference.
    "fitbrowser+transposon",
    "pseudomonad-reference",  # a named pseudomonad reference strain, unedited
    "engineered-chassis",  # named production strain plus heterologous cassettes
    # The chassis plus a guide array: the pathway cassettes are heterologous and the
    # perturbation on top of them leaves the genome unedited. Distinct from
    # "KT2440+guide", where the host is the bare reference.
    "engineered-chassis+guide",
    "evolved-WGS",  # a resequenced evolved clone; unsequenced populations are excluded
]

Basis = Literal["reported", "product", "estimate"]

# The report's own verification labels, carried through rather than restated. A count is
# never promoted: a product of two verified numbers stays "inferred", because the product
# is an assumption about completeness that the source does not state.
#   verified   read on the source in session
#   snippet    a search snippet only; the source page itself was not opened
#   inferred   derived, including every count that is a product of two verified numbers
Verified = Literal["verified", "snippet", "inferred"]

# Ingestion state. "built" leaves the ranking and is reported as such: the row stays in
# the pool so the judgment that it is an overlap rather than a candidate is recorded.
Status = Literal["candidate", "blocked", "built", "aggregation"]

# How well a row's numbers and citation were checked.
Confidence = Literal["sourced", "recall"]

# How the per-record values are obtained, which is the axis that decides what the
# campaign does first.
#   retrievable   programmatic: a plain HTTP fetch, a repository API, or a git clone
#   manual_fetch  needs a human with a browser: a 403 to every automated client, a
#                 publisher supplement, or a JavaScript-rendered deposit page
#   ask           needs a request to a named person or group; nothing public exists
Retrievability = Literal["retrievable", "manual_fetch", "ask"]

# Priority band, applied BEFORE tier and scale. See the module docstring for why each
# exists; "scale" is the default and the right default.
Band = Literal["host bootstrap", "regulatory DNA", "cross-host fitness", "scale"]

# Band order, and it is a decision rather than a score. Regulatory DNA leads because it
# is where the wave adds measurements, cross-host fitness follows because it is where the
# wave adds rows, and host bootstrap sits third rather than first on purpose: the band
# holds 33 rows, so leading with it would hand two thirds of the recommended fifty to one
# low-data host and leave the fitness band, which the row count depends on, entirely
# below the line. Third place still lifts every one of its rows above all 73 scale rows,
# which is what the band is for, and QUOTA_1 is the floor that makes the guarantee
# explicit rather than incidental.
BAND_ORDER: dict[str, int] = {
    "regulatory DNA": 0,
    "cross-host fitness": 1,
    "host bootstrap": 2,
    "scale": 3,
}

BAND_LETTER: dict[str, str] = {
    "host bootstrap": "H",
    "regulatory DNA": "R",
    "cross-host fitness": "F",
    "scale": "S",
}


class Candidate(BaseModel):
    """One dataset proposed for ingestion in the cross-host wave."""

    model_config = {"frozen": True}

    name: str
    organism: Organism
    citation: str
    url: str
    klass: Klass
    tier: int = Field(ge=1, le=4)
    genotypes_n: int | None
    genotypes: str
    env_n: int | None
    env: str
    instances_n: int | None
    instances_basis: Basis
    phenotype: str
    shape: str
    dim: int = 1  # phenotype vector length; 1 for a scalar label
    seq_basis: SeqBasis
    why: str
    accession: str
    # Which annotation the RELEASED gene ids speak. A required field rather than a
    # remark, because one species in this wave has four live namespaces whose model
    # counts differ by 2,182 genes, so "the gene id" is not a well-formed phrase until
    # the namespace is named. Empty only for a row whose readout is not gene-indexed.
    gene_namespace: str
    retrievability: Retrievability = "retrievable"
    # The named person or group when retrievability is "ask", else empty. Validated
    # against each other in _check_rows, so an "ask" row cannot be anonymous.
    ask_of: str = ""
    # The schema class this row needs that the served schema does not hold, or empty.
    new_class: str = ""
    verified: Verified = "verified"
    status: Status = "candidate"
    confidence: Confidence = "sourced"
    band: Band = "scale"
    band_why: str = ""  # why this row is out of the scale band; required when it is
    time_axis: str = ""

    @property
    def measurements(self) -> int | None:
        """Instances times phenotype dimensionality.

        Ranking on instances alone silently prefers a scalar-fitness screen over a
        vector-valued omics panel of the same size: 414 isolate transcriptomes is 414
        instances but roughly 2.4 million numbers.
        """
        return None if self.instances_n is None else self.instances_n * self.dim

    @property
    def scale_key(self) -> tuple[int, float]:
        """Tier, then measurements descending: the yeast script's ordering."""
        return (self.tier, -math.log10(max(self.measurements or 1, 1)))

    @property
    def sort_key(self) -> tuple[int, int, float]:
        """Band first, then the scale key inside it."""
        return (BAND_ORDER[self.band],) + self.scale_key


class Excluded(BaseModel):
    """A dataset considered and dropped, with the rule that dropped it."""

    model_config = {"frozen": True}

    name: str
    reason: str
    rule: Literal[
        # No route to the strain's genomic content, so there is no genotype at all.
        "no-sequence",
        # A real dataset in an organism outside this wave's hosts.
        "off-host",
        # A model, a reconstruction, a review, a design program or a prior over the
        # genome: not a genotype-by-environment measurement.
        "not-a-dataset",
        # A real experiment whose released form is a figure or a summary rather than
        # per-record values.
        "no-per-record-data",
        # Already served, or already in the raw mirror, so it is an overlap.
        "already-built",
        # Re-serves or re-processes another dataset's values; not net new.
        "derivative",
    ]


class Ask(BaseModel):
    """A row whose per-record values exist only behind a request to a named party.

    ``request`` is what to ask for, worded so it can be sent as written; ``unlocks``
    is what the campaign gains if the answer is yes. Separated from the ranked rows
    because an ask is an action with an owner, and a backlog loses those silently.
    """

    model_config = {"frozen": True}

    dataset: str
    organism: Organism | None  # None when the ask spans hosts
    who: str
    request: str
    unlocks: str


class Namespace(BaseModel):
    """One gene-identifier namespace a host's released data speaks.

    ``carries`` is the model or gene count the namespace itself holds, and ``joins`` is
    what maps it to another namespace, or the statement that nothing does. A crosswalk
    that has to be built by protein BLAST is not a join that exists today, and this
    field says so rather than implying one.
    """

    model_config = {"frozen": True}

    organism: Organism
    namespace: str
    spelling: str
    carries: str
    joins: str


class SchemaNeed(BaseModel):
    """One schema or candidate-machinery change the wave requires.

    ``symbol`` is the exact module path and symbol, so the change is addressable rather
    than described. ``policy`` is what the change triggers under the schema-change
    policy: a record, an impact report, a label, a rebuild scope.
    """

    model_config = {"frozen": True}

    symbol: str
    what: str
    policy: str
    additive: bool  # True when the change adds a class and moves no served closure


# ---------------------------------------------------------------------------
# What the wave changes. Listed so none of it surfaces late: the ranked list is
# unimplementable without the first two, and the last is the most expensive thing in
# the document.
# ---------------------------------------------------------------------------

SCHEMA_NEEDS: list[SchemaNeed] = [
    SchemaNeed(
        symbol="torchcell/datamodels/schema.py::Genotype (perturbation identity)",
        what=(
            "Admit a perturbation whose identity is NOT a gene. Eleven of the twenty "
            "largest yeast rows in this fill, and most of the E. coli regulatory-DNA "
            "rows, carry no gene identifier anywhere in their bulk data: the record key "
            "is a lineage barcode, a raw measured sequence, or a residue substitution "
            "string. A barcoded lineage is the hardest case, because the perturbation "
            "is UNKNOWN by construction and is identified only by its tag, which is a "
            "typed absence on the genotype rather than a missing field"
        ),
        policy=(
            "Served-vocabulary change. The honest route is a typed ProvenanceGap on the "
            "gene-identity field rather than a forced mapping to a gene, which would "
            "invent a locus the release never names. Schema-change record and impact "
            "report; additive if it lands as a new perturbation sibling, breaking if it "
            "loosens a required field on the served Genotype"
        ),
        additive=False,
    ),
    SchemaNeed(
        symbol="torchcell/datamodels/schema.py::SourceType",
        what=(
            "Add a patent member. The enum holds journal_article, dissertation, "
            "preliminary_report and in_house, and the identity rule for every "
            "non-journal member is title plus the deposited document's "
            "mirror-relative path and sha256. A patent fits that rule exactly and has "
            "no DOI, so the three ranked 3-HP patents can be deposited and hashed but "
            "not yet declared. Without the member the only honest alternatives are to "
            "drop eleven strains with titer and yield, or to file a patent as a "
            "journal article, which would be false"
        ),
        policy=(
            "Served-vocabulary change, but additive: no served record carries the new "
            "member, so the closure of every built dataset is unchanged. Schema-change "
            "record and impact report; no schema-breaking label and no rebuild, since "
            "nothing served reserializes"
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/data/experiment_dataset.py::Organism",
        what=(
            "Add a pkudriavzevii member. The enum holds exactly scerevisiae, ecoli and "
            "pputida today, and ALL_ORGANISMS is frozenset(Organism), so no loader can "
            "declare the host and dataset_adapter_map can hold nothing for it"
        ),
        policy=(
            "Served-vocabulary change: schema-change record, impact report, the served "
            "datasets named in the PR, and the schema-breaking label if a served "
            "closure changes. The genome side is already done, which is what makes "
            "this the whole cost: registry.py pins PKUDRIAVZEVII_SD108 = "
            "pkudriavzevii_SD108_v1 and gates.py already lists it in "
            "READABLE_ASSEMBLY_SETS with the comment that no host token names it yet."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/candidates/gates.py::HOSTS, _ORGANISM_OF_TABLE_ROW",
        what=(
            "Add HostTokens for SD108 and CBS573, and admit 'P. kudriavzevii' to the "
            "set of organisms a table row may name, which holds E. coli, P. putida and "
            "S. cerevisiae today"
        ),
        policy=(
            "Gate-code change, not a schema change. Until it lands hosts_named() "
            "raises ValueError on any P. kudriavzevii row, so G2 cannot even be "
            "measured and no verdict can be written for one. The eight pseudomonad "
            "hosts need the same treatment, one token per reference strain."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/data/experiment_dataset.py::Organism (pseudomonads)",
        what=(
            "Nine or ten further members for the Fitness Browser organisms and "
            "P. aeruginosa, plus the matching reference genomes in the genomes tier"
        ),
        policy=(
            "The same served-vocabulary policy, ten times over. Worth staging as one "
            "deliberate change rather than ten accidental ones, because the Fitness "
            "Browser rows arrive in one artifact and the marginal work per organism "
            "after the first is a genome registration."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/candidates/verdict.py::CandidateTable",
        what=(
            "A third literal for this pool. The type is Literal['bacteria', 'yeast'] "
            "and it is also the table field of every one of the 52 stored verdicts"
        ),
        policy=(
            "Record-model change on a frozen, extra-forbidding model, so it is a "
            "policy decision rather than a widening: until it lands, every row in "
            "Table 11 reads 'no verdict' and candidate-gate can read this pool's rows "
            "but cannot write a verdict for one. gates.py::CandidateRow.table and "
            "gates.py::TEXT_FIELDS carry the same literal and the same keys."
        ),
        additive=False,
    ),
    SchemaNeed(
        symbol="torchcell/candidates/cli.py::TABLE_SCRIPTS",
        what=(
            "Register this script as a third pool, so candidate-gate can load its rows "
            "by name the way it loads the bacteria and yeast pools"
        ),
        policy=(
            "CLI change only. Done. The gate subcommand's --table choices still admit "
            "two values, so passing this pool's name needs the CandidateTable decision "
            "above first."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="experiments/database/scripts/build_candidate_datasets_table.py::CANDIDATES",
        what=(
            "A separate pool rather than more rows in the yeast one. The yeast "
            "Candidate has no organism field at all, the bacteria one admits only "
            "E. coli and P. putida, and the yeast main() raises on a pool over 250 "
            "rows against 169 literals held"
        ),
        policy=(
            "Splitting by wave is forced by the guard, not merely tidier. This script "
            "is that split, and its own guard is 400 because the pool spans eight "
            "hosts."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="experiments/database/scripts/build_hosts_candidate_datasets_table.py::SeqBasis",
        what=(
            "Host-specific sequence bases. The yeast SeqBasis is eleven values, every "
            "one S288C- or yeast-panel-relative, so isolate-WGS transfers to a Pichia "
            "or pseudomonad isolate panel and an SD108 designed edit does not"
        ),
        policy=(
            "Script-level vocabulary, enforced as a validation error on any unlisted "
            "value. Declared explicitly per host so a bacterial row cannot be written "
            "against a yeast-shaped basis by accident."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/datamodels/schema.py (phenotype leaves)",
        what=(
            "New leaves for TSS initiation efficiency, antifungal and antibiotic MIC, "
            "Biolog respiration per well, plate-reader reporter ratio, and 13C flux"
        ),
        policy=(
            "By the placement rule a different technology is its own leaf with an "
            "explicit conversion, so a plate-reader GFP ratio and a sort-seq bin "
            "average are different leaves rather than one class with a note. Each is "
            "additive and admissible on its own and none weakens an existing required "
            "field."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/datamodels/schema.py (perturbation leaves)",
        what=(
            "New leaves for a 5' UTR variant, a 3' end variant, a promoter swap "
            "upstream of a native gene, a tRNA point substitution, and an amino-acid "
            "substitution for the ProteinGym subset"
        ),
        policy=(
            "Additive leaves, so each should clear the admission check on its own "
            "closure. In the served graph the perturbation subclasses collapse into "
            "one Perturbation node whose kind is a property, so a query naming every "
            "deletion names the kinds one by one."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/datamodels/schema.py (GenePerturbation namespaces)",
        what=(
            "Gene-identifier namespaces for a JGI gNNNN model id, an INSDC SD108 locus "
            "tag, a PA or PA14 number, a Bakta locus tag and a Fitness Browser locusId"
        ),
        policy=(
            "The same additive change the bacterial layer already made for b-numbers "
            "and PP_ tags. The genome never gates a record, so a released name the "
            "reference does not hold is a report-class verification finding and never "
            "a dropped row."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="torchcell/verification/sourced.py::SourcedValue",
        what=(
            "Extend it to the phenotype value and the perturbation. "
            "ontology_checks.sourced_value_fields() reports exactly nine fields that "
            "can carry one today, all of them media, strain background, culture format "
            "or essentiality, so a quoted n_samples, uncertainty type, unit or "
            "threshold has nowhere typed to be written"
        ),
        policy=(
            "The most expensive change here: it moves classes many served closures "
            "hold, so the record's rebuild_scope reads full_kg_rebuild and the PR "
            "carries the schema-breaking label. It is also the blocking change for an "
            "artifact-to-ontology pipeline, whose entire output is that class of value."
        ),
        additive=False,
    ),
    SchemaNeed(
        symbol="notes/paper.north-star.dataset-triage.md (guide-level exclusion)",
        what=(
            "An explicit decision on gRNA-library rows. The standing rule excludes a "
            "dataset whose primary label is per-guide activity or cleavage efficiency "
            "rather than a per-strain phenotype"
        ),
        policy=(
            "The landing-pad screen's cutting and integration efficiency fall under "
            "the rule while its fitness and GFP readouts do not, and the PAO1 and "
            "KT2440 CRISPRi libraries are guide-level by construction, where the guide "
            "sequence is part of the genotype. This needs a decision, not an inherited "
            "rule."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="experiments/database/scripts/build_candidate_datasets_table.py::EXCLUDED",
        what=(
            "Revisit the standing off-species exclusion of the K. phaffii screen and "
            "the Y1000+ panel, whose note reads 'Revisit if the generalization axis "
            "widens to other hosts'"
        ),
        policy=(
            "A cross-host wave is that widening, so the note needs a decision. The "
            "iIsor850 exclusion stands on its own merits under not-a-dataset: a "
            "reconstruction is not an experiment."
        ),
        additive=True,
    ),
    SchemaNeed(
        symbol="$DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018",
        what=(
            "Bring the key into the manifest before extending it. Six of its nine "
            "files are unhashed over 10.21 GB, and a parallel non-pydantic convention "
            "sits beside it: DEPOSIT.md, feba.db.sha256 and SHA256SUMS.txt, which no "
            "program reads"
        ),
        policy=(
            "A precondition of the cross-host fitness band rather than a schema "
            "change: those rows extend this key, and an ingested fitness value has to "
            "carry the release date and sha256 of the db file it came from, because the "
            "bulk page itself says the 2018 release's sucrose and D-mannitol data "
            "should be disregarded in favor of fresh stock data."
        ),
        additive=True,
    ),
]


# ---------------------------------------------------------------------------
# The gene-identifier namespaces. Four of them are live for one species, which is why
# this is a table rather than a remark.
# ---------------------------------------------------------------------------

NAMESPACES: list[Namespace] = [
    Namespace(
        organism="P. kudriavzevii",
        namespace="JGI Issorie2 gene models",
        spelling="gNNNN",
        carries="4,925 gene models",
        joins=(
            "Nothing published. A crosswalk to the INSDC set has to be built by protein "
            "BLAST, and 119 of 130 scaffolds are predicted to be an alternate "
            "haplotype, so the gap is not only an identifier gap"
        ),
    ),
    Namespace(
        organism="P. kudriavzevii",
        namespace="MAESTRO gene spelling",
        spelling="geneNNNN",
        carries="the same JGI model set under a different spelling",
        joins=(
            "Assumed equal to the JGI spelling and not verified: g4260 must be checked "
            "against gene4260 rather than assumed equal, which is exactly the pair the "
            "MAESTRO CNV finding rests on"
        ),
    ),
    Namespace(
        organism="P. kudriavzevii",
        namespace="INSDC / Ensembl Fungi SD108",
        spelling="locus tag",
        carries="7,107 coding genes",
        joins=(
            "The 4,925-versus-7,107 gap is large enough that the two sets are not "
            "trivially inter-mappable. This is the set the genomes tier pins as "
            "pkudriavzevii_SD108_v1"
        ),
    ),
    Namespace(
        organism="P. kudriavzevii",
        namespace="CBS573 / YGAP annotation",
        spelling="YGAP id",
        carries="the clinical-isolate annotation",
        joins=(
            "Used by the clinical work, including the MIC panel and the ERG11 and FKS1 "
            "variant calls, so a clinical row and an SD108 engineering row do not share "
            "a gene id at all"
        ),
    ),
    Namespace(
        organism="P. putida",
        namespace="KT2440 locus tags",
        spelling="PP_NNNN",
        carries="5,729 genes in the served reference",
        joins=(
            "Already served: the bacterial layer namespaces its five perturbation types "
            "by locus tag, so a PP_ row writes the record type the schema holds"
        ),
    ),
    Namespace(
        organism="P. putida",
        namespace="Fitness Browser locusId",
        spelling="orgId, locusId",
        carries="one row per gene per browser organism",
        joins=(
            "refseq_mapping.tsv maps org and locusId to a RefSeq accession and "
            "refseq_assemblies maps orgId to an NCBI assembly, so the identifier bridge "
            "is supplied by the source rather than built"
        ),
    ),
    Namespace(
        organism="P. aeruginosa",
        namespace="PAO1 and PA14 reference numbers",
        spelling="PA NNNN",
        carries="the two reference annotations the Tn-seq literature uses",
        joins=(
            "GOLD_84 carries PAO1 locus tags and GOLD_115 is PA14 in LB, so the two "
            "gene lists are in different namespaces and must not be merged into one "
            "boolean"
        ),
    ),
    Namespace(
        organism="P. aeruginosa",
        namespace="Bakta de novo annotation",
        spelling="per-isolate tag",
        carries="one annotation per newly sequenced isolate",
        joins=(
            "Nothing maps a Bakta tag to a PA number, so an isolate survey annotated "
            "this way forces the alias layer rather than joining on a reference id"
        ),
    ),
    Namespace(
        organism="S. cerevisiae",
        namespace="SGD S288C R64-4-1",
        spelling="YALNNNC",
        carries="6,607 genes in the shared vocabulary",
        joins=(
            "The served namespace. A regulatory-DNA row is not gene-indexed at all: its "
            "identifier is the measured sequence, which is why those rows carry a "
            "reporter-locus or plasmid basis rather than a gene id"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Candidates.
#
# TIER RULE, stated once so the ordering is reproducible rather than a matter of taste.
# Within a tier, rows sort by MEASUREMENTS (instances x phenotype dimensionality),
# descending.
#   1  clears all three bars: >=1e3 genotypes, >=1e4 instances, a clean sequence basis
#   2  clears two of the three; or is the ONLY dataset covering a host or a band
#   3  a modality or reference backbone the substrate needs, without itself carrying a
#      large genotype axis
#   4  real, but small n, a coarse label, heavy extraction overhead, or an unresolved
#      deposit
#
# ``verified`` labels the MEASUREMENT figure, which is the quantity rows are ranked on,
# and ``instances_basis`` labels the instance count separately. A row whose instance
# count was read on the source but whose measurement figure is a product of two numbers
# therefore reads instances_basis="reported" with verified="inferred", which is exactly
# the report's own pairing and is never collapsed into one label.
# ---------------------------------------------------------------------------

CANDIDATES: list[Candidate] = [
    # -- Regulatory DNA ----------------------------------------------------
    Candidate(
        name="de Boer 2020 (GPRA sort-seq, 100M random promoters)",
        organism="S. cerevisiae",
        citation="de Boer CG, Vaishnav ED, Sadeh R, Abeyta EL, Friedman N, Regev A. Nat Biotechnol 2020;38:56-65.",
        url="https://doi.org/10.1038/s41587-019-0315-8",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=102371025,
        genotypes="102,371,025 promoter sequences",
        env_n=3,
        env="YPD, YPGal, YPGly",
        instances_n=102371025,
        instances_basis="reported",
        phenotype="expression driven, 18-bin sort-seq YFP/RFP",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The largest sequence-to-phenotype resource in yeast by two orders of magnitude, and every genotype is an exactly known 80mer at a known position in a known scaffold. The perturbation space is continuous rather than a gene list, which is what a model of regulatory grammar needs and no deletion collection can supply.",
        accession="GEO GSE104878 and GSE104903; Addgene 127546; which accession holds which library is unresolved and one page check settles it",
        gene_namespace="none; the identifier is the measured 80mer, not a gene id",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: one row carrying more measurements than every other row in this wave combined, on a schema that already serves its child, so the admission is a download decision rather than a design one.",
        new_class="",
    ),
    Candidate(
        name="Vaishnav 2022 deferred training libraries",
        organism="S. cerevisiae",
        citation="Vaishnav ED, de Boer CG, Molinet J, Yassour M, Fan L, Adiconis X, Thompson DA, Levin JZ, Cubillos FA, Regev A. Nature 2022;603:455-463.",
        url="https://doi.org/10.1038/s41586-022-04506-6",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=51339035,
        genotypes="30,722,376 complex + 20,616,659 defined",
        env_n=2,
        env="YPD complex, SD-Ura defined",
        instances_n=51339035,
        instances_basis="reported",
        phenotype="expression driven, 18-bin sort-seq YFP",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The cleanest possible superset admission: the loader, the schema, the medium objects, the strain reasoning and the provenance record all exist, and the 71,158 served records are a proven subset, so the admission check can prove the superset by content id. The two files are 3.59 GB and 2.47 GB on Zenodo, which makes it a download decision rather than a schema one.",
        accession="Zenodo 4436477 (complex_media_training_data_Glu.txt 3.59 GB, defined_media_training_data_SC_Ura.txt 2.47 GB); GEO GSE163045 and GSE163866",
        gene_namespace="none; the identifier is the measured 80mer",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band, and the single lowest-effort large win in the wave: every piece of machinery it needs is already written and serving its subset.",
        new_class="",
    ),
    Candidate(
        name="DREAM 2022 random promoter challenge (Rafi 2024)",
        organism="S. cerevisiae",
        citation="Rafi AM, et al. A community effort to optimize sequence-based deep learning models of gene regulation. Nat Biotechnol 2024.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE254493",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=6810361,
        genotypes="6,739,258 train + 71,103 test",
        env_n=1,
        env="1 medium",
        instances_n=6810361,
        instances_basis="reported",
        phenotype="expression driven, FACS, about 100 cells per sequence",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The held-out benchmark every published yeast expression model is scored on, and the only count in this wave with two independent confirmations. It is a curated slice of the same GPRA platform rather than an independent experiment, so it needs an AggregationRecord at G1 and partial nesting with de Boer and Vaishnav is expected.",
        accession="GEO GSE254493; Zenodo 10633252 (v2, about 4.0 GB), superseding Zenodo 7395397",
        gene_namespace="none; the identifier is the measured 110 bp promoter",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the benchmark slice, which is what makes a model comparison possible at all, and it carries its own eight-category test design.",
        new_class="",
    ),
    Candidate(
        name="Zhu 2024 (Pol II MASTER, TSS initiation efficiency)",
        organism="S. cerevisiae",
        citation="Zhu C, Vvedenskaya IO, Sze SH, Nickels BE, Kaplan CD. Nat Struct Mol Biol 2024;31:190-202.",
        url="https://doi.org/10.1038/s41594-023-01171-9",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=1000000,
        genotypes="1,000,000 TSS sequences",
        env_n=1,
        env="defined promoter context, NTP levels varied",
        instances_n=1000000,
        instances_basis="reported",
        phenotype="transcription initiation efficiency per TSS",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="A million exactly specified TSS-window sequences read out mechanistically rather than through a reporter, deposited as both processed values and raw reads. The 2021 preprint of the same work describes about 80,000 variants, so the published figure is the one to pin and the discrepancy travels with the row.",
        accession="GEO GSE185290 (processed); BioProject PRJNA766624 (raw)",
        gene_namespace="none; the identifier is the measured TSS window",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: a different readout from reporter level, so it extends the axis rather than deepening it, and at a million records it is the third-largest row in the wave.",
        new_class="TSS initiation efficiency phenotype leaf",
    ),
    Candidate(
        name="Kotopka and Smolke 2020 (artificial promoter libraries)",
        organism="S. cerevisiae",
        citation="Kotopka BJ, Smolke CD. Model-driven generation of artificial yeast promoters. Nat Commun 2020;11:2113.",
        url="https://doi.org/10.1038/s41467-020-15977-4",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=1002000,
        genotypes="over 675,000 constitutive + over 327,000 inducible",
        env_n=2,
        env="constitutive and inducible arms",
        instances_n=1002000,
        instances_basis="reported",
        phenotype="promoter activity",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="A million-scale synthetic promoter library from a platform and a group independent of the de Boer lineage, so it is not nested with the three rows above, and it carries an inducible arm that the GPRA libraries lack. Both counts are lower bounds as the paper words them.",
        accession="Zenodo 3735426 (Promoter Design Data_2020-03.zip, about 1.1 GB)",
        gene_namespace="none; the identifier is the measured promoter",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the only million-scale promoter library outside the GPRA lineage, which is what makes it a check on it rather than more of it.",
        new_class="plate-reader reporter activity leaf, if the readout is not FACS-binned",
    ),
    Candidate(
        name="Cuperus 2017 (random 5' UTR library)",
        organism="S. cerevisiae",
        citation="Cuperus JT, Groves B, Kuchina A, Rosenberg AB, Jojic N, Fields S, Seelig G. Genome Res 2017;27:2015-2028.",
        url="https://genome.cshlp.org/content/27/12/2015",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=489348,
        genotypes="489,348 detected random 50-nt 5' UTRs",
        env_n=1,
        env="medium lacking histidine",
        instances_n=489348,
        instances_basis="reported",
        phenotype="competitive growth enrichment, HIS3",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Half a million exactly specified 5' UTR sequences, and it changes both axes at once: the sequence variable moves from the promoter to the 5' UTR and the readout moves from a sorted reporter to competitive growth. A separate leaf by the measurement rule, since growth and a bin average are different technologies.",
        accession="GEO GSE104252",
        gene_namespace="none; the identifier is the measured 5' UTR",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the only large 5' UTR library, and the only one in the band whose readout is growth rather than fluorescence.",
        new_class="5' UTR variant perturbation leaf; pooled competitive-growth reporter leaf",
    ),
    Candidate(
        name="Shalem 2015 (3' end FACS-seq library)",
        organism="S. cerevisiae",
        citation="Shalem O, et al. 3' end determinants. PLoS Genet 2015;11:e1005147.",
        url="https://doi.org/10.1371/journal.pgen.1005147",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=13000,
        genotypes="13,000 designed 3' ends, 102 bp variable",
        env_n=1,
        env="SC-URA + galactose, 30 C",
        instances_n=13000,
        instances_basis="reported",
        phenotype="YFP/mCherry ratio, 16-bin FACS-seq",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="Sequences and measured values sit in one XLSX, it extends coverage from the promoter to the terminator, and it is one of the eight benchmark tasks published models are scored on. Its only uncertainty figure is a library-wide 13.2 percent technical RSD rather than a per-construct error, and no raw reads were deposited anywhere.",
        accession="PLoS Genet S3 Table (XLSX), no accession; needs a supplement fetch and a sha256 pin",
        gene_namespace="none; the identifier is the measured 3' end",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the terminator end of the same axis, with the sequence and the value in one released file.",
        new_class="3' end variant perturbation leaf",
    ),
    Candidate(
        name="Lubliner 2015 (core promoter variant panel)",
        organism="S. cerevisiae",
        citation="Lubliner S, Keren L, Segal E, et al. Core promoter sequence in yeast is a major determinant of expression level. Genome Res 2015;25:1008-1017.",
        url="https://genome.cshlp.org/content/25/7/1008",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=7536,
        genotypes="7,536 core promoter variants, several hundred native",
        env_n=1,
        env="1",
        instances_n=7536,
        instances_basis="reported",
        phenotype="YFP/mCherry ratio, 16-bin FACS",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="Isolates the core promoter as the sole variable under a constant strong upstream region, which no other row in the band does. The per-variant values are reported to be in Supplemental Table 1, which was not opened, so the release is very likely rather than confirmed.",
        accession="Genome Res Supplemental Table 1; needs a supplement fetch",
        gene_namespace="none; the identifier is the measured core promoter",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the core promoter is a distinct regulatory element from the upstream activating region every other row varies.",
        new_class="core promoter variant perturbation leaf",
    ),
    Candidate(
        name="Sharon 2012 (designed promoter library)",
        organism="S. cerevisiae",
        citation="Sharon E, Kalma Y, Sharp A, et al., Segal E. Nat Biotechnol 2012;30:521-530.",
        url="https://doi.org/10.1038/nbt.2205",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=6500,
        genotypes="6,500 designed promoters, 55 percent Gal4/Gcn4 sites",
        env_n=2,
        env="GAL1-10 and HIS3 reporter backgrounds",
        instances_n=6500,
        instances_basis="reported",
        phenotype="expression, sorting plus barcode sequencing",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The origin of the GPRA vector the three largest rows use, and the nearest thing to a synthetic transcription-factor library in yeast, which the triage records as a genuine gap rather than a missed search. Designed rather than random, so site number, affinity, spacing and orientation vary one at a time.",
        accession="Nat Biotechnol journal supplement, no accession; needs a supplement fetch",
        gene_namespace="none; the identifier is the designed promoter and its barcode",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the controlled counterpart the random libraries lack, and the platform's own origin.",
        new_class="",
    ),
    Candidate(
        name="Renganaath 2020 (native promoter variant MPRA)",
        organism="S. cerevisiae",
        citation="Renganaath K, Cheung R, Day L, Kosuri S, Kruglyak L, Albert FW. eLife 2020;9:e62669.",
        url="https://doi.org/10.7554/eLife.62669",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=5832,
        genotypes="5,832 variants in 2,503 gene promoters",
        env_n=1,
        env="1",
        instances_n=5832,
        instances_basis="reported",
        phenotype="MPRA mRNA/DNA allelic cis effect",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The only dataset found that pins a native variant to a measured cis effect on a named gene with the tested 144 bp window released, which is what joins the synthetic libraries to the natural-isolate panels. 451 variants were called causal, 359 were tested twice in opposite orientation and 527 in both library designs, so the measurement count exceeds the variant count.",
        accession="GEO GSE155944 (raw plus barcode assignments); the MPRA design and processed tables are eLife Supplementary Files",
        gene_namespace="SGD S288C gene names for the target promoters",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: real alleles rather than sampled sequence, measured on the same readout, which is the bridge between the synthetic and the natural axes.",
        new_class="native promoter variant perturbation leaf; MPRA allelic ratio phenotype leaf",
    ),
    Candidate(
        name="Arg-CCU tRNA deep mutational scan",
        organism="S. cerevisiae",
        citation="Deep mutational scan of a single-copy Arg-CCU tRNA over 14 substitutions seen across post-whole-genome-duplication Saccharomyces. PMC6193533.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6193533",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=4176,
        genotypes="4,176 of 5,184 genotypes after filtering",
        env_n=2,
        env="high temperature, 1 M NaCl",
        instances_n=4176,
        instances_basis="reported",
        phenotype="fitness under selection",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A regulatory-element deep mutational scan, complete over a 14-substitution combinatorial space with a direct fitness readout, and outside the protein-substitution scope of the aggregates. It needs its own loader for exactly that reason.",
        accession="journal supplement, no accession located; needs a supplement fetch",
        gene_namespace="SGD S288C for the tRNA locus",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: point substitutions in a non-coding gene, which is the one DMS shape the protein aggregates cannot hold.",
        new_class="non-coding RNA point substitution perturbation leaf",
    ),
    Candidate(
        name="Keren 2016 (promoter panel x named gene, pooled fitness)",
        organism="S. cerevisiae",
        citation="Keren L, et al. Massively parallel interrogation of the effects of gene expression levels on fitness. Cell 2016;166:1282-1294.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE83936",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=10000,
        genotypes="130 promoters over a 500-fold range x about 81 genes",
        env_n=2,
        env="glucose, galactose",
        instances_n=20000,
        instances_basis="product",
        phenotype="pooled growth fitness per strain",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A sequence-level promoter swap at a named native locus against a per-strain fitness, which is the shape the existing typed record fits most closely of anything in this band. The gene count is reported inconsistently as 81, 96 and about 100 across the paper, the commentary and the two deposit records, so it is unresolved and carried as such.",
        accession="GEO GSE83936; ArrayExpress E-GEOD-83936; BioProject PRJNA329129",
        gene_namespace="SGD S288C gene names for the swapped loci",
        verified="inferred",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the only large tabulated panel that links expression level to fitness gene by gene, and the only one whose perturbation is a swap at a native locus.",
        new_class="native-locus promoter swap perturbation leaf",
    ),
    Candidate(
        name="Gui 2021 (position effects with transcriptomes)",
        organism="S. cerevisiae",
        citation="Gui, et al. Transcriptome analysis in yeast reveals the externality of position effects. Mol Biol Evol 2021.",
        url="https://www.biorxiv.org/content/10.1101/2020.04.02.021162",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=250,
        genotypes="about 250 GFP integration loci",
        env_n=1,
        env="1",
        instances_n=250,
        instances_basis="reported",
        phenotype="full transcriptome per integration locus",
        shape="vector over genes",
        dim=6000,
        seq_basis="S288C+integration-locus",
        why="The only position-effect panel with a gene-level readout rather than one reporter value, which is what makes the integration coordinate a usable sequence variable. Changes in neighboring genes rather than in the reporter itself altered growth rate, and no accession was located.",
        accession="no accession found; needs a browser check of the published supplement",
        gene_namespace="SGD S288C for the transcriptome and the integration coordinates",
        retrievability="manual_fetch",
        verified="inferred",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the sequence variable is purely the integration coordinate, and this is the one row in that tier whose phenotype is a vector.",
        new_class="integration locus perturbation leaf",
    ),
    Candidate(
        name="Wu 2017 (genome-wide position effects, 1,044 loci)",
        organism="S. cerevisiae",
        citation="Wu XL, et al. Genome-wide landscape of position effects on heterogeneous gene expression in Saccharomyces cerevisiae. Biotechnol Biofuels 2017.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5516366/",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=1044,
        genotypes="1,044 evenly spaced integration loci",
        env_n=3,
        env="promoter, reporter and carbon-source control arms",
        instances_n=1044,
        instances_basis="reported",
        phenotype="RFP intensity, plate reader",
        shape="scalar",
        seq_basis="S288C+integration-locus",
        why="The largest position-effect panel, with a greater than 13-fold range and location dominating promoter, reporter and carbon source. Each strain also carries a knockout-collection deletion, so the genotype is a deletion plus an insertion rather than a regulatory element, and Table S2 is stated to be available from the corresponding author on request.",
        accession="Table S2, stated available from the corresponding author on reasonable request",
        gene_namespace="SGD S288C for the integration coordinates and the deletion",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band by its sequence variable, the integration coordinate, even though the readout is one reporter value.",
        new_class="integration locus perturbation leaf",
    ),
    Candidate(
        name="Zhang 2016 (genome-wide GFP position scan)",
        organism="S. cerevisiae",
        citation="Zhang, et al. Genome-wide GFP position-effect scan using the yeast knockout collection. Chin J Biotech 2016.",
        url="https://wsws.ijournals.cn/cjbcn/article/pdf/gc16070901",
        klass="Regulatory DNA",
        tier=4,
        genotypes_n=872,
        genotypes="428 high- and 444 low-expressed sites",
        env_n=1,
        env="1",
        instances_n=872,
        instances_basis="reported",
        phenotype="GFP by FACS",
        shape="scalar",
        seq_basis="S288C+integration-locus",
        why="A genome-wide integration-site scan whose reported sites split cleanly into high and low classes, adding a FACS readout to a tier that is otherwise plate-reader. The total number of sites screened is not stated and the journal is Chinese-language, so a browser fetch comes first.",
        accession="no accession located; Chinese-language journal PDF, needs a browser fetch",
        gene_namespace="SGD S288C for the integration coordinates",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band by its sequence variable, and the only FACS-read position-effect scan.",
        new_class="integration locus perturbation leaf",
    ),
    Candidate(
        name="Chen (earlier position-effect study, 482 loci)",
        organism="S. cerevisiae",
        citation="An earlier Chen study characterizing position effects at 482 loci, reported inside Wu 2017; primary citation not located.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5516366/",
        klass="Regulatory DNA",
        tier=4,
        genotypes_n=482,
        genotypes="482 loci, about a third on chromosomes I and VI",
        env_n=None,
        env="not reported",
        instances_n=482,
        instances_basis="reported",
        phenotype="reporter expression per locus",
        shape="scalar",
        seq_basis="S288C+integration-locus",
        why="A third independent position-effect panel, which matters because the tier's whole claim is that chromosomal location dominates the designed sequence. The count reaches this table only through Wu 2017's citation of it, so the primary paper has to be identified before the row can be filled.",
        accession="not located; the primary paper must be identified first",
        gene_namespace="SGD S288C for the integration coordinates",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band by its sequence variable, the integration coordinate.",
        new_class="integration locus perturbation leaf",
    ),
    Candidate(
        name="Keren 2013 (native promoter panel across conditions)",
        organism="S. cerevisiae",
        citation="Keren L, et al. Mol Syst Biol 2013.",
        url="https://spell.yeastgenome.org/",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=900,
        genotypes="about 900 native promoters",
        env_n=10,
        env="10 conditions",
        instances_n=9000,
        instances_basis="product",
        phenotype="promoter activity, YFP with a fixed ADH1 terminator",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="Native promoters measured across ten conditions is the only genuine environment axis on native regulatory sequence in this band, and it is one of the eight published benchmark tasks. Both counts reach this table through two secondary sources rather than the paper, and one validation used 625 of the promoters.",
        accession="not located; reached only through the Zrimec and Karollus validation sets",
        gene_namespace="SGD S288C gene names for the native promoters",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the only row in it with a real condition axis on native sequence.",
        new_class="",
    ),
    Candidate(
        name="Dvir 2013 (5' UTR positions -10 to -1)",
        organism="S. cerevisiae",
        citation="Dvir S, Velten L, Sharon E, Zeevi D, Carey LB, Weinberger A, Segal E. Proc Natl Acad Sci USA 2013;110:E2792-E2801.",
        url="https://doi.org/10.1073/pnas.1222534110",
        klass="Regulatory DNA",
        tier=3,
        genotypes_n=2041,
        genotypes="2,041 5' UTR variants at positions -10 to -1",
        env_n=1,
        env="1",
        instances_n=2041,
        instances_basis="reported",
        phenotype="reporter fluorescence",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The cleanest minimal-window 5' UTR panel available: ten bases immediately before the start codon, with the -3 to -1 context dominant, then secondary structure, then out-of-frame start codons, explaining two-thirds of expression variation. A second independently constructed library was held out as a test set.",
        accession="PNAS journal supplement, no accession; needs a supplement fetch",
        gene_namespace="none; the identifier is the measured 5' UTR window",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: a controlled ten-base window, which is the designed counterpart to the half-million random 5' UTRs above it.",
        new_class="5' UTR variant perturbation leaf",
    ),
    Candidate(
        name="Lubliner 2013 (native core promoter maximal strengths)",
        organism="S. cerevisiae",
        citation="Lubliner S, Keren L, Segal E. Nucleic Acids Res 2013.",
        url="https://academic.oup.com/nar",
        klass="Regulatory DNA",
        tier=4,
        genotypes_n=None,
        genotypes="native core promoters, count not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="native core promoter maximal strength",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The primary source behind one of the published core-promoter models, measured over a 90 bp window from -70 to +20 of the transcription start site. No count, accession or deposit was located, so the row exists to keep the source identified rather than to be built next.",
        accession="not located",
        gene_namespace="SGD S288C gene names for the native core promoters",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the native counterpart of the core-promoter tier, and the only source for its reference values.",
        new_class="core promoter variant perturbation leaf",
    ),
    Candidate(
        name="Wei 2017 (synthetic terminator panel)",
        organism="S. cerevisiae",
        citation="Wei, et al. Yeast synthetic terminators: fine regulation of strength through linker sequences. ChemBioChem 2017.",
        url="https://doi.org/10.1002/cbic.201700516",
        klass="Engineering parts",
        tier=4,
        genotypes_n=100,
        genotypes="100 terminators, 45 strong, 31 moderate, 24 weak",
        env_n=1,
        env="1",
        instances_n=100,
        instances_basis="reported",
        phenotype="eGFP strength relative to the PGK1t standard",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="A parts panel whose values are fold-strengths against a named standard part, spanning 0.0613 to 1.8002 relative to PGK1t, with the linker sequence as the designed variable. No repository deposit was found for any terminator panel, so every one of them is supplement-only.",
        accession="ChemBioChem journal supplement, no repository deposit found",
        gene_namespace="none; the identifier is the designed terminator",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the terminator parts tier, whose values are relative to a standard part rather than absolute.",
        new_class="reporter fold-strength against a named standard part leaf",
    ),
    Candidate(
        name="Yamanishi 2013 (genomic terminatome)",
        organism="S. cerevisiae",
        citation="Yamanishi, et al. Characterization of heterologous protein expression using over 3,000 yeast genomic terminators. SGD reference S000206762.",
        url="https://www.yeastgenome.org/reference/S000206762",
        klass="Engineering parts",
        tier=3,
        genotypes_n=3000,
        genotypes="over 3,000 genomic terminators",
        env_n=1,
        env="fixed TDH3 promoter, GFP",
        instances_n=3000,
        instances_basis="estimate",
        phenotype="GFP output under a fixed promoter",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The largest terminator panel found and one of the eight published benchmark tasks. The count is reported as over 3,000 by one route and over 5,000 by another, with a validation set of 4,005, so the figure is unresolved and the primary table was never located.",
        accession="primary table not located; needs the paper identified, then a supplement fetch",
        gene_namespace="SGD S288C gene names for the native terminators",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the largest native-terminator panel, which is the one element of the axis no library in the band randomizes.",
        new_class="",
    ),
    Candidate(
        name="Morse (short synthetic terminators)",
        organism="S. cerevisiae",
        citation="Morse NJ, et al. Short synthetic terminators, 35 to 70 bp; primary citation not located.",
        url="https://scite.ai/authors/nicholas-j-morse-1g5Qrn",
        klass="Engineering parts",
        tier=4,
        genotypes_n=None,
        genotypes="short synthetic terminators, 35 to 70 bp, count not reported",
        env_n=1,
        env="1",
        instances_n=None,
        instances_basis="estimate",
        phenotype="fluorescent protein output",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The best short-terminator panel named anywhere in the triage, with the best member reported at 3.7-fold over CYC1. The source is the weakest in the whole wave, an author aggregator page, so none of its numbers may be promoted and the primary paper has to be identified first.",
        accession="no repository deposit found; primary paper not identified",
        gene_namespace="none; the identifier is the designed terminator",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band: the short-terminator corner of the parts tier, held so the lead is not lost.",
        new_class="",
    ),
    Candidate(
        name="ProteinGym yeast subset",
        organism="S. cerevisiae",
        citation="ProteinGym: deep mutational scanning scores from 217 assays over 186 proteins, distributed through ProteinGymR and ProteinGymPy.",
        url="https://proteingym.org",
        klass="Deep mutational scan",
        tier=3,
        genotypes_n=None,
        genotypes="yeast subset of 217 assays over 186 proteins; yeast count needs a live query",
        env_n=None,
        env="one selection per assay",
        instances_n=None,
        instances_basis="estimate",
        phenotype="per-assay deep mutational scanning score",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The cheapest route to coverage of the yeast mutational-scan tier: filtering to the YEAST identifier token isolates the subset with uniform formatting and a fixed reference per assay and no per-paper retrieval. The yeast-only count was not obtained, so it is reported as needing a live query rather than as a number.",
        accession="ProteinGymR and ProteinGymPy package loaders; archived on Zenodo, record number not captured",
        gene_namespace="UniProt identifiers, with a YEAST organism token",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band as the aggregate route into the mutational-scan tier, which is the sequence-to-function axis in coding space rather than regulatory space.",
        new_class="per-assay functional score leaf; one leaf across 217 selections would be wrong",
    ),
    Candidate(
        name="MaveDB yeast score sets",
        organism="S. cerevisiae",
        citation="Esposito D, et al. MaveDB: an open-source platform to distribute and interpret data from multiplexed assays of variant effect. Genome Biol 2019;20:223.",
        url="https://doi.org/10.1186/s13059-019-1845-6",
        klass="Deep mutational scan",
        tier=4,
        genotypes_n=None,
        genotypes="yeast score sets; count needs a live query",
        env_n=None,
        env="one selection per score set",
        instances_n=None,
        instances_basis="estimate",
        phenotype="variant effect score",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The second curated aggregate into the same tier, with an organism filter already built in and over a million variant effect scores across the whole database. The documented search endpoint returns 404, so the yeast count has to come from the live site and the target count is itself reported inconsistently as 39 and 45.",
        accession="mavedb.org; the documented search API endpoint returns HTTP 404, so the organism filter needs a live query",
        gene_namespace="MaveDB target identifiers",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="regulatory DNA",
        band_why="Regulatory DNA band as the second aggregate route into the mutational-scan tier, held because the two aggregates do not hold the same assays.",
        new_class="per-score-set functional score leaf",
    ),
    Candidate(
        name="Kita 2017 (context-stratified yeast eQTLs)",
        organism="S. cerevisiae",
        citation="Kita R, et al. Proc Natl Acad Sci USA 2017;114.",
        url="https://www.pnas.org/doi/10.1073/pnas.1717421114",
        klass="Natural variation",
        tier=4,
        genotypes_n=683,
        genotypes="683 selected of 1,640 eQTLs",
        env_n=None,
        env="not reported",
        instances_n=683,
        instances_basis="reported",
        phenotype="eQTL effect on expression",
        shape="scalar",
        seq_basis="isolate-WGS",
        why="The only eQTL set found that is stratified by regulatory context, across promoter, 5' UTR, 3' UTR and open reading frame, which makes it usable as a variant-effect evaluation set rather than a mapping table. It is a summary table rather than per-record measurements, so it is a borderline row and is ranked as one.",
        accession="PNAS supplementary data file pnas.1717421114.sd01; needs a supplement fetch",
        gene_namespace="SGD S288C gene names",
        retrievability="manual_fetch",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA band: variants stratified by the regulatory element they sit in, which is the only released set with that stratification.",
        new_class="",
    ),
    # -- Host bootstrap: P. kudriavzevii -----------------------------------
    Candidate(
        name="MAESTRO 2025 (170-isolate WGS, 57-stressor growth panel)",
        organism="P. kudriavzevii",
        citation="Hsieh PH, Sasaki Y, Yang CI, Wu ZY, Steindorff AS, Haridas S, Ke J, Fatma Z, Zhao Z, Opulente DA, Deng S, Hittinger CT, Grigoriev IV, Dien B, Zhao H, Li YP, Yoshikuni Y. bioRxiv 2025;2025.08.29.673034.",
        url="https://doi.org/10.1101/2025.08.29.673034",
        klass="Natural variation",
        tier=2,
        genotypes_n=161,
        genotypes="170 WGS isolates, 161 phenotyped",
        env_n=57,
        env="57 stressors: heat, low pH, organic acids, inhibitors, antifungals",
        instances_n=9177,
        instances_basis="product",
        phenotype="growth under each stressor",
        shape="scalar",
        seq_basis="Pk-isolate-WGS",
        why="The single highest-measurement phenotype resource in this host, with whole-genome sequence per genotype and a copy-number model reaching a median Pearson r of 0.92 across the 57 conditions. Copy-number variation of tandem paralogs is the reported mechanism, validated by a gene4260 and gene4261 deletion, and no accession is in hand.",
        accession="no accession retrieved: bioRxiv returns 403 and Europe PMC fullTextXML for PPR1078985 returns 500; per-isolate IO_NNN BioProjects are inferred, not confirmed",
        gene_namespace="MAESTRO geneNNNN; whether gene4260 equals JGI g4260 is unverified",
        retrievability="ask",
        ask_of="Ping-Hung Hsieh (LBNL), corresponding; also Yoshikuni, Zhao, Hittinger, Opulente, Grigoriev, Leakey",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: this host has no genome-wide screen of any kind, so its largest resource is a 161-isolate panel that the measurement rule ranks below every pseudomonad screen. The band is what keeps it in the first tranche.",
        new_class="Organism member; growth-under-stressor leaf matching the released statistic",
    ),
    Candidate(
        name="Dubinkina 2024 (12-strain low-pH transcriptomic atlas)",
        organism="P. kudriavzevii",
        citation="Dubinkina V, Bhogale S, Hsieh PH, Dibaeinia P, Nambiar S, Maslov S, Yoshikuni Y, Sinha S. Microbiol Spectr 2024.",
        url="https://doi.org/10.1128/spectrum.02536-23",
        klass="Transcriptome",
        tier=2,
        genotypes_n=12,
        genotypes="12 strains, 6 acid-tolerant and 6 susceptible",
        env_n=4,
        env="pH 5.5 control and pH 1.5 at 0 h and 1 h, plus two organic-acid arms",
        instances_n=48,
        instances_basis="product",
        phenotype="gene expression count matrix",
        shape="vector over genes",
        dim=4925,
        seq_basis="Pk-isolate-WGS",
        why="The one omics resource in this host built as a strain-by-condition matrix with a released processed count table rather than raw reads only, which makes it the most directly ingestible thing the host has. Stb5, Mac1 and Rtg1/Rtg3 are the implicated regulators, and the sample and replicate counts are not stated on the deposit page.",
        accession="Illinois Data Bank DOI 10.13012/B2IDB-1336177_V1, holding strain regulators, strains and gene sets, and gene expression counts; the page returns 403 to every automated client",
        gene_namespace="most likely JGI Issorie2 gNNNN; not stated on the deposit page",
        retrievability="manual_fetch",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the only multi-strain expression matrix this host has, and the only deposit that already carries processed values rather than reads.",
        new_class="Organism member; StrainBackground for 12 strains with no per-strain assembly",
    ),
    Candidate(
        name="Fatma 2023 (landing-pad locus screen)",
        organism="P. kudriavzevii",
        citation="Fatma Z, Tan SI, Boob A, Zhao H. A landing pad system for multicopy gene integration in Issatchenkia orientalis. bioRxiv 2023;2023.05.21.541627.",
        url="https://doi.org/10.1101/2023.05.21.541627",
        klass="CRISPR library screen",
        tier=3,
        genotypes_n=47,
        genotypes="47 gRNAs across 21 intergenic regions",
        env_n=4,
        env="4 readouts per locus",
        instances_n=188,
        instances_basis="product",
        phenotype="cutting efficiency, integration efficiency, fitness, GFP level",
        shape="4 readouts per locus",
        seq_basis="SD108+guide",
        why="The only measured per-locus fitness data in this species, and genotype-precise: the locus and the cassette are both named. Two of its four readouts, cutting and integration efficiency, fall under the standing guide-level exclusion while its fitness and GFP readouts do not, so the row forces that decision rather than inheriting it.",
        accession="a 23 KB package at uofi.box.com (strains, plasmids, guide RNAs, primers) plus github.com/HuiminZhao/Landing-pad-model; no DOI and no archival deposit",
        gene_namespace="JGI Issorie2 coordinates; the loci are intergenic, so not gNNNN gene ids",
        retrievability="ask",
        ask_of="Huimin Zhao lab (UIUC)",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only measured per-locus fitness, which no scale ranking would ever place in a first tranche at 188 records.",
        new_class="Organism member; cutting-efficiency and integration-efficiency leaves; integration-site expression leaf",
    ),
    Candidate(
        name=DOUGLASS_MIC,
        organism="P. kudriavzevii",
        citation="Douglass AP, Offei B, Braun-Galleani S, Coughlan AY, Martos AAR, Ortiz-Merino RA, Byrne KP, Wolfe KH. PLoS Pathog 2018;14:e1007138.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6053246/",
        klass="Natural variation",
        tier=3,
        genotypes_n=32,
        genotypes="32 sequenced isolates, 20 clinical and 12 environmental",
        env_n=4,
        env="fluconazole, flucytosine, amphotericin B, micafungin",
        instances_n=128,
        instances_basis="product",
        phenotype="minimum inhibitory concentration, mg/L",
        shape="scalar",
        seq_basis="Pk-isolate-WGS",
        why="The cleanest fully public, fully provenanced genotype-to-phenotype benchmark this host has, with ploidy, mating type, Illumina coverage and ERG11, FKS1 and ABC11 variants per strain beside the MIC table. Fluconazole MICs span 8 to 128 mg/L with a mode of 32, and the whole table was read in session.",
        accession="SRA SRP139299 (32 strains); GenBank CP028773-CP028778 (CBS573 reference) and CP028531-CP028535 (CBS5147); MIC table in the open-access article body",
        gene_namespace="CBS573 / YGAP annotation",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: at 128 values it is the host's best public benchmark, which is the measure of how thin the host is and why a band rather than a score has to carry it.",
        new_class="Organism member; MIC phenotype leaf carrying the susceptibility standard",
    ),
    Candidate(
        name="Douglass 2018 (per-strain SNP density and loss of heterozygosity)",
        organism="P. kudriavzevii",
        citation="Douglass AP, et al., Wolfe KH. PLoS Pathog 2018;14:e1007138, S2 Table.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6053246/",
        klass="Modality / backbone",
        tier=4,
        genotypes_n=32,
        genotypes="32 isolates",
        env_n=1,
        env="none; a genome-architecture statistic",
        instances_n=64,
        instances_basis="product",
        phenotype="SNP density and loss-of-heterozygosity extent per strain",
        shape="2 statistics per strain",
        seq_basis="Pk-isolate-WGS",
        why="The per-isolate genomic-background descriptors that make the MIC rows interpretable, alongside the ploidy and mating-type calls in the paper's main table. Not a growth phenotype, which is why it is a backbone row rather than a screen.",
        accession="S2 Table of the open-access article",
        gene_namespace="CBS573 / YGAP coordinates",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: a genome-architecture layer for the host's only sequenced panel, which nothing else supplies.",
        new_class="Organism member",
    ),
    Candidate(
        name="Douglass 2018 (CBS573 RNA-seq)",
        organism="P. kudriavzevii",
        citation="Douglass AP, et al., Wolfe KH. PLoS Pathog 2018;14:e1007138; SRA SRP139056.",
        url="https://www.ncbi.nlm.nih.gov/sra/?term=SRP139056",
        klass="Transcriptome",
        tier=4,
        genotypes_n=1,
        genotypes="CBS573 type strain",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=1,
        seq_basis="CBS573-reference",
        why="The only expression data pinned to the haploid-resolved CBS573 reference, which makes it the natural bridge for building the clinical side of a gene-identifier crosswalk. It is raw reads, so admitting it is a build rather than an ingest of released values.",
        accession="SRA SRP139056",
        gene_namespace="CBS573 / YGAP annotation",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: the only expression data on the host's clean reference, and the anchor for the namespace crosswalk.",
        new_class="Organism member",
    ),
    Candidate(
        name="MAESTRO 2025 (gene4260/gene4261 paralog-pair deletion)",
        organism="P. kudriavzevii",
        citation="Hsieh PH, et al., Yoshikuni Y. bioRxiv 2025;2025.08.29.673034, CRISPR validation arm.",
        url="https://doi.org/10.1101/2025.08.29.673034",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=1,
        genotypes="one tandem paralog pair deleted",
        env_n=None,
        env="several stresses, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="stress-tolerance effect of the deletion",
        shape="scalar",
        seq_basis="SD108+designed-edit",
        why="One of only three sources of measured gene-perturbation phenotype in this host, and the designed-edit arm that validates the copy-number mechanism the isolate panel infers. Whether the deletion is one double mutant or two singles is not stated.",
        accession="same unresolved MAESTRO deposit; needs the browser fetch of the preprint",
        gene_namespace="MAESTRO geneNNNN",
        retrievability="ask",
        ask_of="Ping-Hung Hsieh (LBNL), corresponding",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a measured gene perturbation in a host that has none at genome scale.",
        new_class="Organism member",
    ),
    Candidate(
        name="Cao 2020 (constitutive promoter-GFP panel)",
        organism="P. kudriavzevii",
        citation="Cao M, Fatma Z, Song X, Hsieh PH, Tran VG, Lyon WL, Sayadi M, Shao Z, Yoshikuni Y, Zhao H. Metab Eng 2020;59:87-97.",
        url="https://doi.org/10.1016/j.ymben.2020.01.005",
        klass="Engineering parts",
        tier=4,
        genotypes_n=16,
        genotypes="16 named promoters plus an unstated number more",
        env_n=None,
        env="different culture conditions, count not reported",
        instances_n=16,
        instances_basis="estimate",
        phenotype="GFP level per promoter construct",
        shape="scalar",
        seq_basis="SD108-reference",
        why="The only promoter-strength panel in this host, and the row type is exactly the one the schema already serves: a named promoter driving a reporter under stated media. It also anchors the partial map from JGI gNNNN ids to gene names, since g247 is TDH3, g853 is GPM1 and g5025 is SED1.",
        accession="Illinois Data Bank IDB-8429399 (part sequences plus a top-10 promoter list) and the Elsevier supplement zip; both return 403 to automated clients",
        gene_namespace="JGI Issorie2 gNNNN",
        retrievability="manual_fetch",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only parts panel, and the source of its gene-name to model-id map.",
        new_class="Organism member",
    ),
    Candidate(
        name="Cao 2020 (pooled four-condition RNA-seq)",
        organism="P. kudriavzevii",
        citation="Cao M, et al., Zhao H. Metab Eng 2020;59:87-97, promoter-nomination RNA-seq.",
        url="https://doi.org/10.1016/j.ymben.2020.01.005",
        klass="Transcriptome",
        tier=4,
        genotypes_n=1,
        genotypes="SD108 wild type",
        env_n=4,
        env="glucose and glucose plus inhibitors, each aerobic and anaerobic",
        instances_n=1,
        instances_basis="reported",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=4925,
        seq_basis="SD108-reference",
        why="Supplies the reference expression layer the promoter panel was selected from, under four named inhibitor and oxygen conditions with recipes given to the gram. It is a single library pooled from all four conditions, so there is no per-condition matrix and the honest record is one row whose environment is a pool.",
        accession="Illinois Data Bank IDB-8429399, Transcriptomics tag; no SRA or GEO accession named; the page returns 403",
        gene_namespace="JGI Issorie2 gNNNN by lineage inference",
        retrievability="manual_fetch",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the host's reference expression layer, and the one row that forces a decision about an environment that is a pool.",
        new_class="Organism member; an Environment construct for a pooled condition",
    ),
    Candidate(
        name="Tran 2023 (succinate end-to-end pipeline)",
        organism="P. kudriavzevii",
        citation="Tran VG, Mishra S, Bhagwat SS, Shafaei S, Shen Y, Allen JL, Crosly BA, Tan SI, Fatma Z, Rabinowitz JD, Guest JS, Singh V, Zhao H. Nat Commun 2023;14.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10547785/",
        klass="Production campaign",
        tier=4,
        genotypes_n=10,
        genotypes="at least 10 genotypes countable from the body text",
        env_n=3,
        env="sugar media at pH 3 fed-batch, sugarcane juice, pilot batch",
        instances_n=30,
        instances_basis="estimate",
        phenotype="titer, rate and yield",
        shape="3 quantities per arm",
        seq_basis="SD108+designed-edit",
        why="The deepest genotype-precise engineering record in this host, with named deletions and chromosomal integrations that are reconstructable, reaching 63.1 g/L at pilot scale. Supplementary Table 5 lists all strains and was not read, so ten is a floor and the often-quoted 100 to 250 genotype figure for the whole lineage is an upper-range estimate.",
        accession="no sequence or raw-data accession; all values in the Supplementary Information and the Source Data file, with plasmid sequences in Supplementary Data 4",
        gene_namespace="JGI Issorie2 gNNNN, with loci also named by homology as PkJEN2-1 and PkJEN2-2",
        retrievability="manual_fetch",
        verified="inferred",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's deepest named-genotype record, which is worth ingesting for the genotype side rather than for record count.",
        new_class="Organism member",
    ),
    Candidate(
        name="Xiao 2014 (SD108 reductive-TCA succinate)",
        organism="P. kudriavzevii",
        citation="Xiao H, et al., Zhao H. Microb Cell Fact 2014;13:121.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4244060",
        klass="Production campaign",
        tier=4,
        genotypes_n=2,
        genotypes="SD108 wild type and a uracil-auxotroph rTCA integrant",
        env_n=4,
        env="4 acid-tolerance conditions plus shake-flask batch",
        instances_n=8,
        instances_basis="estimate",
        phenotype="titer, yield, productivity, and acid-tolerance growth calls",
        shape="scalar",
        seq_basis="SD108-reference",
        why="The provenance origin of the whole SD108 lineage: this is where the strain was isolated and typed, and it supplies the acid-tolerance thresholds the later work builds on, at 30 g/L succinic acid, 30 g/L itaconic, 23 g/L adipic and 20 g/L acetic. Reported at 11.63 g/L titer in shake-flask batch.",
        accession="no accession named; open-access article",
        gene_namespace="none released; the rTCA genes are named functionally",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the strain-provenance root of every other engineering row in this host.",
        new_class="Organism member",
    ),
    Candidate(
        name="Illinois Data Bank IDB-0538107 (succinate pipeline deposit)",
        organism="P. kudriavzevii",
        citation="Illinois Data Bank deposit, authors include Huimin Zhao; published March 2026.",
        url="https://databank.illinois.edu/datasets/IDB-0538107",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not reported on the deposit page",
        env_n=3,
        env="fed-batch minimal medium, fed-batch sugarcane juice, pilot batch, all at pH 3",
        instances_n=None,
        instances_basis="estimate",
        phenotype="titer",
        shape="scalar",
        seq_basis="SD108+designed-edit",
        why="The fed-batch titers here, 109.5 g/L in minimal medium and 104.6 g/L in sugarcane juice, are higher than anything in the paper body this wave read, so the deposit holds process records the paper rows do not. Its techno-economic range is model output and must not be stored as a measurement.",
        accession="Illinois Data Bank IDB-0538107; the page returns 403 to automated clients",
        gene_namespace="not stated; JGI Issorie2 gNNNN by lineage inference",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's deposited process data, which the journal pages do not carry.",
        new_class="Organism member",
    ),
    Candidate(
        name="CABBI inorganic carbon sources (low-pH succinate kinetics)",
        organism="P. kudriavzevii",
        citation="Data for influence of inorganic carbon sources on low-pH succinic acid production by Issatchenkia orientalis: process insights and kinetic analysis. CABBI dataset page.",
        url="https://cabbi.bio/datasets/",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not reported on the dataset page",
        env_n=None,
        env="inorganic carbon sources at low pH, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="succinate process and kinetic data",
        shape="not reported",
        seq_basis="SD108+designed-edit",
        why="A carbon-source by process matrix on this host's flagship product, which no other row supplies. Only the dataset page was verified to exist, so the contents, the deposit target and the counts all need one browser visit.",
        accession="CABBI dataset page; the deposit target was not read",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a process-condition axis on the host's main product.",
        new_class="Organism member",
    ),
    Candidate(
        name="Xylose 3-HP from cellulosic hydrolysates (2026)",
        organism="P. kudriavzevii",
        citation="Xylose metabolic engineering of Issatchenkia orientalis for 3-hydroxypropionic acid production. Bioresour Technol 2026;455:134783.",
        url="https://doi.org/10.1016/j.biortech.2026.134783",
        klass="Production campaign",
        tier=4,
        genotypes_n=4,
        genotypes="1AP and three xylose derivatives, 1APX, 1APX-X2 and 1APX-X23, off the SD108 lineage of the Tan 2026 parent",
        env_n=2,
        env="xylose and cellulosic hydrolysates without nutrient supplementation",
        instances_n=8,
        instances_basis="product",
        phenotype="titer, yield and productivity",
        shape="3 quantities",
        dim=3,
        seq_basis="SD108+designed-edit",
        why="Adds a xylose and hydrolysate substrate axis to the host's 3-HP genotypes, which is the feedstock question the whole product line exists to answer. Table 1 carries all four strains at 21.2 to 53.5 g/L, 0.30 to 0.46 g/g and 0.20 to 0.31 g/L/h, with the XYL1, XYL2 and XYL3 cassettes from Scheffersomyces stipitis at the named GD15 and GD24 loci. The article says only that data will be made available on request, so the deposit is the route.",
        accession="open full text at OSTI 3391126; deposit 10.13012/B2IDB-1480828_V1, whose file list is unverified because the Illinois Data Bank returns 403 to every automated client",
        gene_namespace="gene-name abbreviations, not JGI Issorie2 gNNNN; no 3-HP paper in this species uses the g-prefixed namespace, and the cassettes are named by source organism",
        retrievability="manual_fetch",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: a feedstock axis on the host, which the glucose rows cannot supply.",
        new_class="Organism member",
    ),
    Candidate(
        name="Jeong 2025 (acetate metabolism during xylose fermentation)",
        organism="P. kudriavzevii",
        citation="Jeong D, Jia L, Zhang M, Oh EJ, Zhao J. Acetate metabolism during xylose fermentation enhances 3-hydroxypropionic acid production in acid-tolerant Issatchenkia orientalis. Bioresour Technol 2025;437:133113.",
        url="https://doi.org/10.1016/j.biortech.2025.133113",
        klass="Production campaign",
        tier=4,
        genotypes_n=1,
        genotypes="IoDY01H, a named production background with no reference assembly in the genomes tier",
        env_n=2,
        env="hemp stalk hydrolysate by separate hydrolysis and fermentation, at pH 3.5 and pH 5.5",
        instances_n=2,
        instances_basis="product",
        phenotype="titer",
        shape="scalar",
        seq_basis="Pk-named+designed-edit",
        why="A third independent 3-HP arm with a distinct carbon and acetate axis, and the row that closes a stale identity: the preprint this wave recorded is now published, which moves it off bioRxiv and onto a journal DOI. One strain at 8.7 g/L, with pH 5.5 beating pH 3.5, so the released numbers are a pH pair rather than a strain panel.",
        accession="journal DOI 10.1016/j.biortech.2025.133113, PMID 40774436; no repository deposit found",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: a third carbon arm on the host's second product.",
        new_class="Organism member",
    ),
    Candidate(
        name="Glycolic acid overproduction (2026)",
        organism="P. kudriavzevii",
        citation="Xylose-assisted glycolic acid overproduction in engineered Issatchenkia orientalis. J Fungi 2026;12:751.",
        url="https://doi.org/10.3390/jof12100751",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=1,
        env="xylose-assisted culture",
        instances_n=None,
        instances_basis="estimate",
        phenotype="glycolic acid titer",
        shape="scalar",
        seq_basis="SD108+designed-edit",
        why="A fourth product on this host, extending the product axis past the C4 acids, and it is open access so the values are reachable without a browser. Contents were not read, so the counts are open.",
        accession="open-access article (PMC13705920); no separate deposit named",
        gene_namespace="not reported",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a fourth product class on the host, and one of the few rows in it that is openly retrievable.",
        new_class="Organism member",
    ),
    Candidate(
        name="Wu 2023 (piggyBac citramalate copy-number and locus variants)",
        organism="P. kudriavzevii",
        citation="Wu, et al. Metab Eng Commun 2023;16:e00220.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9969067",
        klass="Production campaign",
        tier=4,
        genotypes_n=3,
        genotypes="SD108, SD108 ura3, SD108 ura3 leu2, plus integrants varying cimA copy number and locus",
        env_n=1,
        env="glucose, 48 h batch",
        instances_n=None,
        instances_basis="estimate",
        phenotype="citramalate titer and molar yield",
        shape="scalar",
        seq_basis="SD108+designed-edit",
        why="The one row in this host that varies copy number and integration site at the same time, which is a dose-response axis nothing else has, reaching 2.0 g/L in 48 h at up to 7 mol percent on glucose. The piggyBac element here is a copy-number and placement tool, not a mutagenesis library.",
        accession="Illinois Data Bank IDB-6723545 (published October 2025, contact Yoshikuni); the page returns 403; the article is open access",
        gene_namespace="not stated; JGI Issorie2 gNNNN by lineage inference",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a copy-number by locus dose axis, which is the only quantitative genotype axis the host offers.",
        new_class="Organism member",
    ),
    Candidate(
        name="Mitochondrial decompartmentalization (2025, with GNPS metabolomics)",
        organism="P. kudriavzevii",
        citation="Decompartmentalization of the yeast mitochondrial metabolism to improve chemical production in Issatchenkia orientalis. Nat Commun 2025.",
        url="https://doi.org/10.1038/s41467-025-62304-w",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="metabolite intensities",
        shape="vector over metabolites",
        dim=1,
        seq_basis="SD108+designed-edit",
        why="The only metabolomics deposition in this host, attached to a named engineering intervention. The perturbation is a relocalization of an enzyme out of the mitochondrion, which is a genotype shape nothing in the schema expresses today.",
        accession="a GNPS deposition named on the CABBI dataset page but not read; a further Illinois Data Bank deposit surfaces only through OSTI as DOI 10.13012/B2IDB-9575028_V1",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only metabolomics layer.",
        new_class="Organism member; relocalization perturbation leaf",
    ),
    Candidate(
        name="Xylose utilization across four natural backgrounds",
        organism="P. kudriavzevii",
        citation="Cas9 metabolic-engineering work placing xylose reductase, xylitol dehydrogenase and xylulokinase from Scheffersomyces stipitis into four I. orientalis backgrounds; primary citation not located.",
        url="https://core.ac.uk/works/286100561",
        klass="Production campaign",
        tier=4,
        genotypes_n=4,
        genotypes="SD108, IO21, IO45, IO46, each with the same three-gene cassette",
        env_n=1,
        env="xylose",
        instances_n=4,
        instances_basis="reported",
        phenotype="xylose utilization",
        shape="scalar",
        seq_basis="SD108+designed-edit",
        why="The only row in the whole wave that puts one identical construct into four different natural backgrounds, which is the background-effect axis a cross-host substrate exists to measure. The IO21, IO45 and IO46 naming matches the IO_NNN resequencing projects, so this is also the bridge between the engineering line and the natural panel.",
        accession="not located; only an aggregator record was reached, so the primary paper must be identified first",
        gene_namespace="Scheffersomyces stipitis gene symbols for the heterologous genes; host namespace not stated",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: one construct in four backgrounds is the background-effect design, and it exists nowhere else in this wave.",
        new_class="Organism member; StrainBackground for three isolates with no public assembly",
    ),
    Candidate(
        name="GSE262836 (Delhi NICU, fluconazole-resistance expression)",
        organism="P. kudriavzevii",
        citation="Genomic characterization and transcriptomic profile of Candida krusei in a neonatal intensive care unit in Delhi, India. GEO series GSE262836, 2025.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE262836",
        klass="Transcriptome",
        tier=4,
        genotypes_n=2,
        genotypes="one fluconazole-susceptible and one resistant isolate",
        env_n=1,
        env="1",
        instances_n=4,
        instances_basis="reported",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=4925,
        seq_basis="Pk-isolate-WGS",
        why="With its sibling series it is one of only two deposits in this host that pair a drug phenotype with expression, and both surfaced only through a taxid query rather than a name search. 178 genes were called differentially expressed, which is a derived contrast rather than per-record data, so the counts matrix is what is ingestible.",
        accession="GEO GSE262836, platform GPL34351; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: one of two rows in this host pairing a drug phenotype with expression, and openly retrievable.",
        new_class="Organism member",
    ),
    Candidate(
        name="GSE282019 (Delhi NICU five-year outbreak)",
        organism="P. kudriavzevii",
        citation="Genomic analyses reveal high diversity and rapid evolution of Pichia kudriavzevii within a neonatal intensive care unit in Delhi, India. GEO series GSE282019, 2024; PMID 39853119.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE282019",
        klass="Transcriptome",
        tier=4,
        genotypes_n=3,
        genotypes="3 isolates with differing fluconazole MICs",
        env_n=1,
        env="1",
        instances_n=6,
        instances_basis="reported",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=4925,
        seq_basis="Pk-isolate-WGS",
        why="The second of the two drug-stratified expression series in this host, with ERG5 and FCR1 differences reported and a published identifier behind it. Whole-genome sequencing covers bloodstream outbreaks over five years in two genetically distinct clusters.",
        accession="GEO GSE282019, platform GPL34351; programmatically retrievable",
        gene_namespace="not stated; ERG5 and FCR1 are symbols, not systematic ids",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only published drug-stratified expression series, and openly retrievable.",
        new_class="Organism member",
    ),
    Candidate(
        name="GSE202655 (L-malic acid, EcSthA transhydrogenase)",
        organism="P. kudriavzevii",
        citation="Metabolic engineering of the acid-tolerant yeast Pichia kudriavzevii for efficient L-malic acid production at low pH. GEO series GSE202655, 2023; PMID 36566973.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE202655",
        klass="Transcriptome",
        tier=4,
        genotypes_n=2,
        genotypes="MA009-5 original and MA009-10 carrying E. coli SthA",
        env_n=1,
        env="low pH",
        instances_n=6,
        instances_basis="reported",
        phenotype="transcript abundance, with titer and conversion reported separately",
        shape="vector over genes",
        dim=4925,
        seq_basis="engineered-chassis",
        why="A fifth product on this host, L-malic acid, in a background outside the SD108 lineage entirely, and the only engineered-strain expression series not from that lab. The MA009 background has no reference assembly anywhere in the triage, so its strain record carries a provenance gap rather than an assembly pin.",
        accession="GEO GSE202655, platform GPL32243; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: an engineered-strain expression series in a non-SD108 background, which the host otherwise lacks.",
        new_class="Organism member; StrainBackground for MA009 with no public assembly",
    ),
    Candidate(
        name="GSE159457 (CY902 C4 dicarboxylic acid transporters)",
        organism="P. kudriavzevii",
        citation="Identification and characterization of C4 dicarboxylic acid transporters in Pichia kudriavzevii CY902. GEO series GSE159457, 2021; PMID 33629807.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE159457",
        klass="Transcriptome",
        tier=4,
        genotypes_n=1,
        genotypes="CY902 ura3",
        env_n=3,
        env="pH 5.0 on glucose, succinic acid, or both",
        instances_n=6,
        instances_basis="reported",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=4925,
        seq_basis="engineered-chassis",
        why="A clean one-strain carbon-source by expression matrix in a third background, and the only series in this host with an explicit raw-read study identifier. CY902 has no reference assembly in the triage.",
        accession="GEO GSE159457, platform GPL29244; raw reads at SRA SRP287255; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: a carbon-source axis on expression in a third background, fully deposited.",
        new_class="Organism member; StrainBackground for CY902 with no public assembly",
    ),
    Candidate(
        name="GSE39558 (ethanol-stress expression)",
        organism="P. kudriavzevii",
        citation="GEO series GSE39558, ethanol-stress RNA-seq; species assignment reported by a search index only.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39558",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=1,
        env="ethanol stress",
        instances_n=None,
        instances_basis="estimate",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=1,
        seq_basis="Pk-isolate-WGS",
        why="Would add an ethanol-stress expression arm, and it is a single call to verify. The series was not among the four confirmed by the taxid sweep, so even its species assignment is open.",
        accession="GEO GSE39558; one call verifies contents, sample count and strain",
        gene_namespace="not reported",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a stress axis the host does not otherwise have in expression, and the cheapest row in the wave to verify.",
        new_class="Organism member",
    ),
    Candidate(
        name="PXD069498 (extracellular vesicle proteome)",
        organism="P. kudriavzevii",
        citation="PRIDE project PXD069498, submitter Chiara D'Ambrosio, April 2026.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD069498",
        klass="Proteome",
        tier=4,
        genotypes_n=1,
        genotypes="wild type, strain not reported",
        env_n=1,
        env="vesicle purification",
        instances_n=189,
        instances_basis="reported",
        phenotype="protein identification in purified vesicles",
        shape="identification list",
        seq_basis="Pk-reference",
        why="189 proteins identified from 33 raw files, which is thin on purpose: an identification list with no genotype and no environment axis is a protein-presence set rather than a genotype-by-environment record. Held so the deposit is accounted for, and deferred rather than built.",
        accession="PRIDE PXD069498; retrievable through the PRIDE API",
        gene_namespace="a protein-identifier namespace, not stated",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only vesicle proteome, held for completeness at the bottom of the band.",
        new_class="Organism member; a leaf for an identification list with no abundance",
    ),
    Candidate(
        name="PXD041801 (flocculin expression and floating biofilms)",
        organism="P. kudriavzevii",
        citation="Raw proteomics for a 2023 PLoS Pathogens study of flocculin expression and floating biofilms; submitted via MassIVE.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD041801",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="protein abundance associated with biofilm formation",
        shape="vector over proteins",
        dim=1,
        seq_basis="Pk-reference",
        why="A biofilm and adhesion phenotype axis absent everywhere else in this host, if the deposit carries per-protein quantification and a genotype contrast. Only the accession was verified.",
        accession="ProteomeXchange PXD041801 via MassIVE; retrievable",
        gene_namespace="not reported",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a biofilm axis the host has nowhere else.",
        new_class="Organism member",
    ),
    Candidate(
        name="PXD039973 (succinylation and amphotericin B resistance)",
        organism="P. kudriavzevii",
        citation="PRIDE/iProX project PXD039973, protein succinylation linked to amphotericin B resistance.",
        url="https://www.omicsdi.org/dataset/iprox/PXD039973",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="protein succinylation site intensity",
        shape="vector over sites",
        dim=1,
        seq_basis="Pk-reference",
        why="The only post-translational modification dataset in this host, and it is tied to an antifungal-resistance phenotype, so it pairs conceptually with the MIC panel. Succinylation is a different modification from phosphorylation and has no leaf, so this is a real schema change rather than a trivially additive member.",
        accession="PRIDE/iProX PXD039973; iProX retrievability unverified",
        gene_namespace="not reported",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only modification-site layer, and the only one tied to a drug phenotype.",
        new_class="Organism member; a succinylation-site leaf, since the phosphosite family is phospho-only",
    ),
    Candidate(
        name="PXD075355 (Pichia against Botrytis cinerea)",
        organism="P. kudriavzevii",
        citation="PRIDE project PXD075355, native Pichia against Botrytis cinerea, 2026; lists both P. kudriavzevii and P. kluyveri.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD075355",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=1,
        env="antagonism against Botrytis cinerea",
        instances_n=None,
        instances_basis="estimate",
        phenotype="protein abundance under antagonism",
        shape="vector over proteins",
        dim=1,
        seq_basis="Pk-reference",
        why="A biotic-interaction environment nothing else in this wave supplies. The deposit mixes two species, so records have to be split by organism before anything is written.",
        accession="PRIDE PXD075355; retrievable, but the deposit holds two species",
        gene_namespace="not reported",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the only biotic-interaction environment in the whole wave.",
        new_class="Organism member",
    ),
    Candidate(
        name="Pk E1 toolkit expansion (mycoprotein, 2026)",
        organism="P. kudriavzevii",
        citation="Expanding the genetic toolkit of Pichia kudriavzevii E1 for enhanced mycoprotein production. Microb Biotechnol 2026;19.",
        url="https://doi.org/10.1111/1751-7915.70447",
        klass="Engineering parts",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="parts characterization and mycoprotein output",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="A second, independent parts panel in this host in a background outside the SD108 lineage, plus a product class the host has nowhere else. Only the record was verified; the contents need reading before it is a built row.",
        accession="open-access article (PMC13578925); no separate deposit named",
        gene_namespace="not reported; E1 has no reference assembly in the triage",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: a second parts panel in a second background, which is what tests whether the first one transfers.",
        new_class="Organism member; StrainBackground for E1 with no public assembly",
    ),
    Candidate(
        name="China fermented-vegetable and clinical panel (2024)",
        organism="P. kudriavzevii",
        citation="Comparative genomics and susceptibility of fermented-vegetable and clinical Candida krusei isolates. Virulence 2024;15.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11487970/",
        klass="Natural variation",
        tier=4,
        genotypes_n=None,
        genotypes="isolate count not captured; about a quarter of clinical and 87.5 percent of fermented-vegetable isolates are triploid",
        env_n=6,
        env="fluconazole, voriconazole, itraconazole, amphotericin B, caspofungin, posaconazole",
        instances_n=None,
        instances_basis="estimate",
        phenotype="antifungal susceptibility",
        shape="scalar",
        seq_basis="Pk-isolate-WGS",
        why="Would roughly double this host's clinical panel and add two further drugs plus an isolation-source axis, conditional on the supplement carrying per-isolate values. As read, the susceptibility results are group-level comparisons rather than per-record data, which is why the row is blocked rather than ranked on its axes.",
        accession="open-access article; per-isolate MIC tables and genome accessions not verified",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        status="blocked",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only second clinical panel, held because one supplement check decides whether it is a dataset.",
        new_class="Organism member; MIC phenotype leaf",
    ),
    Candidate(
        name="Indonesian clinical genomes (PRJNA819536)",
        organism="P. kudriavzevii",
        citation="Genome note on whole-genome sequencing of 16 clinical isolates from Jakarta hospitals. F1000Research 2022;11:593.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10320322/",
        klass="Natural variation",
        tier=4,
        genotypes_n=16,
        genotypes="16 clinical isolates",
        env_n=None,
        env="resistance phenotypes intended, drug count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="resistance phenotype, if a public table exists",
        shape="scalar",
        seq_basis="Pk-isolate-WGS",
        why="Sixteen more sequenced clinical genomes for a host whose entire sequenced panel is 32 isolates, which is a meaningful fraction. Whether a public susceptibility table accompanies the sequence, rather than being available on request, is explicitly unknown.",
        accession="BioProject PRJNA819536 for the sequence; the phenotype table's location is unknown and may be an ask",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: it would add half again as many sequenced isolates to the host's only genotyped panel.",
        new_class="Organism member; MIC phenotype leaf",
    ),
    Candidate(
        name="Unidentified Illinois Data Bank CRISPR/Cas9 gene-deletion deposit",
        organism="P. kudriavzevii",
        citation="An Illinois Data Bank CRISPR/Cas9 gene-deletion dataset surfacing only through an OSTI bibliographic record; deposit identifier not confirmed.",
        url="https://www.osti.gov/biblio/3013685",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="not reported",
        shape="not reported",
        seq_basis="SD108+designed-edit",
        why="The only title in the whole triage that sounds like a multi-gene deletion set in this host, which is exactly the resource two search passes concluded does not exist. Worth resolving precisely because the headline finding is a negative.",
        accession="an Illinois Data Bank deposit whose identifier is unconfirmed; needs an OSTI and a repository browser visit",
        gene_namespace="not reported; JGI Issorie2 gNNNN by lineage inference",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
        band="host bootstrap",
        band_why="Host bootstrap: the one lead that could overturn the finding that this host has no multi-gene deletion resource.",
        new_class="Organism member",
    ),
    # -- Cross-host fitness: the LBL Fitness Browser, one row per organism --
    #
    # One row per browser row rather than one per artifact, because each organism needs
    # its own Organism member and its own reference genome and those are the real costs.
    # The experiment counts are the live public page's, read 2026-11-19; they supersede
    # every search summary, the KBase mirror and the non-public fitprivate instance,
    # whose figures must never be mixed in.
    #
    # dim is 4,530 on every one of these rows, and it is INFERRED, not a per-organism
    # gene count: it is the per-gene depth implied by the report's own aggregate, 6.7e6
    # measurements over 1,479 non-KT2440 experiments. The report gives no per-organism
    # gene count, and the only gene count known anywhere is KT2440's 4,732 with data.
    Candidate(
        name="Fitness Browser RCH2 (Stutzerimonas stutzeri_AE)",
        organism="S. stutzeri",
        citation="Price MN, Wetmore KM, Waters RJ, Callaghan M, Ray J, Liu H, Kuehl JV, Melnyk RA, Lamson JS, Suh Y, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=350,
        env="30 carbon, 19 nitrogen, 50 stress, 38 other",
        instances_n=350,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="The largest single-organism experiment count in the browser, 350 experiments with the broadest stress panel of any pseudomonad in it, and the strain the historical name P. stutzeri points at. It arrives in the same feba.db as every other row in this band, so its marginal cost is one Organism member and one reference genome.",
        accession="feba.db at 10,433,708,032 bytes, Last-Modified 2026-10-27, HTTP 200 to a plain curl; orgId psRCH2; refseq_mapping.tsv bridges locusId to RefSeq",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq by refseq_mapping.tsv",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: nine new hosts arrive inside one artifact under one schema, one condition vocabulary and an identifier bridge the source supplies, so the band exists to keep them together rather than scattered down a measurement ranking.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser FW300-N2E3 (Aquipseudomonas helvetica)",
        organism="P. fluorescens",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=211,
        env="46 carbon, 30 nitrogen, 42 stress, 12 other",
        instances_n=211,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="The second-largest pseudomonad row in the browser and the most balanced of the four FW300 isolates across all four condition classes. It also carries 115 updated annotations, which is the browser's own measure of how much a row taught about unannotated genes.",
        accession="feba.db, orgId pseudo3_N2E3; the historical binomial is P. fluorescens and the current one is A. helvetica",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: part of the single-artifact block of nine hosts, whose shared schema is the reason to admit them together.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser FW300-N2E2 (Aquipseudomonas fluorescens_Q)",
        organism="P. fluorescens",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=203,
        env="41 carbon, 25 nitrogen, 34 stress, 7 other",
        instances_n=203,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="A sibling FW300 isolate at 203 experiments, which is what makes the block a comparative resource rather than nine unrelated screens: four isolates of one species over overlapping condition panels is a within-species transfer test the wave has nowhere else.",
        accession="feba.db, orgId pseudo6_N2E2; 116 updated annotations",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: one of four isolates of one species in the block, which is the within-species comparison the band makes possible.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser WCS417 (Aquipseudomonas simiae)",
        organism="P. simiae",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=201,
        env="51 carbon, 11 nitrogen, 30 stress, 8 other",
        instances_n=201,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="The widest carbon panel of any pseudomonad in the browser at 51 sources, and a plant-associated strain, so it is the one row in the block whose native context is a root rather than groundwater. 118 updated annotations.",
        accession="feba.db, orgId WCS417; historical binomial P. simiae, current A. simiae",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: the plant-associated member of the single-artifact block, with its widest carbon panel.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser FW300-N2C3 (Aquipseudomonas fluorescens_AA)",
        organism="P. fluorescens",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=184,
        env="45 carbon, 19 nitrogen, 36 stress, 16 other",
        instances_n=184,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="A third FW300 isolate at 184 experiments with 122 updated annotations, the most of any row in the block. The four FW300 isolates together are the only place in this wave where one species is screened four times over the same condition vocabulary.",
        accession="feba.db, orgId pseudo5_N2C3_1",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: part of the four-isolate within-species block, and the row that taught the most unannotated genes.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser FW300-N1B4 (Aquipseudomonas sp.)",
        organism="P. fluorescens",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=2,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=147,
        env="39 carbon, 21 nitrogen, 34 stress, 10 other",
        instances_n=147,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="The fourth FW300 isolate, and the one whose current binomial the browser leaves at genus level, so it is the row that forces an Organism member for a strain with no species name. 147 experiments over all four condition classes.",
        accession="feba.db, orgId pseudo1_N1B4",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: completes the four-isolate within-species block, and names the case of a host with no species assignment.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser B728a (Aquipseudomonas fragariae pv. syringae)",
        organism="P. syringae",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=2,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=46,
        env="17 stress, 8 other; no carbon or nitrogen panel",
        instances_n=46,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="A plant pathogen rather than an environmental isolate, screened on stress only, and with a deletion background screened beside it. The B728a pair is the browser's own worked example that it already models genotype by condition across two genotypes of one strain over the same stress set.",
        accession="feba.db, orgId SyringaeB728a; the GTDB rename to A. fragariae pv. syringae means both binomials are needed as aliases",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: the pathogen member of the block, and half of the one strain in it screened in two genetic backgrounds.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    Candidate(
        name="Fitness Browser B728a mexB deletion background",
        organism="P. syringae",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=3,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library in a mexB deletion background",
        env_n=23,
        env="17 stress, 1 other",
        instances_n=23,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="The only transposon library in the browser screened in a defined deletion background against its own parent over the same stress panel, which makes the pair a background-by-perturbation interaction rather than two independent screens. 23 experiments.",
        accession="feba.db, orgId SyringaeB728a_mexBdelta",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: the background-crossed half of the B728a pair, which is the only interaction design in the block.",
        new_class="Organism member; a transposon-in-deletion-background genotype",
    ),
    Candidate(
        name="Fitness Browser PA14 (Pseudomonas aeruginosa)",
        organism="P. aeruginosa",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Fitness / chemical genomics",
        tier=4,
        genotypes_n=None,
        genotypes="pooled barcoded transposon library, gene count not reported per organism",
        env_n=8,
        env="3 other; no carbon, nitrogen or stress panel",
        instances_n=8,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene, with a t statistic",
        shape="vector over genes",
        dim=4530,
        seq_basis="fitbrowser+transposon",
        why="A token presence rather than a compendium, at 8 experiments and no updated annotations, and it is the evidence that the browser is not a usable source of fitness for this species. It is ranked because the species enters the wave through its essentiality benchmark instead.",
        accession="feba.db, orgId PA14; the browser does not rename this organism",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        verified="inferred",
        band="cross-host fitness",
        band_why="Cross-host fitness: the same artifact and schema, kept visible because its size is the finding about this species rather than an omission.",
        new_class="Organism member; a pooled barcoded transposon perturbation leaf",
    ),
    # -- Scale: the pseudomonad literature ---------------------------------
    Candidate(
        name="Khaledi 2020 (414-isolate clinical transcriptome panel)",
        organism="P. aeruginosa",
        citation="Khaledi A, Weimann A, Schniederjans M, Asgari E, Kuo TH, Oliver A, Cabot G, Kola A, Gastmeier P, Hogardt M, Jonas D, Mofrad MRK, Bremges A, McHardy AC, Haussler S. EMBO Mol Med 2020;12:e10264.",
        url="https://doi.org/10.15252/emmm.201910264",
        klass="Transcriptome",
        tier=1,
        genotypes_n=414,
        genotypes="414 clinical isolates, each with a genome and a transcriptome",
        env_n=1,
        env="1 culture condition",
        instances_n=414,
        instances_basis="reported",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=5900,
        seq_basis="pseudomonad-isolate-WGS",
        why="The only panel in this wave that needs no provenance reconstruction: the reference genome, the locus-tag scheme, the susceptibility standard and the replicate rule are all stated, and every layer sits under its own accession. 414 isolate transcriptomes is the largest measurement count on the pseudomonad side.",
        accession="GEO GSE123544 (RNA-seq); BioProject PRJNA526797 (DNA-seq); Zenodo 10.5281/zenodo.3464542 (classifier inputs); reference NC_008463.1",
        gene_namespace="PA14 locus tags against NC_008463.1, with some markers carrying Prokka, Roary or CARD identifiers",
        verified="inferred",
    ),
    Candidate(
        name="putidaPRECISE321 (Lim 2022)",
        organism="P. putida",
        citation="Lim HG, Rychel K, Sastry AV, Mueller J, Niu W, Feist AM, Palsson BO. Metab Eng 2022;72.",
        url="https://doi.org/10.1016/j.ymben.2022.04.004",
        klass="Transcriptome",
        tier=1,
        genotypes_n=None,
        genotypes="KT2440 and engineered strains; strain-genotype count not reported",
        env_n=118,
        env="118 unique conditions across 21 projects",
        instances_n=321,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-reference",
        why="A 5,564 by 321 matrix in one MIT-licensed repository with 118 typed conditions, structurally parallel to the fitness compendium, so one condition vocabulary can serve both. The per-sample accession list lives in a supplementary spreadsheet rather than the text, which is the one provenance item still open.",
        accession="github.com/SBRG/modulome_ppu (MIT); named GEO deposits GSE198395 (81 samples) and GSE193493 (22); the full per-sample GEO list is in the SI spreadsheet",
        gene_namespace="KT2440 PP_ locus tags",
        verified="inferred",
    ),
    Candidate(
        name="Borchert 2024 (KT2440 fitness compendium)",
        organism="P. putida",
        citation="Borchert AJ, Bleem AC, Lim HG, Rychel K, Dooley KD, Kellermyer ZA, Hodges TL, Palsson BO, Beckham GT. mSystems 2024;9:e00942-23.",
        url="https://doi.org/10.1128/msystems.00942-23",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=4732,
        genotypes="4,732 genes with data of 5,564 protein-coding",
        env_n=332,
        env="332 samples; unique conditions reported as 179 in the abstract and 183 in the methods",
        instances_n=332,
        instances_basis="reported",
        phenotype="RB-TnSeq gene fitness with t statistics",
        shape="vector over genes",
        dim=4732,
        seq_basis="KT2440+transposon",
        why="A ready-made 4,732 by 332 matrix with per-sample metadata in one repository, which makes it the most ingestible fitness artifact for this host. 254 of the 332 samples come from the Fitness Browser, so de-duplication must be by experiment identity and never by count, and the paper states no genome assembly version.",
        accession="github.com/beckham-lab/fModule (fitness values, statistics, metadata); the non-browser samples are at SRA PRJNA809672, PRJNA856070, PRJNA1011287",
        gene_namespace="KT2440 PP_ locus tags; no assembly version stated",
        verified="verified",
    ),
    Candidate(
        name="Schmidt 2022 (KT2440 nitrogen BarSeq)",
        organism="P. putida",
        citation="Schmidt M, Pearson AN, Incha MR, et al. Appl Environ Microbiol 2022;88:e02430-21.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9004399",
        klass="Transposon fitness",
        tier=2,
        genotypes_n=672,
        genotypes="672 genes with a significant phenotype, 529 of them UniProt-unreviewed",
        env_n=71,
        env="52 nitrogen compounds plus 19 amino-acid dropouts",
        instances_n=71,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440+transposon",
        why="The largest single-axis nitrogen panel for any pseudomonad, and it nearly doubled the public BarSeq data for this host, with 256 of its hits having had no significant phenotype in any previous study. It ships an interactive view that links into the Fitness Browser, so the values are very likely inside feba.db, but no accession is stated.",
        accession="no accession stated; the study links into the Fitness Browser, so its experiments are most likely inside feba.db",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="inferred",
    ),
    Candidate(
        name="Dotsch 2009 (arrayed PA14 library against 19 antibiotics)",
        organism="P. aeruginosa",
        citation="Dotsch A, Becker T, Pommerenke C, Janowska Z, Jansch L, Haussler S. Antimicrob Agents Chemother 2009;53:2522-2531.",
        url="https://doi.org/10.1128/AAC.00035-09",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=5459,
        genotypes="arrayed PA14 non-redundant mutants; the number screened is not stated",
        env_n=19,
        env="19 clinically used antibiotics",
        instances_n=5459,
        instances_basis="estimate",
        phenotype="semiautomated antimicrobial susceptibility",
        shape="scalar",
        seq_basis="PA14+transposon",
        why="If the full matrix was released it is the densest arrayed-library phenotype matrix in this species, and the highest-value unresolved pseudomonad item in the triage. The number of mutants actually screened is not stated, so the instance figure is the library size rather than a count, and the location and format of the values are unconfirmed.",
        accession="value location unconfirmed; the PA14 non-redundant library backbone is at the PATIMDB portal, whose downloads page was not read",
        gene_namespace="PA14 predicted gene numbers; the scheme is not stated verbatim",
        retrievability="ask",
        ask_of="the study's authors (Dotsch, Haussler), or whoever now holds the arrayed PA14 library data",
        verified="inferred",
        confidence="recall",
    ),
    Candidate(
        name="Kaczmarczyk 2025 (PAO1 genome-wide CRISPRi)",
        organism="P. aeruginosa",
        citation="Kaczmarczyk A, Klotz A, Manfredi P, Jenal U. bioRxiv 2025;2025.08.12.669819.",
        url="https://doi.org/10.1101/2025.08.12.669819",
        klass="CRISPR library screen",
        tier=2,
        genotypes_n=80000,
        genotypes="over 80,000 sgRNAs over virtually all coding sequences and intergenic regions",
        env_n=2,
        env="glycolytic and gluconeogenic growth",
        instances_n=80000,
        instances_basis="estimate",
        phenotype="depletion and enrichment fitness per guide",
        shape="scalar",
        seq_basis="PAO1+guide",
        why="The only pseudomonad CRISPRi score table plausibly downloadable today, covering the essential genes a transposon library structurally cannot score, with more than a thousand genes showing a phenotype. Three Zenodo records are cited without labels, so which one holds the gene-level scores is unknown until they are opened.",
        accession="Zenodo 10.5281/zenodo.15591742, .15592238 and .15593361, contents unlabeled; no SRA or GEO accession",
        gene_namespace="PAO1 PA numbers by organism inference; not stated",
        retrievability="ask",
        ask_of="Jenal lab (Kaczmarczyk, Klotz, Manfredi, Jenal)",
        verified="inferred",
    ),
    Candidate(
        name="Thompson 2020 (KT2440 fatty acids and alcohols)",
        organism="P. putida",
        citation="Thompson MG, Incha MR, Pearson AN, et al. Appl Environ Microbiol 2020;86:e01665-20; correction at 10.1128/AEM.00177-21.",
        url="https://pubmed.ncbi.nlm.nih.gov/32826213",
        klass="Transposon fitness",
        tier=3,
        genotypes_n=None,
        genotypes="strong phenotypes for hundreds of genes; no count stated",
        env_n=23,
        env="13 fatty acids and 10 alcohols",
        instances_n=23,
        instances_basis="reported",
        phenotype="RB-TnSeq gene fitness",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440+transposon",
        why="A named carbon-class panel that most likely supplies part of this host's browser carbon-source experiments, and the row with the clearest de-duplication question in the band. A published correction exists, so the loader has to pin which version it read.",
        accession="not stated; the values most likely reach feba.db",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Thompson 2019 (KT2440 lysine metabolism)",
        organism="P. putida",
        citation="Thompson MG, et al. Massively parallel fitness profiling reveals multiple novel enzymes in Pseudomonas putida lysine metabolism. mBio 2019.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6509195",
        klass="Transposon fitness",
        tier=3,
        genotypes_n=39,
        genotypes="39 genes with fitness below -2 on a lysine route and above -0.5 on glucose",
        env_n=4,
        env="5-aminovalerate, D-lysine, L-lysine, glucose",
        instances_n=4,
        instances_basis="reported",
        phenotype="RB-TnSeq gene fitness",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440+transposon",
        why="Small, but it is the only one of the condition-panel papers in this host whose quantitative table format is named: Table S1 is a released CSV of the RB-TnSeq results. That makes it the cheapest row in the band to build against a stated file.",
        accession="Table S1, a CSV supplementary file; no repository",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="KT2440 aromatics RB-TnSeq (2019 preprint)",
        organism="P. putida",
        citation="RB-TnSeq of Pseudomonas putida KT2440 on coumarate and seven related aromatics. bioRxiv 2019;753889.",
        url="https://www.biorxiv.org/content/10.1101/753889v1.full-text",
        klass="Transposon fitness",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=8,
        env="coumarate plus seven related aromatics",
        instances_n=8,
        instances_basis="reported",
        phenotype="RB-TnSeq gene fitness",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440+transposon",
        why="The aromatic carbon panel for this host, which is the feedstock class its whole engineering literature targets. Authors, journal version, counts and accession were all unresolved, so the row holds the condition axis and nothing more.",
        accession="not reported",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="KT2440 lignin-stream tolerance RB-TnSeq (2023)",
        organism="P. putida",
        citation="RB-TnSeq identifies genetic targets for improved tolerance of lignin-related streams. Metab Eng 2023.",
        url="https://research-hub.nrel.gov/en/publications/rb-tnseq-identifies-genetic-targets-for-improved-tolerance-of-seu/",
        klass="Transposon fitness",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="lignin-related compounds, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="RB-TnSeq fitness under stress",
        shape="vector over genes",
        dim=1,
        seq_basis="KT2440+transposon",
        why="The one transposon study in this host that the triage links directly to downstream engineering, through deletion or constitutive expression of the genes it named, so it is a perturbation-to-engineering bridge rather than a condition panel. Counts and accession unresolved.",
        accession="not reported",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="KT2440 metal-excess Tn-seq (2024)",
        organism="P. putida",
        citation="Genes required for growth of Pseudomonas putida KT2440 under metal excess. bioRxiv 2024;2024.04.12.589247; journal version in Environ Microbiol 2025.",
        url="https://doi.org/10.1101/2024.04.12.589247",
        klass="Transposon fitness",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="metal excess, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="conditional essentiality under metal excess",
        shape="vector over genes",
        dim=1,
        seq_basis="KT2440+transposon",
        why="A metal-stress axis this host has nowhere else, and conditional essentiality rather than a baseline map, which is the distinction the essentiality rows in this band turn on. Counts, authors and accession unresolved.",
        accession="not reported",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="KT2440 mini-Tn5 knockout collection (2010)",
        organism="P. putida",
        citation="Genome-wide mini-Tn5 knockout collection of Pseudomonas putida KT2440 screened on minimal glucose medium; the cited record should be re-checked against the intended work.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC5814926",
        klass="Essentiality",
        tier=4,
        genotypes_n=68,
        genotypes="24 mutants in genes proposed conditionally essential plus 24 newly implicated genes, against 68 predicted essential",
        env_n=1,
        env="minimal glucose medium",
        instances_n=68,
        instances_basis="reported",
        phenotype="conditional essentiality call",
        shape="boolean per gene",
        seq_basis="KT2440+transposon",
        why="The only pre-RB-TnSeq essentiality attempt in this host, and the only thing resembling a baseline call for it, since no modern genome-wide essentiality map for this strain was found at all. The cited record is flagged as possibly misattributed and must be re-checked before use.",
        accession="not reported; the cited identifier is flagged for re-checking",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="KT2440 pooled transposon library (Putida_ML5 metadata)",
        organism="P. putida",
        citation="Pooled RB-TnSeq library metadata for Pseudomonas putida KT2440, one mapped version named Putida_ML5.",
        url="https://nrel.gov/docs/fy22osti/83183.pdf",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=4778,
        genotypes="4,778 non-essential genes, about 100,000 unique insertion mutants in one pool",
        env_n=1,
        env="LB plus kanamycin, time-zero cultures",
        instances_n=4778,
        instances_basis="reported",
        phenotype="insertion position and barcode per mutant",
        shape="genotype record",
        seq_basis="KT2440+transposon",
        why="The genotype-side backbone every fitness row in this host indexes against, and the one place a per-mutant insertion position exists rather than a per-gene average. A downstream bioreactor study built a small barcoded deletion set from the pooled hits, which is the record shape sequence-level genotype fidelity wants.",
        accession="NREL methods documentation plus the mapping archives shipped with the bigfit release",
        gene_namespace="KT2440 PP_ locus tags",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Mueller 2025 (24-strain P. putida Biolog panel)",
        organism="P. putida",
        citation="Mueller J, Krishnan J, Wei Q, Hefner Y, Monk JM, Verkler H, Tibocha-Bonilla JD, et al. bioRxiv 2025;2025.11.17.688870.",
        url="https://doi.org/10.1101/2025.11.17.688870",
        klass="Natural variation",
        tier=2,
        genotypes_n=24,
        genotypes="24 strains including KT2440",
        env_n=205,
        env="Biolog PM01 and PM02A, 190 carbon compounds, plus 15 aromatics",
        instances_n=4920,
        instances_basis="product",
        phenotype="Biolog respiration per well",
        shape="scalar",
        seq_basis="pseudomonad-isolate-WGS",
        why="The only resource that lets this host enter as a species with intraspecific variation rather than one reference strain, and the only multi-strain Biolog matrix for any pseudomonad: 123 of the 205 compounds supported activity in at least one strain. bioRxiv rate-limited four attempts, so the deposit is unconfirmed.",
        accession="unconfirmed: bioRxiv returned HTTP 429 on four attempts, so no BioProject, Zenodo record or machine-readable Table S2 is confirmed",
        gene_namespace="none; the record is strain-level, and the genome accessions for 40 newly sequenced strains are unconfirmed",
        retrievability="ask",
        ask_of="the authors, or a browser visit when bioRxiv is not rate-limiting",
        verified="verified",
    ),
    Candidate(
        name="Bartell 2019 (longitudinal cystic fibrosis panel)",
        organism="P. aeruginosa",
        citation="Bartell JA, Sommer LM, Haagensen JAJ, Loch A, Espinosa R, Molin S, Johansen HK. Nat Commun 2019;10:629.",
        url="https://doi.org/10.1038/s41467-019-08504-7",
        klass="Natural variation",
        tier=2,
        genotypes_n=443,
        genotypes="443 longitudinal isolates, 52 clone types, 39 patients",
        env_n=8,
        env="2 growth media plus the assay conditions for 6 further readouts",
        instances_n=443,
        instances_basis="reported",
        phenotype="growth rate in 2 media, 2 E-test MICs, protease, mucoidity, adhesion, aggregation",
        shape="vector of 8 phenotypes",
        dim=8,
        seq_basis="pseudomonad-isolate-WGS",
        why="The whole dataset sits in one spreadsheet, with eight distinct phenotype types and a longitudinal within-patient axis no other panel has, at 4 technical replicates per non-MIC assay. Isolate counts are internally inconsistent on the page, given as 474, 373, 389 and 443, so they must be reconciled against the released file before ingesting.",
        accession="Supplementary Data 1 (xlsx) holds the entire dataset and the isolate accessions; genomes at SRA study ERP004853",
        gene_namespace="not stated for this panel",
        retrievability="manual_fetch",
        verified="inferred",
    ),
    Candidate(
        name="Penaranda 2025 (125-isolate diversity survey)",
        organism="P. aeruginosa",
        citation="Penaranda C, Brenner EP, Clatworthy AE, Cosimi LA, Ravi J, Hung DT. bioRxiv 2025;2025.09.22.677895.",
        url="https://doi.org/10.1101/2025.09.22.677895",
        klass="Natural variation",
        tier=2,
        genotypes_n=128,
        genotypes="125 newly sequenced isolates, 97 clinical and 28 environmental, plus PAO1, PA14 and PAK",
        env_n=13,
        env="5 phenotype assays plus disc-diffusion against 8 antibiotics",
        instances_n=128,
        instances_basis="reported",
        phenotype="pyocyanin, motility, biofilm, cytotoxicity at 4 h and 24 h, 8-drug susceptibility",
        shape="vector of 13 phenotypes",
        dim=13,
        seq_basis="pseudomonad-isolate-WGS",
        why="The only panel with an environmental against clinical isolation-source axis and host-cell cytotoxicity readouts, which are phenotype families the substrate has never held. Its genes are annotated with de novo locus tags rather than reference numbers while variants are called against the reference, so it forces the alias layer rather than joining on a reference identifier.",
        accession="Supplemental Data 1 (xlsx) holds genome accessions, metadata and phenotypes; the BioProject number is inside that file and unconfirmed",
        gene_namespace="Bakta v1.9.4 per-isolate locus tags, with variants called against PAO1",
        retrievability="manual_fetch",
        verified="inferred",
    ),
    Candidate(
        name="Khaledi 2020 (1,656-record agar-dilution MIC panel)",
        organism="P. aeruginosa",
        citation="Khaledi A, et al., Haussler S. EMBO Mol Med 2020;12:e10264, susceptibility layer.",
        url="https://doi.org/10.15252/emmm.201910264",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=414,
        genotypes="414 clinical isolates",
        env_n=4,
        env="tobramycin, ceftazidime, ciprofloxacin, meropenem",
        instances_n=1656,
        instances_basis="product",
        phenotype="minimum inhibitory concentration, agar dilution per CLSI 2018",
        shape="scalar",
        seq_basis="pseudomonad-isolate-WGS",
        why="The best-provenanced susceptibility layer in the wave: the standard is named, the replicate rule is stated as triplicate with up to five replicates when discordant and an inclusion rule of three concordant results, and the phenotype table is a named dataset. The replicates themselves are not released, so a record is one isolate by one drug with the rule recorded as metadata.",
        accession="phenotype table as Dataset EV1; Zenodo 10.5281/zenodo.3464542 for the classifier inputs",
        gene_namespace="PA14 locus tags against NC_008463.1",
        verified="inferred",
    ),
    Candidate(
        name="Gao 2020 (KT2440 targeted SRM proteomics)",
        organism="P. putida",
        citation="Gao Y, Fillmore TL, Munoz N, Bentley GJ, Johnson CW, et al. Front Bioeng Biotechnol 2020;8:603488.",
        url="https://doi.org/10.3389/fbioe.2020.603488",
        klass="Proteome",
        tier=3,
        genotypes_n=1,
        genotypes="KT2440 wild type",
        env_n=8,
        env="8 growth conditions, 30 biological samples",
        instances_n=132,
        instances_basis="reported",
        phenotype="absolute protein quantification, with units",
        shape="vector over enzymes",
        dim=8,
        seq_basis="KT2440-reference",
        why="It carries units rather than ratios, which is disproportionately valuable to a typed store, over an assay library of 339 peptides representing 132 enzymes. It is deposited to Panorama Public with no ProteomeXchange accession at all, so a proteomics crawl that keys on PXD identifiers misses it entirely.",
        accession="Panorama Public, no PXD accession; monitored peptides in Supplementary Tables S1 and S2, limits of quantification in S3",
        gene_namespace="KT2440 PP_ locus tags, through peptide to protein mapping",
        verified="inferred",
    ),
    Candidate(
        name="GOLD_115 and GOLD_84 essential-gene sets",
        organism="P. aeruginosa",
        citation="Van Maele C, Caboche S, Nicolau-Guillaumet N, Muggeo A, Guillard T. PLoS Comput Biol 2026;22:e1013945.",
        url="https://doi.org/10.1371/journal.pcbi.1013945",
        klass="Essentiality",
        tier=3,
        genotypes_n=199,
        genotypes="115 PA14-in-LB plus 84 cross-strain gene calls",
        env_n=2,
        env="GOLD_115 is PA14 in LB; GOLD_84 is cross-strain and cross-medium",
        instances_n=199,
        instances_basis="reported",
        phenotype="essentiality call, scoped by strain, medium and calling method",
        shape="boolean per gene",
        seq_basis="PA14+transposon",
        why="The only curated essentiality benchmark in the wave, and it is already scoped the way placement by physical measurement requires: GOLD_115 names its strain and its medium, and GOLD_84 carries reference locus tags so it is strain-portable. Recovery of the gold set ranges from 0 to 100 percent depending on the calling tool, which is why a call must carry its method and must never be merged across studies into one boolean.",
        accession="S2 Table, Gold Standard List; the paper's own reads at SRA PRJNA1240204; reference NC_008463.1",
        gene_namespace="PA14 locus tags for GOLD_115, with PAO1 locus tags included in GOLD_84",
        retrievability="manual_fetch",
        verified="verified",
    ),
    Candidate(
        name="Liberati 2006 (PA14 non-redundant transposon library and essentiality)",
        organism="P. aeruginosa",
        citation="Liberati NT, Urbach JM, Miyata S, Lee DG, Drenkard E, Wu G, Villanueva J, Wei T, Ausubel FM. Proc Natl Acad Sci USA 2006;103:2833-2838.",
        url="https://doi.org/10.1073/pnas.0511100103",
        klass="Essentiality",
        tier=3,
        genotypes_n=5459,
        genotypes="arrayed non-redundant mutant set; 335 genes called essential",
        env_n=1,
        env="LB",
        instances_n=335,
        instances_basis="reported",
        phenotype="essentiality call",
        shape="boolean per gene",
        seq_basis="PA14+transposon",
        why="The arrayed-library genotype backbone three downstream screens index against, and the oldest of the five essentiality calls the gold-standard sets intersect. It is the one place in this species where a genotype is a single named isolate rather than a pool member, and the strains are order-only.",
        accession="the PATIMDB portal, with Search, BLAST and Insertion Site Map pages; whether per-mutant tables are bulk-downloadable is unconfirmed",
        gene_namespace="PA14 predicted gene numbers",
        retrievability="manual_fetch",
        verified="verified",
    ),
    Candidate(
        name="Skurnik 2013 (PA14 essentiality)",
        organism="P. aeruginosa",
        citation="Skurnik D, et al. Transposon sequencing of Pseudomonas aeruginosa PA14. 2013; full citation not captured.",
        url="https://doi.org/10.1371/journal.pcbi.1013945",
        klass="Essentiality",
        tier=4,
        genotypes_n=634,
        genotypes="634 genes called essential",
        env_n=1,
        env="LB",
        instances_n=634,
        instances_basis="reported",
        phenotype="essentiality call",
        shape="boolean per gene",
        seq_basis="PA14+transposon",
        why="The largest of the five essential-gene calls that the gold-standard intersection reads, at 634 genes against another study's 335 on the same strain in the same medium. That spread is the evidence that an essentiality call is a threshold and pipeline artifact as much as a biological one.",
        accession="not given by the compiling paper; the accession must be chased from the primary study",
        gene_namespace="PA14 locus tags",
        retrievability="manual_fetch",
        verified="verified",
        confidence="recall",
    ),
    Candidate(
        name="Turner 2015 (PA14 and PAO1 essentiality across four arms)",
        organism="P. aeruginosa",
        citation="Turner KH, Wessel AK, Palmer GC, Murray JL, Whiteley M. Proc Natl Acad Sci USA 2015.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC4386324",
        klass="Essentiality",
        tier=3,
        genotypes_n=352,
        genotypes="352 general plus 199 condition-specific essential genes; 336 of 5,606 PAO1 genes",
        env_n=4,
        env="PA14 in brain-heart-infusion agar and sputum; PAO1 in the agar and in MOPS-succinate",
        instances_n=352,
        instances_basis="reported",
        phenotype="essentiality call, general and condition-specific",
        shape="boolean per gene",
        seq_basis="PA14+transposon",
        why="The only one of the five studies that separates general from condition-specific essentiality and that covers both reference strains, from over a million unique insertion positions, and the only one with a deposited read set. Its four arms are what make GOLD_84 cross-medium rather than cross-strain only.",
        accession="SRA PRJNA265367",
        gene_namespace="PA14 and PAO1 locus tags",
        verified="snippet",
    ),
    Candidate(
        name="Lee 2015 (PAO1 essentiality across three media)",
        organism="P. aeruginosa",
        citation="Lee SA, et al. Transposon sequencing of Pseudomonas aeruginosa PAO1 across three media. 2015; full citation not captured.",
        url="https://doi.org/10.1371/journal.pcbi.1013945",
        klass="Essentiality",
        tier=4,
        genotypes_n=352,
        genotypes="352 genes called essential",
        env_n=3,
        env="three media",
        instances_n=352,
        instances_basis="reported",
        phenotype="essentiality call",
        shape="boolean per gene",
        seq_basis="PAO1+transposon",
        why="The PAO1 contribution to the cross-strain gold set, and one of only two of the five studies scoped to that reference, which is what lets GOLD_84 carry PAO1 locus tags at all. Its accession is not given by the compiling paper.",
        accession="not given by the compiling paper; the accession must be chased from the primary study",
        gene_namespace="PAO1 PA numbers",
        retrievability="manual_fetch",
        verified="verified",
        confidence="recall",
    ),
    Candidate(
        name="Poulsen 2019 (nine-strain Tn-seq core essentiality)",
        organism="P. aeruginosa",
        citation="Poulsen BE, et al. Defining the core essential genome of Pseudomonas aeruginosa. bioRxiv 2019;396689.",
        url="https://www.biorxiv.org/content/10.1101/396689v2.full.pdf",
        klass="Essentiality",
        tier=2,
        genotypes_n=321,
        genotypes="321-gene core set across nine strains",
        env_n=5,
        env="five media",
        instances_n=321,
        instances_basis="reported",
        phenotype="essentiality call, three calling methods",
        shape="boolean per gene",
        seq_basis="PA14+transposon",
        why="The closest thing to sequenced-panel functional genomics in this species: nine strains in five media, with three calling methods reported separately rather than merged, which is exactly the shape a typed essentiality record wants. It also introduced the calling tool the gold-standard comparison benchmarks against.",
        accession="not given by the compiling paper; the preprint is open",
        gene_namespace="PA14 locus tags",
        retrievability="manual_fetch",
        verified="snippet",
    ),
    Candidate(
        name="Breidenstein 2008 (ciprofloxacin resistome)",
        organism="P. aeruginosa",
        citation="Breidenstein EBM, Khaira BK, Wiegand I, Overhage J, Hancock REW. Antimicrob Agents Chemother 2008;52:4486-4491.",
        url="https://doi.org/10.1128/AAC.00222-08",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=114,
        genotypes="114 mutants with altered susceptibility, 35 more and 79 less susceptible",
        env_n=1,
        env="ciprofloxacin",
        instances_n=114,
        instances_basis="reported",
        phenotype="altered antimicrobial susceptibility",
        shape="scalar",
        seq_basis="PA14+transposon",
        why="Small, fully released inside the article's own tables, and the canonical worked example of the namespace mismatch an alias layer has to handle: it reports a PA14 library's hits under the other reference strain's gene numbers. The total number of mutants screened is not stated.",
        accession="per-mutant values in Tables 1 and 2 of the article; no repository",
        gene_namespace="PAO1 PA numbers used for a PA14 library, which is the alias hazard itself",
        verified="verified",
    ),
    Candidate(
        name="PAO1 arrayed transposon libraries (Jacobs 2003, Held 2012)",
        organism="P. aeruginosa",
        citation="Jacobs MA, et al. Proc Natl Acad Sci USA 2003;100:14339-14344; Held K, Ramage E, Jacobs M, Gallagher L, Manoil C. J Bacteriol 2012;194.",
        url="https://doi.org/10.1128/JB.01479-12",
        klass="Modality / backbone",
        tier=3,
        genotypes_n=9437,
        genotypes="42,240 archived mutants, reduced to a sequence-verified two-allele set of 9,437",
        env_n=1,
        env="none; a genotype resource",
        instances_n=9437,
        instances_basis="reported",
        phenotype="insertion position per mutant, sequence-verified",
        shape="genotype record",
        seq_basis="PAO1+transposon",
        why="The genotype backbone for the other reference strain, with two alleles for most genes and roughly 8 percent of the original insertion assignments corrected, which is the kind of correction a sequence-level store has to carry rather than average away. The downloadable part is the insertion metadata; the strains are order-only.",
        accession="insertion-position metadata from the library index page; strains are order-only through a cost center and distribution has moved labs",
        gene_namespace="PAO1 PA numbers",
        retrievability="manual_fetch",
        verified="verified",
    ),
    Candidate(
        name="BACTOME (101 clinical genomes with eight phenotypes)",
        organism="P. aeruginosa",
        citation="Hornischer K, et al. BACTOME: a reference database to explore the sequence- and gene expression-variation landscape of Pseudomonas aeruginosa clinical isolates. Nucleic Acids Res 2019;47:D716.",
        url="https://doi.org/10.1093/nar/gky895",
        klass="Natural variation",
        tier=3,
        genotypes_n=101,
        genotypes="99 clinical isolates plus the two reference strains; 18,319 pangenome genes; 96 transcriptomes",
        env_n=1,
        env="LB at OD 2.0",
        instances_n=101,
        instances_basis="reported",
        phenotype="Galleria virulence, biofilm, colony morphology, resistance to 5 drugs",
        shape="vector of 8 phenotypes",
        dim=8,
        seq_basis="pseudomonad-isolate-WGS",
        why="A single stated environment across a hundred isolates is close to ideal for a typed record, and it is the only row carrying an insect-virulence readout. No bulk download is mentioned and no sequence accession appears on the landing page, so it is web-interface-only and only core genes identical in length across all genomes are included.",
        accession="no bulk download and no ENA, SRA or GEO accession on the landing page; not confirmed bulk-retrievable",
        gene_namespace="the PA14 reference used for mapping, with no stated scheme",
        retrievability="ask",
        ask_of="Helmholtz Centre for Infection Research, the BACTOME maintainers",
        verified="inferred",
    ),
    Candidate(
        name="Kohlstedt 2019 (13C metabolic flux analysis, two hosts)",
        organism="P. putida",
        citation="Kohlstedt M, Wittmann C. Metab Eng 2019;54:35-53.",
        url="https://doi.org/10.1016/j.ymben.2019.01.008",
        klass="Metabolome / flux",
        tier=3,
        genotypes_n=2,
        genotypes="KT2440 and PAO1, both wild type",
        env_n=1,
        env="glucose",
        instances_n=534,
        instances_basis="reported",
        phenotype="intracellular flux distribution from mass isotopomers",
        shape="vector over reactions",
        seq_basis="KT2440-reference",
        why="The only side-by-side flux comparison of the two hosts, resolving the parallel and cyclic glucose routes and showing the EDEMP cycle to be essential for pentose-phosphate supply. Where the numeric flux values live, in figures or in a machine-readable supplement, is unconfirmed because both the publisher and the institutional copy refused the fetch.",
        accession="unconfirmed: the publisher returned 403 and the institutional PDF reset the connection",
        gene_namespace="not applicable; the record is reaction-level",
        retrievability="manual_fetch",
        verified="snippet",
    ),
    Candidate(
        name="Wilkes 2026 (muconate multi-omics with fluxomics)",
        organism="P. putida",
        citation="Wilkes RA, Suthers PF, Borchert AJ, Callaghan MM, Thusoo E, Giannone RJ, et al. Metab Eng 2026;99:102573.",
        url="https://doi.org/10.1016/j.ymben.2026.102573",
        klass="Metabolome / flux",
        tier=3,
        genotypes_n=2,
        genotypes="KT2440 wild type and the muconate producer CJ781",
        env_n=2,
        env="glucose and 4-hydroxybenzoate",
        instances_n=4,
        instances_basis="product",
        phenotype="proteome, metabolome and 13C flux per strain and condition",
        shape="three layers per arm",
        dim=3,
        seq_basis="engineered-chassis",
        why="The only multi-omics plus flux dataset on a named engineered producer in this wave, reporting roughly threefold higher intracellular ATP and 36.3 g/L muconate at 1.1 g/L/h. The record is not open access and the index reports neither data nor supplementary files, so its accessions are the highest-value open ones left to chase.",
        accession="none confirmed; the bibliographic record reports no data and no supplementary files",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        retrievability="manual_fetch",
        verified="verified",
    ),
    Candidate(
        name="Werner 2023 (beta-ketoadipate from lignin)",
        organism="P. putida",
        citation="Werner AZ, Cordell WT, Lahive CW, Klein BC, Singer CA, Tan ECD, et al., Beckham GT. Sci Adv 2023;9:eadj0053.",
        url="https://doi.org/10.1126/sciadv.adj0053",
        klass="Production campaign",
        tier=3,
        genotypes_n=11,
        genotypes="10 named engineered genotypes plus wild type, with named deletions and chromosomal insertions",
        env_n=8,
        env="shake flask, bioreactor on 3 feedstocks, two fed-batch modes, pretreatment liquor, corn stover",
        instances_n=30,
        instances_basis="estimate",
        phenotype="titer, rate and molar yield",
        shape="3 quantities per arm",
        dim=3,
        seq_basis="engineered-chassis",
        why="The one engineered-strain paper whose data-availability statement was read verbatim, and the best-documented genotype set in the band: named deletions and insertions at named loci give real sequence-level material, reaching 44.5 g/L in fed-batch. Every number is in one supplement archive with no repository of any kind, and the strains themselves are request-only under a material transfer agreement.",
        accession="sciadv.adj0053_data_s1.zip plus the SI PDF; no repository; plasmids deferred to Tables S1 to S3",
        gene_namespace="KT2440 gene names rather than locus tags",
        retrievability="manual_fetch",
        verified="verified",
    ),
    Candidate(
        name="Kohlstedt 2022 (genome-reduced MA-10 muconate for PET)",
        organism="P. putida",
        citation="Kohlstedt M, Weimer A, Weiland F, Stolzenberger J, Selzer M, Sanz M, Kramps L, Wittmann C. Metab Eng 2022;72:337-352.",
        url="https://doi.org/10.1016/j.ymben.2022.05.001",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="genome-reduced MA-10 lineage; distinct-genotype count unconfirmed",
        env_n=1,
        env="catechol plus glucose",
        instances_n=3,
        instances_basis="estimate",
        phenotype="titer and productivity",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="The highest muconate titer in the lineage at up to 74 g/L and 1.4 g/L/h, on a genome-reduced chassis, which makes the genome reduction itself a perturbation the schema would have to express. Strain-table completeness and supplement format are unconfirmed because the publisher refused every fetch.",
        accession="unconfirmed; publisher returned 403",
        gene_namespace="gene names rather than locus tags",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Kohlstedt 2018 (MA-6 muconate to nylon)",
        organism="P. putida",
        citation="Kohlstedt M, Starck S, Barton N, Stolzenberger J, Selzer M, Mehlmann K, et al., Wittmann C. Metab Eng 2018;47:279-293.",
        url="https://doi.org/10.1016/j.ymben.2018.03.003",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="MA-6 lineage; distinct-genotype count unconfirmed",
        env_n=1,
        env="catechol, fed-batch",
        instances_n=3,
        instances_basis="estimate",
        phenotype="titer and productivity",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="The 64.2 g/L predecessor of the genome-reduced strain above, which together give a two-point lineage on the same product and the same feedstock. The honest number for a real lignin stream is roughly 13 g/L against 65 from added catechol, which is the gap any production record has to carry.",
        accession="unconfirmed; publisher returned 403",
        gene_namespace="gene names rather than locus tags",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Vardon 2015 (adipic acid from lignin)",
        organism="P. putida",
        citation="Vardon DR, et al. Adipic acid production from lignin. Energy Environ Sci 2015.",
        url="https://doi.org/10.1039/C4EE03230F",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=1,
        env="a model lignin monomer",
        instances_n=2,
        instances_basis="estimate",
        phenotype="titer over time",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="The earliest point on the muconate lineage at 13.5 g/L in 78.5 hours, which is what makes the later 64 and 74 g/L figures a trajectory rather than isolated records. Contents unconfirmed because the publisher refused the fetch.",
        accession="unconfirmed; publisher returned 403",
        gene_namespace="gene names rather than locus tags",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Woodworth 2021 (syringyl lignin to pyrone dicarboxylic acid)",
        organism="P. putida",
        citation="Woodworth BD, et al., Beckham GT. Metabolism of syringyl lignin-derived compounds in Pseudomonas putida enables convergent production of 2-pyrone-4,6-dicarboxylic acid. Metab Eng 2021.",
        url="https://www.sciencedirect.com/science/article/abs/pii/S1096717621000276",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=1,
        env="a lignin-monomer mixture",
        instances_n=1,
        instances_basis="estimate",
        phenotype="molar conversion to product",
        shape="scalar",
        seq_basis="engineered-chassis",
        why="A second product class on the same feedstock, with 93 percent of a lignin-monomer mixture converted, which is the convergent-catabolism claim the host is engineered for. The identifier is unconfirmed and the values were not located.",
        accession="unconfirmed; identifier not confirmed",
        gene_namespace="gene names rather than locus tags",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Isoprenol CRISPRi DBTL proteome ladder (7 PRIDE deposits)",
        organism="P. putida",
        citation="Automation of a CRISPRi platform for enhanced isoprenol production in Pseudomonas putida; PRIDE projects PXD063733, PXD063737, PXD063738, PXD063740, PXD063743, PXD063744, PXD063746.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD063733",
        klass="Proteome",
        tier=3,
        genotypes_n=None,
        genotypes="one engineered genotype set per design round; counts not reported per round",
        env_n=7,
        env="7 design-build-test-learn rounds",
        instances_n=7,
        instances_basis="reported",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=5564,
        seq_basis="engineered-chassis+guide",
        why="The most database-shaped proteomics series in this host: seven deposits indexed one per design round, so the engineered genotype and the molecular readout are both public and already separated by round. It is the best candidate in the wave for a genotype by environment by molecular-phenotype record, and the campaign is CRISPRi-driven, so the guide is part of the genotype.",
        accession="PRIDE PXD063733, PXD063737, PXD063738, PXD063740, PXD063743, PXD063744 and PXD063746, all submitted 2025-05-07 and 08 on an Orbitrap Exploris 480",
        gene_namespace="KT2440 PP_ locus tags through peptide to protein mapping",
        verified="inferred",
    ),
    Candidate(
        name="PXD067010 (lignocellulosic sugars to an aviation-fuel precursor)",
        organism="P. putida",
        citation="PRIDE project PXD067010, multi-layered metabolic remodeling for lignocellulosic sugars to an aviation-fuel precursor, 2025-08-06.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD067010",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="lignocellulosic sugar feeds, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="engineered-chassis",
        why="The proteomics layer under the host's aviation-fuel precursor line, on a lignocellulosic feed rather than glucose, which is the feedstock the engineering program is judged on. Only the deposit record was read.",
        accession="PRIDE PXD067010; retrievable through the PRIDE API",
        gene_namespace="KT2440 PP_ locus tags through peptide to protein mapping",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PXD054609 (isoprenol tolerance, evolution plus omics)",
        organism="P. putida",
        citation="PRIDE project PXD054609, adaptive laboratory evolution plus omics for isoprenol tolerance engineering, 2024-08-05.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD054609",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="evolved clones; count not reported",
        env_n=None,
        env="isoprenol challenge, count not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="evolved-WGS",
        why="The tolerance half of the isoprenol program, where the molecule is the stressor rather than the product, with evolved clones behind it. An evolved genotype only has a sequence basis if the clones were resequenced, which the deposit record does not state.",
        accession="PRIDE PXD054609; retrievable through the PRIDE API",
        gene_namespace="KT2440 PP_ locus tags through peptide to protein mapping",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PXD045993 (polyhydroxyalkanoate biology in KT2440)",
        organism="P. putida",
        citation="Kelly S, et al. Comprehensive polyhydroxyalkanoate biology in Pseudomonas putida KT2440. Mol Cell Proteomics 2024;23:100765; PRIDE PXD045993.",
        url="https://doi.org/10.1016/j.mcpro.2024.100765",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="KT2440-reference",
        why="The one PRIDE deposit in this host with a named peer-reviewed paper behind it, on the polymer pathway that is the host's oldest product line. Counts and conditions were not read.",
        accession="PRIDE PXD045993; retrievable through the PRIDE API",
        gene_namespace="KT2440 PP_ locus tags through peptide to protein mapping",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Remaining P. putida PRIDE proteomics projects",
        organism="P. putida",
        citation="About 35 PRIDE projects returned by a keyword search of the PRIDE v3 REST API; 26 were read and named, 7 remain unverified, and no second pass was made on the strain-name keywords.",
        url="https://www.ebi.ac.uk/pride/ws/archive/v3/search/projects?keyword=Pseudomonas%20putida",
        klass="Proteome",
        tier=4,
        genotypes_n=None,
        genotypes="varies per project; three projects hold several organisms and must be split",
        env_n=None,
        env="carbon sources, fluoride, phage infection, pathway engineering",
        instances_n=26,
        instances_basis="estimate",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="KT2440-reference",
        why="The corpus behind the named deposits, which is not net new until it is split by source and by organism: three of its projects hold several species and two are in backgrounds other than the reference strain. The API is the only workable route because the aggregator portal renders in JavaScript.",
        accession="PRIDE v3 REST API; seven candidate identifiers are unverified and must be re-checked before any ingest",
        gene_namespace="KT2440 PP_ locus tags where the organism is the reference strain",
        status="aggregation",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PXD074559 (JAG1 proteome, acetate against glucose)",
        organism="P. putida",
        citation="PRIDE project PXD074559, JAG1 proteome on acetate against glucose in a KT2440 background, 2026-02-17.",
        url="https://www.ebi.ac.uk/pride/archive/projects/PXD074559",
        klass="Proteome",
        tier=4,
        genotypes_n=1,
        genotypes="JAG1, a KT2440-background strain",
        env_n=2,
        env="acetate and glucose",
        instances_n=2,
        instances_basis="reported",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="engineered-chassis",
        why="A clean two-condition carbon-source contrast on one genotype, which is the simplest shape a proteome row can take and therefore the easiest to verify a loader against. Only the deposit record was read.",
        accession="PRIDE PXD074559; retrievable through the PRIDE API",
        gene_namespace="KT2440 PP_ locus tags through peptide to protein mapping",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="n-butanol KT2440 proteome (2015)",
        organism="P. putida",
        citation="Analysis of the molecular response of Pseudomonas putida KT2440 to the next-generation biofuel n-butanol. J Proteomics 2015.",
        url="https://biochemie.uni-hohenheim.de/en/organization/publication/analysis-of-the-molecular-response-of-pseudomonas-putida-kt2440-to-the-next-generation-biofuel-n-butanol",
        klass="Proteome",
        tier=4,
        genotypes_n=1,
        genotypes="KT2440 wild type",
        env_n=1,
        env="n-butanol challenge",
        instances_n=1467,
        instances_basis="reported",
        phenotype="relative protein abundance",
        shape="vector over proteins",
        dim=1,
        seq_basis="KT2440-reference",
        why="1,467 proteins quantified, about 28 percent of this host's proteome, under a solvent stress the biofuel program cares about directly. Authors, identifier, condition count and accession were all unresolved.",
        accession="not reported",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="QconCAT absolute quantification workflow (2023)",
        organism="P. putida",
        citation="QconCAT absolute-quantification workflow for Pseudomonas putida KT2440. STAR Protoc 2023.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC9881405",
        klass="Proteome",
        tier=4,
        genotypes_n=1,
        genotypes="KT2440",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="absolute protein quantification",
        shape="vector over proteins",
        dim=1,
        seq_basis="KT2440-reference",
        why="The second absolute-quantification route in this host, so it carries units rather than ratios, and its code is public. No proteomics accession was found, which is the same gap the other absolute-quantification row has.",
        accession="code at biosustain/QconCATquantSTAR; no PRIDE accession found",
        gene_namespace="KT2440 PP_ locus tags",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="MTBLS12468 (GAPDH homologs on glucose against ferulate)",
        organism="P. putida",
        citation="MetaboLights study MTBLS12468, GAPDH homologs switching between glucose and ferulate metabolism in KT2440, LC-MS negative mode, 2025-11-03.",
        url="https://www.ebi.ac.uk/metabolights/MTBLS12468",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=2,
        env="glucose and ferulate",
        instances_n=None,
        instances_basis="estimate",
        phenotype="metabolite intensities",
        shape="vector over metabolites",
        dim=1,
        seq_basis="KT2440-reference",
        why="A metabolome layer on the aromatic against sugar switch that the flux rows describe, in the host's reference strain. The accession reached this table through an aggregator snippet rather than the study page, so it must be verified at the repository before use.",
        accession="MetaboLights MTBLS12468; unverified, reached through an aggregator",
        gene_namespace="not applicable; the record is metabolite-level",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="MTBLS1715 (13C metabolomics under iron limitation)",
        organism="P. putida",
        citation="MetaboLights study MTBLS1715, 13C metabolomics of iron-replete against iron-limited pseudomonads including KT2440 and S12, 2020-12-05.",
        url="https://www.ebi.ac.uk/metabolights/MTBLS1715",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=None,
        genotypes="several pseudomonads including KT2440 and S12",
        env_n=2,
        env="iron-replete and iron-limited",
        instances_n=None,
        instances_basis="estimate",
        phenotype="labeled metabolite intensities",
        shape="vector over metabolites",
        dim=1,
        seq_basis="KT2440-reference",
        why="A micronutrient-limitation axis with isotope labeling, across more than one strain, which no other metabolome row in the wave carries. The deposit mixes strains and the accession is unverified, so it must be split and re-checked.",
        accession="MetaboLights MTBLS1715; unverified, reached through an aggregator",
        gene_namespace="not applicable; the record is metabolite-level",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="MTBLS319 and MTBLS320 (DOT-T1E solvent and drug exposure)",
        organism="P. putida",
        citation="MetaboLights studies MTBLS319 (toluene) and MTBLS320 (propranolol) in Pseudomonas putida DOT-T1E, 2016-08-23.",
        url="https://www.ebi.ac.uk/metabolights/MTBLS319",
        klass="Metabolome / flux",
        tier=4,
        genotypes_n=1,
        genotypes="DOT-T1E, not the reference strain",
        env_n=2,
        env="toluene and propranolol",
        instances_n=None,
        instances_basis="estimate",
        phenotype="metabolite intensities",
        shape="vector over metabolites",
        dim=1,
        seq_basis="pseudomonad-reference",
        why="The solvent-tolerant strain of this species under two chemically unrelated exposures, which widens the host from its reference strain on the metabolome axis. Both accessions are unverified and neither strain is the reference, so they cannot be merged with the KT2440 rows.",
        accession="MetaboLights MTBLS319 and MTBLS320; unverified, reached through an aggregator",
        gene_namespace="not applicable; the record is metabolite-level",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="GSE198395 (putidaPRECISE321 carbon sources)",
        organism="P. putida",
        citation="GEO series GSE198395, KT2440 and engineered strains on different carbon sources, part of putidaPRECISE321, released 2022-04-28.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198395",
        klass="Transcriptome",
        tier=3,
        genotypes_n=None,
        genotypes="KT2440 and engineered strains; genotype count not reported",
        env_n=None,
        env="different carbon sources, count not reported",
        instances_n=81,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="engineered-chassis",
        why="The largest single deposit inside the host's expression compendium, at 81 samples, and one of only two of its deposits named in the text rather than in a spreadsheet. It is nested inside the compendium row, so admitting both would double-count unless the compendium is split by deposit.",
        accession="GEO GSE198395; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE193493 (putidaPRECISE321 second deposit)",
        organism="P. putida",
        citation="GEO series GSE193493, part of putidaPRECISE321, released 2022-04-28.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193493",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="KT2440 and engineered strains; genotype count not reported",
        env_n=None,
        env="not reported",
        instances_n=22,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="engineered-chassis",
        why="The second named deposit of the expression compendium, which together with its sibling accounts for 103 of the 321 profiles and leaves the rest addressable only through a spreadsheet. Same nesting caveat.",
        accession="GEO GSE193493; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE213139 (low-dose ionizing radiation)",
        organism="P. putida",
        citation="GEO series GSE213139, low-dose ionizing radiation, released 2025-03-31.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213139",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="radiation doses, count not reported",
        instances_n=30,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-reference",
        why="A physical stressor rather than a chemical one, which is an environment class the whole wave otherwise lacks, at 30 samples. Sample metadata was read from the index rather than the series page.",
        accession="GEO GSE213139; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE131229 (non-antibiotic pharmaceutical stress)",
        organism="P. putida",
        citation="GEO SuperSeries GSE131229, non-antibiotic pharmaceutical stress.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131229",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="pharmaceutical exposures, count not reported",
        instances_n=30,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-reference",
        why="A chemical-exposure panel outside the antibiotic class, at 30 samples, which widens the environment vocabulary rather than the genotype one. It is a SuperSeries, so it has to be split into its component series before ingest.",
        accession="GEO GSE131229; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE248909 (micro and nanoplastic stress)",
        organism="P. putida",
        citation="GEO series GSE248909, micro and nanoplastic stress.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE248909",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="plastic particle exposures, count not reported",
        instances_n=28,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-reference",
        why="A particulate rather than a dissolved stressor, at 28 samples, on the host whose industrial interest is plastic depolymerization. Metadata was read from the index.",
        accession="GEO GSE248909; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE221044 (kaiC deletion)",
        organism="P. putida",
        citation="GEO series GSE221044, a kaiC deletion background.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE221044",
        klass="Transcriptome",
        tier=4,
        genotypes_n=2,
        genotypes="wild type and a kaiC deletion",
        env_n=None,
        env="not reported",
        instances_n=22,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-KO",
        why="One of the few expression deposits in this host whose variable is a named deletion rather than a condition, which is the genotype axis the fitness rows have and the expression rows mostly do not. 22 samples.",
        accession="GEO GSE221044; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE275525 (HexR regulatory landscape)",
        organism="P. putida",
        citation="GEO series GSE275525, the HexR regulatory landscape, released 2026-08-20.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE275525",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="not reported",
        instances_n=16,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-KO",
        why="A regulator-centered deposit, which is the shape that makes a regulatory network recoverable rather than inferred, and recent enough that no compendium holds it. 16 samples.",
        accession="GEO GSE275525; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE309641 (fluoride adaptive evolution)",
        organism="P. putida",
        citation="GEO series GSE309641, fluoride adaptive evolution, released 2026-02-26.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE309641",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="evolved clones; count not reported",
        env_n=None,
        env="fluoride, count not reported",
        instances_n=16,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="evolved-WGS",
        why="An evolution deposit with an expression readout, which is the one combination that can connect a mutation to a transcriptional state in this host. Whether the evolved clones were resequenced, which is what gives the row a sequence basis at all, is not stated on the series record.",
        accession="GEO GSE309641; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE155767 (evolved KT2440 on non-native sugars)",
        organism="P. putida",
        citation="GEO series GSE155767, evolved KT2440 on non-native sugars.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155767",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="evolved clones; count not reported",
        env_n=None,
        env="non-native sugars, count not reported",
        instances_n=14,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="evolved-WGS",
        why="A substrate-range evolution deposit, which is the experiment the host's feedstock flexibility claims rest on. The same resequencing question applies as for the fluoride row.",
        accession="GEO GSE155767; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE147420 (ammonium against nitrate)",
        organism="P. putida",
        citation="GEO series GSE147420, ammonium against nitrate.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE147420",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=2,
        env="ammonium and nitrate",
        instances_n=14,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="KT2440-reference",
        why="A nitrogen-source contrast on expression, which pairs directly with the 71-condition nitrogen fitness panel on the same host and the same gene axis. 14 samples.",
        accession="GEO GSE147420; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="GSE144045 (solvent-tolerance evolution in P. putida S12)",
        organism="P. putida",
        citation="GEO series GSE144045, solvent-tolerance adaptive evolution in Pseudomonas putida S12.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144045",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="evolved S12 clones; count not reported",
        env_n=None,
        env="solvent challenge, count not reported",
        instances_n=18,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="pseudomonad-reference",
        why="The only expression deposit in a non-reference strain of this host, which is what keeps the species from collapsing to one genome on the expression axis. 18 samples.",
        accession="GEO GSE144045; programmatically retrievable",
        gene_namespace="not stated; the S12 annotation is not the reference strain's",
        verified="verified",
    ),
    Candidate(
        name="GSE229763 (one-carbon metabolism)",
        organism="P. putida",
        citation="GEO series GSE229763, one-carbon metabolism.",
        url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE229763",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="not reported",
        env_n=None,
        env="one-carbon substrates, count not reported",
        instances_n=12,
        instances_basis="reported",
        phenotype="RNA-seq expression",
        shape="vector over genes",
        dim=5564,
        seq_basis="engineered-chassis",
        why="A one-carbon substrate arm, which is the frontier feedstock class for this host and absent from every other row. 12 samples.",
        accession="GEO GSE229763; programmatically retrievable",
        gene_namespace="not stated on the series record",
        verified="verified",
    ),
    Candidate(
        name="Remaining P. putida GEO expression series",
        organism="P. putida",
        citation="118 GEO series are returned for this organism; 40 identifiers were enumerated, 20 of them with 12 or more samples, and 78 series remain unexamined.",
        url="https://www.ncbi.nlm.nih.gov/gds/?term=%22Pseudomonas+putida%22%5BOrganism%5D",
        klass="Transcriptome",
        tier=4,
        genotypes_n=None,
        genotypes="varies per series",
        env_n=None,
        env="varies per series",
        instances_n=78,
        instances_basis="estimate",
        phenotype="RNA-seq or array expression",
        shape="vector over genes",
        dim=1,
        seq_basis="KT2440-reference",
        why="The unexamined two thirds of this host's expression corpus, held so the number is on the record rather than implied. One enumerated series carries a summary describing a mammalian assay inconsistent with its own taxon, so the corpus needs per-series checking and not a bulk pull.",
        accession="GEO, by organism query; 78 of 118 series unexamined",
        gene_namespace="varies per series",
        status="aggregation",
        verified="verified",
        confidence="recall",
    ),
    Candidate(
        name="Sigma-factor regulon reconstruction (2026)",
        organism="P. putida",
        citation="System-level reconstruction of RpoD, RpoS and RpoN regulatory networks in Pseudomonas putida KT2440. J Biol Eng 2026.",
        url="https://doi.org/10.1186/s13036-026-00769-z",
        klass="Transcriptome",
        tier=4,
        genotypes_n=3,
        genotypes="three sigma-factor backgrounds",
        env_n=None,
        env="not reported",
        instances_n=None,
        instances_basis="estimate",
        phenotype="binding occupancy plus expression",
        shape="two layers over genes",
        dim=1,
        seq_basis="KT2440-KO",
        why="The only row in this host that pairs direct binding with expression over three global regulators, which is what separates a measured regulon from an inferred one. Accessions and counts are unconfirmed and one candidate series is a plausible but unverified match.",
        accession="unconfirmed; GSE322955 is a plausible but unverified match",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Fenster 2022 (KT2440 CRISPRi toolset with a product readout)",
        organism="P. putida",
        citation="Fenster J, Werner A, Tay JW, et al. Metab Eng Commun 2022;15:e00204.",
        url="https://doi.org/10.1016/j.mec.2022.e00204",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=None,
        genotypes="guide constructs against named targets; count not reported",
        env_n=1,
        env="p-coumarate",
        instances_n=None,
        instances_basis="estimate",
        phenotype="beta-ketoadipate production plus single-cell repression",
        shape="scalar",
        seq_basis="engineered-chassis+guide",
        why="The only row that pairs an engineered-product phenotype with a defined CRISPRi perturbation in this host, with repression characterized dynamically and at single-cell resolution including against essential genes. A toolset rather than a library, so its value is the paired readout and not the scale.",
        accession="open-access article; no deposit named",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Batianis 2020 (expanded CRISPRi toolbox)",
        organism="P. putida",
        citation="Batianis C, et al. An expanded CRISPRi toolbox for tunable control of gene expression in Pseudomonas putida. bioRxiv 2020;2020.03.04.975888.",
        url="https://www.biorxiv.org/content/10.1101/2020.03.04.975888v1",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=2,
        genotypes="knockdowns of pyrF and ftsZ",
        env_n=None,
        env="not reported",
        instances_n=2,
        instances_basis="reported",
        phenotype="growth rate and cell morphology under knockdown",
        shape="scalar",
        seq_basis="KT2440+guide",
        why="Two named essential-gene knockdowns with both a growth and a morphology readout, which is the only morphology measurement on this host anywhere in the wave. Tiny, and held because tunable repression of an essential gene is the modality the missing genome-wide library would use.",
        accession="preprint only; no deposit named",
        gene_namespace="KT2440 gene names",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Rhizosphere CRISPRi (2024)",
        organism="P. putida",
        citation="Optimized CRISPR interference system for investigating Pseudomonas alloputida genes involved in rhizosphere microbiome assembly. 2024; PMID 39163848.",
        url="https://pubmed.ncbi.nlm.nih.gov/39163848/",
        klass="CRISPR library screen",
        tier=4,
        genotypes_n=None,
        genotypes="essential-gene knockdowns; count not reported",
        env_n=1,
        env="plant rhizosphere",
        instances_n=None,
        instances_basis="estimate",
        phenotype="cells per root",
        shape="scalar",
        seq_basis="KT2440+guide",
        why="The only measurement in the wave taken in a plant rhizosphere rather than a flask, reporting a 10- to 100-fold reduction in cells per root under essential-gene repression. The environment is a living root, which is an environment class the schema has never held.",
        accession="not reported",
        gene_namespace="KT2440 PP_ locus tags by organism inference; not stated",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PAO1 ppk1 and ppk2 Biolog panels",
        organism="P. aeruginosa",
        citation="Polyphosphate kinase mutants of Pseudomonas aeruginosa PAO1 on Biolog PM1 to PM3. Biol Res 2015;48.",
        url="https://doi.org/10.1186/s40659-015-0012-0",
        klass="Natural variation",
        tier=4,
        genotypes_n=3,
        genotypes="ppk1 and ppk2 mutants plus the parent",
        env_n=3,
        env="Biolog PM1 and PM2 carbon plus PM3 nitrogen",
        instances_n=3,
        instances_basis="reported",
        phenotype="Biolog respiration per well",
        shape="vector over wells",
        dim=1,
        seq_basis="PAO1-reference",
        why="The only phenotype-microarray data found for this species, and it is a mutant against parent comparison rather than a wild-type panel, which makes it a perturbation row rather than a substrate-range row. Well counts and the released format are not reported.",
        accession="open-access article; deposit and format not reported",
        gene_namespace="PAO1 PA numbers",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="P. putida S12 phenotype microarray (190 carbon sources)",
        organism="P. putida",
        citation="Genome-scale model validation against a 190-carbon-source phenotype microarray in Pseudomonas putida S12. BMC Genomics 2023;24.",
        url="https://doi.org/10.1186/s12864-023-09940-y",
        klass="Natural variation",
        tier=4,
        genotypes_n=1,
        genotypes="S12, not the reference strain",
        env_n=190,
        env="190 carbon sources",
        instances_n=190,
        instances_basis="reported",
        phenotype="phenotype-microarray carbon utilization",
        shape="scalar",
        seq_basis="pseudomonad-reference",
        why="A 190-source substrate range on a non-reference strain of this host, which is the comparison that tells whether the reference strain's catabolic breadth is typical. The release format was not confirmed.",
        accession="open-access article; deposit and format not reported",
        gene_namespace="not stated; the S12 annotation is not the reference strain's",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PMkbase v1.0 (phenotype-microarray database)",
        organism="P. putida",
        citation="PMkbase v1.0, a phenotype-microarray knowledge base made interoperable with sequence information, 2025.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12584640/",
        klass="Natural variation",
        tier=4,
        genotypes_n=None,
        genotypes="strain list not confirmed; whether any pseudomonad is included is unconfirmed",
        env_n=None,
        env="carbon substrates, count not reported per strain",
        instances_n=41664,
        instances_basis="reported",
        phenotype="phenotype-microarray carbon utilization",
        shape="scalar",
        seq_basis="pseudomonad-reference",
        why="Conceptually the closest thing to a ready-made ingestion source for the phenotype-microarray layer, at roughly 41,664 carbon-substrate data points already joined to sequence. Worth exactly one manual visit, because the publisher served a challenge and the identifier, the strain list, the resource address and the format are all unconfirmed.",
        accession="unconfirmed: the article page served a challenge, so no identifier, strain list, resource address or download format was read",
        gene_namespace="not confirmed",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Genome Medicine 2024 (1,877-strain genome and MIC set)",
        organism="P. aeruginosa",
        citation="Paired whole-genome and susceptibility data for 1,877 diverse strains plus a 102-strain validation set, with MICs reclassified per CLSI M100S-Ed32:2022. Genome Med 2024;16.",
        url="https://doi.org/10.1186/s13073-024-01346-z",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=1979,
        genotypes="1,877 strains plus a 102-strain validation set",
        env_n=None,
        env="antibiotic panel not read",
        instances_n=1979,
        instances_basis="reported",
        phenotype="susceptibility calls, reclassified per a named standard",
        shape="scalar",
        seq_basis="pseudomonad-isolate-WGS",
        why="The largest paired genome-plus-susceptibility set found anywhere in the wave, and it is blocked: the publisher redirects to an authentication endpoint, so the accessions, the antibiotic list and whether raw concentrations rather than category calls are released are all unconfirmed. That makes it the highest-value unresolved item on the panel side.",
        accession="unconfirmed; the publisher redirects to an authentication endpoint",
        gene_namespace="not confirmed",
        retrievability="manual_fetch",
        status="blocked",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="IPCD (International P. aeruginosa Consortium Database)",
        organism="P. aeruginosa",
        citation="Freschi L, Jeukens J, Kukavica-Ibrulj I, Boyle B, Dupont MJ, Laroche J, et al., Levesque RC. Front Microbiol 2015;6:1036.",
        url="https://doi.org/10.3389/fmicb.2015.01036",
        klass="Natural variation",
        tier=4,
        genotypes_n=3038,
        genotypes="3,038 isolates with draft genomes for all of them",
        env_n=None,
        env="phenotype fields not enumerated",
        instances_n=3038,
        instances_basis="reported",
        phenotype="linked phenotype, genotype and clinical data; fields not enumerated",
        shape="not reported",
        seq_basis="pseudomonad-isolate-WGS",
        why="The largest isolate count in the wave by an order of magnitude, and the clearest case of the pattern the curated portals share: good curation, no bulk export. It advertises no bulk table download and requires a sign-in, so its phenotype fields are not even enumerable without an ask.",
        accession="no advertised bulk download; sign-in required; not confirmed publicly bulk-downloadable",
        gene_namespace="not reported",
        retrievability="ask",
        ask_of="the IPCD consortium at Universite Laval (Levesque lab)",
        verified="verified",
    ),
    Candidate(
        name="Pirnay 2009 (328-isolate population survey)",
        organism="P. aeruginosa",
        citation="Pirnay JP, Bilocq F, Pot B, Cornelis P, Zizi M, Van Eldere J, et al. PLoS ONE 2009;4:e7740.",
        url="https://doi.org/10.1371/journal.pone.0007740",
        klass="Natural variation",
        tier=4,
        genotypes_n=328,
        genotypes="328 unrelated isolates from 69 localities in 30 countries",
        env_n=8,
        env="8 parameters per isolate, type not resolved",
        instances_n=328,
        instances_basis="reported",
        phenotype="8 parameters per isolate; whether they are measurements or typing markers is unresolved",
        shape="vector of 8",
        dim=8,
        seq_basis="pseudomonad-isolate-WGS",
        why="The broadest geographic sampling of the species anywhere in the wave, 30 countries, from the pre-sequencing era. Whether its eight parameters are measured phenotypes or typing markers was not resolved, so the row cannot yet be classified as a phenotype panel and is blocked on that question.",
        accession="open-access article; whether the 8 parameters are phenotypes is unresolved",
        gene_namespace="none; the study predates whole-genome sequencing of the panel",
        status="blocked",
        verified="verified",
    ),
    Candidate(
        name="Chandler 2023 (128 functionally phenotyped isolates)",
        organism="P. aeruginosa",
        citation="Chandler CE, et al. Microbiol Spectr 2023;11:e01556-23.",
        url="https://doi.org/10.1128/spectrum.01556-23",
        klass="Natural variation",
        tier=4,
        genotypes_n=128,
        genotypes="128 isolates sequenced with functional phenotyping",
        env_n=None,
        env="not reported",
        instances_n=128,
        instances_basis="reported",
        phenotype="functional phenotypes, not enumerated",
        shape="not reported",
        seq_basis="pseudomonad-isolate-WGS",
        why="A sequenced panel of the same order as the two best-documented clinical panels, so if its phenotype set overlaps theirs it extends them rather than duplicating. Data availability is unconfirmed and the phenotype list was not read.",
        accession="unconfirmed",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Nunez-Garcia 2024 (39 isolates over ten years)",
        organism="P. aeruginosa",
        citation="Nunez-Garcia L, et al. Microbiol Spectr 2024;12.",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC11619361/",
        klass="Natural variation",
        tier=4,
        genotypes_n=39,
        genotypes="39 isolates collected over ten years",
        env_n=1,
        env="growth-rate assay",
        instances_n=39,
        instances_basis="reported",
        phenotype="growth rate",
        shape="scalar",
        seq_basis="pseudomonad-isolate-WGS",
        why="A second longitudinal within-patient series beside the larger one, which is what would let a within-host adaptation signal be tested across cohorts rather than inside one. The sequence accession is unconfirmed.",
        accession="unconfirmed",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="Pediatric cystic fibrosis susceptibility panel (2025)",
        organism="P. aeruginosa",
        citation="Pediatric cystic fibrosis isolates with EUCAST susceptibility testing and whole-genome sequencing. Antibiotics 2025;14:302.",
        url="https://doi.org/10.3390/antibiotics14030302",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=None,
        genotypes="isolate count unconfirmed",
        env_n=None,
        env="EUCAST panel, drug list not read",
        instances_n=None,
        instances_basis="estimate",
        phenotype="susceptibility per a named standard",
        shape="scalar",
        seq_basis="pseudomonad-isolate-WGS",
        why="The only panel in the wave scored under the European rather than the American standard, which matters because the two standards place breakpoints differently and a merged susceptibility column would hide that. Isolate count unconfirmed.",
        accession="unconfirmed",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="PA14 non-redundant phage-resistance screen (2024)",
        organism="P. aeruginosa",
        citation="A phage-resistance screen of the PA14 non-redundant transposon library, 2024; authors and identifier unconfirmed.",
        url="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11241198/",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=None,
        genotypes="arrayed PA14 mutants; count unconfirmed",
        env_n=None,
        env="phage challenge, count unconfirmed",
        instances_n=None,
        instances_basis="estimate",
        phenotype="phage resistance per mutant",
        shape="scalar",
        seq_basis="PA14+transposon",
        why="A biotic selection against an arrayed library, which would be the only phage phenotype in the wave and the third screen indexed against that library's genotypes. The publisher served a challenge, so authors, identifier, mutant count, phage count and the location of the results are all unconfirmed.",
        accession="unconfirmed; the article page served a challenge",
        gene_namespace="PA14 predicted gene numbers",
        retrievability="manual_fetch",
        status="blocked",
        verified="snippet",
        confidence="recall",
    ),
    Candidate(
        name="OSTI barcoded P. fluorescens transposon libraries",
        organism="P. fluorescens",
        citation="A conference record describing barcoded transposon insertion libraries in Pseudomonas fluorescens with over 100,000 independent insertions, used to study root colonization; the strain is not stated.",
        url="https://www.osti.gov/biblio/1236050",
        klass="Transposon fitness",
        tier=4,
        genotypes_n=100000,
        genotypes="over 100,000 independent insertions; strain not stated",
        env_n=1,
        env="root colonization",
        instances_n=None,
        instances_basis="estimate",
        phenotype="colonization fitness",
        shape="vector over genes",
        dim=1,
        seq_basis="pseudomonad-reference",
        why="A barcoded library in this species screened in a plant root rather than a flask, which is the environment the Fitness Browser rows for it do not cover. The strain is not stated anywhere in the record, so the row cannot be assigned a reference until that is resolved.",
        accession="not reported; the strain is not stated",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        confidence="recall",
    ),
    # -- Already built: kept in the pool so the judgment is recorded --------
    #
    # ranked() drops these, and they are rows rather than Excluded entries because the
    # finding is that each LOOKED like a candidate: two appear in the triage notes as
    # new datasets and one is the artifact the whole cross-host fitness band extends.
    Candidate(
        name="Nadal-Ribelles 2025 (single-cell genotype-phenotype map)",
        organism="S. cerevisiae",
        citation="Nadal-Ribelles M, et al., Posas F. A single-cell resolved genotype-phenotype map using genome-wide genetic and environmental perturbations. Nat Commun 2025;16:2645.",
        url="https://doi.org/10.1038/s41467-025-57600-4",
        klass="Transcriptome",
        tier=1,
        genotypes_n=3500,
        genotypes="about 3,500 barcoded deletion mutants",
        env_n=2,
        env="control and stress",
        instances_n=6188,
        instances_basis="reported",
        phenotype="single-cell transcriptome",
        shape="vector over genes",
        dim=5818,
        seq_basis="S288C-KO",
        why="Already served at 6,188 instances with a 5,818-vector phenotype. It appears in the yeast gene-level-readout note as a new 3,500-mutant atlas, which is what made it look like a candidate, and it is an overlap. The actionable item is that its 6.6 GB in the raw mirror carries no manifest at all, not an ingest.",
        accession="served; the raw mirror holds 6.6 GB under nadalRibelles2025 with no manifest",
        gene_namespace="SGD S288C systematic gene names",
        status="built",
        verified="verified",
    ),
    Candidate(
        name="Price 2018 (feba.db, already in the raw tree)",
        organism="P. putida",
        citation="Price MN, Wetmore KM, Waters RJ, et al., Deutschbauer AM, Arkin AP. Nature 2018;557:503-509.",
        url="https://doi.org/10.1038/s41586-018-0124-0",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=11779,
        genotypes="11,779 genes with a phenotype and no specific annotation, 2,316 of them high confidence",
        env_n=None,
        env="the 32-organism June 2017 release",
        instances_n=11779,
        instances_basis="reported",
        phenotype="BarSeq competition fitness per gene",
        shape="vector over genes",
        dim=1,
        seq_basis="fitbrowser+transposon",
        why="Already in the raw tree as the 10.2 GB feba.db under priceMutantPhenotypesThousands2018, which is exactly what makes the cross-host fitness band an extension of a held key rather than a new ingest. Six of its nine files are unhashed, so the precondition for that band is the manifest, not the download.",
        accession="held: $DATA_ROOT/torchcell-raw/priceMutantPhenotypesThousands2018, 6 of 9 files unhashed over 10.21 GB",
        gene_namespace="Fitness Browser orgId plus locusId, bridged to RefSeq",
        status="built",
        verified="verified",
    ),
    Candidate(
        name="Vaishnav 2022 served subset",
        organism="S. cerevisiae",
        citation="Vaishnav ED, de Boer CG, Molinet J, et al., Regev A. Nature 2022;603:455-463; served by vaishnav2022.py.",
        url="https://doi.org/10.1038/s41586-022-04506-6",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=71158,
        genotypes="71,158 seqID groups of the defined-medium subset",
        env_n=1,
        env="SD-Ura defined",
        instances_n=71158,
        instances_basis="reported",
        phenotype="expression driven, 18-bin sort-seq YFP",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="Served: 71,158 records of VariablePromoterReporterPerturbation by PooledSortSeqExpressionPhenotype, 71,054 carrying an estimate and 104 seqID groups carrying a typed ProvenanceGap for verbatim NA cells. The candidate is the two deferred training libraries, which are a superset admission against exactly this subset.",
        accession="served; GEO GSE163045, the defined-medium molecular-evolution table",
        gene_namespace="none; the identifier is the measured 80mer",
        status="built",
        verified="verified",
    ),
    # -- The 3-HP product line, found by the 2026-11-24 fill pass -----------
    Candidate(
        name="Tan 2026 (I. orientalis 3-HP strain panel, Nat Commun)",
        organism="P. kudriavzevii",
        citation="Tan SI, Bhagwat SS, Martin TA, Suthers PF, Tran VG, Tang W, Fatma Z, Maranas CD, Guest JS, Zhao H. High yield production of 3-hydroxypropionic acid using Issatchenkia orientalis. Nat Commun 2026;17:899.",
        url="https://doi.org/10.1038/s41467-025-67621-8",
        klass="Production campaign",
        tier=4,
        genotypes_n=21,
        genotypes="21 strain configurations off SD108 ura3 and the LPB landing pad, each specified by gene, copy number and integration locus; the total is read off the figure's columns and is never stated",
        env_n=3,
        env="glucose in CSM and in corn steep liquor, across shake flask, 300 mL DASbox and 3 L bioreactor",
        instances_n=25,
        instances_basis="estimate",
        phenotype="titer, yield and productivity",
        shape="3 quantities",
        dim=3,
        seq_basis="SD108+designed-edit",
        why="The largest 3-HP strain panel in this host and the row the wave was missing: open access with a Source Data file, so the per-strain numbers are released rather than figure-bound. The panel spans 0.86 to 29 g/L and the fed-batch arms reach 92 g/L, 0.7 g/g and 0.55 g/L/h. Every genotype is written as gene plus copy number plus locus, and the heterologous decarboxylase and transaminase steps name their source organism and locus tag, so the perturbation is specified at the level the schema wants.",
        accession="open access at PMC12830792, with a 1.7 MB Source Data workbook and eight supplementary data files; the data-availability statement names NO repository, so the companion deposit is findable only through a DOI registry",
        gene_namespace="gene-name abbreviations with occasional accessions, including GE061_003755 of Apolygus lucorum; not the JGI gNNNN namespace",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the host's largest released strain panel for its second product, and the only one whose per-strain numbers are in a Source Data file rather than a figure.",
        new_class="Organism member",
    ),
    Candidate(
        name="Jia 2025 (3-HP from brewer's spent grain)",
        organism="P. kudriavzevii",
        citation="Jia L, Jeong D, Zhang M, Oh EJ, Zhao J. 3-hydroxypropionic acid production from Brewer's spent grain with an engineered Issatchenkia orientalis. J Biotechnol 2025;407:70-77.",
        url="https://doi.org/10.1016/j.jbiotec.2025.08.009",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="the IoDY01H background, off the NRRL Y-27441 lineage rather than SD108; the strain count is not stated",
        env_n=4,
        env="four comparison axes: the pH-adjusting acid, nitrogen supplementation, bicarbonate, and deproteinization of the hydrolysate",
        instances_n=None,
        instances_basis="estimate",
        phenotype="titer and yield",
        shape="2 quantities",
        dim=2,
        seq_basis="Pk-named+designed-edit",
        why="A fourth feedstock for the host's 3-HP line, brewer's spent grain, which no ranked row covers, at 7.8 and 8.7 g/L. Its value is the substrate axis and not a genotype axis: the strain count is never stated, the two titers appear in the abstract alone, and the indexer reports neither supplementary files nor data for it, so the released form is probably figure-only. It is also the row that establishes the second lineage, since this group builds on NRRL Y-27441 and not on SD108.",
        accession="no PMCID and no repository deposit found; the per-condition values are in the article body",
        gene_namespace="not reported; the xylose and beta-alanine genes are named without loci",
        retrievability="manual_fetch",
        verified="inferred",
        band="host bootstrap",
        band_why="Host bootstrap: the only brewer's spent grain feedstock row on the host.",
        new_class="Organism member",
    ),
    Candidate(
        name="Zhang 2026 (green pretreatment, 3-HP and ethanol)",
        organism="P. kudriavzevii",
        citation="Zhang M, Hwang HJ, Jeong D, Jia L, Oh EJ, Liu D, Zhao J. A green biomass pretreatment strategy to produce 3-hydroxypropionic acid and ethanol. Bioresour Technol 2026;441:133576.",
        url="https://doi.org/10.1016/j.biortech.2025.133576",
        klass="Production campaign",
        tier=4,
        genotypes_n=1,
        genotypes="one engineered strain, presumed the NRRL Y-27441 lineage and not stated in the reachable text",
        env_n=6,
        env="corn stover and brewer's spent grain, each at 5, 10 and 20 wt% solids, with and without carbonation",
        instances_n=6,
        instances_basis="product",
        phenotype="titer",
        shape="scalar",
        seq_basis="Pk-named+designed-edit",
        why="The solids-loading axis of the host's 3-HP line, at 6.9 g/L for this host at 20 wt% with 18.4 g/L ethanol beside it. The paper is multi-organism and only this arm is in scope: it carries a Rhodotorula toruloides 3-HP arm and an S. cerevisiae ethanol arm as well, so a loader has to split the release by organism rather than ingest it whole. The strain background could not be read, which is the row's other gap.",
        accession="no PMCID and no deposit found; values are in the article body",
        gene_namespace="not reported",
        retrievability="manual_fetch",
        verified="snippet",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only solids-loading and co-product row.",
        new_class="Organism member",
    ),
    Candidate(
        name="US9845484B2 (Novozymes and Cargill 3-HP strain tables)",
        organism="P. kudriavzevii",
        citation="US9845484B2. 3-hydroxypropionic acid production by recombinant yeasts expressing an insect ... (title truncated in the retrieved record). Novozymes A/S and Cargill Inc.",
        url="https://patents.google.com/patent/US9845484B2/en",
        klass="Production campaign",
        tier=4,
        genotypes_n=11,
        genotypes="11 named strains in the CNB1 background built from CD1822, each with locus-explicit edits carrying SEQ ID numbers",
        env_n=1,
        env="one production condition per table",
        instances_n=11,
        instances_basis="reported",
        phenotype="titer in g/L and yield",
        shape="2 quantities",
        dim=2,
        seq_basis="Pk-named+designed-edit",
        why="The second largest released 3-HP strain panel in this host after Tan 2026, with Tables 9, 10 and 15 giving concentration in g/L and yield per strain, and Table 18 giving g/L per OD. MeJi524 reaches 22.8 g/L. It is blocked on identity rather than on data: the served SourceType enum has no patent member, so this row cannot be cited until that is added.",
        accession="Tables 9, 10, 15 and 18 of the patent text; no repository deposit and no DOI",
        gene_namespace="SEQ ID numbers in the patent, with no public locus namespace",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: eleven strains with both titer and yield, which only Tan 2026 beats on this host.",
        new_class="SourceType patent member; Organism member",
    ),
    Candidate(
        name="US9365875B2 (Novozymes 3-HP per-biomass panel)",
        organism="P. kudriavzevii",
        citation="US9365875B2. 3-hydroxypropionic acid production by recombinant yeasts. Novozymes Inc.",
        url="https://patents.google.com/patent/US9365875B2/en",
        klass="Production campaign",
        tier=4,
        genotypes_n=4,
        genotypes="4 named CNB1-background strains with locus-explicit edits",
        env_n=1,
        env="one production condition",
        instances_n=4,
        instances_basis="reported",
        phenotype="3-HP per dry cell weight",
        shape="scalar",
        seq_basis="Pk-named+designed-edit",
        why="A small panel whose readout is normalized per biomass rather than per volume, so it cannot be pooled with the g/L rows without a stated conversion. MBin558 reads 2.96 against a 129 percent improvement over MeJi461. Blocked on the same missing SourceType member as its sibling patent.",
        accession="Tables 5 and 6 of the patent text; no deposit and no DOI",
        gene_namespace="SEQ ID numbers in the patent",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only per-biomass 3-HP normalization, which fixes the unit question for the other rows.",
        new_class="SourceType patent member; Organism member",
    ),
    Candidate(
        name="US10633664B2 (Cargill 3-HP tolerance dose response)",
        organism="P. kudriavzevii",
        citation="US10633664B2. Compositions and methods for 3-hydroxypropionic acid production. Cargill Inc.",
        url="https://patents.google.com/patent/US10633664B2/en",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=2,
        genotypes="2 named CNB1-background strains plus a null control, locus-explicit with SEQ ID numbers",
        env_n=3,
        env="3-HP in the medium at 0, 35 and 75 g/L",
        instances_n=9,
        instances_basis="product",
        phenotype="growth rate under 3-HP, with enzyme activity reported beside it",
        shape="scalar",
        seq_basis="Pk-named+designed-edit",
        why="A dose-response tolerance panel rather than a titer panel, which is the shape the host has almost none of: growth graded at three 3-HP concentrations, with strain PTA-6658 at 2.34. Titers appear only in prose, so the tolerance table is the ingestible part. Blocked on the missing SourceType member.",
        accession="the tolerance and enzyme-activity tables of the patent text; titers in prose only; no deposit and no DOI",
        gene_namespace="SEQ ID numbers in the patent",
        verified="verified",
        band="host bootstrap",
        band_why="Host bootstrap: the host's only graded 3-HP dose-response tolerance data.",
        new_class="SourceType patent member; Organism member",
    ),
    Candidate(
        name="Tan 2026 (corn kernels, 3-HP and ethanol coproduction)",
        organism="P. kudriavzevii",
        citation="Tan Y, Poudel A, Jeong D, Peng H, Qi J, Arabi N, Babaei S, Jia L, Zhang M, Arvelli S, Oh EJ, Zhao J. Biocatal Agric Biotechnol 2026;71:103906.",
        url="https://doi.org/10.1016/j.bcab.2025.103906",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="not established; the article could not be reached",
        env_n=None,
        env="corn kernels as feedstock; condition count not established",
        instances_n=None,
        instances_basis="estimate",
        phenotype="3-HP and ethanol titer",
        shape="2 quantities",
        seq_basis="Pk-named+designed-edit",
        why="Ranked for how it was found rather than for what is known about it, which is almost nothing: it is absent from both PubMed and Europe PMC, and a 31-hit cross-query over the species synonyms missed it entirely. It surfaced only through a DOI registry, its abstract is null in two indexes, and the publisher refuses automated fetch, so no titer, strain count or condition count is established. The lesson travels further than the row: any sweep built on Europe PMC alone will keep missing this journal, which is how a whole paper in a tracked product line stayed invisible.",
        accession="no deposit found; the abstract is null in two indexes and the publisher returns 403, so the article needs a browser or institutional access",
        gene_namespace="not established",
        retrievability="manual_fetch",
        verified="snippet",
        band="host bootstrap",
        band_why="Host bootstrap: a sixth 3-HP study on the host, and the one that proves the search surface itself was incomplete.",
        new_class="Organism member",
    ),
    # -- E. coli, the primary screens the LASER index pointed at ------------
    Candidate(
        name="Kim 2013 (472 knockouts, five fermentation products)",
        organism="E. coli",
        citation="Kim J, Hou W, Lee S, Kim H, Lee S, Lee J. Genome-wide analysis of redox reactions reveals metabolic engineering targets for d-lactate overproduction in Escherichia coli. Metab Eng 2013;18:44-52.",
        url="https://doi.org/10.1016/j.ymben.2013.03.004",
        klass="Fitness / chemical genomics",
        tier=4,
        genotypes_n=472,
        genotypes="472 single-gene knockouts from a cataloged collection",
        env_n=1,
        env="anaerobic fermentation",
        instances_n=472,
        instances_basis="product",
        phenotype="secreted lactate, succinate, acetate, formate and ethanol",
        shape="5 quantities",
        dim=5,
        seq_basis="ecoli-KO",
        why="The largest per-strain panel the LASER index points at, roughly 2,360 values behind the seven strains LASER itself curated, and the clearest case for reading an index rather than ingesting one. The abstract states the design in the authors' own words, analysing the genome-wide anaerobic fermentation products of 472 single-gene knockouts. The table holding the panel is UNLOCATED: the publisher returns 403 and there is no PMC copy, so this row is blocked on an institutional fetch rather than on a decision.",
        accession="no deposit; the 472-strain table was not located because the publisher refuses automated fetch and there is no PMC copy",
        gene_namespace="E. coli b-numbers through the knockout collection's own naming",
        retrievability="manual_fetch",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Valle 2015 (150 knockouts, hydrogen and ethanol on glycerol)",
        organism="E. coli",
        citation="Valle A, Cabrera G, Cantero D, Bolivar J. Identification of enhanced hydrogen and ethanol Escherichia coli producer strains in a glycerol-based medium by screening in single-knock out mutant collections. Microb Cell Fact 2015;14:93.",
        url="https://doi.org/10.1186/s12934-015-0285-6",
        klass="Fitness / chemical genomics",
        tier=4,
        genotypes_n=150,
        genotypes="150 single-gene knockouts drawn from both the Keio and the Yale CGSC collections, each in triplicate",
        env_n=1,
        env="glycerol-based medium",
        instances_n=150,
        instances_basis="product",
        phenotype="hydrogen, ethanol and glycerol consumption",
        shape="3 quantities",
        dim=3,
        seq_basis="ecoli-KO",
        why="The one screen behind the LASER index whose per-strain numbers are already public and already tabulated: Additional file 2 lists the strains, Additional file 3 the per-strain significance for all three analytes, and Additional file 5 the response across carbon sources and pH. LASER curated 12 confirmed hits from it and discarded the other 138, which is the index's whole failure mode in one row.",
        accession="open access at PMC4485358; per-strain values in Additional files 2, 3 and 5",
        gene_namespace="E. coli b-numbers, with strains named by both collections' identifiers",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Raman 2014 (multiplexed mutagenesis, naringenin and glucaric acid)",
        organism="E. coli",
        citation="Raman S, Rogers JK, Taylor ND, Church GM. Evolution-guided optimization of biosynthetic pathways. Proc Natl Acad Sci USA 2014;111:17803-17808.",
        url="https://doi.org/10.1073/pnas.1409523111",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="clones selected from a library the record gives as 10,000 members; how many carry a released phenotype was not established",
        env_n=1,
        env="one selection condition per product",
        instances_n=None,
        instances_basis="estimate",
        phenotype="naringenin and glucaric acid production",
        shape="scalar",
        seq_basis="evolved-WGS",
        why="The only entry in the LASER index whose selected genotypes were both tabulated and sequenced, with designed genotypes in Table S7, untargeted mutations in Table S8 and reads under BioProject 267705, which is what separates it from the enrichment selections that published a hit list alone. The count of strains carrying a released number is unresolved, so it is ranked without one rather than with a guess.",
        accession="open access at PMC4273373; genotypes in Table S7, untargeted mutations in Table S8, reads under BioProject 267705",
        gene_namespace="E. coli gene names in the supplementary tables",
        verified="snippet",
        band="scale",
    ),
    # -- S. cerevisiae 3-HP, found by the 2026-11-24 fill pass ---------------
    Candidate(
        name="Li 2025 (xylose deregulation panel with 3-HP readout)",
        organism="S. cerevisiae",
        citation="Li X, et al. Modular deregulation of central carbon metabolism for efficient xylose utilization in Saccharomyces cerevisiae. Nat Commun 2025;16:4551.",
        url="https://doi.org/10.1038/s41467-025-59966-x",
        klass="Production campaign",
        tier=4,
        genotypes_n=177,
        genotypes="177 labeled strains off the IMX581 CEN.PK113-5D lineage, each a stated module of central-carbon deregulations",
        env_n=1,
        env="xylose and glucose",
        instances_n=1194,
        instances_basis="product",
        phenotype="3-HP titer with xylose and glucose consumption",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The largest released S. cerevisiae 3-HP strain panel found anywhere, and one of only four in that whole literature whose per-strain numbers are machine readable rather than bar heights: the Source Data file carries 398 labeled rows at three replicates, which is 1,194 records under the one-record-per-released-replicate rule. The title leads with xylose, which is why two prior passes missed it, but 3-HP is named 104 times in the text and GSE151478 deposits the transcriptomes beside it.",
        accession="open access at PMC12084563; per-strain values in the Source Data file; transcriptomes at GEO GSE151478",
        gene_namespace="SGD systematic names for host loci; source-organism names for the heterologous steps",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Qin 2024 (CO2 fixation route to 3-HP)",
        organism="S. cerevisiae",
        citation="Qin N, et al. Increased CO2 fixation enables high carbon-yield production of 3-hydroxypropionic acid in yeast. Nat Commun 2024;15:1591.",
        url="https://doi.org/10.1038/s41467-024-45557-9",
        klass="Production campaign",
        tier=4,
        genotypes_n=71,
        genotypes="71 strains, QLW1 through QLW71 with no gaps, off CEN.PK113-11C, each with a cumulative genotype string",
        env_n=1,
        env="glucose with carbon-fixation modules",
        instances_n=71,
        instances_basis="reported",
        phenotype="titer and carbon yield",
        shape="2 quantities",
        dim=2,
        seq_basis="S288C+designed-edit",
        why="The first S. cerevisiae 3-HP row to ingest, and the only one whose Source Data holds per-REPLICATE values rather than bar heights, with a cumulative genotype string for every strain. 71 strains run QLW1 through QLW71 with no gaps, and titer moves from 0.14 to 11.25 g/L. Two passes missed it because its title says yeast rather than Saccharomyces. One parser trap travels with the row: QLW2 is written QLW2-HCO3- in the file, so a strain name has to be matched as a prefix or records drop silently.",
        accession="open access; per-strain, per-replicate values in the Source Data file; no repository deposit",
        gene_namespace="common gene names only, which is true of every yeast 3-HP source except Kildegaard 2014",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Tong 2021 (oxaloacetate route to 3-HP)",
        organism="S. cerevisiae",
        citation="Tong T, et al. A biosynthesis pathway for 3-hydroxypropionic acid production in genetically engineered Saccharomyces cerevisiae. Green Chem 2021;23:4502-4509.",
        url="https://doi.org/10.1039/d0gc04431h",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="a strain series decarboxylating oxaloacetate through KdcA; counts not reachable",
        env_n=1,
        env="glucose",
        instances_n=None,
        instances_basis="estimate",
        phenotype="titer and yield",
        shape="2 quantities",
        dim=2,
        seq_basis="S288C+designed-edit",
        why="A THIRD 3-HP route, through oxaloacetate rather than beta-alanine or malonyl-CoA, which is a separate pathway and so a separate schema leaf rather than another row on an existing one. It reaches 18.1 g/L at 0.125 g/g. It is invisible to both PubMed and Europe PMC because Green Chemistry is indexed in neither, which is how two passes missed an entire route, and the publisher returns 403 to every automated client.",
        accession="no deposit; the publisher returns 403 to automated fetch, so the tables need a browser or institutional access",
        gene_namespace="SGD systematic names; KdcA named by source organism",
        retrievability="manual_fetch",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Liu 2026 (transcription-factor rewiring for 3-HP)",
        organism="S. cerevisiae",
        citation="Liu J, et al. Global transcriptional rewiring and dual-compartment engineering for efficient 3-hydroxypropionic acid production. Metab Eng 2026;98:102527.",
        url="https://doi.org/10.1016/j.ymben.2026.102527",
        klass="Production campaign",
        tier=4,
        genotypes_n=None,
        genotypes="a transcription-factor mutagenesis series; counts not reachable behind the paywall",
        env_n=1,
        env="glucose",
        instances_n=None,
        instances_basis="estimate",
        phenotype="titer",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The highest S. cerevisiae 3-HP titer found, 81.8 g/L, and a mutagenesis design that makes it the likeliest large strain panel in the yeast half of this product line. Nothing about its size is verified: the article is paywalled with no repository copy, so the row exists to name the fetch rather than to claim a count.",
        accession="no deposit and no open copy; the article is paywalled",
        gene_namespace="SGD systematic names",
        retrievability="manual_fetch",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Borodina 2015 (beta-alanine route strain series)",
        organism="S. cerevisiae",
        citation="Borodina I, et al. Establishing a synthetic pathway for high-level production of 3-hydroxypropionic acid in Saccharomyces cerevisiae via beta-alanine. Metab Eng 2015;27:57-64.",
        url="https://doi.org/10.1016/j.ymben.2014.10.003",
        klass="Production campaign",
        tier=4,
        genotypes_n=35,
        genotypes="35 strains carrying the beta-alanine route, with at least 38 once Supplementary Figure S3 is counted",
        env_n=1,
        env="glucose",
        instances_n=35,
        instances_basis="reported",
        phenotype="titer",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The paper that established the beta-alanine route and the largest strain count in the yeast 3-HP literature outside Li 2025, but every per-strain value is a bar height: there is no table and no deposit, and the article has no PMC identifier despite a CC BY license. An ask is the honest route, since digitizing bars would invent precision the source never stated.",
        accession="CC BY with no PMCID; per-strain values exist only as figure bars, so the numbers behind them have to be requested",
        gene_namespace="SGD systematic names",
        retrievability="ask",
        ask_of="Irina Borodina, DTU Biosustain, corresponding author",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Jessop-Fabre 2019 (Crabtree-negative hydroxy-acid strain transcriptomes)",
        organism="S. cerevisiae",
        citation="Jessop-Fabre MM, et al. The transcriptome and flux profiling of Crabtree-negative hydroxy acid-producing strains of Saccharomyces cerevisiae. Biotechnol J 2019;14:e1900013.",
        url="https://doi.org/10.1002/biot.201900013",
        klass="Transcriptome",
        tier=4,
        genotypes_n=5,
        genotypes="5 hydroxy-acid-producing strains in a TAM pdc-null background",
        env_n=1,
        env="one growth condition, three replicates",
        instances_n=15,
        instances_basis="product",
        phenotype="transcript abundance",
        shape="vector over genes",
        dim=6607,
        seq_basis="S288C+designed-edit",
        why="The only 3-HP strain set in the yeast literature with a real sequence deposit rather than a figure, which is what makes it ingestible at all: ArrayExpress E-MTAB-7600 under PRJEB30893, five strains at three replicates. Its value is the expression axis on a production background, not the titer. Note that ENA PRJEB31874 cites this group's protocols but uses different strains, so it must not be attributed here.",
        accession="ArrayExpress E-MTAB-7600, BioProject PRJEB30893",
        gene_namespace="SGD systematic names",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Zhang 2023 (mitochondrial 3-HP compartment panel)",
        organism="S. cerevisiae",
        citation="Zhang Y, et al. Engineering yeast mitochondrial metabolism for 3-hydroxypropionate production. Biotechnol Biofuels Bioprod 2023;16:64.",
        url="https://doi.org/10.1186/s13068-023-02309-z",
        klass="Production campaign",
        tier=4,
        genotypes_n=38,
        genotypes="38 strains off the CEN.PK113-5D lineage, differing by which pathway steps are targeted to the mitochondrion",
        env_n=1,
        env="glucose",
        instances_n=22,
        instances_basis="reported",
        phenotype="titer",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The compartment axis of the yeast 3-HP line, which no other row carries: the same pathway run in the mitochondrion rather than the cytosol. It is PARTLY tabulated, which is why it is ranked rather than dropped, with 22 of its 38 strains readable as values and 7 of those in a supplementary table. The other 16 are figure-bound and need the same ask as Borodina 2015.",
        accession="open access at PMC10082987; 22 strains tabulated, 7 of them in the supplementary table, the rest figure-bound",
        gene_namespace="common gene names only",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Xu 2023 (heterologous carbon-fixation route, 3-HP and mesaconate)",
        organism="S. cerevisiae",
        citation="Xu S, et al. Exploiting a heterologous construction of the 3-hydroxypropionic acid carbon fixation pathway in Saccharomyces cerevisiae. Bioresour Bioprocess 2023;10:33.",
        url="https://doi.org/10.1186/s40643-023-00652-5",
        klass="Production campaign",
        tier=4,
        genotypes_n=59,
        genotypes="59 tabulated strains off CEN.PK2-1D, with the explicit genotype of each in Supplementary Table S2",
        env_n=1,
        env="glucose",
        instances_n=42,
        instances_basis="reported",
        phenotype="3-HP titer, with mesaconate titer on a subset",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The largest TABULATED strain list in the yeast 3-HP literature, 59 rows with an explicit genotype for every one in Supplementary Table S2, which is the field this product line is otherwise missing. 19 of them carry a 3-HP value and 42 measurements exist once mesaconate is counted, so the genotype axis is wider than the phenotype axis and the row is ranked on the smaller number.",
        accession="open access at PMC10991142; genotypes in Supplementary Table S2, values partly figure-bound",
        gene_namespace="common gene names only",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Qiu 2022 (biosensor-coupled in vivo mutagenesis, malonyl-CoA)",
        organism="S. cerevisiae",
        citation="Qiu C, et al. Biosensor-coupled in vivo mutagenesis for malonyl-CoA pathway optimization. mSystems 2022;7:e01366-21.",
        url="https://doi.org/10.1128/msystems.01366-21",
        klass="Production campaign",
        tier=4,
        genotypes_n=12,
        genotypes="12 mutants recovered from biosensor-coupled in vivo mutagenesis",
        env_n=1,
        env="glucose",
        instances_n=12,
        instances_basis="reported",
        phenotype="product titer with paired transcriptome and genome sequence",
        shape="scalar",
        seq_basis="evolved-WGS",
        why="The only yeast row in this product line whose selected mutants were both phenotyped and SEQUENCED, under BioProjects PRJNA750225 and PRJNA750524, which is what makes a mutagenesis output a genotype rather than a strain label. It surfaced only by reading the paper: both accessions are titled for malonyl-CoA and never say 3-HP, so searching an archive for the compound misses them, which is a general lesson for any downstream product.",
        accession="BioProject PRJNA750225 and PRJNA750524; values in the article",
        gene_namespace="common gene names in the text; the sequencing resolves to systematic names",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Kildegaard 2014 (3-HP tolerance evolution, resequenced clones)",
        organism="S. cerevisiae",
        citation="Kildegaard KR, et al. Evolution reveals a glutathione-dependent mechanism of 3-hydroxypropionic acid tolerance. Metab Eng 2014;26:57-66.",
        url="https://doi.org/10.1016/j.ymben.2014.09.004",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=3,
        genotypes="3 resequenced clones from 30 isolated across 3 independent lineages, with 8 mutated genes listed",
        env_n=1,
        env="3-HP in the medium",
        instances_n=3,
        instances_basis="reported",
        phenotype="3-HP tolerance",
        shape="scalar",
        seq_basis="evolved-WGS",
        why="The ONLY adaptive-evolution experiment in the yeast 3-HP literature, and the only source anywhere in it that names genes systematically: Table S1 gives the 8 mutated genes as ORFs, including SFA1 as YDL168W, where every other yeast 3-HP paper uses common names alone. It evolved tolerance in non-producers, so it is a tolerance row and not a production row. The reads were never deposited, which is what caps it at 3 genotypes.",
        accession="Table S1 of the article; the whole-genome reads were never deposited",
        gene_namespace="SGD systematic ORF names, uniquely in this literature",
        retrievability="manual_fetch",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Shi 2014 (Snf1-dependent Acc1 regulation abolished)",
        organism="S. cerevisiae",
        citation="Shi S, Chen Y, Siewers V, Nielsen J. Improving production of malonyl coenzyme A-derived metabolites by abolishing Snf1-dependent regulation of Acc1. mBio 2014;5:e01130-14.",
        url="https://doi.org/10.1128/mbio.01130-14",
        klass="Production campaign",
        tier=4,
        genotypes_n=4,
        genotypes="4 strains carrying Acc1 phosphosite substitutions off CEN.PK113-11C",
        env_n=1,
        env="glucose",
        instances_n=4,
        instances_basis="reported",
        phenotype="titer",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="Small but fully tabulated, in Table 4 and Supplementary Table S2, which puts it in the four-paper set whose yeast 3-HP numbers are machine readable. Its perturbation is a defined phosphosite substitution rather than a deletion or an overexpression, so it exercises an allele the schema has to represent as a sequence edit.",
        accession="open access; values in Table 4 and Supplementary Table S2",
        gene_namespace="common gene names only",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Chen 2018 (Acc1 phosphorylation-site mutant panel)",
        organism="S. cerevisiae",
        citation="Chen X, et al. Screening phosphorylation site mutations in yeast acetyl-CoA carboxylase using malonyl-CoA sensor to improve malonyl-CoA-derived product. Front Microbiol 2018;9:47.",
        url="https://doi.org/10.3389/fmicb.2018.00047",
        klass="Deep mutational scan",
        tier=4,
        genotypes_n=13,
        genotypes="13 ACC1 phosphorylation-site substitutions off CEN.PK102-5B",
        env_n=1,
        env="glucose with a malonyl-CoA biosensor readout",
        instances_n=13,
        instances_basis="reported",
        phenotype="biosensor output per substitution",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="A site-by-site substitution panel on one gene, which is a different genotype shape from every other row in this product line and the closest thing it has to a mutational scan. The per-substitution values are in Figure 4 rather than a table, so it is ranked on the genotype axis and its phenotype needs the same ask as the other figure-bound rows.",
        accession="open access; per-substitution values in Figure 4",
        gene_namespace="common gene names, with substitutions named by residue",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Kildegaard 2015 (3-HP from glucose and xylose)",
        organism="S. cerevisiae",
        citation="Kildegaard KR, et al. Production of 3-hydroxypropionic acid from glucose and xylose by metabolically engineered Saccharomyces cerevisiae. Metab Eng Commun 2015;2:132-136.",
        url="https://doi.org/10.1016/j.meteno.2015.10.001",
        klass="Production campaign",
        tier=4,
        genotypes_n=3,
        genotypes="3 strains carrying the beta-alanine route with xylose utilization added",
        env_n=2,
        env="glucose and xylose",
        instances_n=6,
        instances_basis="product",
        phenotype="titer and yield",
        shape="2 quantities",
        dim=2,
        seq_basis="S288C+designed-edit",
        why="The xylose arm of the yeast beta-alanine route, which pairs with the Pk xylose row on the same substrate question across two hosts, and that cross-host pairing is worth more than its three strains. Open access in a journal with no PMC deposit, so the tables need a fetch rather than an ask.",
        accession="open access article; no repository deposit",
        gene_namespace="common gene names only",
        retrievability="manual_fetch",
        verified="snippet",
        band="scale",
    ),
    # -- E. coli regulatory DNA, found by the 2026-11-24 fill pass -----------
    Candidate(
        name="Hollerer 2023 (5' UTR and CDS library, uASPIre)",
        organism="E. coli",
        citation="Hollerer S, Jeschek M. Nucleic Acids Res 2023;51:2377-2396.",
        url="https://doi.org/10.1093/nar/gkad040",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=1214438,
        genotypes="1,214,438 exactly specified 5' UTR and coding-sequence variants on a plasmid in TOP10 delta-rhaA",
        env_n=1,
        env="one condition, read out at 6 timepoints",
        instances_n=1214438,
        instances_basis="reported",
        phenotype="relative translation rate",
        shape="scalar plus a 6-point time course",
        dim=6,
        seq_basis="plasmid-reporter-library",
        time_axis="6 timepoints per variant",
        why="The largest E. coli sequence-to-phenotype library in the wave and the single biggest row outside the yeast promoter libraries, with every measured sequence exactly specified. The released file was downloaded and counted at 1,214,438 rows, which settles an inconsistency the source itself carries: the text says 9,589,692 data points and the abstract says more than 8.8 million. Deposited twice, on Zenodo and in a repository that both answer a script.",
        accession="Zenodo 10.5281/zenodo.7536586 and the JeschekLab/uASPIre_UTR_CDS repository, both reachable",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: the band's whole purpose, an axis that adds measurements rather than rows, and this row adds more of them than any other non-yeast candidate.",
    ),
    Candidate(
        name="Kuo 2020 (Shine-Dalgarno 9-mer library in four contexts)",
        organism="E. coli",
        citation="Kuo ST, Jahn RL, Cheng YJ, Chen YH, Lee YJ, Hollfelder F, Wen JD, Chou HHD. Global fitness landscapes of the Shine-Dalgarno sequence. Genome Res 2020;30:711-723.",
        url="https://doi.org/10.1101/gr.260182.119",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=262144,
        genotypes="the complete 262,144-member space of 9-nucleotide Shine-Dalgarno sequences, each measured in 4 ribosome-binding-site contexts",
        env_n=4,
        env="4 ribosome-binding-site contexts",
        instances_n=1023931,
        instances_basis="reported",
        phenotype="log reporter output",
        shape="scalar, 3 replicates",
        seq_basis="plasmid-reporter-library",
        why="A COMPLETE sequence space rather than a sample of one, which is a property almost nothing in the wave has: all 262,144 nine-mers, measured in four contexts, with 1,023,931 of the 1,048,576 possible rows carrying a value when the release is counted. The publisher refuses automated fetch, but the supplementary bundle comes through Europe PMC at 48.4 MB, so the route exists.",
        accession="Europe PMC supplementary files under PMC7263185, 48,362,217 bytes; reads at PRJNA516114; the publisher's own site returns 403",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: a complete nine-mer space in four contexts, which is the band's axis measured exhaustively rather than sampled.",
    ),
    Candidate(
        name="Hollerer 2020 (ribosome binding site library, uASPIre)",
        organism="E. coli",
        citation="Hollerer S, Papaxanthos L, Gumpinger AC, Fischer K, Beisel C, Borgwardt K, Benenson Y, Jeschek M. Large-scale DNA-based phenotypic recording and deep learning enable highly accurate sequence-function mapping. Nat Commun 2020;11:3551.",
        url="https://doi.org/10.1038/s41467-020-17222-4",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=303503,
        genotypes="303,503 exactly specified ribosome-binding-site variants on a plasmid in TOP10 delta-rhaA",
        env_n=1,
        env="one condition, three independent replicates",
        instances_n=841882,
        instances_basis="reported",
        phenotype="recombinase-recorded reporter output",
        shape="scalar plus a 9-point recording",
        dim=9,
        seq_basis="plasmid-reporter-library",
        why="The parent library of the 2023 row and a different measurement principle from every other MPRA here: the phenotype is recorded into DNA by a recombinase rather than sorted, so the readout is a time integral. Counted from the release at 841,882 rows across three replicates, which is the figure to store under the one-record-per-replicate rule; the paper's own variant count is internally inconsistent, 303,503 against more than 313,000 in its discussion.",
        accession="the JeschekLab/uASPIre repository, reachable",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: a 300,000-member binding-site library whose readout is a DNA-recorded time integral, a measurement principle no other row in the band carries.",
    ),
    Candidate(
        name="Urtecho 2020 (genome-wide promoter MPRA)",
        organism="E. coli",
        citation="Urtecho G, et al. Genome-wide functional characterization of Escherichia coli promoters and regulatory elements responsible for their function. eLife Reviewed Preprint 2023;12:92558.",
        url="https://doi.org/10.7554/eLife.92558.1",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=321123,
        genotypes="321,123 genomic fragments tiling the MG1655 genome, plus 17,635 transcription-start-site constructs across 3 landing pads",
        env_n=2,
        env="2 media",
        instances_n=794709,
        instances_basis="reported",
        phenotype="RNA over DNA promoter activity",
        shape="scalar",
        seq_basis="MG1655+reporter-locus",
        why="The genome-wide promoter map of this host, tiling the whole chromosome rather than a designed library, which makes it the bacterial counterpart of the yeast promoter rows and a native-sequence set rather than a synthetic one. Counted at 794,709 released rows. Two cautions travel with it: there is still no final journal version, only a reviewed preprint, and its coordinates are U00096.2 positions rather than b-numbers, so it joins on the assembly and not on a gene id.",
        accession="GEO GSE144621, reachable over FTP; the accession display page is behind a challenge and the authors' own database fails TLS",
        gene_namespace="U00096.2 genome coordinates, not b-numbers",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: the only row that tiles a whole bacterial genome for promoter activity, so its sequences are native rather than designed.",
    ),
    Candidate(
        name="Zhou 2025 (massively parallel bacterial promoter library)",
        organism="E. coli",
        citation="Zhou Y, et al. Nucleic Acids Res 2025;53:gkaf863.",
        url="https://doi.org/10.1093/nar/gkaf863",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=233469,
        genotypes="233,469 promoters in the released file, against 112,955 headlined in the text",
        env_n=1,
        env="one condition, three replicates",
        instances_n=700407,
        instances_basis="reported",
        phenotype="promoter activity",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="A quarter-million-promoter library whose release is roughly twice what its own text advertises: 233,469 promoters at three replicates is 700,407 measurements, counted from the file, against a headline figure of 112,955. The discrepancy is the reason to pin the release rather than the paper. One real gap: the strain is given as BL21(DE3) for storage and the measurement host is never clearly stated, so the sequence basis is a plasmid library and not a named chromosome.",
        accession="Europe PMC supplementary files under PMC12407095; reads at PRJNA1189185",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: 700,407 promoter measurements, and the clearest case in the wave of a release exceeding its own paper's claim.",
    ),
    Candidate(
        name="Fuqua 2024 (promoter-island mutational landscape)",
        organism="E. coli",
        citation="Fuqua T, et al. Dense and pleiotropic regulatory information in a developmental enhancer-like bacterial promoter island. eLife 2024;13:98654.",
        url="https://doi.org/10.7554/eLife.98654",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=245639,
        genotypes="245,639 mutated daughter sequences derived from 25 promoter-island templates",
        env_n=1,
        env="one condition, two reporter channels",
        instances_n=245639,
        instances_basis="reported",
        phenotype="sort-seq expression in two channels",
        shape="8-bin distribution per channel",
        dim=8,
        seq_basis="plasmid-reporter-library",
        why="A mutational landscape around native promoter islands rather than a random library, and the row carries a BIN DISTRIBUTION per channel rather than a single mean, which is the shape the schema should store rather than collapse. Counted at 245,639 rows with the reads deposited. The measurement host is not stated anywhere reachable, which is the row's one gap.",
        accession="the tfuqua95/promoter_islands repository, reachable; reads at SRA PRJNA1071572",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: a quarter-million mutated native promoter sequences released as per-bin distributions rather than means.",
    ),
    Candidate(
        name="Cambray 2018 (244,000 designed sequences, multi-phenotype)",
        organism="E. coli",
        citation="Cambray G, Guimaraes JC, Arkin AP. Evaluation of 244,000 synthetic sequences reveals design principles to optimize translation in Escherichia coli. Nat Biotechnol 2018;36:1005-1015.",
        url="https://doi.org/10.1038/nbt.4238",
        klass="Regulatory DNA",
        tier=2,
        genotypes_n=244000,
        genotypes="exactly 244,000 designed sequences, matching the paper's own headline",
        env_n=4,
        env="4 growth conditions for the growth readout",
        instances_n=244000,
        instances_basis="reported",
        phenotype="protein level, polysome profile, RNA over DNA, RNA half-life and growth",
        shape="103 released columns per sequence",
        dim=103,
        seq_basis="plasmid-reporter-library",
        why="The most MULTI-PHENOTYPE row in the whole wave: one library of 244,000 sequences carrying protein level, a 5-point polysome vector, a 12-point RNA over DNA vector, RNA half-life and growth in four conditions, as 244,000 rows by 103 columns counted from the deposit. That makes it a single source where one genotype axis maps to five different readouts, which is exactly the input-representation-to-readout axis this wave exists to widen. The strain is not stated in any reachable source.",
        accession="an OSF deposit at 130,122,456 bytes that answers a script, where the publisher's own link redirects; reads at SRP086076",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: one designed library of 244,000 sequences read out five different ways, the widest phenotype fan-out in the wave.",
    ),
    Candidate(
        name="Yu 2021 (promoter library across induction states)",
        organism="E. coli",
        citation="Yu H, et al. Dynamic and quantitative characterization of promoters in Escherichia coli. Nat Commun 2021;12:325.",
        url="https://doi.org/10.1038/s41467-020-20094-3",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=8269,
        genotypes="8,269 promoters at a chromosomal landing pad between essQ and cspB in an MG1655 derivative",
        env_n=2,
        env="with and without induction",
        instances_n=33808,
        instances_basis="reported",
        phenotype="promoter activity and its fold change on induction",
        shape="scalar per induction state",
        dim=2,
        seq_basis="MG1655+reporter-locus",
        why="A promoter library measured in TWO induction states rather than one, so each sequence carries a response and not just a level, and it sits at a named chromosomal landing pad rather than on a plasmid. The deposited table carries 33,808 condition-level measurements against an abstract that headlines 8,269 promoters, so the release is four times the advertised size.",
        accession="GEO GSE145630, reachable over FTP",
        gene_namespace="none; the identifier is the measured sequence",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: a chromosomal promoter library carrying an induction response per sequence rather than a single level.",
    ),
    # -- E. coli scale rows, same fill pass ----------------------------------
    Candidate(
        name="BV-BRC E. coli antimicrobial susceptibility panel",
        organism="E. coli",
        citation="Olson RD, et al. Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC). Nucleic Acids Res 2023;51:D678-D689.",
        url="https://doi.org/10.1093/nar/gkac1003",
        klass="Natural variation",
        tier=1,
        genotypes_n=13742,
        genotypes="13,742 distinct sequenced E. coli genomes with a laboratory-measured susceptibility record",
        env_n=104,
        env="104 antibiotic labels, before vocabulary normalization",
        instances_n=176318,
        instances_basis="reported",
        phenotype="minimum inhibitory concentration, mg/L",
        shape="scalar",
        seq_basis="ecoli-isolate-WGS",
        why="The largest genotype-to-phenotype resource for this host anywhere in the wave, and it answers a plain HTTP query: filtering to laboratory-measured evidence gives 266,659 susceptibility records of which 176,318 carry a numeric MIC, over 13,742 sequenced genomes. Every row carries a source identifier, so provenance pins per primary study rather than to the aggregator. Two cautions: the unfiltered table for this taxon is 7,260,599 rows INCLUDING computational predictions, which must never be mixed into a measured count, and the antibiotic vocabulary is unnormalized, with three spellings of one drug combination.",
        accession="the BV-BRC genome_amr API, fully scriptable as JSON; counts measured live on 2026-11-24",
        gene_namespace="BV-BRC genome identifiers, not b-numbers",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Melnikov 2014 (aminoglycoside resistance deep mutational scan)",
        organism="E. coli",
        citation="Melnikov A, Rogov P, Wang L, Gnirke A, Mikkelsen TS. Comprehensive mutational scanning of a kinase in vivo reveals substrate-dependent fitness landscapes. Nucleic Acids Res 2014;42:e112.",
        url="https://doi.org/10.1093/nar/gku511",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=4993,
        genotypes="4,993 of 4,997 designed substitutions detected in an aminoglycoside kinase on a plasmid",
        env_n=24,
        env="6 aminoglycosides at 4 doses each",
        instances_n=299580,
        instances_basis="product",
        phenotype="change in amino-acid frequency under selection",
        shape="scalar",
        seq_basis="plasmid-coding-variant",
        why="A substrate-dependent fitness landscape measured against six drugs at four doses, which is a dose axis almost nothing else in the wave carries. The phenotype needs stating precisely, and this is why the row is worth its verification note: the released quantity is a CHANGE IN AMINO-ACID FREQUENCY, not a fitness score and not an inhibitory concentration, released as a wide matrix of 332,640 values that has to be reshaped before it is records. Storing it as a fitness would misname the measurement.",
        accession="Europe PMC supplementary files under PMC4132701, 4.03 MB; reads at SRP042113",
        gene_namespace="gene name only, with substitutions named by residue",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Johns 2018 (29,249 expression constructs across conditions)",
        organism="E. coli",
        citation="Johns NI, et al. Metagenomic mining of regulatory elements enables programmable species-selective gene expression. Nat Methods 2018;15:323-329.",
        url="https://doi.org/10.1038/nmeth.4633",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=29249,
        genotypes="29,249 regulatory constructs mined from metagenomes",
        env_n=4,
        env="4 conditions",
        instances_n=116996,
        instances_basis="product",
        phenotype="expression, with a transcription-start-site readout beside it",
        shape="scalar plus a variable-length start-site vector",
        seq_basis="plasmid-reporter-library",
        why="A metagenome-mined regulatory library, so its sequences are natural and drawn from outside this host entirely, which is a different sequence distribution from every designed library in the band. The row exists because reading the release corrected the paper twice: the abstract headlines only the 11,319-member converged set rather than all 29,249 constructs, and two supplementary sheets duplicate one condition, so it is 4 conditions and not 5. The data-availability statement is unreadable because the full-text service errors, and no deposit accession was found.",
        accession="two publisher supplementary files, both reachable; no GEO or SRA accession found, and the full-text service returns an error on the availability statement",
        gene_namespace="repository and reference identifiers rather than b-numbers",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: the only library in the band whose sequences were mined from metagenomes rather than designed or tiled from one genome.",
    ),
    Candidate(
        name="Mateus 2020 (121 mutants, proteome abundance and thermal stability)",
        organism="E. coli",
        citation="Mateus A, et al. The functional proteome landscape of Escherichia coli. Nature 2020;588:473-478.",
        url="https://doi.org/10.1038/s41586-020-3002-5",
        klass="Proteome",
        tier=2,
        genotypes_n=121,
        genotypes="121 mutants, mostly cataloged single-gene deletions in BW25113",
        env_n=10,
        env="10 temperatures",
        instances_n=121,
        instances_basis="reported",
        phenotype="protein abundance and thermal stability",
        shape="two vectors over 1,764 proteins",
        dim=1764,
        seq_basis="ecoli-KO",
        why="Two proteome-wide readouts per mutant rather than one, abundance and thermal stability over 1,764 proteins, which makes a deletion panel into a 170,150-value matrix and adds a biophysical axis the database has nowhere else. Deposited three ways, in two proteomics repositories and a code repository, so the retrieval route is not in doubt.",
        accession="PRIDE PXD016589, MassIVE MSV000084632, and the authors' code repository",
        gene_namespace="UniProt accessions against proteome UP000000625",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Galardini 2017 ECOREF (894 strains across 214 conditions)",
        organism="E. coli",
        citation="Galardini M, et al. Phenotype inference in an Escherichia coli strain panel. eLife 2017;6:e31035.",
        url="https://doi.org/10.7554/eLife.31035",
        klass="Natural variation",
        tier=2,
        genotypes_n=869,
        genotypes="869 strains in the released matrix, of 894 phenotyped and 696 sequenced",
        env_n=214,
        env="214 conditions",
        instances_n=114004,
        instances_basis="reported",
        phenotype="colony-size S-score",
        shape="scalar",
        seq_basis="ecoli-isolate-WGS",
        why="The host's natural-isolate panel, and the direct counterpart of the yeast 1,011-isolate rows this database already serves, which is what makes it a cross-host transfer set rather than one more screen. The release is a plain delimited file that answers a script: downloaded and counted at 114,004 rows over 214 conditions and 869 strains, matching the paper's own stated figure of 114,004 single measurements while differing from its strain count, which is the kind of mismatch the pin exists to record.",
        accession="a plain tab-separated release at 6,690,205 bytes that answers a script; genomes at ENA PRJEB20550",
        gene_namespace="strain identifiers; the sequenced subset resolves to its own assemblies",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Schramm 2023 and Lubrano 2025 (15,120 essential-gene substitutions)",
        organism="E. coli",
        citation="Schramm T, et al. Mapping temperature-sensitive mutations at a genome scale. Mol Syst Biol 2023;19:e11596; Lubrano P, et al. Mol Syst Biol 2025;21:5.",
        url="https://doi.org/10.15252/msb.202311596",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=15120,
        genotypes="15,120 strains, each carrying one distinct single amino-acid substitution across 346 essential proteins, from 16,038 designed",
        env_n=4,
        env="30 and 42 degrees, plus carbenicillin and gentamicin in the companion study",
        instances_n=60480,
        instances_basis="product",
        phenotype="pooled barcoded fitness",
        shape="scalar",
        seq_basis="ecoli+designed-edit",
        why="The closest thing to a genome-scale deep mutational scan in this host: one substitution per strain across 346 essential proteins, phenotyped for temperature sensitivity in one paper and for two antibiotics in its companion, which is why the two are one row. It found 1,269 temperature-sensitive mutants and 8,290 low-fitness mutations. One caveat decides the ingest: neither paper says which supplementary file holds the full per-mutant fitness table, and one candidate file is a document rather than data, so the matrix may have to be recomputed from barcode counts.",
        accession="ENA PRJEB64015 and ERP159795, three MassIVE deposits, and a structured-data record; the per-mutant table is not identified among the supplementary files",
        gene_namespace="gene symbols with amino-acid substitution notation",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Chen 2020 (VIM-2 beta-lactamase deep mutational scan)",
        organism="E. coli",
        citation="Chen JZ, Fowler DM, Tokuriki N. Comprehensive exploration of the translocation, stability and substrate recognition requirements in VIM-2 lactamase. eLife 2020;9:e56707.",
        url="https://doi.org/10.7554/eLife.56707",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=5535,
        genotypes="5,535 of 5,607 designed variants covered, on a plasmid in an E. coli cloning host",
        env_n=9,
        env="3 antibiotics across temperatures, 9 selection conditions",
        instances_n=49941,
        instances_basis="reported",
        phenotype="log2 enrichment fitness with its standard deviation",
        shape="scalar plus dispersion",
        seq_basis="plasmid-coding-variant",
        why="A resistance-gene landscape across nine selection conditions, released through an aggregator API that serves scores directly, with a dispersion beside every value, which most scans do not release. One classification trap worth recording: the aggregator files it under Pseudomonas aeruginosa because the GENE is from that organism, while the measurement host is an E. coli strain, so an organism-keyed query would miss it entirely.",
        accession="a variant-effect aggregator score-set API that answers a script; reads at PRJNA606894",
        gene_namespace="gene name with substitution notation; no host gene id",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Stiffler 2015 (TEM-1 beta-lactamase across seven selections)",
        organism="E. coli",
        citation="Stiffler MA, Hekstra DR, Ranganathan R. Evolvability as a function of purifying selection in TEM-1 beta-lactamase. Cell 2015;160:882-892.",
        url="https://doi.org/10.1016/j.cell.2015.01.035",
        klass="Deep mutational scan",
        tier=2,
        genotypes_n=4997,
        genotypes="4,997 substitution variants of TEM-1 on a plasmid",
        env_n=7,
        env="7 selection strengths",
        instances_n=34979,
        instances_basis="product",
        phenotype="log10 enrichment fitness",
        shape="scalar",
        seq_basis="plasmid-coding-variant",
        why="A dose-response landscape on the best-studied resistance gene there is, at seven selection strengths, which turns a scan into a titration. Served through an aggregator as seven linked score sets that answer a script, so the retrieval is settled even though the article is paywalled and refuses automated fetch, which is also why the host strain could not be established.",
        accession="seven linked aggregator score sets that answer a script; the article itself is paywalled",
        gene_namespace="gene name with substitution notation",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Carrera 2014 EcoMAC and EcoPhe compendium",
        organism="E. coli",
        citation="Carrera J, Estrela R, Luo J, Rai N, Tsoukalas A, Tagkopoulos I. An integrative, multi-scale, genome-wide model reveals the phenotypic landscape of Escherichia coli. Mol Syst Biol 2014;10:735.",
        url="https://doi.org/10.15252/msb.20145108",
        klass="Transcriptome",
        tier=3,
        genotypes_n=31,
        genotypes="31 strains including cataloged knockouts, with 141 perturbed genes across the compendium",
        env_n=15,
        env="more than 15 media, as the source states it",
        instances_n=2198,
        instances_basis="reported",
        phenotype="transcript abundance, with growth for a subset",
        shape="vector over 4,189 genes",
        dim=4189,
        seq_basis="ecoli-KO",
        why="The host's assembled expression compendium, 2,198 arrays drawn from 127 separate articles with a paired growth readout for 616 of them, which is the modality backbone a condition-aware model needs rather than a large genotype axis. Its size comes from the vector, not the rows. The supplement is not script-accessible, so the eleven datasets behind it need a browser, and the total value count is arithmetic rather than a reported figure.",
        accession="eleven publisher supplementary datasets; the supplement URL answers with a web page rather than the files, so it needs a browser",
        gene_namespace="gene symbols",
        retrievability="manual_fetch",
        verified="inferred",
        band="scale",
    ),
    Candidate(
        name="Suzuki 2014 (40 evolved resistant strains, expression and mutations)",
        organism="E. coli",
        citation="Suzuki S, Horinouchi T, Furusawa C. Prediction of antibiotic resistance by gene expression profiles. Nat Commun 2014;5:5792.",
        url="https://doi.org/10.1038/ncomms6792",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=40,
        genotypes="40 evolved resistant strains from 10 drugs across 4 independent lines, resequenced",
        env_n=1,
        env="one profiling condition",
        instances_n=41,
        instances_basis="reported",
        phenotype="transcript abundance, with 447 released mutation calls",
        shape="vector over 4,444 genes",
        dim=4444,
        seq_basis="evolved-WGS",
        why="Evolved resistance with the genotype, the expression state and the resistance level in one study, which is the three-way join this database has almost nowhere. Counted at 182,204 expression values and 447 mutation rows. The catch is the phenotype that matters most: inhibitory concentrations for 25 antibiotics WERE measured and released only as a supplementary figure, so the resistance axis is figure-bound while the expression axis is not.",
        accession="Europe PMC supplementary files, GEO GSE59408 and a sequence archive deposit; the susceptibility values exist only in a figure",
        gene_namespace="gene symbols",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Zampieri 2017 (190 resequenced evolved populations, metabolome)",
        organism="E. coli",
        citation="Zampieri M, Enke T, Chubukov V, Ricci V, Piddock L, Sauer U. Metabolic constraints on the evolution of antibiotic resistance. Mol Syst Biol 2017;13:917.",
        url="https://doi.org/10.15252/msb.20167028",
        klass="Metabolome / flux",
        tier=2,
        genotypes_n=190,
        genotypes="190 resequenced evolved populations, which the methods also give as 193, across 4 lineages",
        env_n=6,
        env="3 antibiotics in 2 carbon sources",
        instances_n=190,
        instances_basis="reported",
        phenotype="intracellular and extracellular ion intensities",
        shape="two vectors, 413 intracellular and 392 extracellular",
        dim=805,
        seq_basis="evolved-WGS",
        why="Both an evolution set with resequenced endpoints and a metabolomics panel, which is why it is worth more than either alone: 190 populations each carrying 413 intracellular and 392 extracellular measurements. The source contradicts itself on the population count, 190 against 193, so the release is what gets pinned.",
        accession="a publisher dataset file and ENA PRJEB19222",
        gene_namespace="metabolite and ion identifiers; the genotypes resolve through the resequencing",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Furusawa 2020 (95 stressors, 192 resequenced clones, 47-chemical cross-resistance)",
        organism="E. coli",
        citation="Iwasawa J, Maeda T, Shibai A, Dojima H, Furusawa C, Kashiwagi A. Analysis of the evolution of resistance to multiple antibiotics enables prediction of the Escherichia coli phenotype-genotype landscape. Nat Commun 2020;11:5970.",
        url="https://doi.org/10.1038/s41467-020-19713-w",
        klass="Tolerance / robustness",
        tier=2,
        genotypes_n=192,
        genotypes="192 resequenced clones from 95 stressors across 6 independent lines, plus 64 reconstructed single mutants",
        env_n=47,
        env="cross-resistance measured against 47 chemicals",
        instances_n=192,
        instances_basis="reported",
        phenotype="relative inhibitory concentration",
        shape="vector over 47 chemicals",
        dim=47,
        seq_basis="evolved-WGS",
        why="The best-shaped row in the E. coli half of the wave even though it is not the biggest: evolved clones, resequenced, then cross-phenotyped against 47 chemicals, AND 64 of the implicated mutations reconstructed as single mutants and re-measured, which is the causal step almost no evolution study releases. 12,223 phenotype numbers and 1,448 mutation rows, all counted from the downloaded files, deposited four separate ways.",
        accession="seven publisher supplementary files, GEO GSE137348, a sequence archive deposit and a Zenodo record, all reachable",
        gene_namespace="gene names",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Lipworth 2024 (8,555 clinical isolates with linked susceptibility)",
        organism="E. coli",
        citation="Lipworth S, et al. Ten-year longitudinal molecular epidemiology study of Escherichia coli and Klebsiella species bloodstream infections in Oxfordshire, UK. Lancet Microbe 2024;5:100913.",
        url="https://doi.org/10.1016/s2666-5247(24)00152-6",
        klass="Natural variation",
        tier=2,
        genotypes_n=8555,
        genotypes="8,555 assembled bloodstream isolates with linked susceptibility, of 9,001 assembled",
        env_n=7,
        env="7 antibiotics",
        instances_n=59885,
        instances_basis="product",
        phenotype="resistant or susceptible call on a clinical standard",
        shape="binary",
        seq_basis="ecoli-isolate-WGS",
        why="A ten-year clinical isolate series with sequencing and susceptibility linked per isolate, which is a sampling frame no laboratory panel reproduces. Two properties cap it: the calls are BINARY rather than numeric concentrations, so it cannot be pooled with the measured-concentration rows, and the paper prints two different isolate counts without reconciling them, 9,001 against 8,945 and 8,555 against 8,586. The total number of calls is not stated, so the instance figure is an upper bound.",
        accession="two figshare records and five sequence-archive BioProjects",
        gene_namespace="isolate identifiers",
        verified="snippet",
        band="scale",
    ),
    Candidate(
        name="Lazar 2014 (antibiotic resistance evolution with b-numbered mutations)",
        organism="E. coli",
        citation="Lazar V, et al. Genome-wide analysis captures the determinants of the antibiotic cross-resistance interaction network. Nat Commun 2014;5:4352.",
        url="https://doi.org/10.1038/ncomms5352",
        klass="Tolerance / robustness",
        tier=4,
        genotypes_n=63,
        genotypes="63 sequenced evolved lines, from 120 phenotyped populations",
        env_n=12,
        env="12 antibiotics",
        instances_n=63,
        instances_basis="reported",
        phenotype="cross-resistance and collateral sensitivity",
        shape="vector over 12 antibiotics",
        dim=12,
        seq_basis="evolved-WGS",
        why="Small, but it is the ONLY source found in this whole pass that ships b-numbers, mapping 153 genes to their systematic identifiers, which is the join key every other E. coli candidate here lacks and the reason it is ranked rather than dropped at 1,867 phenotype numbers. Its supplementary route has a specific trap: the request fails unless inline images are excluded, which is the kind of detail a retrieval record has to carry.",
        accession="Europe PMC supplementary files under PMC4102323, which error unless the inline-image parameter is set",
        gene_namespace="b-numbers, uniquely among this pass's E. coli finds",
        verified="verified",
        band="scale",
    ),
    # -- S. cerevisiae fill, the 2026-11-24 unserved-dataset sweep ----------
    Candidate(
        name="Liu 2020 (PPiSeq accessory interactome)",
        organism="S. cerevisiae",
        citation="Liu Z, Miller D, Li F, Liu X, Levy SF. eLife 2020;9:e62365.",
        url="https://doi.org/10.7554/eLife.62365",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=1660856,
        genotypes="1,660,856 protein pairs as BY4741 by BY4742 diploids, each carrying two cassettes at the HO locus",
        env_n=9,
        env="9 environments, including peroxide, hydroxyurea, forskolin, doxorubicin, raffinose, FK506, salt and cold",
        instances_n=14947704,
        instances_basis="reported",
        phenotype="relative fitness under selection, with a p-value and a binary call",
        shape="scalar per pair and environment",
        seq_basis="S288C+integration-locus",
        why="The largest single release in the entire wave, counted from the deposit at 14,947,704 values, and the only row whose genotype is a PAIR of perturbations measured across nine environments, which is the interaction-by-environment shape the database has nowhere else. The paper's own headline of 44 million is the replicate-level barcode count and 14.9 million is the collapsed per-pair-per-environment figure, so the release has two defensible sizes and the loader has to say which it stores. Identifiers are systematic and ordered pairs.",
        accession="an OSF deposit serving a 132,711,841-byte archive to a plain client, unpacking to 362 MB; reads at SRA PRJNA630095",
        gene_namespace="SGD systematic names, as ordered pairs",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Aggeli 2021 (second-step adaptation, barcoded lineages)",
        organism="S. cerevisiae",
        citation="Aggeli D, Li Y, Sherlock G. Nat Commun 2021;12:5193.",
        url="https://doi.org/10.1038/s41467-021-25440-7",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="6 barcoded populations from 3 first-step mutant ancestors in replicate; lineage barcodes carry no gene identifier",
        env_n=1,
        env="one environment over 160 generations",
        instances_n=4427058,
        instances_basis="reported",
        phenotype="barcode read-count trajectory, with fitness and establishment time per adaptive lineage",
        shape="8 to 22 timepoints per lineage",
        seq_basis="S288C+integration-locus",
        time_axis="8 to 22 sequenced timepoints over 160 generations",
        why="Second-step adaptation, meaning evolution starting FROM known adaptive mutants rather than from a clean ancestor, which is a question none of the other lineage-tracking rows asks. Counted at 4,427,058 count rows and 24,929 lineages carrying a fitness and an establishment time with errors.",
        accession="a publisher supplementary archive at 91,285,884 bytes, unpacking to 425 MB; reads at SRA PRJNA641174",
        gene_namespace="ancestors by standard name; the barcode rows carry none",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Michel 2017 (SATAY transposon saturation)",
        organism="S. cerevisiae",
        citation="Michel AH, Hatakeyama R, Kimmig P, Arter M, Peter M, Matos J, De Virgilio C, Kornmann B. eLife 2017;6:e23570.",
        url="https://doi.org/10.7554/eLife.23570",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=None,
        genotypes="8 libraries across 7 genetic backgrounds plus a rapamycin arm, at 169,000 to 495,000 mapped insertions each",
        env_n=2,
        env="rich medium and a rapamycin arm",
        instances_n=2588216,
        instances_basis="reported",
        phenotype="insertion density and read count as a fitness proxy",
        shape="2 values per gene and library",
        dim=2,
        seq_basis="S288C+transposon",
        why="The founding saturated-transposon screen in this organism, and a perturbation class the database does not hold at all: an insertion SITE rather than a cataloged deletion, which reaches intergenic and essential-gene-internal positions no deletion collection can. Counted at 2,588,216 insertion records and 52,824 gene-level records. Note that the publisher record carries no sequence accession, so the processed supplement is the only release.",
        accession="a publisher supplementary archive, downloaded and unpacked; the publisher's own data-availability record is empty, and no sequence accession exists",
        gene_namespace="mixed, standard name where one exists and systematic otherwise",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Sharon 2018 (CRISPEY natural-variant precision editing)",
        organism="S. cerevisiae",
        citation="Sharon E, Chen SAA, Khosla NM, Smith JD, Pritchard JK, Fraser HB. Cell 2018;175:544-557.e16.",
        url="https://doi.org/10.1016/j.cell.2018.08.057",
        klass="CRISPR library screen",
        tier=1,
        genotypes_n=16006,
        genotypes="16,006 natural variants installed one at a time into BY4742, each an RM11-1a allele, through 32,000 guide and donor pairs",
        env_n=1,
        env="one selective medium over about 25 generations",
        instances_n=48006,
        instances_basis="reported",
        phenotype="log2 fold change per generation, with standard error and two significance measures",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The cleanest causal design in the yeast half of the wave: every row is ONE natural variant installed into a common background, so the effect is attributable to that variant rather than to a haplotype, which is exactly what a segregant panel cannot do. 2,240,000 read-count cells reduce to 48,006 scores. One release discrepancy travels with it: the methods say 31,870 guides and the released table has exactly 32,000 rows.",
        accession="the publisher's own supplementary table at 20,453,894 bytes answers a plain client, while the article page refuses one and an open-archive path returns a decoy; reads at SRP126558",
        gene_namespace="none; keyed by chromosome and position on the sacCer3 assembly",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Blundell 2019 (adaptive diversity in two limitations)",
        organism="S. cerevisiae",
        citation="Blundell JR, Schwartz K, Francois D, Fisher DS, Sherlock G, Levy SF. Nat Ecol Evol 2019;3:293-301.",
        url="https://doi.org/10.1038/s41559-018-0758-1",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=494066,
        genotypes="494,066 released barcoded lineages, against about 500,000 claimed",
        env_n=2,
        env="carbon-limited and nitrogen-limited",
        instances_n=2084171,
        instances_basis="reported",
        phenotype="barcode read-count trajectory, with a separate fitness table",
        shape="14 to 33 timepoint columns",
        seq_basis="S288C+integration-locus",
        why="The two-limitation counterpart of the single-environment lineage-tracking rows, and the row that carries the fitness estimates its own parent study never released, 107,907 of them. Counted at 2,084,171 rows. It shares all 494,066 barcode identities with Levy 2015 and matches it value for value on the first data row, so the two must be deduplicated by barcode rather than ingested as independent panels. No repository accession exists anywhere readable, so the processed tables are the only record, and the file extension is load bearing: one form answers and the other is refused.",
        accession="five publisher supplementary files, reachable only as delimited text and not as workbooks; no repository accession exists",
        gene_namespace="none in the bulk data",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Nguyen Ba 2019 (travelling wave of adaptation)",
        organism="S. cerevisiae",
        citation="Nguyen Ba AN, Cvijovic I, Rojas Echenique JI, Lawrence KR, Rego-Costa A, Liu X, Levy SF, Desai MM. Nature 2019;575:494-499.",
        url="https://doi.org/10.1038/s41586-019-1749-3",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=None,
        genotypes="about 50,000 barcodes per population, RE-BARCODED every 100 generations so lineages nest",
        env_n=2,
        env="rich medium and rich medium with acetic acid",
        instances_n=1901267,
        instances_basis="reported",
        phenotype="barcode frequency trajectory, with clone fitness and confidence intervals",
        shape="101 frequency points per barcode",
        seq_basis="S288C+integration-locus",
        time_axis="110 sequenced timepoints over 1,000 generations",
        why="A thousand generations with the barcodes RENEWED every hundred, which is what lets a lineage be followed through successive sweeps instead of being lost when an earlier one fixes, and it is the only row here with that nested design. Counted at 1,901,267 trajectory rows across 36 files, 942,476 of them in the two primary populations. The ancestor is a diploid and is not named in the main text, which is the row's one identity gap.",
        accession="a code repository serving every file to a plain client; reads at PRJNA559526",
        gene_namespace="none",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Reichling 2023 (dynamic metabolome across 164 deletions)",
        organism="S. cerevisiae",
        citation="Reichling S, Doubleday PF, Germade T, Bergmann A, Loewith R, Sauer U, Holbrook-Smith D. eLife 2023;12:e84295.",
        url="https://doi.org/10.7554/eLife.84295",
        klass="Metabolome / flux",
        tier=2,
        genotypes_n=164,
        genotypes="164 cataloged deletion mutants from the BY4741 collection, restored to prototrophy by a plasmid",
        env_n=5,
        env="5 timepoints after rapamycin, at 0, 5, 30, 60 and 90 minutes",
        instances_n=8876,
        instances_basis="reported",
        phenotype="ion intensities with mass, formula, adduct and pathway identifiers",
        shape="vector over 192 ions",
        dim=192,
        seq_basis="S288C-KO",
        time_axis="5 timepoints after rapamycin addition",
        why="A TIME COURSE of the metabolome across a deletion panel, which is a shape the database holds nowhere: most metabolomics here is one snapshot per genotype. Counted at 1,704,192 released values, or 1,351,296 restricted to the 164 mutants, roughly 240 times the figure the abstract gives, because the abstract counts injections and not values. Both of the paper's companion web applications are dead, so the supplement and the raw deposits are the whole record.",
        accession="a publisher supplementary workbook through an open-archive bundle; raw spectra on MassIVE under four deposits",
        gene_namespace="standard gene names only",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Litsios 2024 (cell-cycle proteome movements, GFP collection)",
        organism="S. cerevisiae",
        citation="Litsios A, Grys BT, Kraus OZ, Friesen H, Ross C, Masinas MPD, et al. Cell 2024;187:1490-1507.",
        url="https://doi.org/10.1016/j.cell.2024.02.014",
        klass="Proteome",
        tier=1,
        genotypes_n=3902,
        genotypes="3,902 GFP-tagged open reading frames in BY4741, 3,806 of them carrying a concentration",
        env_n=6,
        env="one growth condition resolved into 6 cell-cycle stages, in triplicate",
        instances_n=3902,
        instances_basis="reported",
        phenotype="localization class fractions and protein concentration per cell-cycle stage",
        shape="23 values per stage, 138 per strain",
        dim=138,
        seq_basis="S288C+designed-edit",
        why="Protein localization AND concentration resolved through the cell cycle for essentially every tagged protein, which turns a static atlas into a trajectory and is the only row here with an explicit cell-cycle axis. Counted at 1,613,700 values. Two cautions: the per-cell features behind the summary are not released, and the source states its own class count two ways, 22 in the text against about 20 in the abstract.",
        accession="an author-hosted download serving 22,556,413 bytes to a plain client with no token, where the publisher refuses one and the open-archive full text errors",
        gene_namespace="both systematic and standard names, in dedicated columns",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Chen 2022 (transposon mutagenesis of 16 natural isolates)",
        organism="S. cerevisiae",
        citation="Chen P, Michel AH, Zhang J. Nat Commun 2022;13:1490.",
        url="https://doi.org/10.1038/s41467-022-29228-1",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=16,
        genotypes="16 haploid natural isolates including S288C and RM11, each transposon-mutagenized across 5,813 genes",
        env_n=1,
        env="one condition",
        instances_n=93008,
        instances_basis="product",
        phenotype="essentiality verdict with a 12-value insertion-statistics vector",
        shape="12 statistics per gene and isolate",
        dim=12,
        seq_basis="isolate-WGS",
        why="Gene essentiality measured independently in SIXTEEN genetic backgrounds, which is the conditional-essentiality question that a single-reference screen cannot answer and the direct yeast counterpart of the cross-host transfer this wave is built for. Counted at 1,133,415 metric values over 93,000 verdicts, on 5,403,391 mapped insertion sites. The browser session holding the full insertion map refuses automated clients, so the per-site layer needs the raw reads.",
        accession="a publisher supplementary workbook at 11,818,899 bytes; reads at SRA PRJNA776744 across 48 runs",
        gene_namespace="systematic names with standard names beside them",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Levy 2015 (high-resolution lineage tracking)",
        organism="S. cerevisiae",
        citation="Levy SF, Blundell JR, Venkataram S, Petrov DA, Fisher DS, Sherlock G. Nature 2015;519:181-186.",
        url="https://doi.org/10.1038/nature14279",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=494066,
        genotypes="494,066 released barcoded lineages in a BY4709 landing-pad background",
        env_n=1,
        env="one environment, two replicate experiments",
        instances_n=988132,
        instances_basis="reported",
        phenotype="barcode read-count trajectory",
        shape="12 and 10 timepoints",
        seq_basis="S288C+integration-locus",
        why="The founding barcoded-lineage dataset, and the reason to ingest it is also its warning: its 25,000 and 14,000 per-lineage FITNESS estimates were never released, only plotted, so the fitness has to come from Blundell 2019 or be recomputed. It has no accession, no data-availability statement, and an archive search on its title returns nothing, so figure source data is the entire release.",
        accession="two figure source-data files at 19,348,822 and 15,377,229 bytes; no accession and no data-availability statement exist",
        gene_namespace="none",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Pokusaeva 2019 (HIS3 combinatorial fitness landscape)",
        organism="S. cerevisiae",
        citation="Pokusaeva VO, Usmanova DR, Putintseva EV, Espinar L, Sarkisyan KS, Mishin AS, et al. PLoS Genet 2019;15:e1008079.",
        url="https://doi.org/10.1371/journal.pgen.1008079",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=875151,
        genotypes="875,151 unique amino-acid sequences of one protein, from 4,018,105 nucleotide genotypes",
        env_n=1,
        env="histidine-free medium over 168 hours",
        instances_n=956648,
        instances_basis="reported",
        phenotype="fitness, with a frequency trajectory",
        shape="scalar plus 3 timepoints",
        seq_basis="plasmid-coding-variant",
        why="A COMBINATORIAL landscape rather than a single-substitution scan: variants carry multiple mutations at once, which is the only way to measure epistasis inside a protein and the shape every other scan in the wave lacks. Counted at 956,648 amino-acid-level rows in the release, which reconciles the paper's 4.0 million figure as a nucleotide-genotype count. Retrieval has a specific trap worth recording: the archive's file path serves every file to a plain client while its display page interposes a challenge.",
        accession="GEO GSE99990, reachable over FTP while the accession display page is behind a challenge",
        gene_namespace="none; keyed by the variant sequence",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Savinov 2021 (random 50-mer 3' UTR library)",
        organism="S. cerevisiae",
        citation="Savinov A, Brandsen BM, Angell BE, Cuperus JT, Fields S. Genome Biol 2021;22:293.",
        url="https://doi.org/10.1186/s13059-021-02509-6",
        klass="Regulatory DNA",
        tier=1,
        genotypes_n=867386,
        genotypes="867,386 scored random 50-nucleotide 3' UTR elements across two libraries, from 4.6 million constructed",
        env_n=2,
        env="two library designs under one selection each",
        instances_n=867386,
        instances_basis="reported",
        phenotype="log2 enrichment, read out as growth through a reporter",
        shape="scalar",
        seq_basis="plasmid-reporter-library",
        why="The 3' UTR counterpart of the promoter and 5' UTR libraries this wave already ranks, which closes the regulatory axis at the other end of the transcript: the database has sequence-to-expression data for what starts transcription and nothing for what terminates or stabilizes it. Counted at 590,024 and 277,362 rows against an abstract that rounds to 590,000 and 280,000.",
        accession="a figshare record serving a 36,920,254-byte archive through its API; reads at SRA PRJNA750726",
        gene_namespace="none; keyed by the 50-nucleotide element",
        verified="verified",
        band="regulatory DNA",
        band_why="Regulatory DNA: the 3' UTR end of the transcript, the one regulatory axis no other row in the band measures.",
    ),
    Candidate(
        name="Schmidlin 2024 (drug-resistance tradeoffs across 12 environments)",
        organism="S. cerevisiae",
        citation="Schmidlin K, Apodaca S, Newell D, Sastokas A, Kinsler G, Geiler-Samerotte K. eLife 2024;13:RP94144.",
        url="https://doi.org/10.7554/eLife.94144",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=774,
        genotypes="774 adaptive strains evolved in 12 antifungal conditions, on a landing-pad background carrying 4 reported fixed mutations",
        env_n=12,
        env="12 environments spanning two antifungals",
        instances_n=9288,
        instances_basis="reported",
        phenotype="fitness per lineage, condition and replicate, with barcode counts beneath it",
        shape="scalar",
        seq_basis="S288C+integration-locus",
        why="A full tradeoff matrix: every adaptive strain scored in all twelve environments, which is what turns a collection of resistant mutants into a measurement of the cost of resistance. 719,872 barcodes across 409 samples reduce to 9,288 fitness rows, all counted. It is also the single most retrievable row in the wave, with an API that lists recursively and served a 595 MB table end to end.",
        accession="an OSF deposit whose API lists recursively and serves a 595,538,135-byte table; sequencing at SRA PRJNA1023288",
        gene_namespace="standard gene names with amino-acid changes, not systematic",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Karadzas 2026 (SATAY chemogenomics across 20 antifungals)",
        organism="S. cerevisiae",
        citation="Karadzas MT, Michel AH, Mosbach A, Giannakopoulos G, McGettigan R, Scalliet G, Kornmann B. eLife 2026;14:RP105333.",
        url="https://doi.org/10.7554/eLife.105333",
        klass="Transposon fitness",
        tier=1,
        genotypes_n=None,
        genotypes="5 drug-sensitive BY-derived backgrounds, transposon-mutagenized, across 27 library-condition screens",
        env_n=21,
        env="21 named compounds plus controls, which the abstract rounds to 20",
        instances_n=674217,
        instances_basis="reported",
        phenotype="transposon count and read count per feature and condition",
        shape="2 values per feature",
        dim=2,
        seq_basis="S288C+transposon",
        why="Saturated transposon screening across a whole antifungal panel, which pairs directly with the P. kudriavzevii antifungal rows this wave pins at rank 1 and makes the drug axis cross-host. Counted at 674,217 records: 178,200 per-gene rows plus three per-promoter window tables. Its own compound count differs between the abstract and the supplement, 20 against 21, so the release is what gets pinned.",
        accession="ENA PRJEB129441 and PRJEB35463, both served by the portal API; processed tables in a publisher archive",
        gene_namespace="systematic names in the per-gene files",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Vieitez 2022 (5,362 strains across 102 conditions)",
        organism="S. cerevisiae",
        citation="Vieitez C, Busby BP, Ochoa D, Mateus A, Memon D, Galardini M, et al. Nat Biotechnol 2022;40:382-390.",
        url="https://doi.org/10.1038/s41587-021-01051-x",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=5362,
        genotypes="5,362 strains, being 4,888 deletions plus 474 phosphosite point mutants, in two different backgrounds",
        env_n=102,
        env="102 conditions",
        instances_n=540817,
        instances_basis="reported",
        phenotype="quantile-normalized colony-size score with a q-value",
        shape="scalar",
        seq_basis="S288C+designed-edit",
        why="The widest CONDITION axis in the wave at 102, and the only row that puts cataloged deletions and designed phosphosite substitutions on the same plate, so a gene's loss and the loss of one of its regulatory sites are measured against each other. Counted at 540,817 scores plus 78,664. The abstract states the 474 point mutants and the 102 conditions but never the 4,888-deletion arm, which is most of the release.",
        accession="a publisher supplementary workbook at 29,260,525 bytes through an open-archive bundle; proteomics at PRIDE PXD017929",
        gene_namespace="both, with a dedicated systematic-name column",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Duan 2025 (Pol II trigger loop across nine conditions)",
        organism="S. cerevisiae",
        citation="Duan B, Qiu C, Sze SH, Kaplan C. Nat Commun 2025;16:7993.",
        url="https://doi.org/10.1038/s41467-025-63304-6",
        klass="Deep mutational scan",
        tier=1,
        genotypes_n=15174,
        genotypes="15,174 substitution variants of one RNA polymerase domain across ten libraries, 6,054 of them viable",
        env_n=9,
        env="9 conditions including two drugs, manganese, two carbon sources, a lysine dropout and formamide",
        instances_n=136566,
        instances_basis="product",
        phenotype="fitness per variant and condition",
        shape="scalar, 3 replicates",
        seq_basis="plasmid-coding-variant",
        why="A deep mutational scan crossed with NINE conditions, which almost no scan does: it measures how a protein's mutational tolerance changes with the environment rather than in one medium. Counted at 409,698 released values over 136,566 variant-condition cells, with every condition read off the file's own header. Deposited on Zenodo, which answers a plain client.",
        accession="Zenodo 10.5281/zenodo.16370006, a 3,854,097-byte archive served to a plain client; reads at SRA PRJNA948661",
        gene_namespace="none; keyed by substitution strings on RPB1",
        verified="verified",
        band="scale",
    ),
    Candidate(
        name="Guenole 2013 (multiconditional DNA-damage E-MAP)",
        organism="S. cerevisiae",
        citation="Guenole A, Srivas R, Vreeken K, Wang ZZ, Wang S, Krogan NJ, Ideker T, van Attikum H. Mol Cell 2013;49:346-358.",
        url="https://doi.org/10.1016/j.molcel.2012.11.023",
        klass="Fitness / chemical genomics",
        tier=1,
        genotypes_n=97578,
        genotypes="97,578 gene pairs in the released grid, which the prose gives as a slightly different array",
        env_n=4,
        env="4 conditions, three DNA-damaging agents and a vehicle control",
        instances_n=97578,
        instances_basis="reported",
        phenotype="colony-size interaction score",
        shape="scalar per pair and condition",
        dim=4,
        seq_basis="S288C-KO",
        why="Genetic interactions measured under DNA damage rather than in rich medium, so it is a condition-dependent epistasis map and the only row of that kind here. Counted at 390,312 scores with ZERO missing values, plus 292,734 differential values, and its supplement answers a plain client with no wall. One gap is worth stating: the host background is not named anywhere in the paper or the supplement.",
        accession="a publisher supplementary workbook at 12,229,267 bytes, served without a wall; the gene mapping is in a sibling file",
        gene_namespace="standard names only, with the mapping in a companion file",
        verified="verified",
        band="scale",
    ),
]


# ---------------------------------------------------------------------------
# Considered and dropped. The rows the report explicitly says not to ingest are here
# with the reason, rather than absent: a dataset that was looked at and rejected is a
# result, and leaving it out makes it look unexamined.
# ---------------------------------------------------------------------------

EXCLUDED: list[Excluded] = [
    Excluded(
        name="Beltran 2025 aPCA domainome",
        reason=(
            "563,534 variants across 522 protein domains, measured in BY4741 yeast "
            "cells with a yeast growth readout, which is why it surfaced as a yeast "
            "candidate at all. 503 of those domains are HUMAN proteins. The cells are "
            "yeast and the genotype is not: the perturbed sequence is a human domain "
            "carried on a plasmid, keyed by UniProt accession and Pfam position, so "
            "ingesting it under this organism would file half a million human variants "
            "as S. cerevisiae genotypes. It is a strong dataset for a human-domain "
            "tier and a category error for this one"
        ),
        rule="off-host",
    ),
    Excluded(
        name="Erijman 2020 (random activation-domain peptides)",
        reason=(
            "About a million random peptides scored for transcription activation, which "
            "would have ranked tenth by size. Two things block it. The release count is "
            "never stated, so the million is the paper's own figure rather than a "
            "counted release, and the processed tables are not reachable: the path "
            "returns a 1,817-byte challenge page rather than data, and the code "
            "repository holds model weights only. The reads are deposited, so the "
            "phenotype would have to be recomputed from them, which makes it a "
            "reanalysis and not an ingest"
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Li 2019 (Pareto fronts in trait space)",
        reason=(
            "259,357 barcode count rows and 3,717 clones scored across four "
            "serial-transfer conditions, fully retrievable and well deposited. It is "
            "dropped on OVERLAP rather than on quality: 3,048 of its 3,717 clones, "
            "which is 82 percent, come from a dataset already on this database's "
            "exclusion list, so most of what it would add is a second scoring of "
            "strains already accounted for. The four-condition fitness vector on the "
            "remaining 669 clones is the only net-new part, which is too thin to carry "
            "a row"
        ),
        rule="derivative",
    ),
    Excluded(
        name="Yeast fill-pass measured negatives (nine sources checked and unusable)",
        reason=(
            "Recorded because each was checked and the result is a finding rather than "
            "an absence. Dudley 2005 releases a three-level ordinal phenotype in twelve "
            "PDF supplements, and its only data pointer is a dead host. Roy 2024 names "
            "no processed table and its BioProject returns zero experiments. Bakerlee "
            "2022 has the right shape but every route to its fitness tables is refused "
            "and its deposit holds genotyping alone. Kryazhimskiy 2014, Jerison 2017 and "
            "Johnson 2021 are not barcoded at all, which was checked by searching their "
            "body text, and use flow-cytometry competition instead, so they are good "
            "fitness releases for a different wave. Collins 2007, Fiedler 2009 and "
            "Bodenmiller 2010 have no locatable data file or a dead host. Further "
            "negatives worth keeping: no yeast messenger-RNA-decay library exists, no "
            "yeast degron atlas exceeds 23,600 peptides because the large ones are "
            "human, and no genome-wide chemical or nutrient panel above roughly 300,000 "
            "records appeared for 2015 to 2026 beyond the 102-condition row ranked above"
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="ALEdb (Phaneuf 2019), the adaptive-laboratory-evolution database",
        reason=(
            "Checked live on 2026-11-24 because it was the one aggregator that could "
            "plausibly have closed this wave's row gap on its own, and it releases "
            "ZERO phenotype numbers. The site states 3,721,969 instances of 1,035,104 "
            "unique mutations from 13,425 isolates across 271 projects, roughly 94 "
            "times the 11,792 mutations its own 2019 paper reported, so it has grown "
            "enormously. But a grep for growth rate, fitness, inhibitory "
            "concentration and titer across every reachable page and export returned "
            "nothing: the exports carry mutations, media, temperature, strain and "
            "sequencing quality only. Retrieval is also degraded, with no API, an "
            "export route that errors even with a fresh token, and only 107 of the 271 "
            "projects browsable anonymously. It is a GENOTYPE source for evolved "
            "clones, which is how a later pass should use it, not a dataset. One "
            "naming trap: one of its project codes is a Plasmodium study, not E. coli"
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="COLOMBOS v3, Bastkowski 2026, Yasir 2026, Reg-Seq, Good 2017 and Tenaillon 2016",
        reason=(
            "Six E. coli objects that each looked larger than anything ranked and "
            "released no per-record phenotype. COLOMBOS v3 would have been the biggest "
            "expression object in the wave at 4,077 condition contrasts, but its domain "
            "times out over both protocols, so it has no resolvable URL at all. "
            "Bastkowski 2026 deposits only raw reads for its 13 antibiotics at 4 "
            "concentrations, with no per-gene fitness table. Yasir 2026 is a "
            "754-sample transposon compendium, the highest-count such object found, "
            "and gives no accession of any kind. Reg-Seq releases sequencing reads and "
            "information-footprint matrices rather than per-variant values. Good 2017 "
            "has 41,962 rows but no phenotype, and its genotype is a population allele "
            "frequency rather than a clone, which fails the definition this wave "
            "ranks on. Tenaillon 2016 releases no per-clone fitness, only a 791-byte "
            "summary file"
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Kosuri 2013, Goodman 2013 and Moradigaravand 2018",
        reason=(
            "Three sets reachable only by a route this database does not take. The "
            "first two are blocked at the publisher, and the copies that do answer are "
            "a third-party compilation and a re-inferred mirror, so ingesting either "
            "would store another analysis's numbers rather than the release. "
            "Moradigaravand 2018's only surfaced deposit route is Mendeley Data, which "
            "is ruled out as a live source. In all three cases the primary release, "
            "not the row, is what is missing"
        ),
        rule="derivative",
    ),
    Excluded(
        name="Kildegaard 2016 and Chen 2026 (figure-bound 3-HP panels)",
        reason=(
            "Two real open-access S. cerevisiae 3-HP experiments with no per-strain "
            "numbers to read. Kildegaard 2016 (10.1186/s12934-016-0451-5) measured 14 "
            "strains and tabulated only the one fed-batch strain in Table 1. Chen 2026 "
            "(10.1038/s41467-026-76008-2) reports 48 colonies, and it DOES ship a "
            "Source Data file, which is why a first read called it ingestible, but the "
            "values in the fig4g sheet are negative, so they are a relative scale "
            "rather than titers and cannot be stored as a concentration. Digitizing a "
            "bar, or storing a relative value as though it were a measurement, would "
            "both invent precision the source never stated. Two passes over this "
            "literature agreed on the structural finding behind these rows: of roughly "
            "twenty yeast 3-HP papers whose full text was read, only four ship "
            "machine-readable per-strain numbers"
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="3-HP in adjacent yeast hosts (O. polymorpha, K. phaffii, S. pombe, Y. lipolytica, R. toruloides, K. marxianus, D. hansenii)",
        reason=(
            "Several reach titers no S. cerevisiae result matches, led by Ogataea "
            "polymorpha at 79.6 g/L and Debaryomyces hansenii at 62.4 g/L, the latter "
            "from a natural isolate plus ion mutagenesis rather than engineering. That "
            "last figure does NOT survive arithmetic: Li 2020 reports 62.42 g/L from "
            "40 g/L of total substrate, which fails mass balance, so it is recorded as "
            "a flagged number and not as a target. They "
            "are recorded rather than dropped because the 3-HP product line is the "
            "reason this wave touched the chemistry at all, and these are the rows a "
            "later wave would add. Admitting any of them means seven new Organism "
            "members and seven reference releases pinned in the genomes tier, which is "
            "a scope decision for the owner and not a fill"
        ),
        rule="off-host",
    ),
    Excluded(
        name="Illinois Data Bank IDB-9063411 (high-yield 3-HP)",
        reason=(
            "Ranked in this wave until the fill pass read it, and it is not a "
            "phenotype dataset at all: the deposit holds the technoeconomic model and "
            "the genome-scale model code, as two source repositories, not the strain "
            "table. The measurements the row was ranked for, 92 g/L at 0.7 g/g and "
            "0.55 g/L/h, are in Tan 2026, which is now ranked on its own identity with "
            "the Source Data file behind it. Keeping both would have counted one "
            "experiment twice"
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="LASER strain-engineering database (Winkler 2015, 2016)",
        reason=(
            "Counted from the bytes of the only live release, on Bitbucket: 456 "
            "curated papers, 638 designs, 441 E. coli and 197 S. cerevisiae, frozen "
            "since 2016-04-27, and the web interface the paper names no longer "
            "resolves. Every number in it is re-extracted from a primary paper, so "
            "ingesting it would re-serve other datasets' values. It is also thinner "
            "than it looks: titer is populated on 181 of 197 yeast designs, yield on "
            "58, there is no rate field at all, and no replicates, no n, no "
            "uncertainty, no sequence-level edit and no strain accession. Its own "
            "total field sums to 3,472,162 strains built against 638 curated, because "
            "it keeps champion strains and discards the screen populations. Its value "
            "is as an index onto primary papers, which is how the fill pass used it, "
            "and three of those papers are ranked above"
        ),
        rule="derivative",
    ),
    Excluded(
        name="Charusanti 2010 (pgi-deletion adaptive evolution)",
        reason=(
            "The largest pure evolution entry the LASER index points at, with ten "
            "endpoint clones and nine resequenced genomes, so the GENOTYPES are "
            "released. The phenotype is not: the supplement holds a peroxidase assay "
            "and primers, and the per-clone growth rates exist only inside Figure 2A. "
            "A figure is not a released per-record value, and digitizing one would "
            "invent precision the source never stated"
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="scYeast (Fan 2027) figshare deposit",
        reason=(
            "A model and its training corpus rather than a new measurement. Its "
            "attention prior is built from the documented-targets subset of YEASTRACT+, "
            "about 170,000 pairs, which is a prior over the genome and not a "
            "genotype-by-environment record, and the expression it trains on is the "
            "built yeast compendia. Nothing in the deposit is a released per-record "
            "value this store does not already hold."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Shorkie induction RNA-seq tracks (gs://shorkie-paper/)",
        reason=(
            "The 3,053 induction RNA-seq tracks generated at Calico are the one "
            "genuinely new measurement in the paper and they are distributed only as "
            "BigWig and TFRecord on a Google Cloud Storage bucket, with no GEO, SRA, "
            "Zenodo or Hugging Face accession anywhere in the paper. A sha256-pinned "
            "mirror is the only reproducible route and there is no archive fallback if "
            "the bucket moves, so the right action is the ask in Table 8, not a parse."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Zrimec 2020 (expression prediction from regulatory sequence)",
        reason=(
            "A modeling paper over expression values already held: its records are the "
            "public compendia re-encoded as model inputs, so admitting it would "
            "re-serve measurements the store has rather than add any."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Gu 2025",
        reason=(
            "A model trained on the regulatory-DNA libraries this wave ranks, not a new "
            "measurement. Its contribution is an architecture and a benchmark score, so "
            "there is no per-record value to store that is not already in de Boer, "
            "Vaishnav or the DREAM release."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Lei 2025",
        reason=(
            "Same class as Gu 2025: a sequence-to-expression model scored on the "
            "existing libraries. The report ranks the libraries and not the models, "
            "because a model's output is a prediction and the substrate stores "
            "measurements."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Cheng 2025",
        reason=(
            "A model over the same corpus again. Named here rather than omitted so the "
            "boundary is explicit: a benchmark result on a held-out split of a dataset "
            "already in the pool is not a second dataset."
        ),
        rule="derivative",
    ),
    Excluded(
        name="iIsor850 genome-scale model (Issatchenkia orientalis)",
        reason=(
            "A reconstruction, not an experiment. Its 850 gene calls are model output, "
            "which is the only form essentiality exists in for this species, and that "
            "is exactly why it cannot stand in for a screen. The yeast pool already "
            "excludes it under the same rule and the exclusion stands on its own merits "
            "even though this wave widens the host axis."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Wetmore 2015 RB-TnSeq methodology release",
        reason=(
            "3,903 genome-wide fitness experiments over 96,024 non-essential genes in "
            "25 bacteria, and already in the database. It is the method paper the "
            "cross-host fitness band's artifact comes from, so admitting it again would "
            "re-serve values the store holds."
        ),
        rule="already-built",
    ),
    Excluded(
        name="Fitness Browser GW456-L13 (Aquipseudomonas jessenii)",
        reason=(
            "A real row in the same artifact as the nine hosts this wave admits, at 106 "
            "experiments with 80 updated annotations, and its species has no member in "
            "this pool's declared Organism vocabulary. Folding it into a neighboring "
            "species would be a silent sequence-level error in a substrate whose claim "
            "is sequence-level genotype fidelity, so it waits on the Organism decision "
            "rather than being mislabeled. Admit it in the same change that adds the "
            "other nine."
        ),
        rule="off-host",
    ),
    Excluded(
        name="Gong 2018 (48-strain microsatellite panel, 9 antifungals)",
        reason=(
            "432 potential susceptibility values against nine drugs, which is more than "
            "twice the drug breadth of the sequenced panel, and the genotype is a "
            "microsatellite marker fingerprint rather than a sequence. There is no "
            "route to the strain's genomic content, so it cannot be joined to a genome "
            "even in principle."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="Wiehlmann 2015 (2,921-isolate 58-marker survey)",
        reason=(
            "2,921 isolates from 1,448 habitats in 323 clone types, and the 58-marker "
            "binary profile is a genotype fingerprint rather than a sequence or a "
            "measured phenotype. It must not be ingested as either."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="PkAC-7, PkAC-8 and PkAC-9 acetic-acid evolved strains",
        reason=(
            "Three strains from acetic acid stepped 7 to 9 g/L, with tolerance measured "
            "against six stresses, and no resequencing, no variant table and no "
            "accession in either search pass. The genotype is undefined, so there is "
            "nothing to map a phenotype from; the 70-gene expression panel on PkAC-9 "
            "inherits the same gap."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="RWT glucose osmotic-stress evolved mutant",
        reason=(
            "Two strains, tolerant to 35 percent glucose with best growth at 20 "
            "percent, and the mutations were never characterized. Same gap as the other "
            "evolved panels, and the osmotic axis is already covered by the 57-stressor "
            "isolate panel."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="Candida glycerinogenes as an ingestible host",
        reason=(
            "A proprietary, independently owned industrial strain with no public "
            "assembly, and the species name was folded into P. kudriavzevii by the "
            "population-genomics study that showed one species under four names. Its "
            "published values cannot be tied to a public genome."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="Blazeck 2012 (modular UAS and core hybrid promoters)",
        reason=(
            "A real S. cerevisiae regulatory-parts panel in two backgrounds with "
            "biological triplicate and standard deviations, and every value exists only "
            "as a bar height: the supplementary table is primers. Unusable as released, "
            "and worth re-checking only if the hybrid-promoter design becomes the "
            "question."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Nikel 2015 (EDEMP cycle flux)",
        reason=(
            "Reports that about 90 percent of consumed sugar is converted to gluconate "
            "and about 10 percent of triose phosphates are recycled to hexose "
            "phosphates, and the numbers sit in paper figures. Flux is the layer where "
            "the provenance discipline bites hardest, so the honest outcome is that the "
            "values are not retrievable rather than a thin extraction."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Puchalka 2008 (legacy KT2440 model validation)",
        reason=(
            "Validated a genome-scale model against continuous culture and substrate "
            "utilization assays, and the utilization matrix itself is not confirmed to "
            "be a downloadable table. The validation is a model result; the matrix is "
            "what would be ingestible and it was not located."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="PA14 pooled essential-gene knockdown library (dissertation only)",
        reason=(
            "A pooled library with one essential gene knocked down per strain, tested "
            "in a murine pneumonia model, described only in a dissertation. No "
            "deposited score table was confirmed, so it is a lead rather than a "
            "citable dataset."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Mobile-CRISPRi and P. aeruginosa CRISPRi optimization methods",
        reason=(
            "Vector systems and tuning studies: conjugation-transferable integrating "
            "CRISPRi vectors, dCas9 expression tuned to avoid toxicity, and guide "
            "designs for common laboratory strains. Methods rather than screens, with "
            "no per-record score tables."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Ozer 2019 (739-genome population structure)",
        reason=(
            "739 isolate genomes with source metadata, reporting that one lineage is "
            "99.4 percent exoS-positive and the other 95.2 percent exoU-positive. "
            "Genotype only, with no measured phenotype, so it belongs beside the "
            "reference genomes rather than in a phenotype table."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="43-strain international reference panel (BCCM/LMG)",
        reason=(
            "43 strains spanning cystic fibrosis, other clinical and environmental "
            "sources, 33 of them sequenced alongside 7 reference genomes. It is an "
            "order-the-strains resource rather than a phenotype table, and the genome "
            "accessions were not confirmed."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Murepavadin 414-isolate panel",
        reason=(
            "Reported as 414 cystic-fibrosis isolates against 14 antibiotics with 412 "
            "genomes, and the citation itself is unconfirmed. The isolate count "
            "coincides with the best-documented clinical panel's, which is reason to "
            "resolve the citation before treating it as a second dataset."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="SBW25 barcoded populations",
        reason=(
            "Chromosomal barcodes for lineage tracking, which the authors explicitly "
            "distinguish from transposon sequencing: nothing is mutagenized. A "
            "different assay, whose value would be evolution and lineage phenotypes "
            "rather than gene fitness."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Pseudomonas Genome DB and the Pseudomonas Ortholog Database",
        reason=(
            "Community annotation and orthology over 14,229 genomes, with ortholog "
            "groups and batch downloads, and no evidence it serves a phenotype table at "
            "all. It is the mapping substrate for the alias layer, which is what the "
            "Bakta and PA-number rows need, not a phenotype source."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="GSE139245 (data-integrity flag)",
        reason=(
            "A 15-sample series whose index record carries a summary describing "
            "mammalian histone-modification profiling, inconsistent with both its own "
            "title and its taxon. Treated as suspect rather than ingested."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Illinois Data Bank IDB-2926789 (constraint-based organic-acid potential)",
        reason=(
            "Predicted production potential for 22 organic acids from a genome-scale "
            "model, which is the same class as the model-derived essential-gene list: "
            "model output, not measurement. The growth-coupled knockout designs built "
            "on the same model fall under this too."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Mendeley Data methylotrophic-protein deposit",
        reason=(
            "A P. kudriavzevii deposit for open production of methylotrophic protein. "
            "Project policy disallows this repository as a source, so the deposit is "
            "recorded and not recommended."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="PXD044970 and PXD048012",
        reason=(
            "The first lists a different species entirely, Saccharomyces kudriavzevii, "
            "and the second lists P. kudriavzevii while its displayed abstract belongs "
            "to an unrelated glycolysis study, so it may be mislabeled. Both are "
            "suspect and neither may be used without a check."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Komagataella phaffii methanol cross-tolerance study",
        reason=(
            "A different species. Named because it surfaced repeatedly in searches for "
            "this wave's low-data host and is not it."
        ),
        rule="off-host",
    ),
    Excluded(
        name="681-member environmental yeast isolate collection",
        reason=(
            "Multi-species rather than specific to this wave's host, so its records "
            "cannot be assigned to one organism without splitting the collection first."
        ),
        rule="off-host",
    ),
    Excluded(
        name="FUN-PROSE expression-prediction preprint",
        reason=(
            "A modeling paper in the same group's orbit as the host's expression work, "
            "not a dataset: its records are the matrices it trains on."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Karollus 2024 benchmark deposit",
        reason=(
            "Model weights plus an unitemized supporting-data bundle. Every measured "
            "dataset behind it belongs to the terminator, promoter and 3' end rows this "
            "pool already ranks, so the right action is to take those."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Martinez-Carranza 2019 essentiality comparison",
        reason=(
            "A comparison of published essentiality screens, reporting 336 essential in "
            "one reference strain and 434 in the other with 20 having no ortholog, and "
            "only 192 genes shared by all screens in one strain against 17 in the "
            "other. Those figures are the reason a call must carry its method, medium "
            "and strain; the comparison itself re-serves the five studies' calls."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Rossi 2021 ChIP-exo and ChIP-MNase track set",
        reason=(
            "1,229 datasets over 791 targets, 400 of them reproducible, and not a "
            "genotype-by-environment record at all: it is a molecular-layer track set. "
            "It belongs in the artifact tier as a feature source if anywhere, which is "
            "a features decision rather than a dataset admission."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="mLife 2026 genome-editing system",
        reason=(
            "A genome-editing system in a nonconventional yeast whose organism is not "
            "confirmed to be this wave's host, so the row cannot even be assigned a "
            "host. One record check resolves it."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Lactic-acid stress comparative transcriptomics",
        reason=(
            "A fermented-grain isolate against a type strain under lactic-acid stress, "
            "with differential expression in pyruvate metabolism, transporters and "
            "amino acid biosynthesis. Comparative rather than evolved, and neither "
            "strain's assembly was located, so there is no route to either genotype."
        ),
        rule="no-sequence",
    ),
    Excluded(
        name="Sc2.0 synthetic-genome program; CNGB CNP0002230",
        reason=(
            "The 2023 Cell papers' phenotype claims appear as figures, and CNP0002230 "
            "lists 231 samples across genome sequencing, proteome and transcriptome "
            "with no phenotype data type at all. A genome-engineering program is not "
            "gene-indexed perturbation-phenotype data, and there is no per-record value "
            "to ingest today."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Genome-wide KT2440 CRISPRi library (78,932 members)",
        reason=(
            "The highest-value missing dataset in the survey and effectively "
            "request-only: 10 to 15 gRNAs per gene with 798 non-targeting controls, "
            "screened on glucose, acetate and p-coumarate at two concentrations, "
            "existing only as a Department of Energy abstract with no publication, no "
            "repository and no accession. It cannot be ranked against rows whose "
            "numbers were read off a file, so it is an ask rather than a row."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Werner 2023 and the engineered-strain product literature",
        reason=(
            "Supplementary-table-only across the board. In Werner 2023, the one paper "
            "whose data-availability statement was read verbatim, every number is in "
            "sciadv.adj0053_data_s1.zip and the SI PDF, there is no repository of any "
            "kind, and the strains are request-only under a material transfer "
            "agreement. The tier is worth ingesting for its named genotypes, which is "
            "why those rows are in the pool, and not for record count."
        ),
        rule="no-per-record-data",
    ),
    Excluded(
        name="Fitness Browser fitprivate instance (104 organisms, 989 KT2440 experiments)",
        reason=(
            "A non-public instance whose counts disagree with the live public page's 74 "
            "organisms and 314 KT2440 experiments. The live public page supersedes every "
            "other count, and fitprivate numbers must never be mixed in, so the instance "
            "is named here to keep its figures out of any total."
        ),
        rule="not-a-dataset",
    ),
    Excluded(
        name="Fitness Browser KBase mirror (48 organisms) and the June 2017 release",
        reason=(
            "Stale mirrors of the same artifact. The bulk page itself says the 2018 "
            "release's sucrose and D-mannitol data should be disregarded in favor of "
            "fresh stock data, so an older copy is not an independent dataset and "
            "ingesting one would serve values the source has retracted."
        ),
        rule="derivative",
    ),
    Excluded(
        name="Wheeldon genome-scale CRISPR libraries (Y. lipolytica, K. phaffii)",
        reason=(
            "Real genome-scale resources, including a 6-fold-coverage K. phaffii library "
            "of 30,848 active guides covering more than 99 percent of the genome, and in "
            "neither case the host this wave bootstraps. Searched specifically because a "
            "Pichia screen would be the single most valuable row available; the platform "
            "has never been extended to P. kudriavzevii."
        ),
        rule="off-host",
    ),
    Excluded(
        name="ProteomeCentral P. putida project pages",
        reason=(
            "JavaScript-rendered, so a crawl returns nothing. This is a retrieval route "
            "and not a dataset: the same projects are reachable through the PRIDE v3 "
            "REST API, which is how the proteomics rows in the pool are counted."
        ),
        rule="not-a-dataset",
    ),
]


# ---------------------------------------------------------------------------
# The asks. The shortest list in this document and the highest leverage: each is one
# message, and for three of these rows it is the only route to the data at all.
# ---------------------------------------------------------------------------

ASKS: list[Ask] = [
    Ask(
        dataset="IoSL resequencing series (PRJNA1464946/47/48)",
        organism="P. kudriavzevii",
        who=(
            "DOE Joint Genome Institute, submitter of PRJNA1464946, PRJNA1464947 and "
            "PRJNA1464948, registered 2026-05-11"
        ),
        request=(
            "How many strains the IoSL series covers, what perturbation produced them, "
            "and whether any phenotype was measured alongside the sequencing"
        ),
        unlocks=(
            "The only remaining candidate for a genome-wide P. kudriavzevii strain "
            "library. Three sampled BioProjects are titled Issatchenkia orientalis SD108 "
            "IoSL10, IoSL16 and IoSL19 resequencing, with no publication attached; a "
            "series sequenced at that scale is either a mutagenized or an evolved SD108 "
            "library, and if it carries phenotypes it is the resource this wave cannot "
            "otherwise find"
        ),
    ),
    Ask(
        dataset="MAESTRO 170-isolate panel",
        organism="P. kudriavzevii",
        who=(
            "Ping-Hung Hsieh (LBNL), corresponding; also Yasuo Yoshikuni (LBNL/JGI), "
            "Huimin Zhao (UIUC/CABBI), Chris Todd Hittinger and Dana Opulente for the "
            "isolates, Igor Grigoriev (JGI) for the genomes, Andrew Leakey (CABBI)"
        ),
        request=(
            "The 161-isolate by 57-stressor phenotype matrix as a file, the complete "
            "isolate-to-accession map, and the per-isolate VCF and CNV tables"
        ),
        unlocks=(
            "This host's largest phenotype resource by measurements, and the per-isolate "
            "variant calls behind the tandem-paralog CNV finding, none of which has a "
            "public accession. bioRxiv returns 403 for the full text and Europe PMC's "
            "fullTextXML for PPR1078985 returns 500, so the supplementary table is "
            "unreachable without a person"
        ),
    ),
    Ask(
        dataset="I. orientalis 5,140-gene by 69-condition expression matrix",
        organism="P. kudriavzevii",
        who="Saurabh Sinha's group at UIUC, with Payam Dibaeinia",
        request=(
            "Whose data the matrix is, whether a release exists, and under what terms it "
            "could be deposited with a provenance record"
        ),
        unlocks=(
            "A 69-condition expression matrix for this host. The senior thesis carrying "
            "it has no data-availability, code, deposition or supplementary section of "
            "any kind, so it is reachable by asking and by nothing else; with a real "
            "provenance record it would be worth more to this host's program than "
            "anything else in that corpus"
        ),
    ),
    Ask(
        dataset="Shorkie IDEA induction RNA-seq tracks",
        organism="S. cerevisiae",
        who="Calico Life Sciences (Chao, Magzoub, Stoops, Hackett, Linder, Kelley)",
        request=(
            "A GEO or SRA deposit for the 3,053 induction RNA-seq tracks, or a stable "
            "archival copy with a checksum manifest"
        ),
        unlocks=(
            "The one genuinely new measurement in that paper, currently distributed only "
            "as BigWig and TFRecord on a Google Cloud Storage bucket with no accession "
            "anywhere in the paper and no archive fallback if the bucket moves"
        ),
    ),
    Ask(
        dataset="Genome-wide KT2440 CRISPRi library (78,932 members)",
        organism="P. putida",
        who=(
            "CU Boulder, ORNL and the Center for Bioenergy Innovation: Fenster, Hren, "
            "Spangler, Alexander, Fox, Cameron, Eckert, Tuskan"
        ),
        request=(
            "Gene-level or guide-level scores from the library, and whether a deposit "
            "exists anywhere, for the glucose, acetate and p-coumarate selections"
        ),
        unlocks=(
            "The complement to RB-TnSeq, covering the essential genes a transposon "
            "library cannot score by construction. Abstract-only today: no publication, "
            "no repository, no accession, so no file exists to hash"
        ),
    ),
    Ask(
        dataset="Landing-pad locus screen (Fatma 2023)",
        organism="P. kudriavzevii",
        who="Huimin Zhao lab (UIUC)",
        request=(
            "An archival deposit for the landing-pad package, currently a lab-hosted "
            "uofi.box.com URL plus a GitHub repository"
        ),
        unlocks=(
            "The only measured per-locus fitness data in this species, and "
            "genotype-precise, behind a URL that can rot. An institutional Box link is "
            "not an archive and nothing mirrors it"
        ),
    ),
    Ask(
        dataset="Fitness Browser licensing",
        organism=None,
        who="LBL Fitness Browser maintainers",
        request=(
            "The license and terms of use for the live feba.db release and for the June "
            "2017 archival release"
        ),
        unlocks=(
            "The whole cross-host fitness band, nine organisms, gated by a Cloudflare "
            "challenge rather than by anything substantive: help.cgi and the organism "
            "list return 403 with Cf-Mitigated: challenge to every non-browser client "
            "and the Figshare archival record page returns 403 as well, so nobody has "
            "read a license string for either release"
        ),
    ),
    Ask(
        dataset="PAO1 genome-wide CRISPRi (Kaczmarczyk 2025)",
        organism="P. aeruginosa",
        who="Jenal lab (Kaczmarczyk, Klotz, Manfredi, Jenal)",
        request=(
            "Which of the three Zenodo records holds the gene-level score table, if "
            "opening them does not settle it"
        ),
        unlocks=(
            "The only pseudomonad CRISPRi score table plausibly downloadable today, "
            "covering the essential genes RB-TnSeq structurally cannot"
        ),
    ),
    Ask(
        dataset="Vaishnav 2022 deferred libraries",
        organism="S. cerevisiae",
        who="Vaishnav 2022 authors",
        request="The meaning of sdEL, relativeSD and sdScale",
        unlocks=(
            "Already an open owner question in vaishnav2022.py and listed under issue "
            "1172. de Boer's Supplementary Tables may answer it without an ask, since "
            "Vaishnav's Methods defer to de Boer, so one page check comes first"
        ),
    ),
    Ask(
        dataset="Dotsch 2009 arrayed PA14 screen",
        organism="P. aeruginosa",
        who="The study's authors, or whoever now holds the arrayed PA14 library data",
        request=(
            "Whether the full 5,459-mutant by 19-antibiotic susceptibility matrix was "
            "released, and where"
        ),
        unlocks=(
            "If the full matrix exists it is the densest arrayed-library phenotype "
            "matrix in the species. This is the highest-value unresolved pseudomonad "
            "item and the value location is unconfirmed, so the row cannot be costed "
            "until someone answers"
        ),
    ),
]


# ---------------------------------------------------------------------------
# Rendering. The same helpers and table shapes both siblings use, so the three
# documents read as one family and a reader who knows one table can read the others.
# ---------------------------------------------------------------------------


def tex_escape(s: str) -> str:
    """Escape every TeX special in free text.

    The backslash goes FIRST, or it would double-escape the backslashes the later
    replacements introduce. ``^`` and ``~`` are here because they are not merely
    special but silently fatal in text mode, which is how the bacteria table found
    them: a source note reading "about 5 by 10^5 mutants" halts the engine.
    """
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ]:
        s = s.replace(a, b)
    return s


def href_target(url: str) -> str:
    r"""A URL as ``\href`` can read it.

    ``\href`` does not read its first argument verbatim, so a literal ``%`` comments
    out the rest of the line and a literal ``#`` is a parameter character. Both appear
    in this wave's accession links: a percent-encoded NCBI query carries ``%22`` and
    ``%5B``. Only these two characters are touched, because the target has to resolve
    byte for byte.
    """
    return url.replace("%", r"\%").replace("#", r"\#")


# Characters after which TeX may break a URL or an accession. The display forms are the
# post-escape spellings, since tex_escape runs first. A break opportunity costs nothing:
# \allowbreak inserts no character, so the string a reader copies is unchanged.
_BREAK_AFTER: tuple[str, ...] = ("/", "?", r"\&", "=", r"\%", r"\_", "-", ".", ":", "+")
# Longest run of characters with no break opportunity before one is forced in. A
# percent-encoded query string such as ?term=%22Pseudomonas+putida%22%5BOrganism%5D has
# long alphabetic runs between its separators, and a 46 mm overflow past the text block
# is what happens without this.
_CHUNK = 8


def _chunk_runs(s: str) -> str:
    r"""Insert a break opportunity inside any long run of unbreakable characters.

    Walks the already-escaped string and counts since the last break opportunity,
    emitting ``\allowbreak`` once the run reaches ``_CHUNK``. TeX control sequences
    introduced by escaping (``\&``, ``\%``, ``\_``, ``\textbackslash{}``) must not be
    split, so a backslash starts a protected span that runs to the end of its command
    name and of any following braces.
    """
    out: list[str] = []
    run = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\":
            # Copy the whole control sequence, then its argument braces if present.
            j = i + 1
            while j < len(s) and s[j].isalpha():
                j += 1
            while j < len(s) and s[j] in "{}":
                j += 1
            out.append(s[i:j])
            run = 0
            i = j
            continue
        if run >= _CHUNK:
            out.append(r"\allowbreak ")
            run = 0
        out.append(ch)
        run = 0 if ch in "/?=-.:+" else run + 1
        i += 1
    return "".join(out)


def breakable(s: str) -> str:
    r"""Escape a path, accession or URL and give TeX somewhere to break it.

    Breaking only at slashes is not enough. This wave's links carry query strings and
    percent-encoded arguments, not just DOIs: an NCBI organism query and a PRIDE
    keyword search are each one unbreakable token to TeX and ran up to 131.7 pt, about
    46 mm, past the text block. So every separator is a break opportunity and any long
    run between separators gets one too.

    Nothing is added to the text the reader copies: ``\allowbreak`` is a break
    opportunity and not a character, so no hyphen, space, ellipsis or truncation enters
    the URL. That is the whole reason to use it rather than shortening the string.
    """
    out = tex_escape(s)
    for tok in _BREAK_AFTER:
        out = out.replace(tok, tok + r"\allowbreak ")
    return _chunk_runs(out)


def link_tex(url: str) -> str:
    """A clickable, short display form of a source URL.

    The full URL is too wide for a table column and mostly boilerplate, so the
    common prefixes collapse to a label while the href keeps the real target.
    """
    shown = url
    for prefix, label in (
        ("https://doi.org/", "doi:"),
        ("https://pubmed.ncbi.nlm.nih.gov/", "PMID "),
        ("https://pmc.ncbi.nlm.nih.gov/articles/", ""),
        ("https://www.ncbi.nlm.nih.gov/pmc/articles/", ""),
        ("https://www.", ""),
        ("https://", ""),
    ):
        if shown.startswith(prefix):
            shown = label + shown[len(prefix) :]
            break
    shown = shown.rstrip("/")
    return r"\href{" + href_target(url) + r"}{" + breakable(shown) + r"}"


def narrow_tex(s: str) -> str:
    r"""Free text for a narrow column, with a break opportunity after each hyphen.

    TeX disables hyphenation inside a word that already contains an explicit hyphen, so
    "4-hydroxybenzoate" and "5-aminovalerate" are each one unbreakable token in a 17 mm
    column and ran a few points past the text block. ``\allowbreak`` after the hyphen
    restores a break point and adds no character, where a discretionary hyphen would
    print a second one.

    Applied only to the narrow axis columns, not to the reason columns: those are wide
    enough to break on spaces, and breaking "genome-wide" mid-table there buys nothing.
    """
    return tex_escape(s).replace("-", r"-\allowbreak ")


def status_tex(status: str) -> str:
    """Marker for a row that is not a plain candidate."""
    return {
        "candidate": "",
        "blocked": r"\,\textsuperscript{\textbf{B}}",
        "built": r"\,\textsuperscript{\textbf{X}}",
        "aggregation": r"\,\textsuperscript{\textbf{A}}",
    }[status]


def retrieval_tex(how: str) -> str:
    """Marker for how the per-record values are obtained."""
    return {
        "retrievable": "",
        "manual_fetch": r"\,\textsuperscript{\textbf{M}}",
        "ask": r"\,\textsuperscript{\textbf{R}}",
    }[how]


def verified_tex(label: str) -> str:
    """The report's verification label, as the one-letter mark it uses."""
    return {"verified": "V", "snippet": "S", "inferred": "I"}[label]


def seq_tex(basis: str) -> str:
    """Sequence-basis label with break points at + and -.

    "plasmid-reporter-library" is one unbreakable token to TeX, so a narrow column
    cannot wrap it and the row runs off the text block. Hyphenation does not help:
    the string is not a word.
    """
    return (
        tex_escape(basis).replace("+", r"+\allowbreak ").replace("-", r"-\allowbreak ")
    )


def org_tex(organism: str) -> str:
    """Host abbreviation, italic per house style for an organism name."""
    return r"\org{" + tex_escape(organism) + r"}"


def sci(n: int | None) -> str:
    """A count as an order-of-magnitude figure, the way the built table reports it."""
    if n is None:
        return "--"
    if n < 1000:
        return f"{n:,}"
    exp = int(math.floor(math.log10(n)))
    mant = n / 10**exp
    return f"${mant:.1f}\\times 10^{{{exp}}}$"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


def _check_rows() -> None:
    """Fail loudly on the defects a frozen row literal can still carry.

    Every check here exists because the alternative is a silent defect: a band with no
    reason, an anonymous ask, a duplicate name that would make two rows one, or an
    excluded entry that is also a ranked row.
    """
    by_name = {c.name: c for c in CANDIDATES}
    if len(by_name) != len(CANDIDATES):
        counts = Counter(c.name for c in CANDIDATES)
        dupes = sorted(name for name, n in counts.items() if n > 1)
        raise SystemExit(f"duplicate candidate name: {dupes}")
    for c in CANDIDATES:
        if c.band != "scale" and not c.band_why:
            raise SystemExit(f"{c.name}: banded out of scale with no reason")
        if (c.retrievability == "ask") != bool(c.ask_of):
            raise SystemExit(
                f"{c.name}: retrievability {c.retrievability!r} and ask_of "
                f"{c.ask_of!r} disagree; an ask row names who to ask"
            )
    for c in CANDIDATES:
        if not c.url.startswith("http"):
            raise SystemExit(f"{c.name}: url is not resolvable: {c.url!r}")
    for name, rank, _ in OWNER_PINS:
        if name not in by_name or by_name[name].status == "built":
            raise SystemExit(f"owner pin names no ranked row: {name!r}")
        if not 1 <= rank <= len(by_name):
            raise SystemExit(f"owner pin rank {rank} is outside the pool: {name!r}")
    overlap = sorted({e.name for e in EXCLUDED} & set(by_name))
    if overlap:
        raise SystemExit(f"named both ranked and excluded: {overlap}")
    for a in ASKS:
        if a.dataset in by_name and by_name[a.dataset].retrievability != "ask":
            raise SystemExit(
                f"{a.dataset}: in ASKS but its row is {by_name[a.dataset].retrievability}"
            )


def _lift(
    rows: list[Candidate],
    boundary: int,
    wanted: Callable[[Candidate], bool],
    protected: Callable[[Candidate], bool],
    reason: str,
    moves: list[tuple[str, str, int, int]],
    limit: int,
) -> None:
    """Move up to ``limit`` rows satisfying ``wanted`` inside the first ``boundary``.

    One lift per iteration, strongest row below the boundary first, displacing the
    weakest row inside it that nothing protects. Every move is appended with its before
    and after rank, so the printed order can be reconciled against the pure measurement
    order row by row.
    """
    for _ in range(limit):
        below = [c for c in rows[boundary:] if wanted(c)]
        if not below:
            return
        promote = below[0]
        inside = [c for c in rows[:boundary] if not protected(c)]
        if not inside:
            return  # the boundary is entirely protected; nothing left to displace
        demote = max(inside, key=lambda c: c.sort_key)
        old = rows.index(promote) + 1
        rows.remove(promote)
        rows.insert(rows.index(demote), promote)
        rows.remove(demote)
        rows.insert(boundary, demote)
        moves.append((promote.name, reason, old, rows.index(promote) + 1))


def ranked() -> tuple[list[Candidate], list[tuple[str, str, int, int]]]:
    """Rank by band, then the tier rule, then the per-host floor, then owner pins.

    Returns the ordered rows and the floor's moves as (name, reason, from_rank,
    to_rank), so the document can name every row that was lifted rather than hiding
    the reordering inside a score. A row with ``status="built"`` leaves the ranking:
    it is an overlap, not a candidate.
    """
    rows = sorted(
        (c for c in CANDIDATES if c.status != "built"), key=lambda c: c.sort_key
    )
    moves: list[tuple[str, str, int, int]] = []

    def of_host(host: str) -> Callable[[Candidate], bool]:
        """A predicate for one host, bound by a factory so its type is inferable."""
        return lambda c: c.organism == host

    def at_floor(c: Candidate) -> bool:
        """Whether this row's host is at or under its floor inside the tranche.

        A row already counted toward another host's floor is protected, so one host's
        promotion never costs a row the other host's floor is holding. Read live against
        ``rows``, because the tranche changes as lifts are applied.
        """
        inside = sum(1 for d in rows[:TRANCHE_1] if d.organism == c.organism)
        return inside <= QUOTA_1.get(c.organism, 0)

    for organism, quota in QUOTA_1.items():
        have = sum(1 for c in rows[:TRANCHE_1] if c.organism == organism)
        _lift(
            rows,
            TRANCHE_1,
            of_host(organism),
            at_floor,
            f"{organism} floor",
            moves,
            max(0, quota - have),
        )
    for name, rank, _ in OWNER_PINS:
        row = next(c for c in rows if c.name == name)
        old = rows.index(row) + 1
        rows.remove(row)
        rows.insert(rank - 1, row)
        moves.append((name, "owner pin", old, rank))
    return rows, moves


def priority_tex(c: Candidate) -> str:
    """Why the row sits where it does: the pin, the band reason, or the scale rule."""
    if c.name in PIN_REASON:
        return tex_escape(PIN_REASON[c.name])
    if c.band != "scale":
        return tex_escape(c.band_why)
    return tex_escape(
        f"Scale band: tier {c.tier}, ordered by measurements ({c.measurements:,})."
        if c.measurements
        else f"Scale band: tier {c.tier}."
    )


def _dataset_cell(c: Candidate) -> str:
    """The dataset name, its marks, and the class and phenotype on a second line."""
    return (
        r"\textbf{"
        + tex_escape(c.name)
        + r"}"
        + status_tex(c.status)
        + retrieval_tex(c.retrievability)
        + r"\newline {\scriptsize "
        + tex_escape(c.klass)
        + "; "
        + tex_escape(c.phenotype)
        + "}"
    )


LEGEND = (
    r"Superscript \textbf{M} marks a row whose values need a human with a browser and "
    r"\textbf{R} a row that needs a request to a named party (Table~\ref{tab:asks}); "
    r"\textbf{B} marks a row whose per-record values are not released and \textbf{A} a "
    r"corpus that re-serves other papers. \emph{Ver.} carries the triage report's own "
    r"verification label for the counts: \textbf{V} read on the source in session, "
    r"\textbf{S} a search snippet only, \textbf{I} inferred, which includes every count "
    r"that is the product of two verified numbers."
)


def render_final(rows: list[Candidate]) -> str:
    """The recommended fifty and ten extra, every stat the ranking used beside the reason."""
    final = rows[:FINAL]
    cols = (
        r"@{}r@{\hspace{3pt}} L{38mm} L{11mm} L{14mm} L{18mm} L{20mm} r@{\hspace{4pt}} "
        r"r@{\hspace{4pt}} c@{\hspace{4pt}} L{20mm} L{66mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset (class; phenotype)} & \textbf{Band, tier} & "
        r"\textbf{Host} & \textbf{Genotypes} & \textbf{Env} & \textbf{Inst.} & "
        r"\textbf{Meas.} & \textbf{Ver.} & \textbf{Sequence basis} & \textbf{Why} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The recommended first fifty of the cross-host wave, then ten extra in rank
order in case a row above proves unreachable. \emph{Band, tier}: \emph{R} is regulatory
DNA, the one axis that adds measurements rather than rows; \emph{F} is cross-host
fitness, the Fitness Browser rows that supply the row count; \emph{H} is host bootstrap,
which exists because \org{P. kudriavzevii} cannot win a measurement contest against a
pseudomonad fitness compendium under any ordering; \emph{S} is scale. The band is applied
before tier, and inside a band the tier rule orders, then measurements. A row marked
\emph{pin} was placed at its rank by the owner and not by that rule
(Sec.~\ref{sec:pin}), and Table~\ref{tab:sources} prints the reason. \emph{Genotypes} and \emph{Env} are the perturbation and condition axes.
\emph{Inst.} is genotype$\times$environment records, $\dagger$ where it is the product of
the two axes rather than a reported count and $\ddagger$ where it is an
order-of-magnitude estimate. \emph{Meas.} is instances times phenotype dimensionality,
the quantity rows are ranked on. \emph{Sequence basis} is the route to each strain's
total genomic content; a row with no route is excluded rather than ranked
(Table~\ref{tab:excluded}). """
        + LEGEND
        + r""" Citations, links and data locations are in
Table~\ref{tab:sources}; the gene-identifier namespace each host's ids speak is in
Table~\ref{tab:namespaces}; what the wave changes in the schema is in
Table~\ref{tab:schema}.}
\label{tab:final}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{11}{@{}l}{\footnotesize\emph{Table~\ref{tab:final}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for i, c in enumerate(final, start=1):
        if i == WAVE_1 + 1:
            lines.append(
                r"\midrule \multicolumn{11}{@{}l}{\textbf{Ten extra, in rank order: rows "
                + str(WAVE_1 + 1)
                + r"--"
                + str(FINAL)
                + r" replace a row above that proves unreachable.}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        lines.append(
            " & ".join(
                [
                    str(i),
                    _dataset_cell(c),
                    BAND_LETTER[c.band]
                    + ", "
                    + str(c.tier)
                    + (", pin" if c.name in PIN_REASON else ""),
                    org_tex(c.organism),
                    narrow_tex(c.genotypes),
                    narrow_tex(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    verified_tex(c.verified),
                    seq_tex(c.seq_basis),
                    tex_escape(c.why),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_candidates(rows: list[Candidate]) -> str:
    r"""The whole pool in rank order.

    Landscape. Portrait gives 182 mm, which would force eleven columns so narrow that
    every free-text cell hyphenates two or three times and the table stops being
    scannable. lscape swaps \textwidth for \textheight, so the block is 261 mm here,
    and every free-text column is ragged-right rather than justified: justification in a
    20 mm column is what produces stretched, over-hyphenated lines.
    """
    cols = (
        r"@{}r@{\hspace{4pt}} L{40mm} L{13mm} L{21mm} L{21mm} L{21mm} r@{\hspace{5pt}} "
        r"r@{\hspace{5pt}} c@{\hspace{5pt}} L{24mm} L{21mm} L{28mm}@{}"
    )
    hdr = (
        r"\textbf{\#} & \textbf{Dataset} & \textbf{Host} & \textbf{Class} & "
        r"\textbf{Genotypes} & \textbf{Env} & \textbf{Instances} & \textbf{Meas.} & "
        r"\textbf{Ver.} & \textbf{Phenotype} & \textbf{Sequence basis} & \textbf{Link} \\"
    )
    shortfall = CUT - len(rows)
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{Every candidate in the cross-host pool, in rank order: band first, then tier,
then measurements descending. \emph{Genotypes} and \emph{Env} are the perturbation and
condition axes. \emph{Instances} is the number of genotype$\times$environment records,
marked $\dagger$ where it is the product of the two axes rather than a figure the source
reports, and $\ddagger$ where it is an order-of-magnitude estimate. \emph{Meas.} is
instances times phenotype dimensionality, which is what a vector-valued panel actually
contributes and what rows are ranked on. \emph{Sequence basis} is how the total genomic
content of one strain would be reconstructed; a row with no such route is excluded rather
than ranked (Table~\ref{tab:excluded}). """
        + LEGEND
        + r""" \emph{Link} resolves to the source and is clickable; the full citation, the
reason for the position and the data location are in Table~\ref{tab:sources-all}. The
RECOMMENDED set is the fifty of Table~\ref{tab:final} and its ten spares, not this table:
rows 1--"""
        + str(min(CUT, len(rows)))
        + r""" here are the rows above the cut, meaning the additions this wave targets"""
        + (
            r"; the pool holds "
            + str(len(rows))
            + r" rows, "
            + str(shortfall)
            + r" short of the "
            + str(CUT)
            + r" this wave adds, so the shortfall is open discovery, not a reserve"
            if shortfall > 0
            else ""
        )
        + r""".}
\label{tab:candidates}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{12}{@{}l}{\footnotesize\emph{Table~\ref{tab:candidates}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    dividers = {
        WAVE_1 + 1: (
            r"End of wave 1. Rows 1--"
            + str(WAVE_1)
            + r" are the recommended next builds."
        ),
        WAVE_2 + 1: (
            r"End of wave 2. Rows "
            + str(WAVE_1 + 1)
            + r"--"
            + str(WAVE_2)
            + r" are the bench, promoted as soon as a wave-1 row proves unreachable."
        ),
        CUT + 1: (
            r"Long-run cut. Rows above are the "
            + str(CUT)
            + r" this wave adds; rows below are the ranked reserve."
        ),
    }
    lines = []
    for i, c in enumerate(rows, start=1):
        if i in dividers:
            lines.append(
                r"\midrule \multicolumn{12}{@{}l}{\textbf{"
                + dividers[i]
                + r"}}\\ \midrule"
            )
        mark = {"reported": "", "product": r"$\dagger$", "estimate": r"$\ddagger$"}[
            c.instances_basis
        ]
        lines.append(
            " & ".join(
                [
                    str(i),
                    r"\textbf{"
                    + tex_escape(c.name)
                    + r"}"
                    + status_tex(c.status)
                    + retrieval_tex(c.retrievability),
                    org_tex(c.organism),
                    tex_escape(c.klass),
                    narrow_tex(c.genotypes),
                    narrow_tex(c.env),
                    sci(c.instances_n) + mark,
                    sci(c.measurements),
                    verified_tex(c.verified),
                    narrow_tex(c.phenotype),
                    seq_tex(c.seq_basis),
                    link_tex(c.url),
                ]
            )
            + r" \\"
        )
        # The same 5 pt gap the other two documents use, and for the same reason: every
        # row here is several lines of wrapped text, so without it adjacent rows run
        # together and the eye cannot find where one ends.
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_sources(
    rows: list[Candidate],
    *,
    label: str = "tab:sources",
    of: str = "tab:final",
    scope: str = "the recommended fifty and the ten extra",
) -> str:
    """Citation, link, why the row is ordered where it is, and where the data lives."""
    hdr = (
        r"\textbf{\#} & \textbf{Dataset, citation and link} & "
        r"\textbf{Why it is ordered here} & \textbf{Where the per-record data lives} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}r@{\hspace{4pt}} L{78mm} L{72mm} L{76mm}@{}}
\caption[]{Sources for """
        + scope
        + r""" of Table~\ref{"""
        + of
        + r"""}, in the same order.
\emph{Why it is ordered here} is the band reason for a host-bootstrap, regulatory-DNA or
cross-host-fitness row, and the tier and measurement rule for a scale row. \emph{Where
the per-record data lives} is the deposit, with its retrieval class: a plain entry is
fetched programmatically, \emph{browser} needs a human because the host returns 403 or
renders in JavaScript, and \emph{ask} means no public copy exists and the request is in
Table~\ref{tab:asks}. Every link is clickable.}
\label{"""
        + label
        + r"""}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{4}{@{}l}{\footnotesize\emph{Table~\ref{"""
        + label
        + r"""}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    note = {
        "retrievable": "",
        "manual_fetch": r"\newline {\scriptsize (browser)}",
        "ask": r"\newline {\scriptsize (ask)}",
    }
    lines = []
    for i, c in enumerate(rows, start=1):
        cite = (
            r"\textbf{"
            + tex_escape(c.name)
            + r"} "
            + org_tex(c.organism)
            + r"\newline "
            + tex_escape(c.citation)
            + r"\newline "
            + link_tex(c.url)
        )
        acc = breakable(c.accession) + note[c.retrievability]
        lines.append(" & ".join([str(i), cite, priority_tex(c), acc]) + r" \\")
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_counts(rows: list[Candidate]) -> str:
    """Class by tranche and host by tranche, off the same ordering."""
    head = r"""\begin{table}[H]\centering
\small
\caption[]{Candidates by class and by host, split at the two wave lines.
\emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the per-row axes over the whole
pool; a row with no count contributes nothing, so every total is a lower bound.
\emph{Meas.} is instances times phenotype dimensionality, which is why the per-host
ordering of the measurement column does not follow the row counts: one yeast
regulatory-DNA row carries more measurements than every pseudomonad row together.}
\label{tab:counts}
\begin{tabular}{@{}l r r r r r r@{}}
\toprule
 & Wave 1 & Wave 2 & Reserve & Genotypes & Instances & Meas. \\
\midrule
"""

    # Sequence rather than list, so a sorted set of Literal values can be passed
    # without mypy complaining that list is invariant.
    def block(key: str, values: Sequence[str]) -> list[str]:
        out = []
        for v in values:
            members = [(i, c) for i, c in enumerate(rows, 1) if getattr(c, key) == v]
            if not members:
                continue
            w1 = [c for i, c in members if i <= WAVE_1]
            w2 = [c for i, c in members if WAVE_1 < i <= WAVE_2]
            rest = [c for i, c in members if i > WAVE_2]
            g = sum(c.genotypes_n or 0 for _i, c in members)
            n = sum(c.instances_n or 0 for _i, c in members)
            m = sum(c.measurements or 0 for _i, c in members)
            out.append(
                f"{tex_escape(v)} & {len(w1)} & {len(w2)} & {len(rest)} & "
                f"{g:,} & {sci(n)} & {sci(m)} \\\\"
            )
        return out

    klasses = sorted({c.klass for c in rows})
    hosts = sorted({c.organism for c in rows})
    lines = block("klass", klasses)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{The same rows, by host}}\\")
    lines += block("organism", hosts)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{7}{@{}l}{\emph{The same rows, by band}}\\")
    lines += block("band", list(BAND_ORDER))
    g_all = sum(c.genotypes_n or 0 for c in rows)
    n_all = sum(c.instances_n or 0 for c in rows)
    m_all = sum(c.measurements or 0 for c in rows)
    lines.append(r"\midrule")
    lines.append(
        f"Total & {min(WAVE_1, len(rows))} & "
        f"{max(min(WAVE_2, len(rows)) - WAVE_1, 0)} & "
        f"{max(len(rows) - WAVE_2, 0)} & "
        f"{g_all:,} & {sci(n_all)} & {sci(m_all)} \\\\"
    )
    return head + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"


def render_summary(rows: list[Candidate]) -> str:
    """Summary of the recommended set, split at the fifty line.

    Every figure here is computed from the same rows the final table prints, so the two
    cannot disagree. Genotype, instance and measurement sums are lower bounds: a row
    with no count contributes nothing.
    """
    final = rows[:FINAL]
    top = final[:WAVE_1]
    extra = final[WAVE_1:]

    def count(pred: Any) -> str:
        a = sum(1 for c in top if pred(c))
        b = sum(1 for c in extra if pred(c))
        g = sum(c.genotypes_n or 0 for c in final if pred(c))
        n = sum(c.instances_n or 0 for c in final if pred(c))
        m = sum(c.measurements or 0 for c in final if pred(c))
        return f"{a} & {b} & {g:,} & {sci(n)} & {sci(m)} \\\\"

    def block(title: str, items: list[tuple[str, Any]]) -> list[str]:
        out = [r"\midrule", r"\multicolumn{6}{@{}l}{\emph{" + title + r"}}\\"]
        for lbl, pred in items:
            if sum(1 for c in final if pred(c)) == 0:
                continue
            out.append(tex_escape(lbl) + " & " + count(pred))
        return out

    hosts = sorted({c.organism for c in final})
    klasses = sorted({c.klass for c in final})
    bases = sorted({c.seq_basis for c in final})
    lines: list[str] = []
    lines += block("By band", [(b, (lambda c, b=b: c.band == b)) for b in BAND_ORDER])
    lines += block(
        "By tier", [(f"tier {t}", (lambda c, t=t: c.tier == t)) for t in (1, 2, 3, 4)]
    )
    lines += block("By host", [(o, (lambda c, o=o: c.organism == o)) for o in hosts])
    lines += block("By class", [(k, (lambda c, k=k: c.klass == k)) for k in klasses])
    lines += block(
        "By sequence basis", [(b, (lambda c, b=b: c.seq_basis == b)) for b in bases]
    )
    lines += block(
        "By retrieval class",
        [
            (
                "retrievable programmatically",
                lambda c: c.retrievability == "retrievable",
            ),
            (
                "needs a human with a browser",
                lambda c: c.retrievability == "manual_fetch",
            ),
            ("needs an ask to a named party", lambda c: c.retrievability == "ask"),
        ],
    )
    lines += block(
        "By verification label",
        [
            ("counts read on the source", lambda c: c.verified == "verified"),
            ("counts from a search snippet", lambda c: c.verified == "snippet"),
            (
                "counts inferred, including a product",
                lambda c: c.verified == "inferred",
            ),
        ],
    )
    lines += block(
        "Other attributes",
        [
            ("needs a schema class the substrate lacks", lambda c: bool(c.new_class)),
            ("carries a time axis", lambda c: bool(c.time_axis)),
            ("figures sourced this pass", lambda c: c.confidence == "sourced"),
            ("figures from recall, to confirm", lambda c: c.confidence == "recall"),
            ("per-record values not released", lambda c: c.status == "blocked"),
            ("a corpus, not net new until split", lambda c: c.status == "aggregation"),
            ("instances a reported count", lambda c: c.instances_basis == "reported"),
            (
                "instances a product of the axes",
                lambda c: c.instances_basis == "product",
            ),
            ("instances an estimate", lambda c: c.instances_basis == "estimate"),
        ],
    )
    lines.append(r"\midrule")
    lines.append("Total & " + count(lambda c: True))
    n_new = len({c.new_class for c in final if c.new_class})
    # A longtable rather than a table[H]: the eight blocks come to 66 rows, which is
    # taller than a page, and a non-floating table that tall pushes content past the
    # bottom margin instead of breaking. A longtable repeats its header and pages.
    hdr = r" & Top 50 & Extra & Genotypes & Instances & Meas. \\"
    head = (
        r"""\begingroup
\small
\begin{longtable}{@{}l r r r r r@{}}
\caption[]{Summary of the recommended set, split at the fifty line. \emph{Top 50} and
\emph{Extra} count rows; \emph{Genotypes}, \emph{Instances} and \emph{Meas.} sum the row
axes over all """
        + str(len(final))
        + r""" and are lower bounds, since a row with no count contributes
nothing. The set names """
        + str(n_new)
        + r""" distinct schema classes the substrate does not hold
(Table~\ref{tab:schema}).}
\label{tab:summary}\\
\toprule
"""
        + hdr
        + r"""
\endfirsthead
\multicolumn{6}{@{}l}{\footnotesize\emph{Table~\ref{tab:summary}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\endhead
\bottomrule
\endfoot
"""
    )
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_excluded() -> str:
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{58mm} L{26mm} L{88mm}@{}}
\caption[]{Considered and dropped. \emph{no-sequence} is the hard gate: without a route
to the strain's genomic content there is no genotype to map a phenotype from.
\emph{off-host} is a real dataset in an organism outside this wave's hosts.
\emph{not-a-dataset} is a model, a design program, a review or a prior over the genome.
\emph{no-per-record-data} is a real experiment whose released form is a figure or a
summary rather than per-record values. \emph{already-built} is served, or already in the
raw mirror, so it is an overlap rather than a candidate. \emph{derivative} re-serves or
re-processes another dataset's values, so its records are not net new.}
\label{tab:excluded}\\
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endfirsthead
\toprule
Dataset or group & Rule & Reason \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for e in EXCLUDED:
        lines.append(
            " & ".join([tex_escape(e.name), tex_escape(e.rule), tex_escape(e.reason)])
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_asks() -> str:
    """The rows gated on a request, with the owner of each request.

    Printed in full rather than counted, because an ask is an action with a named owner
    and an unlocated deposit is what a backlog loses silently.
    """
    hdr = (
        r"\textbf{Dataset} & \textbf{Host} & \textbf{Who to ask} & "
        r"\textbf{What exactly to ask for} & \textbf{What it unlocks} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{@{}L{38mm} L{16mm} L{40mm} L{78mm} L{70mm}@{}}
\caption[]{The rows gated on a request to a named person or group, which is the shortest
list in this document and the highest leverage: each of these is one message, and three
of them are the only route to the dataset at all. A dash in \emph{Host} marks an ask
that spans hosts.}
\label{tab:asks}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{5}{@{}l}{\footnotesize\emph{Table~\ref{tab:asks}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for a in ASKS:
        lines.append(
            " & ".join(
                [
                    r"\textbf{" + tex_escape(a.dataset) + r"}",
                    org_tex(a.organism) if a.organism else "--",
                    tex_escape(a.who),
                    tex_escape(a.request),
                    tex_escape(a.unlocks),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_namespaces() -> str:
    """The gene-identifier namespaces per host, and what joins them."""
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{22mm} L{34mm} L{20mm} L{30mm} L{66mm}@{}}
\caption[]{The gene-identifier namespaces this wave's released data speaks, per host.
\emph{Carries} is the model or gene count the namespace itself holds and \emph{Joins} is
what maps it to another namespace. One species in this wave has four live namespaces
whose model counts differ by more than two thousand genes, so a released gene id is not
a well-formed identifier until the namespace is named. A \emph{Joins} entry that says a
crosswalk has to be built is saying no join exists today: it is not a pointer to one.}
\label{tab:namespaces}\\
\toprule
Host & Namespace & Spelling & Carries & Joins \\
\midrule
\endfirsthead
\toprule
Host & Namespace & Spelling & Carries & Joins \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for ns in NAMESPACES:
        lines.append(
            " & ".join(
                [
                    org_tex(ns.organism),
                    tex_escape(ns.namespace),
                    r"\texttt{" + tex_escape(ns.spelling) + r"}",
                    tex_escape(ns.carries),
                    tex_escape(ns.joins),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_schema() -> str:
    """Every schema or candidate-machinery change the wave requires, and its policy."""
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}L{50mm} L{48mm} L{74mm}@{}}
\caption[]{Every schema or candidate-machinery change this wave requires, with the
symbol it lands on and the policy it triggers. \emph{Symbol} is the exact module path
and name, so the change is addressable rather than described. A change marked additive
adds a class or a vocabulary member that only its own closure holds, so it enters the
served graph by incremental admission; a change to a class other closures hold is a full
rebuild of every dataset whose closure holds it. The first two are the critical path:
without them no loader can declare a host and the candidate gate cannot even be
measured.}
\label{tab:schema}\\
\toprule
Symbol & What changes & Policy it triggers \\
\midrule
\endfirsthead
\toprule
Symbol & What changes & Policy it triggers \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for s in SCHEMA_NEEDS:
        # \file is a \DeclareUrlCommand, so its argument is read with URL catcodes and
        # must NOT be tex_escape'd: an escaped underscore renders as a literal
        # backslash plus underscore.
        lines.append(
            " & ".join(
                [
                    r"\file{" + s.symbol + r"}",
                    tex_escape(s.what),
                    tex_escape(s.policy)
                    + (" Additive." if s.additive else " Not additive."),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_verdicts(rows: list[Candidate]) -> str:
    """The candidate-gate verdict of every ranked row, read from ``database/candidates/``.

    The verdict store is authoritative (plan decision 7 of
    [[plan.dataset-admission-pipeline.2026.10.10]]): this column renders it and no row
    literal carries a verdict, so ``sort_key`` never sees one and a verdict can never
    re-rank a row another branch owns.

    Every row here prints ``no verdict`` today, and that is the honest state rather than
    a failure: ``verdict.CandidateTable`` is ``Literal["bacteria", "yeast"]``, so no
    stored verdict can carry this table's name until that literal is widened, which is a
    policy decision because the field is the ``table`` of all 52 stored verdicts.
    ``verdicts_by_row`` filters on an equality rather than validating its argument, so
    it accepts the name and returns an empty mapping.
    """
    verdicts = verdicts_by_row(TABLE)
    head = r"""\begingroup
\footnotesize
\begin{longtable}{@{}r L{76mm} L{20mm} L{36mm} L{24mm}@{}}
\caption[]{Candidate-gate verdict per ranked row, rendered from the verdict store
(\texttt{database/candidates/}). \emph{no verdict} means the row has not been gated, and
every row reads that way here: the stored verdict's \emph{table} field admits only the
two existing pools, so a row of this table cannot yet carry one
(Table~\ref{tab:schema}).}
\label{tab:verdicts}\\
\toprule
Rank & Dataset & Host & Verdict & Decided \\
\midrule
\endfirsthead
\toprule
Rank & Dataset & Host & Verdict & Decided \\
\midrule
\endhead
\bottomrule
\endfoot
"""
    lines = []
    for rank, c in enumerate(rows, start=1):
        v = verdicts.get(c.name)
        outcome = v.outcome if v is not None else "no verdict"
        decided = v.decided_at[:10] if v is not None else ""
        lines.append(
            " & ".join(
                [
                    str(rank),
                    tex_escape(c.name),
                    org_tex(c.organism),
                    tex_escape(outcome),
                    decided,
                ]
            )
            + r" \\"
        )
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def verdict_column(rows: list[Candidate]) -> dict[str, str]:
    """Row name -> verdict outcome, for the JSON dump (``no verdict`` when ungated)."""
    verdicts = verdicts_by_row(TABLE)
    return {
        c.name: verdicts[c.name].outcome if c.name in verdicts else "no verdict"
        for c in rows
    }


def main() -> None:
    _check_rows()
    rows, quota_moves = ranked()
    shortfall = max(0, CUT - len(rows))
    # The yeast pool's ceiling is 250 because it holds one host. This pool spans eight,
    # and the Fitness Browser alone contributes one row per organism, so a cross-host
    # pool is legitimately larger. 400 is a guard against a duplicated paste or a
    # runaway generated block, not a target: the pool is well under it and the shortfall
    # against ADDITIONS is reported rather than padded.
    if len(rows) > 400:
        raise SystemExit(f"{len(rows)} candidates exceeds the 400-row ceiling")

    write(TEX_DIR / "final.tex", render_final(rows))
    write(TEX_DIR / "sources.tex", render_sources(rows[:FINAL]))
    write(TEX_DIR / "candidates.tex", render_candidates(rows))
    write(
        TEX_DIR / "sources_all.tex",
        render_sources(
            rows, label="tab:sources-all", of="tab:candidates", scope="the whole pool"
        ),
    )
    write(TEX_DIR / "counts.tex", render_counts(rows))
    write(TEX_DIR / "summary.tex", render_summary(rows))
    write(TEX_DIR / "excluded.tex", render_excluded())
    write(TEX_DIR / "asks.tex", render_asks())
    write(TEX_DIR / "namespaces.tex", render_namespaces())
    write(TEX_DIR / "schema.tex", render_schema())
    write(TEX_DIR / "verdicts.tex", render_verdicts(rows))

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(
            {
                "table": TABLE,
                "additions": ADDITIONS,
                "target_count": TARGET_COUNT,
                "cut": CUT,
                "shortfall": shortfall,
                "wave_1": WAVE_1,
                "wave_2": WAVE_2,
                "final": FINAL,
                "quota_1": QUOTA_1,
                "n_candidates": len(rows),
                "n_built": sum(1 for c in CANDIDATES if c.status == "built"),
                "quota_moves": quota_moves,
                "candidates": [c.model_dump() for c in rows],
                "built": [c.model_dump() for c in CANDIDATES if c.status == "built"],
                "excluded": [e.model_dump() for e in EXCLUDED],
                "asks": [a.model_dump() for a in ASKS],
                "namespaces": [n.model_dump() for n in NAMESPACES],
                "schema_needs": [s.model_dump() for s in SCHEMA_NEEDS],
                "verdicts": verdict_column(rows),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {JSON_OUT.relative_to(REPO)}")

    for name, why, old, new in quota_moves:
        print(f"{why} lifted {name!r} from rank {old} to {new}")
    print(
        f"{len(rows)} candidates ranked, "
        f"{sum(1 for c in CANDIDATES if c.status == 'built')} dropped as already built; "
        f"wave 1 = {WAVE_1}, wave 2 ends at {WAVE_2}, the cut is {CUT}; "
        f"shortfall against {ADDITIONS} additions is {shortfall}"
    )
    for organism in sorted({c.organism for c in rows}):
        n_all = sum(1 for c in rows if c.organism == organism)
        n_1 = sum(1 for c in rows[:WAVE_1] if c.organism == organism)
        print(f"  {organism}: {n_all} in the pool, {n_1} in the recommended fifty")
    n_band = sum(c.band != "scale" for c in rows)
    n_ask = sum(c.retrievability == "ask" for c in rows)
    n_manual = sum(c.retrievability == "manual_fetch" for c in rows)
    n_new = len({c.new_class for c in rows if c.new_class})
    print(
        f"{n_band} rows banded out of scale; {n_manual} need a browser, {n_ask} need an "
        f"ask; {n_new} distinct new schema classes; {len(EXCLUDED)} excluded"
    )
    for rank, c in enumerate(rows[:15], start=1):
        print(
            f"  {rank:2d}. {c.name} [{c.organism}] "
            f"meas {c.measurements:,} ({verified_tex(c.verified)}, {c.band})"
        )


if __name__ == "__main__":
    main()
