# torchcell/metabolism/parameters
# [[torchcell.metabolism.parameters]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/parameters
# Test file: tests/torchcell/metabolism/test_parameters.py

r"""Kinetic and physical parameters for a GEM, database-first and provenance-tagged.

WHY THIS IS THE PORTABILITY LAYER
---------------------------------
An enzyme-constrained model needs three numbers per catalytic edge -- :math:`k_{cat}`,
:math:`K_M`, and the gene product's molecular weight -- and only the third is reliably
tabulated. For *S. cerevisiae*, the best-curated organism there is, the Open Enzyme
Database slice holds **1,126 rows** against **1,161 metabolic genes** and **4,131
reactions**. For a non-model yeast or a new bacterial chassis it holds essentially
nothing.

That is the actual barrier to running this model on another organism, and it is why the
resolution order below is a **policy** rather than a lookup:

1. **experimental** -- BRENDA / Open Enzyme Database, matched on UniProt accession and
   substrate, selected at the assay temperature closest to the phenotype's;
2. **predicted** -- a sequence-based model fills the gap;
3. **default** -- a single organism-level constant, used only where both fail, and
   reported as coverage rather than presented as a measurement.

**Database-first is not a preference, it is a correctness requirement.** A predictor
trained on BRENDA will happily reproduce a value it memorized, so using a prediction where
a measurement exists both discards information and inflates any apparent agreement between
the two.

THE PREDICTORS
--------------
Three sequence-based models, registered here by capability rather than hardcoded, because
which one is appropriate depends on what the organism has:

======================  =================================  ==========================
predictor               inputs                             emits
======================  =================================  ==========================
**KcatNet**             protein sequence + substrate SMILES :math:`k_{cat}`
**RealKcat**            protein sequence + substrate SMILES :math:`k_{cat}`, :math:`K_M`
**DEKP**                sequence + optional structure file  :math:`k_{cat}`
======================  =================================  ==========================

RealKcat is the one that closes the :math:`K_M` gap, which matters because promiscuity is
a :math:`K_M` effect: Wu et al. (2026) measured underground reactions at ~2x higher
:math:`K_M` with **indistinguishable** :math:`k_{cat}`, so a :math:`k_{cat}`-only model
routes promiscuous flux at full native capacity for free. DEKP is structure-aware, which is
the right choice when a predicted structure exists and the sequence is far from anything in
the training set -- the usual situation for a novel chassis.

None of the three is invoked from this module. :class:`KcatPredictor` is a protocol with a
registry, so a run records **which predictor produced which value** and a
"published-only vs published-plus-predicted" ablation is a filter on the provenance tag
rather than a re-run of retrieval. Wiring an actual checkpoint in is deliberately a
separate, GPU-bearing step.

WHAT IS REAL TODAY, MEASURED
----------------------------
* **Molecular weight: complete.** ``data/databases/swissprot.tsv`` ships MW for all 6,721
  proteins, joined on the systematic ORF name. Nothing is defaulted.
* **Metabolite concentrations: partial and real.** ``YMDBconcentrations.csv`` holds 867
  measured intracellular concentrations in uM with a mean/max/min, which anchor the learned
  :math:`\ln c` of the thermodynamic term instead of letting it float in its box.
* **k_cat: thin.** The OED slice is the only source, and its coverage against the GEM's
  catalytic units is computed by :func:`resolve_kcat_table` and must be reported.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os.path as osp
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import CatalyticUnits, GemTensors, TableCoverage

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra
from torchcell.metabolism.enzyme_kinetics import (
    KineticKind,
    OedKineticRecord,
    index_by_uniprot,
    load_mirrored_records,
    resolve_parameter,
)

#: Canonical on-disk home of the Open Enzyme Database organism slice.
DEFAULT_OED_MIRROR = "data/enzyme_kinetics/open_enzyme_database"

#: Median k_cat of the OED S. cerevisiae slice is computed at load time; this is only the
#: floor used when the slice itself is empty, i.e. for an organism with no measurements.
FALLBACK_KCAT_PER_S = 13.7


class ParameterProvenance(StrEnum):
    """Where one parameter value came from. Every value carries exactly one of these."""

    BRENDA = "brenda"
    OPEN_ENZYME_DATABASE = "open_enzyme_database"
    KCATNET = "kcatnet"
    REALKCAT = "realkcat"
    DEKP = "dekp"
    #: The median over the sequence-based predictors the within-class study admits for this
    #: organism. Which predictors those are is not baked in here: it is read from the study's
    #: own result file and recorded per run in :class:`PredictorSelection`.
    PREDICTED_MEDIAN = "predicted_median"
    #: A measurement of an enzyme in ANOTHER organism carrying the same EC number and
    #: assayed on a substrate this reaction consumes. It is evidence about the chemistry,
    #: not about this protein, so it sits BETWEEN the measured and the predicted tier and
    #: is never counted as a measurement of the organism being modeled. For a chassis with
    #: no assays of its own this is the only measured-adjacent tier available at all.
    OPEN_ENZYME_DATABASE_EC_HOMOLOG = "open_enzyme_database_ec_homolog"
    ORGANISM_DEFAULT = "organism_default"
    SWISSPROT = "swissprot"
    YMDB = "ymdb"


#: Which provenance tags are measurements rather than model output. The
#: published-only ablation is exactly a filter on this set.
#:
#: **The homolog tag is deliberately absent.** A cross-organism EC match is a real
#: measurement of a real enzyme, but not of the enzyme in this model, and
#: :func:`~torchcell.metabolism.flux_layer._tier_counts` reads this set to decide what
#: counts as measured. Including it would inflate the experimental fraction that 027's
#: preflight asserts on, which exists to catch a missing mirror; the honest coarse
#: placement of a homolog value is with the predictions, and the fine tier stays readable
#: in the provenance list and in :meth:`ParameterTable.tier_counts`.
EXPERIMENTAL_SOURCES = frozenset(
    {
        ParameterProvenance.BRENDA,
        ParameterProvenance.OPEN_ENZYME_DATABASE,
        ParameterProvenance.SWISSPROT,
        ParameterProvenance.YMDB,
    }
)


@runtime_checkable
class KcatPredictor(Protocol):
    """A sequence-based turnover predictor.

    Deliberately minimal. Any of KcatNet, RealKcat or DEKP satisfies it, and a predictor
    that also emits :math:`K_M` advertises that through :attr:`emits_km` so the caller can
    close the affinity gap in the same pass rather than running two models.
    """

    name: str
    emits_km: bool

    def predict(
        self, sequence: str, substrate_smiles: str | None = None
    ) -> tuple[float, float | None]:
        """Return ``(kcat_per_s, km_mM_or_None)`` for one (protein, substrate) pair."""
        ...


class PredictorRegistry(BaseModel):
    """The predictors a run is allowed to use, in priority order.

    Empty by default: a run that fills gaps with predictions has to say so, and the
    resulting parameter table records which predictor supplied each value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    predictors: list[object] = Field(default_factory=list)

    def first_emitting_km(self) -> object | None:
        """The highest-priority registered predictor that also produces ``K_M``."""
        for p in self.predictors:
            if getattr(p, "emits_km", False):
                return p
        return None


class ParameterTable(BaseModel):
    """One resolved parameter per entity, with a per-entity provenance tag.

    The provenance array is the point of this class. A capacity constraint where 95 % of
    turnover numbers are an organism default is a uniform rescaling of the flux box, not an
    enzyme constraint, and the two are indistinguishable from a loss curve.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: torch.Tensor = Field(description="[n] resolved values.")
    provenance: list[ParameterProvenance]
    known_mask: torch.Tensor = Field(
        description="[n] bool, True where the value is NOT an organism default."
    )
    experimental_mask: torch.Tensor = Field(
        description="[n] bool, True where the value is a measurement, not a prediction."
    )
    unit: str
    coverage: TableCoverage
    experimental_coverage: TableCoverage
    notes: str = ""

    def tier_counts(self) -> dict[str, int]:
        """The four-way split the coverage report has to print.

        ``measured`` is a measurement of THIS organism's protein on a substrate the
        reaction consumes, ``homolog`` a measurement of the same chemistry in another
        organism, ``predicted`` a sequence-model value, ``default`` the organism constant.
        Reporting only ``known`` merges the first three, and a "published-only" ablation
        has to separate all four.

        ``homolog`` is reported separately here and folded in with the predictions by
        :func:`~torchcell.metabolism.flux_layer._tier_counts`, which reads
        :data:`EXPERIMENTAL_SOURCES` and does not know about the finer split. The two
        reports therefore agree on ``measured`` and on ``default``, and the flux layer's
        ``predicted`` is this table's ``predicted`` plus ``homolog``.
        """
        counts = {"measured": 0, "homolog": 0, "predicted": 0, "default": 0}
        for p in self.provenance:
            if p in EXPERIMENTAL_SOURCES:
                counts["measured"] += 1
            elif p is ParameterProvenance.OPEN_ENZYME_DATABASE_EC_HOMOLOG:
                counts["homolog"] += 1
            elif p is ParameterProvenance.ORGANISM_DEFAULT:
                counts["default"] += 1
            else:
                counts["predicted"] += 1
        return counts

    @classmethod
    def build(
        cls,
        values: torch.Tensor,
        provenance: list[ParameterProvenance],
        unit: str,
        notes: str = "",
    ) -> ParameterTable:
        """Derive both masks and both coverages from the provenance list."""
        known = torch.tensor(
            [p is not ParameterProvenance.ORGANISM_DEFAULT for p in provenance]
        )
        experimental = torch.tensor([p in EXPERIMENTAL_SOURCES for p in provenance])
        return cls(
            values=values,
            provenance=provenance,
            known_mask=known,
            experimental_mask=experimental,
            unit=unit,
            coverage=TableCoverage.of(known.numpy()),
            experimental_coverage=TableCoverage.of(experimental.numpy()),
            notes=notes,
        )


#: A *S. cerevisiae* systematic ORF name: nuclear ``YKL019W`` / ``YDR034C-A``, or a
#: mitochondrial ``Q0045``. Matching this pattern is what distinguishes the systematic name
#: from the standard name and the ordered-locus aliases in the same column.
SYSTEMATIC_NAME_RE = re.compile(r"^(Y[A-P][LR]\d{3}[WC](?:-[A-Z])?|Q\d{4})$")


def load_swissprot(model_dir: str) -> dict[str, dict[str, str]]:
    """Read ``data/databases/swissprot.tsv``, keyed by systematic ORF name.

    **The ``gene_id`` column is an alias LIST, not a two-field pair**, and reading it as a
    pair is a quiet 63 % data loss. It holds every name UniProt knows for the locus, space
    separated and in no fixed order or length -- ``"RAM2 YKL019W"`` is two tokens, but
    ``"ERG20 BOT3 FDS1 FPP1 YJL167W J0525"`` is six, with the systematic name fifth and an
    ordered-locus alias last. Token counts across the file run from 1 to 11.

    Taking the last token therefore matches only **430 of the GEM's 1,161 genes (37 %)**,
    and taking the first matches 38. Selecting the token that *is* a systematic ORF name
    matches **1,161 of 1,161**. The lesson is the general one: a column whose separator is
    a space is a list until proven otherwise, and the check that catches it is joining
    against a known key set and looking at the hit rate rather than at the first few rows.
    """
    path = osp.join(model_dir, "data", "databases", "swissprot.tsv")
    out: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            for token in (row.get("gene_id") or "").split():
                if SYSTEMATIC_NAME_RE.match(token):
                    out.setdefault(token, row)
    return out


def molecular_weight_table(
    model_dir: str, gene_ids: list[str], default_da: float = 40000.0
) -> ParameterTable:
    """Per-gene molecular weight in kDa, from SwissProt.

    Args:
        model_dir: GEM checkout root.
        gene_ids: GEM gene ids in tensor-index order.
        default_da: Fallback in daltons for a gene SwissProt does not carry. ~40 kDa is
            the median yeast protein; it is a default and is tagged as one.
    """
    swissprot = load_swissprot(model_dir)
    values: list[float] = []
    provenance: list[ParameterProvenance] = []
    for gid in gene_ids:
        row = swissprot.get(gid)
        raw = (row or {}).get("MW")
        if raw:
            values.append(float(raw) / 1000.0)
            provenance.append(ParameterProvenance.SWISSPROT)
        else:
            values.append(default_da / 1000.0)
            provenance.append(ParameterProvenance.ORGANISM_DEFAULT)
    return ParameterTable.build(
        torch.tensor(values, dtype=torch.float32),
        provenance,
        unit="kDa",
        notes="yeast-GEM data/databases/swissprot.tsv, joined on systematic ORF name.",
    )


def uniprot_for_genes(model_dir: str, gene_ids: list[str]) -> dict[str, str]:
    """Systematic ORF name -> UniProt accession, the join key into any kinetics database."""
    swissprot = load_swissprot(model_dir)
    return {
        gid: swissprot[gid]["uniprot"]
        for gid in gene_ids
        if gid in swissprot and swissprot[gid].get("uniprot")
    }


#: Where the per-(gene, reaction) predictor tables live, relative to ``DATA_ROOT``. One
#: parquet per predictor per parameter, written by
#: ``experiments/026-metabolism-flux/scripts/run_<predictor>_pinned.py``.
KINETICS_MIRROR = "data/torchcell/kinetics"

#: The within-class plausibility study's own result file, relative to ``EXPERIMENT_ROOT``.
#: It is the only place the predictor ranking comes from; nothing here hardcodes a winner.
PLAUSIBILITY_RESULT = "026-metabolism-flux/results/within_class_plausibility.json"

#: Version subdirectory of each predictor's mirror that the resolver reads. ``verified``
#: holds the tables built on substrate structures whose composition was checked against the
#: GEM's formula; ``processed`` holds the unchecked tables it supersedes, kept on disk so a
#: before-and-after is reproducible. Written by
#: ``experiments/026-metabolism-flux/scripts/improve_kinetic_coverage.py --stage verify``.
KINETICS_VERSION = "verified"

#: The verified ``(unit, gene, substrate, SMILES)`` table, relative to the kinetics mirror.
#: It is what makes a substrate-matched measured tier possible without a cobra model in
#: hand: it names, per catalytic unit, the substrates whose structure survived the check.
VERIFIED_PAIRS_FEATURE = "_features/verified_pairs.parquet"


def _sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class PredictorSelection(BaseModel):
    r"""Which sequence predictors this run is allowed to fill gaps with, and on what evidence.

    **The rule, stated once and applied to both parameters.** Rank the predictors by the
    ``own_class_band`` reading of
    ``experiments/026-metabolism-flux/results/within_class_plausibility.json``: each measured
    yeast pair is scored against the standard deviation of **its own** EC4 class, which is
    the strictest of the four readings that study computes (a shared median band pools unlike
    chemistry and is wider, so it flatters every predictor). Keep a predictor when the
    **Wilson 95 percent lower bound** of its fraction exceeds the point fraction of the
    **EC-median lookup**, the baseline that knows the enzyme's function and nothing else.

    Two consequences of that rule, both measured rather than chosen:

    * for :math:`k_{cat}` it admits **EZKinetics, EITLEM and UniKP** and rejects DLKcat,
      TurNuP and DeepEnzyme, whose lower bounds fall below the lookup;
    * for :math:`K_M` it admits **EZKinetics alone**, so the median over the admitted set
      degenerates to that one predictor's value. That is the honest outcome of a ranking on
      47 scorable pairs, and it is recorded here rather than papered over by admitting a
      second predictor the evidence does not support.

    A lower bound is used rather than the point estimate because the denominators are 44 to
    47: three of the six :math:`k_{cat}` predictors beat the lookup on the point estimate and
    none of those three beats it once the interval is taken.
    """

    model_config = ConfigDict(extra="forbid")

    parameter: str
    statistic: str = "own_class_band"
    baseline_fraction: float = Field(
        description="The EC-median lookup's fraction on the same reading."
    )
    ranked: list[tuple[str, float, float]] = Field(
        description="(predictor, fraction, Wilson 95 percent lower bound), best first."
    )
    selected: list[str]
    source_path: str
    sha256: str

    @classmethod
    def from_study(cls, parameter: str, plausibility_json: str) -> PredictorSelection:
        """Read the ranking out of the study's result file. No predictor name is hardcoded."""
        with open(plausibility_json) as handle:
            report = json.load(handle)
        block = report["parameters"][parameter]
        baseline = float(block["ec_median_lookup"]["own_class"]["frac"])
        rows: list[tuple[str, float, float]] = []
        for name, entry in block["own_class_band"]["predictors"].items():
            rows.append((name, float(entry["frac"]), float(entry["ci95"][0])))
        rows.sort(key=lambda r: -r[1])
        return cls(
            parameter=parameter,
            baseline_fraction=baseline,
            ranked=rows,
            selected=[name for name, _, lower in rows if lower > baseline],
            source_path=plausibility_json,
            sha256=_sha256_of(plausibility_json),
        )


class SubstrateMatch(BaseModel):
    """Per catalytic unit, the identity of the substrates its reaction consumes.

    **An empty instance means no substrate is known, and therefore no measurement can be
    accepted.** That is the correct semantics rather than a missing feature: a database row
    is a measurement of one enzyme acting on one molecule, and without knowing which
    molecules the reaction consumes there is no way to tell whether the row describes this
    reaction or a different activity of the same protein. Half the measured tier turned out
    to be the second case.

    Two keys, because neither alone is enough. The InChIKey connectivity block is
    protonation- and stereo-tolerant and is what matches a database's neutral acid to a
    GEM's anion; the lowercased name catches the rows whose SMILES does not parse.
    """

    model_config = ConfigDict(extra="forbid")

    keys: dict[int, set[str]] = Field(default_factory=dict)
    names: dict[int, set[str]] = Field(default_factory=dict)
    #: EC numbers annotated on each unit's reaction, the key the homolog tier matches on.
    ec: dict[int, set[str]] = Field(default_factory=dict)
    source_path: str = ""
    sha256: str = ""
    #: Accept every row without checking its substrate. This reproduces the accession-only
    #: key the resolver used before 2026-09-11 and exists ONLY so a before-and-after can be
    #: computed; the name is deliberately ugly, because a table built with it contains
    #: turnover numbers assayed on molecules the reaction does not consume.
    accept_all_unchecked: bool = False

    def matches(self, unit_id: int, key: str | None, name: str | None) -> str | None:
        """How this row's substrate matches the unit's, or None when it does not."""
        if self.accept_all_unchecked:
            return "unchecked"
        if key and key in self.keys.get(unit_id, set()):
            return "inchikey_block"
        if name and name.lower() in self.names.get(unit_id, set()):
            return "substrate_name"
        return None

    @classmethod
    def from_verified_pairs(cls, path: str) -> SubstrateMatch:
        """Build from the verified pair table, keyed on the structures it carries."""
        from torchcell.metabolism.kinetics_pipeline import inchikey_block

        frame = pd.read_parquet(
            path, columns=["unit_id", "substrate_name", "smiles", "reaction_ec"]
        ).drop_duplicates()
        # An empty SMILES is a substrate with no verified structure. It contributes its
        # NAME to the match and no structural key, which is the honest reading: the GEM
        # says what the reaction consumes, and we could not find its structure.
        blocks = {s: inchikey_block(s) if s else None for s in frame["smiles"].unique()}
        keys: dict[int, set[str]] = {}
        names: dict[int, set[str]] = {}
        ec: dict[int, set[str]] = {}
        for unit_id, name, smiles, codes in frame.itertuples(index=False):
            block = blocks[smiles]
            if block:
                keys.setdefault(int(unit_id), set()).add(block)
            names.setdefault(int(unit_id), set()).add(str(name).lower())
            ec.setdefault(int(unit_id), set()).update(
                c for c in str(codes or "").split(";") if c
            )
        return cls(
            keys=keys, names=names, ec=ec, source_path=path, sha256=_sha256_of(path)
        )


def _substrate_key(
    record: OedKineticRecord, cache: dict[str, str | None]
) -> str | None:
    """The InChIKey connectivity block of a database row's assayed substrate."""
    from torchcell.metabolism.kinetics_pipeline import inchikey_block

    if not record.smiles:
        return None
    if record.smiles not in cache:
        cache[record.smiles] = inchikey_block(record.smiles)
    return cache[record.smiles]


def _database_tiers(
    units: CatalyticUnits,
    gene_to_uniprot: dict[str, str],
    organism_records: list[OedKineticRecord],
    homolog_records: list[OedKineticRecord],
    substrates: SubstrateMatch,
    kind: KineticKind,
    target_temperature_c: float,
) -> dict[int, tuple[float, ParameterProvenance, str]]:
    r"""Per catalytic unit, the best database value and which tier it came from.

    Two tiers, tried in order and never merged.

    **Measured**, the accession key: rows for a member gene's own accession, filtered to
    those assayed on a substrate the unit's reaction consumes. The substrate filter is
    applied BEFORE the temperature cascade, so a row assayed on an unrelated substrate can
    no longer win the tie-break and then be discarded, which would leave the unit unmatched
    while a valid row existed.

    **Homolog**, the EC key: rows in ANY organism whose EC number is annotated on the
    unit's reaction and whose substrate matches. This is evidence about the chemistry
    rather than about this protein, and it is tagged
    :attr:`ParameterProvenance.OPEN_ENZYME_DATABASE_EC_HOMOLOG` so the two never merge.

    A unit's value is the minimum over member genes for turnover and the maximum for
    affinity: a complex cannot turn over faster than its slowest subunit, and its weakest
    binding step sets where it saturates.
    """
    by_accession = index_by_uniprot(organism_records)
    by_ec: dict[str, list[OedKineticRecord]] = {}
    for record in homolog_records:
        if record.ec:
            by_ec.setdefault(record.ec, []).append(record)

    unit_genes: dict[int, list[str]] = {}
    for u, g in zip(
        units.unit_gene_index[0].tolist(), units.unit_gene_index[1].tolist()
    ):
        unit_genes.setdefault(u, []).append(units.gene_ids[g])

    cache: dict[str, str | None] = {}
    take = min if kind is KineticKind.KCAT else max
    out: dict[int, tuple[float, ParameterProvenance, str]] = {}
    for unit_id in range(units.n_units):
        hits: list[tuple[float, str]] = []
        for gene in unit_genes.get(unit_id, []):
            accession = gene_to_uniprot.get(gene)
            if not accession:
                continue
            candidates = [
                r
                for r in by_accession.get(accession, [])
                if substrates.matches(unit_id, _substrate_key(r, cache), r.substrate)
            ]
            if not candidates:
                continue
            resolved = resolve_parameter(
                candidates,
                kind,
                uniprot=accession,
                target_temperature_c=target_temperature_c,
            )
            if resolved is not None:
                hits.append((float(resolved.value), resolved.substrate or ""))
        if hits:
            value, substrate = take(hits, key=lambda h: h[0])
            out[unit_id] = (value, ParameterProvenance.OPEN_ENZYME_DATABASE, substrate)
            continue
        candidates = [
            r
            for code in substrates.ec.get(unit_id, set())
            for r in by_ec.get(code, [])
            if substrates.matches(unit_id, _substrate_key(r, cache), r.substrate)
        ]
        if not candidates:
            continue
        resolved = resolve_parameter(
            candidates, kind, uniprot="", target_temperature_c=target_temperature_c
        )
        if resolved is not None:
            out[unit_id] = (
                float(resolved.value),
                ParameterProvenance.OPEN_ENZYME_DATABASE_EC_HOMOLOG,
                resolved.substrate or "",
            )
    return out


class PredictedUnitTable(BaseModel):
    r"""A predicted value per catalytic unit, with the predictors behind each one.

    The values are a median over the admitted predictors in :math:`\log_{10}` space and
    back-transformed. Log space is not cosmetic: a turnover distribution spans six decades,
    and an arithmetic mean over predictors that disagree by two decades is dominated by the
    largest of them.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: torch.Tensor = Field(
        description="[n_units] predicted value, 0 where unknown."
    )
    mask: torch.Tensor = Field(
        description="[n_units] bool, True where a prediction exists."
    )
    n_predictors: torch.Tensor = Field(
        description="[n_units] how many admitted predictors contributed."
    )
    selection: PredictorSelection
    unit: str
    source_paths: dict[str, str]
    sha256: dict[str, str]
    #: Which version subdirectory of the mirror the values came from.
    version: str = KINETICS_VERSION
    #: The substrates each unit's reaction consumes, as InChIKey connectivity blocks, from
    #: the same verified pair table the predictions were computed on. Carried here because
    #: it is the only object the resolver already receives that knows the structures, and a
    #: measured value cannot be accepted without checking what it was assayed on.
    substrates: SubstrateMatch = Field(default_factory=lambda: SubstrateMatch())


def _per_unit_from_predictor(
    parquet_path: str, column: str, n_units: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""One predictor's :math:`\log_{10}` value per catalytic unit.

    Two aggregations, in this order and for stated reasons. **Over substrates**, the median:
    a reaction with several substrates gets one prediction each and none of them is the
    reaction's value, so the central one is taken rather than the largest. **Over the unit's
    genes**, the minimum: a complex cannot turn over faster than its slowest subunit, which is
    the same rule :func:`resolve_kcat_table` applies to the measured tier and the same logic
    the availability softmin uses one level up.

    ``unit_id`` in these tables is the index
    :func:`~torchcell.metabolism.constraints.parse_catalytic_units` assigns, verified against
    the real GEM by ``test_predicted_unit_ids_align_with_the_gpr_parse``.
    """
    frame = pd.read_parquet(parquet_path, columns=["unit_id", "gene_id", column])
    frame = frame[np.isfinite(frame[column]) & (frame[column] > 0)]
    per_gene = frame.groupby(["unit_id", "gene_id"], as_index=False)[column].median()
    per_unit = per_gene.groupby("unit_id")[column].min()
    values = np.full(n_units, np.nan)
    index = per_unit.index.to_numpy().astype(int)
    if index.size and index.max() >= n_units:
        raise ValueError(
            f"{parquet_path} carries unit_id {int(index.max())} for a GEM with {n_units} "
            "catalytic units. The predictor table was built against a different model."
        )
    values[index] = np.log10(per_unit.to_numpy())
    return values, np.isfinite(values)


def load_predicted_unit_table(
    parameter: str,
    n_units: int,
    kinetics_dir: str,
    plausibility_json: str,
    version: str = KINETICS_VERSION,
) -> PredictedUnitTable:
    """Median prediction per catalytic unit over the predictors the study admits.

    Reads the **verified** version of each predictor's table by default: the one built on
    substrate structures whose composition was checked against the GEM's formula. The
    unverified tables remain on disk under ``processed/`` and are reachable by passing
    ``version="processed"``, which is what a before-and-after comparison does; nothing
    silently falls back to them, because a value computed for a molecule the GEM does not
    contain is not a value for that reaction.

    Args:
        parameter: ``"k_cat"`` or ``"K_M"``; the parquet basename and the study's key.
        n_units: Number of catalytic units in the GEM, from ``CatalyticUnits.n_units``.
        kinetics_dir: ``$DATA_ROOT/data/torchcell/kinetics``.
        plausibility_json: The within-class study's result file.
        version: Version subdirectory of each predictor's mirror.
    """
    selection = PredictorSelection.from_study(parameter, plausibility_json)
    if not selection.selected:
        raise ValueError(
            f"the within-class study admits no predictor for {parameter}: none of "
            f"{[r[0] for r in selection.ranked]} beats the EC-median lookup "
            f"({selection.baseline_fraction:.3f}) on its Wilson lower bound."
        )
    stacked: list[np.ndarray] = []
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for name in selection.selected:
        path = osp.join(kinetics_dir, name, version, f"{parameter}.parquet")
        values, _ = _per_unit_from_predictor(path, parameter, n_units)
        stacked.append(values)
        paths[name] = path
        hashes[name] = _sha256_of(path)
    matrix = np.vstack(stacked)
    n_predictors = np.isfinite(matrix).sum(axis=0)
    mask = n_predictors > 0
    log_median = np.full(n_units, np.nan)
    log_median[mask] = np.nanmedian(matrix[:, mask], axis=0)
    values = np.zeros(n_units)
    values[mask] = np.power(10.0, log_median[mask])
    return PredictedUnitTable(
        values=torch.from_numpy(values).float(),
        mask=torch.from_numpy(mask),
        n_predictors=torch.from_numpy(n_predictors.astype(np.int64)),
        selection=selection,
        unit="1/s" if parameter == "k_cat" else "mM",
        source_paths=paths,
        sha256=hashes,
        version=version,
        substrates=SubstrateMatch.from_verified_pairs(
            osp.join(kinetics_dir, VERIFIED_PAIRS_FEATURE)
        ),
    )


def _organism_default(
    values: list[float], fallback: float, name: str
) -> tuple[float, str]:
    """The organism constant, and the sentence that says what it is.

    ONE definition for both parameters and for both the in-place resolver and the portable
    pipeline: the median of the values the DATABASE TIERS actually placed in the table, per
    catalytic unit. Per-unit rather than per-gene, because the table's entries are per unit
    and a per-gene median silently reweights complexes; measured-and-homolog rather than
    measured alone, because the default's job is to be on the right scale for the reactions
    that have no number, and a 94-unit median is a thinner estimate than a 900-unit one.
    With no database rows at all it is the literature fallback, which is the only case where
    a constant enters that the organism's own data did not set.
    """
    if not values:
        return fallback, f"the literature fallback {fallback:.3g} {name} (empty slice)"
    median = float(np.median(np.asarray(values, dtype=float)))
    return median, (
        f"the median of the {len(values)} database-tier values, {median:.3g} {name}"
    )


def resolve_kcat_table(
    units: CatalyticUnits,
    model_dir: str,
    oed_mirror_dir: str,
    *,
    registry: PredictorRegistry | None = None,
    predicted: PredictedUnitTable | None = None,
    substrates: SubstrateMatch | None = None,
    homolog_mirror_dir: str | None = None,
    target_temperature_c: float = 30.0,
) -> ParameterTable:
    r"""One :math:`k_{cat}` per catalytic unit, database-first and substrate-matched.

    FOUR TIERS, IN ORDER
    --------------------
    1. **measured** -- a row for a member gene's own accession, in this organism, assayed on
       a substrate the unit's reaction consumes.
    2. **homolog** -- a row in any organism whose EC number is annotated on the reaction and
       whose substrate matches. Evidence about the chemistry, not about this protein, and
       tagged apart so it is never counted as a measurement of this organism.
    3. **predicted** -- the median over the admitted sequence predictors.
    4. **default** -- the organism constant, see :func:`_organism_default`.

    **The substrate match is what makes tier 1 mean something.** Keying on the accession
    alone let a multi-substrate enzyme contribute a turnover number measured on a molecule
    the reaction does not touch, which was the case for 74 of the 148 units the accession
    key admitted. Without ``substrates`` no row can be checked, so no row is accepted: an
    empty :class:`SubstrateMatch` yields an empty measured tier rather than an unchecked one.

    A unit's value is the **minimum** over its member genes, because a complex cannot turn
    over faster than its slowest constituent, and this is the same "scarcest subunit limits"
    logic the availability softmin uses one level up.

    Returns:
        A :class:`ParameterTable` of length ``units.n_units``, in units of 1/s.
    """
    records = load_mirrored_records(oed_mirror_dir)
    homolog_records = (
        load_mirrored_records(homolog_mirror_dir) if homolog_mirror_dir else []
    )
    # The substrate map comes from THIS argument and nowhere else. Taking it from
    # ``predicted`` when the caller omits it would make the measured tier depend on whether
    # a predicted tier was requested, which is a coupling nobody would expect; and an empty
    # map is the correct answer to "no substrates known", since an unchecked row is not a
    # measurement of this reaction.
    match = substrates if substrates is not None else SubstrateMatch()
    gene_to_uniprot = uniprot_for_genes(model_dir, units.gene_ids)
    database = _database_tiers(
        units,
        gene_to_uniprot,
        records,
        homolog_records,
        match,
        KineticKind.KCAT,
        target_temperature_c,
    )

    # A registered in-process predictor is a different mechanism from the parquet tier and
    # is kept for the protocol's sake; it fills only units the database leaves.
    sequences = {
        gid: row["sequence"]
        for gid, row in load_swissprot(model_dir).items()
        if row.get("sequence")
    }
    unit_genes: dict[int, list[str]] = {}
    for u, g in zip(
        units.unit_gene_index[0].tolist(), units.unit_gene_index[1].tolist()
    ):
        unit_genes.setdefault(u, []).append(units.gene_ids[g])

    default, default_note = _organism_default(
        [v for v, _, _ in database.values()], FALLBACK_KCAT_PER_S, "1/s"
    )

    values: list[float] = []
    provenance: list[ParameterProvenance] = []
    for u in range(units.n_units):
        if u in database:
            value, source, _ = database[u]
            values.append(value)
            provenance.append(source)
        elif predicted is not None and bool(predicted.mask[u]):
            values.append(float(predicted.values[u]))
            provenance.append(ParameterProvenance.PREDICTED_MEDIAN)
        elif registry is not None and registry.predictors:
            genes = [g for g in unit_genes.get(u, []) if g in sequences]
            # The registry holds `object` so any model class can be registered without
            # importing it here; the protocol is what the call site actually requires.
            predictor = cast(KcatPredictor, registry.predictors[0])
            scored = [float(predictor.predict(sequences[g])[0]) for g in genes]
            if scored:
                values.append(min(scored))
                provenance.append(
                    ParameterProvenance(getattr(predictor, "name", "kcatnet"))
                )
            else:
                values.append(default)
                provenance.append(ParameterProvenance.ORGANISM_DEFAULT)
        else:
            values.append(default)
            provenance.append(ParameterProvenance.ORGANISM_DEFAULT)

    predicted_note = (
        "no predicted tier"
        if predicted is None
        else (
            f"predicted tier = median over {', '.join(predicted.selection.selected)} "
            f"({predicted.version} tables, admitted by {predicted.selection.statistic} "
            f"against the EC-median lookup at {predicted.selection.baseline_fraction:.3f})"
        )
    )
    return ParameterTable.build(
        torch.tensor(values, dtype=torch.float32),
        provenance,
        unit="1/s",
        notes=(
            f"Open Enzyme Database first ({len(records)} organism rows, substrate-matched "
            f"against {len(match.keys)} units), then the EC homolog tier "
            f"({len(homolog_records)} rows, any organism), then {predicted_note}, then "
            f"{default_note}. A complex takes the min over its subunits."
        ),
    )


def resolve_km_table(
    units: CatalyticUnits,
    model_dir: str,
    oed_mirror_dir: str,
    *,
    predicted: PredictedUnitTable | None = None,
    substrates: SubstrateMatch | None = None,
    homolog_mirror_dir: str | None = None,
    target_temperature_c: float = 30.0,
) -> ParameterTable:
    r"""One :math:`K_M` per catalytic unit, on the same four tiers as :func:`resolve_kcat_table`.

    The layer stores this and does not yet consume it: the capacity bound in use is
    :math:`|v_j| \le k_{\mathrm{cat}} E`, saturation-free, so :math:`K_M` would enter only
    through a saturation term that has not been written. It is resolved and reported anyway
    because promiscuity is a :math:`K_M` effect, and because a table that arrives with the
    model is a table whose coverage can be argued about before it is used rather than after.

    The affinity of a complex is taken as the **maximum** over its subunits, the mirror of the
    turnover rule: a high :math:`K_M` is a weak binder, and the weakest binding step is what
    sets the concentration at which the complex saturates.
    """
    records = load_mirrored_records(oed_mirror_dir)
    homolog_records = (
        load_mirrored_records(homolog_mirror_dir) if homolog_mirror_dir else []
    )
    # The substrate map comes from THIS argument and nowhere else. Taking it from
    # ``predicted`` when the caller omits it would make the measured tier depend on whether
    # a predicted tier was requested, which is a coupling nobody would expect; and an empty
    # map is the correct answer to "no substrates known", since an unchecked row is not a
    # measurement of this reaction.
    match = substrates if substrates is not None else SubstrateMatch()
    gene_to_uniprot = uniprot_for_genes(model_dir, units.gene_ids)
    database = _database_tiers(
        units,
        gene_to_uniprot,
        records,
        homolog_records,
        match,
        KineticKind.KM,
        target_temperature_c,
    )
    default, default_note = _organism_default(
        [v for v, _, _ in database.values()], 1.0, "mM"
    )

    values: list[float] = []
    provenance: list[ParameterProvenance] = []
    for u in range(units.n_units):
        if u in database:
            value, source, _ = database[u]
            values.append(value)
            provenance.append(source)
        elif predicted is not None and bool(predicted.mask[u]):
            values.append(float(predicted.values[u]))
            provenance.append(ParameterProvenance.PREDICTED_MEDIAN)
        else:
            values.append(default)
            provenance.append(ParameterProvenance.ORGANISM_DEFAULT)

    return ParameterTable.build(
        torch.tensor(values, dtype=torch.float32),
        provenance,
        unit="mM",
        notes=(
            f"Open Enzyme Database first ({len(records)} organism rows, substrate-matched), "
            f"then the EC homolog tier ({len(homolog_records)} rows, any organism), then the "
            f"predicted tier, then {default_note}. A complex takes the max over its subunits. "
            "STORED, NOT USED: the capacity bound is saturation-free."
        ),
    )


_CONC_ID_RE = re.compile(r"CHEBI:(\d+)", re.IGNORECASE)


def load_measured_concentrations(model_dir: str) -> dict[str, float]:
    r"""ChEBI id -> measured intracellular concentration in molar, from YMDB.

    ``YMDBconcentrations.csv`` reports uM with a mean, max and min plus the verbatim
    per-study string. The mean is used; the spread is kept in the file and is the obvious
    place to draw an uncertainty from later. These anchor the learned :math:`\ln c` of the
    anchored thermodynamic mode, which otherwise floats anywhere in its six-order-of-
    magnitude box.
    """
    path = osp.join(model_dir, "data", "databases", "YMDBconcentrations.csv")
    out: dict[str, float] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            chebi = (row.get("chebi") or "").strip()
            mean = (row.get("mean") or "").strip()
            if not chebi or not mean:
                continue
            match = _CONC_ID_RE.search(chebi)
            if match:
                out[match.group(1)] = float(mean) * 1e-6
    return out


def concentration_prior(
    model: cobra.Model, met_ids: list[str], model_dir: str
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Per-metabolite measured concentration prior, in molar, plus its coverage mask.

    Matches on the ChEBI annotation the GEM already carries per metabolite, so no external
    identifier mapping is involved. Returns ``(concentration, mask)``; the concentration is
    meaningless where the mask is False and must not be read there.
    """
    measured = load_measured_concentrations(model_dir)
    values = torch.zeros(len(met_ids))
    mask = torch.zeros(len(met_ids), dtype=torch.bool)
    for i, mid in enumerate(met_ids):
        annotation = model.metabolites.get_by_id(mid).annotation or {}
        chebi = annotation.get("chebi")
        candidates = chebi if isinstance(chebi, list) else [chebi]
        for c in candidates:
            if not c:
                continue
            match = _CONC_ID_RE.search(str(c))
            if match and match.group(1) in measured:
                values[i] = measured[match.group(1)]
                mask[i] = True
                break
    return values, mask


# =====================================================================================
# REACTION THERMODYNAMICS: three routes to one number, and the union of them
# =====================================================================================
#
# yeast-GEM ships a metabolite formation table AND a reaction table, and they came from
# different sources under different match keys, so the reaction column is not the sum of
# the metabolite column (`audit_reaction_thermo_coverage.py` reads the provenance out of
# the GEM's own curation script and cross-tabulates the routes). Our eQuilibrator
# recomputation is a third route. The flux layer has been consuming route one only, the
# shipped metabolite sum, which prices 3,505 of 4,131 reactions; the merged metabolite
# table prices more, and the difference is not a rounding error.

#: Where the eQuilibrator recomputation lives, relative to ``DATA_ROOT``. Written by
#: ``experiments/026-metabolism-flux/scripts/build_equilibrator_thermo.py``.
THERMO_EQUILIBRATOR_MIRROR = "data/torchcell/thermo_equilibrator"

#: Which rows of the shipped REACTION column are already kJ/mol, written by
#: ``experiments/026-metabolism-flux/scripts/thermo_dgpredictor_rows.py`` into the
#: eQuilibrator thermo directory.
#:
#: yeast-GEM 9 copied yETFL's values without converting and yETFL declares kcal/mol, so
#: :func:`~torchcell.metabolism.constraints._read_delta_g_csv` multiplies the whole column
#: by 4.184 on read. That is right for the metabolite column and for 3,177 of the reaction
#: column's 3,210 rows; the other 33 come from dGPredictor, whose driver reports kJ/mol, so
#: those are left 4.184x too large. This file names them, and
#: :func:`resolve_reaction_delta_g` divides them back before the union's last-resort fill.
DGPREDICTOR_ROWS_FILENAME = "dgpredictor_reaction_rows.json"


class ThermoSource(StrEnum):
    """Which route to a reaction's standard energy a run is using.

    ``SHIPPED``       -- sum the GEM's own metabolite table over S. What every 026 and 027
                         run before 2026-09-10 used.
    ``EQUILIBRATOR``  -- sum our eQuilibrator recomputation over S. One self-consistent
                         method, compartment-specific pH, and the only route that carries a
                         formation covariance.
    ``UNION``         -- merge the two metabolite tables (eQuilibrator preferred where both
                         price a species), then fall back to the shipped REACTION column for
                         reactions the merge still cannot price.
    ``UNION_EXTERNAL``-- the same merge, but the eQuilibrator side is the table that also
                         carries the written-structure lipids and the external-pKa targets
                         (``metabolite_dgf_prime_with_external.parquet``, written by
                         ``experiments/026-metabolism-flux/scripts/thermo_union_reprice.py``
                         out of the staged pipeline). A SEPARATE member rather than a
                         redefinition of ``UNION``, so a run that predates the extra rows
                         still reproduces exactly, and so a config records which table it
                         consumed.
    """

    SHIPPED = "shipped"
    EQUILIBRATOR = "equilibrator"
    UNION = "union"
    UNION_EXTERNAL = "union_external"


class ReactionThermoTable(BaseModel):
    r""":math:`\Delta_r G'^\circ` per reaction with a per-reaction provenance flag.

    ``provenance[j]`` is one of ``equilibrator``, ``shipped_metabolite_sum``, ``mixed``,
    ``shipped_reaction_column`` or ``none``. ``mixed`` is the cell that only the union has:
    a reaction that draws one participant from each table and is priceable under neither
    route alone. It is flagged rather than hidden, because a mixed value is two methods
    added together and its uncertainty is not either method's.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    delta_r_g0: torch.Tensor = Field(description="[r] kJ/mol, 0 where masked.")
    mask: torch.Tensor = Field(description="[r] bool, True where a real value exists.")
    provenance: list[str]
    sigma: torch.Tensor = Field(
        description="[r] kJ/mol propagated eQuilibrator uncertainty, 0 where unavailable."
    )
    sigma_mask: torch.Tensor
    source: ThermoSource
    counts: dict[str, int]
    met_counts: dict[str, int]
    source_paths: dict[str, str]
    sha256: dict[str, str]
    notes: str = ""


#: The recomputation table, and the same table extended with the written-structure lipids
#: and the external-pKa targets. Two files rather than one so a run pins which it read.
METABOLITE_TABLE_FILENAME = "metabolite_dgf_prime.parquet"
METABOLITE_TABLE_WITH_EXTERNAL_FILENAME = "metabolite_dgf_prime_with_external.parquet"


def _equilibrator_metabolite_table(
    thermo_dir: str, met_ids: list[str], filename: str = METABOLITE_TABLE_FILENAME
) -> tuple[np.ndarray, np.ndarray, str]:
    """Per-metabolite eQuilibrator formation energy, aligned to the GEM's index order.

    yeast-GEM gives every compartment-specific species its own ``s_`` id, so the parquet's
    ``met_id`` IS the model metabolite id and no compartment stripping is involved.
    """
    path = osp.join(thermo_dir, filename)
    frame = pd.read_parquet(path, columns=["met_id", "dgf_prime_kj_per_mol"])
    lookup = dict(
        zip(frame["met_id"].astype(str), frame["dgf_prime_kj_per_mol"].to_numpy())
    )
    values = np.zeros(len(met_ids))
    mask = np.zeros(len(met_ids), dtype=bool)
    for i, mid in enumerate(met_ids):
        value = lookup.get(mid)
        if value is not None and np.isfinite(value):
            values[i] = float(value)
            mask[i] = True
    if not mask.any():
        raise ValueError(
            f"{path} priced none of the {len(met_ids)} GEM metabolites. The two are keyed "
            "differently, which means this table belongs to another model build."
        )
    return values, mask, path


def _shipped_reaction_column_kj(
    gem: GemTensors, thermo_dir: str
) -> tuple[np.ndarray, np.ndarray, list[str], str]:
    r"""The shipped reaction column with the dGPredictor rows' double conversion undone.

    Returns ``(values, mask, corrected_reaction_ids, path)``.

    The whole column arrives from
    :func:`~torchcell.metabolism.constraints.load_thermo_table` already multiplied by 4.184,
    because yETFL's values are kcal/mol under a kJ/mol header. The 33 rows yeast-GEM filled
    from dGPredictor were kJ/mol before that multiplication, so they are divided back here.

    The correction file is REQUIRED rather than optional. Making it optional would mean the
    union table silently ships 33 reactions whose standard energy is 4.184x too large,
    which for a 30 kJ/mol reaction is a 96 kJ/mol error in the quantity the second-law
    hinge tests the sign of. A missing file is an unbuilt input, not a default.
    """
    path = osp.join(thermo_dir, DGPREDICTOR_ROWS_FILENAME)
    if not osp.isfile(path):
        raise FileNotFoundError(
            f"{path} is missing, so the 33 dGPredictor rows of the shipped reaction column "
            "cannot be told apart from the yETFL rows and would stay over-converted by "
            "4.184. Build it with "
            "experiments/026-metabolism-flux/scripts/thermo_dgpredictor_rows.py."
        )
    with open(path) as handle:
        record = json.load(handle)
    already_kj = set(record["dgpredictor_reaction_ids"])
    factor = float(record["kj_per_kcal"])

    values = gem.thermo.rxn_delta_g.numpy().astype(np.float64).copy()  # type: ignore[union-attr]
    mask = gem.thermo.rxn_mask.numpy().astype(bool)  # type: ignore[union-attr]
    corrected: list[str] = []
    for j, reaction_id in enumerate(gem.rxn_ids):
        if reaction_id in already_kj and mask[j]:
            values[j] = values[j] / factor
            corrected.append(reaction_id)
    return values, mask, corrected, path


def _reaction_sigma(
    thermo_dir: str, s: torch.Tensor, met_ids: list[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    r"""Propagate the eQuilibrator formation covariance to reactions.

    ``formation_sigma.npz`` stores the square-root factor :math:`\Sigma^{1/2}` of the
    component-contribution covariance over the metabolites it could reach, so a reaction's
    standard deviation is :math:`\lVert S_j^\top \Sigma^{1/2}\rVert_2`. This is the
    correlated uncertainty Thermo-Flux's :math:`Qm` offset represents and the flux layer has
    been approximating isotropically. A reaction is given a sigma only when EVERY participant
    is in the factor: one uncovered participant makes the norm an underestimate rather than a
    conservative one.
    """
    path = osp.join(thermo_dir, "formation_sigma.npz")
    payload = np.load(path, allow_pickle=False)
    covered_ids = [str(x) for x in payload["met_ids"]]
    factor = payload["sigma"]
    position = {mid: i for i, mid in enumerate(covered_ids)}
    n_met = len(met_ids)
    full = np.zeros((n_met, factor.shape[1]))
    covered = np.zeros(n_met, dtype=bool)
    for i, mid in enumerate(met_ids):
        j = position.get(mid)
        if j is not None:
            full[i] = factor[j]
            covered[i] = True
    dense = s.to_dense().numpy()
    projected = dense.T @ full
    sigma = np.linalg.norm(projected, axis=1)
    participates = np.abs(dense) > 0
    complete = ~(participates & ~covered[:, None]).any(axis=0)
    complete &= participates.any(axis=0)
    return sigma * complete, complete, path


def resolve_reaction_delta_g(
    gem: GemTensors, thermo_dir: str, source: ThermoSource = ThermoSource.UNION
) -> ReactionThermoTable:
    r"""Build :math:`\Delta_r G'^\circ` for every reaction from the requested route.

    Args:
        gem: A :class:`~torchcell.metabolism.constraints.GemTensors` carrying a
            ``thermo`` table (the shipped route lives there).
        thermo_dir: ``$DATA_ROOT/data/torchcell/thermo_equilibrator``.
        source: Which route.
    """
    if gem.thermo is None:
        raise ValueError(
            "no shipped thermodynamic table on this GEM; build_gem_tensors needs "
            "model_dir so it can read model_metDeltaG.csv."
        )
    met_ids = list(gem.met_ids)
    n_reactions = gem.n_reactions
    shipped_values = gem.thermo.met_delta_g.numpy().astype(np.float64)
    shipped_mask = gem.thermo.met_mask.numpy().astype(bool)
    paths = dict(gem.thermo.source_paths)
    hashes = dict(gem.thermo.sha256)

    if source is ThermoSource.SHIPPED:
        met_values, met_mask = shipped_values * shipped_mask, shipped_mask
        met_tag = np.where(met_mask, "shipped_metabolite_sum", "none")
    else:
        eq_values, eq_mask, eq_path = _equilibrator_metabolite_table(
            thermo_dir,
            met_ids,
            filename=(
                METABOLITE_TABLE_WITH_EXTERNAL_FILENAME
                if source is ThermoSource.UNION_EXTERNAL
                else METABOLITE_TABLE_FILENAME
            ),
        )
        paths["equilibrator_met"] = eq_path
        hashes["equilibrator_met"] = _sha256_of(eq_path)
        if source is ThermoSource.EQUILIBRATOR:
            met_values, met_mask = eq_values * eq_mask, eq_mask
            met_tag = np.where(met_mask, "equilibrator", "none")
        else:
            # eQuilibrator wins where both price a species: it is one method applied at the
            # compartment's own pH and it is the only route carrying a covariance, so a
            # reaction assembled entirely from it has an uncertainty that means something.
            met_values = np.where(eq_mask, eq_values, shipped_values * shipped_mask)
            met_mask = eq_mask | shipped_mask
            met_tag = np.where(
                eq_mask,
                "equilibrator",
                np.where(shipped_mask, "shipped_metabolite_sum", "none"),
            )

    dense = gem.s.to_dense().numpy()
    participates = np.abs(dense) > 0
    delta = dense.T @ (met_values * met_mask)
    n_unknown = (participates & ~met_mask[:, None]).sum(axis=0)
    has_participants = participates.any(axis=0)
    mask = (n_unknown == 0) & has_participants

    provenance: list[str] = []
    for j in range(n_reactions):
        if not mask[j]:
            provenance.append("none")
            continue
        tags = set(met_tag[participates[:, j]])
        provenance.append(tags.pop() if len(tags) == 1 else "mixed")

    delta = delta * mask
    if source in (ThermoSource.UNION, ThermoSource.UNION_EXTERNAL):
        # Last resort: the shipped REACTION column, for reactions no metabolite table can
        # assemble. It is a different method again (yETFL by reaction id, then dG-predictor
        # matched BY ROW POSITION), so it is tagged as its own route and never merged into
        # the metabolite tally.
        #
        # The column also MIXES UNITS, and the mix is corrected before it is used: 33 of
        # its rows come from dGPredictor in kJ/mol while the rest come from yETFL in
        # kcal/mol, and the parser converts the whole column. See
        # `_shipped_reaction_column_kj`.
        rxn_values, rxn_mask, corrected, rows_path = _shipped_reaction_column_kj(
            gem, thermo_dir
        )
        paths["shipped_reaction_unit_correction"] = rows_path
        hashes["shipped_reaction_unit_correction"] = _sha256_of(rows_path)
        fill = rxn_mask & ~mask
        delta = np.where(fill, rxn_values, delta)
        for j in np.nonzero(fill)[0]:
            provenance[int(j)] = "shipped_reaction_column"
        mask = mask | fill
        n_corrected_used = sum(
            1
            for j, reaction_id in enumerate(gem.rxn_ids)
            if reaction_id in set(corrected) and fill[j]
        )

    sigma, sigma_mask, sigma_path = _reaction_sigma(thermo_dir, gem.s, met_ids)
    paths["equilibrator_sigma"] = sigma_path
    hashes["equilibrator_sigma"] = _sha256_of(sigma_path)
    sigma_mask = sigma_mask & mask
    sigma = sigma * sigma_mask

    counts: dict[str, int] = {}
    for tag in provenance:
        counts[tag] = counts.get(tag, 0) + 1
    counts["known"] = int(mask.sum())
    counts["n_reactions"] = int(n_reactions)
    counts["sigma_known"] = int(sigma_mask.sum())
    if source in (ThermoSource.UNION, ThermoSource.UNION_EXTERNAL):
        counts["shipped_reaction_rows_unit_corrected"] = len(corrected)
        counts["shipped_reaction_rows_unit_corrected_and_used"] = n_corrected_used

    met_counts = {
        "n_metabolites": len(met_ids),
        "known": int(met_mask.sum()),
        "shipped": int(shipped_mask.sum()),
    }
    if source is not ThermoSource.SHIPPED:
        met_counts["equilibrator"] = int((met_tag == "equilibrator").sum())

    return ReactionThermoTable(
        delta_r_g0=torch.from_numpy(delta).float(),
        mask=torch.from_numpy(mask),
        provenance=provenance,
        sigma=torch.from_numpy(sigma).float(),
        sigma_mask=torch.from_numpy(sigma_mask),
        source=source,
        counts=counts,
        met_counts=met_counts,
        source_paths=paths,
        sha256=hashes,
        notes=(
            "A reaction is priced only when EVERY participant carries a formation energy; "
            "one unknown participant makes the sum a number that looks like a free energy "
            "and is not one. 'mixed' marks a value assembled from two methods."
        ),
    )
