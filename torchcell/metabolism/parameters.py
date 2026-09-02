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
import os.path as osp
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field

from torchcell.metabolism.constraints import CatalyticUnits, TableCoverage

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
    ORGANISM_DEFAULT = "organism_default"
    SWISSPROT = "swissprot"
    YMDB = "ymdb"


#: Which provenance tags are measurements rather than model output. The
#: published-only ablation is exactly a filter on this set.
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


def resolve_kcat_table(
    units: CatalyticUnits,
    model_dir: str,
    oed_mirror_dir: str,
    *,
    registry: PredictorRegistry | None = None,
    target_temperature_c: float = 30.0,
) -> ParameterTable:
    r"""One :math:`k_{cat}` per catalytic unit, database-first.

    A catalytic unit is an AND-term of genes: a complex has several, an isozyme one. The
    unit's turnover is taken as the **minimum** over its member genes' resolved values,
    because a complex cannot turn over faster than its slowest constituent, and this is the
    same "scarcest subunit limits" logic the availability softmin uses one level up.

    Resolution per gene: UniProt accession from SwissProt -> Open Enzyme Database rows for
    that accession -> the existing temperature-aware selection cascade
    (:func:`~torchcell.metabolism.enzyme_kinetics.resolve_parameter`) -> a registered
    sequence predictor if one is present -> the organism default. The organism default is
    the **median of whatever the database did supply**, not a literature constant, so it is
    at least on the right scale for the organism at hand; with an empty slice it falls back
    to :data:`FALLBACK_KCAT_PER_S`.

    Returns:
        A :class:`ParameterTable` of length ``units.n_units``, in units of 1/s.
    """
    records: list[OedKineticRecord] = []
    try:
        records = load_mirrored_records(oed_mirror_dir)
    except FileNotFoundError:
        records = []
    by_uniprot = index_by_uniprot(records)
    gene_to_uniprot = uniprot_for_genes(model_dir, units.gene_ids)
    sequences = {
        gid: row["sequence"]
        for gid, row in load_swissprot(model_dir).items()
        if row.get("sequence")
    }

    per_gene: dict[int, tuple[float, ParameterProvenance]] = {}
    for g, gid in enumerate(units.gene_ids):
        accession = gene_to_uniprot.get(gid)
        resolved = None
        if accession and accession in by_uniprot:
            resolved = resolve_parameter(
                by_uniprot[accession],
                KineticKind.KCAT,
                uniprot=accession,
                target_temperature_c=target_temperature_c,
            )
        if resolved is not None:
            per_gene[g] = (
                float(resolved.value),
                ParameterProvenance.OPEN_ENZYME_DATABASE,
            )
            continue
        if registry is not None and registry.predictors and gid in sequences:
            predictor = registry.predictors[0]
            kcat, _ = predictor.predict(sequences[gid])  # type: ignore[attr-defined]
            per_gene[g] = (
                float(kcat),
                ParameterProvenance(getattr(predictor, "name", "kcatnet")),
            )

    measured = [v for v, _ in per_gene.values()]
    default = (
        float(torch.tensor(measured).median()) if measured else FALLBACK_KCAT_PER_S
    )

    unit_genes: dict[int, list[int]] = {}
    for u, g in zip(
        units.unit_gene_index[0].tolist(), units.unit_gene_index[1].tolist()
    ):
        unit_genes.setdefault(u, []).append(g)

    values: list[float] = []
    provenance: list[ParameterProvenance] = []
    for u in range(units.n_units):
        hits = [per_gene[g] for g in unit_genes.get(u, []) if g in per_gene]
        if hits:
            value, source = min(hits, key=lambda t: t[0])
            values.append(value)
            provenance.append(source)
        else:
            values.append(default)
            provenance.append(ParameterProvenance.ORGANISM_DEFAULT)

    return ParameterTable.build(
        torch.tensor(values, dtype=torch.float32),
        provenance,
        unit="1/s",
        notes=(
            f"Open Enzyme Database first ({len(records)} organism rows), then registered "
            f"sequence predictors, then the organism default {default:.3g} 1/s = the "
            "median of the resolved values. A complex takes the min over its subunits."
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
