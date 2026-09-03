# torchcell/metabolism/constraints
# [[torchcell.metabolism.constraints]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/metabolism/constraints
# Test file: tests/torchcell/metabolism/test_constraints.py

r"""Genome-scale model -> constraint tensors, as pure functions of a GEM.

WHAT THIS MODULE IS FOR
-----------------------
Everything a differentiable flux layer needs from a genome-scale metabolic model,
extracted once and handed over as dense/sparse tensors with a provenance record. It is
deliberately **organism-agnostic**: nothing here knows the word "yeast". The only inputs
are a ``cobra.Model`` and, optionally, side tables of thermodynamic and kinetic
parameters. That is what makes the port to another yeast or to a bacterium a matter of
supplying a different GEM rather than editing model code.

Five things come out, and they have genuinely different epistemic status, which is why
they are separated rather than bundled into one array:

============================  =========================================  ==============
object                        what it encodes                            status
============================  =========================================  ==============
``S``                         conservation of mass                       physical law
``lb``/``ub``                 directionality + medium                    law + choice
``catalytic units`` (GPR)     which gene product catalyzes what          annotation
:math:`\Delta_f G'^\circ`     formation energy of each metabolite        measurement
:math:`k_{\mathrm{cat}}`      turnover capacity                          measurement
============================  =========================================  ==============

``S`` is never softened. The GPR is a prior whose zeros mean *untested*. The
thermodynamic and kinetic tables are incomplete measurements, so every one of them ships
with an explicit **coverage mask** -- a value is used only where the mask says a value
exists, and the mask is reported, never silently defaulted to zero. A missing
:math:`\Delta_f G'^\circ` imputed as ``0`` would assert that a metabolite has the
formation energy of an element in its standard state, which is a strong and wrong claim.

THE SENTINEL, AND WHY IT MATTERS
--------------------------------
yeast-GEM ships its thermodynamics in ``data/databases/model_metDeltaG.csv`` and
``model_rxnDeltaG.csv`` and NOT in the SBML -- ``grep -c deltaG yeast-GEM.xml`` returns 0,
so a model loaded through ``cobra.io.read_sbml_model`` has no thermodynamics at all. The
CSVs encode "unknown" as the literal value ``10000000``, not as an empty cell or NaN.
Reading them with a naive ``float()`` therefore yields a 10 MJ/mol formation energy that
is numerically finite, silently poisons every downstream sum, and never raises. Treating
that sentinel as data is the single most likely way to get a plausible-looking and
completely wrong thermodynamic term, so it is handled here, once, and the resulting
coverage is recorded in :class:`ThermoTable`.

There is a second missing-value convention on top of the sentinel: 51 metabolites and 120
reactions carry a literal ``NaN``. Both are rejected in :func:`_read_delta_g_csv`, which is
what reproduces the GEM's own curation counts. Measured on yeast-GEM 9.0.2:

============  ===================  ==================  =========================
entity        sentinel-only        sentinel + NaN      what the difference is
============  ===================  ==================  =========================
metabolites   2,440 (87.0 %)       **2,389 (85.1 %)**  51 literal NaN
reactions     3,330 (80.6 %)       **3,210 (77.7 %)**  120 literal NaN
============  ===================  ==================  =========================

THERMODYNAMICS: TWO MODES, AND THE DIFFERENCE IS THE WHOLE POINT
----------------------------------------------------------------
The flux layer can enforce "flux runs downhill" two ways, and this module supplies the
data for both.

**Free mode.** A learned per-metabolite potential :math:`\mu_i = u^\top h^m_i` with

.. math::
    \Delta_j(\mu) = \sum_i S_{ij}\,\mu_i, \qquad v_j\,\Delta_j \le 0 .

Loop-freedom falls out because :math:`\Delta` is a potential difference and therefore sums
to zero around any cycle: no cycle can run downhill everywhere. This needs no data at all
and no integer variables, which is exactly what makes loopless FBA a MILP and this cheap.
But :math:`\mu` is identified only up to an affine map, so it is not a free energy in
kJ/mol and cannot be checked against anything.

**Anchored mode.** Split the potential into the tabulated standard formation energy and a
learned log-concentration,

.. math::
    \mu_i = \Delta_f G'^{\circ}_i + RT \ln c_i,
    \qquad
    \Delta_r G'_j = \Delta_r G'^{\circ}_j + RT \sum_i S_{ij} \ln c_i ,

with :math:`\Delta_r G'^{\circ}_j = \sum_i S_{ij}\,\Delta_f G'^{\circ}_i` supplied by the
table. Now the potential has units, the sign of :math:`\Delta_r G'_j` is a physical claim,
and the learned part is a concentration that can be bounded by physiology (1 uM to 10 mM
is the usual range) and, for 19 amino acids, compared against Mulleder's measured
absolute intracellular mM. **Anchoring is what turns the thermodynamic term from a
regularizer into a measurement.** It only applies on the covered reactions; elsewhere the
free mode is the honest fallback, and the mask says which is which.

WHAT IS NOT HERE
----------------
No pH / ionic-strength Legendre transform is applied. The yeast-GEM table is already a
transformed :math:`\Delta_f G'^\circ`, and re-transforming it would double-count. No
uncertainty propagation on :math:`\Delta_f G'^\circ`: the shipped CSV is point values with
no covariance, so a component-contribution covariance would have to come from
eQuilibrator, which is not installed. Both are recorded as gaps rather than approximated.
"""

from __future__ import annotations

import hashlib
import math
import os.path as osp
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # pragma: no cover - typing only
    import cobra

#: Gas constant times a physiological temperature, kJ/mol. 303.15 K = 30 C, the assay
#: temperature of every phenotype dataset the flux layer is trained against (the pigment
#: screens and Mulleder are 30 C). Using 298.15 K here while the data is at 30 C would put
#: a systematic 1.7 % scale error into every RT ln c term.
R_KJ_PER_MOL_K = 8.314462618e-3
DEFAULT_TEMPERATURE_K = 303.15
RT_KJ_PER_MOL = R_KJ_PER_MOL_K * DEFAULT_TEMPERATURE_K

#: yeast-GEM's "no value" marker in the deltaG CSVs. Not NaN, not blank -- a finite number
#: that arithmetic accepts without complaint. See the module docstring.
DELTA_G_SENTINEL = 1.0e7

#: Physiological bounds on intracellular metabolite concentration, molar. Bounds the
#: learned log-concentration so an anchored delta_r G' cannot be satisfied by an absurd
#: concentration.
#:
#: SOURCED, because the lower bound was previously 1.0e-6 and that is one decade too high.
#: Thermo-Flux (Smith et al. 2026, doi 10.1038/s44320-026-00227-4; mirrored `paper.md`
#: sha256 496019ea07d9d95a63c1f0dbeada666d1e775a0dfcd50e53fe0235b87ead2593), Results,
#: verbatim: "Typically, intracellular metabolite concentrations vary between $0 . 1 \mu
#: \mathsf { M }$ and $1 0 \mathsf { m } M$ (Bennett et al, 2009; Kummel et al, $2 0 0 6 a
#: ,$ ), and this range is chosen as the default concentration range." The Discussion
#: repeats it: "users can define broad metabolite concentration ranges, e.g., from $0 . 1
#: \mu \mathrm { M }$ to $1 0 \mathrm { m M }$".
#:
#: The decade matters rather than being cosmetic. At 303.15 K, RT ln(10) is 5.8 kJ/mol, so
#: a floor set 10x too high removes that much per participating metabolite from the
#: driving force an anchored reaction can reach, which is the quantity the second-law
#: hinge and the dissipation limit are both computed from.
CONC_LOWER_M = 1.0e-7
CONC_UPPER_M = 1.0e-2


class ThermoMode(StrEnum):
    """How the flux layer forms its reaction potential.

    ``FREE``      -- learned per-metabolite potential, no table, no units.
    ``ANCHORED``  -- tabulated standard formation energy + learned log-concentration.
    ``OFF``       -- no thermodynamic term at all (the ablation arm).
    """

    OFF = "off"
    FREE = "free"
    ANCHORED = "anchored"


class TableCoverage(BaseModel):
    """How much of a parameter table is real, stated rather than assumed.

    Every incomplete table in this module reports one of these. A model that silently
    imputes the gaps and a model that reports 80.6 % coverage make the same predictions;
    only the second one can be reviewed.
    """

    n_total: int = Field(description="Entities the table was asked about.")
    n_known: int = Field(description="Entities carrying a real value.")
    fraction: float = Field(description="n_known / n_total.")

    @classmethod
    def of(cls, mask: np.ndarray) -> TableCoverage:
        """Coverage of a boolean mask."""
        n_total = int(mask.size)
        n_known = int(mask.sum())
        return cls(
            n_total=n_total,
            n_known=n_known,
            fraction=(n_known / n_total if n_total else 0.0),
        )


class ThermoTable(BaseModel):
    """Standard transformed Gibbs energies for one GEM, with provenance.

    The stored CSV plus its sha256 is canonical; ``source_path`` is retrieval metadata.
    ``sentinel`` is recorded explicitly so a future reader cannot mistake the convention.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    met_delta_g: torch.Tensor = Field(
        description="delta_f G'^circ per metabolite, kJ/mol; 0 where unknown."
    )
    met_mask: torch.Tensor = Field(description="Bool, True where a real value exists.")
    rxn_delta_g: torch.Tensor = Field(
        description="delta_r G'^circ per reaction, kJ/mol, as SHIPPED; 0 where unknown."
    )
    rxn_mask: torch.Tensor = Field(description="Bool, True where a real value exists.")
    met_coverage: TableCoverage
    rxn_coverage: TableCoverage
    source_paths: dict[str, str]
    sha256: dict[str, str]
    sentinel: float = DELTA_G_SENTINEL
    units: str = "kJ/mol"
    temperature_k: float = DEFAULT_TEMPERATURE_K
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    notes: str = (
        "delta_f G'^circ values are the TRANSFORMED standard energies shipped with the "
        "GEM; no further pH / ionic-strength Legendre transform is applied here, and no "
        "component-contribution covariance is available from this source."
    )


class CatalyticUnits(BaseModel):
    """The GPR, flattened into AND-terms, as index arrays rather than strings.

    A gene-protein-reaction rule is a disjunction of conjunctions: ``(a and b) or c``
    means reaction ``j`` runs if the complex ``{a,b}`` is intact OR the isozyme ``c`` is
    present. Each conjunction is one **catalytic unit**. Complexes take a min over their
    genes (the scarcest subunit limits the complex); isozymes add (their capacities sum).

    Stored as a flat (unit, gene) edge list plus a (unit -> reaction) map, which is the
    shape a scatter-based ``softmin`` / ``index_add`` consumes directly. Storing the
    parsed rule as a string and re-parsing per forward pass would be the same information
    at 10^4 times the cost.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit_gene_index: torch.Tensor = Field(
        description="[2, nnz] long; row 0 = unit id, row 1 = gene id."
    )
    unit_reaction: torch.Tensor = Field(
        description="[n_units] long; the reaction each unit catalyzes."
    )
    n_units: int
    n_multigene_units: int = Field(
        description="Units with >1 gene, i.e. real complexes."
    )
    n_reactions_with_gpr: int
    gene_ids: list[str] = Field(description="GEM gene ids, in tensor-index order.")


class GemTensors(BaseModel):
    """A genome-scale model as tensors: stoichiometry, bounds, GPR, thermodynamics.

    Everything is index-aligned: metabolite ``i`` is ``met_ids[i]`` in every tensor,
    reaction ``j`` is ``rxn_ids[j]``, gene ``g`` is ``catalytic_units.gene_ids[g]``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    s: torch.Tensor = Field(description="[m, r] sparse COO stoichiometric matrix.")
    lb: torch.Tensor = Field(description="[r] lower flux bounds, mmol/gDW/h.")
    ub: torch.Tensor = Field(description="[r] upper flux bounds, mmol/gDW/h.")
    met_ids: list[str]
    rxn_ids: list[str]
    catalytic_units: CatalyticUnits
    thermo: ThermoTable | None = None
    independent_rows: torch.Tensor | None = Field(
        default=None,
        description=(
            "[rank(S)] long index of linearly independent balance rows. The remaining "
            "rows are linear combinations whose residual is already determined by these, "
            "so penalizing them adds noise rather than information."
        ),
    )
    biomass_index: int | None = None
    exchange_indices: torch.Tensor | None = None
    model_id: str = ""
    n_metabolites: int = 0
    n_reactions: int = 0

    # -- derived quantities the flux layer asks for by name -------------------

    @property
    def reversible_mask(self) -> torch.Tensor:
        """True where ``lb < 0 < ub``: the reaction may run in either direction."""
        return (self.lb < 0) & (self.ub > 0)

    def standard_reaction_delta_g(self) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Recompute :math:`\Delta_r G'^\circ_j=\sum_i S_{ij}\Delta_f G'^\circ_i`.

        The GEM ships a reaction table too, but it is recomputed from the metabolite
        table here for a reason worth stating: the reaction value is only meaningful when
        **every** participating metabolite carries a formation energy. Summing over a
        reaction with one unknown participant produces a number that looks like a free
        energy and is not one. The returned mask is therefore the AND over participants,
        which is strictly more conservative than trusting the shipped reaction column, and
        the two are compared in :func:`compare_reaction_delta_g`.

        Returns:
            ``(delta_r_g0, mask)`` -- both ``[r]``, energies in kJ/mol, zero where masked.
        """
        if self.thermo is None:
            raise ValueError("no thermodynamic table attached to this GEM")
        s_dense_abs = torch.sparse.sum(
            torch.sparse_coo_tensor(
                self.s.indices(), self.s.values().abs(), self.s.shape
            ),
            dim=0,
        ).to_dense()
        del s_dense_abs  # participation count is recomputed below on the mask instead
        met_g = self.thermo.met_delta_g * self.thermo.met_mask
        delta = torch.sparse.mm(self.s.t(), met_g.unsqueeze(-1)).squeeze(-1)
        # A participant is unknown if |S_ij| > 0 and met_mask is False. Count them per
        # reaction with one sparse mm against the complement of the mask.
        unknown = (~self.thermo.met_mask).to(self.s.dtype)
        s_abs = torch.sparse_coo_tensor(
            self.s.indices(), self.s.values().abs(), self.s.shape
        ).coalesce()
        n_unknown = torch.sparse.mm(s_abs.t(), unknown.unsqueeze(-1)).squeeze(-1)
        mask = n_unknown == 0
        return delta * mask, mask


def _sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Hash a file's bytes; this is what makes the stored table canonical."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _read_delta_g_csv(path: str) -> dict[str, float]:
    r"""Read a two-column ``id,value`` GEM deltaG CSV, dropping every missing-value form.

    **The file uses TWO missing-value conventions and they are easy to miss.** Most gaps
    are the sentinel ``10000000``; a further 51 metabolites and 120 reactions are the
    literal string ``NaN``. Filtering only the sentinel leaves the NaNs, which then
    propagate silently through every sum -- a reaction touching one gets a NaN
    :math:`\\Delta_r G'^\\circ`, its hinge contributes NaN to the loss, and the whole
    training run produces NaN gradients from what looks like a coverage number of 87 %.

    Rejecting both is what reproduces the counts the GEM's own curation reports: **2,389
    metabolites** and **3,210 reactions**, not the 2,440 / 3,330 a sentinel-only filter
    gives. The discrepancy between those two pairs of numbers IS the NaN set.

    Deliberately not ``pandas.read_csv`` + a filter: the drop has to happen at the point of
    parsing so no code path can ever hold a ``1e7`` or a NaN and call it an energy.
    """
    out: dict[str, float] = {}
    with open(path) as f:
        header = f.readline()
        if "," not in header:
            raise ValueError(f"{path}: expected a two-column CSV, got {header!r}")
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, raw = line.partition(",")
            value = float(raw)
            if value == DELTA_G_SENTINEL or not math.isfinite(value):
                continue
            out[key] = value
    return out


def load_thermo_table(
    model_dir: str, met_ids: list[str], rxn_ids: list[str]
) -> ThermoTable:
    r"""Load :math:`\Delta_f G'^\circ` / :math:`\Delta_r G'^\circ` from a GEM checkout.

    Reads ``<model_dir>/data/databases/model_metDeltaG.csv`` and ``model_rxnDeltaG.csv``,
    the files ``code/missingFields/loadDeltaG.m`` uses to populate the model's
    ``metDeltaG`` / ``rxnDeltaG`` fields. These are absent from the SBML export, which is
    why loading the GEM through cobra alone yields no thermodynamics.

    Metabolite ids in the CSV are bare (``s_0001``) while a cobra SBML model compartmentizes
    them (``s_0001[c]`` or ``s_0001_c``); the compartment tag is stripped for matching and
    the compartment is otherwise ignored, because the shipped table is per-species and
    carries no compartment-specific transform.

    Args:
        model_dir: The GEM checkout root (the directory containing ``data/`` and ``model/``).
        met_ids: Metabolite ids in tensor-index order.
        rxn_ids: Reaction ids in tensor-index order.

    Returns:
        A :class:`ThermoTable` aligned to ``met_ids`` / ``rxn_ids``, with coverage masks.
    """
    met_path = osp.join(model_dir, "data", "databases", "model_metDeltaG.csv")
    rxn_path = osp.join(model_dir, "data", "databases", "model_rxnDeltaG.csv")
    met_raw = _read_delta_g_csv(met_path)
    rxn_raw = _read_delta_g_csv(rxn_path)

    met_g = np.zeros(len(met_ids), dtype=np.float64)
    met_mask = np.zeros(len(met_ids), dtype=bool)
    for i, mid in enumerate(met_ids):
        value = met_raw.get(_strip_compartment(mid))
        if value is not None:
            met_g[i] = value
            met_mask[i] = True

    rxn_g = np.zeros(len(rxn_ids), dtype=np.float64)
    rxn_mask = np.zeros(len(rxn_ids), dtype=bool)
    for j, rid in enumerate(rxn_ids):
        value = rxn_raw.get(rid)
        if value is not None:
            rxn_g[j] = value
            rxn_mask[j] = True

    return ThermoTable(
        met_delta_g=torch.from_numpy(met_g).float(),
        met_mask=torch.from_numpy(met_mask),
        rxn_delta_g=torch.from_numpy(rxn_g).float(),
        rxn_mask=torch.from_numpy(rxn_mask),
        met_coverage=TableCoverage.of(met_mask),
        rxn_coverage=TableCoverage.of(rxn_mask),
        source_paths={"met": met_path, "rxn": rxn_path},
        sha256={"met": _sha256_file(met_path), "rxn": _sha256_file(rxn_path)},
    )


_COMPARTMENT_RE = re.compile(r"(\[[a-z]+\]|_[a-z])$")


def _strip_compartment(met_id: str) -> str:
    """``s_0001[c]`` / ``s_0001_c`` -> ``s_0001``; anything else is returned unchanged."""
    return _COMPARTMENT_RE.sub("", met_id)


def parse_catalytic_units(model: cobra.Model, rxn_ids: list[str]) -> CatalyticUnits:
    """Flatten every GPR into AND-terms indexed against ``rxn_ids``.

    Uses cobra's own parsed ``gene_reaction_rule`` string rather than the model's
    ``genes`` collection, because the collection loses the AND/OR structure that decides
    whether a deletion is lethal: an isozyme pair and a two-subunit complex have the same
    gene set and opposite deletion phenotypes.
    """
    gene_ids: list[str] = sorted(g.id for g in model.genes)
    gene_pos = {g: i for i, g in enumerate(gene_ids)}
    unit_gene: list[tuple[int, int]] = []
    unit_rxn: list[int] = []
    n_multigene = 0
    n_with_gpr = 0
    for j, rid in enumerate(rxn_ids):
        rule = model.reactions.get_by_id(rid).gene_reaction_rule
        if not rule or not rule.strip():
            continue
        n_with_gpr += 1
        for term in _split_or(rule):
            genes = [g for g in _split_and(term) if g in gene_pos]
            if not genes:
                continue
            unit_id = len(unit_rxn)
            unit_rxn.append(j)
            if len(genes) > 1:
                n_multigene += 1
            for g in genes:
                unit_gene.append((unit_id, gene_pos[g]))

    index = torch.tensor(unit_gene, dtype=torch.long).t().contiguous()
    if index.numel() == 0:
        index = torch.zeros(2, 0, dtype=torch.long)
    return CatalyticUnits(
        unit_gene_index=index,
        unit_reaction=torch.tensor(unit_rxn, dtype=torch.long),
        n_units=len(unit_rxn),
        n_multigene_units=n_multigene,
        n_reactions_with_gpr=n_with_gpr,
        gene_ids=gene_ids,
    )


def _split_or(rule: str) -> list[str]:
    """Split a GPR on top-level ``or``. Parentheses are dropped, as cobra rules are flat."""
    return [t for t in re.split(r"\bor\b", rule.replace("(", " ").replace(")", " "))]


def _split_and(term: str) -> list[str]:
    """Split one OR-term on ``and`` into its subunit gene ids."""
    return [g.strip() for g in re.split(r"\band\b", term) if g.strip()]


def independent_balance_rows(
    s: torch.Tensor, tol: float = 1e-9
) -> tuple[torch.Tensor, int]:
    r"""Pick a maximal set of linearly independent rows of ``S``.

    Rank-revealing QR with column pivoting on :math:`S^\top`: the pivot order returned by
    ``scipy.linalg.qr(..., pivoting=True)`` lists columns of :math:`S^\top` -- i.e. rows of
    :math:`S` -- in decreasing order of independence, and the first ``rank`` of them are a
    basis for the row space.

    Why bother: the mass-balance penalty runs over rows of ``S``, and the dependent rows
    (conserved moieties, dead-end metabolites) carry residuals that are exact linear
    combinations of the independent ones. Penalizing all 2,806 rows therefore weights some
    directions of the residual several times over, which is a silent, structured
    mis-weighting of the physics term rather than a harmless redundancy. On yeast-GEM
    9.0.2 the rank is 2,593, so 213 rows are redundant.

    Args:
        s: ``[m, r]`` stoichiometric matrix, sparse or dense.
        tol: Relative tolerance on the R diagonal for the numerical rank.

    Returns:
        ``(rows, rank)`` -- a sorted long tensor of row indices, and the numerical rank.
    """
    from scipy.linalg import qr

    dense = (s.to_dense() if s.is_sparse else s).double().numpy()
    _, r_mat, pivots = qr(dense.T, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r_mat))
    rank = int((diag > tol * diag.max()).sum()) if diag.size else 0
    rows = np.sort(pivots[:rank])
    return torch.from_numpy(rows.astype(np.int64)), rank


def null_space_basis(
    s: torch.Tensor, tol: float = 1e-9, cache_path: str | None = None
) -> torch.Tensor:
    r"""An orthonormal basis :math:`\mathcal{N}` of :math:`\ker S`, so :math:`S\mathcal{N}=0`.

    This is the other half of the exactness budget. A flux written as
    :math:`v=\mathcal{N}z` satisfies mass balance **identically**, to machine precision,
    for any :math:`z` -- there is no residual and no loss weight. What it gives up is the
    box: an orthonormal basis does not preserve :math:`v^\ell\le v\le v^u`, so bounds and
    directionality revert to a penalty.

    The two parameterizations therefore spend exactness on different constraints, and
    which is right is an empirical question rather than a matter of taste:

    ==========================  =====================  =====================
    parameterization            :math:`Sv=0`           box + directionality
    ==========================  =====================  =====================
    box (sigmoid)               soft, one weight       **exact**
    null space                  **exact**              soft, one weight
    ==========================  =====================  =====================

    On yeast-GEM 9.0.2 the basis is :math:`4131\times1538`, so the null-space head is
    2.7x narrower than the per-reaction head, which is a real secondary benefit.

    Computed by SVD rather than by a sparse method: the matrix is 2,806 x 4,131 and dense
    SVD takes seconds, whereas a sparse null-space routine would need a rank decision
    anyway. The result is cached because it depends only on ``S``.
    """
    if cache_path is not None and osp.exists(cache_path):
        return torch.from_numpy(np.load(cache_path))
    dense = (s.to_dense() if s.is_sparse else s).double().numpy()
    _, singular, vt = np.linalg.svd(dense, full_matrices=True)
    rank = int((singular > tol * singular.max()).sum())
    basis = vt[rank:].T.copy()
    if cache_path is not None:
        np.save(cache_path, basis.astype(np.float32))
    return torch.from_numpy(basis.astype(np.float32))


def build_gem_tensors(
    model: cobra.Model,
    model_dir: str | None = None,
    *,
    with_thermo: bool = True,
    with_independent_rows: bool = True,
    biomass_id: str | None = None,
) -> GemTensors:
    """Extract every constraint tensor from a cobra model.

    Args:
        model: Any cobra model. Nothing here is organism-specific.
        model_dir: GEM checkout root, needed only for the thermodynamic CSVs.
        with_thermo: Load the deltaG tables. Requires ``model_dir``.
        with_independent_rows: Run the rank-revealing QR. Costs a few seconds on a 2,806 x
            4,131 model and is cached by the caller, not here.
        biomass_id: Reaction id of the growth reaction. When omitted it is detected by
            objective coefficient, which is how a GEM declares it.

    Returns:
        A fully populated :class:`GemTensors`.
    """
    met_ids = [m.id for m in model.metabolites]
    rxn_ids = [r.id for r in model.reactions]
    met_pos = {m: i for i, m in enumerate(met_ids)}

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for j, rxn in enumerate(model.reactions):
        for met, coef in rxn.metabolites.items():
            rows.append(met_pos[met.id])
            cols.append(j)
            vals.append(float(coef))
    s = torch.sparse_coo_tensor(
        torch.tensor([rows, cols], dtype=torch.long),
        torch.tensor(vals, dtype=torch.float32),
        (len(met_ids), len(rxn_ids)),
    ).coalesce()

    lb = torch.tensor([float(r.lower_bound) for r in model.reactions])
    ub = torch.tensor([float(r.upper_bound) for r in model.reactions])

    thermo = (
        load_thermo_table(model_dir, met_ids, rxn_ids)
        if with_thermo and model_dir is not None
        else None
    )
    rows_idx = None
    if with_independent_rows:
        rows_idx, _ = independent_balance_rows(s)

    if biomass_id is None:
        objective = [
            j
            for j, r in enumerate(model.reactions)
            if abs(float(r.objective_coefficient)) > 0
        ]
        bio_idx = objective[0] if objective else None
    else:
        bio_idx = rxn_ids.index(biomass_id)

    exchange = torch.tensor(
        [j for j, r in enumerate(model.reactions) if len(r.metabolites) == 1],
        dtype=torch.long,
    )

    return GemTensors(
        s=s,
        lb=lb,
        ub=ub,
        met_ids=met_ids,
        rxn_ids=rxn_ids,
        catalytic_units=parse_catalytic_units(model, rxn_ids),
        thermo=thermo,
        independent_rows=rows_idx,
        biomass_index=bio_idx,
        exchange_indices=exchange,
        model_id=str(model.id),
        n_metabolites=len(met_ids),
        n_reactions=len(rxn_ids),
    )


def compare_reaction_delta_g(gem: GemTensors) -> dict[str, Any]:
    r"""Check the recomputed :math:`\Delta_r G'^\circ` against the shipped column.

    Two independent routes to the same quantity: sum the metabolite table over ``S``, or
    read the reaction table. Agreement is evidence the metabolite ids line up and the
    sentinel was handled; disagreement localizes the problem. This is a sanity check with
    a number attached, which is the only kind worth running.

    Returns a dict of counts and the residual distribution over the reactions where both
    routes have a value.
    """
    recomputed, recomputed_mask = gem.standard_reaction_delta_g()
    assert gem.thermo is not None
    shipped, shipped_mask = gem.thermo.rxn_delta_g, gem.thermo.rxn_mask
    both = recomputed_mask & shipped_mask
    residual = (recomputed[both] - shipped[both]).abs()
    return {
        "n_reactions": int(gem.n_reactions),
        "n_shipped": int(shipped_mask.sum()),
        "n_recomputed_all_participants_known": int(recomputed_mask.sum()),
        "n_both": int(both.sum()),
        "abs_residual_median_kj_per_mol": float(residual.median())
        if both.any()
        else None,
        "abs_residual_p95_kj_per_mol": (
            float(residual.quantile(0.95)) if both.any() else None
        ),
        "abs_residual_max_kj_per_mol": float(residual.max()) if both.any() else None,
    }
