# tests/torchcell/metabolism/test_constraints.py
# [[tests.torchcell.metabolism.test_constraints]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/metabolism/test_constraints.py
"""GEM -> tensor extraction on a two-metabolite, two-reaction cobra model built in the test.

The toy: metabolites ``A_c`` and ``B_c``; ``R1: A_c -> B_c`` with bounds ``[-5, 10]``,
GPR ``(g1 and g2) or g3`` and objective coefficient 1; ``EX_B: B_c ->`` with bounds
``[0, 1000]``. So ``S = [[-1, 0], [1, -1]]`` (rank 2, both rows independent), ``R1`` is
reversible and the objective, ``EX_B`` is the only single-metabolite (exchange) reaction.
Genes sort to ``[g1, g2, g3]``; the GPR flattens to unit 0 = {g1, g2} and unit 1 = {g3},
both on reaction 0.

The thermodynamic CSVs written under ``<model_dir>/data/databases/`` give ``A = -10.5``,
``B = 5.0`` kJ/mol plus a sentinel ``10000000`` row and a ``NaN`` row that must be dropped,
and a shipped ``R1 = 15.0`` with ``EX_B`` sentinel. The recomputed reaction energies are
``S^T g = [(-1)(-10.5) + (1)(5.0), (-1)(5.0)] = [15.5, -5.0]``, so the one reaction
present in both routes has ``|15.5 - 15.0| = 0.5`` residual.
"""

import hashlib
from pathlib import Path

import cobra
import numpy as np
import pytest
import torch

from torchcell.metabolism.constraints import (
    DELTA_G_SENTINEL,
    GemTensors,
    TableCoverage,
    _read_delta_g_csv,
    _split_and,
    _split_or,
    _strip_compartment,
    build_gem_tensors,
    compare_reaction_delta_g,
    independent_balance_rows,
    null_space_basis,
)

MET_ROWS = "metabolite,deltaG\nA,-10.5\nB,5.0\n\nC,10000000\nD,NaN\n"
RXN_ROWS = "reaction,deltaG\nR1,15.0\nEX_B,10000000\n"


def _toy_model() -> cobra.Model:
    model = cobra.Model("toy")
    a = cobra.Metabolite("A_c", compartment="c")
    b = cobra.Metabolite("B_c", compartment="c")
    r1 = cobra.Reaction("R1", lower_bound=-5.0, upper_bound=10.0)
    r1.add_metabolites({a: -1.0, b: 1.0})
    r1.gene_reaction_rule = "(g1 and g2) or g3"
    ex = cobra.Reaction("EX_B", lower_bound=0.0, upper_bound=1000.0)
    ex.add_metabolites({b: -1.0})
    model.add_reactions([r1, ex])
    model.objective = "R1"
    return model


def _model_dir(
    tmp_path: Path, met_rows: str = MET_ROWS, rxn_rows: str = RXN_ROWS
) -> str:
    databases = tmp_path / "gem" / "data" / "databases"
    databases.mkdir(parents=True)
    (databases / "model_metDeltaG.csv").write_text(met_rows)
    (databases / "model_rxnDeltaG.csv").write_text(rxn_rows)
    return str(tmp_path / "gem")


def _gem(tmp_path: Path) -> GemTensors:
    return build_gem_tensors(_toy_model(), _model_dir(tmp_path))


def test_read_delta_g_csv_drops_the_sentinel_nan_and_blank_lines(
    tmp_path: Path,
) -> None:
    """Only ``A`` and ``B`` survive; the header must contain a comma."""
    model_dir = _model_dir(tmp_path)
    assert DELTA_G_SENTINEL == 1.0e7
    met_path = f"{model_dir}/data/databases/model_metDeltaG.csv"
    assert _read_delta_g_csv(met_path) == {"A": -10.5, "B": 5.0}
    assert _read_delta_g_csv(f"{model_dir}/data/databases/model_rxnDeltaG.csv") == {
        "R1": 15.0
    }
    bad = tmp_path / "bad.csv"
    bad.write_text("nocomma\nA,1.0\n")
    with pytest.raises(
        ValueError, match=rf"{bad}: expected a two-column CSV, got 'nocomma\\n'"
    ):
        _read_delta_g_csv(str(bad))


@pytest.mark.parametrize(
    ("met_id", "bare"),
    [
        ("s_0001[c]", "s_0001"),
        ("s_0001_c", "s_0001"),
        ("s_0001", "s_0001"),
        ("A_c", "A"),
        ("x[ce]", "x"),
        ("abc_1", "abc_1"),
    ],
)
def test_strip_compartment_removes_one_trailing_tag(met_id: str, bare: str) -> None:
    """A bracketed or single-letter underscore suffix goes; a digit suffix stays."""
    assert _strip_compartment(met_id) == bare


def test_gpr_splitting_keeps_terms_and_trims_genes() -> None:
    """``or`` splits with parentheses blanked; ``and`` splits and strips."""
    assert _split_or("(g1 and g2) or g3") == [" g1 and g2  ", " g3"]
    assert _split_and(" g1 and g2  ") == ["g1", "g2"]
    assert _split_and(" g3") == ["g3"]


def test_stoichiometry_bounds_and_indices_of_the_toy(tmp_path: Path) -> None:
    """``S = [[-1, 0], [1, -1]]``, bounds as declared, R1 reversible, EX_B the exchange."""
    gem = _gem(tmp_path)
    assert gem.s.is_sparse
    assert gem.s.to_dense().tolist() == [[-1.0, 0.0], [1.0, -1.0]]
    assert gem.lb.tolist() == [-5.0, 0.0]
    assert gem.ub.tolist() == [10.0, 1000.0]
    assert gem.met_ids == ["A_c", "B_c"]
    assert gem.rxn_ids == ["R1", "EX_B"]
    assert gem.model_id == "toy"
    assert (gem.n_metabolites, gem.n_reactions) == (2, 2)
    assert gem.biomass_index == 0
    assert gem.exchange_indices is not None
    assert gem.exchange_indices.tolist() == [1]
    assert gem.independent_rows is not None
    assert gem.independent_rows.tolist() == [0, 1]
    assert gem.reversible_mask.tolist() == [True, False]


def test_catalytic_units_flatten_the_gpr_into_two_units(tmp_path: Path) -> None:
    """Unit 0 = {g1, g2} (a complex), unit 1 = {g3}; both catalyze reaction 0."""
    units = _gem(tmp_path).catalytic_units
    assert units.gene_ids == ["g1", "g2", "g3"]
    assert units.unit_gene_index.tolist() == [[0, 0, 1], [0, 1, 2]]
    assert units.unit_reaction.tolist() == [0, 0]
    assert units.n_units == 2
    assert units.n_multigene_units == 1
    assert units.n_reactions_with_gpr == 1


def test_thermo_table_aligns_to_the_model_and_hashes_its_files(tmp_path: Path) -> None:
    """``A_c`` matches ``A`` after stripping ``_c``; coverage 2/2 metabolites, 1/2 reactions."""
    gem = _gem(tmp_path)
    thermo = gem.thermo
    assert thermo is not None
    assert thermo.met_delta_g.tolist() == [-10.5, 5.0]
    assert thermo.met_mask.tolist() == [True, True]
    assert thermo.rxn_delta_g.tolist() == [15.0, 0.0]
    assert thermo.rxn_mask.tolist() == [True, False]
    assert thermo.met_coverage == TableCoverage(n_total=2, n_known=2, fraction=1.0)
    assert thermo.rxn_coverage == TableCoverage(n_total=2, n_known=1, fraction=0.5)
    met_path = str(tmp_path / "gem" / "data" / "databases" / "model_metDeltaG.csv")
    rxn_path = str(tmp_path / "gem" / "data" / "databases" / "model_rxnDeltaG.csv")
    assert thermo.source_paths == {"met": met_path, "rxn": rxn_path}
    assert thermo.sha256 == {
        "met": hashlib.sha256(MET_ROWS.encode()).hexdigest(),
        "rxn": hashlib.sha256(RXN_ROWS.encode()).hexdigest(),
    }
    assert thermo.sentinel == DELTA_G_SENTINEL
    assert thermo.units == "kJ/mol"


def test_recomputed_reaction_energies_are_s_transpose_times_formation_energies(
    tmp_path: Path,
) -> None:
    """``[15.5, -5.0]`` with every participant known; the shipped route agrees to 0.5."""
    gem = _gem(tmp_path)
    delta, mask = gem.standard_reaction_delta_g()
    assert delta.tolist() == [15.5, -5.0]
    assert mask.tolist() == [True, True]
    assert compare_reaction_delta_g(gem) == {
        "n_reactions": 2,
        "n_shipped": 1,
        "n_recomputed_all_participants_known": 2,
        "n_both": 1,
        "abs_residual_median_kj_per_mol": 0.5,
        "abs_residual_p95_kj_per_mol": 0.5,
        "abs_residual_max_kj_per_mol": 0.5,
    }


def test_one_unknown_participant_masks_every_reaction_touching_it(
    tmp_path: Path,
) -> None:
    """With ``B`` sentinel both reactions are masked to 0; no shipped value means None residuals."""
    model_dir = _model_dir(
        tmp_path, "id,v\nA,-10.5\nB,10000000\n", "id,v\nR1,10000000\nEX_B,NaN\n"
    )
    gem = build_gem_tensors(_toy_model(), model_dir)
    assert gem.thermo is not None
    assert gem.thermo.met_mask.tolist() == [True, False]
    assert gem.thermo.rxn_mask.tolist() == [False, False]
    delta, mask = gem.standard_reaction_delta_g()
    assert delta.tolist() == [0.0, 0.0]
    assert mask.tolist() == [False, False]
    assert compare_reaction_delta_g(gem) == {
        "n_reactions": 2,
        "n_shipped": 0,
        "n_recomputed_all_participants_known": 0,
        "n_both": 0,
        "abs_residual_median_kj_per_mol": None,
        "abs_residual_p95_kj_per_mol": None,
        "abs_residual_max_kj_per_mol": None,
    }


def test_build_without_a_model_dir_has_no_thermo_and_takes_the_named_biomass() -> None:
    """``thermo`` and ``independent_rows`` are None; ``biomass_id="EX_B"`` gives index 1."""
    gem = build_gem_tensors(
        _toy_model(), None, with_independent_rows=False, biomass_id="EX_B"
    )
    assert gem.thermo is None
    assert gem.independent_rows is None
    assert gem.biomass_index == 1
    with pytest.raises(ValueError, match="no thermodynamic table attached to this GEM"):
        gem.standard_reaction_delta_g()


def test_no_objective_and_no_gpr_give_none_biomass_and_an_empty_edge_list() -> None:
    """Empty GPRs produce a ``[2, 0]`` long index, zero units, and no reaction with a rule."""
    model = _toy_model()
    model.objective = {}
    for reaction in model.reactions:
        reaction.gene_reaction_rule = ""
    gem = build_gem_tensors(model, None)
    assert gem.biomass_index is None
    units = gem.catalytic_units
    assert units.unit_gene_index.shape == (2, 0)
    assert units.unit_gene_index.dtype == torch.long
    assert units.unit_reaction.tolist() == []
    assert (units.n_units, units.n_multigene_units, units.n_reactions_with_gpr) == (
        0,
        0,
        0,
    )


def test_null_space_basis_is_an_orthonormal_kernel_and_is_cached(
    tmp_path: Path,
) -> None:
    """On the 3x4 cycle network the kernel is 2-D: ``S N = 0`` and ``N^T N = I``.

    The basis is saved as float32 ``.npy`` on first call and read back verbatim afterwards;
    the full-rank 2x2 toy has an empty ``[2, 0]`` kernel.
    """
    s = torch.tensor(
        [[1.0, -1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    basis = null_space_basis(s)
    assert basis.shape == (4, 2)
    assert basis.dtype == torch.float32
    torch.testing.assert_close(s @ basis, torch.zeros(3, 2), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2), atol=1e-6, rtol=0.0)

    cache = tmp_path / "null_space.npy"
    cached = null_space_basis(s.to_sparse_coo(), cache_path=str(cache))
    assert torch.equal(cached, basis)
    assert np.load(cache).dtype == np.float32
    np.save(cache, np.ones((4, 2), dtype=np.float32))
    assert torch.equal(null_space_basis(s, cache_path=str(cache)), torch.ones(4, 2))

    full_rank = torch.tensor([[-1.0, 0.0], [1.0, -1.0]])
    assert null_space_basis(full_rank).shape == (2, 0)


def test_independent_balance_rows_drops_the_zero_row_and_handles_empty_input() -> None:
    """Rows 0 and 1 span the 3x4 network (rank 2); an empty matrix has rank 0."""
    s = torch.tensor(
        [[1.0, -1.0, 1.0, 0.0], [0.0, 1.0, -1.0, -1.0], [0.0, 0.0, 0.0, 0.0]]
    )
    rows, rank = independent_balance_rows(s)
    assert rows.tolist() == [0, 1]
    assert rows.dtype == torch.int64
    assert rank == 2
    empty_rows, empty_rank = independent_balance_rows(torch.zeros(0, 3))
    assert empty_rows.tolist() == []
    assert empty_rank == 0


def test_table_coverage_of_an_empty_mask_is_zero_not_a_division_error() -> None:
    """``n_total = 0`` reports ``fraction = 0.0``; a mask [T, F, T] reports 2/3."""
    assert TableCoverage.of(np.zeros(0, dtype=bool)) == TableCoverage(
        n_total=0, n_known=0, fraction=0.0
    )
    partial = TableCoverage.of(np.array([True, False, True]))
    assert (partial.n_total, partial.n_known) == (3, 2)
    assert partial.fraction == pytest.approx(2 / 3)
