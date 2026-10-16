# tests/torchcell/paper/test_dataset_sequence_fit.py
# [[tests.torchcell.paper.test_dataset_sequence_fit]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/paper/test_dataset_sequence_fit.py
"""The expanded datasets table's two derived columns and its descriptions file.

``sequence_fit`` in ``experiments/database/scripts/build_bacteria_datasets_table.py``
turns one measured row (graph role, the modal edit count of the genotype sample, the
perturbation types, the environment count) into the tier and the label unit the table
prints. The rules are the thing under test: a node-level label keeps its sequence anchor
however many edits the strain carries, while a whole-strain readout loses a tier once it
is a multi-edit strain, and a natural genome has no anchor at all.

``dataset_descriptions.json`` is read as-is: it is the table's one hand-written input, so
the shape the renderer depends on (a non-empty string per class, no pipe to break a
markdown cell, no em-dash) is pinned here rather than discovered in a render.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = (
    REPO / "experiments" / "database" / "scripts" / "build_bacteria_datasets_table.py"
)
DESCRIPTIONS = (
    REPO / "experiments" / "database" / "scripts" / "dataset_descriptions.json"
)


def _load_build_script():
    """Import the experiment script by path (it is a script, not a package module)."""
    spec = importlib.util.spec_from_file_location("_build_datasets_table", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build = _load_build_script()


def row(**kwargs):
    """A probe record with only the fields ``sequence_fit`` reads set."""
    base = dict(
        organism="ecoli",
        dataset_class="X",
        slug="x",
        section="Fitness",
        experiment_type="bacterial_fitness",
        label_name="fitness",
        measurement_type="None",
        shape="scalar",
        graph_role="global",
        built=True,
        probed=True,
        instances=1,
    )
    return build.DatasetProbeRecord(**{**base, **kwargs})


def test_a_natural_genome_has_no_sequence_anchor():
    tier, unit = build.sequence_fit(row(graph_role="global", natural_genome=True))
    assert tier == "none"
    assert "natural genome" in unit


def test_a_thousand_variant_genotype_counts_as_a_natural_genome():
    tier, _ = build.sequence_fit(row(graph_role="node", edits_mode=4993))
    assert tier == "none"


@pytest.mark.parametrize(
    ("role", "unit_word"), [("edge", "gene pair"), ("hyperedge", "gene triple")]
)
def test_an_interaction_is_a_best_fit_on_its_gene_tuple(role, unit_word):
    tier, unit = build.sequence_fit(
        row(graph_role=role, edits_mode=2 if role == "edge" else 3)
    )
    assert tier == "best"
    assert unit_word in unit


def test_a_single_deletion_node_label_is_a_best_fit():
    tier, unit = build.sequence_fit(
        row(graph_role="node", label_name="expression_log2_ratio", edits_mode=1)
    )
    assert tier == "best"
    assert "1 gene" in unit


def test_a_node_label_keeps_a_good_fit_in_a_multi_edit_strain():
    """Protein abundance in an engineered strain: the protein still has a sequence."""
    tier, unit = build.sequence_fit(
        row(
            graph_role="node",
            label_name="protein_abundance",
            edits_mode=8,
            edits_min=6,
            edits_max=9,
            perturbation_types=["heterologous_pathway", "bacterial_deletion"],
        )
    )
    assert tier == "good"
    assert "readout gene x 8 perturbed gene(s)" in unit
    assert "(6-9 edits)" in unit


def test_a_flux_label_names_the_reaction_not_a_metabolite():
    _, unit = build.sequence_fit(
        row(graph_role="bipartite node", label_name="net_flux", edits_mode=1)
    )
    assert unit.startswith("reaction x 1 perturbed gene(s)")


def test_a_metabolite_label_is_a_good_fit():
    tier, unit = build.sequence_fit(
        row(graph_role="bipartite node", label_name="metabolite_level", edits_mode=1)
    )
    assert tier == "good"
    assert unit.startswith("metabolite x 1 perturbed gene(s)")


def test_one_gene_across_conditions_drops_to_good_and_names_the_condition():
    tier, unit = build.sequence_fit(
        row(graph_role="global", edits_mode=1, environments=324)
    )
    assert tier == "good"
    assert unit == "1 gene x condition"


def test_one_gene_in_one_environment_is_a_best_fit():
    tier, unit = build.sequence_fit(
        row(graph_role="global", edits_mode=1, environments=1)
    )
    assert (tier, unit) == ("best", "1 gene")


def test_a_titer_of_a_multi_edit_strain_is_weak_only_through_its_heterologous_genes():
    tier, unit = build.sequence_fit(
        row(
            graph_role="global",
            label_name="titer",
            edits_mode=8,
            perturbation_types=["heterologous_pathway", "bacterial_deletion"],
        )
    )
    assert tier == "weak"
    assert "heterologous" in unit


def test_a_multi_edit_strain_with_only_native_edits_has_no_fit():
    tier, unit = build.sequence_fit(
        row(
            graph_role="global", edits_mode=5, perturbation_types=["bacterial_deletion"]
        )
    )
    assert tier == "none"
    assert "multi-edit strain, 5 edits" in unit


def test_an_unperturbed_strain_readout_has_nothing_to_attach_a_sequence_to():
    tier, _ = build.sequence_fit(row(graph_role="global", edits_mode=0))
    assert tier == "none"


def test_every_description_is_renderable_prose():
    descriptions = json.loads(DESCRIPTIONS.read_text())
    assert descriptions, "the descriptions file is empty"
    for name, text in descriptions.items():
        assert name.endswith("Dataset"), name
        assert 120 <= len(text) <= 480, f"{name}: {len(text)} characters"
        for banned in ("|", "—", "–", "`", "$", "\\"):
            assert banned not in text, f"{name} contains {banned!r}"
        assert text.strip() == text, name
        assert text.endswith("."), name
