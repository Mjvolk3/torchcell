"""The committed Neo4j Browser stylesheet covers every schema node class, keeps the
ancestor rules ahead of the class rules, and matches its generator.
"""

from __future__ import annotations

import re

from torchcell.database.browser_style import (
    ANCESTOR_LABELS,
    DEFAULT_OUTPUT,
    LANE_OF_LABEL,
    lane_of,
    node_rules,
    render,
    schema_node_labels,
)
from torchcell.paper.ontology_graph import LANE_PALETTE_INDEX
from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL


def test_every_schema_node_class_has_a_lane_and_a_rule() -> None:
    labels = schema_node_labels()
    assert "Experiment" in labels and "FitnessPhenotype" in labels
    rule_labels = {rule.label for rule in node_rules()}
    for label, is_a in labels.items():
        assert lane_of(label, is_a) in LANE_PALETTE_INDEX
        assert label in rule_labels


def test_phenotypes_take_the_phenotype_lane_from_the_schema() -> None:
    labels = schema_node_labels()
    phenotypes = [
        label for label, is_a in labels.items() if is_a == "phenotypic feature"
    ]
    assert len(phenotypes) >= 12
    for label in phenotypes:
        assert label not in LANE_OF_LABEL
        assert lane_of(label, "phenotypic feature") == "phenotype"


def test_ancestor_rules_precede_class_rules() -> None:
    labels = [rule.label for rule in node_rules()]
    assert tuple(labels[: len(ANCESTOR_LABELS)]) == ANCESTOR_LABELS
    assert "NamedThing" in labels[: len(ANCESTOR_LABELS)]
    assert set(labels[len(ANCESTOR_LABELS) :]).isdisjoint(ANCESTOR_LABELS)


def test_class_colors_are_the_ontology_lane_colors() -> None:
    by_label = {rule.label: rule for rule in node_rules()}
    for label, lane in [
        ("Experiment", "experiment"),
        ("Genotype", "genotype"),
        ("Media", "environment"),
        ("FitnessPhenotype", "phenotype"),
        ("Publication", "provenance"),
    ]:
        index = LANE_PALETTE_INDEX[lane]
        assert by_label[label].fill == PLOT_PALETTE_FILL[index]
        assert by_label[label].border == PLOT_PALETTE[index]


def test_rendered_text_is_valid_grass_blocks() -> None:
    text = render()
    blocks = re.findall(
        r"^(node(?:\.\w+)?|relationship) \{\n(?:  [\w-]+: [^\n]+;\n)+\}$", text, re.M
    )
    assert blocks[:2] == ["node", "relationship"]
    assert text.count("node.") == len(node_rules())
    assert text.index("node.NamedThing") < text.index("node.Experiment")


def test_committed_stylesheet_is_current() -> None:
    assert DEFAULT_OUTPUT.is_file(), "run python -m torchcell.database.browser_style"
    assert DEFAULT_OUTPUT.read_text(encoding="utf-8") == render()
