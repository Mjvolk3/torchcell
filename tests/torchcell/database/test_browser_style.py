"""The committed Neo4j Browser stylesheet covers every schema node class, keeps the
ancestor rules ahead of the class rules, and matches its generator; the local-storage
seed holds what the Browser's own GraSS importer would store for it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from torchcell.database.browser_style import (
    ANCESTOR_LABELS,
    DEFAULT_OUTPUT,
    DEFAULT_SEED_OUTPUT,
    DIAMETER_TO_SIZE,
    LANE_OF_LABEL,
    SEED_SHA_STORAGE_KEY,
    STYLING_STORAGE_KEY,
    lane_of,
    node_rules,
    persisted_value,
    render,
    render_seed_js,
    schema_node_labels,
    seed_state,
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


def test_seed_state_is_what_the_importer_would_store() -> None:
    rules = node_rules()
    state = seed_state()
    # the importer prepends each rule, so the file's last rule has the top priority
    assert state.stylingPriorityOrder == [rule.label for rule in reversed(rules)]
    assert state.stylingPriorityOrder[-1] == "NamedThing"
    assert state.relStyles == {}
    by_label = {rule.label: rule for rule in rules}
    for label, style in state.nodeStyles.items():
        rule = by_label[label]
        assert style.color == rule.fill
        assert style.size == math.floor(rule.diameter_px / DIAMETER_TO_SIZE)
        assert style.captions[0].type == "property"
        assert "{" + str(style.captions[0].captionKey) + "}" == rule.caption
    assert state.nodeStyles["Experiment"].size == 34  # 65 px
    assert state.nodeStyles["Genotype"].captions[0].captionKey == "perturbed_gene_name"


def test_persisted_value_has_the_redux_persist_shape() -> None:
    outer = json.loads(persisted_value(seed_state()))
    assert set(outer) == {"nodeStyles", "relStyles", "stylingPriorityOrder", "_persist"}
    assert json.loads(outer["_persist"]) == {"version": 1, "rehydrated": True}
    node_styles = json.loads(outer["nodeStyles"])
    assert node_styles["Genotype"] == {
        "color": "#FFE6CC",
        "size": 26,
        "captions": [{"type": "property", "captionKey": "perturbed_gene_name"}],
    }
    assert (
        json.loads(outer["stylingPriorityOrder"]) == seed_state().stylingPriorityOrder
    )


def test_seed_js_carries_the_keys_the_value_and_the_stylesheet_sha() -> None:
    text = render_seed_js()
    assert json.dumps(STYLING_STORAGE_KEY) in text
    assert json.dumps(SEED_SHA_STORAGE_KEY) in text
    assert hashlib.sha256(render().encode("utf-8")).hexdigest() in text
    assert json.dumps(persisted_value(seed_state())) in text
    assert text.count("localStorage.setItem") == 2


def test_committed_seed_is_current() -> None:
    assert DEFAULT_SEED_OUTPUT.is_file(), (
        "run python -m torchcell.database.browser_style"
    )
    assert DEFAULT_SEED_OUTPUT.read_text(encoding="utf-8") == render_seed_js()
