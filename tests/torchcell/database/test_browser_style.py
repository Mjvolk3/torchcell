"""The committed Neo4j Browser stylesheet covers every schema node class, keeps the
ancestor rules ahead of the class rules, and matches its generator; the local-storage
seed holds what the Browser's own GraSS importer would store for it.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

from torchcell.database.browser_style import (
    ANCESTOR_LABELS,
    DEFAULT_OUTPUT,
    DEFAULT_SEED_OUTPUT,
    DIAMETER_TO_SIZE,
    LANE_OF_LABEL,
    SEED_SHA_STORAGE_KEY,
    STYLING_STORAGE_KEY,
    Caption,
    _caption,
    lane_of,
    main,
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
        ("InternedConstant", "experiment"),
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


# Phase 24: main() writes and checks, lane_of and _caption guards


def test_main_writes_both_renders_and_names_the_rule_count(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``main`` writes ``render()`` and ``render_seed_js()`` verbatim, creating parents."""
    out = tmp_path / "conf" / "torchcell.grass"
    seed = tmp_path / "browser" / "seed.js"
    code = main(["--output", str(out), "--seed-output", str(seed)])
    assert code == 0
    assert out.read_text(encoding="utf-8") == render()
    assert seed.read_text(encoding="utf-8") == render_seed_js()
    assert capsys.readouterr().out == (
        f"wrote {out} ({len(node_rules())} node rules) and {seed}\n"
    )


def test_main_check_reports_current_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--check`` on freshly written files returns 0 and prints the current line."""
    out = tmp_path / "torchcell.grass"
    seed = tmp_path / "seed.js"
    out.write_text(render(), encoding="utf-8")
    seed.write_text(render_seed_js(), encoding="utf-8")
    code = main(["--check", "--output", str(out), "--seed-output", str(seed)])
    assert code == 0
    assert capsys.readouterr().out == f"{out} and {seed} are current\n"


def test_main_check_lists_a_missing_and_an_edited_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """One altered seed and one missing stylesheet: both are stale, in output order."""
    out = tmp_path / "missing.grass"
    seed = tmp_path / "seed.js"
    seed.write_text(render_seed_js() + "\n", encoding="utf-8")
    code = main(["--check", "--output", str(out), "--seed-output", str(seed)])
    assert code == 1
    assert capsys.readouterr().out == (
        f"{out} is stale; run python -m torchcell.database.browser_style\n"
        f"{seed} is stale; run python -m torchcell.database.browser_style\n"
    )
    assert not out.exists()


def test_main_check_with_one_current_file_lists_one_stale_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A current stylesheet beside an edited seed prints only the seed path."""
    out = tmp_path / "torchcell.grass"
    seed = tmp_path / "seed.js"
    out.write_text(render(), encoding="utf-8")
    seed.write_text("edited", encoding="utf-8")
    assert main(["--check", "--output", str(out), "--seed-output", str(seed)]) == 1
    assert capsys.readouterr().out == (
        f"{seed} is stale; run python -m torchcell.database.browser_style\n"
    )


def test_lane_of_unknown_label_raises_the_exact_key_error() -> None:
    with pytest.raises(KeyError) as exc:
        lane_of("Foo", None)
    assert exc.value.args[0] == (
        "'Foo' (is_a None) has no ontology lane: add it to LANE_OF_LABEL in "
        "torchcell.database.browser_style or give it a phenotypic feature parent "
        "in the schema config"
    )


def test_caption_id_and_type_and_property_forms() -> None:
    assert _caption("<id>") == Caption(type="id")
    assert _caption("<type>") == Caption(type="type")
    assert _caption("{name}") == Caption(type="property", captionKey="name")


def test_caption_rejects_a_bare_word() -> None:
    with pytest.raises(
        ValueError, match=re.escape("caption 'id' is not one the Browser imports")
    ):
        _caption("id")
