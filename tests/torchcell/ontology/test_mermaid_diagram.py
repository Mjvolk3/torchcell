# tests/torchcell/ontology/test_mermaid_diagram.py
# [[tests.torchcell.ontology.test_mermaid_diagram]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/ontology/test_mermaid_diagram.py
"""``MermaidDiagramGenerator`` on a hand-written BioCypher schema, plus the real schema.

Fixture ``SCHEMA`` (written to ``tmp_path/schema.yaml``) has five nodes, two edges and
two keys the parser must skip (``stray``: null config; ``relation``: ``represented_as``
is neither node nor edge):

- ``experiment`` is_a ``information content entity``; ``fitness phenotype`` is_a
  ``phenotypic feature``; ``strain-genotype`` is_a ``genotype``; ``gene`` and
  ``genotype`` have no ``is_a`` (auto-mapped).
- Biolink classes = the ``is_a`` values = {genotype, information content entity,
  phenotypic feature}; ``genotype`` is therefore BOTH a Biolink class and an auto-mapped
  node, so it is declared twice and styled twice (Finding, same in the real schema).
- Node ids drop every non-word character and capitalize each whitespace word:
  ``strain-genotype`` -> ``Straingenotype`` (one word once the hyphen is gone).
- Edge ``experiment member of`` (is_a ``participates in``) has sources ``[gene,
  strain-genotype]`` and target ``experiment``: 2 x 1 = 2 dotted lines with the label
  ``experiment member of<br/>(is_a: participates in)``. Edge ``phenotype member of``
  (no is_a) has source ``fitness phenotype`` and target ``[experiment]``: one line with
  the bare label. The three data lines are sorted as whole strings, so they order by
  source id (F, G, S), not by edge name.

``EXPECTED_LR`` below is the full diagram, line by line, derived from those rules and the
fixed legend/styling block in the source (lines 214 to 259). The module is imported
inside a fixture that first ``chdir``s into ``tmp_path``, because importing
``torchcell.ontology`` imports biocypher, which writes a ``biocypher-log`` directory into
the working directory.

The real-schema test reads ``biocypher/config/torchcell_schema_config.yaml`` read-only
and pins its 123-line RL diagram (27 nodes, 13 edges, 40 data lines); a schema-config
edit changes these constants on purpose.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

import pytest
import yaml

if TYPE_CHECKING:
    from torchcell.ontology.mermaid_diagram import MermaidDiagramGenerator

REPO = Path(__file__).resolve().parents[3]
REAL_SCHEMA = REPO / "biocypher" / "config" / "torchcell_schema_config.yaml"

SCHEMA: dict[str, object] = {
    "gene": {"represented_as": "node", "preferred_id": "sgd"},
    "genotype": {"represented_as": "node"},
    "experiment": {"represented_as": "node", "is_a": "information content entity"},
    "fitness phenotype": {"represented_as": "node", "is_a": "phenotypic feature"},
    "strain-genotype": {"represented_as": "node", "is_a": "genotype"},
    "experiment member of": {
        "represented_as": "edge",
        "is_a": "participates in",
        "source": ["gene", "strain-genotype"],
        "target": "experiment",
    },
    "phenotype member of": {
        "represented_as": "edge",
        "source": "fitness phenotype",
        "target": ["experiment"],
    },
    "stray": None,
    "relation": {"represented_as": "relationship"},
}

LEGEND_AND_STYLE = [
    "",
    "    %% Legend",
    "    subgraph Legend",
    '        L1["Biolink Class"]',
    '        L2["Direct Biolink Usage"]',
    '        L3["Inherited Torchcell Entity"]',
    '        L4["→ solid = inheritance"]',
    '        L5["-.-> dotted = data relationship"]',
    "    end",
    "",
    "    %% Styling",
    "    classDef biolinkClassStyle fill:#e1f5ff,stroke:#0288d1,stroke-width:2px",
    "    classDef autoMappedStyle fill:#c8e6c9,stroke:#388e3c,stroke-width:2px",
    "    classDef torchcellEntityStyle fill:#fff3e0,stroke:#f57c00,stroke-width:2px",
]

EXPECTED_LR = [
    "graph LR",
    "",
    "    %% Biolink Classes (Parent Entity Types)",
    '    Genotype["genotype"]',
    '    InformationContentEntity["information content entity"]',
    '    PhenotypicFeature["phenotypic feature"]',
    "",
    "    %% Direct Biolink Usage (No Inheritance)",
    '    Gene["gene"]',
    '    Genotype["genotype"]',
    "",
    "    %% Torchcell Entities (Inherited from Biolink)",
    '    Experiment["experiment"]',
    '    FitnessPhenotype["fitness phenotype"]',
    '    Straingenotype["strain-genotype"]',
    "",
    "    %% Class Inheritance",
    "    Genotype -->|is_a| Straingenotype",
    "    InformationContentEntity -->|is_a| Experiment",
    "    PhenotypicFeature -->|is_a| FitnessPhenotype",
    "",
    "    %% Data Relationships",
    '    FitnessPhenotype -.->|"phenotype member of"| Experiment',
    '    Gene -.->|"experiment member of<br/>(is_a: participates in)"| Experiment',
    '    Straingenotype -.->|"experiment member of<br/>(is_a: participates in)"| Experiment',
    *LEGEND_AND_STYLE,
    "    class Genotype,InformationContentEntity,PhenotypicFeature,L1 biolinkClassStyle",
    "    class Gene,Genotype,L2 autoMappedStyle",
    "    class Experiment,FitnessPhenotype,Straingenotype,L3 torchcellEntityStyle",
]


@pytest.fixture
def md(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import the module from inside tmp_path so biocypher's log directory lands there."""
    monkeypatch.chdir(tmp_path)
    import torchcell.ontology.mermaid_diagram as module

    return module


def _write_schema(path: Path, schema: object) -> Path:
    path.write_text(yaml.safe_dump(schema, sort_keys=False))
    return path


@pytest.fixture
def generator(md: ModuleType, tmp_path: Path) -> MermaidDiagramGenerator:
    gen: MermaidDiagramGenerator = md.MermaidDiagramGenerator(
        str(_write_schema(tmp_path / "schema.yaml", SCHEMA))
    )
    return gen


def test_parsing_partitions_nodes_edges_classes_and_auto_mapped(
    generator: MermaidDiagramGenerator,
) -> None:
    """Null configs and non-node/edge entries are skipped; is_a values are the classes."""
    assert sorted(generator.nodes) == [
        "experiment",
        "fitness phenotype",
        "gene",
        "genotype",
        "strain-genotype",
    ]
    assert sorted(generator.edges) == ["experiment member of", "phenotype member of"]
    assert generator.biolink_classes == {
        "genotype",
        "information content entity",
        "phenotypic feature",
    }
    assert generator.auto_mapped_nodes == {"gene", "genotype"}


def test_generate_diagram_emits_every_line_in_order(
    generator: MermaidDiagramGenerator,
) -> None:
    """The full default-orientation diagram equals EXPECTED_LR (derivation in the header).

    Finding: ``genotype`` is both an ``is_a`` target and an auto-mapped node, so
    ``Genotype["genotype"]`` is declared in two sections and listed in two ``class``
    lines (lines 129 to 143 and 239 to 249 never check the overlap). Pinned until the
    overlap is deduplicated.
    """
    assert generator.generate_diagram().split("\n") == EXPECTED_LR


def test_orientation_only_changes_the_header_line(
    generator: MermaidDiagramGenerator,
) -> None:
    """Finding: the default orientation is "LR", while the docstring lists only "RL" and
    "BT", and any string is accepted unvalidated as the header. Pinned until the
    orientation is validated or the docstring names LR.
    """
    body = EXPECTED_LR[1:]
    assert generator.generate_diagram("BT").split("\n") == ["graph BT", *body]
    assert generator.generate_diagram("RL").split("\n") == ["graph RL", *body]
    assert generator.generate_diagram("sideways").split("\n") == [
        "graph sideways",
        *body,
    ]


def test_output_is_independent_of_yaml_key_order(
    md: ModuleType, tmp_path: Path
) -> None:
    """Every section is sorted, so reversing the YAML key order gives the same diagram
    (the module promises deterministic output for clean git diffs).
    """
    reversed_schema = dict(reversed(list(SCHEMA.items())))
    gen = md.MermaidDiagramGenerator(
        str(_write_schema(tmp_path / "reversed.yaml", reversed_schema))
    )
    assert gen.generate_diagram().split("\n") == EXPECTED_LR


def test_empty_mapping_emits_only_header_legend_and_styling(
    md: ModuleType, tmp_path: Path
) -> None:
    """``{}``: no section comments, no ``class`` style lines, just the fixed blocks."""
    gen = md.MermaidDiagramGenerator(str(_write_schema(tmp_path / "e.yaml", {})))
    assert gen.generate_diagram().split("\n") == ["graph LR", *LEGEND_AND_STYLE]


def test_empty_file_raises_attribute_error(md: ModuleType, tmp_path: Path) -> None:
    """Finding: an empty YAML file loads as ``None`` and ``__init__`` calls
    ``None.items()`` (line 47), so the constructor raises instead of giving an empty
    diagram. Pinned until an empty schema is handled or rejected with a named error.
    """
    path = tmp_path / "blank.yaml"
    path.write_text("")
    with pytest.raises(
        AttributeError, match=r"^'NoneType' object has no attribute 'items'$"
    ):
        md.MermaidDiagramGenerator(str(path))


def test_edge_without_endpoints_emits_no_data_section(
    md: ModuleType, tmp_path: Path
) -> None:
    """An edge with no ``source``/``target`` defaults to empty lists, so no line and no
    ``%% Data Relationships`` header; a node with only is_a still gets its class line.
    """
    schema = {
        "widget": {"represented_as": "node", "is_a": "thing"},
        "dangling": {"represented_as": "edge", "is_a": "related to"},
    }
    gen = md.MermaidDiagramGenerator(str(_write_schema(tmp_path / "d.yaml", schema)))
    assert gen.generate_diagram("BT").split("\n") == [
        "graph BT",
        "",
        "    %% Biolink Classes (Parent Entity Types)",
        '    Thing["thing"]',
        "",
        "    %% Torchcell Entities (Inherited from Biolink)",
        '    Widget["widget"]',
        "",
        "    %% Class Inheritance",
        "    Thing -->|is_a| Widget",
        *LEGEND_AND_STYLE,
        "    class Thing,L1 biolinkClassStyle",
        "    class Widget,L3 torchcellEntityStyle",
    ]


@pytest.mark.parametrize(
    ("name", "node_id"),
    [
        ("fitness phenotype", "FitnessPhenotype"),
        ("FITNESS phenotype", "FitnessPhenotype"),
        ("strain-genotype", "Straingenotype"),
        ("gene_set member", "Gene_setMember"),
        ("2micron plasmid", "2micronPlasmid"),
        ("a (b) c", "ABC"),
        ('say "hi"', "SayHi"),
    ],
)
def test_format_node_id(
    generator: MermaidDiagramGenerator, name: str, node_id: str
) -> None:
    """Non-word characters are deleted (underscores and digits are word characters), then
    each whitespace word is ``str.capitalize``d, which also lowercases the rest of a word;
    so ``FITNESS phenotype`` collides with ``fitness phenotype``.
    """
    assert generator._format_node_id(name) == node_id


def test_labels_are_not_escaped(md: ModuleType, tmp_path: Path) -> None:
    """Finding: ``_format_node_label`` returns the name verbatim (line 110) and edge
    labels are interpolated raw (line 200), so a double quote in a name produces a
    Mermaid label with unbalanced quotes. Pinned until labels are escaped.
    """
    schema = {
        'say "hi"': {"represented_as": "node"},
        'e "x"': {"represented_as": "edge", "source": 'say "hi"', "target": 'say "hi"'},
    }
    gen = md.MermaidDiagramGenerator(str(_write_schema(tmp_path / "q.yaml", schema)))
    lines = gen.generate_diagram().split("\n")
    assert lines[1:4] == [
        "",
        "    %% Direct Biolink Usage (No Inheritance)",
        '    SayHi["say "hi""]',
    ]
    assert lines[4:7] == [
        "",
        "    %% Data Relationships",
        '    SayHi -.->|"e "x""| SayHi',
    ]


def test_extract_frontmatter_and_mermaid_content(
    generator: MermaidDiagramGenerator,
) -> None:
    """Frontmatter must start at offset 0; the mermaid block is the first fenced block,
    stripped; no fence gives "".
    """
    content = (
        "---\nid: x\n---\nintro\n```mermaid\n  graph LR\n  A\n```\n```mermaid\nB\n```\n"
    )
    assert generator._extract_frontmatter(content) == (
        "---\nid: x\n---\n",
        "intro\n```mermaid\n  graph LR\n  A\n```\n```mermaid\nB\n```\n",
    )
    assert generator._extract_frontmatter("\n" + content) == ("", "\n" + content)
    assert generator._extract_mermaid_content(content) == "graph LR\n  A"
    assert generator._extract_mermaid_content("no diagram here") == ""


def test_has_changed(generator: MermaidDiagramGenerator, tmp_path: Path) -> None:
    """Missing file -> True; same diagram up to surrounding whitespace -> False; a
    different diagram or a file without a mermaid block -> True.
    """
    path = tmp_path / "note.md"
    assert generator.has_changed("graph LR", str(path)) is True
    path.write_text("---\nid: n\n---\n```mermaid\n\ngraph LR\n  A\n\n```\n")
    assert generator.has_changed("  graph LR\n  A  \n", str(path)) is False
    assert generator.has_changed("graph BT\n  A", str(path)) is True
    path.write_text("---\nid: n\n---\nplain text\n")
    assert generator.has_changed("graph LR", str(path)) is True


def test_write_diagram_creates_file_with_minimal_frontmatter(
    generator: MermaidDiagramGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A new path in a missing directory: parents are created, the id is the stem with
    dots removed, the title is ``stem.replace('.', ' ').title()`` (``str.title`` also
    capitalizes after the underscore), then a blank line and the fenced diagram.
    """
    path = tmp_path / "out" / "nested" / "my.schema_diagram.lr.md"
    assert generator.write_diagram(str(path), "LR") is True
    assert path.read_text() == (
        "---\n"
        "id: myschema_diagramlr\n"
        "title: My Schema_Diagram Lr\n"
        "desc: 'BioCypher Schema Diagram'\n"
        "---\n\n"
        "```mermaid\n" + "\n".join(EXPECTED_LR) + "\n```\n"
    )
    assert capsys.readouterr().out == f"✓ Updated: {path}\n"


def test_write_diagram_is_a_no_op_when_unchanged(
    generator: MermaidDiagramGenerator,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Second write with the same orientation returns False and leaves the bytes alone."""
    path = tmp_path / "d.md"
    generator.write_diagram(str(path), "RL")
    before = path.read_text()
    capsys.readouterr()
    assert generator.write_diagram(str(path), "RL") is False
    assert path.read_text() == before
    assert capsys.readouterr().out == f"✓ No changes: {path}\n"


def test_rewrite_keeps_frontmatter_but_drops_the_blank_line(
    generator: MermaidDiagramGenerator, tmp_path: Path
) -> None:
    """Finding: a fresh file has a blank line after the closing ``---`` (line 354), but
    ``_extract_frontmatter`` keeps only through the closing delimiter line (line 273),
    so the first rewrite of an existing note removes that blank line; custom
    frontmatter keys are preserved. The committed
    ``notes/torchcell.ontology.mermaid_diagram.*.md`` files show exactly this layout.
    Pinned until the separator is normalized.
    """
    path = tmp_path / "d.md"
    path.write_text(
        "---\nid: keep-me\ntitle: Custom\n---\n\nold prose\n```mermaid\ngraph LR\n```\n"
    )
    assert generator.write_diagram(str(path), "BT") is True
    assert path.read_text() == (
        "---\nid: keep-me\ntitle: Custom\n---\n```mermaid\n"
        + "\n".join(["graph BT", *EXPECTED_LR[1:]])
        + "\n```\n"
    )


def test_write_over_file_without_frontmatter_replaces_everything(
    generator: MermaidDiagramGenerator, tmp_path: Path
) -> None:
    """No frontmatter in the existing file: the minimal one is generated and the old body
    (prose and all) is discarded.
    """
    path = tmp_path / "plain.md"
    path.write_text("some hand-written prose\n")
    assert generator.write_diagram(str(path), "LR") is True
    assert path.read_text() == (
        "---\nid: plain\ntitle: Plain\ndesc: 'BioCypher Schema Diagram'\n---\n\n"
        "```mermaid\n" + "\n".join(EXPECTED_LR) + "\n```\n"
    )


def test_main_writes_both_orientations_then_reports_no_changes(
    md: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main`` resolves the project root two levels above the module file; with
    ``__file__`` pointed into tmp_path it reads ``biocypher/config/...yaml`` there and
    writes the RL and BT notes under ``notes/``. The second run changes nothing.
    """
    root = tmp_path / "project"
    config = root / "biocypher" / "config"
    config.mkdir(parents=True)
    _write_schema(config / "torchcell_schema_config.yaml", SCHEMA)
    monkeypatch.setattr(
        md, "__file__", str(root / "torchcell" / "ontology" / "mermaid_diagram.py")
    )
    horizontal = root / "notes" / "torchcell.ontology.mermaid_diagram.horizontal.md"
    vertical = root / "notes" / "torchcell.ontology.mermaid_diagram.vertical.md"

    md.main()
    assert capsys.readouterr().out == (
        f"✓ Updated: {horizontal}\n✓ Updated: {vertical}\n\n✓ Diagram generation complete\n"
    )
    assert horizontal.read_text() == (
        "---\nid: torchcellontologymermaid_diagramhorizontal\n"
        "title: Torchcell Ontology Mermaid_Diagram Horizontal\n"
        "desc: 'BioCypher Schema Diagram'\n---\n\n```mermaid\n"
        + "\n".join(["graph RL", *EXPECTED_LR[1:]])
        + "\n```\n"
    )
    assert vertical.read_text() == (
        "---\nid: torchcellontologymermaid_diagramvertical\n"
        "title: Torchcell Ontology Mermaid_Diagram Vertical\n"
        "desc: 'BioCypher Schema Diagram'\n---\n\n```mermaid\n"
        + "\n".join(["graph BT", *EXPECTED_LR[1:]])
        + "\n```\n"
    )

    md.main()
    assert capsys.readouterr().out == (
        f"✓ No changes: {horizontal}\n✓ No changes: {vertical}\n\n"
        "✓ No changes in ontology since last update\n"
    )


def test_real_schema_diagram(md: ModuleType) -> None:
    """The real config (read-only): 32 nodes (4 auto-mapped: dataset, genome, genotype,
    publication; 28 inherited), 6 Biolink classes, 13 edges expanding to 49 data lines.
    ``interned constant`` (tcdb-002) is a third ``information content entity`` beside
    experiment and experiment reference; the bacterial program added four inherited
    nodes (``bacterial perturbation`` under genotype, and the product titer, protein
    turnover and flux phenotypes) and eight data lines (one more ``perturbation member
    of`` source, one more ``crispr construct member of`` target, and three phenotype
    sources times the two ``phenotype member of`` targets). Its follow-up added
    ``phage perturbation`` under a SIXTH Biolink class, ``biotic exposure`` (Biolink
    defines ``environmental exposure`` as abiotic, so a virion does not belong there),
    and one data line (a second ``environment perturbation member of`` source).
    Line count 143 = 1 header + (2 + 6) + (2 + 4) + (2 + 28) + (2 + 28 is_a lines)
    + (2 + 49) + 9 legend + 5 styling + 3 class lines. ``Genotype`` is declared on
    lines 5 and 14 (Biolink class and auto-mapped node, the duplicate Finding), and
    the list-valued ``source`` of ``genotype member of`` expands to two lines.
    """
    gen = md.MermaidDiagramGenerator(str(REAL_SCHEMA))
    lines = gen.generate_diagram("RL").split("\n")
    raw = yaml.safe_load(REAL_SCHEMA.read_text())
    n_edges = 0
    n_data_lines = 0
    for config in raw.values():
        if config and config.get("represented_as") == "edge":
            n_edges += 1
            src, tgt = config["source"], config["target"]
            n_data_lines += (1 if isinstance(src, str) else len(src)) * (
                1 if isinstance(tgt, str) else len(tgt)
            )
    assert (len(gen.nodes), n_edges, n_data_lines) == (33, 13, 51)
    assert len(lines) == 147
    assert lines[2:9] == [
        "    %% Biolink Classes (Parent Entity Types)",
        '    BioticExposure["biotic exposure"]',
        '    EnvironmentalExposure["environmental exposure"]',
        '    Genotype["genotype"]',
        '    InformationContentEntity["information content entity"]',
        '    NucleicAcidEntity["nucleic acid entity"]',
        '    PhenotypicFeature["phenotypic feature"]',
    ]
    assert lines[10:15] == [
        "    %% Direct Biolink Usage (No Inheritance)",
        '    Dataset["dataset"]',
        '    Genome["genome"]',
        '    Genotype["genotype"]',
        '    Publication["publication"]',
    ]
    assert lines.count('    Genotype["genotype"]') == 2
    assert [line for line in lines if "InternedConstant" in line][:2] == [
        '    InternedConstant["interned constant"]',
        "    InformationContentEntity -->|is_a| InternedConstant",
    ]
    assert "    Genotype -->|is_a| SegregantGenotype" in lines
    assert "    Genotype -->|is_a| BacterialPerturbation" in lines
    # the phage sits under biotic exposure, NOT under environmental exposure
    assert "    BioticExposure -->|is_a| PhagePerturbation" in lines
    assert "    EnvironmentalExposure -->|is_a| PhagePerturbation" not in lines
    assert (
        '    BacterialPerturbation -.->|"perturbation member of<br/>(is_a: genetically associated with)"| Genotype'
        in lines
    )
    assert (
        '    Perturbation -.->|"perturbation member of<br/>(is_a: genetically associated with)"| Genotype'
        in lines
    )
    assert [line for line in lines if "genotype member of" in line] == [
        '    Genotype -.->|"genotype member of<br/>(is_a: participates in)"| Experiment',
        '    SegregantGenotype -.->|"genotype member of<br/>(is_a: participates in)"| Experiment',
    ]
    assert (
        lines[-2] == "    class Dataset,Genome,Genotype,Publication,L2 autoMappedStyle"
    )
