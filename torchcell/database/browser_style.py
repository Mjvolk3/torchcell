# torchcell/database/browser_style.py
# [[torchcell.database.browser_style]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/database/browser_style
"""Neo4j Browser graph stylesheet (GRASS) for the served torchcell graph.

Every node carries its Biolink ancestor labels beside its own class label (an
``Experiment`` is also ``InformationContentEntity``, ``Entity``, ``NamedThing``). The Neo4j
Browser that Neo4j 5.26 serves keeps a label priority list and colors a multi-label node
by its highest-priority label; the list starts in the order the labels were first listed
from the schema, so the ancestor label that came first, ``NamedThing``, painted the whole
graph one color after the 2026.09.17 swap.

Importing a GraSS file rewrites that list: the importer prepends each ``node.<Label>``
rule as it reads the file, so the LAST rule in the file gets the highest priority (read
from the served bundle: the ``stylingPrecedence`` reducer and the ``highestPriorityLabel``
reduction over ``node.labels``; the Browser's own export writes the highest-priority label
last, the same convention). The classic Browser merges matching rules top to bottom with
later properties overriding, so the same order serves both.

The stylesheet written here therefore puts the ancestor labels first, in a receding gray,
and the class labels last, colored by the ontology lane the paper figures use
(:data:`torchcell.paper.ontology_graph.LANE_PALETTE_INDEX` over
:data:`torchcell.utils.PLOT_PALETTE`), so the browser and the figures read the same way:
amber for genotype, brick for environment, wheat for experiment, lilac for phenotype, steel
blue for provenance. The served Browser reads ``color``, ``caption``, ``diameter`` and
``shaft-width`` from a GraSS file and derives border and text colors from ``color``; the
other properties are kept for the classic Browser.

    python -m torchcell.database.browser_style           # writes database/conf/torchcell.grass
    python -m torchcell.database.browser_style --check   # exit 1 when that file is stale

Load it from the graph result's styling panel, "Upload GraSS styles" (the file input
accepts ``.grass``, ``.style``, ``.txt``); "Reset styles to default" undoes it. The
stylesheet lives in that browser's local storage, so each person loads it once.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from torchcell.paper.ontology_graph import LANE_PALETTE_INDEX
from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_CONFIG = REPO_ROOT / "biocypher" / "config" / "torchcell_schema_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "database" / "conf" / "torchcell.grass"

# Biolink 3.2.1 ancestor labels the served nodes carry beside their class label
# (``CALL db.labels()`` on the store served 2026-09-18). Their rules go FIRST so that any
# class rule below overrides them.
ANCESTOR_LABELS: tuple[str, ...] = (
    "NamedThing",
    "Entity",
    "BiologicalEntity",
    "InformationContentEntity",
    "EnvironmentalExposure",
    "Attribute",
    "PhenotypicFeature",
    "DiseaseOrPhenotypicFeature",
    "ChemicalEntity",
    "MolecularEntity",
    "NucleicAcidEntity",
)

# Graph label -> ontology lane, for the classes that are not phenotypes. Phenotype labels
# are read from the schema config (every node class whose ``is_a`` is ``phenotypic
# feature``), so a new phenotype class gets its lane without an edit here.
LANE_OF_LABEL: dict[str, str] = {
    "Genotype": "genotype",
    "Perturbation": "genotype",
    "SegregantGenotype": "genotype",
    "CrisprConstruct": "genotype",
    "Environment": "environment",
    "Media": "environment",
    "Temperature": "environment",
    "EnvironmentPerturbation": "environment",
    "Experiment": "experiment",
    "ExperimentReference": "experiment",
    "Dataset": "provenance",
    "Publication": "provenance",
    "Genome": "provenance",
    "Schema_info": "enum",  # BioCypher's own bookkeeping node
}

# The property shown inside the circle. Everything else falls back to the node id.
CAPTION_OF_LABEL: dict[str, str] = {
    "Genotype": "{perturbed_gene_name}",
    "Perturbation": "{perturbed_gene_name}",
    "SegregantGenotype": "{segregant_id}",
    "CrisprConstruct": "{effector}",
    "Environment": "{media}",
    "Media": "{name}",
    "Temperature": "{value}",
    "EnvironmentPerturbation": "{compound_name}",
    "Publication": "{pubmed_id}",
    "Genome": "{strain}",
}

# Hubs draw larger so the record structure (experiment at the center, its genotype,
# environment and phenotype around it) is readable at a glance.
DIAMETER_OF_LABEL: dict[str, int] = {
    "Experiment": 65,
    "ExperimentReference": 55,
    "Dataset": 65,
}
CLASS_DIAMETER_PX = 50
ANCESTOR_DIAMETER_PX = 35


class NodeRule(BaseModel):
    """One ``node.<Label> { ... }`` block."""

    label: str
    fill: str = Field(pattern=r"^#[0-9A-F]{6}$")
    border: str = Field(pattern=r"^#[0-9A-F]{6}$")
    text: str = Field(default="#000000", pattern=r"^#[0-9A-F]{6}$")
    diameter_px: int = CLASS_DIAMETER_PX
    caption: str = "{id}"

    def render(self) -> str:
        """The GRASS block for this label."""
        return (
            f"node.{self.label} {{\n"
            f"  color: {self.fill};\n"
            f"  border-color: {self.border};\n"
            f"  border-width: 2px;\n"
            f"  text-color-internal: {self.text};\n"
            f"  diameter: {self.diameter_px}px;\n"
            f'  caption: "{self.caption}";\n'
            f"}}\n"
        )


class RelationshipRule(BaseModel):
    """The single ``relationship { ... }`` block: every edge is a membership edge."""

    color: str = Field(default=PLOT_PALETTE[5], pattern=r"^#[0-9A-F]{6}$")
    shaft_width_px: int = 1
    font_size_px: int = 8

    def render(self) -> str:
        """The GRASS block for every relationship type."""
        return (
            "relationship {\n"
            f"  color: {self.color};\n"
            f"  shaft-width: {self.shaft_width_px}px;\n"
            f"  font-size: {self.font_size_px}px;\n"
            "  padding: 3px;\n"
            "  text-color-external: #000000;\n"
            "  text-color-internal: #FFFFFF;\n"
            '  caption: "<type>";\n'
            "}\n"
        )


def _camel(schema_key: str) -> str:
    """``gene interaction phenotype`` -> ``GeneInteractionPhenotype`` (BioCypher labels)."""
    return "".join(word.capitalize() for word in schema_key.split())


def schema_node_labels(schema_config: Path = SCHEMA_CONFIG) -> dict[str, str | None]:
    """Graph label -> ``is_a`` for every node class in the BioCypher schema config."""
    config = yaml.safe_load(schema_config.read_text(encoding="utf-8"))
    return {
        _camel(key): body.get("is_a")
        for key, body in config.items()
        if isinstance(body, dict) and body.get("represented_as") == "node"
    }


def lane_of(label: str, is_a: str | None) -> str:
    """The ontology lane of a graph label; phenotypes come from their ``is_a``."""
    if is_a == "phenotypic feature":
        return "phenotype"
    if label in LANE_OF_LABEL:
        return LANE_OF_LABEL[label]
    raise KeyError(
        f"{label!r} (is_a {is_a!r}) has no ontology lane: add it to LANE_OF_LABEL in "
        f"{__name__} or give it a phenotypic feature parent in the schema config"
    )


def node_rules(schema_config: Path = SCHEMA_CONFIG) -> list[NodeRule]:
    """Ancestor rules first (gray, small), then one colored rule per schema node class."""
    rules = [
        NodeRule(
            label=label,
            fill=PLOT_PALETTE_FILL[5],
            border=PLOT_PALETTE_FILL[11],
            text=PLOT_PALETTE[5],
            diameter_px=ANCESTOR_DIAMETER_PX,
            caption="{id}",
        )
        for label in ANCESTOR_LABELS
    ]
    labels = schema_node_labels(schema_config)
    labels.setdefault("Schema_info", None)
    for label in sorted(labels):
        index = LANE_PALETTE_INDEX[lane_of(label, labels[label])]
        rules.append(
            NodeRule(
                label=label,
                fill=PLOT_PALETTE_FILL[index],
                border=PLOT_PALETTE[index],
                diameter_px=DIAMETER_OF_LABEL.get(label, CLASS_DIAMETER_PX),
                caption=CAPTION_OF_LABEL.get(label, "{id}"),
            )
        )
    return rules


def render(schema_config: Path = SCHEMA_CONFIG) -> str:
    """The complete stylesheet text."""
    header = (
        "/* torchcell Neo4j Browser stylesheet. Generated by\n"
        " * python -m torchcell.database.browser_style; do not edit by hand.\n"
        " * Ancestor labels first so the class rules below win on multi-label nodes;\n"
        " * class colors are the ontology lanes of the paper figures. */\n"
    )
    base = (
        "node {\n"
        f"  color: {PLOT_PALETTE_FILL[5]};\n"
        f"  border-color: {PLOT_PALETTE[5]};\n"
        "  border-width: 2px;\n"
        "  text-color-internal: #000000;\n"
        f"  diameter: {CLASS_DIAMETER_PX}px;\n"
        "  font-size: 10px;\n"
        "}\n"
    )
    blocks = [header, base, RelationshipRule().render()]
    blocks.extend(rule.render() for rule in node_rules(schema_config))
    return "\n".join(blocks)


def main(argv: list[str] | None = None) -> int:
    """Write the stylesheet, or with ``--check`` report whether the written one is current."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.database.browser_style",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if --output differs from the render",
    )
    args = parser.parse_args(argv)
    text = render()
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != text:
            print(
                f"{args.output} is stale; run python -m torchcell.database.browser_style"
            )
            return 1
        print(f"{args.output} is current")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    n_rules = len(node_rules())
    print(f"wrote {args.output} ({n_rules} node rules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
