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

    python -m torchcell.database.browser_style           # writes the stylesheet and the seed
    python -m torchcell.database.browser_style --check   # exit 1 when either file is stale

Two files come out of one render:

``database/conf/torchcell.grass``
    The GraSS stylesheet. Load it by hand from the graph result's styling panel,
    "Upload GraSS styles" (the file input accepts ``.grass``, ``.style``, ``.txt``);
    "Reset styles to default" undoes it. The stylesheet lives in that browser's local
    storage, so each person loads it once.

``database/browser/torchcell-seed.js``
    The same styling as the Browser stores it after an upload, wrapped in a script that
    writes it into local storage before the Browser boots. The Browser keeps its styling
    in a redux-persist slice keyed ``graphStyling`` under the ``nx.v1.nx.`` prefix
    (``W9(rI, "graphStyling", qF)`` in the served ``src.*.js``; whitelist ``nodeStyles``,
    ``relStyles``, ``stylingPriorityOrder``, version 1) and rehydrates it on load, so a
    value planted there first is what every visitor sees. The image built by
    ``database/docker/Dockerfile.tc-neo4j-browser`` adds this file to the Browser jar and
    a ``<script>`` tag for it to ``browser/index.html`` (the page's CSP allows scripts
    from the server itself and nothing inline). The seed runs once per browser per
    stylesheet: it records the stylesheet's sha256 and steps aside while that matches,
    so a person's own restyling survives until the stylesheet changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from torchcell.paper.ontology_graph import LANE_PALETTE_INDEX
from torchcell.utils import PLOT_PALETTE, PLOT_PALETTE_FILL

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_CONFIG = REPO_ROOT / "biocypher" / "config" / "torchcell_schema_config.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "database" / "conf" / "torchcell.grass"
DEFAULT_SEED_OUTPUT = REPO_ROOT / "database" / "browser" / "torchcell-seed.js"

# Local-storage keys of the served Browser: redux-persist key "graphStyling" through the
# framework-scoped storage (createKey(framework, key) -> "nx.v1.nx.<key>", keyPrefix "").
STYLING_STORAGE_KEY = "nx.v1.nx.graphStyling"
SEED_SHA_STORAGE_KEY = "nx.v1.nx.torchcellGrassSha256"
STYLING_PERSIST_VERSION = 1
# The Browser's GraSS importer sizes a node as floor(diameter / 2 / 0.93).
DIAMETER_TO_SIZE = 2 * 0.93

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


class Caption(BaseModel):
    """One caption entry as the Browser stores it (``Z9`` in the served bundle)."""

    type: Literal["id", "type", "property"]
    captionKey: str | None = None  # noqa: N815  # the Browser's own field name


class SeedNodeStyle(BaseModel):
    """What the Browser stores for one label after importing a ``node.<Label>`` rule."""

    color: str = Field(pattern=r"^#[0-9A-F]{6}$")
    size: int
    captions: list[Caption]


class SeedState(BaseModel):
    """The persisted ``graphStyling`` slice: the whitelist of its redux-persist config."""

    nodeStyles: dict[str, SeedNodeStyle]  # noqa: N815
    relStyles: dict[str, dict[str, object]]  # noqa: N815
    stylingPriorityOrder: list[str]  # noqa: N815


def _caption(caption: str) -> Caption:
    """The GraSS caption string as the importer reads it."""
    if caption == "<id>":
        return Caption(type="id")
    if caption == "<type>":
        return Caption(type="type")
    if caption.startswith("{") and caption.endswith("}"):
        return Caption(type="property", captionKey=caption[1:-1])
    raise ValueError(f"caption {caption!r} is not one the Browser imports")


def seed_state(schema_config: Path = SCHEMA_CONFIG) -> SeedState:
    """The styling the Browser would hold after "Upload GraSS styles" of :func:`render`.

    The importer (``phe``) prepends each ``node.<Label>`` rule to the priority list, so
    the last rule in the file is first in the list; the bare ``node`` and ``relationship``
    blocks carry no label and are skipped, so ``relStyles`` stays empty.
    """
    rules = node_rules(schema_config)
    return SeedState(
        nodeStyles={
            rule.label: SeedNodeStyle(
                color=rule.fill,
                size=math.floor(rule.diameter_px / DIAMETER_TO_SIZE),
                captions=[_caption(rule.caption)],
            )
            for rule in rules
        },
        relStyles={},
        stylingPriorityOrder=[rule.label for rule in reversed(rules)],
    )


def persisted_value(state: SeedState) -> str:
    """The local-storage value redux-persist writes.

    Every slice field is JSON-encoded on its own, beside a ``_persist`` record at the
    slice's version.
    """
    fields = {
        name: json.dumps(value, separators=(",", ":"))
        for name, value in state.model_dump(exclude_none=True).items()
    }
    fields["_persist"] = json.dumps(
        {"version": STYLING_PERSIST_VERSION, "rehydrated": True}, separators=(",", ":")
    )
    return json.dumps(fields, separators=(",", ":"))


def stylesheet_sha256(schema_config: Path = SCHEMA_CONFIG) -> str:
    """sha256 of the rendered stylesheet; the seed re-runs when it changes."""
    return hashlib.sha256(render(schema_config).encode("utf-8")).hexdigest()


def render_seed_js(schema_config: Path = SCHEMA_CONFIG) -> str:
    """The script the patched Browser page loads before its own bundle."""
    value = json.dumps(persisted_value(seed_state(schema_config)))
    return (
        "/* torchcell Neo4j Browser styling seed. Generated by\n"
        " * python -m torchcell.database.browser_style; do not edit by hand.\n"
        " * Plants the torchcell.grass styling in this browser's local storage before the\n"
        " * Browser rehydrates its graphStyling slice. Runs once per stylesheet: the\n"
        " * stylesheet's sha256 is recorded and the seed steps aside while it matches. */\n"
        "(function () {\n"
        f"  var STYLING_KEY = {json.dumps(STYLING_STORAGE_KEY)};\n"
        f"  var SHA_KEY = {json.dumps(SEED_SHA_STORAGE_KEY)};\n"
        f"  var SHA = {json.dumps(stylesheet_sha256(schema_config))};\n"
        f"  var VALUE = {value};\n"
        "  if (window.localStorage.getItem(SHA_KEY) === SHA) {\n"
        "    return;\n"
        "  }\n"
        "  window.localStorage.setItem(STYLING_KEY, VALUE);\n"
        "  window.localStorage.setItem(SHA_KEY, SHA);\n"
        "})();\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the stylesheet and the seed, or with ``--check`` report whether both are current."""
    parser = argparse.ArgumentParser(
        prog="python -m torchcell.database.browser_style",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-output", type=Path, default=DEFAULT_SEED_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if --output or --seed-output differs from the render",
    )
    args = parser.parse_args(argv)
    outputs = {args.output: render(), args.seed_output: render_seed_js()}
    if args.check:
        stale = [
            path
            for path, text in outputs.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        for path in stale:
            print(f"{path} is stale; run python -m torchcell.database.browser_style")
        if stale:
            return 1
        print(f"{args.output} and {args.seed_output} are current")
        return 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    n_rules = len(node_rules())
    print(f"wrote {args.output} ({n_rules} node rules) and {args.seed_output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
