# torchcell/knowledge_graphs/supported_queries/cypher_deps.py
# [[torchcell.knowledge_graphs.supported_queries.cypher_deps]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/knowledge_graphs/supported_queries/cypher_deps.py
# Test file: tests/torchcell/knowledge_graphs/supported_queries/test_cypher_deps.py
"""What a Cypher query reads from the graph, extracted from its text.

A supported query depends on node labels, relationship types, properties and literal
values (``dataset.id``, ``graph_level``, media names) that an ontology revision can move.
The dependencies are **extracted, never typed**: :func:`extract_dependencies` reads the
``.cql`` text, so the registry cannot disagree with the query it describes.

The extractor is a set of regular expressions over a normalized text, not a Cypher
parser. Normalization removes ``//`` and ``/* */`` comments and, for every rule that must
not see literal text, replaces the characters inside ``'...'`` and ``"..."`` literals with
spaces (offsets are kept, so a block boundary found in one form is valid in the other).
The query is split into blocks at ``UNION`` / ``UNION ALL``; variables are scoped to a
block, which is how every query in this repo is written.

Rules, per block:

- **node labels**: every ``:Label`` inside a node pattern ``(:Label)``, ``(x:Label)`` or
  ``(x:A:B)``; the variable, when present, is bound to those labels for the block.
- **relationship types**: every type in ``[:Type]``, ``[r:Type]`` or ``[r:A|B]``.
- **properties**: ``x.prop`` where ``x`` is bound in the block by a node pattern
  ``(x...)``, a relationship pattern ``[x...]``, an iteration ``x IN``, or ``AS x``.
  A read through a variable that carries a label is also recorded as ``Label.prop`` in
  ``label_properties`` (one entry per label), which is what the drift check compares
  with the graph schema. Function calls (``x.f(``) are not property reads.
- **dataset ids**: ``x.id = '<lit>'`` and ``x.id IN ['<lit>', ...]`` where ``x`` carries the
  ``Dataset`` label in the block.
- **graph levels**: ``x.graph_level = '<lit>'`` and ``x.graph_level IN [...]``, any ``x``.
- **media names**: ``x.name = '<lit>'`` and ``x.name IN [...]`` where ``x`` carries the
  ``Media`` label in the block.
- **parameters**: every ``$name`` outside a literal.

Documented limits (a query that relies on one of these under-reports its dependencies):

- Labels in a ``WHERE x:Label`` predicate, labels or types quoted with backticks, and
  label expressions (``:A&B``, ``:!A``) are not recognized.
- A literal on the left (``'X' = dataset.id``) and a dataset id passed as a parameter are
  not recognized as dataset ids; a parameter shows up in ``parameters`` only.
- ``CALL { ... }`` subqueries share their enclosing block's variable scope here.
- Property reads through a relationship variable are recorded in ``properties`` but not
  in ``label_properties``: the graph schema lists no edge properties to check them against.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from torchcell.knowledge_graphs.supported_queries.registry import QueryDependencies

__all__ = ["strip_comments", "blank_literals", "split_blocks", "extract_dependencies"]

_IDENT = r"[A-Za-z_]\w*"
_NODE_LABELS = re.compile(
    rf"\(\s*({_IDENT})?\s*((?::\s*{_IDENT}\s*)+)(?=[){{])", re.MULTILINE
)
_LABEL = re.compile(rf":\s*({_IDENT})")
_REL_TYPES = re.compile(
    rf"\[\s*({_IDENT})?\s*:\s*({_IDENT}(?:\s*\|\s*:?\s*{_IDENT})*)", re.MULTILINE
)
_REL_TYPE_SPLIT = re.compile(r"\s*\|\s*:?\s*")
_NODE_VAR = re.compile(rf"\(\s*({_IDENT})\s*(?=[):{{])")
_REL_VAR = re.compile(rf"\[\s*({_IDENT})\s*(?=[:\]*{{])")
_IN_VAR = re.compile(rf"(?<![\w.$])({_IDENT})\s+(?i:IN)\b")
_AS_VAR = re.compile(rf"\b(?i:AS)\s+({_IDENT})")
_PROPERTY = re.compile(rf"(?<![\w.$])({_IDENT})\.({_IDENT})\b(?!\s*\()")
_PARAMETER = re.compile(rf"\$({_IDENT})")
_UNION = re.compile(r"\bUNION(?:\s+ALL)?\b", re.IGNORECASE)
_STRING = r"(?:'([^'\\]*)'|\"([^\"\\]*)\")"


def _literal_rule(prop: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    """``(x.prop = '<lit>', x.prop IN [...])`` patterns; group 1 is the variable."""
    equals = re.compile(rf"(?<![\w.$])({_IDENT})\.{prop}\s*=\s*{_STRING}")
    within = re.compile(rf"(?<![\w.$])({_IDENT})\.{prop}\s+(?i:IN)\s*\[([^\]]*)\]")
    return equals, within


_DATASET_ID = _literal_rule("id")
_GRAPH_LEVEL = _literal_rule("graph_level")
_MEDIA_NAME = _literal_rule("name")
_LIST_ITEM = re.compile(_STRING)


def _scan(text: str, keep_literals: bool) -> str:
    """Drop comments; with ``keep_literals=False`` also blank string-literal contents.

    Blanking replaces each character inside the quotes with a space (a newline stays a
    newline), so offsets are identical between the two forms. Backtick-quoted names are
    passed through untouched: they are identifiers, never comments.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch in "'\"`":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            # An unterminated literal runs to the end of the text.
            close, stop = (ch, j + 1) if j < n else ("", n)
            body = text[i + 1 : stop - len(close)]
            if keep_literals or ch == "`":
                out.append(text[i:stop])
            else:
                blank = "".join("\n" if c == "\n" else " " for c in body)
                out.append(ch + blank + close)
            i = stop
            continue
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j == -1 else j
            out.append(" " * (j - i))
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j == -1 else j + 2
            out.append("".join("\n" if c == "\n" else " " for c in text[i:j]))
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def strip_comments(text: str) -> str:
    """The query with ``//`` and ``/* */`` comments replaced by spaces (offsets kept)."""
    return _scan(text, keep_literals=True)


def blank_literals(text: str) -> str:
    """:func:`strip_comments`, plus every string literal's contents replaced by spaces."""
    return _scan(text, keep_literals=False)


def split_blocks(text: str) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of the ``UNION``-separated blocks of blanked ``text``."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _UNION.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return spans


def _literal_values(
    source: str,
    rule: tuple[re.Pattern[str], re.Pattern[str]],
    variables: Iterable[str] | None,
) -> set[str]:
    """Literal values assigned to ``x.prop`` in ``source`` for ``x`` in ``variables``.

    ``variables=None`` accepts any variable.
    """
    allowed = None if variables is None else set(variables)
    values: set[str] = set()
    equals, within = rule
    for match in equals.finditer(source):
        if allowed is None or match.group(1) in allowed:
            values.add(match.group(2) if match.group(2) is not None else match.group(3))
    for match in within.finditer(source):
        if allowed is None or match.group(1) in allowed:
            for item in _LIST_ITEM.finditer(match.group(2)):
                values.add(
                    item.group(1) if item.group(1) is not None else item.group(2)
                )
    return values


def extract_dependencies(cypher: str) -> QueryDependencies:
    """The labels, types, properties, literals and parameters ``cypher`` depends on."""
    source = strip_comments(cypher)
    blanked = blank_literals(cypher)
    node_labels: set[str] = set()
    relationship_types: set[str] = set()
    properties: set[str] = set()
    label_properties: set[str] = set()
    dataset_ids: set[str] = set()
    graph_levels: set[str] = set()
    media_names: set[str] = set()
    parameters: set[str] = set()
    for start, end in split_blocks(blanked):
        block = blanked[start:end]
        literal_block = source[start:end]
        var_labels: dict[str, set[str]] = {}
        for match in _NODE_LABELS.finditer(block):
            labels = set(_LABEL.findall(match.group(2)))
            node_labels |= labels
            if match.group(1) is not None:
                var_labels.setdefault(match.group(1), set()).update(labels)
        for match in _REL_TYPES.finditer(block):
            relationship_types.update(_REL_TYPE_SPLIT.split(match.group(2).strip()))
        bound = set(var_labels)
        for pattern in (_NODE_VAR, _REL_VAR, _IN_VAR, _AS_VAR):
            bound.update(pattern.findall(block))
        for var, prop in _PROPERTY.findall(block):
            if var not in bound:
                continue
            properties.add(prop)
            for label in var_labels.get(var, ()):
                label_properties.add(f"{label}.{prop}")
        parameters.update(_PARAMETER.findall(block))
        dataset_vars = [v for v, labels in var_labels.items() if "Dataset" in labels]
        media_vars = [v for v, labels in var_labels.items() if "Media" in labels]
        dataset_ids |= _literal_values(literal_block, _DATASET_ID, dataset_vars)
        graph_levels |= _literal_values(literal_block, _GRAPH_LEVEL, None)
        media_names |= _literal_values(literal_block, _MEDIA_NAME, media_vars)
    return QueryDependencies(
        node_labels=sorted(node_labels),
        relationship_types=sorted(relationship_types),
        properties=sorted(properties),
        label_properties=sorted(label_properties),
        dataset_ids=sorted(dataset_ids),
        graph_levels=sorted(graph_levels),
        media_names=sorted(media_names),
        parameters=sorted(parameters),
    )
