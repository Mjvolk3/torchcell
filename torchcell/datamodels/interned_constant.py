# torchcell/datamodels/interned_constant.py
# [[torchcell.datamodels.interned_constant]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/datamodels/interned_constant
"""Content-addressed sub-objects of an Experiment record in the knowledge graph.

An Experiment node's ``serialized_data`` used to inline the whole record: 94% of the
release 1.2 store (645 of 689 GB) was that one property, and 85% of a Costanzo record
was its constant 9.4 KB Environment, repeated 20.7 million times per dataset. The
adapter now writes the large sub-objects of the record once each, as ``interned
constant`` nodes whose id is the sha256 of the exact JSON they hold, and leaves a
pointer ``{"$ref": <id>, "kind": <field>}`` in the Experiment blob where the
sub-object was. The query loader (``torchcell.data.neo4j_query_raw``) fetches the
constants by id, verifies each payload against its id, and splices them back, so a
query returns the same record the inline layout did.

The Experiment node id is unchanged: it is still the sha256 of the fully inlined
record, so incremental admission matches served nodes by the same content id, and
splicing the constants back into a blob must reproduce that id exactly (which the
build's round-trip check and the tests assert).

Which fields become pointers is fixed here, by field and by a size floor, so the
layout is a property of the code and not of the data: ``environment`` (the dataset
LMDB interns it under the same 512-byte floor) and ``genotype`` when it is large
(segregant genotypes carry haplotype blocks, tens of KB, repeated across a cross's
phenotypes). A gene-perturbation genotype (about 1 KB) and every phenotype stay
inline: they vary per record, and a pointer would only move bytes and add a lookup.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

INTERNED_CONSTANT_LABEL = "interned constant"
"""The schema-config node label the adapter emits (PascalCase ``InternedConstant``)."""

INTERNED_CONSTANT_NEO4J_LABEL = "InternedConstant"
"""The label as Neo4j stores it, for the query loader's lookup."""

POINTER_KEY = "$ref"

EXPERIMENT_POINTER_MIN_BYTES: dict[str, int] = {"environment": 512, "genotype": 8192}
"""Top-level fields of ``experiment.model_dump()`` written as pointers, with the
minimum JSON size (bytes) at which a value leaves the blob."""


def constant_id(payload: str) -> str:
    """The node id of an interned constant: sha256 of its exact JSON payload."""
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_experiment_dump(
    dump: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    """Return ``(pointered dump, constants)`` for one ``experiment.model_dump()``.

    ``constants`` is a list of ``(id, kind, payload)`` for every field that became a
    pointer; ``payload`` is ``json.dumps`` of the field's value, byte-identical to the
    bytes that value contributes to ``json.dumps(dump)``, so splicing
    ``json.loads(payload)`` back reproduces the inlined record. ``dump`` is not
    mutated.
    """
    out = dict(dump)
    constants: list[tuple[str, str, str]] = []
    for field, min_bytes in EXPERIMENT_POINTER_MIN_BYTES.items():
        payload = json.dumps(dump[field])
        if len(payload) < min_bytes:
            continue
        ref = constant_id(payload)
        constants.append((ref, field, payload))
        out[field] = {POINTER_KEY: ref, "kind": field}
    return out, constants


def collect_pointers(obj: Any, out: set[str]) -> None:
    """Add every ``$ref`` id reachable in ``obj`` to ``out``."""
    if isinstance(obj, dict):
        ref = obj.get(POINTER_KEY)
        if ref is not None:
            out.add(ref)
            return
        for value in obj.values():
            collect_pointers(value, out)
    elif isinstance(obj, list):
        for value in obj:
            collect_pointers(value, out)


def resolve_pointers(obj: Any, constants: dict[str, Any]) -> Any:
    """Splice constants back: a dict carrying ``$ref`` becomes ``constants[ref]``.

    A missing id raises ``KeyError``: a pointer the store cannot resolve is a broken
    build, not a record to skip.
    """
    if isinstance(obj, dict):
        ref = obj.get(POINTER_KEY)
        if ref is not None:
            return constants[ref]
        return {k: resolve_pointers(v, constants) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_pointers(v, constants) for v in obj]
    return obj


def verified_constant(ref: str, payload: str) -> Any:
    """Parse a fetched payload after checking it hashes to the id it was fetched by."""
    actual = constant_id(payload)
    if actual != ref:
        raise ValueError(
            f"interned constant {ref} holds a payload hashing to {actual}; the store "
            "is corrupt or was written by a different serializer"
        )
    return json.loads(payload)
