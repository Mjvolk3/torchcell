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

The id of an interned constant is namespaced: sha256 over the label, a NUL byte, the
kind, a NUL byte and the payload, never the bare sha256 of the payload. The bare hash
was the id of the sub-object's OWN graph node (a ``genotype`` or ``segregant genotype``
node is ``sha256(json.dumps(genotype.model_dump()))``, the same bytes a whole-genotype
constant holds), and ``torchcell.fast_csv.FastCsvSink`` deduplicates node ids across
every label, so the interned constant (written first, by the experiment node method)
owned the id and the genotype node was never written: on the job 4901 partial 218,258
whole genotypes and 135 Hale 2024 backgrounds existed only as ``InternedConstant``
nodes, with 1,948,098 ``GenotypeMemberOf``, 121,180 ``PerturbationMemberOf`` and
709,716 ``SegregantBackgroundOf`` rows pointing at them, and on the served KG 3.0 store
14,573 Bloom 2019 and Caudal 2024 genotypes (518,883 ``GenotypeMemberOf`` rows). Every
other content id in the graph hashes a JSON document (first byte ``{``), so a preimage
that begins with the label cannot equal one, and the kind is in the preimage because
the same bytes can be a whole ``genotype`` in one record and a ``genotype.background``
in another, and the node's ``kind`` property must be true of the node, not of whichever
record wrote it first (``interned_id_collision_counts.py``, 2026-10-15).
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

NESTED_POINTER_MIN_BYTES: dict[tuple[str, str], int] = {
    ("genotype", "background"): 8192
}
"""``(field, sub-field)`` pairs written as pointers BEFORE the top-level rule runs.

A perturbed segregant genotype (Hale 2024) is a background mosaic shared by every
perturbation measured on that segregant plus a ~1 KB perturbation list. Pointing at the
whole genotype would intern one constant per (segregant, guide) pair, each carrying the
same 8.7 KB mosaic; pointing at the background interns one constant per segregant and
leaves the small, varying perturbation inline. The kind written on the pointer is
``"genotype.background"``. A genotype with no ``background`` key (every other genotype
class) is untouched, so every other dataset's blob is byte-identical to before.
"""


def constant_id(kind: str, payload: str) -> str:
    """The node id of an interned constant: sha256 over label, kind and exact payload.

    The preimage is the label, a NUL byte, the kind, a NUL byte and the payload (UTF-8).
    It starts with the label, so it is never the preimage of a sub-object's own node id
    (the bare ``json.dumps`` of its dump), and it carries the kind, so the same bytes
    interned under two kinds are two nodes, each with a true ``kind`` property.
    """
    digest = hashlib.sha256()
    digest.update(f"{INTERNED_CONSTANT_LABEL}\x00{kind}\x00".encode())
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def split_experiment_dump(
    dump: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str, str]]]:
    """Return ``(pointered dump, constants)`` for one ``experiment.model_dump()``.

    ``constants`` is a list of ``(id, kind, payload)`` for every field (or nested
    sub-field, kind ``"<field>.<sub>"``) that became a pointer; ``payload`` is ``json.dumps`` of the field's value, byte-identical to the
    bytes that value contributes to ``json.dumps(dump)``, so splicing
    ``json.loads(payload)`` back reproduces the inlined record. ``dump`` is not
    mutated.
    """
    out = dict(dump)
    constants: list[tuple[str, str, str]] = []
    # A field that already holds a nested pointer stays inline: pointing at it again
    # would bury that pointer inside a constant, where the query loader (which collects
    # pointers from the Experiment blob only) would never resolve it.
    nested: set[str] = set()
    for (field, sub), min_bytes in NESTED_POINTER_MIN_BYTES.items():
        value = dump.get(field)
        if not isinstance(value, dict) or sub not in value:
            continue
        payload = json.dumps(value[sub])
        if len(payload) < min_bytes:
            continue
        kind = f"{field}.{sub}"
        ref = constant_id(kind, payload)
        constants.append((ref, kind, payload))
        out[field] = {**value, sub: {POINTER_KEY: ref, "kind": kind}}
        nested.add(field)
    for field, min_bytes in EXPERIMENT_POINTER_MIN_BYTES.items():
        if field in nested:
            continue
        payload = json.dumps(dump[field])
        if len(payload) < min_bytes:
            continue
        ref = constant_id(field, payload)
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


def verified_constant(ref: str, kind: str, payload: str) -> Any:
    """Parse a fetched payload after checking that (kind, payload) hashes to its id.

    ``kind`` is the node's own ``kind`` property: a node whose id is not the hash of
    its own kind and payload is corrupt, and a pointer's id was computed from the kind
    it sits under, so a matching id also proves the node's kind is the pointer's.
    """
    actual = constant_id(kind, payload)
    if actual != ref:
        raise ValueError(
            f"interned constant {ref} of kind {kind!r} holds a payload hashing to "
            f"{actual}; the store is corrupt or was written by a different serializer"
        )
    return json.loads(payload)
