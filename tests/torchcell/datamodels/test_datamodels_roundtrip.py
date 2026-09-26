# tests/torchcell/datamodels/test_datamodels_roundtrip.py
# [[tests.torchcell.datamodels.test_datamodels_roundtrip]]
# https://github.com/Mjvolk3/torchcell/tree/main/tests/torchcell/datamodels/test_datamodels_roundtrip.py
"""Serialization round trip for every pydantic model in ``torchcell.datamodels``.

The classes are discovered by walking the package, never listed by hand, so a new model
joins the sweep the day it is defined. Each class gets an instance from ``EXAMPLES`` when
a hand-written dict is committed for it, else from a generic builder that fills every
required field from its annotation (first Literal value, first Enum member, one nested
model, one-element lists and dicts) with a few field-name conventions the schema
validators enforce (``systematic_gene_name`` is a systematic ORF name, ``*so_id`` a
Sequence Ontology CURIE, ``*sha256`` sixty-four hex digits). A class the builder cannot
construct must appear in ``UNCONSTRUCTIBLE`` with the reason; an unexplained failure is a
test failure, so nothing is skipped silently (Phase 1 of
[[plan.test-suite-buildout.2026.09.25]]).

Pinned per class: ``model_validate(model_dump())`` and ``model_validate_json(
model_dump_json())`` reproduce an equal object, ``model_dump()`` is deterministic across
two calls, and ``model_json_schema()`` is JSON-serializable and identical across two
calls.
"""

from __future__ import annotations

import datetime as dt
import enum
import importlib
import inspect
import json
import pkgutil
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

import torchcell.datamodels as datamodels

# Modules whose import has side effects or needs optional dependencies.
SKIP_MODULES = {"torchcell.datamodels.compound_identity_curate"}

# Values the schema validators demand for a field of this name, whatever the class.
FIELD_VALUES: dict[str, Any] = {
    "systematic_gene_name": "YAL001C",
    "perturbed_gene_name": "TFC3",
    "so_id": "SO:0000704",
    "mechanism_so_id": "SO:0000159",
    "mechanism_so_name": "deletion",
    "sequence_sha256": "a" * 64,
    "effector_plasmid_sha256": "a" * 64,
    "pubmed_id": "12345678",
}

# class qualname -> keyword arguments; committed when a validator wants a specific shape.
EXAMPLES: dict[str, dict[str, Any]] = {
    "torchcell.datamodels.schema.Concentration": {"value": 1.0, "unit": "mM"},
    "torchcell.datamodels.schema.EnvironmentResponsePhenotype": {
        "measurement_type": "log2_ratio",
        "environment_response": 0.5,
    },
    "torchcell.datamodels.schema.ExpressionModulationPerturbation": {
        "systematic_gene_name": "YAL001C",
        "perturbed_gene_name": "TFC3",
        "expression_direction": "increased",
        "crispr": {"effector": "dCas9-VPR"},
    },
    "torchcell.datamodels.schema.Media": {
        "name": "YPD",
        "state": "solid",
        "is_synthetic": False,
    },
    "torchcell.datamodels.schema.CalMorphPhenotype": {"calmorph": {"A101_A": 0.5}},
    "torchcell.datamodels.schema.PresenceAbsencePerturbation": {
        "systematic_gene_name": "YAL001C",
        "perturbed_gene_name": "TFC3",
        "state": "present",
        "mechanism_so_id": "SO:0000159",
        "mechanism_so_name": "deletion",
    },
    "torchcell.datamodels.schema.Publication": {
        "pubmed_id": "12345678",
        "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
    },
    "torchcell.datamodels.schema.VisualScorePhenotype": {
        "visual_score": 2.0,
        "n_replicates": 3,
        "score_scale_min": 0,
        "score_scale_max": 4,
        "score_semantics": "higher is more colored",
        "target_product": "beta-carotene",
    },
}

# class qualname -> why no example exists. Strict: a class listed here that DOES round
# trip fails its xfail, so the list decays.
UNCONSTRUCTIBLE: dict[str, str] = {
    "torchcell.datamodels.schema.Phenotype": (
        "abstract base: label_name must be a class attribute of a concrete subclass"
    ),
    "torchcell.datamodels.schema.Experiment": (
        "abstract base: its phenotype field is the abstract Phenotype"
    ),
    "torchcell.datamodels.schema.ExperimentReference": (
        "abstract base: its phenotype_reference field is the abstract Phenotype"
    ),
    "torchcell.datamodels.conversion.ConversionEntry": (
        "holds classes and callables; not JSON-serializable by design"
    ),
    "torchcell.datamodels.conversion.ConversionMap": (
        "holds ConversionEntry callables; not JSON-serializable by design"
    ),
}


class Unbuildable(Exception):
    """The generic builder has no value for an annotation."""


def _models() -> dict[str, type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(datamodels.__path__, "torchcell.datamodels."):
        if info.name in SKIP_MODULES:
            continue
        module = importlib.import_module(info.name)
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj is not BaseModel
                and obj.__module__ == info.name
            ):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    return dict(sorted(found.items()))


MODELS = _models()


def _strip_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        return args[0] if args else type(None)
    if origin is typing.Annotated:
        return _strip_optional(typing.get_args(annotation)[0])
    return annotation


def _value(annotation: Any, depth: int) -> Any:
    """A minimal value for an annotation, or raise Unbuildable."""
    if depth > 8:
        raise Unbuildable(f"nesting deeper than 8 at {annotation!r}")
    annotation = _strip_optional(annotation)
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Literal:
        return args[0]
    if inspect.isclass(annotation):
        if issubclass(annotation, BaseModel):
            return _example(annotation, depth + 1)
        if issubclass(annotation, enum.Enum):
            return next(iter(annotation))
        if annotation is bool:
            return True
        if annotation is int:
            return 1
        if annotation is float:
            return 0.5
        if annotation is str:
            return "x"
        if annotation is bytes:
            return b"x"
        if annotation is Path:
            return Path("x")
        if annotation is dt.datetime:
            return dt.datetime(2026, 1, 2, 3, 4, 5)
        if annotation is dt.date:
            return dt.date(2026, 1, 2)
        if annotation in (dict, list, set, tuple, frozenset):
            return annotation()
    if origin in (list, set, frozenset, typing.Sequence, typing.Iterable):
        return origin([_value(args[0], depth + 1)]) if args else []
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return (_value(args[0], depth + 1),)
        return tuple(_value(a, depth + 1) for a in args)
    if origin is dict or origin is typing.Mapping:
        if len(args) == 2:
            return {_value(args[0], depth + 1): _value(args[1], depth + 1)}
        return {}
    raise Unbuildable(f"no generic value for {annotation!r}")


def _example(cls: type[BaseModel], depth: int = 0) -> BaseModel:
    """An instance of ``cls`` from EXAMPLES or the generic builder."""
    name = f"{cls.__module__}.{cls.__qualname__}"
    if name in EXAMPLES:
        return cls.model_validate(EXAMPLES[name])
    kwargs = {
        field: FIELD_VALUES[field]
        if field in FIELD_VALUES
        else _value(info.annotation, depth)
        for field, info in cls.model_fields.items()
        if info.is_required()
    }
    return cls.model_validate(kwargs)


def _constructible(name: str) -> object:
    if name in UNCONSTRUCTIBLE:
        return pytest.param(
            name, marks=pytest.mark.xfail(strict=True, reason=UNCONSTRUCTIBLE[name])
        )
    return name


@pytest.mark.parametrize("name", [_constructible(n) for n in MODELS])
def test_round_trip_through_dict_and_json(name: str) -> None:
    """validate(dump(obj)) == obj by dict and by JSON; dump and schema are deterministic."""
    cls = MODELS[name]
    obj = _example(cls)
    dumped = obj.model_dump()
    assert cls.model_validate(dumped) == obj
    assert cls.model_validate_json(obj.model_dump_json()) == obj
    assert obj.model_dump() == dumped
    schema = obj.model_json_schema()
    assert json.loads(json.dumps(schema)) == schema
    assert cls.model_json_schema() == schema


def test_sweep_covers_every_model_in_the_package() -> None:
    """At least the 109 models present on 2026.09.26 are collected (guards the walker)."""
    assert len(MODELS) >= 109
    assert all(issubclass(cls, BaseModel) for cls in MODELS.values())


def test_example_and_unconstructible_keys_name_collected_models() -> None:
    """Every committed key refers to a class the walker found (no stale entries)."""
    assert set(EXAMPLES) <= set(MODELS), sorted(set(EXAMPLES) - set(MODELS))
    assert set(UNCONSTRUCTIBLE) <= set(MODELS), sorted(
        set(UNCONSTRUCTIBLE) - set(MODELS)
    )
    assert not set(EXAMPLES) & set(UNCONSTRUCTIBLE)


def test_examples_are_needed() -> None:
    """Every EXAMPLES entry exists because the generic builder fails for that class."""
    for name in EXAMPLES:
        cls = MODELS[name]
        with pytest.raises(Exception):  # noqa: B017 - ValidationError or Unbuildable
            kwargs = {
                field: FIELD_VALUES[field]
                if field in FIELD_VALUES
                else _value(info.annotation, 0)
                for field, info in cls.model_fields.items()
                if info.is_required()
            }
            cls.model_validate(kwargs)
