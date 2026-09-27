# experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_media_components.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.embed_media_components]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/embed_media_components
"""Embed the MEDIA components of the three env-chemgen datasets with every registered
molecule encoder, and report coverage.

``embed_compounds.py`` covers the DOSED perturbation compounds. The medium the cells
grew in is the other half of the environment axis and was never embedded: Vanacloig's
SynBase, Hillenmeyer's YPD / YP-glycerol / SD / SC and the eighteen SC drop-out
variants all resolve to typed ``MediaComponent`` lists that carry a ``Compound`` each.

Enumeration route (measured, two joined halves)
-----------------------------------------------
1. WHICH media were served comes from the flattened parquets written by
   ``flatten_records.py`` (``results/records_<dataset>.parquet``): the distinct
   ``media_name`` and ``ref_media_name`` values, so a medium that appears only as a
   reference environment still counts.
2. WHAT each medium contains comes from the ``Media`` constants in
   ``torchcell/datamodels/media.py`` -- the same objects the loaders pass into the
   records by reference, indexed here by ``Media.name``. A served name that matches no
   constant raises; there is no name-normalizing fallback, because a silent miss would
   under-report the component set.

This avoids re-reading multi-GB LMDBs for objects the loaders hold as module constants,
and the ``Media.name`` join is exact.

Embeddability, and why a component is not embeddable
----------------------------------------------------
A component is embeddable iff its ``Compound`` carries a SMILES (every component that
does also carries an InChIKey). The reasons a component carries none are recorded as a
typed ``ComponentReason``, all four of which are read off the data rather than assumed:

- ``composition_deferred`` -- a DEFINED sub-mix nobody expanded (commercial YNB, the
  SynH3- hydrolysate base). ``ComponentDefinition.composition_deferred``.
- ``intrinsically_undefined`` -- a batch-variable biological digest (peptone, yeast
  extract) that no recipe pins. ``ComponentDefinition.intrinsically_undefined``.
- ``resolved_mixture`` -- the identity table resolved the substance to ChEBI/CID but no
  single-molecule InChIKey exists (``resolution_status="RESOLVED_MIXTURE"``, e.g. agar).
- ``absent_from_identity_table`` -- the resolver returned a bare ``Compound`` with a
  typed ``inchikey`` gap because the name is in no curation input list. A real curation
  gap, not a property of the substance.

``n_failed`` is separate: a component that HAS a SMILES which the encoder's own
``check`` rejects. Each is named in ``results/embeddings_media/failures.json``.

Outputs
-------
- ``results/embeddings_media/<encoder>.npz`` -- ``inchikey`` (str) and ``X`` (float32),
  one row per InChIKey across the union of the three datasets' media components.
- ``results/embeddings_media/failures.json`` -- per encoder, key -> error message.
- ``results/media_component_coverage.csv`` -- one row per (dataset, encoder).
- ``results/media_components.csv`` -- one row per (dataset, component): the enumeration
  itself, with definition class, role, identifiers and reason.

Dropouts are reported alongside but never mixed into the component counts: a dropout is
a compound deliberately OMITTED from the medium, so it is a different axis from what the
flask contains. Their vectors go into the same npz, because an SC drop-out medium is
only defined by naming the molecule that is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import time
from enum import StrEnum

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

import torchcell.datamodels.media as media_lib
from torchcell.datamodels.schema import ComponentDefinition, Media
from torchcell.molecule import ENCODERS, MoleculeEncoder

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)
EMBED_DIR = osp.join(RESULTS_DIR, "embeddings_media")
DATASETS = ["vanacloig2022", "hillenmeyer2008_hom", "hillenmeyer2008_het"]


class ComponentReason(StrEnum):
    """Why a medium component is or is not embeddable."""

    embeddable = "embeddable"
    composition_deferred = "composition_deferred"
    intrinsically_undefined = "intrinsically_undefined"
    resolved_mixture = "resolved_mixture"
    absent_from_identity_table = "absent_from_identity_table"


#: Reason columns of the coverage table, in report order.
GAP_REASONS = [
    ComponentReason.composition_deferred,
    ComponentReason.intrinsically_undefined,
    ComponentReason.resolved_mixture,
    ComponentReason.absent_from_identity_table,
]


class ComponentRow(BaseModel):
    """One enumerated medium constituent of one dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    kind: str = Field(description="'component' (in the medium) or 'dropout' (omitted)")
    name: str
    role: str
    definition: str
    inchikey: str | None
    smiles: str | None
    resolution_status: str | None = Field(
        description="identity-table status; None when the name is in no input list"
    )
    reason: ComponentReason
    n_media: int = Field(description="how many served media of this dataset carry it")


def media_index() -> dict[str, Media]:
    """``Media.name`` -> the library constant, over every ``Media`` the library owns.

    Includes the eighteen ``HILLENMEYER_DROPOUT_MEDIA`` values, which are built by
    ``dropout()`` at import and so are not module-level attributes.
    """
    index: dict[str, Media] = {
        value.name: value
        for value in vars(media_lib).values()
        if isinstance(value, Media)
    }
    for medium in media_lib.HILLENMEYER_DROPOUT_MEDIA.values():
        index.setdefault(medium.name, medium)
    return index


def identity_status() -> dict[str, str]:
    """Compound name -> ``resolution_status`` from the sha256-pinned identity table."""
    from torchcell.datamodels.compound_identity import _TABLE_PATH

    with open(_TABLE_PATH) as f:
        records = json.load(f)["records"]
    return {r["name"]: r["resolution_status"] for r in records}


def served_media(dataset: str, index: dict[str, Media]) -> list[Media]:
    """The ``Media`` objects the dataset's served records actually reference.

    Both the perturbed environment (``media_name``) and the reference environment
    (``ref_media_name``) count. An unmatched name raises.
    """
    df = pd.read_parquet(
        osp.join(RESULTS_DIR, f"records_{dataset}.parquet"),
        columns=["media_name", "ref_media_name"],
    )
    names = {
        name
        for name in set(df["media_name"].dropna()) | set(df["ref_media_name"].dropna())
        if name
    }
    unmatched = sorted(name for name in names if name not in index)
    if unmatched:
        raise KeyError(
            f"{dataset}: served media names absent from the media library: {unmatched}"
        )
    return [index[name] for name in sorted(names)]


def classify(
    definition: ComponentDefinition, smiles: str | None, status: str | None
) -> ComponentReason:
    """The typed reason, read off the component's own definition and identifiers."""
    if smiles:
        return ComponentReason.embeddable
    if definition is ComponentDefinition.composition_deferred:
        return ComponentReason.composition_deferred
    if definition is ComponentDefinition.intrinsically_undefined:
        return ComponentReason.intrinsically_undefined
    if status is None:
        return ComponentReason.absent_from_identity_table
    if status == "RESOLVED_MIXTURE":
        return ComponentReason.resolved_mixture
    raise ValueError(
        f"a 'defined' component with no SMILES and identity status {status!r} is "
        "unaccounted for; classify it before embedding"
    )


def enumerate_rows(
    dataset: str, media: list[Media], status: dict[str, str]
) -> list[ComponentRow]:
    """Every distinct component and dropout of ``dataset``'s served media."""
    counts: dict[tuple[str, str], int] = {}
    seen: dict[tuple[str, str], ComponentRow] = {}
    for medium in media:
        for component in medium.components:
            compound = component.compound
            key = ("component", compound.name)
            counts[key] = counts.get(key, 0) + 1
            seen[key] = ComponentRow(
                dataset=dataset,
                kind="component",
                name=compound.name,
                role=component.role.value,
                definition=component.definition.value,
                inchikey=compound.inchikey,
                smiles=compound.smiles,
                resolution_status=status.get(compound.name),
                reason=classify(
                    component.definition, compound.smiles, status.get(compound.name)
                ),
                n_media=0,
            )
        for compound in medium.dropouts:
            key = ("dropout", compound.name)
            counts[key] = counts.get(key, 0) + 1
            seen[key] = ComponentRow(
                dataset=dataset,
                kind="dropout",
                name=compound.name,
                role="dropout",
                definition=ComponentDefinition.defined.value,
                inchikey=compound.inchikey,
                smiles=compound.smiles,
                resolution_status=status.get(compound.name),
                reason=classify(
                    ComponentDefinition.defined,
                    compound.smiles,
                    status.get(compound.name),
                ),
                n_media=0,
            )
    return [
        row.model_copy(update={"n_media": counts[(row.kind, row.name)]})
        for row in seen.values()
    ]


def embed_all(
    encoder: MoleculeEncoder, smiles_by_key: dict[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Per-molecule ``check`` for coverage accounting, then one batched encode.

    A molecule the encoder rejects is named and counted, never embedded from a
    stand-in. Returns (key -> vector, key -> error message).
    """
    accepted: list[str] = []
    failed: dict[str, str] = {}
    for key, smi in smiles_by_key.items():
        try:
            encoder.check(smi)
        except ValueError as e:
            failed[key] = f"{type(e).__name__}: {e}"
        else:
            accepted.append(key)
    x = encoder.encode([smiles_by_key[k] for k in accepted])
    return dict(zip(accepted, x, strict=True)), failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS)
    )
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    args = ap.parse_args()

    index = media_index()
    status = identity_status()
    per_dataset: dict[str, list[ComponentRow]] = {}
    n_media: dict[str, int] = {}
    for dataset in args.datasets:
        media = served_media(dataset, index)
        rows = enumerate_rows(dataset, media, status)
        per_dataset[dataset] = rows
        n_media[dataset] = len(media)
        comps = [r for r in rows if r.kind == "component"]
        drops = [r for r in rows if r.kind == "dropout"]
        print(
            f"{dataset}: {n_media[dataset]} served media, {len(comps)} components "
            f"({sum(1 for r in comps if r.smiles)} with SMILES), {len(drops)} dropouts "
            f"({sum(1 for r in drops if r.smiles)} with SMILES)"
        )
        for row in rows:
            if row.reason is not ComponentReason.embeddable:
                print(f"  NO STRUCTURE [{row.kind}] {row.name!r}: {row.reason.value}")

    table = pd.DataFrame(
        [row.model_dump(mode="json") for rows in per_dataset.values() for row in rows]
    ).sort_values(["dataset", "kind", "name"])
    table.to_csv(osp.join(RESULTS_DIR, "media_components.csv"), index=False)

    smiles_by_key: dict[str, str] = {
        row.inchikey: row.smiles
        for rows in per_dataset.values()
        for row in rows
        if row.inchikey and row.smiles
    }
    print(f"union: {len(smiles_by_key)} InChIKeys with SMILES over the media sets")

    os.makedirs(EMBED_DIR, exist_ok=True)
    cov_rows = []
    failures: dict[str, dict[str, str]] = {}
    for enc_name in args.encoders:
        t0 = time.time()
        encoder = ENCODERS[enc_name]()
        t_load = time.time() - t0
        t0 = time.time()
        ok, failed = embed_all(encoder, smiles_by_key)
        seconds = time.time() - t0
        failures[enc_name] = failed
        keys_ok = sorted(ok)
        np.savez(
            osp.join(EMBED_DIR, f"{enc_name}.npz"),
            inchikey=np.array(keys_ok, dtype=str),
            X=np.stack([ok[k] for k in keys_ok]).astype(np.float32),
        )
        print(
            f"{enc_name}: dim {encoder.dim}, load {t_load:.1f}s, embedded {len(ok)} "
            f"in {seconds:.1f}s, failed {len(failed)}"
        )
        for key, msg in failed.items():
            print(f"  FAILED {key}: {msg[:160]}")
        for dataset in args.datasets:
            rows = per_dataset[dataset]
            comps = [r for r in rows if r.kind == "component"]
            drops = [r for r in rows if r.kind == "dropout"]
            comp_smiles = [r for r in comps if r.inchikey and r.smiles]
            drop_smiles = [r for r in drops if r.inchikey and r.smiles]
            n_emb = sum(1 for r in comp_smiles if r.inchikey in ok)
            cov_rows.append(
                {
                    "dataset": dataset,
                    "encoder": enc_name,
                    "n_media": n_media[dataset],
                    "n_components": len(comps),
                    "n_with_smiles": len(comp_smiles),
                    "n_embedded": n_emb,
                    "n_failed": len(comp_smiles) - n_emb,
                    **{
                        f"n_{reason.value}": sum(1 for r in comps if r.reason is reason)
                        for reason in GAP_REASONS
                    },
                    "n_dropouts": len(drops),
                    "n_dropouts_with_smiles": len(drop_smiles),
                    "n_dropouts_embedded": sum(
                        1 for r in drop_smiles if r.inchikey in ok
                    ),
                    "dim": encoder.dim,
                    "seconds_union": round(seconds, 3),
                    "seconds_per_100": round(100 * seconds / max(len(ok), 1), 3),
                }
            )
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(osp.join(RESULTS_DIR, "media_component_coverage.csv"), index=False)
    with open(osp.join(EMBED_DIR, "failures.json"), "w") as f:
        json.dump(failures, f, indent=2, sort_keys=True)
    print()
    print(cov.to_string(index=False))


if __name__ == "__main__":
    main()
