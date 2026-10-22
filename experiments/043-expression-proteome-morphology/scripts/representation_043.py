# experiments/043-expression-proteome-morphology/scripts/representation_043.py
# [[experiments.043-expression-proteome-morphology.scripts.representation_043]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/043-expression-proteome-morphology/scripts/representation_043.py
"""The 043 store bound to ``torchcell.data.representation_043``: build, load, resolve.

``build`` (CPU, one slurm job) turns the isolate composite store into the arrays the
resolver reads, under ``$DATA_ROOT/data/torchcell/experiments/043-expression-proteome-
morphology/representation_043/``:

* ``vocab_<encoder>.npy``: the sorted sha256 keys of each encoder, the union of the
  distinct-sequence tables of ``compute_isolate_composite.py`` (isolate alleles,
  accessory ORFs, the S288C context) and the Vaishnav/reporter inputs of
  ``embed_vaishnav_reporter_043.py``. A key's id is its position.
* ``contexts.npz``: per reference context (row 0 = S288C, then the 1,011 isolates of the
  index), per gene of the 6,607 index, the four key ids (-1 = gap), the gap kinds, the
  gene copy number (pangenome matrix; 1 for S288C; NaN when unreleased) and the state.
  Genes outside an isolate's released allele set (596 of the 6,607: the mitochondrial
  genes and SGD annotations the 2018 release does not carry) are ``NO_ISOLATE_ALLELE``.
* ``accessory.parquet``: per (isolate, accessory ORF) the four key ids and gaps.
* ``reporter.parquet`` + ``reporter.json``: the upstream window key id per Vaishnav
  variant and the three constant reporter keys.
* ``build_record.json``: a :class:`RepresentationBuildRecord`.

``load_resolver`` reads them back; ``resolve_genotype`` turns one raw record's genotype
into a :class:`RecordRepresentation`. Vectors stay in the h5 tables and are read through
``AlleleVectorStore`` (``vector_stores``) only when asked; job 6106 may still be filling
them.

    python experiments/043-expression-proteome-morphology/scripts/representation_043.py --stage build
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import os.path as osp
import socket
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data.representation_043 import (
    ENCODERS,
    INDEX_GAP,
    N_ENC,
    S288C_CONTEXT,
    AccessoryRow,
    AlleleVectorStore,
    Block,
    ContextState,
    ContextTables,
    GapKind,
    GenePositions,
    KeyVocabulary,
    RecordRepresentation,
    ReporterKeys,
    RepresentationConfig,
    RepresentationResolver,
    change_type_of,
)
from torchcell.models.genotype_tokens_043 import ChangeType

EXPERIMENT = "043-expression-proteome-morphology"
ISO_REL = "data/scerevisiae/isolate_embeddings/peter2018_1011"
REP_TAG = "representation_043"
#: encoder -> (input kind column of the index, distinct-sequence table)
ENCODER_INPUT: dict[str, tuple[str, str]] = {
    "fudt_species_upstream": ("up_sha256", "sequences/window_up_1003.parquet"),
    "fudt_species_downstream": ("down_sha256", "sequences/window_down_300.parquet"),
    "prot_t5_xl_uniref50": ("protein_sha256", "sequences/protein.parquet"),
    "calm": ("calm_sha256", "sequences/calm.parquet"),
}
#: per-encoder gap columns of the index (written by compute_isolate_composite.py
#: --stage alphabet: protein and CaLM no longer share cds_gap, since an ambiguity code
#: can gap CaLM's CDS while the translated protein is unambiguous)
ENCODER_GAP: dict[str, str] = {
    "fudt_species_upstream": "up_gap",
    "fudt_species_downstream": "down_gap",
    "prot_t5_xl_uniref50": "protein_gap",
    "calm": "calm_gap",
}
VAISHNAV_INPUTS_REL = "sequences/vaishnav2022_inputs.parquet"
STATE_CODE = {
    "present": ContextState.PRESENT,
    "absent": ContextState.ABSENT,
    "not_in_matrix": ContextState.NOT_IN_MATRIX,
}


class RepresentationBuildRecord(BaseModel):
    """``build_record.json``: what the arrays were built from and what they hold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    created_at: str
    wall_seconds: float
    host: str
    slurm_job_id: str | None
    torchcell_commit: str
    isolate_store: str
    index_parquet_bytes: int
    index_parquet_mtime: float
    gene_set_sha256: str
    n_genes: int
    contexts: int
    vocab_sizes: dict[str, int]
    accessory_rows: int
    reporter_variants: int
    off_index_reference_alleles: dict[str, int] = Field(
        description="reference-gene features of the index with no place in the 6,607 "
        "gene index (count of rows, and of distinct features)"
    )
    no_allele_genes_per_isolate: int
    counts: dict[str, int] = Field(default_factory=dict)


def rep_root(data_root: str) -> Path:
    """``$DATA_ROOT/data/torchcell/experiments/<EXPERIMENT>/representation_043``."""
    return Path(data_root) / "data/torchcell/experiments" / EXPERIMENT / REP_TAG


def iso_root(data_root: str) -> Path:
    """The isolate composite store."""
    return Path(data_root) / ISO_REL


def git_commit() -> str:
    """Short HEAD of the checkout this runs from."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=osp.dirname(osp.abspath(__file__)),
    ).stdout.strip()


def gene_set(data_root: str) -> list[str]:
    """The 6,607 gene index of the raw store (sorted)."""
    path = (
        Path(data_root)
        / "data/torchcell/experiments"
        / EXPERIMENT
        / "raw_043/processed/gene_set.json"
    )
    genes: list[str] = json.loads(path.read_text())
    if genes != sorted(genes):
        raise ValueError("gene_set.json is not sorted")
    return genes


def gff_positions(gff_path: str, genes: list[str]) -> GenePositions:
    """Gene spans and chromosome lengths from the SGD GFF (R64-4-1 coordinates)."""
    want = set(genes)
    coords: dict[str, tuple[str, int, int]] = {}
    lengths: dict[str, int] = {}
    with open(gff_path) as handle:
        for line in handle:
            if line.startswith("##FASTA"):
                break
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 9:
                continue
            if cols[2] == "chromosome":
                lengths[cols[0]] = int(cols[4])
            elif cols[2] == "gene":
                attrs = dict(kv.split("=", 1) for kv in cols[8].split(";") if "=" in kv)
                if attrs["ID"] in want:
                    coords[attrs["ID"]] = (cols[0], int(cols[3]), int(cols[4]))
    missing = want - set(coords)
    if missing:
        raise KeyError(f"{len(missing)} genes have no GFF gene feature")
    return GenePositions.build([coords[g] for g in genes], lengths)


def sha_array(values: Any) -> np.ndarray:
    """A pyarrow/pandas string column -> ``S64`` array, nulls as b''."""
    s = pd.Series(values).fillna("")
    return np.asarray(s.to_numpy(dtype=str), dtype="S64")


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def build(data_root: str) -> None:
    """Write the vocabularies, the context tables, the accessory and reporter keys."""
    t0 = time.time()
    store = iso_root(data_root)
    out = rep_root(data_root)
    out.mkdir(parents=True, exist_ok=True)
    genes = gene_set(data_root)
    gene_pos = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)
    vaishnav = pd.read_parquet(store / VAISHNAV_INPUTS_REL)

    vocabs: dict[str, KeyVocabulary] = {}
    for enc in ENCODERS:
        seq_keys = pq.read_table(store / ENCODER_INPUT[enc][1], columns=["sha256"])[
            "sha256"
        ].to_numpy()
        extra = vaishnav.loc[vaishnav["encoder"] == enc, "sha256"].to_numpy()
        vocab = KeyVocabulary(np.concatenate([sha_array(seq_keys), sha_array(extra)]))
        np.save(out / f"vocab_{enc}.npy", vocab.keys)
        vocabs[enc] = vocab
        print(f"vocab {enc}: {len(vocab)} keys", flush=True)

    # S288C context
    ref = pd.read_parquet(store / "index/s288c_reference.parquet")
    ref = ref.set_index("feature_id").loc[genes]
    isolates = sorted(
        pq.read_table(store / "index/per_isolate_counts.parquet", columns=["isolate"])[
            "isolate"
        ].to_pylist()
    )
    context_ids = (S288C_CONTEXT, *isolates)
    ctx_row = {c: i for i, c in enumerate(context_ids)}
    n_ctx = len(context_ids)
    key_ids = np.full((n_ctx, n_genes, N_ENC), -1, dtype=np.int32)
    key_gap = np.full((n_ctx, n_genes, N_ENC), GapKind.NO_ISOLATE_ALLELE, dtype=np.int8)
    gene_copy = np.full((n_ctx, n_genes), np.nan, dtype=np.float32)
    state = np.full((n_ctx, n_genes), ContextState.NO_ALLELE, dtype=np.int8)
    for e, enc in enumerate(ENCODERS):
        ids = vocabs[enc].ids(sha_array(ref[ENCODER_INPUT[enc][0]]))
        key_ids[0, :, e] = ids
        key_gap[0, :, e] = np.where(ids < 0, GapKind.NO_S288C_KEY, GapKind.NONE)
    gene_copy[0] = 1.0
    state[0] = ContextState.PRESENT

    cols = [
        "isolate",
        "feature_id",
        "feature_kind",
        "state",
        "copy_number",
        *sorted(set(ENCODER_GAP.values())),
        *[ENCODER_INPUT[e][0] for e in ENCODERS],
    ]
    pf = pq.ParquetFile(store / "index/isolate_gene.parquet")
    acc_frames: list[pd.DataFrame] = []
    off_index_rows = 0
    off_index_features: set[str] = set()
    n_rows = 0
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        n_rows += len(df)
        ids = {
            enc: vocabs[enc].ids(sha_array(df[ENCODER_INPUT[enc][0]]))
            for enc in ENCODERS
        }
        gaps = {}
        for enc in ENCODERS:
            g = df[ENCODER_GAP[enc]].map(lambda v: int(INDEX_GAP[v]) if v else 0)
            gaps[enc] = g.to_numpy(dtype=np.int8)
            both = (ids[enc] >= 0) & (gaps[enc] > 0)
            neither = (ids[enc] < 0) & (gaps[enc] == 0)
            if both.any() or neither.any():
                raise ValueError(
                    f"{enc}: {int(both.sum())} rows with key and gap, "
                    f"{int(neither.sum())} with neither"
                )
        is_ref = (df["feature_kind"] == "reference_gene").to_numpy()
        gidx = df["feature_id"].map(gene_pos)
        on_index = is_ref & gidx.notna().to_numpy()
        off = is_ref & ~on_index
        off_index_rows += int(off.sum())
        off_index_features |= set(df.loc[off, "feature_id"])
        c = df["isolate"].map(ctx_row).to_numpy()
        if pd.isna(c).any():
            raise KeyError("an index isolate is not among the contexts")
        c = c.astype(np.int64)
        gi = gidx.to_numpy()[on_index].astype(np.int64)
        ci = c[on_index]
        for e, enc in enumerate(ENCODERS):
            key_ids[ci, gi, e] = ids[enc][on_index]
            key_gap[ci, gi, e] = gaps[enc][on_index]
        gene_copy[ci, gi] = df["copy_number"].to_numpy(dtype=np.float32)[on_index]
        state[ci, gi] = (
            df.loc[on_index, "state"].map(lambda s: int(STATE_CODE[s])).to_numpy()
        )
        acc = df.loc[
            df["feature_kind"] == "accessory_orf",
            ["isolate", "feature_id", "copy_number"],
        ].copy()
        sel = (df["feature_kind"] == "accessory_orf").to_numpy()
        for e, enc in enumerate(ENCODERS):
            acc[f"key_{e}"] = ids[enc][sel]
            acc[f"gap_{e}"] = gaps[enc][sel]
        acc_frames.append(acc)
        print(f"row group {rg + 1}/{pf.num_row_groups}: {n_rows} rows", flush=True)

    np.savez(
        out / "contexts.npz",
        key_ids=key_ids,
        key_gap=key_gap,
        gene_copy=gene_copy,
        state=state,
    )
    (out / "contexts.json").write_text(
        json.dumps({"context_ids": list(context_ids), "genes": genes})
    )
    accessory = pd.concat(acc_frames, ignore_index=True)
    accessory.to_parquet(out / "accessory.parquet", index=False)

    # reporter
    windows = pd.read_parquet(
        Path(osp.dirname(osp.dirname(osp.abspath(__file__))))
        / "results/vaishnav_promoter_windows.parquet",
        columns=["variant_id", "upstream_window_sha256"],
    )
    up_ids = vocabs["fudt_species_upstream"].ids(
        sha_array(windows["upstream_window_sha256"])
    )
    pd.DataFrame({"variant_id": windows["variant_id"], "up_key": up_ids}).to_parquet(
        out / "reporter.parquet", index=False
    )
    rep_rows = vaishnav[vaishnav["source"] == "reporter"].set_index("encoder")
    reporter = {
        "feature_id": str(rep_rows["label"].iloc[0]),
        "downstream": int(
            vocabs["fudt_species_downstream"].ids(
                [rep_rows.loc["fudt_species_downstream", "sha256"]]
            )[0]
        ),
        "protein": int(
            vocabs["prot_t5_xl_uniref50"].ids(
                [rep_rows.loc["prot_t5_xl_uniref50", "sha256"]]
            )[0]
        ),
        "calm": int(vocabs["calm"].ids([rep_rows.loc["calm", "sha256"]])[0]),
    }
    (out / "reporter.json").write_text(json.dumps(reporter, indent=1))

    no_allele = (state[1:] == ContextState.NO_ALLELE).sum(axis=1)
    if len(set(no_allele.tolist())) != 1:
        raise ValueError(f"no-allele genes vary by isolate: {set(no_allele.tolist())}")
    st = os.stat(store / "index/isolate_gene.parquet")
    record = RepresentationBuildRecord(
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        wall_seconds=round(time.time() - t0, 1),
        host=socket.gethostname(),
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        torchcell_commit=git_commit(),
        isolate_store=str(store),
        index_parquet_bytes=st.st_size,
        index_parquet_mtime=st.st_mtime,
        gene_set_sha256=hashlib.sha256(json.dumps(genes).encode()).hexdigest(),
        n_genes=n_genes,
        contexts=n_ctx,
        vocab_sizes={e: len(v) for e, v in vocabs.items()},
        accessory_rows=len(accessory),
        reporter_variants=len(windows),
        off_index_reference_alleles={
            "rows": off_index_rows,
            "features": len(off_index_features),
        },
        no_allele_genes_per_isolate=int(no_allele[0]),
        counts={
            "index_rows": n_rows,
            "isolate_gene_slots_with_key": int((key_ids[1:] >= 0).sum()),
            "isolate_gene_slots_gapped": int((key_ids[1:] < 0).sum()),
            "reporter_variants_without_window_key": int((up_ids < 0).sum()),
        },
    )
    (out / "build_record.json").write_text(record.model_dump_json(indent=1))
    print(record.model_dump_json(indent=1))


# --------------------------------------------------------------------------- #
# load and resolve
# --------------------------------------------------------------------------- #
def load_tables(data_root: str) -> tuple[ContextTables, ReporterKeys, list[str]]:
    """The context tables, the reporter keys and the gene list the tables are over."""
    root = rep_root(data_root)
    meta = json.loads((root / "contexts.json").read_text())
    arrays = np.load(root / "contexts.npz")
    acc = pd.read_parquet(root / "accessory.parquet")
    accessory: dict[str, dict[str, AccessoryRow]] = {}
    keys = acc[[f"key_{e}" for e in range(N_ENC)]].to_numpy()
    gaps = acc[[f"gap_{e}" for e in range(N_ENC)]].to_numpy()
    for i, (iso, orf, cn) in enumerate(
        zip(acc["isolate"], acc["feature_id"], acc["copy_number"], strict=True)
    ):
        accessory.setdefault(iso, {})[orf] = AccessoryRow(
            key_ids=tuple(int(k) for k in keys[i]),  # type: ignore[arg-type]
            gaps=tuple(int(g) for g in gaps[i]),  # type: ignore[arg-type]
            copy_number=None if pd.isna(cn) else float(cn),
        )
    tables = ContextTables(
        context_ids=tuple(meta["context_ids"]),
        key_ids=arrays["key_ids"],
        key_gap=arrays["key_gap"],
        gene_copy=arrays["gene_copy"],
        state=arrays["state"],
        accessory=accessory,
    )
    rep = json.loads((root / "reporter.json").read_text())
    win = pd.read_parquet(root / "reporter.parquet")
    reporter = ReporterKeys(
        feature_id=rep["feature_id"],
        upstream_by_variant={
            v: int(k)
            for v, k in zip(win["variant_id"], win["up_key"], strict=True)
            if k >= 0
        },
        downstream=rep["downstream"],
        protein=rep["protein"],
        calm=rep["calm"],
    )
    return tables, reporter, list(meta["genes"])


def load_resolver(
    data_root: str,
    config: RepresentationConfig,
    parent_vocab: list[str],
    genes: list[str],
    gff_path: str,
    tables: tuple[ContextTables, ReporterKeys, list[str]] | None = None,
) -> RepresentationResolver:
    """A resolver over ``genes`` (must equal the gene list the tables were built over)."""
    ctx, reporter, built_genes = (
        tables if tables is not None else load_tables(data_root)
    )
    if built_genes != genes:
        raise ValueError("the representation tables were built over another gene index")
    return RepresentationResolver(
        config, gff_positions(gff_path, genes), ctx, reporter, parent_vocab
    )


def vector_stores(data_root: str) -> dict[str, AlleleVectorStore]:
    """One lazy reader per encoder over ``alleles/<encoder>.h5``."""
    root = rep_root(data_root)
    return {
        enc: AlleleVectorStore(
            str(iso_root(data_root) / "alleles" / f"{enc}.h5"),
            KeyVocabulary(np.load(root / f"vocab_{enc}.npy")),
        )
        for enc in ENCODERS
    }


def gene_vectors(
    r: RecordRepresentation, stores: dict[str, AlleleVectorStore]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per encoder, (gene indices with a key, their fp16 vectors). Gaps are not filled;
    a key whose vector is not in the table raises (``AlleleVectorStore.vectors``).
    """
    out = {}
    for e, enc in enumerate(ENCODERS):
        idx = np.nonzero(r.key_ids[:, e] >= 0)[0]
        out[enc] = (idx, stores[enc].vectors(r.key_ids[idx, e]))
    return out


def isolate_context(name: str, context_ids: tuple[str, ...]) -> str:
    """A record's isolate name -> its context id (the index writes 93 as ``SACE_*``)."""
    if name in context_ids:
        return name
    if f"SACE_{name}" in context_ids:
        return f"SACE_{name}"
    raise KeyError(f"isolate {name!r} has no reference context")


def resolve_genotype(
    resolver: RepresentationResolver,
    genotype: dict[str, Any],
    gene_idx: Callable[[str], int | None],
) -> RecordRepresentation:
    """One raw record's genotype -> its representation under the resolver's config."""
    if "perturbations" not in genotype:
        parents = (genotype["parent_1"], genotype["parent_2"])
        return resolver.segregant(
            [Block.model_validate(b) for b in genotype["blocks"]],
            (
                (parents[0]["name"], parents[0]["peter_strain_id"]),
                (parents[1]["name"], parents[1]["peter_strain_id"]),
            ),
        )
    isolate = genotype.get("natural_isolate")
    marks: list[tuple[int, ChangeType]] = []
    accessory: list[tuple[str, float | None]] = []
    variant_id: str | None = None
    outside = 0
    for p in genotype["perturbations"]:
        ct = change_type_of(p["perturbation_type"])
        if ct == ChangeType.present:
            accessory.append((p["pangenome_orf_id"], p["copy_number"]))
            continue
        if ct == ChangeType.promoter:
            if variant_id is not None:
                raise ValueError("two promoter reporters in one genotype")
            variant_id = p["promoter_variant_id"]
            continue
        if isolate is None and ct in (
            ChangeType.variant,
            ChangeType.loss_of_function,
            ChangeType.absent,
        ):
            raise ValueError(f"{ct} outside a natural isolate")
        idx = gene_idx(p["systematic_gene_name"])
        if idx is None:
            outside += 1
            continue
        marks.append((idx, ct))
    if variant_id is not None:
        if marks or isolate is not None:
            raise ValueError("a promoter record with other marks")
        return resolver.promoter(variant_id)
    if isolate is not None:
        return resolver.isolate(
            isolate_context(isolate["isolate"], resolver.tables.context_ids),
            marks,
            isolate["ploidy"],
            isolate["chromosome_copy_changes"],
            accessory,
            outside,
        )
    if accessory:
        raise ValueError("an accessory ORF outside a natural isolate")
    return resolver.deletion(marks, outside)


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", required=True, choices=["build"])
    args = parser.parse_args()
    load_dotenv()
    if args.stage == "build":
        build(os.environ["DATA_ROOT"])


if __name__ == "__main__":
    main()
