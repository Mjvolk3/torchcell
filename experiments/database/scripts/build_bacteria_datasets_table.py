# experiments/database/scripts/build_bacteria_datasets_table.py
# [[experiments.database.scripts.build_bacteria_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_bacteria_datasets_table
r"""Instances vs Signal (gzip) for the bacterial datasets, plus the Signal controls.

The bacterial counterpart of ``build_supported_datasets_table.py`` (issue #1098), with
the control columns of issue #1099 measured in the same pass. Nothing here is
hand-curated: the row list is the adapter map filtered by organism (what a KG build
serves), the LMDB path is each loader's default root, and every column is read from
the built store.

Per dataset, over the dev-tree LMDB (``$DATA_ROOT/data/torchcell/<slug>/processed/lmdb``):

- ``instances``: record count (LMDB entry count).
- ``genotypes`` / ``environments``: distinct stored genotype and environment payloads.
- ``signal_bytes``: gzip of the concatenated instances (``instance_bytes``, the yeast
  table's definition: perturbation + environment + phenotype, sorted keys).
- ``structure_bytes``: the same serialization with every leaf replaced by a type token
  (``"s"`` string, ``0`` number, ``"b"`` bool, ``null`` None). Keys, nesting and list
  lengths survive; values do not. The cost of the schema copies.
- ``noise_bytes``: every numeric leaf's digits replaced by i.i.d. uniform digits
  (sign, decimal point, exponent and digit count kept; strings kept). Pure noise at the
  stored precision: the gzip ceiling for that structure.
- ``binary_bytes``: every numeric leaf z-scored per key path across the dataset and
  thresholded at 0 (sigmoid then round), serialized as ``1`` / ``0`` in place.
- ``binary_packed_bytes``: gzip of just those bits, packed 8 per byte in record order,
  with no structure around them. The one-bit floor on value-carried information.
- ``shuffled_bytes``: every numeric leaf replaced by a draw without replacement from
  the same key path's values across the dataset (marginals kept, joint structure
  destroyed). Between the real Signal and the noise ceiling.

Two streaming passes per LMDB (``zlib.compressobj`` with the gzip wrapper, level 6,
bounded memory): pass 1 computes the instance/structure/noise streams and the per-path
moments and value pools; pass 2 computes the binary and shuffled streams. Datasets run
in a process pool, largest first. Results are cached per LMDB fingerprint so a rerun
recomputes only rebuilt stores.

Run from the repo root (``$DATA_ROOT`` from .env):
  python experiments/database/scripts/build_bacteria_datasets_table.py
  python experiments/database/scripts/build_bacteria_datasets_table.py --organisms ecoli
  python experiments/database/scripts/build_bacteria_datasets_table.py --workers 24 --max-gb 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import pickle
import random
import re
import time
import zlib
from array import array
from collections.abc import Callable, Iterator
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from torchcell.data.experiment_dataset import ALL_ORGANISMS, Organism, parse_organisms
from torchcell.database.build_dataset_lmdb import dataset_default_root
from torchcell.knowledge_graphs.dataset_adapter_map import build_adapter_map
from torchcell.paper.tables import (
    instance_bytes,
    lmdb_fingerprint,
    phenotype_descriptor,
    read_first_record,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
CACHE_PATH = SCRIPT.parent / "dataset_probe_cache.json"
# Bump when any extractor definition changes: a cached row is only reused when both the
# LMDB fingerprint and this tag match.
PROBE_VARIANT = "probes-v1"
GZIP_LEVEL = 6
SEED = 0
_DIGITS = "0123456789"
_DIGIT_RE = re.compile(r"\d")

# Display section per stored ``experiment_type``. A type not listed here is shown under
# its raw name, so a new experiment class is visible rather than silently grouped.
SECTION_BY_EXPERIMENT_TYPE = {
    "bacterial_environment_response": "Environmental / chemogenomic response",
    "bacterial_fitness": "Fitness",
    "bacterial_gene_essentiality": "Viability",
    "bacterial_gene_interaction": "Gene interaction",
    "bacterial_growth_rate": "Growth rate / doubling time",
    "bacterial_doubling_time": "Growth rate / doubling time",
    "bacterial_rnaseq_expression": "Expression (RNA-seq)",
    "bacterial_protein_abundance": "Protein abundance",
    "bacterial_protein_fold_change": "Protein abundance",
    "bacterial_metabolite": "Metabolite",
    "bacterial_morphology": "Morphology",
    "bacterial_fitness_contrast": "Fitness",
    "bacterial_graded_gene_essentiality": "Viability",
    "bacterial_phosphosite_intensity": "Phosphoproteome",
    "mrna_number_fraction": "Expression (RNA-seq)",
    "promoter_activity": "Promoter activity",
    "protein_synthesis_rate": "Protein synthesis / turnover",
    "protein_turnover": "Protein synthesis / turnover",
    "product_titer": "Product titer",
    "flux": "Flux",
    # yeast experiment types (the 2026-07-17 paper table curates these by hand)
    "fitness": "Fitness",
    "segregant_growth": "Fitness",
    "segregant_perturbation": "Fitness",
    "gene interaction": "Gene interaction",
    "synthetic lethality": "Viability",
    "synthetic rescue": "Viability",
    "gene essentiality": "Viability",
    "environment_response": "Environmental / chemogenomic response",
    "strain_environment_response": "Environmental / chemogenomic response",
    "calmorph": "Morphology",
    "visual_score": "Morphology",
    "microarray_expression": "Expression (microarray)",
    "microarray_expression_score": "Expression (microarray)",
    "induction_timecourse_expression": "Expression (microarray)",
    "rnaseq_expression": "Expression (RNA-seq)",
    "segregant_expression": "Expression (RNA-seq)",
    "pooled_sort_seq_expression": "Expression (RNA-seq)",
    "pseudobulk_expression": "Expression (RNA-seq)",
    "pseudobulk_umi_count": "Expression (RNA-seq)",
    "metabolite": "Metabolite",
    "protein_abundance": "Protein abundance",
    "yeast_protein_turnover": "Protein synthesis / turnover",
    "genome_architecture": "Genome architecture",
    "life_cycle": "Life cycle",
}


def section_for(experiment_type: str) -> str:
    """Display section of a stored ``experiment_type``; unmapped types show raw."""
    return SECTION_BY_EXPERIMENT_TYPE.get(experiment_type, experiment_type)


class Organisms(BaseModel):
    """The organism set a run covers, as the build spells it."""

    members: frozenset[Organism]

    def label(self) -> str:
        """``ecoli,pputida``: the sorted comma-joined names."""
        return ",".join(sorted(o.value for o in self.members))


class PathMoments(BaseModel):
    """Running mean/variance of one numeric key path (Welford), NaN excluded."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, x: float) -> None:
        """Fold one value into the running moments."""
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)


class DatasetProbeRecord(BaseModel):
    """One dataset's measured row: the yeast table's columns plus the controls."""

    organism: str
    dataset_class: str
    slug: str
    section: str
    experiment_type: str
    label_name: str
    measurement_type: str
    shape: str
    graph_role: str
    built: bool
    probed: bool
    instances: int
    genotypes: int = 0
    environments: int = 0
    signal_bytes: int = 0
    structure_bytes: int = 0
    noise_bytes: int = 0
    binary_bytes: int = 0
    binary_packed_bytes: int = 0
    shuffled_bytes: int = 0
    numeric_leaves: int = 0
    numeric_paths: int = 0
    lmdb_bytes: int = 0
    seconds: float = 0.0


class _CacheEntry(BaseModel):
    key: str
    variant: str
    record: DatasetProbeRecord


class ProbeCache(BaseModel):
    """JSON memo of a finished row per slug, keyed by LMDB fingerprint + variant."""

    path: Path
    entries: dict[str, _CacheEntry] = Field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> ProbeCache:
        """An existing cache JSON, or an empty cache for a new path."""
        if path.exists():
            raw = json.loads(path.read_text())
            return cls(path=path, entries={k: _CacheEntry(**v) for k, v in raw.items()})
        return cls(path=path)

    def save(self) -> None:
        """Persist the cache (sorted for a stable diff)."""
        payload = {
            k: v.model_dump(mode="json") for k, v in sorted(self.entries.items())
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


# --- leaf transforms ---------------------------------------------------------


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def structure_view(obj: Any) -> Any:
    """Keys, nesting and list lengths kept; every leaf collapsed to a type token."""
    if isinstance(obj, dict):
        return {k: structure_view(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [structure_view(v) for v in obj]
    if isinstance(obj, bool):
        return "b"
    if _is_number(obj):
        return 0
    if obj is None:
        return None
    return "s"


def _noise_number(x: int | float, rng: random.Random) -> int | float:
    """Same textual shape as ``repr(x)`` with every digit redrawn uniformly."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return x
    s = _DIGIT_RE.sub(lambda _: rng.choice(_DIGITS), repr(x))
    if isinstance(x, int):
        s = s.lstrip("-")
        s = (("-" if x < 0 else "") + (s.lstrip("0") or "0")) if s else "0"
        return int(s)
    return float(s)


def noise_view(obj: Any, rng: random.Random) -> Any:
    """Numeric leaves replaced by digit noise at the stored precision; the rest kept."""
    if isinstance(obj, dict):
        return {k: noise_view(v, rng) for k, v in obj.items()}
    if isinstance(obj, list):
        return [noise_view(v, rng) for v in obj]
    if _is_number(obj):
        return _noise_number(obj, rng)
    return obj


def walk_numeric(
    obj: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], int | float]]:
    """Yield ``(key path, value)`` for every numeric leaf, list indices as ``[]``."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_numeric(v, (*path, str(k)))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_numeric(v, (*path, "[]"))
    elif _is_number(obj):
        yield path, obj


def substitute_numeric(
    obj: Any,
    fn: Callable[[tuple[str, ...], int | float], Any],
    path: tuple[str, ...] = (),
) -> Any:
    """Rebuild ``obj`` with every numeric leaf replaced by ``fn(path, value)``."""
    if isinstance(obj, dict):
        return {k: substitute_numeric(v, fn, (*path, str(k))) for k, v in obj.items()}
    if isinstance(obj, list):
        return [substitute_numeric(v, fn, (*path, "[]")) for v in obj]
    if _is_number(obj):
        return fn(path, obj)
    return obj


def _dumps(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, default=str).encode()


def _digest(obj: Any) -> bytes:
    return hashlib.blake2b(_dumps(obj), digest_size=16).digest()


def _gz() -> Any:
    return zlib.compressobj(GZIP_LEVEL, zlib.DEFLATED, 16 + zlib.MAX_WBITS)


def _pack_bits(bits: bytearray) -> bytes:
    out = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i : i + 8]
        byte = 0
        for b in chunk:
            byte = (byte << 1) | b
        byte <<= 8 - len(chunk)
        out.append(byte)
    return bytes(out)


# --- the two passes ----------------------------------------------------------


def _records(lmdb_dir: str) -> Iterator[dict[str, Any]]:
    env = lmdb.open(lmdb_dir, readonly=True, lock=False, max_readers=2048)
    try:
        with env.begin() as txn:
            for _, v in txn.cursor():
                yield pickle.loads(v)["experiment"]
    finally:
        env.close()


def probe_lmdb(lmdb_dir: str, label: str, log: Callable[[str], None]) -> dict[str, int]:
    """Both passes over one LMDB; returns every measured byte count and the tallies."""
    rng = random.Random(SEED)
    gz_signal, gz_structure, gz_noise = _gz(), _gz(), _gz()
    sig = struct = noise = 0
    genotypes: set[bytes] = set()
    environments: set[bytes] = set()
    moments: dict[tuple[str, ...], PathMoments] = {}
    pools: dict[tuple[str, ...], array] = {}
    n = 0
    t0 = time.time()
    for rec in _records(lmdb_dir):
        n += 1
        sig += len(gz_signal.compress(instance_bytes({"experiment": rec})))
        struct += len(gz_structure.compress(_dumps(structure_view(rec))))
        noise += len(gz_noise.compress(_dumps(noise_view(rec, rng))))
        genotypes.add(_digest(rec["genotype"]))
        environments.add(_digest(rec["environment"]))
        for path, x in walk_numeric(rec):
            pool = pools.get(path)
            if pool is None:
                pool = pools[path] = array("d")
                moments[path] = PathMoments()
            pool.append(float(x))
            if not math.isnan(x):
                moments[path].push(float(x))
        if n % 250_000 == 0:
            log(f"    [{label}] pass 1: {n:,} records ({time.time() - t0:.0f}s)")
    sig += len(gz_signal.flush())
    struct += len(gz_structure.flush())
    noise += len(gz_noise.flush())
    numeric_leaves = sum(len(p) for p in pools.values())
    log(
        f"    [{label}] pass 1 done: {n:,} records, {len(pools):,} numeric paths, "
        f"{numeric_leaves:,} numeric leaves ({time.time() - t0:.0f}s)"
    )

    # Shuffled pools: one permutation per path, consumed in record order in pass 2.
    cursors: dict[tuple[str, ...], int] = {}
    for path, pool in pools.items():
        lst = pool.tolist()
        rng.shuffle(lst)
        pools[path] = array("d", lst)
        cursors[path] = 0
    means = {p: m.mean for p, m in moments.items()}
    # A constant path (variance 0) has no sign to carry; it binarizes to 0 everywhere.
    constant = {p for p, m in moments.items() if m.n < 2 or m.m2 == 0.0}

    def binarize(path: tuple[str, ...], x: int | float) -> int:
        if path in constant or (isinstance(x, float) and math.isnan(x)):
            return 0
        return 1 if x > means[path] else 0

    def shuffled(path: tuple[str, ...], x: int | float) -> int | float:
        i = cursors[path]
        cursors[path] = i + 1
        v = pools[path][i]
        return int(v) if isinstance(x, int) and not math.isnan(v) else v

    gz_binary, gz_shuffled, gz_packed = _gz(), _gz(), _gz()
    binary = shuf = packed = 0
    bits = bytearray()
    t1 = time.time()
    m = 0
    for rec in _records(lmdb_dir):
        m += 1
        binary += len(gz_binary.compress(_dumps(substitute_numeric(rec, binarize))))
        shuf += len(gz_shuffled.compress(_dumps(substitute_numeric(rec, shuffled))))
        for path, x in walk_numeric(rec):
            bits.append(binarize(path, x))
        if len(bits) >= 1 << 20:
            packed += len(gz_packed.compress(_pack_bits(bits)))
            bits = bytearray()
        if m % 250_000 == 0:
            log(f"    [{label}] pass 2: {m:,} records ({time.time() - t1:.0f}s)")
    packed += len(gz_packed.compress(_pack_bits(bits)))
    binary += len(gz_binary.flush())
    shuf += len(gz_shuffled.flush())
    packed += len(gz_packed.flush())
    if m != n:
        raise RuntimeError(f"{label}: pass 2 saw {m} records, pass 1 saw {n}")
    log(f"    [{label}] pass 2 done ({time.time() - t1:.0f}s)")
    return {
        "instances": n,
        "genotypes": len(genotypes),
        "environments": len(environments),
        "signal_bytes": sig,
        "structure_bytes": struct,
        "noise_bytes": noise,
        "binary_bytes": binary,
        "binary_packed_bytes": packed,
        "shuffled_bytes": shuf,
        "numeric_leaves": numeric_leaves,
        "numeric_paths": len(pools),
    }


# --- per-dataset driver --------------------------------------------------------


class Job(BaseModel):
    """One dataset to describe (and probe when its LMDB is under the size cap)."""

    organism: str
    dataset_class: str
    slug: str
    lmdb_dir: str
    lmdb_bytes: int
    probe: bool


def _lmdb_size(lmdb_dir: str) -> int:
    return sum(osp.getsize(osp.join(lmdb_dir, f)) for f in os.listdir(lmdb_dir))


def describe(job: Job) -> DatasetProbeRecord:
    """The schema-derived columns from the first record, with the LMDB entry count."""
    first = read_first_record(job.lmdb_dir)
    exp = first["experiment"]
    shape, role = phenotype_descriptor(first)
    ph = exp["phenotype"]
    env = lmdb.open(job.lmdb_dir, readonly=True, lock=False)
    try:
        n = env.stat()["entries"]
    finally:
        env.close()
    etype = exp["experiment_type"]
    return DatasetProbeRecord(
        organism=job.organism,
        dataset_class=job.dataset_class,
        slug=job.slug,
        section=section_for(etype),
        experiment_type=etype,
        label_name=ph["label_name"],
        measurement_type=str(ph.get("measurement_type")),
        shape=shape,
        graph_role=role,
        built=True,
        probed=False,
        instances=n,
        lmdb_bytes=job.lmdb_bytes,
    )


def run_job(job: Job) -> DatasetProbeRecord:
    """Worker entry: describe, then (when asked) probe; prints progress to stdout."""
    t0 = time.time()
    rec = describe(job)
    if not job.probe:
        print(
            f"  [{job.slug}] described only (LMDB {job.lmdb_bytes / 1e9:.2f} GB over cap)",
            flush=True,
        )
        return rec
    print(
        f"  [{job.slug}] start ({job.lmdb_bytes / 1e6:.0f} MB, {rec.instances:,} records)",
        flush=True,
    )
    measured = probe_lmdb(job.lmdb_dir, job.slug, lambda s: print(s, flush=True))
    if measured["instances"] != rec.instances:
        raise RuntimeError(
            f"{job.slug}: walked {measured['instances']} records, stat says {rec.instances}"
        )
    out = rec.model_copy(
        update={**measured, "probed": True, "seconds": time.time() - t0}
    )
    print(
        f"  [{job.slug}] done in {out.seconds:.0f}s: signal {out.signal_bytes:,} B, "
        f"structure {out.structure_bytes:,} B, noise {out.noise_bytes:,} B, "
        f"binary {out.binary_bytes:,} B, packed {out.binary_packed_bytes:,} B, "
        f"shuffled {out.shuffled_bytes:,} B",
        flush=True,
    )
    return out


def enumerate_jobs(
    data_root: str, organisms: Organisms, max_gb: float | None
) -> tuple[list[Job], list[DatasetProbeRecord]]:
    """One job per mapped dataset whose LMDB exists; a not-built row otherwise."""
    jobs: list[Job] = []
    not_built: list[DatasetProbeRecord] = []
    for cls in build_adapter_map(include_private=False, organisms=organisms.members):
        root = dataset_default_root(cls)
        lmdb_dir = osp.join(data_root, root, "processed", "lmdb")
        slug = osp.basename(osp.normpath(root))
        if not osp.exists(osp.join(lmdb_dir, "data.mdb")):
            not_built.append(
                DatasetProbeRecord(
                    organism=cls.organism.value,
                    dataset_class=cls.__name__,
                    slug=slug,
                    section="not built",
                    experiment_type="",
                    label_name="",
                    measurement_type="",
                    shape="",
                    graph_role="",
                    built=False,
                    probed=False,
                    instances=0,
                )
            )
            continue
        size = _lmdb_size(lmdb_dir)
        jobs.append(
            Job(
                organism=cls.organism.value,
                dataset_class=cls.__name__,
                slug=slug,
                lmdb_dir=lmdb_dir,
                lmdb_bytes=size,
                probe=max_gb is None or size <= max_gb * 1e9,
            )
        )
    jobs.sort(key=lambda j: -j.lmdb_bytes)
    return jobs, not_built


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--organisms",
        default="ecoli,pputida",
        help=f"comma-separated subset of {','.join(o.value for o in ALL_ORGANISMS)}",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--max-gb",
        type=float,
        default=None,
        help="describe but do not probe an LMDB larger than this many GB",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON (default results/pre-build/<date>/<organisms>_datasets.json)",
    )
    args = ap.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    organisms = Organisms(members=parse_organisms(args.organisms))
    jobs, not_built = enumerate_jobs(data_root, organisms, args.max_gb)
    cache = ProbeCache.load(CACHE_PATH)

    pending: list[Job] = []
    records: dict[str, DatasetProbeRecord] = {}
    for job in jobs:
        hit = cache.entries.get(job.slug)
        fp = lmdb_fingerprint(job.lmdb_dir)
        if (
            hit is not None
            and hit.key == fp
            and hit.variant == PROBE_VARIANT
            and (hit.record.probed or not job.probe)
        ):
            records[job.slug] = hit.record.model_copy(
                update={"section": section_for(hit.record.experiment_type)}
            )
        else:
            pending.append(job)
    print(
        f"{len(jobs)} built datasets ({organisms.label()}), {len(records)} cached, "
        f"{len(pending)} to run, {len(not_built)} not built; "
        f"{sum(j.lmdb_bytes for j in pending) / 1e9:.1f} GB to scan on {args.workers} workers",
        flush=True,
    )

    ctx = get_context("fork")
    with ctx.Pool(processes=args.workers) as pool:
        for rec in pool.imap_unordered(run_job, pending):
            records[rec.slug] = rec
            job = next(j for j in pending if j.slug == rec.slug)
            cache.entries[rec.slug] = _CacheEntry(
                key=lmdb_fingerprint(job.lmdb_dir), variant=PROBE_VARIANT, record=rec
            )
            cache.save()

    ordered = [
        records[j.slug]
        for j in sorted(jobs, key=lambda j: (j.organism, j.dataset_class))
    ]
    sections = sorted({r.section for r in ordered})
    out = (
        args.out
        or RESULTS
        / "pre-build"
        / str(date.today())
        / f"{organisms.label().replace(',', '-')}_datasets.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "pre-build",
        "date": str(date.today()),
        "organisms": sorted(o.value for o in organisms.members),
        "probe_variant": PROBE_VARIANT,
        "gzip_level": GZIP_LEVEL,
        "seed": SEED,
        "sections": sections,
        "datasets": [r.model_dump() for r in ordered],
        "not_built": [r.model_dump() for r in not_built],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
