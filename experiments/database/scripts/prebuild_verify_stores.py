# experiments/database/scripts/prebuild_verify_stores.py
# [[experiments.database.scripts.prebuild_verify_stores]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/prebuild_verify_stores
r"""Prove a sample of dev stores built from the integrated ontology branch is RIGHT.

Lane P of the ontology unification (Path U, 2026-10-15). The sweep over 170 dev stores
and the full KG build that follows cost about two days of node time, so before the
owner commits to them a sample of stores is built from the integrated branch under a
private data root (``prebuild_verify_store_array.slurm``), its CSVs are generated
through the real per-adapter array task (``database/slurm/scripts/
gilahyper_generate_adapter_array-slurm_docker.slurm`` in its ``tc-neo4j-gen-<i>``
container, pointed at the private tree; the host's biocypher 0.5.43 lacks the
``_initialize_writer`` the sink needs, so generation runs only in the container), and
this script walks EVERY record of every store and answers, with counts:

* the record count against the served KG 3.0 release and against the shared dev tree
  that `main` built on 2026-10-13 (a count below either is a dropped record);
* every perturbation carries a ``gene_namespace`` and it is the one the dataset's host
  implies; every genome reference carries an ``assembly_set`` or a typed gap with a
  reason and a ``looked_in``;
* every ``GeneAdditionPerturbation`` either resolves its ``placement`` to a deposited
  contig (the construct registry re-hashes the bytes) or carries a typed gap with a
  reason and a ``looked_in``;
* no record carries a retired experiment-type tag;
* the records of the served 3.0 store (``prebuild_verify_served_dump.py``) are all
  still present, compared by CONTENT (gene names, phenotype value, media, temperature,
  environment perturbations), since node ids re-key by design;
* the generated CSVs: rows per label, ``host_genome`` on every perturbation node
  against the ``assembly_set`` of the genome node, experiment-id overlap with 3.0;
* wall time and peak RSS per store build and per generation task, read off the slurm
  logs, against the tier sizes.

It imports the unified schema's ``RETIRED_EXPERIMENT_TYPES`` and ``AssemblySet``-era
record shapes, so it runs on ``ontology-unify`` (or on main once that branch lands) and
stops with an ``ImportError`` on an older schema rather than checking a weaker rule.

Usage (from the repo root, the torchcell environment)::

    python experiments/database/scripts/prebuild_verify_stores.py \\
        --data-root /scratch/projects/torchcell-scratch/dev/prebuild-verify \\
        --shared-data-root /scratch/projects/torchcell-scratch \\
        --served-dir <dir written by prebuild_verify_served_dump.py> \\
        --biocypher-out <dir>/biocypher-out/<name> \\
        --slurm-logs <dir>/slurm-output \\
        --release database/releases/2026.10.06-4b293d34.json \\
        --out experiments/database/results/prebuild_verify_stores.json [--table]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os.path as osp
import pickle
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lmdb
from pydantic import BaseModel, ConfigDict, Field

from torchcell.data.experiment_dataset import InternedDict, resolve_interned
from torchcell.datamodels.interned_constant import POINTER_KEY
from torchcell.datamodels.schema import (
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
    RETIRED_EXPERIMENT_TYPES,
)
from torchcell.sequence.construct.registry import (
    load_construct_manifest,
    resolve_contig,
)

GENE_ADDITION_TYPES = frozenset({"gene_addition", "heterologous_pathway"})

#: The sample: dataset class, store slug, construct set a placement may resolve in,
#: the host namespace every perturbation must carry, the assembly set every reference
#: must pin, and why the dataset is in the sample.
SAMPLE: list[dict[str, Any]] = [
    {
        "dataset": "SmfKuzmin2018Dataset",
        "slug": "smf_kuzmin2018",
        "namespace": "scerevisiae_s288c_sgd_systematic",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "plain yeast SGA deletion store: the P1 class-default path",
    },
    {
        "dataset": "SmfODuibhir2014Dataset",
        "slug": "smf_oduibhir2014",
        "namespace": "scerevisiae_s288c_sgd_systematic",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "yeast loader that can emit the relaxed ReleasedIdentifierPerturbation "
        "leaf (no namespace default, S288C stated explicitly); its released names all "
        "resolve, so the built store holds KanMxDeletionPerturbation only",
    },
    {
        "dataset": "CaudalPanTranscriptome2024Dataset",
        "slug": "caudal_pantranscriptome2024",
        "namespace": "scerevisiae_peter2018_pangenome_orf",
        "also_namespaces": ["scerevisiae_s288c_sgd_systematic"],
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "the only pangenome-namespace member (presence/absence calls), beside "
        "S288C-keyed loss-of-function and deletion calls in the same genotype; "
        "natural isolates, no perturbation lane in its conf; large generation tier",
    },
    {
        "dataset": "BetaxanthinCachera2023Dataset",
        "slug": "betaxanthin_cachera2023",
        "construct_set": "cachera2023_cri_spa_plasmids",
        "namespace": "scerevisiae_s288c_sgd_systematic",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "lane G's one real ConstructPlacement, pointing at the deposited pBTX1",
    },
    {
        "dataset": "CarotenoidOzaydin2013Dataset",
        "slug": "carotenoid_ozaydin2013",
        "namespace": "scerevisiae_s288c_sgd_systematic",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "gene addition whose placement is a typed gap (not_released_by_primary)",
    },
    {
        "dataset": "GrowthRateCampos2018Dataset",
        "slug": "ecoli_growth_rate_campos2018",
        "namespace": "ecoli_k12_bw25113_locus_tag",
        "assembly_set": "ecoli_K12_BW25113_ASM75055v1",
        "why": "bacterial Keio store on the renamed DeletionStrainPerturbation leaf; "
        "bacterial_fitness retired into fitness; assembly pin on the reference",
    },
    {
        "dataset": "EnvChemgenNichols2011Dataset",
        "slug": "ecoli_env_chemgen_nichols2011",
        "namespace": "ecoli_k12_bw25113_locus_tag",
        "assembly_set": "ecoli_K12_BW25113_ASM75055v1",
        "why": "bacterial_environment_response retired into environment_response; "
        "a bacterial released-identifier leaf; the largest store of the sample",
    },
    {
        "dataset": "CrispriChemgenSmith2016Dataset",
        "slug": "crispri_chemgen_smith2016",
        "namespace": "scerevisiae_s288c_sgd_systematic",
        "assembly_set": "sgd_S288C_R64-4-1_20230830",
        "why": "spans a placed CrisprInterferencePerturbation AND a "
        "ReleasedIdentifierPerturbation, a shape no one designed for",
    },
    {
        "dataset": "IsoprenolTiterCarruthers2025Dataset",
        "slug": "isoprenol_titer_carruthers2025",
        "namespace": "pputida_kt2440_locus_tag",
        "assembly_set": "pputida_KT2440_ASM756v2",
        "why": "the dataset whose GeneAdditionPerturbation rows the old conf-keyed "
        "lane dropped (audit C 1.3); P. putida namespace; typed placement gap",
    },
]


class GapCheck(BaseModel):
    """Counts of one gapped field over every occurrence of its owner."""

    model_config = ConfigDict(frozen=True)

    present: int = 0
    gapped_with_reason_and_looked_in: int = 0
    gapped_without_reason: int = 0
    absent_no_gap: int = 0


class PlacementCheck(BaseModel):
    """Gene-addition placements: resolved to a deposited contig or typed gap."""

    model_config = ConfigDict(frozen=True)

    additions: int = 0
    placed: int = 0
    placed_resolved_to_deposit: int = 0
    placed_unresolved: list[str] = Field(default_factory=list)
    gapped_with_reason_and_looked_in: int = 0
    gapped_without_reason: int = 0
    absent_no_gap: int = 0
    gap_reasons: dict[str, int] = Field(default_factory=dict)
    contigs: dict[str, int] = Field(default_factory=dict)


class StoreCheck(BaseModel):
    """Every count this lane measured on one built store."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    slug: str
    why: str
    n_records: int
    n_distinct_experiments: int
    n_served_3_0: int | None
    n_shared_main_tree: int | None
    experiment_types: dict[str, int]
    reference_types: dict[str, int]
    retired_types_seen: list[str]
    unknown_types_seen: list[str]
    perturbation_leaves_manifest: dict[str, list[str]] | None
    perturbation_types: dict[str, int]
    n_perturbations: int
    namespaces: dict[str, int]
    perturbations_missing_namespace: int
    perturbations_off_expected_namespace: int
    expected_namespace: str
    assembly_sets: dict[str, int]
    assembly: GapCheck
    expected_assembly_set: str
    references_off_expected_assembly: int
    placement: PlacementCheck
    content_served_lost: int | None
    content_served_lost_sample: list[str]
    content_new_only: int | None
    content_main_lost: int | None
    content_main_lost_sample: list[str]
    content_new_only_vs_main: int | None
    served_ids_in_new_csv: int | None
    build_wall_s: float | None
    build_time_s: float | None
    build_peak_rss_gb: float | None
    verify_outcome: str | None
    csv: dict[str, Any] | None
    generation_wall_s: float | None
    generation_peak_mem_gb: float | None
    generation_peak_rss_gb: float | None
    generation_time_v_wall_s: float | None


class _TolerantUnpickler(pickle.Unpickler):
    """Unpickle a record whose enum classes the current schema has retired.

    The shared dev tree was written by `main` under the pre-unification schema, whose
    pickles name enum classes the branch removed (``ReleasedGeneKind``). A content
    comparison needs the VALUE the record stored, not the class, so a missing class
    becomes the identity: the enum's pickled value comes back as its plain string.
    """

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except AttributeError:
            return lambda value: value


def _loads(raw: bytes) -> Any:
    import io

    return _TolerantUnpickler(io.BytesIO(raw)).load()


def iter_records(processed_dir: str) -> Iterator[dict[str, Any]]:
    """Yield every record of a store with its interned pointers resolved."""
    interned: dict[str, Any] = {}
    interned_dir = osp.join(processed_dir, "interned")
    if osp.isdir(interned_dir):
        ienv = lmdb.open(interned_dir, readonly=True, lock=False)
        with ienv.begin() as txn:
            for key, value in txn.cursor():
                ref = key.decode()
                loaded = _loads(value)
                interned[ref] = (
                    InternedDict(loaded, ref) if isinstance(loaded, dict) else loaded
                )
        ienv.close()
    env = lmdb.open(osp.join(processed_dir, "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        n = txn.stat()["entries"]
        for i in range(n):
            raw = txn.get(str(i).encode())
            if raw is None:
                raise KeyError(f"record {i} missing from {processed_dir}")
            yield resolve_interned(_loads(raw), interned)
    env.close()


def lmdb_entries(processed_dir: str) -> int | None:
    """Entry count of a store's records env, or None when there is no store."""
    path = osp.join(processed_dir, "lmdb")
    if not osp.isdir(path):
        return None
    env = lmdb.open(path, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.stat()["entries"])
    env.close()
    return n


def _gap_for(owner: dict[str, Any], field: str) -> dict[str, Any] | None:
    for gap in owner.get("provenance_gaps") or []:
        if gap.get("field") == field:
            return dict(gap)
    return None


def _gap_is_typed(gap: dict[str, Any]) -> bool:
    return bool(gap.get("reason")) and gap.get("looked_in") is not None


def _plain(value: Any) -> Any:
    """A JSON-able copy: enums to values, InternedDict to dict."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "value") and not isinstance(value, (str, int, float)):
        return _plain(value.value)
    return value


def content_key(experiment: dict[str, Any]) -> str:
    """A content fingerprint stable across the P1 to P4 re-keying.

    Gene names and perturbed names of the genotype, the phenotype's own label value
    and statistic, the media name, the temperature and the environment perturbation
    names: the facts a record states that no proposal of the unification rewrote.
    """
    genotype = experiment["genotype"]
    perturbations = genotype.get("perturbations")
    if perturbations is None:
        perturbations = []
    phenotype = experiment["phenotype"]
    label = phenotype.get("label_name")
    statistic = phenotype.get("label_statistic_name")
    environment = experiment["environment"]
    media = environment.get("media") or {}
    temperature = environment.get("temperature") or {}
    key = {
        "genes": sorted(str(p.get("systematic_gene_name")) for p in perturbations),
        "names": sorted(str(p.get("perturbed_gene_name")) for p in perturbations),
        "label": label,
        "value": _plain(phenotype.get(label)) if label else None,
        "statistic": _plain(phenotype.get(statistic)) if statistic else None,
        "media": media.get("name"),
        "temperature": _plain(temperature.get("value")),
        "env_perturbations": sorted(
            json.dumps(
                {
                    "name": p.get("name") or p.get("compound_name"),
                    "type": p.get("perturbation_type"),
                },
                sort_keys=True,
            )
            for p in environment.get("perturbations") or []
        ),
    }
    return json.dumps(key, sort_keys=True, default=str)


def resolve_served_blob(
    blob: dict[str, Any], constants: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Inline the ``$ref`` pointers of a served experiment blob."""

    def resolve(value: Any) -> Any:
        if isinstance(value, dict):
            if POINTER_KEY in value:
                ref = value[POINTER_KEY]
                if ref not in constants:
                    raise KeyError(f"served pointer {ref} has no InternedConstant")
                return resolve(json.loads(constants[ref]["payload"]))
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    resolved = resolve(blob)
    if not isinstance(resolved, dict):
        raise TypeError("a served blob resolves to a dict")
    return resolved


def served_content(served_dir: str, dataset: str) -> tuple[Counter[str], set[str]]:
    """Content keys and experiment ids of one dataset's served 3.0 records."""
    path = osp.join(served_dir, f"{dataset}.jsonl")
    constants: dict[str, dict[str, Any]] = {}
    with open(osp.join(served_dir, "interned_constants.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            constants[row["id"]] = row
    keys: Counter[str] = Counter()
    ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            ids.add(row["id"])
            experiment = resolve_served_blob(json.loads(row["blob"]), constants)
            keys[content_key(experiment)] += 1
    return keys, ids


def check_placement(
    perturbation: dict[str, Any],
    construct_set: str | None,
    data_root: str,
    state: dict[str, Any],
) -> None:
    """Classify one gene addition's placement and resolve a placed one on disk."""
    state["additions"] += 1
    placement = perturbation.get("placement")
    if placement is not None:
        state["placed"] += 1
        sha = placement.get("contig_sha256")
        ref = placement.get("contig_ref") or {}
        state["contigs"][sha] += 1
        if sha in state["resolved_cache"]:
            ok = state["resolved_cache"][sha]
        else:
            ok = False
            if construct_set is not None and ref.get("sha256") == sha:
                manifest = load_construct_manifest(construct_set, data_root)
                handles = [c.handle for c in manifest.contigs if c.sha256 == sha]
                if handles:
                    resolve_contig(construct_set, handles[0], data_root=data_root)
                    ok = True
            state["resolved_cache"][sha] = ok
        if ok:
            state["placed_resolved_to_deposit"] += 1
        else:
            state["placed_unresolved"].append(str(sha))
        return
    gap = _gap_for(perturbation, "placement")
    if gap is None:
        state["absent_no_gap"] += 1
    elif _gap_is_typed(gap):
        state["gapped_with_reason_and_looked_in"] += 1
        state["gap_reasons"][str(gap.get("reason"))] += 1
    else:
        state["gapped_without_reason"] += 1


def _read_rows(part_files: list[Path]) -> Iterator[list[str]]:
    """Rows of the part files, split on the tab delimiter, quotes stripped.

    BioCypher writes no quoted delimiter and no multiline field (the import call passes
    no ``--multiline-fields``), so a line is a row and a tab is a field boundary; the
    ``csv`` module is not used because a Caudal 2024 experiment blob exceeds its field
    limit. String properties arrive wrapped in single quotes; ids are bare.
    """
    for part in part_files:
        with part.open(encoding="utf-8", newline="") as f:
            for line in f:
                yield [
                    field[1:-1]
                    if len(field) >= 2 and field[0] == "'" and field[-1] == "'"
                    else field
                    for field in line.rstrip("\r\n").split("\t")
                ]


def _header(adapter_dir: Path, label: str) -> list[str]:
    path = adapter_dir / f"{label}-header.csv"
    with path.open(encoding="utf-8") as f:
        return f.read().strip().split("\t")


def csv_summary(adapter_dir: Path) -> dict[str, Any]:
    """Rows per label, host_genome per perturbation node, assembly_set per genome."""
    labels: dict[str, int] = {}
    parts: dict[str, list[Path]] = {}
    for part in sorted(adapter_dir.glob("*-part*.csv")):
        label = part.name.split("-part")[0]
        parts.setdefault(label, []).append(part)
    for label, files in parts.items():
        labels[label] = sum(1 for _ in _read_rows(files))
    host_genomes: Counter[str] = Counter()
    for label in ("Perturbation", "SequenceVariantPerturbation"):
        if label not in parts:
            continue
        header = _header(adapter_dir, label)
        column = header.index("host_genome")
        for row in _read_rows(parts[label]):
            host_genomes[row[column]] += 1
    assembly_sets: Counter[str] = Counter()
    if "Genome" in parts:
        header = _header(adapter_dir, "Genome")
        column = header.index("assembly_set")
        for row in _read_rows(parts["Genome"]):
            assembly_sets[row[column]] += 1
    experiment_ids: set[str] = set()
    if "Experiment" in parts:
        for row in _read_rows(parts["Experiment"]):
            experiment_ids.add(row[0])
    done = adapter_dir / "DONE"
    marker = json.loads(done.read_text()) if done.exists() else None
    return {
        "rows_per_label": labels,
        "perturbation_host_genome": dict(host_genomes),
        "genome_assembly_set": dict(assembly_sets),
        "host_genome_outside_genome_assembly_sets": sorted(
            set(host_genomes) - set(assembly_sets)
        ),
        "experiment_ids": sorted(experiment_ids),
        "done": marker,
    }


_BUILT_RE = re.compile(r"^BUILT (\S+): (\d+) records at \S+ in (\d+)s", re.M)
_WALL_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)")
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\): (\d+)")
_VERIFY_RE = re.compile(r"VERIFY OUTCOME: (\S+)")
_GEN_RE = re.compile(r"Generation wall (\d+) s, peak memory ([\d.]+) GiB")


def _hms_to_s(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def parse_logs(log_dir: str) -> dict[str, dict[str, Any]]:
    """Per dataset: build and generation timings read off the slurm outputs."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(log_dir).glob("*.out")):
        text = path.read_text(encoding="utf-8", errors="replace")
        built = _BUILT_RE.search(text)
        wall = _WALL_RE.search(text)
        rss = _RSS_RE.search(text)
        if built and "prebuild-verify-store" in path.name:
            entry = out.setdefault(built.group(1), {})
            entry["build_time_s"] = float(built.group(3))
            entry["build_wall_s"] = _hms_to_s(wall.group(1)) if wall else None
            entry["build_peak_rss_gb"] = int(rss.group(1)) / 2**20 if rss else None
            verify = _VERIFY_RE.search(text)
            entry["verify_outcome"] = verify.group(1) if verify else None
        gen = _GEN_RE.search(text)
        dataset = re.search(r"Generating adapter \d+ \S+ \((\S+);", text)
        if gen and dataset and "gen_adapter" in path.name:
            entry = out.setdefault(dataset.group(1), {})
            entry["generation_wall_s"] = float(gen.group(1))
            entry["generation_peak_mem_gb"] = float(gen.group(2))
            entry["generation_time_v_wall_s"] = (
                _hms_to_s(wall.group(1)) if wall else None
            )
            entry["generation_peak_rss_gb"] = int(rss.group(1)) / 2**20 if rss else None
    return out


def check_store(
    spec: dict[str, Any],
    data_root: str,
    shared_data_root: str | None,
    served_dir: str | None,
    release_counts: dict[str, int],
    biocypher_out: str | None,
    logs: dict[str, dict[str, Any]],
) -> StoreCheck:
    """Walk every record of one store and gather the counts."""
    root = osp.join(data_root, "data/torchcell", spec["slug"])
    processed = osp.join(root, "processed")
    manifest_path = osp.join(root, "preprocess", "build_manifest.json")
    manifest = (
        json.loads(Path(manifest_path).read_text()) if osp.exists(manifest_path) else {}
    )
    experiment_types: Counter[str] = Counter()
    reference_types: Counter[str] = Counter()
    perturbation_types: Counter[str] = Counter()
    namespaces: Counter[str] = Counter()
    assembly_sets: Counter[str] = Counter()
    missing_namespace = 0
    off_namespace = 0
    off_assembly = 0
    assembly = {"present": 0, "typed": 0, "untyped": 0, "absent": 0}
    placement_state: dict[str, Any] = {
        "additions": 0,
        "placed": 0,
        "placed_resolved_to_deposit": 0,
        "placed_unresolved": [],
        "gapped_with_reason_and_looked_in": 0,
        "gapped_without_reason": 0,
        "absent_no_gap": 0,
        "gap_reasons": Counter(),
        "contigs": Counter(),
        "resolved_cache": {},
    }
    new_keys: Counter[str] = Counter()
    n = 0
    experiment_digests: set[str] = set()
    for record in iter_records(processed):
        n += 1
        experiment = record["experiment"]
        # The adapter's Experiment id is sha256(json.dumps(model_dump())); two records
        # with identical content become ONE node, so this is the node count the graph
        # can hold for this store.
        experiment_digests.add(
            hashlib.sha256(
                json.dumps(_plain(experiment), sort_keys=True, default=str).encode()
            ).hexdigest()
        )
        reference = record["reference"]
        experiment_types[str(experiment.get("experiment_type"))] += 1
        reference_types[str(reference.get("experiment_reference_type"))] += 1
        for perturbation in experiment["genotype"].get("perturbations") or []:
            perturbation_types[str(perturbation.get("perturbation_type"))] += 1
            namespace = perturbation.get("gene_namespace")
            if namespace is None:
                missing_namespace += 1
            else:
                namespaces[str(namespace)] += 1
                if namespace not in (
                    spec["namespace"],
                    *spec.get("also_namespaces", []),
                ):
                    off_namespace += 1
            if perturbation.get("perturbation_type") in GENE_ADDITION_TYPES:
                check_placement(
                    perturbation, spec.get("construct_set"), data_root, placement_state
                )
        genome = reference.get("genome_reference") or {}
        assembly_set = genome.get("assembly_set")
        if assembly_set is not None:
            assembly["present"] += 1
            assembly_sets[str(assembly_set)] += 1
            if assembly_set != spec["assembly_set"]:
                off_assembly += 1
        else:
            gap = _gap_for(genome, "assembly_set")
            if gap is None:
                assembly["absent"] += 1
            elif _gap_is_typed(gap):
                assembly["typed"] += 1
            else:
                assembly["untyped"] += 1
        if served_dir is not None:
            new_keys[content_key(experiment)] += 1
    retired = sorted(
        t
        for t in set(experiment_types) | set(reference_types)
        if t in RETIRED_EXPERIMENT_TYPES
    )
    unknown = sorted(
        [t for t in experiment_types if t not in EXPERIMENT_TYPE_MAP]
        + [t for t in reference_types if t not in EXPERIMENT_REFERENCE_TYPE_MAP]
    )
    lost: int | None = None
    lost_sample: list[str] = []
    new_only: int | None = None
    served_ids: set[str] = set()
    if served_dir is not None and osp.exists(
        osp.join(served_dir, f"{spec['dataset']}.jsonl")
    ):
        served_keys, served_ids = served_content(served_dir, spec["dataset"])
        missing = served_keys - new_keys
        lost = sum(missing.values())
        lost_sample = [key[:400] for key in list(missing)[:3]]
        new_only = sum((new_keys - served_keys).values())
    main_lost: int | None = None
    main_lost_sample: list[str] = []
    main_only: int | None = None
    shared_processed = (
        osp.join(shared_data_root, "data/torchcell", spec["slug"], "processed")
        if shared_data_root
        else None
    )
    if shared_processed is not None and osp.isdir(osp.join(shared_processed, "lmdb")):
        main_keys: Counter[str] = Counter(
            content_key(record["experiment"])
            for record in iter_records(shared_processed)
        )
        if not new_keys:
            new_keys = Counter(
                content_key(record["experiment"]) for record in iter_records(processed)
            )
        missing_main = main_keys - new_keys
        main_lost = sum(missing_main.values())
        main_lost_sample = [key[:400] for key in list(missing_main)[:3]]
        main_only = sum((new_keys - main_keys).values())
    csv_part: dict[str, Any] | None = None
    overlap: int | None = None
    if biocypher_out is not None:
        matches = [
            m
            for m in sorted(Path(biocypher_out, "adapters").glob("*_*Adapter"))
            if (m / "DONE").exists()
            and json.loads((m / "DONE").read_text())["dataset_class"] == spec["dataset"]
        ]
        if matches:
            csv_part = csv_summary(matches[0])
            ids = set(csv_part.pop("experiment_ids"))
            csv_part["n_experiment_ids"] = len(ids)
            overlap = len(ids & served_ids) if served_ids else None
    shared_n = (
        lmdb_entries(
            osp.join(shared_data_root, "data/torchcell", spec["slug"], "processed")
        )
        if shared_data_root
        else None
    )
    log = logs.get(spec["dataset"], {})
    return StoreCheck(
        dataset=spec["dataset"],
        slug=spec["slug"],
        why=spec["why"],
        n_records=n,
        n_distinct_experiments=len(experiment_digests),
        n_served_3_0=release_counts.get(spec["dataset"]),
        n_shared_main_tree=shared_n,
        experiment_types=dict(experiment_types),
        reference_types=dict(reference_types),
        retired_types_seen=retired,
        unknown_types_seen=unknown,
        perturbation_leaves_manifest=manifest.get("perturbation_leaves"),
        perturbation_types=dict(perturbation_types),
        n_perturbations=sum(perturbation_types.values()),
        namespaces=dict(namespaces),
        perturbations_missing_namespace=missing_namespace,
        perturbations_off_expected_namespace=off_namespace,
        expected_namespace=spec["namespace"],
        assembly_sets=dict(assembly_sets),
        assembly=GapCheck(
            present=assembly["present"],
            gapped_with_reason_and_looked_in=assembly["typed"],
            gapped_without_reason=assembly["untyped"],
            absent_no_gap=assembly["absent"],
        ),
        expected_assembly_set=spec["assembly_set"],
        references_off_expected_assembly=off_assembly,
        placement=PlacementCheck(
            additions=placement_state["additions"],
            placed=placement_state["placed"],
            placed_resolved_to_deposit=placement_state["placed_resolved_to_deposit"],
            placed_unresolved=sorted(set(placement_state["placed_unresolved"])),
            gapped_with_reason_and_looked_in=placement_state[
                "gapped_with_reason_and_looked_in"
            ],
            gapped_without_reason=placement_state["gapped_without_reason"],
            absent_no_gap=placement_state["absent_no_gap"],
            gap_reasons=dict(placement_state["gap_reasons"]),
            contigs=dict(placement_state["contigs"]),
        ),
        content_served_lost=lost,
        content_served_lost_sample=lost_sample,
        content_new_only=new_only,
        content_main_lost=main_lost,
        content_main_lost_sample=main_lost_sample,
        content_new_only_vs_main=main_only,
        served_ids_in_new_csv=overlap,
        build_wall_s=log.get("build_wall_s"),
        build_time_s=log.get("build_time_s"),
        build_peak_rss_gb=log.get("build_peak_rss_gb"),
        verify_outcome=log.get("verify_outcome"),
        csv=csv_part,
        generation_wall_s=log.get("generation_wall_s"),
        generation_peak_mem_gb=log.get("generation_peak_mem_gb"),
        generation_peak_rss_gb=log.get("generation_peak_rss_gb"),
        generation_time_v_wall_s=log.get("generation_time_v_wall_s"),
    )


def _fmt(value: float | int | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def render_tables(checks: list[StoreCheck]) -> str:
    """The markdown tables the note carries."""
    lines = [
        "| dataset | records | distinct experiments | 3.0 served | main tree 10-13 | "
        "build s | wall s | peak RSS GB | verify | gen wall s | gen peak mem GB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for c in checks:
        lines.append(
            f"| {c.dataset} | {c.n_records:,} | {c.n_distinct_experiments:,} | "
            f"{_fmt(c.n_served_3_0)} | "
            f"{_fmt(c.n_shared_main_tree)} | {_fmt(c.build_time_s, 0)} | "
            f"{_fmt(c.build_wall_s, 0)} | {_fmt(c.build_peak_rss_gb, 2)} | "
            f"{c.verify_outcome or '-'} | {_fmt(c.generation_wall_s, 0)} | "
            f"{_fmt(c.generation_peak_mem_gb, 2)} |"
        )
    lines += [
        "",
        "| dataset | perturbations | namespace (count) | missing | off expected | "
        "references pinned | gapped typed | absent no gap | off expected | "
        "retired tags |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for c in checks:
        ns = ", ".join(f"{k} ({v:,})" for k, v in sorted(c.namespaces.items())) or "-"
        lines.append(
            f"| {c.dataset} | {c.n_perturbations:,} | {ns} | "
            f"{c.perturbations_missing_namespace} | "
            f"{c.perturbations_off_expected_namespace} | {c.assembly.present:,} | "
            f"{c.assembly.gapped_with_reason_and_looked_in} | "
            f"{c.assembly.absent_no_gap} | {c.references_off_expected_assembly} | "
            f"{', '.join(c.retired_types_seen) or 'none'} |"
        )
    lines += [
        "",
        "| dataset | additions | placed | resolved to deposit | gapped typed | "
        "gap reasons | absent no gap | 3.0 content lost | new vs 3.0 | "
        "main-tree content lost | new vs main tree | 3.0 ids in new CSV |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for c in checks:
        p = c.placement
        reasons = (
            ", ".join(f"{k} ({v})" for k, v in sorted(p.gap_reasons.items())) or "-"
        )
        lines.append(
            f"| {c.dataset} | {p.additions:,} | {p.placed:,} | "
            f"{p.placed_resolved_to_deposit:,} | "
            f"{p.gapped_with_reason_and_looked_in:,} | {reasons} | "
            f"{p.absent_no_gap} | {_fmt(c.content_served_lost)} | "
            f"{_fmt(c.content_new_only)} | {_fmt(c.content_main_lost)} | "
            f"{_fmt(c.content_new_only_vs_main)} | {_fmt(c.served_ids_in_new_csv)} |"
        )
    lines += [
        "",
        "| dataset | CSV experiment rows | perturbation rows | host_genome (count) | "
        "genome assembly_set (count) | host_genome outside the genome's sets |",
        "|---|---:|---:|---|---|---|",
    ]
    for c in checks:
        if c.csv is None:
            lines.append(f"| {c.dataset} | - | - | - | - | - |")
            continue
        rows = c.csv["rows_per_label"]
        hg = (
            ", ".join(
                f"{k} ({v:,})"
                for k, v in sorted(c.csv["perturbation_host_genome"].items())
            )
            or "-"
        )
        ga = (
            ", ".join(
                f"{k} ({v:,})" for k, v in sorted(c.csv["genome_assembly_set"].items())
            )
            or "-"
        )
        outside = ", ".join(c.csv["host_genome_outside_genome_assembly_sets"]) or "none"
        n_pert = rows.get("Perturbation", 0) + rows.get(
            "SequenceVariantPerturbation", 0
        )
        lines.append(
            f"| {c.dataset} | {rows.get('Experiment', 0):,} | {n_pert:,} | {hg} | "
            f"{ga} | {outside} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--shared-data-root", default=None)
    parser.add_argument("--served-dir", default=None)
    parser.add_argument("--biocypher-out", default=None)
    parser.add_argument("--slurm-logs", default=None)
    parser.add_argument("--release", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--datasets", default=None, help="comma-separated subset")
    args = parser.parse_args(argv)
    release_counts: dict[str, int] = {}
    if args.release:
        release = json.loads(Path(args.release).read_text())
        release_counts = {
            name: int(entry["n_experiments"])
            for name, entry in release["datasets"].items()
        }
    logs = parse_logs(args.slurm_logs) if args.slurm_logs else {}
    wanted = set(args.datasets.split(",")) if args.datasets else None
    checks: list[StoreCheck] = []
    for spec in SAMPLE:
        if wanted is not None and spec["dataset"] not in wanted:
            continue
        check = check_store(
            spec,
            args.data_root,
            args.shared_data_root,
            args.served_dir,
            release_counts,
            args.biocypher_out,
            logs,
        )
        checks.append(check)
        print(
            f"{check.dataset}: {check.n_records:,} records; served 3.0 "
            f"{check.n_served_3_0}; main tree {check.n_shared_main_tree}; "
            f"missing namespace {check.perturbations_missing_namespace}; "
            f"off namespace {check.perturbations_off_expected_namespace}; "
            f"assembly {check.assembly}; placement {check.placement.model_dump()}; "
            f"served content lost {check.content_served_lost}; main-tree content "
            f"lost {check.content_main_lost}; retired "
            f"{check.retired_types_seen}",
            file=sys.stderr,
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps([c.model_dump() for c in checks], indent=1, default=str) + "\n"
    )
    if args.table:
        print(render_tables(checks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
