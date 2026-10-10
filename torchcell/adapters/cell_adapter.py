"""BioCypher adapter that turns a cell dataset into graph nodes and edges."""

# torchcell/adapters/cell_adapter
# [[torchcell.adapters.cell_adapter]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/adapters/cell_adapter
# Test file: tests/torchcell/adapters/test_cell_adapter.py

import copy
import gc
import hashlib
import json
import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from datetime import datetime
from functools import wraps
from itertools import chain
from typing import Any, cast

import wandb
from biocypher._create import BioCypherEdge, BioCypherNode
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from tqdm import tqdm

from torchcell.build_telemetry import BuildPhase
from torchcell.datamodels.identity import (
    environment_identity,
    environment_perturbation_identity,
    identity_sha256,
    media_identity,
    temperature_identity,
)
from torchcell.datamodels.interned_constant import split_experiment_dump
from torchcell.datamodels.schema import (
    BacterialCrisprActivationPerturbation,
    BacterialCrisprInterferencePerturbation,
    BacterialDegronPerturbation,
    BacterialDeletionPerturbation,
    BacterialMarkedAllelePerturbation,
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialSpanDeletionPerturbation,
    HeterologousPathwayPerturbation,
    PhagePerturbation,
    PromoterReplacementPerturbation,
    TransposonInsertionPerturbation,
)
from torchcell.fast_csv import RenderedChunk, RowSpecs
from torchcell.loader import CpuExperimentLoaderMultiprocessing

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

CHUNKS_PER_WORKER = 2

SINGLE_PASS_NODES = "all node types (chunked)"
SINGLE_PASS_EDGES = "all edge types (chunked)"
SINGLE_PASS_CHUNK_BUDGET_BYTES = 48 * 2**20
INPROCESS_MAX_BYTES = 0  # 0: the in-process rule is by record count alone
"""Resolved-record bytes a single-pass chunk may carry.

A folded pass emits every node (or edge) type per record, so a chunk's output scales
with the record's resolved size, not its count. Bloom2019 (segregant genotypes, no
memory reduction factor in its adapter config) got 12,500-record chunks on job 2889;
30 workers plus their loader children took anonymous memory from 36 GB to 104 GB in
20 s and the container was OOM-killed at 128 GB. The budget divides by the sampled
resolved record size so a chunk of big records is proportionally shorter.

r7: the budget is a per-adapter attribute (``single_pass_chunk_budget_bytes``) because
48 MiB cut Costanzo dmf to 2,355-record chunks on job 2905 and its node pass took
5,317 s against 2,794 s at 6,250 records on job 2889 (same code otherwise, same box):
2.7x more chunks, and the pool is rebuilt every chunks_per_worker x workers of them.
"""
SINGLE_PASS_MIN_CHUNK = 256
BACTERIAL_PERTURBATION_LEAVES: tuple[type, ...] = (
    BacterialDeletionPerturbation,
    TransposonInsertionPerturbation,
    BacterialCrisprInterferencePerturbation,
    PromoterReplacementPerturbation,
    HeterologousPathwayPerturbation,
    # round-2 leaves (#749, #792, #799). Each projects the same five properties the
    # `bacterial perturbation` graph class declares; their leaf-specific fields
    # (cassette, tag, degron, insertion_site, collection) carry no node property, as
    # `BacterialDeletionPerturbation.collection` does not, and travel in the
    # serialized record.
    BacterialMarkedAllelePerturbation,
    BacterialDegronPerturbation,
    BacterialCrisprActivationPerturbation,
)
"""The gene-perturbation leaves written as ``bacterial perturbation`` nodes.

The leaves that carry ``gene_namespace`` MINUS the called-variant leaves below, which
have their own class. A yeast leaf is never one of them, so a yeast record emits no
``bacterial perturbation`` node, and ``_perturbation_node`` (served) is not touched to
exclude them: a bacterial adapter conf enables ``bacterial perturbation (chunked)``
instead of ``perturbation (chunked)``.
"""
BACTERIAL_VARIANT_PERTURBATION_LEAVES: tuple[type, ...] = (
    BacterialSequenceVariantPerturbation,
    BacterialSiteVariantPerturbation,
    BacterialSpanDeletionPerturbation,
)
"""The leaves written as ``bacterial sequence variant perturbation`` nodes (#731).

Exactly the leaves that compose a ``BacterialVariantCall``.
``BacterialSpanDeletionPerturbation`` is also a ``BacterialDeletionPerturbation``, so the
``bacterial perturbation`` method excludes this tuple explicitly; without that a span
deletion would be written twice, once under each label.
"""
CGROUP_MEMORY_CURRENT = "/sys/fs/cgroup/memory.current"
CGROUP_MEMORY_MAX = "/sys/fs/cgroup/memory.max"


def cgroup_memory_fraction() -> float:
    """Current cgroup v2 memory use as a fraction of the container's limit.

    Raises when the container has no memory limit (``memory.max`` is ``max``) or the
    cgroup files are absent: the memory-driven pool recycling needs a real limit to
    measure against, and running without one is a configuration error, not a case
    to fall back from.
    """
    with open(CGROUP_MEMORY_MAX) as fh:
        limit = fh.read().strip()
    if limit == "max":
        raise RuntimeError(
            "pool_memory_fraction needs a cgroup memory limit; memory.max is 'max'"
        )
    with open(CGROUP_MEMORY_CURRENT) as fh:
        current = int(fh.read().strip())
    return current / int(limit)


"""Phase names of the r3 single-pass traversals (one per adapter per kind)."""
"""Chunks a pool worker may handle before the pool is rebuilt with fresh workers.

Bounds a worker's heap, which ratchets up across chunks because CPython does not
return freed arenas to the OS. ``max_tasks_per_child`` would express this directly
but raises ValueError under the fork start method, and fork is what lets a worker
inherit the built dataset instead of re-importing and re-opening it.
"""


class CellAdapter:
    """Convert a cell experiment dataset into BioCypher nodes and edges for import."""

    def __init__(
        self,
        config: DictConfig,
        dataset: Dataset,
        process_workers: int,
        io_workers: int,
        chunk_size: int = int(1e4),
        loader_batch_size: int = int(1e3),
        inprocess_max_records: int = 0,
    ):
        """Store config, dataset, and worker/chunk sizes for graph generation.

        Args:
            config: Hydra config holding the adapter's node and edge methods.
            dataset: Dataset whose items are converted to nodes and edges.
            process_workers: Number of processes for parallel chunk processing.
            io_workers: Number of workers used by the experiment data loader.
            chunk_size: Number of dataset items processed per chunk.
            loader_batch_size: Batch size used within each chunk; must not
                exceed ``chunk_size``.
            inprocess_max_records: Datasets with at most this many records run
                every chunked method in THIS process, with no worker pool and no
                loader children. 0 disables the path. On job 2032 a method cost
                about 30 s of fixed overhead (pool fork from a ~100 GB parent,
                loader forks, LMDB open, teardown) whatever the dataset size, so
                33 small datasets took 9.5 h for a median of 1,484 records each.
        """
        if loader_batch_size > chunk_size:
            raise ValueError(
                "chunk_size must be greater than or equal to loader_batch_size."
                "Our recommendation are chunk_size 2-3 order of magnitude in size."
            )
        self.config = config
        self.dataset = dataset
        self.process_workers = process_workers
        self.io_workers = io_workers
        self.chunk_size = chunk_size
        self.loader_batch_size = loader_batch_size
        self.inprocess_max_records = inprocess_max_records
        # r3: one traversal per adapter for every chunked node method (and one for
        # every chunked edge method) instead of one traversal per method. The
        # per-record work is identical; only the number of passes over the LMDB and
        # the pydantic rehydrations change. Set by the build script from the config.
        self.single_pass = False
        self._single_pass_methods: list[tuple[str, Callable[..., Any]]] = []
        # r5: when set, chunk workers render neo4j-admin rows themselves and return one
        # RenderedChunk per chunk; the build's FastCsvSink dedups and appends. None
        # keeps the BioCypherNode/Edge objects flowing to bc.write_nodes/write_edges.
        self.row_specs: RowSpecs | None = None
        self.chunks_per_worker = CHUNKS_PER_WORKER
        self.single_pass_chunk_budget_bytes = SINGLE_PASS_CHUNK_BUDGET_BYTES
        # r10: the in-process rule is by records only unless this is set; then a
        # dataset also has to fit in this many resolved bytes. Job 2959 ran Caudal
        # (943 records of 3 MB), Kemmeren (1,484 of 780 KB), Messner and Nadal-Ribelles
        # in-process on one core for 39 minutes of a 4 h build.
        self.inprocess_max_bytes = INPROCESS_MAX_BYTES
        # r10: yield chunk results as they complete instead of in submission order.
        # In order, the parent blocks on the oldest chunk and submits a replacement
        # only after consuming it, so one slow chunk idles the rest of the pool: job
        # 2959's Costanzo passes averaged 18 of 64 cores with peaks at 50.
        self.completion_order = False
        # r11: recycle a pool once the container's memory is above this fraction of
        # its cgroup limit (0 keeps the fixed chunks_per_worker groups).
        self.pool_memory_fraction = 0.0
        self._record_bytes: int | None = None
        self.event = 0
        wandb.init()
        self.log_method_table()
        wandb.log(
            {
                "current_adapter_dataset_name": self.dataset.name,
                "current_adapter_dataset_start_time": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }
        )

        # Supported methods
        self.node_methods = [
            ("experiment reference", self._get_experiment_reference_nodes),
            ("genome", self._get_genome_nodes),
            ("experiment (chunked)", self._experiment_node),
            ("genotype (chunked)", self._genotype_node),
            ("segregant genotype (chunked)", self._segregant_genotype_node),
            ("perturbation (chunked)", self._perturbation_node),
            ("bacterial perturbation (chunked)", self._bacterial_perturbation_node),
            (
                "bacterial sequence variant perturbation (chunked)",
                self._bacterial_variant_perturbation_node,
            ),
            ("crispr construct (chunked)", self._crispr_construct_node),
            ("environment (chunked)", self._environment_node),
            ("environment reference", self._get_environment_reference_nodes),
            ("media (chunked)", self._media_node),
            ("media reference", self._get_media_reference_nodes),
            ("temperature (chunked)", self._temperature_node),
            ("temperature reference", self._get_temperature_reference_nodes),
            ("environment perturbation (chunked)", self._environment_perturbation_node),
            (
                "environment perturbation reference",
                self._get_environment_perturbation_reference_nodes,
            ),
            ("phage perturbation (chunked)", self._phage_perturbation_node),
            (
                "phage perturbation reference",
                self._get_phage_perturbation_reference_nodes,
            ),
            ("fitness phenotype (chunked)", self._fitness_phenotype_node),
            (
                "gene interaction phenotype (chunked)",
                self._gene_interaction_phenotype_node,
            ),
            (
                "gene essentiality phenotype (chunked)",
                self._gene_essentiality_phenotype_node,
            ),
            (
                "synthetic lethality phenotype (chunked)",
                self._synthetic_lethality_phenotype_node,
            ),
            (
                "synthetic rescue phenotype (chunked)",
                self._synthetic_rescue_phenotype_node,
            ),
            ("calmorph phenotype (chunked)", self._calmorph_phenotype_node),
            (
                "microarray expression phenotype (chunked)",
                self._microarray_expression_phenotype_node,
            ),
            (
                "rnaseq expression phenotype (chunked)",
                self._rnaseq_expression_phenotype_node,
            ),
            (
                "pseudobulk expression phenotype (chunked)",
                self._pseudobulk_expression_phenotype_node,
            ),
            ("visual score phenotype (chunked)", self._visual_score_phenotype_node),
            ("metabolite phenotype (chunked)", self._metabolite_phenotype_node),
            (
                "protein abundance phenotype (chunked)",
                self._protein_abundance_phenotype_node,
            ),
            # --- begin #770: the protein fold-change family ---
            (
                "protein fold change phenotype (chunked)",
                self._protein_fold_change_phenotype_node,
            ),
            # --- end #770 ---
            (
                "environment response phenotype (chunked)",
                self._environment_response_phenotype_node,
            ),
            ("product titer phenotype (chunked)", self._product_titer_phenotype_node),
            (
                "protein turnover phenotype (chunked)",
                self._protein_turnover_phenotype_node,
            ),
            ("flux phenotype (chunked)", self._flux_phenotype_node),
            (
                "promoter activity phenotype (chunked)",
                self._promoter_activity_phenotype_node,
            ),
            (
                "bacterial morphology phenotype (chunked)",
                self._bacterial_morphology_phenotype_node,
            ),
            (
                "mrna number fraction phenotype (chunked)",
                self._mrna_number_fraction_phenotype_node,
            ),
            (
                "fitness phenotype reference",
                self._get_fitness_phenotype_reference_nodes,
            ),
            (
                "gene interaction phenotype reference",
                self._get_gene_interaction_phenotype_reference_nodes,
            ),
            (
                "gene essentiality phenotype reference",
                self._get_gene_essentiality_phenotype_reference_nodes,
            ),
            (
                "synthetic lethality phenotype reference",
                self._get_synthetic_lethality_phenotype_reference_nodes,
            ),
            (
                "synthetic rescue phenotype reference",
                self._get_synthetic_rescue_phenotype_reference_nodes,
            ),
            (
                "calmorph phenotype reference",
                self._get_calmorph_phenotype_reference_nodes,
            ),
            (
                "microarray expression phenotype reference",
                self._get_microarray_expression_phenotype_reference_nodes,
            ),
            (
                "rnaseq expression phenotype reference",
                self._get_rnaseq_expression_phenotype_reference_nodes,
            ),
            (
                "pseudobulk expression phenotype reference",
                self._get_pseudobulk_expression_phenotype_reference_nodes,
            ),
            (
                "visual score phenotype reference",
                self._get_visual_score_phenotype_reference_nodes,
            ),
            (
                "metabolite phenotype reference",
                self._get_metabolite_phenotype_reference_nodes,
            ),
            (
                "protein abundance phenotype reference",
                self._get_protein_abundance_phenotype_reference_nodes,
            ),
            # --- begin #770: the protein fold-change family ---
            (
                "protein fold change phenotype reference",
                self._get_protein_fold_change_phenotype_reference_nodes,
            ),
            # --- end #770 ---
            (
                "environment response phenotype reference",
                self._get_environment_response_phenotype_reference_nodes,
            ),
            (
                "product titer phenotype reference",
                self._get_product_titer_phenotype_reference_nodes,
            ),
            (
                "protein turnover phenotype reference",
                self._get_protein_turnover_phenotype_reference_nodes,
            ),
            ("flux phenotype reference", self._get_flux_phenotype_reference_nodes),
            (
                "promoter activity phenotype reference",
                self._get_promoter_activity_phenotype_reference_nodes,
            ),
            (
                "bacterial morphology phenotype reference",
                self._get_bacterial_morphology_phenotype_reference_nodes,
            ),
            (
                "mrna number fraction phenotype reference",
                self._get_mrna_number_fraction_phenotype_reference_nodes,
            ),
            ("dataset", self._get_dataset_nodes),
            ("publication (chunked)", self._publication_node),
        ]
        self.edge_methods = [
            (
                "experiment reference to dataset",
                self._get_experiment_reference_to_dataset_edges,
            ),
            ("experiment to dataset (chunked)", self._experiment_to_dataset_edge),
            (
                "experiment reference to experiment (chunked)",
                self._experiment_reference_to_experiment_edge,
            ),
            ("genotype to experiment (chunked)", self._genotype_to_experiment_edge),
            (
                "perturbation to genotype (chunked)",
                self._perturbation_to_genotype_edges,
            ),
            (
                "crispr construct to perturbation (chunked)",
                self._crispr_construct_to_perturbation_edges,
            ),
            (
                "environment to experiment (chunked)",
                self._environment_to_experiment_edge,
            ),
            (
                "environment to experiment reference",
                self._get_environment_to_experiment_reference_edges,
            ),
            ("phenotype to experiment (chunked)", self._phenotype_to_experiment_edge),
            ("media to environment (chunked)", self._media_to_environment_edge),
            (
                "temperature to environment (chunked)",
                self._temperature_to_environment_edge,
            ),
            (
                "environment perturbation to environment (chunked)",
                self._environment_perturbation_to_environment_edges,
            ),
            (
                "environment perturbation to environment reference",
                self._get_environment_perturbation_to_environment_reference_edges,
            ),
            (
                "genome to experiment reference",
                self._get_genome_to_experiment_reference_edges,
            ),
            (
                "phenotype to experiment reference",
                self._get_phenotype_to_experiment_reference_edges,
            ),
            (
                "publication to experiment (chunked)",
                self._publication_to_experiment_edge,
            ),
        ]

    def log_method_table(self) -> None:
        """Log a wandb table summarizing the configured node and edge methods."""
        methods: list[Iterable[Any]] = []
        simulated_event_counter = 0
        for method in (
            self.config.cell_adapter.node_methods
            + self.config.cell_adapter.edge_methods
        ):
            simulated_event_counter += 1
            method_name = method["method_name"]
            data_type = (
                "node" if method in self.config.cell_adapter.node_methods else "edge"
            )

            if "(chunked)" in method_name:
                if "memory_reduction_factor" in method:
                    memory_reduction_factor = method["memory_reduction_factor"]
                else:
                    memory_reduction_factor = 1.0
            else:
                memory_reduction_factor = float("nan")

            method_info = [
                simulated_event_counter,
                method_name,
                data_type,
                memory_reduction_factor,
            ]
            methods.append(method_info)

        columns: list[str | int] = [
            "event",
            "method",
            "data_type",
            "memory_reduction_factor",
        ]
        method_table = wandb.Table(columns=columns, data=methods)
        wandb.log({f"{self.dataset.name}_method_table": method_table})

    def get_data_by_type(
        self,
        chunk_processing_func: Callable[..., Any],
        method_name: str,
        is_edge: bool = False,
    ) -> Iterator[Any]:
        """Yield processed data by chunking the dataset and mapping over processes.

        Submission is windowed: at most ``process_workers + 2`` chunks are in flight,
        and a new one is submitted only as an earlier result is consumed. Results are
        still yielded in submission order, so callers see no behavioral change.

        The window is what keeps peak memory a function of chunk size rather than
        dataset size. Submitting every chunk up front lets workers run arbitrarily far
        ahead of the consumer, and because results are yielded in order, each finished
        chunk is held in this process until its turn -- for a 20.7M-record dataset at
        chunk_size 1e5 that is 207 results of ~1 GB each. Job 1543 died that way, with
        a worker's fork for its loader failing ENOMEM against the container's limit.

        The window bounds what THIS process holds; ``CHUNKS_PER_WORKER`` bounds what a
        worker holds, by rebuilding the pool every group of chunks. Job 1545 needed both:
        windowed but long-lived workers still ratcheted to 38-50 GB of heap apiece.
        """
        memory_reduction_factor = self.get_memory_reduction_factor(method_name, is_edge)
        chunk_size = int(self.chunk_size * memory_reduction_factor)
        if method_name in (SINGLE_PASS_NODES, SINGLE_PASS_EDGES):
            record_bytes = self._estimate_record_bytes()
            budget_chunk = max(
                SINGLE_PASS_MIN_CHUNK,
                self.single_pass_chunk_budget_bytes // record_bytes,
            )
            if budget_chunk < chunk_size:
                log.info(
                    "single-pass chunk %d -> %d records (%d resolved bytes per record)",
                    chunk_size,
                    budget_chunk,
                    record_bytes,
                )
                chunk_size = budget_chunk

        # Small datasets: one chunk, this process, no forks. The whole dataset is
        # the chunk, and data_chunker's in-process branch iterates it directly. With
        # inprocess_max_bytes set, "small" also means small in resolved bytes, so a
        # dataset of a few hundred multi-MB expression records goes to the pool.
        inprocess = 0 < len(self.dataset) <= self.inprocess_max_records
        if inprocess and self.inprocess_max_bytes > 0:
            inprocess = (
                len(self.dataset) * self._estimate_record_bytes()
                <= self.inprocess_max_bytes
            )
        if inprocess:
            whole = self.dataset[0 : len(self.dataset)]
            self.dataset.close_lmdb()
            yield from chunk_processing_func(whole, method_name, inprocess=True)
            return

        # Every chunk view is built BEFORE the first submission, and the dataset's LMDB
        # environment is closed once afterwards, so that no chunk is ever built while
        # submissions are in flight. Dataset.__getitem__ routes through len(), which
        # opens that environment and closes it again, and executor.submit() hands the
        # call item to a feeder THREAD that pickles it asynchronously -- the call item
        # holds a bound adapter method, hence the dataset. Interleaving the two lets the
        # feeder pickle a dataset whose env is transiently open, which raises
        # "TypeError: cannot pickle 'Environment' object" (job 1544, four minutes into
        # DmfCostanzo2016Adapter's environment method, after three methods had passed).
        data_chunks = [
            self.dataset[i : i + chunk_size]
            for i in range(0, len(self.dataset), chunk_size)
        ]
        self.dataset.close_lmdb()

        # Chunks are handed out in groups, one pool per group, so that no worker
        # outlives CHUNKS_PER_WORKER tasks. Pool workers are otherwise reused for the
        # method's whole traversal, and CPython does not return freed arenas to the OS,
        # so a worker's heap ratchets up across the chunks it handles. Job 1545 was
        # measured mid-hang with worker heaps at 38-50 GB each (read off the
        # Shared_Dirty of their forked loader children) against a 400 GB container, six
        # OOM kills, and 316 GB of anon memory. Recycling caps that at roughly one
        # group's working set. The group is a multiple of the pool size, so a 414-chunk
        # method rebuilds the pool ~7 times rather than paying a fork storm.
        # r6: the group size is the knob. Job 2889's telemetry (Costanzo, 30 workers,
        # 6,250-record chunks) shows a 47 s cycle: the pool fills memory to the cgroup
        # cap, is torn down, and the container sits at 3 to 10 cores for about 15 s
        # while the next pool forks. With the byte-budgeted chunks the groups would be
        # 2.7x shorter still, so chunks_per_worker (default CHUNKS_PER_WORKER) sets it.
        group_size = self.process_workers * self.chunks_per_worker
        remaining = iter(data_chunks)

        def pool_chunks() -> Iterator[Any]:
            """The chunks one pool handles: at most group_size, fewer under memory pressure.

            r11: with ``pool_memory_fraction`` set, a pool stops taking chunks once the
            container's cgroup memory is above that fraction of its limit (checked at
            each submission after every worker has had one chunk), so the group size
            follows the box instead of a fixed count. Job 3067's telemetry: live
            workers retain about 1 GB per chunk handled and only the pool teardown
            releases it; at 22 workers x 8 chunks that crossed 96 GB in 89 s, while
            22 x 2 peaked near 57 GB per group.
            """
            n = 0
            for chunk in remaining:
                yield chunk
                n += 1
                if n >= group_size:
                    return
                if (
                    self.pool_memory_fraction > 0
                    and n >= self.process_workers
                    and cgroup_memory_fraction() >= self.pool_memory_fraction
                ):
                    log.info(
                        "pool recycled at %d chunks: cgroup memory at %.2f of its limit",
                        n,
                        cgroup_memory_fraction(),
                    )
                    return

        while True:
            group = pool_chunks()
            first = next(group, None)
            if first is None:
                break
            group = chain([first], group)
            # Move everything currently reachable into the GC's permanent generation
            # before forking this group's pool. Workers inherit the parent's heap
            # copy-on-write, but one collection inside a worker traverses every tracked
            # object and writes to its header, so those pages turn private per worker.
            # Measured with a 3.05 GB parent heap: one trivial worker privatized 1.08 GB
            # and six privatized 6.50 GB; with gc.freeze() first, every worker stayed at
            # 0.00 GB. On the real genotype method with a ballasted parent, peak private
            # across 41 processes went 46.2 GB -> 3.4 GB, which extrapolates to 168 GB ->
            # 12 GB at 29 workers. That gap is what killed jobs 1545, 1552 and 1553.
            #
            # This runs per group, not once: the parent keeps allocating (BioCypher's
            # write buffers above all), and each group forks a fresh pool, so anything
            # allocated since the last freeze would otherwise be privatized by the next
            # group's workers.
            gc.collect()
            gc.freeze()

            def submit_next(
                executor: ProcessPoolExecutor, group: Iterator[Any] = group
            ) -> Future[Any] | None:
                """Submit the next chunk, or None once the group is fully submitted."""
                chunk = next(group, None)
                if chunk is None:
                    return None
                return executor.submit(chunk_processing_func, chunk, method_name)

            with ProcessPoolExecutor(max_workers=self.process_workers) as executor:
                in_flight: deque[Future[Any]] = deque()
                for _ in range(self.process_workers + 2):
                    future = submit_next(executor)
                    if future is None:
                        break
                    in_flight.append(future)
                if self.completion_order:
                    # r10: consume whichever chunk finishes first and refill at once,
                    # so the window stays full and no worker waits on the oldest chunk.
                    # Row order across chunks then depends on timing; the writer dedups
                    # by id, and the import does not depend on row order.
                    pending: set[Future[Any]] = set(in_flight)
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for finished in done:
                            yield from finished.result()
                            future = submit_next(executor)
                            if future is not None:
                                pending.add(future)
                    continue
                while in_flight:
                    yield from in_flight.popleft().result()
                    future = submit_next(executor)
                    if future is not None:
                        in_flight.append(future)

    def data_chunker(  # type: ignore[misc]  # in-class decorator factory; first arg is the wrapped method, not self
        data_creation_logic: Callable[..., Any],
    ) -> Callable[..., list[Any]]:
        """Wrap a chunk handler so it loads, transforms, and collects each item."""

        @wraps(data_creation_logic)
        def decorator(
            self: "CellAdapter",
            data_chunk: Any,
            method_name: str,
            inprocess: bool = False,
        ) -> list[Any]:
            if inprocess:
                datas_inproc: list[Any] = []
                for i in range(len(data_chunk)):
                    transformed = data_chunk.transform_item(data_chunk[i])
                    out = data_creation_logic(self, transformed, method_name)
                    if isinstance(out, list):
                        datas_inproc.extend(out)
                    else:
                        datas_inproc.append(out)
                data_chunk.close_lmdb()
                return self._pack_chunk(datas_inproc)
            memory_reduction_factor = self.get_memory_reduction_factor(method_name)
            loader_batch_size = int(self.loader_batch_size * memory_reduction_factor)
            # loader_batch_size = self.loader_batch_size
            data_loader = CpuExperimentLoaderMultiprocessing(
                data_chunk, batch_size=loader_batch_size, num_workers=self.io_workers
            )
            datas = []
            # close() in a finally so an error inside the loop tears the loader's
            # worker processes down instead of orphaning them. Orphaned non-daemon
            # workers block interpreter shutdown, which -- when this runs inside a
            # ProcessPoolExecutor worker -- prevents the worker from ever returning
            # the exception, turning a clean crash into a silent Queue "deadlock".
            try:
                for batch in tqdm(data_loader):
                    for data in batch:
                        transformed_data = data_chunk.transform_item(data)
                        data = data_creation_logic(self, transformed_data, method_name)
                        if isinstance(data, list):
                            datas.extend(data)
                        else:
                            datas.append(data)
            finally:
                data_loader.close()
            return self._pack_chunk(datas)

        return decorator

    def _estimate_record_bytes(self, samples: int = 64) -> int:
        """Median JSON size of a resolved record, from evenly spaced samples (cached)."""
        if self._record_bytes is not None:
            return self._record_bytes
        n = len(self.dataset)
        step = max(1, n // samples)
        sizes = sorted(
            len(json.dumps(self.dataset[i], default=str)) for i in range(0, n, step)
        )
        # get() leaves the LMDB environment open. A full dataset closes it again on
        # the next slice (its len() runs through indices()), but a subset view has
        # stored indices, so every chunk view below would shallow-copy the open
        # environment and the pool's feeder could not pickle it (jobs 2918-2921,
        # Costanzo capped to 2M: "cannot pickle 'Environment' object").
        self.dataset.close_lmdb()
        self._record_bytes = max(1, sizes[len(sizes) // 2])
        return self._record_bytes

    def __getstate__(self) -> dict[str, Any]:
        """Pickle for a pool task WITHOUT the parent's per-record indexes.

        Every chunk task pickles the bound chunk method, hence this adapter, hence
        ``self.dataset``, and the worker only reads records through the chunk view it
        is handed. Two caches on the dataset are per-record and must not travel:

        - ``_indices``: a subset view (a capped or prefiltered build, the benchmark
          ladder) carries the whole index list, 2M entries and 10 MB per task
          (``experiments/tcdb-002-build-speed/scripts/worker_heap_ratchet.py``).
        - ``_experiment_reference_index``: the reference node method runs in the
          parent before the chunked pass and caches one member index per record, so
          every Costanzo task carried 20.7M integers, 103.5 MB, which the loader's
          per-chunk ``gc.freeze`` then pinned in the worker: 0.85 GB retained per
          chunk per worker, released only by the pool teardown; with it dropped the
          worker stays flat at 0.38 GB
          (``experiments/tcdb-002-build-speed/scripts/pool_worker_retention.py``).
        """
        state = self.__dict__.copy()
        shipped = copy.copy(state["dataset"])
        shipped._indices = None
        shipped._experiment_reference_index = None
        state["dataset"] = shipped
        return state

    def _pack_chunk(self, datas: list[Any]) -> list[Any]:
        """Return a chunk's output as objects, or as one RenderedChunk when rendering."""
        if self.row_specs is None:
            return datas
        return [RenderedChunk.from_rows(datas, self.row_specs)]

    def get_memory_reduction_factor(
        self, method_name: str, is_edge: bool = False
    ) -> float:
        """Return the configured memory reduction factor for a method (default 1.0).

        The single-pass method carries every chunked method's output per record, so
        its factor is the smallest configured factor divided by the number of
        methods folded in: a chunk then holds about as much as one method's chunk did.
        """
        if method_name in (SINGLE_PASS_NODES, SINGLE_PASS_EDGES):
            factors = [
                self.get_memory_reduction_factor(name, is_edge)
                for name, _ in self._single_pass_methods
            ]
            return min(factors) / len(factors)
        method_list = (
            self.config.cell_adapter.edge_methods
            if is_edge
            else self.config.cell_adapter.node_methods
        )
        for method in method_list:
            if method["method_name"] == method_name:
                return cast(float, method.get("memory_reduction_factor", 1.0))
        return 1.0

    @data_chunker
    def _all_chunked(self, data: dict[str, Any], method_name: str) -> list[Any]:
        """Apply every folded chunked method to one record (single-pass body)."""
        out: list[Any] = []
        for _, method in self._single_pass_methods:
            # ``__wrapped__`` is the undecorated per-record function that
            # data_chunker wrapped; calling it directly skips a nested loader.
            result = cast(Any, method).__wrapped__(self, data, method_name)
            if isinstance(result, list):
                out.extend(result)
            else:
                out.append(result)
        return out

    def _yield_methods(
        self,
        methods: list[tuple[str, Callable[..., Any]]],
        config_methods: Any,
        kind: str,
    ) -> Iterator[Any]:
        """Run the enabled methods of one kind, per method or as a single pass."""
        enabled = [
            (name, method)
            for name, method in methods
            if name in [i["method_name"] for i in config_methods]
        ]
        chunked = [(n, m) for n, m in enabled if not m.__name__.startswith("_get_")]
        for method_name, method in enabled:
            if self.single_pass and not method.__name__.startswith("_get_"):
                continue
            log.info(f"Running: {method_name}")
            BuildPhase.set(type(self).__name__, method_name, kind)
            if method.__name__.startswith("_get_"):
                yield from method()
            else:
                yield from self.get_data_by_type(
                    method, method_name, is_edge=kind == "edge"
                )
            self.event += 1
            wandb.log({"event": self.event, "method": method_name, "type": kind})
        if self.single_pass and chunked:
            pass_name = SINGLE_PASS_NODES if kind == "node" else SINGLE_PASS_EDGES
            self._single_pass_methods = chunked
            log.info(f"Running: {pass_name} ({len(chunked)} methods folded)")
            BuildPhase.set(type(self).__name__, pass_name, kind)
            yield from self.get_data_by_type(
                self._all_chunked, pass_name, is_edge=kind == "edge"
            )
            self.event += 1
            wandb.log({"event": self.event, "method": pass_name, "type": kind})

    def get_nodes(self) -> Iterator[BioCypherNode]:
        """Yield BioCypher nodes from every enabled node method in config order."""
        yield from self._yield_methods(
            self.node_methods, self.config.cell_adapter.node_methods, "node"
        )

    def get_edges(self) -> Iterator[BioCypherEdge]:
        """Yield BioCypher edges from every enabled edge method in config order."""
        yield from self._yield_methods(
            self.edge_methods, self.config.cell_adapter.edge_methods, "edge"
        )

    @property
    def supported_node_methods(self) -> list[str]:
        """Return the names of all registered node methods."""
        return [method_name for method_name, _ in self.node_methods]

    @property
    def supported_edge_methods(self) -> list[str]:
        """Return the names of all registered edge methods."""
        return [method_name for method_name, _ in self.edge_methods]

    # nodes
    def _get_experiment_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=experiment_ref_id,
                preferred_id="experiment reference",
                node_label="experiment reference",
                properties={"serialized_data": json.dumps(data.reference.model_dump())},
            )
            nodes.append(node)
        return nodes

    def _get_genome_nodes(self) -> list[BioCypherNode]:
        """One genome node per distinct reference genome.

        ``serialized_data`` is the whole ``genome_reference.model_dump()``, so a
        ``StrainReferenceGenome``'s typed ``background`` rides along in full: its
        ``alleles`` AND its ``integrations`` (the cassettes an engineered host carries at
        a named site) are in the blob with no per-field adapter change, which is how the
        background has always reached the graph. ``species`` and ``strain`` stay the
        queryable scalars.
        """
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            genome_id = hashlib.sha256(
                json.dumps(data.reference.genome_reference.model_dump()).encode("utf-8")
            ).hexdigest()
            if genome_id not in seen_node_ids:
                seen_node_ids.add(genome_id)
                node = BioCypherNode(
                    node_id=genome_id,
                    preferred_id="genome",
                    node_label="genome",
                    properties={
                        "species": data.reference.genome_reference.species,
                        "strain": data.reference.genome_reference.strain,
                        "serialized_data": json.dumps(
                            data.reference.genome_reference.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _experiment_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """The Experiment node plus the interned constants its blob points to.

        The node id is the sha256 of the fully inlined record, as before. The blob
        written to ``serialized_data`` replaces the large sub-objects (environment,
        segregant genotype) with ``{"$ref": <id>}`` pointers and emits each pointed-to
        constant as an ``interned constant`` node once per record; the sink dedups
        nodes by id, so a dataset's constant environment is written once
        (torchcell/datamodels/interned_constant.py).
        """
        dump = data["experiment"].model_dump()
        experiment_id = hashlib.sha256(json.dumps(dump).encode("utf-8")).hexdigest()
        pointered, constants = split_experiment_dump(dump)
        nodes = [
            BioCypherNode(
                node_id=experiment_id,
                preferred_id="experiment",
                node_label="experiment",
                properties={"serialized_data": json.dumps(pointered)},
            )
        ]
        # Literal label: the ontology coherence check reads emitted labels from the
        # source statically (torchcell/datamodels/ontology_checks.py).
        for ref, kind, payload in constants:
            nodes.append(
                BioCypherNode(
                    node_id=ref,
                    preferred_id="interned constant",
                    node_label="interned constant",
                    properties={"kind": kind, "serialized_data": payload},
                )
            )
        return nodes

    # --- No serialized_data on sub-object nodes ---
    # Genotype, segregant genotype, perturbation, crispr construct, environment
    # perturbation and every phenotype node carry ONLY their queryable scalar
    # properties. Their full typed record is a sub-object of the experiment record,
    # so it is already held byte for byte in the Experiment blob (or the interned
    # constant it points to); a reference-side phenotype or environment perturbation
    # is likewise inside the experiment reference blob. The node id is still the
    # sha256 of the sub-object's model_dump, so ids and edges are unchanged.

    @data_chunker
    def _genotype_node(self, data: dict[str, Any], method_name: str) -> BioCypherNode:
        genotype = data["experiment"].genotype
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=genotype_id,
            preferred_id="genotype",
            node_label="genotype",
            properties={
                "systematic_gene_names": genotype.systematic_gene_names,
                "perturbed_gene_names": genotype.perturbed_gene_names,
                "perturbation_types": genotype.perturbation_types,
            },
        )

    # --- Segregant genotypes (haplotype mosaics; a sibling of Genotype) ---
    # A SegregantGenotype has no gene-keyed perturbations, so it gets its own node
    # method (never a branch inside _genotype_node, which every served dataset
    # fingerprints). The node id is the sha256 of the whole model_dump, hashed once;
    # the blocks travel in the Experiment blob's interned-constant genotype.

    @staticmethod
    def _segregant_genotype_node_from(genotype: Any) -> BioCypherNode:
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=genotype_id,
            preferred_id="segregant genotype",
            node_label="segregant genotype",
            properties={
                "cross": genotype.cross,
                "segregant_id": genotype.segregant_id,
                "parent_1": genotype.parent_1.name,
                "parent_2": genotype.parent_2.name,
                "n_blocks": len(genotype.blocks),
            },
        )

    @data_chunker
    def _segregant_genotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        return self._segregant_genotype_node_from(data["experiment"].genotype)

    @data_chunker
    def _perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        perturbations = data["experiment"].genotype.perturbations
        nodes = []
        for perturbation in perturbations:
            perturbation_id = hashlib.sha256(
                json.dumps(perturbation.model_dump()).encode("utf-8")
            ).hexdigest()
            node = BioCypherNode(
                node_id=perturbation_id,
                preferred_id=perturbation.perturbation_type,
                node_label="perturbation",
                properties={
                    "systematic_gene_name": perturbation.systematic_gene_name,
                    "perturbed_gene_name": perturbation.perturbed_gene_name,
                    "perturbation_type": perturbation.perturbation_type,
                    "description": perturbation.description,
                    # strain_id is declared only on the Sga*/Marker/Natural*/
                    # SequenceVariant/CopyNumberVariant leaves -- not on the base
                    # KanMxDeletion/GeneAddition types the metabolite/morphology
                    # datasets use, so read it defensively.
                    "strain_id": getattr(perturbation, "strain_id", None),
                },
            )
            nodes.append(node)
        return nodes

    # --- Bacterial perturbations (a sibling class of `perturbation`) ---
    # A bacterial leaf carries gene_namespace, the host locus-tag space its
    # systematic_gene_name is written in; the served `perturbation` class has no such
    # property and cannot gain one without a full rebuild, so these leaves are their own
    # class. The node id is the sha256 of the leaf's model_dump, the id
    # _perturbation_to_genotype_edges and _crispr_construct_to_perturbation_edges
    # already address, so those edge methods connect these nodes unchanged.

    @staticmethod
    def _bacterial_perturbation_node_from(perturbation: Any) -> BioCypherNode:
        perturbation_id = hashlib.sha256(
            json.dumps(perturbation.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=perturbation_id,
            preferred_id=perturbation.perturbation_type,
            node_label="bacterial perturbation",
            properties={
                "systematic_gene_name": perturbation.systematic_gene_name,
                "perturbed_gene_name": perturbation.perturbed_gene_name,
                "perturbation_type": perturbation.perturbation_type,
                "description": perturbation.description,
                "gene_namespace": perturbation.gene_namespace,
            },
        )

    @data_chunker
    def _bacterial_perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """One node per perturbation of a bacterial leaf class; none for a yeast leaf.

        A called-variant leaf is excluded even when it is a subclass of one of these
        (``BacterialSpanDeletionPerturbation`` is a ``BacterialDeletionPerturbation``):
        it belongs to ``bacterial sequence variant perturbation`` and writing it here
        too would serve one perturbation as two nodes.
        """
        return [
            self._bacterial_perturbation_node_from(perturbation)
            for perturbation in data["experiment"].genotype.perturbations
            if isinstance(perturbation, BACTERIAL_PERTURBATION_LEAVES)
            and not isinstance(perturbation, BACTERIAL_VARIANT_PERTURBATION_LEAVES)
        ]

    # --- Called bacterial variants (issue #731) ---
    # A variant leaf composes a ``BacterialVariantCall``, so the replicon, the 1-based
    # interval, the variant kind and the call frequency are read off ``.call`` rather
    # than off the leaf. The node id is the sha256 of the leaf's model_dump, the same id
    # _perturbation_to_genotype_edges addresses, so that edge method is unchanged.

    @staticmethod
    def _bacterial_variant_perturbation_node_from(perturbation: Any) -> BioCypherNode:
        perturbation_id = hashlib.sha256(
            json.dumps(perturbation.model_dump()).encode("utf-8")
        ).hexdigest()
        call = perturbation.call
        return BioCypherNode(
            node_id=perturbation_id,
            preferred_id=perturbation.perturbation_type,
            node_label="bacterial sequence variant perturbation",
            properties={
                "systematic_gene_name": perturbation.systematic_gene_name,
                "perturbed_gene_name": perturbation.perturbed_gene_name,
                "perturbation_type": perturbation.perturbation_type,
                "description": perturbation.description,
                "gene_namespace": perturbation.gene_namespace,
                "reference_sequence": call.reference_sequence,
                "position_start": call.position_start,
                "position_end": call.position_end,
                "variant_type": str(call.variant_type),
                # None when the release wrote a RANGE rather than one number; the
                # verbatim cell stays in the Experiment blob either way.
                "variant_frequency": call.frequency,
                "call_mode": str(call.call_mode),
            },
        )

    @data_chunker
    def _bacterial_variant_perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """One node per CALLED variant of the genotype; none for any other leaf."""
        return [
            self._bacterial_variant_perturbation_node_from(perturbation)
            for perturbation in data["experiment"].genotype.perturbations
            if isinstance(perturbation, BACTERIAL_VARIANT_PERTURBATION_LEAVES)
        ]

    # Environment.temperature is Optional: a curation layer that never carried a
    # temperature records a typed gap instead of guessing one. Every read of it is
    # therefore bound to a local first and guarded -- an unguarded walk turns a legal
    # record into an AttributeError at KG-build time. A present temperature yields
    # exactly the node, edge and property bytes it did before.

    # --- CRISPR constructs (the reagent a CRISPR perturbation was made with) ---
    # A CrisprConstruct is COMPOSED onto the CRISPR leaves (CrisprDeletionPerturbation
    # and the CRISPRa/CRISPRi expression family), so it is read off the perturbation,
    # not off the genotype. It gets its OWN node class rather than extra properties on
    # `perturbation`: `perturbation` is a served graph class, and adding a property to
    # it would force a full rebuild of all 36 served datasets. `crispr` is declared only
    # on the CRISPR leaves, so -- as with `strain_id` in _perturbation_node -- it is read
    # defensively off a union whose other members do not carry it.

    @staticmethod
    def _crispr_construct_node_from(construct: Any) -> BioCypherNode:
        construct_id = hashlib.sha256(
            json.dumps(construct.model_dump()).encode("utf-8")
        ).hexdigest()
        # The plasmid ArtifactRef is projected flat, as a node property can hold only
        # scalars: its tc:// location string and the sha256 it pins (both None today).
        plasmid = construct.effector_plasmid_ref
        return BioCypherNode(
            node_id=construct_id,
            preferred_id="crispr construct",
            node_label="crispr construct",
            properties={
                "effector": construct.effector,
                "guide_sequence": construct.guide_sequence,
                "n_guides": construct.n_guides,
                "library_pool": construct.library_pool,
                "effector_plasmid_ref": None if plasmid is None else str(plasmid),
                "effector_plasmid_sha256": None if plasmid is None else plasmid.sha256,
            },
        )

    @data_chunker
    def _crispr_construct_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """Emit one node per CRISPR construct carried by this record's perturbations."""
        nodes = []
        for perturbation in data["experiment"].genotype.perturbations:
            construct = getattr(perturbation, "crispr", None)
            if construct is not None:
                nodes.append(self._crispr_construct_node_from(construct))
        return nodes

    @data_chunker
    def _crispr_construct_to_perturbation_edges(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        """Link each CRISPR construct to the perturbation it was used to make."""
        edges = []
        for perturbation in data["experiment"].genotype.perturbations:
            construct = getattr(perturbation, "crispr", None)
            if construct is None:
                continue
            edges.append(
                BioCypherEdge(
                    source_id=hashlib.sha256(
                        json.dumps(construct.model_dump()).encode("utf-8")
                    ).hexdigest(),
                    target_id=hashlib.sha256(
                        json.dumps(perturbation.model_dump()).encode("utf-8")
                    ).hexdigest(),
                    relationship_label="crispr construct member of",
                )
            )
        return edges

    # --- Environment-side node ids: identity by COMPOSITION, not by quote ---
    # A medium, a temperature, an environment perturbation and an environment are
    # persistent entities two datasets can both state, so their ids come from what
    # the entity IS (``torchcell.datamodels.identity``), not from the full pydantic
    # dump. The dump carries the stating dataset's provenance quotes, notes and free
    # text ``name``, so hashing it gave two datasets on the same YPD two media nodes
    # and no cross-dataset aggregate could form. One function per class, called by
    # the node method AND by every edge method, so an edge can never address a node
    # the graph does not contain.

    @staticmethod
    def _media_node_id(media: Any) -> str:
        """Content-address a ``Media`` by its composition."""
        return identity_sha256(media_identity(media))

    @staticmethod
    def _temperature_node_id(temperature: Any) -> str:
        """Content-address a ``Temperature`` by its value and typed unit."""
        return identity_sha256(temperature_identity(temperature))

    @staticmethod
    def _environment_perturbation_node_id(perturbation: Any) -> str:
        """Content-address an environment perturbation by its typed slots."""
        return identity_sha256(environment_perturbation_identity(perturbation))

    @staticmethod
    def _environment_node_id(environment: Any) -> str:
        """Content-address an ``Environment`` by medium, temperature, edits, duration."""
        return identity_sha256(environment_identity(environment))

    @data_chunker
    def _environment_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        environment = data["experiment"].environment
        environment_id = self._environment_node_id(environment)
        media = json.dumps(environment.media.model_dump())
        temperature = environment.temperature
        return BioCypherNode(
            node_id=environment_id,
            preferred_id="environment",
            node_label="environment",
            properties={
                "temperature": (temperature.value if temperature is not None else None),
                "media": media,
                "serialized_data": json.dumps(environment.model_dump()),
            },
        )

    # --- Environment perturbations (the environment axis of Genotype.perturbations) ---
    # An added compound / physical factor / biologic is its own node, content-addressed
    # like a gene perturbation, so a condition such as "YPD + 0.4 M NaCl" is queryable
    # rather than only embedded in the environment's serialized_data.
    #
    # A `PhagePerturbation` is NOT emitted here: it has its own node class and its own
    # method below, and both ids are the same composition projection, so a conf enabling
    # both lanes would otherwise write one content id under two labels (issue #756). The
    # two lanes PARTITION `environment.perturbations` -- phages to `phage perturbation`,
    # every other leaf to `environment perturbation` -- so a dataset whose environment
    # carries a phage AND a compound enables both and each leaf is written exactly once.
    # The filter edits a SERVED method, which is adapter drift on every served dataset
    # enabling it and therefore a full rebuild; it changes no served OUTPUT, since no
    # served dataset's environment carries a phage.

    @staticmethod
    def _environment_perturbation_node_from(perturbation: Any) -> BioCypherNode:
        perturbation_id = CellAdapter._environment_perturbation_node_id(perturbation)
        # A small molecule carries compound + concentration; a physical factor carries
        # factor + magnitude (+ an optional agent, the acid that set the pH); both
        # project onto the same columns so pH 4.5 is as queryable as 0.4 M NaCl.
        compound = getattr(perturbation, "compound", None)
        if compound is None:
            compound = getattr(perturbation, "agent", None)
        dose = getattr(perturbation, "concentration", None)
        if dose is None:
            dose = getattr(perturbation, "magnitude", None)
        factor = getattr(perturbation, "factor", None)
        return BioCypherNode(
            node_id=perturbation_id,
            preferred_id=perturbation.perturbation_type,
            node_label="environment perturbation",
            properties={
                "perturbation_type": perturbation.perturbation_type,
                "description": perturbation.description,
                "factor": str(factor.value) if factor is not None else None,
                "compound_name": compound.name if compound is not None else None,
                "inchikey": compound.inchikey if compound is not None else None,
                "concentration_value": dose.value if dose is not None else None,
                "concentration_unit": (
                    str(dose.unit.value)
                    if dose is not None and dose.unit is not None
                    else None
                ),
            },
        )

    @data_chunker
    def _environment_perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """One node per NON-phage perturbation of the environment; phages have their own."""
        return [
            self._environment_perturbation_node_from(perturbation)
            for perturbation in data["experiment"].environment.perturbations
            if not isinstance(perturbation, PhagePerturbation)
        ]

    def _get_environment_perturbation_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            for perturbation in data.reference.environment_reference.perturbations:
                if isinstance(perturbation, PhagePerturbation):
                    continue
                node = self._environment_perturbation_node_from(perturbation)
                if node.get_id() not in seen_node_ids:
                    seen_node_ids.add(node.get_id())
                    nodes.append(node)
        return nodes

    # --- Phage challenges (the environment axis of a phage-resistance screen) ---
    # A phage gets its OWN node class and method for the same reason `crispr construct`
    # and `bacterial perturbation` do: giving `environment perturbation` the MOI, the
    # taxon and the accession as properties would change a served class, and the dose is
    # not a concentration so it cannot ride the concentration columns.
    # This method and `_environment_perturbation_node` PARTITION the environment's
    # perturbations on `isinstance(..., PhagePerturbation)`, so a conf may enable both
    # lanes and no content id is written under two labels (issue #756). The id is the
    # same composition projection the other environment-side nodes use, so
    # `_environment_perturbation_to_environment_edges` addresses these nodes unchanged.

    @staticmethod
    def _phage_perturbation_node_from(perturbation: Any) -> BioCypherNode:
        perturbation_id = CellAdapter._environment_perturbation_node_id(perturbation)
        return BioCypherNode(
            node_id=perturbation_id,
            preferred_id=perturbation.perturbation_type,
            node_label="phage perturbation",
            properties={
                "perturbation_type": perturbation.perturbation_type,
                "description": perturbation.description,
                # `phage_name`, not `name`: the sibling `environment perturbation` class
                # names its agent column `compound_name` and the served `media` class
                # uses `name` for a medium's label, so the agent's name is qualified by
                # what it names. The pydantic field stays `name`.
                "phage_name": perturbation.name,
                "ncbi_taxid": perturbation.ncbi_taxid,
                "genome_accession": perturbation.genome_accession,
                "multiplicity_of_infection": perturbation.multiplicity_of_infection,
                "titer_pfu_per_ml": perturbation.titer_pfu_per_ml,
            },
        )

    @data_chunker
    def _phage_perturbation_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """One node per phage of the environment; none for any other perturbation."""
        return [
            self._phage_perturbation_node_from(perturbation)
            for perturbation in data["experiment"].environment.perturbations
            if isinstance(perturbation, PhagePerturbation)
        ]

    def _get_phage_perturbation_reference_nodes(self) -> list[BioCypherNode]:
        """The phages of every reference environment, deduplicated by content id."""
        nodes: list[BioCypherNode] = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            for perturbation in data.reference.environment_reference.perturbations:
                if not isinstance(perturbation, PhagePerturbation):
                    continue
                node = self._phage_perturbation_node_from(perturbation)
                if node.get_id() not in seen_node_ids:
                    seen_node_ids.add(node.get_id())
                    nodes.append(node)
        return nodes

    def _get_environment_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            environment = data.reference.environment_reference
            environment_id = self._environment_node_id(environment)
            if environment_id not in seen_node_ids:
                seen_node_ids.add(environment_id)
                media = json.dumps(environment.media.model_dump())
                temperature = environment.temperature
                node = BioCypherNode(
                    node_id=environment_id,
                    preferred_id="environment",
                    node_label="environment",
                    properties={
                        "temperature": (
                            temperature.value if temperature is not None else None
                        ),
                        "media": media,
                        "serialized_data": json.dumps(environment.model_dump()),
                    },
                )
                nodes.append(node)
        return nodes

    @data_chunker
    def _media_node(self, data: dict[str, Any], method_name: str) -> BioCypherNode:
        media_id = self._media_node_id(data["experiment"].environment.media)
        name = data["experiment"].environment.media.name
        state = data["experiment"].environment.media.state
        return BioCypherNode(
            node_id=media_id,
            preferred_id="media",
            node_label="media",
            properties={
                "name": name,
                "state": state,
                "serialized_data": json.dumps(
                    data["experiment"].environment.media.model_dump()
                ),
            },
        )

    def _get_media_reference_nodes(self) -> list[BioCypherNode]:
        seen_node_ids = set()
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            media_id = self._media_node_id(data.reference.environment_reference.media)
            if media_id not in seen_node_ids:
                seen_node_ids.add(media_id)
                name = data.reference.environment_reference.media.name
                state = data.reference.environment_reference.media.state
                node = BioCypherNode(
                    node_id=media_id,
                    preferred_id="media",
                    node_label="media",
                    properties={
                        "name": name,
                        "state": state,
                        "serialized_data": json.dumps(
                            data.reference.environment_reference.media.model_dump()
                        ),
                    },
                )
                nodes.append(node)
        return nodes

    @staticmethod
    def _temperature_node_from(temperature: Any) -> BioCypherNode:
        temperature_id = CellAdapter._temperature_node_id(temperature)
        return BioCypherNode(
            node_id=temperature_id,
            preferred_id="temperature",
            node_label="temperature",
            properties={
                "value": temperature.value,
                "unit": temperature.unit,
                "serialized_data": json.dumps(temperature.model_dump()),
            },
        )

    @data_chunker
    def _temperature_node(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherNode]:
        """Emit the temperature node, or nothing when the record gaps temperature."""
        temperature = data["experiment"].environment.temperature
        if temperature is None:
            return []
        return [self._temperature_node_from(temperature)]

    def _get_temperature_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            temperature = data.reference.environment_reference.temperature
            if temperature is None:
                continue
            node = self._temperature_node_from(temperature)
            if node.get_id() not in seen_node_ids:
                seen_node_ids.add(node.get_id())
                nodes.append(node)
        return nodes

    @data_chunker
    def _fitness_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        fitness = phenotype.fitness
        fitness_std = phenotype.fitness_std

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "fitness": fitness,
            "fitness_std": fitness_std,
            # The screen a measurement came from (Kuzmin 2020 main vs pilot screens,
            # issue #602); None for a source with one screen per measurement.
            "screen_id": phenotype.screen_id,
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="fitness phenotype",
            properties=properties,
        )

    @data_chunker
    def _gene_interaction_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        gene_interaction = phenotype.gene_interaction
        gene_interaction_p_value = phenotype.gene_interaction_p_value

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "gene_interaction": gene_interaction,
            "gene_interaction_p_value": gene_interaction_p_value,
            "screen_id": phenotype.screen_id,
            # #793: the replicate-design quartet, so a sourced eight-colony design is
            # queryable from the graph instead of living beside the build.
            "n_samples": phenotype.n_samples,
            "sample_unit": (
                str(phenotype.sample_unit.value)
                if phenotype.sample_unit is not None
                else None
            ),
            "gene_interaction_uncertainty": phenotype.gene_interaction_uncertainty,
            "gene_interaction_uncertainty_type": (
                str(phenotype.gene_interaction_uncertainty_type.value)
                if phenotype.gene_interaction_uncertainty_type is not None
                else None
            ),
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="gene interaction phenotype",
            properties=properties,
        )

    @data_chunker
    def _gene_essentiality_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        is_essential = phenotype.is_essential

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "is_essential": is_essential,
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="gene essentiality phenotype",
            properties=properties,
        )

    @data_chunker
    def _synthetic_lethality_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        is_synthetic_lethal = phenotype.is_synthetic_lethal
        synthetic_lethality_statistic_score = (
            phenotype.synthetic_lethality_statistic_score
        )

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "is_synthetic_lethal": is_synthetic_lethal,
            "synthetic_lethality_statistic_score": synthetic_lethality_statistic_score,
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="synthetic lethality phenotype",
            properties=properties,
        )

    @data_chunker
    def _synthetic_rescue_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        is_synthetic_rescue = phenotype.is_synthetic_rescue
        synthetic_rescue_statistic_score = phenotype.synthetic_rescue_statistic_score

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "is_synthetic_rescue": is_synthetic_rescue,
            "synthetic_rescue_statistic_score": synthetic_rescue_statistic_score,
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="synthetic rescue phenotype",
            properties=properties,
        )

    # --- Environment response phenotype (chemogenomic / segregant growth) ---
    # The typed record is in the Experiment blob; the scalar response, its SE and two
    # typed axes (measurement_type = WHAT the number is, assay_type = HOW it was
    # measured) are projected so a condition-response query never has to parse JSON.
    # A categorical or ordinal screen carries no number, so its call is projected too:
    # `category` on the shared ResponseCategory axis (what joins across screens) and
    # `category_label` verbatim from the source (what makes the mapping auditable).
    # `screen_id` names the screening run, so two screens of one compound at one dose
    # stay separable.

    @staticmethod
    def _environment_response_properties(phenotype: Any) -> dict[str, Any]:
        assay_type = phenotype.assay_type
        category = phenotype.category
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "environment_response": phenotype.environment_response,
            "environment_response_se": phenotype.environment_response_se,
            "measurement_type": str(phenotype.measurement_type.value),
            "assay_type": str(assay_type.value) if assay_type is not None else None,
            "category": str(category.value) if category is not None else None,
            "category_label": phenotype.category_label,
            "screen_id": phenotype.screen_id,
            # #776: both limits of the released confidence interval plus its level
            # (an asymmetric interval has no half-width), and the replicate id of a
            # per-replicate release.
            "environment_response_lower": phenotype.environment_response_lower,
            "environment_response_upper": phenotype.environment_response_upper,
            "confidence_level": phenotype.confidence_level,
            "replicate_id": phenotype.replicate_id,
            # #863: the released test of the response and its correction.
            "environment_response_p_value": phenotype.environment_response_p_value,
            "environment_response_p_value_adjusted": (
                phenotype.environment_response_p_value_adjusted
            ),
            "p_value_adjustment_method": phenotype.p_value_adjustment_method,
        }

    @data_chunker
    def _environment_response_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="environment response phenotype",
            properties=self._environment_response_properties(phenotype),
        )

    def _get_environment_response_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="environment response phenotype",
                    node_label="environment response phenotype",
                    properties=self._environment_response_properties(phenotype),
                )
            )
        return nodes

    def _get_fitness_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            fitness = phenotype.fitness
            fitness_std = phenotype.fitness_std

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "fitness": fitness,
                "fitness_std": fitness_std,
                "screen_id": phenotype.screen_id,
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="fitness phenotype",
                node_label="fitness phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_gene_interaction_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            gene_interaction = phenotype.gene_interaction
            gene_interaction_p_value = phenotype.gene_interaction_p_value

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "gene_interaction": gene_interaction,
                "gene_interaction_p_value": gene_interaction_p_value,
                "screen_id": phenotype.screen_id,
                # #793: the replicate-design quartet, so a sourced eight-colony design is
                # queryable from the graph instead of living beside the build.
                "n_samples": phenotype.n_samples,
                "sample_unit": (
                    str(phenotype.sample_unit.value)
                    if phenotype.sample_unit is not None
                    else None
                ),
                "gene_interaction_uncertainty": phenotype.gene_interaction_uncertainty,
                "gene_interaction_uncertainty_type": (
                    str(phenotype.gene_interaction_uncertainty_type.value)
                    if phenotype.gene_interaction_uncertainty_type is not None
                    else None
                ),
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="gene interaction phenotype",
                node_label="gene interaction phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_gene_essentiality_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            is_essential = phenotype.is_essential

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "is_essential": is_essential,
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="gene essentiality phenotype",
                node_label="gene essentiality phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_synthetic_lethality_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            is_synthetic_lethal = phenotype.is_synthetic_lethal
            statistic_score = phenotype.synthetic_lethality_statistic_score

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "is_synthetic_lethal": is_synthetic_lethal,
                "synthetic_lethality_statistic_score": statistic_score,
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="synthetic lethality phenotype",
                node_label="synthetic lethality phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    def _get_synthetic_rescue_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            is_synthetic_rescue = phenotype.is_synthetic_rescue
            synthetic_rescue_statistic_score = (
                phenotype.synthetic_rescue_statistic_score
            )

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "is_synthetic_rescue": is_synthetic_rescue,
                "synthetic_rescue_statistic_score": synthetic_rescue_statistic_score,
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="synthetic rescue phenotype",
                node_label="synthetic rescue phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    @data_chunker
    def _calmorph_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()

        graph_level = phenotype.graph_level
        label_name = phenotype.label_name
        label_statistic_name = phenotype.label_statistic_name
        calmorph = phenotype.calmorph
        calmorph_coefficient_of_variation = phenotype.calmorph_coefficient_of_variation

        properties = {
            "graph_level": graph_level,
            "label_name": label_name,
            "label_statistic_name": label_statistic_name,
            "calmorph": json.dumps(calmorph),  # Store as JSON string
            "calmorph_coefficient_of_variation": json.dumps(
                calmorph_coefficient_of_variation
            )
            if calmorph_coefficient_of_variation
            else None,
        }

        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="calmorph phenotype",
            properties=properties,
        )

    def _get_calmorph_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference

            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()

            graph_level = phenotype.graph_level
            label_name = phenotype.label_name
            label_statistic_name = phenotype.label_statistic_name
            calmorph = phenotype.calmorph
            calmorph_coefficient_of_variation = (
                phenotype.calmorph_coefficient_of_variation
            )

            properties = {
                "graph_level": graph_level,
                "label_name": label_name,
                "label_statistic_name": label_statistic_name,
                "calmorph": json.dumps(calmorph),  # Store as JSON string
                "calmorph_coefficient_of_variation": json.dumps(
                    calmorph_coefficient_of_variation
                )
                if calmorph_coefficient_of_variation
                else None,
            }

            node = BioCypherNode(
                node_id=phenotype_id,
                preferred_id="calmorph phenotype",
                node_label="calmorph phenotype",
                properties=properties,
            )
            nodes.append(node)
        return nodes

    # --- Expression / metabolite / visual-score phenotypes (abstract datasets) ---
    # Multi-valued phenotypes (expression, metabolite) serialize their per-key dicts to
    # JSON strings (the CalMorph pattern); the scalar VisualScore stores plain scalars.

    @data_chunker
    def _microarray_expression_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "expression_log2_ratio": json.dumps(phenotype.expression_log2_ratio),
            "expression_log2_ratio_se": (
                json.dumps(phenotype.expression_log2_ratio_se)
                if phenotype.expression_log2_ratio_se is not None
                else None
            ),
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="microarray expression phenotype",
            properties=properties,
        )

    def _get_microarray_expression_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "expression_log2_ratio": json.dumps(phenotype.expression_log2_ratio),
                "expression_log2_ratio_se": (
                    json.dumps(phenotype.expression_log2_ratio_se)
                    if phenotype.expression_log2_ratio_se is not None
                    else None
                ),
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="microarray expression phenotype",
                    node_label="microarray expression phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _rnaseq_expression_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "expression_tpm": json.dumps(phenotype.expression_tpm),
            "measurement_type": phenotype.measurement_type,
            "n_mapped_reads": phenotype.n_mapped_reads,
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="rnaseq expression phenotype",
            properties=properties,
        )

    def _get_rnaseq_expression_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "expression_tpm": json.dumps(phenotype.expression_tpm),
                "measurement_type": phenotype.measurement_type,
                "n_mapped_reads": phenotype.n_mapped_reads,
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="rnaseq expression phenotype",
                    node_label="rnaseq expression phenotype",
                    properties=properties,
                )
            )
        return nodes

    @staticmethod
    def _pseudobulk_expression_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties for a ``PseudobulkExpressionPhenotype`` (experiment or reference).

        The per-gene log2 fold-change dict is stored as a JSON string (the multi-valued
        phenotype convention); ``dispersion`` and ``n_cells`` are the per-genotype
        single-cell scalars and stay typed so they are queryable.
        """
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "expression_log2_ratio": json.dumps(phenotype.expression_log2_ratio),
            "dispersion": phenotype.dispersion,
            "n_cells": phenotype.n_cells,
            "measurement_type": phenotype.measurement_type,
        }

    @data_chunker
    def _pseudobulk_expression_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="pseudobulk expression phenotype",
            properties=self._pseudobulk_expression_properties(phenotype),
        )

    def _get_pseudobulk_expression_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="pseudobulk expression phenotype",
                    node_label="pseudobulk expression phenotype",
                    properties=self._pseudobulk_expression_properties(phenotype),
                )
            )
        return nodes

    @data_chunker
    def _visual_score_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "visual_score": phenotype.visual_score,
            "n_replicates": phenotype.n_replicates,
            "target_product": phenotype.target_product,
            "target_metabolite_id": phenotype.target_metabolite_id,
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="visual score phenotype",
            properties=properties,
        )

    def _get_visual_score_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "visual_score": phenotype.visual_score,
                "n_replicates": phenotype.n_replicates,
                "target_product": phenotype.target_product,
                "target_metabolite_id": phenotype.target_metabolite_id,
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="visual score phenotype",
                    node_label="visual score phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _metabolite_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "metabolite_level": json.dumps(phenotype.metabolite_level),
            "metabolite_level_se": (
                json.dumps(phenotype.metabolite_level_se)
                if phenotype.metabolite_level_se is not None
                else None
            ),
            "measurement_type": phenotype.measurement_type,
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="metabolite phenotype",
            properties=properties,
        )

    def _get_metabolite_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "metabolite_level": json.dumps(phenotype.metabolite_level),
                "metabolite_level_se": (
                    json.dumps(phenotype.metabolite_level_se)
                    if phenotype.metabolite_level_se is not None
                    else None
                ),
                "measurement_type": phenotype.measurement_type,
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="metabolite phenotype",
                    node_label="metabolite phenotype",
                    properties=properties,
                )
            )
        return nodes

    @data_chunker
    def _protein_abundance_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        properties = {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "protein_abundance": json.dumps(phenotype.protein_abundance),
            "protein_abundance_se": (
                json.dumps(phenotype.protein_abundance_se)
                if phenotype.protein_abundance_se is not None
                else None
            ),
            "measurement_type": phenotype.measurement_type,
        }
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="protein abundance phenotype",
            properties=properties,
        )

    def _get_protein_abundance_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            properties = {
                "graph_level": phenotype.graph_level,
                "label_name": phenotype.label_name,
                "label_statistic_name": phenotype.label_statistic_name,
                "protein_abundance": json.dumps(phenotype.protein_abundance),
                "protein_abundance_se": (
                    json.dumps(phenotype.protein_abundance_se)
                    if phenotype.protein_abundance_se is not None
                    else None
                ),
                "measurement_type": phenotype.measurement_type,
            }
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="protein abundance phenotype",
                    node_label="protein abundance phenotype",
                    properties=properties,
                )
            )
        return nodes

    # --- Product titer, protein turnover and flux phenotypes ---
    # Shaped as _fitness_phenotype_node: id = sha256 of the phenotype's model_dump,
    # preferred_id phenotype_<id> on the experiment side and the class name on the
    # reference side, no serialized_data. The properties are every field except
    # provenance_gaps, named by the field: an enum as its value, a dict and the product
    # Compound as a JSON string, None kept as None.

    @staticmethod
    def _product_titer_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``ProductTiterPhenotype`` (experiment or reference)."""
        uncertainty_type = phenotype.titer_uncertainty_type
        sample_unit = phenotype.sample_unit
        yield_unit = phenotype.product_yield_unit
        productivity_unit = phenotype.productivity_unit
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "product": json.dumps(phenotype.product.model_dump()),
            "titer": phenotype.titer,
            "titer_unit": str(phenotype.titer_unit.value),
            "titer_se": phenotype.titer_se,
            "titer_uncertainty": phenotype.titer_uncertainty,
            "titer_uncertainty_type": (
                str(uncertainty_type.value) if uncertainty_type is not None else None
            ),
            "n_samples": phenotype.n_samples,
            "sample_unit": str(sample_unit.value) if sample_unit is not None else None,
            "product_yield": phenotype.product_yield,
            "product_yield_unit": (
                str(yield_unit.value) if yield_unit is not None else None
            ),
            "productivity": phenotype.productivity,
            "productivity_unit": (
                str(productivity_unit.value) if productivity_unit is not None else None
            ),
            "quantification_method": phenotype.quantification_method,
        }

    @data_chunker
    def _product_titer_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="product titer phenotype",
            properties=self._product_titer_properties(phenotype),
        )

    def _get_product_titer_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="product titer phenotype",
                    node_label="product titer phenotype",
                    properties=self._product_titer_properties(phenotype),
                )
            )
        return nodes

    # --- begin #770: the protein fold-change family ---
    @staticmethod
    def _protein_fold_change_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``ProteinFoldChangePhenotype`` (experiment or reference)."""
        standard_errors = phenotype.protein_fold_change_se
        p_values = phenotype.protein_fold_change_p_value
        p_values_adjusted = phenotype.protein_fold_change_p_value_adjusted
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "protein_fold_change": json.dumps(phenotype.protein_fold_change),
            "protein_fold_change_se": (
                json.dumps(standard_errors) if standard_errors is not None else None
            ),
            "protein_fold_change_p_value": (
                json.dumps(p_values) if p_values is not None else None
            ),
            "protein_fold_change_p_value_adjusted": (
                json.dumps(p_values_adjusted) if p_values_adjusted is not None else None
            ),
            "p_value_adjustment_method": phenotype.p_value_adjustment_method,
            "fold_change_scale": str(phenotype.fold_change_scale),
            "reference_basis": phenotype.reference_basis,
            "n_replicates": json.dumps(phenotype.n_replicates),
            "measurement_type": phenotype.measurement_type,
        }

    @data_chunker
    def _protein_fold_change_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="protein fold change phenotype",
            properties=self._protein_fold_change_properties(phenotype),
        )

    def _get_protein_fold_change_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="protein fold change phenotype",
                    node_label="protein fold change phenotype",
                    properties=self._protein_fold_change_properties(phenotype),
                )
            )
        return nodes

    # --- end #770 ---

    @staticmethod
    def _protein_turnover_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``ProteinTurnoverPhenotype`` (experiment or reference)."""
        degradation_rate_se = phenotype.degradation_rate_se
        half_life = phenotype.half_life
        synthesis_rate = phenotype.synthesis_rate
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "degradation_rate": json.dumps(phenotype.degradation_rate),
            "degradation_rate_se": (
                json.dumps(degradation_rate_se)
                if degradation_rate_se is not None
                else None
            ),
            "half_life": json.dumps(half_life) if half_life is not None else None,
            "synthesis_rate": (
                json.dumps(synthesis_rate) if synthesis_rate is not None else None
            ),
            "n_replicates": json.dumps(phenotype.n_replicates),
            "measurement_type": phenotype.measurement_type,
            # --- begin #753: the published interval and the censoring flag ---
            "degradation_rate_lower": (
                json.dumps(phenotype.degradation_rate_lower)
                if phenotype.degradation_rate_lower is not None
                else None
            ),
            "degradation_rate_upper": (
                json.dumps(phenotype.degradation_rate_upper)
                if phenotype.degradation_rate_upper is not None
                else None
            ),
            "confidence_level": phenotype.confidence_level,
            "interval_method": phenotype.interval_method,
            "censoring": (
                json.dumps(
                    {key: str(value) for key, value in phenotype.censoring.items()}
                )
                if phenotype.censoring is not None
                else None
            ),
            # --- end #753 ---
        }

    @data_chunker
    def _protein_turnover_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="protein turnover phenotype",
            properties=self._protein_turnover_properties(phenotype),
        )

    def _get_protein_turnover_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="protein turnover phenotype",
                    node_label="protein turnover phenotype",
                    properties=self._protein_turnover_properties(phenotype),
                )
            )
        return nodes

    @staticmethod
    def _flux_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``FluxPhenotype`` (experiment or reference)."""
        lower = phenotype.net_flux_lower
        upper = phenotype.net_flux_upper
        sample_unit = phenotype.sample_unit
        target_reaction_ids = phenotype.target_reaction_ids
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "net_flux": json.dumps(phenotype.net_flux),
            "net_flux_lower": json.dumps(lower) if lower is not None else None,
            "net_flux_upper": json.dumps(upper) if upper is not None else None,
            "confidence_level": phenotype.confidence_level,
            "measurement_type": phenotype.measurement_type,
            "n_samples": phenotype.n_samples,
            "sample_unit": str(sample_unit.value) if sample_unit is not None else None,
            "target_reaction_ids": (
                json.dumps(target_reaction_ids)
                if target_reaction_ids is not None
                else None
            ),
        }

    @data_chunker
    def _flux_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="flux phenotype",
            properties=self._flux_properties(phenotype),
        )

    def _get_flux_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="flux phenotype",
                    node_label="flux phenotype",
                    properties=self._flux_properties(phenotype),
                )
            )
        return nodes

    @staticmethod
    def _promoter_activity_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``PromoterActivityPhenotype`` (experiment or reference).

        Every value is a scalar: a record measures one promoter, so this family has no
        dict-valued field to JSON-encode the way the profile phenotypes do.
        """
        uncertainty_type = phenotype.promoter_activity_uncertainty_type
        sample_unit = phenotype.sample_unit
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "promoter_activity": phenotype.promoter_activity,
            "promoter_activity_se": phenotype.promoter_activity_se,
            "promoter_activity_uncertainty": phenotype.promoter_activity_uncertainty,
            "promoter_activity_uncertainty_type": (
                uncertainty_type.value if uncertainty_type is not None else None
            ),
            "n_samples": phenotype.n_samples,
            "sample_unit": sample_unit.value if sample_unit is not None else None,
            "promoter_name": phenotype.promoter_name,
            "promoter_gene": phenotype.promoter_gene,
            "readout": phenotype.readout.value,
            "reporter_gene": phenotype.reporter_gene,
            "activity_units": phenotype.activity_units,
            "well_id": phenotype.well_id,
        }

    @data_chunker
    def _promoter_activity_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="promoter activity phenotype",
            properties=self._promoter_activity_properties(phenotype),
        )

    def _get_promoter_activity_phenotype_reference_nodes(self) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="promoter activity phenotype",
                    node_label="promoter activity phenotype",
                    properties=self._promoter_activity_properties(phenotype),
                )
            )
        return nodes

    @staticmethod
    def _bacterial_morphology_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``BacterialMorphologyPhenotype`` (experiment or reference).

        The two dict-valued fields serialize to JSON strings, the CalMorph convention,
        and ``assay`` rides beside them because the keys inside those strings are only
        interpretable against the assay vocabulary that named them.
        """
        coefficients = phenotype.morphology_coefficient_of_variation
        sample_unit = phenotype.sample_unit
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "assay": phenotype.assay,
            "morphology": json.dumps(phenotype.morphology),
            "morphology_coefficient_of_variation": (
                json.dumps(coefficients) if coefficients else None
            ),
            "n_samples": phenotype.n_samples,
            "sample_unit": sample_unit.value if sample_unit is not None else None,
        }

    @data_chunker
    def _bacterial_morphology_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="bacterial morphology phenotype",
            properties=self._bacterial_morphology_properties(phenotype),
        )

    def _get_bacterial_morphology_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="bacterial morphology phenotype",
                    node_label="bacterial morphology phenotype",
                    properties=self._bacterial_morphology_properties(phenotype),
                )
            )
        return nodes

    @staticmethod
    def _mrna_number_fraction_properties(phenotype: Any) -> dict[str, Any]:
        """Node properties of a ``MrnaNumberFractionPhenotype`` (experiment or reference).

        The per-gene dict serializes to a JSON string, the multi-valued phenotype
        convention; ``n_libraries`` and ``measurement_type`` stay typed so a query can
        tell a single library from a replicate mean without parsing the dict.
        """
        return {
            "graph_level": phenotype.graph_level,
            "label_name": phenotype.label_name,
            "label_statistic_name": phenotype.label_statistic_name,
            "mrna_number_fraction": json.dumps(phenotype.mrna_number_fraction),
            "n_libraries": phenotype.n_libraries,
            "measurement_type": phenotype.measurement_type,
        }

    @data_chunker
    def _mrna_number_fraction_phenotype_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        phenotype = data["experiment"].phenotype
        phenotype_id = hashlib.sha256(
            json.dumps(phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherNode(
            node_id=phenotype_id,
            preferred_id=f"phenotype_{phenotype_id}",
            node_label="mrna number fraction phenotype",
            properties=self._mrna_number_fraction_properties(phenotype),
        )

    def _get_mrna_number_fraction_phenotype_reference_nodes(
        self,
    ) -> list[BioCypherNode]:
        nodes = []
        seen_node_ids: set[str] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            phenotype = data.reference.phenotype_reference
            phenotype_id = hashlib.sha256(
                json.dumps(phenotype.model_dump()).encode("utf-8")
            ).hexdigest()
            if phenotype_id in seen_node_ids:
                continue
            seen_node_ids.add(phenotype_id)
            nodes.append(
                BioCypherNode(
                    node_id=phenotype_id,
                    preferred_id="mrna number fraction phenotype",
                    node_label="mrna number fraction phenotype",
                    properties=self._mrna_number_fraction_properties(phenotype),
                )
            )
        return nodes

    def _get_dataset_nodes(self) -> list[BioCypherNode]:
        nodes = [
            BioCypherNode(
                node_id=self.dataset.name,
                preferred_id=self.dataset.name,
                node_label="dataset",
            )
        ]
        return nodes

    @data_chunker
    def _publication_node(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherNode:
        publication = data["publication"]
        publication_id = hashlib.sha256(
            json.dumps(publication.model_dump()).encode("utf-8")
        ).hexdigest()
        # A non-journal source (a dissertation, a preliminary-exam report, an in-house
        # measurement) has no PubMed id and no DOI, so the preferred id falls back to
        # the DOI and then to the deposited document's identifier (path + sha256). The
        # fallback order is pubmed -> doi -> identifier, and a source carrying none of
        # the three cannot exist: Publication's validator requires a doi/pmid for a
        # journal article and an identifier for every other source type.
        preferred = publication.pubmed_id or publication.doi or publication.identifier

        return BioCypherNode(
            node_id=publication_id,
            preferred_id=f"publication_{preferred}",
            node_label="publication",
            properties={
                "pubmed_id": publication.pubmed_id,
                "pubmed_url": publication.pubmed_url,
                "doi": publication.doi,
                "doi_url": publication.doi_url,
                "source_type": publication.source_type.value,
                "title": publication.title,
                "identifier": publication.identifier,
                "identifier_url": publication.identifier_url,
                "serialized_data": json.dumps(publication.model_dump()),
            },
        )

    # edges
    def _get_experiment_reference_to_dataset_edges(self) -> list[BioCypherEdge]:
        edges = []
        for data in self.dataset.experiment_reference_index:
            reference_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            edge = BioCypherEdge(
                source_id=reference_id,
                target_id=self.dataset.name,
                relationship_label="experiment reference member of",
            )
            edges.append(edge)
        return edges

    @data_chunker
    def _experiment_to_dataset_edge(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_id,
            target_id=self.dataset.name,
            relationship_label="experiment member of",
        )
        return edge  # type: ignore[no-any-return]  # BioCypherEdge is Any (biocypher untyped)

    @data_chunker
    def _experiment_reference_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        experiment_ref_id = hashlib.sha256(
            json.dumps(data["reference"].model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=experiment_ref_id,
            target_id=experiment_id,
            relationship_label="experiment reference of",
        )
        return edge

    @data_chunker
    def _genotype_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        genotype = data["experiment"].genotype
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=genotype_id,
            target_id=experiment_id,
            relationship_label="genotype member of",
        )
        return edge

    @data_chunker
    def _perturbation_to_genotype_edges(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        edges = []
        genotype = data["experiment"].genotype
        # genotype_id is invariant across this genotype's perturbations, so hash it
        # ONCE. Recomputing it inside the loop re-serializes the entire genotype
        # (all perturbations) every iteration -- O(P^2) per record, catastrophic for
        # natural isolates (~5k perturbations each: ~33 h for Caudal vs seconds).
        genotype_id = hashlib.sha256(
            json.dumps(genotype.model_dump()).encode("utf-8")
        ).hexdigest()
        for perturbation in genotype.perturbations:
            perturbation_id = hashlib.sha256(
                json.dumps(perturbation.model_dump()).encode("utf-8")
            ).hexdigest()
            edges.append(
                BioCypherEdge(
                    source_id=perturbation_id,
                    target_id=genotype_id,
                    relationship_label="perturbation member of",
                )
            )
        return edges

    @data_chunker
    def _environment_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        environment_id = self._environment_node_id(data["experiment"].environment)
        edge = BioCypherEdge(
            source_id=environment_id,
            target_id=experiment_id,
            relationship_label="environment member of",
        )
        return edge

    def _get_environment_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_environment_experiment_ref_pairs: set[tuple[str, str]] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            environment_id = self._environment_node_id(
                data.reference.environment_reference
            )
            env_experiment_ref_pair = (environment_id, experiment_ref_id)
            if env_experiment_ref_pair not in seen_environment_experiment_ref_pairs:
                seen_environment_experiment_ref_pairs.add(env_experiment_ref_pair)

                edge = BioCypherEdge(
                    source_id=environment_id,
                    target_id=experiment_ref_id,
                    relationship_label="environment member of",
                )
                edges.append(edge)
        return edges

    @data_chunker
    def _phenotype_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        phenotype_id = hashlib.sha256(
            json.dumps(data["experiment"].phenotype.model_dump()).encode("utf-8")
        ).hexdigest()
        edge = BioCypherEdge(
            source_id=phenotype_id,
            target_id=experiment_id,
            relationship_label="phenotype member of",
        )
        return edge

    @data_chunker
    def _media_to_environment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        environment_id = self._environment_node_id(data["experiment"].environment)
        media_id = self._media_node_id(data["experiment"].environment.media)
        edge = BioCypherEdge(
            source_id=media_id,
            target_id=environment_id,
            relationship_label="media member of",
        )
        return edge

    @data_chunker
    def _temperature_to_environment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        """Link temperature to environment, or nothing when temperature is gapped."""
        environment = data["experiment"].environment
        temperature = environment.temperature
        if temperature is None:
            return []
        environment_id = self._environment_node_id(environment)
        temperature_id = self._temperature_node_id(temperature)
        return [
            BioCypherEdge(
                source_id=temperature_id,
                target_id=environment_id,
                relationship_label="temperature member of",
            )
        ]

    @data_chunker
    def _environment_perturbation_to_environment_edges(
        self, data: dict[str, Any], method_name: str
    ) -> list[BioCypherEdge]:
        environment = data["experiment"].environment
        environment_id = self._environment_node_id(environment)
        return [
            BioCypherEdge(
                source_id=self._environment_perturbation_node_id(perturbation),
                target_id=environment_id,
                relationship_label="environment perturbation member of",
            )
            for perturbation in environment.perturbations
        ]

    def _get_environment_perturbation_to_environment_reference_edges(
        self,
    ) -> list[BioCypherEdge]:
        edges = []
        seen_pairs: set[tuple[str, str]] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            environment = data.reference.environment_reference
            environment_id = self._environment_node_id(environment)
            for perturbation in environment.perturbations:
                perturbation_id = self._environment_perturbation_node_id(perturbation)
                pair = (perturbation_id, environment_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    edges.append(
                        BioCypherEdge(
                            source_id=perturbation_id,
                            target_id=environment_id,
                            relationship_label="environment perturbation member of",
                        )
                    )
        return edges

    def _get_genome_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_genome_experiment_ref_pairs: set[tuple[str, str]] = set()
        for i, data in tqdm(enumerate(self.dataset.experiment_reference_index)):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            genome_id = hashlib.sha256(
                json.dumps(data.reference.genome_reference.model_dump()).encode("utf-8")
            ).hexdigest()
            genome_experiment_ref_pair = (genome_id, experiment_ref_id)
            if genome_experiment_ref_pair not in seen_genome_experiment_ref_pairs:
                seen_genome_experiment_ref_pairs.add(genome_experiment_ref_pair)
                edge = BioCypherEdge(
                    source_id=genome_id,
                    target_id=experiment_ref_id,
                    relationship_label="genome member of",
                )
                edges.append(edge)
        return edges

    def _get_phenotype_to_experiment_reference_edges(self) -> list[BioCypherEdge]:
        edges = []
        seen_phenotype_experiment_ref_pairs: set[tuple[str, str]] = set()
        for data in tqdm(self.dataset.experiment_reference_index):
            experiment_ref_id = hashlib.sha256(
                json.dumps(data.reference.model_dump()).encode("utf-8")
            ).hexdigest()
            phenotype_id = hashlib.sha256(
                json.dumps(data.reference.phenotype_reference.model_dump()).encode(
                    "utf-8"
                )
            ).hexdigest()
            phenotype_experiment_ref_pair = (phenotype_id, experiment_ref_id)
            if phenotype_experiment_ref_pair not in seen_phenotype_experiment_ref_pairs:
                seen_phenotype_experiment_ref_pairs.add(phenotype_experiment_ref_pair)
                edge = BioCypherEdge(
                    source_id=phenotype_id,
                    target_id=experiment_ref_id,
                    relationship_label="phenotype member of",
                )
                edges.append(edge)
        return edges

    @data_chunker
    def _publication_to_experiment_edge(
        self, data: dict[str, Any], method_name: str
    ) -> BioCypherEdge:
        experiment_id = hashlib.sha256(
            json.dumps(data["experiment"].model_dump()).encode("utf-8")
        ).hexdigest()
        publication_id = hashlib.sha256(
            json.dumps(data["publication"].model_dump()).encode("utf-8")
        ).hexdigest()
        return BioCypherEdge(
            source_id=publication_id,
            target_id=experiment_id,
            relationship_label="mentions",
        )


if __name__ == "__main__":
    pass
