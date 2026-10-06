"""Generate the API reference pages ``docs/source/modules/<package>.rst``.

One page per documented ``torchcell`` subpackage. Each page opens with the hand-written
description in ``DESC`` and then lists the package's public classes, functions and
constants in ``autosummary`` blocks, which Sphinx expands into one page per member under
``docs/source/generated/``. The member lists are computed by importing the package, so
run this with the full torchcell environment, from the repository root::

    PYTHONPATH=$PWD python docs/gen_api_pages.py            # rewrite the pages
    PYTHONPATH=$PWD python docs/gen_api_pages.py --check    # exit 1 if any page is stale

``MODE`` picks how a package's members are listed: from its ``__all__`` (the default),
module by module (every public class and function each submodule defines), or the
special dataset layout (embedding datasets plus every class in ``dataset_registry``).
Scratch, demo and deprecated submodules are left out, and so is any submodule whose
import fails; both are named in a "Not documented" section. Because import success
depends on the installed packages, ``--check`` is only meaningful in the same
environment the pages were generated in.
"""

import argparse
import importlib
import inspect
import os
import re
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "docs", "source", "modules")

DESC: dict[str, str] = {
    "adapters": (
        "BioCypher adapters that turn a built torchcell dataset into knowledge-graph "
        "nodes and edges. There is one adapter class per registered dataset (for "
        "example ``SmfCostanzo2016Adapter`` or ``BetaxanthinCachera2023Adapter``), and "
        "most inherit from :class:`~torchcell.adapters.CellAdapter`, which reads the "
        "dataset's LMDB records and yields ``BioCypherNode`` and ``BioCypherEdge`` "
        "objects through ``get_nodes`` and ``get_edges``. The knowledge-graph builders "
        "in :mod:`torchcell.knowledge_graphs` pair each dataset class with its adapter "
        "through ``dataset_adapter_map``."
    ),
    "data": (
        "The data layer between the knowledge graph and model training. "
        ":class:`~torchcell.data.ExperimentDataset` is the abstract LMDB-backed base "
        "for experiment datasets, :class:`~torchcell.data.Neo4jCellDataset` queries "
        "Neo4j, caches the raw records in LMDB and turns each experiment into a "
        "perturbed cell graph, and the graph processors (for example "
        ":class:`~torchcell.data.SubgraphRepresentation` and "
        ":class:`~torchcell.data.LazySubgraphRepresentation`) define how a genotype "
        "perturbation is applied to that graph. Deduplicators and aggregators merge "
        "repeated measurements of the same genotype, and the reference-index helpers "
        "group experiments by their shared reference state."
    ),
    "database": (
        "Commands that build and serve the torchcell Neo4j knowledge graph. The "
        "package exports :class:`~torchcell.database.build_command.BuildCommand`, a "
        "Cliff command "
        "that runs the database image build scripts; ``tcdb`` (the console script "
        "declared in ``pyproject.toml``) is the Cliff entry point in "
        "``torchcell.database.tcdb``. Other modules create the Neo4j directory tree, "
        "combine BioCypher output directories into one import set, build a single "
        "registered dataset's LMDB for admission, and hold the client connection "
        "settings and the Neo4j Browser stylesheet."
    ),
    "datamodels": (
        "The pydantic schema of a torchcell experiment. A record is a typed "
        "``genotype x environment -> phenotype`` experiment: ``Genotype`` holds a "
        "list of gene perturbations (for example ``SgaKanMxDeletionPerturbation``), "
        "``Environment`` holds the medium, an optional temperature and any "
        "environment perturbations, and each phenotype family (fitness, gene "
        "interaction, CalMorph morphology, microarray, RNA-seq and pseudobulk "
        "expression, metabolite, protein abundance, visual score, environment "
        "response and others) has its own phenotype, experiment and "
        "experiment-reference classes in ``schema``. The classes derive from the "
        "strict bases in ``pydant`` (``ModelStrict`` and ``ModelStrictArbitrary``). "
        "Dataset loaders emit these objects, and the converter modules map one "
        "experiment type onto another (for example gene essentiality onto fitness). "
        "The package namespace re-exports a subset (``torchcell.datamodels.__all__``); "
        "every class below is listed under the module that defines it."
    ),
    "datamodules": (
        "PyTorch Lightning data modules for training on cell datasets. "
        ":class:`~torchcell.datamodules.CellDataModule` builds train, validation and "
        "test splits of a cell dataset and caches the split indices, and the "
        "``perturbation_subset`` module's data module draws a "
        "size-limited subset of those splits. The split-index records "
        "(:class:`~torchcell.datamodules.DataModuleIndex` and related models) are "
        "pydantic objects, cached as JSON in the data module's cache directory."
    ),
    "datasets": (
        "Dataset loaders. The top level of ``torchcell.datasets`` holds the "
        "per-gene embedding datasets (sequence language-model embeddings, codon "
        "frequencies, one-hot and random baselines), which derive from the "
        "in-memory embedding base in ``torchcell.data.embedding``, and "
        "``NodeEmbeddingBuilder``, which builds them from a configuration. The "
        "experiment datasets live in ``torchcell.datasets.scerevisiae``: one module "
        "per source publication, each reading that publication's released data, "
        "converting its records into :mod:`torchcell.datamodels` experiments and "
        "storing them in LMDB. A class decorated with "
        ":func:`~torchcell.datasets.dataset_registry.register_dataset` is added to "
        "``dataset_registry``, the name-to-class map that ``knowledge_graphs.create_kg`` "
        "and ``knowledge_graphs.kg_manifest`` look datasets up in; "
        "the table below lists every registered class."
    ),
    "graph": (
        "Gene graphs for *S. cerevisiae*. "
        ":class:`~torchcell.graph.SCerevisiaeGraph` assembles the Gene Ontology DAG, "
        "SGD physical and genetic interaction networks, regulatory networks and "
        "STRING networks over the genes of a genome; "
        ":class:`~torchcell.graph.GeneGraph` and "
        ":class:`~torchcell.graph.GeneMultiGraph` wrap the resulting NetworkX graphs "
        "as named, typed containers that the cell datasets consume. The ``filter_*`` "
        "functions prune the GO DAG (by annotation date, by contained genes, by "
        "evidence code, and by redundant terms)."
    ),
    "knowledge_graphs": (
        "Builders for the BioCypher knowledge graph and the manifest that governs "
        "changes to the served graph. ``dataset_adapter_map`` pairs each dataset "
        "class with its adapter, and the ``create_*`` modules run a BioCypher build "
        "over a configured set of datasets. ``kg_manifest`` records, per served "
        "dataset, the schema fingerprints and adapter code the graph was built "
        "under, and decides whether a new dataset can be admitted by incremental "
        "import or requires a full rebuild; ``incremental_import`` turns one "
        "BioCypher output directory into a Neo4j incremental import; ``releases`` "
        "names and lists served releases. The package resolves its two listed "
        "submodules lazily, because each imports every dataset loader and adapter."
    ),
    "literature": (
        "The literature capture subsystem. It reads papers and supplementary files "
        "from Zotero (``zotero``), OCRs or extracts them into markdown (``ocr``, "
        "``extract``, ``scanned``), fetches supplementary data (``si_data``), and "
        "records a sha256 provenance manifest for every stored artifact "
        "(``manifest``, ``provenance``) in the on-disk mirror. ``server`` is the "
        "read-only HTTP endpoint (``tc-lit-server``) that serves the mirrored "
        "artifacts, and ``sync`` reconciles the mirror against a Zotero collection."
    ),
    "loader": (
        "Batch loaders for experiment datasets. "
        ":class:`~torchcell.loader.cpu_experiment_loader.CpuExperimentLoaderMultiprocessing` "
        "prefetches "
        "batches in worker processes on the CPU side; the ``dense_padding_data_loader`` "
        "module pads batches to a fixed node count for models that need dense "
        "inputs."
    ),
    "losses": (
        "Loss functions. The package exports "
        ":class:`~torchcell.losses.list_mle.ListMLELoss`, a listwise ranking loss, and "
        ":class:`~torchcell.losses.dcell.DCellLoss`, the root plus subsystem loss of "
        "the DCell model. The remaining modules (``multi_dim_nan_tolerant``, "
        "``distributional``, ``mle_wasserstein``, ``point_dist_graph_reg`` and "
        "others) are imported by module path from the experiment training scripts "
        "and are not re-exported. Every class below is listed under the module "
        "that defines it."
    ),
    "metabolism": (
        "Genome-scale metabolic modeling. ``yeast_GEM`` wraps the yeast-GEM model "
        "(through cobra) and derives its reaction and metabolite graphs; "
        "``constraints`` turns a genome-scale model into constraint tensors; "
        "``flux_layer`` is a differentiable, enzyme-constrained flux layer; ``media`` "
        "maps an ontology ``Media`` object onto exchange bounds; ``pathway`` adds a "
        "heterologous pathway to a model as a typed perturbation; and "
        "``enzyme_kinetics`` and ``parameters`` hold kinetic parameters with their "
        "provenance. Nothing is re-exported at the package level, because "
        "``yeast_GEM`` imports cobra and downloads the model on first use; import "
        "the submodule you need."
    ),
    "models": (
        "Model implementations, one module per architecture. They range from "
        "sequence language-model wrappers (Nucleotide Transformer, ESM-2, ProtT5, "
        "the fungal up/down-stream transformer) through set and graph models "
        "(DeepSet, GCN and GAT encoders, DiffPool and SAGPool variants) to the cell "
        "models used in the experiments: :class:`~torchcell.models.dcell.DCell`, "
        "the DANGO family (for example "
        ":class:`~torchcell.models.hetero_cell_bipartite_dango_gi.GeneInteractionDango` "
        "in ``hetero_cell_bipartite_dango_gi``), and the Cell Graph Transformer "
        "(:class:`~torchcell.models.cell_graph_transformer.CellGraphTransformer`, with "
        "its equivariant variant "
        ":class:`~torchcell.models.equivariant_cell_graph_transformer.CellGraphTransformer`). "
        "The package namespace re-exports a "
        "small subset (``torchcell.models.__all__``); every class below is listed "
        "under the module that defines it."
    ),
    "nn": (
        "Neural-network layers shared by the models. ``hetero_nsa``, ``nsa_encoder``, "
        "``masked_attention_block`` and ``self_attention_block`` build node-set "
        "attention over graphs with FlexAttention adjacency masks; "
        "``masked_gin_conv`` is a GIN convolution for the lazy, masked subgraph "
        "representation; and ``stoichiometric_hypergraph_conv`` is a "
        "stoichiometry-aware hypergraph convolution over metabolic networks."
    ),
    "ontology": (
        "Helpers for inspecting the torchcell BioCypher ontology. The "
        "``print_*`` functions in ``tc_ontology`` print the ontology structure, a "
        "summary and the schema mappings; the ``mermaid_diagram`` module (not "
        "re-exported) renders the BioCypher schema as a Mermaid diagram."
    ),
    "paper": (
        "Code behind the manuscript's generated artifacts. ``tables`` holds "
        "primitives that emit one table as both markdown and LaTeX; "
        "``ontology_graph`` introspects the pydantic schema into a typed graph and "
        "``ontology_svg`` lays it out as the zoomable SVG published at "
        "`/ontology/ <https://mjvolk3.github.io/torchcell/ontology/>`_; ``signal`` "
        "is a command-line tool that computes a built dataset's gzip signal and its "
        "derived shape and graph role."
    ),
    "profiling": (
        "A timing decorator for profiling. :func:`~torchcell.profiling.time_method` "
        "records the wall time of each call of the decorated function in a "
        "module-level store when the environment variable ``TORCHCELL_DEBUG_TIMING`` "
        "is ``1`` (otherwise it calls the function directly), and "
        ":func:`~torchcell.profiling.get_timings`, "
        ":func:`~torchcell.profiling.get_timing_summary` and "
        ":func:`~torchcell.profiling.print_timing_summary` read it back."
    ),
    "provenance": (
        "Schema-dependency tracking, which decides when a built dataset must be "
        "rebuilt because the schema changed. ``schema_impact`` diffs the working-tree "
        "schema against a git ref, classifies each change as breaking or stale, and "
        "names the loaders that must rebuild (it runs as a pre-commit hook). "
        "``build_manifest`` stores, next to each built LMDB, the contract "
        "fingerprints of the schema symbols its loader depends on, and its ``check`` "
        "reports which built datasets are stale. A fingerprint hashes a symbol's "
        "contract (field shape and validators), not its source text, so a docstring "
        "edit flags nothing. ``schema_deps`` is the shared AST analysis."
    ),
    "scheduler": (
        "Learning-rate schedulers. "
        ":class:`~torchcell.scheduler.cosine_annealing_warmup.CosineAnnealingWarmupRestarts` "
        "is cosine annealing with a linear warmup and cyclic restarts."
    ),
    "sequence": (
        "Genome and sequence data structures. :class:`~torchcell.sequence.Genome` is "
        "the abstract genome interface (lazy gene-set and sequence access), "
        ":class:`~torchcell.sequence.Gene` a gene within it, and "
        ":class:`~torchcell.sequence.DnaWindowResult` and "
        ":class:`~torchcell.sequence.DnaSelectionResult` the sequence windows the "
        "embedding datasets consume. The *S. cerevisiae* S288C implementation lives in "
        "``torchcell.sequence.genome.scerevisiae``, and ``torchcell.sequence.genome."
        "registry`` resolves the sha256-pinned reference genome files."
    ),
    "sga": (
        "A colony-fitness pipeline adapted from SGAtools for single CRISPR knockouts "
        "dispensed onto plates by an ECHO acoustic liquid handler. It reads the "
        "gitter colony-size file and the ECHO picklist (``io``), corrects "
        "positional plate artifacts (``normalize``), and scores each knockout as "
        "fitness relative to the on-plate BY4741 wild type (``score``). ``image`` "
        "and ``cellpose_seg`` measure colony sizes from plate photographs, and "
        "``viz`` draws plate heatmaps and histograms. The typical call sequence is "
        ":func:`~torchcell.sga.read_gitter_dat`, "
        ":func:`~torchcell.sga.read_echo_picklist`, "
        ":func:`~torchcell.sga.merge_layout`, "
        ":func:`~torchcell.sga.normalize_plate`, then "
        ":func:`~torchcell.sga.score_plate`."
    ),
    "trainers": (
        "PyTorch Lightning training tasks. The package exports the regression task "
        "(:class:`~torchcell.trainers.neo_regression.RegressionTask`), a simple linear "
        "baseline, and two DCell tasks. The ``fit_int_*`` and ``int_*`` modules hold "
        "the tasks used by individual experiments and are imported by module path "
        "from the experiment scripts. Every class below is listed under the module "
        "that defines it."
    ),
    "transforms": (
        "PyG transforms applied to cell graphs. ``regression_to_classification`` "
        "and its COO variants normalize regression labels, bin them into "
        "classification targets, and invert the binning; ``hetero_to_dense_mask`` "
        "converts the sparse adjacencies of a ``HeteroData`` graph into boolean masks."
    ),
    "utils": (
        "Shared helpers. ``paths`` resolves output directories relative to the "
        "current checkout (so a script run from a git worktree writes into that "
        "worktree); ``utils`` holds the repository-wide figure standards (the "
        "ordered plot palette, Nature panel widths in millimeters, "
        ":func:`~torchcell.utils.savefig_true_size_svg` and "
        ":func:`~torchcell.utils.apply_paper_style`); and ``file_lock`` provides "
        ":class:`~torchcell.utils.FileLockHelper` for locked JSON reads and writes."
    ),
    "verification": (
        "Record-level verification of built datasets at five levels, L0 to L4. "
        "``levels`` holds the reusable checks and ``common`` the rules every "
        "dataset family shares; each family module (``fitness``, ``expression``, "
        "``morphology``, ``metabolite``, ``protein`` and others) applies them to one "
        "phenotype type. ``report`` defines the pydantic verification report, "
        "``sourced`` binds a single extracted value to its source quote and sha256, "
        "and ``runners`` runs the verifiers over built datasets."
    ),
    "viz": (
        "Plotting helpers used during training and analysis: predicted-versus-"
        "measured fitness and genetic-interaction plots, dataset split "
        "visualizations, graph-regularization and edge-recovery plots, "
        "transformer diagnostics, oversmoothing and oversquashing measures, and "
        "regression diagnostics logged to Weights & Biases."
    ),
}

# How the member list is built.
#   all       -> the package's __all__
#   modules   -> every public class/function defined in each submodule
#   datasets  -> embedding datasets + dataset_registry
MODE = {
    "database": "modules",
    "datamodels": "modules",
    "knowledge_graphs": "modules",
    "loader": "modules",
    "losses": "modules",
    "metabolism": "modules",
    "models": "modules",
    "nn": "modules",
    "paper": "modules",
    "scheduler": "modules",
    "trainers": "modules",
    "transforms": "modules",
    "viz": "modules",
    "datasets": "datasets",
}

# Submodules left out of "modules" mode: scratch sketches, demos and deprecated code.
SKIP_MODULE = re.compile(r"(DEPRECATED|deprecated|scratch|tutorial)")
# Script entry points (`main`, `main_incidence`, ...) are not API.
SKIP_FUNC = re.compile(r"^main(_|$)")

HEADS = {"cls": "Classes", "fn": "Functions", "data": "Constants"}


def submodules(pkg: str) -> list[str]:
    """Dotted names of every module and subpackage below ``pkg``, sorted."""
    root = os.path.join(REPO_ROOT, *pkg.split("."))
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(
            d
            for d in dns
            if d != "__pycache__" and os.path.exists(os.path.join(dp, d, "__init__.py"))
        )
        base = os.path.relpath(dp, REPO_ROOT).replace(os.sep, ".")
        for fn in sorted(fns):
            if fn.endswith(".py") and fn != "__init__.py":
                out.append(base + "." + fn[:-3])
        if dp != root:
            out.append(base)
    return sorted(set(out))


def classify(obj: object) -> str:
    """Return ``cls``, ``fn``, ``mod`` or ``data`` for an exported object."""
    if inspect.isclass(obj):
        return "cls"
    if inspect.isfunction(obj):
        return "fn"
    if isinstance(obj, types.ModuleType):
        return "mod"
    return "data"


def block(names: list[str], kind: str) -> str:
    """One ``autosummary`` directive over ``names``."""
    if not names:
        return ""
    lines = [".. autosummary::", "   :nosignatures:", "   :toctree: ../generated"]
    if kind == "cls":
        lines.append("   :template: autosummary/class.rst")
    lines.append("")
    lines += [f"   {n}" for n in names]
    return "\n".join(lines) + "\n\n"


def members_block(entries: dict[str, list[str]], level: str) -> str:
    """Classes, functions and constants, each under its own heading."""
    s = ""
    for kind in ("cls", "fn", "data"):
        if entries.get(kind):
            h = HEADS[kind]
            s += f"{h}\n{level * len(h)}\n\n" + block(entries[kind], kind)
    return s


def module_sections(full: str, skipped: list[str]) -> tuple[str, int]:
    """One section per importable, non-skipped submodule of ``full``."""
    s = ""
    count = 0
    for m in submodules(full):
        rel = m[len(full) + 1 :]
        if m.endswith(".conf"):
            skipped.append(f"{rel} (configuration files only)")
            continue
        if SKIP_MODULE.search(m):
            skipped.append(f"{rel} (scratch, demo or deprecated)")
            continue
        try:
            sm = importlib.import_module(m)
        except BaseException as e:  # recorded on the page; a failing module is skipped
            skipped.append(f"{rel} (import fails: {type(e).__name__})")
            continue
        entries: dict[str, list[str]] = {"cls": [], "fn": [], "data": []}
        for k, v in vars(sm).items():
            if k.startswith("_") or getattr(v, "__module__", None) != m:
                continue
            c = classify(v)
            if c == "cls":
                entries["cls"].append(f"{rel}.{k}")
            elif c == "fn" and not SKIP_FUNC.match(k):
                entries["fn"].append(f"{rel}.{k}")
        n = len(entries["cls"]) + len(entries["fn"])
        if n == 0:
            continue
        count += n
        head = f"``{rel}``"
        s += f"{head}\n{'-' * len(head)}\n\n"
        doc = (inspect.getdoc(sm) or "").strip().split("\n\n")[0].replace("\n", " ")
        if doc:
            s += doc + "\n\n"
        s += members_block(entries, "~")
    return s, count


def dataset_sections(mod: types.ModuleType, skipped: list[str]) -> tuple[str, int]:
    """Embedding datasets, the registry, and every registered experiment dataset."""
    emb = list(mod.embedding_datasets)
    s = "Embedding datasets\n------------------\n\n" + block(emb, "cls")
    s += (
        "Dataset registry\n----------------\n\n"
        ".. currentmodule:: torchcell.datasets.dataset_registry\n\n"
        + block(["register_dataset"], "fn")
        + block(["dataset_registry"], "data")
    )
    from torchcell.datasets.dataset_registry import dataset_registry

    for m in submodules("torchcell.datasets.scerevisiae"):
        if SKIP_MODULE.search(m):
            continue
        try:
            importlib.import_module(m)
        except BaseException as e:  # recorded on the page; a failing module is skipped
            name = m.rsplit(".", 1)[1]
            skipped.append(f"scerevisiae.{name} (import fails: {type(e).__name__})")
    reg = sorted((v.__module__, k) for k, v in dataset_registry.items())
    prefix = "torchcell.datasets.scerevisiae."
    names = [f"{m[len(prefix) :]}.{k}" for m, k in reg]
    s += (
        "Registered *S. cerevisiae* experiment datasets\n"
        "----------------------------------------------\n\n"
        f"The {len(names)} classes in ``dataset_registry``, grouped by source "
        "module (one module per publication).\n\n"
        ".. currentmodule:: torchcell.datasets.scerevisiae\n\n" + block(names, "cls")
    )
    return s, len(emb) + 2 + len(names)


def page(pkg: str) -> tuple[str, int]:
    """Render ``modules/<pkg>.rst``; return the text and the member count."""
    full = f"torchcell.{pkg}"
    mod = importlib.import_module(full)
    s = f"{full}\n{'=' * len(full)}\n\n.. module:: {full}\n\n"
    s += f".. currentmodule:: {full}\n\n{DESC[pkg]}\n\n"
    s += ".. contents:: Contents\n    :local:\n\n"
    mode = MODE.get(pkg, "all")
    skipped: list[str] = []
    if mode == "all":
        entries: dict[str, list[str]] = {"cls": [], "fn": [], "data": []}
        for name in mod.__all__:
            k = classify(getattr(mod, name))
            if k == "mod":
                skipped.append(name)
                continue
            entries[k].append(name)
        count = sum(len(v) for v in entries.values())
        s += members_block(entries, "-")
    elif mode == "modules":
        body, count = module_sections(full, skipped)
        s += body
    else:
        body, count = dataset_sections(mod, skipped)
        s += body
    if skipped:
        s += "Not documented\n--------------\n\n"
        if mode == "all":
            s += "Names in ``__all__`` that are submodules, not objects: "
            s += ", ".join(f"``{x}``" for x in skipped) + ".\n\n"
        else:
            s += "Submodules left out of this page:\n\n"
            for x in skipped:
                name, reason = x.split(" ", 1)
                s += f"- ``{name}`` {reason}\n"
            s += "\n"
    return s.rstrip() + "\n", count


def main() -> int:
    """Write every page, or with ``--check`` report the pages that would change."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if any page differs from the regenerated text",
    )
    parser.add_argument("packages", nargs="*", help="subset of packages (default: all)")
    args = parser.parse_args()
    stale = []
    for pkg in args.packages or sorted(DESC):
        text, count = page(pkg)
        path = os.path.join(OUT_DIR, f"{pkg}.rst")
        current = open(path).read() if os.path.exists(path) else None
        if args.check:
            if current != text:
                stale.append(path)
            continue
        if current != text:
            with open(path, "w") as f:
                f.write(text)
        print(f"{pkg}: {count} members")
    if stale:
        print("stale API pages (run docs/gen_api_pages.py):", *stale, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
