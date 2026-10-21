# experiments/database/scripts/build_ecoli_genslm_benchmark_table.py
# [[experiments.database.scripts.build_ecoli_genslm_benchmark_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_ecoli_genslm_benchmark_table
r"""The E. coli datasets table for notes-tex, and the short list of them that can
benchmark a gene-sequence encoding (GenSLM and its siblings).

Reads the newest E. coli pre-build snapshot written by
``build_bacteria_datasets_table.py --organisms ecoli`` (the per-store probe: instances,
genotypes, environments, Signal, the sequence-only fit tier) and writes, under
``notes-tex/database/ecoli-datasets/tables/``:

- ``datasets.tex``: every served E. coli dataset, one row each, grouped by phenotype
  family, with the sequence-only fit tier and the label unit of each.
- ``tiers.tex``: datasets and instances per sequence-only fit tier.
- ``shortlist.tex``: the benchmark short list (``PICKS`` below), with the columns
  that are MEASURED from each store: distinct genes carrying a label, the number of
  gene-attached labels, the identifier namespace the store keys genes by, and the
  distinct environments.
- ``numbers.tex``: ``\newcommand`` macros for every number the document's prose
  quotes, so a figure in a sentence is as generated as a cell in a table.

It also writes ``experiments/database/results/ecoli_genslm_benchmark_shortlist.json``
(the measured short list, with the LMDB fingerprint each measurement was read from)
and the generated block of ``notes/experiments.database.ecoli-datasets.md``.

What is hand-authored here is the pick list itself: which datasets, in what order, the
task each poses and the reason. Everything numeric is read from a built store.

The short list is for a GENE-level encoding: one vector per coding sequence, as
``torchcell.datasets.genslm.GenSLMDataset`` builds it. A dataset qualifies when a
label attaches to one gene (or one gene pair) and enough genes carry a label for a
held-out-gene split. A label that sits on a 20-nt guide, on a whole engineered strain,
or on a strain with no edit has no single CDS to attach to and is not a primary pick.

Run from the repo root (``$DATA_ROOT`` from .env):
  python experiments/database/scripts/build_bacteria_datasets_table.py --organisms ecoli
  python experiments/database/scripts/build_ecoli_genslm_benchmark_table.py
  python experiments/database/scripts/build_ecoli_genslm_benchmark_table.py --reuse-measurements
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import os.path as osp
import pickle
import re
import sys
from datetime import date
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Literal

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel, Field

SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(SCRIPT.parent))

from build_bacteria_datasets_table import DatasetProbeRecord  # noqa: E402
from render_bacteria_datasets_table import (  # noqa: E402
    SEQ_FIT_LABEL,
    SEQ_TIER_MEANING,
    breakable_phenotype,
    short_name,
)

from torchcell.data.experiment_dataset import resolve_interned  # noqa: E402
from torchcell.paper.tables import (  # noqa: E402
    Cell,
    Column,
    PaperTable,
    Row,
    lmdb_fingerprint,
    read_frontmatter,
    scientific,
    tex_escape,
)

REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database" / "ecoli-datasets" / "tables"
NOTE = REPO / "notes" / "experiments.database.ecoli-datasets.md"
MEASUREMENTS = RESULTS / "ecoli_genslm_benchmark_shortlist.json"
BITS_PER_BYTE = 8
NOTE_START = (
    "<!-- GENERATED: shortlist start (build_ecoli_genslm_benchmark_table.py) -->"
)
NOTE_END = "<!-- GENERATED: shortlist end -->"


# --- the short list --------------------------------------------------------------


class BenchmarkPick(BaseModel):
    """One dataset on the short list and the benchmark task it poses."""

    dataset_class: str
    tier: Literal["core", "second"]
    task: str = Field(description="The prediction problem, in a few words.")
    label: str = Field(description="What one label is, and what it attaches to.")
    split: str = Field(description="The held-out unit a benchmark score is read on.")
    alternates: list[str] = Field(
        description="Served E. coli datasets that pose the same task."
    )
    why: str = Field(description="Why this row, in one or two sentences.")


PICKS: list[BenchmarkPick] = [
    BenchmarkPick(
        dataset_class="GeneEssentialityGoodall2018Dataset",
        tier="core",
        task="binary classification",
        label="essential or not, one TraDIS call per gene and medium",
        split="held-out genes",
        alternates=[
            "GeneEssentialityBaba2006Dataset",
            "GeneEssentialityGerdes2003Dataset",
            "GeneEssentialityPrice2018EcoliDataset",
        ],
        why=(
            "The simplest gene-level label in the host, and the one with three "
            "independent calls to agree with (Keio construction, genetic footprinting, "
            "TnSeq). A classifier on one vector per gene is the whole model."
        ),
    ),
    BenchmarkPick(
        dataset_class="EnvChemgenNichols2011Dataset",
        tier="core",
        task="regression, gene x condition",
        label="a colony-size S-score per (deletion, condition)",
        split="held-out genes, every condition",
        alternates=[
            "RbTnseqPrice2018EcoliDataset",
            "EnvChemgenShiver2016Dataset",
            "CarbonSourceTong2020Dataset",
            "EnvChemgenGirgis2009Dataset",
        ],
        why=(
            "The widest condition panel on one gene set. A held-out gene's whole "
            "profile is the target, and the condition enters as a second input the "
            "gene encoding does not have to carry."
        ),
    ),
    BenchmarkPick(
        dataset_class="ProteomeSchmidt2016Dataset",
        tier="core",
        task="regression, protein x condition",
        label="absolute copies per cell of one protein in one growth condition",
        split="held-out genes, every condition",
        alternates=[
            "ProteomeMori2021Dataset",
            "ProteomeCaglar2017Dataset",
            "ProteomeIshii2007Dataset",
        ],
        why=(
            "A molecular level of the gene's own product, unperturbed, across "
            "conditions. Expected (untested) to be where a codon model separates from "
            "a one-hot control, because the input is the coding sequence whose product "
            "is measured."
        ),
    ),
    BenchmarkPick(
        dataset_class="GeneInteractionBabu2014Dataset",
        tier="core",
        task="regression, gene pair",
        label="a signed colony-size S score per (query, array) deletion pair",
        split="held-out genes (both pair members unseen), and one member unseen",
        alternates=[
            "GeneInteractionButland2008Dataset",
            "CrispriCrossRachwalski2024Dataset",
        ],
        why=(
            "The only pairwise label at scale in the host. It asks whether two gene "
            "vectors compose, which no single-gene task does."
        ),
    ),
    BenchmarkPick(
        dataset_class="ProteinSynthesisRateLi2014Dataset",
        tier="second",
        task="regression, gene x medium",
        label="absolute protein synthesis rate per gene from ribosome profiling",
        split="held-out genes",
        alternates=["ProteinTurnoverGupta2024Dataset"],
        why=(
            "The phenotype closest to what a codon model reads: translation of the "
            "coding sequence itself, in three media."
        ),
    ),
    BenchmarkPick(
        dataset_class="RnaseqLamoureux2023Dataset",
        tier="second",
        task="regression, gene x library",
        label="TPM of one gene in one RNA-seq library",
        split="held-out genes, every library",
        alternates=["RnaseqPublicK12Lamoureux2023Dataset", "RnaseqCaglar2017Dataset"],
        why=(
            "The transcript-level counterpart of the proteome pick, over a thousand "
            "libraries, so the per-gene target is a long expression profile."
        ),
    ),
    BenchmarkPick(
        dataset_class="MorphologyCampos2018Dataset",
        tier="second",
        task="multi-output regression",
        label="a vector of cell-shape and cell-cycle features per deletion",
        split="held-out genes",
        alternates=["GrowthRateCampos2018Dataset"],
        why=(
            "The one imaging phenotype in the host: a different label family from "
            "growth and abundance, on the same Keio gene set."
        ),
    ),
    BenchmarkPick(
        dataset_class="CrispriGeneFitnessWang2018Dataset",
        tier="second",
        task="regression, gene x screen",
        label="gene-level CRISPRi fitness per (gene, screen)",
        split="held-out genes",
        alternates=[
            "CrispriKnockdownCui2018Dataset",
            "CrispriChemgenChoe2025Dataset",
            "MismatchCrispriFitnessHawkins2020Dataset",
        ],
        why=(
            "The knockdown counterpart of the deletion screens, already aggregated to "
            "the gene by its authors, so it covers essential genes the deletion "
            "collections cannot."
        ),
    ),
]


# --- measurement -----------------------------------------------------------------


_KEY_CLASSES: list[tuple[str, re.Pattern[str]]] = [
    ("b-number", re.compile(r"^b\d{4}$")),
    ("BW25113 locus tag", re.compile(r"^BW25113_\d+$")),
    ("ECK accession", re.compile(r"^ECK\d{4}$")),
]


def classify_key(key: str) -> str:
    for name, pat in _KEY_CLASSES:
        if pat.match(key):
            return name
    return "gene symbol"


def _is_number(v: Any) -> bool:
    return (
        isinstance(v, (int, float))
        and not isinstance(v, bool)
        and not (isinstance(v, float) and math.isnan(v))
    )


class PickMeasurement(BaseModel):
    """What one shortlisted store holds, read from every record of its LMDB."""

    dataset_class: str
    slug: str
    lmdb_fingerprint: str
    instances: int
    graph_role: str
    label_name: str
    labeled_genes: int = Field(description="Distinct gene ids that carry a label.")
    labels: int = Field(description="Gene-attached numeric or boolean labels.")
    identifier: str = Field(
        description="How the store names its genes: namespace, and kind if released."
    )
    identifier_share: float = Field(
        description="Fraction of labeled gene ids in the dominant identifier class."
    )
    environments: int = Field(
        description="Distinct resolved environment payloads, the probe's definition."
    )


def _env_key(env: dict[str, Any]) -> str:
    """A digest of the whole resolved environment payload, so two conditions that
    share a medium but differ in a dose or a stress count as two environments.
    """
    return hashlib.sha256(
        json.dumps(env, sort_keys=True, default=str).encode()
    ).hexdigest()


_NAMESPACE_SHORT = {
    "ecoli_k12_mg1655_bnumber": "MG1655 b-number",
    "ecoli_k12_bw25113_locus_tag": "BW25113 locus tag",
    "ecoli_k12_w3110_locus_tag": "W3110 locus tag",
    "ecoli_b_rel606_locus_tag": "REL606 locus tag",
}


def identifier_text(namespaces: set[str], released_kinds: set[str]) -> str:
    """``MG1655 b-number, released as ECK accession``: the perturbation namespaces,
    then the released identifier kinds (each cut at its first comma, where the
    loaders start their explanation).
    """
    ns = ", ".join(sorted(_NAMESPACE_SHORT.get(n, n) for n in namespaces))
    if not released_kinds:
        return ns
    kinds = "; ".join(sorted(k.split(",")[0].strip() for k in released_kinds))
    return f"{ns}, released as {kinds}"


def measure_store(args: tuple[str, str, str]) -> PickMeasurement:
    """Scan every record of a store and count its gene-attached labels."""
    dataset_class, slug, lmdb_dir = args
    processed = osp.dirname(lmdb_dir)
    interned: dict[str, Any] = {}
    idir = osp.join(processed, "interned")
    if osp.isdir(idir):
        ienv = lmdb.open(idir, readonly=True, lock=False)
        try:
            with ienv.begin() as itxn:
                for key, value in itxn.cursor():
                    interned[key.decode()] = pickle.loads(value)  # noqa: S301
        finally:
            ienv.close()
    genes: set[str] = set()
    key_classes: dict[str, int] = {}
    namespaces: set[str] = set()
    released_kinds: set[str] = set()
    labels = 0
    envs: set[str] = set()
    role = ""
    label_name = ""
    env = lmdb.open(lmdb_dir, readonly=True, lock=False)
    try:
        n = env.stat()["entries"]
        with env.begin() as txn:
            for _, raw in txn.cursor():
                rec = pickle.loads(raw)  # noqa: S301
                exp = resolve_interned(rec["experiment"], interned)
                envs.add(_env_key(exp["environment"]))
                ph = exp["phenotype"]
                role = ph["graph_level"]
                label_name = ph["label_name"]
                value = ph.get(label_name)
                if role == "node" and isinstance(value, dict):
                    # A readout over genes: the dict is keyed by gene id.
                    for gene, v in value.items():
                        if _is_number(v):
                            labels += 1
                            if gene not in genes:
                                genes.add(gene)
                                c = classify_key(gene)
                                key_classes[c] = key_classes.get(c, 0) + 1
                    continue
                # A strain-level or pair label: the genes are the perturbed ones, and
                # a dict-valued label (a feature vector) counts one label per number.
                if isinstance(value, dict):
                    labels += sum(1 for v in value.values() if _is_number(v))
                elif _is_number(value) or isinstance(value, bool):
                    labels += 1
                for p in exp["genotype"].get("perturbations", []):
                    gene = p["systematic_gene_name"]
                    if gene not in genes:
                        genes.add(gene)
                        namespaces.add(p.get("gene_namespace") or "unknown namespace")
                        if p.get("perturbation_type") == "released_identifier":
                            released_kinds.add(str(p.get("identifier_kind")))
                        c = classify_key(gene)
                        key_classes[c] = key_classes.get(c, 0) + 1
    finally:
        env.close()
    if not genes:
        raise ValueError(f"{slug}: no gene carries a label")
    dominant, count = max(key_classes.items(), key=lambda kv: kv[1])
    identifier = (
        identifier_text(namespaces, released_kinds)
        if namespaces
        else f"{dominant} (readout key)"
    )
    return PickMeasurement(
        dataset_class=dataset_class,
        slug=slug,
        lmdb_fingerprint=lmdb_fingerprint(lmdb_dir),
        instances=n,
        graph_role=role,
        label_name=label_name,
        labeled_genes=len(genes),
        labels=labels,
        identifier=identifier,
        identifier_share=count / len(genes),
        environments=len(envs),
    )


class ShortlistSnapshot(BaseModel):
    """The pick list and its measurements, as written to the results JSON."""

    date: str
    source_json: str
    picks: list[BenchmarkPick]
    measurements: list[PickMeasurement]


# --- rendering -------------------------------------------------------------------


def newest_ecoli_json() -> Path:
    cands = sorted(RESULTS.glob("pre-build/*/ecoli_datasets.json"))
    if not cands:
        raise FileNotFoundError(
            "no E. coli snapshot; run build_bacteria_datasets_table.py --organisms ecoli"
        )
    return cands[-1]


def landscape(tex: str) -> str:
    return "\\begin{landscape}\n" + tex + "\\end{landscape}\n"


def source_comment(data_path: Path) -> str:
    return (
        "%% GENERATED FILE -- do not hand-edit.\n"
        "%% SOURCE: experiments/database/scripts/build_ecoli_genslm_benchmark_table.py\n"
        f"%% off {data_path.relative_to(REPO)}\n"
    )


DATASET_COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Genotypes", align="r"),
    Column(header="Env", align="r"),
    Column(header="Instances", align="r"),
    Column(header="Phenotype", align="l", width="34mm"),
    Column(header="Shape", align="l"),
    Column(header="Role", align="l"),
    Column(header="Signal (bits)", align="r"),
    Column(header="Seq. fit", align="l"),
    Column(header="Label unit", align="l", width="40mm"),
]


def datasets_table(
    records: list[DatasetProbeRecord], sections: list[str]
) -> PaperTable:
    rows = [
        Row(
            section=section,
            cells={
                "Dataset": short_name(r),
                "Genotypes": f"{r.genotypes:,}" if r.probed else "\u2014",
                "Env": f"{r.environments:,}" if r.probed else "\u2014",
                "Instances": f"{r.instances:,}",
                "Phenotype": breakable_phenotype(r),
                "Shape": r.shape,
                "Role": r.graph_role,
                "Signal (bits)": scientific(r.signal_bytes * BITS_PER_BYTE)
                if r.signal_bytes
                else "pending",
                "Seq. fit": SEQ_FIT_LABEL[r.seq_fit],
                "Label unit": r.seq_unit,
            },
        )
        for section in sections
        for r in records
        if r.section == section
    ]
    probed = [r for r in records if r.probed]
    footer = Row(
        bold=True,
        cells={
            "Dataset": f"Total ({len(records)} datasets)",
            "Instances": f"{sum(r.instances for r in records):,}",
            "Signal (bits)": scientific(
                sum(r.signal_bytes for r in probed) * BITS_PER_BYTE
            ),
        },
    )
    return PaperTable(columns=DATASET_COLUMNS, rows=rows, footer=footer)


TIER_COLUMNS = [
    Column(header="Sequence-only fit", align="l"),
    Column(header="Datasets", align="r"),
    Column(header="Instances", align="r"),
    Column(header="What the tier means", align="l", width="88mm"),
]


def tiers_table(records: list[DatasetProbeRecord]) -> PaperTable:
    rows = [
        Row(
            cells={
                "Sequence-only fit": SEQ_FIT_LABEL[tier],
                "Datasets": f"{len(group):,}",
                "Instances": f"{sum(r.instances for r in group):,}",
                "What the tier means": SEQ_TIER_MEANING[tier],
            }
        )
        for tier in ("best", "good", "weak", "none")
        if (group := [r for r in records if r.seq_fit == tier])
    ]
    footer = Row(
        bold=True,
        cells={
            "Sequence-only fit": "Total",
            "Datasets": f"{len(records):,}",
            "Instances": f"{sum(r.instances for r in records):,}",
        },
    )
    return PaperTable(columns=TIER_COLUMNS, rows=rows, footer=footer)


# Landscape A4 with the house margins gives a 269 mm text block; the wrapping widths
# plus the four numeric columns and the column separation fit inside it.
SHORTLIST_COLUMNS = [
    Column(header="#", align="r"),
    Column(header="Dataset", align="l", width="25mm"),
    Column(header="Task", align="l", width="20mm"),
    Column(header="Label", align="l", width="29mm"),
    Column(header="Genes", align="r"),
    Column(header="Labels", align="r"),
    Column(header="Env", align="r"),
    Column(header="Gene identifier", align="l", width="25mm"),
    Column(header="Split", align="l", width="23mm"),
    Column(header="Same task", align="l", width="29mm"),
    Column(header="Why", align="l", width="46mm"),
]


def breakable_names(names: list[str]) -> Cell:
    """Class names that may wrap after a capital, with no space in the markdown."""
    md = ", ".join(short_name_of(n) for n in names)
    tex = ", ".join(
        re.sub(
            r"(?<=[a-z0-9])(?=[A-Z])", r"\\allowbreak{}", tex_escape(short_name_of(n))
        )
        for n in names
    )
    return Cell(md=md, tex=tex)


def short_name_of(dataset_class: str) -> str:
    return dataset_class.removesuffix("Dataset")


def shortlist_table(
    picks: list[BenchmarkPick], measured: dict[str, PickMeasurement]
) -> PaperTable:
    rows = []
    for i, p in enumerate(picks, 1):
        m = measured[p.dataset_class]
        rows.append(
            Row(
                section="Core" if p.tier == "core" else "Second tier",
                cells={
                    "#": str(i),
                    "Dataset": breakable_names([p.dataset_class]),
                    "Task": p.task,
                    "Label": p.label,
                    "Genes": f"{m.labeled_genes:,}",
                    "Labels": f"{m.labels:,}",
                    "Env": f"{m.environments:,}",
                    "Gene identifier": m.identifier,
                    "Split": p.split,
                    "Same task": breakable_names(p.alternates),
                    "Why": p.why,
                },
            )
        )
    return PaperTable(columns=SHORTLIST_COLUMNS, rows=rows)


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def numbers_tex(
    records: list[DatasetProbeRecord],
    picks: list[BenchmarkPick],
    measured: dict[str, PickMeasurement],
    payload: dict[str, Any],
) -> str:
    core = [measured[p.dataset_class] for p in picks if p.tier == "core"]
    by_tier = {t: [r for r in records if r.seq_fit == t] for t in SEQ_FIT_LABEL}
    served = {r.dataset_class: r for r in records}
    lines = [
        "%% GENERATED FILE -- do not hand-edit.",
        "%% SOURCE: experiments/database/scripts/build_ecoli_genslm_benchmark_table.py",
        "%% Every number the prose quotes, as a macro, so it is read from the snapshot.",
        _macro("snapshotDate", payload["date"]),
        _macro("ecoliDatasetCount", f"{len(records)}"),
        _macro("ecoliInstanceTotal", f"{sum(r.instances for r in records):,}"),
        _macro("ecoliSectionCount", f"{len(payload['sections'])}"),
        _macro("ecoliBestCount", f"{len(by_tier['best'])}"),
        _macro("ecoliGoodCount", f"{len(by_tier['good'])}"),
        _macro("ecoliWeakCount", f"{len(by_tier['weak'])}"),
        _macro("ecoliNoneCount", f"{len(by_tier['none'])}"),
        _macro("pickCount", f"{len(picks)}"),
        _macro("coreCount", f"{len(core)}"),
        _macro("coreGenesMin", f"{min(m.labeled_genes for m in core):,}"),
        _macro("coreGenesMax", f"{max(m.labeled_genes for m in core):,}"),
        _macro("coreLabelsTotal", f"{sum(m.labels for m in core):,}"),
        _macro("priceEnv", f"{served['RbTnseqPrice2018EcoliDataset'].environments:,}"),
    ]
    short = {
        "GeneEssentialityGoodall2018Dataset": "goodall",
        "EnvChemgenNichols2011Dataset": "nichols",
        "ProteomeSchmidt2016Dataset": "schmidt",
        "GeneInteractionBabu2014Dataset": "babu",
        "ProteinSynthesisRateLi2014Dataset": "li",
        "RnaseqLamoureux2023Dataset": "precise",
        "MorphologyCampos2018Dataset": "campos",
        "CrispriGeneFitnessWang2018Dataset": "wang",
    }
    for cls, stem in short.items():
        m = measured[cls]
        r = served[cls]
        lines += [
            _macro(f"{stem}Genes", f"{m.labeled_genes:,}"),
            _macro(f"{stem}Labels", f"{m.labels:,}"),
            _macro(f"{stem}Env", f"{m.environments:,}"),
            _macro(f"{stem}Instances", f"{r.instances:,}"),
            _macro(f"{stem}Identifier", tex_escape(m.identifier)),
        ]
    lines.append(
        _macro(
            "guideStoreCount",
            f"{sum(1 for r in records if r.dataset_class in GUIDE_LEVEL)}",
        )
    )
    return "\n".join(lines) + "\n"


# Stores whose label sits on a guide RNA rather than on the gene it targets.
GUIDE_LEVEL = {
    "CrispriKnockdownCui2018Dataset",
    "CrispriGuideFitnessWang2018Dataset",
    "CrispriChemgenChoe2025Dataset",
    "CrispriScreenRousset2018Dataset",
    "MismatchCrispriFitnessHawkins2020Dataset",
    "CrispriGuideFfaEnrichmentFang2025Dataset",
}


def write_note_block(
    picks: list[BenchmarkPick], measured: dict[str, PickMeasurement], data_path: Path
) -> None:
    """Replace (or append) the generated block of the paired Dendron note."""
    table = shortlist_table(picks, measured).to_markdown(
        sectioned=True, heading_level=4
    )
    block = "\n".join(
        [
            NOTE_START,
            "",
            "### GenSLM-style benchmark short list, measured from the dev stores",
            "",
            f"Generated {date.today().isoformat()} off `{data_path.relative_to(REPO)}` and a full",
            "scan of each shortlisted LMDB by `build_ecoli_genslm_benchmark_table.py`. *Genes* =",
            "distinct gene ids carrying a label; *Labels* = gene-attached numeric or boolean",
            "values; *Env* = distinct stored environments; *Gene identifier* = the namespace the",
            "store keys genes by. The typeset version is `notes-tex/database/ecoli-datasets/`.",
            "",
            table,
            "",
            NOTE_END,
        ]
    )
    front = read_frontmatter(NOTE, default_title="Ecoli Datasets")
    text = NOTE.read_text() if NOTE.exists() else front + "\n"
    if NOTE_START in text and NOTE_END in text:
        start = text.index(NOTE_START)
        end = text.index(NOTE_END) + len(NOTE_END)
        text = text[:start] + block + text[end:]
    else:
        text = text.rstrip("\n") + "\n\n" + block + "\n"
    NOTE.write_text(text.rstrip("\n") + "\n")
    print(f"Wrote {NOTE.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=None, help="E. coli snapshot JSON")
    ap.add_argument(
        "--reuse-measurements",
        action="store_true",
        help=f"read {MEASUREMENTS.name} instead of scanning the stores",
    )
    ap.add_argument("--workers", type=int, default=len(PICKS))
    args = ap.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    data_path = (args.data or newest_ecoli_json()).resolve()
    payload = json.loads(data_path.read_text())
    if payload["organisms"] != ["ecoli"]:
        raise ValueError(
            f"{data_path} is not an E. coli snapshot: {payload['organisms']}"
        )
    records = [DatasetProbeRecord(**d) for d in payload["datasets"]]
    sections = payload["sections"]
    served = {r.dataset_class: r for r in records}
    missing = [
        n for p in PICKS for n in (p.dataset_class, *p.alternates) if n not in served
    ]
    if missing:
        raise ValueError(f"short list names datasets not in the snapshot: {missing}")

    if args.reuse_measurements:
        snap = ShortlistSnapshot(**json.loads(MEASUREMENTS.read_text()))
        measured = {m.dataset_class: m for m in snap.measurements}
    else:
        jobs = [
            (
                p.dataset_class,
                served[p.dataset_class].slug,
                osp.join(
                    data_root,
                    "data",
                    "torchcell",
                    served[p.dataset_class].slug,
                    "processed",
                    "lmdb",
                ),
            )
            for p in PICKS
        ]
        with get_context("fork").Pool(processes=args.workers) as pool:
            measured = {m.dataset_class: m for m in pool.map(measure_store, jobs)}
        for m in measured.values():
            print(
                f"  {m.dataset_class}: {m.labeled_genes:,} genes, {m.labels:,} labels, "
                f"{m.environments} env, {m.identifier}"
            )
        snap = ShortlistSnapshot(
            date=date.today().isoformat(),
            source_json=str(data_path.relative_to(REPO)),
            picks=PICKS,
            measurements=[measured[p.dataset_class] for p in PICKS],
        )
        MEASUREMENTS.parent.mkdir(parents=True, exist_ok=True)
        MEASUREMENTS.write_text(snap.model_dump_json(indent=2) + "\n")
        print(f"Wrote {MEASUREMENTS.relative_to(REPO)}")

    TEX_DIR.mkdir(parents=True, exist_ok=True)
    src = source_comment(data_path)
    (TEX_DIR / "datasets.tex").write_text(
        src
        + landscape(
            datasets_table(records, sections).to_latex(
                caption=(
                    f"The \\emph{{E. coli}} datasets served from the dev tree, snapshot "
                    f"{payload['date']}, grouped by phenotype family. \\emph{{Genotypes}} "
                    "and \\emph{Env} count distinct stored genotype and environment "
                    "payloads; \\emph{Instances} is the record count; \\emph{Signal} the "
                    "gzip size in bits of the concatenated stored instances (perturbation, "
                    "environment and phenotype). \\emph{Seq.\\ fit} is the sequence-only "
                    "fit tier of Table~\\ref{tab:tiers} and \\emph{Label unit} what one "
                    "label attaches to without the cell graph, both read from the genotype "
                    "of 400 evenly spaced records per store."
                ),
                label="tab:ecoli-datasets",
                table_env="longtable",
                size="scriptsize",
                colsep_pt=3,
            )
        )
    )
    (TEX_DIR / "tiers.tex").write_text(
        src
        + tiers_table(records).to_latex(
            caption=(
                "\\emph{E. coli} datasets and instances per sequence-only fit tier, as "
                "\\texttt{sequence\\_fit} in the probe script derives it from where the "
                "label sits and the modal edit count of the genotype sample."
            ),
            label="tab:tiers",
            table_env="table",
            position="H",
            size="footnotesize",
        )
    )
    (TEX_DIR / "shortlist.tex").write_text(
        src
        + landscape(
            shortlist_table(PICKS, measured).to_latex(
                caption=(
                    "The short list for a gene-sequence encoding benchmark. \\emph{Genes}, "
                    "\\emph{Labels}, \\emph{Env} and \\emph{Gene identifier} are read from "
                    "a full scan of each store: distinct gene ids carrying a label, "
                    "gene-attached numeric or boolean values, distinct stored "
                    "environments, and the namespace the store keys genes by. "
                    "\\emph{Task}, \\emph{Label}, \\emph{Split}, \\emph{Same task} and "
                    "\\emph{Why} are the pick list in the generating script."
                ),
                label="tab:shortlist",
                table_env="longtable",
                size="scriptsize",
                colsep_pt=3,
            )
        )
    )
    (TEX_DIR / "numbers.tex").write_text(numbers_tex(records, PICKS, measured, payload))
    for name in ("datasets", "tiers", "shortlist", "numbers"):
        print(f"Wrote {(TEX_DIR / f'{name}.tex').relative_to(REPO)}")
    write_note_block(PICKS, measured, data_path)


if __name__ == "__main__":
    main()
