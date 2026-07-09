# paper/nature-biotech/scripts/generate_datasets_table.py
# [[paper.nature-biotech.scripts.generate_datasets_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/paper/nature-biotech/scripts/generate_datasets_table.py
r"""Generate the supported-datasets table (LaTeX + markdown mirror) with a
streaming-gzip ``Signal`` column computed from each built LMDB.

This is a THIN consumer of the reusable ``torchcell.paper.tables`` module: it
owns only the dataset registry (the source of truth) and the surrounding prose;
all signal computation, caching, and md/LaTeX rendering live in the module, so
other paper tables can follow the same pattern.

Each row maps to exactly one built LMDB, so every emitted data row gets a real
gzip value (Kolmogorov-complexity proxy). The compacted ``smf/dmf/tmf`` rows in
the old note are split one-per-dataset here.

Outputs (both regenerated from the registry):
  - paper/nature-biotech/sections/datasets_table.tex   (\\input into the SI)
  - notes/paper.supported-datasets-and-databases.md      (readable mirror)

Run from the repo root:
  python paper/nature-biotech/scripts/generate_datasets_table.py            # all (slow: Costanzo dmf/dmi)
  python paper/nature-biotech/scripts/generate_datasets_table.py --max-gb 3 # quick: skip the giant LMDBs
  python paper/nature-biotech/scripts/generate_datasets_table.py --only mulleder
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.paper.tables import (
    Column,
    PaperTable,
    Row,
    SignalCache,
    human_bytes,
    read_frontmatter,
)

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
TEX_OUT = REPO / "paper" / "nature-biotech" / "sections" / "datasets_table.tex"
MD_OUT = REPO / "notes" / "paper.supported-datasets-and-databases.md"
CACHE_PATH = SCRIPT.parent / "dataset_signals_cache.json"

COLUMNS = [
    Column(header="Dataset", align="l"),
    Column(header="Genotypes", align="r"),
    Column(header="Env", align="r"),
    Column(header="Phenotype", align="l"),
    Column(header="Label", align="l"),
    Column(header="Shape", align="l"),
    Column(header="Signal (gzip)", align="r"),
]

SECTIONS = [
    "Fitness + genetic interaction",
    "Viability",
    "Morphology",
    "Expression (microarray)",
    "Expression (RNA-seq)",
    "Metabolite",
    "Protein abundance",
    "Visual / product-proxy score",
]


class DatasetRow(BaseModel):
    """One paper-table row. ``data_subpath`` is relative to ``$DATA_ROOT`` and
    points at the dataset ROOT dir (``processed/lmdb`` is appended). ``None``
    means not built yet -- the dataset is listed but has no gzip value.
    """

    section: str
    name: str
    genotypes: str
    env: str
    phenotype: str
    label: str
    shape: str
    data_subpath: str | None = None


DATASET_ROWS: list[DatasetRow] = [
    DatasetRow(section="Fitness + genetic interaction", name="Costanzo 2016 smf", genotypes="20,484", env="2", phenotype="single-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/smf_costanzo2016"),
    DatasetRow(section="Fitness + genetic interaction", name="Costanzo 2016 dmf", genotypes="20.7M", env="2", phenotype="double-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/dmf_costanzo2016"),
    DatasetRow(section="Fitness + genetic interaction", name="Costanzo 2016 dmi", genotypes="20.7M", env="2", phenotype="digenic interaction", label="edge", shape="n×1", data_subpath="data/torchcell/dmi_costanzo2016"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2018 smf", genotypes="1,539", env="1", phenotype="single-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/smf_kuzmin2018"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2018 dmf", genotypes="410,399", env="1", phenotype="double-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/dmf_kuzmin2018"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2018 tmf", genotypes="91,111", env="1", phenotype="triple-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/tmf_kuzmin2018"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2018 dmi", genotypes="410,399", env="1", phenotype="digenic interaction", label="edge", shape="n×1", data_subpath="data/torchcell/dmi_kuzmin2018"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2018 tmi", genotypes="91,111", env="1", phenotype="trigenic interaction", label="hyperedge", shape="n×1", data_subpath="data/torchcell/tmi_kuzmin2018"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2020 smf", genotypes="480", env="1", phenotype="single-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/smf_kuzmin2020"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2020 dmf", genotypes="256,862", env="1", phenotype="double-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/dmf_kuzmin2020"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2020 tmf", genotypes="537,911", env="1", phenotype="triple-mutant fitness", label="scalar", shape="n×1", data_subpath="data/torchcell/tmf_kuzmin2020"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2020 dmi", genotypes="256,862", env="1", phenotype="digenic interaction", label="edge", shape="n×1", data_subpath="data/torchcell/dmi_kuzmin2020"),
    DatasetRow(section="Fitness + genetic interaction", name="Kuzmin 2020 tmi", genotypes="537,911", env="1", phenotype="trigenic interaction", label="hyperedge", shape="n×1", data_subpath="data/torchcell/tmi_kuzmin2020"),
    DatasetRow(section="Viability", name="SGD essentiality", genotypes="1,329", env="1", phenotype="gene essentiality", label="bool", shape="n×1", data_subpath="data/torchcell/gene_essentiality_sgd"),
    DatasetRow(section="Viability", name="SynLethDB (lethal)", genotypes="14,000", env="1", phenotype="synthetic lethality", label="bool", shape="n×1", data_subpath="data/torchcell/synth_lethality_yeast_synth_leth_db"),
    DatasetRow(section="Viability", name="SynLethDB (rescue)", genotypes="6,948", env="1", phenotype="synthetic rescue", label="bool", shape="n×1", data_subpath="data/torchcell/synth_rescue_yeast_synth_leth_db"),
    DatasetRow(section="Morphology", name="Ohya 2005 (SCMD CalMorph)", genotypes="4,718", env="1", phenotype="cell morphology", label="k-vector", shape="n×k", data_subpath="data/torchcell/scmd_ohya2005"),
    DatasetRow(section="Expression (microarray)", name="Kemmeren 2014", genotypes="1,484", env="1", phenotype="mRNA log2(mut/wt)", label="~6,000-gene vector", shape="n×~6000", data_subpath="data/torchcell/microarray_kemmeren2014"),
    DatasetRow(section="Expression (microarray)", name="Sameith 2015 sm", genotypes="82", env="1", phenotype="mRNA log2(mut/ref)", label="6,169-gene vector", shape="82×6169", data_subpath="data/torchcell/sm_microarray_sameith2015"),
    DatasetRow(section="Expression (microarray)", name="Sameith 2015 dm", genotypes="72", env="1", phenotype="mRNA log2(mut/ref)", label="6,169-gene vector", shape="72×6169", data_subpath="data/torchcell/dm_microarray_sameith2015"),
    DatasetRow(section="Expression (RNA-seq)", name="Caudal 2024 (pan-transcriptome)", genotypes="943", env="1", phenotype="RNA-seq TPM (natural isolates)", label="~6,445-gene vector", shape="943×~6445", data_subpath="data/torchcell/caudal_pantranscriptome2024"),
    DatasetRow(section="Metabolite", name="Cachera 2023 (CRI-SPA betaxanthin)", genotypes="4,735", env="1", phenotype="betaxanthin (product proxy)", label="scalar", shape="n×1", data_subpath="data/torchcell/betaxanthin_cachera2023"),
    DatasetRow(section="Metabolite", name="Mülleder 2016 (amino-acid metabolome)", genotypes="4,678", env="1", phenotype="19 amino-acid concentrations", label="19-vector", shape="4678×19", data_subpath="data/torchcell/amino_acid_mulleder2016"),
    DatasetRow(section="Metabolite", name="Zelezniak 2018 (metabolome)", genotypes="97", env="1", phenotype="~46 metabolite levels", label="~46-vector", shape="97×~46", data_subpath="data/torchcell/metabolite_zelezniak2018"),
    DatasetRow(section="Protein abundance", name="Zelezniak 2018 (SWATH proteome)", genotypes="97", env="1", phenotype="protein abundance", label="726-protein vector", shape="97×726", data_subpath="data/torchcell/proteome_zelezniak2018"),
    DatasetRow(section="Visual / product-proxy score", name="Ozaydin 2013 (β-carotene screen)", genotypes="4,474", env="1", phenotype="colony-color visual score", label="ordinal scalar", shape="n×1", data_subpath="data/torchcell/carotenoid_ozaydin2013"),
]

NOT_BUILT = [
    "Baryshnikova 2010 (smf; class dormant)",
    "Ohnuki 2018 / 2022 (morphology)",
    "O'Duibhir 2014 (expression / fitness)",
    "Wildenhain 2015 (drug tolerance, 195 × 4,915 conditions)",
    "Lian 2017 (AID furfural tolerance)",
    "FitDb (fitness across 1,144 conditions)",
]

MD_INTRO = """\
## Supported datasets + databases (paper table)

<!-- GENERATED by paper/nature-biotech/scripts/generate_datasets_table.py -- do not
edit the tables by hand; edit the DATASET_ROWS registry in that script and rerun.
Reusable machinery lives in torchcell/paper/tables.py. -->

Readable, paper-facing inventory of the datasets currently schematized and built in the
torchcell database (all L0-L4 verified; "supported" is implied). Deliberately less
detailed than the full candidate backlog `[[paper.north-star.dataset-triage]]` -- this
table is for the reader to grasp the **diversity and scale** of the training signal, not
to plan ingestion.

**Columns.** *Genotypes* = distinct perturbed strains/combinations (curated). *Env* =
number of environments/conditions. *Shape* = genotypes × label dimensionality (scalar, a
k-vector, or a graph edge/hyperedge). *Signal (gzip)* = gzip-compressed size of the
serialized per-strain phenotype values -- a Kolmogorov-complexity proxy for the total
information in the dataset (captures breadth × depth in one number). The Signal is
computed **from the built LMDB** by the generator (streaming gzip, so even the 20M+
Costanzo LMDBs get a real value) and refreshes whenever a dataset is (re)built.
"""

MD_TAIL = """\
### In progress (not yet built/verified)

{notbuilt}. See `[[paper.north-star.dataset-triage]]` for the full ~75-candidate backlog.

## Reference databases

Curated resources torchcell reads from or cross-references (identity, annotation,
metabolic-model scaffold). Not per-strain perturbation datasets.

| Database | Type | What it provides | URL |
| :-- | :-- | :-- | :-- |
| SGD | genome / knowledgebase | reference genome, annotation, phenotype + literature curation | yeastgenome.org |
| **SPELL** | expression compendium (SGD) | search engine over 752 datasets / 15,475 arrays / 576 studies of yeast expression microarrays | spell.yeastgenome.org |
| YMDB 2.0 | metabolite database | curated yeast metabolite structures, concentrations, pathways | ymdb.ca |
| YeastNet v3 | functional gene network | probabilistic integrated gene-interaction network | inetbio.org/yeastnet |
| CYCLoPs / LoQAtE | localization + abundance | GFP-collection protein localization/abundance atlases | thecellvision / weizmann |
| TheCellMap.org | genetic-interaction portal | query/download for the Costanzo/Boone global GI network | thecellmap.org |
| Yeast9 GEM | genome-scale metabolic model | consensus stoichiometric reconstruction (metabolite-node IDs) | github SysBioChalmers/yeast-GEM |
| ScRAPdb | pan-omics assembly panel | 142-strain telomere-to-telomere reference panel omics | evomicslab.org/db/ScRAPdb |
| Yeast PeptideAtlas | proteome-observation DB | reprocessed MS peptide/protein observation confidence | peptideatlas.org/builds/yeast |

## Maintenance + provenance

- **Regenerated, never hand-edited.** This note's tables + the paper LaTeX
  (`paper/nature-biotech/sections/datasets_table.tex`) are both emitted by
  `paper/nature-biotech/scripts/generate_datasets_table.py` (thin) on top of
  `torchcell/paper/tables.py` (reusable). After a dataset is (re)built, rerun to refresh
  the *Signal (gzip)* column.
- **Signal caveat.** The gzip number includes phenotype metadata (SE, n_replicates,
  measurement_type) alongside the measured values, so it is a *relative* signal proxy,
  not an absolute information content -- comparable across datasets, not exact.
- The detailed candidate backlog + Qian-2026 cross-reference lives in
  `[[paper.north-star.dataset-triage]]`; the differentiation framing in `[[paper.north-star]]`.
"""


def lmdb_dir(row: DatasetRow, data_root: str) -> Path | None:
    """Absolute ``processed/lmdb`` dir for a row, or None if it isn't built."""
    if row.data_subpath is None:
        return None
    d = Path(data_root) / row.data_subpath / "processed" / "lmdb"
    return d if (d / "data.mdb").exists() else None


def build_table(signals: dict[str, str]) -> PaperTable:
    """Assemble the reusable ``PaperTable`` from the registry + computed signals."""
    rows: list[Row] = []
    for section in SECTIONS:
        for r in (x for x in DATASET_ROWS if x.section == section):
            rows.append(
                Row(
                    section=section,
                    cells={
                        "Dataset": r.name,
                        "Genotypes": r.genotypes,
                        "Env": r.env,
                        "Phenotype": r.phenotype,
                        "Label": r.label,
                        "Shape": r.shape,
                        "Signal (gzip)": signals.get(r.name, "pending"),
                    },
                )
            )
    return PaperTable(columns=COLUMNS, rows=rows)


def main() -> None:
    """Compute signals for every built dataset and (re)write the .tex + .md."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-gb", type=float, default=None,
                    help="Skip LMDBs whose data.mdb exceeds this many GB (marks them 'pending'). Default: compute all.")
    ap.add_argument("--only", type=str, default=None,
                    help="Substring; only (re)compute rows whose name matches (others use cache/pending).")
    ap.add_argument("--no-cache", action="store_true", help="Ignore the signal cache; recompute everything selected.")
    args = ap.parse_args()

    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    cache = SignalCache(path=CACHE_PATH) if args.no_cache else SignalCache.load(CACHE_PATH)

    signals: dict[str, str] = {}
    print(f"DATA_ROOT = {data_root}")
    print(f"Computing gzip signal for {len(DATASET_ROWS)} rows (max_gb={args.max_gb}, only={args.only!r})\n")

    for r in DATASET_ROWS:
        d = lmdb_dir(r, data_root)
        if d is None:
            signals[r.name] = "not built"
            print(f"  - {r.name}: not built (no LMDB)")
            continue

        cached = cache.entries.get(r.data_subpath or "")
        selected = args.only is None or args.only.lower() in r.name.lower()
        if not selected:
            signals[r.name] = human_bytes(cached.bytes) if cached else "pending"
            continue

        size_gb = (d / "data.mdb").stat().st_size / 1e9
        if args.max_gb is not None and size_gb > args.max_gb:
            signals[r.name] = human_bytes(cached.bytes) if cached else "pending"
            print(f"  ~ {r.name}: skipped ({size_gb:.1f} GB > --max-gb {args.max_gb}) -> {signals[r.name]}")
            continue

        assert r.data_subpath is not None
        n, nbytes, from_cache = cache.get_or_compute(r.data_subpath, d, label=r.name)
        signals[r.name] = human_bytes(nbytes)
        tag = "cached" if from_cache else f"computed, {size_gb:.2f} GB LMDB"
        print(f"  {'=' if from_cache else '*'} {r.name}: {signals[r.name]} (n={n:,}, {tag})")

    table = build_table(signals)

    footnote = "In scope but not yet built/verified (no signal): " + "; ".join(NOT_BUILT) + "."
    header_comment = (
        "GENERATED by paper/nature-biotech/scripts/generate_datasets_table.py -- do not edit by hand.\n"
        "Regenerate after any dataset (re)build. If this overflows a page, switch table*->longtable."
    )
    TEX_OUT.write_text(
        table.to_latex(
            caption=(
                "Supported datasets currently schematized and L0--L4 verified in the TorchCell "
                "database. \\emph{Signal (gzip)} is a Kolmogorov-complexity proxy: the gzip size of "
                "the concatenated per-record serialized phenotype values, computed directly from each "
                "built LMDB. It captures breadth $\\times$ depth in one number and is comparable across "
                "datasets (relative, not absolute -- it includes phenotype metadata such as standard "
                "error and replicate counts)."
            ),
            label="tab:supported-datasets",
            footnote=footnote,
            header_comment=header_comment,
        )
    )
    print(f"\nWrote {TEX_OUT.relative_to(REPO)}")

    front = read_frontmatter(MD_OUT, default_title="Supported Datasets and Databases")
    md_body = table.to_markdown(sectioned=True)
    note = "\n".join([front.rstrip("\n"), "", MD_INTRO, md_body, "", MD_TAIL.format(notbuilt=" · ".join(NOT_BUILT))])
    MD_OUT.write_text(note.rstrip("\n") + "\n")
    print(f"Wrote {MD_OUT.relative_to(REPO)}")

    filled = sum(1 for r in DATASET_ROWS if signals[r.name] not in ("pending", "not built"))
    n_notbuilt = sum(1 for r in DATASET_ROWS if signals[r.name] == "not built")
    n_pending = sum(1 for r in DATASET_ROWS if signals[r.name] == "pending")
    print(f"\nSignal filled for {filled}/{len(DATASET_ROWS)} rows ({n_notbuilt} not built, {n_pending} pending).")


if __name__ == "__main__":
    main()
