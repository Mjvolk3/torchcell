# experiments/database/scripts/build_pputida_cell_state_table.py
# [[experiments.database.pputida-cell-state]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/database/scripts/build_pputida_cell_state_table
r"""The served *P. putida* datasets that bear on alcohol exposure and cell state.

The question this answers: which of the 32 *P. putida* datasets in KG 4.0 carry an
alcohol growth or tolerance arm, or a perturbation of cell state, and what does each
one's ``genotype x environment -> phenotype`` record actually look like.

TWO LAYERS, and the split is the point. Every COUNT in the emitted tables is read off
the served store itself, not off a loader docstring: record totals, distinct
environments, media names, environment factor histograms, resolved agent counts,
temperatures, genotype perturbation types, phenotype class, measurement type, assay
type and the genome assembly the reference record pins. The CURATION is the row set
and the relevance sentence, and that is held as data in ``ROWS`` below.

DISTINCT ENVIRONMENTS ARE COUNTED BY VALUE, over a full scan. An ``Environment`` is
interned in a sibling LMDB and carried on the record as ``{"$ref": <sha256>}``, so a
store's interned table is one route to the count. It is the wrong route: the table
holds every interned constant, environments and media and compounds alike, and
nothing guarantees each entry is still referenced. So every record is read, its
environment spliced back, and the spliced object hashed. That is slower by minutes on
the two million-record transposon compendia and it is exact.

THE ISOPRENOL HALF is a second question off the same stores: which non-Carruthers
dataset helps predict the last design-build-test-learn round of the Carruthers
isoprenol campaign. It compares the two served titer campaigns axis by axis, and it
asks whether a served Carruthers record names the DBTL cycle it came from, which is
the label that task needs. It probes about a thousand records rather than 2.7 million,
so it has its own flag.

Scanning the row stores takes long enough that the derived facts are cached. Both
probes merge into the same cache and neither clobbers the other's keys:

  python experiments/database/scripts/build_pputida_cell_state_table.py --probe
      re-read the fifteen row stores
  python experiments/database/scripts/build_pputida_cell_state_table.py --probe-isoprenol
      re-read the titer campaigns and the cycle-label question
  python experiments/database/scripts/build_pputida_cell_state_table.py
      render every table from the cache

Emits:
  - notes-tex/database/pputida-cell-state/tables/candidates.tex  (the row table)
  - notes-tex/database/pputida-cell-state/tables/environment.tex (the environment axis)
  - notes-tex/database/pputida-cell-state/tables/counts.tex      (totals)
  - notes-tex/database/pputida-cell-state/tables/isoprenol.tex   (the two campaigns)
  - notes-tex/database/pputida-cell-state/tables/cycle-label.tex (the DBTL cycle label)
  - <results>/pputida_cell_state_probe.json                      (the cache)

Run from the repo root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any

import lmdb
from dotenv import load_dotenv
from pydantic import BaseModel

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
RESULTS = SCRIPT.parent.parent / "results"
TEX_DIR = REPO / "notes-tex" / "database" / "pputida-cell-state" / "tables"
CACHE = RESULTS / "pputida_cell_state_probe.json"

SOURCE_LINE = (
    "%% SOURCE: experiments/database/scripts/build_pputida_cell_state_table.py"
)

# Factor labels are long in the schema and the table column is narrow. The map is
# display only; the cache keeps the schema spelling.
FACTOR_SHORT = {
    "carbon_source": "carbon",
    "nitrogen_source": "nitrogen",
    "nutrient_dropout": "dropout",
    "small_molecule": "small mol.",
    "osmolarity": "osmolarity",
    "ph": "pH",
    "ionic_strength": "ionic",
    "dissolved_oxygen": "O2",
    "radiation": "radiation",
}

# The two served isoprenol titer campaigns, compared in Sec. isoprenol. Neither is a
# row of the main table: both measure product rather than cell state, and they are
# here because the prediction task is theirs.
TITER_STORES = {
    "Carruthers 2025": "isoprenol_titer_carruthers2025",
    "Yunus 2026": "isoprenol_titer_yunus2026",
}

# Every served Carruthers store, searched for the DBTL cycle label the campaign's
# record identity is defined by.
CYCLE_STORES = [
    "isoprenol_titer_carruthers2025",
    "campaign_proteome_carruthers2025",
    "proteome_carruthers2025",
    "proteome_fold_change_carruthers2025",
]

DBTL_TOKEN = re.compile(r"DBTL\s*\d")

# The two pooled transposon compendia, which between them hold almost every record
# in the row set. Named so the share in tables/counts.tex is generated rather than
# written out and left to drift.
COMPENDIA_STORES = ["rbtnseq_borchert2024", "rbtnseq_fitness_browser_putida"]

# The Dryad deposit's culture metadata, relative to DATA_ROOT. It carries the cycle
# label the served records do not.
RAW_CYCLE_CSV = Path(
    "torchcell-raw/carruthersAutomationMachineLearning2025/data/dryad/"
    "CRISPRi_automation_Pputida_proteomic_metadata.csv"
)


class Row(BaseModel):
    """One served dataset, with the curated half of its row.

    ``store`` is the directory under ``$DATA_ROOT/data/torchcell/``; every number in
    the emitted tables is probed from it. ``genotype_note`` qualifies the probed
    perturbation types with what the probe cannot see, such as whether a transposon
    pool was barcoded, and ``relevance`` is why the row is in this document at all.
    """

    klass: str
    store: str
    genotype_note: str | None = None
    relevance: str


# The row set: every served P. putida dataset carrying an alcohol growth or tolerance
# arm, or a perturbation of cell state. Ordered by record count descending, which is
# how the stores were first surveyed and keeps the two compendia at the top where
# their scale is visible.
ROWS: list[Row] = [
    Row(
        klass="RbTnseqBorchert2024",
        store="rbtnseq_borchert2024",
        genotype_note="gene-level aggregate, one gene per record",
        relevance=(
            "The RB-TnSeq compendium, and it carries Thompson 2020's alcohol panel. "
            "Nine alcohol and diol carbon sources, 2-methyl-1-butanol among them. "
            "These arms are growth ON the alcohol as the sole carbon source, so they "
            "measure catabolism, not tolerance to a solvent stress"
        ),
    ),
    Row(
        klass="FitnessBrowserPutida",
        store="rbtnseq_fitness_browser_putida",
        genotype_note="gene-level aggregate, one locus per record",
        relevance=(
            "The Fitness Browser deposit over the same alcohol panel, on 4,778 loci, "
            "and it is the one compendium that carries a moderated t statistic per "
            "record alongside the fitness value"
        ),
    ),
    Row(
        klass="FitnessContrastBorchert2023",
        store="fitness_contrast_borchert2023",
        relevance=(
            "The significance test of a lignin-related stressor arm against its "
            "reference arm, stored as the t statistic rather than as a fitness value"
        ),
    ),
    Row(
        klass="EnvMetalTnseqRoyet2025",
        store="pputida_env_metal_tnseq_royet2025",
        genotype_note="non-barcoded mariner pool, gene-level aggregate",
        relevance=(
            "Cobalt, copper, zinc and cadmium at sub-inhibitory dose over 12 "
            "generations. A metal stress rather than a solvent one, and the cleanest "
            "served example of a graded cell-state perturbation on a transposon pool"
        ),
    ),
    Row(
        klass="GradedGeneEssentialityRoyet2025",
        store="graded_gene_essentiality_royet2025",
        relevance=(
            "The unstressed essentiality baseline the metal contrasts above are read "
            "against, so the two rows are used together or not at all"
        ),
    ),
    Row(
        klass="RbTnseqBorchert2023",
        store="rbtnseq_borchert2023",
        relevance=(
            "The loci Borchert 2023 scored that the compendium eliminated. Disjoint "
            "from the compendium row by construction, so it adds fitness values "
            "rather than duplicating them"
        ),
    ),
    Row(
        klass="CampaignProteomeCarruthers2025",
        store="campaign_proteome_carruthers2025",
        relevance=(
            "Cell state under CRISPRi knockdown across the six automated "
            "design-build-test-learn cycles, as a proteome panel. The paired feature "
            "table for the isoprenol titer campaign"
        ),
    ),
    Row(
        klass="PutidaPrecise321Lim2022",
        store="putida_precise321_lim2022",
        relevance=(
            "The only served butanol exposure, in the compendium's Fuel project, "
            "beside an isopentanol arm, a three-point growth-phase series, a "
            "multistress panel of salt, imipenem and peroxide, an ionic-liquid "
            "adaptation series and two bioreactor runs. The broadest cell-state axis "
            "in the served set"
        ),
    ),
    Row(
        klass="IsoprenolSelectionMenasalvas2025",
        store="isoprenol_selection_menasalvas2025",
        relevance=(
            "Isoprenol production coupled to growth through a pyrF biosensor, so the "
            "readout is a categorical enrichment call rather than a titer"
        ),
    ),
    Row(
        klass="ProteomeMenasalvas2025",
        store="proteome_menasalvas2025",
        relevance=(
            "Growth phase against production phase over 2,090 loci, with the two "
            "phases stored as distinct environments rather than as one condition"
        ),
    ),
    Row(
        klass="ProteomeFoldChangeLim2025",
        store="proteome_fold_change_lim2025",
        relevance=(
            "The proteome response to isoprenol stress, as a log2 fold change with "
            "its Welch test, which is the closest served analog of a solvent "
            "tolerance signature"
        ),
    ),
    Row(
        klass="IsoprenolToleranceLim2025",
        store="isoprenol_tolerance_lim2025",
        relevance=(
            "The only served alcohol tolerance growth phenotype: specific growth rate "
            "of a tolerized lineage against wild type at the same alcohol dose"
        ),
    ),
    Row(
        klass="ProteomeLim2025",
        store="proteome_lim2025",
        relevance=(
            "Absolute abundances on the same tolerized lineages as the row above, so "
            "the tolerance phenotype has a proteome beside it"
        ),
    ),
    Row(
        klass="MetaboliteGrowthPhaseMenasalvas2025",
        store="metabolite_growth_menasalvas2025",
        relevance=(
            "Intracellular pool sizes in growth phase, the metabolite half of the "
            "same phase contrast"
        ),
    ),
    Row(
        klass="MetaboliteProductionPhaseMenasalvas2025",
        store="metabolite_production_menasalvas2025",
        relevance=(
            "The paired production-phase pools, which is the environment the titer "
            "campaigns actually sample"
        ),
    ),
]


def data_root() -> Path:
    """The dev build tree, which is the tree a loader writes and a probe may read.

    Not ``$DATA_ROOT/database/data/torchcell``: that is the KG build tree, owned by
    the build user and read-only here. The two were verified identical in record
    count against the build manifest when this document was written.
    """
    load_dotenv(REPO / ".env")
    root = os.environ["DATA_ROOT"]
    return Path(root) / "data" / "torchcell"


def interned_table(store_dir: Path) -> dict[str, Any]:
    """The store's interned constants, keyed by the sha256 a record refers to.

    A store whose records hold no interned reference has no sibling table at all, and
    ``putida_precise321_lim2022`` is one: its 321 samples carry 118 environments
    between them, so interning would save nothing. An absent directory therefore
    means an empty table, not a broken store.
    """
    path = store_dir / "processed" / "interned"
    if not path.is_dir():
        return {}
    env = lmdb.open(str(path), readonly=True, lock=False, subdir=True)
    out: dict[str, Any] = {}
    with env.begin() as txn:
        for key, value in txn.cursor():
            out[key.decode()] = pickle.loads(value)
    env.close()
    return out


def splice(obj: Any, table: dict[str, Any]) -> Any:
    """Replace every ``{"$ref": sha}`` with the interned constant it names."""
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if isinstance(ref, str) and ref in table:
            return splice(table[ref], table)
        return {k: splice(v, table) for k, v in obj.items()}
    if isinstance(obj, list):
        return [splice(v, table) for v in obj]
    return obj


def stable_hash(obj: Any) -> str:
    """A value identity for a spliced environment, so distinctness is by content."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


def probe(store_dir: Path) -> dict[str, Any]:
    """Read one served store and return every fact the tables print.

    One full pass. Environments are hashed by value rather than by their ``$ref``,
    because two refs can splice to the same object and the count the document claims
    is a count of distinct conditions.
    """
    table = interned_table(store_dir)
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )

    # A pooled screen repeats one environment across every gene, so splicing and
    # hashing it per record does the same work several thousand times. The memo is
    # keyed on the reference a record carries and caches what that reference splices
    # to, which leaves the count by value: two references whose spliced objects are
    # equal still hash the same and collapse to one environment.
    env_memo: dict[str, tuple[str, Any]] = {}
    env_hashes: set[str] = set()
    media: set[str] = set()
    factors: dict[str, int] = {}
    agents: set[str] = set()
    temperatures: set[str] = set()
    genotype_types: set[str] = set()
    phenotype: dict[str, Any] = {}
    genome: dict[str, Any] = {}
    records = 0

    with env.begin() as txn:
        records = txn.stat()["entries"]
        for _, value in txn.cursor():
            rec = pickle.loads(value)
            experiment = rec["experiment"]

            raw_env = experiment["environment"]
            memo_key = raw_env.get("$ref") if isinstance(raw_env, dict) else None
            if memo_key in env_memo:
                digest, environment = env_memo[memo_key]
            else:
                environment = splice(raw_env, table)
                digest = stable_hash(environment)
                if memo_key is not None:
                    env_memo[memo_key] = (digest, environment)
            env_hashes.add(digest)

            name = (environment.get("media") or {}).get("name")
            if name:
                media.add(name)
            temp = environment.get("temperature")
            temperatures.add(
                f"{temp['value']:g} {temp['unit']}" if temp else "not reported"
            )
            for pert in environment.get("perturbations") or []:
                label = pert.get("factor") or pert.get("perturbation_type")
                factors[label] = factors.get(label, 0) + 1
                agent = pert.get("agent") or pert.get("compound") or {}
                if agent.get("name"):
                    agents.add(agent["name"])

            for pert in (
                splice(experiment["genotype"], table).get("perturbations") or []
            ):
                genotype_types.add(pert["perturbation_type"])

            if not phenotype:
                pheno = splice(experiment["phenotype"], table)
                phenotype = {
                    "label_name": pheno.get("label_name"),
                    "measurement_type": pheno.get("measurement_type"),
                    "assay_type": pheno.get("assay_type"),
                    "graph_level": pheno.get("graph_level"),
                }
                genome = splice(rec["reference"], table)["genome_reference"]

    env.close()
    return {
        "records": records,
        "distinct_environments": len(env_hashes),
        "n_media": len(media),
        "media": sorted(media)[:4],
        "factors": dict(sorted(factors.items(), key=lambda kv: -kv[1])),
        "n_agents": len(agents),
        "temperatures": sorted(temperatures),
        "genotype_types": sorted(genotype_types),
        "phenotype": phenotype,
        "genome_species": genome.get("species"),
        "genome_strain": genome.get("strain"),
        "genome_assembly": genome.get("assembly_accession"),
        "genome_assembly_set": genome.get("assembly_set"),
    }


def probe_titer(store_dir: Path) -> dict[str, Any]:
    """One isoprenol titer campaign, for the transfer comparison of Sec. isoprenol.

    The target set is the CRISPRi half of the genotype only. A Carruthers record also
    carries the heterologous isoprenol pathway and, on some strains, a deletion or a
    gene addition, and counting those as targets would overstate the overlap between
    the two campaigns.
    """
    table = interned_table(store_dir)
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )

    crispri: set[str] = set()
    pert_types: set[str] = set()
    titers: list[float] = []
    with_uncertainty = 0
    media: set[str] = set()
    temperatures: set[str] = set()
    durations: set[float] = set()
    units: set[str] = set()
    methods: set[str] = set()
    replicates: set[str] = set()
    records = 0

    with env.begin() as txn:
        records = txn.stat()["entries"]
        for _, value in txn.cursor():
            rec = splice(pickle.loads(value), table)
            experiment = rec["experiment"]
            for pert in experiment["genotype"]["perturbations"]:
                pert_types.add(pert["perturbation_type"])
                if pert["perturbation_type"] == "crispr_interference":
                    crispri.add(pert["systematic_gene_name"])
            pheno = experiment["phenotype"]
            titers.append(pheno["titer"])
            if pheno.get("titer_uncertainty") is not None:
                with_uncertainty += 1
            units.add(pheno["titer_unit"])
            methods.add(pheno["quantification_method"])
            replicates.add(
                f"{pheno['n_samples']} {pheno['sample_unit'].replace('_', ' ')}s"
            )
            environment = experiment["environment"]
            media.add((environment.get("media") or {}).get("name"))
            temp = environment.get("temperature")
            temperatures.add(
                f"{temp['value']:g} {temp['unit']}" if temp else "not reported"
            )
            durations.add(environment.get("duration_hours"))

    env.close()
    return {
        "records": records,
        "n_crispri_targets": len(crispri),
        "crispri_targets": sorted(crispri),
        "perturbation_types": sorted(pert_types),
        "titer_min": min(titers),
        "titer_max": max(titers),
        "records_with_uncertainty": with_uncertainty,
        "titer_units": sorted(units),
        "quantification_methods": sorted(methods),
        "replicates": sorted(replicates),
        "media": sorted(m for m in media if m),
        "temperatures": sorted(temperatures),
        "duration_hours": sorted(d for d in durations if d is not None),
    }


def probe_cycle_label(store_dir: Path) -> dict[str, Any]:
    """Whether a served Carruthers record names the DBTL cycle it came from.

    The loader's record identity is one per construct and DBTL cycle, so the cycle is
    implied. This asks whether it was ever written down: every record is serialized
    and searched for a DBTL token, and ``screen_id`` is counted separately because it
    is the field a campaign round would naturally use.
    """
    table = interned_table(store_dir)
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )

    records = 0
    with_token = 0
    with_screen_id = 0
    with env.begin() as txn:
        records = txn.stat()["entries"]
        for _, value in txn.cursor():
            rec = splice(pickle.loads(value), table)
            blob = json.dumps(rec, default=str)
            if DBTL_TOKEN.search(blob):
                with_token += 1
            if rec["experiment"]["phenotype"].get("screen_id"):
                with_screen_id += 1
    env.close()
    return {
        "records": records,
        "records_naming_a_cycle": with_token,
        "records_with_screen_id": with_screen_id,
    }


def probe_raw_cycles() -> dict[str, Any]:
    """The cycle label as the raw mirror holds it, which is the recovery path.

    Rows are cultures and include the per-cycle controls, so these counts are not
    record counts; what they establish is that the label exists upstream and is
    joinable on the construct name.
    """
    load_dotenv(REPO / ".env")
    path = Path(os.environ["DATA_ROOT"]) / RAW_CYCLE_CSV
    # utf-8-sig, because the deposit's CSV opens with a byte-order mark and the
    # first column name would otherwise read as "﻿DBTL_Cycle".
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["DBTL_Cycle"]] = counts.get(row["DBTL_Cycle"], 0) + 1
    return {
        "path": str(RAW_CYCLE_CSV),
        "rows": len(rows),
        "join_key": "Line_name",
        "counts": dict(sorted(counts.items())),
    }


def run_probe() -> dict[str, Any]:
    """Probe every row's store and write the cache."""
    base = data_root()
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    for row in ROWS:
        print(f"probing {row.store} ...", flush=True)
        out[row.store] = probe(base / row.store)
        print(f"  {json.dumps(out[row.store])}", flush=True)
    _save(out)
    return out


def run_probe_isoprenol() -> dict[str, Any]:
    """Probe the isoprenol transfer comparison and the cycle-label question.

    Separate from ``--probe`` because these stores hold about a thousand records
    between them while the row probe reads 2.7 million, and the two answers are
    revised on different occasions.
    """
    base = data_root()
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    titer: dict[str, Any] = {}
    for label, store in TITER_STORES.items():
        print(f"probing titer {store} ...", flush=True)
        titer[label] = probe_titer(base / store) | {"store": store}
    shared = set(titer["Carruthers 2025"]["crispri_targets"]) & set(
        titer["Yunus 2026"]["crispri_targets"]
    )
    cycles: dict[str, Any] = {}
    for store in CYCLE_STORES:
        print(f"probing cycle label {store} ...", flush=True)
        cycles[store] = probe_cycle_label(base / store)
    out["isoprenol"] = {
        "campaigns": titer,
        "shared_crispri_targets": len(shared),
        "yunus_only": sorted(
            set(titer["Yunus 2026"]["crispri_targets"])
            - set(titer["Carruthers 2025"]["crispri_targets"])
        ),
    }
    out["cycle_label"] = {"served": cycles, "raw": probe_raw_cycles()}
    _save(out)
    return out


def _save(out: dict[str, Any]) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {CACHE.relative_to(REPO)}")


def tex_escape(s: str) -> str:
    r"""Escape every TeX special in free text.

    The backslash goes first so the later replacements are not escaped twice.
    ``^`` and ``~`` are here because they are fatal rather than merely special: an
    unescaped one halts the engine with "Missing $ inserted".
    """
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in [
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
        ("$", r"\$"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("^", r"\textasciicircum{}"),
        ("~", r"\textasciitilde{}"),
    ]:
        s = s.replace(a, b)
    return s


def breakable(s: str) -> str:
    r"""Escape an identifier and let TeX break it.

    A class name like ``GradedGeneEssentialityRoyet2025`` and an accession like
    ``GCA_000007565.2`` are each one unbreakable token, wide enough to run off a
    narrow column. \allowbreak permits a break rather than forcing one.
    """
    out = tex_escape(s)
    for tok in (r"\_", "-", "."):
        out = out.replace(tok, tok + r"\allowbreak ")
    return out


def class_tex(name: str) -> str:
    r"""A dataset class name with break opportunities between its humps.

    ``MetaboliteProductionPhaseMenasalvas2025`` carries no underscore, hyphen or dot,
    so \breakable leaves it one 39-character token and it runs 36 mm past a 34 mm
    column. A break is offered before every capital that follows a lower-case letter
    or a digit, which is where a reader would break it anyway.
    """
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", r"\\allowbreak ", breakable(name))


def factor_tex(factors: dict[str, int]) -> str:
    """The environment factor histogram, shortest label first in the display map."""
    if not factors:
        return "none"
    return ", ".join(
        f"{FACTOR_SHORT.get(k, k.replace('_', ' '))} {v:,}" for k, v in factors.items()
    )


def temperature_tex(temps: list[str]) -> str:
    """Temperatures as the store holds them, including a typed absence."""
    return ", ".join(t.replace(" Celsius", r"\,\textdegree C") for t in temps)


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "%% GENERATED FILE -- do not hand-edit.\n" + SOURCE_LINE + "\n" + body
    )
    print(f"Wrote {path.relative_to(REPO)}")


def render_candidates(cache: dict[str, Any]) -> str:
    """The row table: genotype, environment, phenotype and genome per dataset."""
    cols = r"@{}L{30mm} r@{\hspace{5pt}} L{30mm} L{17mm} L{34mm} L{20mm} L{66mm}@{}"
    hdr = (
        r"\textbf{Dataset} & \textbf{Records} & \textbf{Genotype} & "
        r"\textbf{Env.} & \textbf{Phenotype} & \textbf{Genome} & \textbf{Why it is here} \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The served \org{P. putida} datasets bearing on alcohol exposure or cell
state, ordered by record count. Every column but the last is read off the served store
(Sec.~\ref{sec:method}). \emph{Genotype} gives the perturbation types the records
carry, with what the probe cannot see added in italic. \emph{Env.} is the number of
distinct \texttt{Environment} objects, counted by value over a full scan, and
Table~\ref{tab:pcsenv} breaks each one into its media, factors and agents.
\emph{Phenotype} is the stored class, then its measurement type and assay type.
\emph{Genome} is the assembly the reference record pins; the experiment-level
\texttt{genome\_reference} is null throughout, by design, since the genome never gates
a record.}
\label{tab:pcsrows}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{7}{@{}l}{\footnotesize\emph{Table~\ref{tab:pcsrows}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for row in ROWS:
        probed = cache[row.store]
        genotype = (
            ", ".join(breakable(t.replace("_", " ")) for t in probed["genotype_types"])
            or "none"
        )
        if row.genotype_note:
            genotype += (
                r"\newline {\scriptsize\emph{" + tex_escape(row.genotype_note) + r"}}"
            )
        pheno = probed["phenotype"]
        phenotype = (
            r"\textbf{"
            + breakable(str(pheno["label_name"]))
            + r"}\newline {\scriptsize "
            + breakable(str(pheno["measurement_type"]))
            + " / "
            + breakable(str(pheno["assay_type"]))
            + r"}"
        )
        genome = (
            breakable(str(probed["genome_strain"]))
            + r"\newline {\scriptsize "
            + breakable(str(probed["genome_assembly"]))
            + r"}"
        )
        lines.append(
            " & ".join(
                [
                    r"\textbf{" + class_tex(row.klass) + r"}",
                    f"{probed['records']:,}",
                    genotype,
                    f"{probed['distinct_environments']:,}",
                    phenotype,
                    genome,
                    tex_escape(row.relevance),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[5pt]")
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_environment(cache: dict[str, Any]) -> str:
    """The environment axis, one dataset per row, factors broken out."""
    cols = r"@{}L{34mm} r@{\hspace{5pt}} r@{\hspace{5pt}} L{48mm} r@{\hspace{5pt}} L{26mm}@{}"
    hdr = (
        r"\textbf{Dataset} & \textbf{Env.} & \textbf{Media} & "
        r"\textbf{Factors (perturbations carrying each)} & \textbf{Agents} & "
        r"\textbf{Temperature} \\"
    )
    head = (
        r"""\begingroup
\footnotesize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.15}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The environment axis of each row in Table~\ref{tab:pcsrows}. \emph{Env.} is
distinct \texttt{Environment} objects and \emph{Media} distinct media names.
\emph{Factors} counts the environment perturbations carrying each factor, summed over
records, so a dataset of many records shows large counts for a few conditions.
\emph{Agents} is distinct compounds resolved to an InChIKey and a PubChem identifier,
which is what makes an alcohol arm queryable by structure rather than by string.
\emph{Temperature} prints every value the store holds, a typed absence included.}
\label{tab:pcsenv}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{6}{@{}l}{\footnotesize\emph{Table~\ref{tab:pcsenv}, continued}}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endhead
\bottomrule
\endfoot
"""
    )
    lines = []
    for row in ROWS:
        probed = cache[row.store]
        lines.append(
            " & ".join(
                [
                    class_tex(row.klass),
                    f"{probed['distinct_environments']:,}",
                    f"{probed['n_media']:,}",
                    factor_tex(probed["factors"]),
                    f"{probed['n_agents']:,}",
                    temperature_tex(probed["temperatures"]),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[3pt]")
    return head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n"


def render_counts(cache: dict[str, Any]) -> str:
    """Totals, so the document's headline numbers are generated too."""
    records = sum(cache[r.store]["records"] for r in ROWS)
    environments = sum(cache[r.store]["distinct_environments"] for r in ROWS)
    assemblies = {cache[r.store]["genome_assembly"] for r in ROWS}
    pheno_classes = {cache[r.store]["phenotype"]["label_name"] for r in ROWS}
    genotype_types: set[str] = set()
    for row in ROWS:
        genotype_types.update(cache[row.store]["genotype_types"])
    # The two pooled compendia against everything else, because the ratio is the
    # reason record count is a cost estimate rather than a ranking.
    compendia = sum(cache[s]["records"] for s in COMPENDIA_STORES)
    share = 100.0 * compendia / records
    body = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{longtable}{@{}L{86mm} r@{}}",
        r"\caption[]{Totals over the rows of Table~\ref{tab:pcsrows}, generated with "
        r"them.}\label{tab:pcscounts}\\",
        r"\toprule",
        r"\textbf{Quantity} & \textbf{Value} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
        rf"Datasets & {len(ROWS)} \\",
        rf"Records & {records:,} \\",
        rf"Records in the two pooled transposon compendia & {compendia:,} \\",
        rf"Their share of all records & {share:.1f}\% \\",
        rf"Distinct environments, summed over datasets & {environments:,} \\",
        rf"Distinct phenotype classes & {len(pheno_classes)} \\",
        rf"Distinct genotype perturbation types & {len(genotype_types)} \\",
        rf"Distinct genome assemblies & {len(assemblies)} \\",
        r"\end{longtable}",
        r"\endgroup",
    ]
    return "\n".join(body) + "\n"


def render_isoprenol(cache: dict[str, Any]) -> str:
    """The two isoprenol campaigns side by side, axis by axis."""
    iso = cache["isoprenol"]
    carr = iso["campaigns"]["Carruthers 2025"]
    yun = iso["campaigns"]["Yunus 2026"]

    def both(key: str, fmt: Any = str) -> tuple[str, str]:
        return fmt(carr[key]), fmt(yun[key])

    def joined(key: str) -> tuple[str, str]:
        return (
            ", ".join(tex_escape(str(v)) for v in carr[key]),
            ", ".join(tex_escape(str(v)) for v in yun[key]),
        )

    # A row whose right cell is None spans both campaign columns, for a value the two
    # share rather than one each.
    rows: list[tuple[str, str, str | None]] = [
        ("Records", f"{carr['records']:,}", f"{yun['records']:,}"),
        (
            "CRISPRi targets",
            f"{carr['n_crispri_targets']:,}",
            f"{yun['n_crispri_targets']:,}",
        ),
        (
            "Targets shared by both campaigns",
            r"\multicolumn{2}{@{}l@{}}{" + f"{iso['shared_crispri_targets']:,}" + r"}",
            None,
        ),
        (
            "Genotype perturbation types",
            ", ".join(
                breakable(t.replace("_", " ")) for t in carr["perturbation_types"]
            ),
            ", ".join(
                breakable(t.replace("_", " ")) for t in yun["perturbation_types"]
            ),
        ),
        ("Medium", *joined("media")),
        (
            "Temperature",
            *(temperature_tex(c) for c in (carr["temperatures"], yun["temperatures"])),
        ),
        (
            "Duration",
            ", ".join(f"{d:g} h" for d in carr["duration_hours"]),
            ", ".join(f"{d:g} h" for d in yun["duration_hours"]),
        ),
        ("Replicates", *joined("replicates")),
        ("Quantification", *joined("quantification_methods")),
        ("Titer unit", *joined("titer_units")),
        (
            "Titer range",
            f"{carr['titer_min']:.2f} to {carr['titer_max']:,.1f}",
            f"{yun['titer_min']:.2f} to {yun['titer_max']:,.1f}",
        ),
        (
            "Records carrying an uncertainty",
            f"{carr['records_with_uncertainty']:,} of {carr['records']:,}",
            f"{yun['records_with_uncertainty']:,} of {yun['records']:,}",
        ),
    ]
    body = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{@{}L{44mm} L{62mm} L{62mm}@{}}",
        r"\caption[]{The two served isoprenol titer campaigns, axis by axis, every "
        r"value read off the store. The product is the same compound in both, "
        r"isoprenol at InChIKey CPJRRXSHAYUTGL-UHFFFAOYSA-N and PubChem 12988. "
        r"\emph{Targets shared} is the intersection of the two CRISPRi target sets, "
        r"counting the CRISPRi half of the genotype only, since a Carruthers record "
        r"also carries the heterologous pathway.}\label{tab:pcsiso}\\",
        r"\toprule",
        r"\textbf{Axis} & \textbf{Carruthers 2025} & \textbf{Yunus 2026} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for label, left, right in rows:
        cells = f"{left}" if right is None else f"{left} & {right}"
        body.append(f"{label} & {cells} " + r"\\")
        body.append(r"\addlinespace[2pt]")
    body += [r"\end{longtable}", r"\endgroup"]
    extra = iso["yunus_only"]
    if extra:
        body.append(
            r"\noindent\emph{Targets in Yunus 2026 and not in Carruthers 2025:} "
            + ", ".join(r"\gene{" + breakable(t) + r"}" for t in extra)
            + "."
        )
    return "\n".join(body) + "\n"


def render_cycle_label(cache: dict[str, Any]) -> str:
    """The cycle-label absence, per served store, with the recovery path beside it."""
    served = cache["cycle_label"]["served"]
    raw = cache["cycle_label"]["raw"]
    body = [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{longtable}{@{}L{62mm} r@{\hspace{8pt}} r@{\hspace{8pt}} r@{}}",
        r"\caption[]{The DBTL cycle label in the served Carruthers stores. "
        r"\emph{Naming a cycle} counts records whose full serialization matches "
        r"\texttt{DBTL\textbackslash s*\textbackslash d} anywhere, so it is as "
        r"permissive a search as the store allows.}\label{tab:pcscycle}\\",
        r"\toprule",
        r"\textbf{Store} & \textbf{Records} & \textbf{Naming a cycle} & "
        r"\textbf{With a screen id} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for store, probed in served.items():
        body.append(
            " & ".join(
                [
                    breakable(store),
                    f"{probed['records']:,}",
                    f"{probed['records_naming_a_cycle']:,}",
                    f"{probed['records_with_screen_id']:,}",
                ]
            )
            + r" \\"
        )
    body += [r"\end{longtable}", ""]
    body += [
        r"\begingroup",
        r"\footnotesize",
        r"\begin{longtable}{@{}L{62mm} r@{}}",
        r"\caption[]{The same label in the raw mirror, which is the recovery path. "
        r"Rows are cultures and include each cycle's controls, so these are not "
        r"record counts; they establish that the label exists upstream and joins on "
        r"\texttt{" + breakable(raw["join_key"]) + r"}.}\label{tab:pcsrawcycle}\\",
        r"\toprule",
        r"\textbf{Cycle} & \textbf{Culture rows} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for cycle, count in raw["counts"].items():
        body.append(f"DBTL{tex_escape(cycle)} & {count:,} " + r"\\")
    body.append(r"\midrule")
    body.append(r"\textbf{Total} & \textbf{" + f"{raw['rows']:,}" + r"} \\")
    body += [r"\end{longtable}", r"\endgroup", r"\endgroup"]
    return "\n".join(body) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="re-read the fifteen row stores, 2.7 million records, and update the cache",
    )
    parser.add_argument(
        "--probe-isoprenol",
        action="store_true",
        help="re-read the isoprenol campaigns and the cycle-label question, and update the cache",
    )
    args = parser.parse_args()

    if args.probe:
        run_probe()
    if args.probe_isoprenol:
        run_probe_isoprenol()
    cache = json.loads(CACHE.read_text())

    write(TEX_DIR / "candidates.tex", render_candidates(cache))
    write(TEX_DIR / "environment.tex", render_environment(cache))
    write(TEX_DIR / "counts.tex", render_counts(cache))
    write(TEX_DIR / "isoprenol.tex", render_isoprenol(cache))
    write(TEX_DIR / "cycle-label.tex", render_cycle_label(cache))


if __name__ == "__main__":
    main()
