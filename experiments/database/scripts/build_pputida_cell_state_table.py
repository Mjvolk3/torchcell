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
  python experiments/database/scripts/build_pputida_cell_state_table.py --probe-correlates
      test every served gene-level readout against the Carruthers target effect;
      reads both compendia in full, so it takes a few minutes, and it needs the
      --probe-roles cache for the class names
  python experiments/database/scripts/build_pputida_cell_state_table.py
      render every table from the cache

Emits:
  - notes-tex/database/pputida-cell-state/tables/candidates.tex  (the row table)
  - notes-tex/database/pputida-cell-state/tables/environment.tex (the environment axis)
  - notes-tex/database/pputida-cell-state/tables/counts.tex      (totals)
  - notes-tex/database/pputida-cell-state/tables/isoprenol.tex   (the two campaigns)
  - notes-tex/database/pputida-cell-state/tables/cycle-label.tex (the DBTL cycle label)
  - notes-tex/database/pputida-cell-state/tables/correlates.tex  (gene-level correlates)
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
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from scipy.stats import kruskal, pearsonr, rankdata, spearmanr

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

# Carruthers' own four stores. The question the transfer table answers is which OTHER
# served data can help that campaign, so the campaign is the thing joined to and never
# a row of it.
CARRUTHERS_STORES = {
    "isoprenol_titer_carruthers2025",
    "campaign_proteome_carruthers2025",
    "proteome_carruthers2025",
    "proteome_fold_change_carruthers2025",
}

# The two pooled transposon compendia, which between them hold almost every record
# in the row set. Named so the share in tables/counts.tex is generated rather than
# written out and left to drift.
COMPENDIA_STORES = ["rbtnseq_borchert2024", "rbtnseq_fitness_browser_putida"]

# Isoprenol, by structure. Matching on the name alone would both miss a record that
# spells it 3-methyl-3-buten-1-ol and hit DL-3-aminoisobutyric acid on a looser stem,
# so the InChIKey skeleton decides and the names are only a secondary net.
ISOPRENOL_INCHIKEY = "CPJRRXSHAYUTGL"
ISOPRENOL_NAMES = ("isoprenol", "3-methyl-3-buten-1-ol")

# Every served P. putida store, to answer "which of these touch isoprenol" over the
# whole organism rather than over the cell-state row set.
PPUTIDA_STORES = [
    "campaign_proteome_carruthers2025",
    "crispri_array_yunus2026",
    "crispri_differential_proteome_yunus2026",
    "crispri_knockdown_yunus2026",
    "crispri_panel_proteome_yunus2026",
    "fitness_contrast_borchert2023",
    "graded_gene_essentiality_royet2025",
    "isoprenol_selection_menasalvas2025",
    "isoprenol_titer_banerjee2024",
    "isoprenol_titer_carruthers2025",
    "isoprenol_titer_desiqueira2025",
    "isoprenol_titer_yunus2026",
    "isoprenol_tolerance_lim2025",
    "isoprenyl_acetate_titer_kang2026",
    "lactam_growth_rate_thompson2019",
    "metabolite_growth_menasalvas2025",
    "metabolite_production_menasalvas2025",
    "pputida_env_metal_tnseq_royet2025",
    "proteome_banerjee2025",
    "proteome_carruthers2025",
    "proteome_desiqueira2025",
    "proteome_fold_change_carruthers2025",
    "proteome_fold_change_lim2025",
    "proteome_lim2025",
    "proteome_log10_percent_desiqueira2025",
    "proteome_menasalvas2025",
    "proteome_percent_desiqueira2025",
    "putida_precise321_lim2022",
    "rbtnseq_borchert2023",
    "rbtnseq_borchert2024",
    "rbtnseq_fitness_browser_putida",
    "valerolactam_titer_thompson2019",
]

# The pooled screens, decided on their interned table rather than record by record.
# A store that interns its Environment holds every compound it uses as an interned
# constant, so a compound absent from that table is absent from the store. Reading
# five million records to learn which of a few hundred environments carry one
# compound is the same answer at a hundred times the cost.
POOLED_STORES = {
    "rbtnseq_borchert2024",
    "rbtnseq_fitness_browser_putida",
    "rbtnseq_borchert2023",
    "fitness_contrast_borchert2023",
    "pputida_env_metal_tnseq_royet2025",
    "graded_gene_essentiality_royet2025",
}

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
    strains: set[str] = set()
    records = 0

    with env.begin() as txn:
        records = txn.stat()["entries"]
        for _, value in txn.cursor():
            rec = splice(pickle.loads(value), table)
            strains.add(rec["reference"]["genome_reference"]["strain"])
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
        "chassis_strains": sorted(strains),
        "media": sorted(m for m in media if m),
        "temperatures": sorted(temperatures),
        "duration_hours": sorted(d for d in durations if d is not None),
    }


def mentions_isoprenol(obj: Any) -> bool:
    """Whether a spliced object names isoprenol by structure or by name."""
    blob = json.dumps(obj, default=str)
    return ISOPRENOL_INCHIKEY in blob or any(n in blob.lower() for n in ISOPRENOL_NAMES)


def probe_isoprenol_roles(store_dir: Path, pooled: bool) -> dict[str, Any]:
    """The four roles isoprenol can play in one store, counted by record.

    ``product`` is the compound a titer measures, ``carbon_source`` the substrate the
    pool was grown on, ``stressor`` a compound added to the medium, and ``pathway``
    the heterologous cassette the strain carries. The first three are decided on the
    InChIKey; ``pathway`` is decided on the genotype text, since a cassette names the
    product it makes rather than carrying its structure.

    A pooled store is decided on its interned table: see ``POOLED_STORES``. For one
    that does carry isoprenol, the records are then counted by matching each record's
    environment reference against the interned environments that hold it, which is
    one cheap pass rather than a splice per record.
    """
    table = interned_table(store_dir)
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )
    roles = {"product": 0, "carbon_source": 0, "stressor": 0, "pathway": 0}
    dataset_name = None
    # Collected over the isoprenol records only, since that is the subset a model
    # would train on: a store's other records answer another question.
    strains: set[str] = set()
    environments: set[str] = set()
    namespaces: set[str] = set()
    label_names: set[str] = set()
    units: set[str] = set()
    # Genes the isoprenol records perturb, and genes they report a value for. Kept
    # apart because only the first gives a target a second measurement of its effect.
    perturbed: set[str] = set()
    measured: set[str] = set()

    def note_genotype(genotype: dict[str, Any]) -> None:
        """Record one isoprenol record's strain identity, namespace and genes."""
        perts = genotype["perturbations"]
        for pert in perts:
            if pert.get("systematic_gene_name"):
                perturbed.add(pert["systematic_gene_name"])
        strains.add(
            stable_hash(
                sorted(
                    (p.get("systematic_gene_name"), p["perturbation_type"])
                    for p in perts
                )
            )
        )
        for pert in perts:
            if pert.get("gene_namespace"):
                namespaces.add(pert["gene_namespace"])

    def note_phenotype(pheno: dict[str, Any]) -> None:
        """Record what one isoprenol label is, its unit, and the loci it keys by."""
        for field in ("protein_abundance", "protein_fold_change"):
            keys = pheno.get(field)
            if isinstance(keys, dict):
                measured.update(keys)
        label_names.add(str(pheno.get("label_name")))
        unit = pheno.get("titer_unit") or pheno.get("measurement_type")
        units.add(str(unit))

    if pooled:
        refs = set()
        for digest, const in table.items():
            if not (isinstance(const, dict) and "media" in const):
                continue
            for pert in const.get("perturbations") or []:
                agent = pert.get("agent") or pert.get("compound") or {}
                if ISOPRENOL_INCHIKEY in str(agent.get("inchikey") or ""):
                    refs.add((digest, pert.get("factor") or pert["perturbation_type"]))
        by_digest = {d: f for d, f in refs}
        with env.begin() as txn:
            total = txn.stat()["entries"]
            for _, value in txn.cursor():
                rec = pickle.loads(value)
                dataset_name = dataset_name or rec["experiment"].get("dataset_name")
                ref = rec["experiment"]["environment"].get("$ref")
                factor = by_digest.get(ref)
                if factor is None:
                    continue
                roles["carbon_source" if factor == "carbon_source" else "stressor"] += 1
                environments.add(ref)
                note_genotype(splice(rec["experiment"]["genotype"], table))
                note_phenotype(splice(rec["experiment"]["phenotype"], table))
        env.close()
        return {
            "records": total,
            "roles": roles,
            "dataset_name": dataset_name,
            "decided_on": "interned_table",
            "strains": len(strains),
            "environments": len(environments),
            "namespaces": sorted(namespaces),
            "label_names": sorted(label_names),
            "units": sorted(units),
            "genes_perturbed": sorted(perturbed),
            "genes_measured": sorted(measured),
        }

    with env.begin() as txn:
        total = txn.stat()["entries"]
        for _, value in txn.cursor():
            rec = splice(pickle.loads(value), table)
            experiment = rec["experiment"]
            dataset_name = dataset_name or experiment.get("dataset_name")
            pheno = experiment["phenotype"]
            matched = False
            if ISOPRENOL_INCHIKEY in str(
                (pheno.get("product") or {}).get("inchikey") or ""
            ):
                roles["product"] += 1
                matched = True
            for pert in experiment["environment"].get("perturbations") or []:
                agent = pert.get("compound") or pert.get("agent") or {}
                if ISOPRENOL_INCHIKEY in str(agent.get("inchikey") or ""):
                    key = (
                        "carbon_source"
                        if pert.get("factor") == "carbon_source"
                        else "stressor"
                    )
                    roles[key] += 1
                    matched = True
            if any(
                mentions_isoprenol(p) for p in experiment["genotype"]["perturbations"]
            ):
                roles["pathway"] += 1
                matched = True
            if matched:
                environments.add(stable_hash(experiment["environment"]))
                note_genotype(experiment["genotype"])
                note_phenotype(pheno)
    env.close()
    return {
        "records": total,
        "roles": roles,
        "dataset_name": dataset_name,
        "decided_on": "every_record",
        "strains": len(strains),
        "environments": len(environments),
        "namespaces": sorted(namespaces),
        "label_names": sorted(label_names),
        "units": sorted(units),
        "genes_perturbed": sorted(perturbed),
        "genes_measured": sorted(measured),
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


def run_probe_roles() -> dict[str, Any]:
    """Ask every served P. putida store what role isoprenol plays in it."""
    base = data_root()
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    roles: dict[str, Any] = {}
    for store in PPUTIDA_STORES:
        print(f"probing isoprenol roles {store} ...", flush=True)
        roles[store] = probe_isoprenol_roles(base / store, store in POOLED_STORES)
    out["isoprenol_roles"] = roles
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
        ("Chassis strain", *joined("chassis_strains")),
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


ROLE_LABEL = {
    "product": "Isoprenol is the measured product",
    "carbon_source": "Isoprenol is the carbon source: catabolism, not production",
    "stressor": "Isoprenol is the stressor",
    "pathway": "Isoprenol is the pathway the strain carries, and something else is measured",
}
ROLE_ORDER = ["product", "carbon_source", "stressor", "pathway"]

# The subtotal label that closes each band.
ROLE_TOTAL = {
    "product": "Product subtotal",
    "carbon_source": "Carbon-source subtotal",
    "stressor": "Stressor subtotal",
    "pathway": "Pathway subtotal",
}


class IsoprenolPick(BaseModel):
    """One isoprenol dataset and the modeling problem it poses.

    The same shape as the E. coli benchmark short list: what is derived from
    the store stays out of this object, and what is a judgment call lives here so it
    is versioned beside the numbers it sits next to.
    """

    store: str
    task: str = Field(description="The prediction problem, in a few words.")
    label: str = Field(description="What one label is, and what it attaches to.")
    split: str = Field(description="The held-out unit a score would be read on.")
    alternates: list[str] = Field(description="Served datasets posing the same task.")
    why: str = Field(description="Why this row, in one or two sentences.")
    helps: str = Field(
        default="",
        description=(
            "What this dataset does for the Carruthers campaign, in one sentence. "
            "Empty for Carruthers' own stores, which are joined to rather than "
            "listed."
        ),
    )


PICKS: list[IsoprenolPick] = [
    IsoprenolPick(
        store="isoprenol_titer_carruthers2025",
        task="regression, strain x cycle",
        label="isoprenol titer in ug/mL per (CRISPRi construct, DBTL cycle)",
        split="held-out cycle, the last of six",
        alternates=["IsoprenolTiterYunus2026"],
        why=(
            "The target campaign. The only served isoprenol data with a repeated "
            "design-build-test-learn structure, and the only one carrying a per-record "
            "sample SD, so a model can be weighted by measurement precision. The cycle "
            "the split needs is not on the record (Table~\\ref{tab:pcscycle})."
        ),
    ),
    IsoprenolPick(
        store="isoprenol_titer_yunus2026",
        task="regression, strain",
        label="isoprenol titer in ug/mL per CRISPRi strain",
        split="held-out condition, against Carruthers",
        alternates=["IsoprenolTiterCarruthers2025"],
        why=(
            "All but one of its targets are Carruthers targets, with the same GC-FID "
            "readout, on a different chassis designation (IY1452 against IY1449b) "
            "and in a different medium and temperature. Measured against the DBTL0 "
            "single knockdowns, it ranks the shared targets in unrelated order "
            "(Table~\\ref{tab:pcscorr}), so it does not test a transferred ranking."
        ),
        helps=(
            "A second titer for 120 of the 135 targets, in another medium, "
            "temperature and chassis. Its ranking of those targets does not track the "
            "campaign's (Table~\\ref{tab:pcscorr}), so it says how condition-bound a "
            "target effect is rather than scoring a model trained on the early cycles."
        ),
    ),
    IsoprenolPick(
        store="isoprenyl_acetate_titer_kang2026",
        task="regression, strain",
        label="titer in ug/mL per strain, isoprenol on 7 records and the ester on 19",
        split="held-out strains",
        alternates=["IsoprenolTiterBanerjee2024", "IsoprenolTiterDeSiqueira2025"],
        why=(
            "An ester campaign on the isoprenol chassis, so its isoprenol records are "
            "the upstream intermediate rather than the target. Useful as the one "
            "served case where draining the isoprenol pool is the intent."
        ),
        helps=(
            "One shared target. Its value is the design rather than the overlap: the one "
            "served case where draining the isoprenol pool on purpose is the goal."
        ),
    ),
    IsoprenolPick(
        store="isoprenol_titer_banerjee2024",
        task="regression, strain x medium",
        label="isoprenol titer in ug/mL per (strain, condition family)",
        split="held-out strains",
        alternates=["IsoprenolTiterDeSiqueira2025", "IsoprenylAcetateTiterKang2026"],
        why=(
            "Model-guided deletions and pathway swaps rather than CRISPRi, so it "
            "shares no target set with the campaigns above. Every number is read from "
            "the article prose against the pinned OCR, since no measurement table was "
            "released."
        ),
        helps=(
            "Three shared targets, and a set of deletions and pathway swaps the campaign "
            "never tried. Useful as an out-of-design check, not as training rows."
        ),
    ),
    IsoprenolPick(
        store="isoprenol_titer_desiqueira2025",
        task="regression, strain",
        label="isoprenol titer in ug/mL per tolerized isolate",
        split="held-out strains",
        alternates=["IsoprenolTiterBanerjee2024"],
        why=(
            "Acetate-tolerized isolates of a catabolism-deficient parent, with a "
            "resequenced genotype per isolate. The one served row where the strain "
            "difference is an evolved variant call rather than a designed edit."
        ),
        helps=(
            "Two shared targets on evolved isolates with resequenced genotypes, so it "
            "says what tolerization does to titer where the campaign only varies "
            "knockdowns."
        ),
    ),
    IsoprenolPick(
        store="rbtnseq_borchert2024",
        task="regression, gene",
        label="log2 fitness per gene, growing on isoprenol as the sole carbon source",
        split="held-out genes",
        alternates=["FitnessBrowserPutida"],
        why=(
            "Genome-wide loss-of-function fitness on isoprenol at 10 mM. It names the "
            "catabolism genes, which is the deletion target for stopping a producer "
            "consuming its own product, and it is the largest isoprenol block in the "
            "served data by three orders of magnitude."
        ),
        helps=(
            "The same prior over a slightly different locus set, and the arm that names "
            "the catabolism genes: deleting one stops the chassis consuming the product "
            "the campaign is optimizing."
        ),
    ),
    IsoprenolPick(
        store="rbtnseq_fitness_browser_putida",
        task="regression, gene",
        label="log2 fitness per locus on the same panel, with a moderated t",
        split="held-out genes",
        alternates=["RbTnseqBorchert2024"],
        why=(
            "The same arm in the Fitness Browser deposit over a slightly different "
            "locus set, and the only one of the two carrying a per-record t statistic, "
            "so a gene measured in a sparse pool can be down-weighted."
        ),
        helps=(
            "A growth-on-isoprenol fitness value for 86 of the 135 targets, with a "
            "moderated t so a weakly measured gene can be down-weighted. A per-target "
            "prior that needs no new experiment."
        ),
    ),
    IsoprenolPick(
        store="isoprenol_tolerance_lim2025",
        task="regression, strain",
        label="log2 growth rate against wild type at the same isoprenol dose",
        split="held-out strains",
        alternates=["ProteomeLim2025", "ProteomeFoldChangeLim2025"],
        why=(
            "The only served alcohol tolerance growth phenotype. Three records, so it "
            "is a check on a tolerance model rather than training data for one."
        ),
        helps=(
            "One shared target. Three records, so it is a sanity check on whether a "
            "high-titer strain is tolerating its own product, not a source of signal."
        ),
    ),
    IsoprenolPick(
        store="proteome_lim2025",
        task="regression, protein x strain",
        label="absolute Top3 abundance per protein in a tolerized lineage",
        split="held-out strains",
        alternates=["ProteomeFoldChangeLim2025", "ProteomeMenasalvas2025"],
        why=(
            "The feature side of the tolerance phenotype above, on the same lineages, "
            "so the pair supports a proteome-to-tolerance map at three points."
        ),
        helps=(
            "Abundances for 111 of the 135 targets under isoprenol stress, so a target's "
            "protein level is known in a condition the campaign never ran."
        ),
    ),
    IsoprenolPick(
        store="proteome_fold_change_lim2025",
        task="regression, protein x strain",
        label="log2 fold change per protein under isoprenol stress, with a Welch test",
        split="held-out strains",
        alternates=["ProteomeLim2025"],
        why=(
            "The stress response as a contrast rather than a level, which is the "
            "closest served analog of a solvent tolerance signature."
        ),
        helps=(
            "The same 111 targets as a stress contrast, which separates a target that "
            "responds to isoprenol from one that is merely abundant."
        ),
    ),
    IsoprenolPick(
        store="campaign_proteome_carruthers2025",
        task="regression, protein x strain",
        label="DIA-NN Top3 percent of proteome per protein and campaign strain",
        split="held-out cycle",
        alternates=["CrispriPanelProteomeYunus2026", "ProteomeCarruthers2025"],
        why=(
            "The features paired with the target campaign, cycle by cycle. The natural "
            "input to a titer model, and the row whose cycle label is missing for the "
            "same reason the titer's is."
        ),
    ),
    IsoprenolPick(
        store="isoprenol_selection_menasalvas2025",
        task="binary classification",
        label="enriched or not, one biosensor call per CRISPRi target and round",
        split="held-out targets",
        alternates=["IsoprenolTiterCarruthers2025"],
        why=(
            "Growth-coupled selection, so it scales where a titer assay does not. The "
            "loader is explicit that the titer it proxies is not measured per clone, "
            "so it is a ranking label and never a quantitative one."
        ),
        helps=(
            "No shared target, but a growth-coupled assay that scales past what a titer "
            "campaign can measure, so it is the route to a target list larger than 135."
        ),
    ),
    IsoprenolPick(
        store="proteome_carruthers2025",
        task="regression, protein x strain",
        label="absolute abundance per protein and campaign strain",
        split="held-out strains",
        alternates=["CampaignProteomeCarruthers2025"],
        why=(
            "The absolute-scale companion to the campaign proteome, on 21 records. A "
            "calibration check rather than a feature table."
        ),
    ),
    IsoprenolPick(
        store="proteome_fold_change_carruthers2025",
        task="regression, protein x strain",
        label="log2 fold change per protein against the per-cycle control",
        split="held-out strains",
        alternates=["ProteomeFoldChangeLim2025"],
        why=(
            "The campaign proteome as a contrast against the controls the cycles "
            "carry, which is the form a knockdown effect is read in."
        ),
    ),
    IsoprenolPick(
        store="proteome_menasalvas2025",
        task="regression, protein x phase",
        label="summed DIA protein counts per protein, growth or production phase",
        split="held-out phase",
        alternates=["ProteomeCarruthers2025"],
        why=(
            "The phase contrast over 2,090 loci, with the two phases stored as "
            "distinct environments rather than one condition, so the phase is "
            "queryable instead of implied."
        ),
        helps=(
            "Abundances for 100 of the 135 targets split by growth and production phase, "
            "which is the axis the campaign's single endpoint cannot resolve."
        ),
    ),
    IsoprenolPick(
        store="metabolite_growth_menasalvas2025",
        task="regression, metabolite x phase",
        label="intracellular concentration in uM per metabolite, growth phase",
        split="held-out phase",
        alternates=["MetaboliteProductionPhaseMenasalvas2025"],
        why=(
            "Pathway intermediates rather than the product, which is where a flux "
            "model is checked against something other than the titer."
        ),
        helps=(
            "Four shared targets, and pathway intermediates rather than product, which is "
            "where a flux explanation of a knockdown can be checked against something "
            "other than titer."
        ),
    ),
    IsoprenolPick(
        store="metabolite_production_menasalvas2025",
        task="regression, metabolite x phase",
        label="the same pools in production phase, the paired environment",
        split="held-out phase",
        alternates=["MetaboliteGrowthPhaseMenasalvas2025"],
        why=(
            "The phase the titer campaigns actually sample, so it is the arm that "
            "says what the cell looks like while it is making product."
        ),
        helps=(
            "The same four targets in the phase the campaign actually samples, so it is "
            "the closer of the two to the campaign's own condition."
        ),
    ),
]


def render_carruthers_transfer(cache: dict[str, Any]) -> str:
    """What can help the Carruthers campaign, ranked by how much of it it covers.

    Carruthers' own four stores are the thing joined to, so none of them is a row.
    The ranking key is the larger of the two coverage counts, because a dataset is
    useful here in proportion to how much of the campaign's target set it reaches,
    by either route.
    """
    roles = cache["isoprenol_roles"]
    picks = {p.store: p for p in PICKS}
    targets = set(cache["isoprenol"]["campaigns"]["Carruthers 2025"]["crispri_targets"])

    rows = []
    for store, probed in roles.items():
        if store in CARRUTHERS_STORES or not any(probed["roles"].values()):
            continue
        pick = picks[store]
        role = next(k for k in ROLE_ORDER if probed["roles"].get(k))
        perturbed = len(set(probed["genes_perturbed"]) & targets)
        measured = len(set(probed["genes_measured"]) & targets)
        rows.append(
            (
                max(perturbed, measured),
                probed["roles"][role],
                store,
                pick,
                probed,
                role,
                perturbed,
                measured,
            )
        )
    rows.sort(key=lambda r: (-r[0], -r[1]))
    blank = [r[2] for r in rows if not r[3].helps]
    if blank:
        raise RuntimeError(f"transfer rows with no 'helps' text: {blank}")

    cols = (
        r"@{}r >{\raggedright\arraybackslash}p{29mm} r "
        r">{\raggedright\arraybackslash}p{40mm} r r "
        r">{\raggedright\arraybackslash}p{88mm}@{}"
    )
    hdr = (
        r"\# & Dataset & Rec. & What one row gives you & Pert. & Meas. & "
        r"How it helps the campaign \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The served \org{P. putida} isoprenol data that is NOT Carruthers', ranked by
how much of that campaign's target set it reaches. \emph{Rec.} is the dataset's
isoprenol records. \emph{Pert.} and \emph{Meas.} count, of the """
        + str(len(targets))
        + r""" CRISPRi targets
the campaign knocks down, how many this dataset also perturbs and how many it reports a
value for, such as a proteome keyed by locus tag. The two are kept apart because only
the first gives a target a second measurement of its effect; the second gives it a
feature. All four columns are read from a full scan. \emph{What one row gives you} and
\emph{How it helps the campaign} are the pick list in the generating script.}
\label{tab:pcstransfer}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{7}{@{}l}{\footnotesize\emph{Table~\ref{tab:pcstransfer}, continued}}\\
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
    for i, (_, iso, store, pick, probed, role, perturbed, measured) in enumerate(
        rows, start=1
    ):
        lines.append(
            " & ".join(
                [
                    str(i),
                    class_tex(str(probed["dataset_name"]).removesuffix("Dataset")),
                    f"{iso:,}",
                    tex_escape(pick.label),
                    f"{perturbed:,}",
                    f"{measured:,}",
                    tex_escape(pick.helps),
                ]
            )
            + r" \\"
        )
        lines.append(r"\addlinespace[2pt]")
    total = sum(r[1] for r in rows)
    lines.append(r"\midrule")
    lines.append(
        r" & \textbf{All "
        + str(len(rows))
        + r" datasets} & \textbf{"
        + f"{total:,}"
        + r"} & & & & \\"
    )
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


def render_isoprenol_datasets(cache: dict[str, Any]) -> str:
    """The isoprenol short list, in the shape of the E. coli benchmark table.

    Banded by the role isoprenol plays, because the role decides what the dataset can
    be used for: only the product band holds a titer label, and the carbon-source band
    answers the opposite engineering question.
    """
    roles = cache["isoprenol_roles"]
    picks = {p.store: p for p in PICKS}
    rows: list[tuple[str, IsoprenolPick, dict[str, Any]]] = []
    for pick in PICKS:
        probed = roles[pick.store]
        hit = {k: v for k, v in probed["roles"].items() if v}
        rows.append((next(k for k in ROLE_ORDER if k in hit), pick, probed))
    rows.sort(key=lambda r: ROLE_ORDER.index(r[0]))
    missing = sorted(
        s for s, p in roles.items() if any(p["roles"].values()) and s not in picks
    )
    if missing:
        raise RuntimeError(f"isoprenol datasets with no pick entry: {missing}")

    cols = (
        r"@{}r >{\raggedright\arraybackslash}p{27mm} "
        r">{\raggedright\arraybackslash}p{22mm} "
        r">{\raggedright\arraybackslash}p{36mm} r r r r "
        r">{\raggedright\arraybackslash}p{23mm} "
        r">{\raggedright\arraybackslash}p{29mm} "
        r">{\raggedright\arraybackslash}p{64mm}@{}"
    )
    hdr = (
        r"\# & Dataset & Task & Label & Rec. & Iso. & Strains & Env & Split & "
        r"Same task & Why \\"
    )
    head = (
        r"""\begin{landscape}
\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{"""
        + cols
        + r"""}
\caption[]{The served \org{P. putida} isoprenol data as a modeling short list, banded
by the role isoprenol plays in each store. \emph{Rec.}, \emph{Iso.}, \emph{Strains}
and \emph{Env} are read from a full scan: records in the store, records in which
isoprenol plays that role, distinct genotypes among those records, and distinct stored
environments among them. \emph{Task}, \emph{Label}, \emph{Split}, \emph{Same task}
and \emph{Why} are the pick list in the generating script. A compound is matched on
its InChIKey, so a record spelling it 3-methyl-3-buten-1-ol is found and
DL-3-aminoisobutyric acid is not; the pathway band is the exception, since a cassette
names the product it makes rather than carrying its structure. The 15 served
\org{P. putida} datasets in which isoprenol never appears are not listed. \\emph{Rec.} and
\\emph{Iso.} are summed per band and overall; \\emph{Strains} and \\emph{Env} are
not, since each counts distinct values inside one store and two stores can hold the
same strain or the same condition.}
\label{tab:pcsisodata}\\
\toprule
"""
        + hdr
        + r"""
\midrule
\endfirsthead
\multicolumn{11}{@{}l}{\footnotesize\emph{Table~\ref{tab:pcsisodata}, continued}}\\
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
    lines: list[str] = []
    seen = None
    band_records = 0
    band_isoprenol = 0

    def band_total(role: str) -> str:
        """The subtotal line that closes a band.

        Records and isoprenol records sum; strains and environments do not, because
        each is a count of distinct values within one store and two stores can hold
        the same strain or the same condition.
        """
        return (
            r"\cmidrule(l{2pt}r{2pt}){5-6}"
            + r" & \multicolumn{3}{@{}l}{\emph{"
            + ROLE_TOTAL[role]
            + r"}} & \emph{"
            + f"{band_records:,}"
            + r"} & \emph{"
            + f"{band_isoprenol:,}"
            + r"} & & & & & \\"
        )

    for i, (role, pick, probed) in enumerate(rows, start=1):
        if role != seen:
            if seen is not None:
                lines.append(band_total(seen))
                lines.append(r"\addlinespace[3pt]")
            band_records = 0
            band_isoprenol = 0
            lines.append(
                r"\multicolumn{11}{@{}l}{\textbf{" + ROLE_LABEL[role] + r"}} \\"
            )
            seen = role
        band_records += probed["records"]
        band_isoprenol += probed["roles"][role]
        lines.append(
            " & ".join(
                [
                    str(i),
                    class_tex(str(probed["dataset_name"]).removesuffix("Dataset")),
                    tex_escape(pick.task),
                    tex_escape(pick.label),
                    f"{probed['records']:,}",
                    f"{probed['roles'][role]:,}",
                    f"{probed['strains']:,}",
                    f"{probed['environments']:,}",
                    tex_escape(pick.split),
                    ", ".join(class_tex(a) for a in pick.alternates) or "--",
                    pick.why,
                ]
            )
            + r" \\"
        )
    lines.append(band_total(seen))
    total_records = sum(p["records"] for _, _, p in rows)
    total_isoprenol = sum(p["roles"][r] for r, _, p in rows)
    lines.append(r"\midrule")
    lines.append(
        r" & \multicolumn{3}{@{}l}{\textbf{All "
        + str(len(rows))
        + r" datasets}} & \textbf{"
        + f"{total_records:,}"
        + r"} & \textbf{"
        + f"{total_isoprenol:,}"
        + r"} & & & & & \\"
    )
    return (
        head + "\n".join(lines) + "\n\\end{longtable}\n\\endgroup\n\\end{landscape}\n"
    )


# ---------------------------------------------------------------------------------
# Gene-level correlates of the Carruthers campaign
# ---------------------------------------------------------------------------------
#
# The transfer table counts how many Carruthers targets another dataset reaches. This
# block asks the next question: over those shared targets, does the other dataset's
# value TRACK the campaign's titer effect? A target set that overlaps but ranks the
# targets in unrelated order transfers nothing.

CARRUTHERS_TITER = "isoprenol_titer_carruthers2025"
YUNUS_TITER = "isoprenol_titer_yunus2026"
YUNUS_KNOCKDOWN = "crispri_knockdown_yunus2026"

# Every served non-Carruthers store whose phenotype is keyed by gene, either as a
# per-gene dict on one record (a proteome, a transcriptome) or as one scalar per
# single-gene record (a transposon screen). The readout text is what the table says
# one value is; each claim is checked against the record shape when the store is read.
CORRELATE_STORES: dict[str, str] = {
    "rbtnseq_fitness_browser_putida": "log2 transposon fitness per gene, one vector per experiment",
    "rbtnseq_borchert2024": "log2 transposon fitness per gene, one vector per experiment",
    "rbtnseq_borchert2023": "log2 transposon fitness per gene, one vector per experiment",
    "fitness_contrast_borchert2023": "t statistic of a two-arm fitness contrast, one vector per contrast",
    "pputida_env_metal_tnseq_royet2025": "Tn-seq log2 fold change under a metal, one vector per metal",
    "putida_precise321_lim2022": "RNA-seq TPM, one vector per sample",
    "proteome_lim2025": "absolute Top3 abundance, one vector per lineage",
    "proteome_fold_change_lim2025": "log2 fold change against the parent, one vector per lineage",
    "proteome_menasalvas2025": "summed DIA protein counts, one vector per strain and phase",
    "proteome_banerjee2025": "log2 LFQ intensity, one vector per strain",
    "proteome_desiqueira2025": "Top3 peptide signal, one vector per isolate",
    "proteome_percent_desiqueira2025": "percent of proteome, one vector per isolate",
    "proteome_log10_percent_desiqueira2025": "log10 percent of proteome, one vector per isolate",
    "crispri_panel_proteome_yunus2026": "Top3 signal on a 207-protein panel, one vector per CRISPRi strain",
    "crispri_differential_proteome_yunus2026": "fold change against the control, one knockdown strain",
}
GENE_DICT_FIELDS = ("protein_abundance", "protein_fold_change", "expression_tpm")
GENE_SCALAR_LABELS = ("environment_response", "t_statistic")

# Below this many shared targets a rank correlation is not reported: with 15 targets
# a rho of 0.5 is already p of about 0.06, and the column would mostly print noise.
MIN_SHARED_TARGETS = 15
# Permutations for the best-of-many null. A dataset with 310 conditions offers 310
# chances at a large rho, so its best one is compared against the best one under
# shuffled target labels, not against the single-test p.
N_PERMUTATIONS = 2000
PERMUTATION_SEED = 0
# Ridge penalty for the all-strain additive fit. Small: it only keeps the 135-column
# indicator design solvable where a target appears once.
ADDITIVE_RIDGE = 1.0


class CorrelateRow(BaseModel):
    """One served store tested against the Carruthers per-target effect."""

    store: str
    readout: str
    n_conditions: int
    n_tested: int
    shared_targets: int = Field(description="median shared targets over tested vectors")
    rho: float | None = Field(description="best rho against the DBTL0 effect")
    p: float | None
    family_p: float | None = Field(
        description="permutation p of the best over all vectors"
    )
    rho_additive: float | None = Field(
        description="the same vector against the all-strain fit"
    )
    best_condition: str | None


def gene_vectors(store_dir: Path) -> dict[str, dict[str, float]]:
    """Every per-condition gene vector a store holds.

    A record whose phenotype is a gene-keyed dict is one vector. A record that holds
    one scalar for one perturbed gene contributes that gene's entry to the vector of
    its condition, and the condition is the screen id, else the contrast label, else
    the interned environment reference, in that order: a compendium runs many
    experiments in one medium, so the environment alone would merge replicates of
    different experiments.
    """
    table = interned_table(store_dir)
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )
    vectors: dict[str, dict[str, float]] = {}
    with env.begin() as txn:
        for n, (_, value) in enumerate(txn.cursor()):
            rec = pickle.loads(value)
            experiment = rec["experiment"]
            pheno = splice(experiment["phenotype"], table)
            field = next(
                (f for f in GENE_DICT_FIELDS if isinstance(pheno.get(f), dict)), None
            )
            if field is not None:
                vectors[f"record {n}"] = {
                    g: float(v) for g, v in pheno[field].items() if v is not None
                }
                continue
            label = pheno["label_name"]
            if label not in GENE_SCALAR_LABELS or pheno.get(label) is None:
                continue
            perts = splice(experiment["genotype"], table)["perturbations"]
            if len(perts) != 1:
                continue
            raw_env = experiment["environment"]
            key = (
                pheno.get("screen_id")
                or pheno.get("contrast_label")
                or raw_env.get("$ref")
                or stable_hash(splice(raw_env, table))
            )
            vectors.setdefault(key, {})[perts[0]["systematic_gene_name"]] = float(
                pheno[label]
            )
    env.close()
    return vectors


def isoprenol_conditions(store_dir: Path) -> set[str]:
    """The screen ids of a compendium's experiments grown on isoprenol, by InChIKey."""
    table = interned_table(store_dir)
    refs = {
        digest
        for digest, const in table.items()
        if isinstance(const, dict)
        and "media" in const
        and any(
            ISOPRENOL_INCHIKEY
            in str(((p.get("agent") or p.get("compound") or {}).get("inchikey")) or "")
            for p in const.get("perturbations") or []
        )
    }
    env = lmdb.open(
        str(store_dir / "processed" / "lmdb"), readonly=True, lock=False, subdir=True
    )
    out: set[str] = set()
    with env.begin() as txn:
        for _, value in txn.cursor():
            experiment = pickle.loads(value)["experiment"]
            if experiment["environment"].get("$ref") in refs:
                out.add(splice(experiment["phenotype"], table)["screen_id"])
    env.close()
    return out


def carruthers_effects(base: Path) -> dict[str, Any]:
    """The campaign's per-target titer effect, estimated two ways.

    The titer store holds 502 strains: single knockdowns and combinations of two to
    four, and no unedited control. The PRIMARY estimate is the single-knockdown
    titer, the mean where a target was built twice. Its cycle is read off the raw
    mirror by target set, and every single knockdown that assigns lands in DBTL0, so
    it is a within-round estimate with no batch term to fit. The SECONDARY estimate
    is an additive ridge fit over all 502 strains, one indicator per target plus one
    for a deletion and one for a gene addition. It has no cycle term, because a third
    of the records match more than one cycle on target set and assigning those would
    be a guess, so it is reported as a robustness check rather than as the answer.
    """
    table = interned_table(base / CARRUTHERS_TITER)
    env = lmdb.open(
        str(base / CARRUTHERS_TITER / "processed" / "lmdb"),
        readonly=True,
        lock=False,
        subdir=True,
    )
    strains: list[dict[str, Any]] = []
    with env.begin() as txn:
        for _, value in txn.cursor():
            experiment = splice(pickle.loads(value), table)["experiment"]
            perts = experiment["genotype"]["perturbations"]
            pheno = experiment["phenotype"]
            strains.append(
                {
                    "targets": sorted(
                        p["systematic_gene_name"]
                        for p in perts
                        if p["perturbation_type"] == "crispr_interference"
                    ),
                    "other": sorted(
                        {p["perturbation_type"] for p in perts}
                        - {"crispr_interference", "heterologous_pathway"}
                    ),
                    "titer": pheno["titer"],
                    "se": pheno["titer_se"],
                }
            )
    env.close()

    load_dotenv(REPO / ".env")
    with (Path(os.environ["DATA_ROOT"]) / RAW_CYCLE_CSV).open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        raw = list(csv.DictReader(handle))
    set_cycles: dict[tuple[str, ...], set[int]] = {}
    for row in raw:
        key = tuple(sorted(re.findall(r"PP_\d+", row["Genes_targeted_for_CRISPRi"])))
        set_cycles.setdefault(key, set()).add(int(row["DBTL_Cycle"]))

    single = [s for s in strains if len(s["targets"]) == 1 and not s["other"]]
    single_cycles: dict[str, int] = {}
    for s in single:
        cycles = set_cycles.get(tuple(s["targets"]), set())
        label = f"DBTL{min(cycles)}" if len(cycles) == 1 else "unassigned or several"
        single_cycles[label] = single_cycles.get(label, 0) + 1
    by_target: dict[str, list[float]] = {}
    for s in single:
        by_target.setdefault(s["targets"][0], []).append(s["titer"])
    primary = {g: float(np.mean(v)) for g, v in by_target.items()}

    titers = np.array([s["titer"] for s in single])
    ses = np.array([s["se"] for s in single])
    reliability = 1 - float(np.mean(ses**2)) / float(np.var(titers, ddof=1))

    genes = sorted({g for s in strains for g in s["targets"]})
    col = {g: i for i, g in enumerate(genes)}
    design = np.zeros((len(strains), len(genes) + 3))
    design[:, 0] = 1
    for i, s in enumerate(strains):
        for g in s["targets"]:
            design[i, 1 + col[g]] = 1
        design[i, -2] = "deletion_strain" in s["other"]
        design[i, -1] = "gene_addition" in s["other"]
    penalty = ADDITIVE_RIDGE * np.eye(design.shape[1])
    penalty[0, 0] = 0
    beta = np.linalg.solve(
        design.T @ design + penalty, design.T @ np.array([s["titer"] for s in strains])
    )
    additive = {g: float(beta[1 + col[g]]) for g in genes}

    return {
        "n_strains": len(strains),
        "n_single": len(single),
        "single_cycles": single_cycles,
        "n_targets_single": len(primary),
        "n_targets_all": len(additive),
        "single_titer_sd": float(np.std(titers, ddof=1)),
        "single_mean_se": float(np.mean(ses)),
        "reliability": reliability,
        "primary": primary,
        "additive": additive,
    }


def best_correlate(
    effect: dict[str, float],
    vectors: dict[str, dict[str, float]],
    perms: np.ndarray,
    genes: list[str],
) -> tuple[list[tuple[float, float, int, str]], np.ndarray]:
    """Spearman of the effect against each vector, and the null of the best |rho|.

    The null permutes the effect over ALL its targets and then takes each vector's
    shared subset, so a vector covering fewer targets is held to its own null rather
    than to a pooled one.
    """
    idx = {g: i for i, g in enumerate(genes)}
    e = np.array([effect[g] for g in genes])
    shuffled = e[perms]
    results: list[tuple[float, float, int, str]] = []
    null_best = np.zeros(len(perms))
    for key, vec in vectors.items():
        shared = [g for g in genes if g in vec]
        if len(shared) < MIN_SHARED_TARGETS:
            continue
        y = np.array([vec[g] for g in shared])
        if np.ptp(y) == 0:
            continue
        ix = np.array([idx[g] for g in shared])
        rho, p = spearmanr(e[ix], y)
        ry = rankdata(y)
        ry = (ry - ry.mean()) / ry.std()
        rp = np.apply_along_axis(rankdata, 1, shuffled[:, ix])
        rp = (rp - rp.mean(1, keepdims=True)) / rp.std(1, keepdims=True)
        null_best = np.maximum(null_best, np.abs((rp * ry).mean(1)))
        results.append((float(rho), float(p), len(shared), key))
    results.sort(key=lambda r: -abs(r[0]))
    return results, null_best


def run_probe_correlates() -> dict[str, Any]:
    """Test every served gene-level readout against the Carruthers per-target effect."""
    base = data_root()
    out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    print("estimating the Carruthers per-target effect ...", flush=True)
    effects = carruthers_effects(base)
    primary, additive = effects["primary"], effects["additive"]

    stores: dict[str, tuple[str, dict[str, dict[str, float]]]] = {}
    yunus: dict[str, float] = {}
    table = interned_table(base / YUNUS_TITER)
    env = lmdb.open(
        str(base / YUNUS_TITER / "processed" / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        for _, value in txn.cursor():
            experiment = splice(pickle.loads(value), table)["experiment"]
            (gene,) = [
                p["systematic_gene_name"]
                for p in experiment["genotype"]["perturbations"]
                if p["perturbation_type"] == "crispr_interference"
            ]
            yunus[gene] = experiment["phenotype"]["titer"]
    env.close()
    stores[YUNUS_TITER] = ("isoprenol titer per CRISPRi strain", {"titer": yunus})

    knockdown: dict[str, float] = {}
    table = interned_table(base / YUNUS_KNOCKDOWN)
    env = lmdb.open(
        str(base / YUNUS_KNOCKDOWN / "processed" / "lmdb"), readonly=True, lock=False
    )
    with env.begin() as txn:
        for _, value in txn.cursor():
            pheno = splice(pickle.loads(value), table)["experiment"]["phenotype"]
            knockdown.update(pheno["protein_fold_change"])
    env.close()
    stores[YUNUS_KNOCKDOWN] = (
        "the target's own protein against the non-targeting control",
        {"depth": knockdown},
    )
    for store, readout in CORRELATE_STORES.items():
        print(f"reading gene vectors {store} ...", flush=True)
        stores[store] = (readout, gene_vectors(base / store))

    rng = np.random.default_rng(PERMUTATION_SEED)
    p_genes, a_genes = sorted(primary), sorted(additive)
    p_perms = np.array([rng.permutation(len(p_genes)) for _ in range(N_PERMUTATIONS)])
    a_perms = np.array([rng.permutation(len(a_genes)) for _ in range(N_PERMUTATIONS)])
    rows: list[CorrelateRow] = []
    for store, (readout, vectors) in stores.items():
        res, null = best_correlate(primary, vectors, p_perms, p_genes)
        res_a, _ = best_correlate(additive, vectors, a_perms, a_genes)
        by_key_a = {r[3]: r[0] for r in res_a}
        shared = [len(set(primary) & set(v)) for v in vectors.values()]
        top = res[0] if res else None
        rows.append(
            CorrelateRow(
                store=store,
                readout=readout,
                n_conditions=len(vectors),
                n_tested=len(res),
                shared_targets=int(np.median([r[2] for r in res]))
                if res
                else max(shared, default=0),
                rho=top[0] if top else None,
                p=top[1] if top else None,
                family_p=float((null >= abs(top[0])).mean()) if top else None,
                rho_additive=by_key_a.get(top[3]) if top else None,
                best_condition=top[3] if top else None,
            )
        )
        print(f"  {rows[-1].model_dump_json()}", flush=True)

    # The compendia's isoprenol arms, by name, since the best-of-many row hides them.
    isoprenol_arms = []
    for store in COMPENDIA_STORES:
        vectors = stores[store][1]
        for screen in sorted(isoprenol_conditions(base / store)):
            res, _ = best_correlate(
                primary, {screen: vectors[screen]}, p_perms, p_genes
            )
            isoprenol_arms.append(
                {
                    "store": store,
                    "screen": screen,
                    "rho": res[0][0],
                    "p": res[0][1],
                    "n": res[0][2],
                }
            )

    # Aggregate gene priors: each is one vector, so each is one test.
    priors = []

    def prior(name: str, vec: dict[str, float]) -> None:
        shared = sorted(set(primary) & set(vec))
        rho, p = spearmanr([primary[g] for g in shared], [vec[g] for g in shared])
        rho_a, _ = spearmanr(
            [additive[g] for g in sorted(set(additive) & set(vec))],
            [vec[g] for g in sorted(set(additive) & set(vec))],
        )
        priors.append(
            {
                "name": name,
                "n": len(shared),
                "rho": float(rho),
                "p": float(p),
                "rho_additive": float(rho_a),
            }
        )

    for store in COMPENDIA_STORES:
        vectors = stores[store][1]
        values: dict[str, list[float]] = {}
        for vec in vectors.values():
            for g, v in vec.items():
                values.setdefault(g, []).append(v)
        label = class_name(out, store)
        prior(
            f"{label}: mean fitness over all {len(vectors)} experiments",
            {g: float(np.mean(v)) for g, v in values.items()},
        )
        prior(
            f"{label}: share of experiments with |fitness| above 1",
            {g: float(np.mean(np.abs(v) > 1)) for g, v in values.items()},
        )
    precise = stores["putida_precise321_lim2022"][1]
    genes_tpm: dict[str, list[float]] = {}
    for vec in precise.values():
        for g, v in vec.items():
            genes_tpm.setdefault(g, []).append(np.log2(v + 1))
    prior(
        f"PRECISE321: mean log2(TPM + 1) over {len(precise)} samples",
        {g: float(np.mean(v)) for g, v in genes_tpm.items()},
    )

    calls: dict[str, str] = {}
    table = interned_table(base / "graded_gene_essentiality_royet2025")
    env = lmdb.open(
        str(base / "graded_gene_essentiality_royet2025" / "processed" / "lmdb"),
        readonly=True,
        lock=False,
    )
    with env.begin() as txn:
        for _, value in txn.cursor():
            experiment = splice(pickle.loads(value), table)["experiment"]
            gene = experiment["genotype"]["perturbations"][0]["systematic_gene_name"]
            calls[gene] = str(experiment["phenotype"]["essentiality_call"])
    env.close()
    groups: dict[str, list[float]] = {}
    for g, v in primary.items():
        if g in calls:
            groups.setdefault(calls[g], []).append(v)
    essentiality = {
        "groups": {
            k: {"n": len(v), "median": float(np.median(v))}
            for k, v in sorted(groups.items())
        },
        "kruskal_p": float(kruskal(*[v for v in groups.values() if len(v) > 2]).pvalue),
    }

    shared_y = sorted(set(primary) & set(yunus))
    a = np.array([primary[g] for g in shared_y])
    y = np.array([yunus[g] for g in shared_y])
    q = len(shared_y) // 4
    names = np.array(shared_y)
    yunus_detail = {
        "n": len(shared_y),
        "pearson": float(pearsonr(a, y)[0]),
        "carruthers_median": float(np.median(a)),
        "yunus_median": float(np.median(y)),
        "quartile": q,
        "top_overlap": len(
            set(names[np.argsort(-a)[:q]]) & set(names[np.argsort(-y)[:q]])
        ),
        "bottom_overlap": len(
            set(names[np.argsort(a)[:q]]) & set(names[np.argsort(y)[:q]])
        ),
        "chance": q * q / len(shared_y),
    }

    out["correlates"] = {
        "effects": {
            k: v for k, v in effects.items() if k not in ("primary", "additive")
        },
        "rows": [r.model_dump() for r in rows],
        "isoprenol_arms": isoprenol_arms,
        "priors": priors,
        "essentiality": essentiality,
        "yunus": yunus_detail,
        "n_permutations": N_PERMUTATIONS,
        "min_shared": MIN_SHARED_TARGETS,
    }
    _save(out)
    return out


def class_name(cache: dict[str, Any], store: str) -> str:
    """A store's dataset class, read off the roles probe rather than restated here."""
    name: str = cache["isoprenol_roles"][store]["dataset_name"]
    return name.removesuffix("Dataset")


def signed(x: float) -> str:
    """A correlation in math mode, so its minus is a minus and a column aligns."""
    # A value that rounds to zero prints unsigned rather than as -0.00.
    if round(x, 2) == 0:
        return "$0.00$"
    return f"${x:+.2f}$"


def p_tex(p: float) -> str:
    """A p-value at two significant figures, in scientific form below 0.001."""
    if p < 0.001:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"${mantissa}\\times 10^{{{int(exponent)}}}$"
    return f"{p:.2g}"


def render_correlates(cache: dict[str, Any]) -> str:
    """Whether another served gene-level readout tracks the Carruthers target effect."""
    c = cache["correlates"]
    fx = c["effects"]
    rows = sorted(
        c["rows"],
        key=lambda r: (r["family_p"] is None, r["family_p"] or 0, -abs(r["rho"] or 0)),
    )
    head = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}L{40mm} L{54mm} r r r r r r@{}}",
        r"\caption[]{Does another served dataset's gene-level value track the "
        r"Carruthers per-target titer effect? The effect is the DBTL0 "
        r"single-knockdown titer, "
        + f"{fx['n_targets_single']} targets"
        + r". Each dataset is a set of gene vectors, one per experiment, sample or "
        r"strain. \emph{Vec.} is the number of vectors tested out of the total, "
        r"where a vector is tested when it shares at least "
        + f"{c['min_shared']}"
        + r" targets. \emph{Shared} is the median shared targets. "
        r"$\rho$ and \emph{p} are the Spearman correlation and its single-test "
        r"p-value for the best vector. \emph{Best-of p} compares that best $\rho$ "
        r"against the best under "
        + f"{c['n_permutations']:,}"
        + r" shuffles of the target labels, which is the honest p when a dataset "
        r"offers hundreds of vectors. \emph{All} is the same vector against the "
        r"additive fit over all "
        + f"{fx['n_strains']}"
        + r" strains. A dash means no vector shares enough targets.}"
        r"\label{tab:pcscorr}\\",
        r"\toprule",
        r"Dataset & One value is & Vec. & Shared & $\rho$ & \emph{p} & Best-of \emph{p} & All \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{8}{@{}l}{\footnotesize\emph{Table~\ref{tab:pcscorr}, continued}}\\",
        r"\toprule",
        r"Dataset & One value is & Vec. & Shared & $\rho$ & \emph{p} & Best-of \emph{p} & All \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    body = []
    for r in rows:
        name = class_tex(class_name(cache, r["store"]))
        if r["rho"] is None:
            cells = [
                name,
                tex_escape(r["readout"]),
                f"0/{r['n_conditions']:,}",
                f"{r['shared_targets']}",
                "--",
                "--",
                "--",
                "--",
            ]
        else:
            cells = [
                name,
                tex_escape(r["readout"]),
                f"{r['n_tested']:,}/{r['n_conditions']:,}",
                f"{r['shared_targets']}",
                signed(r["rho"]),
                p_tex(r["p"]),
                f"{r['family_p']:.3f}",
                signed(r["rho_additive"]) if r["rho_additive"] is not None else "--",
            ]
        body.append(" & ".join(cells) + r" \\")
    first = "\n".join(head + body + [r"\end{longtable}", r"\endgroup"])

    ess = c["essentiality"]
    single = [
        r"\begingroup",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{@{}L{112mm} r r r r@{}}",
        r"\caption[]{Single tests against the same effect: the compendia's two "
        r"growth-on-isoprenol experiments, named because the best-of row in "
        r"Table~\ref{tab:pcscorr} hides them, and aggregate gene priors, each one "
        r"vector. The priors were not chosen in advance and the two compendia share "
        r"most of their experiments, so their p-values are not corrected and are "
        r"not independent.}\label{tab:pcspriors}\\",
        r"\toprule",
        r"Test & Shared & $\rho$ & \emph{p} & All \\",
        r"\midrule",
        r"\endfirsthead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for arm in c["isoprenol_arms"]:
        single.append(
            " & ".join(
                [
                    class_tex(class_name(cache, arm["store"]))
                    + ": growth on isoprenol, "
                    + breakable(arm["screen"]),
                    f"{arm['n']}",
                    signed(arm["rho"]),
                    p_tex(arm["p"]),
                    "",
                ]
            )
            + r" \\"
        )
    single.append(r"\addlinespace[2pt]")
    for pr in c["priors"]:
        single.append(
            " & ".join(
                [
                    tex_escape(pr["name"]).replace("|fitness|", r"$|$fitness$|$"),
                    f"{pr['n']}",
                    signed(pr["rho"]),
                    p_tex(pr["p"]),
                    signed(pr["rho_additive"]),
                ]
            )
            + r" \\"
        )
    # The four TRANSIT HMM states, spelled as the schema's EssentialityCall names them.
    call_names = {
        "ES": "essential",
        "GD": "growth defect",
        "NE": "nonessential",
        "GA": "growth advantage",
    }
    groups = ", ".join(
        f"{call_names[k]} {v['median']:.0f} (n {v['n']})"
        for k, v in ess["groups"].items()
    )
    single.append(r"\addlinespace[2pt]")
    single.append(
        r"Royet 2025 essentiality call: median DBTL0 titer in ug/mL by class, "
        + groups
        + r"; Kruskal--Wallis & & & "
        + p_tex(ess["kruskal_p"])
        + r" & \\"
    )
    second = "\n".join(single + [r"\end{longtable}", r"\endgroup"])
    return first + "\n\n" + second + "\n"


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
    parser.add_argument(
        "--probe-roles",
        action="store_true",
        help="ask every served P. putida store what role isoprenol plays in it",
    )
    parser.add_argument(
        "--probe-correlates",
        action="store_true",
        help="test every served gene-level readout against the Carruthers target effect",
    )
    args = parser.parse_args()

    if args.probe:
        run_probe()
    if args.probe_isoprenol:
        run_probe_isoprenol()
    if args.probe_roles:
        run_probe_roles()
    if args.probe_correlates:
        run_probe_correlates()
    cache = json.loads(CACHE.read_text())

    write(TEX_DIR / "candidates.tex", render_candidates(cache))
    write(TEX_DIR / "environment.tex", render_environment(cache))
    write(TEX_DIR / "counts.tex", render_counts(cache))
    write(TEX_DIR / "isoprenol.tex", render_isoprenol(cache))
    write(TEX_DIR / "cycle-label.tex", render_cycle_label(cache))
    write(TEX_DIR / "carruthers-transfer.tex", render_carruthers_transfer(cache))
    write(TEX_DIR / "isoprenol-datasets.tex", render_isoprenol_datasets(cache))
    write(TEX_DIR / "correlates.tex", render_correlates(cache))


if __name__ == "__main__":
    main()
