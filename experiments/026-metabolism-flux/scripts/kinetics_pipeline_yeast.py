# experiments/026-metabolism-flux/scripts/kinetics_pipeline_yeast.py
# [[experiments.026-metabolism-flux.scripts.kinetics_pipeline_yeast]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/kinetics_pipeline_yeast.py

r"""Run the portable kinetics pipeline on yeast-GEM 9.0.2, as its test case.

:mod:`torchcell.metabolism.kinetics_pipeline` takes a GEM, a proteome and a set of mirrors
and knows nothing about yeast. This script is the GEM-specific glue that says where yeast's
copies of those things are, and nothing else: it writes the accession map out of the
``swissprot.tsv`` yeast-GEM ships, points the config at the model's SBML, the genome's
protein dump, the MetaNetX mirror, the lipid writer's structures, the Open Enzyme Database
slice and the predictor parquets, then runs it.

Porting to another organism replaces this file, not the module. What has to exist:

======================  =================================================================
input                   what it is for another organism
======================  =================================================================
``sbml_path``           the GEM being parameterized
``proteome_path``       a reference proteome FASTA, or gene -> sequence JSON
``accessions_path``     gene -> database accession, only for the accession-keyed tier
``chem_prop/chem_xref`` the structure database and its cross-reference, organism-free
``measured_mirror``     a kinetics database slice, or nothing at all
``predictor_tables``    whatever sequence predictors were run for that organism
======================  =================================================================

``--measured-key`` selects the matcher. Compare ``uniprot`` (the flux layer's current key)
against ``ec_substrate_organism`` and ``ec_substrate_any_organism`` to see what an organism
with no protein-level measurements would still get.

Run from the worktree root::

    PYTHONPATH=$PWD ~/miniconda3/envs/torchcell/bin/python \
        experiments/026-metabolism-flux/scripts/kinetics_pipeline_yeast.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp

from dotenv import load_dotenv

from torchcell.metabolism.kinetics_pipeline import (
    KineticsPipelineConfig,
    MeasuredKey,
    PredictorTable,
    resolve_kinetics,
)
from torchcell.metabolism.parameters import (
    KINETICS_MIRROR,
    KINETICS_VERSION,
    PLAUSIBILITY_RESULT,
    SYSTEMATIC_NAME_RE,
)
from torchcell.metabolism.yeast_GEM import YeastGEM

load_dotenv()

DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_ROOT = osp.join(DATA_ROOT, KINETICS_MIRROR)
#: Every predictor whose table exists, per parameter. Which of them a run is allowed to use
#: is decided by the within-class study, not by this list.
PREDICTORS = {
    "k_cat": ["ezkinetics", "eitlem", "unikp", "dlkcat", "turnup", "deepenzyme"],
    "K_M": ["ezkinetics", "eitlem", "unikp", "boost_km"],
}


def write_accession_map(model_dir: str, path: str) -> int:
    """Systematic ORF name -> UniProt accession, from the table yeast-GEM ships.

    The ``gene_id`` column is a whitespace-separated ALIAS LIST of unpredictable length, so
    the systematic name is selected by pattern rather than by position; taking the last
    token matches 430 of 1,161 genes and taking the first matches 38.
    """
    out: dict[str, str] = {}
    with open(osp.join(model_dir, "data", "databases", "swissprot.tsv")) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            accession = (row.get("uniprot") or "").strip()
            if not accession:
                continue
            for token in (row.get("gene_id") or "").split():
                if SYSTEMATIC_NAME_RE.match(token):
                    out.setdefault(token, accession)
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2)
    return len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--measured-key",
        default=MeasuredKey.UNIPROT_SUBSTRATE.value,
        choices=[k.value for k in MeasuredKey],
    )
    parser.add_argument(
        "--no-homolog-tier",
        action="store_true",
        help="Drop the cross-organism EC-plus-substrate stage, leaving one database tier.",
    )
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()
    key = MeasuredKey(arguments.measured_key)

    source = YeastGEM()
    accessions_path = osp.join(RESULTS_DIR, "kinetic_pipeline_yeast_accessions.json")
    n_accessions = write_accession_map(source.model_dir, accessions_path)
    print(f"{n_accessions} gene -> accession pairs")

    # An EC key asks about the reaction's chemistry rather than about this protein, so the
    # organism filter is dropped only for the deliberately cross-organism key.
    organism = None if key is MeasuredKey.EC_SUBSTRATE_ANY_ORGANISM else "Saccharomyces cerevisiae"
    mirror = "all" if organism is None else "scerevisiae"

    config = KineticsPipelineConfig(
        sbml_path=osp.join(source.model_dir, "model", "yeast-GEM.xml"),
        proteome_path=osp.join(
            KINETICS_ROOT, "_features", "gem_protein_sequences.json"
        ),
        accessions_path=accessions_path,
        name_table_path=osp.join(source.model_dir, "data", "databases", "smilesDB.tsv"),
        chem_prop_path=osp.join(DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_prop.tsv"),
        chem_xref_path=osp.join(DATA_ROOT, "data/enzyme_kinetics/metanetx/chem_xref.tsv"),
        supplied_structures_path=osp.join(RESULTS_DIR, "lipid_structures.json"),
        measured_mirror_dir=osp.join(
            DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database", mirror
        ),
        measured_key=key,
        organism=organism,
        homolog_key=(
            None if arguments.no_homolog_tier else MeasuredKey.EC_SUBSTRATE_ANY_ORGANISM
        ),
        homolog_mirror_dir=(
            None
            if arguments.no_homolog_tier
            else osp.join(DATA_ROOT, "data/enzyme_kinetics/open_enzyme_database/all")
        ),
        predictor_tables={
            parameter: [
                PredictorTable(
                    name=name,
                    path=osp.join(
                        KINETICS_ROOT, name, KINETICS_VERSION, f"{parameter}.parquet"
                    ),
                )
                for name in names
                if osp.exists(
                    osp.join(KINETICS_ROOT, name, KINETICS_VERSION, f"{parameter}.parquet")
                )
            ]
            for parameter, names in PREDICTORS.items()
        },
        plausibility_json=osp.join(EXPERIMENT_ROOT, PLAUSIBILITY_RESULT),
    )
    result = resolve_kinetics(config)

    out_path = arguments.out or osp.join(
        RESULTS_DIR, f"kinetic_pipeline_yeast_{key.value}.json"
    )
    with open(out_path, "w") as handle:
        handle.write(result.model_dump_json(indent=2))

    print(f"catalytic units {result.n_catalytic_units}")
    print(f"substrates resolved {result.n_substrates_resolved} of {result.n_metabolites}")
    print(f"structures by route {result.structures_by_route}")
    print(f"input blockers {result.input_blockers}")
    for parameter, block in result.parameters.items():
        print(
            f"{parameter:6} {block.coarse_tier_counts()} "
            f"{block.tier_counts()} predictors {block.selected_predictors} "
            f"default {block.default_value:.4g} {block.unit}"
        )
        print(f"       blockers {block.blocker_counts()}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
