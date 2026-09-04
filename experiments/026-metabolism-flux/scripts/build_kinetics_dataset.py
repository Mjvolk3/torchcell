# experiments/026-metabolism-flux/scripts/build_kinetics_dataset.py
# [[experiments.026-metabolism-flux.scripts.build_kinetics_dataset]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/build_kinetics_dataset.py

r"""Run one kinetic predictor over every (enzyme, substrate) pair in yeast-GEM.

Assembles the pair table -- catalytic unit, ORF, UniProt accession, protein sequence,
substrate metabolite and its SMILES -- then hands blocks of it to the named predictor and
writes a parquet keyed on the pair. The build is cached on the ORF permanently, so a
rerun recomputes nothing already present.

The protein sequence comes from the genome object, via ``embed_prot_t5.py
--dump-sequences``, so a predictor sees the same protein the perturbation ontology edits
rather than one bundled in a third-party table.

    python experiments/026-metabolism-flux/scripts/build_kinetics_dataset.py \
        --predictor dlkcat --device cuda:0
"""

import argparse
import json
import os
import os.path as osp
from datetime import UTC, datetime
from typing import cast

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from torchcell.data.kinetics import KineticParameter, KineticsDatasetSummary
from torchcell.datasets.kinetics import KINETICS_CONFIGS

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
GEM_DIR = osp.join(DATA_ROOT, "data", "torchcell", "yeast-GEM", "yeast-GEM-9.0.2")
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")
KINETICS_ROOT = osp.join(DATA_ROOT, "data", "torchcell", "kinetics")
GENOME_SEQUENCES = osp.join(KINETICS_ROOT, "_features", "gem_protein_sequences.json")


def read_sequences(path: str) -> dict[str, str]:
    """Systematic ORF name -> protein sequence, as dumped from the genome object.

    The sequences come from ``SCerevisiaeGenome`` via ``embed_prot_t5.py
    --dump-sequences``, NOT from the ``swissprot.tsv`` yeast-GEM bundles. Everything in
    torchcell derives from the S288C reference FASTA plus the GAF, so a predictor has to
    see the same protein the perturbation ontology edits. All 1,161 GEM genes resolve to a
    current R64 ORF with a protein sequence, so a miss here is a real inconsistency rather
    than an expected gap.
    """
    with open(path) as handle:
        return {k.upper(): v for k, v in json.load(handle).items()}


def load_pairs(inputs_csv: str, sequences: dict[str, str]) -> pd.DataFrame:
    """The predictor input table, with sequences joined on and unresolved rows dropped."""
    pairs = pd.read_csv(inputs_csv)
    pairs["sequence"] = pairs["gene_id"].str.upper().map(sequences)
    missing = int(pairs["sequence"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing} pairs have no protein sequence. The genome resolves all 1,161 "
            "GEM genes, so a miss means the pair table and the genome disagree."
        )
    return pairs


def main() -> None:
    """Build one predictor's parameter tables and write a coverage summary beside them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor", required=True, choices=sorted(KINETICS_CONFIGS))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test on N pairs.")
    parser.add_argument("--block-size", type=int, default=500)
    args = parser.parse_args()

    sequences = read_sequences(GENOME_SEQUENCES)
    pairs = load_pairs(osp.join(RESULTS, "kinetics_predictor_inputs.csv"), sequences)
    # The denominator is the GEM's catalytic units, NOT the units present in the input
    # file. The input only holds units whose substrate resolved to a SMILES, so scoring
    # against it reports 100% coverage while 176 units have no prediction at all.
    with open(osp.join(RESULTS, "gem_audit.json")) as handle:
        n_units_total = int(json.load(handle)["gem"]["n_catalytic_units"])
    if args.limit:
        pairs = pairs.head(args.limit)
    print(f"{len(pairs)} pairs, {pairs['unit_id'].nunique()} catalytic units", flush=True)

    dataset = KINETICS_CONFIGS[args.predictor](
        root=KINETICS_ROOT, inputs=pairs, data_root=DATA_ROOT, device=args.device
    )

    blocks = []
    for start in range(0, len(pairs), args.block_size):
        block = pairs.iloc[start : start + args.block_size]
        blocks.append(dataset.predict_batch(block))
        done = min(start + args.block_size, len(pairs))
        print(f"  {done}/{len(pairs)}", flush=True)
    predictions = pd.concat(blocks) if blocks else pd.DataFrame()
    table = pairs.join(predictions)

    weights_sha256 = dataset._weights_sha256
    table["predictor"] = args.predictor
    table["weights_sha256"] = weights_sha256
    table["provenance"] = "predicted"

    os.makedirs(dataset.processed_dir, exist_ok=True)
    for parameter in dataset.EMITS:
        column = parameter.value
        ok = table[table[column].notna()]
        out_path = dataset.processed_path(parameter)
        ok.to_parquet(out_path, index=False)

        failed = table[table[column].isna()]
        reasons = (
            failed["failure"].value_counts().to_dict() if "failure" in failed else {}
        )
        summary = KineticsDatasetSummary(
            predictor=args.predictor,
            parameter=parameter,
            n_rows=len(ok),
            n_units=int(ok["unit_id"].nunique()),
            n_units_total=n_units_total,
            n_genes=int(ok["gene_id"].nunique()),
            n_substrates=int(ok["substrate_met_id"].nunique()),
            coverage_frac=float(ok["unit_id"].nunique() / n_units_total),
            value_median=float(np.median(ok[column])) if len(ok) else float("nan"),
            value_p05=float(np.percentile(ok[column], 5)) if len(ok) else float("nan"),
            value_p95=float(np.percentile(ok[column], 95)) if len(ok) else float("nan"),
            n_failed=len(failed),
            failure_reasons={str(k): int(v) for k, v in reasons.items()},
            weights_sha256=weights_sha256,
            built_at=datetime.now(UTC).isoformat(),
        )
        with open(dataset.summary_path(parameter), "w") as handle:
            handle.write(summary.model_dump_json(indent=2))
        with open(
            osp.join(RESULTS, f"kinetics_{args.predictor}_{column}_summary.json"), "w"
        ) as handle:
            handle.write(summary.model_dump_json(indent=2))
        print(json.dumps(json.loads(summary.model_dump_json()), indent=2))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
