# experiments/010-kuzmin-tmi/scripts/validation_panel_smf_reference.py
# [[experiments.010-kuzmin-tmi.scripts.validation_panel_smf_reference]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/010-kuzmin-tmi/scripts/validation_panel_smf_reference
"""Single-mutant-fitness (SMF) reference table for the droplet validation panel.

For each mutant in the droplet-assay panel this queries fitness ± s.d. from the
three SMF datasets (Costanzo2016, Kuzmin2018, Kuzmin2020) so the wet-lab droplet
measurements have a public-data SMF comparison column. Unlike the panel-12
inference tables, this list includes YLR313C/SPH1 (the authentic gene) and
YPL056C/LCL1, which the inference roster never queried.

Values are looked up from the small Smf LMDB datasets (~20k rows each, ~1 s),
keyed by the perturbed systematic gene name — the same mechanism as
investigate_YLR313C_smf_and_interactions.py. A gene absent from a study is
written as an empty cell (honest missing, never guessed).

Run from repo root:
  ~/miniconda3/envs/torchcell/bin/python \
    experiments/010-kuzmin-tmi/scripts/validation_panel_smf_reference.py

Output:
  results/validation_panel_smf_costanzo_kuzmin.csv
"""
import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datamodels.schema import UncertaintyType, derive_se
from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset, SmfKuzmin2018Dataset
from torchcell.datasets.scerevisiae.costanzo2016 import N_SAMPLES_QUERY_SMF_SCREENS
from torchcell.datasets.scerevisiae.kuzmin2020 import (
    N_SAMPLES_COMBINED_MUTANT as N_KUZMIN2020,
)
from torchcell.datasets.scerevisiae.kuzmin2020 import SmfKuzmin2020Dataset

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]

RESULTS_DIR = osp.join(EXPERIMENT_ROOT, "010-kuzmin-tmi", "results")

# Droplet validation panel: (common name, systematic ORF), in the wet-lab table's
# row order. Common name "" where SGD R64 has no standard name.
PANEL: list[tuple[str, str]] = [
    ("SPH1", "YLR313C"),
    ("LCL1", "YPL056C"),
    ("COS111", "YBR203W"),
    ("", "YKL033W-A"),
    ("CBF1", "YJR060W"),
    ("RPS9A", "YPL081W"),
    ("", "YER079W"),
    ("MMS2", "YGL087C"),
    ("YEH1", "YLL012W"),
    ("ELC1", "YPL046C"),
    ("YOS9", "YDR057W"),
    ("", "YLR312C-B"),
]

SMF_CONFIGS = [
    ("SmfCostanzo2016", SmfCostanzo2016Dataset, "smf_costanzo2016"),
    ("SmfKuzmin2018", SmfKuzmin2018Dataset, "smf_kuzmin2018"),
    ("SmfKuzmin2020", SmfKuzmin2020Dataset, "smf_kuzmin2020"),
]

# What each source's SMF uncertainty IS, from the loaders' sourced constants.
# These are NOT all the same statistic -- naming the type is what makes the number
# comparable to a wet-lab assay SD:
#   Costanzo2016 -> BOOTSTRAP SE over control screens (already an SE; never /sqrt(n)).
#     Costanzo SI si1.md line 94: "...bootstrapped means, instead of medians, across
#     replicates were used in variance estimation and final fitness values."
#   Kuzmin2018   -> loader stores NO single-mutant uncertainty (fitness_std=None);
#     the "12-24 colony" figure in their SI is the query-fitness column, not this one.
#   Kuzmin2020   -> sample SD over colony replicates -> SE = SD/sqrt(n).
SMF_DESIGN: dict[str, tuple[UncertaintyType, int, str] | None] = {
    "SmfCostanzo2016": (UncertaintyType.bootstrap_se, N_SAMPLES_QUERY_SMF_SCREENS, "screen"),
    "SmfKuzmin2018": None,
    "SmfKuzmin2020": (UncertaintyType.sample_sd, N_KUZMIN2020, "colony"),
}


def build_single_index(dataset, name: str) -> dict:
    """frozenset({gene}) -> {fitness, fitness_std, strain_id} for single KOs."""
    index = {}
    for i in tqdm(range(len(dataset)), desc=f"Indexing {name}"):
        item = dataset[i]
        perts = item["experiment"]["genotype"]["perturbations"]
        genes = frozenset(p["systematic_gene_name"] for p in perts)
        ph = item["experiment"]["phenotype"]
        index[genes] = {
            "fitness": ph["fitness"],
            "fitness_std": ph["fitness_std"],
            "strain_id": perts[0]["strain_id"],
        }
    return index


def query_panel() -> pd.DataFrame:
    genes = [orf for _, orf in PANEL]
    rows = {orf: {"common_name": common, "ORF": orf} for common, orf in PANEL}
    for name, cls, subdir in SMF_CONFIGS:
        print(f"\nProcessing {name}...")
        ds = cls(root=osp.join(DATA_ROOT, f"data/torchcell/{subdir}"), io_workers=4)
        idx = build_single_index(ds, name)
        for g in genes:
            data = idx.get(frozenset([g]))
            rows[g][f"{name}_fitness"] = data["fitness"] if data else None
            rows[g][f"{name}_std"] = data["fitness_std"] if data else None
            rows[g][f"{name}_strain_id"] = data["strain_id"] if data else None
        hits = sum(rows[g][f"{name}_fitness"] is not None for g in genes)
        print(f"  Matches found: {hits}/{len(genes)}")
    df = pd.DataFrame([rows[orf] for _, orf in PANEL])
    return annotate_uncertainty(df)


def annotate_uncertainty(df: pd.DataFrame) -> pd.DataFrame:
    """Say what each reported uncertainty IS, and derive the SE accordingly.

    Uses the schema's own ``derive_se`` so the conversion cannot drift from the
    uncertainty ontology: bootstrap_se is used as-is, sample_sd is divided by
    sqrt(n). A source that stores no uncertainty (Kuzmin2018) gets empty columns
    rather than a fabricated type.
    """
    for src, design in SMF_DESIGN.items():
        std_col = f"{src}_std"
        if std_col not in df.columns:
            continue
        if design is None:
            df[f"{src}_uncertainty_type"] = None
            df[f"{src}_n_samples"] = np.nan
            df[f"{src}_sample_unit"] = None
            df[f"{src}_se"] = np.nan
            continue
        unc_type, n, unit = design
        has = df[std_col].notna()
        df[f"{src}_uncertainty_type"] = np.where(has, str(unc_type), None)
        df[f"{src}_n_samples"] = np.where(has, n, np.nan)
        df[f"{src}_sample_unit"] = np.where(has, unit, None)
        df[f"{src}_se"] = [
            derive_se(v, unc_type, n) if pd.notna(v) else np.nan for v in df[std_col]
        ]
    return df


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(r, name):
        f, s = r[f"{name}_fitness"], r[f"{name}_std"]
        if pd.isna(f):
            return "—"
        return f"{float(f):.3f} ± {float(s):.3f}" if not pd.isna(s) else f"{float(f):.3f}"

    lines = [
        "| Common | ORF | Costanzo2016 SMF ± SD | Kuzmin2018 SMF ± SD | Kuzmin2020 SMF ± SD |",
        "|--------|-----|-----------------------|---------------------|---------------------|",
    ]
    for _, r in df.iterrows():
        common = r["common_name"] if r["common_name"] else "—"
        lines.append(
            f"| {common} | {r['ORF']} | {fmt(r, 'SmfCostanzo2016')} | "
            f"{fmt(r, 'SmfKuzmin2018')} | {fmt(r, 'SmfKuzmin2020')} |"
        )
    return "\n".join(lines)


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = query_panel()
    out = osp.join(RESULTS_DIR, "validation_panel_smf_costanzo_kuzmin.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}\n")
    print(markdown_table(df))


if __name__ == "__main__":
    main()
