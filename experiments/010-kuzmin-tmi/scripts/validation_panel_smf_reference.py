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

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from torchcell.datasets.scerevisiae import SmfCostanzo2016Dataset, SmfKuzmin2018Dataset
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
    return pd.DataFrame([rows[orf] for _, orf in PANEL])


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
