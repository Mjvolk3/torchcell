# experiments/040-inhibitor-synergy-wetlab/scripts/wetlab_table.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.wetlab_table]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/wetlab_table
"""The 2021 Bioscreen wells as one tidy table, read from the private dev LMDB.

One row per served record of ``InhibitorBioscreenVolk2021Dataset`` (977 wells over the
runs ex21, ex23, ex26, ex27, ex28): the run, its length, the dosed compounds (names
exactly as the loader gives them), the dose of each of the six inhibitors in g/L and in
mM, the relative growth rate (``fitness``; 1 = the run's mean wild-type rate) and the
no-growth call. The biological replicate of each well is taken from the loader's own
plate layout (``bioscreen.layout``), which the records do not carry, so the bootstraps of
the downstream scripts resample real replicates.

GROWTH CALL. The primary call is the served one (the loader's raw-curve derivation). For
ex23 the Bioscreen software's generation times (``bioscreen.software_generation_times``,
the source 039 used) are added as a sensitivity call, on their own scale
(``fitness_software`` = mean software WT generation time / the well's). The two calls
disagree on 12 of ex23's 197 wells, all in HMF combinations; they are listed in
``results/wetlab_growth_call_check.csv``.

Writes ``results/wetlab_wells.csv``, ``results/wetlab_growth_call_check.csv`` and
``results/wetlab_provenance.json`` (pydantic), prints the counts by number of compounds and
by run (a well-count mismatch raises), and prints the ex23 condition-level summary under
both calls beside experiment 039's published numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import os.path as osp
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel

from torchcell.datasets.private_torchcell import bioscreen as b
from torchcell.datasets.private_torchcell.volk2021_inhibitor_bioscreen import (
    CONSUMED_SHA256,
    RESOLVER_NAME,
    InhibitorBioscreenVolk2021Dataset,
)

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
DATASET_ROOT = osp.join(
    os.environ["DATA_ROOT"], "data/torchcell/inhibitor_bioscreen_volk2021"
)
RAW_MIRROR = b.raw_mirror_dir(os.environ["DATA_ROOT"])

#: Short column tag per inhibitor, keyed by the compound name the loader serves.
ABBR: dict[str, str] = {RESOLVER_NAME[i]: i.value for i in b.INHIBITORS}
COMPOUNDS: list[str] = [RESOLVER_NAME[i] for i in b.INHIBITORS]
#: Standard formula weights (g/mol) of the six inhibitors, used only to convert the
#: served g/L doses to mM: furfural C5H4O2, acetic acid C2H4O2, 5-HMF C6H6O3, formic
#: acid CH2O2, levulinic acid C5H8O3, lactic acid C3H6O3.
MOLAR_MASS_G_PER_MOL: dict[str, float] = {
    "FF": 96.08,
    "AA": 60.05,
    "HMF": 126.11,
    "FA": 46.03,
    "LVA": 116.12,
    "LA": 90.08,
}

#: The counts this table must reproduce (the task statement and the loader's oracles).
EXPECTED_BY_N_COMPOUNDS = {0: 32, 1: 288, 2: 531, 3: 60, 4: 45, 5: 18, 6: 3}
#: Experiment 039's ex23 summary (notes/experiments.039-inhibitor-combinations-wetlab,
#: computed there from the Bioscreen SOFTWARE's generation times): per number of
#: inhibitors, (combinations, combinations with at least one grown well, mean fitness of
#: the grown combinations, each the mean of its grown wells).
EXPECTED_039_EX23 = {
    1: (6, 6, 0.788),
    2: (15, 14, 0.465),
    3: (20, 5, 0.461),
    4: (15, 0, None),
    5: (6, 0, None),
    6: (1, 0, None),
}


class WetlabProvenance(BaseModel):
    """Where ``wetlab_wells.csv`` came from: the store, its source and the raw mirror."""

    dataset_class: str
    dataset_root: str
    n_records: int
    publication_title: str
    publication_identifier: str
    report_pdf_sha256: str
    raw_mirror_manifest_path: str
    raw_mirror_manifest_sha256: str
    consumed_raw_sha256: dict[str, str]
    trait_source: str
    sensitivity_trait_source: str
    replicate_ids_from: str
    molar_mass_g_per_mol: dict[str, float]
    generating_script: str


def sha256_of(path: Path) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replicate_ids() -> dict[str, int]:
    """``<run>:well<n>`` -> biological replicate id, from the loader's plate layouts."""
    raw = Path(DATASET_ROOT) / "raw"
    return {
        f"{run.value}:well{w.well}": w.biological_replicate_id
        for run in b.Run
        for w in b.layout(run, raw).wells
    }


def wells_table(dataset: InhibitorBioscreenVolk2021Dataset) -> pd.DataFrame:
    """One row per record."""
    replicate = replicate_ids()
    rows = []
    for i in range(len(dataset)):
        e = dataset.transform_item(dataset[i])["experiment"]
        doses = {c: 0.0 for c in COMPOUNDS}
        for p in e.environment.perturbations:
            if p.concentration.unit.value != "g/L":
                raise ValueError(
                    f"{e.phenotype.screen_id}: unit {p.concentration.unit}"
                )
            doses[p.compound.name] = float(p.concentration.value)
        present = sorted(c for c, d in doses.items() if d > 0)
        screen_id = e.phenotype.screen_id
        run, well = screen_id.split(":well")
        response = e.phenotype.environment_response
        row = {
            "well": screen_id,
            "run": run,
            "well_number": int(well),
            "biological_replicate_id": replicate[screen_id],
            "duration_h": float(e.environment.duration_hours),
            "n_compounds": len(present),
            "compounds": "|".join(present),
        }
        for c in COMPOUNDS:
            row[f"dose_g_per_l_{ABBR[c]}"] = doses[c]
        for c in COMPOUNDS:
            row[f"dose_mM_{ABBR[c]}"] = (
                doses[c] / MOLAR_MASS_G_PER_MOL[ABBR[c]] * 1000.0
            )
        row["fitness"] = float(response) if response is not None else np.nan
        row["grew"] = response is not None
        row["category_label"] = e.phenotype.category_label
        rows.append(row)
    return pd.DataFrame(rows)


def ex23_conditions(wells: pd.DataFrame) -> pd.DataFrame:
    """039's ex23 condition table: grew = any replicate grew, fitness = mean of grown."""
    ex23 = wells[(wells["run"] == "ex23") & (wells["n_compounds"] > 0)].assign(
        grew_software=lambda d: d["grew_software"].astype(bool)
    )
    return (
        ex23.groupby("compounds")
        .agg(
            n_compounds=("n_compounds", "first"),
            n_wells=("well", "size"),
            n_grew=("grew", "sum"),
            fitness_mean_grown=("fitness", "mean"),
            n_grew_software=("grew_software", "sum"),
            fitness_software_mean_grown=("fitness_software", "mean"),
        )
        .assign(
            grew=lambda d: d["n_grew"] > 0,
            grew_software=lambda d: d["n_grew_software"] > 0,
        )
        .reset_index()
    )


def add_software_call(wells: pd.DataFrame) -> pd.DataFrame:
    """ex23's sensitivity call: the Bioscreen software's generation time, on its scale.

    ``fitness_software`` = mean software generation time of ex23's grown WT wells / the
    well's; ``grew_software`` = the software assigned a generation time. NaN for the
    other runs (the analysis uses the software call only on ex23).
    """
    raw = Path(DATASET_ROOT) / "raw"
    gt = b.software_generation_times(b.Run.ex23, raw)
    ex23 = wells["run"] == "ex23"
    numbers = wells.loc[ex23, "well_number"]
    sw = numbers.map(lambda w: np.nan if gt[w] is None else gt[w]).astype(float)
    wt = sw[wells.loc[ex23, "n_compounds"] == 0].mean()
    out = wells.copy()
    out["software_generation_time_h"] = np.nan
    out.loc[ex23, "software_generation_time_h"] = sw
    out["fitness_software"] = np.nan
    out.loc[ex23, "fitness_software"] = wt / sw
    out["grew_software"] = pd.NA
    out.loc[ex23, "grew_software"] = sw.notna()
    return out


def growth_call_check(wells: pd.DataFrame) -> pd.DataFrame:
    """ex23 wells where the served (raw-curve) and the software growth calls differ."""
    raw = Path(DATASET_ROOT) / "raw"
    curves = b.read_raw(raw / b.RAW_CSV["ex23"])
    ex23 = wells[wells["run"] == "ex23"]
    disputed = ex23[ex23["grew"] != ex23["grew_software"].astype(bool)].copy()
    disputed["od_rise"] = [
        float(
            (curves[w].to_numpy(float) - np.median(curves[w].to_numpy(float)[:3])).max()
        )
        for w in disputed["well_number"]
    ]
    return disputed[
        [
            "well",
            "compounds",
            "biological_replicate_id",
            "od_rise",
            "software_generation_time_h",
            "grew",
            "grew_software",
        ]
    ].rename(columns={"grew": "grew_served", "compounds": "combination"})


def check_039(cond: pd.DataFrame, grew: str, fitness: str) -> pd.DataFrame:
    """The ex23 summary under one growth call beside 039's (039 used the software)."""
    rows = []
    for k, (n_comb, n_grew, mean_039) in EXPECTED_039_EX23.items():
        g = cond[cond["n_compounds"] == k]
        grown = g[g[grew]]
        mean_here = float(grown[fitness].mean()) if len(grown) else None
        rows.append(
            {
                "n_compounds": k,
                "combinations": len(g),
                "combinations_039": n_comb,
                "grew": int(g[grew].sum()),
                "grew_039": n_grew,
                "mean_fitness_grown": mean_here,
                "mean_fitness_grown_039": mean_039,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(RESULTS, exist_ok=True)
    dataset = InhibitorBioscreenVolk2021Dataset(root=DATASET_ROOT)
    wells = add_software_call(wells_table(dataset))
    wells.to_csv(osp.join(RESULTS, "wetlab_wells.csv"), index=False)
    disputed = growth_call_check(wells)
    disputed.to_csv(osp.join(RESULTS, "wetlab_growth_call_check.csv"), index=False)
    print(
        f"ex23 wells whose served and software growth calls differ: {len(disputed)} of "
        f"{int((wells['run'] == 'ex23').sum())}\n" + disputed.round(3).to_string()
    )
    ex23_wt = wells[(wells["run"] == "ex23") & (wells["n_compounds"] == 0)]
    wt_soft = ex23_wt["software_generation_time_h"].mean()
    # the loader writes each run's served (raw-curve) WT generation time to preprocess/
    with open(
        osp.join(DATASET_ROOT, "preprocess", "wild_type_generation_time.json")
    ) as h:
        wt_served = json.load(h)["ex23"]
    print(
        f"ex23 control generation time: served (raw curve) {wt_served:.3f} h, software "
        f"{wt_soft:.3f} h (n = {len(ex23_wt)} control wells); each call is scored on its "
        "own scale"
    )

    by_n = Counter(wells["n_compounds"])
    print("wells by number of compounds:", dict(sorted(by_n.items())))
    if dict(by_n) != EXPECTED_BY_N_COMPOUNDS:
        raise RuntimeError(f"{dict(by_n)} != {EXPECTED_BY_N_COMPOUNDS}")
    by_run = wells.groupby("run").agg(
        wells=("well", "size"),
        grew=("grew", "sum"),
        duration_h=("duration_h", "first"),
        controls=("n_compounds", lambda s: int((s == 0).sum())),
    )
    print("wells by run:\n" + by_run.to_string())
    print(
        "wells by run and number of compounds:\n"
        + pd.crosstab(wells["run"], wells["n_compounds"]).to_string()
    )

    cond = ex23_conditions(wells)
    for label, grew, fitness in (
        ("served (raw-curve) call", "grew", "fitness_mean_grown"),
        ("software call", "grew_software", "fitness_software_mean_grown"),
    ):
        print(
            f"ex23 under the {label} vs 039 (039 used the software call):\n"
            + check_039(cond, grew, fitness).round(3).to_string()
        )

    publication = dataset.transform_item(dataset[0])["publication"]
    manifest = RAW_MIRROR / "manifest.json"
    provenance = WetlabProvenance(
        dataset_class=InhibitorBioscreenVolk2021Dataset.__name__,
        dataset_root=DATASET_ROOT,
        n_records=len(dataset),
        publication_title=publication.title,
        publication_identifier=publication.identifier,
        report_pdf_sha256=b.REPORT_PDF_SHA256,
        raw_mirror_manifest_path=str(manifest),
        raw_mirror_manifest_sha256=sha256_of(manifest),
        consumed_raw_sha256=CONSUMED_SHA256,
        trait_source=b.TraitSource.raw_curve.value,
        sensitivity_trait_source=(
            "ex23 only: bioscreen.software_generation_times (the gt column of "
            f"{b.EX23_PREPROCESSED}), fitness on its own WT scale"
        ),
        replicate_ids_from="torchcell.datasets.private_torchcell.bioscreen.layout",
        molar_mass_g_per_mol=MOLAR_MASS_G_PER_MOL,
        generating_script="experiments/040-inhibitor-synergy-wetlab/scripts/wetlab_table.py",
    )
    if b.REPORT_PDF_SHA256 not in publication.identifier:
        raise RuntimeError("publication identifier does not carry the report sha256")
    with open(osp.join(RESULTS, "wetlab_provenance.json"), "w") as handle:
        handle.write(provenance.model_dump_json(indent=2))
    print(f"wrote {len(wells)} wells and the provenance record to {RESULTS}")


if __name__ == "__main__":
    main()
