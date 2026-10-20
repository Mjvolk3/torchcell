# experiments/040-inhibitor-synergy-wetlab/scripts/r3_host_predictions.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.r3_host_predictions]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/r3_host_predictions
"""Every arm's host-head prediction on every Bioscreen medium, as one long table.

``train_mixture.py`` writes, per config and fold, the seed-mean host prediction over
``MixtureData.host`` (the public IC30 anchors, the ex21 titration, the ex23 combination
plate and the three isobole grids, in the trainer's own record order) to
``$DATA_ROOT/experiments/040-inhibitor-synergy-wetlab/predictions/<sweep>/
<config>_fold<k>_seed<fold_seed>_host.npy``. That order is not written beside the
vector, so this script assembles the same host records the trainer did (the served
call, the Vanacloig source alone, which is enough for the records) and joins each
prediction to its medium: the run, the compounds, their doses in mM and in g/L (read
back from ``results/wetlab_wells.csv`` through the same medium key the trainer builds),
the observed fitness and growth call, and which split the record is in.

Which records an arm TRAINED on depends on its ``host_train``: ``anchors`` trains on the
anchors alone, so its ex21 rows are predictions of singles it never saw;
``anchors+ex21`` and ``finetune_ex21`` train on the ex21 wells, so their ex21 rows are
fits. The ex23 and isobole rows are predictions for every arm. The table says which
through ``trained_on``.

Writes ``results/host_predictions.csv``.

    PYTHONPATH=$PWD python experiments/040-inhibitor-synergy-wetlab/scripts/r3_host_predictions.py
"""

from __future__ import annotations

import glob
import json
import os
import os.path as osp
import re
import sys

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from mixture_data import ABBREVIATION, _well_doses, assemble  # noqa: E402

load_dotenv()
EXPERIMENT = osp.dirname(osp.dirname(osp.abspath(__file__)))
RESULTS = osp.join(EXPERIMENT, "results")
CONF = osp.join(EXPERIMENT, "conf", "mixture")
WELLS_CSV = osp.join(RESULTS, "wetlab_wells.csv")
#: the per-well prediction roots: this machine's, and the Delta tree's, rsynced from
#: ``/scratch/bbub/mjvolk3/torchcell/experiments/040-inhibitor-synergy-wetlab/
#: predictions/`` into ``predictions-delta/`` beside it (no checkpoints). The same arm
#: can have run on both; the ``machine`` column keeps the runs apart and a reader that
#: averages an arm takes every run it has.
PREDICTIONS = {
    machine: osp.join(
        os.environ["DATA_ROOT"], "experiments", "040-inhibitor-synergy-wetlab", sub
    )
    for machine, sub in (("gilahyper", "predictions"), ("delta", "predictions-delta"))
}
FILE = re.compile(r"^(?P<name>.+)_fold(?P<fold>\d+)_seed(?P<fold_seed>\d+)_host\.npy$")
ABBR_OF = ABBREVIATION  # long name to the wells table column suffix


def config_of(sweep: str, name: str) -> dict:
    """One config, read from its sweep file."""
    with open(osp.join(CONF, f"{sweep}.yaml")) as fh:
        for cfg in yaml.safe_load(fh):
            if cfg["name"] == name:
                return dict(cfg)
    raise KeyError(f"{name} not in {sweep}.yaml")


def main() -> None:
    # The host records depend on the arm's sources: every IC30-dosed source adds its
    # anchors (Vanacloig 29, Hoepfner 74 more), so the records are assembled once per
    # distinct source set and the prediction vector is matched to its own set.
    hosts: dict[tuple[str, ...], list] = {}

    def host_for(sources: list[str]) -> list:
        key = tuple(sources)
        if key not in hosts:
            hosts[key] = assemble(
                list(sources), WELLS_CSV, ["fcfp4_count"], None, True, "served"
            ).host
        return hosts[key]

    wells = pd.read_csv(WELLS_CSV)
    wells["medium"] = wells.apply(
        lambda r: json.dumps(_well_doses(r, ABBREVIATION)), axis=1
    )
    g_per_l: dict[str, dict[str, float]] = {}
    well_ids: dict[str, list[str]] = {}
    for (run, medium), group in wells.groupby(["run", "medium"]):
        key = f"{run}:{medium}"
        first = group.iloc[0]
        g_per_l[key] = {
            name: float(first[f"dose_g_per_l_{ABBR_OF[name]}"])
            for name, _ in json.loads(medium)
        }
        well_ids[key] = list(group["well"])

    rows = []
    paths = [
        (machine, path)
        for machine, root in PREDICTIONS.items()
        for path in sorted(glob.glob(osp.join(root, "*", "*_host.npy")))
    ]
    for machine, path in paths:
        sweep = osp.basename(osp.dirname(path))
        # round 1 ran under the pool-and-last protocol before the Hoepfner anchors
        # joined the host set, so its 733-record vectors do not join today's records;
        # no round reads it through this table
        if "smoke" in sweep or sweep.startswith("r1_"):
            continue
        m = FILE.match(osp.basename(path))
        assert m, path
        pred = np.load(path)
        cfg = config_of(sweep, m["name"])
        host = host_for(list(cfg.get("sources", ["vanacloig"])))
        assert len(pred) == len(host), f"{path}: {len(pred)} predictions"
        trained = str(cfg.get("host_train", "anchors"))
        for record, value in zip(host, pred, strict=True):
            trained_on = record.split == "anchor" or (
                record.split == "ex21" and trained != "anchors"
            )
            rows.append(
                {
                    "machine": machine,
                    "sweep": sweep,
                    "name": m["name"],
                    "fold": int(m["fold"]),
                    "fold_seed": int(m["fold_seed"]),
                    "host_train": trained,
                    "key": record.key,
                    "split": record.split,
                    "run": record.run,
                    "compounds": "|".join(record.compounds),
                    "n_compounds": len(record.compounds),
                    "mM": "|".join(f"{10 ** (x + 3):.6g}" for x in record.log10_molar),
                    "g_per_l": "|".join(
                        f"{g_per_l.get(record.key, {}).get(c, float('nan')):.6g}"
                        for c in record.compounds
                    ),
                    "wells": "|".join(well_ids.get(record.key, [])),
                    "observed": record.fitness,
                    "observed_grown": record.fitness_grown,
                    "grew": record.grew,
                    "trained_on": trained_on,
                    "predicted": float(value),
                }
            )
    table = pd.DataFrame(rows)
    out = osp.join(RESULTS, "host_predictions.csv")
    table.to_csv(out, index=False)
    print(table.groupby(["sweep", "machine"]).size().to_string())
    print(f"{len(table)} rows, {table.name.nunique()} configs -> {out}")


if __name__ == "__main__":
    main()
