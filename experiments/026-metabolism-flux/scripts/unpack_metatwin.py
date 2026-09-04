# experiments/026-metabolism-flux/scripts/unpack_metatwin.py
# [[experiments.026-metabolism-flux.scripts.unpack_metatwin]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/unpack_metatwin.py

r"""Unpack Wu et al.'s released archive and locate their per-pair kinetic predictions.

Their git repository ships the figure notebooks and the figure PDFs but not the prediction
tables the notebooks read: those live in ``Results/kcat_km_predict/`` inside the 6.6 GB
Zenodo archive. Without them the underground arm of Fig.~3 cannot be drawn at all, since
generating the Yeast-MetaTwin reaction set is a separate retrobiosynthesis pipeline.

The notebook names the files it wants, so the mapping is read from their code rather than
guessed:

* ``yeast8U_kcat_predict.csv`` -- DLKcat, column ``kcat``
* ``yeast8U_unikp.csv`` -- UniKP, used for both parameters in their notebook
* ``yeast8U_kcat_result_TurNuP.csv`` -- TurNuP, column ``kcat``
* ``yeast8U_km_km_predict.csv`` -- Boost_KM, column ``KM``

Every table is keyed by ``rea_id``, and their convention is that an id containing ``rxn``
is a predicted (underground) reaction while anything else is a curated Yeast9 one. That
split is the entire core-versus-underground distinction in the figure, so it is recorded
in the emitted mapping rather than re-derived at plot time.

The archive is RAR. ``rarfile`` drives whichever extractor is present; ``bsdtar`` ships
with the conda base here and reads RAR through libarchive.
"""

import argparse
import json
import os
import os.path as osp
from typing import cast

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
METATWIN = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "yeast_metatwin")
EXPERIMENT_ROOT = cast(str, os.getenv("EXPERIMENT_ROOT"))
RESULTS = osp.join(EXPERIMENT_ROOT, "026-metabolism-flux", "results")

# Filename fragment -> (predictor, parameter, value column), taken from the input paths in
# Code/kcatkm_prediction/Yeast-MetaTwin-05.Fig3abcde.ipynb.
WANTED: list[tuple[str, str, str, str]] = [
    ("yeast8U_kcat_predict", "dlkcat", "k_cat", "kcat"),
    ("yeast8U_unikp", "unikp", "k_cat", "kcat"),
    ("yeast8U_unikp", "unikp", "K_M", "km"),
    ("yeast8U_kcat_result_TurNuP", "turnup", "k_cat", "kcat"),
    ("yeast8U_km_km_predict", "boost_km", "K_M", "KM"),
    ("eitlem", "eitlem", "k_cat", "kcat"),
    ("eitlem", "eitlem", "K_M", "km"),
    ("deepenzyme", "deepenzyme", "k_cat", "kcat"),
]
ID_COLUMN = "rea_id"


def extract(archive: str, target: str) -> None:
    """Unpack the archive, preferring rarfile and falling back to bsdtar."""
    os.makedirs(target, exist_ok=True)
    import rarfile

    # libarchive via bsdtar is the extractor available here; unrar is not installed.
    rarfile.BSDTAR_TOOL = osp.expanduser("~/miniconda3/bin/bsdtar")
    rarfile.PREFER_BSDTAR = True
    with rarfile.RarFile(archive) as handle:
        handle.extractall(target)


def find_tables(root: str) -> list[str]:
    """Every CSV under the unpacked tree, so a moved file is still found."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".csv"):
                found.append(osp.join(dirpath, name))
    return sorted(found)


def main() -> None:
    """Unpack, locate the prediction tables, and write the plot-time mapping."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default=osp.join(METATWIN, "metatwin.rar"))
    parser.add_argument("--target", default=osp.join(METATWIN, "unpacked"))
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()

    if not args.skip_extract:
        print(f"extracting {args.archive} ...", flush=True)
        extract(args.archive, args.target)

    tables = find_tables(args.target)
    print(f"{len(tables)} CSV files under {args.target}")

    mapping: dict[str, dict[str, str]] = {}
    report: list[dict[str, object]] = []
    for fragment, predictor, parameter, column in WANTED:
        matches = [t for t in tables if fragment.lower() in osp.basename(t).lower()]
        if not matches:
            report.append({"predictor": predictor, "parameter": parameter,
                           "status": "no file matching", "fragment": fragment})
            continue
        path = matches[0]
        head = pd.read_csv(path, nrows=5)
        if column not in head.columns or ID_COLUMN not in head.columns:
            report.append({"predictor": predictor, "parameter": parameter,
                           "status": "columns absent", "path": path,
                           "columns": list(head.columns)})
            continue
        frame = pd.read_csv(path, usecols=[ID_COLUMN, column])
        ids = frame[ID_COLUMN].astype(str)
        underground = int(ids.str.contains("rxn").sum())
        mapping[f"{predictor}:{parameter}"] = {
            "path": path,
            "column": column,
            "id_column": ID_COLUMN,
        }
        report.append({
            "predictor": predictor,
            "parameter": parameter,
            "status": "ok",
            "path": path,
            "n_rows": int(len(frame)),
            "n_underground": underground,
            "n_core": int(len(frame) - underground),
        })

    with open(osp.join(METATWIN, "paper_tables.json"), "w") as handle:
        json.dump(mapping, handle, indent=2)
    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "metatwin_tables.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
