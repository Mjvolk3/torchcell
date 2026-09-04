# experiments/026-metabolism-flux/scripts/mirror_kinetic_models.py
# [[experiments.026-metabolism-flux.scripts.mirror_kinetic_models]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/mirror_kinetic_models.py

r"""Move every kinetic predictor into the standard mirror and pin its bytes.

The predictors arrived as ad-hoc git clones under ``data/enzyme_kinetics/predictors``,
which is not a location anything can be rebuilt from: nothing records which commit was
cloned, and nothing detects an upstream force-push. This promotes them to
``$DATA_ROOT/models/kinetics/<name>/`` with a ``manifest.json`` per model, so a dataset
built from one can name the exact weights that produced it.

Run once per predictor, and again whenever a weight archive is added. Re-running is safe:
hashing is idempotent and an existing mirror is re-verified rather than re-copied.
"""

import argparse
import json
import os
import os.path as osp
import shutil
import subprocess
from typing import cast

from dotenv import load_dotenv

from torchcell.data.model_mirror import (
    ModelMirror,
    list_mirrors,
    mirror_summary,
    read_manifest,
    scan_files,
    utc_now,
    write_manifest,
)
from torchcell.literature.manifest import RetrievalMethod

load_dotenv()
DATA_ROOT = cast(str, os.getenv("DATA_ROOT"))
CLONE_ROOT = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "predictors")
WEIGHTS_ROOT = osp.join(DATA_ROOT, "data", "enzyme_kinetics", "weights")

# Role assignment per model, by path prefix within the mirror. Anything unmatched is
# recorded as ``source``, so no file is ever implicitly promoted to a weight.
ROLES: dict[str, dict[str, str]] = {
    "dlkcat": {
        "DeeplearningApproach/Results/output/all--radius2": "weights",
        "DeeplearningApproach/Data/input/atom_dict.pickle": "dictionary",
        "DeeplearningApproach/Data/input/bond_dict.pickle": "dictionary",
        "DeeplearningApproach/Data/input/edge_dict.pickle": "dictionary",
        "DeeplearningApproach/Data/input/fingerprint_dict.pickle": "dictionary",
        "DeeplearningApproach/Data/input/sequence_dict.pickle": "vocabulary",
        "DeeplearningApproach/Data/input/": "training_data",
        "DeeplearningApproach/Code/": "source",
        "BayesianApproach/Data/": "training_data",
    },
    "unikp": {
        "trfm_12_23000.pkl": "weights",
        "vocab.pkl": "vocabulary",
        "UniKP for ": "weights",
        "datasets/": "training_data",
    },
    "turnup": {"data.zip": "archive", "data/": "weights"},
    "eitlem": {"Weights/": "weights", "Data/": "training_data", "Code/": "source"},
    "boost_km": {
        "notebooks_and_code/additional_code/unirep/1900_weights/": "weights",
        "notebooks_and_code/additional_code/UniRep50/unirep/1900_weights/": "weights",
        "datasets/": "training_data",
    },
    "deepenzyme": {"Weights/": "weights", "Data/": "training_data", "Code/": "source"},
}

# One entry per predictor: where it came from, what it can produce, and -- when it cannot
# run -- what specifically blocks it. ``runnable=False`` with a stated blocker is a
# measurement; silently omitting the model would leave the gap looking like a choice.
SPECS: dict[str, dict[str, object]] = {
    "dlkcat": {
        "clone_dir": "DLKcat",
        "display_name": "DLKcat",
        "source_url": "https://github.com/SysBioChalmers/DLKcat",
        "method": RetrievalMethod.git_clone,
        "emits": ["k_cat"],
        "inputs": ["sequence", "substrate_smiles"],
        "runnable": True,
        "notes": (
            "The trained model is Results/output/all--radius2--...--iteration50, a "
            "state_dict with no file extension. An earlier audit read the extensionless "
            "name as 'ships no weights' and queued a 18 GB Zenodo download that was never "
            "needed."
        ),
    },
    "unikp": {
        "clone_dir": "UniKP",
        "display_name": "UniKP",
        "source_url": "https://github.com/Luo-SynBioLab/UniKP",
        "method": RetrievalMethod.git_clone,
        "emits": ["k_cat", "K_M"],
        "inputs": ["sequence", "substrate_smiles"],
        "runnable": True,
        "extra_command": (
            "curl -L -o 'UniKP for kcat.pkl' "
            "https://huggingface.co/HanselYu/UniKP/resolve/main/UniKP%20for%20kcat.pkl "
            "(likewise 'UniKP for Km.pkl' and 'UniKP for kcat_Km.pkl')"
        ),
        "notes": (
            "The repository ships the SMILES transformer (trfm_12_23000.pkl) and its "
            "vocabulary. The three fitted ExtraTrees regressors are NOT in the repository; "
            "they are released separately on HuggingFace at HanselYu/UniKP and are "
            "retrieved by the extra command recorded here. They were pickled under "
            "scikit-learn earlier than 1.3, so they load only in the pinned environment."
        ),
    },
    "turnup": {
        "clone_dir": "kcat_prediction",
        "display_name": "TurNuP",
        "source_url": "https://github.com/AlexanderKroll/kcat_prediction",
        "method": RetrievalMethod.git_clone,
        "emits": ["k_cat"],
        "inputs": ["sequence", "reaction_smiles"],
        "runnable": True,
        "extra_command": (
            "unzip data.zip  # Zenodo 8367052, the fitted xgboost boosters; the ESM-1b "
            "fine-tune ESM1b_ts ships only with Zenodo 8038678, and the 7.83 GB ESM-1b "
            "base was pulled from Meta's CDN and verified against that archive's central "
            "directory by size and CRC32"
        ),
        "notes": (
            "Takes the FULL reaction SMILES, so a reaction is refused when any single "
            "participant has no structure; that is why coverage is 55 percent rather than "
            "95. Output is base 10, explicitly unlike DLKcat's base 2. The deployed "
            "booster was fit on [difference_fp, ESM1b_ts BOS-token], NOT on mean-pooled "
            "ESM-1b: the older DLKcat_comparison notebook feeds the latter and would run "
            "without error while returning numbers that are not TurNuP's."
        ),
    },
    "eitlem": {
        "clone_dir": "EITLEM-Kinetics",
        "display_name": "EITLEM-Kinetics",
        "source_url": "https://github.com/XvQiao/EITLEM-Kinetics",
        "method": RetrievalMethod.git_clone,
        "emits": ["k_cat", "K_M"],
        "inputs": ["sequence", "substrate_smiles"],
        "runnable": True,
        "extra_command": (
            "curl -L -o Weights.zip "
            "https://zenodo.org/api/records/16153803/files/Weights.zip/content  # 12.3 GB, "
            "sha256 7628a5c64a6c88ee7cda79d47e162967f88254583c1fd52de4e2771c1c25b58c"
        ),
        "notes": (
            "Weights are released on Zenodo rather than in the repository. The archive "
            "ships iter1 and iter8 of the 8-round transfer schedule per parameter; iter8 "
            "is the final one and is what the README's usage section names. Output is "
            "log10, established by reproducing the README's worked example at 1.3904 "
            "against its stated 1.39, which log2 could not produce. Sequences are "
            "truncated to 1,022 residues because ESM-1v admits 1,024 and the authors' own "
            "training FASTA maxes at exactly 1,022."
        ),
    },
    "boost_km": {
        "clone_dir": "KM_prediction",
        "display_name": "Boost_KM",
        "source_url": "https://github.com/AlexanderKroll/KM_prediction",
        "method": RetrievalMethod.git_clone,
        "emits": ["K_M"],
        "inputs": ["sequence", "substrate_smiles"],
        "runnable": False,
        "blocked_by": (
            "The 18 shipped weight files are the UniRep 1900 mLSTM FEATURIZER, not a "
            "fitted K_M model. The repository ships training notebooks and the BRENDA/"
            "Sabio training data, so the gradient-boosting model has to be refit before "
            "any prediction exists."
        ),
    },
    "deepenzyme": {
        "clone_dir": "DeepEnzyme",
        "display_name": "DeepEnzyme",
        "source_url": "https://github.com/hongzhonglu/DeepEnzyme",
        "method": RetrievalMethod.git_clone,
        "emits": ["k_cat"],
        "inputs": ["sequence", "substrate_smiles", "structure"],
        "runnable": True,
        "extra_command": (
            "curl -L -o Weights/example "
            "<figshare 10.6084/m9.figshare.25771062.v2>  # md5 "
            "68f83e5d90937e66a6f62b85e8695f38"
        ),
        "notes": (
            "The apex requirement is listed in requirements.txt and imported nowhere in "
            "Code/, so it was a training-time mixed-precision dependency and the network "
            "runs unmodified on current torch, numpy and rdkit. Weights are released on "
            "figshare rather than in the repository; all 137 checkpoint tensors match the "
            "architecture with none missing or unexpected. Output is log2, the same "
            "convention as DLKcat."
        ),
    },
}


def git_revision(path: str) -> str | None:
    """The commit a clone sits on, or None when the directory is not a git checkout."""
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return out.stdout.strip()


def build_mirror(name: str, copy: bool) -> ModelMirror:
    """Populate one mirror from its clone and return the manifest that pins it."""
    spec = SPECS[name]
    source = osp.join(CLONE_ROOT, cast(str, spec["clone_dir"]))
    if not osp.isdir(source):
        raise FileNotFoundError(f"{name}: clone absent at {source}")

    target = osp.join(DATA_ROOT, "models", "kinetics", name)
    revision = git_revision(source)
    if copy and not osp.isdir(target):
        os.makedirs(osp.dirname(target), exist_ok=True)
        # ``.git`` is excluded on purpose: the mirror pins bytes by sha256, and carrying a
        # 100 MB object store for a history nothing reads makes every verify pass slower
        # without making any value more traceable. The commit is recorded instead.
        shutil.copytree(source, target, ignore=shutil.ignore_patterns(".git"))

    # A weight archive downloaded separately lands beside the clone; fold it in so the
    # mirror is one self-contained, hashed unit.
    archive = osp.join(WEIGHTS_ROOT, f"{name}_data.zip")
    if osp.exists(archive) and not osp.exists(osp.join(target, "data.zip")):
        shutil.copy2(archive, osp.join(target, "data.zip"))

    files = scan_files(target, ROLES.get(name, {}))
    return ModelMirror(
        name=name,
        display_name=cast(str, spec["display_name"]),
        emits=cast(list[str], spec["emits"]),
        inputs=cast(list[str], spec["inputs"]),
        method=cast(RetrievalMethod, spec["method"]),
        source_url=cast(str, spec["source_url"]),
        retrieval_command=(
            f"git clone {spec['source_url']} {name}"
            + (f" && {spec['extra_command']}" if spec.get("extra_command") else "")
        ),
        retrieved_at=utc_now(),
        revision=revision,
        files=files,
        runnable=cast(bool, spec.get("runnable", True)),
        blocked_by=cast(str | None, spec.get("blocked_by")),
        notes=cast(str | None, spec.get("notes")),
    )


def main() -> None:
    """Mirror the named predictors, or verify the mirrors that already exist."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=sorted(SPECS))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--no-copy", action="store_true")
    args = parser.parse_args()

    results = osp.join(
        cast(str, os.getenv("EXPERIMENT_ROOT")), "026-metabolism-flux", "results"
    )
    os.makedirs(results, exist_ok=True)

    if args.verify_only:
        report = []
        for mirror in list_mirrors(DATA_ROOT):
            broken = mirror.verify(DATA_ROOT)
            report.append({"name": mirror.name, "n_files": len(mirror.files), "broken": broken})
            state = "OK" if not broken else f"BROKEN ({len(broken)})"
            print(f"{mirror.name:12s} {len(mirror.files):5d} files  {state}")
        with open(osp.join(results, "kinetic_model_mirrors_verify.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        return

    mirrors = []
    for name in args.models:
        print(f"mirroring {name} ...", flush=True)
        mirror = build_mirror(name, copy=not args.no_copy)
        write_manifest(mirror, DATA_ROOT)
        mirrors.append(mirror)
        weight_mb = sum(f.size_bytes for f in mirror.weight_files()) / 1e6
        print(
            f"  {len(mirror.files):5d} files, {weight_mb:9.1f} MB of weights, "
            f"runnable={mirror.runnable}"
        )

    summary = mirror_summary(list_mirrors(DATA_ROOT))
    with open(osp.join(results, "kinetic_model_mirrors.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
