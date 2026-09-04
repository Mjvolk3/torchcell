# experiments/026-metabolism-flux/scripts/run_boost_km_esm1b_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_boost_km_esm1b_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_boost_km_esm1b_pinned.py

r"""Run Boost_KM the way Wu et al. actually ran it: the ESM-1b variant, not the paper's.

WHY THIS SCRIPT EXISTS ALONGSIDE ``run_boost_km_pinned.py``
------------------------------------------------------------
Wu et al.'s Methods cite ``AlexanderKroll/KM_prediction`` for :math:`K_M`, and that repo is
Kroll et al. (2021) as published: a 52-dim message-passing substrate fingerprint
concatenated with a **UniRep 1900** enzyme vector, into an XGBoost regressor. Their released
code runs something else. ``Yeast-MetaTwin-04.KMprediction.ipynb`` vendors a different
variant, ``KM_prediction_function``, whose ``enzyme_representations.py`` loads
``esm1b_t33_650M_UR50S.pt`` and whose regressor is ``xgboost_model_new_KM_esm1b.dat``.
Kroll's own README flags the substitution:

    "the model provided in the repository 'KM_prediction_function' is slighly different to
    the one presented in our paper: Instead of the UniRep model, we are using the ESM-1b
    model."

So the citation and the computation disagree, and reproducing Fig. 3f means following the
code. The two scripts are kept separate rather than merged behind a flag because they are
different published models that happen to share a substrate encoder.

WHAT IS SHARED AND WHAT IS NOT
--------------------------------
The DMPNN substrate checkpoint is **byte-identical** between the two repositories (all three
``saved_model_GNN_best_hyperparameters`` files match on sha256), so the fingerprint stage is
imported from :mod:`run_boost_km_pinned` rather than reimplemented, and it inherits that
script's validation against the authors' own stored ``GNN FP`` column. What changes is the
enzyme half, 1,900 UniRep dimensions becoming 1,280 ESM-1b dimensions, and the regressor,
which takes 1,332 features rather than 1,952.

The practical consequence is speed. UniRep is a TensorFlow 1 mLSTM that runs on CPU and
saturated this machine on a previous attempt; ESM-1b is a transformer that runs on one GPU
in minutes.

UNITS AND BASE
----------------
``KM_prediction.py`` ends with ``KMs = 10**bst.predict(dX)``, so the regressor's target is
:math:`\log_{10} K_M` and the emitted value is in **mM**, matching the UniRep variant and
the two other :math:`K_M` tables in this experiment. The base is read from the authors' call
site, not assumed.
"""

import argparse
import importlib.util
import json
import os
import os.path as osp
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

HERE = osp.dirname(osp.abspath(__file__))
# The ESM-1b variant is vendored inside Wu et al.'s release rather than mirrored on its own,
# because that copy is the one their figure was produced with.
METATWIN_KM = (
    "/scratch/projects/torchcell-scratch/data/enzyme_kinetics/yeast_metatwin/repo/"
    "Code/kcatkm_prediction/KM_prediction"
)
REGRESSOR = osp.join(METATWIN_KM, "data", "saved_models", "xgboost",
                     "xgboost_model_new_KM_esm1b.dat")
ESM1B_CHECKPOINT = (
    "/scratch/projects/torchcell-scratch/models/kinetics/boost_km/esm/"
    "esm1b_t33_650M_UR50S.pt"
)

FINGERPRINT_DIM = 52
ESM1B_DIM = 1280
# ``preprocess_enzymes`` crops every sequence to 1022 residues, which is ESM-1b's limit once
# the start and end tokens are added. Reproduced exactly: a longer crop silently changes the
# mean-pooled vector for every large enzyme.
MAX_RESIDUES = 1022


def load_sibling() -> object:
    """The UniRep script's fingerprint machinery, which this variant shares verbatim."""
    path = osp.join(HERE, "run_boost_km_pinned.py")
    spec = importlib.util.spec_from_file_location("run_boost_km_pinned", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_boost_km_pinned"] = module
    spec.loader.exec_module(module)
    return module


def sequence_key(sequence: str) -> str:
    """Stable content key, so the embedding cache survives a re-run with new pairs."""
    import hashlib

    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def stage_esm1b(fasta: str, output: str, batch_size: int, device: str) -> None:
    """Mean-pooled ESM-1b layer-33 vectors, the authors' enzyme representation.

    Their loop pools ``representations[33][0, 1 : len + 1]``, that is the residue tokens
    only, excluding the start token and anything past the sequence. Batching adds padding,
    so each row is pooled over its own length rather than over the padded width.
    """
    import torch
    from esm import pretrained

    records: list[tuple[str, str]] = []
    with open(fasta) as handle:
        key = None
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                key = line[1:]
            elif line and key is not None:
                records.append((key, line))
                key = None

    model, alphabet = pretrained.load_model_and_alphabet(ESM1B_CHECKPOINT)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()

    # Length-sorted so a batch is not padded to the longest sequence in the whole set.
    records.sort(key=lambda item: len(item[1]))
    keys: list[str] = []
    blocks: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        _, _, tokens = converter(chunk)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[33])["representations"][33]
        for row, (key, sequence) in enumerate(chunk):
            pooled = out[row, 1 : len(sequence) + 1].mean(0).float().cpu().numpy()
            keys.append(key)
            blocks.append(pooled)
        print(f"esm1b: {min(start + batch_size, len(records))}/{len(records)}", flush=True)

    np.savez(
        output,
        keys=np.array(keys, dtype=object),
        vectors=np.stack(blocks).astype(np.float32),
    )
    print(f"esm1b: {len(keys)} sequences -> {output}", flush=True)


def fill_esm1b(sequences: list[str], cache_dir: str, batch_size: int, device: str) -> dict:
    """ESM-1b vectors for every sequence, computing only the ones not already cached."""
    import subprocess

    sibling = sys.modules["run_boost_km_pinned"]
    path = osp.join(cache_dir, "esm1b_1280.npz")
    cache = sibling.read_cache(path)
    missing = [s for s in sequences if sequence_key(s) not in cache]
    if not missing:
        return cache
    print(f"esm1b: {len(missing)} of {len(sequences)} sequences to embed", flush=True)
    os.makedirs(cache_dir, exist_ok=True)
    fasta = osp.join(cache_dir, "_esm1b_input.fasta")
    out = osp.join(cache_dir, "_esm1b_output.npz")
    with open(fasta, "w") as handle:
        handle.write(
            "\n".join(f">{sequence_key(s)}\n{s[:MAX_RESIDUES]}" for s in missing)
        )
    # A separate process because the fingerprint stage needs TensorFlow to own the process
    # and ESM-1b needs torch to own the GPU; sharing one interpreter makes the allocator
    # fight itself.
    subprocess.run(
        [sys.executable, osp.abspath(__file__), "--stage", "esm1b", "--fasta", fasta,
         "--output", out, "--batch-size", str(batch_size), "--device", device],
        check=True,
    )
    cache.update(sibling.read_cache(out))
    sibling.write_cache(path, cache)
    os.remove(fasta)
    os.remove(out)
    return cache


def load_regressor():
    """The vendored ESM-1b gradient-boosting model, checked for the width it should have."""
    with open(REGRESSOR, "rb") as handle:
        booster = pickle.load(handle)
    expected = FINGERPRINT_DIM + ESM1B_DIM
    if booster.num_features() != expected:
        raise RuntimeError(
            f"{REGRESSOR} takes {booster.num_features()} features, expected {expected}. "
            "The vendored regressor is not the one this feature layout describes."
        )
    return booster


def predict(inputs: str, output: str, cache_dir: str, batch_size: int, device: str) -> None:
    """Featurize, predict, and write the results parquet with K_M in mM."""
    sibling = load_sibling()
    pairs = pd.read_parquet(inputs)
    failure = sibling.screen(pairs)
    work = pairs[failure.isna()]
    print(f"{len(work)} usable of {len(pairs)} pairs", flush=True)

    sequences = work["sequence"].drop_duplicates().tolist()
    smiles = work["smiles"].drop_duplicates().tolist()
    print(f"{len(sequences)} distinct proteins, {len(smiles)} distinct molecules", flush=True)

    embeddings = fill_esm1b(sequences, cache_dir, batch_size, device)
    fingerprints = sibling.fill_fingerprints(smiles, cache_dir, batch_size)

    dropped = [s for s in smiles if s not in fingerprints]
    if dropped:
        lost = work["smiles"].isin(dropped)
        failure.loc[work.index[lost]] = "fingerprint_failed"
        work = work[~lost]
        print(f"{len(dropped)} molecules produced no fingerprint", flush=True)

    features = np.concatenate(
        [
            np.stack([fingerprints[s] for s in work["smiles"]]),
            np.stack([embeddings[sequence_key(s)] for s in work["sequence"]]),
        ],
        axis=1,
    )
    print(f"feature matrix {features.shape}", flush=True)

    import xgboost as xgb

    log10_km = load_regressor().predict(xgb.DMatrix(features))
    km_mm = np.power(10.0, log10_km)

    out = pairs.copy()
    out["K_M"] = np.nan
    out.loc[work.index, "K_M"] = km_mm
    out["failure"] = failure.values
    out["predictor"] = "boost_km"
    out.to_parquet(output, index=False)
    print(
        f"K_M mM: median {np.median(km_mm):.4f} p05 {np.percentile(km_mm, 5):.4f} "
        f"p95 {np.percentile(km_mm, 95):.4f}",
        flush=True,
    )
    print(f"wrote {output}")


def self_test(device: str) -> None:
    """Check the three things that would be silently wrong if they were wrong.

    The regressor's own test split is not shipped with the ESM-1b variant, so an R^2 against
    held-out truth is not available here the way it is for the UniRep model. What IS
    checkable is that each stage produces the object the next stage expects, and that the
    substrate encoder is the validated one.
    """
    sibling = load_sibling()
    booster = load_regressor()
    print(f"[1] regressor width {booster.num_features()} = 52 + {ESM1B_DIM}")

    import hashlib

    def digest(path: str) -> str:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()

    shared = "saved_model_GNN_best_hyperparameters"
    same = all(
        digest(osp.join(METATWIN_KM, "data", "saved_models", "GNN", f"{shared}{suffix}"))
        == digest(osp.join(sibling.DATASETS, "model_weights", f"{shared}{suffix}"))
        for suffix in (".index", ".data-00000-of-00002", ".data-00001-of-00002")
    )
    print(f"[2] substrate checkpoint identical to the validated UniRep-variant one: {same}")
    if not same:
        raise RuntimeError(
            "The two repositories ship different DMPNN weights, so the fingerprint stage "
            "cannot be shared and its earlier validation does not transfer."
        )

    import torch
    from esm import pretrained

    model, alphabet = pretrained.load_model_and_alphabet(ESM1B_CHECKPOINT)
    model = model.to(device).eval()
    converter = alphabet.get_batch_converter()
    probe = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"
    _, _, tokens = converter([("probe", probe)])
    with torch.no_grad():
        out = model(tokens.to(device), repr_layers=[33])["representations"][33]
    pooled = out[0, 1 : len(probe) + 1].mean(0)
    print(
        f"[3] esm1b probe: layer-33 width {out.shape[-1]}, pooled {tuple(pooled.shape)}, "
        f"finite {bool(torch.isfinite(pooled).all())}, "
        f"norm {float(pooled.norm()):.3f}"
    )
    if out.shape[-1] != ESM1B_DIM:
        raise RuntimeError(f"ESM-1b returned width {out.shape[-1]}, expected {ESM1B_DIM}.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--output")
    parser.add_argument(
        "--cache-dir",
        default="/scratch/projects/torchcell-scratch/data/torchcell/kinetics/boost_km/features",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stage", choices=["esm1b"])
    parser.add_argument("--fasta")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test(args.device)
    elif args.stage == "esm1b":
        stage_esm1b(args.fasta, args.output, args.batch_size, args.device)
    else:
        predict(args.inputs, args.output, args.cache_dir, args.batch_size, args.device)


if __name__ == "__main__":
    main()
