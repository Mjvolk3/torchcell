# experiments/026-metabolism-flux/scripts/run_turnup_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_turnup_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_turnup_pinned.py

r"""Run TurNuP inside its pinned environment and write predictions back as parquet.

WHAT TurNuP IS, AND WHY ITS INPUT IS NOT A SUBSTRATE
------------------------------------------------------
Kroll et al. (2023), *Nature Communications*. TurNuP represents the REACTION, not one
substrate: every substrate and every product is converted to SMARTS, joined into a
reaction SMARTS, and reduced to a 2,048-dim difference fingerprint. The enzyme is
represented by a 1,280-dim ESM-1b vector from a TASK-SPECIFIC fine-tune (``ESM1b_ts``),
read off the BOS token of layer 33 rather than mean-pooled over residues. A gradient
boosted tree (xgboost) maps the 3,328-dim concatenation to :math:`\log_{10} k_{cat}`.

Base 10, not base 2. The authors' own prediction function exponentiates with ``10**``,
and DLKcat -- the other predictor in this experiment -- uses base 2. Reading one as the
other inflates or deflates every value by a power of about 3.3 and still looks plausible.

WHY THIS IS A SEPARATE PROCESS
--------------------------------
The released booster is a pickle produced by xgboost 1.6, whose payload is a pre-1.6 JSON
model; xgboost warns on it and drops support entirely at 2.3. The project environment
carries no xgboost at all, and fair-esm additionally pins an older torch surface. So
inference runs in an environment pinned to xgboost 1.6.1 / fair-esm 2.0.0 / torch 2.5.1,
invoked as a subprocess, with parquet as the interface -- the same shape ``run_unikp_pinned.py``
uses for UniKP's scikit-learn pin.

WHERE THE WEIGHTS COME FROM
----------------------------
Two Zenodo archives back one model. The mirror's ``data.zip`` (Zenodo 8367052, the paper's
repository) carries the fitted boosters. The ESM-1b base checkpoint and the task-specific
fine-tune ``model_ESM_binary_A100_epoch_1_new_split.pkl`` ship only with the authors'
prediction-function archive (Zenodo 8038678) and are unpacked into the mirror under
``data/saved_models/``. Without the fine-tune there is no way to produce the vector the
booster was fit on: feeding a stock ESM-1b mean-pooled embedding into it would run without
error and return numbers that are not TurNuP's.

    python run_turnup_pinned.py --self-test          # reproduce the authors' example
    python run_turnup_pinned.py --inputs in.parquet --output out.parquet --device cuda:2
"""

import argparse
import json
import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
import torch

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/turnup"
ESM1B_DIR = osp.join(MIRROR, "data", "saved_models", "ESM1b")
XGBOOST_MODEL = osp.join(
    MIRROR, "data", "saved_models", "xgboost", "xgboost_train_and_test.pkl"
)
FINE_TUNE = osp.join(ESM1B_DIR, "model_ESM_binary_A100_epoch_1_new_split.pkl")
ESM1B_BASE = osp.join(ESM1B_DIR, "esm1b_t33_650M_UR50S.pt")

# ESM-1b takes 1,024 positions including the BOS and EOS tokens, so 1,022 residues is the
# authors' crop and also the architectural ceiling.
MAX_RESIDUES = 1022
FP_SIZE = 2048
EMBED_DIM = 1280

# The fine-tune was saved from a wrapper that carried a three-layer classification head on
# top of the encoder. Those tensors have no counterpart in the encoder and are dropped, as
# in the authors' own loading code.
HEAD_KEYS = (
    "module.fc1.weight",
    "module.fc1.bias",
    "module.fc2.weight",
    "module.fc2.bias",
    "module.fc3.weight",
    "module.fc3.bias",
)

# The authors' tutorial: one enzyme, one reaction, and the value their notebook printed.
# Reproducing it end to end is what shows the fingerprint, the embedding and the booster
# are all wired the way the released model expects.
EXAMPLE_SUBSTRATES = (
    "InChI=1S/C7H5NO4/c9-8(10)5-1-2-6-7(3-5)12-4-11-6/h1-3H,4H2;"
    "InChI=1S/H2O2/c1-2/h1-2H"
)
EXAMPLE_PRODUCTS = (
    "InChI=1S/C6H5NO4/c8-5-2-1-4(7(10)11)3-6(5)9/h1-3,8-9H;"
    "InChI=1S/CH2O2/c2-1-3/h1H,(H,2,3);InChI=1S/H2O/h1H2"
)
EXAMPLE_ENZYME = (
    "MKYFPLFPTLVFAARVVAFPAYASLAGLSQQELDAIIPTLEAREPGLPPGPLENSSAKLVNDEAHPWKPLRPGDIRGPCP"
    "GLNTLASHGYLPRNGVATPVQIINAVQEGLNFDNQAAVFATYAAHLVDGNLITDLLSIGRKTRLTGPDPPPPASVGGLNE"
    "HGTFEGDASMTRGDAFFGNNHDFNETLFEQLVDYSNRFGGGKYNLTVAGELRFKRIQDSIATNPNFSFVDFRFFTAYGET"
    "TFPANLFVDGRRDDGQLDMDAARSFFQFSRMPDDFFRAPSPRSGTGVEVVIQAHPMQPGRNVGKINSYTVDPTSSDFSTP"
    "CLMYEKFVNITVKSLYPNPTVQLRKALNTNLDFFFQGVAAGCTQVFPYGRD"
)
EXAMPLE_KCAT = 216.114853


def sequence_key(sequence: str) -> str:
    """Stable key for a protein sequence, so the embedding cache is content-addressed."""
    import hashlib

    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def metabolite_smarts(metabolite: str):
    """One metabolite string to SMARTS, or None when rdkit cannot read it.

    SMILES and InChI are both accepted, as in the authors' ``get_reaction_site_smarts``.
    KEGG compound ids are not: resolving one needs the mol-file directory, and every
    metabolite here arrives as a structure already.
    """
    from rdkit import Chem, RDLogger

    # rdkit narrates every valence and hydrogen complaint on stderr; a metabolite that
    # genuinely fails is reported by the return value instead.
    RDLogger.DisableLog("rdApp.*")

    if metabolite.startswith("InChI="):
        mol = Chem.inchi.MolFromInchi(metabolite)
    else:
        mol = Chem.MolFromSmiles(metabolite)
    if mol is None:
        return None
    return Chem.MolToSmarts(mol)


def reaction_site_smarts(metabolites: str):
    """A ';'-joined side of a reaction to one '.'-joined SMARTS, or None on any failure."""
    parts = []
    for metabolite in metabolites.split(";"):
        smarts = metabolite_smarts(metabolite.strip())
        if smarts is None:
            return None
        parts.append(smarts)
    return ".".join(parts) if parts else None


def difference_fingerprint(substrates: str, products: str):
    """The 2,048-dim reaction difference fingerprint, or None when a side is unreadable."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    left = reaction_site_smarts(substrates)
    right = reaction_site_smarts(products)
    if left is None or right is None:
        return None
    reaction = AllChem.ReactionFromSmarts(left + ">>" + right)
    sparse = Chem.rdChemReactions.CreateDifferenceFingerprintForReaction(reaction)
    fingerprint = np.zeros(FP_SIZE)
    for index, count in sparse.GetNonzeroElements().items():
        fingerprint[index] = count
    return fingerprint


def load_esm1b_ts(device: str):
    """The ESM-1b encoder with the task-specific fine-tune loaded over it.

    The base checkpoint supplies the architecture and the alphabet; every weight is then
    replaced by the fine-tune. ``load_state_dict`` runs strict, so a mismatch in shape or
    in key naming fails here rather than producing a silently different embedding.
    """
    import esm

    model_data = torch.load(ESM1B_BASE, map_location="cpu")
    regression_data = torch.load(
        ESM1B_BASE[:-3] + "-contact-regression.pt", map_location="cpu"
    )
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        "esm1b_t33_650M_UR50S", model_data, regression_data
    )
    fine_tune = torch.load(FINE_TUNE, map_location="cpu")
    weights = {key.split("model.")[-1]: value for key, value in fine_tune.items()}
    for key in HEAD_KEYS:
        del weights[key]
    model.load_state_dict(weights)
    model.eval().to(device)
    return model, alphabet.get_batch_converter()


def embed_sequences(sequences: list[str], device: str) -> dict[str, np.ndarray]:
    """ESM1b_ts vectors for a list of protein sequences, keyed by sequence hash.

    The representation is the BOS token of layer 33, not a mean over residues. That is the
    authors' choice and it is not interchangeable with pooling: the booster was fit on
    this vector.
    """
    model, batch_converter = load_esm1b_ts(device)
    vectors: dict[str, np.ndarray] = {}
    for index, sequence in enumerate(sequences):
        cropped = sequence.upper()[:MAX_RESIDUES]
        _, _, tokens = batch_converter([(f"protein_{index}", cropped)])
        with torch.no_grad():
            out = model(tokens.to(device), repr_layers=[33])
        vectors[sequence_key(sequence)] = (
            out["representations"][33][0][0].cpu().numpy().astype(np.float32)
        )
        if index % 50 == 0:
            print(f"  enzyme {index}/{len(sequences)}", flush=True)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return vectors


def read_cache(path: str | None) -> dict[str, np.ndarray]:
    """Whatever embeddings are already on disk, or an empty map."""
    if not path or not osp.exists(path):
        return {}
    stored = np.load(path, allow_pickle=True)
    return dict(zip(stored["keys"].tolist(), stored["vectors"]))


def write_cache(path: str, cache: dict[str, np.ndarray]) -> None:
    """Rewrite the embedding cache, through a temporary file so a crash cannot truncate it."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    keys = list(cache)
    np.savez(
        path + ".tmp.npz",
        keys=np.array(keys, dtype=object),
        vectors=np.stack([cache[k] for k in keys]),
    )
    os.replace(path + ".tmp.npz", path)


def predict(features: np.ndarray) -> np.ndarray:
    """:math:`k_{cat}` in 1/s from the released booster, exponentiating base 10."""
    import xgboost as xgb

    with open(XGBOOST_MODEL, "rb") as handle:
        booster = pickle.load(handle)
    log10_kcat = booster.predict(xgb.DMatrix(features))
    return np.power(10.0, log10_kcat)


def self_test(device: str) -> None:
    """Reproduce the value the authors' tutorial notebook prints for its one example."""
    fingerprint = difference_fingerprint(EXAMPLE_SUBSTRATES, EXAMPLE_PRODUCTS)
    if fingerprint is None:
        raise RuntimeError("the tutorial reaction did not parse; check the rdkit pin")
    vectors = embed_sequences([EXAMPLE_ENZYME], device)
    embedding = vectors[sequence_key(EXAMPLE_ENZYME)]
    print(f"enzyme rep head: {embedding[:4]}")
    features = np.concatenate([fingerprint, embedding]).reshape(1, -1)
    value = float(predict(features)[0])
    print(f"self-test k_cat {value:.6f} 1/s, authors' notebook {EXAMPLE_KCAT:.6f} 1/s")
    print(f"relative difference {abs(value - EXAMPLE_KCAT) / EXAMPLE_KCAT:.3e}")


def main() -> None:
    """Featurize, predict, and write the results parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--output")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--embedding-cache",
        default=None,
        help="npz of ESM1b_ts vectors keyed by sequence hash. Loading the encoder costs "
        "10 GB of checkpoint reads, so it happens once and every later block reuses it.",
    )
    parser.add_argument(
        "--sequences",
        default=None,
        help="json list of every sequence the whole build will need. Passing the superset "
        "means the encoder loads on the first block instead of on all fifteen.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore")

    if args.self_test:
        self_test(args.device)
        return

    rows = pd.read_parquet(args.inputs)

    wanted = list(rows["sequence"].dropna().drop_duplicates())
    if args.sequences:
        with open(args.sequences) as handle:
            wanted = list(dict.fromkeys(json.load(handle)))
    cache = read_cache(args.embedding_cache)
    missing = [s for s in wanted if sequence_key(s) not in cache]
    print(f"{len(cache)} cached embeddings, {len(missing)} to compute", flush=True)
    if missing:
        cache.update(embed_sequences(missing, args.device))
        if args.embedding_cache:
            write_cache(args.embedding_cache, cache)

    # One fingerprint per distinct reaction, not per row: a reaction is shared by every
    # catalytic unit that carries it, and the SMARTS round trip is the slow step.
    usable = rows["substrate_smiles"].notna() & rows["product_smiles"].notna()
    reactions = (
        rows.loc[usable, ["substrate_smiles", "product_smiles"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    fingerprints = {}
    for substrates, products in reactions:
        fingerprints[(substrates, products)] = difference_fingerprint(
            substrates, products
        )
    n_bad = sum(1 for v in fingerprints.values() if v is None)
    print(f"{len(fingerprints)} distinct reactions, {n_bad} unparseable", flush=True)

    values: list[float] = []
    failures: list[str | None] = []
    feature_rows: list[np.ndarray] = []
    feature_index: list[int] = []
    for position, row in enumerate(rows.itertuples()):
        if not usable.iloc[position]:
            values.append(np.nan)
            failures.append(str(row.reaction_failure))
            continue
        fingerprint = fingerprints[(row.substrate_smiles, row.product_smiles)]
        if fingerprint is None:
            values.append(np.nan)
            failures.append("rdkit_reaction_parse_failed")
            continue
        embedding = cache.get(sequence_key(row.sequence))
        if embedding is None:
            values.append(np.nan)
            failures.append("no_enzyme_embedding")
            continue
        values.append(np.nan)
        failures.append(None)
        feature_rows.append(np.concatenate([fingerprint, embedding]))
        feature_index.append(position)

    if feature_rows:
        features = np.stack(feature_rows)
        print(f"feature matrix {features.shape}", flush=True)
        predicted = predict(features)
        for position, value in zip(feature_index, predicted):
            values[position] = float(value)
        print(f"k_cat: median {np.median(predicted):.3f} 1/s", flush=True)

    out = rows.copy()
    out["k_cat"] = values
    out["failure"] = failures
    out.to_parquet(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
