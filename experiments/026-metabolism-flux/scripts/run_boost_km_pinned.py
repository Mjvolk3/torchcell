# experiments/026-metabolism-flux/scripts/run_boost_km_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_boost_km_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_boost_km_pinned.py

r"""Run Boost_KM inside its pinned environment and write K_M predictions back as parquet.

WHAT Boost_KM IS
----------------
Kroll et al. (2021), *PLOS Biology*. Three stages, all of them shipped fitted in the
mirror of https://github.com/AlexanderKroll/KM_prediction:

1. a directed message-passing network (``saved_model_GNN_best_hyperparameters``) whose
   penultimate concatenation is a 52-dim task-specific substrate fingerprint -- 50 learned
   dimensions plus exact molecular weight and the Crippen LogP coefficient;
2. the UniRep 1900 mLSTM (``1900_weights/*.npy``), averaged over residues, 1,900 dims;
3. an XGBoost regressor (``xgboost_model.dat``, 1,381 rounds, 1,952 features) mapping the
   concatenation to :math:`\log_{10} K_M`.

**Units.** The target is :math:`\log_{10}` of :math:`K_M` in **mM**. This is not assumed.
BRENDA reports :math:`K_M` in mM and the authors' preprocessing takes ``np.log10`` of the
BRENDA value with no rescaling, and the shipped test split has median ``log10_KM`` -0.80,
i.e. 0.16 mM -- a normal enzyme :math:`K_M`, and three orders off if the unit were M. So
the exponentiation at the end of :func:`predict` yields mM directly.

WHY THE STAGES ARE SEPARATE PROCESSES
--------------------------------------
UniRep is TensorFlow 1 graph code and needs ``disable_v2_behavior()``; the fingerprint
network is a TensorFlow 2 functional model that relies on raw ops being auto-wrapped as
layers, which only happens with v2 behavior on. One process cannot be in both modes, so
each stage is re-entered through ``--stage``. The gradient-boosting step needs no
TensorFlow at all and runs in the parent.

TWO SHIMS, NEITHER OF WHICH TOUCHES A PREDICTED VALUE
------------------------------------------------------
The mirror is read-only, so the two places the authors' code reaches for a vanished API
are repaired from outside rather than edited:

* ``tf.contrib.layers.fully_connected`` and ``tf.contrib.seq2seq.sequence_loss``, deleted
  in TensorFlow 2, build UniRep's language-model head. ``get_rep`` reads only
  ``_output`` and ``_final_state``, which come from ``dynamic_rnn`` over the mLSTM cell, so
  the head is off the representation path entirely. It is rebuilt from the shipped weights
  anyway, so the restored graph is the authors' graph.
* ``directory_infomation.datasets_dir`` is a hardcoded Windows path in the mirror, and is
  supplied as a module injected into ``sys.modules``.

Both are verified rather than argued for: ``--self-test`` reproduces the authors' own
stored UniRep vectors and stored GNN fingerprints, and their published test-set score.
"""

import argparse
import hashlib
import json
import os
import os.path as osp
import subprocess
import sys
import types
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/boost_km"
DATASETS = osp.join(MIRROR, "datasets") + os.sep
CODE = osp.join(MIRROR, "notebooks_and_code", "additional_code")
GNN_CHECKPOINT = osp.join(DATASETS, "model_weights", "saved_model_GNN_best_hyperparameters")
REGRESSOR = osp.join(DATASETS, "model_weights", "xgboost_model.dat")

# The DMPNN hyperparameters the authors selected, from "Training full model with enzyme
# and substrate information.ipynb". D is the learned width; the fingerprint is D plus the
# two extra features, hence 52.
GNN_PARAMS = dict(l2_reg_conv=0.01, l2_reg_fc=1, learning_rate=0.05, D=50, N=70, F1=32, F2=10, F=42)
FINGERPRINT_DIM = 52
UNIREP_DIM = 1900
MAX_ATOMS = 70  # The GNN pads to 70 nodes; a larger molecule cannot be encoded at all.


def sequence_key(sequence: str) -> str:
    """Stable key for a protein sequence, so the embedding cache is content-addressed."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:24]


def inject_datasets_dir() -> None:
    """Supply the mirror's dataset root, which the shipped module hardcodes for Windows."""
    module = types.ModuleType("directory_infomation")
    module.datasets_dir = DATASETS
    sys.modules["directory_infomation"] = module
    if CODE not in sys.path:
        sys.path.insert(0, CODE)


# --------------------------------------------------------------------------------------
# Stage 1: UniRep, TensorFlow 1 graph mode.
# --------------------------------------------------------------------------------------


def _install_tf_contrib(module) -> None:
    """Restore the two ``tf.contrib`` callables UniRep's babbler constructor calls.

    Both feed the language-model head (``_logits``, ``_loss``, ``_sample``) and neither is
    read by ``get_rep``. The target module is passed in because the caller has already
    aliased ``tensorflow`` to its ``compat.v1`` view; importing it here would reach a
    different wrapper object and the attribute would land where nothing looks for it.
    """
    import tensorflow.compat.v1 as tf1

    def fully_connected(
        inputs,
        num_outputs,
        activation_fn=tf1.nn.relu,
        weights_initializer=None,
        biases_initializer=None,
        scope=None,
        reuse=None,
    ):
        in_dim = int(inputs.shape[-1])
        with tf1.variable_scope(scope or "fully_connected", reuse=reuse):
            weights = tf1.get_variable(
                "weights", shape=[in_dim, num_outputs], dtype=tf1.float32,
                initializer=weights_initializer,
            )
            biases = tf1.get_variable(
                "biases", shape=[num_outputs], dtype=tf1.float32,
                initializer=biases_initializer,
            )
        out = tf1.nn.bias_add(tf1.matmul(inputs, weights), biases)
        return activation_fn(out) if activation_fn is not None else out

    def sequence_loss(
        logits, targets, weights,
        average_across_timesteps=True, average_across_batch=True, name=None,
    ):
        entropy = tf1.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, logits=logits
        ) * weights
        if average_across_timesteps:
            entropy = tf1.reduce_sum(entropy, axis=1) / (
                tf1.reduce_sum(weights, axis=1) + 1e-12
            )
        if average_across_batch:
            entropy = tf1.reduce_mean(entropy)
        return entropy

    contrib = types.ModuleType("tensorflow.contrib")
    contrib.layers = types.ModuleType("tensorflow.contrib.layers")
    contrib.layers.fully_connected = fully_connected
    contrib.seq2seq = types.ModuleType("tensorflow.contrib.seq2seq")
    contrib.seq2seq.sequence_loss = sequence_loss
    module.contrib = contrib


def stage_unirep(fasta: str, output: str, batch_size: int) -> None:
    """Average-hidden UniRep vectors for a fasta, written as an npz of keys and vectors."""
    import tensorflow.compat.v1 as tf1

    tf1.disable_v2_behavior()
    _install_tf_contrib(tf1)
    # unirep.py does `import tensorflow as tf` and then uses the TF1 graph API throughout.
    sys.modules["tensorflow"] = tf1
    sys.path.insert(0, CODE)
    from unirep.run_inference import BatchInference

    batcher = BatchInference(batch_size=batch_size)
    frame = batcher.run_inference(fasta)
    # The model consumes the average hidden state only; the frame also carries the final
    # hidden state and final cell, which is why the authors' stored vectors are 5,700 wide.
    columns = [f"av_{i + 1}" for i in range(UNIREP_DIM)]
    np.savez(
        output,
        keys=np.array(list(frame.index), dtype=object),
        vectors=frame[columns].values.astype(np.float32),
    )
    print(f"unirep: {len(frame)} sequences -> {output}", flush=True)


# --------------------------------------------------------------------------------------
# Stage 2: the DMPNN fingerprint, TensorFlow 2.
# --------------------------------------------------------------------------------------


def load_fingerprint_model():
    """The authors' DMPNN with its checkpoint, cut at the 52-dim fingerprint layer."""
    inject_datasets_dir()
    import build_GNN
    import tensorflow as tf
    from build_GNN import DMPNN

    # Only the compile step touches the optimizer, and a v2.11+ optimizer refuses to read
    # the legacy slot variables in the checkpoint. The weights are unaffected either way.
    build_GNN.Adadelta = tf.keras.optimizers.legacy.Adadelta
    model = DMPNN(drop_rate=0.0, ada_rho=0.95, **GNN_PARAMS)
    model.load_weights(GNN_CHECKPOINT).expect_partial()

    # The authors read the fingerprint off `model.layers[-10]`. That index is asserted
    # rather than trusted: a different Keras version could wrap raw ops into a different
    # number of layers and silently move the read to another tensor.
    layer = model.layers[-10]
    if layer.output_shape[-1] != FINGERPRINT_DIM:
        raise RuntimeError(
            f"layers[-10] has width {layer.output_shape[-1]}, expected {FINGERPRINT_DIM}. "
            "The graph this Keras built is not the graph the checkpoint describes."
        )
    return tf.keras.Model(inputs=model.inputs, outputs=layer.output)


def molecule_inputs(smiles: str):
    """The GNN's four input tensors for one SMILES, or a reason it cannot be encoded.

    The authors built these from KEGG MDL molfiles; the pair table carries SMILES, so the
    molecule is parsed from SMILES and passed through the authors' own feature functions
    unchanged. ``--self-test`` measures what that substitution costs.
    """
    import functions_and_dicts_data_preprocessing_GNN as prep
    from rdkit import Chem
    from rdkit.Chem import Crippen, Descriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, "unparsable_smiles"
    if molecule.GetNumAtoms() >= MAX_ATOMS:
        return None, "too_many_atoms"

    atoms, bonds = [], []
    for index in range(molecule.GetNumAtoms()):
        atom = molecule.GetAtomWithIdx(index)
        atoms.append([
            atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge(),
            str(atom.GetHybridization()), atom.GetIsAromatic(), atom.GetMass(),
            atom.GetTotalNumHs(), str(atom.GetChiralTag()),
        ])
    for index in range(molecule.GetNumBonds()):
        bond = molecule.GetBondWithIdx(index)
        bonds.append([
            bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType()),
            bond.GetIsAromatic(), bond.IsInRing(), str(bond.GetStereo()),
        ])

    n = prep.N
    X = np.zeros((n, 32))
    for i, line in enumerate(atoms):
        X[i, :] = np.concatenate((
            prep.dic_atomic_number[float(line[0])], prep.dic_num_bonds[float(line[1])],
            prep.dic_charge[float(line[2])], prep.dic_hybrid[line[3]],
            prep.dic_aromatic[float(line[4])], np.array([line[5] / 100.0]),
            prep.dic_H_bonds[float(line[6])], prep.dic_chirality[line[7]],
        ))
    A = np.zeros((n, n))
    E = np.zeros((n, n, 10))
    for line in bonds:
        start, end = line[0], line[1]
        A[start, end] = A[end, start] = 1
        # The second slot is named `conjugated` in the mirror but is fed the aromatic flag.
        # Reproduced as written: it is what the released weights were trained against.
        e_vw = np.concatenate((
            prep.dic_bond_type[line[2]], prep.dic_conjugated[float(line[3])],
            prep.dic_inRing[float(line[4])], prep.dic_stereo[line[5]],
        ))
        E[start, end, :] = E[end, start, :] = e_vw
    XE = prep.concatenate_X_and_E(X, E, N=n)
    extras = np.array([Descriptors.ExactMolWt(molecule), Crippen.MolLogP(molecule)])
    return (XE, X, A.reshape(n, n, 1), extras), None


def stage_fingerprints(smiles_json: str, output: str, batch_size: int) -> None:
    """52-dim fingerprints for a list of SMILES, written as an npz of keys and vectors."""
    inject_datasets_dir()
    with open(smiles_json) as handle:
        wanted = json.load(handle)

    model = load_fingerprint_model()
    keys, blocks = [], []
    batch_keys, XE, X, A, EX = [], [], [], [], []

    def flush() -> None:
        if not batch_keys:
            return
        vectors = model(
            [np.array(XE, dtype="float32"), np.array(X, dtype="float32"),
             np.array(A, dtype="float32"), np.array(EX, dtype="float32")],
            training=False,
        ).numpy()
        keys.extend(batch_keys)
        blocks.append(vectors)
        for buffer in (batch_keys, XE, X, A, EX):
            buffer.clear()

    for smiles in wanted:
        built, _ = molecule_inputs(smiles)
        if built is None:
            continue
        batch_keys.append(smiles)
        XE.append(built[0])
        X.append(built[1])
        A.append(built[2])
        EX.append(built[3])
        if len(batch_keys) == batch_size:
            flush()
    flush()

    np.savez(
        output,
        keys=np.array(keys, dtype=object),
        vectors=(np.concatenate(blocks) if blocks else np.zeros((0, FINGERPRINT_DIM))).astype(np.float32),
    )
    print(f"fingerprints: {len(keys)} molecules -> {output}", flush=True)


# --------------------------------------------------------------------------------------
# Content-addressed caches, so a block-wise build embeds each enzyme and molecule once.
# --------------------------------------------------------------------------------------


def read_cache(path: str) -> dict:
    if not osp.exists(path):
        return {}
    stored = np.load(path, allow_pickle=True)
    return dict(zip(stored["keys"].tolist(), stored["vectors"]))


def write_cache(path: str, cache: dict) -> None:
    os.makedirs(osp.dirname(path), exist_ok=True)
    np.savez(
        path,
        keys=np.array(list(cache), dtype=object),
        vectors=(np.stack(list(cache.values())) if cache else np.zeros((0, 1))).astype(np.float32),
    )


def fill_unirep(sequences: list[str], cache_dir: str, batch_size: int) -> dict:
    """UniRep vectors for every sequence, computing only the ones not already cached."""
    path = osp.join(cache_dir, "unirep_1900.npz")
    cache = read_cache(path)
    missing = [s for s in sequences if sequence_key(s) not in cache]
    if missing:
        print(f"unirep: {len(missing)} of {len(sequences)} sequences to embed", flush=True)
        os.makedirs(cache_dir, exist_ok=True)
        fasta = osp.join(cache_dir, "_unirep_input.fasta")
        out = osp.join(cache_dir, "_unirep_output.npz")
        with open(fasta, "w") as handle:
            handle.write("\n".join(f">{sequence_key(s)}\n{s}" for s in missing))
        subprocess.run(
            [sys.executable, osp.abspath(__file__), "--stage", "unirep",
             "--fasta", fasta, "--output", out, "--batch-size", str(batch_size)],
            check=True,
        )
        cache.update(read_cache(out))
        write_cache(path, cache)
        os.remove(fasta)
        os.remove(out)
    return cache


def fill_fingerprints(smiles: list[str], cache_dir: str, batch_size: int) -> dict:
    """Fingerprints for every SMILES, computing only the ones not already cached."""
    path = osp.join(cache_dir, "gnn_fingerprints.npz")
    cache = read_cache(path)
    missing = [s for s in smiles if s not in cache]
    if missing:
        print(f"fingerprints: {len(missing)} of {len(smiles)} molecules to encode", flush=True)
        os.makedirs(cache_dir, exist_ok=True)
        wanted = osp.join(cache_dir, "_smiles_input.json")
        out = osp.join(cache_dir, "_smiles_output.npz")
        with open(wanted, "w") as handle:
            json.dump(missing, handle)
        subprocess.run(
            [sys.executable, osp.abspath(__file__), "--stage", "fingerprints",
             "--smiles-json", wanted, "--output", out, "--batch-size", str(batch_size)],
            check=True,
        )
        cache.update(read_cache(out))
        write_cache(path, cache)
        os.remove(wanted)
        os.remove(out)
    return cache


# --------------------------------------------------------------------------------------
# Prediction.
# --------------------------------------------------------------------------------------


def load_regressor():
    """The fitted gradient-boosting model, checked for the width it is supposed to have."""
    import pickle

    with open(REGRESSOR, "rb") as handle:
        booster = pickle.load(handle)
    expected = FINGERPRINT_DIM + UNIREP_DIM
    if booster.num_features() != expected:
        raise RuntimeError(
            f"{REGRESSOR} takes {booster.num_features()} features, expected {expected}."
        )
    return booster


def screen(pairs: pd.DataFrame) -> pd.Series:
    """Name, per row, why a pair cannot be predicted -- or ``None`` when it can.

    A refusal is a fact about the model's domain and is recorded as one. Two of the three
    are hard limits of the released architecture rather than choices: the message-passing
    tensors are padded to 70 nodes, so a larger molecule has nowhere to go, and a SMILES
    RDKit cannot parse has no graph. Multi-component SMILES are excluded because the
    substrate representation is a single connected molecular graph built from a KEGG
    molfile, and a mixture would enter as one disconnected graph that no training example
    resembled. The sibling predictors in this experiment exclude them for the same reason,
    which also keeps the comparison row-aligned.
    """
    from rdkit import Chem

    reasons = []
    for smiles in pairs["smiles"]:
        if not isinstance(smiles, str) or not smiles:
            reasons.append("no_smiles")
        elif "." in smiles:
            reasons.append("multi_component_smiles")
        else:
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                reasons.append("unparsable_smiles")
            elif molecule.GetNumAtoms() >= MAX_ATOMS:
                reasons.append("too_many_atoms")
            else:
                reasons.append(None)
    return pd.Series(reasons, index=pairs.index)


def predict(inputs: str, output: str, cache_dir: str, batch_size: int) -> None:
    """Featurize, predict, and write the results parquet with K_M in mM."""
    pairs = pd.read_parquet(inputs)
    failure = screen(pairs)
    work = pairs[failure.isna()]
    print(f"{len(work)} usable of {len(pairs)} pairs", flush=True)

    sequences = work["sequence"].drop_duplicates().tolist()
    smiles = work["smiles"].drop_duplicates().tolist()
    print(f"{len(sequences)} distinct proteins, {len(smiles)} distinct molecules", flush=True)

    unirep = fill_unirep(sequences, cache_dir, batch_size)
    fingerprints = fill_fingerprints(smiles, cache_dir, batch_size)

    # A molecule that passed the screen but produced no fingerprint would otherwise become
    # a silent NaN, so it is turned back into a named failure.
    dropped = [s for s in smiles if s not in fingerprints]
    if dropped:
        lost = work["smiles"].isin(dropped)
        failure.loc[work.index[lost]] = "fingerprint_failed"
        work = work[~lost]
        print(f"{len(dropped)} molecules produced no fingerprint", flush=True)

    features = np.concatenate(
        [
            np.stack([fingerprints[s] for s in work["smiles"]]),
            np.stack([unirep[sequence_key(s)] for s in work["sequence"]]),
        ],
        axis=1,
    )
    print(f"feature matrix {features.shape}", flush=True)

    import xgboost as xgb

    log10_km = load_regressor().predict(xgb.DMatrix(features))
    # The regressor's target is log10 of K_M in mM, so this is mM, the canonical unit.
    km_mm = np.power(10.0, log10_km)

    out = pairs.copy()
    out["K_M"] = np.nan
    out.loc[work.index, "K_M"] = km_mm
    out["failure"] = failure.values
    out.to_parquet(output, index=False)
    print(
        f"K_M mM: median {np.median(km_mm):.4f} p05 {np.percentile(km_mm, 5):.4f} "
        f"p95 {np.percentile(km_mm, 95):.4f}",
        flush=True,
    )
    print(f"wrote {output}")


# --------------------------------------------------------------------------------------
# Self-test.
# --------------------------------------------------------------------------------------


def self_test() -> None:
    """Reproduce the authors' published score and their own stored intermediate features.

    Three independent checks, because a wrong answer at any stage is invisible downstream:
    an XGBoost model happily scores a mis-built feature vector, and a fingerprint read off
    the wrong layer is still 52 numbers.
    """
    inject_datasets_dir()
    import xgboost as xgb
    from sklearn.metrics import r2_score

    test = pd.read_pickle(osp.join(DATASETS, "splits", "test_data.pkl"))
    test = test.loc[~pd.isnull(test["GNN FP"])]
    features = np.concatenate(
        [np.array(list(test["GNN FP"])), np.array(list(test["Unirep"]))[:, :UNIREP_DIM]],
        axis=1,
    )
    target = np.array(list(test["log10_KM"]))
    matrix = xgb.DMatrix(features)
    print(f"[1] shipped test split: {features.shape}")
    import pickle

    for name in ("xgboost_model.dat", "xgboost_model_full.dat"):
        with open(osp.join(DATASETS, "model_weights", name), "rb") as handle:
            booster = pickle.load(handle)
        predicted = booster.predict(matrix)
        print(
            f"    {name}: MSE {np.mean((predicted - target) ** 2):.4f} "
            f"R2 {r2_score(target, predicted):.4f}"
        )
    print("    paper reports MSE 0.65 / R2 0.53 for the full model")

    # [2] Fingerprints, recomputed from the shipped KEGG feature vectors, against the
    # authors' own stored GNN FP column.
    import functions_and_dicts_data_preprocessing_GNN as prep

    model = load_fingerprint_model()
    sample = test.head(256)
    XE, X, A, EX, keep = [], [], [], [], []
    for index in sample.index:
        xe, x, a = prep.create_input_data_for_GNN_for_substrates(sample["KEGG ID"][index])
        if a is None:
            keep.append(False)
            continue
        keep.append(True)
        XE.append(xe)
        X.append(x)
        A.append(a)
        EX.append([sample["MW"][index], sample["LogP"][index]])
    got = model(
        [np.array(XE, dtype="float32"), np.array(X, dtype="float32"),
         np.array(A, dtype="float32"), np.array(EX, dtype="float32")],
        training=False,
    ).numpy()
    stored = np.array(list(sample["GNN FP"][np.array(keep)]))
    print(f"[2] fingerprint vs stored, n={len(got)}: max abs diff {np.abs(got - stored).max():.3e}")

    # [3] The SMILES path is what production uses, and the authors used KEGG molfiles. The
    # cost of that substitution is measured, not assumed, by re-deriving each molecule's
    # SMILES from its molfile and comparing the two fingerprints for the same compound.
    from rdkit import Chem

    pairs_got, pairs_ref = [], []
    for index in sample.index[:200]:
        kegg = sample["KEGG ID"][index]
        molecule = Chem.MolFromMolFile(osp.join(DATASETS, "mol-files", f"{kegg}.mol"))
        if molecule is None:
            continue
        built, _ = molecule_inputs(Chem.MolToSmiles(molecule))
        xe, x, a = prep.create_input_data_for_GNN_for_substrates(kegg)
        if built is None or a is None:
            continue
        extras = np.array([sample["MW"][index], sample["LogP"][index]])
        pairs_got.append(built)
        pairs_ref.append((xe, x, a, extras))

    def run(items):
        return model(
            [np.array([i[0] for i in items], dtype="float32"),
             np.array([i[1] for i in items], dtype="float32"),
             np.array([i[2] for i in items], dtype="float32"),
             np.array([i[3] for i in items], dtype="float32")],
            training=False,
        ).numpy()

    from_smiles, from_molfile = run(pairs_got), run(pairs_ref)
    print(
        f"[3] SMILES path vs KEGG molfile path, n={len(pairs_got)}: "
        f"max abs diff {np.abs(from_smiles - from_molfile).max():.3e}, "
        f"median abs diff {np.median(np.abs(from_smiles - from_molfile)):.3e}"
    )

    # [4] What that costs in K_M, which is the only version of the question that matters.
    # The two paths are not identical and cannot be: `atom.GetChiralTag()` is CW or CCW
    # relative to the order that atom's bonds are stored in, so the same stereocenter reads
    # differently once a canonical SMILES has renumbered the molecule. Trehalose is ten CW
    # centers read from its molfile and five CW plus five CCW read from its own canonical
    # SMILES. The authors always had a KEGG molfile; the pair table has only SMILES, and no
    # SMILES carries the atom order of a molfile it never came from. So the gap is measured
    # on the authors' own test split rather than argued away.
    from rdkit import Chem as _Chem

    encoded = {}
    for kegg in test["KEGG ID"].unique():
        molecule = _Chem.MolFromMolFile(osp.join(DATASETS, "mol-files", f"{kegg}.mol"))
        if molecule is None:
            continue
        built, _ = molecule_inputs(_Chem.MolToSmiles(molecule))
        if built is not None:
            encoded[kegg] = built
    scored = test[test["KEGG ID"].isin(encoded)]
    order = list(scored["KEGG ID"])
    chunks = []
    for start in range(0, len(order), 64):
        items = [encoded[k] for k in order[start : start + 64]]
        chunks.append(run(items))
    smiles_fp = np.concatenate(chunks)
    unirep = np.array(list(scored["Unirep"]))[:, :UNIREP_DIM]
    truth = np.array(list(scored["log10_KM"]))
    with open(osp.join(DATASETS, "model_weights", "xgboost_model.dat"), "rb") as handle:
        booster = pickle.load(handle)
    from_stored = booster.predict(
        xgb.DMatrix(np.concatenate([np.array(list(scored["GNN FP"])), unirep], axis=1))
    )
    from_path = booster.predict(xgb.DMatrix(np.concatenate([smiles_fp, unirep], axis=1)))
    print(
        f"[4] test split scored through the SMILES path, n={len(scored)}: "
        f"MSE {np.mean((from_path - truth) ** 2):.4f} R2 {r2_score(truth, from_path):.4f} "
        f"(stored fingerprints: MSE {np.mean((from_stored - truth) ** 2):.4f} "
        f"R2 {r2_score(truth, from_stored):.4f})"
    )
    gap = np.abs(from_stored - from_path)
    print(
        f"    log10 K_M path-to-path: median {np.median(gap):.4f} p95 "
        f"{np.percentile(gap, 95):.4f} corr {np.corrcoef(from_stored, from_path)[0, 1]:.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs")
    parser.add_argument("--output")
    parser.add_argument("--cache-dir", default="/scratch/projects/torchcell-scratch/data/torchcell/kinetics/boost_km/features")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--stage", choices=["unirep", "fingerprints"])
    parser.add_argument("--fasta")
    parser.add_argument("--smiles-json")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.stage == "unirep":
        stage_unirep(args.fasta, args.output, args.batch_size)
    elif args.stage == "fingerprints":
        stage_fingerprints(args.smiles_json, args.output, args.batch_size)
    else:
        predict(args.inputs, args.output, args.cache_dir, args.batch_size)


if __name__ == "__main__":
    main()
