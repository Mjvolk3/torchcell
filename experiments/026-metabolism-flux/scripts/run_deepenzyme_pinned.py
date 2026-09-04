# experiments/026-metabolism-flux/scripts/run_deepenzyme_pinned.py
# [[experiments.026-metabolism-flux.scripts.run_deepenzyme_pinned]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/026-metabolism-flux/scripts/run_deepenzyme_pinned.py

r"""Run DeepEnzyme in a child process over (enzyme, substrate, structure) triples.

Wang et al. (2024), *Briefings in Bioinformatics*. DeepEnzyme is the structure-informed
member of the predictor set: the substrate is a Weisfeiler-Lehman fingerprint graph, the
enzyme is overlapping 4-grams of its sequence read two ways -- through a transformer
encoder, and through a GCN over the residue contact map taken from an AlphaFold structure
at a 10 Angstrom C-alpha cutoff. The three pooled vectors are concatenated and mapped to a
scalar. The network is imported from the mirror, never transcribed.

THE OUTPUT IS log2, NOT log10
------------------------------
Determined three ways, not assumed. The authors' ``Code/Example/example.py`` converts with
``math.pow(2, prediction)``; ``Code/Model/train.py`` carries commented reporting lines
``math.log10(math.pow(2, ...))``, which only makes sense if the stored target is log2; and
the released training targets in ``Data/Input/logkcat_0612.npy`` have median 1.758 and p95
9.229, so 2 to those powers is 3.4 and 600 per second, while 10 to those powers would put
the 95th percentile at 1.7e9 per second. The conversion happens here, once.

WHY A CHILD PROCESS RATHER THAN AN IMPORT
-------------------------------------------
Not a dependency conflict. ``apex`` appears in the authors' ``requirements.txt`` but is
imported nowhere in ``Code/``, and the network runs unmodified on this project's torch,
numpy and rdkit; the authors' own example reproduces in the torchcell environment. The
separation is for two other reasons that are still real. The network hardcodes ``.cuda()``
in four places, so the only way to choose a GPU is ``CUDA_VISIBLE_DEVICES`` before torch
initializes, which a caller that has already imported torch cannot do. And the mirror's
module is named ``example_model`` at the top level of ``sys.path``, which is not a name to
introduce into a long-lived process. Parquet on each side is the interface, the same shape
``run_unikp_pinned.py`` uses.

THE ONE INPUT-SIDE DEPARTURE FROM THE AUTHORS' CODE, AND WHY
--------------------------------------------------------------
``get_ca_coords`` in the authors' example splits each PDB line on whitespace and reads the
chain from field 4. PDB is a fixed-column format, so at residue 1000 the chain and residue
number run together as ``A1000``, the field-4 test fails, and every residue from 1000 on is
dropped. On yeast that silently truncates 34 of 1,151 enzymes -- the largest, MDN1
(YLR106C) and Ynr016c among them -- to their first 999 residues, after which the authors'
own padding branch fills the remainder with an identity diagonal and no contacts. This
module reads the columns the PDB specification defines instead. Nothing about the model,
its weights, or its forward pass changes; the contact map is simply built from all the
residues in the file rather than the first 999. ``n_ca`` is written per row so the effect
stays auditable.
"""

import argparse
import json
import os
import os.path as osp
import pickle
import sys
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch
from sklearn.metrics import pairwise_distances

MIRROR = "/scratch/projects/torchcell-scratch/models/kinetics/deepenzyme"
STRUCTURES = (
    "/scratch/projects/torchcell-scratch/data/enzyme_kinetics/alphafold/scerevisiae"
)
WEIGHTS = "Weights/example"

# Fixed by the released checkpoint, not free choices. They are the values in the authors'
# example and in the checkpoint filename dim64_lr001_E100_head4_drop3_0612_seed666;
# changing any of them fails to load or silently reshapes an embedding table.
DIM = 64
LAYER_OUTPUT = 3
HIDDEN_DIM1 = 64
HIDDEN_DIM2 = 128
DROPOUT = 0.0
NHEAD = 4
HID_SIZE = 64
LAYERS_TRANS = 3
RADIUS = 2
NGRAM = 4
CONTACT_THRESHOLD = 10.0


def load_pickle(path: str) -> Any:
    """Read one of the frozen vocabularies."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_sequence(sequence: str, word_dict: dict[str, int]) -> tuple[np.ndarray, int]:
    """Overlapping 4-grams as vocabulary indices, plus how many were unseen.

    The authors' ``split_sequence`` writes a missing key into ``word_dict`` with value 0
    and continues. Index 0 is a legitimate learned embedding, so an unseen 4-gram is
    predicted as whatever index 0 means. This produces the same indices without mutating
    the vocabulary, and counts the misses, because the fraction of a protein that was
    never seen in training is the confidence signal DeepEnzyme does not otherwise expose.
    """
    padded = "--" + sequence + "="
    words, unseen = [], 0
    for i in range(len(padded) - NGRAM + 1):
        key = padded[i : i + NGRAM]
        if key in word_dict:
            words.append(word_dict[key])
        else:
            words.append(0)
            unseen += 1
    return np.array(words), unseen


def create_atoms(mol: Any, atom_dict: dict[Any, int]) -> np.ndarray:
    """Atom-type indices, aromatic atoms typed separately, as the authors do."""
    atoms: list[Any] = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        atoms[a.GetIdx()] = (atoms[a.GetIdx()], "aromatic")
    return np.array([atom_dict[a] for a in atoms])


def create_ijbonddict(
    mol: Any, bond_dict: dict[str, int]
) -> dict[int, list[tuple[int, int]]]:
    """Adjacency as an atom-index to (neighbor, bond-type) map."""
    i_jbond: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b = bond_dict[str(bond.GetBondType())]
        i_jbond[i].append((j, b))
        i_jbond[j].append((i, b))
    return i_jbond


def extract_fingerprints(
    atoms: np.ndarray,
    i_jbond_dict: dict[int, list[tuple[int, int]]],
    fingerprint_dict: dict[Any, int],
    edge_dict: dict[Any, int],
) -> tuple[np.ndarray, int]:
    """Weisfeiler-Lehman subgraph fingerprints, plus how many were unseen.

    The authors' function with the silent-miss branches counted rather than swallowed, and
    without writing new keys into the frozen dictionaries.
    """
    unseen = 0
    if len(atoms) == 1 or RADIUS == 0:
        out = []
        for a in atoms:
            if a in fingerprint_dict:
                out.append(fingerprint_dict[a])
            else:
                out.append(0)
                unseen += 1
        return np.array(out), unseen

    nodes: list[int] = list(atoms)
    i_jedge = i_jbond_dict
    fingerprints: list[int] = []
    for _ in range(RADIUS):
        fingerprints = []
        for i, j_edge in i_jedge.items():
            neighbors = [(nodes[j], edge) for j, edge in j_edge]
            key = (nodes[i], tuple(sorted(neighbors)))
            if key in fingerprint_dict:
                fingerprints.append(fingerprint_dict[key])
            else:
                fingerprints.append(0)
                unseen += 1
        nodes = fingerprints
        next_edge: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, j_edge in i_jedge.items():
            for j, edge in j_edge:
                both = tuple(sorted((nodes[i], nodes[j])))
                next_edge[i].append((j, edge_dict.get((both, edge), 0)))
        i_jedge = next_edge
    return np.array(fingerprints), unseen


def ca_coordinates(pdb_path: str) -> np.ndarray:
    """C-alpha coordinates of chain A, read by PDB column positions.

    Columns 13-16 are the atom name, 22 the chain identifier, and 31-38 / 39-46 / 47-54
    the coordinates, per the PDB specification. See the module docstring for why this
    replaces the authors' whitespace split.
    """
    coordinates = []
    with open(pdb_path) as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[12:16] != " CA " or line[21] != "A":
                continue
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
    return np.asarray(coordinates, dtype=float)


def contact_map(pdb_path: str, sequence: str) -> tuple[sparse.csr_matrix, int]:
    """Residue contact map at a 10 Angstrom C-alpha cutoff, shaped to the sequence.

    Returns the map and the number of C-alpha atoms it was built from. When the structure
    is shorter than the sequence the authors pad with zeros and set the diagonal, which is
    kept verbatim. When it is longer -- four yeast enzymes whose UniProt canonical
    sequence disagrees with the genome-derived ORF -- the map is cropped to the sequence,
    since the alternative is a shape error rather than a prediction.
    """
    coordinates = ca_coordinates(pdb_path)
    n_ca = int(coordinates.shape[0])
    contacts = (pairwise_distances(coordinates) < CONTACT_THRESHOLD).astype(float)
    length = len(sequence)
    if n_ca == length:
        return sparse.csr_matrix(contacts), n_ca
    if n_ca > length:
        return sparse.csr_matrix(contacts[:length, :length]), n_ca
    padded = np.zeros((length, length), dtype=float)
    padded[:n_ca, :n_ca] = contacts
    row, col = np.diag_indices_from(padded)
    padded[row, col] = 1.0
    return sparse.csr_matrix(padded), n_ca


def build_model(n_fingerprint: int, n_word: int, weights_path: str) -> Any:
    """Construct the authors' network from the mirror and load the released checkpoint."""
    sys.path.insert(0, MIRROR)
    sys.path.insert(0, osp.join(MIRROR, "Code", "Example"))
    # The authors' module, resolved out of the mirror at runtime rather than vendored.
    from example_model import DeepEnzyme  # type: ignore[import-not-found]

    model = DeepEnzyme(
        n_fingerprint,
        DIM,
        n_word,
        LAYER_OUTPUT,
        HIDDEN_DIM1,
        HIDDEN_DIM2,
        DROPOUT,
        NHEAD,
        HID_SIZE,
        LAYERS_TRANS,
    ).cuda()
    state = torch.load(weights_path, map_location="cuda", weights_only=False)
    # The authors pass strict=False. That tolerates a missing or extra key, so the load is
    # checked here instead: a silently skipped tensor would leave a randomly initialized
    # layer in the forward pass and still return a plausible number.
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint does not match the network: missing {result.missing_keys}, "
            f"unexpected {result.unexpected_keys}"
        )
    model.train(False)
    return model


def main() -> None:
    """Predict k_cat for every usable pair and write the results parquet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--structures", default=STRUCTURES)
    parser.add_argument("--mirror", default=MIRROR)
    args = parser.parse_args()

    from rdkit import Chem, RDLogger

    # rdkit narrates every valence complaint on stderr; the failures that matter are
    # captured per row below.
    RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]

    dictionaries = osp.join(args.mirror, "Data", "Input")
    atom_dict = load_pickle(osp.join(dictionaries, "atom_dict_0612.pickle"))
    bond_dict = load_pickle(osp.join(dictionaries, "bond_dict_0612.pickle"))
    edge_dict = load_pickle(osp.join(dictionaries, "edge_dict_0612.pickle"))
    fingerprint_dict = load_pickle(
        osp.join(dictionaries, "fingerprint_dict_0612.pickle")
    )
    word_dict = load_pickle(osp.join(dictionaries, "sequence_dict_0612.pickle"))

    model = build_model(
        len(fingerprint_dict), len(word_dict), osp.join(args.mirror, WEIGHTS)
    )

    pairs = pd.read_parquet(args.inputs)
    n = len(pairs)
    values: list[float] = [np.nan] * n
    failures: list[str | None] = [None] * n
    unseen_fracs: list[float] = [np.nan] * n
    n_cas: list[float] = [np.nan] * n

    # Featurize each distinct molecule once. The pair table repeats a substrate across
    # every enzyme that acts on it.
    molecule_cache: dict[str, Any] = {}

    # Rows are walked grouped by enzyme so a contact map -- an L-by-L pairwise distance
    # matrix -- is built once per protein rather than once per substrate it acts on.
    order = pairs.reset_index(drop=True).sort_values(["uniprot"]).index
    protein: tuple[str, Any, int] | None = None

    for count, position in enumerate(order):
        row = pairs.iloc[position]
        smiles = row["smiles"]
        sequence = row["sequence"]

        if not isinstance(smiles, str) or not smiles:
            failures[position] = "no_smiles"
            continue
        if "." in smiles:
            # A disconnected fragment has no entry in the bond map, so on the second
            # Weisfeiler-Lehman round the authors' ``nodes`` list is shorter than the atom
            # indices that index it and the featurization walks off the end. DLKcat's own
            # prediction script refuses these for the same reason, and refusing them here
            # keeps the two predictors' coverage directly comparable.
            failures[position] = "multi_component_smiles"
            continue

        if smiles in molecule_cache:
            cached = molecule_cache[smiles]
        else:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                molecule_cache[smiles] = ("rdkit_parse_failed", None)
                cached = molecule_cache[smiles]
            else:
                # Hydrogens are added because the training featurization added them; the
                # same molecule yields different fingerprints without them.
                mol = Chem.AddHs(mol)
                try:
                    atoms = create_atoms(mol, atom_dict)
                except KeyError:
                    molecule_cache[smiles] = ("unseen_atom_type", None)
                    cached = molecule_cache[smiles]
                else:
                    i_jbond = create_ijbonddict(mol, bond_dict)
                    fingerprints, fp_unseen = extract_fingerprints(
                        atoms, i_jbond, fingerprint_dict, edge_dict
                    )
                    if fingerprints.size == 0:
                        molecule_cache[smiles] = ("empty_fingerprint", None)
                    else:
                        molecule_cache[smiles] = (
                            None,
                            (
                                torch.LongTensor(fingerprints).cuda(),
                                torch.FloatTensor(Chem.GetAdjacencyMatrix(mol)).cuda(),
                                int(fingerprints.size),
                                fp_unseen,
                            ),
                        )
                    cached = molecule_cache[smiles]

        reason, molecule = cached
        if reason is not None:
            failures[position] = reason
            continue

        uniprot = row["uniprot"]
        if protein is None or protein[0] != uniprot:
            pdb_path = osp.join(args.structures, f"AF-{uniprot}-F1-model_v6.pdb")
            if not osp.exists(pdb_path):
                protein = (uniprot, None, 0)
            else:
                adjacency, n_ca = contact_map(pdb_path, sequence)
                words, word_unseen = split_sequence(sequence, word_dict)
                protein = (
                    uniprot,
                    (
                        torch.LongTensor(words).cuda(),
                        adjacency,
                        int(words.size),
                        word_unseen,
                    ),
                    n_ca,
                )
        if protein[1] is None:
            failures[position] = "no_structure"
            continue

        word_tensor, protein_adjacency, n_words, word_unseen = protein[1]
        n_cas[position] = protein[2]
        fingerprint_tensor, smiles_adjacency, n_fp, fp_unseen = molecule

        with torch.no_grad():
            log2_kcat = model.forward(
                [fingerprint_tensor, smiles_adjacency, word_tensor, protein_adjacency],
                LAYER_OUTPUT,
                DROPOUT,
            ).item()
        # BASE 2. See the module docstring: this is the authors' own conversion, and
        # reading it as log10 would inflate every value by a power of about 3.3 while
        # still looking like a plausible turnover number.
        values[position] = float(2.0**log2_kcat)
        total = n_fp + n_words
        unseen_fracs[position] = (fp_unseen + word_unseen) / total if total else np.nan

        if (count + 1) % 500 == 0:
            print(f"  {count + 1}/{n}", flush=True)

    out = pd.DataFrame(
        {
            "k_cat": values,
            "failure": failures,
            "unseen_token_frac": unseen_fracs,
            "n_ca": n_cas,
        },
        index=pairs.index,
    )
    out.to_parquet(args.output, index=False)
    finite = out["k_cat"].dropna()
    print(
        json.dumps(
            {
                "rows": n,
                "predicted": int(len(finite)),
                "median_kcat": float(np.median(finite)) if len(finite) else None,
                "failures": {
                    str(k): int(v) for k, v in out["failure"].value_counts().items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        # The network calls .cuda() with no device argument in four places, so the GPU is
        # chosen by masking rather than by an argument. Refusing to guess is deliberate:
        # this box has four GPUs and other work runs on them.
        raise SystemExit(
            "set CUDA_VISIBLE_DEVICES before running; DeepEnzyme hardcodes .cuda()"
        )
    main()
