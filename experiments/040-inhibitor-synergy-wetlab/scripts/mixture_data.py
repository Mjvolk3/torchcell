# experiments/040-inhibitor-synergy-wetlab/scripts/mixture_data.py
# [[experiments.040-inhibitor-synergy-wetlab.scripts.mixture_data]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/040-inhibitor-synergy-wetlab/scripts/mixture_data
"""The data the dose- and mixture-aware trainer reads: gene-level sources and host wells.

TWO TASKS SHARE ONE COMPOUND REPRESENTATION.

1. THE GENE-LEVEL CHEMOGENOMIC TASK. One or more pooled-screen sources from build 002 of
   the 033 cell table. A CONDITION is one compound at one served dose; a CELL is one
   deletion strain in one condition. Each source is a genes-by-conditions matrix.

   ``vanacloig``        Vanacloig 2022, haploid barcoded deletions on the PDR1/PDR3/SNQ2
                        sensitized host, anaerobic SynBase, 48 h, 32 published compounds
                        (``EXCLUDED_CONDITIONS`` drops DMSO and MBO as experiment 038
                        does). Its served ``conc_values`` are empty, so the dose comes
                        from the rebuilt dev store (:func:`vanacloig_doses`).
   ``hoepfner_hop``     Hoepfner 2014's HOMOZYGOUS arm (``barcoded_kanmx_deletion``,
                        copy number 0 of 2), YPD, 30 C, adjusted MADL sensitivity, every
                        served (compound, concentration) its own condition.
   ``hillenmeyer_het``  Hillenmeyer 2008's heterozygous arm, log2 fitness-defect ratio.

   ORIENTATION TO SICK-NEGATIVE IS PER SOURCE AND HAPPENS BEFORE ANYTHING IS POOLED
   (the 031 rule). Vanacloig's log2(inhibitor / control) and Hoepfner's MADL score are
   already negative when the deletion is sick; Hillenmeyer's HET value is
   ``log2(mean control intensity / treatment intensity)``, a fitness DEFECT, so it is
   negated. Each source is then standardized on its own fitted conditions only, and
   carries a learned source token into the readout as experiments 030 and 031 did.

2. THE HOST-GROWTH TASK. The bAID host (BY4742-iAID6) carries no deletion, so a host
   record is a condition with no genotype.

   PUBLIC ANCHORS: every compound whose served dose is an unambiguous IC30 is an anchor
   at host fitness 0.70 (30 percent growth inhibition is what IC30 names), and the
   compound-free medium is an anchor at 1.0. A Hoepfner compound served at several
   concentrations has no unambiguous IC30 among them, so it is NOT anchored.
   PRIVATE WELLS: ``results/wetlab_wells.csv`` (``wetlab_table.py``), served growth call
   primary. ``ex21`` is the six single-agent titrations (162 inhibited wells over 54
   compound-dose cells, plus 18 uninhibited wells); ``ex23`` is 63 fixed-dose
   combinations at 85 h; ``ex26`` to ``ex28`` are pairwise isobole grids, 81 interior
   cells each. ex23 and the isoboles are NEVER trained on: they are the test.

DOSES ENTER AS LOG10 MOLAR. mM, uM and ug/mL convert (ug/mL through the RDKit molecular
weight of the curated-identity SMILES); a percent dose states no basis or density, so it
does not convert, and such a compound is dropped from the panel when the trainer asks for
molar doses (``require_molar_dose``) and counted in :func:`vanacloig_doses`.

FINGERPRINTS are FCFP4 counts from the 031 table, keyed by InChIKey. Formic and lactic
acid have no row there (the private loader carries an identity gap for both), so they are
featurized from SMILES with the recipe ``inhibitor_profiles.py`` verified reproduces every
npz row exactly. Acetic acid, which that script also featurizes from SMILES, DOES have a
row (QTBSBXVTEAMEQO-UHFFFAOYSA-N) and is taken from the table.

ATTRIBUTION: ``VanacloigCells``, ``Fold``, ``load_cells``, ``make_folds``,
``subsample_pool``, ``ceiling`` and ``score_compounds`` are experiment 038's
``scripts/vanacloig_data.py`` (worktree ``exp/038-env-chemgen-vanacloig-cgt-corrected``,
commit 9427c0d4e), restated here so the 040 tree is self-contained on Delta; 038 is a
read-only reference and is not modified. The Vanacloig loading path is unchanged, so the
compound-cold folds and the centered score are the same objects experiment 038 reports.
"""

from __future__ import annotations

import json
import os
import os.path as osp
import pickle
from typing import Literal

import lmdb
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr

from torchcell.datamodels.compound_identity import _TABLE_PATH

VANACLOIG = "EnvChemgenVanacloig2022Dataset"
HOEPFNER = "EnvChemgenHoepfner2014Dataset"
HILLENMEYER = "HetHillenmeyer2008Dataset"

#: Build 002 of the 033 pooled store, as experiment 038 reads it. On Delta the launcher
#: points TC040_CELL_TABLE at the rsynced copy.
CELL_TABLE = os.getenv(
    "TC040_CELL_TABLE",
    "/scratch/projects/torchcell-scratch/experiments/033-env-chemgen-pooled/"
    "cell_table_002/cell_table.parquet",
)
#: The 031 embedding tables; TC040_EMBEDDING_DIR points at the rsynced copy on Delta.
EMBEDDING_DIR = os.getenv(
    "TC040_EMBEDDING_DIR",
    "/home/michaelvolk/Documents/projects/torchcell.worktrees/exp/"
    "031-env-chemgen-vanacloig-hillenmeyer/experiments/"
    "031-env-chemgen-inhibitor-tolerance/results/embeddings",
)
FCFP4_NPZ = osp.join(EMBEDDING_DIR, "fcfp4_count.npz")
#: The rebuilt Vanacloig dev store, whose interned environments hold the Table S1 IC30
#: doses the served cell table leaves empty. TC040_VANACLOIG_STORE overrides it.
VANACLOIG_STORE = os.getenv(
    "TC040_VANACLOIG_STORE",
    osp.join(
        os.environ.get("DATA_ROOT", ""),
        "data",
        "torchcell",
        "env_chemgen_vanacloig2022",
        "processed",
    ),
)
#: The three efflux-regulator deletions of the Vanacloig sensitized host (PDR1, PDR3,
#: SNQ2), on the reference's StrainReferenceGenome rather than in the genotype; the cell
#: lacks them all the same, as experiments 035 and 038 model it.
HOST_GENES: tuple[str, str, str] = ("YGL013C", "YBL005W", "YDR011W")
EXCLUDED_CONDITIONS: frozenset[str] = frozenset(
    {"dimethyl sulfoxide", "2-methyl-3-buten-2-ol"}
)
COUNT_FINGERPRINTS = ("fcfp4_count", "ecfp4_count")
MIN_GENES = 10
CONSTANT_SD = 1e-10
#: Host fitness of an IC30 anchor: IC30 is the dose at 30 percent growth inhibition.
IC30_FITNESS = 0.70
#: Host fitness of the compound-free medium.
CONTROL_FITNESS = 1.0
#: Wells whose growth call is False have no fitness; the model-free scoring of
#: ``mixture_rules.py`` reads them as zero (``w["y"] = w["y_grown"].fillna(0.0)``).
NO_GROWTH_FITNESS = 0.0
#: The isobole runs and the inhibitor titrated against acetic acid in each
#: (``mixture_rules.ISOBOLES``).
ISOBOLE_RUNS = {
    "ex26": "furfural",
    "ex27": "formic acid",
    "ex28": "5-(hydroxymethyl)furfural",
}
#: Structures for the two wet-lab acids with no row in the 031 table, from
#: ``inhibitor_profiles.inhibitors`` (the loader carries an identity gap for both;
#: lactic acid without stereo because the stock sheet does not state the enantiomer).
IDENTITY_GAP_SMILES = {"formic acid": "OC=O", "lactic acid": "CC(O)C(=O)O"}
#: The six 2021 inhibitors, by the private loader's compound name, and the abbreviation
#: the wet-lab table's dose columns carry.
ABBREVIATION = {
    "furfural": "FF",
    "acetic acid": "AA",
    "5-(hydroxymethyl)furfural": "HMF",
    "formic acid": "FA",
    "levulinic acid": "LVA",
    "lactic acid": "LA",
}
#: Molar-convertible served units and the factor to molar.
MOLAR_FACTOR = {"M": 1.0, "mM": 1e-3, "uM": 1e-6, "nM": 1e-9}

GeneSourceName = Literal["vanacloig", "hoepfner_hop", "hillenmeyer_het"]


# ---- fingerprints ---------------------------------------------------------- #
def fcfp4_count(smiles: str) -> NDArray[np.float32]:
    """FCFP4 count fingerprint, 2048 bits: 031 ``torchcell.molecule.encoders.FCFP4Count``.

    The recipe ``inhibitor_profiles.fcfp4_count`` verified reproduces every row of the
    031 npz exactly (``results/profile_fingerprint_check.csv``).
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"unparsable SMILES: {smiles!r}"
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    return gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)


def identity_smiles() -> dict[str, str]:
    """InChIKey -> curated SMILES from the sha256-pinned compound identity table."""
    with open(str(_TABLE_PATH)) as f:
        records = json.load(f)["records"]
    return {
        r["inchikey"]: r["smiles"]
        for r in records
        if r.get("inchikey") and r.get("smiles")
    }


class CompoundTable(BaseModel):
    """Every compound the run can represent, as rows of one feature matrix.

    ROWS ARE KEYED BY INCHIKEY, not by name: Hoepfner serves two names
    ("Cleisthantin derivative", "Valinomycin Derivative") against two structures each.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    names: list[str]
    inchikeys: list[str]
    #: "031 fcfp4_count.npz" or "SMILES (identity gap)"
    feature_source: list[str]
    x_raw: NDArray[np.float64]  # [n_compounds, n_features]

    def row(self, inchikey: str) -> int:
        """The feature-matrix row of ``inchikey``."""
        return self.inchikeys.index(inchikey)


def compound_table(
    names: list[str], inchikeys: list[str], embeddings: list[str], pca_dim: int | None
) -> CompoundTable:
    """Concatenate the named 031 embedding tables for ``names``, before standardization.

    A compound with no row in a table is featurized from its curated SMILES when that
    table is a count fingerprint and the compound is one of ``IDENTITY_GAP_SMILES``;
    anything else missing stops the run.
    """
    assert len(names) == len(inchikeys)
    assert len(set(inchikeys)) == len(inchikeys), "an InChIKey is listed twice"
    blocks = []
    sources = ["031 " + ", ".join(f"{n}.npz" for n in embeddings)] * len(names)
    for table in embeddings:
        data = np.load(osp.join(EMBEDDING_DIR, f"{table}.npz"), allow_pickle=True)
        x_all = data["X"].astype(np.float64)
        x_all = x_all[:, np.isfinite(x_all).all(axis=0)]
        if table in COUNT_FINGERPRINTS:
            x_all = np.log1p(x_all)
        row = {key: i for i, key in enumerate(data["inchikey"])}
        missing = [n for n, k in zip(names, inchikeys, strict=True) if k not in row]
        bad = [n for n in missing if n not in IDENTITY_GAP_SMILES]
        assert not bad, f"{table} has no embedding and no curated SMILES for {bad}"
        assert table == "fcfp4_count" or not missing, (
            f"only fcfp4_count can be re-featurized from SMILES; {table} lacks {missing}"
        )
        raw = []
        for i, (name, key) in enumerate(zip(names, inchikeys, strict=True)):
            if key in row:
                raw.append(x_all[row[key]])
                continue
            vector = np.log1p(fcfp4_count(IDENTITY_GAP_SMILES[name]).astype(np.float64))
            assert vector.shape[0] == x_all.shape[1], (
                f"{name}: the SMILES fingerprint is {vector.shape[0]} wide, the table "
                f"{x_all.shape[1]}"
            )
            raw.append(vector)
            sources[i] = f"SMILES (identity gap): {IDENTITY_GAP_SMILES[name]}"
        block = np.stack(raw)
        if pca_dim is not None and pca_dim < x_all.shape[1]:
            mu, sd = x_all.mean(0), x_all.std(0)
            scale = np.where(sd > 0, sd, 1.0)
            z = (x_all - mu) / scale
            _, _, vt = np.linalg.svd(z - z.mean(0), full_matrices=False)
            block = ((block - mu) / scale) @ vt[:pca_dim].T
        blocks.append(block)
    return CompoundTable(
        names=names,
        inchikeys=inchikeys,
        feature_source=sources,
        x_raw=np.concatenate(blocks, axis=1),
    )


# ---- the Vanacloig doses the served table leaves empty --------------------- #
class VanacloigDose(BaseModel):
    """One Vanacloig condition's Table S1 dose, as the rebuilt dev store holds it."""

    compound: str
    inchikey: str
    value: float
    unit: str
    basis: str
    mM: float | None
    log10_molar: float | None
    molar_source: str
    record_index: int
    n_records: int


def vanacloig_doses(store: str = VANACLOIG_STORE) -> list[VanacloigDose]:
    """Every Vanacloig condition's dose, read from the rebuilt dev LMDB.

    The store interns its environments, so each condition appears once in
    ``processed/interned`` and the records in ``processed/lmdb`` point at it by content
    hash. One record per compound is located by scanning the records for the first that
    references each environment, so a dose is reported with the record it was read from.
    """
    interned_dir, records_dir = osp.join(store, "interned"), osp.join(store, "lmdb")
    assert osp.isdir(interned_dir), f"no interned store at {interned_dir}"
    env = lmdb.open(interned_dir, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        interned = {k.decode(): pickle.loads(v) for k, v in txn.cursor()}
    env.close()

    first: dict[str, int] = {}
    count: dict[str, int] = {}
    env = lmdb.open(records_dir, readonly=True, lock=False, subdir=True)
    with env.begin() as txn:
        for key, value in txn.cursor():
            ref = pickle.loads(value)["experiment"]["environment"]["$ref"]
            count[ref] = count.get(ref, 0) + 1
            first.setdefault(ref, int(key.decode()))
    env.close()

    weights = {
        k: Descriptors.MolWt(Chem.MolFromSmiles(s))
        for k, s in identity_smiles().items()
    }
    out = []
    for ref, obj in interned.items():
        if "perturbations" not in obj:
            continue
        small = [p for p in obj["perturbations"] if p.get("compound")]
        assert len(small) == 1, (
            f"{ref}: {len(small)} compounds in one Vanacloig condition"
        )
        compound = small[0]["compound"]
        conc = small[0]["concentration"]
        unit = str(getattr(conc["unit"], "value", conc["unit"]))
        value = float(conc["value"])
        if unit in MOLAR_FACTOR:
            molar: float | None = value * MOLAR_FACTOR[unit]
            molar_source = f"{unit} -> molar"
        elif unit == "ug/mL":
            weight = weights[compound["inchikey"]]
            molar = value * 1e-3 / weight
            molar_source = f"ug/mL / MolWt {weight:.2f} g/mol (curated SMILES)"
        else:
            molar, molar_source = (
                None,
                f"{unit} states no basis or density; no molar dose",
            )
        out.append(
            VanacloigDose(
                compound=compound["name"],
                inchikey=compound["inchikey"],
                value=value,
                unit=unit,
                basis=str(getattr(conc["basis"], "value", conc["basis"])),
                mM=None if molar is None else molar * 1e3,
                log10_molar=None if molar is None else float(np.log10(molar)),
                molar_source=molar_source,
                record_index=first[ref],
                n_records=count[ref],
            )
        )
    return sorted(out, key=lambda d: d.compound)


def write_vanacloig_doses(path: str, store: str = VANACLOIG_STORE) -> pd.DataFrame:
    doses = vanacloig_doses(store)
    frame = pd.DataFrame([d.model_dump() for d in doses])
    frame["in_panel"] = ~frame["compound"].isin(EXCLUDED_CONDITIONS)
    frame.to_csv(path, index=False)
    return frame


# ---- the gene-level sources ------------------------------------------------ #
class SourceSpec(BaseModel):
    """One pooled screen: how it is filtered and how it is oriented to sick-negative."""

    name: str
    dataset: str
    #: the perturbation type that selects the arm, None for a single-arm store
    perturbation_type: str | None
    #: multiply the served value by this to make a sick deletion NEGATIVE
    sign: float
    orientation: str
    #: genes deleted in every strain of this source beside the queried one
    extra_genes: tuple[str, ...]
    #: doses come from the served table, or from the dev store (Vanacloig)
    dose_from: Literal["table", "dev_store"]
    #: "single" asserts one measurement per cell (Vanacloig, as 038 asserts); "mean"
    #: averages the cell's served measurements, which for Hoepfner and Hillenmeyer are
    #: the same (compound, concentration, gene) read in different screens or scanners
    label_policy: Literal["single", "mean"]


SPECS: dict[str, SourceSpec] = {
    "vanacloig": SourceSpec(
        name="vanacloig",
        dataset=VANACLOIG,
        perturbation_type="barcoded_kanmx_deletion",
        sign=1.0,
        orientation=(
            "log2((TMM-normalized CPM of the inhibitor replicate + 1) / (mean "
            "TMM-normalized CPM of the SAME CG batch's inhibitor-free control columns "
            "+ 1)); negative = sick deletion, kept as served"
        ),
        extra_genes=HOST_GENES,
        dose_from="dev_store",
        label_policy="single",
    ),
    "hoepfner_hop": SourceSpec(
        name="hoepfner_hop",
        dataset=HOEPFNER,
        perturbation_type="barcoded_kanmx_deletion",
        sign=1.0,
        orientation=(
            "adjusted MADL sensitivity score = (r_L - med(r_L)) / MAD(r_L); 'negative = "
            "hypersensitive, positive = resistant' (hoepfner2014.py), kept as served"
        ),
        extra_genes=(),
        dose_from="table",
        label_policy="mean",
    ),
    "hillenmeyer_het": SourceSpec(
        name="hillenmeyer_het",
        dataset=HILLENMEYER,
        perturbation_type="heterozygous_deletion",
        sign=-1.0,
        orientation=(
            "HET fitness-defect log-ratio, log2(mean control intensity / treatment "
            "intensity) (hillenmeyer2008.py); POSITIVE = sick, so the value is NEGATED"
        ),
        extra_genes=(),
        dose_from="table",
        label_policy="mean",
    ),
}


class GeneSource(BaseModel):
    """One source's genes-by-conditions matrix, oriented sick-negative."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: SourceSpec
    index: int  # the source token's index
    genes: list[str]
    compounds: list[str]  # one per condition, a name may repeat across doses
    inchikeys: list[str]
    #: [n_conditions] rows of the shared compound table
    compound_row: NDArray[np.int64]
    #: [n_conditions] log10 molar dose, NaN where the served unit does not convert
    log10_molar: NDArray[np.float64]
    y: NDArray[np.float64]  # [n_genes, n_conditions]
    se: NDArray[np.float64]
    n_cells: int
    n_dropped_cells: int

    def keep_conditions(self, keep: list[int]) -> GeneSource:
        """A copy holding only ``keep``, in that order (used to drop percent doses)."""
        return GeneSource(
            spec=self.spec,
            index=self.index,
            genes=self.genes,
            compounds=[self.compounds[j] for j in keep],
            inchikeys=[self.inchikeys[j] for j in keep],
            compound_row=self.compound_row[keep],
            log10_molar=self.log10_molar[keep],
            y=self.y[:, keep],
            se=self.se[:, keep],
            n_cells=int(np.isfinite(self.y[:, keep]).sum()),
            n_dropped_cells=self.n_dropped_cells
            + int(np.isfinite(self.y).sum())
            - int(np.isfinite(self.y[:, keep]).sum()),
        )


def _condition_frame(spec: SourceSpec, cell_table: str) -> pd.DataFrame:
    """The source's single-compound cells, with a condition key per compound and dose."""
    filters = [("dataset", "==", spec.dataset)]
    if spec.perturbation_type is not None:
        filters.append(("perturbation_type", "==", spec.perturbation_type))
    table = pd.read_parquet(
        cell_table,
        columns=[
            "query_gene",
            "genes",
            "n_measurements",
            "n_compounds",
            "compound_names",
            "inchikeys",
            "conc_values",
            "conc_units",
            "log10_molar",
            "responses",
            "response_ses",
        ],
        filters=filters,
    ).reset_index(drop=True)
    assert len(table), f"{spec.name}: no cells matched {filters}"
    total = len(table)
    table = table[table["n_compounds"] == 1].reset_index(drop=True)
    table = table[table["inchikeys"].astype(bool)].reset_index(drop=True)
    table = table[~table["compound_names"].isin(EXCLUDED_CONDITIONS)].reset_index(
        drop=True
    )
    assert (table["genes"] == table["query_gene"]).all(), (
        f"{spec.name}: a strain carries more than its screened deletion"
    )
    if spec.label_policy == "single":
        assert (table["n_measurements"] == 1).all(), (
            f"{spec.name}: a cell folds measurements"
        )
    table["condition"] = (
        table["compound_names"] + "|" + table["conc_values"].astype(str)
    )
    table["response"] = table["responses"].map(lambda r: float(np.mean(r))) * spec.sign
    table["response_se"] = table["response_ses"].map(lambda s: float(np.mean(s)))
    return table.assign(n_total=total)


def load_gene_source(
    spec: SourceSpec,
    index: int,
    cell_table: str,
    doses: dict[str, float] | None,
    compounds: CompoundTable,
) -> GeneSource:
    """``spec``'s matrix, with every condition's dose as log10 molar.

    ``doses`` maps a compound name to its log10 molar dose and is required when the
    source's doses come from the dev store; a compound missing from it carries NaN.
    """
    table = _condition_frame(spec, cell_table)
    genes = sorted(table["query_gene"].unique())
    keys = (
        table[["condition", "compound_names", "inchikeys", "log10_molar"]]
        .drop_duplicates("condition")
        .sort_values("condition", ignore_index=True)
    )
    gene_index = {g: i for i, g in enumerate(genes)}
    cond_index = {c: i for i, c in enumerate(keys["condition"])}
    rows = table["query_gene"].map(gene_index).to_numpy(dtype=np.int64)
    cols = table["condition"].map(cond_index).to_numpy(dtype=np.int64)
    y = np.full((len(genes), len(keys)), np.nan)
    se = np.full((len(genes), len(keys)), np.nan)
    y[rows, cols] = table["response"].to_numpy()
    se[rows, cols] = table["response_se"].to_numpy()

    if spec.dose_from == "dev_store":
        assert doses is not None, f"{spec.name} needs the dev-store doses"
        log10_molar = np.array(
            [doses.get(name, np.nan) for name in keys["compound_names"]]
        )
    else:
        log10_molar = keys["log10_molar"].to_numpy(dtype=np.float64)
    compound_row = np.array(
        [compounds.row(key) for key in keys["inchikeys"]], dtype=np.int64
    )
    return GeneSource(
        spec=spec,
        index=index,
        genes=genes,
        compounds=list(keys["compound_names"]),
        inchikeys=list(keys["inchikeys"]),
        compound_row=compound_row,
        log10_molar=log10_molar,
        y=y,
        se=se,
        n_cells=int(len(table)),
        n_dropped_cells=int(table["n_total"].iloc[0]) - int(len(table)),
    )


# ---- the Vanacloig folds and score (experiment 038, restated) -------------- #
class Fold(BaseModel):
    """The compounds of one fold, as indices into the Vanacloig condition order."""

    fold: int
    train: list[int]
    val: list[int]
    test: list[int]


def make_folds(n_compounds: int, n_folds: int, n_val: int, seed: int) -> list[Fold]:
    """Compound-cold folds: every compound is tested exactly once (038)."""
    rng = np.random.default_rng(seed)
    groups = np.array_split(rng.permutation(n_compounds), n_folds)
    folds = []
    for k, test in enumerate(groups):
        rest = np.concatenate([g for j, g in enumerate(groups) if j != k])
        val = np.random.default_rng([seed, k]).choice(rest, size=n_val, replace=False)
        train = np.setdiff1d(rest, val)
        folds.append(
            Fold(
                fold=k,
                train=sorted(int(i) for i in train),
                val=sorted(int(i) for i in val),
                test=sorted(int(i) for i in test),
            )
        )
    tested = sorted(i for f in folds for i in f.test)
    assert tested == list(range(n_compounds)), "a compound is tested twice or never"
    return folds


def subsample_pool(fold: Fold, fold_seed: int, n: int | None) -> list[int]:
    """The first ``n`` of a fixed shuffle of the fold's non-test compounds (038)."""
    pool = sorted(fold.train + fold.val)
    if n is None:
        return pool
    assert 2 <= n <= len(pool), f"{n} compounds asked of a pool of {len(pool)}"
    order = np.random.default_rng([fold_seed, fold.fold, 17]).permutation(pool)
    return sorted(int(i) for i in order[:n])


def ceiling(response: NDArray[np.float64], se: NDArray[np.float64]) -> float:
    """Square root of 1 - mean(SE^2) / Var(response); zero where that is negative (038)."""
    ok = np.isfinite(response) & np.isfinite(se)
    if ok.sum() < MIN_GENES:
        return float("nan")
    reliability = 1.0 - float(np.mean(se[ok] ** 2)) / float(
        np.var(response[ok], ddof=1)
    )
    return float(np.sqrt(reliability)) if reliability > 0 else 0.0


def score_compounds(
    source: GeneSource,
    prediction: NDArray[np.float64],
    train: list[int],
    held_out: list[int],
) -> pd.DataFrame:
    """One row per held-out compound and target, as experiment 038 scores it.

    EACH SIDE IS CENTERED BY ITS OWN TRAINING MEAN: the measurement by the measured gene
    mean over the fitted compounds, the prediction by the PREDICTED gene mean over the
    same compounds. Subtracting the measured mean from the prediction makes a model whose
    output barely depends on the gene score well for nothing (038's smoke run, slurm
    3009: 0.356 centered against -0.001 raw).
    """
    measured, se = source.y, source.se
    measured_mean = np.nanmean(measured[:, train], axis=1)
    predicted_mean = np.nanmean(prediction[:, train], axis=1)
    zero = np.zeros_like(measured_mean)
    rows = []
    for j in held_out:
        for target in ("raw", "centered"):
            obs = measured[:, j] - (measured_mean if target == "centered" else zero)
            pred = prediction[:, j] - (predicted_mean if target == "centered" else zero)
            ok = np.isfinite(obs) & np.isfinite(pred)
            constant = ok.sum() < MIN_GENES or np.std(pred[ok]) < CONSTANT_SD
            rows.append(
                {
                    "compound": source.compounds[j],
                    "target": target,
                    "n_genes": int(ok.sum()),
                    "spearman": (
                        float("nan")
                        if constant
                        else float(spearmanr(pred[ok], obs[ok])[0])
                    ),
                    "pearson": (
                        float("nan")
                        if constant
                        else float(pearsonr(pred[ok], obs[ok])[0])
                    ),
                    "ceiling": ceiling(obs, se[:, j]),
                }
            )
    return pd.DataFrame(rows)


# ---- the host-growth records ----------------------------------------------- #
class HostRecord(BaseModel):
    """One host-growth observation: a medium, its fitness, and where it came from."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    key: str
    split: Literal["anchor", "ex21", "ex23", "isobole"]
    run: str
    compounds: list[str]
    #: rows of the shared compound table, one per compound
    compound_row: list[int]
    log10_molar: list[float]
    fitness: float  # mean over replicates, no growth = NO_GROWTH_FITNESS
    fitness_grown: float | None  # None where no replicate grew
    #: every replicate well's fitness, no growth read as NO_GROWTH_FITNESS
    well_fitness: list[float]
    grew: bool
    n_wells: int


def _well_doses(
    row: pd.Series, abbreviation: dict[str, str]
) -> list[tuple[str, float]]:
    """The well's compounds and their mM doses, in the order the wells name them."""
    listed = row["compounds"]
    members = [] if pd.isna(listed) else [c for c in str(listed).split("|") if c]
    assert len(members) == int(row["n_compounds"]), row["well"]
    return [(c, float(row[f"dose_mM_{abbreviation[c]}"])) for c in members]


def load_host_records(
    wells_csv: str,
    compounds: CompoundTable,
    inchikey_of: dict[str, str],
    call: Literal["served", "software"],
) -> list[HostRecord]:
    """Every private well, aggregated to one record per medium.

    A record's replicates are the wells of one run with the same compound set and doses:
    ``grew`` is true when any replicate grew (the model-free scoring's rule),
    ``fitness`` is the mean over replicates with no growth read as zero, and
    ``fitness_grown`` the mean over the replicates that grew.
    """
    fitness_column, grew_column = (
        ("fitness", "grew")
        if call == "served"
        else ("fitness_software", "grew_software")
    )
    wells = pd.read_csv(wells_csv)
    abbreviation = ABBREVIATION
    wells["grew_call"] = wells[grew_column].astype(bool)
    wells["y_grown"] = wells[fitness_column].where(wells["grew_call"])
    wells["y"] = wells["y_grown"].fillna(NO_GROWTH_FITNESS)
    wells["medium"] = wells.apply(
        lambda r: json.dumps(_well_doses(r, abbreviation)), axis=1
    )
    split_of = {
        "ex21": "ex21",
        "ex23": "ex23",
        "ex26": "isobole",
        "ex27": "isobole",
        "ex28": "isobole",
    }
    records = []
    for (run, medium), group in wells.groupby(["run", "medium"]):
        doses = json.loads(medium)
        grown = group["y_grown"].dropna()
        records.append(
            HostRecord(
                key=f"{run}:{medium}",
                split=split_of[run],  # type: ignore[arg-type]
                run=run,
                compounds=[name for name, _ in doses],
                compound_row=[compounds.row(inchikey_of[name]) for name, _ in doses],
                log10_molar=[float(np.log10(mM * 1e-3)) for _, mM in doses],
                fitness=float(group["y"].mean()),
                fitness_grown=None if grown.empty else float(grown.mean()),
                well_fitness=[float(y) for y in group["y"]],
                grew=bool(group["grew_call"].any()),
                n_wells=int(len(group)),
            )
        )
    for run in split_of:
        assert any(r.run == run for r in records), (
            f"{run} has no records in {wells_csv}"
        )
    interior = {
        run: sum(1 for r in records if r.run == run and len(r.compounds) == 2)
        for run in ISOBOLE_RUNS
    }
    assert all(n == 81 for n in interior.values()), f"isobole interiors are {interior}"
    return sorted(records, key=lambda r: r.key)


def anchor_records(
    sources: list[GeneSource], compounds: CompoundTable
) -> list[HostRecord]:
    """The public IC30 anchors at host fitness 0.70 plus the compound-free medium at 1.0.

    A compound served at several concentrations in one source has no unambiguous IC30
    among them, so it is left out; ``results/mixture/<name>_host_records.csv`` counts how
    many each source contributed.
    """
    records = [
        HostRecord(
            key="anchor:control",
            split="anchor",
            run="anchor",
            compounds=[],
            compound_row=[],
            log10_molar=[],
            fitness=CONTROL_FITNESS,
            fitness_grown=CONTROL_FITNESS,
            well_fitness=[],
            grew=True,
            n_wells=0,
        )
    ]
    for source in sources:
        served = pd.Series(source.compounds).value_counts()
        for j, name in enumerate(source.compounds):
            if served[name] != 1 or not np.isfinite(source.log10_molar[j]):
                continue
            records.append(
                HostRecord(
                    key=f"anchor:{source.spec.name}:{name}",
                    split="anchor",
                    run=f"anchor_{source.spec.name}",
                    compounds=[name],
                    compound_row=[int(source.compound_row[j])],
                    log10_molar=[float(source.log10_molar[j])],
                    fitness=IC30_FITNESS,
                    fitness_grown=IC30_FITNESS,
                    well_fitness=[],
                    grew=True,
                    n_wells=0,
                )
            )
    seen: dict[str, HostRecord] = {}
    for record in records:
        seen.setdefault(record.key, record)
    return list(seen.values())


# ---- the cell graph and the strain tensors (038, restated) ----------------- #
def build_cell_graph(graph_names: list[str]):
    """The wildtype cell graph over the S288C gene set, as the dataset class builds it.

    Experiment 038's ``train_vanacloig_cgt.build_cell_graph``, restated.
    """
    from torchcell.data.cell_data import to_cell_data
    from torchcell.data.neo4j_cell import create_graph_from_gene_set
    from torchcell.graph import SCerevisiaeGraph
    from torchcell.graph.graph import build_gene_multigraph
    from torchcell.sequence.genome.scerevisiae.s288c import SCerevisiaeGenome

    data_root = os.environ["DATA_ROOT"]
    genome = SCerevisiaeGenome(
        genome_root=osp.join(data_root, "data/sgd/genome"),
        go_root=osp.join(data_root, "data/go"),
        overwrite=False,
    )
    genome.drop_empty_go()
    graph = SCerevisiaeGraph(
        sgd_root=osp.join(data_root, "data/sgd/genome"),
        string_root=osp.join(data_root, "data/string"),
        tflink_root=osp.join(data_root, "data/tflink"),
        genome=genome,
    )
    multigraph = build_gene_multigraph(graph=graph, graph_names=graph_names)
    assert multigraph is not None, "no gene graphs were named"
    multigraph.graphs["base"] = create_graph_from_gene_set(genome.gene_set)
    return to_cell_data(multigraph)


def strain_indices(source: GeneSource, node_ids: list[str]) -> torch.Tensor:
    """[n_genes, 1 + len(extra_genes)] cell-graph indices, one strain per queried gene."""
    position = {gene: i for i, gene in enumerate(node_ids)}
    missing = [g for g in source.genes if g not in position]
    assert not missing, (
        f"{source.spec.name}: genes outside the cell graph: {missing[:5]}"
    )
    query = np.array([position[g] for g in source.genes])[:, None]
    if not source.spec.extra_genes:
        return torch.tensor(query, dtype=torch.long)
    extra = np.array([position[g] for g in source.spec.extra_genes])
    return torch.tensor(
        np.concatenate([query, np.tile(extra, (len(query), 1))], axis=1),
        dtype=torch.long,
    )


# ---- assembly -------------------------------------------------------------- #
def wetlab_inhibitors() -> dict[str, str]:
    """The six 2021 inhibitors' compound name -> InChIKey of the structure predicted from.

    The private loader serves the structure for four of them; formic and lactic acid
    carry an identity gap there (no InChIKey, no SMILES), so their structures are the
    ones ``inhibitor_profiles.inhibitors`` assigns (``IDENTITY_GAP_SMILES``).
    """
    from torchcell.datasets.private_torchcell import bioscreen as bs
    from torchcell.datasets.private_torchcell import (
        volk2021_inhibitor_bioscreen as volk,
    )

    out = {}
    for inhibitor in bs.INHIBITORS:
        compound = volk.compound(inhibitor)
        if compound.name in IDENTITY_GAP_SMILES:
            assert compound.smiles is None and compound.inchikey is None
            smiles = IDENTITY_GAP_SMILES[compound.name]
        else:
            smiles = compound.smiles
        key = str(Chem.MolToInchiKey(Chem.MolFromSmiles(smiles)))
        if compound.inchikey is not None:
            assert key == compound.inchikey, compound.name
        out[compound.name] = key
    assert set(out) == set(ABBREVIATION), f"the wet-lab inhibitor names moved: {out}"
    return out


class MixtureData(BaseModel):
    """Everything one run reads: the compound table, the sources, the host records."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    compounds: CompoundTable
    sources: list[GeneSource]
    host: list[HostRecord]
    #: the Vanacloig compounds dropped from the scored panel, with the reason
    dropped_compounds: dict[str, str]
    inchikey_of_inhibitor: dict[str, str]
    counts: pd.DataFrame

    def source(self, name: str) -> GeneSource:
        """The loaded source called ``name``."""
        return next(s for s in self.sources if s.spec.name == name)


def assemble(
    source_names: list[str],
    wells_csv: str,
    embeddings: list[str],
    pca_dim: int | None,
    require_molar_dose: bool,
    call: Literal["served", "software"] = "served",
    cell_table: str = CELL_TABLE,
    vanacloig_store: str = VANACLOIG_STORE,
) -> MixtureData:
    """Load every source, the wet-lab wells and one shared compound feature matrix.

    ``source_names`` must start with ``vanacloig``: it carries the compound-cold folds
    the gene-level score is reported on, so it is the task, and the others are auxiliary.
    ``require_molar_dose`` drops the Vanacloig compounds whose served dose is a percent
    from the scored panel, so the dose ablation compares like for like.
    """
    assert source_names and source_names[0] == "vanacloig", (
        f"vanacloig is the scored source and must come first: {source_names}"
    )
    assert len(set(source_names)) == len(source_names), "a source is named twice"
    doses = {
        d.compound: d.log10_molar
        for d in vanacloig_doses(vanacloig_store)
        if d.log10_molar is not None
    }
    frames = {name: _condition_frame(SPECS[name], cell_table) for name in source_names}
    inchikey_of = wetlab_inhibitors()
    keys: dict[str, str] = {}
    for frame in frames.values():
        for name, key in zip(frame["compound_names"], frame["inchikeys"], strict=True):
            keys.setdefault(key, name)
    for name, key in inchikey_of.items():
        keys.setdefault(key, name)
    inchikeys = sorted(keys)
    compounds = compound_table(
        [keys[k] for k in inchikeys], inchikeys, embeddings, pca_dim
    )

    sources = [
        load_gene_source(SPECS[name], i, cell_table, doses, compounds)
        for i, name in enumerate(source_names)
    ]
    vanacloig = sources[0]
    panel = [
        j
        for j in range(len(vanacloig.compounds))
        if not require_molar_dose or np.isfinite(vanacloig.log10_molar[j])
    ]
    assert len(panel) >= 10, f"only {len(panel)} Vanacloig compounds carry a molar dose"
    dropped = {
        vanacloig.compounds[
            j
        ]: "the served dose is a percent and does not convert to molar"
        for j in range(len(vanacloig.compounds))
        if j not in panel
    }
    if dropped:
        sources[0] = vanacloig.keep_conditions(panel)
        vanacloig = sources[0]
    host = load_host_records(wells_csv, compounds, inchikey_of, call) + anchor_records(
        sources, compounds
    )
    counts = pd.DataFrame(
        [
            {
                "source": s.spec.name,
                "n_cells": s.n_cells,
                "n_cells_dropped": s.n_dropped_cells,
                "n_genes": len(s.genes),
                "n_compounds": len(set(s.compounds)),
                "n_conditions": len(s.compounds),
                "n_conditions_with_molar_dose": int(np.isfinite(s.log10_molar).sum()),
                "sign": s.spec.sign,
                "orientation": s.spec.orientation,
            }
            for s in sources
        ]
        + [
            {
                "source": f"host_{split}",
                "n_cells": sum(r.n_wells for r in host if r.split == split),
                "n_cells_dropped": 0,
                "n_genes": 0,
                "n_compounds": len(
                    {c for r in host if r.split == split for c in r.compounds}
                ),
                "n_conditions": sum(1 for r in host if r.split == split),
                "n_conditions_with_molar_dose": sum(
                    1
                    for r in host
                    if r.split == split and all(np.isfinite(r.log10_molar))
                ),
                "sign": 1.0,
                "orientation": "host fitness, 1 = uninhibited growth; no growth = 0",
            }
            for split in ("anchor", "ex21", "ex23", "isobole")
        ]
    )
    return MixtureData(
        compounds=compounds,
        sources=sources,
        host=host,
        dropped_compounds=dropped,
        inchikey_of_inhibitor=inchikey_of,
        counts=counts,
    )
