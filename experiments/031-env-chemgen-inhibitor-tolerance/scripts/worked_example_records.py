# experiments/031-env-chemgen-inhibitor-tolerance/scripts/worked_example_records.py
# [[experiments.031-env-chemgen-inhibitor-tolerance.scripts.worked_example_records]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/031-env-chemgen-inhibitor-tolerance/scripts/worked_example_records
"""One real record from each dataset, carried through every transformation, end to end.

A representation argued for in prose is hard to check. This script takes an actual served
record from each dataset and prints what the model receives and what it is asked to predict,
so the claim that the five map onto one form can be verified against real numbers rather than
believed.

THE TARGET IS A CONTINUOUS REGRESSION TARGET, NOT A CLASS. The per-dataset sign convention
(a sick strain is negative in three datasets and positive in two) is a statement about which
DIRECTION means a fitness defect, and it is applied by multiplying the response by plus or
minus one. Nothing is thresholded, binned or binarized at any point, and no information is
discarded: orienting is a bijection on the real line, and standardizing is an affine map. The
transformed value keeps the full continuous response, which is what makes the compound-level
correlations in the rest of the document the right way to score a model.

THE THREE STEPS, applied in this order:

1. **orient**: multiply by ``o_k``, the orientation factor, so that after the step a more
   NEGATIVE number means a sicker strain in every dataset. That direction is chosen to match
   how fitness is represented everywhere else in torchcell, and it is also the minimal
   change: three of the five sources already store a sick strain as negative and keep
   ``o_k = +1``, while the two Hillenmeyer arms store it as positive and flip.
   ``o_k = -s_k`` where ``s_k`` is the sign a sick strain carries in the SOURCE, so the two
   are easy to confuse and are kept as separate named quantities throughout.
2. **center**: subtract the source's training-split median.
3. **scale**: divide by the source's training-split standard deviation.

Both statistics come from the TRAINING split alone, so no held-out value informs them. Here
they are computed over the whole dataset, because this script illustrates the arithmetic
rather than fitting a model, and the printed statistics say so.

Writes ``results/worked_example_records.csv``, one row per dataset, carrying the record's
identity, its input channels, and the response at each of the three steps.
"""

from __future__ import annotations

import os
import os.path as osp

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EXPERIMENT_ROOT = os.environ["EXPERIMENT_ROOT"]
RESULTS_DIR = osp.join(
    EXPERIMENT_ROOT, "031-env-chemgen-inhibitor-tolerance", "results"
)

#: The four kept datasets plus the dropped homozygous arm, which is shown so the reader can
#: see what was given up.
NAMES = [
    "vanacloig2022",
    "hillenmeyer2008_het",
    "hoepfner2014",
    "wildenhain2015",
    "hillenmeyer2008_hom",
]
LABEL = {
    "vanacloig2022": "Vanacloig 2022",
    "hillenmeyer2008_het": "Hillenmeyer HET",
    "hoepfner2014": "Hoepfner 2014",
    "wildenhain2015": "Wildenhain 2015",
    "hillenmeyer2008_hom": "Hillenmeyer HOM (dropped)",
}
#: Sign a SICK strain carries in each SOURCE, quoted through the loader; see
#: dataset_joinability.py POLARITY for the verbatim definitions. This is a property of the
#: data, not a choice.
SICK_SIGN = {
    "vanacloig2022": -1,
    "hillenmeyer2008_het": +1,
    "hoepfner2014": -1,
    "wildenhain2015": -1,
    "hillenmeyer2008_hom": +1,
}
#: The three deletions every Vanacloig genotype carries on top of the queried gene. They are
#: the efflux-regulator deletions that make the strain drug-sensitive, which is what
#: "sensitized host" means; the queried gene is the fourth.
HOST_GENES = {"YBL005W": "PDR1", "YDR011W": "SNQ2", "YGL013C": "PDR3"}
#: A gene picked because it is measured in every dataset, so the five rows differ only in the
#: dataset rather than also in the strain.
PREFERRED_GENE = "YBR058C"


def orient_factor(name: str) -> int:
    """The multiplier that puts a dataset on the shared fitness-like direction.

    Negated because the shared convention is that a more NEGATIVE value means a sicker
    strain, while ``SICK_SIGN`` records the sign a sick strain carries in the source.
    """
    return -SICK_SIGN[name]


def pick_record(df: pd.DataFrame) -> pd.Series:
    """One record for the preferred gene, else the first single-compound record."""
    single = df[df["n_small_molecules"] == 1]
    hit = single[single["gene"].astype(str).str.contains(PREFERRED_GENE, na=False)]
    chosen = hit if len(hit) else single
    # the strongest-responding record of that gene, so the example is a real phenotype
    # rather than a value sitting at zero where every transformation looks like a no-op
    return chosen.loc[chosen["response"].abs().idxmax()]


def dataset_stats(df: pd.DataFrame, orient: int) -> tuple[float, float]:
    """Median and standard deviation of the ORIENTED response over the dataset."""
    v = orient * df["response"].to_numpy()
    return float(np.median(v)), float(np.std(v, ddof=1))


def main() -> None:
    rows = []
    for name in NAMES:
        df = pd.read_parquet(
            osp.join(RESULTS_DIR, f"records_{name}.parquet"),
            columns=[
                "gene",
                "gene_common",
                "compound",
                "inchikey",
                "response",
                "measurement_type",
                "ref_ploidy",
                "perturbation_type",
                "n_small_molecules",
                "dose_value",
                "dose_unit",
                "dose_basis",
                "media_base",
                "n_genes",
            ],
        )
        sign = SICK_SIGN[name]
        orient = orient_factor(name)
        med, sd = dataset_stats(df, orient)
        r = pick_record(df)

        genes = str(r["gene"]).split("|")
        commons = str(r["gene_common"]).split("|")
        # the common name has to be taken at the QUERIED gene's index: taking element 0
        # names a host deletion instead, which on a Vanacloig record is the wrong gene
        queried = [g for g in genes if g not in HOST_GENES]
        queried_common = [
            commons[i] if i < len(commons) else ""
            for i, g in enumerate(genes)
            if g not in HOST_GENES
        ]
        host = [HOST_GENES[g] for g in genes if g in HOST_GENES]

        raw = float(r["response"])
        oriented = orient * raw
        standardized = (oriented - med) / sd
        dose = (
            f"{r['dose_value']} {r['dose_unit']}"
            if str(r["dose_value"]).strip()
            else "not stated"
        )
        rows.append(
            {
                "dataset": LABEL[name],
                "gene": "|".join(queried),
                "gene_common": "|".join(c for c in queried_common if c),
                "host_deletions": "+".join(host) if host else "none",
                "n_perturbed_genes": int(r["n_genes"]),
                "ploidy": r["ref_ploidy"],
                "perturbation": str(r["perturbation_type"]).split("|")[0],
                "compound": str(r["compound"]),
                "inchikey": str(r["inchikey"]),
                "dose": dose,
                "dose_basis": str(r["dose_basis"]) or "none",
                "medium": r["media_base"],
                "measurement_type": r["measurement_type"],
                "sick_sign_in_source": sign,
                "orientation_factor": orient,
                "response_raw": round(raw, 4),
                "response_oriented": round(oriented, 4),
                "dataset_median_oriented": round(med, 4),
                "dataset_sd_oriented": round(sd, 4),
                "response_standardized": round(standardized, 4),
            }
        )
        print(
            f"{LABEL[name]:26s} {str(r['measurement_type']):18s} "
            f"raw {raw:+8.3f} -> oriented {oriented:+8.3f} -> "
            f"standardized {standardized:+7.3f}   "
            f"(o={orient:+d}, median {med:+.3f}, sd {sd:.3f})"
        )

    out = pd.DataFrame(rows)
    out.to_csv(osp.join(RESULTS_DIR, "worked_example_records.csv"), index=False)
    print()
    print(
        "After orienting, a MORE NEGATIVE value means a sicker strain in every dataset.\n"
        "The target is continuous at every step: orienting multiplies by +/-1 and "
        "standardizing is affine, so nothing is thresholded or binned."
    )


if __name__ == "__main__":
    main()
