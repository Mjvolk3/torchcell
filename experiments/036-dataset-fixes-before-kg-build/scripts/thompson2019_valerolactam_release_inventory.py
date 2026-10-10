# experiments/036-dataset-fixes-before-kg-build/scripts/thompson2019_valerolactam_release_inventory.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.thompson2019_valerolactam_release_inventory]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/thompson2019_valerolactam_release_inventory
"""What Thompson 2019 (valerolactam) released, and which part is already served.

Schedule row 59 of the ranked bacterial candidate table
(``experiments/database/scripts/build_bacteria_candidate_datasets_table.py``) is a
Transposon-fitness row, estimated at 4,778 insertion mutants over 2 conditions. This
script measures the release instead of estimating it, off sha256-pinned bytes, and the
measurement is what decided that the RB-TnSeq half is SUBSUMED and the loader serves the
titer and growth-rate halves:

1. **The release inventory.** Every file the raw mirror holds for this key, with its
   role, byte size and sha256, plus the counts each consumed artifact carries: the eight
   titers the Results state, Table S1's nine growth rates, and the quotes the loader
   binds every value to.
2. **The RB-TnSeq subsumption, measured in both directions.** The Methods describe ONE
   growth (10 mM valerolactam, sole carbon source, 48-well plate, Tecan Infinite F200)
   and the Results name "two valerolactam RB-TnSeq experiments". The Borchert 2024
   compendium release holds exactly two carbon-source samples at ``2-Piperidinone``
   10 mM, and the served ``RbTnseqBorchert2024Dataset`` carries both in full. The
   5-aminovalerate samples of Fig. S2 are attributed to the lysine paper by the SI
   itself, so they are NOT this row's and are reported separately.
3. **What the compendium does not carry**, so the subsumption is not a gesture: the
   per-experiment fitness this paper itself released, which lives only on the Fitness
   Browser with no accession.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/thompson2019_valerolactam_release_inventory.py
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

import torchcell.datasets.pputida.thompson2019_valerolactam as vl  # noqa: E402
from torchcell.verification.report import sha256_file  # noqa: E402

RESULTS = osp.join(
    os.environ["EXPERIMENT_ROOT"], "036-dataset-fixes-before-kg-build", "results"
)
OUT = osp.join(RESULTS, "thompson2019_valerolactam_release_inventory.json")
SAMPLES_CSV = osp.join(RESULTS, "thompson2019_valerolactam_compendium_samples.csv")

#: The compendium release the served Borchert 2024 store is built from.
COMPENDIUM_KEY = "borchertMachineLearningAnalysis2024"
COMPENDIUM_RELEASE = "data/fModule_Metadata.xlsx"
COMPENDIUM_SHA256 = "4d649385ac06684482396a125f135df22a2a5060da73485b2cd14468f8cc8be1"
COMPENDIUM_ROOT_REL = "data/torchcell/rbtnseq_borchert2024"
#: The compendium's own name for valerolactam, and for the 5AVA samples Fig. S2 borrows.
COMPENDIUM_VALEROLACTAM = "2-Piperidinone"
COMPENDIUM_5AVA = "5-Aminovaleric acid"
#: The group the paper's growth is in: a sole CARBON source.
COMPENDIUM_GROUP = "carbon source"
#: What the Methods state, and what the Results state, about the RB-TnSeq arm.
METHODS_CONDITION_MM = 10.0
RESULTS_EXPERIMENTS = 2


def data_root() -> str:
    """``DATA_ROOT`` from the environment."""
    return os.environ["DATA_ROOT"]


def mirror_inventory() -> list[dict[str, Any]]:
    """Every file this key's raw mirror holds, verified against its manifest pin."""
    manifest = vl.load_manifest(data_root())
    root = vl.raw_mirror_dir(data_root())
    rows: list[dict[str, Any]] = []
    for raw in vl.RAW_FILES:
        path = root / raw.relpath
        recorded = vl.manifest_sha256(manifest, raw.relpath)
        observed = sha256_file(path)
        if not (recorded == raw.sha256 == observed):
            raise RuntimeError(
                f"{raw.relpath}: manifest {recorded}, module pin {raw.sha256}, bytes "
                f"{observed}"
            )
        rows.append(
            {
                "relpath": raw.relpath,
                "role": raw.role,
                "bytes": path.stat().st_size,
                "sha256": observed,
                "description": raw.description,
            }
        )
    return rows


def released_values() -> dict[str, Any]:
    """The counts each consumed artifact carries, read off the pinned bytes."""
    si_text = vl.raw_mirror_dir(data_root()) / vl.SI_TEXT.relpath
    growth = vl.read_table_s1(si_text)
    stored = [c for c in vl.TITER_CELLS if c.hours in vl.TITER_HOURS_STORED]
    refused = [c for c in vl.TITER_CELLS if c.hours not in vl.TITER_HOURS_STORED]
    return {
        "titers_stated": len(vl.TITER_CELLS),
        "titers_stored": len(stored),
        "titers_refused": len(refused),
        "titers_refused_reason": vl.TITER_48H_REFUSED,
        "titer_strains": sorted({c.strain for c in vl.TITER_CELLS}),
        "titer_sampling_hours": sorted({c.hours for c in vl.TITER_CELLS}),
        "growth_rows": len(growth),
        "growth_strains": sorted({row.strain for row in growth}),
        "growth_carbon_sources": sorted({row.carbon_source for row in growth}),
        "growth_zero_rates": sum(1 for row in growth if row.rate_per_hour == 0.0),
        "quotes_verified": vl.verify_quotes(data_root()),
        "symbol_locus_tags": _symbols(),
    }


def _symbols() -> dict[str, str]:
    """Every gene symbol the paper names, and the locus tag it is stored as."""
    from torchcell.datasets.bacteria_common import bacterial_genome

    genome = bacterial_genome("pputida", "KT2440", data_root())
    resolution = vl.resolve_symbols(genome, data_root())
    return dict(sorted(resolution.locus_tags.items()))


def compendium_samples() -> tuple[pd.DataFrame, dict[str, Any]]:
    """The compendium's carbon-source samples at this paper's two compounds."""
    path = osp.join(data_root(), "torchcell-raw", COMPENDIUM_KEY, COMPENDIUM_RELEASE)
    observed = sha256_file(path)
    if observed != COMPENDIUM_SHA256:
        raise RuntimeError(f"{path} hashes {observed}, not {COMPENDIUM_SHA256}")
    metadata = pd.read_excel(path, sheet_name="metadata")
    in_group = metadata[metadata["expGroup"] == COMPENDIUM_GROUP]
    frame = in_group[
        in_group["condition_1"].isin({COMPENDIUM_VALEROLACTAM, COMPENDIUM_5AVA})
    ]
    columns = [
        "expName",
        "condition_1",
        "concentration_1",
        "units_1",
        "media",
        "vessel",
        "mutantLibrary",
        "person",
        "dateStarted",
    ]
    table = frame[columns].sort_values("expName").reset_index(drop=True)
    summary = {
        "release_file": f"$DATA_ROOT/torchcell-raw/{COMPENDIUM_KEY}/{COMPENDIUM_RELEASE}",
        "sha256": observed,
        "samples_total": int(len(metadata)),
        "samples_in_group": int(len(in_group)),
        "valerolactam_samples": sorted(
            frame[frame["condition_1"] == COMPENDIUM_VALEROLACTAM]["expName"]
        ),
        "aminovalerate_samples": sorted(
            frame[frame["condition_1"] == COMPENDIUM_5AVA]["expName"]
        ),
        "valerolactam_doses_mm": sorted(
            {
                float(value)
                for value in frame[frame["condition_1"] == COMPENDIUM_VALEROLACTAM][
                    "concentration_1"
                ]
            }
        ),
    }
    return table, summary


def served_coverage(sample_names: list[str]) -> dict[str, Any]:
    """What the served Borchert 2024 store holds for those samples, by content.

    The store is read once and its handle closed: a held handle makes the next reader
    of this store fail with "already open in this process".
    """
    from torchcell.datasets.pputida.borchert2024 import RbTnseqBorchert2024Dataset

    wanted = set(sample_names)
    dataset = RbTnseqBorchert2024Dataset(
        root=osp.join(data_root(), COMPENDIUM_ROOT_REL)
    )
    per_sample: Counter[str] = Counter()
    loci: dict[str, set[str]] = {}
    for index in range(len(dataset)):
        experiment = dataset[index]["experiment"]
        sample = experiment["phenotype"]["screen_id"]
        if sample not in wanted:
            continue
        per_sample[sample] += 1
        loci.setdefault(sample, set()).add(
            experiment["genotype"]["perturbations"][0]["systematic_gene_name"]
        )
    total = len(dataset)
    dataset.close_lmdb()
    return {
        "served_dataset": "RbTnseqBorchert2024Dataset",
        "served_store": f"$DATA_ROOT/{COMPENDIUM_ROOT_REL}",
        "served_records_total": total,
        "records_per_sample": dict(sorted(per_sample.items())),
        "loci_per_sample": {k: len(v) for k, v in sorted(loci.items())},
        "samples_absent": sorted(wanted - set(per_sample)),
    }


def main(argv: list[str] | None = None) -> int:
    """Measure the release and the subsumption, write the results, print the JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)

    table, compendium = compendium_samples()
    valerolactam = list(compendium["valerolactam_samples"])
    coverage = served_coverage(valerolactam)
    if len(valerolactam) != RESULTS_EXPERIMENTS:
        raise RuntimeError(
            f"the compendium holds {len(valerolactam)} valerolactam carbon-source "
            f"samples and the Results state {RESULTS_EXPERIMENTS}"
        )
    if compendium["valerolactam_doses_mm"] != [METHODS_CONDITION_MM]:
        raise RuntimeError(
            f"the compendium doses those samples at {compendium['valerolactam_doses_mm']} "
            f"mM and the Methods state {METHODS_CONDITION_MM} mM"
        )
    if coverage["samples_absent"]:
        raise RuntimeError(
            f"the served store is missing {coverage['samples_absent']}; the "
            "subsumption claim would be false"
        )
    results: dict[str, Any] = {
        "script": "experiments/036-dataset-fixes-before-kg-build/scripts/"
        "thompson2019_valerolactam_release_inventory.py",
        "row": "59 Thompson 2019 valerolactam",
        "citation_key": vl.CITATION_KEY,
        "doi": vl.DOI,
        "measured_at": datetime.now(UTC).isoformat(),
        "mirror": mirror_inventory(),
        "released_values": released_values(),
        "rbtnseq_subsumption": {
            "methods_condition": "10 mM valerolactam, sole carbon source, 48-well "
            "plate, Tecan Infinite F200",
            "results_experiments": RESULTS_EXPERIMENTS,
            "compendium": compendium,
            "served": coverage,
            "verdict": "SUBSUMED at the sample level: both valerolactam carbon-source "
            "samples the Results count are compendium samples the served "
            f"{coverage['served_dataset']} already carries in full "
            f"({coverage['records_per_sample']}), so an RB-TnSeq loader for this row "
            "would store every one of those values a second time. The paper's own "
            f"release is the Fitness Browser alone ({vl.FITNESS_BROWSER}), with no "
            "per-experiment accession, so there are no other bytes to load.",
            "aminovalerate_not_this_paper": "the 5-aminovalerate samples that appear "
            "beside them in Fig. S2 are the lysine paper's, by the SI's own "
            "attribution ('All non-valerolactam fitness experiments are from Thompson "
            "et. al 2019'), so they belong to schedule row 55 and are listed here only "
            "to keep the two rows' samples apart",
            "what_the_compendium_drops": "the compendium eliminated loci lacking a "
            "value in some sample, and this paper released no per-gene file to recover "
            "them from: its only supplementary file is a figure-and-one-table PDF. "
            "That is the same state the Thompson 2020 and Schmidt 2022 rows measured.",
        },
        "loader": {
            "module": "torchcell/datasets/pputida/thompson2019_valerolactam.py",
            "datasets": [
                "ValerolactamTiterThompson2019Dataset",
                "LactamGrowthRateThompson2019Dataset",
            ],
            "expected_records": {
                "ValerolactamTiterThompson2019Dataset": vl.EXPECTED_TITER_RECORDS,
                "LactamGrowthRateThompson2019Dataset": vl.EXPECTED_GROWTH_RECORDS,
            },
        },
    }
    os.makedirs(RESULTS, exist_ok=True)
    table.to_csv(SAMPLES_CSV, index=False)
    with open(OUT, "w") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))
    print(f"wrote {OUT}")
    print(f"wrote {SAMPLES_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
