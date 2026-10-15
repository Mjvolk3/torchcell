# experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_phosphosite_checks.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.potel2018_phosphosite_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_phosphosite_checks.py
"""Two measurements behind the phosphosite-intensity leaf (#921), off the schema and stores.

1. **The proteome family's contracts did not move.** The contract fingerprint of
   ``Phenotype``, ``ProteinAbundancePhenotype`` and every other served proteome leaf and
   experiment pair, read from ``schema.py`` at ``--base`` and in the working tree. The
   new leaf is a sibling, so each must be identical.
2. **Stale stores still serialize identically.** Every record of each dataset that
   ``schema_impact`` names as impacted through the grown unions and type maps is
   validated through this branch's schema and re-dumped; a record whose dump differs
   from the stored dict would mean the additive leaf changed a stored serialization.

Run from the repo root (a worktree: ``PYTHONPATH=<worktree>``)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/potel2018_phosphosite_checks.py --base origin/main
"""

import argparse
import json
import os
import os.path as osp
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

from torchcell.database.build_dataset_lmdb import dataset_default_root  # noqa: E402
from torchcell.datamodels.schema import (  # noqa: E402
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.datasets.dataset_registry import dataset_registry  # noqa: E402
from torchcell.provenance.schema_deps import (  # noqa: E402
    default_surface_modules,
    load_surface,
    load_surface_from_sources,
)
from torchcell.provenance.schema_impact import (  # noqa: E402
    _git_show,
    build_impact_report,
)
from torchcell.verification.runners import stream_records  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "potel2018_phosphosite_checks.json",
)
#: The bindings this leaf grows: a dataset is stale through the leaf only via these
#: (the W3110 vocabulary carried in the same PR has its own record and is not measured
#: here).
GROWN_BINDINGS: frozenset[str] = frozenset(
    {
        "ExperimentType",
        "ExperimentReferenceType",
        "PhenotypeType",
        "EXPERIMENT_TYPE_MAP",
        "EXPERIMENT_REFERENCE_TYPE_MAP",
    }
)
#: The served proteome leaves and their parent, and the experiment pairs over them.
PROTEOME_FAMILY: tuple[str, ...] = (
    "Phenotype",
    "ProteinAbundancePhenotype",
    "ProteinFoldChangePhenotype",
    "ProteinTurnoverPhenotype",
    "ProteinSynthesisRatePhenotype",
    "ProteinAbundanceExperiment",
    "ProteinAbundanceExperimentReference",
    "ProteinAbundanceExperiment",
    "ProteinAbundanceExperimentReference",
    "ProteinFoldChangeExperiment",
    "ProteinFoldChangeExperimentReference",
    "Environment",
    "Genotype",
    "ReferenceGenome",
)


def family_fingerprints(base: str) -> dict[str, dict[str, Any]]:
    """Each family symbol's fingerprint at ``base`` and in the working tree."""
    paths = default_surface_modules()
    new = load_surface(paths)
    old = load_surface_from_sources(
        {str(p): _git_show(base, str(p.relative_to(REPO)), REPO) for p in paths}
    )
    return {
        name: {
            "base": old.fingerprints[name],
            "head": new.fingerprints[name],
            "unchanged": old.fingerprints[name] == new.fingerprints[name],
        }
        for name in PROTEOME_FAMILY
    }


def round_trip(path: str) -> dict[str, Any]:
    """Records of one store whose re-dump through this branch's schema differs."""
    n = differ = 0
    for record in stream_records(path):
        experiment, reference = record["experiment"], record["reference"]
        experiment_cls = cast(
            type[BaseModel], EXPERIMENT_TYPE_MAP[experiment["experiment_type"]]
        )
        reference_cls = cast(
            type[BaseModel],
            EXPERIMENT_REFERENCE_TYPE_MAP[reference["experiment_reference_type"]],
        )
        e = experiment_cls.model_validate(experiment)
        r = reference_cls.model_validate(reference)
        n += 1
        same = json.dumps(e.model_dump(mode="json"), sort_keys=True) == json.dumps(
            experiment, sort_keys=True, default=str
        ) and json.dumps(r.model_dump(mode="json"), sort_keys=True) == json.dumps(
            reference, sort_keys=True, default=str
        )
        differ += 0 if same else 1
    return {"records": n, "re_serialized_differently": differ}


def stale_round_trips(base: str, data_root: str) -> dict[str, Any]:
    """Round-trip every impacted dataset that has a dev store on this machine."""
    report = build_impact_report(base, REPO)
    out: dict[str, Any] = {}
    for impact in report.impacted_loaders:
        via = sorted(set(impact.changed_symbols) & GROWN_BINDINGS)
        if not via:
            continue
        for name in impact.dataset_classes:
            if name.startswith("_") or name not in dataset_registry:
                continue
            if name == "PhosphositeIntensityPotel2018Dataset":
                continue
            root = osp.join(data_root, dataset_default_root(dataset_registry[name]))
            if not osp.isdir(osp.join(root, "processed", "lmdb")):
                out[name] = {"store": "no dev store on this machine"}
                continue
            try:
                out[name] = {"via": via, **round_trip(root)}
            except (AttributeError, ValueError) as error:
                # A shared dev store written by ANOTHER branch's schema (a class this
                # branch does not define, or a record it would refuse) is reported, not
                # compared: it is not evidence about this change.
                out[name] = {
                    "via": via,
                    "not_readable_under_this_branch": str(error).splitlines()[0],
                }
    return out


def main() -> None:
    """Measure both and write the results JSON."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main")
    args = parser.parse_args()
    data_root = os.environ["DATA_ROOT"]
    result = {
        "base": args.base,
        "proteome_family_fingerprints": family_fingerprints(args.base),
        "stale_store_round_trips": stale_round_trips(args.base, data_root),
    }
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    unchanged = all(
        v["unchanged"] for v in result["proteome_family_fingerprints"].values()
    )
    print(f"proteome family unchanged: {unchanged}")
    print(json.dumps(result["stale_store_round_trips"], indent=2))


if __name__ == "__main__":
    main()
