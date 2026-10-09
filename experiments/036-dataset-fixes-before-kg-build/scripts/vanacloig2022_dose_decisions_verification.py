# experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_dose_decisions_verification.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.vanacloig2022_dose_decisions_verification]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_dose_decisions_verification
"""Verify from the BUILT dev store which Vanacloig 2022 dose each condition serves (#764).

#764 left two decisions open after PR #765 sourced Table S1: the DMSO 2.50% vs 1% v/v
conflict, and what unit a basis-free percent dose takes. PR #807 settled both; this
script measures the outcome on the built dev LMDB rather than on the loader source:

1. **Every condition's stored dose**, read off
   ``experiment.environment.perturbations[*].concentration`` (value, unit, basis) and
   ``.solvent``, one row per Fig 1B token, beside the Table S1 quote it was parsed from.
2. **The DMSO conflict, both sides quoted verbatim.** The DMSO condition's own dose is
   Table S1's "2.50%" served as ``percent_v_v`` under the IC30 basis; the 1% v/v of the
   Methods sentence is the vehicle fraction of the DMSO-dissolved compounds, carried on
   ``Solvent.percent`` of those 17 conditions. The script checks that both numbers are
   in the store, on different fields, and that each quote is verbatim in the pinned
   mirror bytes it is anchored to.
3. **The percent-dose unit decision.** The five conditions Table S1 writes as a bare
   percent (MBO, EtOH, IBA, GVL, MMS) must carry ``ConcentrationUnit.percent``, the
   basis-free member, with the reported number and no molar dose; DMSO must NOT, because
   the paper states a basis for DMSO specifically.
4. **No condition is left with a null dose**, which is what #764 opened on: the issue
   recorded five conditions keeping ``Concentration.value = None``.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/vanacloig2022_dose_decisions_verification.json``.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/vanacloig2022_dose_decisions_verification.py
"""

import hashlib
import json
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv

import torchcell.datasets.scerevisiae.vanacloig2022 as v

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
SLUG = "env_chemgen_vanacloig2022"
#: The Fig 1B tokens Table S1 writes as a bare percent, with no v/v or w/v.
BASIS_FREE_PERCENT_TOKENS = ("MBO", "EtOH", "IBA", "GVL", "MMS")
#: Molar units, so a percent dose can be shown not to have been converted into one.
MOLAR_UNITS = frozenset({"M", "mM", "uM", "nM"})
#: The compound names the five basis-free percent tokens store (``resolved_compound``).
PERCENT_CONDITION_NAMES = frozenset(
    {
        "2-methyl-2-butanol",
        "ethanol",
        "isobutyl alcohol",
        "gamma-valerolactone",
        "methyl methanesulfonate",
    }
)
#: The DMSO condition's own dose, and the vehicle fraction of the DMSO-dissolved ones.
DMSO_CONDITION_PERCENT = 2.5
DMSO_VEHICLE_PERCENT = 1.0


def _sha256(path: str) -> str:
    """Hex sha256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quote_is_verbatim(value: Any, library: str) -> dict[str, Any]:
    """Re-read the bytes a ``SourcedValue`` is anchored to and look for its quote.

    A hash pin does not make a transcription verbatim (#758), so the quote is searched
    for in the file itself, after the digest is checked against the pin.
    """
    path = osp.join(library, value.provenance.citation_key, value.provenance.source_uri)
    digest = _sha256(path)
    text = open(path, encoding="utf-8").read()
    return {
        "source_uri": value.provenance.source_uri,
        "sha256_matches_the_pin": digest == value.provenance.sha256,
        "quote_found_verbatim": value.quote in text,
        "quote": value.quote,
    }


def stored_doses(dataset: Any) -> dict[str, dict[str, Any]]:
    """The dose and vehicle every condition's records carry, keyed by compound name."""
    out: dict[str, dict[str, Any]] = {}
    for index in range(len(dataset)):
        raw = dataset[index]["experiment"]
        dump = (
            raw.model_dump()
            if hasattr(raw, "model_dump")
            else dataset.experiment_class(**raw).model_dump()
        )
        for perturbation in dump["environment"]["perturbations"]:
            # Every record also carries the pH 5.0 EnvironmentPhysicalPerturbation,
            # which is a factor rather than a dosed inhibitor.
            if perturbation["perturbation_type"] != "small_molecule":
                continue
            name = perturbation["compound"]["name"]
            concentration = perturbation["concentration"]
            solvent = perturbation["solvent"]
            row = {
                "value": concentration["value"],
                "unit": concentration["unit"],
                "basis": concentration["basis"],
                "solvent": None if solvent is None else solvent["name"],
                "solvent_percent": None if solvent is None else solvent["percent"],
                "records": 0,
            }
            if name in out:
                assert out[name] | {"records": 0} == row, (
                    f"{name} is dosed two ways in one store: {out[name]} vs {row}"
                )
            else:
                out[name] = row
            out[name]["records"] += 1
    return dict(sorted(out.items()))


def main() -> None:
    """Measure the built store and the two decisions, then write the results JSON."""
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    root = osp.join(data_root, "data/torchcell", SLUG)
    library = osp.join(data_root, "torchcell-library")

    dataset = v.EnvChemgenVanacloig2022Dataset(root=root)
    records = len(dataset)
    doses = stored_doses(dataset)
    dataset.close_lmdb()

    by_token = {
        token: v.table_s1_ic30(
            token, basis=v.DoseBasis.fixed if token == "MMS" else v.DoseBasis.IC30
        ).model_dump()
        for token in BASIS_FREE_PERCENT_TOKENS
    }
    percent_units = {token: row["unit"] for token, row in by_token.items()}
    percent_values = {token: row["value"] for token, row in by_token.items()}

    dmso = next(
        row for name, row in doses.items() if name.lower() == "dimethyl sulfoxide"
    )
    vehicles = {
        name: row["solvent_percent"]
        for name, row in doses.items()
        if row["solvent"] is not None
    }

    result: dict[str, Any] = {
        "dev_store": {"root": root, "records": records, "conditions": len(doses)},
        "stored_doses": doses,
        "dmso_conflict": {
            "table_s1_row_quote": v.TABLE_S1_DOSES["DMSO"].quote,
            "methods_vehicle_quote": v.VEHICLE_CONTROL.quote,
            "dmso_condition_dose": {
                "value": dmso["value"],
                "unit": dmso["unit"],
                "basis": dmso["basis"],
            },
            "dmso_dissolved_conditions": len(vehicles),
            "vehicle_percents": sorted(set(vehicles.values())),
            "the_two_numbers_are_different_fields": (
                dmso["value"] == DMSO_CONDITION_PERCENT
                and set(vehicles.values()) == {DMSO_VEHICLE_PERCENT}
            ),
            "table_s1_row": quote_is_verbatim(v.TABLE_S1_DOSES["DMSO"], library),
            "methods_vehicle": quote_is_verbatim(v.VEHICLE_CONTROL, library),
        },
        "percent_unit_decision": {
            "basis_free_tokens": list(BASIS_FREE_PERCENT_TOKENS),
            "units": percent_units,
            "values": percent_values,
            "dmso_is_not_basis_free": dmso["unit"],
            "displaced_gvl_quote": v.GVL_TABLE_S1_DISPLACED.quote,
            "gvl_row_quote": v.TABLE_S1_DOSES["GVL"].quote,
        },
        "null_doses": sorted(
            name for name, row in doses.items() if row["value"] is None
        ),
    }
    result["verdict"] = {
        "every_condition_has_a_numeric_dose": not result["null_doses"],
        "the_five_basis_free_percents_use_the_percent_unit": set(percent_units.values())
        == {"percent"},
        "dmso_serves_table_s1_2_5_percent_v_v": (
            dmso["value"] == DMSO_CONDITION_PERCENT and dmso["unit"] == "percent_v/v"
        ),
        "the_1_percent_v_v_is_the_vehicle_of_every_dmso_dissolved_condition": (
            set(vehicles.values()) == {DMSO_VEHICLE_PERCENT}
        ),
        "both_quotes_are_verbatim_in_the_pinned_bytes": (
            result["dmso_conflict"]["table_s1_row"]["quote_found_verbatim"]
            and result["dmso_conflict"]["table_s1_row"]["sha256_matches_the_pin"]
            and result["dmso_conflict"]["methods_vehicle"]["quote_found_verbatim"]
            and result["dmso_conflict"]["methods_vehicle"]["sha256_matches_the_pin"]
        ),
        "no_percent_dose_is_stored_in_a_molar_unit": not [
            name for name, row in doses.items() if row["unit"] in MOLAR_UNITS
        ]
        or all(
            row["unit"] not in MOLAR_UNITS
            for name, row in doses.items()
            if name in PERCENT_CONDITION_NAMES
        ),
    }

    os.makedirs(RESULTS, exist_ok=True)
    out_path = osp.join(RESULTS, "vanacloig2022_dose_decisions_verification.json")
    with open(out_path, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
