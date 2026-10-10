# experiments/036-dataset-fixes-before-kg-build/scripts/li2014_synthesis_rate_checks.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.li2014_synthesis_rate_checks]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/li2014_synthesis_rate_checks.py
"""Two measurements behind the Li 2014 loader (#857), read off built dev stores.

1. **Duplication by content.** For every record of each served E. coli proteome-like
   store (Schmidt 2016, Ishii 2007, Brunk 2016 proteomes and Gupta 2024 turnover), and
   every Li 2014 medium: the shared protein keys, how many shared keys carry the SAME
   number, and the Spearman correlation of the two label maps. Schmidt and Ishii key
   on BW25113 GenBank tags, so their keys are carried to MG1655 b-numbers through the
   ECK cross-strain synonym both assemblies record (one-to-one ECKs only). A re-release of Li's
   numbers would show many exact matches and a rank correlation near 1.
2. **Served records still serialize identically.** Every record of each dataset the
   schema-impact check names (impacted through the grown ``ExperimentType`` union) is
   validated through this branch's schema and re-dumped; a record whose dump differs
   from the stored dict would mean the additive leaf changed a served serialization.

Run from the repo root::

    python experiments/036-dataset-fixes-before-kg-build/scripts/li2014_synthesis_rate_checks.py
"""

import json
import os
import os.path as osp
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel
from scipy.stats import spearmanr

load_dotenv()

from torchcell.datamodels.schema import (  # noqa: E402
    EXPERIMENT_REFERENCE_TYPE_MAP,
    EXPERIMENT_TYPE_MAP,
)
from torchcell.datasets.bacteria_common import bacterial_genome  # noqa: E402
from torchcell.verification.runners import stream_records  # noqa: E402

LI_ROOT = "data/torchcell/protein_synthesis_rate_li2014"
COMPARED = {
    "ProteomeSchmidt2016Dataset": "data/torchcell/proteome_schmidt2016",
    "ProteomeIshii2007Dataset": "data/torchcell/proteome_ishii2007",
    "ProteomeBrunk2016Dataset": "data/torchcell/proteome_brunk2016",
    "ProteinTurnoverGupta2024Dataset": "data/torchcell/protein_turnover_gupta2024",
}
#: The datasets `scripts/schema_impact_check.py --base origin/main` names as impacted
#: through ExperimentType on this branch (public ones with a dev store).
IMPACTED = {
    "ProteinTurnoverGupta2024Dataset": "data/torchcell/protein_turnover_gupta2024",
    "ProteomeIshii2007Dataset": "data/torchcell/proteome_ishii2007",
    "MetabolomeIshii2007Dataset": "data/torchcell/metabolome_ishii2007",
    "FluxIshii2007Dataset": "data/torchcell/flux_ishii2007",
    "RnaseqCaglar2017Dataset": "data/torchcell/rnaseq_caglar2017",
    "ProteomeCaglar2017Dataset": "data/torchcell/proteome_caglar2017",
    "ProteinFoldChangeCaglar2017Dataset": "data/torchcell/protein_fold_change_caglar2017",
    "MetabolomeSchastnaya2021Dataset": "data/torchcell/metabolome_schastnaya2021",
    "ProteomeDeSiqueira2025Dataset": "data/torchcell/proteome_desiqueira2025",
    "ProteomePercentDeSiqueira2025Dataset": "data/torchcell/proteome_percent_desiqueira2025",
    "ProteomeLog10PercentDeSiqueira2025Dataset": (
        "data/torchcell/proteome_log10_percent_desiqueira2025"
    ),
    "IsoprenolTiterDeSiqueira2025Dataset": "data/torchcell/isoprenol_titer_desiqueira2025",
    "PutidaPrecise321Lim2022Dataset": "data/torchcell/putida_precise321_lim2022",
}
RESULTS = osp.join(
    osp.dirname(osp.dirname(osp.abspath(__file__))),
    "results",
    "li2014_synthesis_rate_checks.json",
)


class Overlap(BaseModel):
    """One (Li medium, compared record) pair."""

    li_medium: str
    compared_record: int
    compared_label: str
    n_shared_keys: int
    n_identical_values: int
    spearman: float | None


def _label_map(record: dict[str, Any]) -> tuple[str, dict[str, float]]:
    phenotype = record["experiment"]["phenotype"]
    name = str(phenotype["label_name"])
    return name, {str(k): float(v) for k, v in phenotype[name].items()}


def _eck(table: Any) -> dict[str, str]:
    """Locus tag -> its ECK synonym, for loci with exactly one, ECK used once."""
    pairs = []
    for tag, synonyms in zip(table["locus_tag"], table["synonyms"], strict=True):
        ecks = [s for s in synonyms if str(s).startswith("ECK")]
        if len(ecks) == 1:
            pairs.append((str(tag), str(ecks[0])))
    counts: dict[str, int] = {}
    for _, eck in pairs:
        counts[eck] = counts.get(eck, 0) + 1
    return {tag: eck for tag, eck in pairs if counts[eck] == 1}


def bw25113_to_mg1655(data_root: str) -> dict[str, str]:
    """BW25113 GenBank tag -> MG1655 b-number, through a shared one-to-one ECK."""
    bw = _eck(bacterial_genome("ecoli", "BW25113", data_root).locus_table)
    mg = {
        eck: tag
        for tag, eck in _eck(
            bacterial_genome("ecoli", "MG1655", data_root).locus_table
        ).items()
    }
    return {tag: mg[eck] for tag, eck in bw.items() if eck in mg}


def overlaps(data_root: str) -> dict[str, Any]:
    """Shared keys, identical values and Spearman per Li medium x compared record."""
    to_mg1655 = bw25113_to_mg1655(data_root)
    li_records = list(stream_records(osp.join(data_root, LI_ROOT)))
    li_maps = {
        str(r["experiment"]["environment"]["media"]["name"]): _label_map(r)[1]
        for r in li_records
    }
    out: dict[str, Any] = {}
    for name, root in COMPARED.items():
        path = osp.join(data_root, root)
        if not osp.isdir(path):
            out[name] = "no dev store on this machine"
            continue
        rows: list[Overlap] = []
        for index, record in enumerate(stream_records(path)):
            label, values = _label_map(record)
            if any(k.startswith("BW25113_") for k in values):
                values = {to_mg1655[k]: v for k, v in values.items() if k in to_mg1655}
            for medium, li_map in li_maps.items():
                shared = sorted(set(li_map) & set(values))
                rho = (
                    float(
                        spearmanr(
                            [li_map[k] for k in shared], [values[k] for k in shared]
                        ).statistic
                    )
                    if len(shared) > 2
                    else None
                )
                rows.append(
                    Overlap(
                        li_medium=medium,
                        compared_record=index,
                        compared_label=label,
                        n_shared_keys=len(shared),
                        n_identical_values=sum(
                            1 for k in shared if li_map[k] == values[k]
                        ),
                        spearman=None if rho is None else round(rho, 4),
                    )
                )
        finite = [r.spearman for r in rows if r.spearman is not None]
        out[name] = {
            "records": len({r.compared_record for r in rows}),
            "max_shared_keys": max(r.n_shared_keys for r in rows),
            "total_identical_values": sum(r.n_identical_values for r in rows),
            "spearman_min": min(finite) if finite else None,
            "spearman_max": max(finite) if finite else None,
            "pairs": [r.model_dump() for r in rows],
        }
    out["bw25113_to_mg1655_mapped_tags"] = len(to_mg1655)
    return out


def round_trips(data_root: str) -> dict[str, Any]:
    """Records whose re-dump through this branch's schema differs from the store."""
    out: dict[str, Any] = {}
    for name, root in IMPACTED.items():
        path = osp.join(data_root, root)
        if not osp.isdir(path):
            out[name] = "no dev store on this machine"
            continue
        n = differ = 0
        refused: str | None = None
        for record in stream_records(path):
            experiment, reference = record["experiment"], record["reference"]
            try:
                e = EXPERIMENT_TYPE_MAP[experiment["experiment_type"]].model_validate(
                    experiment
                )
                r = EXPERIMENT_REFERENCE_TYPE_MAP[
                    reference["experiment_reference_type"]
                ].model_validate(reference)
            except ValueError as error:
                # A store written by ANOTHER branch's schema (shared dev tree) is
                # reported, not compared: it is not evidence about this change.
                refused = str(error).splitlines()[0]
                break
            n += 1
            same = json.dumps(e.model_dump(mode="json"), sort_keys=True) == json.dumps(
                experiment, sort_keys=True, default=str
            ) and json.dumps(r.model_dump(mode="json"), sort_keys=True) == json.dumps(
                reference, sort_keys=True, default=str
            )
            differ += 0 if same else 1
        out[name] = {"records": n, "re_serialized_differently": differ}
        if refused is not None:
            out[name]["not_valid_under_this_branch"] = refused
    return out


def main() -> None:
    """Measure both and write the results JSON."""
    data_root = os.environ["DATA_ROOT"]
    result = {"overlap": overlaps(data_root), "round_trip": round_trips(data_root)}
    os.makedirs(osp.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary = {
        "overlap": {
            k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
            if isinstance(v, dict)
            else v
            for k, v in result["overlap"].items()
        },
        "round_trip": result["round_trip"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
