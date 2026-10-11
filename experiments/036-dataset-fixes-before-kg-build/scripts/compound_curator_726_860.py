# experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.compound_curator_726_860]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860
r"""Run the compound-identity curator over the bacterial condition labels (#726, #860).

The curator (``torchcell.datamodels.compound_identity_curate``) is run on ONE input list,
``compound_identity_inputs/bacterial_conditions.txt``, and its rows are merged into the
pinned table ADDITIVELY, the way the bacterial-media branch added its 32 salt rows: no
existing row's name, InChIKey, CID, ChEBI id or status changes, and nothing is removed.

Two things make the merge more than an append, and both are checked rather than assumed:

- **A label whose CID an existing row already holds is FOLDED into that row as a
  synonym.** Two rows for one CID would be two compound nodes for one substance, which is
  exactly what the curator's own CID collapse exists to prevent (``NaCl`` and ``sodium
  chloride``). The existing row keeps its name; the label gains it. The InChIKeys must
  agree or the run stops.
- **No lookup key may be claimed twice.** A new key that an existing row already answers
  stops the run, since that label would then resolve to a different compound than today.

Every directive is reported with its evidence: the PubChem record the curator adopted
(the exact endpoint, its Title, InChIKey and connectivity SMILES), the CAS number Price
2018 Tables S2-S4 state for the label when they state one and the CID PubChem gives that
CAS, the stores the label appears in (from the committed gap inventory), and the name its
records carry after the merge. A label whose name changes is listed, because that is a
node rename in every record carrying it (#726, the IPTG finding); none of these labels
resolved before, so the rename is only from the release spelling to the row's canonical
spelling. That list covers the spellings the committed gap inventory saw in the bacterial
stores; the measurement over EVERY mapped store, yeast included, is
``compound_curator_726_860_store_reach.py``.

Writes ``experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860.csv``
(one row per input label) and ``.../compound_curator_726_860.json`` (summary, counts,
folds, renames, CAS cross-check, cache and table digests). With ``--write`` it also
rewrites ``torchcell/datamodels/compound_identity_table.json``; the printed sha256 is
then re-pinned into ``compound_identity.py::_TABLE_SHA256`` by hand.

Run from the repo root (the PubChem response cache makes a rerun offline)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/compound_curator_726_860.py \
        --cache $DATA_ROOT/tmp/pubchem_curator_cache_726_860_2026-10-10.json \
        --min-interval 1.0 --write \
        | tee experiments/036-dataset-fixes-before-kg-build/results/compound_curator_726_860_log.txt
"""

import argparse
import csv
import hashlib
import json
import os
import os.path as osp
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import openpyxl
from dotenv import load_dotenv

from torchcell.datamodels.compound_identity_curate import (
    CuratedRow,
    CurationDirective,
    PubChemClient,
    _first_property,
    cid_property_url,
    curate,
    name_property_url,
    serialize,
)

RESULTS = "experiments/036-dataset-fixes-before-kg-build/results"
INPUT_LIST = Path(
    "torchcell/datamodels/compound_identity_inputs/bacterial_conditions.txt"
)
TABLE = Path("torchcell/datamodels/compound_identity_table.json")
GAPS = Path(RESULTS) / "bacterial_compound_identity_gaps.json"
#: Price 2018 Supplementary Tables, the same pinned file media.py quotes Table S18 from.
PRICE_SI3_REL = "torchcell-library/priceMutantPhenotypesThousands2018/si/si3.xlsx"
PRICE_SI3_SHA256 = "e5dbf3d5c97cfc12f49d7fd83f84bc95c16cbe963309a561fff20f442788b879"
PRICE_CAS_SHEETS = ("TableS2_Carbon", "TableS3_Nitrogen", "TableS4_Stress")
#: Labels the committed gap inventory could not see, with where they are used: the 20
#: Borchert 2024 samples on RCH2_defined_noCarbon were dropped, so their compounds were
#: never built, and the medium's salts had no library entry (#860).
EXTRA_USES: dict[str, list[str]] = {
    "4-hydroxyvalerate": ["RbTnseqBorchert2024Dataset (RCH2 samples set1IT082/083)"],
    "potassium acetate": ["RbTnseqBorchert2024Dataset (RCH2 samples set1IT084-087)"],
    "potassium chloride": ["MEDIA_LIBRARY RCH2_DEFINED_NOCARBON_PRICE2018"],
    "sodium dihydrogen phosphate monohydrate": [
        "MEDIA_LIBRARY RCH2_DEFINED_NOCARBON_PRICE2018"
    ],
    "pipes sesquisodium salt": ["MEDIA_LIBRARY RCH2_DEFINED_NOCARBON_PRICE2018"],
}


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of a byte string."""
    return hashlib.sha256(data).hexdigest()


def price_cas_numbers(data_root: str) -> dict[str, list[dict[str, Any]]]:
    """Casefolded compound label -> every (sheet, row, CAS) Price 2018 Tables S2-S4 state.

    Re-hashes the workbook first: a CAS number is evidence only against the pinned bytes.
    """
    path = Path(data_root) / PRICE_SI3_REL
    digest = sha256_bytes(path.read_bytes())
    if digest != PRICE_SI3_SHA256:
        raise RuntimeError(f"{path}: sha256 {digest}, pinned {PRICE_SI3_SHA256}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        workbook = openpyxl.load_workbook(path, read_only=True)
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sheet in PRICE_CAS_SHEETS:
        header_row: int | None = None
        for index, row in enumerate(workbook[sheet].iter_rows(values_only=True)):
            if row and row[0] == "Compound":
                header_row = index
                continue
            if header_row is None or not row or row[0] is None:
                continue
            out[str(row[0]).strip().lower()].append(
                {"sheet": sheet, "row": index, "cas": row[1]}
            )
    return dict(out)


def gap_uses() -> tuple[dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    """Casefolded label -> its stores, and -> each release spelling's own stores.

    Both read the committed gap inventory, whose labels are the names the BUILT records
    carry, so a spelling here is exactly what a record's ``Compound.name`` is today.
    """
    payload = json.loads(GAPS.read_text())
    uses: dict[str, set[str]] = defaultdict(set)
    spellings: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for gap in payload["gaps"]:
        key = gap["label"].strip().lower()
        uses[key].update(gap["stores"])
        spellings[key][gap["label"]].update(gap["stores"])
    for key, extra in EXTRA_USES.items():
        uses[key].update(extra)
    return (
        {key: sorted(value) for key, value in uses.items()},
        {
            key: {spelling: sorted(stores) for spelling, stores in by.items()}
            for key, by in spellings.items()
        },
    )


def merge(
    existing: list[CuratedRow], new: list[CuratedRow], labels: dict[str, str]
) -> tuple[list[CuratedRow], list[dict[str, Any]]]:
    """Existing rows plus the new ones; a new row whose CID is already held is folded.

    ``labels`` maps each casefolded input label to its spelling. A fold adds only those
    labels to the existing row, never the PubChem Title the curator names an uncurated row
    after, so every key this run adds is a label some source writes. A row that is NOT
    folded must be named by an input label for the same reason.

    Raises when a fold would change an existing row's InChIKey, when one CID is held by
    two existing rows (no single row to fold into), when a new row is named by no input
    label, or when any lookup key would be claimed by two rows after the merge.
    """
    by_cid: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(existing):
        if row.pubchem_cid is not None:
            by_cid[row.pubchem_cid].append(index)
    merged = [row.model_copy(deep=True) for row in existing]
    folds: list[dict[str, Any]] = []
    added: list[CuratedRow] = []
    for row in new:
        held = by_cid.get(row.pubchem_cid) if row.pubchem_cid is not None else None
        if not held:
            if row.name.strip().lower() not in labels:
                raise ValueError(
                    f"new row {row.name!r} is named by no input label; mark one of "
                    f"its labels {row.synonyms} canonical"
                )
            added.append(row)
            continue
        if len(held) != 1:
            raise ValueError(
                f"{row.name!r}: CID {row.pubchem_cid} is held by "
                f"{[existing[i].name for i in held]}"
            )
        target = merged[held[0]]
        if row.inchikey != target.inchikey:
            raise ValueError(
                f"{row.name!r} (InChIKey {row.inchikey}) would fold into "
                f"{target.name!r} (InChIKey {target.inchikey})"
            )
        row_labels = sorted(
            {
                labels[key]
                for key in (k.strip().lower() for k in [row.name, *row.synonyms])
                if key in labels
            }
        )
        target_keys = {k.strip().lower() for k in [target.name, *target.synonyms]}
        gained = [
            label for label in row_labels if label.strip().lower() not in target_keys
        ]
        target.synonyms = sorted([*target.synonyms, *gained])
        folds.append(
            {
                "labels": row_labels,
                "into": target.name,
                "pubchem_cid": target.pubchem_cid,
                "inchikey": target.inchikey,
                "synonyms_gained": gained,
            }
        )
    rows = merged + added
    owners: dict[str, str] = {}
    for row in rows:
        for key in {k.strip().lower() for k in [row.name, *row.synonyms]}:
            if key in owners:
                raise ValueError(
                    f"lookup key {key!r} claimed by {owners[key]!r} and {row.name!r}"
                )
            owners[key] = row.name
    rows.sort(key=lambda r: (r.name.lower(), r.pubchem_cid or 0))
    return rows, folds


def adopted_record(
    directive: CurationDirective, cache: dict[str, Any]
) -> dict[str, Any]:
    """The PubChem record the curator adopted for one directive, read from the cache."""
    if directive.skip_lookup:
        return {"endpoint": None}
    url = (
        cid_property_url(directive.cid)
        if directive.cid is not None
        else name_property_url(directive.name_query)
    )
    prop = _first_property(cache.get(url))
    if prop is None:
        return {"endpoint": url, "pubchem_title": None}
    return {
        "endpoint": url,
        "pubchem_title": prop.title,
        "pubchem_record_cid": prop.cid,
        "pubchem_smiles": prop.smiles,
    }


def main() -> None:
    """Curate the list, merge it, and write the evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--min-interval", type=float, default=1.0)
    parser.add_argument("--write", action="store_true", help="rewrite the table")
    args = parser.parse_args()
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]

    old_bytes = TABLE.read_bytes()
    old_payload = json.loads(old_bytes)
    existing = [CuratedRow(**r) for r in old_payload["records"]]
    if serialize(existing).encode("utf-8") != old_bytes:
        raise RuntimeError("the pinned table does not round-trip through serialize()")

    new_rows, directives = curate([INPUT_LIST], [], args.cache, args.min_interval)
    label_spelling = {d.label.strip().lower(): d.label for d in directives}
    rows, folds = merge(existing, new_rows, label_spelling)
    text = serialize(rows)
    new_sha = sha256_bytes(text.encode("utf-8"))

    # Cross-check every label Price 2018 states a CAS for (cached after the first run).
    cas_by_label = price_cas_numbers(data_root)
    client = PubChemClient(args.cache, args.min_interval)
    cache = json.loads(args.cache.read_text())

    key_to_row = {
        k.strip().lower(): row for row in rows for k in [row.name, *row.synonyms]
    }
    folded_label = {
        label.strip().lower(): fold["into"]
        for fold in folds
        for label in fold["labels"]
    }
    uses, spellings = gap_uses()
    report: list[dict[str, Any]] = []
    for directive in directives:
        key = directive.label.strip().lower()
        row = key_to_row[key]
        record = adopted_record(directive, cache)
        cas_rows = cas_by_label.get(key, [])
        cas_numbers = sorted({str(c["cas"]) for c in cas_rows if c["cas"]})
        cas_cids: dict[str, int | None] = {}
        for number in cas_numbers:
            prop = client.property_by_name(number)
            cas_cids[number] = prop.cid if prop is not None else None
        report.append(
            {
                "label": directive.label,
                "stores": uses.get(key, []),
                "directive": "; ".join(
                    part
                    for part in (
                        "canonical" if directive.canonical else "",
                        f"query={directive.query}" if directive.query else "",
                        f"cid={directive.cid}" if directive.cid else "",
                        "mixture" if directive.mixture_reason else "",
                        "undefined" if directive.undefined_reason else "",
                        "unresolved" if directive.unresolved_reason else "",
                    )
                    if part
                ),
                "resolution_status": row.resolution_status,
                "row_name": row.name,
                "record_spellings": spellings.get(key, {directive.label: []}),
                "folded_into_existing_row": folded_label.get(key),
                "inchikey": row.inchikey,
                "pubchem_cid": row.pubchem_cid,
                "chebi_id": row.chebi_id,
                **record,
                "price2018_cas": cas_numbers,
                "price2018_cas_rows": [
                    f"{c['sheet']} row {c['row']}" for c in cas_rows if c["cas"]
                ],
                "price2018_cas_cid": cas_cids,
                "cas_agrees": (
                    None
                    if not cas_cids
                    else any(cid == row.pubchem_cid for cid in cas_cids.values())
                ),
                "unresolved_reason": row.unresolved_reason,
            }
        )
    client.flush()

    if args.write:
        TABLE.write_text(text, encoding="utf-8")

    status_counts = Counter(r["resolution_status"] for r in report)
    # A record's compound name today is its release spelling (no row matched); after the
    # merge it is the row's name. Every spelling that differs is a node rename in the
    # stores that carry it.
    renames = [
        {
            "spelling": spelling,
            "now": r["row_name"],
            "case_only": spelling.lower() == r["row_name"].lower(),
            "stores": stores,
        }
        for r in report
        for spelling, stores in sorted(r["record_spellings"].items())
        if spelling != r["row_name"]
    ]
    summary = {
        "input_list": str(INPUT_LIST),
        "input_list_sha256": sha256_bytes(INPUT_LIST.read_bytes()),
        "pubchem_cache": str(args.cache),
        "pubchem_cache_sha256": sha256_bytes(args.cache.read_bytes()),
        "pubchem_cache_responses": len(json.loads(args.cache.read_text())),
        "price2018_si3": PRICE_SI3_REL,
        "price2018_si3_sha256": PRICE_SI3_SHA256,
        "table_before": {"sha256": sha256_bytes(old_bytes), "records": len(existing)},
        "table_after": {"sha256": new_sha, "records": len(rows)},
        "table_written": args.write,
        "labels": len(report),
        "labels_by_status": dict(sorted(status_counts.items())),
        "labels_with_an_inchikey": sum(1 for r in report if r["inchikey"]),
        "rows_added": len(rows) - len(existing),
        "labels_folded_into_existing_rows": sum(len(f["labels"]) for f in folds),
        "folds": folds,
        "renames": renames,
        "cas_checked": sum(1 for r in report if r["cas_agrees"] is not None),
        "cas_disagreements": [
            {
                "label": r["label"],
                "adopted_cid": r["pubchem_cid"],
                "cas": r["price2018_cas_cid"],
            }
            for r in report
            if r["cas_agrees"] is False
        ],
        "unresolved": [
            {
                "label": r["label"],
                "status": r["resolution_status"],
                "why": r["unresolved_reason"],
            }
            for r in report
            if r["inchikey"] is None
        ],
    }

    os.makedirs(RESULTS, exist_ok=True)
    with open(osp.join(RESULTS, "compound_curator_726_860.json"), "w") as handle:
        json.dump({"summary": summary, "labels": report}, handle, indent=2)
        handle.write("\n")
    columns = [
        "label",
        "resolution_status",
        "row_name",
        "inchikey",
        "pubchem_cid",
        "chebi_id",
        "directive",
        "folded_into_existing_row",
        "pubchem_title",
        "pubchem_smiles",
        "endpoint",
        "price2018_cas",
        "cas_agrees",
        "stores",
        "unresolved_reason",
    ]
    with open(osp.join(RESULTS, "compound_curator_726_860.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in report:
            writer.writerow(
                {
                    **r,
                    "stores": ";".join(r["stores"]),
                    "price2018_cas": ";".join(r["price2018_cas"]),
                }
            )

    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"folds", "renames"}},
            indent=2,
        )
    )
    print(f"folds ({len(folds)}):")
    for fold in folds:
        print(f"  {fold['labels']} -> {fold['into']!r} (CID {fold['pubchem_cid']})")
    print(f"renames ({len(renames)}):")
    for rename in renames:
        print(
            f"  {rename['spelling']!r} -> {rename['now']!r}"
            f"{' (case only)' if rename['case_only'] else ''} in {rename['stores']}"
        )
    print(f"table sha256 = {new_sha}{'' if args.write else ' (not written)'}")


if __name__ == "__main__":
    main()
