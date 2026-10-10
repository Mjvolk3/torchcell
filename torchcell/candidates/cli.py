# torchcell/candidates/cli.py
# [[torchcell.candidates.cli]]
# https://github.com/Mjvolk3/torchcell/tree/main/torchcell/candidates/cli.py
# Test file: tests/torchcell/candidates/test_cli.py
r"""``candidate-gate``: the CLI over the candidate gates and the verdict store.

Run from the repo root (or a worktree with ``PYTHONPATH`` set to it)::

    python -m torchcell.candidates schema                      # JSON schema for prompts
    python -m torchcell.candidates validate verdict.json       # agent output -> typed
    python -m torchcell.candidates gate --row "D2Cell 2026"    # G1, G2, G3, G4-key
    python -m torchcell.candidates gate --citation-key K --phenotype-class C --write
    python -m torchcell.candidates inventory --citation-key K  # G3 off the mirrors
    python -m torchcell.candidates overlap --citation-key K --released r.csv \
        --store $DATA_ROOT/data/torchcell/<slug> --sample-path ... --key-path ... \
        --value-path ...                                       # G4-value, read-only
    python -m torchcell.candidates ledger                      # dated ledger section
    python -m torchcell.candidates audit                       # #758 quote audit
    python -m torchcell.candidates status                      # store summary

``DATA_ROOT`` is read from the environment (``.env`` via ``load_dotenv``) unless
``--data-root`` is passed. Exit codes: 0 success, 1 a refusal or a failed audit, 2 a
usage or input error.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path
from types import ModuleType
from typing import Final

from dotenv import load_dotenv
from pydantic import ValidationError

from torchcell.candidates import gates, ledger, store
from torchcell.candidates.verdict import (
    CandidateTable,
    CandidateVerdict,
    G4ValueRecord,
    GateName,
    GateResult,
    verdict_json_schema,
)
from torchcell.literature.manifest import sha256_file
from torchcell.verification.sourced import SourcedValue

REPO: Final = Path(__file__).resolve().parents[2]
TABLE_SCRIPTS: Final[dict[str, Path]] = {
    "bacteria": REPO
    / "experiments/database/scripts/build_bacteria_candidate_datasets_table.py",
    "yeast": REPO / "experiments/database/scripts/build_candidate_datasets_table.py",
}
DOI_TO_PMID_CACHE: Final = (
    REPO / "experiments/database/results/candidates/doi2pmid.json"
)


class UsageError(Exception):
    """A user input the CLI cannot act on (exit 2)."""


def load_table_module(table: str, path: Path | None = None) -> ModuleType:
    """Import a candidate-table script by path (they are scripts, not a package)."""
    script = path if path is not None else TABLE_SCRIPTS[table]
    spec = importlib.util.spec_from_file_location(f"_candidate_table_{table}", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def table_rows(table: CandidateTable, module: ModuleType) -> list[gates.CandidateRow]:
    """Every row literal of a table script's ``CANDIDATES``."""
    return [gates.row_from_table(table, c) for c in module.CANDIDATES]


def git_commit(repo: Path = REPO) -> str:
    """HEAD of the tree the gates read."""
    return subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def data_root(value: str | None) -> str:
    """``--data-root``, else ``DATA_ROOT`` from the environment or ``.env``."""
    if value is not None:
        return value
    load_dotenv(REPO / ".env")
    root = os.environ.get("DATA_ROOT")
    if not root:
        raise UsageError("DATA_ROOT is not set; pass --data-root")
    return root


def row_for_citation_key(
    rows: Sequence[gates.CandidateRow],
    citation_key: str,
    index: tuple[gates.ModuleKeys, ...],
    root: str,
) -> gates.CandidateRow:
    """The one row whose DOI a module or a mirror files under ``citation_key``."""
    dois = {d for m in index if m.citation_key == citation_key for d in m.own_dois}
    for top in (gates.LIBRARY_DIR, gates.RAW_DIR):
        path = Path(root) / top / citation_key / "manifest.json"
        if path.is_file():
            doi = json.loads(path.read_text())["doi"]
            if doi:
                dois.add(doi)
    lowered = {d.lower() for d in dois}
    hits = [r for r in rows if r.doi is not None and r.doi.lower() in lowered]
    if len(hits) != 1:
        names = ", ".join(repr(r.name) for r in hits) or "none"
        raise UsageError(
            f"{citation_key}: {len(hits)} rows carry its DOI(s) {sorted(dois)} ({names}); "
            "pass --row"
        )
    return hits[0]


def load_doi_to_pmid(path: Path = DOI_TO_PMID_CACHE) -> dict[str, str] | None:
    """check_candidate_overlap.py's offline DOI -> PMID cache, None when never written."""
    if not path.is_file():
        return None
    return {k.lower(): str(v) for k, v in json.loads(path.read_text()).items()}


def render_verdict(verdict: CandidateVerdict) -> str:
    """The human summary ``gate`` prints: one line per gate, then the outcome."""
    lines = [f"{verdict.row_name}  [{verdict.table}]  {verdict.citation_key}"]
    for g in verdict.gates:
        kind = f":{g.blocked_kind}" if g.blocked_kind else ""
        issue = f" (#{g.issue})" if g.issue is not None else ""
        lines.append(f"  {g.gate} {g.outcome}{kind}{issue}: {g.reason}")
    if verdict.aggregation is not None:
        a = verdict.aggregation
        lines.append(
            f"  aggregation: {a.n_source_studies} source studies named by "
            f"{a.attribution_field!r}; values {a.value_origin}; mirrored "
            f"{_count(a.n_sources_mirrored)}; net-new vs served {_count(a.net_new_vs_served)}"
        )
    lines.append(f"  zotero item: {verdict.zotero_item}")
    lines.append(f"  outcome: {verdict.outcome}")
    return "\n".join(lines)


def _count(value: int | None) -> str:
    return "unmeasured" if value is None else str(value)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def cmd_schema(args: argparse.Namespace) -> int:
    """Print the verdict's JSON schema."""
    print(json.dumps(verdict_json_schema(), indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate an agent's JSON; with --write, store it."""
    text = Path(args.path).read_text(encoding="utf-8")
    try:
        verdict = CandidateVerdict.model_validate_json(text)
    except ValidationError as error:
        for item in error.errors():
            where = ".".join(str(p) for p in item["loc"]) or "<root>"
            print(f"refused: {where}: {item['msg']}")
        return 1
    print(render_verdict(verdict))
    if args.write:
        print(f"wrote {store.write_verdict(verdict, Path(args.store_dir))}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Run G1, G2, G3 and G4-key on a row (and G5 when a class is named)."""
    from torchcell.candidates.findings import finding_for

    if args.row is None and args.citation_key is None:
        raise UsageError("gate needs --row or --citation-key")
    root = data_root(args.data_root)
    table: CandidateTable = args.table
    rows = table_rows(table, load_table_module(table))
    index = gates.module_index()
    if args.row is not None:
        row = gates.find_row(rows, args.row)
    else:
        row = row_for_citation_key(rows, args.citation_key, index, root)
    key = args.citation_key
    if key is None:
        if row.doi is None:
            raise UsageError(f"{row.name}: no DOI in its url; pass --citation-key")
        key = gates.citation_key_for_doi(row.doi, index, root)
        if key is None:
            raise UsageError(
                f"{row.name}: no module or mirror files {row.doi}; pass --citation-key"
            )
    finding = finding_for(row.name)
    if finding is not None and finding.citation_key != key:
        raise UsageError(
            f"{row.name}: finding is filed under {finding.citation_key}, not {key}"
        )
    g1 = gates.evaluate_g1(row, finding)
    steps: dict[GateName, Callable[[], gates.GateRun]] = {
        "G2": lambda: gates.evaluate_g2(
            row, root, source_kind=g1.source_kind, verify=not args.no_verify
        ),
        "G3": lambda: gates.evaluate_g3(gates.inventory(key, root)),
        "G4": lambda: gates.evaluate_g4_key(
            row,
            key,
            index,
            doi_to_pmid=load_doi_to_pmid(),
            aggregator_pmids=gates.aggregator_pmid_sets(root),
        ),
    }
    if args.phenotype_class is not None:
        steps["G5"] = lambda: gates.evaluate_g5(
            args.phenotype_class,
            args.compound,
            issue=args.issue,
            proposed_class=args.proposed_class,
        )
    runs: dict[GateName, gates.GateRun] = {}
    stopped = g1.result.outcome in ("fail", "blocked")
    for name, step in steps.items():
        if stopped:
            break
        run = step()
        runs[name] = run
        stopped = run.result.outcome in ("fail", "blocked")
    verdict = gates.compose_verdict(
        row=row,
        citation_key=key,
        g1=g1,
        runs=runs,
        zotero=gates.zotero_item(key, root),
        decided_at=date.today().isoformat(),
        torchcell_commit=git_commit(),
    )
    print(verdict.model_dump_json(indent=2) if args.json else render_verdict(verdict))
    if args.write:
        print(f"wrote {store.write_verdict(verdict, Path(args.store_dir))}")
    return 1 if verdict.outcome == "refused" else 0


def cmd_inventory(args: argparse.Namespace) -> int:
    """Print G3's inventory of a key's deposit."""
    record = gates.inventory(args.citation_key, data_root(args.data_root))
    if record is None:
        print(f"{args.citation_key}: no manifest in the literature or the raw mirror")
        return 1
    print(record.model_dump_json(indent=2))
    print(gates.evaluate_g3(record).result.reason, file=sys.stderr)
    return 0


def read_released(path: Path) -> dict[str, dict[str, float]]:
    """A long released table: header ``column,key,value``, one value per line."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["column", "key", "value"]:
            raise UsageError(
                f"{path}: header must be column,key,value, not {reader.fieldnames}"
            )
        out: dict[str, dict[str, float]] = {}
        for line in reader:
            out.setdefault(line["column"], {})[line["key"]] = float(line["value"])
    return out


def with_g4_value(verdict: CandidateVerdict, value: G4ValueRecord) -> CandidateVerdict:
    """The verdict with G4 recomputed from its key record and ``value``; stop rule kept."""
    if verdict.g4_key is None:
        raise UsageError(
            f"{verdict.citation_key}: G4-key was not run; gate the row first"
        )
    g4 = gates.g4_result(verdict.g4_key, value)
    gate_list: list[GateResult] = []
    for g in verdict.gates:
        if g.gate == "G4":
            gate_list.append(g4)
        elif g.gate == "G5" and g4.outcome == "fail":
            gate_list.append(
                gates.unmeasured("G5", "not evaluated: G4 stopped the run")
            )
        else:
            gate_list.append(g)
    data = verdict.model_dump()
    data.update(
        gates=[g.model_dump() for g in gate_list],
        g4_value=value.model_dump(),
        g5=None if g4.outcome == "fail" else data["g5"],
    )
    return CandidateVerdict.model_validate(data)


def cmd_overlap(args: argparse.Namespace) -> int:
    """G4-value: a released table against a dev store, folded into the stored verdict."""
    verdict = store.read_verdict(args.citation_key, Path(args.store_dir))
    released = read_released(Path(args.released))
    try:
        n, commit, served = gates.read_store_values(
            args.store,
            sample_path=args.sample_path,
            key_path=args.key_path,
            value_path=args.value_path,
        )
    except gates.DirtyStoreError as error:
        print(f"refused: {error}")
        return 1
    record = G4ValueRecord(
        store=args.store,
        store_commit=commit,
        records_read=n,
        tolerance=args.tolerance,
        matches=gates.match_by_value(released, served),
    )
    updated = with_g4_value(verdict, record)
    for m in record.matches:
        margin = "n/a" if m.margin is None else f"{m.margin:.6g}"
        print(
            f"{m.released_column}: best {m.best_sample} max|d| {m.best_max_abs_diff:.6g} "
            f"over {m.keys_compared} keys; runner-up {m.runner_up_sample} margin {margin}"
        )
    print(render_verdict(updated))
    if args.write:
        print(f"wrote {store.write_verdict(updated, Path(args.store_dir))}")
    return 0


def cmd_ledger(args: argparse.Namespace) -> int:
    """Append today's ledger section rendered from the store."""
    verdicts = store.load_store(Path(args.store_dir)).values()
    section = ledger.render_section(verdicts, args.date)
    if args.print:
        print(section)
        return 0
    ledger.append_section(Path(args.note), section)
    print(f"appended {ledger.heading(args.date)!r} to {args.note}")
    return 0


_WHITESPACE: Final = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Whitespace collapsed and soft hyphens dropped: what a quote must survive (#758)."""
    return _WHITESPACE.sub(" ", text.replace("­", "")).strip()


def audit_evidence(value: SourcedValue, root: str) -> tuple[str, str]:
    """``(state, path)``: ok, drift, quote-missing, binary (sha256 only) or absent."""
    key = value.provenance.citation_key
    for top in (gates.LIBRARY_DIR, gates.RAW_DIR):
        path = Path(root) / top / str(key) / value.provenance.source_uri
        if not path.is_file():
            continue
        if sha256_file(path) != value.provenance.sha256:
            return "drift", str(path)
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return "binary", str(path)
        if normalize(value.quote) not in normalize(text):
            return "quote-missing", str(path)
        return "ok", str(path)
    return "absent", f"{key}/{value.provenance.source_uri}"


def cmd_audit(args: argparse.Namespace) -> int:
    """Re-read every quoted evidence slice of every stored verdict against its file."""
    root = data_root(args.data_root)
    states: Counter[str] = Counter()
    bad = 0
    for key, verdict in store.load_store(Path(args.store_dir)).items():
        evidence = [e for g in verdict.gates for e in g.evidence]
        if verdict.aggregation is not None:
            evidence.extend(verdict.aggregation.evidence)
        for value in evidence:
            state, where = audit_evidence(value, root)
            states[state] += 1
            if state in ("drift", "quote-missing", "absent"):
                bad += 1
                print(f"{key}: {state}: {where}")
    print(", ".join(f"{k} {v}" for k, v in sorted(states.items())) or "no evidence")
    return 1 if bad else 0


def cmd_status(args: argparse.Namespace) -> int:
    """Count stored verdicts by outcome."""
    verdicts = store.load_store(Path(args.store_dir))
    counts = Counter(v.outcome for v in verdicts.values())
    print(f"{len(verdicts)} verdict(s) in {args.store_dir}")
    for outcome, n in sorted(counts.items()):
        print(f"  {outcome}: {n}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The ``candidate-gate`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="candidate-gate", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--store-dir", default=str(store.STORE_DIR), help="verdict store"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("schema", help="print the verdict JSON schema").set_defaults(
        func=cmd_schema
    )

    p = sub.add_parser("validate", help="validate a verdict JSON")
    p.add_argument("path")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("gate", help="run the deterministic gates on a row")
    p.add_argument("--row")
    p.add_argument("--citation-key")
    p.add_argument("--table", choices=("bacteria", "yeast"), default="bacteria")
    p.add_argument("--data-root")
    p.add_argument("--phenotype-class")
    p.add_argument("--compound", action="append", default=[])
    p.add_argument("--issue", type=int)
    p.add_argument("--proposed-class")
    p.add_argument(
        "--no-verify", action="store_true", help="skip genome sha256 re-hash"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("inventory", help="G3 inventory of a key's deposit")
    p.add_argument("--citation-key", required=True)
    p.add_argument("--data-root")
    p.set_defaults(func=cmd_inventory)

    p = sub.add_parser("overlap", help="G4-value against a dev store (read-only)")
    p.add_argument("--citation-key", required=True)
    p.add_argument("--released", required=True, help="CSV with header column,key,value")
    p.add_argument("--store", required=True, help="dev store directory")
    p.add_argument("--sample-path", required=True)
    p.add_argument("--key-path", required=True)
    p.add_argument("--value-path", required=True)
    p.add_argument("--tolerance", type=float, default=0.01)
    p.add_argument("--write", action="store_true")
    p.set_defaults(func=cmd_overlap)

    p = sub.add_parser("ledger", help="append a dated ledger section")
    p.add_argument("--date", default=date.today().isoformat())
    p.add_argument("--note", default=str(ledger.LEDGER_NOTE))
    p.add_argument("--print", action="store_true", help="print instead of appending")
    p.set_defaults(func=cmd_ledger)

    p = sub.add_parser("audit", help="re-read every quoted evidence slice (#758)")
    p.add_argument("--data-root")
    p.set_defaults(func=cmd_audit)

    sub.add_parser("status", help="store summary").set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point of ``candidate-gate`` and ``python -m torchcell.candidates``."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (UsageError, LookupError) as error:
        print(f"candidate-gate: {error}", file=sys.stderr)
        return 2
