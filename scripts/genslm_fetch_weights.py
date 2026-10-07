# scripts/genslm_fetch_weights.py
# [[scripts.genslm_fetch_weights]]
# https://github.com/Mjvolk3/torchcell/tree/main/scripts/genslm_fetch_weights.py
"""Fetch GenSLM foundation-model checkpoints from the authors' Globus endpoint into
``$DATA_ROOT/models/genslm/`` and pin them in ``manifest.json``.

The checkpoints are released only through Globus endpoint
``25918ad0-2a4e-4f37-bcfc-8183b19c3150`` (the ``ramanathanlab/genslm`` README), which
needs a logged-in Globus identity, so the transfer is driven through the Globus CLI
and a one-time interactive login::

    # one-time, interactive (prints a URL; log in with the Illinois identity; paste the code)
    $DATA_ROOT/envs/globus-cli/bin/globus login --no-local-server

    # see what the endpoint holds
    python scripts/genslm_fetch_weights.py ls
    python scripts/genslm_fetch_weights.py ls --path /genslm/models/

    # transfer one model's checkpoint to this host's Globus Connect Personal
    # endpoint (the weights directory must be inside its accessible paths), wait for
    # the task, verify, then pin it
    python scripts/genslm_fetch_weights.py transfer --model genslm_25M_patric \
        --path /genslm/models/25M/

    # pin a checkpoint that is already on disk (any route), recording how it got there
    python scripts/genslm_fetch_weights.py record --model genslm_25M_patric \
        --source-path /genslm/models/25M/patric_25m_epoch01-val_loss_0.57_bias_removed.pt \
        --retriever "globus transfer <task-id>"

The Globus CLI moves bytes only endpoint to endpoint, so this host needs a Globus
Connect Personal endpoint (``globusconnectpersonal -setup`` once, interactive; then
``-start``); ``transfer`` reads its id from ``globus endpoint local-id``. ``record``
is what ties the bytes to the provenance model: a checkpoint without a manifest entry
is refused by ``torchcell.models.genslm.GenSLM`` at load time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from torchcell.literature.manifest import RetrievalMethod, RetrievalRecord, sha256_file
from torchcell.models.genslm import (
    GENSLM_MODELS,
    GLOBUS_ENDPOINT_ID,
    WEIGHTS_MANIFEST,
    GenSLMWeightsManifest,
    default_weights_dir,
)


def globus_cli() -> str:
    """The Globus CLI executable: ``$GLOBUS_CLI`` or the scratch venv."""
    return os.environ.get(
        "GLOBUS_CLI",
        osp.join(os.environ["DATA_ROOT"], "envs", "globus-cli", "bin", "globus"),
    )


def run(args: list[str]) -> str:
    """Run a Globus CLI command and return stdout, failing loudly on error."""
    proc = subprocess.run(
        [globus_cli(), *args], capture_output=True, text=True, check=False
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"globus {' '.join(args)} failed ({proc.returncode})")
    return proc.stdout


def cmd_ls(args: argparse.Namespace) -> None:
    """List a path on the endpoint."""
    sys.stdout.write(run(["ls", f"{GLOBUS_ENDPOINT_ID}:{args.path}", "--long"]))


def cmd_transfer(args: argparse.Namespace) -> None:
    """Transfer one checkpoint to this host's endpoint, wait, and pin it."""
    spec = GENSLM_MODELS[args.model]
    weights_dir = Path(args.weights_dir or default_weights_dir()).resolve()
    weights_dir.mkdir(parents=True, exist_ok=True)
    source_path = args.path.rstrip("/") + "/" + spec.weights_file
    target = weights_dir / spec.weights_file
    local_id = args.dest_endpoint or run(["endpoint", "local-id"]).strip()
    submit = run(
        [
            "transfer",
            "--sync-level",
            "checksum",
            "--label",
            f"torchcell {spec.model_id}",
            "--format",
            "json",
            f"{GLOBUS_ENDPOINT_ID}:{source_path}",
            f"{local_id}:{target}",
        ]
    )
    task_id = json.loads(submit)["task_id"]
    print(f"submitted task {task_id}; waiting")
    run(["task", "wait", task_id, "--polling-interval", "30"])
    status = json.loads(run(["task", "show", task_id, "--format", "json"]))["status"]
    if status != "SUCCEEDED":
        raise SystemExit(f"task {task_id} ended {status}")
    write_record(
        weights_dir,
        spec.weights_file,
        source_path=source_path,
        retriever=f"globus transfer {task_id}",
    )


def cmd_record(args: argparse.Namespace) -> None:
    """Pin a checkpoint already on disk."""
    spec = GENSLM_MODELS[args.model]
    weights_dir = Path(args.weights_dir or default_weights_dir())
    write_record(
        weights_dir,
        spec.weights_file,
        source_path=args.source_path,
        retriever=args.retriever,
    )


def write_record(
    weights_dir: Path, weights_file: str, *, source_path: str, retriever: str
) -> None:
    """Hash the checkpoint and add (or replace) its record in the manifest."""
    path = weights_dir / weights_file
    if not path.is_file():
        raise FileNotFoundError(f"{path} is not on disk; nothing to record")
    manifest_path = weights_dir / WEIGHTS_MANIFEST
    manifest = (
        GenSLMWeightsManifest.model_validate_json(manifest_path.read_text())
        if manifest_path.is_file()
        else GenSLMWeightsManifest()
    )
    record = RetrievalRecord(
        method=RetrievalMethod.globus,
        source_url=f"globus://{GLOBUS_ENDPOINT_ID}{source_path}",
        retriever=retriever,
        params={"endpoint_id": GLOBUS_ENDPOINT_ID, "path": source_path},
        sha256=sha256_file(path),
        retrieved_at=dt.date.today().isoformat(),
    )
    manifest.files[weights_file] = record
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    print(f"{weights_file}: sha256 {record.sha256} -> {manifest_path}")


def main() -> None:
    """CLI entry point."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ls = sub.add_parser("ls", help="list a path on the GenSLM Globus endpoint")
    p_ls.add_argument("--path", default="/")
    p_ls.set_defaults(func=cmd_ls)

    p_tr = sub.add_parser("transfer", help="transfer one checkpoint here and pin it")
    p_tr.add_argument("--model", choices=list(GENSLM_MODELS), required=True)
    p_tr.add_argument("--path", required=True, help="endpoint directory of the file")
    p_tr.add_argument("--weights-dir", default=None)
    p_tr.add_argument(
        "--dest-endpoint", default=None, help="default: `globus endpoint local-id`"
    )
    p_tr.set_defaults(func=cmd_transfer)

    p_rec = sub.add_parser("record", help="pin a checkpoint already on disk")
    p_rec.add_argument("--model", choices=list(GENSLM_MODELS), required=True)
    p_rec.add_argument("--source-path", required=True, help="endpoint path of the file")
    p_rec.add_argument("--retriever", required=True, help="the command that fetched it")
    p_rec.add_argument("--weights-dir", default=None)
    p_rec.set_defaults(func=cmd_record)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
