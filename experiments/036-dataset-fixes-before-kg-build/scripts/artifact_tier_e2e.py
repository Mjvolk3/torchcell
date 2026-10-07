# experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_e2e.py
# [[experiments.036-dataset-fixes-before-kg-build.scripts.artifact_tier_e2e]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_e2e.py
"""Phase 6 step 1 of the artifact tier: the Caudal pointer resolves locally, then over HTTP.

The loop under test (plan.artifact-tier.2026.10.07, phase 6): a Caudal 2024 record's
``ArtifactRef`` (the Peter 2018 gene-keyed tarball, member ``<gene>.fasta#<token>``)
resolves through the local genomes tier; with the local tier hidden (``data_root`` pointed
at an empty directory) the same ref resolves through a tc-data server started by this
script on a free port, lands in the artifact cache, and verifies; a second remote resolve
is a cache hit; ``check`` consults only the remote manifest and downloads nothing; a ref
whose sha256 disagrees with the manifest raises ``ArtifactIntegrityError`` at once.

Every claim in ``results/artifact_tier_e2e.json`` is measured here: sources, verified
flags, byte counts, cache paths and wall-clock seconds per step. The tc-data server is
``torchcell.datasets.server`` run as a subprocess with the local tiers as its roots and a
key minted for this run, so the bytes travel over a real socket.

Usage (from the repo root, the torchcell env)::

    python experiments/036-dataset-fixes-before-kg-build/scripts/artifact_tier_e2e.py
"""

from __future__ import annotations

import json
import os
import os.path as osp
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from torchcell.api_keys import mint_key
from torchcell.artifacts import ArtifactIntegrityError, TcDataSource, check, resolve
from torchcell.datasets.scerevisiae.caudal2024 import (
    refgene_member_ref,
    refgene_tarball_ref,
)

RESULTS = Path(__file__).resolve().parents[1] / "results" / "artifact_tier_e2e.json"
SLICE = (
    Path(__file__).resolve().parents[1] / "results" / "artifact_ref_loader_slice.json"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def first_member(slice_path: Path) -> tuple[str, str]:
    """``(gene, token)`` of the first ``<gene>.fasta#<token>`` member the phase 3 loader
    slice recorded (the member is everything after the tarball's ``#``).
    """
    data = json.loads(slice_path.read_text(encoding="utf-8"))
    stack: list[object] = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and item.startswith("tc://genomes/"):
            member = item.split("#", 1)[1] if "#" in item else ""
            if ".fasta#" in member:
                gene, token = member.split(".fasta#", 1)
                return gene, token
    raise LookupError(f"{slice_path} records no <gene>.fasta#<token> member ref")


def start_server(data_root: str, port: int, api_key: str) -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        {
            "TC_DATA_ROOT": os.environ.get("TC_DATA_E2E_STORE", "/bulk/tc-data"),
            "TC_DATA_RAW_ROOT": osp.join(data_root, "torchcell-raw"),
            "TC_DATA_GENOMES_ROOT": osp.join(data_root, "torchcell-genomes"),
            "TC_DATA_OBJECTS_ROOT": osp.join(data_root, "torchcell-objects"),
            "TC_DATA_API_KEYS": f"e2e:{api_key}",
            "TC_DATA_HOST": "127.0.0.1",
            "TC_DATA_PORT": str(port),
        }
    )
    env.pop("TC_DATA_KEYS_FILE", None)
    proc = subprocess.Popen(
        [sys.executable, "-m", "torchcell.datasets.server"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"tc-data exited with {proc.returncode} before /health")
        time.sleep(0.5)
    proc.kill()
    raise RuntimeError("tc-data did not answer /health within 60 s")


def main() -> int:
    load_dotenv()
    data_root = os.environ["DATA_ROOT"]
    tarball = refgene_tarball_ref(data_root)
    gene, token = first_member(SLICE)
    ref = refgene_member_ref(tarball, gene, token)
    out: dict[str, object] = {"ref": str(ref), "sha256": ref.sha256, "bytes": ref.bytes}

    t0 = time.perf_counter()
    local = resolve(ref, data_root=data_root)
    out["local"] = {
        "path": str(local.path),
        "source": local.source,
        "verified": local.verified,
        "seconds": round(time.perf_counter() - t0, 3),
    }

    wrong = ref.model_copy(update={"sha256": "0" * 64})
    try:
        resolve(wrong, data_root=data_root)
        out["wrong_sha256"] = "resolved (WRONG: must raise)"
    except ArtifactIntegrityError as exc:
        out["wrong_sha256"] = f"ArtifactIntegrityError: {exc}"[:200]

    port = free_port()
    api_key, _ = mint_key("e2e")
    proc = start_server(data_root, port, api_key)
    try:
        url = f"http://127.0.0.1:{port}"
        client = TcDataSource(url, api_key)
        with tempfile.TemporaryDirectory(prefix="artifact-e2e-") as empty:
            t0 = time.perf_counter()
            ok = check(ref, data_root=empty, client=client)
            out["remote_check"] = {
                "resolvable": ok,
                "seconds": round(time.perf_counter() - t0, 3),
                "cache_files_after": sorted(
                    str(p.relative_to(empty))
                    for p in Path(empty).rglob("*")
                    if p.is_file()
                ),
            }
            t0 = time.perf_counter()
            remote = resolve(ref, data_root=empty, client=client)
            seconds = time.perf_counter() - t0
            size = remote.path.stat().st_size
            out["remote"] = {
                "path": str(remote.path.relative_to(empty)),
                "source": remote.source,
                "verified": remote.verified,
                "bytes": size,
                "seconds": round(seconds, 3),
                "mb_per_s": round(size / seconds / 1e6, 1),
            }
            t0 = time.perf_counter()
            again = resolve(ref, data_root=empty, client=client)
            out["remote_again"] = {
                "same_path": again.path == remote.path,
                "source": again.source,
                "verified": again.verified,
                "seconds": round(time.perf_counter() - t0, 3),
            }
    finally:
        proc.terminate()
        proc.wait(timeout=20)

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
