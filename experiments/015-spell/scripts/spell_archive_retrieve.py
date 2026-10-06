# experiments/015-spell/scripts/spell_archive_retrieve
# [[experiments.015-spell.scripts.spell_archive_retrieve]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/015-spell/scripts/spell_archive_retrieve

"""
Retrieve everything SGD publishes under its expression/microarray download
area, and write a provenance manifest for it.

SGD serves downloads from the public S3 bucket `sgd-archive.yeastgenome.org`
(us-west-2); `downloads.yeastgenome.org/expression/microarray/` is a JavaScript
browser over the same bucket. The area holds:

    all_spell_datasets.tar.gz      every study folder (README + PCL + zip)
    all_spell_readmes.tar.gz       the READMEs alone
    all_readmes.tar.gz             an older README bundle
    dual_channel_arrays.tar.gz     two-channel studies
    single_channel_arrays.tar.gz   one-channel studies
    all_spell_exprconn_pcl.tar.gz  the retired Expression Connection data
    archive/...                    Expression Connection files, one per dataset
    <Study>_<year>_PMID_<pmid>/    the same study folders, unpacked

The per-study folders repeat the contents of all_spell_datasets.tar.gz and are
not fetched. `../README.html` (the area's own README) is fetched.

`all_spell_datasets.tar.gz` was first downloaded by hand on 2025-12-21 with no
record. It is not fetched again: the local file is hashed and entered in the
manifest with `retrieved_at` taken from its filesystem birth time and
`retrieval_method = manual_browser`, and its size is checked against upstream.

Outputs ($DATA_ROOT/data/sgd/spell/):
    sgd_download/<key under expression/>   each retrieved file
    sgd_download/manifest.json             one provenance record per file

Usage:
    python experiments/015-spell/scripts/spell_archive_retrieve.py
"""

import hashlib
import html
import os
import os.path as osp
import re
from datetime import datetime, timezone
from enum import StrEnum

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]

SPELL_DIR = osp.join(DATA_ROOT, "data/sgd/spell")
DOWNLOAD_DIR = osp.join(SPELL_DIR, "sgd_download")
MANIFEST_PATH = osp.join(DOWNLOAD_DIR, "manifest.json")

BUCKET_URL = "https://s3-us-west-2.amazonaws.com/sgd-archive.yeastgenome.org"
PREFIX = "expression/"
HAND_DOWNLOADED_KEY = "expression/microarray/all_spell_datasets.tar.gz"
HAND_DOWNLOADED_PATH = osp.join(SPELL_DIR, "all_spell_datasets.tar.gz")
STUDY_FOLDER = re.compile(r"expression/microarray/[^/]+_\d{4}_PMID_\d+/")


class RetrievalMethod(StrEnum):
    direct_url = "direct_url"
    manual_browser = "manual_browser"


class RetrievedFile(BaseModel):
    key: str
    local_path: str
    source_url: str
    retrieval_method: RetrievalMethod
    retrieval_command: str
    retrieved_at: str
    retrieved_at_basis: str
    sha256: str
    bytes: int
    upstream_last_modified: str
    upstream_etag: str


class ArchiveManifest(BaseModel):
    bucket_url: str
    prefix: str
    listed_at: str
    n_study_folders_upstream: int
    files: list[RetrievedFile]


class UpstreamObject(BaseModel):
    key: str
    last_modified: str
    etag: str
    size: int


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_bucket(prefix: str) -> list[UpstreamObject]:
    objects: list[UpstreamObject] = []
    params = {"list-type": "2", "prefix": prefix}
    while True:
        response = requests.get(BUCKET_URL, params=params, timeout=120)
        response.raise_for_status()
        for block in re.findall(r"<Contents>(.*?)</Contents>", response.text, re.S):
            objects.append(
                UpstreamObject(
                    key=html.unescape(re.search(r"<Key>(.*?)</Key>", block).group(1)),
                    last_modified=re.search(
                        r"<LastModified>(.*?)</LastModified>", block
                    ).group(1),
                    etag=html.unescape(
                        re.search(r"<ETag>(.*?)</ETag>", block).group(1)
                    ).strip('"'),
                    size=int(re.search(r"<Size>(\d+)</Size>", block).group(1)),
                )
            )
        token = re.search(
            r"<NextContinuationToken>(.*?)</NextContinuationToken>", response.text
        )
        if token is None:
            return objects
        params["continuation-token"] = html.unescape(token.group(1))


def download(obj: UpstreamObject) -> RetrievedFile:
    url = f"{BUCKET_URL}/{obj.key}"
    local_path = osp.join(DOWNLOAD_DIR, obj.key[len(PREFIX) :])
    os.makedirs(osp.dirname(local_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)
    assert osp.getsize(local_path) == obj.size, obj.key
    return RetrievedFile(
        key=obj.key,
        local_path=osp.relpath(local_path, SPELL_DIR),
        source_url=url,
        retrieval_method=RetrievalMethod.direct_url,
        retrieval_command=f"requests.get('{url}', stream=True)",
        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        retrieved_at_basis="clock at download",
        sha256=sha256_file(local_path),
        bytes=obj.size,
        upstream_last_modified=obj.last_modified,
        upstream_etag=obj.etag,
    )


def hand_downloaded(obj: UpstreamObject) -> RetrievedFile:
    assert osp.getsize(HAND_DOWNLOADED_PATH) == obj.size
    birth = datetime.fromtimestamp(
        os.stat(HAND_DOWNLOADED_PATH).st_birthtime, timezone.utc
    )
    return RetrievedFile(
        key=obj.key,
        local_path=osp.relpath(HAND_DOWNLOADED_PATH, SPELL_DIR),
        source_url=(
            "http://sgd-archive.yeastgenome.org/expression/microarray/"
            "all_spell_datasets.tar.gz"
        ),
        retrieval_method=RetrievalMethod.manual_browser,
        retrieval_command=(
            "curl -O http://sgd-archive.yeastgenome.org/expression/microarray/"
            "all_spell_datasets.tar.gz  (recipe printed by "
            "torchcell/datasets/scerevisiae/spell.py; the actual command was "
            "not recorded)"
        ),
        retrieved_at=birth.isoformat(timespec="seconds"),
        retrieved_at_basis="filesystem birth time of the local file",
        sha256=sha256_file(HAND_DOWNLOADED_PATH),
        bytes=obj.size,
        upstream_last_modified=obj.last_modified,
        upstream_etag=obj.etag,
    )


def main() -> None:
    listed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    objects = list_bucket(PREFIX + "README.html") + list_bucket(PREFIX + "microarray/")
    study_folders = {
        m.group(0) for o in objects if (m := STUDY_FOLDER.match(o.key)) is not None
    }
    wanted = [o for o in objects if STUDY_FOLDER.match(o.key) is None and o.size > 0]
    print(f"upstream objects: {len(objects)}")
    print(f"study folders (not fetched, same as the tarball): {len(study_folders)}")
    print(f"files to record: {len(wanted)}, {sum(o.size for o in wanted):,} bytes")

    files = []
    for obj in wanted:
        if obj.key == HAND_DOWNLOADED_KEY:
            record = hand_downloaded(obj)
        else:
            record = download(obj)
        print(f"{record.bytes:>12,}  {record.sha256[:12]}  {record.key}")
        files.append(record)

    manifest = ArchiveManifest(
        bucket_url=BUCKET_URL,
        prefix=PREFIX,
        listed_at=listed_at,
        n_study_folders_upstream=len(study_folders),
        files=files,
    )
    with open(MANIFEST_PATH, "w") as f:
        f.write(manifest.model_dump_json(indent=1))
    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
