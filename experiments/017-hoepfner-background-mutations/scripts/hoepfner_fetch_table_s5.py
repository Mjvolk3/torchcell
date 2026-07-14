# experiments/017-hoepfner-background-mutations/scripts/hoepfner_fetch_table_s5.py
# [[experiments.017-hoepfner-background-mutations.scripts.hoepfner_fetch_table_s5]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/017-hoepfner-background-mutations/scripts/hoepfner_fetch_table_s5
"""Retrieve Hoepfner 2014 Table_S5 (the 157 affected-HIP-strain list) + sha256-pin it.

Table_S5 = "Summary of identified HIP strains with common background mutations" lives in the
SAME sha256-pinned Dryad deposit we already build from (doi:10.5061/dryad.v5m8v, file id
4834604 -> /downloads/file_stream/4834604), so it is scriptable via the loader's existing
Anubis-PoW Dryad path -- clean, reproducible provenance. Deposit it into the library mirror
`si/` subdir (the canonical home for non-PDF supplements) and print its sha256 for pinning.
Then dump the sheet layout so the cross-validation pass can map columns.
"""

import hashlib
import os
import os.path as osp

import pandas as pd
import requests
from dotenv import load_dotenv

from torchcell.datasets.scerevisiae.hoepfner2014 import _UA, _dryad_get

load_dotenv()
DATA_ROOT = os.environ["DATA_ROOT"]
SI_DIR = osp.join(
    DATA_ROOT, "torchcell-library/hoepfnerHighresolutionChemicalDissection2014/si"
)
os.makedirs(SI_DIR, exist_ok=True)
DEST = osp.join(SI_DIR, "Table_S5.xls")
URL = "https://datadryad.org/downloads/file_stream/4834604"


def main() -> None:
    if not osp.exists(DEST):
        session = requests.Session()
        session.headers.update({"User-Agent": _UA})
        resp = _dryad_get(session, URL)
        digest = hashlib.sha256()
        with open(DEST, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
                    digest.update(chunk)
        resp.close()
        print(f"downloaded {DEST}")
    with open(DEST, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    print(f"Table_S5.xls  bytes={osp.getsize(DEST)}  sha256={sha}")
    print(f"source: {URL}  (Dryad doi:10.5061/dryad.v5m8v, file id 4834604)")

    xl = pd.ExcelFile(DEST)
    print(f"\nsheets: {xl.sheet_names}")
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=0)
        print(f"\n=== sheet '{sheet}'  shape={df.shape} ===")
        print("columns:", list(df.columns))
        print(df.head(10).to_string(max_colwidth=28))


if __name__ == "__main__":
    main()
